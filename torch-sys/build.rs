// The LIBTORCH environment variable can be used to specify the directory
// where libtorch has been installed.
// When not specified this script downloads the cpu version for libtorch
// and extracts it in OUT_DIR.
//
// On Linux, the TORCH_CUDA_VERSION environment variable can be used,
// like 9.0, 90, or cu90 to specify the version of CUDA to use for libtorch.

use anyhow::{Context, Result};
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::{env, fs, io};
use walkdir::WalkDir;

const TORCH_VERSION: &str = "2.0.0";
const PYTHON_PRINT_PYTORCH_DETAILS: &str = r"
import torch
from torch.utils import cpp_extension
print('LIBTORCH_VERSION:', torch.__version__.split('+')[0])
print('LIBTORCH_CXX11:', torch._C._GLIBCXX_USE_CXX11_ABI)
for include_path in cpp_extension.include_paths():
  print('LIBTORCH_INCLUDE:', include_path)
for library_path in cpp_extension.library_paths():
  print('LIBTORCH_LIB:', library_path)
";

const PYTHON_PRINT_INCLUDE_PATH: &str = r"
import sysconfig
print('PYTHON_INCLUDE:', sysconfig.get_path('include'))
";

const NO_DOWNLOAD_ERROR_MESSAGE: &str = r"
Cannot find a libtorch install, you can either:
- Install libtorch manually and set the LIBTORCH environment variable to appropriate path.
- Use a system wide install in /usr/lib/libtorch.so.
- Use a Python environment with PyTorch installed by setting LIBTORCH_USE_PYTORCH=1

See the readme for more details:
https://github.com/LaurentMazare/tch-rs/blob/main/README.md
";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinkType {
    Dynamic,
    Static,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Os {
    Linux,
    Macos,
    Windows,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct SystemInfo {
    os: Os,
    python_interpreter: PathBuf,
    cxx11_abi: String,
    libtorch_include_dirs: Vec<PathBuf>,
    libtorch_lib_dir: PathBuf,
    link_type: LinkType,
}

// 负责利用cmake编译c++依赖库
#[derive(Default)]
struct TchCmakeBuilder {
    // 编译选项
    compile_flags: Vec<String>,
    // 链接选项
    link_flags: Vec<String>,
    // 头文件目录
    header_dirs: Vec<PathBuf>,
    // 源文件
    src: Vec<PathBuf>,
    // cuda头文件目录
    cuda_include_dirs: Vec<PathBuf>,
    // libtorch cmake文件的目录,用于cmake的find_package命令
    torch_cmake_dir: PathBuf,
}

struct SourceFileManager {
    libtch_project_dir: PathBuf,
}

#[cfg(feature = "ureq")]
fn download<P: AsRef<Path>>(source_url: &str, target_file: P) -> anyhow::Result<()> {
    let f = fs::File::create(&target_file)?;
    let mut writer = io::BufWriter::new(f);
    let response = ureq::get(source_url).call()?;
    let response_code = response.status();
    if response_code != 200 {
        anyhow::bail!("Unexpected response code {} for {}", response_code, source_url)
    }
    let mut reader = response.into_reader();
    std::io::copy(&mut reader, &mut writer)?;
    Ok(())
}

#[cfg(not(feature = "ureq"))]
fn download<P: AsRef<Path>>(_source_url: &str, _target_file: P) -> anyhow::Result<()> {
    anyhow::bail!("cannot use download without the ureq feature")
}

#[cfg(not(feature = "download-libtorch"))]
fn get_pypi_wheel_url_for_aarch64_macosx() -> anyhow::Result<String> {
    anyhow::bail!("cannot get pypi wheel url without the ureq feature")
}

#[cfg(feature = "download-libtorch")]
#[derive(serde::Deserialize, Debug)]
struct PyPiPackageUrl {
    url: String,
    filename: String,
}
#[cfg(feature = "download-libtorch")]
#[derive(serde::Deserialize, Debug)]
struct PyPiPackage {
    urls: Vec<PyPiPackageUrl>,
}
#[cfg(feature = "download-libtorch")]
fn get_pypi_wheel_url_for_aarch64_macosx() -> anyhow::Result<String> {
    let pypi_url = format!("https://pypi.org/pypi/torch/{TORCH_VERSION}/json");
    let response = ureq::get(pypi_url.as_str()).call()?;
    let response_code = response.status();
    if response_code != 200 {
        anyhow::bail!("Unexpected response code {} for {}", response_code, pypi_url)
    }
    let pypi_package: PyPiPackage = response.into_json()?;
    let urls = pypi_package.urls;
    let expected_filename = format!("torch-{TORCH_VERSION}-cp311-none-macosx_11_0_arm64.whl");
    let url = urls.iter().find_map(|pypi_url: &PyPiPackageUrl| {
        if pypi_url.filename == expected_filename {
            Some(pypi_url.url.clone())
        } else {
            None
        }
    });
    url.context("Failed to find arm64 macosx wheel from pypi")
}

fn extract<P: AsRef<Path>>(filename: P, outpath: P) -> anyhow::Result<()> {
    let file = fs::File::open(&filename)?;
    let buf = io::BufReader::new(file);
    let mut archive = zip::ZipArchive::new(buf)?;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = outpath.as_ref().join(file.mangled_name());
        if !file.name().ends_with('/') {
            println!(
                "File {} extracted to \"{}\" ({} bytes)",
                i,
                outpath.as_path().display(),
                file.size()
            );
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(p)?;
                }
            }
            let mut outfile = fs::File::create(&outpath)?;
            io::copy(&mut file, &mut outfile)?;
        }
    }

    // This is if we're unzipping a python wheel.
    if outpath.as_ref().join("torch").exists() {
        fs::rename(outpath.as_ref().join("torch"), outpath.as_ref().join("libtorch"))?;
    }
    Ok(())
}

fn env_var_rerun(name: &str) -> Result<String, env::VarError> {
    // 避免rebuild
    // println!("cargo:rerun-if-env-changed={name}");
    env::var(name)
}

fn version_check(version: &str) -> Result<()> {
    if env_var_rerun("LIBTORCH_BYPASS_VERSION_CHECK").is_ok() {
        return Ok(());
    }
    let version = version.trim();
    // Typical version number is 2.0.0+cpu or 2.0.0+cu117
    let version = match version.split_once('+') {
        None => version,
        Some((version, _)) => version,
    };
    if version != TORCH_VERSION {
        anyhow::bail!("this tch version expects PyTorch {TORCH_VERSION}, got {version}, this check can be bypassed by setting the LIBTORCH_BYPASS_VERSION_CHECK environment variable")
    }
    Ok(())
}

fn get_out_dir() -> &'static Path {
    use std::cell::OnceCell;
    use std::sync::Mutex;
    static OUT_DIR: Mutex<OnceCell<PathBuf>> = Mutex::new(OnceCell::new());

    let path = {
        let out_dir = OUT_DIR.lock().unwrap();
        out_dir.get_or_init(|| {
            PathBuf::from(
                std::env::var("OUT_DIR").context("failed to get cargo `OUT_DIR`").unwrap(),
            )
        }) as *const PathBuf
    };

    unsafe { &*path }
}

// 递归获得指定目录下的所有文件
fn get_all_file_path(dir: impl AsRef<Path>) -> Vec<PathBuf> {
    WalkDir::new(dir)
        .into_iter()
        .filter_map(|item| {
            let entry = item.context("failed to get `DirEntry`").unwrap();
            let path = entry.path();
            if path.is_file() {
                Some(path.to_owned())
            } else {
                None
            }
        })
        .collect()
}

impl SourceFileManager {
    // 生成cxx相关的源文件
    fn generate_cxx_wrapper(
        &self,
        cxx_rs_relative_dir: impl AsRef<Path>,
        cxx_wrapper_file_name: &str,
    ) -> cc::Build {
        cxx_build::bridges(get_all_file_path(cxx_rs_relative_dir).iter().filter(|path| {
            if path.is_dir() {
                return false;
            }

            path.file_name()
                .map(|file_name| file_name == cxx_wrapper_file_name)
                .context("failed to get file name")
                .unwrap()
        }))
    }
    // 生成autocxx相关的源文件
    fn generate_autocxx_wrapper<P: AsRef<Path>, I: IntoIterator<Item = P>>(
        &self,
        header_dirs: I,
        autocxx_rs_file: impl AsRef<Path>,
    ) -> cc::Build {
        let header_dirs: Vec<P> = header_dirs.into_iter().collect();
        autocxx_build::Builder::new(&autocxx_rs_file, header_dirs.iter().map(AsRef::as_ref))
            .extra_clang_args(&["-D", "USE_C10D_NCCL=1"]) // 解析头文件时所需要的额外参数
            .build()
            .context("failed to generate autocxx binding!")
            .unwrap()
    }
}

impl SourceFileManager {
    fn add_basic_src_and_header_dirs(
        &self,
        src: &mut Vec<PathBuf>,
        header_dirs: &mut Vec<PathBuf>,
        use_cuda: bool,
        use_hip: bool,
    ) {
        let original_src_len = src.len();
        let original_dirs_len = header_dirs.len();

        // 源文件
        let cuda_dependency = if use_cuda || use_hip {
            "src/dummy_cuda_dependency.cpp"
        } else {
            "src/fake_cuda_dependency.cpp"
        };
        src.extend([
            "src/torch_api_generated.cpp".into(),
            "src/torch_api.cpp".into(),
            cuda_dependency.into(),
        ]);
        if cfg!(feature = "python-extension") {
            src.push("src/torch_python.cpp".into());
            // 默认使用conda安装的python的头文件目录(编写python插件的cxx源文件需要相关头文件)
            header_dirs.push("/opt/conda/include/python3.10".into());
        }

        header_dirs.push("include".into());

        // 补全头文件目录与源文件的路径
        src[original_src_len..]
            .iter_mut()
            .chain(header_dirs[original_dirs_len..].iter_mut())
            .for_each(|path| {
                *path = self.libtch_project_dir.join(&path);
            });
    }

    fn add_cxx_wrapper_src_and_header_dirs(
        &self,
        src: &mut Vec<PathBuf>,
        header_dirs: &mut Vec<PathBuf>,
        cxx_rs_relative_dir: impl AsRef<Path>,
        cxx_wrapper_file_name: &str,
    ) {
        // 产生cxx相关的源文件
        self.generate_cxx_wrapper(&cxx_rs_relative_dir, cxx_wrapper_file_name);

        // 加入cxx相关的源文件,包括自定义存放在libtch目录下的一部分,以及cxx生成的、包装
        // 这一部分代码,为rust侧提供binding的另一部分
        src.extend(get_all_file_path(self.libtch_project_dir.join("src/wrappers")));

        // cxx所生产的代码的布局相关的注释如下,
        // 其中`...`为用于生成cxx代码的rust源文件所在的目录的相对路径,即为`cxx_rs_relative_dir`
        // We lay out the OUT_DIR as follows. Everything is namespaced under a cxxbridge
        // subdirectory to avoid stomping on other things that the caller's build script
        // might be doing inside OUT_DIR.
        //
        //     $OUT_DIR/
        //        cxxbridge/
        //           crate/
        //              $CARGO_PKG_NAME -> $CARGO_MANIFEST_DIR
        //           include/
        //              rust/
        //                 cxx.h
        //              $CARGO_PKG_NAME/
        //                 .../
        //                    lib.rs.h
        //           sources/
        //              $CARGO_PKG_NAME/
        //                 .../
        //                    lib.rs.cc
        //
        // The crate/ and include/ directories are placed on the #include path for the
        // current build as well as for downstream builds that have a direct dependency
        // on the current crate.
        // 加入cxx生成的源文件
        let out_dir = get_out_dir();
        src.extend(get_all_file_path(out_dir.join("cxxbridge/sources/torch-sys")));

        // 加入cxx生成的头文件的目录
        header_dirs.extend([
            out_dir.join("cxxbridge/include/torch-sys/src"),
            out_dir.join("cxxbridge/include/rust"),
        ]);
    }

    fn add_autocxx_wrapper_src_and_header_dirs(
        &self,
        src: &mut Vec<PathBuf>,
        header_dirs: &mut Vec<PathBuf>,
        cuda_include_dirs: &[impl AsRef<Path>],
        torch_include_dirs: &[impl AsRef<Path>],
        autocxx_rs_file: impl AsRef<Path>,
    ) {
        self.generate_autocxx_wrapper(
            header_dirs
                .iter()
                .map(AsRef::as_ref)
                .chain(cuda_include_dirs.iter().map(AsRef::as_ref))
                // torch相关的头文件
                .chain(torch_include_dirs.iter().map(AsRef::as_ref)),
            autocxx_rs_file,
        );
        let out_dir = get_out_dir();
        let autocxx_dir = out_dir.join("autocxx-build-dir");
        // 加入源文件
        src.extend(get_all_file_path(autocxx_dir.join("cxx")));
        // 加入头文件
        header_dirs.push(autocxx_dir.join("include"));
    }

    // 由于需要使用torch.cmake没有提供分布式训练相关的头文件目录(c10d),
    // 由该函数进行添加
    fn add_extra_libtorch_header_dirs(
        &self,
        header_dirs: &mut Vec<PathBuf>,
        libtorch_src_dir: impl AsRef<Path>,
    ) {
        // 添加c10d相关的头文件目录
        header_dirs.push(libtorch_src_dir.as_ref().join("include/torch/csrc/distributed/c10d"));
    }
}

impl TchCmakeBuilder {
    // 将结构体实例中保存的参数转换为cmake文件所需的参数
    fn config_cmake(
        &self,
        cmake_config: &mut cmake::Config,
        lib_name: &str,
        static_link: bool,
        use_cuda: bool,
        use_hip: bool,
    ) {
        #[inline]
        fn join_pathbufs(pathbufs: &[PathBuf], split: impl AsRef<OsStr>) -> OsString {
            pathbufs.iter().map(|dir| dir.as_os_str()).collect::<Vec<_>>().join(split.as_ref())
        }

        const CMAKE_LIST_SPLIT: &str = ";";
        // 设置表示是rust侧发起的编译的flag
        cmake_config.define("CARGO_BUILD", "ON");
        // 表示进行测试
        // if cfg!(test) {
        //     cmake_config.define("CARGO_TEST", "ON");
        // }
        // 表示进行静态链接
        if static_link {
            cmake_config.define("CARGO_STATIC_lINK", "ON");
        }
        // release模式下不允许动态链接
        if !cfg!(debug_assertions) {
            if !static_link {
                panic!("dynamic linking is not allowed in release mode!")
            }
            // release profile下启用跨语言的lto优化
            cmake_config.define("CARGO_LTO", "ON");
        }
        // 设置lib名称
        cmake_config.define("CARGO_CXX_LIB_NAME", lib_name);
        // 设置编译选项
        cmake_config.define("CARGO_CXX_COMPILE_FLAGS", self.compile_flags.join(CMAKE_LIST_SPLIT));
        // 设置链接选项
        cmake_config.define("CARGO_CXX_LINK_FLAGS", self.link_flags.join(CMAKE_LIST_SPLIT));

        // 设置头文件目录
        cmake_config
            .define("CARGO_CXX_INCLUDE_DIRS", join_pathbufs(&self.header_dirs, CMAKE_LIST_SPLIT));
        // 设置所有的源文件
        cmake_config.define("CARGO_CXX_SRC", join_pathbufs(&self.src, CMAKE_LIST_SPLIT));
        // 传递使用cuda的选项并设置cuda头文件目录
        if use_cuda || use_hip {
            cmake_config.define("CARGO_USE_CUDA", "ON");
            cmake_config.define(
                "CARGO_CUDA_INCLUDE_DIRS",
                join_pathbufs(&self.cuda_include_dirs, CMAKE_LIST_SPLIT),
            );
        }
        // 设置libtorch的cmake文件的目录
        cmake_config.define("CARGO_TORCH_DIR", &self.torch_cmake_dir);
    }

    fn build(
        &self,
        cmake_file_dir: PathBuf,
        lib_name: &str,
        static_link: bool,
        use_cuda: bool,
        use_hip: bool,
    ) {
        let mut cmake_config = cmake::Config::new(cmake_file_dir);
        self.config_cmake(&mut cmake_config, lib_name, static_link, use_cuda, use_hip);

        let dst = cmake_config.build();
        dst.try_exists().expect("failed to cmake build");
        println!("cargo:rustc-link-search=native={}", dst.display());

        println!("cargo:rerun-if-changed=libtch/CMakeLists.txt");
        self.header_dirs
            .iter()
            .flat_map(get_all_file_path)
            .chain(self.src.iter().cloned())
            .for_each(|path| {
                // 确保源文件和头文件都为绝对路径
                assert!(path.is_absolute());
                println!("cargo:rerun-if-changed={}", path.display());
            });
    }
}
impl SystemInfo {
    fn new() -> Result<Self> {
        let os = match env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS").as_str() {
            "linux" => Os::Linux,
            "windows" => Os::Windows,
            "macos" => Os::Macos,
            os => anyhow::bail!("unsupported TARGET_OS '{os}'"),
        };
        // Locate the currently active Python binary, similar to:
        // https://github.com/PyO3/maturin/blob/243b8ec91d07113f97a6fe74d9b2dcb88086e0eb/src/target.rs#L547
        let python_interpreter = match os {
            Os::Windows => PathBuf::from("python.exe"),
            Os::Linux | Os::Macos => {
                if env::var_os("VIRTUAL_ENV").is_some() {
                    PathBuf::from("python")
                } else {
                    PathBuf::from("python3")
                }
            }
        };
        let mut libtorch_include_dirs = vec![];
        if cfg!(feature = "python-extension") {
            let output = std::process::Command::new(&python_interpreter)
                .arg("-c")
                .arg(PYTHON_PRINT_INCLUDE_PATH)
                .output()
                .with_context(|| format!("error running {python_interpreter:?}"))?;
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                if let Some(path) = line.strip_prefix("PYTHON_INCLUDE: ") {
                    libtorch_include_dirs.push(PathBuf::from(path))
                }
            }
        }
        let mut libtorch_lib_dir = None;
        let cxx11_abi = if env_var_rerun("LIBTORCH_USE_PYTORCH").is_ok() {
            let output = std::process::Command::new(&python_interpreter)
                .arg("-c")
                .arg(PYTHON_PRINT_PYTORCH_DETAILS)
                .output()
                .with_context(|| format!("error running {python_interpreter:?}"))?;
            let mut cxx11_abi = None;
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                if let Some(version) = line.strip_prefix("LIBTORCH_VERSION: ") {
                    version_check(version)?
                }
                match line.strip_prefix("LIBTORCH_CXX11: ") {
                    Some("True") => cxx11_abi = Some("1".to_owned()),
                    Some("False") => cxx11_abi = Some("0".to_owned()),
                    _ => {}
                }
                if let Some(path) = line.strip_prefix("LIBTORCH_INCLUDE: ") {
                    libtorch_include_dirs.push(PathBuf::from(path))
                }
                if let Some(path) = line.strip_prefix("LIBTORCH_LIB: ") {
                    libtorch_lib_dir = Some(PathBuf::from(path))
                }
            }
            match cxx11_abi {
                Some(cxx11_abi) => cxx11_abi,
                None => anyhow::bail!("no cxx11 abi returned by python {output:?}"),
            }
        } else {
            let libtorch = Self::prepare_libtorch_dir(os)?;
            let includes = env_var_rerun("LIBTORCH_INCLUDE")
                .map(PathBuf::from)
                .unwrap_or_else(|_| libtorch.clone());
            let lib = env_var_rerun("LIBTORCH_LIB")
                .map(PathBuf::from)
                .unwrap_or_else(|_| libtorch.clone());
            let mut version_file = libtorch;
            version_file.push("build-version");
            if version_file.exists() {
                if let Ok(version) = std::fs::read_to_string(&version_file) {
                    version_check(&version)?
                }
            }
            libtorch_include_dirs.push(includes.join("include"));
            libtorch_include_dirs.push(includes.join("include/torch/csrc/api/include"));
            libtorch_lib_dir = Some(lib.join("lib"));
            env_var_rerun("LIBTORCH_CXX11_ABI").unwrap_or_else(|_| "1".to_owned())
        };
        let libtorch_lib_dir = libtorch_lib_dir.expect("no libtorch lib dir found");
        let link_type = match env_var_rerun("LIBTORCH_STATIC").as_deref() {
            Ok("0") | Ok("false") | Ok("FALSE") => LinkType::Dynamic,
            // 默认静态链接
            _ => LinkType::Static,
        };
        Ok(Self {
            os,
            python_interpreter,
            cxx11_abi,
            libtorch_include_dirs,
            libtorch_lib_dir,
            link_type,
        })
    }

    fn check_system_location(os: Os) -> Option<PathBuf> {
        match os {
            Os::Linux => Path::new("/usr/lib/libtorch.so").exists().then(|| PathBuf::from("/usr")),
            _ => None,
        }
    }

    fn prepare_libtorch_dir(os: Os) -> Result<PathBuf> {
        if let Ok(libtorch) = env_var_rerun("LIBTORCH") {
            Ok(PathBuf::from(libtorch))
        } else if let Some(pathbuf) = Self::check_system_location(os) {
            Ok(pathbuf)
        } else {
            if !cfg!(feature = "download-libtorch") {
                anyhow::bail!(NO_DOWNLOAD_ERROR_MESSAGE)
            }

            let device = match env_var_rerun("TORCH_CUDA_VERSION") {
                Ok(cuda_env) => match os {
                    Os::Linux | Os::Windows => cuda_env
                        .trim()
                        .to_lowercase()
                        .trim_start_matches("cu")
                        .split('.')
                        .take(2)
                        .fold("cu".to_owned(), |mut acc, curr| {
                            acc += curr;
                            acc
                        }),
                    os => anyhow::bail!(
                        "CUDA was specified with `TORCH_CUDA_VERSION`, but pre-built \
                 binaries with CUDA are only available for Linux and Windows, not: {os:?}.",
                    ),
                },
                Err(_) => "cpu".to_owned(),
            };

            let libtorch_dir = get_out_dir().join("libtorch");
            if !libtorch_dir.exists() {
                fs::create_dir(&libtorch_dir).unwrap_or_default();
                let libtorch_url = match os {
                Os::Linux => format!(
                    "https://download.pytorch.org/libtorch/{}/libtorch-cxx11-abi-shared-with-deps-{}{}.zip",
                    device, TORCH_VERSION, match device.as_ref() {
                        "cpu" => "%2Bcpu",
                        "cu102" => "%2Bcu102",
                        "cu113" => "%2Bcu113",
                        "cu116" => "%2Bcu116",
                        "cu117" => "%2Bcu117",
                        "cu118" => "%2Bcu118",
                        _ => anyhow::bail!("unsupported device {device}, TORCH_CUDA_VERSION may be set incorrectly?"),
                    }
                ),
                Os::Macos => {
                    if env::var("CARGO_CFG_TARGET_ARCH") == Ok(String::from("aarch64")) {
                        get_pypi_wheel_url_for_aarch64_macosx().expect(
                            "Failed to retrieve torch from pypi.  Pre-built version of libtorch for apple silicon are not available.
                            You can install torch manually following the indications from https://github.com/LaurentMazare/tch-rs/issues/629
                            pip3 install torch=={TORCH_VERSION}
                            Then update the following environment variables:
                            export LIBTORCH=$(python3 -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
                            export DYLD_LIBRARY_PATH=${{LIBTORCH}}/lib
                            ")
                    } else {
                        format!("https://download.pytorch.org/libtorch/cpu/libtorch-macos-{TORCH_VERSION}.zip")
                    }
                },
                Os::Windows => format!(
                    "https://download.pytorch.org/libtorch/{}/libtorch-win-shared-with-deps-{}{}.zip",
                    device, TORCH_VERSION, match device.as_ref() {
                        "cpu" => "%2Bcpu",
                        "cu102" => "%2Bcu102",
                        "cu113" => "%2Bcu113",
                        "cu116" => "%2Bcu116",
                        "cu117" => "%2Bcu117",
                        "cu118" => "%2Bcu118",
                        _ => ""
                    }),
            };

                let filename = libtorch_dir.join(format!("v{TORCH_VERSION}.zip"));
                download(&libtorch_url, &filename)?;
                extract(&filename, &libtorch_dir)?;
            }
            Ok(libtorch_dir.join("libtorch"))
        }
    }

    #[allow(dead_code)]
    fn make(&self, lib_name: &str, use_cuda: bool, use_hip: bool) {
        let cuda_dependency = if use_cuda || use_hip {
            "libtch/src/dummy_cuda_dependency.cpp"
        } else {
            "libtch/src/fake_cuda_dependency.cpp"
        };
        println!("cargo:rerun-if-changed={}", cuda_dependency);
        println!("cargo:rerun-if-changed=libtch/src/torch_python.cpp");
        println!("cargo:rerun-if-changed=libtch/include/torch_python.h");
        println!("cargo:rerun-if-changed=libtch/src/torch_api_generated.cpp");
        println!("cargo:rerun-if-changed=libtch/include/torch_api_generated.h");
        println!("cargo:rerun-if-changed=libtch/src/torch_api.cpp");
        println!("cargo:rerun-if-changed=libtch/include/torch_api.h");
        println!("cargo:rerun-if-changed=libtch/include/stb_image_write.h");
        println!("cargo:rerun-if-changed=libtch/include/stb_image_resize.h");
        println!("cargo:rerun-if-changed=libtch/include/stb_image.h");
        let mut c_files =
            vec!["libtch/src/torch_api.cpp", "libtch/src/torch_api_generated.cpp", cuda_dependency];
        if cfg!(feature = "python-extension") {
            c_files.push("libtch/src/torch_python.cpp")
        }

        match self.os {
            Os::Linux | Os::Macos => {
                // Pass the libtorch lib dir to crates that use torch-sys. This will be available
                // as DEP_TORCH_SYS_LIBTORCH_LIB, see:
                // https://doc.rust-lang.org/cargo/reference/build-scripts.html#the-links-manifest-key
                println!("cargo:libtorch_lib={}", self.libtorch_lib_dir.display());
                cc::Build::new()
                    .cpp(true)
                    .pic(true)
                    .warnings(false)
                    .include("libtch/include")
                    .includes(&self.libtorch_include_dirs)
                    .flag(&format!("-Wl,-rpath={}", self.libtorch_lib_dir.display()))
                    .flag("-std=c++14")
                    .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", self.cxx11_abi))
                    .files(&c_files)
                    .compile(lib_name);
            }
            Os::Windows => {
                // TODO: Pass "/link" "LIBPATH:{}" to cl.exe in order to emulate rpath.
                //       Not yet supported by cc=rs.
                //       https://github.com/alexcrichton/cc-rs/issues/323
                cc::Build::new()
                    .cpp(true)
                    .pic(true)
                    .warnings(false)
                    .includes(&self.libtorch_include_dirs)
                    .files(&c_files)
                    .compile("tch");
            }
        };
    }

    // 利用cmake编译封装libtorch的代码,包括cxx生成的相关源文件
    #[allow(dead_code)]
    fn cmake(&self, lib_name: &str, use_cuda: bool, use_hip: bool) {
        // libtorch的源码目录
        let libtorch_src_dir =
            self.libtorch_lib_dir.parent().context("failed to find the libtorch src dir").unwrap();
        // 关于libtorch的rust binding的目录,即torch-sys/libtch
        let libtorch_binding_dir = PathBuf::from(
            env_var_rerun("CARGO_MANIFEST_DIR")
                .context("failed to get `CARGO_MANIFEST_DIR`")
                .unwrap(),
        )
        .join("libtch");
        // 用于生成cxx相关的源文件的rs代码的相对目录
        let cxx_rs_relative_dir = "src/wrappers";
        // 用于编写cxx wrapper的rs源文件的名称
        let cxx_wrapper_file_name = "cxx_wrappers.rs";
        // 用于生成autocxx binding的rust源文件
        let autocxx_rs_file = "src/wrappers/autocxx_wrappers.rs";
        // cuda的头文件目录
        let cuda_include_dirs = [PathBuf::from("/usr/local/cuda/include")];

        let source_file_manager =
            SourceFileManager { libtch_project_dir: libtorch_binding_dir.clone() };
        let (mut src, mut header_dirs) = (vec![], vec![]);

        source_file_manager.add_basic_src_and_header_dirs(
            &mut src,
            &mut header_dirs,
            use_cuda,
            use_hip,
        );
        source_file_manager.add_cxx_wrapper_src_and_header_dirs(
            &mut src,
            &mut header_dirs,
            cxx_rs_relative_dir,
            cxx_wrapper_file_name,
        );
        source_file_manager.add_extra_libtorch_header_dirs(&mut header_dirs, libtorch_src_dir);
        // 因为autocxx需要利用bindgen去读取c++头文件,注意在所有头文件目录添加完毕后才去生成
        source_file_manager.add_autocxx_wrapper_src_and_header_dirs(
            &mut src,
            &mut header_dirs,
            &cuda_include_dirs,
            &self.libtorch_include_dirs,
            autocxx_rs_file,
        );

        match self.os {
            Os::Linux | Os::Macos => {
                // Pass the libtorch lib dir to crates that use torch-sys. This will be available
                // as DEP_TORCH_SYS_LIBTORCH_LIB, see:
                // https://doc.rust-lang.org/cargo/reference/build-scripts.html#the-links-manifest-key
                println!("cargo:libtorch_lib={}", self.libtorch_lib_dir.display());

                let mut cmake_builder = TchCmakeBuilder::default();
                // 编译选项
                cmake_builder.compile_flags.extend([
                    "-fPIC".into(),
                    "-std=c++14".into(),
                    format!("-D_GLIBCXX_USE_CXX11_ABI={}", self.cxx11_abi),
                ]);
                // 链接选项
                cmake_builder
                    .link_flags
                    // 指定搜索动态库的目录
                    .push(format!("-Wl,-rpath={}", self.libtorch_lib_dir.display()));
                // torch cmake文件目录
                cmake_builder.torch_cmake_dir = libtorch_src_dir.join("share/cmake/Torch");
                // cuda头文件目录,使用本地环境的cuda
                // TODO 添加相关的环境变量?
                cmake_builder.cuda_include_dirs.extend(cuda_include_dirs);
                // 头文件目录
                cmake_builder.header_dirs.extend(header_dirs);
                // 源文件
                cmake_builder.src.extend(src);
                // 进行编译
                cmake_builder.build(
                    libtorch_binding_dir,
                    lib_name,
                    matches!(self.link_type, LinkType::Static),
                    use_cuda,
                    use_hip,
                );
            }
            Os::Windows => unimplemented!(),
        };
    }
    fn link(&self, lib_name: &str) {
        match self.link_type {
            LinkType::Dynamic => println!("cargo:rustc-link-lib={lib_name}"),
            LinkType::Static => {
                // TODO: whole-archive might only be necessary for libtorch_cpu?
                println!("cargo:rustc-link-lib=static:+whole-archive,-bundle={lib_name}")
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    if !cfg!(feature = "doc-only") {
        let system_info = SystemInfo::new()?;
        // use_cuda is a hacky way to detect whether cuda is available and
        // if it's the case link to it by explicitly depending on a symbol
        // from the torch_cuda library.
        // It would be better to use -Wl,--no-as-needed but there is no way
        // to specify arbitrary linker flags at the moment.
        //
        // Once https://github.com/rust-lang/cargo/pull/8441 is available
        // we should switch to using rustc-link-arg instead e.g. with the
        // following flags:
        //   -Wl,--no-as-needed -Wl,--copy-dt-needed-entries -ltorch
        //
        // This will be available starting from cargo 1.50 but will be a nightly
        // only option to start with.
        // https://github.com/rust-lang/cargo/blob/master/CHANGELOG.md
        //
        // Update: The above doesn't seem to propagate to the downstream binaries
        // so doesn't really help, the comment has been kept though to keep track
        // if this issue.
        // TODO: Try out the as-needed native link modifier when it lands.
        // https://github.com/rust-lang/rust/issues/99424
        let si_lib = &system_info.libtorch_lib_dir;
        let use_cuda =
            si_lib.join("libtorch_cuda.so").exists() || si_lib.join("torch_cuda.dll").exists();
        let use_cuda_cu = si_lib.join("libtorch_cuda_cu.so").exists()
            || si_lib.join("torch_cuda_cu.dll").exists();
        let use_cuda_cpp = si_lib.join("libtorch_cuda_cpp.so").exists()
            || si_lib.join("torch_cuda_cpp.dll").exists();
        let use_hip =
            si_lib.join("libtorch_hip.so").exists() || si_lib.join("torch_hip.dll").exists();
        println!("cargo:rustc-link-search={}", si_lib.display());

        if system_info.link_type != LinkType::Static {
            system_info.cmake("tch", use_cuda, use_hip);
            println!("cargo:rustc-link-lib=tch");
        } else {
            system_info.cmake("tch", use_cuda, use_hip);
            println!("cargo:rustc-link-lib=static=tch");

            if use_cuda {
                println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
                system_info.link("torch_cuda");
                system_info.link("c10_cuda");
                system_info.link("cudart");
            }
            if use_cuda_cu {
                system_info.link("torch_cuda_cu")
            }
            if use_cuda_cpp {
                system_info.link("torch_cuda_cpp")
            }
            if use_hip {
                system_info.link("torch_hip")
            }
            if cfg!(feature = "python-extension") {
                system_info.link("torch_python")
            }

            // TODO: this has only be tried out on the cpu version. Check that it works
            // with cuda too and maybe just try linking all available files?
            // system_info.link("asmjit");
            // system_info.link("clog");
            // system_info.link("cpuinfo");
            // system_info.link("dnnl");
            // system_info.link("dnnl_graph");
            // system_info.link("fbgemm");
            // system_info.link("gloo");
            // system_info.link("kineto");
            // system_info.link("nnpack");
            // system_info.link("onnx");
            // system_info.link("onnx_proto");
            // system_info.link("protobuf");
            // system_info.link("pthreadpool");
            // system_info.link("pytorch_qnnpack");
            // system_info.link("sleef");
            // system_info.link("tensorpipe");
            // system_info.link("tensorpipe_uv");
            // system_info.link("XNNPACK");

            system_info.link("torch_cpu");
            system_info.link("torch");
            system_info.link("c10");
            if use_hip {
                system_info.link("c10_hip");
            }
        }

        let target = env::var("TARGET").context("TARGET variable not set")?;

        if !target.contains("msvc") && !target.contains("apple") {
            println!("cargo:rustc-link-lib=gomp");
        }
    }
    Ok(())
}
