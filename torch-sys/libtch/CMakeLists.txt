cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

include(CMakeDependentOption)
option(CARGO_BUILD "build with cargo" OFF)
CMAKE_DEPENDENT_OPTION(CARGO_LTO "compile with link time optimization" OFF "CARGO_BUILD" OFF)
CMAKE_DEPENDENT_OPTION(CARGO_TEST "build with cargo test" OFF "CARGO_BUILD" OFF)
CMAKE_DEPENDENT_OPTION(USE_CUDA "enable cuda" ON "CARGO_BUILD" ON)

# 使用clang编译以启用rust跨语言的lto
# (注:这里使用的是clang-10,以编译cuda相关文件,但rust那边是使用的llvm-16即
# 应对应clang-16,但没有报错)
set(CMAKE_C_COMPILER "/usr/bin/clang-12")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-12")

project(tch)

if (CARGO_LTO)
    include(CheckIPOSupported)
    check_ipo_supported()
endif ()

# 设置c++标准
set(CMAKE_CXX_STANDARD 17)

# 关闭警告
add_definitions(-w)

if (CARGO_BUILD)
    # 库名称
    set(lib_name "${CARGO_CXX_LIB_NAME}")
    # 编译选项
    set(compile_flags "${CARGO_CXX_COMPILE_FLAGS}")
    # 链接选项
    set(link_flags "${CARGO_CXX_LINK_FLAGS}")
    # 头文件目录
    set(header_dirs "${CARGO_CXX_INCLUDE_DIRS}")
    # 所有的源文件
    set(src "${CARGO_CXX_SRC}")
    # cuda头文件目录
    if (DEFINED CARGO_USE_CUDA)
        set(USE_CUDA ON)
        set(CUDA_INCLUDE_DIRS "${CARGO_CUDA_INCLUDE_DIRS}")
    endif ()
    # libtorch cmake文件目录
    set(Torch_DIR "${CARGO_TORCH_DIR}")
else ()
    set(lib_name "tch")
    # 默认使用conda安装的Torch库
    set(CONDA_TORCH_DIR "/opt/conda/lib/python3.10/site-packages/torch")
    set(Torch_DIR "${CONDA_TORCH_DIR}/share/cmake/Torch")
    # 默认编译选项
    set(compile_flags "-fPIC")
    # 默认链接选项
    set(link_flags "-Wl,-rpath=${CONDA_TORCH_DIR}/lib")
    # 默认为include、cxx-wrapper目录以及cxx头文件目录
    set(CARGO_OUT_DIR "/workspaces/tch-rs/target")
    set(header_dirs
            "${CMAKE_SOURCE_DIR}/include"
            "${CARGO_OUT_DIR}/cxxbridge/torch-sys/src"
            "${CARGO_OUT_DIR}/cxxbridge/rust"
            # 默认使用conda安装的python的头文件目录(编写python插件的cxx源文件需要相关头文件)
            #            "/opt/conda/include/python3.10"
            # c10d命名空间相关的头文件
            "${CONDA_TORCH_DIR}/include/torch/csrc/distributed/c10d"
    )

    # 默认为src目录下的所有文件以及cargo OUT_DIR目录下的cxx源文件
    file(GLOB_RECURSE src "${CMAKE_SOURCE_DIR}/src/*" "${CARGO_OUT_DIR}/cxxbridge/torch-sys/src/cxx_wrapper/*.cc")
    # 过滤掉fake的cuda文件
    list(FILTER src EXCLUDE REGEX ".*fake_cuda_dependency.cpp$")
    list(FILTER src EXCLUDE REGEX ".*torch_python.cpp$")

    # 默认启用cuda,且使用本地的cuda头文件
    set(CUDA_INCLUDE_DIRS "/usr/local/cuda")
endif ()

# Finds the Torch library
#
# This will define the following variables:
#
#   TORCH_FOUND        -- True if the system has the Torch library
#   TORCH_INCLUDE_DIRS -- The include directories for torch
#   TORCH_LIBRARIES    -- Libraries to link against
#   TORCH_CXX_FLAGS    -- Additional (required) compiler flags
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if (USE_CUDA)
    # 导入cuda相关的头文件
    include_directories("${CUDA_INCLUDE_DIRS}")
endif ()

# 导入libtorch头文件
include_directories("${TORCH_INCLUDE_DIRS}")

# 导入自定义的头文件
include_directories("${header_dirs}")

# 创建动态库
add_library("${lib_name}" SHARED "${src}")

if (CARGO_LTO)
    # 开启lto所需的编译与链接选项
    list(APPEND compile_flags "-flto=thin")
    list(APPEND compile_flags "-v")
    list(APPEND link_flags "-flto=thin")
    list(APPEND link_flags "-v")
    list(APPEND link_flags "-fuse-ld=lld-12")
    set_property(TARGET ${lib_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif ()

# 添加编译选项与链接选项
#list(JOIN link_flags " " link_flags)
target_compile_options("${lib_name}" PUBLIC "${compile_flags}")
target_link_options("${lib_name}" PUBLIC "${link_flags}")
#set_target_properties("${lib_name}" PROPERTIES LINK_FLAGS "${link_flags}")

target_link_libraries("${lib_name}" "${TORCH_LIBRARIES}")
install(TARGETS "${lib_name}" DESTINATION .)

if(NOT CARGO_BUILD)
   # 用于cpp侧专门的debug 
   add_subdirectory(tests/playground)
endif()
