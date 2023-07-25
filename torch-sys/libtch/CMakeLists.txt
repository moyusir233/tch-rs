cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(tch)

if (DEFINED CARGO_BUILD)
    # 头文件目录
    set(header_dirs "${CARGO_CXX_INCLUDE_DIRS}")
    # 源文件目录
    set(src "${CARGO_CXX_SRC_DIRS}")
    # cuda头文件目录
    if (DEFINED CARGO_USE_CUDA)
        set(USE_CUDA "")
        set(CUDA_INCLUDE_DIRS "${CARGO_CUDA_INCLUDE_DIRS}")
    endif()
    # libtorch头文件目录
    set(TORCH_INCLUDE_DIRS "${CARGO_TORCH_INCLUDE_DIRS}")
    # libtroch库目录
    set(TORCH_LIBRARIES "${CARGO_TORCH_LIBRARIES}")
else()
    # 默认为include目录
    set(header_dirs "include")
    # 默认为src目录
    set(src "src")
    # 默认启用cuda,且使用conda安装的cuda头文件目录
    set(USE_CUDA "")
    set(CUDA_INCLUDE_DIRS "/opt/conda/include")
    # 默认使用conda安装的Torch库的头文件目录与库文件目录
    set(CONDA_TORCH_DIR "/opt/conda/lib/python3.10/site-packages/torch")
    set(TORCH_INCLUDE_DIRS "${CONDA_TORCH_DIR}/include" "${CONDA_TORCH_DIR}/include/torch/csrc/api/include")
    set(TORCH_LIBRARIES "${CONDA_TORCH_DIR}/lib")
    # 默认使用conda安装的python的头文件目录(编写python插件的cxx源文件需要相关头文件)
    list(APPEND header_dirs "/opt/conda/include/python3.10")
endif()

# 导入pytorch和cuda相关的头文件
include_directories("${TORCH_INCLUDE_DIRS}")
if (DEFINED USE_CUDA)
    include_directories("${CUDA_INCLUDE_DIRS}")
endif()

# 导入自定义的头文件
include_directories("${header_dirs}")

# 搜索源文件
file(GLOB_RECURSE cxx_sources "${src}/*")
if (DEFINED USE_CUDA)
    # 过滤掉fake的cuda文件
    list(FILTER cxx_sources EXCLUDE REGEX ".*fake_cuda_dependency.cpp$")
else()
    # 过滤掉使用cuda的源文件
    list(FILTER cxx_sources EXCLUDE REGEX ".*dummy_cuda_dependency.cpp$")
endif()

add_library(tch SHARED ${cxx_sources})
target_link_libraries(tch "${TORCH_LIBRARIES}")
set_property(TARGET tch PROPERTY CXX_STANDARD 14)
install(TARGETS tch DESTINATION .)