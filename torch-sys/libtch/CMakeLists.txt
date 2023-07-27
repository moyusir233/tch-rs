cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(tch)

if (DEFINED CARGO_BUILD)
    # 库名称
    set(lib_name "${CARGO_CXX_LIB_NAME}")
    # 编译选项
    set(compile_flags "${CARGO_CXX_COMPILE_FLAGS}")
    # 头文件目录
    set(header_dirs "${CARGO_CXX_INCLUDE_DIRS}")
    # 所有的源文件
    set(src "${CARGO_CXX_SRC}")
    # cuda头文件目录
    if (DEFINED CARGO_USE_CUDA)
        set(USE_CUDA "")
        set(CUDA_INCLUDE_DIRS "${CARGO_CUDA_INCLUDE_DIRS}")
    endif()
    # libtorch cmake文件目录
    set(Torch_DIR "${CARGO_TORCH_DIR}")
    # # libtorch头文件目录
    # set(TORCH_INCLUDE_DIRS "${CARGO_TORCH_INCLUDE_DIRS}")
    # # libtroch库文件
    # set(TORCH_LIBRARIES "${CARGO_TORCH_LIBRARIES}")
else()
    set(lib_name "tch")
    # 默认编译选项
    set(compile_flags "-fPIC")
    # 默认为include、cxx-wrapper目录以及cxx头文件目录
    set(CARGO_OUT_DIR "/workspace/tch-rs/target" )
    set(header_dirs 
            "${CMAKE_SOURCE_DIR}/include" 
            "${CMAKE_SOURCE_DIR}/include/cxx-wrapper"
            "${CARGO_OUT_DIR}/cxxbridge/torch-sys/src/cxx_wrapper"
            "${CARGO_OUT_DIR}/cxxbridge/rust"
            # 默认使用conda安装的python的头文件目录(编写python插件的cxx源文件需要相关头文件)
            "/opt/conda/include/python3.10"
    )

    # 默认为src目录下的所有文件以及cargo OUT_DIR目录下的cxx源文件
    file(GLOB_RECURSE src "${CMAKE_SOURCE_DIR}/src/*" "${CARGO_OUT_DIR}/cxxbridge/torch-sys/src/cxx_wrapper/*.cc")
    # 过滤掉fake的cuda文件
    list(FILTER src EXCLUDE REGEX ".*fake_cuda_dependency.cpp$")

    # 默认启用cuda,且使用conda安装的cuda头文件目录
    set(USE_CUDA "")
    set(CUDA_INCLUDE_DIRS "/opt/conda/include")

    # 默认使用conda安装的Torch库
    set(Torch_DIR "/opt/conda/lib/python3.10/site-packages/torch/share/cmake/Torch")
    # set(CONDA_TORCH_DIR "/opt/conda/lib/python3.10/site-packages/torch")
    # set(TORCH_INCLUDE_DIRS "${CONDA_TORCH_DIR}/include" "${CONDA_TORCH_DIR}/include/torch/csrc/api/include")
    # file(GLOB_RECURSE TORCH_LIBRARIES "${CONDA_TORCH_DIR}/lib/*")
endif()

# Finds the Torch library
#
# This will define the following variables:
#
#   TORCH_FOUND        -- True if the system has the Torch library
#   TORCH_INCLUDE_DIRS -- The include directories for torch
#   TORCH_LIBRARIES    -- Libraries to link against
#   TORCH_CXX_FLAGS    -- Additional (required) compiler flags
find_package(Torch REQUIRED)

# 导入pytorch和cuda相关的头文件
if (DEFINED USE_CUDA)
    include_directories("${CUDA_INCLUDE_DIRS}")
endif()
include_directories("${TORCH_INCLUDE_DIRS}")

# 导入自定义的头文件
include_directories("${header_dirs}")

# 添加编译flag
add_compile_options("${compile_flags}")

add_library("${lib_name}" SHARED "${src}")
target_link_libraries("${lib_name}" "${TORCH_LIBRARIES}")
set_property(TARGET "${lib_name}" PROPERTY CXX_STANDARD 14)
install(TARGETS "${lib_name}" DESTINATION .)