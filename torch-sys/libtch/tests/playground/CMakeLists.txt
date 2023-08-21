cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(playground)

add_executable(playground playground.cpp)
target_link_libraries(playground "${TORCH_LIBRARIES}")