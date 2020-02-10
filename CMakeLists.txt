cmake_minimum_required(VERSION 3.12)
project(world-seed-tester-gpu CUDA CXX)

set(CMAKE_C_STANDARD 11)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA support")
else()
    message(STATUS "No CUDA support")
endif()

add_executable(world-seed-tester-gpu main.cu)