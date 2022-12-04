cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(CUDA_Freshman CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-arch=compute_75 -g -G -O3")
add_executable(stencil_1d_constant_read_only stencil_1d_constant_read_only.cu)
