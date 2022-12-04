cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(CUDA_Freshman CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-arch=compute_75 -g -G -O3  -Xptxas -dlcm=cg")
add_executable(sum_array_offset sum_array_offset.cu)
