cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(CUDA_Freshman CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-arch=compute_75 -g -G -O3")
add_executable(stream_call_back stream_call_back.cu)
