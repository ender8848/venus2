set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
cmake_minimum_required(VERSION 3.22)
project(gemmc CUDA)

set(CMAKE_CUDA_STANDARD 11)

set(CMAKE_CUDA_ARCHITECTURES 61)

add_executable(
        CutlassTest main.cu
        src/gemmGPU.cuh
        src/Interval.cuh
        test/IntervalTest.cuh
        test/gemmGPUCTest.cuh
        test/gemmGPUPyTest.cuh src/util.cuh src/mma.cuh test/mmaGPUCTest.cuh test/mmaGPUPyTest.cuh)

set_target_properties(CutlassTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

