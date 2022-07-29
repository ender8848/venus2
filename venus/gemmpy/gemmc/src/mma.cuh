//
// Created by hao on 23/07/22.
//

#ifndef CUTLASSTEST_MMA_CUH
#define CUTLASSTEST_MMA_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include "util.cuh"
#include "Interval.cuh"
#include "datatype.cuh"


// kernel function
// each thread calculates one item in matrix
template<typename T>
__global__ void mma2DKernel(T * MatA, T * MatB, T * MatC, int nx, int ny)
{
    int ix = threadIdx.x+blockDim.x*blockIdx.x;
    int iy = threadIdx.y+blockDim.y*blockIdx.y;
    int idx = ix+iy*ny;
    if (ix<nx && iy<ny)
    {
        MatC[idx] = MatA[idx]+MatB[idx];
    }
}


/*
 * perform GPU matrix addition
 * supported datatypes are float, double, Interval<float> and Interval<double>
 * @param dest_dev: destination matrix holding the calculation result
 * @param A_dev: device pointer to matrix A
 * @param B_dev: device pointer to matrix B
 * @param nx: number of rows in matrix A and matrix B
 * @param ny: number of columns in matrix A and matrix B
 */
template<typename T>
void mma2DGPUC(T *dest_dev, T *A_dev, T *B_dev, int nx, int ny) {
    // 2-d bolck ，32×32
    dim3 block(32, 32);
    // 2-d grid，128×128
    dim3 grid((nx-1)/block.x+1, (ny-1)/block.y+1);
    // execute kernel function in grid
    mma2DKernel<<<grid, block>>>(A_dev, B_dev, dest_dev, nx, ny);
    cudaDeviceSynchronize();
}



extern "C" {
/*
 * API that is used by python to do matrix addition (mainly on interval matrix)
 * @param dest_dev: destination matrix holding the calculation result
 * @param A_dev: device pointer to matrix A
 * @param B_dev: device pointer to matrix B
 * @param nx: number of rows in matrix A and matrix B
 * @param ny: number of columns in matrix A and matrix B
 * @param dtype: datatype of matrix A and matrix B
 */
void mma2DGPUPy(void* dest_dev, void* A_dev, void* B_dev, int nx, int ny, datatype dtype) {
    if (dtype == datatype::FLOAT) {
        auto A_dev_ = static_cast<float*>(A_dev);
        auto B_dev_ = static_cast<float*>(B_dev);
        auto dest_dev_ = static_cast<float*>(dest_dev);
        mma2DGPUC<float>(dest_dev_, A_dev_, B_dev_, nx, ny);
        return;
    }
    if (dtype == datatype::DOUBLE) {
        auto A_dev_ = static_cast<double*>(A_dev);
        auto B_dev_ = static_cast<double*>(B_dev);
        auto dest_dev_ = static_cast<double*>(dest_dev);
        mma2DGPUC<double>(dest_dev_, A_dev_, B_dev_, nx, ny);
        return;
    }
    if (dtype == datatype::INTV_FLOAT) {
        auto A_dev_ = static_cast<Interval<float>*>(A_dev);
        auto B_dev_ = static_cast<Interval<float>*>(B_dev);
        auto dest_dev_ = static_cast<Interval<float>*>(dest_dev);
        mma2DGPUC<Interval<float>>(dest_dev_, A_dev_, B_dev_, nx, ny);
        return;
    }
    if (dtype == datatype::INTV_DOUBLE) {
        auto A_dev_ = static_cast<Interval<double>*>(A_dev);
        auto B_dev_ = static_cast<Interval<double>*>(B_dev);
        auto dest_dev_ = static_cast<Interval<double>*>(dest_dev);
        mma2DGPUC<Interval<double>>(dest_dev_, A_dev_, B_dev_, nx, ny);
        return;
    }
}
}



#endif //CUTLASSTEST_MMA_CUH
