//
// Created by hao on 23/07/22.
//

#ifndef CUTLASSTEST_MMAGPUCTEST_CUH
#define CUTLASSTEST_MMAGPUCTEST_CUH

#include "../src/mma.cuh"
#include "../cutlass/numeric_types.h"
#include "../cutlass/gemm/device/gemm.h"
#include "../cutlass/util/host_tensor.h"
#include "../src/Interval.cuh"
#include "../src/util.cuh"


/*
 * Test the calculation core -- mmaGPUC
 * Results are also printed
 */
template<typename T>
void mmaGPUCalculatesCorrectly() {
    printf("Testing type -- %s\n", typeid(T).name());

    // Define the problem size and params
    int M = 4;
    int N = 6;

    // Allocate device tensor
    cutlass::HostTensor<T, cutlass::layout::RowMajor> A({M, N});
    cutlass::HostTensor<T, cutlass::layout::RowMajor> B({M, N});
    cutlass::HostTensor<T, cutlass::layout::RowMajor> dest({M, N});

    T *A_dev = A.device_data();
    T *B_dev = B.device_data();
    T *dest_dev = dest.device_data(); // use dest_dev as D_dev

    // init will make dest zero, which is same as pyTorch init
    float a = 1.f;
    float b = 2.f;

    // inefficient initialization, but fine for testing
    initMatrix<T><<<1, 1>>>(A_dev, M, N, {a});
    cudaDeviceSynchronize();
    printMatrix<T><<<1, 1>>>(A_dev, M, N);
    cudaDeviceSynchronize();
    printf("add...\n");
    initMatrix<T><<<1, 1>>>(B_dev, M, N, b);
    cudaDeviceSynchronize();
    printMatrix<T><<<1, 1>>>(B_dev, M, N);
    cudaDeviceSynchronize();

    // test the API
    mma2DGPUC<T>(dest_dev, A_dev, B_dev, M, N);

    printf("equals to:\n");
    printMatrix<T><<<1, 1>>>(dest_dev, M, N);
    cudaDeviceSynchronize();
    printf("\n");
}




#endif //CUTLASSTEST_MMAGPUCTEST_CUH
