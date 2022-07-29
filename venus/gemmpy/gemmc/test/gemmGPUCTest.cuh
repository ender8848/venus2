//
// Created by hao on 07/07/22.
//

#ifndef CUTLASSTEST_GEMMGPUCTEST_CUH
#define CUTLASSTEST_GEMMGPUCTEST_CUH

#include "../src/gemmGPU.cuh"
#include "../cutlass/numeric_types.h"
#include "../cutlass/gemm/device/gemm.h"
#include "../cutlass/util/host_tensor.h"
#include "../src/Interval.cuh"
#include "../src/util.cuh"


/*
 * Test the calculation core -- gemmGPUCUsingGPUPtr
 * Results are also printed
 */
template<typename T>
void GemmGPUCalculatesCorrectly() {
    printf("Testing type -- %s\n", typeid(T).name());

    // Define the problem size and params
    int M = 2;
    int N = 4;
    int K = 4;

    // Allocate device memory
    cutlass::HostTensor<T, cutlass::layout::RowMajor> A({M, K});
    cutlass::HostTensor<T, cutlass::layout::RowMajor> B({K, N});
    cutlass::HostTensor<T, cutlass::layout::RowMajor> C({M, N});

    T *A_dev = A.device_data();
    T *B_dev = B.device_data();
    T *C_dev = C.device_data(); // use C_dev as D_dev

    // init will make dest zero, which is same as pyTorch init
    float a = 1.f;
    float b = 2.f;

    // inefficient initialization, but fine for testing
    initMatrix<T><<<1, 1>>>(A_dev, M, K, {a});
    cudaDeviceSynchronize();
    printMatrix<T><<<1, 1>>>(A_dev, M, K);
    cudaDeviceSynchronize();
    printf("times...\n");
    initMatrix<T><<<1, 1>>>(B_dev, K, N, b);
    cudaDeviceSynchronize();
    printMatrix<T><<<1, 1>>>(B_dev, K, N);
    cudaDeviceSynchronize();

    // test the API
    gemmGPUCUsingGPUPtr<T>(A_dev, B_dev, C_dev, M, N, K, C_dev);

    printf("equals to:\n");
    printMatrix<T><<<1, 1>>>(C_dev, M, N);
    cudaDeviceSynchronize();
    printf("\n");
}

/*
 * Test the calculation core -- gemmGPUCUsingCPUPtr
 * since that is only an adapter of gemmGPUCUsingGPUPtr, results are not printed
 */
template<typename T>
void GemmCPUCalculatesCorrectly() {
    // Define the problem size and params
    int M = 2;
    int N = 4;
    int K = 4;

    // Allocate device memory
    cutlass::HostTensor<T, cutlass::layout::RowMajor> A({M, K});
    cutlass::HostTensor<T, cutlass::layout::RowMajor> B({K, N});
    cutlass::HostTensor<T, cutlass::layout::RowMajor> C({M, N});

    // test the API
    gemmGPUCUsingCPUPtr<T>(A.host_data(), B.host_data(), C.host_data(),
                           M, N, K, C.host_data());
}


/*
 * consists of two tests
 * Results for gemmGPUCUsingGPUPtr (which is calculation core) is printed
 */
template<typename T>
void GemmCalculatesCorrectly() {
    GemmGPUCalculatesCorrectly<T>();
    GemmCPUCalculatesCorrectly<T>();
}


#endif //CUTLASSTEST_GEMMGPUCTEST_CUH
