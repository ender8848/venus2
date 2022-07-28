//
// Created by hao on 08/07/22.
//

#ifndef CUTLASSTEST_GEMMGPUPYTEST_CUH
#define CUTLASSTEST_GEMMGPUPYTEST_CUH

#include "gemmGPUCTest.cuh"
#include "../src/gemmGPU.cuh"

/// a wrapped system test

template<typename T>
void canCallGemmGPUC() {
    printf("Testing type -- %s\n", typeid(T).name());
    int datatype;
    if (typeid(T) == typeid(float)) {
        datatype = datatype::FLOAT;
    } else if (typeid(T) == typeid(double)) {
        datatype = datatype::DOUBLE;
    } else if (typeid(T) == typeid(Interval<float>)) {
        datatype = datatype::INTV_FLOAT;
    } else if (typeid(T) == typeid(Interval<double>)) {
        datatype = datatype::INTV_DOUBLE;
    } else {
        printf("Error: unsupported datatype\n");
        exit(1);
    }

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
    gemmGPUPy(A_dev, B_dev, C_dev, M, N, K, datatype, C_dev);

    printf("equals to:\n");
    printMatrix<T><<<1, 1>>>(C_dev, M, N);
    cudaDeviceSynchronize();
    printf("\n");
}



#endif //CUTLASSTEST_GEMMGPUPYTEST_CUH
