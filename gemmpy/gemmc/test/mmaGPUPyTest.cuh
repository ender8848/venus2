//
// Created by hao on 23/07/22.
//

#ifndef CUTLASSTEST_MMAGPUPYTEST_CUH
#define CUTLASSTEST_MMAGPUPYTEST_CUH

#include "../src/mma.cuh"

/// a wrapped system test

template<typename T>
void canCallmmaGPUC() {
    printf("Testing type -- %s\n", typeid(T).name());
    datatype dtype;
    if (typeid(T) == typeid(float)) {
        dtype = datatype::FLOAT;
    } else if (typeid(T) == typeid(double)) {
        dtype = datatype::DOUBLE;
    } else if (typeid(T) == typeid(Interval<float>)) {
        dtype = datatype::INTV_FLOAT;
    } else if (typeid(T) == typeid(Interval<double>)) {
        dtype = datatype::INTV_DOUBLE;
    } else {
        printf("Error: unsupported dtype\n");
        exit(1);
    }

    // Define the problem size and params
    int M = 2;
    int N = 6;

    // Allocate device memory
    cutlass::HostTensor<T, cutlass::layout::RowMajor> A({M, N});
    cutlass::HostTensor<T, cutlass::layout::RowMajor> B({M, N});
    cutlass::HostTensor<T, cutlass::layout::RowMajor> dest({M, N});

    T *A_dev = A.device_data();
    T *B_dev = B.device_data();
    T *dest_dev = dest.device_data();

    // init will make dest zero, which is same as pyTorch init
    float a = 1.f;
    float b = 2.f;

    initMatrix<T><<<1, 1>>>(A_dev, M, N, {a});
    cudaDeviceSynchronize();
    printMatrix<T><<<1, 1>>>(A_dev, M, N);
    cudaDeviceSynchronize();
    printf("times...\n");
    initMatrix<T><<<1, 1>>>(B_dev, M, N, b);
    cudaDeviceSynchronize();
    printMatrix<T><<<1, 1>>>(B_dev, M, N);
    cudaDeviceSynchronize();

    // test the API
    mma2DGPUPy(dest_dev, A_dev, B_dev, M, N, dtype);

    printf("equals to:\n");
    printMatrix<T><<<1, 1>>>(dest_dev, M, N);
    cudaDeviceSynchronize();
    printf("\n");
}


#endif //CUTLASSTEST_MMAGPUPYTEST_CUH
