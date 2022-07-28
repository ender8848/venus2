//
// Created by hao on 07/07/22.
//

#ifndef CUTLASSTEST_GEMMGPU_CUH
#define CUTLASSTEST_GEMMGPU_CUH
#include "../cutlass/numeric_types.h"
#include "../cutlass/gemm/device/gemm.h"
#include "../cutlass/util/host_tensor.h"
#include "Interval.cuh"
#include "util.cuh"
#include "datatype.cuh"

/*
 * C interface to do gemm on GPU
 * Performs dest_dev = A_dev @ B_dev + bias_dev
 * Normally bias_dev is all 0
 * @param A_dev: pointer to the device memory of matrix A
 * @param B_dev: pointer to the device memory of matrix B
 * @param dest_dev: pointer to the device memory of matrix dest, which is the output matrix
 * @param M: number of rows of A and dest
 * @param N: number of columns of B and dest
 * @param K: number of columns of A and rows of B
 * @param bias_dev: pointer to the device memory of bias matrix
 */
template<typename T>
void gemmGPUCUsingGPUPtr(T* A_dev, T* B_dev, T* dest_dev, int M, int N, int K, T* bias_dev) {
    using Gemm = cutlass::gemm::device::Gemm<
            T,                                    // ElementA, namely type of A
            cutlass::layout::RowMajor,            // LayoutA, column major means colum of A is contiguous in memory
            T,                                    // ElementB
            cutlass::layout::RowMajor,            // LayoutB
            T,                                    // ElementOutput
            cutlass::layout::RowMajor,            // LayoutOutput
            T,                                    // ElementAccumulator
            cutlass::arch::OpClassSimt,           // tag indicating GPU opcode class, architecture-dependent
            cutlass::arch::Sm61                   // tag indicating GPU compute compatability, architecture-dependent
    >;
    Gemm gemm;
    cutlass::Status status;

    T alpha = T(1.);    // Define alpha and beta, this controls dest = alpha * A @ B + beta * bias
    T beta = T(1.);     // use 1 here to get dest = A @ B + bias
    int lda = K;        // leading dimension of A, namely the number of cols of A
    int ldb = N;        // leading dimension of B, namely the number of cols of B
    int ld_dest = N;    // leading dimension of dest, namely the number of cols of dest
    int ld_bias = N;    // leading dimension of bias, namely the number of cols of bias

    status = gemm({
        {M,        N, K},
        {A_dev,    lda},            // TensorRef to A device tensor
        {B_dev,    ldb},            // TensorRef to B device tensor
        {bias_dev, ld_dest},        // TensorRef to C device tensor
        {dest_dev, ld_bias},        // TensorRef to D device tensor - may be the same as C (depending on passed value)
        {alpha,    beta}            // epilogue operation arguments
    });

    if (status != cutlass::Status::kSuccess) {
        printf("GEMM failed\n");
        printf("status: %d\n", static_cast<int>(status));
        printf("\n");
    }
}

/// note that this API is provided but not used.
/// This is left for future extensibility in case venus is switched to C++
/*
 * C interface to do gemm on GPU, but receives host tensor as input
 * Performs dest_host = A_host @ B_host + bias_host
 * Normally bias_host is all 0
 * This function is just an adapter which create device tensor from host tensor and call gemmGPUCUsingGPUPtr
 * @param A_host: pointer to the host memory of matrix A
 * @param B_host: pointer to the host memory of matrix B
 * @param dest_host: pointer to the host memory of matrix dest, which is the output matrix
 * @param M: number of rows of A and dest
 * @param N: number of columns of B and dest
 * @param K: number of columns of A and rows of B
 * @param bias_host: pointer to the host memory of bias matrix, default to NULL
 */
template<typename T>
void gemmGPUCUsingCPUPtr(T* A_host, T* B_host, T* dest_host, int M, int N, int K, T* bias_host) {
    // create device tensor in GPU with same size as host tensor
    T * A_dev = nullptr;
    T * B_dev = nullptr;
    T * dest_dev = nullptr;
    T * bias_dev = nullptr;
    cudaMalloc((void**)&A_dev, sizeof(T)*M*K);
    cudaMalloc((void**)&B_dev, sizeof(T)*K*N);
    cudaMalloc((void**)&dest_dev, sizeof(T)*M*N);
    if (bias_host == dest_host) bias_dev = dest_dev;
    else cudaMalloc((void**)&bias_dev, sizeof(T)*M*N);
    // copy host tensor to device tensor
    cudaMemcpy(A_dev, A_host, sizeof(T)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host, sizeof(T)*K*N, cudaMemcpyHostToDevice);

    // init dest tensor to 0
    cudaMemset(dest_dev, 0, sizeof(T)*M*N);
    if (bias_host != dest_host) cudaMemcpy(bias_dev, bias_host, sizeof(T)*M*N, cudaMemcpyHostToDevice);
    
    // delegate to gemmGPUCUsingGPUPtr
    gemmGPUCUsingGPUPtr(A_dev, B_dev, dest_dev, M, N, K, bias_dev);

    // copy dest tensor back to host tensor
    cudaMemcpy(dest_host, dest_dev, sizeof(T)*M*N, cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(dest_dev);
    if (bias_host != dest_host) cudaFree(bias_dev);
}


extern "C" {
/*
* API that is used by Python to do gemm
* dest_host = A @ B + bias if bias is not NULL
* dest_host = A @ B + dest if bias is NULL, hence make sure in this case dest is all 0
* @param A: pointer to the memory of matrix A, can be host or device
* @param B: pointer to the memory of matrix B, can be host or device
* @param dest: pointer to the memory of matrix dest, can be host or device
* @param M: number of rows of A and dest
* @param N: number of columns of B and dest
* @param K: number of columns of A and rows of B
* @param datatype: datatype of A, B, dest, and bias, this is a protocol that python code should also follow
* @param bias: pointer to the memory of bias matrix, can be host or device, default to NULL
* @param is_host: whether A, B, dest, and bias are host or device, default to true
*/
void gemmGPUPy(void* A, void* B, void* dest, int M, int N, int K, int datatype, void* bias = nullptr, bool is_host = true) {
    if (bias == nullptr) bias = dest;
    if (is_host && datatype == datatype::FLOAT) {
        gemmGPUCUsingCPUPtr<>((float*)A,(float*)B,(float*)dest,
                              M, N, K,(float*)bias);
        return;
    }
    if (is_host && datatype == datatype::DOUBLE) {
        gemmGPUCUsingCPUPtr<>((double*)A, (double*)B, (double*)dest,
                              M, N, K, (double*)bias);
        return;
    }
    if (is_host && datatype == datatype::INTV_FLOAT) {
        gemmGPUCUsingCPUPtr<>((Interval<float>*)A, (Interval<float>*)B, (Interval<float>*)dest,
                              M, N, K, (Interval<float>*)bias);
        return;
    }
    if (is_host && datatype == datatype::INTV_DOUBLE) {
        gemmGPUCUsingCPUPtr<>((Interval<double>*)A, (Interval<double>*)B, (Interval<double>*)dest,
                              M, N, K, (Interval<double>*)bias);
        return;
    }
    if (!is_host && datatype == datatype::FLOAT) {
        gemmGPUCUsingGPUPtr<>((float*)A, (float*)B, (float*)dest,
                              M, N, K, (float*)bias);
        return;
    }
    if (!is_host && datatype == datatype::DOUBLE) {
        gemmGPUCUsingGPUPtr<>((double*)A, (double*)B, (double*)dest,
                              M, N, K, (double*)bias);
        return;
    }
    if (!is_host && datatype == datatype::INTV_FLOAT) {
        gemmGPUCUsingGPUPtr<>((Interval<float>*)A, (Interval<float>*)B, (Interval<float>*)dest,
                              M, N, K, (Interval<float>*)bias);
        return;
    }
    if (!is_host && datatype == datatype::INTV_DOUBLE) {
        gemmGPUCUsingGPUPtr<>((Interval<double>*)A, (Interval<double>*)B, (Interval<double>*)dest,
                              M, N, K, (Interval<double>*)bias);
        return;
    }
}
}

#endif //CUTLASSTEST_GEMMGPU_CUH
