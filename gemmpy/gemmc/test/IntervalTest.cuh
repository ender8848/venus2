//
// Created by hao on 06/07/22.
//

#include "../src/Interval.cuh"
#include "../../../../../usr/include/c++/11/iostream"
#include "../../../../../usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h"
#include "../../../../../usr/include/stdio.h"
#include "../../../../../usr/include/c++/11/cassert"

template<typename T>
__global__ void mul(Interval<T>* dest_dev, Interval<T>* rhs_dev, bool print = false) {
    int i = threadIdx.x;
    if (i > 0 ) return;
    dest_dev[i] *= rhs_dev[i];
    if (print) {
        printf("%.16f %.16f\n", dest_dev[0].lower, dest_dev[0].upper);
    }
}

__global__ void hasSaferMultiplication() {
    printf("safe multiplication test:\n");
    Interval<float> f1(5., 5.);
    Interval<float> f2(1.-0.0000001, 1+0.0000001);
    float upper = 5.;
    float lower = 5.;
    for (int i = 0; i < 100000; ++i) {
        f1 *= f2;
        upper *= f2.upper;
        lower *= f2.lower;
    }
    assert(f1.lower < lower);
    assert(f1.upper > upper);
    printf("Safe Interval Arithmetic Bounds: [");
    printf("%.8f %.8f]\n", f1.lower, f1.upper);
    printf("Normal Float  Arithmetic Bounds: [");
    printf("%.8f %.8f]\n", lower, upper);
    printf("\n");
}

__global__ void hasSaferAddition() {
    printf("safe addition test:\n");
    float upper = 6.;
    float lower = 5.;
    Interval<float> f1(lower, upper);
    for (int i = 0; i < 1000; ++i) {
        f1 += static_cast<float>(i/10.);
        f1 += static_cast<float>(-i/10.);
    }

    assert(f1.lower < lower);
    assert(f1.upper > upper);
    printf("Safe Interval Arithmetic Bounds: [");
    printf("%.8f %.8f]\n", f1.lower, f1.upper);
    printf("Normal Float  Arithmetic Bounds: [");
    printf("%.8f %.8f]\n", lower, upper);
    printf("\n");
}




