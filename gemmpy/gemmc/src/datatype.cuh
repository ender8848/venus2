//
// Created by hao on 23/07/22.
//

#ifndef CUTLASSTEST_DATATYPE_CUH
#define CUTLASSTEST_DATATYPE_CUH

enum datatype {
    FLOAT = 0,
    DOUBLE = 1,
    INTV_FLOAT = 2,
    INTV_DOUBLE = 3
};

#include "../cutlass/numeric_types.h"
#include "../cutlass/gemm/device/gemm.h"
#include "../cutlass/util/host_tensor.h"
#include "Interval.cuh"
#include "util.cuh"

#endif //CUTLASSTEST_DATATYPE_CUH
