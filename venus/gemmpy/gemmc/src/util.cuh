//
// Created by hao on 17/07/22.
//

#ifndef CUTLASSTEST_UTIL_CUH
#define CUTLASSTEST_UTIL_CUH
#include "Interval.cuh"

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

template<typename T>
__global__ void printMatrix(T* M_dev, int row, int col);

template<>
__global__ void printMatrix(float * M_dev, int row, int col) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            printf("%.2f ", M_dev[i*col + j]);
        }
        printf("\n");
    }
}


template<>
__global__ void printMatrix(double * M_dev, int row, int col) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            printf("%.2f ", M_dev[i*col + j]);
        }
        printf("\n");
    }
}

template<>
__global__ void printMatrix(Interval<float>* M_dev, int row, int col) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            printf("(%.2f, %.2f) ", M_dev[i*col + j].lower, M_dev[i*col + j].upper);
        }
        printf("\n");
    }
}

template<>
__global__ void printMatrix(Interval<double>* M_dev, int row, int col) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            printf("(%.2f, %.2f) ", M_dev[i*col + j].lower, M_dev[i*col + j].upper);
        }
        printf("\n");
    }
}

template<typename T>
__global__ void initMatrix(T* M_dev, int row, int col, T val) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            M_dev[i*col + j] = val;
        }
    }
}





#endif //CUTLASSTEST_UTIL_CUH
