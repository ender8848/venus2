//
// Created by hao on 06/07/22.
//

#ifndef CUTLASSTEST_INTERVAL_CUH
#define CUTLASSTEST_INTERVAL_CUH

#include <cuda.h>

/*
 * Interval class
 * - lower: lower bound of the interval
 * - upper: upper bound of the interval
 * Interval ensures the floating point soundness during calculation
 * Interval contains basic operator overloading so that Interval works like a real-valued data type
 * operator overloading for * and + use template-based simulated dynamic binding instead of inheritance
 * for better run-time performance
 */
template<typename T>
struct Interval {
    T lower = 0.; /// lower bound

    T upper = 0.; /// upper bound

    /// constructors
    __device__ __host__ Interval() = default;
    __device__ __host__ Interval(const T &lower, const T &upper) : lower(lower), upper(upper) {}
    __device__ __host__ Interval(const T &num) : lower(num), upper(num) {} // cannot be marked as explicit!

    /// Multiplication with an interval, need different implementation for different data types
    __device__ Interval<T> operator*(const Interval<T> &rhs) const;

    /// Multiplication with a number, delegate to the operator*(const Interval<T> &)
    __device__ Interval<T> operator*(const T &rhs) const {
        return *this * Interval<T>(rhs);
    }

    /// Inplace multiplication with an interval, delegate to operator *
    __device__ Interval<T> &operator*=(const Interval<T> &rhs) {
        *this = *this * rhs;
        return *this;
    }

    /// Inplace multiplication with a number, delegate to operator *=
    __device__ Interval<T> &operator*=(const T &rhs) {
        *this = *this * rhs;
        return *this;
    }

    /// Addition with an interval, need different implementation for different data types
    __device__ Interval<T> operator+(const Interval<T> &rhs) const;

    /// Addition with a number, delegate to the operator+(const Interval<T> &)
    __device__ Interval<T> operator+(const T &rhs) const {
        return *this + Interval<T>(rhs);
    }

    /// Inplace addition with an interval, delegate to operator +
    __device__ Interval<T> &operator+=(const Interval<T> &rhs) {
        *this = *this + rhs;
        return *this;
    }
    /// Inplace addition with a number, delegate to operator +=
    __device__ Interval<T> &operator+=(const T &rhs) {
        *this = *this + rhs;
        return *this;
    }

    /// equivalent
    __device__ bool operator==(const Interval<T> &rhs) const {
        return (lower == rhs.lower) && (upper == rhs.upper);
    }

    /// different, delegate to operator ==
    __device__ bool operator!=(const Interval<T> &rhs) const {
        return !(*this == rhs);
    }

    /// assignment using an int
    __device__ __host__ Interval<T> &operator=(const int &rhs) {
        lower = rhs;
        upper = rhs;
        return *this;
    }

    /// assignment using a double
    __device__ __host__ Interval<T> &operator=(const double &src) {
        lower = src;
        upper = src;
        return *this;
    }

    /// assignment using a float
    __device__ __host__ Interval<T> &operator=(const float &src) {
        lower = src;
        upper = src;
        return *this;
    }

    /// deep assignment using an Interval
    __device__ __host__ Interval<T> &operator=(const Interval<T> &src) {
        lower = src.lower;
        upper = src.upper;
        return *this;
    }

    /// deep copy constructor, delegate to deep assignment
    __device__ __host__ Interval<T>(const Interval<T> &src) {
        *this = src;
    }

};

template<>
__device__ Interval<double> Interval<double>::operator*(const Interval<double> &rhs) const {
    return {min(min(__dmul_rd(this->lower, rhs.lower), __dmul_rd(this->lower, rhs.upper)),
                 min(__dmul_rd(this->upper, rhs.lower), __dmul_rd(this->upper, rhs.upper))),
            max(max(__dmul_ru(this->lower, rhs.lower), __dmul_ru(this->lower, rhs.upper)),
                 max(__dmul_ru(this->upper, rhs.lower), __dmul_ru(this->upper, rhs.upper)))};
}

template<>
__device__ Interval<float> Interval<float>::operator*(const Interval<float> &rhs) const {
    return {min(min(__fmul_rd(this->lower, rhs.lower), __fmul_rd(this->lower, rhs.upper)),
                  min(__fmul_rd(this->upper, rhs.lower), __fmul_rd(this->upper, rhs.upper))),
            max(max(__fmul_ru(this->lower, rhs.lower), __fmul_ru(this->lower, rhs.upper)),
                  max(__fmul_ru(this->upper, rhs.lower), __fmul_ru(this->upper, rhs.upper)))};
}

template<>
__device__ Interval<float> Interval<float>::operator+(const Interval<float> &rhs) const {
    return {__fadd_rd(this->lower, rhs.lower),
            __fadd_ru(this->upper, rhs.upper)};
}

template<>
__device__ Interval<double> Interval<double>::operator+(const Interval<double> &rhs) const {
    return {__dadd_rd(this->lower, rhs.lower),
            __dadd_ru(this->upper, rhs.upper)};
}

#endif //CUTLASSTEST_INTERVAL_CUH
