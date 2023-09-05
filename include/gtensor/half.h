#ifndef GTENSOR_HALF_H
#define GTENSOR_HALF_H

#include <iostream>

#if __has_include(<cuda_fp16.h>)
#include <cuda_fp16.h>
#define GTENSOR_FP16_CUDA_HEADER
#elif CXX_FLAG_FLOAT16 // TODO check if other fp16 type available, e.g., _Float16
#define GTENSOR_FLOAT16_TYPE_AVAILABLE
#else
#error "GTENSOR_ENABLE_FP16=ON, but no 16-bit FP type available!"
#endif

#include <gtensor/macros.h>

namespace gt
{

// ======================================================================
// half

#if defined(GTENSOR_FP16_CUDA_HEADER)
using storage_type = __half;
#elif defined(GTENSOR_FLOAT16_TYPE_AVAILABLE)
using storage_type = _Float16;
#else
#error "GTENSOR_ENABLE_FP16=ON, but no 16-bit FP type available!"
#endif

#if defined(GTENSOR_FP16_CUDA_HEADER) \
    && defined(__CUDA_ARCH__) \
    && (__CUDA_ARCH__ >= 530)
using compute_type = __half;
#elif defined(GTENSOR_FLOAT16_TYPE_AVAILABLE)
using compute_type = _Float16;
#else
using compute_type = float;
#endif

class half
{
public:
    half() = default;

#if defined(GTENSOR_FLOAT16_TYPE_AVAILABLE)
    GT_INLINE half(int x) : x(static_cast<storage_type>(x)) {};
    GT_INLINE half(double x) : x(static_cast<storage_type>(x)) {};
#endif

    GT_INLINE half(float x) : x(static_cast<storage_type>(x)) {};
    GT_INLINE half(storage_type x) : x(x) {};

    GT_INLINE const half& operator=(const float f) { x = static_cast<storage_type>(f); return *this; }
    GT_INLINE compute_type Get() const { return static_cast<compute_type>(x); }
private:
    storage_type x;
};

#define PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR(op) \
    GT_INLINE half operator op(const half& lhs, const half& rhs) \
    { return half( lhs.Get() op rhs.Get() ); }

#define PROVIDE_HALF_BINARY_UPDATE_OPERATOR(op) \
    GT_INLINE half& operator op(half& lhs, const half& rhs) \
    { auto tmp = lhs.Get(); tmp op rhs.Get(); lhs = tmp; return lhs; }

#define PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(op, fp_type) \
    \
    GT_INLINE fp_type operator op(const half& lhs, const fp_type& rhs) \
    { return static_cast<fp_type>(lhs.Get()) op rhs; } \
    \
    GT_INLINE fp_type operator op(const fp_type& lhs, const half& rhs) \
    { return lhs op static_cast<fp_type>(rhs.Get()); }

PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR(+);
PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR(-);
PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR(*);
PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR(/);

PROVIDE_HALF_BINARY_UPDATE_OPERATOR(+=);
PROVIDE_HALF_BINARY_UPDATE_OPERATOR(-=);
PROVIDE_HALF_BINARY_UPDATE_OPERATOR(*=);
PROVIDE_HALF_BINARY_UPDATE_OPERATOR(/=);

PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(+, float);
PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(-, float);
PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(*, float);
PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(/, float);

PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(+, double);
PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(-, double);
PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(*, double);
PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(/, double);

#define PROVIDE_HALF_COMPARISON_OPERATOR(op) \
    GT_INLINE bool operator op(const half& lhs, const half& rhs) \
    { return lhs.Get() op rhs.Get(); }

#define PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(op, fp_type) \
    \
    GT_INLINE bool operator op(const half& lhs, const fp_type& rhs) \
    { return static_cast<fp_type>(lhs.Get()) op rhs; } \
    \
    GT_INLINE bool operator op(const fp_type& lhs, const half& rhs) \
    { return lhs op static_cast<fp_type>(rhs.Get()); }

PROVIDE_HALF_COMPARISON_OPERATOR(==);
PROVIDE_HALF_COMPARISON_OPERATOR(!=);
PROVIDE_HALF_COMPARISON_OPERATOR(<);
PROVIDE_HALF_COMPARISON_OPERATOR(<=);
PROVIDE_HALF_COMPARISON_OPERATOR(>);
PROVIDE_HALF_COMPARISON_OPERATOR(>=);

PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(==, float);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(!=, float);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(<, float);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(<=, float);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(>, float);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(>=, float);

PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(==, double);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(!=, double);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(<, double);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(<=, double);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(>, double);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(>=, double);

std::ostream& operator<<(std::ostream& s, const half& h)
{ s << static_cast<float>(h.Get()); return s; }

} // namespace gt

#undef PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR
#undef PROVIDE_HALF_BINARY_UPDATE_OPERATOR
#undef PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR
#undef PROVIDE_HALF_COMPARISON_OPERATOR
#undef PROVIDE_MIXED_HALF_COMPARISON_OPERATOR

#endif
