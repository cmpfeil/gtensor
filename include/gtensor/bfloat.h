#ifndef GTENSOR_BFLOAT_H
#define GTENSOR_BFLOAT_H

#include <iostream>
#include <cuda_bf16.h>

namespace gt
{

// ======================================================================
// bfloat
#ifdef __CUDA_ARCH__
#define TARGET_ARCH __host__ __device__
#else
#define TARGET_ARCH
#endif

#if (__CUDA_ARCH__ >= 800)
using compute_type = __nv_bfloat16;
#else
using compute_type = float;
#endif

class bfloat
{
public:
    bfloat() = default;
    TARGET_ARCH bfloat(float x) : x(x) {};
    TARGET_ARCH bfloat(__nv_bfloat16 x) : x(x) {};

    TARGET_ARCH const bfloat& operator=(const float f) { x = f; return *this; }
    TARGET_ARCH compute_type Get() const { return static_cast<compute_type>(x); }
private:
    __nv_bfloat16 x;
};

#define PROVIDE_BFLOAT_BINARY_ARITHMETIC_OPERATOR(op) \
    TARGET_ARCH bfloat operator op(const bfloat& lhs, const bfloat& rhs) \
    { return bfloat( lhs.Get() op rhs.Get() ); }

#define PROVIDE_MIXED_BFLOAT_BINARY_ARITHMETIC_OPERATOR(op, fp_type) \
    \
    TARGET_ARCH fp_type operator op(const bfloat& lhs, const fp_type& rhs) \
    { return static_cast<fp_type>(lhs.Get()) op rhs; } \
    \
    TARGET_ARCH fp_type operator op(const fp_type& lhs, const bfloat& rhs) \
    { return lhs op static_cast<fp_type>(rhs.Get()); }

PROVIDE_BFLOAT_BINARY_ARITHMETIC_OPERATOR(+);
PROVIDE_BFLOAT_BINARY_ARITHMETIC_OPERATOR(-);
PROVIDE_BFLOAT_BINARY_ARITHMETIC_OPERATOR(*);
PROVIDE_BFLOAT_BINARY_ARITHMETIC_OPERATOR(/);

PROVIDE_MIXED_BFLOAT_BINARY_ARITHMETIC_OPERATOR(+, float);
PROVIDE_MIXED_BFLOAT_BINARY_ARITHMETIC_OPERATOR(-, float);
PROVIDE_MIXED_BFLOAT_BINARY_ARITHMETIC_OPERATOR(*, float);
PROVIDE_MIXED_BFLOAT_BINARY_ARITHMETIC_OPERATOR(/, float);

PROVIDE_MIXED_BFLOAT_BINARY_ARITHMETIC_OPERATOR(+, double);
PROVIDE_MIXED_BFLOAT_BINARY_ARITHMETIC_OPERATOR(-, double);
PROVIDE_MIXED_BFLOAT_BINARY_ARITHMETIC_OPERATOR(*, double);
PROVIDE_MIXED_BFLOAT_BINARY_ARITHMETIC_OPERATOR(/, double);

#define PROVIDE_BFLOAT_COMPARISON_OPERATOR(op) \
    TARGET_ARCH bool operator op(const bfloat& lhs, const bfloat& rhs) \
    { return lhs.Get() op rhs.Get(); }

#define PROVIDE_MIXED_BFLOAT_COMPARISON_OPERATOR(op, fp_type) \
    \
    TARGET_ARCH bool operator op(const bfloat& lhs, const fp_type& rhs) \
    { return static_cast<fp_type>(lhs.Get()) op rhs; } \
    \
    TARGET_ARCH bool operator op(const fp_type& lhs, const bfloat& rhs) \
    { return lhs op static_cast<fp_type>(rhs.Get()); }

PROVIDE_BFLOAT_COMPARISON_OPERATOR(==);
PROVIDE_BFLOAT_COMPARISON_OPERATOR(!=);
PROVIDE_BFLOAT_COMPARISON_OPERATOR(<);
PROVIDE_BFLOAT_COMPARISON_OPERATOR(<=);
PROVIDE_BFLOAT_COMPARISON_OPERATOR(>);
PROVIDE_BFLOAT_COMPARISON_OPERATOR(>=);

PROVIDE_MIXED_BFLOAT_COMPARISON_OPERATOR(==, float);
PROVIDE_MIXED_BFLOAT_COMPARISON_OPERATOR(!=, float);
PROVIDE_MIXED_BFLOAT_COMPARISON_OPERATOR(<, float);
PROVIDE_MIXED_BFLOAT_COMPARISON_OPERATOR(<=, float);
PROVIDE_MIXED_BFLOAT_COMPARISON_OPERATOR(>, float);
PROVIDE_MIXED_BFLOAT_COMPARISON_OPERATOR(>=, float);

PROVIDE_MIXED_BFLOAT_COMPARISON_OPERATOR(==, double);
PROVIDE_MIXED_BFLOAT_COMPARISON_OPERATOR(!=, double);
PROVIDE_MIXED_BFLOAT_COMPARISON_OPERATOR(<, double);
PROVIDE_MIXED_BFLOAT_COMPARISON_OPERATOR(<=, double);
PROVIDE_MIXED_BFLOAT_COMPARISON_OPERATOR(>, double);
PROVIDE_MIXED_BFLOAT_COMPARISON_OPERATOR(>=, double);

std::ostream& operator<<(std::ostream& s, const bfloat& h)
{ s << static_cast<float>(h.Get()); return s; }

#undef TARGET_ARCH
#undef PROVIDE_BFLOAT_BINARY_ARITHMETIC_OPERATOR
#undef PROVIDE_MIXED_BFLOAT_BINARY_ARITHMETIC_OPERATOR
#undef PROVIDE_BFLOAT_COMPARISON_OPERATOR
#undef PROVIDE_MIXED_BFLOAT_COMPARISON_OPERATOR

} // namespace gt

#endif
