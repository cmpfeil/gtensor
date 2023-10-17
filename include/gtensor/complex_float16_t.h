#ifndef GTENSOR_COMPLEX_FLOAT16T_H
#define GTENSOR_COMPLEX_FLOAT16T_H

#include "complex.h"
#include "float16_t.h"
#include <cmath>
#include <iostream>

#if __has_include(<cuda_fp16.h>)
#include <cuda_fp16.h>
#define GTENSOR_FP16_CUDA_HEADER
#elif 0 // TODO check if other fp16 type available, e.g., _Float16
#else
#error "GTENSOR_ENABLE_FP16=ON, but no 16-bit FP type available!"
#endif

namespace gt
{

// ======================================================================
// complex_float16_t
// ... adapted from the C++ <complex>, see e.g.,
// header https://en.cppreference.com/w/cpp/header/complex [2023/10/17]

class complex_float16_t
{
public:
  typedef float16_t value_type;
  GT_INLINE complex_float16_t(const float16_t& re = float16_t(),
                              const float16_t& im = float16_t())
  : _real(re), _imag(im) {}
  constexpr complex_float16_t(const complex_float16_t&) = default;
  template <class X>
  constexpr complex_float16_t(const complex<X>&);

  GT_INLINE float16_t real() const { return _real; }
  constexpr void real(float16_t);
  GT_INLINE float16_t imag() const { return _imag; }
  constexpr void imag(float16_t);

  constexpr complex_float16_t& operator=(const float16_t&);
  constexpr complex_float16_t& operator+=(const float16_t&);
  constexpr complex_float16_t& operator-=(const float16_t&);
  constexpr complex_float16_t& operator*=(const float16_t&);
  constexpr complex_float16_t& operator/=(const float16_t&);

  constexpr complex_float16_t& operator=(const complex_float16_t&);
  template <class X>
  constexpr complex_float16_t& operator=(const complex<X>&);
  template <class X>
  constexpr complex_float16_t& operator+=(const complex<X>&);
  template <class X>
  constexpr complex_float16_t& operator-=(const complex<X>&);
  template <class X>
  constexpr complex_float16_t& operator*=(const complex<X>&);
  template <class X>
  constexpr complex_float16_t& operator/=(const complex<X>&);

private:
  float16_t _real;
  float16_t _imag;
};

// operators:
constexpr complex_float16_t operator+(const complex_float16_t&,
                                      const complex_float16_t&);
constexpr complex_float16_t operator+(const complex_float16_t&,
                                      const float16_t&);
constexpr complex_float16_t operator+(const float16_t&,
                                      const complex_float16_t&);

constexpr complex_float16_t operator-(const complex_float16_t&,
                                      const complex_float16_t&);
constexpr complex_float16_t operator-(const complex_float16_t&,
                                      const float16_t&);
constexpr complex_float16_t operator-(const float16_t&,
                                      const complex_float16_t&);

constexpr complex_float16_t operator*(const complex_float16_t&,
                                      const complex_float16_t&);
constexpr complex_float16_t operator*(const complex_float16_t&,
                                      const float16_t&);
constexpr complex_float16_t operator*(const float16_t&,
                                      const complex_float16_t&);

constexpr complex_float16_t operator/(const complex_float16_t&,
                                      const complex_float16_t&);
constexpr complex_float16_t operator/(const complex_float16_t&,
                                      const float16_t&);
constexpr complex_float16_t operator/(const float16_t&,
                                      const complex_float16_t&);

constexpr complex_float16_t operator+(const complex_float16_t&);
constexpr complex_float16_t operator-(const complex_float16_t&);

GT_INLINE bool operator==(const complex_float16_t& lhs, const complex_float16_t& rhs)
{
    return lhs.real() == rhs.real() && lhs.imag() == rhs.imag();
}
GT_INLINE bool operator==(const complex_float16_t& lhs, const float16_t& rhs)
{
    return lhs.real() == rhs && lhs.imag() == 0;
}
GT_INLINE bool operator==(const float16_t& lhs, const complex_float16_t& rhs)
{
    return lhs == rhs.real() && 0 == rhs.imag();
}

GT_INLINE bool operator!=(const complex_float16_t& lhs, const complex_float16_t& rhs)
{
    return lhs.real() != rhs.real() || lhs.imag() != rhs.imag();
}
GT_INLINE bool operator!=(const complex_float16_t& lhs, const float16_t& rhs)
{
    return lhs.real() != rhs || lhs.imag() != 0;
}
GT_INLINE bool operator!=(const float16_t& lhs, const complex_float16_t& rhs)
{
    return lhs != rhs.real() || 0 != rhs.imag();
}

template <class CharT, class Traits>
std::basic_istream<CharT, Traits>& operator>>(std::basic_istream<CharT, Traits>&,
                                         complex_float16_t&);

template <class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& s,
                                         const complex_float16_t& z)
{ return s << "TODO LATER"; }

// values:
constexpr float16_t real(const complex_float16_t&);
constexpr float16_t imag(const complex_float16_t&);

float16_t abs(const complex_float16_t&);
float16_t arg(const complex_float16_t&);
constexpr float16_t norm(const complex_float16_t&);

constexpr complex_float16_t conj(const complex_float16_t&);
complex_float16_t proj(const complex_float16_t&);
complex_float16_t polar(const float16_t&, const float16_t& = 0);

// transcendentals:
complex_float16_t acos(const complex_float16_t&);
complex_float16_t asin(const complex_float16_t&);
complex_float16_t atan(const complex_float16_t&);

complex_float16_t acosh(const complex_float16_t&);
complex_float16_t asinh(const complex_float16_t&);
complex_float16_t atanh(const complex_float16_t&);

complex_float16_t cos(const complex_float16_t&);
complex_float16_t cosh(const complex_float16_t&);
complex_float16_t exp(const complex_float16_t&);
complex_float16_t log(const complex_float16_t&);
complex_float16_t log10(const complex_float16_t&);

complex_float16_t pow(const complex_float16_t&, const float16_t&);
complex_float16_t pow(const complex_float16_t&, const complex_float16_t&);
complex_float16_t pow(const float16_t&, const complex_float16_t&);

complex_float16_t sin(const complex_float16_t&);
complex_float16_t sinh(const complex_float16_t&);
complex_float16_t sqrt(const complex_float16_t&);
complex_float16_t tan(const complex_float16_t&);
complex_float16_t tanh(const complex_float16_t&);

// complex literals:
inline namespace literals
{
inline namespace complex_literals
{
constexpr complex_float16_t operator""_ih(long double);
constexpr complex_float16_t operator""_ih(unsigned long long);
} // namespace complex_literals
} // namespace literals

} // namespace gt

#endif // GTENSOR_COMPLEX_FLOAT16T_H
