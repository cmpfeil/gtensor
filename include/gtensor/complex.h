
#ifndef GTENSOR_COMPLEX_H
#define GTENSOR_COMPLEX_H

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)
#include <thrust/complex.h>
#elif defined(GTENSOR_DEVICE_SYCL)
#define _SYCL_CPLX_NAMESPACE gt::sycl_cplx
#include "sycl_ext_complex.hpp"
#else
#include <complex>
#endif

#if defined(GTENSOR_ENABLE_FP16)
#include "gtensor/complex_float16_t.h"
#include "gtensor/float16_t.h"
#endif

#include "gtensor/device_ptr.h"
#include "gtensor/meta.h"

namespace gt
{

namespace detail
{

// NOTE: always use thrust complex for CUDA and HIP, regardless of storage
// backend. Depending on ROCm and CUDA verison, using std::complex could
// cause device versions of operators to not be defined properly.
#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)

template <typename T>
using classic_complex = thrust::complex<T>;

#elif defined(GTENSOR_DEVICE_SYCL)

// TODO: this will hopefully be standardized soon and be sycl::complex
template <typename T>
using classic_complex = gt::sycl_cplx::complex<T>;

#else // fallback to std::complex, e.g. for host backend

template <typename T>
using classic_complex = std::complex<T>;

#endif

template <typename T>
struct ComplexAliasTypedef
{
  typedef classic_complex<T> type; //TODO: T --> std::enable_if_t<std::is_floating_point_v<T>, T>
};

#if defined(GTENSOR_ENABLE_FP16)
template <>
struct ComplexAliasTypedef<gt::float16_t>
{
  typedef gt::complex_float16_t type;
};
#endif

} // namespace detail

template <typename T>
using complex = typename detail::ComplexAliasTypedef<T>::type;

// ======================================================================
// is_complex

template <typename T>
struct is_complex : public std::false_type
{};

template <typename T>
struct is_complex<detail::classic_complex<T>> : public std::true_type
{};

#if defined(GTENSOR_ENABLE_FP16)
template <>
struct is_complex<gt::complex_float16_t> : public std::true_type
{};
#endif

template <typename T>
constexpr bool is_complex_v = is_complex<T>::value;

// ======================================================================
// has_complex_value_type

template <typename C, typename = void>
struct has_complex_value_type : std::false_type
{};

template <typename C>
struct has_complex_value_type<
  C, gt::meta::void_t<std::enable_if_t<is_complex_v<typename C::value_type>>>>
  : std::true_type
{};

template <typename T>
constexpr bool has_complex_value_type_v = has_complex_value_type<T>::value;

// ======================================================================
// complex_subtype

template <typename T>
struct complex_subtype
{
  using type = T;
};

template <typename R>
struct complex_subtype<gt::detail::classic_complex<R>>
{
  using type = R;
};

#if defined(GTENSOR_ENABLE_FP16)
template <>
struct complex_subtype<gt::complex_float16_t>
{
  using type = gt::float16_t;
};
#endif

template <typename T>
using complex_subtype_t = typename complex_subtype<T>::type;

// ======================================================================
// container_complex_subtype

template <typename C>
struct container_complex_subtype
{
  using type = complex_subtype_t<typename C::value_type>;
};

template <typename T>
using container_complex_subtype_t = typename container_complex_subtype<T>::type;

#ifdef GTENSOR_DEVICE_SYCL

// oneMKL and possibly other APIs taking std::complex pointers do not work
// well with sycl::ext::cplx::complex yet. As a stopgap, this provides
// routines for casting between the two types.

namespace complex_cast
{

// map gt::complex to std::complex, leave other types unchanged
template <typename T>
inline auto std_cast(T* p)
{
  return p;
}

template <typename T>
inline auto std_cast(T** p)
{
  return p;
}

template <typename T>
inline auto std_cast(gt::backend::device_ptr<T> p)
{
  return p.get();
}

template <typename T>
inline auto std_cast(gt::complex<T>* p)
{
  return reinterpret_cast<std::complex<T>*>(p);
}

template <typename T>
inline auto std_cast(const gt::complex<T>* p)
{
  return reinterpret_cast<const std::complex<T>*>(p);
}

template <typename T>
inline auto std_cast(gt::complex<T>** p)
{
  return reinterpret_cast<std::complex<T>**>(p);
}

template <typename T>
inline auto std_cast(const gt::complex<T>** p)
{
  return reinterpret_cast<const std::complex<T>**>(p);
}

template <typename T>
inline auto std_cast(gt::backend::device_ptr<gt::complex<T>> p)
{
  return reinterpret_cast<std::complex<T>*>(p.get());
}

template <typename T>
struct make_std
{
  using type = T;
};

template <typename R>
struct make_std<gt::complex<R>>
{
  using type = std::complex<R>;
};

} // namespace complex_cast

#endif // GTENSOR_DEVICE_SYCL

} // namespace gt

#endif
