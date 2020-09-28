
#ifndef GTENSOR_SPAN_H
#define GTENSOR_SPAN_H

#include <cassert>

#if __cplusplus >= 202000L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202000L)
#include <span>
#endif

#include "defs.h"

#ifdef GTENSOR_HAVE_DEVICE
#ifdef GTENSOR_USE_THRUST
#include <thrust/device_ptr.h>
#else
#include "gtensor_storage.h"
#endif
#endif

namespace gt
{

// ======================================================================
// span
//
// very minimal, just enough to support making a gtensor_view
// Note that the span has pointer semantics, in that coyping does
// not copy the underlying data, just the pointer and size, and
// requesting access to the underlying data from a const instance
// via data() and operator[] returns a non-const reference allowing
// modification. This is consistent with the C++20 standardized
// span and with gsl::span. To not allow modification, the underlying
// data type can be const.

#if __cplusplus >= 202000L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202000L)

template <typename T>
using span = std::span<T>;

#else // not C++ 20, define subset of span we care about

template <typename T>
class span
{
public:
  using element_type = T;
  using value_type = typename std::remove_cv<T>::type;

  using pointer = typename std::add_pointer<element_type>::type;
  using const_pointer = typename std::add_const<pointer>::type;
  using reference = typename std::add_lvalue_reference<element_type>::type;
  using const_reference = typename std::add_const<reference>::type;
  using size_type = gt::size_type;

  span() = default;
  span(pointer data, size_type size) : data_{data}, size_{size} {}

  span(const span& other) = default;

  template <class OtherT,
            std::enable_if_t<std::is_convertible<OtherT (*)[], T (*)[]>::value,
                             int> = 0>
  span(const span<OtherT>& other) : data_{other.data()}, size_{other.size()}
  {}

  span& operator=(const span& other) = default;

  GT_INLINE pointer data() const { return data_; }
  GT_INLINE size_type size() const { return size_; }

  GT_INLINE reference operator[](size_type i) const { return data_[i]; }

private:
  pointer data_ = nullptr;
  size_type size_ = 0;
};

#endif // C++20

#ifdef GTENSOR_HAVE_DEVICE

// ======================================================================
// device_span
//
// for a gtensor_view of device memory

#ifdef GTENSOR_USE_THRUST

template <typename T>
class device_span
{
public:
  using element_type = T;
  using value_type = typename std::remove_cv<T>::type;

  using pointer = thrust::device_ptr<T>;
  using const_pointer = typename std::add_const<pointer>::type;
  using reference = thrust::device_reference<T>;
  using const_reference = typename std::add_const<reference>::type;
  using size_type = gt::size_type;

  device_span() = default;
  device_span(pointer data, size_type size) : data_{data}, size_{size} {}

  device_span(const device_span& other) = default;

  template <class OtherT,
            std::enable_if_t<std::is_convertible<OtherT (*)[], T (*)[]>::value,
                             int> = 0>
  device_span(const device_span<OtherT>& other)
    : data_{other.data()}, size_{other.size()}
  {}

  device_span& operator=(const device_span& other) = default;

  GT_INLINE pointer data() const { return data_; }
  GT_INLINE size_type size() const { return size_; }

  GT_INLINE reference operator[](size_type i) const { return data_[i]; }

private:
  pointer data_;
  size_type size_ = 0;
};

#else // not GTENSOR_USE_THRUST

template <typename T>
using device_span = span<T>;

#endif // GTENSOR_USE_THRUST

#endif // GTENSOR_HAVE_DEVICE

} // namespace gt

#endif // GTENSOR_SPAN_H
