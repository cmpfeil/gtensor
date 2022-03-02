#include <gtest/gtest.h>

#include <stdint.h>

#include <gtensor/gtensor.h>

#include "test_debug.h"

#define MAX_DEVICES 100

#ifdef GTENSOR_HAVE_DEVICE

TEST(device_backend, list_devices)
{
  int n_devices = gt::backend::clib::device_get_count();
  uint32_t vendor_id[MAX_DEVICES];

  ASSERT_LE(n_devices, MAX_DEVICES);

  for (int i = 0; i < n_devices; i++) {
    vendor_id[i] = gt::backend::clib::device_get_vendor_id(i);
    GT_DEBUG_PRINTLN("device[" << i << "]: 0x" << std::setfill('0')
                               << std::setw(8) << std::hex << vendor_id[i]
                               << std::dec << std::endl);
    for (int j = i - 1; j >= 0; j--) {
      EXPECT_NE(vendor_id[i], vendor_id[j]);
    }
  }
}

#define N 10
TEST(device_backend, managed_allocate)
{
  using allocator = gt::backend::clib::gallocator<gt::space::clib_managed>;
  double* a = allocator::allocate<double>(N);
  for (int i = 0; i < N; i++) {
    a[i] = ((double)i) / N;
  }
  auto aview = gt::adapt_device(a, gt::shape(N));
  aview = aview + 1.0;
  gt::synchronize();
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(a[i], 1.0 + ((double)i) / N);
  }
  allocator::deallocate(a);
}

#ifdef GTENSOR_DEVICE_SYCL

TEST(device_backend, sycl_new_stream_queue)
{
  cl::sycl::queue& q0 = gt::backend::sycl::get_queue();
  cl::sycl::queue& q1 = gt::backend::sycl::new_stream_queue();

  EXPECT_NE(q0, q1);

  auto q0_view = gt::stream_view(q0);
  auto q1_view = gt::stream_view(q1);

  EXPECT_TRUE(q0_view.is_default());
  EXPECT_FALSE(q1_view.is_default());

  auto a = gt::zeros_device<int>({5});
  auto b = gt::full_like(a, 1);
  auto c = gt::full_like(a, 2);
  auto h_a = gt::empty<int>(a.shape());
  auto h_b = gt::empty_like(h_a);
  auto h_c = gt::empty_like(h_a);

  gt::copy(a, h_a);
  gt::copy(b, h_b);
  gt::copy(c, h_c);

  EXPECT_NE(h_b, h_a);
  EXPECT_NE(h_c, h_a);

  gt::assign(b, a, q1);
  q1.wait();
  gt::assign(c, b, q0);
  q0.wait();

  gt::copy(a, h_a);
  gt::copy(b, h_b);
  gt::copy(c, h_c);

  EXPECT_EQ(h_b, h_a);
  EXPECT_EQ(h_c, h_a);

  gt::backend::sycl::delete_stream_queue(q1);

  EXPECT_FALSE(gt::backend::sycl::has_open_stream_queues());
}

#endif // GTENSOR_DEVICE_SYCL

#endif // GTENSOR_HAVE_DEVICE
