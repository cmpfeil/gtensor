#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include <gtensor/complex_float16_t.h>
#include <gtensor/float16_t.h>

TEST(complex_float16_t, comparison_operators)
{
  gt::complex_float16_t a{7.0, -2.0};
  gt::complex_float16_t b{6.0, -3.0};
  gt::complex_float16_t c{7.0, -3.0};
  gt::complex_float16_t d{6.0, -2.0};

  EXPECT_EQ(a, a);
  EXPECT_NE(a, b);
  EXPECT_NE(a, c);
  EXPECT_NE(a, d);

  gt::complex_float16_t e{3.0, 0.0};
  gt::complex_float16_t f{3.0, 1.0};
  gt::float16_t s{3.0};
  gt::float16_t t{4.0};

  EXPECT_EQ(e, s);
  EXPECT_EQ(s, e);
  EXPECT_NE(f, s);
  EXPECT_NE(s, f);
  EXPECT_NE(e, t);
  EXPECT_NE(t, e);
  EXPECT_NE(f, t);
  EXPECT_NE(t, f);

}

TEST(complex_float16_t, constructors)
{
  gt::complex_float16_t a{7.0, -2.0};
  gt::complex_float16_t b{a};
  gt::complex<float> c{7.0, -2.0};
  gt::complex_float16_t d{c};

  EXPECT_EQ(a, b);
  EXPECT_EQ(a, d);

}
