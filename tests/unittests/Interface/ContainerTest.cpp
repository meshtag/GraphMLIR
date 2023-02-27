#include "Interface/Container.h"
#include <gtest/gtest.h>

template <typename T>
void ASSERT_ARRAY_EQ(const T *x, const T *y, const size_t n) {
  if (std::is_integral<T>::value) {
    for (size_t i = 0; i < n; i++) {
      ASSERT_EQ(x[i], y[i]);
    }
  } else if (std::is_same<T, float>::value) {
    for (size_t i = 0; i < n; i++) {
      ASSERT_FLOAT_EQ(x[i], y[i]);
    }
  } else if (std::is_same<T, double>::value) {
    for (size_t i = 0; i < n; i++) {
      ASSERT_DOUBLE_EQ(x[i], y[i]);
    }
  }
}

class MemRefTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Copy constructor.
TEST_F(MemRefTest, CopyConstructor2DMemref) {
  // new hard codede MemRef object.
  float aligned[] = {0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0,
                     0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
  intptr_t sizes[2] = {6, 6};
  MemRef<float, 2> m(aligned, sizes, 0);
  MemRef<float, 2> copy(m);
  EXPECT_EQ(m == copy, true);
}

// Copy assignment operator.
TEST_F(MemRefTest, CopyAssignment2DMemref) {
  // new hard codede MemRef object.
  float aligned[] = {0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0,
                     0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
  intptr_t sizes[2] = {6, 6};
  MemRef<float, 2> m(aligned, sizes, 0);
  MemRef<float, 2> copy = m;
  EXPECT_EQ(m == copy, true);
}

// Move constructor.
TEST_F(MemRefTest, MoveConstructor2DMemref) {
  // new hard codede MemRef object.
  float aligned[] = {0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0,
                     0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
  intptr_t sizes[2] = {6, 6};
  MemRef<float, 2> m1(aligned, sizes, 0);
  MemRef<float, 2> m2(aligned, sizes, 0);

  // Construct using move constructor.
  MemRef<float, 2> move = std::move(m1);

  // test
  EXPECT_EQ(m2 == move, true);
}

// Move assignment operator.
TEST_F(MemRefTest, MoveAssignment2DMemref) {
  // new hard codede MemRef object.
  float aligned[] = {0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0,
                     0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
  intptr_t sizes[2] = {6, 6};
  MemRef<float, 2> m1(aligned, sizes, 0);
  MemRef<float, 2> m2(aligned, sizes, 0);

  MemRef<float, 2> move;
  // Assignment using the move assignment operator.
  move = std::move(m1);

  // test
  EXPECT_EQ(m2 == move, true);
}
