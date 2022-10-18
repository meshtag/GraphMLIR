//===- Container.cpp ------------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the container descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef CORE_CONTAINER_DEF
#define CORE_CONTAINER_DEF

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numeric>
#include <stdexcept>

#include "Interface/Container.h"

/**
 * @brief MemRef Shape Constructor. Construct a MemRef object from the data
 * shape and initial value. The default initial value is 0.
 * @tparam T represents the datatype to be used
 * @tparam N represents the number of dimensions
 */
template <typename T, std::size_t N>
MemRef<T, N>::MemRef(intptr_t sizes[N], T init) {
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setStrides();
  size = product(sizes);
  allocated = new T[size];
  aligned = allocated;
  std::fill(aligned, aligned + size, init);
}

/**
 * @brief MemRef Array Constructor. Construct a MemRef object from the data
 * pointer, sizes, and offset. The default offset is 0.
 * @tparam T represents the datatype to be used
 * @tparam N represents the number of dimensions
 */
template <typename T, std::size_t N>
MemRef<T, N>::MemRef(const T *data, intptr_t sizes[N], intptr_t offset) {
  this->offset = offset;
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setStrides();
  size = product(sizes);
  allocated = new T[size];
  aligned = allocated;
  std::copy(data, data + size, aligned);
}

/**
 * @brief MemRef Array Constructor. Construct a MemRef object from a unique_ptr,
 * sizes, and offset. The default offset is 0.
 * @tparam T represents the datatype to be used
 * @tparam N represents the number of dimensions
 */
template <typename T, size_t N>
MemRef<T, N>::MemRef(std::unique_ptr<T> &uptr, intptr_t *sizes,
                     intptr_t offset) {
  if (!uptr)
    assert(0 && "Taking over an empty unique pointer.");
  T *data = uptr.release();
  this->aligned = data;
  this->allocated = data;
  this->offset = offset;
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setStrides();
  size = product(sizes);
}

/**
 * @brief Copy Constructor. This constructor is used to initialize a MemRef
 object with another MemRef object.
        - Copy `offset` and `size` directly.
 *  - Elementwise copy `sizes` array.
 *  - Calculate `strides`.
 *  - Allocate new space.
 *  - Deep copy the data from the original object.
 * @tparam T represents the datatype to be used
 * @tparam N represents the number of dimensions
 */
template <typename T, std::size_t N>
MemRef<T, N>::MemRef(const MemRef<T, N> &other)
    : offset(other.offset), size(other.size) {
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = other.sizes[i];
  }
  setStrides();
  allocated = new T[size];
  aligned = allocated;
  std::copy(other.aligned, other.aligned + size, aligned);
}

/**
 * @brief Copy Assignment Operator..
 * - Check if they are the same object.
 * - Copy `offset` and `size` directly.
 * - Elementwise copy `sizes`.
 * - Calculate the `strides`.
 * - Free the data space of this object to avoid memory leaks.
 * - Allocate new space and deep copy.
 * @tparam T represents the datatype to be used
 * @tparam N represents the number of dimensions
 */
template <typename T, std::size_t N>
MemRef<T, N> &MemRef<T, N>::operator=(const MemRef<T, N> &other) {
  this->offset = other.offset;
  this->size = other.size;
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = other.sizes[i];
  }
  setStrides();
  // Free the original aligned and allocated space.
  delete[] allocated;
  // Allocate new space and deep copy.
  T *ptr = new T[size];
  std::copy(other.aligned, other.aligned + size, ptr);
  aligned = ptr;
  allocated = ptr;
  return *this;
}

/**
 * @brief Move Constructor. This constructor is used to initialize a MemRef
 * object from a rvalue. The move constructor steals the resources of the
 * original object. Note that the original object no longer owns the members and
 * spaces.
 * - Steal members from the original object.
 * - Assign the NULL pointer to the original aligned and allocated members to
 *   avoid the double free error.
 * @tparam T represents the datatype to be used
 * @tparam N represents the number of dimensions
 */
template <typename T, std::size_t N>
MemRef<T, N>::MemRef(MemRef<T, N> &&other) noexcept
    : allocated(other.allocated), aligned(other.aligned), offset(other.offset),
      size(other.size) {
  std::swap(this->sizes, other.sizes);
  std::swap(this->strides, other.strides);
  // Assign the NULL pointer to the original aligned and allocated members to
  // avoid the double free error.
  other.allocated = other.aligned = nullptr;
}

/**
 * @brief Move Assignment Operator. Note that the original object no longer owns
 * the members and spaces.
 * - Check if they are the same object.
 * - Free the data space of this object to avoid memory leaks.
 * - Steal members from the original object.
 * - Assign the NULL pointer to the original aligned and allocated members to
 *   avoid the double free error.
 * @tparam T represents the datatype to be used
 * @tparam N represents the number of dimensions
 */
template <typename T, std::size_t N>
MemRef<T, N> &MemRef<T, N>::operator=(MemRef<T, N> &&other) noexcept {
  // Free the original aligned and allocated space.
  delete[] allocated;
  // Copy members of the original object.
  MemRef<T, N>::MemRef(other);
  return *this;
}

/**
 * @brief MemRef Destructor. Note that the original object no longer owns the
 * members and spaces. Note that the `allocated` and `aligned` point to the same
 * address, so it is enough to release the space of the `allocated` pointer in
 * the destructor.
 * @tparam T represents the datatype to be used
 * @tparam N represents the number of dimensions
 */
template <typename T, std::size_t N> MemRef<T, N>::~MemRef() {
  if (allocated)
    delete allocated;
}

/**
 * @brief Get the data pointer.
 * Return the `aligned` pointer if the container data size is greater than zero.
 * If the data size is negative or zero, which means no space is allocated for
 * the container data pointer, the function does not allow to return the data
 * pointer.
 * @tparam T represents the datatype to be used
 * @tparam N represents the number of dimensions
 */
template <typename T, std::size_t N> T *MemRef<T, N>::getData() {
  assert((size > 0) && "Invalid container data size.");
  return aligned;
}
/**
 * @brief Get the element at index.
 * Return a const refrence of specific element if the container data size is
 * greater than zero. If the data size is negative or zero, which means no space
 * is allocated for the container data pointer, this operator does not allow to
 * return the data element.
 * @tparam T represents the datatype to be used
 * @tparam N represents the number of dimensions
 */
template <typename T, std::size_t N>
const T &MemRef<T, N>::operator[](size_t index) const {
  assert((size > 0) && "Invalid container data size.");
  return aligned[index + offset];
}

/**
 * @brief Get the element at index.
 * Return a non-const refrence of specific element if the container data size is
 * greater than zero. If the data size is negative or zero, which means no space
 * is allocated for the container data pointer, this operator does not allow to
 * return the data element.
 * @tparam T represents the datatype to be used
 * @tparam N represents the number of dimensions
 */
template <typename T, std::size_t N> T &MemRef<T, N>::operator[](size_t index) {
  assert((size > 0) && "Invalid container data size.");
  return aligned[index + offset];
}

/**
 * @brief Calculate the stride values for each dimension based on the sizes.
 * @tparam T represents the datatype to be used
 * @tparam N represents the number of dimensions
 */
template <typename T, std::size_t N> void MemRef<T, N>::setStrides() {
  assert((N > 0) && "Invalid container number of dims");
  strides[N - 1] = 1;
  if (N < 2)
    return;
  for (int i = N - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * sizes[i + 1];
  }
}

/**
 * @brief Calculate the total number of elements in the MemRef container.
 * @tparam T represents the datatype to be used
 * @tparam N represents the number of dimensions
 */
template <typename T, std::size_t N>
size_t MemRef<T, N>::product(intptr_t sizes[N]) const {
  size_t size = 1;
  for (size_t i = 0; i < N; i++)
    size *= sizes[i];
  return size;
}

/**
 * @brief Release the aligned and allocated field.
 * @tparam T represents the datatype to be used
 * @tparam N represents the number of dimensions
 */
template <typename T, size_t N> T *MemRef<T, N>::release() {
  T *temp = aligned;
  aligned = nullptr;
  allocated = nullptr;
  return temp;
}

template <typename T, size_t N>
bool MemRef<T, N>::operator==(const MemRef<T, N> &other) {
  intptr_t x1 = this->sizes[0];
  intptr_t y1 = this->sizes[1];
  intptr_t x2 = other.sizes[0];
  intptr_t y2 = other.sizes[1];

  // compare the sizes array and size
  if (x1 != x2 || y1 != y2 || this->size != other.size) {
    return false;
  }

  // compare the strides
  if (this->strides[0] != this->strides[0] ||
      other.strides[1] != other.strides[1]) {
    return false;
  }

  for (intptr_t i = 0; i < x1; i++) {
    for (intptr_t j = 0; j < y1; j++) {
      if (this->allocated[i * x1 + y1] != other.allocated[i * x1 + y1]) {
        return false;
      }
      // if(this->aligned[i * x1 + y1] != other.aligned[i * x1 + y1]) {
      // 	return false;
      // }
    }
  }

  return true;
}

#endif // CORE_CONTAINER_DEF
