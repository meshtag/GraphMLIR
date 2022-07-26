 
//===- Container.h --------------------------------------------------------===//
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
// Container descriptor.
//
//===----------------------------------------------------------------------===//


#ifndef INTERFACE_GRAPH_CORE_CONTAINER_H
#define INTERFACE_GRAPH_CORE_CONTAINER_H

#include <memory>
#include <stdint.h>
#include <vector>
#include <list>

// MemRef descriptor.
// - T represents the type of the elements.
// - N represents the number of dimensions.
template <typename T, size_t N> class MemRef {

public:
  // Default constructor.
  MemRef() {};

  // Data.
  // The `aligned` and `allocated` members point to the same address, `aligned`
  // member is responsible for handling data, and `allocated` member is
  // resposible for handling the memory space.
  T *allocated;
  T *aligned;
  // Offset.
  intptr_t offset = 0;
  // Shape.
  intptr_t sizes[N];
  // Strides.
  intptr_t strides[N];
  // Number of elements.
  size_t size;
};
#include "Interface/Container.cpp"
#endif // INTERFACE_GRAPH_CORE_CONTAINER. 
