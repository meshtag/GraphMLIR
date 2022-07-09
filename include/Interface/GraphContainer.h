//===- GraphContainer.h ---------------------------------------------------===//
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
// Graph container descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef INTERFACE_GRAPHCONTAINER_H
#define INTERFACE_GRAPHCONTAINER_H

#include "Interface/Container.h"

// Graph container.
// - T represents the type of the elements.
// - N represents the number of dimensions.
template <typename T, size_t N> class Graph : public MemRef<T, N> {
    // V is the number of vertices/nodes.
    int64_t V,E;
    int ** Matrix;

public:
  // Default Constructor
  Graph();
  // For Adjaceny Matrix.
  Graph(int64_t vertices, int64_t edges);

  // TODO:
  // Add Different contructors.
 int ** Insert(int64_t edges, int64_t vertices, std::string method="incidence",std::string type="undirected");
#include "Interface/GraphContainer.cpp"

#endif // INTERFACE_GRAPHCONTAINER_H 
