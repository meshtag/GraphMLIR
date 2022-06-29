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

#include "Interface/core/Container.h"

// Node struct for Adjacency list representation

struct Node{
    int Vertex; 
    Node* next;
};
// Graph container.
// - T represents the type of the elements.
// - N represents the number of dimensions.
template <typename T, size_t N> class Graph : public MemRef<T, N> {
public:
  // For Adjaceny List
  Graph();
  Graph(std::vector<Node*> adjList);
  ~Graph();
};

#include "Interface/core/GraphContainer.cpp"
#endif // INTERFACE_GRAPHCONTAINER_H 