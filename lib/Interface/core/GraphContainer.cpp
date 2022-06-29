//===- GraphContainer.cpp -------------------------------------------------===//
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
// This file implements the Graph container descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPH_CONTAINER_DEF
#define GRAPH_CONTAINER_DEF

#include "Interface/core/Container.h"
#include "Interface/core/GraphContainer.h"


template <typename T, size_t N> Graph<T, N>::Graph(std::vector<Node*> adjList) : MemRef<T, N> {
    this->size = adjList.size();
    this->allocated = new T[this->size];
    this->aligned = this->allocated;

    //implementation to fill the aligned
}
//explicit template instantiation 
// template class Graph<float,1>;
#endif // GRAPH_CONTAINER_DEF 