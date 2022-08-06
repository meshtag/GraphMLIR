
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
  protected:
    uint16_t grap_type;
    T V;
    std::vector< std::vector<T> > nodes;
    std::vector< std::vector<std::pair<T,T> > > weighted_nodes;

    // template<typename T, size_t N> friend MemRef_descriptor GraphToMemrefConversion <>(Graph<T,N> &g);
    // template<typename T, size_t N> friend void PrintGraphInMemrefConversion<> (Graph<T,N> &g);
public:
  // Constructor for any graph type with V nodes .
  Graph(uint16_t graph_type, T V);
  // Graph(T V);
  // Implementation functions to take in nodes and edges
  void AddEdge(T Node1, T Node2);
  void AddEdge(T Node1,T Node2, T EdgeWeight); 
  //debugging function
  void PrintGraph();
  // TODO-done: Added conversion to memref functions
  MemRef_descriptor GraphToMemrefConversion();
  void PrintGraphInMemrefConversion ();
};


#include "../lib/Interface/GraphContainer.cpp"

#endif // INTERFACE_GRAPHCONTAINER_H