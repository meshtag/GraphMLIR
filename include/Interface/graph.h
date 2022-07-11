//===- graph.h
//--------------------------------------------------------------===//
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
// Header file for graph dialect specific operations and other entities.
//
//===----------------------------------------------------------------------===//

#ifndef INCLUDE_INTERFACE_GRAPH_H
#define INCLUDE_INTERFACE_GRAPH_H

#include <Interface/memref.h>
#include <map>
#include <list>
#include <iostream>

namespace graph {
class Graph;

namespace detail {
enum class GRAPH_TYPE { ADJACENCY_LIST };

// Functions present inside graph::detail are not meant to be called by users
// directly.
// Declare the BFS C interface.
extern "C" {
void _mlir_ciface_bfs(MemRef_descriptor graph1, MemRef_descriptor graph2,
                      MemRef_descriptor graph3);
}
} // namespace detail

void graph_bfs(MemRef_descriptor graph1, MemRef_descriptor graph2,
               MemRef_descriptor graph3) {
  detail::_mlir_ciface_bfs(graph1, graph2, graph3);
}

class Graph {
public:
    std::map<int, std::list<int>> adj;
    detail::GRAPH_TYPE graph_type;
 
    // function to add an edge to graph
    void addEdge(int v, int w);
 
    // DFS traversal of the vertices
    // reachable from v
    void DFS(int v);
};
 
void Graph::addEdge(int v, int w)
{
    adj[v].push_back(w); // Add w to v’s list.
}

namespace detail {

void ConvertGraphToMemRef(Graph g)
{
  switch(g.graph_type)
  {
    case GRAPH_TYPE::ADJACENCY_LIST:
    {
      intptr_t graphSize[2] = {g.adj.size(), g.adj.size()};
      intptr_t graphStrides[2] = {g.adj.size(), g.adj.size()};
      float *allocationPointer = (float *)malloc(sizeof(float));
      float *graphAlign = (float *)malloc(graphSize[0] * graphSize[1] * sizeof(float));

      for (unsigned int i = 0; i < graphSize[0]; ++i)
        for (unsigned int j = 0; j < graphSize[1]; ++j)
          graphAlign[i * graphSize[0] + j] = -1;

      for (unsigned int i = 0; i < graphSize[0]; ++i)
      {
        graphAlign[i * graphSize[0]] = i;
        for (auto val = g.adj[i].begin(); val != g.adj[i].end(); ++val)
        {
          if (*val)
            graphAlign[i * graphSize[0] + *val] = 1;
        }
      }

      for (unsigned int i = 0; i < graphSize[0] * graphSize[1]; ++i)
      {
        if (graphAlign[i] == -1)
          graphAlign[i] = 0;
      }

      // for (unsigned int i = 0; i < graphSize[0] * graphSize[1]; ++i)
      // {
      //   std::cout << graphAlign[i] << " ";
      // }
      // std::cout << "\n";

      for (unsigned int i = 0; i < graphSize[0]; ++i)
      {
        for (unsigned int j = 0; j < graphSize[1]; ++j)
          std::cout << graphAlign[i * graphSize[0] + j];
        std::cout << "\n";
      }
      std::cout << "\n\n";

      // MemRef_descriptor graphMemRef =
      //     MemRef_Descriptor(allocationPointer, sample_graph1_array, 0,
      //                   sample_graph_sizes, sample_graph_strides);
    }
  }
}

} // namespace detail

} // namespace graph
#endif
