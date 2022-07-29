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
#include <iostream>
namespace graph {
namespace detail {
// Functions present inside graph::detail are not meant to be called by users
// directly.
enum graph_type {
  GRAPH_ADJ_LIST_UNDIRECTED_UNWEIGHTED,
  GRAPH_ADJ_LIST_UNDIRECTED_WEIGHTED,
  GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED,
  GRAPH_ADJ_LIST_DIRECTED_WEIGHTED,
  GRAPH_ADJ_MATRIX_UNDIRECTED_UNWEIGHTED,
  GRAPH_ADJ_MATRIX_UNDIRECTED_WEIGHTED,
  GRAPH_ADJ_MATRIX_DIRECTED_UNWEIGHTED,
  GRAPH_ADJ_MATRIX_DIRECTED_WEIGHTED,
};

// // // creating function to convert to MemRef 
// // // that takes in instance of Graph class as input
// template <typename T, size_t N> MemRef_descriptor GraphToMemrefConversion (graph::Graph<T,N> g)
// {
//   switch(g.grap_type)
//   {
//     // case to convert Directed Unweighted Adj_List  
//     case graph_type::GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED:
//     {
//       // allocating memory and member data types to store 
//       // new created adjacency matrix
//       intptr_t graphSize[2] = {g.size , g.size};
//       intptr_t graphStrides[2] = {g.size , g.size};
//       float *allocationPointer =  (float *)malloc(sizeof(float));
//       // Storing a 2d-matrix in a 1-d format to store as memref
//       // do for example a 3x3 matrix becomes a matrix of 9 elements 
//       float *graphAlign = (float *)malloc(graphSize[0] * graphSize[1] * sizeof(float));
      
//       for(unsigned int i=0; i< graphSize[0]; ++i)
//         for(unsigned int j = 0; j < graphSize[1]; ++j)
//           // first making all edges as zero
//           graphAlign[i * graphSize[0] + j] = 0;

//       // accessing adjacency list
//       for (unsigned int i = 0; i < graphSize[0]; ++i)
//       {
//         for (auto x : g.nodes[i])
//         {
//           graphAlign[i * graphSize[0] + x] = 1;
//         }
//       }

//       //printing adjacency matrix for debug
//       for(unsigned int i=0; i< graphSize[0]; ++i)
//       {
//         for(unsigned int j = 0; j < graphSize[1]; ++j)
//           // first making all edges as zero
//           std::cout<<graphAlign[i * graphSize[0] + j];
//         std::cout<<"\n";
//       }
//       std::cout<<"\n \n";
//   }
//   break;
// }
// }
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
} // namespace graph

#endif
