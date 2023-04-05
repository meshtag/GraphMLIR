//===- graph.h
//-------------------------------------------------------------===//
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

#include <Interface/Container.h>

namespace graph {
namespace detail {

enum graph_type {
  GRAPH_ADJ_LIST_UNDIRECTED_UNWEIGHTED,
  GRAPH_ADJ_LIST_UNDIRECTED_WEIGHTED,
  GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED,
  GRAPH_ADJ_LIST_DIRECTED_WEIGHTED,
  GRAPH_ADJ_MATRIX_UNDIRECTED_UNWEIGHTED,
  GRAPH_ADJ_MATRIX_UNDIRECTED_WEIGHTED,
  GRAPH_ADJ_MATRIX_DIRECTED_UNWEIGHTED,
  GRAPH_ADJ_MATRIX_DIRECTED_WEIGHTED,
  GRAPH_INC_MATRIX_UNDIRECTED_UNWEIGHTED,
  GRAPH_INC_MATRIX_UNDIRECTED_WEIGHTED,
  GRAPH_INC_MATRIX_DIRECTED_UNWEIGHTED,
  GRAPH_INC_MATRIX_DIRECTED_WEIGHTED,
};

// Functions present inside graph::detail are not meant to be called by users
// directly.
// Declare the BFS C interface.
extern "C" {
void _mlir_ciface_bfs(MemRef<int, 1> *weights, MemRef<int, 1> *cnz,
                      MemRef<int, 1> *cidx, MemRef<int, 1> *parent,
                      MemRef<int, 1> *distance);
}

extern "C" {
void _mlir_ciface_floyd_warshall(MemRef<float, 2> *graph1,
                                 MemRef<float, 2> *graph2);
}

} // namespace detail

void inline graph_bfs(MemRef<int, 1> *weights, MemRef<int, 1> *cnz,
                      MemRef<int, 1> *cidx, MemRef<int, 1> *parent,
                      MemRef<int, 1> *distance) {
  detail::_mlir_ciface_bfs(weights, cnz, cidx, parent, distance);
}

void inline floyd_warshall(MemRef<float, 2> *input, MemRef<float, 2> *output) {
  detail::_mlir_ciface_floyd_warshall(input, output);
}
} // namespace graph

#endif
