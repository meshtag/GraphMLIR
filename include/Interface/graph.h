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

namespace graph {
namespace detail {

  enum class graph_type {
      GRAPH_ADJ_LIST_UNDIRECTED_UNWEIGHTED,
  GRAPH_ADJ_LIST_UNDIRECTED_WEIGHTED,
  GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED,
  GRAPH_ADJ_LIST_DIRECTED_WEIGHTED,
  GRAPH_ADJ_MATRIX_UNDIRECTED_UNWEIGHTED,
  GRAPH_ADJ_MATRIX_UNDIRECTED_WEIGHTED,
  GRAPH_ADJ_MATRIX_DIRECTED_UNWEIGHTED,
  GRAPH_ADJ_MATRIX_DIRECTED_WEIGHTED,
  GRAPH_INC_MATRIX_DIRECTED_WEIGHTED,
  GRAPH_INC_MATRIX_DIRECTED_UNWEIGHTED,
  GRAPH_INC_MATRIX_UNDIRECTED_WEIGHTED,
  GRAPH_INC_MATRIX_UNDIRECTED_UNWEIGHTED
  };
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
} // namespace graph

#endif
