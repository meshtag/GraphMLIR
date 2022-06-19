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

#ifndef INCLUDE_GRAPH
#define INCLUDE_GRAPH

#include <Interface/memref.h>

namespace graph {
namespace detail {
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
