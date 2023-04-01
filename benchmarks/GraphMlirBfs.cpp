//===- GraphMlirBfs.cpp
//----------------------------------===//
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
// This file implements the benchmark for GraphMLIR Floyd Warshall Benchmark.
//
//===----------------------------------------------------------------------===//

#include <Interface/GraphContainer.h>
#include <Interface/graph.h>
#include <Utility/Utils.h>
#include <benchmark/benchmark.h>

using namespace std;

namespace {
Graph<int, 2> sample_graph(graph::detail::GRAPH_ADJ_MATRIX_UNDIRECTED_WEIGHTED,
                           100);
intptr_t size[2], opsize[1];
MemRef<int, 2> *input;
} // namespace

void initializeGraphMLIRBfs() {
  graph::generateRandomGraph(&sample_graph, 100);

  input = &sample_graph.get_Memref();

  opsize[0] = 101;

  MemRef<int, 1> parent = MemRef<int, 1>(opsize);
  MemRef<int, 1> distance = MemRef<int, 1>(opsize);
}

static void GraphMLIR_Bfs(benchmark::State &state) {
  for (auto _ : state) {
    MemRef<int, 1> parent = MemRef<int, 1>(opsize);
    MemRef<int, 1> distance = MemRef<int, 1>(opsize);
    for (int i = 0; i < state.range(0); ++i) {
      graph::graph_bfs(input, &parent, &distance);
    }
  }
}

BENCHMARK(GraphMLIR_Bfs)->Arg(1);

void generateResultGraphMLIRBfs() {
  initializeGraphMLIRBfs();
  cout << "-------------------------------------------------------\n";
  cout << "[ GraphMLIR BFS Result Information ]\n";
  MemRef<int, 1> parent = MemRef<int, 1>(opsize);
  MemRef<int, 1> distance = MemRef<int, 1>(opsize);
  graph::graph_bfs(input, &parent, &distance);

  // auto y = generateResult.getData();

  // for(int i=0; i<size[0]; i++){
  //     for(int j=0; j<size[1]; j++){
  //         std::cout<<y[i*size[0] + j]<<" ";
  //     }
  //     std::cout<<"\n";
  // }
  cout << "BFS operation finished!\n";
}