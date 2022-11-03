//===- GraphMlirFloydWarshallBenchmark.cpp
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
#include <benchmark/benchmark.h>

using namespace std;

namespace {
Graph<int, 2> sample_graph(graph::detail::GRAPH_ADJ_MATRIX_UNDIRECTED_WEIGHTED,
                           4);
intptr_t size[2];
MemRef<int, 2> *input;
} // namespace

void initializeGraphMLIRFloydWarshall() {
  sample_graph.addEdge(0, 1, 4);
  sample_graph.addEdge(1, 2, 3);
  sample_graph.addEdge(2, 3, 3);
  sample_graph.addEdge(3, 0, 6);
  sample_graph.addEdge(0, 2, 2);
  sample_graph.addEdge(1, 3, 2);

  input = &sample_graph.get_Memref();

  size[0] = 4;
  size[1] = 4;

  MemRef<int, 2> output = MemRef<int, 2>(size);
}

// Benchmarking function.
static void GraphMLIR_FloydWarshall(benchmark::State &state) {
  for (auto _ : state) {
    MemRef<int, 2> output = MemRef<int, 2>(size);
    for (int i = 0; i < state.range(0); ++i) {
      graph::floyd_warshall(input, &output);
    }
  }
}

// Register benchmarking function.
BENCHMARK(GraphMLIR_FloydWarshall)->Arg(1);

void generateResultGraphMLIRFloydWarshall() {
  initializeGraphMLIRFloydWarshall();
  cout << "-------------------------------------------------------\n";
  cout << "[ GraphMLIR Floyd Warshall Result Information ]\n";
  MemRef<int, 2> generateResult(size);
  graph::floyd_warshall(input, &generateResult);

  // auto y = generateResult.getData();

  // for(int i=0; i<size[0]; i++){
  //     for(int j=0; j<size[1]; j++){
  //         std::cout<<y[i*size[0] + j]<<" ";
  //     }
  //     std::cout<<"\n";
  // }
  cout << "FLoyd Warshall operation finished!\n";
}
