//===- GraphMlirMinSpanningTreeBenchmark.cpp
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
// This file implements the benchmark for GraphMLIR Minimum Spanning Tree.
//
//===----------------------------------------------------------------------===//

#include <Interface/GraphContainer.h>
#include <Interface/graph.h>
#include <benchmark/benchmark.h>
#include <Utility/Utils.h>

#define V 10
#define MAX_WEIGHT 1000

using namespace std;

namespace {
Graph<int, 2> g(graph::detail::GRAPH_ADJ_MATRIX_UNDIRECTED_WEIGHTED, V);
MemRef<int, 2> *input;
intptr_t size[1];
} // namespace

void initializeGraphMlirMinSpanningTree() {
  graph::generateRandomGraphI(&g, V);
  input = &g.get_Memref();
  size[0] = V;

  MemRef<int, 1> cost = MemRef<int, 1>(size, MAX_WEIGHT);
  MemRef<int, 1> visited = MemRef<int, 1>(size, 0);
  MemRef<int, 1> output = MemRef<int, 1>(size, -1);
}

// Benchmarking function.
static void GraphMlirMinSpanningTree(benchmark::State &state) {
  for (auto _ : state) {
    MemRef<int, 1> output = MemRef<int, 1>(size, -1);
    MemRef<int, 1> visited = MemRef<int, 1>(size, 0);
    MemRef<int, 1> cost = MemRef<int, 1>(size, MAX_WEIGHT);
    for (int i = 0; i < state.range(0); ++i) {
      graph::min_spanning_tree(input, &output, &visited, &cost);
    }
  }
}

// Register benchmarking function.
BENCHMARK(GraphMlirMinSpanningTree)->Arg(1);

void generateResultGraphMlirMinSpanningTree() {
  initializeGraphMlirMinSpanningTree();
  MemRef<int, 1> output(size, -1);
  MemRef<int, 1> visited(size, 0);
  MemRef<int, 1> cost(size, MAX_WEIGHT);

  std::cout << "-------------------------------------------------------\n";
  std::cout << "[ GraphMLIR Minimum Spanning Tree Result Information ]\n";
  graph::min_spanning_tree(input, &output, &visited, &cost);

  auto parent = output.getData();
  for (int i = 0; i < V; i++) {
    std::cout << "p[" << i << "] = " << parent[i] << ", ";
  }

  std::cout << "GraphMLIR Minimum Spanning Tree operation finished!\n";
}
