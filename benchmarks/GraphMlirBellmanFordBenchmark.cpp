//===- GraphMlirBellmanFordBenchmark.cpp
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
// This file implements the benchmark for GraphMLIR Bellman Ford Benchmark.
//
//===----------------------------------------------------------------------===//

#include <Interface/GraphContainer.h>
#include <Interface/graph.h>
#include <Utility/Utils.h>
#include <benchmark/benchmark.h>

#define V 100
#define NUM_EDGE V *(V - 1) / 2

using namespace std;

namespace {
// Graph<float, 2>
// sample_graph(graph::detail::GRAPH_ADJ_MATRIX_DIRECTED_WEIGHTED,
//                            100);
intptr_t size[1];
// // MemRef<float, 2> *input;
MemRef<int, 1> *start;
MemRef<int, 1> *e;
MemRef<int, 1> *dist;
} // namespace

void initializeGraphMLIRBellmanFord() {

  // graph::generateRandomGraph(&sample_graph, V);

  // input = &sample_graph.get_Memref();

  // size = 100;

  // MemRef<float, 1> output = MemRef<float, 1>(size);

  std::vector<int> st, ed, distance(V, INT32_MAX);

  srand(time(0));

  for (int i = 0; i < NUM_EDGE; i++) {
    int u = rand() % V;
    int v = rand() % V;
    int d = (rand() % 100);

    st.push_back(u);
    ed.push_back(v);
    distance.push_back(d);
  }
  size[0] = V;

  MemRef<int, 1> temp1 = MemRef<int, 1>(st);
  start = &temp1;
  MemRef<int, 1> temp2 = MemRef<int, 1>(ed);
  e = &temp2;
  MemRef<int, 1> temp3 = MemRef<int, 1>(distance);
  dist = &temp3;
  // start = MemRef<int, 1>(st);
  // e = MemRef<int, 1>(ed);
  // dist = MemRef<int, 1>(distance);
}

// Benchmarking function.
static void GraphMLIR_BellmanFord(benchmark::State &state) {
  for (auto _ : state) {
    std::vector<int> op(V, INT32_MAX);
    MemRef<int, 1> output = MemRef<int, 1>(op);
    for (int i = 0; i < state.range(0); ++i) {
      graph::bellman_ford(start, e, dist, &output);
    }
  }
}

// Register benchmarking function.
BENCHMARK(GraphMLIR_BellmanFord)->Arg(1);

void generateResultGraphMLIRBellmanFord() {
  initializeGraphMLIRBellmanFord();
  cout << "-------------------------------------------------------\n";
  cout << "[ GraphMLIR Bellman Ford Result Information ]\n";
  MemRef<int, 1> output(size);
  // graph::bellman_ford(&start, &end, &dist, &output);

  cout << "Bellman Ford operation finished!\n";
}
