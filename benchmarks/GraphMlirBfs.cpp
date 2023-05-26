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

#define V 1000

namespace {
int g[V][V];
intptr_t opsize[1];
std::vector<int> a, ia, ca;
MemRef<int, 1> *weights, *cnz, *cidx;
} // namespace

// Generate compressed sparse matrix
void generateCSR(int graph[V][V], std::vector<int> &a, std::vector<int> &ia,
                 std::vector<int> &ca) {
  int count = 0;
  ia.push_back(0);

  for (int r = 0; r < V; r++) {
    for (int c = 0; c < V; c++) {
      if (graph[r][c] != 0) {
        a.push_back(graph[r][c]);
        ca.push_back(c);

        count++;
      }
    }

    ia.push_back(count);
  }
}

void initializeGraphMLIRBfs() {
  int MAX_EDGES = V * (V - 1) / 2;
  int NUMEDGES = MAX_EDGES;

  for (int i = 0; i < NUMEDGES; i++) {
    int u = rand() % V;
    int v = rand() % V;
    int d = rand() % 100 + 1;

    if (g[u][v] == 0)
      g[u][v] = d;
  }

  std::vector<int> a, ia, ca;

  generateCSR(g, a, ia, ca);

  opsize[0] = V + 1;

  weights = new MemRef<int, 1>(a);
  cnz = new MemRef<int, 1>(ia);
  cidx = new MemRef<int, 1>(ca);
  MemRef<int, 1> parent = MemRef<int, 1>(opsize);
  MemRef<int, 1> distance = MemRef<int, 1>(opsize);
}

static void GraphMLIR_Bfs(benchmark::State &state) {
  for (auto _ : state) {
    MemRef<int, 1> parent = MemRef<int, 1>(opsize);
    MemRef<int, 1> distance = MemRef<int, 1>(opsize);
    for (int i = 0; i < state.range(0); ++i) {
      graph::graph_bfs(weights, cnz, cidx, &parent, &distance);
    }
  }
}

BENCHMARK(GraphMLIR_Bfs)->Arg(1);

void generateResultGraphMLIRBfs() {
  initializeGraphMLIRBfs();

  cout << "-------------------------------------------------------\n";
  cout << "[ GraphMLIR BFS Result Information ]\n";

  MemRef<int, 1> parent(opsize);
  MemRef<int, 1> distance(opsize);

  graph::graph_bfs(weights, cnz, cidx, &parent, &distance);

  cout << "BFS operation finished!\n";
}