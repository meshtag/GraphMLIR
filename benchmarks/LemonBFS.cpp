//===- lemonBFS.cpp -------------------------------------------------------===//
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
// This file implements the benchmark for Lemon BFS example benchmark.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <lemon/bfs.h>
#include <lemon/list_graph.h>

using namespace std;
using namespace lemon;

namespace {
ListDigraph g;
ListDigraph::Node source = g.addNode();
} // namespace

void initializeLemonBFS() {
  ListDigraph::Node source = g.addNode();
  ListDigraph::Node y = g.addNode();
  ListDigraph::Node z = g.addNode();
  ListDigraph::Node w = g.addNode();

  g.addArc(source, y);
  g.addArc(y, z);
  g.addArc(z, w);
  g.addArc(w, source);
  g.addArc(source, z);
  g.addArc(y, w);

  Bfs<ListDigraph> bfs(g);
}

// Benchmarking function.
static void Lemon_BFS(benchmark::State &state) {
  for (auto _ : state) {
    Bfs<ListDigraph> bfs(g);
    for (int i = 0; i < state.range(0); ++i) {
      bfs.run(source);
    }
  }
}

// Register benchmarking function.
BENCHMARK(Lemon_BFS)->Arg(1);

void generateResultLemonBFS() {
  initializeLemonBFS();
  cout << "-------------------------------------------------------\n";
  cout << "[ LEMON BFS Result Information ]\n";
  Bfs<ListDigraph> output(g);
  output.run(source);
  cout << "Lemon bfs operation finished!\n";
}
