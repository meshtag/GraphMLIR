//===- LemonMinSpanningTree.cpp --------------------------------------------------===//
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
// This file implements the benchmark for LEMON Minimum Spanning Tree.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <bits/stdc++.h>
#include <lemon/maps.h>
#include <lemon/kruskal.h>
#include <lemon/list_graph.h>

using namespace std;
using namespace lemon;

#define V 10
#define MaxWeight 1000
#define UpperLimit 100
#define LowerLimit 2

typedef ListGraph::Node Node;
typedef ListGraph::Edge Edge;
typedef ListGraph::NodeIt NodeIt;
typedef ListGraph::EdgeIt EdgeIt;
typedef ListGraph::EdgeMap<int> ECostMap;

namespace {
ListGraph g;
ListGraph::Node nodes[V];
ECostMap edge_cost_map(g);
} // namespace

void initializeLemonMinSpanningTree() {
  for (int i = 0; i < V; i++) {
    nodes[i] = g.addNode();
  }
  std::set<std::pair<int, int>> container;
  std::set<std::pair<int, int>>::iterator it;
  srand(time(NULL));

  int NUM = V; // Number of Vertices
  int MAX_EDGES = V * (V - 1) / 2;
  int NUMEDGE = MAX_EDGES; // Number of Edges
  for (int j = 1; j <= NUMEDGE; j++) {
    int a = rand() % NUM;
    int b = rand() % NUM;

    std::pair<int, int> p = std::make_pair(a, b);
    std::pair<int, int> reverse_p = std::make_pair(b, a);

    while (container.find(p) != container.end() ||
           container.find(reverse_p) != container.end() || a==b) {
      a = rand() % NUM;
      b = rand() % NUM;
      p = std::make_pair(a, b);
      reverse_p = std::make_pair(b, a);
    }

    container.insert(p);
    container.insert(reverse_p);
    edge_cost_map.set(g.addEdge(nodes[a], nodes[b]), 1 + rand() % MaxWeight);
  }
}

// Benchmarking function.
static void LemonMinSpanningTree(benchmark::State &state) {
  for (auto _ : state) {
    vector<Edge> tree_edge_vec;
    for (int i = 0; i < state.range(0); ++i) {
      kruskal(g, edge_cost_map, std::back_inserter(tree_edge_vec));
    }
  }
}

BENCHMARK(LemonMinSpanningTree)->Arg(1);

void generateResultLemonMinSpanningTree() {
  initializeLemonMinSpanningTree();
  cout << "-------------------------------------------------------\n";
  cout << "[ LEMON Kruskal Result Information ]\n";

  vector<Edge> tree_edge_vec;
  std::cout << "The weight of the minimum spanning tree is "
            << kruskal(g, edge_cost_map, std::back_inserter(tree_edge_vec))
            << std::endl;

  std::cout << "The edges of the tree are: ";
  for (int i = tree_edge_vec.size() - 1; i >= 0; i--)
    std::cout << g.id(tree_edge_vec[i]) << ";";
  std::cout << std::endl;
  std::cout << "The size of the tree is: " << tree_edge_vec.size()
            << std::endl;

  cout << "Lemon Kruskal Operation Completed!\n";
}
