//===- BoostMinSpanningTree.cpp
//-------------------------------------------------------===//
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
// This file implements the benchmark for Boost Minimum Spanning Tree.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <benchmark/benchmark.h>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/exterior_property.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <vector>
#include <Utility/Utils.h>

#define V 10
#define MAX_WEIGHT 1000

using namespace std;

namespace {
typedef int t_weight;

// define the graph type
typedef boost::property<boost::edge_weight_t, t_weight> EdgeWeightProperty;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                              boost::no_property, EdgeWeightProperty>
    Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor VerticesMap;

Graph g;
} // namespace

void initializeBoostMinSpanningTree() {

  const int vertices = V;
  int num_edges = V * (V -1) / 2;

  std::vector<int> edges;
  std::vector<int> weight;

  graph::generateRandomGraph(edges, weight, vertices, MAX_WEIGHT);

  for (std::size_t k = 0; k < num_edges; ++k) {
    boost::add_edge(edges[k * 2] - 1, edges[k * 2 + 1] - 1, weight[k], g);
  }

  // set the parent vector to receive the minimum spanning tree output
  std::vector<VerticesMap> p(num_vertices(g));
}

// Benchmarking function.
static void BoostMinSpanningTree(benchmark::State &state) {

  for (auto _ : state) {
    // set the parent vector to receive the minimum spanning tree output
    std::vector<VerticesMap> p(num_vertices(g));

    for (int i = 0; i < state.range(0); ++i) {
      boost::prim_minimum_spanning_tree(g, &p[0]);
    }
  }
}

// Register benchmarking function.
BENCHMARK(BoostMinSpanningTree)->Arg(1);

void generateResultBoostMinSpanningTree() {
    initializeBoostMinSpanningTree();
  
    // set the parent vector to receive the minimum spanning tree output
    std::vector<VerticesMap> p(num_vertices(g));
    std::cout << "-------------------------------------------------------\n";
    std::cout << "[ Boost Minimum Spanning Tree Result Information ]\n";
    boost::prim_minimum_spanning_tree(g, &p[0]);
    for (int i = 0; i < p.size(); i++) {
      std::cout << "p[" << i << "] = " << p[i] << ", ";
    }
    std::cout << "Boost Minimum Spanning Tree operation finished!\n";
}
