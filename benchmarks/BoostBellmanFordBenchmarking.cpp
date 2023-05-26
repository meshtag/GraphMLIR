//===- BoostBellmanforBenchmarking.cpp -----------------------------------===//
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
// This file implements the benchmark for Boost Bellmanford example benchmark.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <benchmark/benchmark.h>
#include <boost/config.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/exterior_property.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/bellman_ford_shortest_paths.hpp>
#include <vector>
#include <Utility/Utils.h>

using namespace std;

#define SIZE 100

namespace {
typedef int t_weight;

// define the graph type
typedef boost::property<boost::edge_weight_t, t_weight> EdgeWeightProperty;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                              boost::no_property, EdgeWeightProperty>
    Graph;

typedef boost::property_map<Graph, boost::edge_weight_t>::type WeightMap;


Graph g;

std::vector<int> distnce(SIZE, (std::numeric_limits<int>::max)());
int source_node_index = 0;
std::vector<std::size_t> parent(SIZE);

} // namespace

void initializeBoostBellmanFord() {

  const int vertices = SIZE;
  int num_edges = vertices * (vertices -1) / 2;

  // define edges
  // int edges[] = {1, 2, 2, 3, 3, 4, 4, 1, 1, 3, 2, 4};

  // t_weight weight[] = {4, 3, 3, 6, 2, 2};

  std::vector<int> edges;
  std::vector<int> weight;

  graph::generateRandomGraph(edges, weight, vertices);

  for (std::size_t k = 0; k < num_edges; ++k)
    boost::add_edge(edges[k * 2] - 1, edges[k * 2 + 1] - 1, weight[k], g);
  WeightMap weight_pmap = boost::get(boost::edge_weight, g);


  for (int i = 0; i < num_vertices(g); ++i)
    parent[i] = i;
  distnce[source_node_index] = 0;

 
}

// Benchmarking function.
static void Boost_BellmanFord(benchmark::State &state) {

  for (auto _ : state) {
    WeightMap weight_pmap = boost::get(boost::edge_weight, g);

    for (int i = 0; i < state.range(0); ++i) {
        bool r = bellman_ford_shortest_paths(g, num_vertices(g), weight_pmap, &parent[0],
        &distnce[0], boost::closed_plus< int >(), std::less< int >(),
        boost::default_bellman_visitor());
    }
  }
}

// Register benchmarking function.
BENCHMARK(Boost_BellmanFord)->Arg(1);

void generateResultBoostBellmanFord() {
  initializeBoostBellmanFord();
  WeightMap weight_pmap = boost::get(boost::edge_weight, g);
  for (int i = 0; i < num_vertices(g); ++i)
    parent[i] = i;
  distnce[source_node_index] = 0;

  cout << "-------------------------------------------------------\n";
  cout << "[ BOOST FloydWarshall Result Information ]\n";
  bool r = bellman_ford_shortest_paths(g, num_vertices(g), weight_pmap, &parent[0],
        &distnce[0], boost::closed_plus< int >(), std::less< int >(),
        boost::default_bellman_visitor());
  cout << "Boost FloydWarshall operation finished!\n";
}