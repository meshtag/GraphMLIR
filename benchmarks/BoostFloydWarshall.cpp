//===- BoostFloyWarshall.cpp
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
// This file implements the benchmark for Boost FloydWarshall example benchmark.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <benchmark/benchmark.h>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/exterior_property.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <vector>
#include <Utility/Utils.h>

using namespace std;

namespace {
typedef int t_weight;

// define the graph type
typedef boost::property<boost::edge_weight_t, t_weight> EdgeWeightProperty;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                              boost::no_property, EdgeWeightProperty>
    Graph;

typedef boost::property_map<Graph, boost::edge_weight_t>::type WeightMap;

// Declare a matrix type and its corresponding property map that
// will contain the distances between each pair of vertices.
typedef boost::exterior_vertex_property<Graph, t_weight> DistanceProperty;
typedef DistanceProperty::matrix_type DistanceMatrix;
typedef DistanceProperty::matrix_map_type DistanceMatrixMap;
Graph g;
} // namespace

void initializeBoostFLoydWarshall() {

  const int vertices = 4;
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
  // set the distance matrix to receive the floyd warshall output
  DistanceMatrix distances(num_vertices(g));
  DistanceMatrixMap dm(distances, g);

 
}

// Benchmarking function.
static void Boost_FloydWarshall(benchmark::State &state) {

  for (auto _ : state) {
    WeightMap weight_pmap = boost::get(boost::edge_weight, g);

    // set the distance matrix to receive the floyd warshall output
    DistanceMatrix distances(num_vertices(g));
    DistanceMatrixMap dm(distances, g);
    for (int i = 0; i < state.range(0); ++i) {
      bool valid = floyd_warshall_all_pairs_shortest_paths(
          g, dm, boost::weight_map(weight_pmap));
    }
  }
}

// Register benchmarking function.
BENCHMARK(Boost_FloydWarshall)->Arg(1);

void generateResultBoostFLoydWarshall() {
  initializeBoostFLoydWarshall();
  WeightMap weight_pmap = boost::get(boost::edge_weight, g);

  // set the distance matrix to receive the floyd warshall output
  DistanceMatrix distances(num_vertices(g));
  DistanceMatrixMap dm(distances, g);
  cout << "-------------------------------------------------------\n";
  cout << "[ BOOST FloydWarshall Result Information ]\n";
  bool valid = floyd_warshall_all_pairs_shortest_paths(
      g, dm, boost::weight_map(weight_pmap));
  cout << "Boost FloydWarshall operation finished!\n";
}
