#include <benchmark/benchmark.h>
#include <bits/stdc++.h>
#include <lemon/bellman_ford.h>
#include <lemon/list_graph.h>

using namespace std;
using namespace lemon;

#define V 100
#define MaxWeight 900
#define UpperLimit 100
#define LowerLimit 2

typedef ListDigraph::ArcMap<int> LengthMap;

namespace {
ListDigraph g;
ListDigraph::Node source;
LengthMap length(g);
ListDigraph::Node nodes[V];
} // namespace

void initializeLemonBellmanFord() {

  //   LengthMap length(g);

  for (int i = 0; i < V; i++) {
    nodes[i] = g.addNode();
  }

  source = nodes[0];

  //   length[g.addArc(nodes[0], nodes[1])] = 4;
  //   length[g.addArc(nodes[1], nodes[2])] = 3;
  //   length[g.addArc(nodes[2], nodes[3])] = 3;
  //   length[g.addArc(nodes[3], nodes[0])] = 6;
  //   length[g.addArc(nodes[0], nodes[2])] = 2;
  //   length[g.addArc(nodes[1], nodes[3])] = 2;

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
           container.find(reverse_p) != container.end()) {
      a = rand() % NUM;
      b = rand() % NUM;
      p = std::make_pair(a, b);
      reverse_p = std::make_pair(b, a);
    }

    container.insert(p);
    length[g.addArc(nodes[a], nodes[b])] = 1 + rand() % MaxWeight;
  }
  BellmanFord<ListDigraph, LengthMap> bf(g, length);
}

// Benchmarking function.
static void Lemon_BellmanFord(benchmark::State &state) {
  for (auto _ : state) {
    BellmanFord<ListDigraph, LengthMap> bf(g, length);
    for (int i = 0; i < state.range(0); ++i) {
      bf.run(source);
    }
  }
}

BENCHMARK(Lemon_BellmanFord)->Arg(1);

void generateResultLemonBellmanFord() {
  initializeLemonBellmanFord();
  cout << "-------------------------------------------------------\n";
  cout << "[ LEMON Bellman Ford Result Information ]\n";
  BellmanFord<ListDigraph, LengthMap> output(g, length);
  output.run(source);
  
    LengthMap::Value value = output.dist(nodes[2]);
        std::cout << "The distance of node t from node s: "
                << output.dist(nodes[3]) << std::endl;

  cout << "Lemon Bellman Ford Operation Completed!\n";
}