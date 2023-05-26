//====- bfs.cpp - Example of graph-opt tool ========================//
//
// The graph.bfs operation will be compiled into an object file with the
// graph-opt tool.
// This file will be linked with the object file to generate the executable
// file.
//
//===----------------------------------------------------------------------===//

#include <Interface/Container.h>
#include <Interface/GraphContainer.h>
#include <Interface/graph.h>
#include <iostream>
#include <vector>

#define V 5

// Generate compressed sparse matrix
void generateCSR(std::vector<std::vector<int>> &graph, std::vector<int> &a,
                 std::vector<int> &ia, std::vector<int> &ca) {
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

int main() {
  std::vector<std::vector<int>> graph(V, std::vector<int>(V, 0));

  int MAX_EDGES = V * (V - 1) / 2;
  int NUMEDGES = MAX_EDGES;

  for (int i = 0; i < NUMEDGES; i++) {
    int u = rand() % V;
    int v = rand() % V;
    int d = rand() % 100 + 1;

    if (graph[u][v] == 0)
      graph[u][v] = d;
  }

  std::vector<int> a, ia, ca;

  generateCSR(graph, a, ia, ca);

  MemRef<int, 1> weights = MemRef<int, 1>(a);
  MemRef<int, 1> cnz = MemRef<int, 1>(ia);
  MemRef<int, 1> cidx = MemRef<int, 1>(ca);
  MemRef<int, 1> parent = MemRef<int, 1>(std::vector<int>(V, -1));
  MemRef<int, 1> distance = MemRef<int, 1>(std::vector<int>(V, INT32_MAX));

  graph::graph_bfs(&weights, &cnz, &cidx, &parent, &distance);

  auto y = parent.getData();

  // Print parents
  for (size_t i = 0; i < V; i++) {
    std::cout << "parent(" << i << ")"
              << " = " << parent[i] << std::endl;
  }

  y = distance.getData();

  // Print distance
  for (size_t i = 0; i < V; i++) {
    std::cout << "distance(" << i << ")"
              << " = " << distance[i] << std::endl;
  }
}