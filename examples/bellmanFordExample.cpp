#include <Interface/Container.h>
#include <Interface/GraphContainer.h>
#include <Interface/graph.h>
#include <iostream>
#include <vector>

#define V 100

// Bellman Ford basic implementation
void bellman_ford(std::vector<int> &st, std::vector<int> &ed,
                  std::vector<int> &dist, std::vector<int> &output1) {
  int E = st.size();
  output1[0] = 0;

  for (int i = 1; i < V; i++) {
    for (int j = 0; j < E; j++) {
      int u = st[j];
      int v = ed[j];
      int d = dist[j];

      if (output1[u] != INT16_MAX && output1[u] + d < output1[v])
        output1[v] = output1[u] + d;
    }
  }

  for (int j = 0; j < E; j++) {
    int u = st[j];
    int v = ed[j];
    int d = dist[j];

    if (output1[u] != INT16_MAX && output1[u] + d < output1[v])
      return;
  }
}

int main() {
  int MAX_EDGES = V * (V - 1) / 2;
  int NUMEDGES = MAX_EDGES;

  std::vector<int> st, ed, dist, op(V, INT16_MAX);

  for (int i = 0; i < NUMEDGES; i++) {
    int u = rand() % V;
    int v = rand() % V;
    int d = (rand() % 100) - 50;

    st.push_back(u);
    ed.push_back(v);
    dist.push_back(d);
  }

  std::vector<int> output1(V, INT16_MAX);
  bellman_ford(st, ed, dist, output1);

  std::cout << st[0] << " " << ed[0] << " " << dist[0] << std::endl;

  intptr_t size[1] = {V};

  MemRef<int, 1> start = MemRef<int, 1>(st);
  MemRef<int, 1> end = MemRef<int, 1>(ed);
  MemRef<int, 1> distance = MemRef<int, 1>(dist);
  MemRef<int, 1> output = MemRef<int, 1>(op);

  graph::bellman_ford(&start, &end, &distance, &output);

  auto y = output.getData();

  for (int i = 0; i < V; i++) {
    std::cout << (y[i] == output1[i]) << " ";
  }
  std::cout << std::endl;
}