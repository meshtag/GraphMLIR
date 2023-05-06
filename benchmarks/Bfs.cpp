#include <benchmark/benchmark.h>
#include <bits/stdc++.h>

using namespace std;

#define V 1000

namespace {
int graph[V][V];
int parent[V];
int dist[V];
} // namespace

void bfs(int graph[V][V], int *parent, int *dist) {
  bool visited[V];

  for (int i = 0; i < V; i++)
    visited[i] = false;

  queue<int> q;

  visited[0] = true;
  q.push(0);

  while (!q.empty()) {
    int u = q.front();
    q.pop();

    for (int v = 0; v < V; v++) {
      if (visited[v] == false && graph[u][v] != 0) {
        visited[v] = true;

        dist[v] = dist[u] + graph[u][v];
        parent[v] = u;

        q.push(v);
      }
    }
  }
}

void initializeBfs() {
  int MAX_EDGES = V * (V - 1) / 2;

  for (int i = 0; i < MAX_EDGES; i++) {
    int u = rand() % V;
    int v = rand() % V;
    int d = rand() % 100;

    graph[u][v] = d;
  }
}

static void Bfs(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); i++) {
      bfs(graph, parent, dist);
    }
  }
}

BENCHMARK(Bfs)->Arg(1);

void generateResultBfs() {
  initializeBfs();
  cout << "-------------------------------------------------------\n";
  cout << "BFS Result Information ]\n";

  bfs(graph, parent, dist);

  for (int i = 0; i < V; i++)
    cout << parent[i] << " ";
  cout << endl;

  for (int i = 0; i < V; i++)
    cout << dist[i] << " ";
  cout << endl;

  cout << "BFS operation finished!\n";
}