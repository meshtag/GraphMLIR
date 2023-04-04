#include <benchmark/benchmark.h>
#include <bits/stdc++.h>

using namespace std;

#define V 100

namespace {
int graph[V][V];
int parent[V];
int dist[V];
} // namespace

void bfs(int graph[V][V], int parent[V], int dist[V]) {
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
  for (int i = 0; i < (100 * 99) / 2; i++) {
    int u = rand() % 100;
    int v = rand() % 100;
    int d = rand() % 100;

    graph[u][v] = d;
    graph[v][u] = d;
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

  for (int x : parent)
    cout << x << " ";
  cout << endl;

  for (int x : dist)
    cout << x << " ";
  cout << endl;
  cout << "BFS operation finished!\n";
}