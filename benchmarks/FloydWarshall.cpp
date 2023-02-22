//===- FloydWarshall.cpp --------------------------------------------------===//
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
// This file implements the benchmark for Floyd Warshall Benchmark.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <cstring>
#include <iostream>
// #include <Utility/Utils.h>

using namespace std;

#define V 100

namespace {
int input[V][V];
int output[V][V];
} // namespace

void floydWarshall(int graph[][V], int dist[][V]) {

  int i, j, k;

  for (i = 0; i < V; i++)
    for (j = 0; j < V; j++)
      dist[i][j] = graph[i][j];

  for (k = 0; k < V; k++) {
    for (i = 0; i < V; i++) {
      for (j = 0; j < V; j++) {
        if (dist[i][j] > (dist[i][k] + dist[k][j]))
          dist[i][j] = dist[i][k] + dist[k][j];
      }
    }
  }
}

void initializeFloydWarshall() {
  int data[4][4] = {{0, 4, 2, 6}, {4, 0, 3, 2}, {2, 3, 0, 3}, {5, 2, 3, 0}};

  for (int i = 0; i < V; i++) {
    for (int j = 0; j < V; j++) {
      input[i][j] = 100;
    }
  }
  memset(output, 0, sizeof(output));
}

// Benchmarking function.
static void FloydWarshall(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      floydWarshall(input, output);
    }
  }
}

// Register benchmarking function.
BENCHMARK(FloydWarshall)->Arg(1);

void generateResultFloydWarshall() {
  initializeFloydWarshall();
  cout << "-------------------------------------------------------\n";
  cout << "[Floyd Warshall Result Information ]\n";

  floydWarshall(input, output);

  for (int i = 0; i < V; i++) {
    for (int j = 0; j < V; j++) {
      cout << output[i][j] << " ";
    }
    cout << "\n";
  }
  cout << "FLoyd Warshall operation finished!\n";
}
