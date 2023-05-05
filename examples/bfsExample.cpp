//====- bfs.cpp - Example of graph-opt tool ========================//
//
// The graph.bfs operation will be compiled into an object file with the
// graph-opt tool.
// This file will be linked with the object file to generate the executable
// file.
//
//===----------------------------------------------------------------------===//

#include <Interface/graph.h>
#include <Interface/memref.h>
#include </home/andros/GraphMLIR/include/Interface/Container.h>
#include </home/andros/GraphMLIR/include/Interface/GraphContainer.h>
#include <iostream>

int main() {
  // std::cout << "Reached here !!!\n";

  // float sample_graph1_array[9] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
  // intptr_t sample_graph_length = 3;
  // intptr_t sample_graph_width = 3;
  // float *allocation_pointer = (float *)malloc(sizeof(float));
  // intptr_t sample_graph_sizes[2] = {sample_graph_width, sample_graph_length};
  // intptr_t sample_graph_strides[2] = {sample_graph_width, sample_graph_length};

  // MemRef_descriptor sample_graph =
  //     MemRef_Descriptor(allocation_pointer, sample_graph1_array, 0,
  //                       sample_graph_sizes, sample_graph_strides);

  // graph::graph_bfs(sample_graph, sample_graph, sample_graph);

  int n, m;
  float adjMatrix[1000][1000] = {0};

  std::cout << "Enter the number of nodes and edges: \n";
  std::cin >> n >> m;

  std::cout << "Enter the edges of the graph.\n";
  for (int i = 0; i < m; ++i) {
    int x, y;
    std::cin >> x >> y;
    adjMatrix[x][y] = 1;
    adjMatrix[y][x] = 1;
  }

  int inputSize = n * n;
  float *inputGraph = (float *)malloc(inputSize * sizeof(float));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      inputGraph[n * i + j] = (float)adjMatrix[i][j];
    }
  }

  intptr_t nodes = n;

  float *allocation_pointer = (float *)malloc(sizeof(float));

  intptr_t graph_sizes[2] = {nodes, nodes};
  intptr_t graph_strides[2] = {nodes, nodes};

  Graph<float, 1> adjMatrix_graph(nodes, &inputGraph);

  MemRef_descriptor input_graph = MemRef_Descriptor(
    allocation_pointer, inputGraph, 0, graph_sizes, graph_strides);

  graph::graph_bfs(input_graph, input_graph, input_graph);

  std::cout << "Graph using adjacency matrix created! \n";
}
