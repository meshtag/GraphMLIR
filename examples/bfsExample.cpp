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
#include <Interface/Container.h>
#include <Interface/GraphContainer.h>
#include <vector>
#include <iostream>

int main() {
  // float sample_graph1_array[9] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
  // intptr_t sample_graph_length = 3;
  // intptr_t sample_graph_width = 3;
  // float *allocation_pointer = (float *)malloc(sizeof(float));
  // intptr_t sample_graph_sizes[2] = {sample_graph_width, sample_graph_length};
  // intptr_t sample_graph_strides[2] = {sample_graph_width, sample_graph_length};
  
  Graph<float, 4> sample_graph(graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_UNWEIGHTED);

  sample_graph.addEdge(0,1);
  sample_graph.addEdge(0,2);
  sample_graph.addEdge(1,2);
  sample_graph.addEdge(2,0);
  sample_graph.addEdge(2,3);
  sample_graph.addEdge(3,3);

  sample_graph.printGraph();
}
