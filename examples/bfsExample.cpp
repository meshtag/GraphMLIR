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
#include <iostream>
#include "../lib/Interface/GraphContainer.cpp"

int main() {
  std::cout << "Reached here !!!\n";

  // float sample_graph1_array[9] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
  // intptr_t sample_graph_length = 3;
  // intptr_t sample_graph_width = 3;
  // float *allocation_pointer = (float *)malloc(sizeof(float));
  // intptr_t sample_graph_sizes[2] = {sample_graph_width, sample_graph_length};
  // intptr_t sample_graph_strides[2] = {sample_graph_width, sample_graph_length};
  intptr_t no_of_elements;
  std::cout<<"enter the number of elements in graph"<<std::endl;
  std::cin>>no_of_elements;

  AdjGraph<float> my_graph(no_of_elements);
  //TODO for Adjacency List Directed 
  // Call functions to create graph
  // Convert data to Memref for passing in mlir
  // MemRef_descriptor sample_graph =
  //     MemRef_Descriptor(allocation_pointer, sample_graph1_array, 0,
  //                       sample_graph_sizes, sample_graph_strides);

  // graph::graph_bfs(sample_graph, sample_graph, sample_graph);
}
