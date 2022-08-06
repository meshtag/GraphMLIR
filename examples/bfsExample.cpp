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
  intptr_t no_of_elements = 3;
  // std::cout<<"enter the number of elements in graph"<<std::endl;
  // std::cin>>no_of_elements;

  
  std::cout<<"Example for weighted graph : \n";
  //TODO for Adjacency List Directed 
  // Call functions to create weighted graph
  Graph<float, 2> my_graph(3, no_of_elements);
  my_graph.AddEdge(0, 1, 2);
  my_graph.AddEdge(0, 2, 3);
  my_graph.AddEdge(1, 2, 4);
  my_graph.AddEdge(2, 0, 6);
  // std::cout<<"Example for unweighted graph : \n";
  // // Call functions to create unweighted graph
  // Graph<float, 2> my_graph(2, no_of_elements);
  // my_graph.AddEdge(0, 1);
  // my_graph.AddEdge(0, 2);
  // my_graph.AddEdge(1, 2);
  // my_graph.AddEdge(2, 0);
  std::cout<<"Graph in Adjacency List format : \n";
  my_graph.PrintGraph();
  GraphToMemrefConversion(my_graph);
  std::cout<<"Graph in Adjacency Matrix Format : \n";
  PrintGraphInMemrefConversion(my_graph);
  // Convert data to Memref for passing in mlir
  // MemRef_descriptor sample_graph =
  //     MemRef_Descriptor(allocation_pointer, sample_graph1_array, 0,
  //                       sample_graph_sizes, sample_graph_strides);

  // graph::graph_bfs(sample_graph, sample_graph, sample_graph);
}
