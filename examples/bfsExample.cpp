//====- bfs.cpp - Example of graph-opt tool ========================//
//
// The graph.bfs operation will be compiled into an object file with the
// graph-opt tool.
// This file will be linked with the object file to generate the executable
// file.
//
//===----------------------------------------------------------------------===//

#include <Interface/graph/graph.h>
#include <Interface/graph/memref.h>
#include <Interface/core/Container.h>
#include <Interface/core/GraphContainer.h>
#include <vector>
#include <iostream>

int main() {
  // float sample_graph1_array[9] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
  // intptr_t sample_graph_length = 3;
  // intptr_t sample_graph_width = 3;
  // float *allocation_pointer = (float *)malloc(sizeof(float));
  // intptr_t sample_graph_sizes[2] = {sample_graph_width, sample_graph_length};
  // intptr_t sample_graph_strides[2] = {sample_graph_width, sample_graph_length};
  intptr_t no_of_elements;
  std::cout<<"enter the number of elements in graph"<<std::endl;
  std::cin>>no_of_elements;
  std::vector<Node*> adjListVec;
  Node* new_node;
  new_node->Vertex = 4;
  new_node->next = nullptr;
  adjListVec.push_back(new_node);
  
  Graph<float, 1> my_graph(adjListVec);

  MemRef_descriptor sample_graph =
      MemRef_Descriptor(my_graph.allocated, my_graph.aligned, 0,
                        my_graph.sizes, my_graph.strides);

  graph::graph_bfs(sample_graph, sample_graph, sample_graph);
}
