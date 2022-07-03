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

  AdjList_Unweighted_Undirected sample_AdjList;
  Graph<float,1> sample_graph(sample_AdjList);


  MemRef_descriptor sample_graph_memref =
      MemRef_Descriptor(sample_graph.allocated, sample_graph.aligned, 0,
                        sample_graph.sizes, sample_graph.strides);

  graph::graph_bfs(sample_graph_memref, sample_graph_memref, sample_graph_memref);
}
