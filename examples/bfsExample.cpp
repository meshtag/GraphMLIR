#include <Interface/graph.h>
#include <Interface/memref.h>
#include <Interface/Container.h>
#include <Interface/GraphContainer.h>
#include <vector>
#include <iostream>

int main() {
  /*std::cout << "Reached here !!!\n";

  float sample_graph1_array[9] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
  intptr_t sample_graph_length = 3;
  intptr_t sample_graph_width = 3;
  float *allocation_pointer = (float *)malloc(sizeof(float));
  intptr_t sample_graph_sizes[2] = {sample_graph_width, sample_graph_length};
  intptr_t sample_graph_strides[2] = {sample_graph_width, sample_graph_length};

  MemRef_descriptor sample_graph =
      MemRef_Descriptor(allocation_pointer, sample_graph1_array, 0,
                        sample_graph_sizes, sample_graph_strides);

  graph::graph_bfs(sample_graph, sample_graph, sample_graph);*/

  	Graph<float, 2> sample_graph(graph::detail::GRAPH_INC_MATRIX_UNDIRECTED_UNWEIGHTED, 4);	

	sample_graph.addEdge(0,2);
	sample_graph.addEdge(1,2);
	sample_graph.addEdge(2,3);
	sample_graph.addEdge(3,2);

	auto memref = graph_to_MemRef_descriptor(sample_graph);

	//this will print the linear 2d matrix in 2d form.
	sample_graph.printGraph();
}