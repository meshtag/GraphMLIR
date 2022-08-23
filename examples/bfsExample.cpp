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
  
  	Graph<float, 2> sample_graph(graph::detail::GRAPH_ADJ_LIST_DIRECTED_WEIGHTED, 4);	

	// //use for unweighted graphs 
	// sample_graph.addEdge(0,2);
	// sample_graph.addEdge(2,3);
	// sample_graph.addEdge(3,2);
	// sample_graph.addEdge(2,2);
	// sample_graph.addEdge(1,2);
	
	//use for weighted graphs
	sample_graph.addEdge(0,2,1);
	sample_graph.addEdge(2,3,3);
	sample_graph.addEdge(3,2,3);
	sample_graph.addEdge(2,2,6);
	sample_graph.addEdge(1,2,2);

	//this will print the original graph.
	std::cout<<"Printing graph in format it was entered ( GRAPH_ADJ_LIST_DIRECTED_WEIGHTED )\n";
	sample_graph.printGraphOg();
	
	auto memref = sample_graph.graph_to_MemRef_descriptor();
	
	//this will print the linear 2d matrix in 2d form.

	std::cout<<"Printing graph in form of 2d matrix after conversion to memref\n";
	sample_graph.printGraph();
}
