//====- bfs.cpp - Example of graph-opt tool ========================//
//
// The graph.bfs operation will be compiled into an object file with the
// graph-opt tool.
// This file will be linked with the object file to generate the executable
// file.
//
//===----------------------------------------------------------------------===//

#include <Interface/Container.h>
#include <Interface/GraphContainer.h>
#include <Interface/graph.h>
#include <iostream>
#include <vector>

int main() {

  // Graph<float, 2>
  // sample_graph(graph::detail::GRAPH_ADJ_MATRIX_DIRECTED_WEIGHTED, 5);

  // use for unweighted graph
  // sample_graph.addEdge(0,2);
  // sample_graph.addEdge(2,3);
  // sample_graph.addEdge(3,2);
  // sample_graph.addEdge(2,2);
  // sample_graph.addEdge(1,2);

  // use for weighted graph
  Graph<float, 2> sample_graph(
      graph::detail::GRAPH_ADJ_MATRIX_DIRECTED_WEIGHTED, 5);
  sample_graph.addEdge(0, 2, 1);
  sample_graph.addEdge(2, 3, 3);
  sample_graph.addEdge(3, 2, 3);
  sample_graph.addEdge(2, 2, 6);
  sample_graph.addEdge(1, 2, 2);

  // this will print the original graph.
  std::cout << "Printing graph in format it was entered ( "
               "GRAPH_ADJ_MARIX_DIRECTED_WEIGHTED )\n";
  sample_graph.printGraphOg();

  auto x = sample_graph.get_Memref();

  // this will print the linear 2d matrix in 2d form.

  std::cout
      << "Printing graph in form of 2d matrix after conversion to memref\n";
  sample_graph.printGraph();
  graph::graph_bfs(x, x, x);
  x.release();
  std::cout << "End of the program! \n";
}
