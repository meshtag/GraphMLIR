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
  // use for weighted graph
  Graph<int, 2> sample_graph(graph::detail::GRAPH_ADJ_MATRIX_DIRECTED_WEIGHTED,
                             5);

  sample_graph.addEdge(0, 1, 1);
  sample_graph.addEdge(1, 2, 3);
  sample_graph.addEdge(2, 3, 3);
  sample_graph.addEdge(3, 4, 6);

  // this will print the original graph.
  std::cout << "Printing graph in format it was entered ( "
               "GRAPH_ADJ_MARIX_DIRECTED_WEIGHTED )\n";
  sample_graph.printGraphOg();

  auto graph = sample_graph.get_Memref();

  // Distance and Parent vector
  intptr_t size[1] = {5};

  MemRef<int, 1> parent = MemRef<int, 1>(size);
  MemRef<int, 1> distance = MemRef<int, 1>(size);

  // this will print the linear 2d matrix in 2d form.
  std::cout
      << "Printing graph in form of 2d matrix after conversion to memref\n";
  sample_graph.printGraph();

  graph::graph_bfs(&graph, &parent, &distance);

  std::cout << "Distance\n";
  for (int i = 0; i < 5; i++) {
    std::cout << distance[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "\nParent\n";
  for (int i = 0; i < 5; i++) {
    std::cout << parent[i] << " ";
  }
  std::cout << std::endl;

  graph.release();
  parent.release();
  distance.release();

  std::cout << "End of the program! \n";
}
