//====- floydWarshallExample.cpp =============================================//
//
// The graph.bfs operation will be compiled into an object file with the
// graph-opt tool.
// This file will be linked with the object file to generate the executable
// file.
//
//===----------------------------------------------------------------------===//

#include <Interface/GraphContainer.h>
#include <Interface/graph.h>
#include <Utility/Utils.h>
#include <iostream>
#include <vector>


int main() {

  int MAX_VERTICES = 20;
  Graph<float, 2> sample_graph(
      graph::detail::GRAPH_ADJ_MATRIX_UNDIRECTED_WEIGHTED, MAX_VERTICES);
  // sample_graph.addEdge(0, 1, 4);
  // sample_graph.addEdge(1, 2, 3);
  // sample_graph.addEdge(2, 3, 3);
  // sample_graph.addEdge(3, 0, 6);
  // sample_graph.addEdge(0, 2, 2);
  // sample_graph.addEdge(1, 3, 2);

  graph::generateRandomGraph(&sample_graph, MAX_VERTICES);

  // this will print the original graph.
  std::cout << "Printing graph in format it was entered ( "
               "GRAPH_ADJ_MARIX_UNDIRECTED_WEIGHTED )\n";
  sample_graph.printGraphOg();

  auto x = sample_graph.get_Memref();

  // this will print the linear 2d matrix in 2d form.
  std::cout
      << "Printing graph in form of 2d matrix after conversion to memref\n";
  sample_graph.printGraph();
  int vert = MAX_VERTICES;
  intptr_t size[2];
  size[0] = vert;
  size[1] = vert;
  MemRef<float, 2> output = MemRef<float, 2>(size);

  graph::floyd_warshall(&x, &output);
  auto y = output.getData();

  std::cout<<"Floyd Warshall Output!"<<"\n";
  
  for(int i=0; i<vert; i++){
    for(int j=0; j<vert; j++){
      std::cout<<y[i*vert + j]<<" ";
    }
    std::cout<<"\n";
  }
  x.release();

  std::cout << "End of the program! \n";
}
