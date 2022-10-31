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
#include <iostream>
#include <vector>

int main() {

  Graph<int, 2> sample_graph(
      graph::detail::GRAPH_ADJ_MATRIX_UNDIRECTED_WEIGHTED, 4);
  sample_graph.addEdge(0, 1, 4);
  sample_graph.addEdge(1, 2, 3);
  sample_graph.addEdge(2, 3, 3);
  sample_graph.addEdge(3, 0, 6);
  sample_graph.addEdge(0, 2, 2);
  sample_graph.addEdge(1, 3, 2);

  // this will print the original graph.
  std::cout << "Printing graph in format it was entered ( "
               "GRAPH_ADJ_MARIX_UNDIRECTED_WEIGHTED )\n";
  sample_graph.printGraphOg();

  auto x = sample_graph.get_Memref();

  // this will print the linear 2d matrix in 2d form.
  std::cout
      << "Printing graph in form of 2d matrix after conversion to memref\n";
  sample_graph.printGraph();

  intptr_t size[2];
  size[0] = 4;
  size[1] = 4;
  MemRef<int, 2> output = MemRef<int, 2>(size);

  graph::_mlir_ciface_floyd_warshall(&x, &output);
  auto y = output.getData();

  std::cout<<"Floyd Warshall Output!"<<"\n";
  
  for(int i=0; i<4; i++){
    for(int j=0; j<4; j++){
      std::cout<<y[i*4 + j]<<" ";
    }
    std::cout<<"\n";
  }
  x.release();

  std::cout << "End of the program! \n";
}
