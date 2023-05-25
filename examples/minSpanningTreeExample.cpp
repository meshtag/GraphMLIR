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

  int MAX_VERTICES = 5;
  Graph<int, 2> sample_graph(
      graph::detail::GRAPH_ADJ_MATRIX_UNDIRECTED_WEIGHTED, MAX_VERTICES);

    sample_graph.addEdge(0, 2, 1);
    sample_graph.addEdge(1, 3, 1);
    sample_graph.addEdge(1, 4, 2);
    sample_graph.addEdge(2, 1, 6);
    sample_graph.addEdge(2, 3, 3);
    sample_graph.addEdge(3, 4, 1);
    sample_graph.addEdge(4, 0, 1);

  std::cout << "Printing graph in format it was entered ( "
               "GRAPH_ADJ_MARIX_UNDIRECTED_WEIGHTED )\n";
  sample_graph.printGraphOg();
  std::cout
      << "Printing graph in form of 2d matrix after conversion to memref\n";
  sample_graph.printGraph();

  int V = MAX_VERTICES;
  intptr_t size[1];
  size[0] = V;

  MemRef<int, 1> output = MemRef<int, 1>(size, -1);
  
  // visited tracks vertices that have been visited so far. 0 for unvisited, 1 for visited
  MemRef<int, 1> visited = MemRef<int, 1>(size, 0);
  //  // key tracks the minimum weighted edge corresponding to each vertex, initially infinity (1000)
  MemRef<int, 1> cost = MemRef<int, 1>(size, 1000);

  // source vertex cost is set to 0 so that it is picked first
  cost[0] = 0;

  auto x = sample_graph.get_Memref();
  graph::min_spanning_tree(&x, &output, &visited, &cost);
  
  auto parent = output.getData();
  auto y = visited.getData();
  auto z = cost.getData();

  // expected output - [0 3 0 4 0]
  std::cout<<"\nMinimum Spanning Tree\n";
  for(int i=0; i<V; i++) {
    std::cout<<"parent[" << i << "] = " << parent[i] <<"\n";
  }
  cout << "\nVisited\n";
  for(int i=0; i<V; i++) {
    std::cout<<"visited[" << i << "] = " << y[i] <<"\n";
  }
  cout<<"\nCost\n";
  for(int i=0; i<V; i++) {
    std::cout<<"cost[" << i << "] = " << z[i] <<"\n";
  }
  x.release();
}
