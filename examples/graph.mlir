func.func @bfs(%graph : memref<?x?xi32>, %parent : memref<?xi32>, %distance : memref<?xi32>)
{
  graph.bfs %graph, %parent, %distance : memref<?x?xi32>, memref<?xi32>, memref<?xi32> 
  %c0 = arith.constant 0 : index
  return 
}

func.func @floyd_warshall(%input : memref<?x?xf32>, %output : memref<?x?xf32>)
{
  graph.FloydWarshall %input, %output : memref<?x?xf32>, memref<?x?xf32>
  return
}

