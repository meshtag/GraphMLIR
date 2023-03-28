func.func @bfs(%m1 : memref<?x?xf32>, %m2 : memref<?x?xf32>, %m3 : memref<?x?xf32>)
{
  graph.bfs %m1, %m2, %m3 : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32> 
  %c0 = arith.constant 0 : index
  return 
}

func.func @floyd_warshall(%input : memref<?x?xf32>, %output : memref<?x?xf32>)
{
  graph.FloydWarshall %input, %output : memref<?x?xf32>, memref<?x?xf32>
  return
}

