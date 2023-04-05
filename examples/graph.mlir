func.func @bfs(%weights : memref<?xi32>, %cnz : memref<?xi32>, %cidx : memref<?xi32>, %parent : memref<?xi32>, %distance : memref<?xi32>)
{
  graph.bfs %weights, %cnz, %cidx, %parent, %distance : memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32> 
  return 
}

func.func @floyd_warshall(%input : memref<?x?xf32>, %output : memref<?x?xf32>)
{
  graph.FloydWarshall %input, %output : memref<?x?xf32>, memref<?x?xf32>
  return
}

