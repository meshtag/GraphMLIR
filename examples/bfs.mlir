func @bfs(%m1 : memref<?x?xf32>)
{
  graph.bfs %m1 : memref<?x?xf32> 
  %c0 = arith.constant 0 : index
  return 
}
