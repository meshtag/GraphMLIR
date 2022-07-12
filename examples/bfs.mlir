// func @bfs_cycle_detection(%m1 : memref<?x?xf32>)
// {
//   graph.bfs DETECT_CYCLE %m1 : memref<?x?xf32> 
//   %c0 = arith.constant 0 : index
//   return 
// }

func @bfs_cycle_detection(%m1 : memref<?x?xf32>)
{
  graph.bfs %m1 : memref<?x?xf32> 
  %c0 = arith.constant 0 : index
  return 
}
