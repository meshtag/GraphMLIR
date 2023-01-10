memref.global "private" @kernel_4 : memref<4x4xf32> = dense<[[0.0,4.0,2.0,6.0],[4.0,0.0,3.0,2.0],[2.0,3.0,0.0,3.0],[6.0,2.0,3.0,0.0]]>
func.func private @printMemrefF32(memref<*xf32>)

func.func @main() -> () {
  %0 = arith.constant dense<[[1,1,1,1,1,41],[2,1,1,10,1,1]]> : vector<2x6xi32>
  %acc0 = arith.constant dense<[0,0,0,0,0,0]> : vector<6xi32>
  %res = vector.multi_reduction <maxui>, %0, %acc0 [0] : vector<2x6xi32> to vector<6xi32>
  vector.print %res : vector<6xi32>
  // %0 = arith.constant dense<[[1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0]]> : vector<2x6xf32>
  // %1 = arith.constant dense<[2.0,2.0,2.0,2.0,2.0,2.0]> : vector<6xf32>
  // %2 = vector.insert %1, %0[0] : vector<6xf32> into vector<2x6xf32> 
  // vector.print %0 : vector<2x6xf32>
  %krn0 = memref.get_global @kernel_4 : memref<4x4xf32>
  %krn = memref.cast %krn0 : memref<4x4xf32> to memref<?x?xf32>
  %output0 = memref.alloc() : memref<4x4xf32>
  %output = memref.cast %output0: memref<4x4xf32> to memref<?x?xf32>
  graph.FloydWarshall %krn, %output : memref<?x?xf32>, memref<?x?xf32>

  %print_mem =  memref.cast %output : memref<?x?xf32> to memref<*xf32>
  func.call @printMemrefF32(%print_mem) : (memref<*xf32>) -> ()
  return
}