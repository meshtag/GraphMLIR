
  memref.global "private" @in : memref<5x5xf32> = dense<[[0. , 0. , 1. , 0. , 1.],
                                                        [0., 0., 7., 1. , 2.],
                                                        [1., 7., 0., 3. , 0.],
                                                        [0., 1., 3., 0. , 1.],
                                                        [1., 2., 0., 1. , 0.]]>

  memref.global "private" @out : memref<5xi32> = dense<[-1, -1, -1, -1, -1]>


func.func @min_spanning_tree(%input : memref<5x5xf32>, %output : memref<5xi32>)
{
  graph.MinSpanningTree %input, %output : memref<5x5xf32>, memref<5xi32>
  return
}


  func.func @main() {
    %input = memref.get_global @in : memref<5x5xf32>
    %output = memref.get_global @out : memref<5xi32>
    func.call @min_spanning_tree(%input, %output) : (memref<5x5xf32>, memref<5xi32>) -> ()
    func.return
    }