# find_package(LEMON)

# add_subdirectory(${LEMON_DIR} ./lemon)
# include_directories(${LEMON_DIR})

add_custom_command(OUTPUT graph.o
COMMAND ${GraphMLIR_BINARY_DIR}/graph-opt ${GraphMLIR_EXAMPLES_DIR}/graph.mlir 
        -lower-graph -lower-affine -convert-scf-to-cf 
        -convert-vector-to-llvm -convert-memref-to-llvm 
        -convert-func-to-llvm='emit-c-wrappers=1' 
        -reconcile-unrealized-casts | 
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${GraphMLIR_OPT_TRIPLE} -mattr=${GraphMLIR_OPT_ATTR} 
        --filetype=obj -o ${GraphMLIR_BINARY_DIR}/../benchmarks/graph.o
DEPENDS graph-opt)

add_library(graph STATIC graph.o)

SET_TARGET_PROPERTIES(
graph
PROPERTIES
LINKER_LANGUAGE C)

add_executable(graph-processing-benchmark
  GraphMlirFloydWarshallBenchmark.cpp
  FloydWarshall.cpp
  BoostFloydWarshall.cpp
  LemonBFS.cpp
  Main.cpp

)

# target_link_directories(graph-processing-benchmark PRIVATE ${LEMON_DIR}/build/lemon/)
# target_link_directories(graph-processing-benchmark PRIVATE ${GraphMLIR_BINARY_DIR}/../build/lemon/lemon/)

find_package(Boost 1.40 COMPONENTS program_options REQUIRED)

target_link_libraries(graph-processing-benchmark
  PRIVATE
  emon
  Boost::program_options
  graph
  GoogleBenchmark
)
