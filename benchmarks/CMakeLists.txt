ADD_SUBDIRECTORY(${LEMON_DIR} ./lemon)
SET(LEMON_INCLUDE_DIRS
  ${LEMON_DIR}
  ${CMAKE_BINARY_DIR}/benchmarks/lemon
)
SET(LEMON_LIBRARIES lemon)
UNSET(LEMON_ROOT_DIR CACHE)
UNSET(LEMON_DIR CACHE)
UNSET(LEMON_INCLUDE_DIR CACHE)
UNSET(LEMON_LIBRARY CACHE)

INCLUDE_DIRECTORIES(
     ${LEMON_INCLUDE_DIRS}
)

add_custom_command(OUTPUT graph.o
COMMAND ${GraphMLIR_BINARY_DIR}/graph-opt ${GraphMLIR_EXAMPLES_DIR}/graph.mlir 
        -lower-graph -lower-affine -convert-scf-to-cf 
        -test-vector-multi-reduction-lowering-patterns 
        -convert-vector-to-llvm 
        -convert-memref-to-llvm
        -llvm-request-c-wrappers
        -convert-func-to-llvm
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
  FloydWarshall.cpp
  BoostFloydWarshall.cpp
  LemonBFS.cpp
  GraphMlirFloydWarshallBenchmark.cpp
  Main.cpp

)


find_package(Boost 1.40 COMPONENTS program_options REQUIRED)

target_link_libraries(graph-processing-benchmark
  PRIVATE
  ${LEMON_LIBRARIES}
  Boost::program_options
  graph
  GoogleBenchmark
)
