add_custom_command(OUTPUT graph.o
COMMAND ${GraphMLIR_BINARY_DIR}/graph-opt ${GraphMLIR_EXAMPLES_DIR}/graph.mlir 
        -lower-graph -lower-affine -convert-scf-to-cf 
        -convert-vector-to-llvm -convert-memref-to-llvm 
        -convert-func-to-llvm='emit-c-wrappers=1' 
        -reconcile-unrealized-casts | 
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${GraphMLIR_OPT_TRIPLE} -mattr=${GraphMLIR_OPT_ATTR} 
        --filetype=obj -o ${GraphMLIR_BINARY_DIR}/../examples/graph.o
DEPENDS graph-opt)

add_library(GRAPH STATIC graph.o)

SET_TARGET_PROPERTIES(
GRAPH
PROPERTIES
LINKER_LANGUAGE C)

add_executable(bfsExample bfsExample.cpp)
add_dependencies(bfsExample graph-opt)
target_link_libraries(bfsExample GRAPH)

add_executable(floydWarshallExample floydWarshallExample.cpp)
add_dependencies(floydWarshallExample graph-opt)
target_link_libraries(floydWarshallExample GRAPH)
