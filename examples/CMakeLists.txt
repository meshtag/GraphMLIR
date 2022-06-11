add_custom_command(OUTPUT bfs.o
COMMAND ${GraphMLIR_BINARY_DIR}/graph-opt ${GraphMLIR_EXAMPLES_DIR}/bfs.mlir -lower-graph -lower-affine -convert-scf-to-cf -convert-vector-to-llvm -convert-memref-to-llvm -convert-func-to-llvm='emit-c-wrappers=1' -reconcile-unrealized-casts | 
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${GraphMLIR_OPT_TRIPLE} -mattr=${GraphMLIR_OPT_ATTR} --filetype=obj -o ${GraphMLIR_BINARY_DIR}/../examples/bfs.o
DEPENDS graph-opt)

add_library(BFS STATIC bfs.o)

SET_TARGET_PROPERTIES(
BFS
PROPERTIES
LINKER_LANGUAGE C)

find_package(OpenCV REQUIRED CONFIG)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(bfsExample bfsExample.cpp)
add_dependencies(bfsExample graph-opt)
target_link_libraries(bfsExample ${OpenCV_LIBS} BFS)
