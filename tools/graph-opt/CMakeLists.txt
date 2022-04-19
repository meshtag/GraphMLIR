get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_tool(graph-opt
  graph-opt.cpp
)

target_link_libraries(graph-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  
  LowerGraphPass
  )
