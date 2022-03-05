add_mlir_dialect_library(Graph
  GraphDialect.cpp
  GraphOps.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  )
