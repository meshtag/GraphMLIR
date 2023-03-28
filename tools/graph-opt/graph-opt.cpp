//====- graph-opt.cpp - The driver of graph-mlir --------------------------===//
//
// This file is the dialect and oprimization driver of graph-mlir project.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Graph/GraphDialect.h"
#include "Graph/GraphOps.h"

// using namespace llvm;
// using namespace mlir;

namespace mlir {
namespace graph {
void registerLowerGraphPass();
} // namespace graph
namespace test {
  void registerTestVectorLowerings();
}
} // namespace mlir

namespace test {
void registerTestDialect(mlir::DialectRegistry &);
void registerTestTransformDialectExtension(mlir::DialectRegistry &);
} 

int main(int argc, char **argv) {
  // Register all MLIR passes.
  mlir::registerAllPasses();
  mlir::graph::registerLowerGraphPass();
  mlir::test::registerTestVectorLowerings();
  mlir::DialectRegistry registry;
  // Register all MLIR core dialects.
  registerAllDialects(registry);
  ::test::registerTestDialect(registry);
  ::test::registerTestTransformDialectExtension(registry);
  // Register dialects in graph-mlir project.
  // clang-format off
  registry.insert<graph::GraphDialect>();
  // clang-format on

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "graph-mlir optimizer driver", registry));
}
