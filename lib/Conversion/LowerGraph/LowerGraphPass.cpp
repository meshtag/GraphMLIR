//====- LowerGraphPass.cpp - graph Dialect Lowering Pass
//---------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines Graph dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "Graph/GraphDialect.h"
#include "Graph/GraphOps.h"

using namespace mlir;
using namespace graph;
using namespace vector;
using namespace mlir::arith;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class GraphBFSLowering : public OpRewritePattern<graph::BFSOp> {
public:
  using OpRewritePattern<graph::BFSOp>::OpRewritePattern;

  explicit GraphBFSLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(graph::BFSOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Registering operand types
    Value graph = op->getOperand(0);
    Value parent = op->getOperand(1);
    Value distance = op->getOperand(2);

    Value idx0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value idx1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value V = rewriter.create<memref::DimOp>(loc, graph, idx0);

    IndexType idxt = IndexType::get(ctx);
    IntegerType it32 = IntegerType::get(ctx, 32);
    VectorType vt32 = VectorType::get({1000}, it32);

    Value minusOne = rewriter.create<ConstantIntOp>(loc, int(-1), it32);
    Value zero = rewriter.create<ConstantIntOp>(loc, int(0), it32);
    Value one = rewriter.create<ConstantIntOp>(loc, int(1), it32);

    // Queue implementation
    VectorType qt = VectorType::get({1000}, idxt);
    Value queue = rewriter.create<vector::BroadcastOp>(loc, qt, idx0);
    Value front = rewriter.create<ConstantIntOp>(loc, int(0), it32);
    Value rear = rewriter.create<ConstantIntOp>(loc, int(0), it32);

    // Visited array
    Value visited = rewriter.create<vector::BroadcastOp>(loc, vt32, zero);

    queue = rewriter.create<vector::InsertElementOp>(loc, idx0, queue, rear);
    rear = rewriter.create<AddIOp>(loc, rear, one);

    SmallVector<Value> operands = {queue, front, rear, visited};
    SmallVector<Type> types = {qt, it32, it32, vt32};
    SmallVector<Location> locations = {loc, loc, loc, loc};

    SmallVector<Value> lbs = {idx0};
    SmallVector<Value> ubs = {V};
    SmallVector<int64_t> steps = {1};

    // While loop
    auto whileOp = rewriter.create<scf::WhileOp>(loc, types, operands);
    Block *before =
        rewriter.createBlock(&whileOp.getBefore(), {}, types, locations);
    Block *after =
        rewriter.createBlock(&whileOp.getAfter(), {}, types, locations);

    // Before block - Condition
    {
      rewriter.setInsertionPointToStart(&whileOp.getBefore().front());
      Value front = before->getArgument(1);
      Value rear = before->getArgument(2);

      Value queueNotEmpty =
          rewriter.create<CmpIOp>(loc, CmpIPredicate::ne, front, rear);
      rewriter.create<scf::ConditionOp>(loc, queueNotEmpty,
                                        before->getArguments());
    }
    // After block
    {
      rewriter.setInsertionPointToStart(&whileOp.getAfter().front());
      Value queue = after->getArgument(0);
      Value front = after->getArgument(1);
      Value rear = after->getArgument(2);
      Value visited = after->getArgument(3);

      // Code logic here
      Value u = rewriter.create<vector::ExtractElementOp>(loc, queue, front);
      front = rewriter.create<AddIOp>(loc, front, one);

      buildAffineLoopNest(
          rewriter, loc, lbs, ubs, steps,
          [&](OpBuilder &builder, Location loc, ValueRange ivr) {
            Value edge = builder.create<memref::LoadOp>(loc, graph,
                                                        ValueRange{u, ivr[0]});
            Value visited =
                builder.create<vector::ExtractElementOp>(loc, visited, ivr[0]);
          });

      // Termination step for after block
      rewriter.create<scf::YieldOp>(loc,
                                    ValueRange({queue, front, rear, visited}));
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};
} // end anonymous namespace

void populateLowerGraphConversionPatterns(RewritePatternSet &patterns,
                                          int64_t stride) {
  patterns.add<GraphBFSLowering>(patterns.getContext(), stride);
}

//===----------------------------------------------------------------------===//
// LowerGraphPass
//===----------------------------------------------------------------------===//

namespace {
class LowerGraphPass
    : public PassWrapper<LowerGraphPass, OperationPass<ModuleOp>> {
public:
  LowerGraphPass() = default;
  LowerGraphPass(const LowerGraphPass &) {}
  explicit LowerGraphPass(int64_t strideParam) { stride = strideParam; }

  StringRef getArgument() const final { return "lower-graph"; }
  StringRef getDescription() const final { return "Lower Graph Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<graph::GraphDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect, VectorDialect,
                    AffineDialect, arith::ArithmeticDialect>();
  }

  Option<int64_t> stride{*this, "Graph-strip-mining",
                         llvm::cl::desc("Strip mining size."),
                         llvm::cl::init(32)};
};
} // end anonymous namespace.

void LowerGraphPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<AffineDialect, scf::SCFDialect, func::FuncDialect,
                         memref::MemRefDialect, VectorDialect,
                         arith::ArithmeticDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();

  RewritePatternSet patterns(context);
  populateLowerGraphConversionPatterns(patterns, stride);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace graph {
void registerLowerGraphPass() { PassRegistration<LowerGraphPass>(); }
} // namespace graph
} // namespace mlir
