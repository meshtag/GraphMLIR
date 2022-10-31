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

    // Create constant indices.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    // Register operand values.
    Value m1 = op->getOperand(0);
    Value m2 = op->getOperand(1);
    Value m3 = op->getOperand(2);

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};

class GraphFloydWarshallLowering : public OpRewritePattern<graph::FloydWarshallOp> {
public:
  using OpRewritePattern<graph::FloydWarshallOp>::OpRewritePattern;

  explicit GraphFloydWarshallLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(graph::FloydWarshallOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Register operand values.
    Value input = op->getOperand(0);
    Value output = op->getOperand(1);

    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value V = rewriter.create<memref::DimOp>(loc, input, c0);

    SmallVector<int64_t, 8> step{1,1};

    buildAffineLoopNest(
    rewriter, loc, ValueRange{c0, c0}, ValueRange{V, V}, step,
    [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value x = builder.create<memref::LoadOp>(loc, input, ValueRange{ivs[0], ivs[1]});
        builder.create<memref::StoreOp>(loc, x, output, ValueRange{ivs[0], ivs[1]});
    });
    
    SmallVector<Value, 8> lowerBounds(3, c0);
    SmallVector<Value, 8> upperBounds(3, V);
    SmallVector<int64_t, 8> steps{1,1,1};
    
    buildAffineLoopNest(
    rewriter, loc, lowerBounds, upperBounds, steps,
    [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value x = builder.create<memref::LoadOp>(loc, output, ValueRange{ivs[1], ivs[0]});
        Value y = builder.create<memref::LoadOp>(loc, output, ValueRange{ivs[0], ivs[2]});
        Value z = builder.create<memref::LoadOp>(loc, output, ValueRange{ivs[1], ivs[2]});

        Value temp = builder.create<AddIOp>(loc, x, y);

        Value checkCond = builder.create<CmpIOp>(loc, CmpIPredicate::slt, temp, z);

        builder.create<scf::IfOp>(
          loc, checkCond, [&](OpBuilder &builder, Location loc) {
            builder.create<memref::StoreOp>(loc, temp, output, ValueRange{ivs[1], ivs[2]});
            builder.create<scf::YieldOp>(loc);
          }
          // [&](OpBuilder &builder, Location loc){
          //   builder.create<scf::YieldOp>(loc);
          // }
        );

    });


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
  patterns.add<GraphFloydWarshallLowering>(patterns.getContext(), stride);
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
