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

    // Register operands
    Value graph = op->getOperand(0);
    Value parent = op->getOperand(1);
    Value distance = op->getOperand(2);

    // Types
    IndexType idxt = IndexType::get(ctx);
    IntegerType it32 = IntegerType::get(ctx, 32);
    VectorType vt32 = VectorType::get({1000}, it32);
    VectorType qt = VectorType::get({1000}, idxt);

    // Constants
    Value idx0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value idx1 = rewriter.create<ConstantIndexOp>(loc, 1);

    Value V = rewriter.create<memref::DimOp>(loc, graph, idx0);

    Value zero = rewriter.create<ConstantIntOp>(loc, int(0), it32);
    Value one = rewriter.create<ConstantIntOp>(loc, int(1), it32);
    Value minusOne = rewriter.create<ConstantIntOp>(loc, int(-1), it32);
    Value five = rewriter.create<ConstantIntOp>(loc, int(5), it32);
    // Queue
    Value queue = rewriter.create<vector::BroadcastOp>(loc, qt, idx0);
    Value front = rewriter.create<ConstantIntOp>(loc, int(0), it32);
    Value rear = rewriter.create<ConstantIntOp>(loc, int(1), it32);

    // Visited array
    Value visited = rewriter.create<vector::BroadcastOp>(loc, vt32, zero);

    queue = rewriter.create<vector::InsertElementOp>(loc, idx0, queue, rear);
    rear = rewriter.create<AddIOp>(loc, rear, one);

    visited = rewriter.create<vector::InsertElementOp>(loc, one, visited, idx0);

    // While loop
    SmallVector<Value> operands = {queue, front, rear, visited};
    SmallVector<Type> types = {qt, it32, it32, vt32};
    SmallVector<Location> locations = {loc, loc, loc, loc};

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

      Value notEmpty =
          rewriter.create<CmpIOp>(loc, CmpIPredicate::ne, front, rear);

      rewriter.create<scf::ConditionOp>(loc, notEmpty, before->getArguments());
    }

    {
      rewriter.setInsertionPointToStart(&whileOp.getAfter().front());

      Value queue = after->getArgument(0);
      Value front = after->getArgument(1);
      Value rear = after->getArgument(2);
      Value visited = after->getArgument(3);

      Value u = rewriter.create<vector::ExtractElementOp>(loc, queue, front);
      front = rewriter.create<AddIOp>(loc, front, one);

      auto loop = rewriter.create<scf::ForOp>(
          loc, idx0, V, idx1, ValueRange{queue, front, rear, visited},
          [&](OpBuilder &builder, Location loc, Value v, ValueRange args) {
            Value queue = args[0];
            Value front = args[1];
            Value rear = args[2];
            Value visited = args[3];

            Value edge =
                builder.create<memref::LoadOp>(loc, graph, ValueRange{u, v});
            Value vis =
                builder.create<vector::ExtractElementOp>(loc, visited, v);

            Value present =
                builder.create<CmpIOp>(loc, CmpIPredicate::ne, edge, zero);
            Value nvisited =
                builder.create<CmpIOp>(loc, CmpIPredicate::eq, vis, zero);

            Value condition = builder.create<AndIOp>(loc, present, nvisited);

            scf::IfOp ifop = builder.create<scf::IfOp>(
                loc, TypeRange{qt, it32, it32, vt32}, condition, true);
            // Then block
            {
              builder.setInsertionPointToStart(ifop.thenBlock());

              // Logic
              Value dist = builder.create<memref::LoadOp>(loc, distance, u);
              Value p = builder.create<IndexCastOp>(loc, it32, u);
              dist = builder.create<AddIOp>(loc, dist, edge);

              builder.create<memref::StoreOp>(loc, dist, distance, v);
              builder.create<memref::StoreOp>(loc, p, parent, v);

              Value nvisited =
                  builder.create<vector::InsertElementOp>(loc, one, visited, v);
              Value nqueue =
                  builder.create<vector::InsertElementOp>(loc, v, queue, rear);
              Value nrear = builder.create<AddIOp>(loc, rear, one);

              builder.create<scf::YieldOp>(
                  loc, ValueRange{nqueue, front, nrear, nvisited});
            }
            // Else block
            {
              builder.setInsertionPointToStart(ifop.elseBlock());
              builder.create<scf::YieldOp>(
                  loc, ValueRange{queue, front, rear, visited});
            }

            builder.setInsertionPointAfter(ifop);
            ValueRange results = ifop.getResults();

            builder.create<scf::YieldOp>(loc, results);
          });

      ValueRange lresults = loop.getResults();

      rewriter.create<scf::YieldOp>(loc, lresults);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};

class GraphFloydWarshallLowering
    : public OpRewritePattern<graph::FloydWarshallOp> {
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

    SmallVector<int64_t, 8> step{1, 1};

    buildAffineLoopNest(
        rewriter, loc, ValueRange{c0, c0}, ValueRange{V, V}, step,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          Value x = builder.create<memref::LoadOp>(loc, input,
                                                   ValueRange{ivs[0], ivs[1]});
          builder.create<memref::StoreOp>(loc, x, output,
                                          ValueRange{ivs[0], ivs[1]});
        });

    SmallVector<Value, 8> lowerBounds(3, c0);
    SmallVector<Value, 8> upperBounds(3, V);
    SmallVector<int64_t, 8> steps{1, 1, 4};

    IntegerType i32 = IntegerType::get(ctx, 32);
    FloatType f32 = FloatType::getF32(ctx);
    VectorType vectorTy32 = VectorType::get({4}, f32);
    VectorType vectorred = VectorType::get({2, 4}, f32);
    Value one = rewriter.create<ConstantFloatOp>(loc, APFloat(float(1)), f32);
    Value mx =
        rewriter.create<ConstantFloatOp>(loc, APFloat(float(10000)), f32);
    Value vecOne = rewriter.create<vector::BroadcastOp>(loc, vectorTy32, one);
    Value vecMx = rewriter.create<vector::BroadcastOp>(loc, vectorTy32, mx);
    Value temp = rewriter.create<vector::BroadcastOp>(loc, vectorred, one);
    // rewriter.create<vector::PrintOp>(loc, vecOne);
    buildAffineLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          Value x = builder.create<memref::LoadOp>(loc, output,
                                                   ValueRange{ivs[1], ivs[0]});
          Value y = builder.create<memref::LoadOp>(loc, output,
                                                   ValueRange{ivs[0], ivs[2]});
          Value z = builder.create<memref::LoadOp>(loc, output,
                                                   ValueRange{ivs[1], ivs[2]});

          Value temp = builder.create<AddFOp>(loc, x, y);

          Value checkCond =
              builder.create<CmpFOp>(loc, CmpFPredicate::OLT, temp, z);

          builder.create<scf::IfOp>(
              loc, checkCond,
              [&](OpBuilder &builder, Location loc) {
                builder.create<memref::StoreOp>(loc, temp, output,
                                                ValueRange{ivs[1], ivs[2]});
                builder.create<scf::YieldOp>(loc);
              }
              // [&](OpBuilder &builder, Location loc){
              //   builder.create<scf::YieldOp>(loc);
              // }
          );
          // Value x = builder.create<memref::LoadOp>(loc, output,
          // ValueRange{ivs[1], ivs[0]}); Value vecik =
          // builder.create<vector::BroadcastOp>(loc, vectorTy32, x); Value
          // vecij = builder.create<vector::LoadOp>(loc, vectorTy32, output,
          // ValueRange{ivs[1], ivs[2]}); Value veckj =
          // builder.create<vector::LoadOp>(loc, vectorTy32, output,
          // ValueRange{ivs[0], ivs[2]}); Value vecikj =
          // builder.create<vector::FMAOp>(loc, veckj, vecOne, vecik); Value y =
          // builder.create<vector::InsertOp>(loc, vecij, temp,
          // ArrayRef<int64_t>{0}); Value z =
          // builder.create<vector::InsertOp>(loc, vecikj, y,
          // ArrayRef<int64_t>{1}); Value res =
          // builder.create<vector::MultiDimReductionOp>(loc, z, vecMx,
          // ArrayRef<bool>{true,false}, vector::CombiningKind::MINF);
          // // builder.create<vector::PrintOp>(loc, vecik);
          // // builder.create<vector::PrintOp>(loc, vecij);
          // // builder.create<vector::PrintOp>(loc, veckj);
          // // builder.create<vector::PrintOp>(loc, vecikj);
          // // builder.create<vector::PrintOp>(loc, res);
          // builder.create<vector::StoreOp>(loc, res, output,
          // ValueRange{ivs[1], ivs[2]});
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
