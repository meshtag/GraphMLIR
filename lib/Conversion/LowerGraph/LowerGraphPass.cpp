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
#include "mlir/IR/ImplicitLocOpBuilder.h"

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
    SmallVector<int64_t, 8> steps{1,1,4};

    IntegerType i32 = IntegerType::get(ctx, 32);
    FloatType f32 = FloatType::getF32(ctx);
    VectorType vectorTy32 = VectorType::get({4}, f32);
    VectorType vectorred = VectorType::get({2,4}, f32);
    Value one = rewriter.create<ConstantFloatOp>(loc, APFloat(float(1)), f32);
    Value mx = rewriter.create<ConstantFloatOp>(loc, APFloat(float(10000)), f32);
    Value vecOne = rewriter.create<vector::BroadcastOp>(loc, vectorTy32, one);
    Value vecMx = rewriter.create<vector::BroadcastOp>(loc, vectorTy32, mx);
    Value temp = rewriter.create<vector::BroadcastOp>(loc, vectorred, one);
    // rewriter.create<vector::PrintOp>(loc, vecOne);
    buildAffineLoopNest(
    rewriter, loc, lowerBounds, upperBounds, steps,
    [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value x = builder.create<memref::LoadOp>(loc, output, ValueRange{ivs[1], ivs[0]});
        Value y = builder.create<memref::LoadOp>(loc, output, ValueRange{ivs[0], ivs[2]});
        Value z = builder.create<memref::LoadOp>(loc, output, ValueRange{ivs[1], ivs[2]});

        Value temp = builder.create<AddFOp>(loc, x, y);

        Value checkCond = builder.create<CmpFOp>(loc, CmpFPredicate::OLT, temp, z);

        builder.create<scf::IfOp>(
          loc, checkCond, [&](OpBuilder &builder, Location loc) {
            builder.create<memref::StoreOp>(loc, temp, output, ValueRange{ivs[1], ivs[2]});
            builder.create<scf::YieldOp>(loc);
          }
          // [&](OpBuilder &builder, Location loc){
          //   builder.create<scf::YieldOp>(loc);
          // }
        );
        // Value x = builder.create<memref::LoadOp>(loc, output, ValueRange{ivs[1], ivs[0]});
        // Value vecik = builder.create<vector::BroadcastOp>(loc, vectorTy32, x);
        // Value vecij = builder.create<vector::LoadOp>(loc, vectorTy32, output, ValueRange{ivs[1], ivs[2]});
        // Value veckj = builder.create<vector::LoadOp>(loc, vectorTy32, output, ValueRange{ivs[0], ivs[2]});
        // Value vecikj = builder.create<vector::FMAOp>(loc, veckj, vecOne, vecik);
        // Value y = builder.create<vector::InsertOp>(loc, vecij, temp, ArrayRef<int64_t>{0});
        // Value z = builder.create<vector::InsertOp>(loc, vecikj, y, ArrayRef<int64_t>{1});
        // Value res = builder.create<vector::MultiDimReductionOp>(loc, z, vecMx, ArrayRef<bool>{true,false}, vector::CombiningKind::MINF);
        // // builder.create<vector::PrintOp>(loc, vecik);
        // // builder.create<vector::PrintOp>(loc, vecij);
        // // builder.create<vector::PrintOp>(loc, veckj);
        // // builder.create<vector::PrintOp>(loc, vecikj);
        // // builder.create<vector::PrintOp>(loc, res);
        // builder.create<vector::StoreOp>(loc, res, output, ValueRange{ivs[1], ivs[2]});

    });


    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};

class GraphMinSpanningTreeLowering : public OpRewritePattern<graph::MinSpanningTreeOp> {
public:
  using OpRewritePattern<graph::MinSpanningTreeOp>::OpRewritePattern;

  explicit GraphMinSpanningTreeLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(graph::MinSpanningTreeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();
    // ImplicitLocOpBuilder rewriter(loc, rewriter);
    
    /* operands */
    Value input = op->getOperand(0);
    Value output = op->getOperand(1);
    Value visited = op->getOperand(2);
    Value cost = op->getOperand(3);
    
    /* types */
    IndexType idx = IndexType::get(ctx);
    IntegerType i32 = IntegerType::get(ctx, 32);
    VectorType vi32 = VectorType::get({1000}, i32);
    VectorType vidx = VectorType::get({1000}, idx);

    /* constants */
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value zeroI = rewriter.create<arith::ConstantIntOp>(loc, int(0), i32);
    Value oneI = rewriter.create<arith::ConstantIntOp>(loc, int(1), i32);
    Value minusOneI = rewriter.create<arith::ConstantIntOp>(loc, int(-1), i32);
    Value minusTwoI = rewriter.create<arith::ConstantIntOp>(loc, int(-2), i32);
    Value maxI = rewriter.create<arith::ConstantIntOp>(loc, int(1000), i32);

    /* loop bounds */
    Value V = rewriter.create<memref::DimOp>(loc, input, c0);
    Value vAsInt = rewriter.create<IndexCastOp>(loc, i32, V);
    // since MST has V-1 edges
    Value eAsInt = rewriter.create<arith::AddIOp>(loc, vAsInt, minusOneI);
    Value E = rewriter.create<IndexCastOp>(loc, rewriter.getIndexType(), eAsInt);

    /* initial condition */
    // parent of root is root itself
    rewriter.create<memref::StoreOp>(loc, zeroI, output, c0);
  
    // loop through V-1 times since MST will have at least V-1 edges
    rewriter.create<scf::ForOp>(
      loc, c0, E, c1, ValueRange{},
      [&](OpBuilder &builder, Location loc, Value inductionVar, ValueRange iterArgs) {
        Value minCost = maxI;
        Value minIndex = minusTwoI;

        // finding the minimum weighted edge
        scf::ForOp forOp = builder.create<scf::ForOp>(
          loc, c0, V, c1, ValueRange{cost, visited, minIndex, minCost},
          [&](OpBuilder &builder, Location loc, Value iv, ValueRange args) {
            Value cost = args[0];
            Value visited = args[1];
            Value minIndex = args[2];
            Value minCost = args[3];

            Value costArg = builder.create<memref::LoadOp>(loc, cost, iv);
            Value visitedArg = builder.create<memref::LoadOp>(loc, visited, iv);

            // if vertex is unvisited and cost is lesser than current minimum cost
            Value visitedCondition = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, visitedArg, zeroI);
            Value costCondition = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, costArg, minCost);
            Value condition = builder.create<AndIOp>(loc, visitedCondition, costCondition);

            scf::IfOp ifOp = builder.create<scf::IfOp>(loc, TypeRange{i32, i32}, condition,
              [&](OpBuilder &builder, Location loc) {
                Value temp = builder.create<IndexCastOp>(loc, i32, iv);
                builder.create<scf::YieldOp>(loc, ValueRange{temp, costArg});
              },
              // else block
              [&](OpBuilder &builder, Location loc) {
                builder.create<scf::YieldOp>(loc, ValueRange{minIndex, minCost});
              });
            minIndex = ifOp.getResult(0);
            minCost = ifOp.getResult(1);

            builder.create<scf::YieldOp>(loc, ValueRange{cost, visited, minIndex, minCost});
          });
        Value minIndexFoundAsInt = forOp.getResult(2);
        Value minIndexFound = builder.create<IndexCastOp>(loc, builder.getIndexType(), minIndexFoundAsInt);
 
        // mark vertex as visited
        builder.create<memref::StoreOp>(loc, oneI, visited, minIndexFound);
        
        // adding the edge to output and updating weights
        builder.create<scf::ForOp>(
          loc, c0, V, c1, ValueRange{cost, visited, minIndexFound},
          [&](OpBuilder &builder, Location loc, Value iv, ValueRange args) {
            Value costVal = args[0];
            Value visited = args[1];
            Value minIndex = args[2];

            Value costArg = builder.create<memref::LoadOp>(loc, cost, iv);
            Value visitedArg = builder.create<memref::LoadOp>(loc, visited, iv);
            Value weight = builder.create<memref::LoadOp>(loc, input, ValueRange{minIndex, iv});

            // if vertex is unvisited, edge between current vertex and minIndex vertex exists, and edge weight is lesser than current cost for that vertex
            Value visitedCondition = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, visitedArg, zeroI);
            Value costCondition = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, weight, costArg);
            Value existsCondition = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, weight, zeroI);
            Value condition = builder.create<AndIOp>(loc, existsCondition, builder.create<AndIOp>(loc, visitedCondition, costCondition));
            
            builder.create<scf::IfOp>(loc, condition,
            [&](OpBuilder &builder, Location loc) {
              Value minIndexAsInt = builder.create<IndexCastOp>(loc, i32, minIndex);
              builder.create<memref::StoreOp>(loc, minIndexAsInt, output, iv);
              builder.create<memref::StoreOp>(loc, weight, cost, iv);
              builder.create<scf::YieldOp>(loc);
            });

          builder.create<scf::YieldOp>(loc, ValueRange{cost, visited, minIndex});
          });

        builder.create<scf::YieldOp>(loc);
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
  patterns.add<GraphMinSpanningTreeLowering>(patterns.getContext(), stride);
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
