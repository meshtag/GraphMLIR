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

// Size for visited and queue vector
#define MAXSIZE 1000

// Cast index to i32
Value indexToI32(OpBuilder &builder, Location loc, Value v) {
  return builder.create<IndexCastOp>(loc, builder.getI32Type(), v);
}

// Cast i32 to index
Value I32ToIndex(OpBuilder &builder, Location loc, Value v) {
  return builder.create<IndexCastOp>(loc, builder.getIndexType(), v);
}

// Add to index
Value addIndex(OpBuilder &builder, Location loc, Value u, Value d) {
  Value ans = indexToI32(builder, loc, u);
  ans = builder.create<AddIOp>(loc, ans, d);

  return I32ToIndex(builder, loc, ans);
}

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

    Value weights = op->getOperand(0);
    Value cnz = op->getOperand(1);
    Value cidx = op->getOperand(2);
    Value parent = op->getOperand(3);
    Value distance = op->getOperand(4);

    // Types
    IndexType idxt = IndexType::get(ctx);
    IntegerType it32 = IntegerType::get(ctx, 32);
    VectorType vt32 = VectorType::get({MAXSIZE}, it32);
    VectorType qt = VectorType::get({MAXSIZE}, idxt);

    // Constants
    Value idx0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value idx1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value cnzsize = rewriter.create<memref::DimOp>(loc, cnz, idx0);
    Value zero = rewriter.create<ConstantIntOp>(loc, int(0), it32);
    Value one = rewriter.create<ConstantIntOp>(loc, int(1), it32);
    Value two = rewriter.create<ConstantIntOp>(loc, int(2), it32);
    Value minusOne = rewriter.create<ConstantIntOp>(loc, int(-1), it32);

    // Number of vertices as index
    Value V = indexToI32(rewriter, loc, cnzsize);
    V = rewriter.create<AddIOp>(loc, V, minusOne);
    V = I32ToIndex(rewriter, loc, V);

    // Queue
    Value queue = rewriter.create<vector::BroadcastOp>(loc, qt, idx0);
    Value front = rewriter.create<ConstantIntOp>(loc, int(0), it32);
    Value rear = rewriter.create<ConstantIntOp>(loc, int(0), it32);

    // Visited
    // 0 = not discovered = white (not added to the queue)
    // 1 = discovered but no explored = grey (added to the queue)
    // 2 = discovered and explored = black (removed from the queue)
    Value visited = rewriter.create<vector::BroadcastOp>(loc, vt32, zero);

    /*
      queue[rear] = 0
      rear++
      visited[0] = 1
      distance[0] = 0
      parent[0] = -1
    */
    queue = rewriter.create<vector::InsertElementOp>(loc, idx0, queue, rear);
    rear = rewriter.create<AddIOp>(loc, rear, one);
    visited = rewriter.create<vector::InsertElementOp>(loc, one, visited, idx0);
    rewriter.create<memref::StoreOp>(loc, zero, distance, idx0);
    rewriter.create<memref::StoreOp>(loc, minusOne, parent, idx0);

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
    // while(front != rear)
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

      /*
        u = queue[front]
        z = u + 1
        front++
        visited[u] = 2
      */
      Value u = rewriter.create<vector::ExtractElementOp>(loc, queue, front);
      Value z = addIndex(rewriter, loc, u, one);

      front = rewriter.create<AddIOp>(loc, front, one);
      visited = rewriter.create<vector::InsertElementOp>(loc, two, visited, u);

      Value s = rewriter.create<memref::LoadOp>(loc, cnz, u);
      Value e = rewriter.create<memref::LoadOp>(loc, cnz, z);
      s = I32ToIndex(rewriter, loc, s);
      e = I32ToIndex(rewriter, loc, e);

      // for(i = s; i < e; i++)
      auto loop = rewriter.create<scf::ForOp>(
          loc, s, e, idx1, ValueRange{queue, front, rear, visited},
          [&](OpBuilder &builder, Location loc, Value i, ValueRange args) {
            Value queue = args[0];
            Value front = args[1];
            Value rear = args[2];
            Value visited = args[3];

            /*
              v = cidx[i]
              color = visited[v]
              condition = (color == 0)
            */
            Value v = rewriter.create<memref::LoadOp>(loc, cidx, i);
            v = I32ToIndex(builder, loc, v);
            Value color =
                rewriter.create<vector::ExtractElementOp>(loc, visited, v);
            Value condition =
                rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, color, zero);

            scf::IfOp ifop = builder.create<scf::IfOp>(
                loc, TypeRange{qt, it32, it32, vt32}, condition, true);

            // Then block
            {
              builder.setInsertionPointToStart(ifop.thenBlock());
              /*
                d = distance[u]
                e = weights[i]
                f = d + e
                p = u
              */
              Value d = rewriter.create<memref::LoadOp>(loc, distance, u);
              Value e = rewriter.create<memref::LoadOp>(loc, weights, i);
              Value f = rewriter.create<AddIOp>(loc, d, e);
              Value p = rewriter.create<IndexCastOp>(loc, it32, u);

              /*
                distance[v] = f
                parent[v] = u
              */
              rewriter.create<memref::StoreOp>(loc, f, distance, v);
              rewriter.create<memref::StoreOp>(loc, p, parent, v);

              /*
                visited[v] = 1
                queue[rear] = v
                rear++
              */
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

class GraphBellmanFordLowering : public OpRewritePattern<graph::BellmanFordOp> {
public:
  using OpRewritePattern<graph::BellmanFordOp>::OpRewritePattern;

  explicit GraphBellmanFordLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(graph::BellmanFordOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    Value start = op->getOperand(0);
    Value end = op->getOperand(1);
    Value distance = op->getOperand(2);
    Value output = op->getOperand(3);

    // Types
    IndexType idxt = IndexType::get(ctx);
    IntegerType it32 = IntegerType::get(ctx, 32);
    VectorType vt32 = VectorType::get({100}, it32);
    VectorType qt = VectorType::get({100}, idxt);

    Value idx0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value idx1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value V = rewriter.create<memref::DimOp>(loc, output, idx0);
    Value E = rewriter.create<memref::DimOp>(loc, start, idx0);

    Value zero = rewriter.create<ConstantIntOp>(loc, int(0), it32);
    Value one = rewriter.create<ConstantIntOp>(loc, int(1), it32);
    Value two = rewriter.create<ConstantIntOp>(loc, int(2), it32);
    Value minusOne = rewriter.create<ConstantIntOp>(loc, int(-1), it32);
    Value maxInt = rewriter.create<memref::LoadOp>(loc, output, idx0);

    SmallVector<Value, 8> lowerBounds{idx1, idx0};
    SmallVector<Value, 8> upperBounds{V, E};
    SmallVector<int64_t, 8> steps{1, 1};

    rewriter.create<memref::StoreOp>(loc, zero, output, idx0);

    buildAffineLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          Value u =
              rewriter.create<memref::LoadOp>(loc, start, ValueRange{ivs[1]});
          Value v =
              rewriter.create<memref::LoadOp>(loc, end, ValueRange{ivs[1]});
          Value d = rewriter.create<memref::LoadOp>(loc, distance,
                                                    ValueRange{ivs[1]});

          Value uidx = I32ToIndex(builder, loc, u);
          Value vidx = I32ToIndex(builder, loc, v);

          Value id = rewriter.create<memref::LoadOp>(loc, output, uidx);
          // distance[u] + d
          Value fd = rewriter.create<AddIOp>(loc, id, d);
          // distance[v]
          Value vd = rewriter.create<memref::LoadOp>(loc, output, vidx);

          Value condition1 =
              rewriter.create<CmpIOp>(loc, CmpIPredicate::ne, id, maxInt);
          Value condition2 =
              rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, fd, vd);
          Value condition =
              rewriter.create<AndIOp>(loc, condition1, condition2);

          builder.create<scf::IfOp>(
              loc, condition, [&](OpBuilder &builder, Location loc) {
                builder.create<memref::StoreOp>(loc, fd, output, vidx);
                builder.create<scf::YieldOp>(loc);
              });
        });

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
              loc, checkCond, [&](OpBuilder &builder, Location loc) {
                builder.create<memref::StoreOp>(loc, temp, output,
                                                ValueRange{ivs[1], ivs[2]});
                builder.create<scf::YieldOp>(loc);
              });
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
  patterns.add<GraphBellmanFordLowering>(patterns.getContext(), stride);
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
