//===- GraphOps.td - graph Dialect Ops -------------------------*- tablegen -*-===//
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
// This is the top level file for operations in the Graph dialect.
//
//===----------------------------------------------------------------------===//

#ifndef Graph_GraphOPS_TD
#define Graph_GraphOPS_TD

include "GraphDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Graph_BFSOp : Graph_Op<"bfs">
{
  let summary = [{
      abcde
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       Arg<AnyRankedOrUnrankedMemRef, "kernelMemref",
                           [MemRead]>:$memrefK,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCO);

  let assemblyFormat = [{
    $memrefI `,` $memrefK `,` $memrefCO attr-dict `:` type($memrefI) `,` type($memrefK) `,` type($memrefCO) 
  }];
}

def Graph_FloydWarshallOp : Graph_Op<"FloydWarshall">
{
  let summary = [{The Floyd Warshall Algorithm is for solving all pairs shortest path problems. 
  The problem is to find the shortest distances between every pair of vertices in a given 
  edge-weighted directed Graph. 
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCO);

  let assemblyFormat = [{
    $memrefI `,` $memrefCO attr-dict `:` type($memrefI) `,` type($memrefCO) 
  }];
}

def Graph_MinSpanningTreeOp: Graph_Op<"MinSpanningTree">
{
  let summary = [{
  Minimum spanning tree (MST) is a subset of the edges of the graph that includes all vertices of the graph, 
  forming a tree such that it has the sum of edge weights is the minimum.
  This implementation uses the Prim-Jarnik's algorithm for evaluating the MST.
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref", 
                            [MemRead]>:$memrefI, 
                      Arg<AnyRankedOrUnrankedMemRef, "outputMemref", 
                            [MemRead]>:$memrefO,
                      Arg<AnyRankedOrUnrankedMemRef, "visitedMemref",
                            [MemRead]>:$memrefV,
                      Arg<AnyRankedOrUnrankedMemRef, "costMemref",
                            [MemRead]>:$memrefC);
                            
  let assemblyFormat = [{
    $memrefI `,` $memrefO `,` $memrefV `,` $memrefC attr-dict `:` type($memrefI) `,` type($memrefO) `,` type($memrefV) `,` type($memrefC)
  }];
}

#endif // Graph_GraphOPS_TD
