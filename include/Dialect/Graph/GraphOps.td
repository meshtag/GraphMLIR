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

#endif // Graph_GraphOPS_TD
