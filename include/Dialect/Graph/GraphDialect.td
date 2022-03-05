//===- GraphDialect.td - graph Dialect Definition --------------*- tablegen -*-===//
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
// This is the top level file for the Graph dialect.
//
//===----------------------------------------------------------------------===//

#ifndef Graph_GraphDIALECT_TD
#define Graph_GraphDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Graph Dialect Definition.
//===----------------------------------------------------------------------===//

def Graph_Dialect : Dialect {
  let name = "graph";
  let summary = "The graph Dialect.";
  let description = [{
    Dialect for optimizing graph algorithms with improved IR processing/generation
    via MLIR.
  }];
  let cppNamespace = "::graph";
}

//===----------------------------------------------------------------------===//
// Base Graph Operation Definition.
//===----------------------------------------------------------------------===//

class Graph_Op<string mnemonic, list<Trait> traits = []> :
    Op<Graph_Dialect, mnemonic, traits>;

#endif // Graph_GraphDIALECT_TD
