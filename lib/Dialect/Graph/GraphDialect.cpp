//===- GraphDialect.cpp - graph Dialect Definition-------------------*- C++
//-*-===//
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
// This file defines Graph dialect.
//
//===----------------------------------------------------------------------===//

#include "Graph/GraphDialect.h"
#include "Graph/GraphOps.h"

using namespace mlir;
using namespace graph;

#include "Graph/GraphOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Graph dialect.
//===----------------------------------------------------------------------===//

void GraphDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Graph/GraphOps.cpp.inc"
      >();
}
