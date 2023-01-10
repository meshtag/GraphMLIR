//====- Utils.cpp ---------------------------------------------------------===//
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
// This file implements generic utility functions.
//
//===----------------------------------------------------------------------===//

#ifndef UTILS_UTILS
#define UTILS_UTILS

#include <Interface/GraphContainer.h>
#include <iostream>
#include <vector>
#include <bits/stdc++.h>

using namespace std;

namespace graph {
void generateRandomGraph(Graph<float, 2> *graph, int vertices, int maxWeight = 900, int randomUpperLimit = 100, int randomLowerLimit = 2);
void generateRandomGraph(vector<int> &edge, vector<int> &weight, int vertices, int maxWeight = 900, int randomUpperLimit = 100, int randomLowerLimit = 2);
}
#include <Utility/Utils.cpp>

#endif