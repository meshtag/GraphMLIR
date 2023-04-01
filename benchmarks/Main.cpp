//===- Main.cpp -----------------------------------------------------------===//
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
// This is the main file of the Graph processing Algorithm benchmark.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>

void initializeGraphMLIRFloydWarshall();
void initializeFloydWarshall();
void initializeLemonBFS();
void initializeBoostFLoydWarshall();
void initializeGraphMLIRBfs();

void generateResultGraphMLIRFloydWarshall();
void generateResultFloydWarshall();
void generateResultLemonBFS();
void generateResultBoostFLoydWarshall();
void generateResultGraphMLIRBfs();

int main(int argc, char **argv) {

  initializeGraphMLIRFloydWarshall();
  initializeFloydWarshall();
  initializeLemonBFS();
  initializeBoostFLoydWarshall();
  initializeGraphMLIRBfs();

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  generateResultGraphMLIRFloydWarshall();
  generateResultFloydWarshall();
  generateResultLemonBFS();
  generateResultBoostFLoydWarshall();
  generateResultGraphMLIRBfs();

  return 0;
}
