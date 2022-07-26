//===- GraphContainer.h ---------------------------------------------------===//
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
// Graph container descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef INTERFACE_GRAPHCONTAINER_H
#define INTERFACE_GRAPHCONTAINER_H

#include "Interface/Container.h"
#include "Interface/memref.h"
#include <cstddef>
#include <iostream>

template <typename T, size_t N> class Graph : public MemRef<T, N> {
	public:
		//the graph type and representation
		uint16_t graph_type;
		int edgeCount = 0;
		
		std::vector<std::vector<T>> adjList;
		Graph(uint16_t graph_type, size_t size);
		void addEdge(int a, int b);
		void printGraph();
};

template<typename T, size_t N>
MemRef_descriptor graph_to_MemRef_descriptor(Graph<T, N> &graph);

#include "Interface/GraphContainer.cpp"
#endif // INTERFACE_GRAPHCONTAINER_H 
