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
#include <iostream>

template <typename T, size_t N> class Graph : public MemRef<T, N> {
	public:
		//the graph type and representation
		uint16_t grap_type;
		long unsigned int graph_size;
		int edgeCount = 0;
		
		T *graphInternal;
		
		std::vector<std::vector<T>> adjList;
	
		Graph(uint16_t graph_type);
		void addEdge(int a, int b);
		void printGraph();
};

#include "Interface/GraphContainer.cpp"
#endif // INTERFACE_GRAPHCONTAINER_H 
