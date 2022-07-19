//===- GraphContainer.cpp -------------------------------------------------===//
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
// This file implements the Graph container descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPH_CONTAINER_DEF
#define GRAPH_CONTAINER_DEF

#include "Interface/GraphContainer.h"
#include "Interface/Container.h"
#include "Interface/graph.h"
#include <cstddef>

template <typename T, size_t N> Graph<T, N>::Graph(uint16_t graph_type, size_t size) {
	
	this->graph_type = graph_type;

	switch (graph_type) {
		case graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_UNWEIGHTED:
		case graph::detail::GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED:
			this->adjList.resize(this->size);
			break;

		case graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_WEIGHTED:
		case graph::detail::GRAPH_ADJ_LIST_DIRECTED_WEIGHTED:

		default:
			std::cout<<"Unknown graph container"<<std::endl;
	}
}

template <typename T, size_t N> void Graph<T, N>::addEdge(int p, int q) {

	switch (this->graph_type) {
		case graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_UNWEIGHTED:
			this->adjList[p].push_back(q);
			this->adjList[q].push_back(p);
			this->edgeCount++;
			break;

		case graph::detail::GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED: 
			this->adjList[p].push_back(q);
			this->edgeCount++;
			break;
		
		//TODO- For default add edges into adjacency matrix.
		default:
			this->edgeCount++;
	}
}

template <typename T, size_t N> void Graph<T, N>::printGraph() {
	for (int v = 0; v < this->graph_size; ++v) {
		std::cout << "\n Adjacency list of vertex " << v << "\n head ";
		for (auto x : this->adjList[v])
			 std::cout << "-> " << x;
		printf("\n");
	}
}

#endif // GRAPH_CONTAINER_DEF
