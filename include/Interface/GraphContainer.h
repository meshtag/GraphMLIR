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
/**
 * The Graph Class ; the object of this class will be used to call functions.
 * 
 * @tparam T represents the datatype to be used.
 * @tparam N represents the number of dimensions.
 */
template <typename T, size_t N> class Graph : public MemRef<T, N> {
	protected:
		//the graph type and representation
		uint16_t graph_type;

		//the count of number of edges added in the graph.
		int edgeCount = 0;
	
		//adjacency list representation of graph.
		std::vector<std::vector<T>> adjList;

		// incidence matrix representation of graph.
		std::vector<std::vector<T>> incMat;

		// adjacency matrix representation of graph
		std::vector<std::vector<T>> adjMat;

		//weighted adjacency list representation of graph.
		std::vector<std::vector<std::pair<T,T>>> adjList_weighted;
	public:
		//Constructor
		Graph(uint16_t graph_type, size_t size);

		//Function to add edges in graph.
		void addEdge(T a, T b);
		void addEdge(T Node1,T Node2, T EdgeWeight); 

		//Function to print the linear 2d graph.
		void printGraphOg();

		//converter from graph to MemRef_descriptor
		void graph_to_MemRef_descriptor();

		//Function to print the linear 2d graph after conversion.
		void printGraph();
};

#include "Interface/GraphContainer.cpp"
#endif // INTERFACE_GRAPHCONTAINER_H 
