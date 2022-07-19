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

#ifndef INTERFACE_GRAPH_CONTAINER_DEF
#define INTERFACE_GRAPH_CONTAINER_DEF

#include "Interface/GraphContainer.h"
#include "Interface/Container.h"
#include "Interface/graph.h"
#include <cstdint>

template <typename T, size_t N> Graph<T, N>::Graph(uint16_t graph_type, size_t size)
{
	
	// Assign the grah type.
	this->graph_type = graph_type;

	// Assign the memebers of MefRef.
	this->size = size;
	this->allocated = (T *)malloc(sizeof(T));
	this->sizes[0] = size;
	this->sizes[1] = size;
	this->strides[0] = size;
	this->strides[1] = size;
	
	//resize the adjacency list according to the number of nodes.
	switch (graph_type) {
		case graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_UNWEIGHTED:
		case graph::detail::GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED:
			this->adjList.resize(this->size);
			break;

		case graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_WEIGHTED:
		case graph::detail::GRAPH_ADJ_LIST_DIRECTED_WEIGHTED:
			this->adjList_weighted.resize(this->size);
			break;

		default:
			std::cout<<"Unknown graph container"<<std::endl;
	}
}

template <typename T, size_t N> void Graph<T, N>::addEdge(T Node1, T Node2)
{

	switch (this->graph_type) {
		case graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_UNWEIGHTED:
			this->adjList[Node1].push_back(Node2);
			this->adjList[Node2].push_back(Node1);
			this->edgeCount++;
			break;

		case graph::detail::GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED: 
			this->adjList[Node1].push_back(Node2);
			this->edgeCount++;
			break;
		
		default:
			this->edgeCount++;
	}
}

// Overloading function for weighted graphs, currently assuming edges are of the same type as nodes
template <typename T, size_t N> void Graph<T, N>::addEdge(T Node1,T Node2, T EdgeWeight)
{
    //Add an edge between any two nodes
    switch (this->graph_type) {
        case graph::detail::GRAPH_ADJ_LIST_DIRECTED_WEIGHTED:
            this->adjList_weighted[Node1].push_back( std::make_pair(Node2, EdgeWeight));
            break; 
        case graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_WEIGHTED:
            this->adjList_weighted[Node1].push_back( std::make_pair(Node2, EdgeWeight));
            this->adjList_weighted[Node2].push_back( std::make_pair(Node1, EdgeWeight));
            break; 
    }
}

template <typename T, size_t N> void Graph<T, N>::printGraph() 
{
	intptr_t size = this->size;
	for (int v = 0; v < this->sizes[0]; ++v) {
		for (int w = 0; w < this->sizes[1]; ++w) {
			std::cout<<this->aligned[this->sizes[0]*v + w]<<" ";
		}
		std::cout<<std::endl;
	}
}


template <typename T, size_t N> 
MemRef_descriptor Graph<T, N>::graph_to_MemRef_descriptor()
{
	intptr_t x = this->sizes[0];
	intptr_t y = this->sizes[1];
	T* linear = (T *)malloc(sizeof(T) * x * y);

	for (intptr_t i = 0; i < x; i++){
		for (intptr_t j = 0; j < y; j++) {
			linear[i * x + j] = 0;
		}
	}
	
	switch (this->graph_type) {
		case graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_UNWEIGHTED:
			for (intptr_t i = 0; i < x; i++){
				intptr_t neighbour_count = this->adjList[i].size();
				for (intptr_t j = 0; j < neighbour_count; j++) {

					T n = this->adjList[i][j];	
					linear[i * x + (int)n] = 1;
					linear[(int)n * x + i] = 1;
				}
			}
			break;

		case graph::detail::GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED: 
			for (intptr_t i = 0; i < x; i++){
				intptr_t neighbour_count = this->adjList[i].size();
				for (intptr_t j = 0; j < neighbour_count; j++) {

					T n = this->adjList[i][j];
					linear[i * x + (int)n] = 1;
				}
			}
			break;

  		case graph::detail::GRAPH_ADJ_LIST_DIRECTED_WEIGHTED:
      		for (unsigned int i = 0; i < x; ++i){
        		for (auto X : this->adjList_weighted[i]) {
          			linear[i * x + int(X.first)] = X.second;
				}
			}
  			break;

  		case graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_WEIGHTED:
      		for (unsigned int i = 0; i < x; ++i) {
        		for (auto X : this->adjList_weighted[i]) {
          			linear[i * x + int(X.first)] = X.second;
					linear[i + x * int(X.first)] = X.second;
				} 
			} 
  			break;
				
		default:
			std::cout<<"Unknown graph type"<<std::endl;
			break;
	}

	this->aligned = linear;

	MemRef_descriptor sample_graph_memref = MemRef_Descriptor(this->allocated, this->aligned, this->offset,
							this->sizes, this->strides);

	return sample_graph_memref;
}

#endif // INTERFACE_GRAPH_CONTAINER_DEF
