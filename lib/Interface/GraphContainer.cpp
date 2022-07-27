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

template <typename T, size_t N> Graph<T, N>::Graph(uint16_t graph_type, size_t size) {

	this->graph_type = graph_type;
    

	// Assign the memebers of MefRef.
	this->size = size;
	this->allocated = (T *)malloc(sizeof(T));
    this->sizes[1] = size;
	this->sizes[0] = size;
	this->strides[0] = size;
    this->strides[1] = size;

	switch (graph_type) {
		case graph::detail::GRAPH_INC_MATRIX_UNDIRECTED_UNWEIGHTED:
		    this->IncidenceMat.resize(this->size);
			break;
		case graph::detail::GRAPH_INC_MATRIX_DIRECTED_UNWEIGHTED:
			this->IncidenceMat.resize(this->size);
			break;

		case graph::detail::GRAPH_INC_MATRIX_UNDIRECTED_WEIGHTED:
		   this->IncidenceMat.resize(this->size);
		   break;
		case graph::detail::GRAPH_INC_MATRIX_DIRECTED_WEIGHTED:
		   this->IncidenceMat.resize(this->size);
		   break;

		default:
			std::cout<<"Unknown graph container"<<std::endl;
	}
}

template <typename T, size_t N> void Graph<T, N>::addEdge(int p, int q) {

	switch (this->graph_type) {
		case graph::detail::GRAPH_INC_MATRIX_UNDIRECTED_UNWEIGHTED:
                    this->edgeCount++;
			this->IncidenceMat[p].push_back(1);
			this->IncidenceMat[q].push_back(1);

			break;

		case graph::detail::GRAPH_INC_MATRIX_DIRECTED_UNWEIGHTED: 
            this->edgeCount++;
			this->IncidenceMat[p].push_back(1);
            this->IncidenceMat[q].push_back(-1);
           
			break;

      /*  case graph::detail::GRAPH_INC_MATRIX_DIRECTED_WEIGHTED:
            this->edgeCount++;
            std::cout<<"Enter weight of the edge"<<std::endl;
            float weight;
            std::cin>>weight;
            this->IncidenceMat[p].push_back(weight);
            this->IncidenceMat[q].push_back(-weight);    
            break;

        case graph::detail::GRAPH_INC_MATRIX_UNDIRECTED_WEIGHTED:
            this->edgeCount++;
            float weight;
            std::cout<<"Enter the weight of the edge"<<std::endl;
            std::cin>>weight;
            this->IncidenceMat[p].push_back(weight);
            this->IncidenceMat[q].push_back(weight);*/


		//TODO- For default add edges into adjacency matrix.
		default :
			this->edgeCount++;
			break;
	}
}

template <typename T, size_t N> void Graph<T, N>::printGraph() {

	for (int v = 0; v < this->sizes[0]; ++v) {
		for (int w = 0; w < this->sizes[1]; ++w) {
			std::cout<<this->aligned[this->sizes[0]*v + w]<<" ";
		}
		std::cout<<std::endl;
	}
}


//This funciton will convert the graph implementation to linear 2d matrix
template<typename T, size_t N>
void graph_container_to_linear_2d(Graph<T, N> &g) {

	intptr_t x = g.sizes[0];
	intptr_t y = g.sizes[1];
	T* linear = (T *)malloc(sizeof(T) * x * y);

	for (intptr_t i = 0; i < x; i++){
		for (intptr_t j = 0; j < y; j++) {
			linear[i * x + j] = 0;
		}
	}

	switch (g.graph_type) {
		case graph::detail::GRAPH_INC_MATRIX_UNDIRECTED_UNWEIGHTED:
            intptr_t w;
			for (intptr_t i = 0; i < g.IncidenceMat.size(); i++){
				for (intptr_t j = 0; j < g.IncidenceMat[i].size(); j++) {
				if(g.IncidenceMat[i][j] == 1)
                {
                    for(intptr_t k = i+1;i<g.IncidenceMat.size();++i)
                    {
                        if(g.IncidenceMat[k][j] == 1)
                        {
                            
                            w = k;
                            g.IncidenceMat[k][j] = 0;
                        }
                        linear[i*x + w] = 1;
                        linear[w*x + i] = 1;
                    }
                }
				}
			}
			break;

		case graph::detail::GRAPH_INC_MATRIX_DIRECTED_UNWEIGHTED: 

			for (intptr_t i = 0; i < g.IncidenceMat.size(); i++){
				for (intptr_t j = 0; j < g.IncidenceMat[i].size(); j++) {

					if(g.IncidenceMat[i][j] == 1)
                {
                    for(intptr_t k = i+1;i<g.IncidenceMat.size();++i)
                    {
                        if(g.IncidenceMat[k][j] == -1)
                        {
                            
                            w = k;
                            g.IncidenceMat[k][j] = 0;
                        }
                        linear[i*x + w] = 1;
                    }
                }
				}
			}
			break;

		//TODO- For default add edges into adjacency matrix.
		default:
			std::cout<<"Unknown graph type"<<std::endl;
			break;
	}

	g.aligned = linear;
}

template< typename T, size_t N>
MemRef_descriptor graph_to_MemRef_descriptor(Graph<T, N> &graph) {

	graph_container_to_linear_2d(graph); 

	MemRef_descriptor sample_graph_memref = MemRef_Descriptor(graph.allocated, graph.aligned, graph.offset,
							graph.sizes, graph.strides);

	return sample_graph_memref;
}

#endif // GRAPH_CONTAINER_DEF