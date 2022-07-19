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

#include "Interface/Container.h"
#include "Interface/GraphContainer.h"

//Common constructor for graph with V Nodes.
template <typename T, size_t N> Graph<T, N>::Graph(uint16_t graph_type, T V) : MemRef<T, N>() {
    this->grap_type = graph_type;
    this->size = V;
    this->allocated = new T[this->size];

    // if adj_list_directed_unwieghted
    if(grap_type == 2){
        this->nodes.resize(this->size);
    }
    else if(grap_type == 3){
        this->weighted_nodes.resize(this->size);
    }
    //TODO
    // Add the implementation for passing data to alignned.
};


// TODO
// Functions to populate 2d vector
template <typename T, size_t N> void Graph<T, N>::AddEdge(T Node1,T Node2) {
    //Add an edge between any two nodes
    switch (this->grap_type) {
        case graph::detail::GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED :
            this->nodes[Node1].push_back(Node2);
            break;
        
    }
};
// Overloading function for weighted graphs, currently assuming edges are of the same type as nodes
template <typename T, size_t N> void Graph<T, N>::AddEdge(T Node1,T Node2, T EdgeWeight) {
    //Add an edge between any two nodes
    switch (this->grap_type) {
        case graph::detail::GRAPH_ADJ_LIST_DIRECTED_WEIGHTED :
            this->weighted_nodes[Node1].push_back( std::make_pair(Node2, EdgeWeight) );
            break;
        
    }
};
template <typename T, size_t N> void Graph<T, N>::PrintGraph(){
    std::cout<< "Nodes -> Edges \n";
    for (size_t i = 0; i < this->size; i++) {
        std::cout << i << "     -> ";
        switch (this-> grap_type) {
            case graph::detail::GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED :
                for (T x : this->nodes[i]) {
                    std::cout << x << " ";
                }
                break;
            case graph::detail::GRAPH_ADJ_LIST_DIRECTED_WEIGHTED :
                // for (std:vector : x : this->weighted_nodes.at(i)) {
                for(size_t j = 0; j<this->weighted_nodes[i].size(); j++){
                    std::cout << this->weighted_nodes[i].at(j).first;
                    std::cout << " Weight(" << this->weighted_nodes[i].at(j).second <<") | ";
                }
                break;
        }
        std::cout << std::endl;
    }
}
#endif // GRAPH_CONTAINER_DEF