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

// // creating function to convert to MemRef 
// // that takes in instance of Graph class as input
template <typename T, size_t N> MemRef_descriptor GraphToMemrefConversion (Graph<T,N> g)
{
  uint16_t graph_type = 
  switch(g.grap_type)
  {
    // case to convert Directed Unweighted Adj_List  
    case graph_type::GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED:
    {
      // allocating memory and member data types to store 
      // new created adjacency matrix
      intptr_t graphSize[2] = {g.size , g.size};
      intptr_t graphStrides[2] = {g.size , g.size};
      float *allocationPointer =  (float *)malloc(sizeof(float));
      // Storing a 2d-matrix in a 1-d format to store as memref
      // do for example a 3x3 matrix becomes a matrix of 9 elements 
      float *graphAlign = (float *)malloc(graphSize[0] * graphSize[1] * sizeof(float));
      
      for(unsigned int i=0; i< graphSize[0]; ++i)
        for(unsigned int j = 0; j < graphSize[1]; ++j)
          // first making all edges as zero
          graphAlign[i * graphSize[0] + j] = 0;

      // accessing adjacency list
      for (unsigned int i = 0; i < graphSize[0]; ++i)
      {
        for (auto x : g.nodes[i])
        {
          graphAlign[i * graphSize[0] + x] = 1;
        }
      }

      //printing adjacency matrix for debug
      for(unsigned int i=0; i< graphSize[0]; ++i)
      {
        for(unsigned int j = 0; j < graphSize[1]; ++j)
          // first making all edges as zero
          std::cout<<graphAlign[i * graphSize[0] + j];
        std::cout<<"\n";
      }
      std::cout<<"\n \n";
  }
  break;
}
}

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
};
#endif // GRAPH_CONTAINER_DEF