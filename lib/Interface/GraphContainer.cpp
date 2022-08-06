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
    this->sizes[0]=V;
    this->sizes[1]=V;
    this->strides[0]=V;
    this->strides[1]=V;

    // if adj_list_directed_unwieghted
    if(grap_type == 2){
        this->nodes.resize(this->size);
    }
    else if(grap_type == 3){
        this->weighted_nodes.resize(this->size);
    }
    //TODO - done
    // Implementation for passing data to alignned
    // added in GraphTomemrefConversion function
};


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
template <typename T, size_t N> MemRef_descriptor Graph<T, N>::GraphToMemrefConversion (Graph<T,N> &g)
{
  // allocating memory and member data types to store 
  // new created adjacency matrix
  intptr_t graphSize[2] = {g.sizes[0] , g.sizes[1]};
  intptr_t graphStrides[2] = {g.strides[0] , g.strides[1]};
  float *allocationPointer =  (float *)malloc(sizeof(float));
  // Storing a 2d-matrix in a 1-d format to store as memref
  // do for example a 3x3 matrix becomes a matrix of 9 elements 
  float *graphAlign = (float *)malloc(graphSize[0] * graphSize[1] * sizeof(float));
  g.aligned = graphAlign;

  switch(g.grap_type)
  {
    // case to convert Directed Unweighted Adj_List  
    case graph::detail::GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED:
    {
      
      for(unsigned int i=0; i< graphSize[0]; ++i)
        for(unsigned int j = 0; j < graphSize[1]; ++j)
          // first making all edges as zero
          graphAlign[i * graphSize[0] + j] = 0;

      // accessing adjacency list
      for (unsigned int i = 0; i < graphSize[0]; ++i)
      {
        for (auto x : g.nodes[i])
        {
          graphAlign[(i * graphSize[0]) + int(x)] = 1;
        }
      }
  }
  // case to convert Directed Weighted Adj_List
  case graph::detail::GRAPH_ADJ_LIST_DIRECTED_WEIGHTED:
  {
      
      for(unsigned int i=0; i< graphSize[0]; ++i)
        for(unsigned int j = 0; j < graphSize[1]; ++j)
          // first making all edges as zero
          graphAlign[i * graphSize[0] + j] = 0;

      // accessing adjacency list
      for (unsigned int i = 0; i < graphSize[0]; ++i)
      {
        for (auto x : g.weighted_nodes[i])
        {
          graphAlign[(i * graphSize[0]) + int(x.first)] = x.second;
        }
      }
  }
  default:
    int i = 0;
  break;
}

MemRef_descriptor sample_graph_memref = MemRef_Descriptor(g.allocated, g.aligned, g.offset,
							g.sizes, g.strides);;
return sample_graph_memref;
}

template <typename T, size_t N> void Graph<T, N>::PrintGraphInMemrefConversion (Graph<T,N> &g){
      //printing adjacency matrix for debug
      std::cout<<"   ";
      for(unsigned int i=0; i< g.sizes[0]; ++i)
        std::cout<<i<<" ";
      std::cout<<"\n |-------- \n";
      for(unsigned int i=0; i< g.sizes[0]; ++i)
      {
        std::cout<<i<<"| ";
        for(unsigned int j = 0; j < g.sizes[1]; ++j)
        {
          std::cout<<g.aligned[i * g.sizes[0] + j]<<" ";
        }
        std::cout<<"\n";
      }
      std::cout<<"\n \n";
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