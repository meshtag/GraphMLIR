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

#include "Interface/core/Container.h"
#include "Interface/core/GraphContainer.h"


template <typename T, size_t N> Graph<T, N>::Graph(AdjList_Unweighted_Undirected adjList) : MemRef<T, N> {

    /* Number of vertices in the adjacency list will be the number of vertices in the graph.*/
    this->NoOfVetrices = adjList.size();

    /* allocated will hold the actual data the vertices of the graph will hold. 
    It will be a linear array with elements of type T */
    this->allocated = new T[this->size];
    for(intptr_t i = 0; i < adjList.size(), i++){
        std::cout<<"Enter the Vertice "<<i<<" :";
        std::cin>>this->allocated[i];
    }

    /* for now aligned and allocated will point to the same memory address. */
    this->aligned = this->allocated;

    //TODO: implementation to fill the aligned
}

// TODO for adjacency List.
//add the support for unweighted undirected
//add the support for unweighted directed
//add the support for weighted undirected
//add the support for weighted directed.

#endif // GRAPH_CONTAINER_DEF 