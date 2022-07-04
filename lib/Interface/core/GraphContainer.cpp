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

#include <iostream>
#include "Interface/core/Container.h"
#include "Interface/core/GraphContainer.h"


template <typename T, size_t N> Graph<T, N>::Graph(AdjList_Unweighted_Undirected adjList) : MemRef<T, N>() {

    /* Number of vertices in the adjacency list will be the number of vertices in the graph.*/
    this->NoOfVetrices = adjList.size;

    /* allocated will hold the actual data the vertices of the graph will hold. 
    It will be a linear array with elements of type T */
    this->allocated = new T[this->size];
    for(intptr_t i = 0; i < this->NoOfVetrices; i++){
        std::cout<<"Enter the Vertice "<<i<<" :";
        std::cin>>this->allocated[i];
    }

    /* for now aligned and allocated will point to the same memory address. */
    this->aligned = this->allocated;

    //TODO: implementation to fill the aligned
}

/* This default constructor will construct a adjacency list for a unweighted undirected graph, taking user input. */
AdjList_Unweighted_Undirected::AdjList_Unweighted_Undirected() : AdjListUnweighted(){

    intptr_t Vetrices;
    std::cout<<"Enter the number of Vertices in the Graph: "<<std::endl;
    std::cin>>Vetrices;

    /*adjList is a vector of linked lists. The liked list at the index i will 
    store the neighbours of the i th vertex in the graph.*/

    /* This for loop will initialize the list */
    for(intptr_t i = 0; i < Vetrices; i++){
        NodeUnweighted* head = new NodeUnweighted;
        head->Vertex = i;
        head->next = nullptr;
        adjList.push_back(head);
    }

    int toStop = 0;
    int counter = 0;
    int maxNoOfEdges = (Vetrices*(Vetrices - 1))/2;

    /* This while loop will take in the edges of the adjacency list. */
    while(toStop != -1 && counter <= maxNoOfEdges) {
        std::cout<<"Enter the edges of the graph"<<std::endl;
        std::cout<<"After entering all the edges enter -1 for start and end value."<<std::endl;


        intptr_t start,end;
        std::cout<<"Enter start: "<<std::endl;
        std::cin>>start;
        NodeUnweighted* node_start = new NodeUnweighted;
        node_start->next = nullptr;
        node_start->Vertex = start;
        adjList[start]->next = node_start;
        
        std::cout<<"Enter end: "<<std::endl;
        std::cin>>end;
        NodeUnweighted* node_end = new NodeUnweighted;
        node_end->next = nullptr;
        node_end->Vertex = end;
        adjList[end]->next = node_end;

        if(start == -1 && end == -1)
            toStop = -1;
 
        counter++;

        if(counter > maxNoOfEdges)
            std::cout<<"You have reached the limit of forming edgies for a Graph with "<<Vetrices<<" no of Vertices."<<std::endl;
    }
}

// TODO for adjacency List.
//add the support for unweighted directed
//add the support for weighted undirected
//add the support for weighted directed.

#endif // GRAPH_CONTAINER_DEF 