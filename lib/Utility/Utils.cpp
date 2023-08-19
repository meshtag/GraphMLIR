//====- Utils.cpp ---------------------------------------------------------===//
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
// This file implements generic utility functions for the buddy compiler
// ecosystem.
//
//===----------------------------------------------------------------------===//

#ifndef UTILS_UTILS_DEF
#define UTILS_UTILS_DEF

#include <Interface/GraphContainer.h>
#include <Utility/Utils.h>
#include <iostream>
#include <vector>
#include <bits/stdc++.h>

using namespace std;
namespace graph{
void inline generateRandomGraph(Graph<float, 2> *graph, int vertices, int maxWeight, int randomUpperLimit, int randomLowerLimit){
    // printf("Inside the function create_graph\n");
    std::set<std::pair<int, int>> container;
    std::set<std::pair<int, int>>::iterator it;
    // printf("Inside the function create_graph 1\n");
    srand(time(NULL));
    // printf("Inside the function create_graph 2\n");
    int NUM = vertices;    // Number of Vertices
    int MAX_EDGES = vertices * (vertices-1) /2;
    int NUMEDGE = MAX_EDGES; // Number of Edges
    
        
    // Then print the edges of the form (a b)
    // where 'a' is connected to 'b'
    for (int j=1; j<=NUMEDGE; j++)
    {            
        int a = rand() % NUM;
        int b = rand() % NUM;
        std::pair<int, int> p = std::make_pair(a, b);
        std::pair<int, int> reverse_p = std::make_pair(b, a);

        while (container.find(p) != container.end() || container.find(reverse_p) != container.end() || a==b)
        {
            a = rand() % NUM;
            b = rand() % NUM;
            p = std::make_pair(a, b);
            reverse_p = std::make_pair(b,a);
        }

        container.insert(p);
        // int wt = 1 + rand() % MAXWEIGHT;

        graph->addEdge(a, b, 1 + rand() % maxWeight);
    }
    // for (it=container.begin(); it!=container.end(); ++it)
    //     printf("%d %d\n", it->first, it->second);
            
    container.clear();
    printf("\n");           
//  return graph;
    // }
}

void inline generateRandomGraphI(Graph<int, 2> *graph, int vertices){
    // printf("Inside the function create_graph\n");
    std::set<std::pair<int, int>> container;
    std::set<std::pair<int, int>>::iterator it;
    // printf("Inside the function create_graph 1\n");
    srand(time(NULL));
    // printf("Inside the function create_graph 2\n");
    int NUM = vertices;    // Number of Vertices
    int MAX_EDGES = vertices * (vertices-1) /2;
    int NUMEDGE = MAX_EDGES; // Number of Edges
    
        
    // Then print the edges of the form (a b)
    // where 'a' is connected to 'b'
    for (int j=1; j<=NUMEDGE; j++)
    {            
        int a = rand() % NUM;
        int b = rand() % NUM;
        std::pair<int, int> p = std::make_pair(a, b);
        std::pair<int, int> reverse_p = std::make_pair(b, a);

        while (container.find(p) != container.end() || container.find(reverse_p) != container.end() || a==b)
        {
            a = rand() % NUM;
            b = rand() % NUM;
            p = std::make_pair(a, b);
            reverse_p = std::make_pair(b,a);
        }

        container.insert(p);
        // int wt = 1 + rand() % MAXWEIGHT;

        graph->addEdge(a, b, 1 + rand() % 1000);
    }
    // for (it=container.begin(); it!=container.end(); ++it)
    //     printf("%d %d\n", it->first, it->second);
            
    container.clear();
    printf("\n");           
//  return graph;
    // }
}

void inline generateRandomGraph(std::vector<int> &edge, std::vector<int> &weight, int vertices, int maxWeight, int randomUpperLimit, int randomLowerLimit){
    // printf("Inside the function create_graph\n");
    std::set<std::pair<int, int>> container;
    std::set<std::pair<int, int>>::iterator it;
    // printf("Inside the function create_graph 1\n");
    srand(time(NULL));
    // printf("Inside the function create_graph 2\n");
    int NUM = vertices;    // Number of Vertices
    int MAX_EDGES = vertices * (vertices-1) /2;
    int NUMEDGE = MAX_EDGES; // Number of Edges
    
        
    // Then print the edges of the form (a b)
    // where 'a' is connected to 'b'
    for (int j=1; j<=NUMEDGE; j++)
    {            
        int a = 1 + rand() % NUM;
        int b = 1 + rand() % NUM;
        std::pair<int, int> p = std::make_pair(a, b);
        std::pair<int, int> reverse_p = std::make_pair(b, a);
  
        while (container.find(p) != container.end() || container.find(reverse_p) != container.end())
        {
            a = 1 + rand() % NUM;
            b = 1 + rand() % NUM;
            p = std::make_pair(a, b);
            reverse_p = std::make_pair(b,a);
        }
            //   cout<<"Inside here"<<"\n";
        container.insert(p);
        int wt = 1 + rand() % maxWeight;

        edge.push_back(a);
        edge.push_back(b);
        weight.push_back(wt);
    }
    // for (it=container.begin(); it!=container.end(); ++it)
    //     printf("%d %d\n", it->first, it->second);
            
    container.clear();
    printf("\n");           
//  return graph;
    // }
}
}
#endif