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
#include "include/Interface/GraphContainer.h"


// Insert Method (takes input from the user)
 

// Incidence Matrix constructor.
template <typename T, size_t N> Graph<T, N>::Graph(int64_t vertices,int64_t edges) {

  this->V = vertices;
  this->E = edges;
};

 template<typename T,size_t N> int ** Graph<T,N>::Insert(int64_t edges,int64_t vertices, std::string method="incidence",std::string type="undirected")
{
   Graph(vertices,edges);
   Matrix = new int *[vertices+1];
   for(int i =0;i<vertices;i++)
   {
    Matrix[i] = new int[edges+1];
   }
   for(int j=1;j<=edges;++j)
   { int x,y;
    std::cout<<"Enter the vertices for edges"<<std::endl;
    std::cin>>x>>y:
    std::endl;
    Matrix[x][j] = 1;
    Matrix[y][j] = 1;
  }
    return Matrix;
}




// TODO
// Implement the contructor for different implementation.

#endif // GRAPH_CONTAINER_DEF 