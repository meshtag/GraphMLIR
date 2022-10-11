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
#include <cmath>
#include <cstdint>

/**
 * @brief Construct a new Graph< T,  N>:: Graph object
 *
 * @tparam T represents the datatype to be used
 * @tparam N represents the number of dimensions
 * @param graph_type represents the type of graph
 * @param size reopresents the number of nodes in the graph
 */
template <typename T, size_t N>
Graph<T, N>::Graph(uint16_t graph_type, size_t size) {

  // Assign the grah type.
  this->graph_type = graph_type;
  this->size = size;
  this->sizes[0] = size;
  this->sizes[1] = size;

  size_t maxEdges = ((this->size) * (this->size - 1)) / 2;

  // resize the adjacency list according to the number of nodes.
  switch (graph_type) {
  case graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_UNWEIGHTED:
    this->adjList.resize(this->size);
    break;
  case graph::detail::GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED:
    this->adjList.resize(this->size);
    break;

  case graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_WEIGHTED:
    this->adjList_weighted.resize(this->size);
    break;
  case graph::detail::GRAPH_ADJ_LIST_DIRECTED_WEIGHTED:
    this->adjList_weighted.resize(this->size);
    break;

  case graph::detail::GRAPH_INC_MATRIX_UNDIRECTED_UNWEIGHTED:
    this->incMat.resize(this->size);
    for (size_t i = 0; i < this->size; i++) {
      this->incMat[i].resize(maxEdges);
    }
    break;
  case graph::detail::GRAPH_INC_MATRIX_DIRECTED_UNWEIGHTED:
    this->incMat.resize(this->size);
    for (size_t i = 0; i < this->size; i++) {
      this->incMat[i].resize(maxEdges);
    }
    break;
  case graph::detail::GRAPH_INC_MATRIX_UNDIRECTED_WEIGHTED:
    this->incMat.resize(this->size);
    for (size_t i = 0; i < this->size; i++) {
      this->incMat[i].resize(maxEdges);
    }
    break;
  case graph::detail::GRAPH_INC_MATRIX_DIRECTED_WEIGHTED:
    this->incMat.resize(this->size);
    for (size_t i = 0; i < this->size; i++) {
      this->incMat[i].resize(maxEdges);
    }
    break;
  case graph::detail::GRAPH_ADJ_MATRIX_DIRECTED_WEIGHTED:
    this->adjMat.resize(this->size);
    for (size_t i = 0; i < this->size; i++) {
      this->adjMat[i].resize(this->size);
    }
    break;
  case graph::detail::GRAPH_ADJ_MATRIX_UNDIRECTED_WEIGHTED:
    this->adjMat.resize(this->size);
    for (size_t i = 0; i < this->size; i++) {
      this->adjMat[i].resize(this->size);
    }
    break;
  case graph::detail::GRAPH_ADJ_MATRIX_DIRECTED_UNWEIGHTED:
    this->adjMat.resize(this->size);
    for (size_t i = 0; i < this->size; i++) {
      this->adjMat[i].resize(this->size);
    }
    break;
  case graph::detail::GRAPH_ADJ_MATRIX_UNDIRECTED_UNWEIGHTED:
    this->adjMat.resize(this->size);
    for (size_t i = 0; i < this->size; i++) {
      this->adjMat[i].resize(this->size);
    }
    break;

  default:
    std::cout << "Unknown graph container" << std::endl;
  }
}
/**
 * @brief This function provides the functionality to add edges to the graph
 *
 * @tparam T represnts the datatype used.
 * @tparam N represnts the number of dimensions.
 * @param Node1 The first node for inserting an edge
 * @param Node2 The second node for inserting an edge.
 */

template <typename T, size_t N> void Graph<T, N>::addEdge(T Node1, T Node2) {

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

  case graph::detail::GRAPH_INC_MATRIX_UNDIRECTED_UNWEIGHTED:
    this->incMat[Node1][edgeCount] = 1;
    this->incMat[Node2][edgeCount] = 1;
    this->edgeCount += 1;
    break;

  case graph::detail::GRAPH_INC_MATRIX_DIRECTED_UNWEIGHTED:
    if (Node1 == Node2) {
      this->incMat[Node1][edgeCount] = 2;
      this->incMat[Node2][edgeCount] = 2;
      this->edgeCount += 1;
    } else {
      this->incMat[Node1][edgeCount] = 1;
      this->incMat[Node2][edgeCount] = -1;
      this->edgeCount += 1;
    }
    break;

  case graph::detail::GRAPH_ADJ_MATRIX_UNDIRECTED_UNWEIGHTED:
    if (Node1 == Node2) {
      this->adjMat[Node1][Node1] = 1;
      this->edgeCount++;
    } else {
      this->adjMat[Node1][Node2] = 1;
      this->adjMat[Node2][Node1] = 1;
      this->edgeCount++;
    }
    break;

  case graph::detail::GRAPH_ADJ_MATRIX_DIRECTED_UNWEIGHTED:
    if (Node1 == Node2) {
      this->adjMat[Node1][Node1] = 1;
      this->edgeCount++;
    } else {
      this->adjMat[Node1][Node2] = 1;
      this->edgeCount++;
    }
    break;

  default:
    this->edgeCount++;
  }
}

/**
 * @brief Overloading function for weighted graphs, currently assuming edges are
 * of the same type as nodes
 *
 * @tparam T Represnts the datatype used
 * @tparam N Represents the nnumber of  dimensions
 * @param Node1 First Node for creating an edge
 * @param Node2 Second node for creating an edge
 * @param EdgeWeight The edgeweight for the edge.
 */
template <typename T, size_t N>
void Graph<T, N>::addEdge(T Node1, T Node2, T EdgeWeight) {
  // Add an edge between any two nodes
  switch (this->graph_type) {
  case graph::detail::GRAPH_ADJ_LIST_DIRECTED_WEIGHTED:
    this->adjList_weighted[Node1].push_back(std::make_pair(Node2, EdgeWeight));
    break;
  case graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_WEIGHTED:
    this->adjList_weighted[Node1].push_back(std::make_pair(Node2, EdgeWeight));
    this->adjList_weighted[Node2].push_back(std::make_pair(Node1, EdgeWeight));
    break;
  case graph::detail::GRAPH_INC_MATRIX_UNDIRECTED_WEIGHTED:
    this->edgeCount += 1;
    this->incMat[Node1][edgeCount] = EdgeWeight;
    this->incMat[Node2][edgeCount] = EdgeWeight;
    break;
  case graph::detail::GRAPH_INC_MATRIX_DIRECTED_WEIGHTED:
    EdgeWeight = std::abs(sqrt(EdgeWeight));
    this->incMat[Node1][edgeCount] = EdgeWeight;
    this->incMat[Node2][edgeCount] = -EdgeWeight;
    this->edgeCount += 1;
    break;
  case graph::detail::GRAPH_ADJ_MATRIX_DIRECTED_WEIGHTED:
    this->edgeCount++;
    this->adjMat[Node1][Node2] = EdgeWeight;
    break;
  case graph::detail::GRAPH_ADJ_MATRIX_UNDIRECTED_WEIGHTED:
    this->edgeCount++;
    this->adjMat[Node1][Node2] = EdgeWeight;
    this->adjMat[Node2][Node1] = EdgeWeight;
    break;
  }
}
/**
 * @brief Prints the Graph in its original form before its conversion to a
 * linear memref descriptor
 *
 * @tparam T represents the datatype used
 * @tparam N represnts the number of dimensions.
 */
template <typename T, size_t N> void Graph<T, N>::printGraphOg() {
  std::cout << "Nodes -> Edges \n";
  for (size_t i = 0; i < this->size; i++) {
    std::cout << i << "     -> ";
    switch (this->graph_type) {
    case graph::detail::GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED:
      for (T x : this->adjList[i]) {
        std::cout << x << " ";
      }
      break;
    case graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_UNWEIGHTED:
      for (T x : this->adjList[i]) {
        std::cout << x << " ";
      }
      break;
    case graph::detail::GRAPH_ADJ_LIST_DIRECTED_WEIGHTED:
      // for (std:vector : x : this->weighted_nodes.at(i)) {
      for (size_t j = 0; j < this->adjList_weighted[i].size(); j++) {
        std::cout << this->adjList_weighted[i].at(j).first;
        std::cout << " Weight(" << this->adjList_weighted[i].at(j).second
                  << ") | ";
      }
      break;
    case graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_WEIGHTED:
      for (size_t j = 0; j < this->adjList_weighted[i].size(); j++) {
        std::cout << this->adjList_weighted[i].at(j).first;
        std::cout << " Weight(" << this->adjList_weighted[i].at(j).second
                  << ") | ";
      }
      break;
    case graph::detail::GRAPH_INC_MATRIX_DIRECTED_UNWEIGHTED:
      for (T x : this->incMat[i]) {
        std::cout << x << "  ";
      }
      break;
    case graph::detail::GRAPH_INC_MATRIX_UNDIRECTED_UNWEIGHTED:
      for (T x : this->incMat[i]) {
        std::cout << x << " ";
      }
      break;
    case graph::detail::GRAPH_INC_MATRIX_UNDIRECTED_WEIGHTED:
      for (T x : this->incMat[i]) {
        std::cout << x << " ";
      }
      break;
    case graph::detail::GRAPH_INC_MATRIX_DIRECTED_WEIGHTED:
      for (T x : this->incMat[i]) {
        std::cout << x << " ";
      }
      break;
    case graph::detail::GRAPH_ADJ_MATRIX_DIRECTED_WEIGHTED:
      for (T x : this->adjMat[i]) {
        std::cout << x << " ";
      }
      break;
    case graph::detail::GRAPH_ADJ_MATRIX_UNDIRECTED_WEIGHTED:
      for (T x : this->adjMat[i]) {
        std::cout << x << " ";
      }
      break;
    case graph::detail::GRAPH_ADJ_MATRIX_DIRECTED_UNWEIGHTED:
      for (T x : this->adjMat[i]) {
        std::cout << x << " ";
      }
      break;
    case graph::detail::GRAPH_ADJ_MATRIX_UNDIRECTED_UNWEIGHTED:
      for (T x : this->adjMat[i]) {
        std::cout << x << " ";
      }
      break;
    }
    std::cout << std::endl;
  }
};
// /**
//  * @brief Function to print the graph using the memref descriptor
//  *
//  * @tparam T Repesents the datatype used
//  * @tparam N represents the number of dimensions.
//  */
template <typename T, size_t N> void Graph<T, N>::printGraph() {
  if (!this->data) {
    std::cout << "Graph is not converted into memref! \n";
    return;
  }
  auto y = this->data->getData();
  for (int v = 0; v < this->sizes[0]; ++v) {
    for (int w = 0; w < this->sizes[1]; ++w) {
      std::cout << y[this->sizes[0] * v + w] << " ";
    }
    std::cout << "\n";
  }
}
/**
 * @brief Converts the various Graph representations types to a linear memref
 * descriptor
 *
 * @tparam T represents the datatype used
 * @tparam N represents the number of dimensions.
 * @return MemRef_descriptor returns a memref descriptor of the respective graph
 * representation type after conversion.
 */

template <typename T, size_t N> void Graph<T, N>::graph_to_MemRef_descriptor() {
  // Assign the memebers of MefRef.
  intptr_t x = this->sizes[0];
  intptr_t y = this->sizes[1];
  T *linear = (T *)malloc(sizeof(T) * x * y);
  size_t maxEdges = ((this->size) * (this->size - 1)) / 2;
  T flag = -2;

  for (intptr_t i = 0; i < x; i++) {
    for (intptr_t j = 0; j < y; j++) {
      linear[i * x + j] = 0;
    }
  }

  switch (this->graph_type) {
  case graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_UNWEIGHTED:
    for (intptr_t i = 0; i < x; i++) {
      intptr_t neighbour_count = this->adjList[i].size();
      for (intptr_t j = 0; j < neighbour_count; j++) {

        T n = this->adjList[i][j];
        linear[i * x + (int)n] = 1;
        linear[(int)n * x + i] = 1;
      }
    }
    break;

  case graph::detail::GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED:
    for (intptr_t i = 0; i < x; i++) {
      intptr_t neighbour_count = this->adjList[i].size();
      for (intptr_t j = 0; j < neighbour_count; j++) {

        T n = this->adjList[i][j];
        linear[i * x + (int)n] = 1;
      }
    }
    break;

  case graph::detail::GRAPH_ADJ_LIST_DIRECTED_WEIGHTED:
    for (unsigned int i = 0; i < x; ++i) {
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

  case graph::detail::GRAPH_INC_MATRIX_UNDIRECTED_UNWEIGHTED:
    for (size_t j = 0; j < maxEdges; j++) {
      flag = -2;
      for (size_t i = 0; i < this->incMat.size() && flag != -1; i++) {
        if (this->incMat[i][j] == 1 && flag == -2) {
          flag = i;
        } else if (this->incMat[i][j] == 1 && flag != -2) {
          linear[int(flag) * x + i] = 1;
          linear[i * x + int(flag)] = 1;
          flag = -1;
        }
        // covering corner case of self-loop
        if (i == this->incMat.size() - 1 && (flag != -1 && flag != -2)) {
          linear[int(flag) * x + int(flag)] = 1;
          flag = -1;
        }
      }
    }
    break;
  case graph::detail::GRAPH_INC_MATRIX_UNDIRECTED_WEIGHTED:
    for (size_t j = 0; j < maxEdges; j++) {
      flag = -2;
      for (size_t i = 0; i < this->incMat.size() && flag != -1; i++) {
        if (this->incMat[i][j] != 0 && flag == -2) {
          flag = i;
        } else if (this->incMat[i][j] != 0 && flag != -2) {
          linear[int(flag) * x + i] = incMat[i][j];
          linear[i * x + int(flag)] = incMat[i][j];
          flag = -1;
        }
        // covering corner case of self-loop
        if (i == this->incMat.size() - 1 && (flag != -1 && flag != -2)) {
          linear[int(flag) * x + int(flag)] = incMat[int(flag)][j];
          flag = -1;
        }
      }
    }
    break;
  case graph::detail::GRAPH_INC_MATRIX_DIRECTED_UNWEIGHTED:
    for (size_t j = 0; j < maxEdges; j++) {
      flag = -2;
      for (size_t i = 0; i < this->incMat.size() && flag != -1; i++) {
        if ((this->incMat[i][j] == 1 || this->incMat[i][j] == -1) &&
            flag == -2) {
          flag = i;
        } else if ((this->incMat[i][j] == 1 || this->incMat[i][j] == -1) &&
                   flag != -2) {
          if (this->incMat[i][j] == -1)
            linear[int(flag) * x + i] = 1;
          else
            linear[i * x + int(flag)] = 1;
          flag = -1;
        }
        // case for self loops
        if (this->incMat[i][j] == 2) {
          linear[i * x + i] = 1;
          flag = -1;
        }
      }
    }
    break;
  case graph::detail::GRAPH_INC_MATRIX_DIRECTED_WEIGHTED:
    for (size_t j = 0; j < maxEdges; j++) {
      flag = -2;
      for (size_t i = 0; i < this->incMat.size() && flag != -1; i++) {
        if ((this->incMat[i][j] != 0) && flag == -2) {
          flag = i;
          size_t k;
          for (k = flag + 1; k < this->incMat.size() && flag != -2; k++) {
            if ((this->incMat[k][j] != 0) && flag != -1) {
              if (this->incMat[k][j] < this->incMat[int(flag)][j])
                linear[int(flag) * x + k] = pow(incMat[k][j], 2);
              else
                linear[k * x + int(flag)] = pow(incMat[k][j], 2);
              flag = -1;
            }
          }
          // case of self loop
          if (k == incMat.size() && flag != -1) {
            linear[i * x + i] = pow(incMat[i][j], 2);
          }
        }
      }
    }
    break;
  case graph::detail::GRAPH_ADJ_MATRIX_DIRECTED_UNWEIGHTED:
    for (intptr_t i = 0; i < x; ++i) {
      for (intptr_t j = 0; j < x; ++j) {
        T n = adjMat[i][j];
        linear[i * x + j] = n;
      }
    }
    break;
  case graph::detail::GRAPH_ADJ_MATRIX_DIRECTED_WEIGHTED:
    for (intptr_t i = 0; i < x; ++i) {
      for (intptr_t j = 0; j < x; ++j) {
        T n = adjMat[i][j];
        linear[i * x + j] = n;
      }
    }
    break;
  case graph::detail::GRAPH_ADJ_MATRIX_UNDIRECTED_UNWEIGHTED:
    for (intptr_t i = 0; i < x; ++i) {
      for (intptr_t j = 0; j < x; ++j) {
        T n = adjMat[i][j];
        linear[i * x + j] = n;
      }
    }
    break;
  case graph::detail::GRAPH_ADJ_MATRIX_UNDIRECTED_WEIGHTED:
    for (intptr_t i = 0; i < x; ++i) {
      for (intptr_t j = 0; j < x; ++j) {
        T n = adjMat[i][j];
        linear[i * x + j] = n;
      }
    }
    break;

  default:
    std::cout << "Unknown graph type" << std::endl;
    break;
  }

  if (data)
    delete data;
  std::cout << "Inside the graph to memref descriptor! \n";
  data = new MemRef<T, N>(linear, this->sizes);

  // return *data;
}

#endif // INTERFACE_GRAPH_CONTAINER_DEF
