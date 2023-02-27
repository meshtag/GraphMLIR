#include "Interface/GraphContainer.h"
#include <gtest/gtest.h>

class GraphContainerTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(GraphContainerTest, adjListUndirectedUnweighted) {

  // Create object of the Graph class and add edges.
  Graph<float, 2> graph(graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_UNWEIGHTED, 6);
  graph.addEdge(0, 1);
  graph.addEdge(1, 3);
  graph.addEdge(1, 5);
  graph.addEdge(1, 4);
  graph.addEdge(2, 4);

  // convert the graph to MemRef using the functions in GraphContainer.cpp
  auto memref2 = graph.get_Memref();

  // new hard coded MemRef object.
  float aligned[] = {0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0,
                     0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
  intptr_t sizes[2] = {6, 6};
  MemRef<float, 2> memref1(aligned, sizes, 0);

  // Test
  EXPECT_EQ(memref1 == memref2, true);
}

TEST_F(GraphContainerTest, adjListDirectedUnweighted) {

  // Create object of the Graph class and add edges.
  Graph<float, 2> graph(graph::detail::GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED, 6);
  graph.addEdge(1, 0);
  graph.addEdge(3, 1);
  graph.addEdge(4, 1);
  graph.addEdge(4, 2);
  graph.addEdge(5, 1);

  // convert the graph to MemRef using the functions in GraphContainer.cpp
  auto memref2 = graph.get_Memref();

  // new hard coded MemRef object.
  float aligned[] = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
  intptr_t sizes[2] = {6, 6};
  MemRef<float, 2> memref1(aligned, sizes, 0);

  // Test
  EXPECT_EQ(memref1 == memref2, true);
}

TEST_F(GraphContainerTest, adjListUndirectedWeighted) {

  // Create object of the Graph class and add edges.
  Graph<float, 2> graph(graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_WEIGHTED, 6);
  graph.addEdge(1, 0, 2);
  graph.addEdge(3, 1, 3);
  graph.addEdge(4, 1, 4);
  graph.addEdge(4, 2, 5);
  graph.addEdge(5, 1, 6);

  // convert the graph to MemRef using the functions in GraphContainer.cpp
  auto memref2 = graph.get_Memref();

  // new hard coded MemRef object.
  float aligned[] = {0, 2, 0, 0, 0, 0, 2, 0, 0, 3, 4, 6, 0, 0, 0, 0, 5, 0,
                     0, 3, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 6, 0, 0, 0, 0};
  intptr_t sizes[2] = {6, 6};
  MemRef<float, 2> memref1(aligned, sizes, 0);

  // Test
  EXPECT_EQ(memref1 == memref2, true);
}

TEST_F(GraphContainerTest, adjListDirectedWeighted) {

  // Create object of the Graph class and add edges.
  Graph<int, 2> graph(graph::detail::GRAPH_ADJ_LIST_DIRECTED_WEIGHTED, 6);
  graph.addEdge(1, 0, 2);
  graph.addEdge(3, 1, 3);
  graph.addEdge(4, 1, 4);
  graph.addEdge(4, 2, 5);
  graph.addEdge(5, 1, 6);

  // convert the graph to MemRef using the functions in GraphContainer.cpp
  auto memref2 = graph.get_Memref();

  // new hard coded MemRef object.
  int aligned[] = {0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 3, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 6, 0, 0, 0, 0};
  intptr_t sizes[2] = {6, 6};
  MemRef<int, 2> memref1(aligned, sizes, 0);

  // Test
  EXPECT_EQ(memref1 == memref2, true);
}

TEST_F(GraphContainerTest, adjMatrixUndirectedUnweighted) {

  // Create object of the Graph class and add edges.
  Graph<float, 2> graph(graph::detail::GRAPH_ADJ_MATRIX_UNDIRECTED_UNWEIGHTED,
                        6);
  graph.addEdge(0, 1);
  graph.addEdge(1, 3);
  graph.addEdge(1, 5);
  graph.addEdge(1, 4);
  graph.addEdge(2, 4);

  // convert the graph to MemRef using the functions in GraphContainer.cpp and
  auto memref2 = graph.get_Memref();

  // new hard coded MemRef object.
  float aligned[] = {0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0,
                     0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
  intptr_t sizes[2] = {6, 6};
  MemRef<float, 2> memref1(aligned, sizes, 0);

  // Test
  EXPECT_EQ(memref1 == memref2, true);
}

TEST_F(GraphContainerTest, adjMatrixDirectedUnweighted) {

  // Create object of the Graph class and add edges.
  Graph<float, 2> graph(graph::detail::GRAPH_ADJ_MATRIX_DIRECTED_UNWEIGHTED, 6);
  graph.addEdge(1, 0);
  graph.addEdge(3, 1);
  graph.addEdge(4, 1);
  graph.addEdge(4, 2);
  graph.addEdge(5, 1);

  // convert the graph to MemRef using the functions in GraphContainer.cpp
  auto memref2 = graph.get_Memref();

  // new hard coded MemRef object.
  float aligned[] = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
  intptr_t sizes[2] = {6, 6};
  MemRef<float, 2> memref1(aligned, sizes, 0);

  // Test
  EXPECT_EQ(memref1 == memref2, true);
}

TEST_F(GraphContainerTest, adjMatrixUndirectedWeighted) {

  // Create object of the Graph class and add edges.
  Graph<float, 2> graph(graph::detail::GRAPH_ADJ_MATRIX_UNDIRECTED_WEIGHTED, 6);
  graph.addEdge(1, 0, 2);
  graph.addEdge(3, 1, 3);
  graph.addEdge(4, 1, 4);
  graph.addEdge(4, 2, 5);
  graph.addEdge(5, 1, 6);

  // convert the graph to MemRef using the functions in GraphContainer.cpp
  auto memref2 = graph.get_Memref();

  // new hard coded MemRef object.
  float aligned[] = {0, 2, 0, 0, 0, 0, 2, 0, 0, 3, 4, 6, 0, 0, 0, 0, 5, 0,
                     0, 3, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 6, 0, 0, 0, 0};
  intptr_t sizes[2] = {6, 6};
  MemRef<float, 2> memref1(aligned, sizes, 0);

  // Test
  EXPECT_EQ(memref1 == memref2, true);
}

TEST_F(GraphContainerTest, adjMatrixDirectedWeighted) {

  // Create object of the Graph class and add edges.
  Graph<float, 2> graph(graph::detail::GRAPH_ADJ_MATRIX_DIRECTED_WEIGHTED, 6);
  graph.addEdge(1, 0, 2);
  graph.addEdge(3, 1, 3);
  graph.addEdge(4, 1, 4);
  graph.addEdge(4, 2, 5);
  graph.addEdge(5, 1, 6);

  // convert the graph to MemRef using the functions in GraphContainer.cpp
  auto memref2 = graph.get_Memref();

  // new hard coded MemRef object.
  float aligned[] = {0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 3, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 6, 0, 0, 0, 0};
  intptr_t sizes[2] = {6, 6};
  MemRef<float, 2> memref1(aligned, sizes, 0);

  // Test
  EXPECT_EQ(memref1 == memref2, true);
}

TEST_F(GraphContainerTest, incMatrixUndirectedUnweighted) {

  // Create object of the Graph class and add edges.
  Graph<float, 2> graph(graph::detail::GRAPH_INC_MATRIX_UNDIRECTED_UNWEIGHTED,
                        6);
  graph.addEdge(0, 1);
  graph.addEdge(1, 3);
  graph.addEdge(1, 5);
  graph.addEdge(1, 4);
  graph.addEdge(2, 4);

  // convert the graph to MemRef using the functions in GraphContainer.cpp
  auto memref2 = graph.get_Memref();

  // new hard coded MemRef object.
  float aligned[] = {0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0,
                     0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
  intptr_t sizes[2] = {6, 6};
  MemRef<float, 2> memref1(aligned, sizes, 0);

  // Test
  EXPECT_EQ(memref1 == memref2, true);
}

TEST_F(GraphContainerTest, incMatrixDirectedUnweighted) {

  // Create object of the Graph class and add edges.
  Graph<float, 2> graph(graph::detail::GRAPH_INC_MATRIX_DIRECTED_UNWEIGHTED, 6);
  graph.addEdge(1, 0);
  graph.addEdge(3, 1);
  graph.addEdge(4, 1);
  graph.addEdge(4, 2);
  graph.addEdge(5, 1);

  // convert the graph to MemRef using the functions in GraphContainer.cpp
  auto memref2 = graph.get_Memref();

  // new hard coded MemRef object.
  float aligned[] = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
  intptr_t sizes[2] = {6, 6};
  MemRef<float, 2> memref1(aligned, sizes, 0);

  // Test
  EXPECT_EQ(memref1 == memref2, true);
}

TEST_F(GraphContainerTest, incMatrixUndirectedWeighted) {

  // Create object of the Graph class and add edges.
  Graph<float, 2> graph(graph::detail::GRAPH_INC_MATRIX_UNDIRECTED_WEIGHTED, 6);
  graph.addEdge(1, 0, 2);
  graph.addEdge(3, 1, 3);
  graph.addEdge(4, 1, 4);
  graph.addEdge(4, 2, 5);
  graph.addEdge(5, 1, 6);

  // convert the graph to MemRef using the functions in GraphContainer.cpp
  auto memref2 = graph.get_Memref();

  // new hard coded MemRef object.
  float aligned[] = {0, 2, 0, 0, 0, 0, 2, 0, 0, 3, 4, 6, 0, 0, 0, 0, 5, 0,
                     0, 3, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 6, 0, 0, 0, 0};
  intptr_t sizes[2] = {6, 6};
  MemRef<float, 2> memref1(aligned, sizes, 0);

  // Test
  EXPECT_EQ(memref1 == memref2, true);
}

TEST_F(GraphContainerTest, incMatrixDirectedWeighted) {

  // Create object of the Graph class and add edges.
  Graph<int, 2> graph(graph::detail::GRAPH_INC_MATRIX_DIRECTED_WEIGHTED, 6);
  graph.addEdge(1, 0, 2);
  graph.addEdge(3, 1, 3);
  graph.addEdge(4, 1, 4);
  graph.addEdge(4, 2, 5);
  graph.addEdge(5, 1, 6);

  // convert the graph to MemRef using the functions in GraphContainer.cpp
  auto memref2 = graph.get_Memref();

  // new hard coded MemRef object.
  int aligned[] = {0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 3, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 6, 0, 0, 0, 0};
  intptr_t sizes[2] = {6, 6};
  MemRef<int, 2> memref1(aligned, sizes, 0);

  // Test
  EXPECT_EQ(memref1 == memref2, true);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
