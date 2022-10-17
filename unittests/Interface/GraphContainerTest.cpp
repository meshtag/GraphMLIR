#include "Interface/GraphContainer.h"
#include<gtest/gtest.h>

class GraphContainerTest : public ::testing::Test {
protected:
	void SetUp() override {

	}
	void TearDown() override{
		
	}
};

TEST_F(GraphContainerTest, adjListUndirectedUnweighted) {

	Graph<float, 2> graph(graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_UNWEIGHTED, 5);
	graph.addEdge(0,2);
	graph.addEdge(1,2);
	graph.addEdge(3,2);
	graph.addEdge(0,3);
	graph.addEdge(1,3);

	//test only the allocated field of the memref class.
	MemRef<float, 2> memref1;
	auto memref2 = graph.get_Memref();

	bool isEqual;

	if(memref1 == memref2) {
		isEqual = true;
	}
	EXPECT_EQ(isEqual, true);
	
}

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

