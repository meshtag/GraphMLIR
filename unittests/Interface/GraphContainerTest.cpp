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

	//Create object of the Graph class and add edges.
	Graph<float, 2> graph(graph::detail::GRAPH_ADJ_LIST_UNDIRECTED_UNWEIGHTED, 6);
	graph.addEdge(0,1);
	graph.addEdge(1,3);
	graph.addEdge(1,5);
	graph.addEdge(1,4);
	graph.addEdge(2,4);
	
	//Print the orignal grpah according to the representaion.
	std::cout<<"Orignal Grpah: "<<std::endl;
	graph.printGraphOg();

	//convert the graph to MemRef using the functions in GraphContainer.cpp and print the memref
	auto memref2 = graph.get_Memref();
	std::cout<<"Graph in linera 2D form: "<<std::endl;
	graph.printGraph();

	//new hard codede MemRef object.
	float aligned[] =  {0,1,0,0,0,0,
						1,0,0,1,1,1, 
						0,0,0,0,1,0,
						0,1,0,0,0,0,
						0,1,1,0,0,0,
						0,1,0,0,0,0};
	intptr_t sizes[2] = {6,6};
	MemRef<float, 2> memref1(aligned, sizes, 0);

	//Test
	EXPECT_EQ(memref1 == memref2, true);
}

TEST_F(GraphContainerTest, adjListUndirectedWeighted) {

	//Create object of the Graph class and add edges.
	Graph<float, 2> graph(graph::detail::GRAPH_ADJ_LIST_DIRECTED_UNWEIGHTED, 6);
	graph.addEdge(1,0);
	graph.addEdge(3,1);
	graph.addEdge(4,1);
	graph.addEdge(4,2);
	graph.addEdge(5,1);
	
	//Print the orignal grpah according to the representaion.
	std::cout<<"Orignal Grpah: "<<std::endl;
	graph.printGraphOg();

	//convert the graph to MemRef using the functions in GraphContainer.cpp and print the memref
	auto memref2 = graph.get_Memref();
	std::cout<<"Graph in linera 2D form: "<<std::endl;
	graph.printGraph();

	//new hard codede MemRef object.
	float aligned[] =  {0,0,0,0,0,0,
						1,0,0,0,0,0, 
						0,0,0,0,0,0,
						0,1,0,0,0,0,
						0,1,1,0,0,0,
						0,1,0,0,0,0};
	intptr_t sizes[2] = {6,6};
	MemRef<float, 2> memref1(aligned, sizes, 0);

	//Test
	EXPECT_EQ(memref1 == memref2, true);
}


int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

