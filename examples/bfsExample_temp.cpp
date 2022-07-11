#include <Interface/graph.h>
#include <Interface/memref.h>
#include <iostream>

int main() {
    // Create a graph.
    graph::Graph g;
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 2);
    g.addEdge(2, 0);
    g.addEdge(2, 3);
    g.addEdge(3, 3);

    g.graph_type = graph::detail::GRAPH_TYPE::ADJACENCY_LIST;

    graph::detail::ConvertGraphToMemRef(g);
}
