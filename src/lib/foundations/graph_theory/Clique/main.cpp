#include "RainbowClique.h"

using std::cout;
using std::endl;


int main () {

	// The graph class is templated such that the vertex labels can be
	// whatever is specified in the template specifiers denoted by the
	// "<>" symbols. For this graph class, the vertex labels needs to
	// be numeric types and so does the vertex colors. Both the labels
	// and the colors needs to be indexed from zero.

	// In the following example we are creating a graph whose vertex
	// labels are characters. This graph is uncolored.
	Graph<> G (5, 3); // Generate the graph with 5 vertices and
							// three color classes.

	// Here, for demonstration purposes, we are creating a graph with
	// a star topology by setting the following edges:
	// * node 0 is connected to nodes: 4,3,2,1
	// * node 1 is connected to nodes: 0,4,3
	// * node 2 is connected to nodes: 0,4
	// * node 3 is connected to nodes: 0,1
	// * node 4 is connected to nodes: 0,1,2
	G.set_edge(0,4); G.set_edge(0,3); G.set_edge(0,2); G.set_edge(0,1);
	G.set_edge(1,0); G.set_edge(1,4); G.set_edge(1,3);
	G.set_edge(2,0); G.set_edge(2,4);
	G.set_edge(3,0); G.set_edge(3,1);
	G.set_edge(4,0); G.set_edge(4,1); G.set_edge(4,2);

	// Print the adjacency matrix of the graph:
	printf("Adjacency matrix of the current graph:\n");
	G.print_adj_matrix();

	// Set the vertex labels
	G.vertex_label[0] = 0;
	G.vertex_label[1] = 1;
	G.vertex_label[2] = 2;
	G.vertex_label[3] = 3;
	G.vertex_label[4] = 4;


	// Set the color of each vertex. note that the coloring has to
	// be numerical in order to use the rainbow clique algorithm
	G.vertex_color[0] = 0;
	G.vertex_color[1] = 2;
	G.vertex_color[2] = 2;
	G.vertex_color[3] = 1;
	G.vertex_color[4] = 1;


	// Create the solution storage. The base type of the solution
	// storage must be the same as data type of the vertex label
	// in the graph
	std::vector<std::vector<unsigned int>> solutions;

	// Call the Rainbow Clique finding algorithm
	RainbowClique::find_cliques(G, solutions, 0);

	// Print the solutions
	printf("Found %ld solution(s).\n", solutions.size());
	for (size_t i=0; i<solutions.size(); ++i) {
		for (size_t j=0; j<solutions[i].size(); ++j) {
			cout << solutions[i][j] << " ";
		} cout << endl;
	}

}
