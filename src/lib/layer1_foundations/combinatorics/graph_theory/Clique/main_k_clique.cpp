#include "../../combinatorics/graph_theory/Clique/KClique.h"

using std::cout;
using std::endl;

//logger_init();

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
	G.set_edge(0,4);
	G.set_edge(0,3);
	G.set_edge(0,2);
	G.set_edge(0,1);
	G.set_edge(1,0);
	G.set_edge(1,4);
	G.set_edge(1,3);
	G.set_edge(2,0);
	G.set_edge(2,4);
	G.set_edge(3,0);
	G.set_edge(3,1);
	G.set_edge(4,0);
	G.set_edge(4,1);
	G.set_edge(4,2);
	G.set_edge(3,2);
	G.set_edge(2,3);
	G.set_edge(4,3);
	G.set_edge(3,4);
	G.set_edge(1,2);
	G.set_edge(2,1);

	// Print the adjacency matrix of the graph:
	printf("Adjacency matrix of the current graph:\n");
	G.print_adj_matrix();

	// Set the vertex labels
	G.vertex_label[0] = 0;
	G.vertex_label[1] = 1;
	G.vertex_label[2] = 2;
	G.vertex_label[3] = 3;
	G.vertex_label[4] = 4;


	// Create the solution storage. The base type of the solution
	// storage must be the same as data type of the vertex label
	// in the graph
	std::vector<std::vector<unsigned int>> solutions;

	// Call the K-Clique finding algorithm where k is 3
	KClique::find_cliques(G, solutions, 3, 2);

	// Print the solutions
	printf("Found %ld solution(s).\n", solutions.size());
	for (size_t i=0; i<solutions.size(); ++i) {
		for (size_t j=0; j<solutions[i].size(); ++j) {
			cout << solutions[i][j] << " ";
		} cout << endl;
	}

}
