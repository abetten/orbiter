#include "RainbowClique.h"

using std::cout;
using std::endl;


int main () {

	Graph G (5); // Generate the graph

	std::vector<std::vector<size_t>> v; // Vector containing solutions

	RainbowClique::find_cliques(G, v);
}
