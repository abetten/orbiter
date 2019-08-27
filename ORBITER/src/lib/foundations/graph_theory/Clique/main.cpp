#include "RainbowClique.h"

using std::cout;
using std::endl;


int main () {

//	typedef uint32_t _vertex_lbl_type_;
//	typedef uint16_t _vertex_color_type_;

	Graph G (5); // Generate the graph

	std::vector<std::vector<uint32_t>> v; // Vector containing solutions

	RainbowClique::find_cliques(G, v);
}
