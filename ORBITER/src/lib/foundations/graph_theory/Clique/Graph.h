/*
 * Graph.h
 *
 *  Created on: Aug 25, 2019
 *      Author: sajeeb
 */


#include "bitset.h"

#ifndef ORBITER_SRC_LIB_FOUNDATIONS_GRAPH_THEORY_CLIQUE_GRAPH_H_
#define ORBITER_SRC_LIB_FOUNDATIONS_GRAPH_THEORY_CLIQUE_GRAPH_H_

class Graph {
public:

	__forceinline__
	Graph (size_t _nb_vertices_, size_t _nb_colors_=0) {
		nb_vertices = _nb_vertices_;
		nb_colors = _nb_colors_;

		adjacency.init(nb_vertices);
	}

	~Graph () {
	}


	__forceinline__ size_t get_color(size_t vertex) {
		return vertex_color[vertex];
	}

	__forceinline__ size_t get_label(size_t vertex) {
		return vertex_label[vertex];
	}

	__forceinline__ void set_edge (size_t i, size_t j) {
		adjacency.set(i*nb_vertices+j);
	}

	__forceinline__ void unset_edge (size_t i, size_t j) {
		adjacency.unset(i*nb_vertices+j);
	}

	__forceinline__ bool is_adjacent (size_t i, size_t j) {
		return adjacency[i*nb_vertices+j];
	}

	void print_adj_matrix () {
		for (size_t i=0; i<nb_vertices; ++i) {
			for (size_t j=0; j<nb_vertices; ++j) {
				if (is_adjacent(i,j)) std::cout << "1 " ;
				else std::cout << "0 ";
			}
			std::cout << std::endl;
		}
	}


	size_t nb_colors = 0;
	size_t nb_vertices = 0;
	bitset adjacency;
	size_t* vertex_color = NULL;
	size_t* vertex_label = NULL;

};

#endif /* ORBITER_SRC_LIB_FOUNDATIONS_GRAPH_THEORY_CLIQUE_GRAPH_H_ */
