/*
 * Graph.h
 *
 *  Created on: Aug 25, 2019
 *      Author: sajeeb
 */


#include "../../../other/BitSet/bitset.h"
#include "../../../foundations.h"
#include <fstream>
#include <thread>
#include <vector>
#include <algorithm>
#include <iterator>
#include <limits>

#ifndef ORBITER_SRC_LIB_FOUNDATIONS_GRAPH_THEORY_CLIQUE_GRAPH_H_
#define ORBITER_SRC_LIB_FOUNDATIONS_GRAPH_THEORY_CLIQUE_GRAPH_H_

template <typename T=uint32_t, typename U=uint32_t>
class Graph {
public:

    Graph() {}

    __forceinline__
    Graph (size_t _nb_vertices_, size_t _nb_colors_=0, size_t nb_colors_per_vertex=1) {
        this->init(_nb_vertices_, _nb_colors_, nb_colors_per_vertex);
    }

    __forceinline__
    void init (size_t _nb_vertices_, size_t _nb_colors_=0, size_t _nb_colors_per_vertex_=1) {
        nb_vertices = _nb_vertices_;
        nb_colors = _nb_colors_;
        nb_colors_per_vertex = _nb_colors_per_vertex_;

        adjacency.init(nb_vertices * nb_vertices);

        vertex_label = new T [nb_vertices];
        if (_nb_colors_!=0) vertex_color = new U [nb_vertices * nb_colors_per_vertex];
    }

    ~Graph () {
        if (vertex_label) delete [] vertex_label;
        if (vertex_color) delete [] vertex_color;
    }

    /**
     * get the jth color of vertex
     */
    __forceinline__ U get_color(size_t vertex, size_t j=0) const {
        return vertex_color [vertex * nb_colors_per_vertex + j];
    }

    /**
     * get the label of vertex
     */
    __forceinline__ T get_label(size_t vertex) const {
        return vertex_label[vertex];
    }

    /**
     *
     */
    __forceinline__
    void set_vertex_labels(T* lbl) {
    	for (size_t i=0; i<nb_vertices; ++i) {
    		vertex_label[i] = lbl[i];
    	}
    }

    /**
     *
     */
    __forceinline__
	void set_vertex_colors(U* colors) {
    	for (size_t i=0; i<nb_vertices; ++i) { // Anton: found an error, it was nb_colors, which is wrong.
    		vertex_color[i] = colors[i];
    	}
    }

    /**
     * create an edge between vertex i and j
     */
    __forceinline__ void set_edge (size_t i, size_t j) {
        adjacency.set(i*nb_vertices+j);
    }

    /**
     * remove the edge between vertex i and j 
     */
    __forceinline__ void unset_edge (size_t i, size_t j) {
        adjacency.unset(i*nb_vertices+j);
    }

    /**
     * i -> vertex
     * j -> the jth color of the vertex i
     * color -> jth color of vertex i
     */
    __forceinline__ void set_vertex_color (U color, size_t i, size_t j=0) {
    	vertex_color [i * nb_colors_per_vertex + j] = color;
    }

    /**
     * label -> label of the ith vertex
     * i -> the ith vertex in the graph
     */ 
    __forceinline__ void set_vertex_label (T label, size_t i) {
    	vertex_label [i] = label;
    }

    /**
     * check if vertex i is adjacent to vertex j
     */
    __forceinline__ bool is_adjacent (size_t i, size_t j) const {
        return adjacency[i*nb_vertices+j];
    }

    /**
     * The following function sets the edges in the graph from the bitvector
     * adjacency. 'vl' is the verbose level.
     */
    __forceinline__ void set_edge_from_bitvector_adjacency(orbiter::layer1_foundations::other::data_structures::bitvector *Bitvec, int vl=0) {
        if (vl - 2) printf("%s: %d: set_edge_from_bitvector_adjacency\n", __FILE__, __LINE__);
        size_t nThreads = std::thread::hardware_concurrency();
        printf("%s: %d: hardware_concurrency = %ld\n",
        		__FILE__, __LINE__, (long int) std::thread::hardware_concurrency());
        printf("%s: %d: nThreads = %ld\n",
        		__FILE__, __LINE__, (long int) nThreads);
        const size_t n = this->nb_vertices;
	    std::thread threads [nThreads];
        bitset adj[nThreads];
        for (size_t tID=0; tID<nThreads; ++tID) {
            adj[tID].init(adjacency.size());
		    threads[tID] = std::thread([=, &adj] {
                for (size_t i = 0, k = 0; i < n; i++) {
                    if ((i % nThreads) == tID) {
                        for (size_t j = i + 1; j < n; j++, k++) {
                            const int aij = Bitvec->s_i(k);
                            if (aij) {
                                adj[tID].set(i*n+j);
                                adj[tID].set(j*n+i);
                            }
                        }
                    } else {
                        k += n - (i+1);
                    }
                }
            });
        }
        for (size_t i=0; i<nThreads; ++i) threads[i].join();
        for (size_t i=0; i<nThreads; ++i) adjacency |= adj[i];
        if (vl - 2) printf("%s: %d: set_edge_from_bitvector_adjacency Done.\n", __FILE__, __LINE__);
    }

    /**
     *
     */
    void print_adj_matrix () const {
        for (size_t i=0; i<nb_vertices; ++i) {
            for (size_t j=0; j<nb_vertices; ++j) {
                if (is_adjacent(i,j)) std::cout << "1 " ;
                else std::cout << "0 ";
            }
            std::cout << std::endl;
        }
    }

    /**
     * dump the contents of this class in a file 
     */
    void dump(const char* filename) {
    	std::ofstream file;
    	file.open (filename);

    	// nb_vertices, nb_colors_per_vertex, nb_colors
    	file << nb_vertices << " " << nb_colors_per_vertex << " " << nb_colors << "\n";

    	// dump adjacency matrix bitset
    	for (size_t i=0; i < adjacency.data_size(); ++i) {
    		file << adjacency.data(i);
    		if (i+1 < adjacency.data_size()) file << " ";
    	}
    	file << "\n";

    	// dump vertex label
    	for (size_t i=0; i < nb_vertices; ++i) {
    		file << vertex_label[i];
    		if (i+1 < nb_vertices) file << " ";
    	}
    	file << "\n";

    	// dump vertex color
    	for (size_t i=0; i < nb_vertices; ++i) {
    		for (size_t j=0; j < nb_colors_per_vertex; ++j) {
    			file << vertex_color[i*nb_vertices + j];
    			if (j+1 < nb_colors_per_vertex) file << " ";
    		}
    		file << "\n";
    	}

    	file.close();
    }

    size_t nb_colors = 0;
    size_t nb_colors_per_vertex = 0;
    size_t nb_vertices = 0;
    bitset adjacency;
    T* vertex_label = NULL;
    U* vertex_color = NULL;
};

#endif /* ORBITER_SRC_LIB_FOUNDATIONS_GRAPH_THEORY_CLIQUE_GRAPH_H_ */
