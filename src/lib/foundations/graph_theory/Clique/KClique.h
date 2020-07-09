/*
 * KClique.h
 *
 *  Created on: Jun 16, 2020
 *      Author: sajeeb
 */

#include <iostream>
#include "Graph.h"
#include <thread>
#include <vector>
#include <algorithm>
#include <iterator>
#include <limits>
#include <unordered_set>
#include <string>
#include <math.h>

#ifndef SRC_LIB_FOUNDATIONS_GRAPH_THEORY_CLIQUE_KCLIQUE_H_
#define SRC_LIB_FOUNDATIONS_GRAPH_THEORY_CLIQUE_KCLIQUE_H_

using std::vector;
using std::cout;
using std::endl;

class KClique {
public:
	template <typename T, typename U>
	__forceinline__
	static void find_cliques (Graph<T,U>& G, vector<vector<T>>& soln, unsigned int k, size_t n_threads=0) {
		const size_t nThreads = (n_threads == 0) ? std::thread::hardware_concurrency() : n_threads;
		std::thread threads [nThreads];
		PARAMS<T> params [nThreads];

		for (size_t i=0; i<nThreads; ++i) {
			params[i].init(i, k, G.nb_vertices, nThreads);
			threads[i] = std::thread(find_cliques_parallel<T,U>, 0, std::ref(params[i]), std::ref(G));
		}

		for (size_t i=0; i<nThreads; ++i) threads[i].join();

		// Find the total number of solutions
		size_t nb_sols = 0;
		for (size_t i=0; i<nThreads; ++i) {
			nb_sols += params[i].t_solutions.size();
		}
		soln.reserve(nb_sols);

		for (size_t i=0; i<nThreads; ++i) {
			std::move(params[i].t_solutions.begin(), params[i].t_solutions.end(), std::back_inserter(soln));
		}
	}


private:
	template <typename T>
	class PARAMS {
	public:
		~PARAMS() {
			if (current_cliques) delete [] current_cliques;
			if (candidates) delete [] candidates;
		}
		PARAMS() {}

		void init(uint8_t tid, unsigned int k, uint32_t num_nodes, uint8_t n_threads) {
			this->tid = tid;
			this->n_threads = n_threads;
			this->k = k;
			this->num_nodes = num_nodes;
			current_cliques= new T [k];
			memset(current_cliques, -1, sizeof(T)*k);
			candidates = new T [k * (num_nodes+1)] ();
			for (unsigned int i=0; i<num_nodes; ++i) candidates[i] = i;
		}

		__forceinline__
		T* get_candidates(size_t depth) {
			return candidates + depth * (num_nodes + 1);
		}

		uint8_t tid = 0;	//
		T* current_cliques = NULL;	// Index of current clique
		T* candidates = NULL;
		size_t nb_sol = 0; // number of solutions found by a thread
		uint8_t n_threads = 0;
		unsigned int k = 0;
		size_t depth = 0;
		size_t num_nodes = 0;
		vector<vector<T>> t_solutions;
	};

	/**
	 *
	 */
	template <typename T, typename U>
	static void find_cliques_parallel (size_t depth, PARAMS<T>& param, Graph<T,U>& G) {
		if (depth == param.k) {
			param.nb_sol += 1;
			param.t_solutions.emplace_back(vector<T>());
			for (size_t i=0; i<depth; ++i)
				param.t_solutions.at(param.t_solutions.size()-1).emplace_back(
						G.get_label(param.current_cliques[i])
				);
			return;
		}

		if (depth == 0) {
			for (T pt=0; pt < G.nb_vertices; ++pt) {
				if ((pt % param.n_threads) == param.tid) {
					param.current_cliques[depth] = pt;
					populate_adjacency(depth, param, G, pt);
					find_cliques_parallel(depth+1, param, G);
				}
			}
		} else {
			T* candidate_nodes = param.get_candidates(depth-1);
			T num_candidate_nodes = candidate_nodes[0];
			for (T i=0; i < num_candidate_nodes; ++i) {
				T pt = candidate_nodes[i+1];
				param.current_cliques[depth] = pt;
				populate_candidates(depth, param, G, pt);
				find_cliques_parallel(depth+1, param, G);
			}
		}
	}

	/**
	 *
	 */
	template<typename T, typename U>
	__forceinline__
	static void populate_candidates(size_t depth, PARAMS<T>& param, Graph<T,U>& G, T node) {
		register T* candidates = param.get_candidates(depth);
		register T* candidates_prev_depth = param.get_candidates(depth-1);
		register T candidates_prev_depth_size = candidates_prev_depth[0];
		register T k = 1;
		for (T i=0; i<candidates_prev_depth_size; ++i) {
			T pt = candidates_prev_depth[i+1];
			if (G.is_adjacent(pt, node) && pt > node) {
				candidates[k++] = pt;
			}
		}
		candidates[0] = k-1;
	}

	/**
	 *
	 */
	template<typename T, typename U>
	__forceinline__
	static void populate_adjacency(size_t depth, PARAMS<T>& param, Graph<T,U>& G, T node) {
		register T* candidates = param.get_candidates(depth);
		register T k = 1;
		for (T i=0; i<G.nb_vertices; ++i) {
			if (G.is_adjacent(node, i) && i > node) {
				candidates[k++] = i;
			}
		}
		candidates[0] = k-1;
	}
};

#endif /* SRC_LIB_FOUNDATIONS_GRAPH_THEORY_CLIQUE_KCLIQUE_H_ */
