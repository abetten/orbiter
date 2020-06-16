/*
 * KClique.h
 *
 *  Created on: Jun 16, 2020
 *      Author: sajeeb
 */

#include <iostream>
#include "Graph.h"
//#include "chrono.h"
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
	static void find_cliques (Graph<T,U>& G, vector<vector<T>>& soln, uint k, size_t n_threads=0) {
		const size_t nThreads = (n_threads == 0) ? std::thread::hardware_concurrency() : n_threads;
		std::thread threads [nThreads];
		PARAMS<T> params [nThreads];


		for (size_t i=0; i<nThreads; ++i) {
			params[i].init(i, k, G.nb_vertices, nThreads);
			threads[i] = std::thread(find_cliques_parallel<T,U>, 0, 0, std::ref(params[i]), std::ref(G));
		}


		for (size_t i=0; i<nThreads; ++i) threads[i].join();

		// Find the total number of solutions
		size_t nb_sols = 0;
		for (size_t i=0; i<nThreads; ++i) {
			nb_sols += params[i].t_solutions.size();
		}
		soln.reserve(nb_sols);

		std::unordered_set<std::string> solutions_set;
		std::string s;

		for (size_t i=0; i<nThreads; ++i) {
			std::move(params[i].t_solutions.begin(), params[i].t_solutions.end(),
						  std::back_inserter(soln));
		}
	}


private:
	template <typename T>
	class PARAMS {
	public:
		~PARAMS() {
			if (current_cliques) delete [] current_cliques;
			if (visited_nodes) delete [] visited_nodes;
		}
		PARAMS() {}

		void init(uint8_t tid, uint k, uint32_t num_nodes, uint8_t n_threads) {
			this->tid = tid;
			this->n_threads = n_threads;
			this->k = k;
			current_cliques= new T [k];
			memset(current_cliques, -1, sizeof(T)*k);
			visited_nodes = new bool [num_nodes] ();
			node_adjacency = new T [num_nodes];
			for (uint i=0; i<num_nodes; ++i) node_adjacency[i] = i;
		}

		uint8_t tid = 0;	//
		T* current_cliques = NULL;	// Index of current clique
		T* node_adjacency = NULL;
		size_t nb_sol = 0; // number of solutions found by a thread
		bool* visited_nodes = NULL;
		uint8_t n_threads = 0;
		uint k = 0;
		vector<vector<T>> t_solutions;
	};

	/**
	 *
	 */
	template <typename T, typename U>
	static void find_cliques_parallel (uint depth, T end, PARAMS<T>& param, Graph<T,U>& G) {
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
					T end_adj = adjacency_cluster(param, G, depth+1, G.nb_vertices, pt);
					find_cliques_parallel(depth+1, end_adj, param, G);
				}
			}
		} else {
			for (T i=depth; i < end; ++i) {
				T pt = param.node_adjacency[i];
				param.current_cliques[depth] = pt;
				T end_adj = adjacency_cluster(param, G, depth+1, end, pt);
				find_cliques_parallel(depth+1, end_adj, param, G);
			}
		}
	}

	/**
	 *
	 */
	template<typename T, typename U>
	__forceinline__
	static T adjacency_cluster(PARAMS<T>& param, Graph<T,U>& G, uint start, size_t end, T node) {
		T* node_adjacency = param.node_adjacency;
		uint size = start;
		for (T i=start; i<end; ++i) {
			if (G.is_adjacent(node, node_adjacency[i]) && node_adjacency[i] > node) {
				if (size != i) std::swap(node_adjacency[i], node_adjacency[size]);
				size++;
			}
		}
		return size;
	}
};

#endif /* SRC_LIB_FOUNDATIONS_GRAPH_THEORY_CLIQUE_KCLIQUE_H_ */
