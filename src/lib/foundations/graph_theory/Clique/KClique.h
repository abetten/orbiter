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
#include <atomic>
#include <string>
#include <math.h>
#include <CPPLOGGER_ASYNC.h>

#ifndef SRC_LIB_FOUNDATIONS_GRAPH_THEORY_CLIQUE_KCLIQUE_H_
#define SRC_LIB_FOUNDATIONS_GRAPH_THEORY_CLIQUE_KCLIQUE_H_

using std::vector;
using std::cout;
using std::endl;
using std::atomic;

class KClique {
public:
	template <typename T, typename U>
	__forceinline__
	static void find_cliques (Graph<T,U>& G, vector<vector<T>>& soln, uint k, size_t n_threads=0) {
		current_progress = 0;
		total_progress = G.nb_vertices;

		const register size_t nThreads = (n_threads == 0) ? std::thread::hardware_concurrency() : n_threads;
		std::thread threads [nThreads];
		PARAMS<T> params [nThreads];

		// Initialize the params for every thread
		logger_info("Initializing params for each thread");
		for (size_t i=0; i<nThreads; ++i) {
			params[i].init(i, k, G.nb_vertices, nThreads);
			threads[i] = std::thread(find_cliques_parallel<T,U>, 0, std::ref(params[i]), std::ref(G));
		}
		logger_info("Done initializing params for each thread");

		// start the worker threads
		logger_info("Starting worker threads");
		for (size_t i=0; i<nThreads; ++i) threads[i].join();

		logger_info("All worker threads done.");

		// Find the total number of solutions
		logger_info("Reserving the solutions vector");
		register size_t nb_sols = 0;
		for (size_t i=0; i<nThreads; ++i) {
			nb_sols += params[i].t_solutions.size();
		}
		soln.reserve(nb_sols);

		logger_info("Adding solutions to the solutions vector");
		for (size_t i=0; i<nThreads; ++i) {
			std::move(params[i].t_solutions.begin(), params[i].t_solutions.end(), std::back_inserter(soln));
		}

		logger_info("Done finding cliques.");
	}


private:
	static atomic<size_t> current_progress;
	static size_t total_progress;

	template <typename T>
	class PARAMS {
	public:
		~PARAMS() {
			if (current_cliques) delete [] current_cliques;
			if (candidates) delete [] candidates;
		}
		PARAMS() {}

		void init(uint8_t tid, uint k, uint32_t num_nodes, uint8_t n_threads) {
			this->tid = tid;
			this->n_threads = n_threads;
			this->k = k;
			this->num_nodes = num_nodes;
			current_cliques= new T [k];
			memset(current_cliques, -1, sizeof(T)*k);
			candidates = new T [k * (num_nodes+1)] ();
			for (uint i=0; i<num_nodes; ++i) candidates[i] = i;
			t_solutions.reserve(100);
		}

		__forceinline__
		T* get_candidates(size_t depth) {
			return candidates + depth * (num_nodes + 1);
		}

		uint8_t tid = 0;	//
		T* current_cliques = NULL;	// Index of current clique
		T* candidates = NULL;
		uint8_t n_threads = 0;
		uint k = 0;
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
			param.t_solutions.emplace_back(vector<T>(param.current_cliques, param.current_cliques+param.k));
			return;
		}
		if (depth == 0) {
			register size_t nb_vertices = G.nb_vertices;
			for (T pt=0; pt < nb_vertices; ++pt) {
				if ((pt % param.n_threads) == param.tid) {
					printf("Progress: %3.2f%%    \r", current_progress/(double)total_progress*100.0);
					param.current_cliques[depth] = pt;
					populate_adjacency(depth, param, G, pt);
					find_cliques_parallel(depth+1, param, G);
					current_progress += 1; // increment current progress
				}
			}
		} else {
			register T* candidate_nodes = param.get_candidates(depth-1);
			register T num_candidate_nodes = candidate_nodes[0];
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
