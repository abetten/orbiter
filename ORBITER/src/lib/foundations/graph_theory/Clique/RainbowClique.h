#include "Graph.h"
#include <thread>
#include <vector>
#include <algorithm>
#include <iterator>

#ifndef _RAINBOW_CLIQUE_
#define _RAINBOW_CLIQUE_

class RainbowClique {
public:

	class PARAMS {
	public:
		~PARAMS() {
			if (live_pts) delete [] live_pts;
			if (current_cliques) delete [] current_cliques;
			if (color_frequency) delete [] color_frequency;
		}

		uint8_t tid;	//
		size_t* live_pts;	// Store index of points in graph
		size_t* current_cliques;	// Index of current clique
		size_t nb_sol;
		bitset color_satisfied;
		size_t* color_frequency;	//
		uint8_t n_threads;
		std::vector<std::vector<size_t>> t_solutions;
						// store the vertex label of the points in current_clique
	} ;


	__forceinline__
	static void find_cliques (Graph& G, std::vector<std::vector<size_t>>& soln) {
		const size_t nThreads = std::thread::hardware_concurrency();
		std::thread threads [nThreads];
		PARAMS params [nThreads];

		#pragma unroll
		for (size_t i=0; i<nThreads; ++i) {
			params[i].tid = i;
			params[i].live_pts = new size_t [G.nb_vertices] ();
			params[i].current_cliques = new size_t [G.nb_colors] ();
			params[i].nb_sol = 0;
			params[i].color_satisfied.init(G.nb_colors);
			params[i].color_frequency = new size_t [G.nb_colors] ();
			params[i].n_threads = nThreads;

			threads[i] = std::thread(find_cliques_parallel, 0, 0, 0, std::ref(params[i]),
																				std::ref(G));
		}

		#pragma unroll
		for (size_t i=0; i<nThreads; ++i) threads[i].join();

		// Find the total number of solutions
		size_t nb_sols = 0;
		#pragma unroll
		for (size_t i=0; i<nThreads; ++i) nb_sols += params[i].t_solutions.size();
		soln.reserve(nb_sols);


		#pragma unroll
		for (size_t i=0; i<nThreads; ++i) {
			// use std::move to avoid performing copy
			std::move(params[i].t_solutions.begin(), params[i].t_solutions.end(),
						std::back_inserter(soln));
			params[i].t_solutions.clear();
		}
	}

	__forceinline__
	static void find_cliques_parallel (size_t depth, size_t start, size_t end,
																	PARAMS& param, Graph& G) {

		if (depth == G.nb_colors) {
			param.nb_sol += 1;
			param.t_solutions.emplace_back(std::vector<size_t>());
			#pragma unroll
			for (size_t i=0; i<depth; ++i)
				param.t_solutions.at(param.t_solutions.size()-1).emplace_back(
						G.get_label(param.current_cliques[i])
				);
			return;
		}

		size_t end_adj = 0, end_color_class = 0;

		if (depth > 0) {
			size_t pt = param.current_cliques[depth-1];
			end_adj = clump_by_adjacency(G, param.live_pts, start, end, pt);
		} else {
			#pragma unroll
			for (size_t i=0; i<G.nb_vertices; ++i) param.live_pts[i] = i;
			end_adj = G.nb_vertices;
		}

		size_t lowest_color = get_color_with_lowest_frequency_(G, param.live_pts,
																param.color_frequency,
																param.color_satisfied,
																start, end_adj);

		end_color_class = clump_color_class(G, param.live_pts, start, end_adj, lowest_color);


		param.color_satisfied.set(lowest_color);

		// find how many points are there with the lowest value at current depth
		if (depth == 0) {
			#pragma unroll
			for (size_t i=start; i<end_color_class; ++i) {
				if ((i % param.n_threads) == param.tid) {
					param.current_cliques[depth] = param.live_pts[i];
					find_cliques_parallel(depth+1, end_color_class, end_adj, param, G);
					std::cout<<"Progress: "<<((double)i+1)/(end_color_class-start)<<"% \r";
					std::cout << std::flush;
				}
			}
		} else {
			#pragma unroll
			for (size_t i=start; i<end_color_class; ++i) {
				param.current_cliques[depth] = param.live_pts[i];
				find_cliques_parallel(depth+1, end_color_class, end_adj, param, G);
			}
		}

		param.color_satisfied.unset(lowest_color);

	}

	__forceinline__
	static inline size_t clump_by_adjacency(Graph& G, size_t* live_pts, size_t start,
																	size_t end, size_t node) {
		#pragma unroll
		for (size_t i = start; i<end; ++i) {
			if (G.is_adjacent(node, live_pts[i])) {
				if (start != i) std::swap(live_pts[i], live_pts[start]);
				start++;
			}
		}
		return start;
	}

	__forceinline__
	static inline void create_color_freq_of_live_points(Graph& G, size_t* live_pts,
										 size_t* color_frequency, size_t start, size_t end) {
		// any point in the graph that is dead will have a negative value in the
		// live_point array

		// reset color_frequency stats
		memset(color_frequency, 0, sizeof(size_t)*G.nb_colors);

		#pragma unroll
		for (size_t i = start; i < end; ++i) {
			size_t point_color = G.vertex_color [live_pts[i]];
			color_frequency[point_color] += 1;
		}
	}

	__forceinline__
	static inline size_t get_color_with_lowest_frequency_(Graph& G, size_t* live_pts,
				size_t* color_frequency, bitset& color_satisfied, size_t start, size_t end) {

		create_color_freq_of_live_points(G, live_pts, color_frequency, start, end);

		// returns index of the lowest value in t he array
		size_t max = INT64_MAX;
		size_t min_element = max;
		size_t return_value = -1;
		#pragma unroll
		for (size_t i = 0; i < G.nb_colors; ++i) {
			if (color_frequency[i] < min_element && !color_satisfied[i]) {
				min_element = color_frequency[i];
				return_value = i;
			}
		}
		return return_value;

	}

	__forceinline__
	static inline size_t clump_color_class(Graph& G, size_t* live_pts, size_t start,
																size_t end, size_t color) {
		#pragma unroll
		for (size_t i = start; i<end; ++i) {
			if (G.get_color(live_pts[i]) == color) {
				std::swap(live_pts[start], live_pts[i]);
				start += 1;
			}
		}
		return start;
	}

};

#endif
