#include "Graph.h"
#include <thread>
#include <vector>
#include <algorithm>
#include <iterator>
#include <limits>

#ifndef _RAINBOW_CLIQUE_
#define _RAINBOW_CLIQUE_


class RainbowClique {
public:

    template <typename T, typename U>
    __forceinline__
    static void find_cliques (Graph<T,U>& G, std::vector<std::vector<T>>& soln, size_t n_threads=0) {
        const size_t nThreads = (n_threads == 0) ? std::thread::hardware_concurrency() : n_threads;
        std::thread threads [nThreads];
        PARAMS<T> params [nThreads];

        #pragma unroll
        for (size_t i=0; i<nThreads; ++i) {
            params[i].tid = i;
            params[i].live_pts = new T [G.nb_vertices] ();
            params[i].current_cliques = new T [G.nb_colors] ();
            params[i].nb_sol = 0;
            params[i].color_satisfied = new bool [G.nb_colors] ();
            params[i].color_frequency = new T [G.nb_colors] ();
            params[i].n_threads = nThreads;

            threads[i] = std::thread(find_cliques_parallel<T,U>, 0, 0, 0, std::ref(params[i]),
                                     &(params[0]), std::ref(G));
        }

        #pragma unroll
        for (size_t i=0; i<nThreads; ++i) threads[i].join();

        // Find the total number of solutions
        size_t nb_sols = 0;
        #pragma unroll
        for (size_t i=0; i<nThreads; ++i) {
            nb_sols += params[i].t_solutions.size();
        }
        soln.reserve(nb_sols);

        #pragma unroll
        for (size_t i=0; i<nThreads; ++i) {
            // use std::move to avoid performing intermediate copy ops when
            // putting solutions into soln vector
            std::move(params[i].t_solutions.begin(), params[i].t_solutions.end(),
                      std::back_inserter(soln));
            params[i].t_solutions.clear();
        }
    }

private:

    template <typename T>
    class PARAMS {
    public:
        ~PARAMS() {
            if (live_pts) delete [] live_pts;
            if (current_cliques) delete [] current_cliques;
            if (color_frequency) delete [] color_frequency;
            if (color_satisfied) delete [] color_satisfied;
        }

        uint8_t tid;	//
        T* live_pts;	// Store index of points in graph
        T* current_cliques;	// Index of current clique
        size_t nb_sol;
        bool* color_satisfied;
        T* color_frequency;	//
        uint8_t n_threads;
        std::vector<std::vector<T>> t_solutions;
    };

    template <typename T, typename U>
    static void find_cliques_parallel (size_t depth, T start, T end, 
                                        PARAMS<T>& param, PARAMS<T>* params, Graph<T,U>& G) {

        if (depth == G.nb_colors/G.nb_colors_per_vertex) {
            param.nb_sol += 1;
            param.t_solutions.emplace_back(std::vector<T>(G.nb_colors));
            #pragma unroll
            for (size_t i=0; i<depth; ++i)
                param.t_solutions.at(param.t_solutions.size()-1).emplace_back(
                        G.get_label(param.current_cliques[i])
                );
            if (param.tid == 0) {
                printf("                                                        ");
                fflush(stdout);
                size_t ns = 0;
                for (size_t j=0; j<param.n_threads; ++j) ns += params[j].nb_sol;
                printf("\rnb_sol: %ld                                         \r", ns);
                fflush(stdout);
            }
            return;
        }

        T end_adj = 0;
        T end_color_class = 0;

        if (depth > 0) {
            auto pt = param.current_cliques[depth-1];
            end_adj = clump_by_adjacency(G, param.live_pts, start, end, pt);
        } else {
            #pragma unroll
            for (size_t i=0; i<G.nb_vertices; ++i) param.live_pts[i] = i;
            end_adj = G.nb_vertices;
        }

        U lowest_color = get_color_with_lowest_frequency_(G, param.live_pts,
                                                          param.color_frequency,
                                                          param.color_satisfied,
                                                          start, end_adj);

        end_color_class = clump_color_class(G, param.live_pts, start, 
                                            end_adj, lowest_color, param.color_satisfied);

//        param.color_satisfied[lowest_color] = true;


        // find how many points are there with the lowest value at current depth
        if (depth == 0) {
            #pragma unroll
            for (size_t i=start; i<end_color_class; ++i) {
            	if (param.tid == 0) {
                    size_t ns = 0;
                    for (size_t j=0; j<param.n_threads; ++j) ns += params[j].nb_sol;
					printf("%ld\tof\t%ld\t", i, end_color_class);
					printf("Progress: %.2f%", i/double(end_color_class-start)*100);
                    
                    if (i == 0)
                        printf("\t\tn_sol: %ld                    \r", 0);
                    else
                        printf("\t\tn_sol: %ld                    \r", ns);
					
                    fflush(stdout);
            	}
                if ((i % param.n_threads) == param.tid) {
                    satisfy_color(G, param, i, true);
                    param.current_cliques[depth] = param.live_pts[i];
                    find_cliques_parallel(depth+1, end_color_class, end_adj, param, params, G);
                    satisfy_color(G, param, i, false);
                }
            }
        } else {
            #pragma unroll
            for (size_t i=start; i<end_color_class; ++i) {
            	satisfy_color(G, param, i, true);
                param.current_cliques[depth] = param.live_pts[i];
                find_cliques_parallel(depth+1, end_color_class, end_adj, param, params, G);
                satisfy_color(G, param, i, false);
            }
        }

//        param.color_satisfied[lowest_color] = false;

    }

    /**
     * Mark all the colors associated with node i in the graph as satisfied
     */
    template <typename T, typename U>
	__forceinline__
	static inline void satisfy_color (Graph<T,U>& G, PARAMS<T>& param, size_t i, bool value) {
		#pragma unroll
		for (size_t j=0; j < G.nb_colors_per_vertex; ++j) {
			param.color_satisfied[G.get_color(param.live_pts[i], j)] = value;
		}
    }

    template <typename T, typename U>
    __forceinline__
    static inline T clump_by_adjacency(Graph<T,U>& G, T* live_pts, T start,
                                       T end, T node) {
        #pragma unroll
        for (T i = start; i<end; ++i) {
            if (G.is_adjacent(node, live_pts[i])) {
                if (start != i) std::swap(live_pts[i], live_pts[start]);
                start++;
            }
        }
        return start;
    }

    template <typename T, typename U>
    __forceinline__
    static inline void create_color_freq_of_live_points(Graph<T,U>& G, T* live_pts,
                                                        T* color_frequency, T start, T end) {
        // any point in the graph that is dead will have a negative value in the
        // live_point array

        // reset color_frequency stats
        memset(color_frequency, 0, sizeof(T)*G.nb_colors);

        #pragma unroll
        for (size_t i = start; i < end; ++i) {
        	for (size_t j=0; j < G.nb_colors_per_vertex; ++j) {
        		const U point_color = G.get_color(live_pts[i], j);
        		color_frequency[point_color] += 1;
        	}
        }
    }

    template <typename T, typename U>
    __forceinline__
    static inline U get_color_with_lowest_frequency_(Graph<T,U>& G, T* live_pts, T* color_frequency,
													 	 	 bool* color_satisfied, T start, T end) {

        create_color_freq_of_live_points(G, live_pts, color_frequency, start, end);

        // returns index of the lowest value in t he array
        T min_element = std::numeric_limits<T>::max();
        U return_value = 0;
        #pragma unroll
        for (U i=0; i < G.nb_colors; ++i) {
            if (color_frequency[i] < min_element && !color_satisfied[i]) {
                min_element = color_frequency[i];
                return_value = i;
            }
        }
        return return_value;

    }

    #if 1
    template <typename T, typename U>
    __forceinline__
    static inline T clump_color_class(Graph<T,U>& G, T* live_pts, T start, T end, U color, 
                                                                        bool* color_satisfied) {
        #pragma unroll
        for (size_t i=start; i<end; ++i) {
            const T pt = live_pts[i];
            bool pick_vertex = false;
            #pragma unroll
            for (size_t j=0; j<G.nb_colors_per_vertex; ++j) {
                const U pt_color = G.get_color(pt, j);
            	if (pt_color == color) {
                    pick_vertex = true;
				} else if (color_satisfied[pt_color]) {
                    pick_vertex = false;
                    break;
                }
            }
            if (pick_vertex) {
                std::swap(live_pts[start], live_pts[i]);
                start += 1;
            }
        }
        return start;
    }
    #else
    template <typename T, typename U>
    __forceinline__
    static inline T clump_color_class(Graph<T,U>& G, T* live_pts, T start, T end, U color, 
                                                                        bool* color_satisfied) {
        #pragma unroll
        for (size_t i=start; i<end; ++i) {
            const T pt = live_pts[i];
            #pragma unroll
            for (size_t j=0; j<G.nb_colors_per_vertex; ++j) {
            	if (G.get_color(pt, j) == color) {
                    std::swap(live_pts[start], live_pts[i]);
                    start += 1;
                    break;
                }
            }
        }
        return start;
    }
    #endif

};

#endif
