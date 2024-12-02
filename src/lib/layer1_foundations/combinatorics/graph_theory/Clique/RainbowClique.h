/*
 * RainbowClique.h
 *
 *  Created on: Aug 25, 2019
 *      Author: sajeeb
 */


#include <thread>
#include <vector>
#include <algorithm>
#include <iterator>
#include <limits>
#include <cstdint>

#include "../../../combinatorics/graph_theory/Clique/Graph.h"
	// for uint32_t


#ifndef _RAINBOW_CLIQUE_
#define _RAINBOW_CLIQUE_

// #define COLLECT_RUNTIME_STATS 
#define COLLECT_RUNTIME_STATS_LIMIT 99999999 // number of sample data points

class RainbowClique {
public:

    template <typename T, typename U>
    __forceinline__
    static void find_cliques (Graph<T,U>& G, std::vector<std::vector<T>>& soln, size_t n_threads=0) {
        size_t nThreads = (n_threads == 0) ? std::thread::hardware_concurrency() : n_threads;
        std::thread threads [nThreads];
        PARAMS<T> params [nThreads];


        for (size_t i=0; i<nThreads; ++i) {
            params[i].tid = i;
            params[i].live_pts = new T [G.nb_vertices] ();
            params[i].current_cliques = new T [G.nb_colors] ();
            params[i].nb_sol = 0;
            params[i].color_satisfied = new bool [G.nb_colors] ();
            params[i].color_frequency = new T [G.nb_colors] ();
            params[i].n_threads = nThreads;

            for (size_t j = 0; j < G.nb_colors; j++) {
            	params[i].color_satisfied[j] = false;
            }

            threads[i] = std::thread(find_cliques_parallel<T,U>, 0, 0, 0, std::ref(params[i]),
                                     &(params[0]), std::ref(G));
        }


        for (size_t i=0; i<nThreads; ++i) threads[i].join();

        // Find the total number of solutions
        size_t nb_sols = 0;

        for (size_t i=0; i<nThreads; ++i) {
            nb_sols += params[i].t_solutions.size();
        }
        soln.reserve(nb_sols);


        for (size_t i=0; i<nThreads; ++i) {
            // use std::move to avoid performing intermediate copy ops when
            // putting solutions into soln vector
            std::move(params[i].t_solutions.begin(), params[i].t_solutions.end(),
                      std::back_inserter(soln));
            params[i].t_solutions.clear();
        }

        #ifdef COLLECT_RUNTIME_STATS
        RainbowClique::dump_runtime_stats("runtime_stats.bin", &params[0], nThreads);
        #endif
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

        #ifdef COLLECT_RUNTIME_STATS
        std::vector<uint64_t> satisfy_color;
        std::vector<uint64_t> adjacency_cluster;
        std::vector<uint64_t> live_pt_color_frequency;
        std::vector<uint64_t> lowest_color_frequency;
        std::vector<uint64_t> cluster_color;
        #endif

    };

    template <typename T, typename U>
    static void find_cliques_parallel (size_t depth, T start, T end, 
                                        PARAMS<T>& param, PARAMS<T>* params, Graph<T,U>& G) {

        if (depth == G.nb_colors/G.nb_colors_per_vertex) {
            param.nb_sol += 1;
            param.t_solutions.emplace_back(std::vector<T>());

            for (size_t i=0; i<depth; ++i)
                param.t_solutions.at(param.t_solutions.size()-1).emplace_back(
                        /*G.get_label(*/param.current_cliques[i]/*)*/
                );

            if (param.tid == 0) {
                //printf("                                                        ");
                //fflush(stdout);
                size_t ns = 0;
           		printf("nb_sol=");
                for (size_t j=0; j<param.n_threads; ++j) {
                	ns += params[j].nb_sol;
                	printf("%ld", (long int) params[j].nb_sol);
                	if (j < param.n_threads - 1) {
                		printf("+");
                	}
                }
        		printf("=%ld\n", (long int) ns);
                fflush(stdout);
            }
            return;
        }

        T end_adj = 0;
        T end_color_class = 0;

        if (depth > 0) {
            T pt = param.current_cliques[depth-1];
            end_adj = clump_by_adjacency(G, param, start, end, pt);
        } else {

            for (size_t i=0; i<G.nb_vertices; ++i) param.live_pts[i] = i;
            end_adj = G.nb_vertices;
        }

        U lowest_color = get_color_with_lowest_frequency_(G, param, start, end_adj);

#if 0
        if (param.tid == 0) {
            printf("thread %ld "
            		"level %ld "
            		"start = %ld "
               		"end_adj = %ld "
               		"lowest_color = %ld "
            		"\n",
					(long int) param.tid, (long int) depth, (long int) start, (long int) end_adj, (long int) lowest_color);
            fflush(stdout);
        }
#endif

        end_color_class = clump_color_class(G, param, start, end_adj, lowest_color);

#if 0
        if (param.tid == 0) {
            printf("thread %ld "
            		"level %ld "
            		"end_color_class = %ld "
            		"G.nb_colors = %ld "
            		"G.nb_colors_per_vertex = %ld"
            		"\n",
					(long int) param.tid, (long int) depth, (long int) end_color_class, (long int) G.nb_colors, (long int) G.nb_colors_per_vertex);
            fflush(stdout);
        }
#endif


//        param.color_satisfied[lowest_color] = true;


        #ifdef COLLECT_RUNTIME_STATS
        if (param.satisfy_color.size() > COLLECT_RUNTIME_STATS_LIMIT) {
            return;
        }
        #endif


        // find how many points are there with the lowest value at current depth
        if (depth == 0) {

            for (size_t i=start; i<end_color_class; ++i) {
                if ((i % param.n_threads) == param.tid) {

#if 1
                    if (param.tid == 0) {
                        printf("thread %ld "
                        		"level %ld "
                        		"at %ld of %ld"
                        		"\n",
								(long int) param.tid, (long int) depth, (long int) i, (long int) end_color_class);
                        fflush(stdout);
                    }
#endif

                	satisfy_color(G, param, i, true);
                    param.current_cliques[depth] = param.live_pts[i];
                    find_cliques_parallel(depth+1, end_color_class, end_adj, param, params, G);
                    satisfy_color(G, param, i, false);
                    #ifdef COLLECT_RUNTIME_STATS
                    if (param.satisfy_color.size() > COLLECT_RUNTIME_STATS_LIMIT) {
                        if (param.tid == 0)
                            RainbowClique::dump_runtime_stats("runtime_stats.bin", params, param.n_threads);
                        return;
                    }
                    #endif
                }
            }
        } else {

            for (size_t i=start; i<end_color_class; ++i) {
            	satisfy_color(G, param, i, true);
                param.current_cliques[depth] = param.live_pts[i];
                find_cliques_parallel(depth+1, end_color_class, end_adj, param, params, G);
                satisfy_color(G, param, i, false);
                #ifdef COLLECT_RUNTIME_STATS
                if (param.satisfy_color.size() > COLLECT_RUNTIME_STATS_LIMIT) {
                    return;
                }
                #endif
            }
        }

//        param.color_satisfied[lowest_color] = false;

    }

    /**
     * Mark all the colors associated with node i in the graph as satisfied
     */
    template <typename T, typename U>
	__forceinline__
	static void satisfy_color (Graph<T,U>& G, PARAMS<T>& param, size_t i, bool value) {
		
        #ifdef COLLECT_RUNTIME_STATS
        chrono_ C;
        #endif
        

		for (size_t j=0; j < G.nb_colors_per_vertex; ++j) {
			param.color_satisfied[G.get_color(param.live_pts[i], j)] = value;
		}

        #ifdef COLLECT_RUNTIME_STATS
        param.satisfy_color.emplace_back(C.calculateDuration(chrono_()));
        #endif        
    }

    template <typename T, typename U>
    __forceinline__
    static T clump_by_adjacency(Graph<T,U>& G, PARAMS<T>& param, T start, T end, T node) {
        T* live_pts = param.live_pts;

        #ifdef COLLECT_RUNTIME_STATS
        chrono_ C;
        #endif


        for (T i = start; i<end; ++i) {
            if (G.is_adjacent(node, live_pts[i])) {
                if (start != i) std::swap(live_pts[i], live_pts[start]);
                start++;
            }
        }

        #ifdef COLLECT_RUNTIME_STATS
        param.adjacency_cluster.emplace_back(C.calculateDuration(chrono_()));
        #endif

        return start;
    }

    template <typename T, typename U>
    __forceinline__
    static void create_color_freq_of_live_points(Graph<T,U>& G, PARAMS<T>& param, T start, T end) {
        // any point in the graph that is dead will have a negative value in the
        // live_point array

        T* live_pts = param.live_pts;
        T* color_frequency = param.color_frequency;

        #ifdef COLLECT_RUNTIME_STATS
        chrono_ C;
        #endif

        // reset color_frequency stats
        memset(color_frequency, 0, sizeof(T)*G.nb_colors);


        for (size_t i = start; i < end; ++i) {
        	for (size_t j=0; j < G.nb_colors_per_vertex; ++j) {
        		const U point_color = G.get_color(live_pts[i], j);
        		color_frequency[point_color] += 1;
        	}
        }

        #ifdef COLLECT_RUNTIME_STATS
        param.live_pt_color_frequency.emplace_back(C.calculateDuration(chrono_()));
        #endif
    }

    template <typename T, typename U>
    __forceinline__
    static U get_color_with_lowest_frequency_(Graph<T,U>& G, PARAMS<T>& param, T start, T end) {

        T* color_frequency = param.color_frequency;
        bool* color_satisfied = param.color_satisfied; 

        create_color_freq_of_live_points(G, param, start, end);

        #ifdef COLLECT_RUNTIME_STATS
        chrono_ C;
        #endif

        // returns index of the lowest value in t he array
        T min_element = std::numeric_limits<T>::max();
        U return_value = 0;

        for (U i=0; i < G.nb_colors; ++i) {
            if (color_frequency[i] < min_element && !color_satisfied[i]) {
                min_element = color_frequency[i];
                return_value = i;
            }
        }

        #ifdef COLLECT_RUNTIME_STATS
        param.lowest_color_frequency.emplace_back(C.calculateDuration(chrono_()));
        #endif

        return return_value;
    }

    #if 1
    template <typename T, typename U>
    __forceinline__
    static T clump_color_class(Graph<T,U>& G, PARAMS<T>& param, T start, T end, U color) {
        
        T* live_pts = param.live_pts;
        bool* color_satisfied = param.color_satisfied;

        #ifdef COLLECT_RUNTIME_STATS
        chrono_ C;
        #endif

#if 0
        if (param.tid == 0) {
            printf("thread %ld "
            		"clump_color_class "
               		"start = %ld "
               		"end = %ld "
               		"color = %ld "
            		"\n",
					(long int) param.tid, (long int) start, (long int) end, (long int) color);
            fflush(stdout);
        }
#endif

        for (size_t i=start; i<end; ++i) {
            const T pt = live_pts[i];
            bool pick_vertex = false;


#if 0
            if (param.tid == 0) {
                printf("thread %ld "
                		"clump_color_class "
                   		"i = %ld "
                   		"pt = %ld "
                		"\n",
    					(long int) param.tid, (long int) i, (long int) pt);
                fflush(stdout);
            }
#endif


            for (size_t j=0; j<G.nb_colors_per_vertex; ++j) {
                const U pt_color = G.get_color(pt, j);


#if 0
                if (param.tid == 0) {
                    printf("thread %ld "
                    		"clump_color_class "
                       		"i = %ld "
                       		"pt = %ld "
                       		"pt_color = %ld "
                    		"\n",
        					(long int) param.tid, (long int) i, (long int) pt, (long int) pt_color);
                    fflush(stdout);
                }
#endif

                if (pt_color == color) {
                    pick_vertex = true;
				}
            	if (pick_vertex && color_satisfied[pt_color]) {
                    pick_vertex = false;

#if 0
                    if (param.tid == 0) {
                        printf("thread %ld "
                        		"clump_color_class "
                           		"pt = %ld "
                           		"j = %ld "
                           		"pt_color = %ld "
                           		"rejected b/c color satisfied "
                        		"\n",
            					(long int) param.tid, (long int) pt, (long int) j, (long int) pt_color);
                        fflush(stdout);
                    }
#endif


                    break;
                }
            }
            if (pick_vertex) {
                std::swap(live_pts[start], live_pts[i]);
                start += 1;
            }
        }

        #ifdef COLLECT_RUNTIME_STATS
        param.cluster_color.emplace_back(C.calculateDuration(chrono_()));
        #endif

        return start;
    }
    #else
    template <typename T, typename U>
    __forceinline__
    static T clump_color_class(Graph<T,U>& G, T* live_pts, T start, T end, U color,
                                                                        bool* color_satisfied) {

        for (size_t i=start; i<end; ++i) {
            const T pt = live_pts[i];

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

    template<typename T>
    __forceinline__
    static void dump_runtime_stats(char* filename, PARAMS<T>* params, size_t nThreads) {
        std::ofstream file;
    	file.open (filename);

        for (size_t i=0; i<nThreads; ++i) {
            printf("%-10ld", params[i].satisfy_color.size());
            printf("%-10ld", params[i].adjacency_cluster.size());
            printf("%-10ld", params[i].live_pt_color_frequency.size());
            printf("%-10ld", params[i].lowest_color_frequency.size());
            printf("%-10ld", params[i].cluster_color.size());
            printf("                                    \r");
            fflush(stdout);
        }

        // satisfy_color
        for (size_t i=0; i<nThreads; ++i) {
            for (size_t j=0; j<params[i].satisfy_color.size(); ++j) {
                file << params[i].satisfy_color[j];
                if (j+1 < params[i].satisfy_color.size()) file << " ";
            }
            if (i+1 < nThreads) file << " ";
        }
        file << "\n";

        // adjacency_cluster
        for (size_t i=0; i<nThreads; ++i) {
            for (size_t j=0; j<params[i].adjacency_cluster.size(); ++j) {
                file << params[i].adjacency_cluster[j];
                if (j+1 < params[i].adjacency_cluster.size()) file << " ";
            }
            if (i+1 < nThreads) file << " ";
        }
        file << "\n";

        // live_pt_color_frequency
        for (size_t i=0; i<nThreads; ++i) {
            for (size_t j=0; j<params[i].live_pt_color_frequency.size(); ++j) {
                file << params[i].live_pt_color_frequency[j];
                if (j+1 < params[i].live_pt_color_frequency.size()) file << " ";
            }
            if (i+1 < nThreads) file << " ";
        }
        file << "\n";

        // lowest_color_frequency
        for (size_t i=0; i<nThreads; ++i) {
            for (size_t j=0; j<params[i].lowest_color_frequency.size(); ++j) {
                file << params[i].lowest_color_frequency[j];
                if (j+1 < params[i].lowest_color_frequency.size()) file << " ";
            }
            if (i+1 < nThreads) file << " ";
        }
        file << "\n";

        // cluster_color
        for (size_t i=0; i<nThreads; ++i) {
            for (size_t j=0; j<params[i].cluster_color.size(); ++j) {
                file << params[i].cluster_color[j];
                if (j+1 < params[i].cluster_color.size()) file << " ";
            }
            if (i+1 < nThreads) file << " ";
        }
        file << "\n";

        file.close();
    }

};

#endif
