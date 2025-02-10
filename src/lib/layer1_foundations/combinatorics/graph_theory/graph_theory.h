// graph_theory.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005


#ifndef ORBITER_SRC_LIB_FOUNDATIONS_GRAPH_THEORY_GRAPH_THEORY_H_
#define ORBITER_SRC_LIB_FOUNDATIONS_GRAPH_THEORY_GRAPH_THEORY_H_



namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace graph_theory {



// #############################################################################
// clique_finder_control.cpp
// #############################################################################


#define CLIQUE_FINDER_CONTROL_MAX_RESTRICTIONS 100

//! settings to control the clique finding process


class clique_finder_control {

public:

	// doc/tables/commands/clique_finder_control.csv


	int f_rainbow;

	int f_target_size;
	int target_size;

	int f_weighted;
	int weights_total;
	int weights_offset;
	std::string weights_string;
	std::string weights_bounds;


	int f_Sajeeb;
	int f_nonrecursive;
	int f_output_solution_raw;
	int f_store_solutions;

	int f_output_file;
	std::string output_file;

	int f_maxdepth;
	int maxdepth;

	int f_restrictions;
	int nb_restrictions;
	int restrictions[CLIQUE_FINDER_CONTROL_MAX_RESTRICTIONS * 3];

	int f_tree;
	int f_decision_nodes_only;
	std::string fname_tree;

	int print_interval;

	// extra stuff for the clique finder
	// that does not come from the command line:

	int f_has_additional_test_function;
	void (*call_back_additional_test_function)(
		rainbow_cliques *R, void *user_data,
		int current_clique_size, int *current_clique,
		int nb_pts, int &reduced_nb_pts,
		int *pt_list, int *pt_list_inv,
		int verbose_level);
	void *additional_test_function_data; // previously user_data

	int f_has_print_current_choice_function;
	void (*call_back_print_current_choice)(
			clique_finder *CF,
		int depth, void *user_data, int verbose_level);
	void *print_current_choice_data; // previously user_data



	// output variables:
	unsigned long int nb_search_steps;
	unsigned long int nb_decision_steps;
	int dt;

	int *Sol; // [nb_sol * target_size]
	long int nb_sol;


	clique_finder_control();
	~clique_finder_control();
	int parse_arguments(
			int argc, std::string *argv);
	void print();

};





// #############################################################################
// clique_finder.cpp
// #############################################################################

//! finds all cliques of a certain size in a graph



class clique_finder {
public:


	clique_finder_control *Control;


	std::string label;
	int n; // number of points


	int f_write_tree;
	std::string fname_tree;
	std::ofstream *fp_tree;


	int *point_labels;
	int *point_is_suspicious;
	
	int verbose_level;
	

	int f_has_adj_list;
	int *adj_list_coded;
	int f_has_bitvector;
	other::data_structures::bitvector *Bitvec_adjacency;

	int f_has_row_by_row_adjacency_matrix;
	char **row_by_row_adjacency_matrix; // [n][n]


	int *pt_list;
	int *pt_list_inv;
	int *nb_points;
	int *candidates; // [max_depth * n]
	int *nb_candidates; // [max_depth]
	int *current_choice; // [max_depth]
	int *level_counter; // [max_depth] (added Nov 8, 2014)

	// restrictions for partial search
	int *f_level_mod; // [max_depth] (added Nov 8, 2014)
	int *level_r; // [max_depth] (added Nov 8, 2014)
	int *level_m; // [max_depth] (added Nov 8, 2014)

	int *current_clique; // [max_depth]

	unsigned long int counter; // number of backtrack nodes
	unsigned long int decision_step_counter;
		// number of backtrack nodes that are decision nodes

	// solution storage:
	std::deque<std::vector<int> > solutions;
	long int nb_sol;


	// callbacks:
	void (*call_back_clique_found)(
			clique_finder *CF, int verbose_level);
	
	// added May 26, 2009:
	void (*call_back_add_point)(
			clique_finder *CF,
		int current_clique_size, int *current_clique, 
		int pt, int verbose_level);
	void (*call_back_delete_point)(
			clique_finder *CF,
		int current_clique_size, int *current_clique, 
		int pt, int verbose_level);
	int (*call_back_find_candidates)(
			clique_finder *CF,
		int current_clique_size, int *current_clique, 
		int nb_pts, int &reduced_nb_pts, 
		int *pt_list, int *pt_list_inv, 
		int *candidates, int verbose_level);
		// Jan 2, 2012: added reduced_nb_pts and pt_list_inv

	int (*call_back_is_adjacent)(
			clique_finder *CF,
		int pt1, int pt2, int verbose_level);

	// added Oct 2011:
	void (*call_back_after_reduction)(
			clique_finder *CF,
		int depth, int nb_points, int verbose_level);


	void *call_back_clique_found_data1;
	void *call_back_clique_found_data2;
	
	
	clique_finder();
	~clique_finder();
	void null();
	void free();
	void init(
			clique_finder_control *Control,
			std::string &label, int n,
			int f_has_adj_list, int *adj_list_coded,
			int f_has_bitvector,
			other::data_structures::bitvector *Bitvec_adjacency,
			int verbose_level);
	void init_restrictions(
			int *restrictions, int verbose_level);
	void init_point_labels(
			int *pt_labels);
	void init_suspicious_points(
			int nb, int *point_list);
	void clique_finder_backtrack_search(
			int depth, int verbose_level);
	int solve_decision_problem(
			int depth, int verbose_level);
		// returns true if we found a solution
	void open_tree_file(
			std::string &fname_base);
	void close_tree_file();
	void get_solutions(
			int *&Sol, long int &nb_solutions, int &clique_sz,
		int verbose_level);
	void print_suspicious_points();
	void print_set(
			int size, int *set);
	void print_suspicious_point_subset(
			int size, int *set);
	void log_position_and_choice(
			int depth,
			unsigned long int  counter_save,
			unsigned long int counter);
	void log_position(
			int depth,
			unsigned long int  counter_save,
			unsigned long int counter);
	void log_choice(
			int depth);
	void swap_point(
			int idx1, int idx2);
	void degree_of_point_statistic(
			int depth, int nb_points,
		int verbose_level);
	int degree_of_point(
			int depth, int i, int nb_points);
	int is_suspicious(
			int i);
	int point_label(
			int i);
	int is_adjacent(
			int depth, int i, int j);
	int is_viable(
			int depth, int pt);
	void write_entry_to_tree_file(
			int depth, int verbose_level);
	int s_ij(
			int i, int j);
	void delinearize_adjacency_list(
			int verbose_level);
	void write_solutions(
			std::string &fname,
			int verbose_level);

private:
	void parallel_delinearize_adjacency_list();
};

// #############################################################################
// colored_graph_cliques.cpp
// #############################################################################


//! clique finding activities for graphs


class colored_graph_cliques {
public:


	colored_graph *CG;

	colored_graph_cliques();
	~colored_graph_cliques();
	void init(
			colored_graph *CG, int verbose_level);

	void early_test_func_for_clique_search(
			long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void early_test_func_for_coclique_search(
			long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void all_cliques(
			clique_finder_control *Control,
			std::string &graph_label,
			std::vector<std::string> &feedback,
			clique_finder *&CF,
			int verbose_level);
	void all_cliques_rainbow(
			clique_finder_control *Control,
			int verbose_level);
	void all_cliques_black_and_white(
			clique_finder_control *Control,
			clique_finder *&CF,
			int verbose_level);
	void do_Sajeeb(
			clique_finder_control *Control,
			std::vector<std::vector<unsigned int> > &solutions,
			int verbose_level);
	void do_Sajeeb_black_and_white(
			clique_finder_control *Control,
			std::vector<std::vector<unsigned int> >& solutions,
			int verbose_level);
	void all_cliques_weighted_with_two_colors(
			clique_finder_control *Control,
			int verbose_level);
	void all_cliques_of_size_k_ignore_colors(
			clique_finder_control *Control,
			clique_finder *&CF,
			int verbose_level);
	void all_rainbow_cliques(
			clique_finder_control *Control,
			int verbose_level);
	int test_Neumaier_property(
			int &regularity,
			int &lambda_value,
			int clique_size,
			int &nexus,
			int verbose_level);
	int test_if_clique_is_regular(
			int *clique, int sz, int &nexus, int verbose_level);

};





// #############################################################################
// colored_graph.cpp
// #############################################################################


//! a graph with a vertex coloring


class colored_graph {
public:


	std::string fname_base;

	std::string label;
	std::string label_tex;

	int nb_points;
	int nb_colors;
	int nb_colors_per_vertex; // = 1 by default
	
	long int L;
	
	long int *points; // [nb_points]
	int *point_color; // [nb_points * nb_colors_per_vertex]
	

	int user_data_size;
	long int *user_data; // [user_data_size]

	int f_ownership_of_bitvec;
	other::data_structures::bitvector *Bitvec;

	int f_has_list_of_edges;
	int nb_edges;
	int *list_of_edges;
		// used in early_test_func_for_path_and_cycle_search

	colored_graph_cliques *Colored_graph_cliques;

	colored_graph();
	~colored_graph();
	void compute_edges(
			int verbose_level);
	int is_adjacent(
			int i, int j);
	void set_adjacency(
			int i, int j, int a);
	void set_adjacency_k(
			long int k, int a);
	long int hash();
	std::string stringify_adjacency_list();
	void partition_by_color_classes(
		int *&partition, int *&partition_first, 
		int &partition_length, 
		int verbose_level);
	colored_graph *sort_by_color_classes(
			int verbose_level);
	colored_graph *subgraph_by_color_classes(
			int c, int verbose_level);
	colored_graph *subgraph_by_color_classes_with_condition(
			int *seed_pts, int nb_seed_pts,
			int c, int verbose_level);
	void print();
	void print_points_and_colors();
	void print_adjacency_list();
	void init_basic(
			int nb_points,
		std::string &label, std::string &label_tex,
		int verbose_level);
	void init_from_bitvector(
			int nb_points, int nb_colors, int nb_colors_per_vertex,
		int *colors, other::data_structures::bitvector *Bitvec,
		int f_ownership_of_bitvec,
		std::string &label, std::string &label_tex,
		int verbose_level);
	void init_with_point_labels(
			int nb_points, int nb_colors, int nb_colors_per_vertex,
		int *colors, other::data_structures::bitvector *Bitvec,
		int f_ownership_of_bitvec,
		long int *point_labels,
		std::string &label, std::string &label_tex,
		int verbose_level);
	void init_no_colors(
			int nb_points, other::data_structures::bitvector *Bitvec,
		int f_ownership_of_bitvec, 
		std::string &label, std::string &label_tex,
		int verbose_level);
	void init_adjacency(
			int nb_points, int nb_colors, int nb_colors_per_vertex,
		int *colors, int *Adj,
		std::string &label, std::string &label_tex,
		int verbose_level);
	void init_adjacency_upper_triangle(
			int nb_points, int nb_colors, int nb_colors_per_vertex,
		int *colors, int *Adj,
		std::string &label, std::string &label_tex,
		int verbose_level);
	void init_from_adjacency_no_colors(
			int nb_points, int *Adj,
			std::string &label, std::string &label_tex,
		int verbose_level);
	void init_adjacency_two_colors(
			int nb_points,
		int *Adj, int *subset, int sz,
		std::string &label, std::string &label_tex,
		int verbose_level);
	void init_user_data(
			long int *data, int data_size, int verbose_level);
	void save(
			std::string &fname, int verbose_level);
	void save_DIMACS(
			std::string &fname, int verbose_level);
	void load(
			std::string &fname, int verbose_level);
	void draw_on_circle(
			std::string &fname,
			other::graphics::layered_graph_draw_options *Draw_options,
		int verbose_level);
	void draw_on_circle_2(
			other::graphics::mp_graphics &G,
			other::graphics::layered_graph_draw_options *Draw_options);
	void create_bitmatrix(
			other::data_structures::bitmatrix *&Bitmatrix,
		int verbose_level);
	void draw(
			std::string &fname,
			other::graphics::layered_graph_draw_options *Draw_options,
		int verbose_level);
	void draw_Levi(
			std::string &fname,
			other::graphics::layered_graph_draw_options *Draw_options,
		int f_partition, int nb_row_parts, int *row_part_first, 
		int nb_col_parts, int *col_part_first, 
		int m, int n, int f_draw_labels, 
		int verbose_level);
	void draw_with_a_given_partition(
			std::string &fname,
			other::graphics::layered_graph_draw_options *Draw_options,
			int *parts, int nb_parts,
			int verbose_level);
	void draw_partitioned(
			std::string &fname,
			other::graphics::layered_graph_draw_options *Draw_options,
		int f_labels,
		int verbose_level);
	colored_graph *compute_neighborhood_subgraph(
		int pt,
		other::data_structures::fancy_set *&vertex_subset,
		other::data_structures::fancy_set *&color_subset,
		int verbose_level);
	colored_graph *compute_neighborhood_subgraph_based_on_subset(
		long int *subset, int subset_sz,
		other::data_structures::fancy_set *&vertex_subset,
		other::data_structures::fancy_set *&color_subset,
		int verbose_level);
	void common_neighbors(
		int *Pts, int nb_pts,
		other::data_structures::fancy_set *&vertex_subset,
		int verbose_level);
	void export_to_magma(
			std::string &fname, int verbose_level);
	void export_to_maple(
			std::string &fname, int verbose_level);
	void export_to_file(
			std::string &fname, int verbose_level);
	void export_to_text(
			std::string &fname, int verbose_level);
	void export_laplacian_to_file(
			std::string &fname,
		int verbose_level);
	void export_to_file_matlab(
			std::string &fname, int verbose_level);
	void export_to_csv(
			std::string &fname, int verbose_level);
	void export_to_graphviz(
			std::string &fname, int verbose_level);
	void early_test_func_for_path_and_cycle_search(
			long int *S, int len,
			long int *candidates, int nb_candidates,
			long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	int is_cycle(
			int nb_e, long int *edges, int verbose_level);
	void find_subgraph(
			std::string &subgraph_label, int verbose_level);
	void find_subgraph_E6(
			int verbose_level);
	void find_subgraph_An(
			int n,
			std::vector<std::vector<int> > &Solutions,
			int verbose_level);
	void find_subgraph_An_recursion(
			int n,
			int *T,
			std::vector<int> &Candidates,
			std::vector<std::vector<int> > &Solutions,
			int current_depth, int *subgraph,
			int verbose_level);
	void write_solutions_to_csv_file(
			std::string &fname_sol,
			std::vector<std::vector<unsigned int> >& solutions,
			clique_finder_control *Control,
			int verbose_level);
	void complement(
			int verbose_level);
	int test_SRG_property(
			int &lambda_value,
			int &mu_value,
			int verbose_level);
	int test_lambda_property(
			int &lambda_value,
			int verbose_level);
	int test_mu_property(
			int &mu_value,
			int verbose_level);
	int partial_lambda_test(
			int from, int to, int lambda,
			int verbose_level);
	void compute_lambda_matrix(
			int *&Lambda,
			int verbose_level);
	void compute_mu_matrix(
			int *&Mu,
			int verbose_level);
	int test_automorphism_property(
			int *perm, int degree,
			int verbose_level);
	void distance_2(
			int verbose_level);
	void reorder(
			std::string &perm_label,
			int verbose_level);
	int get_nb_edges(
			int verbose_level);
	void compute_degree_sequence(
			int *Degree,
			int verbose_level);
	int test_if_regular(
			int &regularity,
			int verbose_level);
	void properties(
			int verbose_level);
	int is_regular(
			int *Degree,
			int &regularity,
			int verbose_level);
	int is_strongly_regular(
			int &k, int &lambda, int &mu,
			int verbose_level);
	void compute_A_and_A_square(
			int *&A,
			int *&A_square,
			int verbose_level);
	int test_distinguishing_property(
			long int *set, int sz,
			int verbose_level);
	void all_distinguishing_codes(
			long int *set, int sz,
			int *code,
			int verbose_level);
	// code[nb_points], where nb_points = number of vertices
	void distinguishing_code_frequency(
			long int *set, int sz,
			int *frequency, int N,
			int verbose_level);
	// frequency[N], where N = 1 << sz
	int distinguishing_code_wrt_set(
			long int *set, int sz, int i);
	void eigenvalues(
			double *&E, int verbose_level);
	void Laplace_eigenvalues(
			double *&E, int verbose_level);

};



// #############################################################################
// graph_layer.cpp
// #############################################################################


//! part of the data structure layered_graph



class graph_layer {
public:
	int id_of_first_node;
	int nb_nodes;
	graph_node *Nodes;
	double y_coordinate;

	graph_layer();
	~graph_layer();
	void init(
			int nb_nodes,
			int id_of_first_node, int verbose_level);
	void place(
			int verbose_level);
	void place_with_grouping(
			int *group_size, int nb_groups,
		double x_stretch, int verbose_level);
	void scale_x_coordinates(
			double x_stretch, int verbose_level);
	void write_memory_object(
			other::orbiter_kernel_system::memory_object *m,
			int verbose_level);
	void read_memory_object(
			other::orbiter_kernel_system::memory_object *m,
			int verbose_level);
};

// #############################################################################
// graph_node.cpp
// #############################################################################

//! part of the data structure layered_graph


class graph_node {
public:
	std::string label;
	int id;

	int f_has_data1;
	int data1;

	int f_has_data2;
	int data2;

	int f_has_data3;
	int data3;

	int f_has_vec_data;
	long int *vec_data;
	int vec_data_len;

	int f_has_distinguished_element; // refers to vec_data
	int distinguished_element_index;

	int layer;
	int neighbor_list_allocated;
	int nb_neighbors;
	int *neighbor_list; // [nb_neighbors]
	int *Edge_color; // [nb_neighbors]
	double x_coordinate;
	
	// added June 28, 2016:
	int nb_children;
	int nb_children_allocated;
	int *child_id; // [nb_children]
	int weight_of_subtree;
	double width;
	int depth_first_node_rank;

	// added May 25, 2017
	double radius_factor;

	graph_node();
	~graph_node();
	void add_neighbor(
			int l, int n, int id, int edge_color);
	void add_text(
			std::string &text);
	void add_vec_data(
			long int *v, int len);
	void set_distinguished_element(
			int idx);
	void add_data1(
			int data);
	void add_data2(
			int data);
	void add_data3(
			int data);
	void write_memory_object(
			other::orbiter_kernel_system::memory_object *m,
			int verbose_level);
	void read_memory_object(
			other::orbiter_kernel_system::memory_object *m,
			int verbose_level);
	void allocate_tree_structure(
			int verbose_level);
	int remove_neighbor(
			layered_graph *G, int id,
			int verbose_level);
	void find_all_parents(
			layered_graph *G, std::vector<int> &All_Parents,
			int verbose_level);
	int find_parent(
			layered_graph *G, int verbose_level);
	void register_child(
			layered_graph *G, int id_child,
			int verbose_level);
	void place_x_based_on_tree(
			layered_graph *G,
			double left, double right,
		int verbose_level);
	void depth_first_rank_recursion(
			layered_graph *G, int &r,
		int verbose_level);
	void scale_x_coordinate(
			double x_stretch, int verbose_level);

};


// #############################################################################
// graph_theory_domain.cpp
// #############################################################################


//! various functions related to graph theory


class graph_theory_domain {
public:
	graph_theory_domain();
	~graph_theory_domain();

	void colored_graph_draw(
			std::string &fname,
			other::graphics::layered_graph_draw_options *Draw_options,
			int f_labels,
			int verbose_level);
	void colored_graph_all_cliques(
			clique_finder_control *Control,
			std::string &fname,
			int f_output_solution_raw,
			int f_output_fname, std::string &output_fname,
			int verbose_level);
#if 0
	void colored_graph_all_cliques_list_of_cases(
			clique_finder_control *Control,
			long int *list_of_cases, int nb_cases,
			std::string &fname_template,
			std::string &fname_sol,
			std::string &fname_stats,
			int f_split, int split_r, int split_m,
			int f_prefix, std::string &prefix,
			int verbose_level);
#endif
	void save_as_colored_graph_easy(
			std::string &fname_base,
			int n, int *Adj, int verbose_level);
	void save_colored_graph(
			std::string &fname,
			int nb_vertices, int nb_colors,
			int nb_colors_per_vertex,
			long int *points, int *point_color,
			long int *data, int data_sz,
			other::data_structures::bitvector *Bitvec,
			int verbose_level);
	void load_colored_graph(
			std::string &fname,
			int &nb_vertices, int &nb_colors,
			int &nb_colors_per_vertex,
			long int *&vertex_labels,
			int *&vertex_colors, long int *&user_data,
			int &user_data_size,
			other::data_structures::bitvector *&Bitvec,
			int verbose_level);
	int is_association_scheme(
			int *color_graph, int n, int *&Pijk,
		int *&colors, int &nb_colors,
		int verbose_level);
	void print_Pijk(
			int *Pijk, int nb_colors);
	void compute_decomposition_of_graph_wrt_partition(
			int *Adj, int N,
		int *first, int *len, int nb_parts, int *&R,
		int verbose_level);
	void draw_bitmatrix(
			std::string &fname_base,
			other::graphics::layered_graph_draw_options *Draw_options,
			int f_dots,
			int f_partition, int nb_row_parts, int *row_part_first,
			int nb_col_parts, int *col_part_first,
			int f_row_grid, int f_col_grid,
			int f_bitmatrix, other::data_structures::bitmatrix *Bitmatrix,
			int *M, int m, int n,
			int f_has_labels, int *labels,
			int verbose_level);
	void list_parameters_of_SRG(
			int v_max, int verbose_level);
	void load_dimacs(
			int *&Adj, int &N,
			std::string &fname,
			int verbose_level);
	void make_cycle_graph(
			int *&Adj, int &N,
			int n, int verbose_level);
	void make_inversion_graph(
			int *&Adj, int &N,
			int *perm, int n, int verbose_level);
	void make_Hamming_graph(
			int *&Adj, int &N,
			int n, int q, int verbose_level);
	void make_Johnson_graph(
			int *&Adj, int &N,
			int n, int k, int s, int verbose_level);
	void make_Paley_graph(
			int *&Adj, int &N,
			algebra::field_theory::finite_field *Fq, int verbose_level);
	void make_Schlaefli_graph(
			int *&Adj, int &N,
			algebra::field_theory::finite_field *F,
			int verbose_level);
	void make_Winnie_Li_graph(
			int *&Adj, int &N,
			algebra::field_theory::finite_field *Fq,
			int index, int verbose_level);
	void make_Grassmann_graph(
			int *&Adj, int &N,
			int n, int k,
			algebra::field_theory::finite_field *F,
			int r, int verbose_level);
	void make_tritangent_plane_disjointness_graph(
			int *&Adj, int &N,
			int verbose_level);
	void make_trihedral_pair_disjointness_graph(
			int *&Adj, int &N,
			int verbose_level);
	void make_non_attacking_queens_graph(
			int *&Adj, int &N,
			int n, int verbose_level);
	void make_disjoint_sets_graph(
			int *&Adj, int &N,
			std::string &fname, int verbose_level);
	void make_Neumaier_graph_16(
			int *&Adj, int &N,
			int verbose_level);
	void make_Neumaier_graph_25(
			int *&Adj, int &N,
			int verbose_level);
	void make_chain_graph(
			int *&Adj, int &N,
			int *part1, int sz1,
			int *part2, int sz2,
			int verbose_level);
	void make_collinearity_graph(
			int *&Adj, int &N,
			int *Inc, int nb_rows, int nb_cols,
			int verbose_level);
	void make_adjacency_bitvector(
			int *&Adj, int *v, int N,
			int verbose_level);
#if 0
	void compute_adjacency_matrix_for_disjoint_sets_graph(
			int *Table, int nb_sets, int set_size,
			std::string &prefix_for_graph,
			data_structures::bitvector *&B,
			int verbose_level);
	void make_graph_of_disjoint_sets_from_rows_of_matrix(
		int *M, int m, int n,
		int *&Adj, int verbose_level);
	void all_cliques_of_given_size(
			int *Adj,
			int nb_pts, int clique_sz, int *&Sol, long int &nb_sol,
			int f_write_cliques, std::string &fname_cliques,
			int verbose_level);
#endif
	void eigenvalues(
			combinatorics::graph_theory::colored_graph *CG,
			int verbose_level);
	void find_subgraph(
			int nb, colored_graph **CG,
			std::string &subgraph_label, int verbose_level);
	void find_subgraph_An(
			int n,
			int nb, colored_graph **CG,
			std::vector<std::vector<int> > &Solutions,
			int verbose_level);
	void find_subgraph_An_recursion(
			int n,
			int nb, colored_graph **CG,
			std::vector<int> &Candidates,
			std::vector<std::vector<int> > &Solutions,
			int current_depth, int *subgraph,
			int verbose_level);
	void find_subgraph_Dn(
			int n,
			int nb, colored_graph **CG,
			std::vector<std::vector<int> > &Solutions,
			int verbose_level);
	void find_subgraph_Dn_recursion_level_0(
			int n,
			int nb, colored_graph **CG,
			std::vector<int> &Candidates,
			std::vector<std::vector<int> > &Solutions,
			int *subgraph,
			int verbose_level);
	void find_subgraph_Dn_recursion_level_1(
			int n,
			int nb, colored_graph **CG,
			std::vector<int> &Candidates,
			std::vector<std::vector<int> > &Solutions,
			int *subgraph,
			int verbose_level);
	void find_subgraph_Dn_recursion_level_2(
			int n,
			int nb, colored_graph **CG,
			std::vector<int> &Candidates,
			std::vector<std::vector<int> > &Solutions,
			int *subgraph,
			int verbose_level);
	void find_subgraph_Dn_recursion_level_3_and_above(
			int n,
			int nb, colored_graph **CG,
			std::vector<int> &Candidates,
			std::vector<std::vector<int> > &Solutions,
			int current_depth, int *subgraph,
			int verbose_level);

};




// #############################################################################
// layered_graph.cpp
// #############################################################################


//! a data structure to store layered graphs or Hasse diagrams


class layered_graph {
public:
	int nb_layers;
	int nb_nodes_total;
	int id_of_first_node;
	graph_layer *L;
	std::string fname_base;
	int f_has_data1;
	int data1;

	layered_graph();
	~layered_graph();
	void init(
			int nb_layers, int *Nb_nodes_layer,
			std::string &fname_base, int verbose_level);
	int nb_nodes();
	void print_nb_nodes_per_level();
	double average_word_length();
	void place(
			int verbose_level);
	void place_upside_down(
			int verbose_level);
	void place_with_y_stretch(
			double y_stretch, int verbose_level);
	void scale_x_coordinates(
			double x_stretch, int verbose_level);
	void place_with_grouping(
			int **Group_sizes, int *Nb_groups,
		double x_stretch,
		int verbose_level);
	void add_edge(
			int l1, int n1, int l2, int n2,
			int edge_color,
			int verbose_level);
	void add_text(
			int l, int n, std::string &text,
			int verbose_level);
	void add_data1(
			int data, int verbose_level);
	void add_node_vec_data(
			int l, int n, long int *v, int len,
		int verbose_level);
	void set_distinguished_element_index(
			int l, int n,
		int index, int verbose_level);
	void add_node_data1(
			int l, int n, int data,
			int verbose_level);
	void add_node_data2(
			int l, int n, int data,
			int verbose_level);
	void add_node_data3(
			int l, int n, int data,
			int verbose_level);
	void draw_with_options(
			std::string &fname,
			other::graphics::layered_graph_draw_options *O,
			int verbose_level);
	void draw_edges(
			other::graphics::layered_graph_draw_options *O,
			other::graphics::mp_graphics *G,
			int verbose_level);
	void draw_vertices(
			other::graphics::layered_graph_draw_options *O,
			other::graphics::mp_graphics *G,
			int verbose_level);
	void draw_level_info(
			other::graphics::layered_graph_draw_options *O,
			other::graphics::mp_graphics *G,
			int verbose_level);
	void coordinates_direct(
			double x_in, double y_in,
		int x_max, int y_max, int f_rotated,
		int &x, int &y);
	void coordinates(
			int id, int x_max, int y_max,
		int f_rotated,
		int &x, int &y);
	void find_node_by_id(
			int id, int &l, int &n);
	void write_file(
			std::string &fname, int verbose_level);
	void read_file(
			std::string &fname, int verbose_level);
	void write_memory_object(
			other::orbiter_kernel_system::memory_object *m,
			int verbose_level);
	void read_memory_object(
			other::orbiter_kernel_system::memory_object *m,
			int verbose_level);
	void remove_edges(
			int layer1, int node1,
			int layer2, int node2,
			std::vector<std::vector<int> > &All_Paths,
			int verbose_level);
	void remove_edge(
			int layer1, int node1,
			int layer2, int node2,
			int verbose_level);
	void find_all_paths_between(
			int layer1, int node1,
			int layer2, int node2,
			std::vector<std::vector<int> > &All_Paths,
			int verbose_level);
	void find_all_paths_between_recursion(
			int layer1, int node1,
			int layer2, int node2,
			int l0, int n0,
			std::vector<std::vector<int> > &All_Paths,
			std::vector<int> &Path,
			int verbose_level);
	void create_spanning_tree(
			int f_place_x, int verbose_level);
	void compute_depth_first_ranks(
			int verbose_level);
	void set_radius_factor_for_all_nodes_at_level(
			int lvl,
		double radius_factor, int verbose_level);
	void make_subset_lattice(
			int n, int depth, int f_tree,
		int f_depth_first, int f_breadth_first,
		int verbose_level);
	void init_poset_from_file(
			std::string &fname,
			int f_grouping, double x_stretch,
			int verbose_level);
};


// #############################################################################
// rainbow_cliques.cpp
// #############################################################################



//! to search for rainbow cliques in graphs




class rainbow_cliques {
public:

	clique_finder_control *Control;

	//std::ostream *ost_sol;
	
	colored_graph *graph;
	clique_finder *CF;
	int *f_color_satisfied;
	int *color_chosen_at_depth;
	int *color_frequency;

	rainbow_cliques();
	~rainbow_cliques();

	void search(
			clique_finder_control *Control,
			colored_graph *graph,
			//std::ostream &ost_sol,
			int verbose_level);
	int find_candidates(
		int current_clique_size, int *current_clique, 
		int nb_pts, int &reduced_nb_pts, 
		int *pt_list, int *pt_list_inv, 
		int *candidates, int verbose_level);
	void clique_found(
			int *current_clique,
			int verbose_level);
	void clique_found_record_in_original_labels(
			int *current_clique,
		int verbose_level);

};



}}}}



#endif /* ORBITER_SRC_LIB_FOUNDATIONS_GRAPH_THEORY_GRAPH_THEORY_H_ */




