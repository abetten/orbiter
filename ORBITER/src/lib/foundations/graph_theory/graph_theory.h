// graph_theory.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005

namespace orbiter {
namespace foundations {


// #############################################################################
// clique_finder.C
// #############################################################################

//! A class that can be used to find cliques in graphs



class clique_finder {
public:
	char label[1000];
	int n; // number of points
	
	int print_interval;
	
	int f_write_tree;
	int f_decision_nodes_only;
	char fname_tree[1000];
	ofstream *fp_tree;

	
	int f_maxdepth;
	int maxdepth;
	
	int *point_labels;
	int *point_is_suspicous;
	
	int target_depth;
	int verbose_level;
	

	int f_has_bitmatrix;
	int bitmatrix_m;
	int bitmatrix_n;
	int bitmatrix_N;
	uchar *bitmatrix_adjacency;

	int f_has_adj_list;
	int *adj_list_coded;
	int f_has_bitvector;
	uchar *bitvector_adjacency;

	int f_has_row_by_row_adjacency_matrix;
	char **row_by_row_adjacency_matrix; // [n][n]


	int *pt_list;
	int *pt_list_inv;
	int *nb_points;
	int *candidates; // [max_depth * n]
	int *nb_candidates; // [max_depth]
	int *current_choice; // [max_depth]
	int *level_counter; // [max_depth] (added Nov 8, 2014)
	int *f_level_mod; // [max_depth] (added Nov 8, 2014)
	int *level_r; // [max_depth] (added Nov 8, 2014)
	int *level_m; // [max_depth] (added Nov 8, 2014)

	int *current_clique; // [max_depth]

	uint counter; // number of backtrack nodes
	uint decision_step_counter;
		// number of backtrack nodes that are decision nodes

	// solution storage:
	int f_store_solutions;
	deque<vector<int> > solutions;
	int nb_sol;


	// callbacks:
	void (*call_back_clique_found)(clique_finder *CF, int verbose_level);
	
	// added May 26, 2009:
	void (*call_back_add_point)(clique_finder *CF, 
		int current_clique_size, int *current_clique, 
		int pt, int verbose_level);
	void (*call_back_delete_point)(clique_finder *CF, 
		int current_clique_size, int *current_clique, 
		int pt, int verbose_level);
	int (*call_back_find_candidates)(clique_finder *CF, 
		int current_clique_size, int *current_clique, 
		int nb_pts, int &reduced_nb_pts, 
		int *pt_list, int *pt_list_inv, 
		int *candidates, int verbose_level);
		// Jan 2, 2012: added reduced_nb_pts and pt_list_inv

	int (*call_back_is_adjacent)(clique_finder *CF, 
		int pt1, int pt2, int verbose_level);
	// added Oct 2011:
	void (*call_back_after_reduction)(clique_finder *CF, 
		int depth, int nb_points, int verbose_level);

	// added Nov 2014:
	int f_has_print_current_choice_function;
	void (*call_back_print_current_choice)(clique_finder *CF, 
		int depth, void *user_data, int verbose_level);
	void *print_current_choice_data;
	
	void *call_back_clique_found_data1;
	void *call_back_clique_found_data2;
	
	
	void open_tree_file(const char *fname_base, 
		int f_decision_nodes_only);
	void close_tree_file();
	void init(const char *label, int n, 
		int target_depth, 
		int f_has_adj_list, int *adj_list_coded, 
		int f_has_bitvector, uchar *bitvector_adjacency, 
		int print_interval, 
		int f_maxdepth, int maxdepth, 
		int f_store_solutions, 
		int verbose_level);
	void delinearize_adjacency_list(int verbose_level);
	void allocate_bitmatrix(int verbose_level);
	void init_restrictions(int *restrictions, int verbose_level);
	clique_finder();
	~clique_finder();
	void null();
	void free();
	void init_point_labels(int *pt_labels);
	void init_suspicous_points(int nb, int *point_list);
	void print_suspicous_points();
	void print_set(int size, int *set);
	void print_suspicous_point_subset(int size, int *set);
	void log_position_and_choice(int depth, int counter_save, 
		int counter);
	void log_position(int depth, int counter_save, int counter);
	void log_choice(int depth);
	void swap_point(int idx1, int idx2);
	void degree_of_point_statistic(int depth, int nb_points, 
		int verbose_level);
	int degree_of_point(int depth, int i, int nb_points);
	//int degree_of_point_verbose(int i, int nb_points);
	int is_suspicous(int i);
	int point_label(int i);
	int is_adjacent(int depth, int i, int j);
	int is_viable(int depth, int pt);
	void write_entry_to_tree_file(int depth, int verbose_level);
	void m_iji(int i, int j, int a);
	int s_ij(int i, int j);
	void backtrack_search(int depth, int verbose_level);
	int solve_decision_problem(int depth, int verbose_level);
		// returns TRUE if we found a solution
	void get_solutions(int *&Sol, int &nb_solutions, int &clique_sz, 
		int verbose_level);
	void backtrack_search_not_recursive(int verbose_level);
};

void all_cliques_of_given_size(int *Adj, int nb_pts, int clique_sz, 
	int *&Sol, int &nb_sol, int verbose_level);



// #############################################################################
// clique_finder_control.C
// #############################################################################


#define CLIQUE_FINDER_CONTROL_MAX_RESTRICTIONS 100

//! a class that controlls the clique finding process


class clique_finder_control {

public:
	int f_rainbow;
	int f_file;
	const char *fname_graph;
	int f_weighted;
	const char *weights_string;
	int f_nonrecursive;
	int f_output_solution_raw;
	int f_output_file;
	const char *output_file;
	int f_maxdepth;
	int maxdepth;
	int f_restrictions;
	int nb_restrictions;
	int restrictions[CLIQUE_FINDER_CONTROL_MAX_RESTRICTIONS * 3];
	int f_tree;
	int f_decision_nodes_only;
	const char *fname_tree;
	int print_interval;
	int nb_search_steps;
	int nb_decision_steps;
	int nb_sol;
	int dt;


	clique_finder_control();
	~clique_finder_control();
	int parse_arguments(
			int argc, const char **argv);
	void all_cliques(
		int verbose_level);
	void all_cliques_weighted(colored_graph *CG,
		const char *fname_sol,
		int verbose_level);
};



// #############################################################################
// colored_graph.C
// #############################################################################


//! a graph with a vertex coloring


class colored_graph {
public:

	char fname_base[1000];
	
	int nb_points;
	int nb_colors;
	
	int bitvector_length;
	int L;
	
	int *points; // [nb_points]
	int *point_color; // [nb_points]
	

	int user_data_size;
	int *user_data; // [user_data_size]

	int f_ownership_of_bitvec;
	uchar *bitvector_adjacency;

	int f_has_list_of_edges;
	int nb_edges;
	int *list_of_edges;

	colored_graph();
	~colored_graph();
	void null();
	void freeself();
	void compute_edges(int verbose_level);
	int is_adjacent(int i, int j);
	void set_adjacency(int i, int j, int a);
	void partition_by_color_classes(
		int *&partition, int *&partition_first, 
		int &partition_length, 
		int verbose_level);
	colored_graph *sort_by_color_classes(int verbose_level);
	colored_graph *subgraph_by_color_classes(
			int c, int verbose_level);
	colored_graph *subgraph_by_color_classes_with_condition(
			int *seed_pts, int nb_seed_pts,
			int c, int verbose_level);
	void print();
	void print_points_and_colors();
	void print_adjacency_list();
	void init(int nb_points, int nb_colors, 
		int *colors, uchar *bitvec, int f_ownership_of_bitvec, 
		int verbose_level);
	void init_with_point_labels(int nb_points, int nb_colors, 
		int *colors, uchar *bitvec, int f_ownership_of_bitvec, 
		int *point_labels, 
		int verbose_level);
	void init_no_colors(int nb_points, uchar *bitvec, 
		int f_ownership_of_bitvec, 
		int verbose_level);
	void init_adjacency(int nb_points, int nb_colors, 
		int *colors, int *Adj, int verbose_level);
	void init_adjacency_upper_triangle(int nb_points, int nb_colors, 
		int *colors, int *Adj, int verbose_level);
	void init_adjacency_no_colors(int nb_points, int *Adj, 
		int verbose_level);
	void init_user_data(int *data, int data_size, int verbose_level);
	void save(const char *fname, int verbose_level);
	void load(const char *fname, int verbose_level);
	void all_cliques_of_size_k_ignore_colors(
		int target_depth,
		int *&Sol, int &nb_solutions,
		int &decision_step_counter,
		int verbose_level);
	void all_cliques_of_size_k_ignore_colors_and_write_solutions_to_file(
		int target_depth, 
		const char *fname, 
		int f_restrictions, int *restrictions, 
		int &nb_sol, int &decision_step_counter, 
		int verbose_level);
	void all_rainbow_cliques(ofstream *fp, int f_output_solution_raw, 
		int f_maxdepth, int maxdepth, 
		int f_restrictions, int *restrictions, 
		int f_tree, int f_decision_nodes_only, const char *fname_tree,  
		int print_interval, 
		int &search_steps, int &decision_steps, int &nb_sol, int &dt, 
		int verbose_level);
	void all_rainbow_cliques_with_additional_test_function(ofstream *fp, 
		int f_output_solution_raw, 
		int f_maxdepth, int maxdepth, 
		int f_restrictions, int *restrictions, 
		int f_tree, int f_decision_nodes_only, const char *fname_tree,  
		int print_interval, 
		int f_has_additional_test_function,
		void (*call_back_additional_test_function)(rainbow_cliques *R, 
			void *user_data, 
			int current_clique_size, int *current_clique, 
			int nb_pts, int &reduced_nb_pts, 
			int *pt_list, int *pt_list_inv, 
			int verbose_level), 
		int f_has_print_current_choice_function,
		void (*call_back_print_current_choice)(clique_finder *CF, 
			int depth, void *user_data, int verbose_level), 
		void *user_data, 
		int &search_steps, int &decision_steps, int &nb_sol, int &dt, 
		int verbose_level);
	void draw_on_circle(char *fname, 
		int xmax_in, int ymax_in, int xmax_out, int ymax_out,
		int f_radius, double radius, 
		int f_labels, int f_embedded, int f_sideways, 
		double tikz_global_scale, double tikz_global_line_width);
	void draw_on_circle_2(mp_graphics &G, int f_labels, 
		int f_radius, double radius);
	void draw(const char *fname, 
		int xmax_in, int ymax_in, int xmax_out, int ymax_out,
		double scale, double line_width, 
		int verbose_level);
	void draw_Levi(const char *fname, 
		int xmax_in, int ymax_in, int xmax_out, int ymax_out,
		int f_partition, int nb_row_parts, int *row_part_first, 
		int nb_col_parts, int *col_part_first, 
		int m, int n, int f_draw_labels, 
		double scale, double line_width, 
		int verbose_level);
	void draw_with_a_given_partition(const char *fname, 
		int xmax_in, int ymax_in, int xmax_out, int ymax_out,
		int *parts, int nb_parts, 
		double scale, double line_width, 
		int verbose_level);
	void draw_partitioned(const char *fname, 
		int xmax_in, int ymax_in, int xmax_out, int ymax_out,
		int f_labels, 
		double scale, double line_width, 
		int verbose_level);
	colored_graph *compute_neighborhood_subgraph(int pt, 
		fancy_set *&vertex_subset, fancy_set *&color_subset, 
		int verbose_level);
	colored_graph *
	compute_neighborhood_subgraph_with_additional_test_function(
		int pt, 
		fancy_set *&vertex_subset, fancy_set *&color_subset, 
		int (*test_function)(colored_graph *CG, int test_point, 
		int pt, void *test_function_data, int verbose_level),
		void *test_function_data, 
		int verbose_level);
	void export_to_magma(const char *fname, int verbose_level);
	void export_to_maple(const char *fname, int verbose_level);
	void export_to_file(const char *fname, int verbose_level);
	void export_to_text(const char *fname, int verbose_level);
	void export_laplacian_to_file(const char *fname, 
		int verbose_level);
	void export_to_file_matlab(const char *fname, int verbose_level);
	void early_test_func_for_clique_search(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		int verbose_level);
	void early_test_func_for_coclique_search(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		int verbose_level);
	void early_test_func_for_path_and_cycle_search(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		int verbose_level);
	int is_cycle(int nb_e, int *edges, int verbose_level);
	void draw_it(const char *fname_base, 
		int xmax_in, int ymax_in, int xmax_out, int ymax_out, 
		double scale, double line_width);
	int rainbow_cliques_nonrecursive(int &nb_backtrack_nodes, int verbose_level);

};

// global functions in colored_graph.C:

void colored_graph_draw(const char *fname, 
	int xmax_in, int ymax_in, int xmax_out, int ymax_out, 
	double scale, double line_width, 
	int verbose_level);
void colored_graph_all_cliques(const char *fname, int f_output_solution_raw, 
	int f_output_fname, const char *output_fname, 
	int f_maxdepth, int maxdepth, 
	int f_restrictions, int *restrictions, 
	int f_tree, int f_decision_nodes_only, const char *fname_tree,  
	int print_interval, 
	int &search_steps, int &decision_steps, int &nb_sol, int &dt, 
	int verbose_level);
void colored_graph_all_cliques_list_of_cases(int *list_of_cases, int nb_cases, 
	int f_output_solution_raw, 
	const char *fname_template, 
	const char *fname_sol, const char *fname_stats, 
	int f_split, int split_r, int split_m, 
	int f_maxdepth, int maxdepth, 
	int f_prefix, const char *prefix, 
	int print_interval, 
	int verbose_level);
void colored_graph_all_cliques_list_of_files(int nb_cases, 
	int *Case_number, const char **Case_fname, 
	int f_output_solution_raw, 
	const char *fname_sol, const char *fname_stats, 
	int f_maxdepth, int maxdepth, 
	int f_prefix, const char *prefix, 
	int print_interval, 
	int verbose_level);
void call_back_clique_found_using_file_output(clique_finder *CF, 
	int verbose_level);
int colored_graph_all_rainbow_cliques_nonrecursive(const char *fname, 
	int &nb_backtrack_nodes, 
	int verbose_level);

// #############################################################################
// graph_layer.C
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
	void null();
	void freeself();
	void init(int nb_nodes, int id_of_first_node, int verbose_level);
	void place(int verbose_level);
	void place_with_grouping(int *group_size, int nb_groups, 
		double x_stretch, int verbose_level);
	void write_memory_object(memory_object *m, int verbose_level);
	void read_memory_object(memory_object *m, int verbose_level);
};

// #############################################################################
// graph_node.C
// #############################################################################

//! part of the data structure layered_graph


class graph_node {
public:
	char *label;
	int id;

	int f_has_data1;
	int data1;

	int f_has_data2;
	int data2;

	int f_has_data3;
	int data3;

	int f_has_vec_data;
	int *vec_data;
	int vec_data_len;

	int f_has_distinguished_element; // refers to vec_data
	int distinguished_element_index;
		
	int layer;
	int neighbor_list_allocated;
	int nb_neighbors;
	int *neighbor_list;
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
	void null();
	void freeself();
	void add_neighbor(int l, int n, int id);
	void add_text(const char *text);
	void add_vec_data(int *v, int len);
	void set_distinguished_element(int idx);
	void add_data1(int data);
	void add_data2(int data);
	void add_data3(int data);
	void write_memory_object(memory_object *m, int verbose_level);
	void read_memory_object(memory_object *m, int verbose_level);
	void allocate_tree_structure(int verbose_level);
	int find_parent(layered_graph *G, int verbose_level);
	void register_child(layered_graph *G, int id_child, int verbose_level);
	void place_x_based_on_tree(layered_graph *G, double left, double right, 
		int verbose_level);
	void depth_first_rank_recursion(layered_graph *G, int &r, 
		int verbose_level);
};

// #############################################################################
// layered_graph.C
// #############################################################################


//! a data structure to store partially ordered sets


class layered_graph {
public:
	int nb_layers;
	int nb_nodes_total;
	int id_of_first_node;
	graph_layer *L;
	char fname_base[1000];
	int data1;

	layered_graph();
	~layered_graph();
	void null();
	void freeself();
	void init(int nb_layers, int *Nb_nodes_layer, 
		const char *fname_base, int verbose_level);
	int nb_nodes();
	double average_word_length();
	void place(int verbose_level);
	void place_with_y_stretch(double y_stretch, int verbose_level);
	void place_with_grouping(int **Group_sizes, int *Nb_groups, 
		double x_stretch, int verbose_level);
	void add_edge(int l1, int n1, int l2, int n2, int verbose_level);
	void add_text(int l, int n, const char *text, int verbose_level);
	void add_data1(int data, int verbose_level);
	void add_node_vec_data(int l, int n, int *v, int len, 
		int verbose_level);
	void set_distinguished_element_index(int l, int n, 
		int index, int verbose_level);
	void add_node_data1(int l, int n, int data, int verbose_level);
	void add_node_data2(int l, int n, int data, int verbose_level);
	void add_node_data3(int l, int n, int data, int verbose_level);
	void draw_with_options(const char *fname, 
		layered_graph_draw_options *O, int verbose_level);
	void coordinates_direct(double x_in, double y_in, 
		int x_max, int y_max, int f_rotated, int &x, int &y);
	void coordinates(int id, int x_max, int y_max, 
		int f_rotated, int &x, int &y);
	void find_node_by_id(int id, int &l, int &n);
	void write_file(char *fname, int verbose_level);
	void read_file(const char *fname, int verbose_level);
	void write_memory_object(memory_object *m, int verbose_level);
	void read_memory_object(memory_object *m, int verbose_level);
	void create_spanning_tree(int f_place_x, int verbose_level);
	void compute_depth_first_ranks(int verbose_level);
	void set_radius_factor_for_all_nodes_at_level(int lvl, 
		double radius_factor, int verbose_level);
};

// #############################################################################
// layered_graph_draw_options.C
// #############################################################################

//! options for drawing an object of type layered_graph

class layered_graph_draw_options {
public:

	int xmax;
	int ymax;
	int x_max;
	int y_max;
	int rad;
	
	int f_circle;
	int f_corners;
	int f_nodes_empty;
	int f_select_layers;
	int nb_layer_select;
	int *layer_select;


	int f_has_draw_begining_callback;
	void (*draw_begining_callback)(layered_graph *LG, mp_graphics *G, 
		int x_max, int y_max, int f_rotated, int dx, int dy);
	int f_has_draw_ending_callback;
	void (*draw_ending_callback)(layered_graph *LG, mp_graphics *G, 
		int x_max, int y_max, int f_rotated, int dx, int dy);
	int f_has_draw_vertex_callback;
	void (*draw_vertex_callback)(layered_graph *LG, mp_graphics *G, 
		int layer, int node, int x, int y, int dx, int dy);
	
	int f_show_level_info;
	int f_embedded;
	int f_sideways;
	int f_label_edges;
	int f_rotated;
	

	double global_scale;
	double global_line_width;

	layered_graph_draw_options();
	~layered_graph_draw_options();
	void init(
		int xmax, int ymax, int x_max, int y_max, int rad, 
		int f_circle, int f_corners, int f_nodes_empty, 
		int f_select_layers, int nb_layer_select, int *layer_select, 
		int f_has_draw_begining_callback, 
		void (*draw_begining_callback)(layered_graph *LG, 
			mp_graphics *G, 
			int x_max, int y_max, int f_rotated, 
			int dx, int dy), 
		int f_has_draw_ending_callback, 
		void (*draw_ending_callback)(layered_graph *LG, 
			mp_graphics *G, 
			int x_max, int y_max, int f_rotated, 
			int dx, int dy), 
		int f_has_draw_vertex_callback, 
		void (*draw_vertex_callback)(layered_graph *LG, 
			mp_graphics *G, 
			int layer, int node, 
			int x, int y, int dx, int dy), 
		int f_show_level_info, 
		int f_embedded, int f_sideways, 
		int f_label_edges, 
		int f_rotated, 
		double global_scale, double global_line_width);
};

// #############################################################################
// rainbow_cliques.C
// #############################################################################



//! to search for rainbow cliques in graphs




class rainbow_cliques {
public:

	rainbow_cliques();
	~rainbow_cliques();
	void null();
	void freeself();

	ofstream *fp_sol;
	int f_output_solution_raw;
	
	colored_graph *graph;
	clique_finder *CF;
	int *f_color_satisfied;
	int *color_chosen_at_depth;
	int *color_frequency;
	int target_depth;

	// added November 5, 2014:
	int f_has_additional_test_function;
	void (*call_back_additional_test_function)(rainbow_cliques *R, 
		void *user_data, 
		int current_clique_size, int *current_clique, 
		int nb_pts, int &reduced_nb_pts, 
		int *pt_list, int *pt_list_inv, 
		int verbose_level);
	void *user_data;


	void search(colored_graph *graph, ofstream *fp_sol, 
		int f_output_solution_raw, 
		int f_maxdepth, int maxdepth, 
		int f_restrictions, int *restrictions, 
		int f_tree, int f_decision_nodes_only, 
		const char *fname_tree,  
		int print_interval, 
		int &search_steps, 
		int &decision_steps, int &nb_sol, int &dt, 
		int verbose_level);
	void search_with_additional_test_function(colored_graph *graph, 
		ofstream *fp_sol, int f_output_solution_raw, 
		int f_maxdepth, int maxdepth, 
		int f_restrictions, int *restrictions, 
		int f_tree, int f_decision_nodes_only, 
		const char *fname_tree,  
		int print_interval, 
		int f_has_additional_test_function,
		void (*call_back_additional_test_function)(
			rainbow_cliques *R, 
			void *user_data, 
			int current_clique_size, int *current_clique, 
			int nb_pts, int &reduced_nb_pts, 
			int *pt_list, int *pt_list_inv, 
			int verbose_level), 
		int f_has_print_current_choice_function,
		void (*call_back_print_current_choice)(clique_finder *CF, 
			int depth, void *user_data, int verbose_level), 
		void *user_data, 
		int &search_steps, int &decision_steps, 
		int &nb_sol, int &dt, 
		int verbose_level);
	int find_candidates(
		int current_clique_size, int *current_clique, 
		int nb_pts, int &reduced_nb_pts, 
		int *pt_list, int *pt_list_inv, 
		int *candidates, int verbose_level);
	void clique_found(int *current_clique, int verbose_level);
	void clique_found_record_in_original_labels(int *current_clique, 
		int verbose_level);

};

void call_back_colored_graph_clique_found(clique_finder *CF, 
	int verbose_level);
void call_back_colored_graph_add_point(clique_finder *CF, 
	int current_clique_size, int *current_clique, 
	int pt, int verbose_level);
void call_back_colored_graph_delete_point(clique_finder *CF, 
	int current_clique_size, int *current_clique, 
	int pt, int verbose_level);
int call_back_colored_graph_find_candidates(clique_finder *CF, 
	int current_clique_size, int *current_clique, 
	int nb_pts, int &reduced_nb_pts, 
	int *pt_list, int *pt_list_inv, 
	int *candidates, int verbose_level);


}
}

