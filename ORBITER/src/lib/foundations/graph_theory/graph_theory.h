// graph_theory.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005

// #############################################################################
// clique_finder.C
// #############################################################################

//! A class that can be used to find cliques in graphs



class clique_finder {
public:
	BYTE label[1000];
	INT n; // number of points
	
	INT print_interval;
	
	INT f_write_tree;
	INT f_decision_nodes_only;
	BYTE fname_tree[1000];
	ofstream *fp_tree;

	
	INT f_maxdepth;
	INT maxdepth;
	
	INT *point_labels;
	INT *point_is_suspicous;
	
	INT target_depth;
	INT verbose_level;
	

	INT f_has_bitmatrix;
	INT bitmatrix_m;
	INT bitmatrix_n;
	INT bitmatrix_N;
	UBYTE *bitmatrix_adjacency;

	INT f_has_adj_list;
	INT *adj_list_coded;
	INT f_has_bitvector;
	UBYTE *bitvector_adjacency;

	INT f_has_row_by_row_adjacency_matrix;
	BYTE **row_by_row_adjacency_matrix; // [n][n]


	INT *pt_list;
	INT *pt_list_inv;
	INT *nb_points;
	INT *candidates; // [max_depth * n]
	INT *nb_candidates; // [max_depth]
	INT *current_choice; // [max_depth]
	INT *level_counter; // [max_depth] (added Nov 8, 2014)
	INT *f_level_mod; // [max_depth] (added Nov 8, 2014)
	INT *level_r; // [max_depth] (added Nov 8, 2014)
	INT *level_m; // [max_depth] (added Nov 8, 2014)

	INT *current_clique; // [max_depth]

	UINT counter; // number of backtrack nodes
	UINT decision_step_counter;
		// number of backtrack nodes that are decision nodes

	// solution storage:
	INT f_store_solutions;
	deque<vector<int> > solutions;
	INT nb_sol;


	// callbacks:
	void (*call_back_clique_found)(clique_finder *CF, INT verbose_level);
	
	// added May 26, 2009:
	void (*call_back_add_point)(clique_finder *CF, 
		INT current_clique_size, INT *current_clique, 
		INT pt, INT verbose_level);
	void (*call_back_delete_point)(clique_finder *CF, 
		INT current_clique_size, INT *current_clique, 
		INT pt, INT verbose_level);
	INT (*call_back_find_candidates)(clique_finder *CF, 
		INT current_clique_size, INT *current_clique, 
		INT nb_pts, INT &reduced_nb_pts, 
		INT *pt_list, INT *pt_list_inv, 
		INT *candidates, INT verbose_level);
		// Jan 2, 2012: added reduced_nb_pts and pt_list_inv

	INT (*call_back_is_adjacent)(clique_finder *CF, 
		INT pt1, INT pt2, INT verbose_level);
	// added Oct 2011:
	void (*call_back_after_reduction)(clique_finder *CF, 
		INT depth, INT nb_points, INT verbose_level);

	// added Nov 2014:
	INT f_has_print_current_choice_function;
	void (*call_back_print_current_choice)(clique_finder *CF, 
		INT depth, void *user_data, INT verbose_level);
	void *print_current_choice_data;
	
	void *call_back_clique_found_data1;
	void *call_back_clique_found_data2;
	
	
	void open_tree_file(const BYTE *fname_base, 
		INT f_decision_nodes_only);
	void close_tree_file();
	void init(const BYTE *label, INT n, 
		INT target_depth, 
		INT f_has_adj_list, INT *adj_list_coded, 
		INT f_has_bitvector, UBYTE *bitvector_adjacency, 
		INT print_interval, 
		INT f_maxdepth, INT maxdepth, 
		INT f_store_solutions, 
		INT verbose_level);
	void delinearize_adjacency_list(INT verbose_level);
	void allocate_bitmatrix(INT verbose_level);
	void init_restrictions(INT *restrictions, INT verbose_level);
	clique_finder();
	~clique_finder();
	void null();
	void free();
	void init_point_labels(INT *pt_labels);
	void init_suspicous_points(INT nb, INT *point_list);
	void print_suspicous_points();
	void print_set(INT size, INT *set);
	void print_suspicous_point_subset(INT size, INT *set);
	void log_position_and_choice(INT depth, INT counter_save, 
		INT counter);
	void log_position(INT depth, INT counter_save, INT counter);
	void log_choice(INT depth);
	void swap_point(INT idx1, INT idx2);
	void degree_of_point_statistic(INT depth, INT nb_points, 
		INT verbose_level);
	INT degree_of_point(INT depth, INT i, INT nb_points);
	//INT degree_of_point_verbose(INT i, INT nb_points);
	INT is_suspicous(INT i);
	INT point_label(INT i);
	INT is_adjacent(INT depth, INT i, INT j);
	INT is_viable(INT depth, INT pt);
	void write_entry_to_tree_file(INT depth, INT verbose_level);
	void m_iji(INT i, INT j, INT a);
	INT s_ij(INT i, INT j);
	void backtrack_search(INT depth, INT verbose_level);
	INT solve_decision_problem(INT depth, INT verbose_level);
		// returns TRUE if we found a solution
	void get_solutions(INT *&Sol, INT &nb_solutions, INT &clique_sz, 
		INT verbose_level);
	void backtrack_search_not_recursive(INT verbose_level);
};

void all_cliques_of_given_size(INT *Adj, INT nb_pts, INT clique_sz, 
	INT *&Sol, INT &nb_sol, INT verbose_level);



// #############################################################################
// colored_graph.C
// #############################################################################


//! a graph with a vertex coloring


class colored_graph {
public:

	BYTE fname_base[1000];
	
	INT nb_points;
	INT nb_colors;
	
	INT bitvector_length;
	INT L;
	
	INT *points; // [nb_points]
	INT *point_color; // [nb_points]
	

	INT user_data_size;
	INT *user_data; // [user_data_size]

	INT f_ownership_of_bitvec;
	UBYTE *bitvector_adjacency;

	INT f_has_list_of_edges;
	INT nb_edges;
	INT *list_of_edges;

	colored_graph();
	~colored_graph();
	void null();
	void freeself();
	void compute_edges(INT verbose_level);
	INT is_adjacent(INT i, INT j);
	void set_adjacency(INT i, INT j, INT a);
	void partition_by_color_classes(
		INT *&partition, INT *&partition_first, 
		INT &partition_length, 
		INT verbose_level);
	colored_graph *sort_by_color_classes(INT verbose_level);
	void print();
	void print_points_and_colors();
	void print_adjacency_list();
	void init(INT nb_points, INT nb_colors, 
		INT *colors, UBYTE *bitvec, INT f_ownership_of_bitvec, 
		INT verbose_level);
	void init_with_point_labels(INT nb_points, INT nb_colors, 
		INT *colors, UBYTE *bitvec, INT f_ownership_of_bitvec, 
		INT *point_labels, 
		INT verbose_level);
	void init_no_colors(INT nb_points, UBYTE *bitvec, 
		INT f_ownership_of_bitvec, 
		INT verbose_level);
	void init_adjacency(INT nb_points, INT nb_colors, 
		INT *colors, INT *Adj, INT verbose_level);
	void init_adjacency_upper_triangle(INT nb_points, INT nb_colors, 
		INT *colors, INT *Adj, INT verbose_level);
	void init_adjacency_no_colors(INT nb_points, INT *Adj, 
		INT verbose_level);
	void init_user_data(INT *data, INT data_size, INT verbose_level);
	void save(const BYTE *fname, INT verbose_level);
	void load(const BYTE *fname, INT verbose_level);
	void all_cliques_of_size_k_ignore_colors(INT target_depth, 
		INT &nb_sol, INT &decision_step_counter, INT verbose_level);
	void all_cliques_of_size_k_ignore_colors_and_write_solutions_to_file(
		INT target_depth, 
		const BYTE *fname, 
		INT f_restrictions, INT *restrictions, 
		INT &nb_sol, INT &decision_step_counter, 
		INT verbose_level);
	void all_rainbow_cliques(ofstream *fp, INT f_output_solution_raw, 
		INT f_maxdepth, INT maxdepth, 
		INT f_restrictions, INT *restrictions, 
		INT f_tree, INT f_decision_nodes_only, const BYTE *fname_tree,  
		INT print_interval, 
		INT &search_steps, INT &decision_steps, INT &nb_sol, INT &dt, 
		INT verbose_level);
	void all_rainbow_cliques_with_additional_test_function(ofstream *fp, 
		INT f_output_solution_raw, 
		INT f_maxdepth, INT maxdepth, 
		INT f_restrictions, INT *restrictions, 
		INT f_tree, INT f_decision_nodes_only, const BYTE *fname_tree,  
		INT print_interval, 
		INT f_has_additional_test_function,
		void (*call_back_additional_test_function)(rainbow_cliques *R, 
			void *user_data, 
			INT current_clique_size, INT *current_clique, 
			INT nb_pts, INT &reduced_nb_pts, 
			INT *pt_list, INT *pt_list_inv, 
			INT verbose_level), 
		INT f_has_print_current_choice_function,
		void (*call_back_print_current_choice)(clique_finder *CF, 
			INT depth, void *user_data, INT verbose_level), 
		void *user_data, 
		INT &search_steps, INT &decision_steps, INT &nb_sol, INT &dt, 
		INT verbose_level);
	void draw_on_circle(char *fname, 
		INT xmax_in, INT ymax_in, INT xmax_out, INT ymax_out,
		INT f_radius, double radius, 
		INT f_labels, INT f_embedded, INT f_sideways, 
		double tikz_global_scale, double tikz_global_line_width);
	void draw_on_circle_2(mp_graphics &G, INT f_labels, 
		INT f_radius, double radius);
	void draw(const BYTE *fname, 
		INT xmax_in, INT ymax_in, INT xmax_out, INT ymax_out,
		double scale, double line_width, 
		INT verbose_level);
	void draw_Levi(const BYTE *fname, 
		INT xmax_in, INT ymax_in, INT xmax_out, INT ymax_out,
		INT f_partition, INT nb_row_parts, INT *row_part_first, 
		INT nb_col_parts, INT *col_part_first, 
		INT m, INT n, INT f_draw_labels, 
		double scale, double line_width, 
		INT verbose_level);
	void draw_with_a_given_partition(const BYTE *fname, 
		INT xmax_in, INT ymax_in, INT xmax_out, INT ymax_out,
		INT *parts, INT nb_parts, 
		double scale, double line_width, 
		INT verbose_level);
	void draw_partitioned(const BYTE *fname, 
		INT xmax_in, INT ymax_in, INT xmax_out, INT ymax_out,
		INT f_labels, 
		double scale, double line_width, 
		INT verbose_level);
	colored_graph *compute_neighborhood_subgraph(INT pt, 
		fancy_set *&vertex_subset, fancy_set *&color_subset, 
		INT verbose_level);
	colored_graph *
	compute_neighborhood_subgraph_with_additional_test_function(
		INT pt, 
		fancy_set *&vertex_subset, fancy_set *&color_subset, 
		INT (*test_function)(colored_graph *CG, INT test_point, 
		INT pt, void *test_function_data, INT verbose_level),
		void *test_function_data, 
		INT verbose_level);
	void export_to_magma(const BYTE *fname, INT verbose_level);
	void export_to_maple(const BYTE *fname, INT verbose_level);
	void export_to_file(const BYTE *fname, INT verbose_level);
	void export_to_text(const BYTE *fname, INT verbose_level);
	void export_laplacian_to_file(const BYTE *fname, 
		INT verbose_level);
	void export_to_file_matlab(const BYTE *fname, INT verbose_level);
	void early_test_func_for_clique_search(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
	void early_test_func_for_coclique_search(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
	void early_test_func_for_path_and_cycle_search(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
	INT is_cycle(INT nb_e, INT *edges, INT verbose_level);
	void draw_it(const BYTE *fname_base, 
		INT xmax_in, INT ymax_in, INT xmax_out, INT ymax_out, 
		double scale, double line_width);
	INT rainbow_cliques_nonrecursive(INT &nb_backtrack_nodes, INT verbose_level);

};

// global functions in colored_graph.C:

void colored_graph_draw(const BYTE *fname, 
	INT xmax_in, INT ymax_in, INT xmax_out, INT ymax_out, 
	double scale, double line_width, 
	INT verbose_level);
void colored_graph_all_cliques(const BYTE *fname, INT f_output_solution_raw, 
	INT f_output_fname, const BYTE *output_fname, 
	INT f_maxdepth, INT maxdepth, 
	INT f_restrictions, INT *restrictions, 
	INT f_tree, INT f_decision_nodes_only, const BYTE *fname_tree,  
	INT print_interval, 
	INT &search_steps, INT &decision_steps, INT &nb_sol, INT &dt, 
	INT verbose_level);
void colored_graph_all_cliques_list_of_cases(INT *list_of_cases, INT nb_cases, 
	INT f_output_solution_raw, 
	const BYTE *fname_template, 
	const BYTE *fname_sol, const BYTE *fname_stats, 
	INT f_split, INT split_r, INT split_m, 
	INT f_maxdepth, INT maxdepth, 
	INT f_prefix, const BYTE *prefix, 
	INT print_interval, 
	INT verbose_level);
void colored_graph_all_cliques_list_of_files(INT nb_cases, 
	INT *Case_number, const BYTE **Case_fname, 
	INT f_output_solution_raw, 
	const BYTE *fname_sol, const BYTE *fname_stats, 
	INT f_maxdepth, INT maxdepth, 
	INT f_prefix, const BYTE *prefix, 
	INT print_interval, 
	INT verbose_level);
void call_back_clique_found_using_file_output(clique_finder *CF, 
	INT verbose_level);
INT colored_graph_all_rainbow_cliques_nonrecursive(const BYTE *fname, 
	INT &nb_backtrack_nodes, 
	INT verbose_level);

// #############################################################################
// graph_layer.C
// #############################################################################


//! part of the data structure layered_graph



class graph_layer {
public:
	INT id_of_first_node;
	INT nb_nodes;
	graph_node *Nodes;
	double y_coordinate;

	graph_layer();
	~graph_layer();
	void null();
	void freeself();
	void init(INT nb_nodes, INT id_of_first_node, INT verbose_level);
	void place(INT verbose_level);
	void place_with_grouping(INT *group_size, INT nb_groups, 
		double x_stretch, INT verbose_level);
	void write_memory_object(memory_object *m, INT verbose_level);
	void read_memory_object(memory_object *m, INT verbose_level);
};

// #############################################################################
// graph_node.C
// #############################################################################

//! part of the data structure layered_graph


class graph_node {
public:
	BYTE *label;
	INT id;

	INT f_has_data1;
	INT data1;

	INT f_has_data2;
	INT data2;

	INT f_has_data3;
	INT data3;

	INT f_has_vec_data;
	INT *vec_data;
	INT vec_data_len;

	INT f_has_distinguished_element; // refers to vec_data
	INT distinguished_element_index;
		
	INT layer;
	INT neighbor_list_allocated;
	INT nb_neighbors;
	INT *neighbor_list;
	double x_coordinate;
	
	// added June 28, 2016:
	INT nb_children;
	INT nb_children_allocated;
	INT *child_id; // [nb_children]
	INT weight_of_subtree;
	double width;
	INT depth_first_node_rank;

	// added May 25, 2017
	double radius_factor;

	graph_node();
	~graph_node();
	void null();
	void freeself();
	void add_neighbor(INT l, INT n, INT id);
	void add_text(const BYTE *text);
	void add_vec_data(INT *v, INT len);
	void set_distinguished_element(INT idx);
	void add_data1(INT data);
	void add_data2(INT data);
	void add_data3(INT data);
	void write_memory_object(memory_object *m, INT verbose_level);
	void read_memory_object(memory_object *m, INT verbose_level);
	void allocate_tree_structure(INT verbose_level);
	INT find_parent(layered_graph *G, INT verbose_level);
	void register_child(layered_graph *G, INT id_child, INT verbose_level);
	void place_x_based_on_tree(layered_graph *G, double left, double right, 
		INT verbose_level);
	void depth_first_rank_recursion(layered_graph *G, INT &r, 
		INT verbose_level);
};

// #############################################################################
// layered_graph.C
// #############################################################################


//! a data structure to store partially ordered sets


class layered_graph {
public:
	INT nb_layers;
	INT nb_nodes_total;
	INT id_of_first_node;
	graph_layer *L;
	BYTE fname_base[1000];
	INT data1;

	layered_graph();
	~layered_graph();
	void null();
	void freeself();
	void init(INT nb_layers, INT *Nb_nodes_layer, 
		const BYTE *fname_base, INT verbose_level);
	void place(INT verbose_level);
	void place_with_y_stretch(double y_stretch, INT verbose_level);
	void place_with_grouping(INT **Group_sizes, INT *Nb_groups, 
		double x_stretch, INT verbose_level);
	void add_edge(INT l1, INT n1, INT l2, INT n2, INT verbose_level);
	void add_text(INT l, INT n, const BYTE *text, INT verbose_level);
	void add_data1(INT data, INT verbose_level);
	void add_node_vec_data(INT l, INT n, INT *v, INT len, 
		INT verbose_level);
	void set_distinguished_element_index(INT l, INT n, 
		INT index, INT verbose_level);
	void add_node_data1(INT l, INT n, INT data, INT verbose_level);
	void add_node_data2(INT l, INT n, INT data, INT verbose_level);
	void add_node_data3(INT l, INT n, INT data, INT verbose_level);
	void draw_with_options(const char *fname, 
		layered_graph_draw_options *O, INT verbose_level);
	void coordinates_direct(double x_in, double y_in, 
		INT x_max, INT y_max, INT f_rotated, INT &x, INT &y);
	void coordinates(INT id, INT x_max, INT y_max, 
		INT f_rotated, INT &x, INT &y);
	void find_node_by_id(INT id, INT &l, INT &n);
	void write_file(BYTE *fname, INT verbose_level);
	void read_file(const BYTE *fname, INT verbose_level);
	void write_memory_object(memory_object *m, INT verbose_level);
	void read_memory_object(memory_object *m, INT verbose_level);
	void create_spanning_tree(INT f_place_x, INT verbose_level);
	void compute_depth_first_ranks(INT verbose_level);
	void set_radius_factor_for_all_nodes_at_level(INT lvl, 
		double radius_factor, INT verbose_level);
};

// #############################################################################
// layered_graph_draw_options.C
// #############################################################################

//! options for drawing an object of type layered_graph

class layered_graph_draw_options {
public:

	INT xmax;
	INT ymax;
	INT x_max;
	INT y_max;
	INT rad;
	
	INT f_circle;
	INT f_corners;
	INT f_nodes_empty;
	INT f_select_layers;
	INT nb_layer_select;
	INT *layer_select;


	INT f_has_draw_begining_callback;
	void (*draw_begining_callback)(layered_graph *LG, mp_graphics *G, 
		INT x_max, INT y_max, INT f_rotated, INT dx, INT dy);
	INT f_has_draw_ending_callback;
	void (*draw_ending_callback)(layered_graph *LG, mp_graphics *G, 
		INT x_max, INT y_max, INT f_rotated, INT dx, INT dy);
	INT f_has_draw_vertex_callback;
	void (*draw_vertex_callback)(layered_graph *LG, mp_graphics *G, 
		INT layer, INT node, INT x, INT y, INT dx, INT dy);
	
	INT f_show_level_info;
	INT f_embedded;
	INT f_sideways;
	INT f_label_edges;
	INT f_rotated;
	
	double global_scale;
	double global_line_width;

	layered_graph_draw_options();
	~layered_graph_draw_options();
	void init(
		INT xmax, INT ymax, INT x_max, INT y_max, INT rad, 
		INT f_circle, INT f_corners, INT f_nodes_empty, 
		INT f_select_layers, INT nb_layer_select, INT *layer_select, 
		INT f_has_draw_begining_callback, 
		void (*draw_begining_callback)(layered_graph *LG, 
			mp_graphics *G, 
			INT x_max, INT y_max, INT f_rotated, 
			INT dx, INT dy), 
		INT f_has_draw_ending_callback, 
		void (*draw_ending_callback)(layered_graph *LG, 
			mp_graphics *G, 
			INT x_max, INT y_max, INT f_rotated, 
			INT dx, INT dy), 
		INT f_has_draw_vertex_callback, 
		void (*draw_vertex_callback)(layered_graph *LG, 
			mp_graphics *G, 
			INT layer, INT node, 
			INT x, INT y, INT dx, INT dy), 
		INT f_show_level_info, 
		INT f_embedded, INT f_sideways, 
		INT f_label_edges, 
		INT f_rotated, 
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
	INT f_output_solution_raw;
	
	colored_graph *graph;
	clique_finder *CF;
	INT *f_color_satisfied;
	INT *color_chosen_at_depth;
	INT *color_frequency;
	INT target_depth;

	// added November 5, 2014:
	INT f_has_additional_test_function;
	void (*call_back_additional_test_function)(rainbow_cliques *R, 
		void *user_data, 
		INT current_clique_size, INT *current_clique, 
		INT nb_pts, INT &reduced_nb_pts, 
		INT *pt_list, INT *pt_list_inv, 
		INT verbose_level);
	void *user_data;


	void search(colored_graph *graph, ofstream *fp_sol, 
		INT f_output_solution_raw, 
		INT f_maxdepth, INT maxdepth, 
		INT f_restrictions, INT *restrictions, 
		INT f_tree, INT f_decision_nodes_only, 
		const BYTE *fname_tree,  
		INT print_interval, 
		INT &search_steps, 
		INT &decision_steps, INT &nb_sol, INT &dt, 
		INT verbose_level);
	void search_with_additional_test_function(colored_graph *graph, 
		ofstream *fp_sol, INT f_output_solution_raw, 
		INT f_maxdepth, INT maxdepth, 
		INT f_restrictions, INT *restrictions, 
		INT f_tree, INT f_decision_nodes_only, 
		const BYTE *fname_tree,  
		INT print_interval, 
		INT f_has_additional_test_function,
		void (*call_back_additional_test_function)(
			rainbow_cliques *R, 
			void *user_data, 
			INT current_clique_size, INT *current_clique, 
			INT nb_pts, INT &reduced_nb_pts, 
			INT *pt_list, INT *pt_list_inv, 
			INT verbose_level), 
		INT f_has_print_current_choice_function,
		void (*call_back_print_current_choice)(clique_finder *CF, 
			INT depth, void *user_data, INT verbose_level), 
		void *user_data, 
		INT &search_steps, INT &decision_steps, 
		INT &nb_sol, INT &dt, 
		INT verbose_level);
	INT find_candidates(
		INT current_clique_size, INT *current_clique, 
		INT nb_pts, INT &reduced_nb_pts, 
		INT *pt_list, INT *pt_list_inv, 
		INT *candidates, INT verbose_level);
	void clique_found(INT *current_clique, INT verbose_level);
	void clique_found_record_in_original_labels(INT *current_clique, 
		INT verbose_level);

};

void call_back_colored_graph_clique_found(clique_finder *CF, 
	INT verbose_level);
void call_back_colored_graph_add_point(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT pt, INT verbose_level);
void call_back_colored_graph_delete_point(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT pt, INT verbose_level);
INT call_back_colored_graph_find_candidates(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT nb_pts, INT &reduced_nb_pts, 
	INT *pt_list, INT *pt_list_inv, 
	INT *candidates, INT verbose_level);




