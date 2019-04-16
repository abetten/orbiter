/*
 * snakes_and_ladders.h
 *
 *  Created on: Aug 9, 2018
 *      Author: Anton Betten
 *
 * started:  September 20, 2007
 * pulled out of snakesandladders.h: Aug 9, 2018
 */

namespace orbiter {
namespace classification {



// #############################################################################
// extension.C:
// #############################################################################

#define NB_EXTENSION_TYPES 5

#define EXTENSION_TYPE_UNPROCESSED  0
#define EXTENSION_TYPE_EXTENSION 1
#define EXTENSION_TYPE_FUSION 2
#define EXTENSION_TYPE_PROCESSING 3
#define EXTENSION_TYPE_NOT_CANONICAL 4


//! represents a flag in the poset classification algorithm; related to poset_orbit_node



class extension {

public:
	int pt;
	int orbit_len;
	int type;
		// EXTENSION_TYPE_UNPROCESSED = unprocessed
		// EXTENSION_TYPE_EXTENSION = extension node
		// EXTENSION_TYPE_FUSION = fusion node
		// EXTENSION_TYPE_PROCESSING = currently processing
		// EXTENSION_TYPE_NOT_CANONICAL = no extension formed 
		// because it is not canonical
	int data;
		// if EXTENSION_TYPE_EXTENSION: a handle to the next 
		//  poset_orbit_node
		// if EXTENSION_TYPE_FUSION: a handle to a fusion element
	int data1;
		// if EXTENSION_TYPE_FUSION: node to which we are fusing
	int data2;
		// if EXTENSION_TYPE_FUSION: extension within that 
		// node to which we are fusing

	extension();
	~extension();
};


void print_extension_type(std::ostream &ost, int t);


// #############################################################################
// orbit_based_testing.cpp
// #############################################################################


#define MAX_CALLBACK 100


//! maintains a list of test functions which define a G-invariant poset

class orbit_based_testing {

public:

	poset_classification *PC;
	int max_depth;
	int *local_S; // [max_depth]
	int nb_callback;
	int (*callback_testing[MAX_CALLBACK])(orbit_based_testing *Obt,
			int *S, int len, void *data, int verbose_level);
	void *callback_data[MAX_CALLBACK];

	int nb_callback_no_group;
	void (*callback_testing_no_group[MAX_CALLBACK])(
			int *S, int len,
			int *candidates, int nb_candidates,
			int *good_candidates, int &nb_good_candidates,
			void *data, int verbose_level);
	void *callback_data_no_group[MAX_CALLBACK];

	orbit_based_testing();
	~orbit_based_testing();
	void null();
	void freeself();
	void init(
			poset_classification *PC,
			int max_depth,
			int verbose_level);
	void add_callback(
			int (*func)(orbit_based_testing *Obt,
					int *S, int len, void *data, int verbose_level),
			void *data,
			int verbose_level);
	void add_callback_no_group(
			void (*func)(int *S, int len,
					int *candidates, int nb_candidates,
					int *good_candidates, int &nb_good_candidates,
					void *data, int verbose_level),
			void *data,
			int verbose_level);
	void early_test_func(
		int *S, int len,
		int *candidates, int nb_candidates,
		int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void early_test_func_by_using_group(
		int *S, int len,
		int *candidates, int nb_candidates,
		int *good_candidates, int &nb_good_candidates,
		int verbose_level);
};


// #############################################################################
// poset.C:
// #############################################################################

//! a poset on which a group acts


class poset {
public:
	poset_description *description;

	int f_subset_lattice;
	int n;

	int f_subspace_lattice;
	vector_space *VS;
	//int vector_space_dimension;

	action *A; // the action in which the group is given
	action *A2; // the action in which we do the search

	strong_generators *Strong_gens;
	longinteger_object go;

	int f_has_orbit_based_testing;
	orbit_based_testing *Orbit_based_testing;

	poset();
	~poset();
	void null();
	void freeself();
	void init_subset_lattice(action *A, action *A2,
			strong_generators *Strong_gens,
			int verbose_level);
	void init_subspace_lattice(action *A, action *A2,
			strong_generators *Strong_gens,
			vector_space *VS,
			int verbose_level);
	void init(poset_description *description,
		action *A, // the action in which the group is given
		action *A2, // the action in which we do the search
		strong_generators *Strong_gens,
		int verbose_level);
	void add_independence_condition(
			int independence_value,
			int verbose_level);
	void add_testing(
			int (*func)(orbit_based_testing *Obt,
					int *S, int len, void *data, int verbose_level),
			void *data,
			int verbose_level);
	void add_testing_without_group(
			void (*func)(int *S, int len,
					int *candidates, int nb_candidates,
					int *good_candidates, int &nb_good_candidates,
					void *data, int verbose_level),
			void *data,
			int verbose_level);
	void print();
	void early_test_func(
		int *S, int len,
		int *candidates, int nb_candidates,
		int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void unrank_point(int *v, int rk);
	int rank_point(int *v);
	void orbits_on_k_sets(
		int k, int *&orbit_reps, int &nb_orbits, int verbose_level);
	poset_classification *orbits_on_k_sets_compute(
		int k, int verbose_level);
};

int callback_test_independence_condition(orbit_based_testing *Obt,
					int *S, int len, void *data, int verbose_level);



// #############################################################################
// poset_description.C:
// #############################################################################

//! description of a poset from the command line


class poset_description {
public:
	int f_subset_lattice;

	int f_subspace_lattice;
	int dimension;
	int q;


	char label[1000];

	int f_independence_condition;
	int independence_condition_value;


	poset_description();
	~poset_description();
	void null();
	void freeself();
	void read_arguments_from_string(
			const char *str, int verbose_level);
	int read_arguments(int argc, const char **argv,
		int verbose_level);
};





// #############################################################################
// poset_classification.C
// #############################################################################

//! the poset classification algorithm (aka Snakes and Ladders)


class poset_classification {

public:
	int t0;

	char problem_label[1000];
	
	poset *Poset;


	schreier_vector_handler *Schreier_vector_handler;


	// used as storage for the current set:
	int *S; // [sz]
	
	int sz;
		// the target depth
		
	
	int *Elt_memory; // [6 * elt_size_in_int]
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *Elt4;
	int *Elt5;
	int *Elt6; // for poset_orbit_node::read_memory_object / write_memory_object
	
	int *tmp_set_apply_fusion;
		// used in poset_orbit_upstep.C poset_orbit_node::apply_isomorphism


	// for vector space actions, allocated in init:
	int *tmp_find_node_for_subspace_by_rank1;
		// [vector_space_dimension] used in poset_classification_trace.C: 
		// find_node_for_subspace_by_rank
	int *tmp_find_node_for_subspace_by_rank2;
		// [sz * vector_space_dimension] used in poset_classification_trace.C: 
		// find_node_for_subspace_by_rank


	int f_print_function;
	void (*print_function)(std::ostream &ost, int len, int *S, void *data);
	void *print_function_data;
	
	int nb_times_trace;
	int nb_times_trace_was_saved;
	
	// data for recognize:
	vector_ge *transporter; // [sz + 1]
	int **set; // [sz + 1][sz]

	
	
	// the following is maintained
	// by init_poset_orbit_node / exit_poset_orbit_node:
	int nb_poset_orbit_nodes_used;
	int nb_poset_orbit_nodes_allocated;
	int poset_orbit_nodes_increment;
	int poset_orbit_nodes_increment_last;
	
	poset_orbit_node *root;
	
	int *first_poset_orbit_node_at_level;
	int *set0; // [sz + 1] temporary storage
	int *set1; // [sz + 1] temporary storage
	int *set3; // [sz + 1] temporary storage
	
	int *nb_extension_nodes_at_level_total;
	int *nb_extension_nodes_at_level;
	int *nb_fusion_nodes_at_level;
	int *nb_unprocessed_nodes_at_level;


	// command line options:


	int depth; // search depth
	int f_w; // write output in level files (only last level)
	int f_W; // write output in level files (each level)
	int f_write_data_files;
	int f_T; // draw tree file (each level)
	int f_t; // draw tree file (only last level)
	int f_Log; // log nodes (each level)
	int f_log; // log nodes (only last level)
	int f_print_only;
	int f_find_group_order;
	int find_group_order;
	
	int f_has_tools_path;
	const char *tools_path;

	int verbose_level;
	int verbose_level_group_theory;
	
	int xmax, ymax, radius;
	
	int f_recover;
	const char *recover_fname;

	int f_extend;
	int extend_from, extend_to;
	int extend_r, extend_m;
	char extend_fname[1000];

	int f_lex;
	int f_max_depth;
	int max_depth;

	char fname_base[1000]; // = path + prefix
	char prefix[1000]; // = fname_base without prefix
	char path[1000];


	int f_starter;
	int starter_size;
	int *starter;
	strong_generators *starter_strong_gens;
	int *starter_live_points;
	int starter_nb_live_points;
	void *starter_canonize_data;
	int (*starter_canonize)(int *Set, int len, int *Elt, 
		void *data, int verbose_level);
	int *starter_canonize_Elt;
	
	int f_has_invariant_subset_for_root_node;
	int *invariant_subset_for_root_node;
	int invariant_subset_for_root_node_size;
	

	int f_do_group_extension_in_upstep;
		// is TRUE by default

	int f_allowed_to_show_group_elements;
	int downstep_orbits_print_max_orbits;
	int downstep_orbits_print_max_points_per_orbit;
	


	int nb_times_image_of_called0;
	int nb_times_mult_called0;
	int nb_times_invert_called0;
	int nb_times_retrieve_called0;
	int nb_times_store_called0;
	
	double progress_last_time;
	double progress_epsilon;

	int f_export_schreier_trees;
	int f_draw_schreier_trees;
		char schreier_tree_prefix[1000];
		int schreier_tree_xmax; // = 1000000;
		int schreier_tree_ymax; // =  500000;
		int schreier_tree_f_circletext; // = TRUE;
		int schreier_tree_rad; // = 25000;
		int schreier_tree_f_embedded;
		int schreier_tree_f_sideways;
		double schreier_tree_scale;
		double schreier_tree_line_width;



	// poset_classification.C:
	int nb_orbits_at_level(int level);
	int nb_flag_orbits_up_at_level(int level);
	poset_orbit_node *get_node_ij(int level, int node);
	int poset_structure_is_contained(int *set1, int sz1, 
		int *set2, int sz2, int verbose_level);
	void print_progress_by_extension(int size, int cur, 
		int prev, int cur_ex, int nb_ext_cur, int nb_fuse_cur);
	void print_progress(int size, int cur, int prev, 
		int nb_ext_cur, int nb_fuse_cur);
	void print_progress(double progress);
	void print_progress_by_level(int lvl);
	void print_orbit_numbers(int depth);
	void print();
	void print_statistic_on_callbacks_naked();
	void print_statistic_on_callbacks();
	orbit_transversal *get_orbit_transversal(
			int level, int verbose_level);
	set_and_stabilizer *get_set_and_stabilizer(int level, 
		int orbit_at_level, int verbose_level);
	void get_set_by_level(int level, int node, int *set);
	void get_set(int node, int *set, int &size);
	void get_set(int level, int orbit, int *set, int &size);
	void print_set_verbose(int node);
	void print_set_verbose(int level, int orbit);
	void print_set(int node);
	void print_set(int level, int orbit);
	
	int find_poset_orbit_node_for_set(int len, int *set, 
		int f_tolerant, int verbose_level);
	int find_poset_orbit_node_for_set_basic(int from, 
		int node, int len, int *set, int f_tolerant, 
		int verbose_level);
	void poset_orbit_node_depth_breadth_perm_and_inverse(
		int max_depth,
		int *&perm, int *&perm_inv, int verbose_level);
	int count_extension_nodes_at_level(int lvl);
	double level_progress(int lvl);
	void count_automorphism_group_orders(int lvl, int &nb_agos, 
		longinteger_object *&agos, int *&multiplicities, 
		int verbose_level);
	void compute_and_print_automorphism_group_orders(int lvl, 
			std::ostream &ost);
	void stabilizer_order(int node, longinteger_object &go);
	void orbit_length(int orbit_at_level, int level,
		longinteger_object &len);
	void get_orbit_length_and_stabilizer_order(int node, int level, 
		longinteger_object &stab_order, longinteger_object &len);
	int orbit_length_as_int(int orbit_at_level, int level);
	void print_representatives_at_level(int lvl);
	void print_lex_rank(int *set, int sz);
	void print_problem_label();
	void print_level_info(int prev_level, int prev);
	void print_level_extension_info(int prev_level,
		int prev, int cur_extension);
	void print_level_extension_coset_info(int prev_level,
		int prev, int cur_extension, int coset, int nb_cosets);
	void recreate_schreier_vectors_up_to_level(int lvl, 
		int verbose_level);
	void recreate_schreier_vectors_at_level(int i,
		int verbose_level);
	void print_node(int node);
	void print_tree();
	void get_table_of_nodes(int *&Table, int &nb_rows, int &nb_cols, 
		int verbose_level);
	int count_live_points(int level, int node_local,
		int verbose_level);
	void find_automorphism_group_of_order(int level, int order);
	void get_stabilizer_order(int level, int orbit_at_level, 
		longinteger_object &go);
	void get_stabilizer_group(group *&G,  
		int level, int orbit_at_level, int verbose_level);
	void get_stabilizer_generators_cleaned_up(strong_generators *&gens,
		int level, int orbit_at_level, int verbose_level);
	void get_stabilizer_generators(strong_generators *&gens,
		int level, int orbit_at_level, int verbose_level);
	void change_extension_type(int level, int node, int cur_ext, 
		int type, int verbose_level);
	void orbit_element_unrank(int depth, int orbit_idx, 
		int rank, int *set, int verbose_level);
	void orbit_element_rank(int depth, int &orbit_idx, 
		int &rank, int *set, 
		int verbose_level);
		// used in M_SYSTEM/global_data::do_puzzle
		// and in UNITALS/test_lines_data::get_orbit_of_sets
	void coset_unrank(int depth, int orbit_idx, int rank, 
		int *Elt, int verbose_level);
	int coset_rank(int depth, int orbit_idx, int *Elt, 
		int verbose_level);
	void list_all_orbits_at_level(int depth, 
		int f_has_print_function, 
		void (*print_function)(std::ostream &ost, int len, int *S, void *data),
		void *print_function_data, 
		int f_show_orbit_decomposition, int f_show_stab, 
		int f_save_stab, int f_show_whole_orbit);
	void compute_integer_property_of_selected_list_of_orbits(
		int depth, 
		int nb_orbits, int *Orbit_idx, 
		int (*compute_function)(int len, int *S, void *data), 
		void *compute_function_data,
		int *&Data);
	void list_selected_set_of_orbits_at_level(int depth, 
		int nb_orbits, int *Orbit_idx, 
		int f_has_print_function, 
		void (*print_function)(std::ostream &ost, int len, int *S, void *data),
		void *print_function_data, 
		int f_show_orbit_decomposition, int f_show_stab, 
		int f_save_stab, int f_show_whole_orbit);
	void test_property(int depth, 
		int (*test_property_function)(int len, int *S, void *data), 
		void *test_property_data, 
		int &nb, int *&Orbit_idx);
	void list_whole_orbit(int depth, int orbit_idx, 
		int f_has_print_function, 
		void (*print_function)(std::ostream &ost, int len, int *S, void *data),
		void *print_function_data, 
		int f_show_orbit_decomposition, int f_show_stab, 
		int f_save_stab, int f_show_whole_orbit);
	void get_whole_orbit(
		int depth, int orbit_idx,
		int *&Orbit, int &orbit_length, int verbose_level);
	void print_extensions_at_level(std::ostream &ost, int lvl);
	void map_to_canonical_k_subset(int *the_set, int set_size, 
		int subset_size, int subset_rk, 
		int *reduced_set, int *transporter, int &local_idx, 
		int verbose_level);
		// fills reduced_set[set_size - subset_size], 
		// transporter and local_idx
		// local_idx is the index of the orbit that the 
		// subset belongs to 
		// (in the list of orbit of subsets of size subset_size)
	void get_representative_of_subset_orbit(int *set, int size, 
		int local_orbit_no, 
		strong_generators *&Strong_gens, 
		int verbose_level);
	void print_fusion_nodes(int depth);
	void find_interesting_k_subsets(int *the_set, int n, int k, 
		int *&interesting_sets, int &nb_interesting_sets, 
		int &orbit_idx, int verbose_level);
	void classify_k_subsets(int *the_set, int n, int k, 
		classify *&C, int verbose_level);
	void trace_all_k_subsets(int *the_set, int n, int k, 
		int &nCk, int *&isotype, int verbose_level);
	void get_orbit_representatives(int level, int &nb_orbits, 
		int *&Orbit_reps, int verbose_level);
	void unrank_point(int *v, int rk);
	int rank_point(int *v);
	void unrank_basis(int *Basis, int *S, int len);
	void rank_basis(int *Basis, int *S, int len);

	// poset_classification_init.C:
	poset_classification();
	~poset_classification();
	void null();
	void freeself();
	void usage();
	void read_arguments(int argc, const char **argv, 
		int verbose_level);
	void init(poset *Poset,
		int sz, int verbose_level);
	void initialize(poset *Poset,
		int depth, 
		const char *path, const char *prefix, int verbose_level);
	void initialize_with_starter(poset *Poset,
		int depth, 
		char *path, 
		char *prefix, 
		int starter_size, 
		int *starter, 
		strong_generators *Starter_Strong_gens, 
		int *starter_live_points, 
		int starter_nb_live_points, 
		void *starter_canonize_data, 
		int (*starter_canonize)(int *Set, int len, int *Elt, 
			void *data, int verbose_level), 
		int verbose_level);
	void init_root_node_invariant_subset(
		int *invariant_subset, int invariant_subset_size, 
		int verbose_level);
	void init_root_node(int verbose_level);
	void init_poset_orbit_node(int nb_poset_orbit_nodes,
		int verbose_level);
	void exit_poset_orbit_node();
	void reallocate();
	void reallocate_to(int new_number_of_nodes, int verbose_level);
	void init_starter(int starter_size, 
		int *starter, 
		strong_generators *starter_strong_gens, 
		int *starter_live_points, 
		int starter_nb_live_points, 
		void *starter_canonize_data, 
		int (*starter_canonize)(int *Set, int len, int *Elt, 
			void *data, int verbose_level), 
		int verbose_level);
		// Does not initialize the first starter nodes. 
		// This is done in init_root_node 

	// poset_classification_classify.C
	void compute_orbits_on_subsets(
		int target_depth,
		const char *prefix,
		int f_W, int f_w,
		poset *Poset,
		int verbose_level);
	int compute_orbits(int from_level, int to_level, 
		int verbose_level);
		// returns TRUE if there is at least one orbit 
		// at level to_level, 
		// FALSE otherwise
	int main(int t0, 
		int schreier_depth, 
		int f_use_invariant_subset_if_available, 
		int f_debug, 
		int verbose_level);
		// f_use_invariant_subset_if_available is 
		// an option that affects the downstep.
		// if FALSE, the orbits of the stabilizer 
		// on all points are computed. 
		// if TRUE, the orbits of the stabilizer 
		// on the set of points that were 
		// possible in the previous level are computed only 
		// (using Schreier.orbits_on_invariant_subset_fast).
		// The set of possible points is stored 
		// inside the schreier vector data structure (sv).
	void extend_level(int size, 
		int f_create_schreier_vector, 
		int f_use_invariant_subset_if_available, 
		int f_debug, 
		int f_write_candidate_file, 
		int verbose_level);
		// calls compute_flag_orbits, upstep
	void compute_flag_orbits(int size,
		int f_create_schreier_vector,
		int f_use_invariant_subset_if_available, 
		int verbose_level);
		// calls root[prev].downstep_subspace_action 
		// or root[prev].downstep
	void upstep(int size, int f_debug, 
		int verbose_level);
		// calls extend_node
	void extend_node(int size,
		int prev, int &cur,
		int f_debug, 
		int f_indicate_not_canonicals, FILE *fp,
		int verbose_level);
		// called by poset_classification::upstep
		// Uses an upstep_work structure to handle the work.
		// Calls upstep_work::handle_extension



	// poset_classification_combinatorics.C:
	void Plesken_matrix_up(int depth, 
		int *&P, int &N, int verbose_level);
	void Plesken_matrix_down(int depth, 
		int *&P, int &N, int verbose_level);
	void Plesken_submatrix_up(int i, int j, 
		int *&Pij, int &N1, int &N2, int verbose_level);
	void Plesken_submatrix_down(int i, int j,
		int *&Pij, int &N1, int &N2, int verbose_level);
	int count_incidences_up(int lvl1, int po1, 
		int lvl2, int po2, int verbose_level);
	int count_incidences_down(int lvl1, 
		int po1, int lvl2, int po2, int verbose_level);
	void Asup_to_Ainf(int t, int k, int *M_sup, 
		int *M_inf, int verbose_level);
	void test_for_multi_edge_in_classification_graph(
		int depth, int verbose_level);


	// poset_classification_trace.C:
	int find_isomorphism(int *set1, int *set2, int sz, 
		int *transporter, int &orbit_idx, int verbose_level);
	set_and_stabilizer *identify_and_get_stabilizer(
		int *set, int sz, int *transporter, 
		int &orbit_at_level, 
		int verbose_level);
	void identify(int *data, int sz, int *transporter, 
		int &orbit_at_level, int verbose_level);
	void test_identify(int level, int nb_times, int verbose_level);
	void poset_classification_apply_isomorphism_no_transporter(
		int cur_level, int size, int cur_node, int cur_ex, 
		int *set_in, int *set_out, 
		int verbose_level);
	int poset_classification_apply_isomorphism(int level, int size, 
		int current_node, int current_extension, 
		int *set_in, int *set_out, int *set_tmp, 
		int *transporter_in, int *transporter_out, 
		int f_tolerant, 
		int verbose_level);
	int trace_set_recursion(int cur_level, int cur_node, 
		int size, int level, 
		int *canonical_set, int *tmp_set1, int *tmp_set2, 
		int *Elt_transporter, int *tmp_Elt1, 
		int f_tolerant, 
		int verbose_level);
		// called by poset_classification::trace_set
		// returns the node in the poset_classification that corresponds 
		// to the canonical_set
		// or -1 if f_tolerant and the node could not be found
	int trace_set(int *set, int size, int level, 
		int *canonical_set, int *Elt_transporter, 
		int verbose_level);
		// returns the case number of the canonical set
	int find_node_for_subspace_by_rank(int *set, int len, 
		int verbose_level);
	
	// poset_classification_draw.C:
	void report_schreier_trees(std::ostream &ost, int verbose_level);
	void write_treefile_and_draw_tree(char *fname_base, 
		int lvl, int xmax, int ymax, int rad, 
		int f_embedded, int verbose_level);
	int write_treefile(char *fname_base, int lvl, 
		int verbose_level);
	void draw_tree(char *fname_base, int lvl, 
		int xmax, int ymax, int rad, int f_embedded, 
		int f_sideways, int verbose_level);
	void draw_tree_low_level(char *fname, int nb_nodes, 
		int *coord_xyw, int *perm, int *perm_inv, 
		int f_draw_points, int f_draw_extension_points, 
		int f_draw_aut_group_order, 
		int xmax, int ymax, int rad, int f_embedded, 
		int f_sideways, int verbose_level);
	void draw_tree_low_level1(mp_graphics &G, int nb_nodes, 
		int *coords, int *perm, int *perm_inv, 
		int f_draw_points, int f_draw_extension_points, 
		int f_draw_aut_group_order, 
		int radius, int verbose_level);
	void draw_poset_full(const char *fname_base, int depth, 
		int data, int f_embedded, int f_sideways, 
		double x_stretch, int verbose_level);
	void draw_poset_fname_base_aux_poset(
			char *fname, int depth);
	void draw_poset_fname_base_poset_lvl(
			char *fname, int depth);
	void draw_poset_fname_base_tree_lvl(
			char *fname, int depth);
	void draw_poset_fname_base_poset_detailed_lvl(
			char *fname, int depth);
	void draw_poset(const char *fname_base, int depth, 
		int data1, int f_embedded, int f_sideways, 
		int verbose_level);
	void draw_level_graph(const char *fname_base, int depth, 
		int data, int level, int f_embedded, int f_sideways, 
		int verbose_level);
	void make_full_poset_graph(int depth, layered_graph *&LG, 
		int data1, double x_stretch, 
		int verbose_level);
	void make_auxiliary_graph(int depth, 
		layered_graph *&LG, int data1, 
		int verbose_level);
	void make_graph(int depth, layered_graph *&LG, 
		int data1, int f_tree, int verbose_level);
	void make_level_graph(int depth, layered_graph *&LG, 
		int data1, int level, int verbose_level);
	void print_data_structure_tex(int depth, int verbose_level);
	void make_poset_graph_detailed(layered_graph *&LG, 
		int data1, int max_depth, int verbose_level);


	// in poset_classification_io.C:
	void read_data_file(int &depth_completed,
		const char *fname, int verbose_level);
	void write_data_file(int depth_completed,
		const char *fname_base, int verbose_level);
	void write_file(std::ofstream &fp, int depth_completed,
		int verbose_level);
	void read_file(std::ifstream &fp, int &depth_completed,
		int verbose_level);
	void read_memory_object(int &depth_completed,
		memory_object *m, int &nb_group_elements, int verbose_level);
	void write_memory_object(int depth_completed,
		memory_object *m, int &nb_group_elements, int verbose_level);
	int calc_size_on_file(int depth_completed, int verbose_level);
	void report(std::ostream &ost);
	void housekeeping(int i, int f_write_files, int t0, 
		int verbose_level);
	void housekeeping_no_data_file(int i, int t0, int verbose_level);
	int test_sv_level_file_binary(int level, char *fname_base);
	void read_sv_level_file_binary(int level, char *fname_base, 
		int f_split, int split_mod, int split_case, 
		int f_recreate_extensions, int f_dont_keep_sv, 
		int verbose_level);
	void write_sv_level_file_binary(int level, char *fname_base, 
		int f_split, int split_mod, int split_case, 
		int verbose_level);
	void read_sv_level_file_binary2(int level, FILE *fp, 
		int f_split, int split_mod, int split_case, 
		int f_recreate_extensions, int f_dont_keep_sv, 
		int verbose_level);
	void write_sv_level_file_binary2(int level, FILE *fp, 
		int f_split, int split_mod, int split_case, 
		int verbose_level);
	void read_level_file_binary(int level, char *fname_base, 
		int verbose_level);
	void write_level_file_binary(int level, char *fname_base, 
		int verbose_level);
	void read_level_file_binary2(int level, FILE *fp, 
		int &nb_group_elements, int verbose_level);
	void write_level_file_binary2(int level, FILE *fp, 
		int &nb_group_elements, int verbose_level);
	void write_candidates_binary_using_sv(char *fname_base, 
		int lvl, int t0, int verbose_level);
	void read_level_file(int level, char *fname, int verbose_level);
	void recover(const char *recover_fname, 
		int &depth_completed, int verbose_level);
	void write_lvl_file_with_candidates(char *fname_base, 
		int lvl, int t0, int verbose_level);
	void write_lvl_file(char *fname_base, int lvl, 
		int t0, int f_with_stabilizer_generators, int f_long_version,
		int verbose_level);
	void write_lvl(std::ostream &f, int lvl, int t0,
		int f_with_stabilizer_generators, int f_long_version,
		int verbose_level);
	void log_nodes_for_treefile(int cur, int depth, 
			std::ostream &f, int f_recurse, int verbose_level);
	void Log_nodes(int cur, int depth, std::ostream &f,
		int f_recurse, int verbose_level);
	void log_current_node(std::ostream &f, int size);
	void make_spreadsheet_of_orbit_reps(spreadsheet *&Sp, 
		int max_depth);
	void make_spreadsheet_of_level_info(spreadsheet *&Sp, 
		int max_depth, int verbose_level);
	void generate_source_code(int level, int verbose_level);
	void create_schreier_tree_fname_mask_base(
			char *fname_mask, int node);
	void create_shallow_schreier_tree_fname_mask_base(
			char *fname_mask, int node);
	void make_fname_candidates_file_default(char *fname, int level);
	void wedge_product_export_magma(
			int n, int q, int vector_space_dimension,
			int level, int verbose_level);

	// poset_classification_recognize.C:
	void recognize_start_over(
		int size, int f_implicit_fusion,
		int lvl, int current_node,
		int &final_node, int verbose_level);
	// Called from poset_orbit_node::recognize_recursion
	// when trace_next_point returns FALSE
	// This can happen only if f_implicit_fusion is TRUE
	void recognize_recursion(
		int size, int f_implicit_fusion,
		int lvl, int current_node, int &final_node,
		int verbose_level);
	// this routine is called by upstep_work::recognize
	// we are dealing with a set of size len + 1.
	// but we can only trace the first len points.
	// the tracing starts at lvl = 0 with current_node = 0
	void recognize(
		int *the_set, int size, int *transporter, int f_implicit_fusion,
		int &final_node, int verbose_level);

};


const char *trace_result_as_text(trace_result r);
int trace_result_is_no_result(trace_result r);



// #############################################################################
// poset_orbit_node.C, poset_orbit_node_io.C, poset_orbit_node_upstep.C,
// poset_orbit_node_upstep_subspace_action.C,
// poset_orbit_node_downstep.C, poset_orbit_node_downstep_subspace_action.C:
// #############################################################################


//! to represent one poset orbit; related to the class poset_classification


class poset_orbit_node {
public:
	int node;
	int prev;
	
	int pt;
	int nb_strong_generators;
	int *hdl_strong_generators;
	int *tl;
	
	int nb_extensions;
	extension *E;
	
	schreier_vector *Schreier_vector;
	
	action *A_on_upset;

	// poset_orbit_node.C:
	poset_orbit_node();
	~poset_orbit_node();
	void null();
	void freeself();
	void init_root_node(poset_classification *gen, int verbose_level);
		// copies gen->SG0 and gen->tl into the poset_orbit_node 
		// structure using store_strong_generators
	int get_level(poset_classification *gen);
	int get_node_in_level(poset_classification *gen);
	int *live_points();
	int get_nb_of_live_points();
	int get_nb_of_orbits_under_stabilizer();
	void poset_orbit_node_depth_breadth_perm_and_inverse(
		poset_classification *gen,
		int max_depth, 
		int &idx, int hdl, int cur_depth, int *perm, int *perm_inv);
	int find_extension_from_point(poset_classification *gen, int pt, 
		int verbose_level);
	void print_extensions(std::ostream &ost);
	void log_current_node_without_group(poset_classification *gen, 
		int s, std::ostream &f, int verbose_level);
	void log_current_node(poset_classification *gen, int s, 
			std::ostream &f, int f_with_stabilizer_poset_classifications,
		int verbose_level);
	void log_current_node_after_applying_group_element(
		poset_classification *gen, int s, std::ostream &f, int hdl,
		int verbose_level);
	void log_current_node_with_candidates(poset_classification *gen, 
		int lvl, std::ostream &f, int verbose_level);
	int depth_of_node(poset_classification *gen);
	void store_set(poset_classification *gen, int i);
		// stores a set of size i + 1 to gen->S
	void store_set_with_verbose_level(poset_classification *gen, int i, 
		int verbose_level);
	// stores a set of size i + 1 to gen->S[]
	void store_set_to(poset_classification *gen, int i, int *to);
	void store_set_to(poset_classification *gen, int *to);
	int check_node_and_set_consistency(poset_classification *gen,
		int i, int *set);
	void print_set_verbose(poset_classification *gen);
	void print_set(poset_classification *gen);
	void print_node(poset_classification *gen);
	void print_extensions(poset_classification *gen);
	void reconstruct_extensions_from_sv(poset_classification *gen,
		int verbose_level);
	int nb_extension_points();
		// sums up the lengths of orbits in all extensions

	// in poset_orbit_node_group_theory.C:
	void store_strong_generators(poset_classification *gen,
		strong_generators *Strong_gens);
	void get_stabilizer_order(poset_classification *gen,
		longinteger_object &go);
	void get_stabilizer(poset_classification *gen,
		group &G, longinteger_object &go_G,
		int verbose_level);
	void get_stabilizer_generators(poset_classification *gen,
		strong_generators *&Strong_gens,
		int verbose_level);
	void init_extension_node_prepare_G(
		poset_classification *gen,
		int prev, int prev_ex, int size, group &G,
		longinteger_object &go_G,
		int verbose_level);
		// sets up the group G using the strong
		// poset_classifications that are stored
	void init_extension_node_prepare_H(
		poset_classification *gen,
		int prev, int prev_ex, int size,
		group &G, longinteger_object &go_G,
		group &H, longinteger_object &go_H,
		int pt, int pt_orbit_len,
		int verbose_level);
		// sets up the group H which is the stabilizer
		// of the point pt in G
	void compute_point_stabilizer_in_subspace_setting(
		poset_classification *gen,
		int prev, int prev_ex, int size,
		group &G, longinteger_object &go_G,
		group &H, longinteger_object &go_H,
		int pt, int pt_orbit_len,
		int verbose_level);
	void compute_point_stabilizer_in_standard_setting(
		poset_classification *gen,
		int prev, int prev_ex, int size,
		group &G, longinteger_object &go_G,
		group &H, /* longinteger_object &go_H, */
		int pt, int pt_orbit_len,
		int verbose_level);
	void create_schreier_vector_wrapper(
		poset_classification *gen,
		int f_create_schreier_vector,
		schreier &Schreier, int verbose_level);
		// called from downstep_orbit_test_and_schreier_vector
		// calls Schreier.get_schreier_vector
	void create_schreier_vector_wrapper_subspace_action(
		poset_classification *gen,
		int f_create_schreier_vector,
		schreier &Schreier,
		action *A_factor_space, action_on_factor_space *AF,
		int verbose_level);


	// in poset_orbit_node_io.C:
	void read_memory_object(
		poset_classification *PC,
		action *A, memory_object *m,
		int &nb_group_elements, int verbose_level);
	void write_memory_object(
		poset_classification *PC,
		action *A, memory_object *m,
		int &nb_group_elements, int verbose_level);
	int calc_size_on_file(
		action *A, int verbose_level);
	void sv_read_file(
		poset_classification *PC,
		FILE *fp, int verbose_level);
	void sv_write_file(
		poset_classification *PC,
		FILE *fp, int verbose_level);
	void read_file(
		action *A, FILE *fp, int &nb_group_elements,
		int verbose_level);
	void write_file(
		action *A, FILE *fp, int &nb_group_elements,
		int verbose_level);
	void save_schreier_forest(
		poset_classification *PC,
		schreier *Schreier,
		int verbose_level);
	void save_shallow_schreier_forest(
		poset_classification *PC,
		int verbose_level);
	void draw_schreier_forest(
		poset_classification *PC,
		schreier *Schreier,
		int f_using_invariant_subset, action *AR,
		int verbose_level);


	// poset_orbit_node_upstep.C:
	int apply_isomorphism(poset_classification *gen, 
		int lvl, int current_node, 
		int current_extension, int len, int f_tolerant, 
		int verbose_level);
		// returns next_node
	void install_fusion_node(poset_classification *gen, 
		int lvl, int current_node, 
		int my_node, int my_extension, int my_coset, 
		int pt0, int current_extension, 
		int f_debug, int f_implicit_fusion, 
		int verbose_level);
		// Called from poset_orbit_node::handle_last_level
	int trace_next_point_wrapper(poset_classification *gen, int lvl, 
		int current_node, 
		int len, int f_implicit_fusion, int &f_failure_to_find_point, 
		int verbose_level);
		// Called from upstep_work::recognize_recursion
		// applies the permutation which maps the point with index lvl 
		// (i.e. the lvl+1-st point) to its orbit representative.
		// also maps all the other points under that permutation.
		// we are dealing with a set of size len + 1
		// returns FALSE if we are using implicit fusion nodes 
		// and the set becomes lexicographically
		// less than before, in which case trace has to be restarted.
	int trace_next_point_in_place(poset_classification *gen, 
		int lvl, int current_node, int size, 
		int *cur_set, int *tmp_set,
		int *cur_transporter, int *tmp_transporter, 
		int f_implicit_fusion, int &f_failure_to_find_point, 
		int verbose_level);
		// called by poset_classification::trace_set_recursion
	void trace_starter(poset_classification *gen, int size, 
		int *cur_set, int *next_set,
		int *cur_transporter, int *next_transporter, 
		int verbose_level);
	int trace_next_point(poset_classification *gen, 
		int lvl, int current_node, int size, 
		int *cur_set, int *next_set,
		int *cur_transporter, int *next_transporter, 
		int f_implicit_fusion, int &f_failure_to_find_point, 
		int verbose_level);
		// Called by poset_orbit_node::trace_next_point_wrapper 
		// and by poset_orbit_node::trace_next_point_in_place
		// returns FALSE only if f_implicit_fusion is TRUE and
		// the set becomes lexcographically less 
	int orbit_representative_and_coset_rep_inv(poset_classification *gen, 
		int lvl, int pt_to_trace, int &pt0, int *&cosetrep, 
		int verbose_level);
		// called by poset_orbit_node::trace_next_point
		// FALSE means the point to trace was not found. 
		// This can happen if nodes were eliminated due to clique_test

	// poset_orbit_node_upstep_subspace_action.C:
	void orbit_representative_and_coset_rep_inv_subspace_action(
		poset_classification *gen, 
		int lvl, int pt_to_trace, int &pt0, int *&cosetrep, 
		int verbose_level);
		// called by poset_orbit_node::trace_next_point
		

	// poset_orbit_node_downstep.C
	// top level functions:
	void compute_flag_orbits(
		poset_classification *gen,
		int lvl, 
		int f_create_schreier_vector,
		int f_use_invariant_subset_if_available, 
		int f_implicit_fusion, 
		int verbose_level);
		// Called from poset_classification::compute_flag_orbits
		// if we are acting on sets
		// (i.e., not on subspaces).
		// Calls downstep_orbits, 
		// downstep_orbit
	void compute_schreier_vector(poset_classification *gen, 
		int lvl, int verbose_level);
		// called from poset_classification::recreate_schreier_vectors_at_level
		// and from poset_classification::count_live_points
		// calls downstep_apply_early_test
		// and check_orbits
		// and Schreier.get_schreier_vector

		// 1st level under downstep:
	void schreier_forest(
		poset_classification *gen, schreier &Schreier, action *&AR,
		int lvl, 
		int f_use_invariant_subset_if_available, 
		int &f_using_invariant_subset, 
		int verbose_level);
		// calls downstep_get_invariant_subset, 
		// downstep_apply_early_test, 
		// and AR.induced_action_by_restriction
		// if f_use_invariant_subset_if_available and 
		// f_using_invariant_subset
		//
		// Sets up the schreier data structure Schreier 
		// If f_using_invariant_subset, we will use the 
		// restricted action AR, otherwise the action gen->A2
		// In this action, the orbits are computed using 
		// Schreier.compute_all_point_orbits
		// and possibly printed using downstep_orbits_print
	void downstep_orbit_test_and_schreier_vector(
		poset_classification *gen, schreier &Schreier, action *AR,
		int lvl, 
		int f_use_invariant_subset_if_available, 
		int f_using_invariant_subset,
		int f_create_schreier_vector,
		int &nb_good_orbits, int &nb_points, 
		int verbose_level);
		// called from downstep once downstep_orbits is completed
		// Calls check_orbits_wrapper and 
		// create_schreier_vector_wrapper
		// The order in which these two functions are called matters.
	void downstep_implicit_fusion(
		poset_classification *gen, schreier &Schreier, action *AR,
		int f_using_invariant_subset,
		int lvl, 
		int f_implicit_fusion, 
		int good_orbits1, int nb_points1, 
		int verbose_level);
		// called from downstep, 
		// once downstep_orbit_test_and_schreier_vector is done
		// calls test_orbits_for_implicit_fusion
	void find_extensions(poset_classification *gen, 
		schreier &O, action *AR, int f_using_invariant_subset,
		int lvl, 
		int verbose_level);
		// called by downstep
		// prepares all extension nodes and marks them as unprocessed.
		// we are at depth lvl, i.e., currently, 
		// we have a set of size lvl.


		// second level under downstep:
	int downstep_get_invariant_subset(
		poset_classification *gen, 
		int lvl, 
		int &n, int *&subset, int &f_subset_is_allocated, 
		int verbose_level);
		// called from downstep_orbits
		// Gets the live points at the present node.
	void downstep_apply_early_test(
		poset_classification *gen, 
		int lvl, 
		int n, int *subset, 
		int *candidates, int &nb_candidates, 
		int verbose_level);
		// called from downstep_orbits, compute_schreier_vector  
		// calls the callback early test function if available
		// and calls test_point_using_check_functions otherwise

	void check_orbits_wrapper(poset_classification *gen, 
		schreier &Schreier, action *AR, int f_using_invariant_subset,
		int lvl, 
		int &nb_good_orbits1, int &nb_points1, 
		int f_use_incremental_test_func_if_available, 
		int verbose_level);
		// called from downstep_orbit_test_and_schreier_vector
		// This function and create_schreier_vector_wrapper 
		// are used in pairs.
		// Except, the order in which the function is used matters.
		// Calls check_orbits

	void test_orbits_for_implicit_fusion(poset_classification *gen, 
		schreier &Schreier, action *AR,
		int f_using_invariant_subset, 
		int lvl, int verbose_level);
		// called from downstep_implicit_fusion
		// eliminates implicit fusion orbits from the 
		// Schreier data structure, 
	void check_orbits(poset_classification *gen, 
		schreier &Schreier, action *AR,
		int f_using_invariant_subset, 
		int lvl, 
		int f_use_incremental_test_func_if_available, 
		int verbose_level);
		// called from compute_schreier_vector 
		// and check_orbits_wrapper (which is called from 
		// downstep_orbit_test_and_schreier_vector)
		// calls test_point_using_check_functions
		// eliminates bad orbits from the Schreier data structure, 
		// does not eliminate implicit fusion orbits
	int test_point_using_check_functions(poset_classification *gen, 
		int lvl, int rep, int *the_set, 
		int verbose_level);
		// called by check_orbits and downstep_apply_early_test 
		// Calls gen->check_the_set_incrementally 
		// (if gen->f_candidate_incremental_check_func).
		// Otherwise, calls gen->check_the_set 
		// (if gen->f_candidate_check_func).
		// Otherwise accepts any point.
	void relabel_schreier_vector(action *AR, int verbose_level);
		// called from compute_schreier_vector, 
		// downstep_orbit_test_and_schreier_vector
	void downstep_orbits_print(poset_classification *gen, 
		schreier &Schreier, action *AR,
		int lvl, 
		int f_using_invariant_subset, int f_print_orbits, 
		int max_orbits, int max_points_per_orbit);


	// poset_orbit_node_downstep_subspace_action.C
	void compute_flag_orbits_subspace_action(
		poset_classification *gen,
		int lvl,
		int f_create_schreier_vector,
		int f_use_invariant_subset_if_available,
		int f_implicit_fusion,
		int verbose_level);
		// called from poset_classification::downstep
		// creates action *A_factor_space
		// and action_on_factor_space *AF
		// and disposes them at the end.
	void setup_factor_space_action_light(
		poset_classification *gen,
		action_on_factor_space &AF, 
		int lvl, int verbose_level);
	void setup_factor_space_action_with_early_test(
		poset_classification *gen,
		action_on_factor_space &AF, action &A_factor_space, 
		int lvl, int verbose_level);
	void setup_factor_space_action(
		poset_classification *gen,
		action_on_factor_space &AF, action &A_factor_space, 
		int lvl, int f_compute_tables, 
		int verbose_level);
	void downstep_subspace_action_print_orbits(
		poset_classification *gen, schreier &Schreier, 
		int lvl, 
		int f_print_orbits, 
		int verbose_level);
	void downstep_orbits_subspace_action(
		poset_classification *gen, schreier &Schreier, 
		int lvl, 
		int f_use_invariant_subset_if_available, 
		int &f_using_invariant_subset, 
		int verbose_level);
	void find_extensions_subspace_action(
		poset_classification *gen, schreier &O,
		action *A_factor_space, action_on_factor_space *AF, 
		int lvl, int f_implicit_fusion, int verbose_level);
};





// #############################################################################
// upstep_work.C:
// #############################################################################


typedef struct coset_table_entry coset_table_entry;

//! a helper class for the poset classification algorithm

struct coset_table_entry {
	int coset;
		// as in the loop in upstep_work::upstep_subspace_action
		// goes from 0 to degree - 1.

	int node; // = final_node as computed by recognize
	int ex; // = final_ex as computed by recognize
	int type; // = return value of recognize

	int nb_times_image_of_called;
	int nb_times_mult_called;
	int nb_times_invert_called;
	int nb_times_retrieve_called;
};


//! a helper class for the poset classification algorithm


class upstep_work {
public:
	poset_classification *gen;
	int size; // size = size of the object at prev
	int prev;
	int prev_ex;
	int cur;
	int nb_fusion_nodes;
	int nb_fuse_cur;
	int nb_ext_cur;
	int f_debug;
	int f_implicit_fusion;
	int f_indicate_not_canonicals;
	int mod_for_printing;


	int pt;
	int pt_orbit_len;
	
	int *path; // [size + 1] 
		// path[i] is the node that represents set[0,..,i-1]


	
	poset_orbit_node *O_prev;
	poset_orbit_node *O_cur;

	group *G;
	group *H;	
	longinteger_object go_G, go_H;

	int coset;
	
	int nb_cosets;
	int nb_cosets_processed;
	coset_table_entry *coset_table;

	FILE *f;


	upstep_work();
	~upstep_work();
	void init(poset_classification *gen, 
		int size,
		int prev,
		int prev_ex,
		int cur,
		int f_debug,
		int f_implicit_fusion,
		int f_indicate_not_canonicals, 
		FILE *fp, 
		int verbose_level);
		// called from poset_classification::extend_node
	void handle_extension(int &nb_fuse_cur, int &nb_ext_cur, 
		int verbose_level);
		// called from poset_classification::extend_node
		// Calls handle_extension_fusion_type 
		// or handle_extension_unprocessed_type
		//
		// Handles the extension 'cur_ex' in node 'prev'.
		// We are extending a set of size 'size' 
		// to a set of size 'size' + 1. 
		// Calls poset_orbit_node::init_extension_node for the 
		// n e w node that is (possibly) created
	void handle_extension_fusion_type(int verbose_level);
		// called from upstep_work::handle_extension
		// Handles the extension 'cur_ex' in node 'prev'.
	void handle_extension_unprocessed_type(int verbose_level);
		// called from upstep_work::handle_extension
		// calls init_extension_node
	int init_extension_node(int verbose_level);
		// Called from upstep_work::handle_extension_unprocessed_type
		// Calls upstep_subspace_action or upstep_for_sets, 
		// depending on the type of action
		// then changes the type of the extension to 
		// EXTENSION_TYPE_EXTENSION

		// Establishes a n e w node at depth 'size'
		// (i.e., a set of size 'size') as an extension 
		// of a previous node (prev) at depth size - 1 
		// with respect to a given point (pt).
		// This function is to be called for the next 
		// free poset_orbit_node node which will 
		// become the descendant of the previous node (prev).
		// the extension node corresponds to the point pt. 
		// returns FALSE if the set is not canonical 
		// (provided f_indicate_not_canonicals is TRUE)
	int upstep_for_sets(int verbose_level);
		// This routine is called from upstep_work::init_extension_node
		// It is testing a set of size 'size'. 
		// The newly added point is in gen->S[size - 1]
		// returns FALSE if the set is not canonical 
		// (provided f_indicate_not_canonicals is TRUE)
	//void print_level_extension_info_original_size();
	void print_level_extension_info();
	void print_level_extension_coset_info();

	// upstep_work_subspace_action.C:
	int upstep_subspace_action(int verbose_level);
		// This routine is called from upstep_work::init_extension_node
		// It computes coset_table.
		// It is testing a set of size 'size'. 
		// The newly added point is in gen->S[size - 1]
		// The extension is initiated from node 'prev' 
		// and from extension 'prev_ex' 
		// The node 'prev' is at depth 'size' - 1 
		// returns FALSE if the set is not canonical 
		// (provided f_indicate_not_canonicals is TRUE)


	// upstep_work_trace.C:

	trace_result recognize(
		int &final_node, int &final_ex, int f_tolerant, 
		int verbose_level);
	trace_result recognize_recursion(
		int lvl, int current_node, int &final_node, int &final_ex, 
		int f_tolerant, int verbose_level);
	trace_result handle_last_level(
		int lvl, int current_node, int current_extension, int pt0, 
		int &final_node, int &final_ex,  
		int verbose_level);
	trace_result start_over(
		int lvl, int current_node, 
		int &final_node, int &final_ex, 
		int f_tolerant, int verbose_level);
};

// in upstep_work.C:
void print_coset_table(coset_table_entry *coset_table, int len);

}}



