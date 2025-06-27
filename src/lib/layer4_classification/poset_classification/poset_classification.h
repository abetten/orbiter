/*
 * poset_classification.h
 *
 *  Created on: Aug 9, 2018
 *      Author: Anton Betten
 *
 * started:  September 20, 2007
 * pulled out of snakesandladders.h: Aug 9, 2018
 */


#ifndef ORBITER_SRC_LIB_CLASSIFICATION_POSET_CLASSIFICATION_POSET_CLASSIFICATION_H_
#define ORBITER_SRC_LIB_CLASSIFICATION_POSET_CLASSIFICATION_POSET_CLASSIFICATION_H_



namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


// #############################################################################
// classification_base_case.cpp
// #############################################################################



//! represents a known classification with constructive recognition, to be used as base case for poset_classification



class classification_base_case {

public:
	poset_with_group_action *Poset;

	int size;
	long int *orbit_rep; // [size]
	groups::strong_generators *Stab_gens;
	long int *live_points;
	int nb_live_points;
	void *recognition_function_data;
	int (*recognition_function)(
			long int *Set, int len, int *Elt,
		void *data, int verbose_level);
	int *Elt;

	classification_base_case();
	~classification_base_case();
	void init(
			poset_with_group_action *Poset,
			int size, long int *orbit_rep,
			long int *live_points, int nb_live_points,
			groups::strong_generators *Stab_gens,
			void *recognition_function_data,
			int (*recognition_function)(long int *Set, int len,
					int *Elt, void *data, int verbose_level),
			int verbose_level);
	int invoke_recognition(
			long int *Set, int len,
			int *Elt, int verbose_level);
};

// #############################################################################
// extension.cpp
// #############################################################################

#define NB_EXTENSION_TYPES 5

#define EXTENSION_TYPE_UNPROCESSED  0
#define EXTENSION_TYPE_EXTENSION 1
#define EXTENSION_TYPE_FUSION 2
#define EXTENSION_TYPE_PROCESSING 3
#define EXTENSION_TYPE_NOT_CANONICAL 4


//! represents a flag in the poset classification algorithm; related to poset_orbit_node



class extension {

private:
	long int pt;
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

public:

	extension();
	~extension();
	int get_pt();
	void set_pt(
			int pt);
	int get_type();
	void set_type(
			int type);
	int get_orbit_len();
	void set_orbit_len(
			int orbit_len);
	int get_data();
	void set_data(
			int data);
	int get_data1();
	void set_data1(
			int data1);
	int get_data2();
	void set_data2(
			int data1);
};


void print_extension_type(
		std::ostream &ost, int t);


// #############################################################################
// orbit_based_testing.cpp
// #############################################################################


#define MAX_CALLBACK 100


//! maintains a list of test functions which define a G-invariant poset

class orbit_based_testing {

public:

	poset_classification *PC;
	int max_depth;
	long int *local_S; // [max_depth]
	int nb_callback;
	int (*callback_testing[MAX_CALLBACK])(orbit_based_testing *Obt,
			long int *S, int len, void *data, int verbose_level);
	void *callback_data[MAX_CALLBACK];

	int nb_callback_no_group;
	void (*callback_testing_no_group[MAX_CALLBACK])(
			long int *S, int len,
			long int *candidates, int nb_candidates,
			long int *good_candidates, int &nb_good_candidates,
			void *data, int verbose_level);
	void *callback_data_no_group[MAX_CALLBACK];

	orbit_based_testing();
	~orbit_based_testing();
	void init(
			poset_classification *PC,
			int max_depth,
			int verbose_level);
	void add_callback(
			int (*func)(orbit_based_testing *Obt,
					long int *S, int len,
					void *data, int verbose_level),
			void *data,
			int verbose_level);
	void add_callback_no_group(
			void (*func)(long int *S, int len,
					long int *candidates, int nb_candidates,
					long int *good_candidates,
					int &nb_good_candidates,
					void *data, int verbose_level),
			void *data,
			int verbose_level);
	void early_test_func(
		long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void early_test_func_by_using_group(
		long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
};


// #############################################################################
// orbit_tracer.cpp
// #############################################################################

//! to trace a set in the poset classification algorithm


class orbit_tracer {

private:

	poset_classification *PC;

	// data for recognize:

	data_structures_groups::vector_ge *Transporter; // [PC->sz + 1]

	long int **Set; // [PC->sz + 1][PC->max_set_size]

	int *Elt1;
	int *Elt2;
	int *Elt3;


public:

	orbit_tracer();
	~orbit_tracer();
	void init(
			poset_classification *PC, int verbose_level);
	data_structures_groups::vector_ge *get_transporter();
	long int *get_set_i(
			int i);

	void recognize_start_over(
		int size,
		int lvl, int current_node,
		int &final_node, int verbose_level);
	// Called from poset_orbit_node::recognize_recursion
	// when trace_next_point returns false
	// This can happen only if f_implicit_fusion is true
	void recognize_recursion(
		int size,
		int lvl, int current_node, int &final_node,
		int verbose_level);
	// this routine is called by upstep_work::recognize
	// we are dealing with a set of size len + 1.
	// but we can only trace the first len points.
	// the tracing starts at lvl = 0 with current_node = 0
	// The input set the_set[] is not modified.
	void recognize(
		long int *the_set, int size, int *transporter,
		int &final_node, int verbose_level);
	void identify(
			long int *data, int sz,
			int *transporter, int &orbit_at_level,
			int verbose_level);



};




// #############################################################################
// poset_classification_activity_description.cpp
// #############################################################################


//! description on an activity for the poset classification after it has been computed


class poset_classification_activity_description {

public:

	// TABLES/poset_classification_activity.tex

	int f_report;
	poset_classification_report_options *report_options;

	int f_export_level_to_cpp;
	int export_level_to_cpp_level;
	int f_export_history_to_cpp;
	int export_history_to_cpp_level;


	int f_write_tree; // create a tree
	std::string write_tree_draw_options;

	int f_find_node_by_stabilizer_order;
	int find_node_by_stabilizer_order;


	int f_draw_poset;
	std::string draw_poset_draw_options;

	int f_draw_full_poset;
	std::string draw_full_poset_draw_options;

	int f_plesken;


	int f_print_data_structure;

	int f_list;
	int f_list_all;
	int f_table_of_nodes;
	int f_make_relations_with_flag_orbits;

	int f_level_summary_csv;
	int f_orbit_reps_csv;


	int f_node_label_is_group_order;
	int f_node_label_is_element;

	int f_show_orbit_decomposition;
	int f_show_stab;
	int f_save_stab;
	int f_show_whole_orbits;



	int f_export_schreier_trees;
	int f_draw_schreier_trees;
	std::string schreier_tree_prefix;
			// comes after problem_label_with_path

	int f_test_multi_edge_in_decomposition_matrix;



	std::vector<std::string> recognize;


	poset_classification_activity_description();
	~poset_classification_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// poset_classification_activity.cpp
// #############################################################################


//! an activity for the poset classification after it has been computed


class poset_classification_activity {

public:

	poset_classification_activity_description *Descr;
	poset_classification *PC;
	int actual_size;


	poset_classification_activity();
	~poset_classification_activity();
	void init(
			poset_classification_activity_description *Descr,
			poset_classification *PC,
			int actual_size,
			int verbose_level);
	void perform_work(
			int verbose_level);

	void compute_Kramer_Mesner_matrix(
			int t, int k,
			int verbose_level);
	void Plesken_matrix_up(
			int depth,
		int *&P, int &N, int verbose_level);
	void Plesken_matrix_down(
			int depth,
		int *&P, int &N, int verbose_level);
	void Plesken_submatrix_up(
			int i, int j,
		int *&Pij, int &N1, int &N2, int verbose_level);
	void Plesken_submatrix_down(
			int i, int j,
		int *&Pij, int &N1, int &N2, int verbose_level);
	int count_incidences_up(
			int lvl1, int po1,
		int lvl2, int po2, int verbose_level);
	int count_incidences_down(
			int lvl1,
		int po1, int lvl2, int po2, int verbose_level);
	void Asup_to_Ainf(
			int t, int k,
		long int *M_sup, long int *&M_inf,
		int verbose_level);
	void test_for_multi_edge_in_classification_graph(
		int depth, int verbose_level);
	void Kramer_Mesner_matrix_neighboring(
			int level, long int *&M,
			int &nb_rows, int &nb_cols, int verbose_level);
	void Mtk_via_Mtr_Mrk(
			int t, int r, int k,
			long int *Mtr, long int *Mrk, long int *&Mtk,
			int nb_r1, int nb_c1, int nb_r2, int nb_c2,
			int &nb_r3, int &nb_c3,
			int verbose_level);
	// Computes $M_{tk}$ via a recursion formula:
	// $M_{tk} = {{k - t} \choose {k - r}} \cdot M_{t,r} \cdot M_{r,k}$.
	void Mtk_from_MM(
			long int **pM,
		int *Nb_rows, int *Nb_cols,
		int t, int k,
		long int *&Mtk, int &nb_r, int &nb_c,
		int verbose_level);

	// poset_classification_activity_export_source_code.cpp:
	void generate_source_code(
			int level, int verbose_level);
	void generate_history(
			int level, int verbose_level);



};


// #############################################################################
// poset_classification_control.cpp
// #############################################################################


//! to control the behavior of the poset classification algorithm


class poset_classification_control {

public:

	// TABLES/poset_classification_control.tex

	int f_problem_label;
	std::string problem_label;

	int f_path;
	std::string path;

	int f_depth;
	int depth;

	int f_verbose_level;
	int verbose_level;

	int f_verbose_level_group_theory;
	int verbose_level_group_theory;

	int f_recover;
	std::string recover_fname;

	int f_extend;
	int extend_from, extend_to;
	int extend_r, extend_m;
	std::string extend_fname;

	int f_lex;


	int f_w; // write output in level files (only last level)
	int f_W; // write output in level files (each level)
	int f_write_data_files;

	int f_T; // draw tree file (each level)
	int f_t; // draw tree file (only last level)


	int f_draw_options;
	std::string draw_options_label;
	//graphics::layered_graph_draw_options *draw_options;
		// used for write_treefile in poset_classification_io




	int f_preferred_choice;
	std::vector<std::vector<int> > preferred_choice;

	int f_clique_test;
	std::string clique_test_graph;
	combinatorics::graph_theory::colored_graph *clique_test_CG;


	int f_test_mindist_nonlinear;
	int mindist_nonlinear;
	poset_classification *test_mindist_nonlinear_PC;
	int *test_mindist_nonlinear_word1;
	int *test_mindist_nonlinear_word2;




	int f_has_invariant_subset_for_root_node;
	int *invariant_subset_for_root_node;
	int invariant_subset_for_root_node_size;


	int f_do_group_extension_in_upstep;
		// is true by default

	int f_allowed_to_show_group_elements;
	int downstep_orbits_print_max_orbits;
	int downstep_orbits_print_max_points_per_orbit;




	poset_classification_control();
	~poset_classification_control();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();
	void prepare(
			poset_classification *PC, int verbose_level);
	void early_test_func_for_clique_search(
		long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void early_test_func_for_mindist_nonlinear(
		long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void init_root_node_invariant_subset(
		int *invariant_subset, int invariant_subset_size,
		int verbose_level);

};

void poset_classification_control_preferred_choice_function(
		int pt, int &pt_pref,
		groups::schreier *Sch,
		void *data, int data2,
		int verbose_level);





// #############################################################################
// poset_classification.cpp
// #############################################################################

//! the poset classification algorithm


class poset_classification {

private:
	int t0;

	poset_classification_control *Control;


	std::string problem_label;
		// = Control->problem_label
	std::string problem_label_with_path;
		// Control->path + Control->problem_label

	
	poset_with_group_action *Poset;


	int f_base_case;
	classification_base_case *Base_case;


	data_structures_groups::schreier_vector_handler
		*Schreier_vector_handler;


	int depth;


	// used as storage for the current set:
	// poset_orbit_node::store_set stores to set_S[]
	long int *set_S; // [sz]
	
	int sz; // = depth, the target depth
	int max_set_size; // Poset->A2->degree

	
	orbit_tracer *Orbit_tracer;

	poset_of_orbits *Poo;




	long int nb_times_image_of_called0;
	long int nb_times_mult_called0;
	long int nb_times_invert_called0;
	long int nb_times_retrieve_called0;
	long int nb_times_store_called0;
	
	double progress_last_time;
	double progress_epsilon;

public:



	// poset_classification.cpp:

	poset_of_orbits *get_Poo();
	orbit_tracer *get_Orbit_tracer();
	std::string &get_problem_label_with_path();
	std::string &get_problem_label();
	int first_node_at_level(
			int i);
	poset_orbit_node *get_node(
			int node_idx);
	data_structures_groups::vector_ge *get_transporter();
	int *get_transporter_i(
			int i);
	int get_sz();
	int get_max_set_size();
	long int *get_S();
	long int *get_set_i(
			int i);
	long int *get_set0();
	long int *get_set1();
	long int *get_set3();
	int allowed_to_show_group_elements();
	int do_group_extension_in_upstep();
	poset_with_group_action *get_poset();
	poset_classification_control *get_control();
	actions::action *get_A();
	actions::action *get_A2();
	algebra::linear_algebra::vector_space *get_VS();
	data_structures_groups::schreier_vector_handler
		*get_schreier_vector_handler();
	int &get_depth();
	int has_base_case();
	int has_invariant_subset_for_root_node();
	int size_of_invariant_subset_for_root_node();
	int *get_invariant_subset_for_root_node();
	classification_base_case *get_Base_case();
	int node_has_schreier_vector(
			int node_idx);
	int max_number_of_orbits_to_print();
	int max_number_of_points_to_print_in_orbit();
	void invoke_early_test_func(
			long int *the_set, int lvl,
			long int *candidates,
			int nb_candidates,
			long int *good_candidates,
			int &nb_good_candidates,
			int verbose_level);
	int nb_orbits_at_level(
			int level);
	long int nb_flag_orbits_up_at_level(
			int level);
	poset_orbit_node *get_node_ij(
			int level, int node);
	int poset_structure_is_contained(
			long int *set1, int sz1,
		long int *set2, int sz2, int verbose_level);
	data_structures_groups::orbit_transversal
		*get_orbit_transversal(
			int level, int verbose_level);
	int test_if_stabilizer_is_trivial(
			int level, int orbit_at_level, int verbose_level);
	data_structures_groups::set_and_stabilizer
		*get_set_and_stabilizer(
				int level,
		int orbit_at_level, int verbose_level);
	void get_set_by_level(
			int level, int node, long int *set);
	void get_set(
			int node, long int *set, int &size);
	void get_set(
			int level, int orbit, long int *set, int &size);
	
	int find_poset_orbit_node_for_set(
			int len, long int *set,
		int f_tolerant, int verbose_level);
	int find_poset_orbit_node_for_set_basic(
			int from,
		int node, int len, long int *set, int f_tolerant,
		int verbose_level);
	long int count_extension_nodes_at_level(
			int lvl);
	double level_progress(
			int lvl);
	void count_automorphism_group_orders(
			int lvl, int &nb_agos,
			algebra::ring_theory::longinteger_object *&agos,
			int *&multiplicities,
		int verbose_level);
	void compute_and_print_automorphism_group_orders(
			int lvl,
			std::ostream &ost);
	void stabilizer_order(
			int node, algebra::ring_theory::longinteger_object &go);
	void orbit_length(
			int orbit_at_level, int level,
			algebra::ring_theory::longinteger_object &len);
	void get_orbit_length_and_stabilizer_order(
			int node, int level,
			algebra::ring_theory::longinteger_object &stab_order,
			algebra::ring_theory::longinteger_object &len);
	int orbit_length_as_int(
			int orbit_at_level, int level);
	void recreate_schreier_vectors_up_to_level(
			int lvl,
		int verbose_level);
	void recreate_schreier_vectors_at_level(
			int level,
		int verbose_level);
	void find_node_by_stabilizer_order(
			int level, int order, int verbose_level);
	void get_all_stabilizer_orders_at_level(
			int level,
			long int *&Ago, int &nb, int verbose_level);
	void get_stabilizer_order(
			int level, int orbit_at_level,
			algebra::ring_theory::longinteger_object &go);
	long int get_stabilizer_order_lint(
			int level,
			int orbit_at_level);
	void get_stabilizer_group(
			data_structures_groups::group_container *&G,
		int level, int orbit_at_level, int verbose_level);
	void get_stabilizer_generators_cleaned_up(
			groups::strong_generators *&gens,
		int level, int orbit_at_level, int verbose_level);
	void get_stabilizer_generators(
			groups::strong_generators *&gens,
		int level, int orbit_at_level, int verbose_level);
	void orbit_element_unrank(
			int depth, int orbit_idx,
		long int rank, long int *set, int verbose_level);
	void orbit_element_rank(
			int depth, int &orbit_idx,
		long int &rank, long int *set,
		int verbose_level);
	void coset_unrank(
			int depth, int orbit_idx, long int rank,
		int *Elt, int verbose_level);
	long int coset_rank(
			int depth, int orbit_idx, int *Elt,
		int verbose_level);
	void list_all_orbits_at_level(
			int depth,
		int f_has_print_function, 
		void (*print_function)(std::ostream &ost,
				int len, long int *S, void *data),
		void *print_function_data, 
		int f_show_orbit_decomposition, int f_show_stab, 
		int f_save_stab, int f_show_whole_orbit);
	void compute_integer_property_of_selected_list_of_orbits(
		int depth, 
		int nb_orbits, int *Orbit_idx, 
		int (*compute_function)(
				int len, long int *S, void *data),
		void *compute_function_data,
		int *&Data);
	void list_selected_set_of_orbits_at_level(
			int depth,
		int nb_orbits, int *Orbit_idx, 
		int f_has_print_function, 
		void (*print_function)(std::ostream &ost,
				int len, long int *S, void *data),
		void *print_function_data, 
		int f_show_orbit_decomposition, int f_show_stab, 
		int f_save_stab, int f_show_whole_orbit);
	void test_property(
			int depth,
		int (*test_property_function)(
				int len, long int *S, void *data),
		void *test_property_data, 
		int &nb, int *&Orbit_idx);
	void list_whole_orbit(
			int depth, int orbit_idx,
		int f_has_print_function, 
		void (*print_function)(std::ostream &ost,
				int len, long int *S, void *data),
		void *print_function_data, 
		int f_show_orbit_decomposition, int f_show_stab, 
		int f_save_stab, int f_show_whole_orbit);
	void get_whole_orbit(
		int depth, int orbit_idx,
		long int *&Orbit, int &orbit_length, int verbose_level);
	void map_to_canonical_k_subset(
			long int *the_set, int set_size,
		int subset_size, int subset_rk, 
		long int *reduced_set, int *transporter, int &local_idx,
		int verbose_level);
		// fills reduced_set[set_size - subset_size], 
		// transporter and local_idx
		// local_idx is the index of the orbit that the 
		// subset belongs to 
		// (in the list of orbit of subsets of size subset_size)
	void get_representative_of_subset_orbit(
			long int *set, int size,
		int local_orbit_no, 
		groups::strong_generators *&Strong_gens,
		int verbose_level);
	void find_interesting_k_subsets(
			long int *the_set, int n, int k,
		int *&interesting_sets, int &nb_interesting_sets, 
		int &orbit_idx, int verbose_level);
	void classify_k_subsets(
			long int *the_set, int n, int k,
			other::data_structures::tally *&C, int verbose_level);
	void trace_all_k_subsets_and_compute_frequencies(
			long int *the_set,
			int n, int k, int &nCk,
			int *&isotype, int *&orbit_frequencies, int &nb_orbits,
			int verbose_level);
	void trace_all_k_subsets(
			long int *the_set, int n, int k,
		int &nCk, int *&isotype, int verbose_level);
	void get_orbit_representatives(
			int level, int &nb_orbits,
		long int *&Orbit_reps, int verbose_level);
	void unrank_point(
			int *v, long int rk);
	long int rank_point(
			int *v);
	void unrank_basis(
			int *Basis, long int *S, int len);
	void rank_basis(
			int *Basis, long int *S, int len);

	// poset_classification_init.cpp:
	poset_classification();
	~poset_classification();
	void init_internal(
		poset_classification_control *PC_control,
		poset_with_group_action *Poset,
		int sz, int verbose_level);
	void initialize_and_allocate_root_node(
		poset_classification_control *PC_control,
		poset_with_group_action *Poset,
		int depth, 
		int verbose_level);
	void initialize_with_base_case(
		poset_classification_control *PC_control,
		poset_with_group_action *Poset,
		int depth,
		classification_base_case *Base_case,
		int verbose_level);
	void init_base_case(
			classification_base_case *Base_case,
		int verbose_level);
		// Does not initialize the first starter nodes. 
		// This is done in init_root_node 

	// poset_classification_classify.cpp:
	void compute_orbits_on_subsets(
		int target_depth,
		poset_classification_control *PC_control,
		poset_with_group_action *Poset,
		int verbose_level);
	int main(
			int t0,
		int schreier_depth, 
		int f_use_invariant_subset_if_available, 
		int f_debug, 
		int verbose_level);
		// f_use_invariant_subset_if_available is 
		// an option that affects the downstep.
		// if false, the orbits of the stabilizer 
		// on all points are computed. 
		// if true, the orbits of the stabilizer 
		// on the set of points that were 
		// possible in the previous level are computed only 
		// (using Schreier.orbits_on_invariant_subset_fast).
		// The set of possible points is stored 
		// inside the schreier vector data structure (sv).
	int compute_orbits(
			int from_level, int to_level,
			int schreier_depth,
			int f_use_invariant_subset_if_available,
			int verbose_level);
			// returns the last level that has at least one orbit
	//void post_processing(int actual_size, int verbose_level);
	void recognize(
			std::string &set_to_recognize,
			int h, int nb_to_recognize, int verbose_level);
	void extend_level(
			int size,
		int f_create_schreier_vector, 
		int f_use_invariant_subset_if_available, 
		int f_debug, 
		int f_write_candidate_file, 
		int verbose_level);
		// calls compute_flag_orbits, upstep
	void compute_flag_orbits(
			int size,
		int f_create_schreier_vector,
		int f_use_invariant_subset_if_available, 
		int verbose_level);
		// calls root[prev].downstep_subspace_action 
		// or root[prev].downstep
	void upstep(
			int size, int f_debug,
		int verbose_level);
		// calls extend_node
	void extend_node(
			int size,
		int prev, int &cur,
		int f_debug, 
		int f_indicate_not_canonicals,
		int verbose_level);
		// called by poset_classification::upstep
		// Uses an upstep_work structure to handle the work.
		// Calls upstep_work::handle_extension





	// poset_classification_draw.cpp:
	void draw_poset_fname_base_aux_poset(
			std::string &fname, int depth);
	void draw_poset_fname_base_poset_lvl(
			std::string &fname, int depth);
	void draw_poset_fname_base_tree_lvl(
			std::string &fname, int depth);
	void draw_poset_fname_base_poset_detailed_lvl(
			std::string &fname, int depth);
	void draw_poset_fname_aux_poset(
			std::string &fname, int depth);
	void draw_poset_fname_poset(
			std::string &fname, int depth);
	void draw_poset_fname_tree(
			std::string &fname, int depth);
	void draw_poset_fname_poset_detailed(
			std::string &fname, int depth);
	void write_treefile(
			std::string &fname_base,
		int lvl,
		other::graphics::layered_graph_draw_options *draw_options,
		int verbose_level);
	int write_treefile(
			std::string &fname_base, int lvl,
		int verbose_level);
	void draw_tree(
			std::string &fname_base, int lvl,
			other::graphics::tree_draw_options *Tree_draw_options,
			other::graphics::layered_graph_draw_options *Draw_options,
		int xmax, int ymax, int rad, int f_embedded,
		int f_sideways, int verbose_level);
	void draw_tree_low_level(
			std::string &fname,
			other::graphics::tree_draw_options *Tree_draw_options,
			other::graphics::layered_graph_draw_options *Draw_options,
			int nb_nodes,
		int *coord_xyw, int *perm, int *perm_inv, 
		int f_draw_points, int f_draw_extension_points,
		int f_draw_aut_group_order,
		int xmax, int ymax, int rad, int f_embedded,
		int f_sideways,
		int verbose_level);
	void draw_tree_low_level1(
			other::graphics::mp_graphics &G, int nb_nodes,
		int *coords, int *perm, int *perm_inv, 
		int f_draw_points, int f_draw_extension_points, 
		int f_draw_aut_group_order, 
		int radius, int verbose_level);
	void draw_poset_full(
			std::string &fname_base,
			int depth, int data,
			other::graphics::layered_graph_draw_options *LG_Draw_options,
			double x_stretch,
			int verbose_level);
	void draw_poset(
			std::string &fname_base,
			int depth, int data,
			other::graphics::layered_graph_draw_options *LG_Draw_options,
			int verbose_level);
	void draw_level_graph(
			std::string &fname_base,
			int depth, int data, int level,
			other::graphics::layered_graph_draw_options *LG_Draw_options,
			int verbose_level);
	void make_flag_orbits_on_relations(
			int depth,
			std::string &fname_prefix, int verbose_level);
	void make_full_poset_graph(
			int depth,
			combinatorics::graph_theory::layered_graph *&LG,
		int data1, double x_stretch, 
		int verbose_level);
		// Draws the full poset: each element of each orbit is drawn.
		// The orbits are indicated by grouping the elements closer together.
		// Uses int_vec_sort_and_test_if_contained to test containment relation.
		// This is only good for actions on sets, not for actions on subspaces
	void make_auxiliary_graph(
			int depth,
			combinatorics::graph_theory::layered_graph *&LG, int data1,
		int verbose_level);
		// makes a graph of the poset of orbits with 2 * depth + 1 layers.
		// The middle layers represent the flag orbits.
	void make_graph(
			int depth, combinatorics::graph_theory::layered_graph *&LG,
		int data1, int f_tree, int verbose_level);
		// makes a graph  of the poset of orbits with depth + 1 layers.
	void make_level_graph(
			int depth,
			combinatorics::graph_theory::layered_graph *&LG,
		int data1, int level, int verbose_level);
		// makes a graph with 4 levels showing the relation between
		// orbits at level 'level' and orbits at level 'level' + 1
	void make_poset_graph_detailed(
			combinatorics::graph_theory::layered_graph *&LG,
		int data1, int max_depth, int verbose_level);
		// creates the poset graph, with two middle layers at each level.
		// In total, the graph that is created will have 3 * depth + 1 layers.
	void print_data_structure_tex(
			int depth, int verbose_level);


	// in poset_classification_io.cpp:
	void print_set_verbose(
			int node);
	void print_set_verbose(
			int level, int orbit);
	void print_set(
			int node);
	void print_set(
			int level, int orbit);
	void print_progress_by_extension(
			int size, int cur,
		int prev, int cur_ex, int nb_ext_cur, int nb_fuse_cur);
	void print_progress(
			int size, int cur, int prev,
		int nb_ext_cur, int nb_fuse_cur);
	void print_progress(
			double progress);
	void print_progress_by_level(
			int lvl);
	void print_orbit_numbers(
			int depth);
	void print();
	void print_statistic_on_callbacks_bare();
	void print_statistic_on_callbacks();
	void prepare_fname_data_file(
			std::string &fname,
			std::string &fname_base, int depth_completed);
	void print_representatives_at_level(
			int lvl);
	void print_lex_rank(
			long int *set, int sz);
	void print_problem_label();
	void print_level_info(
			int prev_level, int prev);
	void print_level_extension_info(
			int prev_level,
		int prev, int cur_extension);
	void print_level_extension_coset_info(
			int prev_level,
		int prev, int cur_extension, int coset, int nb_cosets);
	void print_node(
			int node);
	void print_extensions_at_level(
			std::ostream &ost, int lvl);
	void print_fusion_nodes(
			int depth);
	void read_data_file(
			int &depth_completed,
		std::string &fname, int verbose_level);
	void write_data_file(
			int depth_completed,
			std::string &fname_base, int verbose_level);
	void write_file(
			std::ofstream &fp, int depth_completed,
		int verbose_level);
	void read_file(
			std::ifstream &fp, int &depth_completed,
		int verbose_level);
	void housekeeping(
			int i, int f_write_files, int t0,
		int verbose_level);
	void housekeeping_no_data_file(
			int i, int t0, int verbose_level);
	void create_fname_sv_level_file_binary(
			std::string &fname,
			std::string &fname_base, int level);
	int test_sv_level_file_binary(
			int level, std::string &fname_base);
	void read_sv_level_file_binary(
			int level, std::string &fname_base,
		int f_split, int split_mod, int split_case, 
		int f_recreate_extensions, int f_dont_keep_sv, 
		int verbose_level);
	void write_sv_level_file_binary(
			int level, std::string &fname_base,
		int f_split, int split_mod, int split_case, 
		int verbose_level);
	void read_level_file_binary(
			int level, std::string &fname_base,
		int verbose_level);
	void write_level_file_binary(
			int level, std::string &fname_base,
		int verbose_level);
	void recover(
			std::string &recover_fname,
		int &depth_completed, int verbose_level);
	void make_fname_lvl_file_candidates(
			std::string &fname,
			std::string &fname_base, int lvl);
	void make_fname_lvl_file(
			std::string &fname,
			std::string &fname_base, int lvl);
	void make_fname_lvl_reps_file(
			std::string &fname,
			std::string &fname_base, int lvl);
	void log_current_node(
			std::ostream &f, int size);

	void make_spreadsheet_of_orbit_reps(
			other::data_structures::spreadsheet *&Sp,
		int max_depth);
	void make_spreadsheet_of_level_info(
			other::data_structures::spreadsheet *&Sp,
		int max_depth, int verbose_level);

	void create_schreier_tree_fname_mask_base(
			std::string &fname_mask);
	void create_schreier_tree_fname_mask_base_tex(
			std::string &fname_mask);
	void create_shallow_schreier_tree_fname_mask_base(
			std::string &fname_mask);
	void create_shallow_schreier_tree_fname_mask(
			std::string &fname, int node);
	void make_fname_candidates_file_default(
			std::string &fname, int level);
	void wedge_product_export_magma(
			int n, int q, int vector_space_dimension,
			int level, int verbose_level);
	void write_reps_csv(
			int lvl, int verbose_level);
	void write_reps_csv_fname(
			std::string &fname,
			int lvl, int verbose_level);
	void export_something(
			std::string &what, int data1,
			std::string &fname, int verbose_level);
	void export_something_worker(
			std::string &what, int data1,
			std::string &fname,
			int verbose_level);



	// in poset_classification_report.cpp:
	void report(
			poset_classification_report_options *Opt,
			int verbose_level);
	void report2(
			std::ostream &ost,
			poset_classification_report_options *Opt,
			int verbose_level);
	void report_orbits_in_detail(
			std::ostream &ost,
			poset_classification_report_options *Opt,
			int verbose_level);
	void report_number_of_orbits_at_level(
			std::ostream &ost,
			poset_classification_report_options *Opt,
			int verbose_level);
	void report_orbits_summary(
			std::ostream &ost,
			poset_classification_report_options *Opt,
			int verbose_level);
	void report_poset_of_orbits(
			std::ostream &ost,
			poset_classification_report_options *Opt,
			int verbose_level);
	void report_orbit(
			int level, int orbit_at_level,
			poset_classification_report_options *Opt,
			std::ostream &ost, int verbose_level);

	// poset_classification_trace.cpp:
	int find_isomorphism(
			long int *set1, long int *set2, int sz,
		int *transporter, int &orbit_idx,
		int verbose_level);
	void identify_and_get_stabilizer(
			long int *set, int sz, int *transporter,
			int &orbit_at_level,
			data_structures_groups::set_and_stabilizer
				*&Set_and_stab_original,
			data_structures_groups::set_and_stabilizer
				*&Set_and_stab_canonical,
			int verbose_level);
	void test_identify(
			int level, int nb_times, int verbose_level);
	void poset_classification_apply_isomorphism_no_transporter(
		int cur_level, int size, int cur_node, int cur_ex,
		long int *set_in, long int *set_out,
		int verbose_level);
	int poset_classification_apply_isomorphism(
			int level, int size,
		int current_node, int current_extension,
		long int *set_in, long int *set_out, long int *set_tmp,
		int *transporter_in, int *transporter_out,
		int f_tolerant,
		int verbose_level);
	int trace_set_recursion(
			int cur_level, int cur_node,
		int size, int level,
		long int *canonical_set,
		long int *tmp_set1, long int *tmp_set2,
		int *Elt_transporter, int *tmp_Elt1, int *tmp_Elt2,
		int f_tolerant,
		int verbose_level);
		// called by poset_classification::trace_set
		// returns the node in the poset_classification that corresponds
		// to the canonical_set
		// or -1 if f_tolerant and the node could not be found
	int trace_set(
			long int *set, int size, int level,
		long int *canonical_set, int *Elt_transporter,
		int verbose_level);
	// Elt_transporter maps the given set in set[]
	// to the canonical set in canonical_set[]
	// return value is the local orbit index of the canonical set
	long int find_node_for_subspace_by_rank(
			long int *set, int len,
		int verbose_level);

};


const char *trace_result_as_text(
		trace_result r);
int trace_result_is_no_result(
		trace_result r);



// #############################################################################
// poset_classification_report_options.cpp
// #############################################################################


//! to control the behavior of the poset classification report function


class poset_classification_report_options {

public:

	// TABLES/poset_classification_report_options.tex

	int f_select_orbits_by_level;
	int select_orbits_by_level_level;

	int f_select_orbits_by_stabilizer_order;
	int select_orbits_by_stabilizer_order_so;

	int f_select_orbits_by_stabilizer_order_multiple_of;
	int select_orbits_by_stabilizer_order_so_multiple_of;

	int f_include_projective_stabilizer;

	int f_draw_poset;

	int f_type_aux;
	int f_type_ordinary;
	int f_type_tree;
	int f_type_detailed;


	int f_fname;
	std::string fname;

	int f_draw_options;
	std::string draw_options_label;


	poset_classification_report_options();
	~poset_classification_report_options();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();
	int is_selected_by_group_order(
			long int so);

};




// #############################################################################
// poset_of_orbits.cpp
// #############################################################################

//! the data structure for the poset of orbits in the poset classification algorithm


class poset_of_orbits {

private:

	poset_classification *PC;

	int sz;
	int max_set_size;
	long int t0;

	long int nb_poset_orbit_nodes_used;
	long int nb_poset_orbit_nodes_allocated;
	long int poset_orbit_nodes_increment;
	long int poset_orbit_nodes_increment_last;

	poset_orbit_node *root;

	long int *first_poset_orbit_node_at_level;

	long int *nb_extension_nodes_at_level_total;
	long int *nb_extension_nodes_at_level;
	long int *nb_fusion_nodes_at_level;
	long int *nb_unprocessed_nodes_at_level;

public:
	long int *set0; // [max_set_size] temporary storage
	long int *set1; // [max_set_size] temporary storage
	long int *set3; // [max_set_size] temporary storage


	poset_of_orbits();
	~poset_of_orbits();
	void init(
			poset_classification *PC,
			int nb_poset_orbit_nodes,
			int sz, int max_set_size, long int t0,
			int verbose_level);
	void init_poset_orbit_node(
			int nb_poset_orbit_nodes,
			int verbose_level);
	void reallocate();
	void reallocate_to(
			long int new_number_of_nodes,
			int verbose_level);
	int get_max_set_size();
	long int get_nb_poset_orbit_nodes_allocated();
	long int get_nb_extension_nodes_at_level_total(
			int level);
	void set_nb_poset_orbit_nodes_used(
			int value);
	int first_node_at_level(
			int i);
	void set_first_node_at_level(
			int i, int value);
	poset_orbit_node *get_node(
			int node_idx);
	long int *get_set0();
	long int *get_set1();
	long int *get_set3();
	int nb_orbits_at_level(
			int level);
	long int nb_flag_orbits_up_at_level(
			int level);
	poset_orbit_node *get_node_ij(
			int level, int node);
	int node_get_nb_of_extensions(
			int node);
	void get_set(
			int node, long int *set, int &size);
	void get_set(
			int level, int orbit, long int *set, int &size);
	int find_extension_from_point(
			int node_idx,
			long int pt, int verbose_level);
	long int count_extension_nodes_at_level(
			int lvl);
	double level_progress(
			int lvl);
	void change_extension_type(
			int level,
			int node, int cur_ext, int type, int verbose_level);
	void get_table_of_nodes(
			long int *&Table,
			int &nb_rows, int &nb_cols, int verbose_level);
	int count_live_points(
			int level,
			int node_local, int verbose_level);
	void print_progress_by_level(
			int lvl);
	void print_tree();
	void init_root_node_from_base_case(
			int verbose_level);
	void init_root_node(
			int verbose_level);
	void make_tabe_of_nodes(
			int verbose_level);
	void poset_orbit_node_depth_breadth_perm_and_inverse(
		int max_depth,
		int *&perm, int *&perm_inv, int verbose_level);
	void read_memory_object(
			int &depth_completed,
			other::orbiter_kernel_system::memory_object *m,
			int &nb_group_elements,
			int verbose_level);
	void write_memory_object(
			int depth_completed,
			other::orbiter_kernel_system::memory_object *m,
			int &nb_group_elements,
			int verbose_level);
	long int calc_size_on_file(
			int depth_completed,
			int verbose_level);
	void read_sv_level_file_binary2(
		int level, std::ifstream &fp,
		int f_split, int split_mod, int split_case,
		int f_recreate_extensions, int f_dont_keep_sv,
		int verbose_level);
	void write_sv_level_file_binary2(
		int level, std::ofstream &fp,
		int f_split, int split_mod, int split_case,
		int verbose_level);
	void read_level_file_binary2(
		int level, std::ifstream &fp,
		int &nb_group_elements, int verbose_level);
	void write_level_file_binary2(
		int level, std::ofstream &fp,
		int &nb_group_elements, int verbose_level);
	void write_candidates_binary_using_sv(
			std::string &fname_base,
			int lvl, int t0, int verbose_level);
	void read_level_file(
			int level,
			std::string &fname, int verbose_level);
	void write_lvl_file_with_candidates(
			std::string &fname_base, int lvl, int t0,
			int verbose_level);
	void get_orbit_reps_at_level(
			int lvl, long int *&Data,
			int &nb_reps, int verbose_level);
	void write_orbit_reps_at_level(
			std::string &fname_base,
			int lvl,
			int verbose_level);
	void write_lvl_file(
			std::string &fname_base,
			int lvl, int t0, int f_with_stabilizer_generators,
			int f_long_version,
			int verbose_level);
	void write_lvl(
			std::ostream &f, int lvl, int t0,
			int f_with_stabilizer_generators, int f_long_version,
			int verbose_level);
	void log_nodes_for_treefile(
			int cur, int depth,
			std::ostream &f, int f_recurse, int verbose_level);
	void save_representatives_at_level_to_csv(
			std::string &fname,
			int lvl, int verbose_level);
	void get_set_orbits_at_level(
			int lvl, other::data_structures::set_of_sets *&SoS,
			int verbose_level);

};







// #############################################################################
// poset_orbit_node.cpp, poset_orbit_node_io.cpp, poset_orbit_node_upstep.cpp,
// poset_orbit_node_upstep_subspace_action.cpp,
// poset_orbit_node_downstep.cpp, poset_orbit_node_downstep_subspace_action.cpp:
// #############################################################################


//! to represent one poset orbit; related to the class poset_classification


class poset_orbit_node {

private:
	int node;
	int prev;
	
	long int pt;
	int nb_strong_generators;
	int first_strong_generator_handle;
		// only if there is a generator
	int *tl;
		// only if the group is not trivial
	
	int nb_extensions;
	extension *E;
	
	data_structures_groups::schreier_vector *Schreier_vector;
	
	actions::action *A_on_upset;
	// only used for actions on subspace lattices
	// it records the action on the factor space
	// modulo the current subspace
	// used in poset_orbit_node_downstep_subspace_action.cpp
	// and poset_orbit_node_upstep_subspace_action.cpp

public:

	// poset_orbit_node.cpp:
	poset_orbit_node();
	~poset_orbit_node();
	void null();
	void freeself();
	void init_root_node(
			poset_classification *gen, int verbose_level);
		// copies gen->SG0 and gen->tl into the poset_orbit_node 
		// structure using store_strong_generators
	void init_node(
			int node, int prev, long int pt,
			int verbose_level);
	int get_node();
	void set_node(
			int node);
	void delete_Schreier_vector();
	void allocate_E(
			int nb_extensions, int verbose_level);
	int get_level(
			poset_classification *gen);
	int get_node_in_level(
			poset_classification *gen);
	void get_strong_generators_handle(
			std::vector<int> &gen_hdl, int verbose_level);
	void get_tl(
			std::vector<int> &tl,
			poset_classification *PC, int verbose_level);
	int get_tl(
			int i);
	int has_Schreier_vector();
	data_structures_groups::schreier_vector
		*get_Schreier_vector();
	int get_nb_strong_generators();
	int *live_points();
	int get_nb_of_live_points();
	int get_nb_of_orbits_under_stabilizer();
	int get_nb_of_extensions();
	extension *get_E(
			int idx);
	long int get_pt();
	void set_pt(
			long int pt);
	int get_prev();
	void set_prev(
			int prev);
	void poset_orbit_node_depth_breadth_perm_and_inverse(
		poset_classification *gen,
		int max_depth, 
		int &idx, int hdl, int cur_depth,
		int *perm, int *perm_inv);
	int find_extension_from_point(
			poset_classification *gen, long int pt,
		int verbose_level);
	void print_extensions(
			std::ostream &ost);
	void log_current_node_without_group(
			poset_classification *gen,
		int s, std::ostream &f, int verbose_level);
	void log_current_node(
			poset_classification *gen,
			int s,
			std::ostream &f,
			int f_with_stabilizer_poset_classifications,
		int verbose_level);
	void log_current_node_after_applying_group_element(
		poset_classification *gen, int s,
		std::ostream &f, int hdl,
		int verbose_level);
	void log_current_node_with_candidates(
			poset_classification *gen,
		int lvl, std::ostream &f, int verbose_level);
	int depth_of_node(
			poset_classification *gen);
	void store_set(
			poset_classification *gen, int i);
		// stores a set of size i + 1 to gen->S
	void store_set_with_verbose_level(
			poset_classification *gen, int i,
		int verbose_level);
	// stores a set of size i + 1 to gen->S[]
	void store_set_to(
			poset_classification *gen, int i, long int *to);
	void store_set_to(
			poset_classification *gen, long int *to);
	int check_node_and_set_consistency(
			poset_classification *gen,
		int i, long int *set);
	void print_set_verbose(
			poset_classification *gen);
	void print_set(
			poset_classification *gen);
	void print_node(
			poset_classification *gen);
	void print_extensions(
			poset_classification *gen);
	void reconstruct_extensions_from_sv(
			poset_classification *gen,
		int verbose_level);
	int nb_extension_points();
		// sums up the lengths of orbits in all extensions

	// in poset_orbit_node_group_theory.cpp:
	void store_strong_generators(
			poset_classification *gen,
			groups::strong_generators *Strong_gens);
	void get_stabilizer_order(
			poset_classification *gen,
			algebra::ring_theory::longinteger_object &go);
	long int get_stabilizer_order_lint(
			poset_classification *PC);
	void get_stabilizer(
			poset_classification *PC,
			data_structures_groups::group_container &G,
			algebra::ring_theory::longinteger_object &go_G,
		int verbose_level);
	int test_if_stabilizer_is_trivial();
	void get_stabilizer_generators(
			poset_classification *PC,
			groups::strong_generators *&Strong_gens,
		int verbose_level);
	void init_extension_node_prepare_G(
		poset_classification *gen,
		int prev, int prev_ex, int size,
		data_structures_groups::group_container &G,
		algebra::ring_theory::longinteger_object &go_G,
		int verbose_level);
		// sets up the group G using the strong
		// poset_classifications that are stored
	void init_extension_node_prepare_H(
		poset_classification *gen,
		int prev, int prev_ex, int size,
		data_structures_groups::group_container &G,
		algebra::ring_theory::longinteger_object &go_G,
		data_structures_groups::group_container &H,
		algebra::ring_theory::longinteger_object &go_H,
		long int pt, int pt_orbit_len,
		int verbose_level);
		// sets up the group H which is the stabilizer
		// of the point pt in G
	void compute_point_stabilizer_in_subspace_setting(
		poset_classification *gen,
		int prev, int prev_ex, int size,
		data_structures_groups::group_container &G,
		algebra::ring_theory::longinteger_object &go_G,
		data_structures_groups::group_container &H,
		algebra::ring_theory::longinteger_object &go_H,
		long int pt, int pt_orbit_len,
		int verbose_level);
	void compute_point_stabilizer_in_standard_setting(
		poset_classification *gen,
		int prev, int prev_ex, int size,
		data_structures_groups::group_container &G,
		algebra::ring_theory::longinteger_object &go_G,
		data_structures_groups::group_container &H,
		int pt, int pt_orbit_len,
		int verbose_level);
	void create_schreier_vector_wrapper(
		poset_classification *gen,
		int f_create_schreier_vector,
		groups::schreier *Schreier, int verbose_level);
		// called from downstep_orbit_test_and_schreier_vector
		// calls Schreier.get_schreier_vector
	void create_schreier_vector_wrapper_subspace_action(
		poset_classification *gen,
		int f_create_schreier_vector,
		groups::schreier &Schreier,
		actions::action *A_factor_space,
		induced_actions::action_on_factor_space *AF,
		int verbose_level);


	// in poset_orbit_node_io.cpp:
	void read_memory_object(
		poset_classification *PC,
		actions::action *A,
		other::orbiter_kernel_system::memory_object *m,
		int &nb_group_elements,
		int *Elt_tmp,
		int verbose_level);
	void write_memory_object(
		poset_classification *PC,
		actions::action *A,
		other::orbiter_kernel_system::memory_object *m,
		int &nb_group_elements,
		int *Elt_tmp,
		int verbose_level);
	long int calc_size_on_file(
			actions::action *A, int verbose_level);
	void sv_read_file(
		poset_classification *PC,
		std::ifstream &fp, int verbose_level);
	void sv_write_file(
		poset_classification *PC,
		std::ofstream &fp, int verbose_level);
	void read_file(
			actions::action *A,
			std::ifstream &fp, int &nb_group_elements,
		int verbose_level);
	void write_file(
			actions::action *A,
			std::ofstream &fp, int &nb_group_elements,
		int verbose_level);
#if 0
	void save_schreier_forest(
		poset_classification *PC,
		groups::schreier *Schreier,
		int verbose_level);
	void save_shallow_schreier_forest(
		poset_classification *PC,
		int verbose_level);
	void draw_schreier_forest(
		poset_classification *PC,
		groups::schreier *Schreier,
		int f_using_invariant_subset, actions::action *AR,
		int verbose_level);
#endif

	// poset_orbit_node_upstep.cpp:
	int apply_isomorphism(
			poset_classification *gen,
		int lvl, int current_node, 
		int current_extension, int len, int f_tolerant, 
		int *Elt_tmp1, int *Elt_tmp2,
		int verbose_level);
		// returns next_node
	void install_fusion_node(
			poset_classification *gen,
		int lvl, int current_node, 
		int my_node, int my_extension, int my_coset, 
		long int pt0, int current_extension,
		int f_debug, int f_implicit_fusion, 
		int *Elt_tmp,
		int verbose_level);
		// Called from poset_orbit_node::handle_last_level
	int trace_next_point_wrapper(
			poset_classification *gen, int lvl,
		int current_node, 
		int len, int f_implicit_fusion,
		int *cosetrep,
		int &f_failure_to_find_point,
		int verbose_level);
		// Called from upstep_work::recognize_recursion
		// applies the permutation which maps the point with index lvl 
		// (i.e. the lvl+1-st point) to its orbit representative.
		// also maps all the other points under that permutation.
		// we are dealing with a set of size len + 1
		// returns false if we are using implicit fusion nodes 
		// and the set becomes lexicographically
		// less than before, in which case trace has to be restarted.
	int trace_next_point_in_place(
			poset_classification *gen,
		int lvl, int current_node, int size, 
		long int *cur_set, long int *tmp_set,
		int *cur_transporter, int *tmp_transporter, 
		int *cosetrep,
		int f_implicit_fusion, int &f_failure_to_find_point, 
		int verbose_level);
		// called by poset_classification::trace_set_recursion
	void trace_starter(
			poset_classification *gen, int size,
		long int *cur_set, long int *next_set,
		int *cur_transporter, int *next_transporter, 
		int verbose_level);
	int trace_next_point(
			poset_classification *gen,
		int lvl, int current_node, int size, 
		long int *cur_set, long int *next_set,
		int *cur_transporter, int *next_transporter, 
		int *cosetrep,
		int f_implicit_fusion, int &f_failure_to_find_point, 
		int verbose_level);
		// Called by poset_orbit_node::trace_next_point_wrapper 
		// and by poset_orbit_node::trace_next_point_in_place
		// returns false only if f_implicit_fusion is true and
		// the set becomes lexicographically less
	int orbit_representative_and_coset_rep_inv(
			poset_classification *gen,
		int lvl, long int pt_to_trace, long int &pt0, int *cosetrep,
		int verbose_level);
		// called by poset_orbit_node::trace_next_point
		// false means the point to trace was not found. 
		// This can happen if nodes were eliminated due to clique_test

	// poset_orbit_node_upstep_subspace_action.cpp:
	void orbit_representative_and_coset_rep_inv_subspace_action(
		poset_classification *gen, 
		int lvl, long int pt_to_trace, long int &pt0, int *cosetrep,
		int verbose_level);
		// called by poset_orbit_node::trace_next_point
		

	// poset_orbit_node_downstep.cpp
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
	void compute_schreier_vector(
			poset_classification *gen,
		int lvl, int verbose_level);
		// called from poset_classification::recreate_schreier_vectors_at_level
		// and from poset_classification::count_live_points
		// calls downstep_apply_early_test
		// and check_orbits
		// and Schreier.get_schreier_vector
	void get_candidates(
		poset_classification *gen,
		int lvl,
		long int *&candidates, int &nb_candidates,
		int verbose_level);

		// 1st level under downstep:
	void schreier_forest(
		poset_classification *gen,
		groups::schreier &Schreier, actions::action *&AR,
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
		poset_classification *gen,
		groups::schreier *Schreier, actions::action *AR,
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
		poset_classification *gen,
		groups::schreier &Schreier, actions::action *AR,
		int f_using_invariant_subset,
		int lvl, 
		int f_implicit_fusion, 
		int good_orbits1, int nb_points1, 
		int verbose_level);
		// called from downstep, 
		// once downstep_orbit_test_and_schreier_vector is done
		// calls test_orbits_for_implicit_fusion
	void find_extensions(
			poset_classification *gen,
			groups::schreier &O,
			actions::action *AR, int f_using_invariant_subset,
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
		int &n, long int *&subset,
		int verbose_level);
		// called from downstep_orbits
		// Gets the live points at the present node.
	void downstep_apply_early_test(
		poset_classification *gen, 
		int lvl, 
		int n, long int *subset,
		long int *candidates, int &nb_candidates,
		int verbose_level);
		// called from downstep_orbits, compute_schreier_vector  
		// calls the callback early test function if available
		// and calls test_point_using_check_functions otherwise

	void check_orbits_wrapper(
			poset_classification *gen,
			groups::schreier *Schreier, actions::action *AR,
		int lvl, 
		int &nb_good_orbits1, int &nb_points1, 
		int verbose_level);
		// called from downstep_orbit_test_and_schreier_vector
		// This function and create_schreier_vector_wrapper 
		// are used in pairs.
		// Except, the order in which the function is used matters.
		// Calls check_orbits

	void test_orbits_for_implicit_fusion(
			poset_classification *gen,
			groups::schreier &Schreier, actions::action *AR,
		int f_using_invariant_subset, 
		int lvl, int verbose_level);
		// called from downstep_implicit_fusion
		// eliminates implicit fusion orbits from the 
		// Schreier data structure, 
	void check_orbits(
			poset_classification *gen,
			groups::schreier *Schreier, actions::action *AR,
		int lvl, 
		int verbose_level);
		// called from compute_schreier_vector 
		// and check_orbits_wrapper (which is called from 
		// downstep_orbit_test_and_schreier_vector)
		// calls test_point_using_check_functions
		// eliminates bad orbits from the Schreier data structure, 
		// does not eliminate implicit fusion orbits
	int test_point_using_check_functions(
			poset_classification *gen,
		int lvl, int rep, int *the_set, 
		int verbose_level);
		// called by check_orbits and downstep_apply_early_test 
		// Calls gen->check_the_set_incrementally 
		// (if gen->f_candidate_incremental_check_func).
		// Otherwise, calls gen->check_the_set 
		// (if gen->f_candidate_check_func).
		// Otherwise accepts any point.
	void relabel_schreier_vector(
			actions::action *AR, int verbose_level);
		// called from compute_schreier_vector, 
		// downstep_orbit_test_and_schreier_vector
	void downstep_orbits_print(
			poset_classification *gen,
			groups::schreier *Schreier, actions::action *AR,
		int lvl, 
		int f_print_orbits,
		int max_orbits, int max_points_per_orbit);


	// poset_orbit_node_downstep_subspace_action.cpp
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
		induced_actions::action_on_factor_space &AF,
		int lvl, int verbose_level);
	void setup_factor_space_action_with_early_test(
		poset_classification *gen,
		induced_actions::action_on_factor_space *AF,
		actions::action *&A_factor_space,
		int lvl, int verbose_level);
	void setup_factor_space_action(
		poset_classification *gen,
		induced_actions::action_on_factor_space *AF,
		actions::action *&A_factor_space,
		int lvl, int f_compute_tables, 
		int verbose_level);
	void downstep_subspace_action_print_orbits(
		poset_classification *gen,
		groups::schreier &Schreier,
		int lvl, 
		int f_print_orbits, 
		int verbose_level);
	void downstep_orbits_subspace_action(
		poset_classification *gen, groups::schreier &Schreier,
		int lvl, 
		int f_use_invariant_subset_if_available, 
		int &f_using_invariant_subset, 
		int verbose_level);
	void find_extensions_subspace_action(
		poset_classification *gen, groups::schreier &O,
		actions::action *A_factor_space,
		induced_actions::action_on_factor_space *AF,
		int lvl, int f_implicit_fusion, int verbose_level);
};


// #############################################################################
// poset_with_group_action.cpp
// #############################################################################

//! a poset with a group action on it


class poset_with_group_action {
public:

	int f_subset_lattice;
	int n;

	int f_subspace_lattice;
	algebra::linear_algebra::vector_space *VS;

	actions::action *A; // the action in which the group is given
	actions::action *A2; // the action in which we do the search

	groups::strong_generators *Strong_gens;
	algebra::ring_theory::longinteger_object go;

	int f_has_orbit_based_testing;
	orbit_based_testing *Orbit_based_testing;

	int f_print_function;
	void (*print_function)(std::ostream &ost,
			int len, long int *S, void *data);
	void *print_function_data;

	poset_with_group_action();
	~poset_with_group_action();
	void init_subset_lattice(
			actions::action *A, actions::action *A2,
			groups::strong_generators *Strong_gens,
			int verbose_level);
	void init_subspace_lattice(
			actions::action *A, actions::action *A2,
			groups::strong_generators *Strong_gens,
			algebra::linear_algebra::vector_space *VS,
			int verbose_level);
	void add_independence_condition(
			int independence_value,
			int verbose_level);
	void add_testing(
			int (*func)(orbit_based_testing *Obt,
					long int *S, int len,
					void *data, int verbose_level),
			void *data,
			int verbose_level);
	void add_testing_without_group(
			void (*func)(long int *S, int len,
					long int *candidates, int nb_candidates,
					long int *good_candidates, int &nb_good_candidates,
					void *data, int verbose_level),
			void *data,
			int verbose_level);
	void print();
	void early_test_func(
		long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void unrank_point(
			int *v, long int rk);
	long int rank_point(
			int *v);
	void orbits_on_k_sets(
			poset_classification_control *Control,
			int k, long int *&orbit_reps,
			int &nb_orbits, int verbose_level);
	poset_classification *orbits_on_k_sets_compute(
			poset_classification_control *Control,
			int k, int verbose_level);
	void invoke_print_function(
			std::ostream &ost, int sz, long int *set);
};




// #############################################################################
// upstep_work.cpp
// #############################################################################


typedef struct coset_table_entry coset_table_entry;

//! auxiliary class to build a coset transversal for the automorphism group in the poset classification algorithm

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


//! auxiliary class for the poset classification algorithm to deal with flag orbits


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

	data_structures_groups::group_container *G;
	data_structures_groups::group_container *H;
	algebra::ring_theory::longinteger_object go_G, go_H;

	int coset;
	
	int nb_cosets;
	int nb_cosets_processed;
	coset_table_entry *coset_table;

	int *Elt1;
	int *Elt2;
	int *Elt3;


	upstep_work();
	~upstep_work();
	void init(
			poset_classification *gen,
		int size,
		int prev,
		int prev_ex,
		int cur,
		int f_debug,
		int f_implicit_fusion,
		int f_indicate_not_canonicals, 
		int verbose_level);
		// called from poset_classification::extend_node
	void handle_extension(
			int &nb_fuse_cur, int &nb_ext_cur,
		int verbose_level);
		// called from poset_classification::extend_node
		// Calls handle_extension_fusion_type 
		// or handle_extension_unprocessed_type
		//
		// Handles the extension 'cur_ex' in node 'prev'.
		// We are extending a set of size 'size' 
		// to a set of size 'size' + 1. 
		// Calls poset_orbit_node::init_extension_node for the 
		// new node that is (possibly) created
	void handle_extension_fusion_type(
			int verbose_level);
		// called from upstep_work::handle_extension
		// Handles the extension 'cur_ex' in node 'prev'.
	void handle_extension_unprocessed_type(
			int verbose_level);
		// called from upstep_work::handle_extension
		// calls init_extension_node
	int init_extension_node(
			int verbose_level);
		// Called from upstep_work::handle_extension_unprocessed_type
		// Calls upstep_subspace_action or upstep_for_sets, 
		// depending on the type of action
		// then changes the type of the extension to 
		// EXTENSION_TYPE_EXTENSION

		// Establishes a new node at depth 'size'
		// (i.e., a set of size 'size') as an extension 
		// of a previous node (prev) at depth size - 1 
		// with respect to a given point (pt).
		// This function is to be called for the next 
		// free poset_orbit_node node which will 
		// become the descendant of the previous node (prev).
		// the extension node corresponds to the point pt. 
		// returns false if the set is not canonical 
		// (provided f_indicate_not_canonicals is true)
	int upstep_for_sets(
			int verbose_level);
		// This routine is called from upstep_work::init_extension_node
		// It is testing a set of size 'size'. 
		// The newly added point is in gen->S[size - 1]
		// returns false if the set is not canonical 
		// (provided f_indicate_not_canonicals is true)
	//void print_level_extension_info_original_size();
	void print_level_extension_info();
	void print_level_extension_coset_info();

	// upstep_work_subspace_action.cpp:
	int upstep_subspace_action(
			int verbose_level);
		// This routine is called from upstep_work::init_extension_node
		// It computes coset_table.
		// It is testing a set of size 'size'. 
		// The newly added point is in gen->S[size - 1]
		// The extension is initiated from node 'prev' 
		// and from extension 'prev_ex' 
		// The node 'prev' is at depth 'size' - 1 
		// returns false if the set is not canonical 
		// (provided f_indicate_not_canonicals is true)


	// upstep_work_trace.cpp:

	trace_result recognize(
		int &final_node, int &final_ex, int f_tolerant, 
		int verbose_level);
	trace_result recognize_recursion(
		int lvl, int current_node, int &final_node, int &final_ex, 
		int f_tolerant,
		int verbose_level);
	trace_result handle_last_level(
		int lvl, int current_node, int current_extension, int pt0, 
		int &final_node, int &final_ex,  
		int verbose_level);
	trace_result start_over(
		int lvl, int current_node, 
		int &final_node, int &final_ex, 
		int f_tolerant, int verbose_level);
};


}}}


#endif /* ORBITER_SRC_LIB_CLASSIFICATION_POSET_CLASSIFICATION_POSET_CLASSIFICATION_H_ */



