/*
 * tl_combinatorics.h
 *
 *  Created on: Oct 27, 2019
 *      Author: betten
 */

#ifndef ORBITER_SRC_LIB_TOP_LEVEL_COMBINATORICS_TL_COMBINATORICS_H_
#define ORBITER_SRC_LIB_TOP_LEVEL_COMBINATORICS_TL_COMBINATORICS_H_


namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {


// #############################################################################
// boolean_function_classify.cpp
// #############################################################################


//! classification of boolean functions


class boolean_function_classify {

public:

	combinatorics::boolean_function_domain *BF;

	// group stuff:
	actions::action *A;
	//data_structures_groups::vector_ge *nice_gens;

	induced_actions::action_on_homogeneous_polynomials *AonHPD;
	groups::strong_generators *SG;
	ring_theory::longinteger_object go;

	actions::action *A_affine;
		// restricted action on affine points

	int nb_sol; // before isomorphism test
	int nb_orbits; // after isomorphism test
	std::vector<int> orbit_first;
	std::vector<int> orbit_length;
	std::vector<std::vector<int> > Bent_function_table;
	std::vector<std::vector<int> > Equation_table;

	std::multimap<uint32_t, int> Hashing;
		// we store the pair (hash, idx)
		// where hash is the hash value of the set and idx is the
		// index in the table Sets where the set is stored.
		//
		// we use a multimap because the hash values are not unique
		// it happens that two sets have the same hash value.
		// map cannot handle that.

	std::vector<void *> Orb_vector; // [nb_orbits]
	std::vector<void *> Stab_gens_vector; // [nb_orbits]

	//orbits_schreier::orbit_of_equations *Orb;
	//groups::strong_generators *Stab_gens;


	boolean_function_classify();
	~boolean_function_classify();

	void init_group(
			combinatorics::boolean_function_domain *BF,
			actions::action *A,
			int verbose_level);
	void search_for_bent_functions(
			int verbose_level);
	void print();
	void print_orbits_sorted();
	void print_orbit_reps_with_minimum_weight();
	void export_orbit(
			int idx,
			data_structures::int_matrix *&M,
			int verbose_level);

};








// #############################################################################
// combinatorial_object_activity_description.cpp
// #############################################################################


//! description of an activity for a combinatorial object

class combinatorial_object_activity_description {
public:


	// options that apply to GOC = geometric_object_create

	int f_save;

	int f_save_as;
	std::string save_as_fname;

	int f_extract_subset;
	std::string extract_subset_set;
	std::string extract_subset_fname;

	int f_line_type_old;

	int f_line_type;

	int f_conic_type;
	int conic_type_threshold;

	int f_non_conical_type;

	int f_ideal;
	std::string ideal_ring_label;


	// options that apply to IS = data_input_stream

	int f_canonical_form_PG;
	std::string canonical_form_PG_PG_label;
	int f_canonical_form_PG_has_PA;
	projective_geometry::projective_space_with_action
		*Canonical_form_PG_PA;
	canonical_form_classification::classification_of_objects_description
		*Canonical_form_PG_Descr;

	int f_canonical_form;
	canonical_form_classification::classification_of_objects_description
		*Canonical_form_Descr;

	int f_report;
	canonical_form_classification::classification_of_objects_report_options
		*Classification_of_objects_report_options;

	int f_draw_incidence_matrices;
	std::string draw_incidence_matrices_prefix;

	int f_test_distinguishing_property;
	std::string test_distinguishing_property_graph;

	int f_covering_type;
	std::string covering_type_orbits;
	int covering_type_size;

	int f_filter_by_Steiner_property;

	int f_compute_frequency;
	std::string compute_frequency_graph;

	int f_unpack_from_restricted_action;
	std::string unpack_from_restricted_action_prefix;
	std::string unpack_from_restricted_action_group_label;

	int f_line_covering_type;
	std::string line_covering_type_prefix;
	std::string line_covering_type_projective_space;
	std::string line_covering_type_lines;

	int f_activity;
	user_interface::activity_description *Activity_description;

	combinatorial_object_activity_description();
	~combinatorial_object_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};

// #############################################################################
// combinatorial_object_activity.cpp
// #############################################################################


//! perform an activity for a combinatorial object

class combinatorial_object_activity {
public:
	combinatorial_object_activity_description *Descr;

	int f_has_geometric_object;
	geometry::geometric_object_create *GOC;

	int f_has_combo;
	apps_combinatorics::combinatorial_object *Combo;


	combinatorial_object_activity();
	~combinatorial_object_activity();
	void init(
			combinatorial_object_activity_description *Descr,
			geometry::geometric_object_create *GOC,
			int verbose_level);
	void init_combo(
			combinatorial_object_activity_description *Descr,
			apps_combinatorics::combinatorial_object *Combo,
			int verbose_level);
	void perform_activity(
			orbiter_kernel_system::activity_output *&AO,
			int verbose_level);
	void perform_activity_geometric_object(
			int verbose_level);
	void perform_activity_combo(
			orbiter_kernel_system::activity_output *&AO,
			int verbose_level);

};




// #############################################################################
// combinatorial_object.cpp
// #############################################################################


//! a list of combinatorial objects all of the same type

class combinatorial_object {
public:

	canonical_form_classification::data_input_stream *IS;

	canonical_form_classification::classification_of_objects *Classification;
	canonical_form::classification_of_combinatorial_objects *Classification_CO;

	combinatorial_object();
	~combinatorial_object();
	void init(
			canonical_form_classification::data_input_stream_description
					*Data_input_stream_description,
			int verbose_level);
	void do_canonical_form_PG(
			projective_geometry::projective_space_with_action *PA,
			canonical_form_classification::classification_of_objects_description
					*Canonical_form_PG_Descr,
			int verbose_level);
	void do_canonical_form_not_PG(
			canonical_form_classification::classification_of_objects_description
				*Canonical_form_Descr,
			int verbose_level);
	void do_test_distinguishing_property(
			graph_theory::colored_graph *CG,
			int verbose_level);
	void do_covering_type(
			orbits::orbits_create *Orb,
			int sz,
			int f_filter_by_Steiner_property,
			orbiter_kernel_system::activity_output *&AO,
			int verbose_level);
	void do_compute_frequency_graph(
			graph_theory::colored_graph *CG,
			int verbose_level);
	void do_compute_ideal(
			ring_theory::homogeneous_polynomial_domain *HPD,
			int verbose_level);
	void do_save(
			std::string &save_as_fname,
			int f_extract,
			long int *extract_idx_set, int extract_size,
			int verbose_level);
	void draw_incidence_matrices(
			std::string &prefix,
			int verbose_level);
	void unpack_from_restricted_action(
			std::string &prefix,
			apps_algebra::any_group *G,
			int verbose_level);
	void line_covering_type(
			std::string &prefix,
			projective_geometry::projective_space_with_action *PA,
			std::string &lines,
			int verbose_level);
	void line_type(
			std::string &prefix,
			projective_geometry::projective_space_with_action *PA,
			int verbose_level);
	void do_activity(
			user_interface::activity_description *Activity_description,
			orbiter_kernel_system::activity_output *&AO,
			int verbose_level);
	void do_graph_theoretic_activity(
			apps_graph_theory::graph_theoretic_activity_description
					*Graph_theoretic_activity_description,
			orbiter_kernel_system::activity_output *&AO,
			int verbose_level);

};


// #############################################################################
// combinatorics_global.cpp
// #############################################################################


//! combinatorics stuff


class combinatorics_global {

public:

	combinatorics_global();
	~combinatorics_global();
	void create_design_table(
			design_create *DC,
			std::string &problem_label,
			design_tables *&T,
			groups::strong_generators *Gens,
			int verbose_level);
	void load_design_table(
			design_create *DC,
			std::string &problem_label,
			design_tables *&T,
			groups::strong_generators *Gens,
			int verbose_level);
	void span_base_blocks(
			apps_algebra::any_group *AG,
			data_structures::set_of_sets *SoS_base_blocks,
			int *Base_block_selection, int nb_base_blocks,
			int &v, int &b, int &k,
			long int *&Blocks,
			int verbose_level);
	// Blocks[b * k]

#if 0
	void Hill_cap56(
			std::string &fname, int &nb_Pts, long int *&Pts,
		int verbose_level);
	void append_orbit_and_adjust_size(
			groups::schreier *Orb,
			int idx, int *set, int &sz);
#endif

};


// #############################################################################
// dd_lifting.cpp
// #############################################################################


//! search for Delandtsheer-Doyen designs, lifting of starter configurations


class dd_lifting {

public:

	delandtsheer_doyen *DD;

	int target_depth;
	int level;

	std::string starter_file;

	long int *Nb_sol;
	long int *Nb_nodes;
	int *Orbit_idx;
	int nb_orbits_not_ruled_out;

	long int nb_sol_total;
	long int nb_nodes_total;


	orbiter_kernel_system::orbiter_data_file *ODF;


	dd_lifting();
	~dd_lifting();
	void perform_lifting(
			delandtsheer_doyen *DD,
			int verbose_level);
	void search_case_singletons_and_count(
			int orbit_idx, long int &nb_sol, long int &nb_nodes,
			int verbose_level);
	void search_case_singletons(
			int orbit_idx, long int &nb_sol, long int &nb_nodes,
			int verbose_level);

};


// #############################################################################
// dd_search_singletons.cpp
// #############################################################################


//! search for Delandtsheer-Doyen designs, lifting of one specific case


class dd_search_singletons {

public:

	dd_lifting *DD_lifting;
	delandtsheer_doyen *DD;


	int orbit_idx;
	int target_depth;
	int level;
	data_structures::set_of_sets *Live_points;
	long int *chosen_set; // [target_depth]
	int *index; // [target_depth]
	long int nb_nodes;
	std::vector<std::vector<int> > Solutions;

	dd_search_singletons();
	~dd_search_singletons();
	void search_case_singletons(
			dd_lifting *DD_lifting,
			int orbit_idx, int verbose_level);
	void search_case_singletons_recursion(
			int level,
			int verbose_level);


};


// #############################################################################
// delandtsheer_doyen_description.cpp
// #############################################################################

#define MAX_MASK_TESTS 1000


//! description of the problem for delandtsheer_doyen


class delandtsheer_doyen_description {
public:

	int f_d1;
	int d1;

	int f_d2;
	int d2;

	int f_q1;
	int q1;

	int f_q2;
	int q2;

	int f_group_label;
	std::string group_label;

	int f_mask_label;
	std::string mask_label;

	int f_problem_label;
	std::string problem_label;



	int DELANDTSHEER_DOYEN_X;
	int DELANDTSHEER_DOYEN_Y;

	int f_K;
	int K;

	int f_pair_search_control;
	std::string pair_search_control_label;

	int f_search_control;
	std::string search_control_label;

	// row intersection type
	int f_R;
	int nb_row_types;
	int *row_type; // [nb_row_types + 1]

	// col intersection type
	int f_C;
	int nb_col_types;
	int *col_type; // [nb_col_types + 1]


	int f_nb_orbits_on_blocks;
	int nb_orbits_on_blocks;


	// mask related test:
	int nb_mask_tests;
	int mask_test_level[MAX_MASK_TESTS];
	int mask_test_who[MAX_MASK_TESTS];
		// 1 = x
		// 2 = y
		// 3 = x+y
		// 4 = singletons
	int mask_test_what[MAX_MASK_TESTS];
		// 1 = eq
		// 2 = ge
		// 3 = le
	int mask_test_value[MAX_MASK_TESTS];

	int f_create_starter;

	int f_create_graphs;

	int f_singletons;
	int singletons_starter_size;

	int f_subgroup;
	std::string subgroup_gens;
	std::string subgroup_order;

	int f_search_wrt_subgroup;


	delandtsheer_doyen_description();
	~delandtsheer_doyen_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// delandtsheer_doyen.cpp
// #############################################################################



//! search for line transitive point imprimitive linear spaces as described by Delandtsheer and Doyen



class delandtsheer_doyen {
public:

	delandtsheer_doyen_description *Descr;

	field_theory::finite_field *F1;
	field_theory::finite_field *F2;

	std::string label;

	int Xsize; // = D = q1 = # of rows
	int Ysize; // = C = q2 = # of cols

	int V; // = Xsize * Ysize
	int b;
	long int *line;        // [K];
	int *row_sum; // [Xsize]
	int *col_sum; // [Ysize]

	poset_classification::poset_classification_control *Search_control;

	algebra::matrix_group *M1;
	algebra::matrix_group *M2;
	actions::action *A1;
	actions::action *A2;

	actions::action *A;
	actions::action *A0;

	groups::strong_generators *SG;
	ring_theory::longinteger_object go;
	group_constructions::direct_product *P;

	poset_classification::poset_classification *Gen;
	poset_classification::poset_with_group_action *Poset_search;


	orbits::orbits_on_pairs *Orbits_on_pairs;
		// Orbits_on_pairs->nb_orbits
		// int *pair_orbit; // [V * V]
		// int nb_orbits;
		// int *orbit_length; // [nb_orbits]


	dd_lifting *DD_Lifting;

	int *orbit_covered; // [nb_orbits]
	int *orbit_covered2; // [nb_orbits]
	int *orbit_covered_max; // [nb_orbits]
		// orbit_covered_max[i] = orbit_length[i] / b;
	int *orbits_covered; // [K * K]


	// intersection type tests:

	int inner_pairs_in_rows;
	int inner_pairs_in_cols;

	// row intersection type
	int *row_type_cur; // [nb_row_types + 1]
	int *row_type_this_or_bigger; // [nb_row_types + 1]

	// col intersection type
	int *col_type_cur; // [nb_col_types + 1]
	int *col_type_this_or_bigger; // [nb_col_types + 1]



	// for testing the mask:
	int *f_row_used; // [Xsize];
	int *f_col_used; // [Ysize];
	int *row_idx; // [Xsize];
	int *col_idx; // [Ysize];
	int *singletons; // [K];

	// temporary data
	int *row_col_idx; // [Xsize];
	int *col_row_idx; // [Ysize];

	long int *live_points; // [V]
	int nb_live_points;

	delandtsheer_doyen();
	~delandtsheer_doyen();
	void init(
			delandtsheer_doyen_description *Descr,
			int verbose_level);
	void show_generators(
			int verbose_level);
	void search_singletons(
			int verbose_level);
	int try_to_increase_orbit_covering_based_on_two_sets(
			long int *pts1, int sz1,
			long int *pts2, int sz2,
			long int pt0);
	void increase_orbit_covering_firm(
			long int *pts, int sz, long int pt0);
	// firm means that an excess in the orbit covering raises an error
	void decrease_orbit_covering(
			long int *pts, int sz, long int pt0);
	void create_starter(
			int verbose_level);
	void create_graphs(
			int verbose_level);
	void create_graph(
			int case_number, long int *line, int s, int s2, int *Covered_orbits,
			int &nb_live_points,
			std::string &fname,
			int verbose_level);
	void setup_orbit_covering(
			int verbose_level);
	groups::strong_generators *scan_subgroup_generators(
			int verbose_level);
	void create_product_action_of_monomial_groups(
			int verbose_level);
	void create_action(
			int verbose_level);
	void compute_live_points_for_singleton_search(
			long int *line0, int len, int verbose_level);
	void print_mask_test_i(
			std::ostream &ost, int i);
	void early_test_func(
			long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	int check_conditions(
			long int *S, int len, int verbose_level);
	int check_orbit_covering(
			long int *line, int len, int verbose_level);
	int check_row_sums(
			long int *line, int len, int verbose_level);
	int check_col_sums(
			long int *line, int len, int verbose_level);
	int check_mask(
			long int *line, int len, int verbose_level);
	void get_mask_core_and_singletons(
		long int *line, int len,
		int &nb_rows_used, int &nb_cols_used,
		int &nb_singletons, int verbose_level);
};




// #############################################################################
// design_activity_description.cpp
// #############################################################################

//! to describe an activity for a design



class design_activity_description {

public:


	int f_load_table;
	std::string load_table_label;
	std::string load_table_group;


	std::string load_table_H_label;
	std::string load_table_H_group_order;
	std::string load_table_H_gens;
	int load_table_selected_orbit_length;


	int f_canonical_form;
	canonical_form_classification::classification_of_objects_description
		*Canonical_form_Descr;

	int f_extract_solutions_by_index_csv;
	int f_extract_solutions_by_index_txt;
	std::string extract_solutions_by_index_label;
	std::string extract_solutions_by_index_group;
	std::string extract_solutions_by_index_fname_solutions_in;
	std::string extract_solutions_by_index_fname_solutions_out;
	std::string extract_solutions_by_index_prefix;

	int f_export_inc;
	int f_export_incidence_matrix;
	int f_export_incidence_matrix_latex;
	int f_intersection_matrix;
	int f_export_blocks;
	int f_row_sums;
	int f_tactical_decomposition;

	design_activity_description();
	~design_activity_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// design_activity.cpp
// #############################################################################

//! an activity for a design



class design_activity {

public:
	design_activity_description *Descr;


	design_activity();
	~design_activity();
	void perform_activity(
			design_activity_description *Descr,
			design_create *DC, int verbose_level);
	void do_extract_solutions_by_index(
			design_create *DC,
			std::string &label,
			std::string &group_label,
			std::string &fname_in,
			std::string &fname_out,
			std::string &prefix_text,
			int f_csv_format,
			int verbose_level);
	void do_create_table(
			design_create *DC,
			std::string &label,
			std::string &group_label,
			int verbose_level);
	void do_load_table(
			design_create *DC,
			std::string &label,
			std::string &group_label,
			std::string &H_label,
			std::string &H_go_text,
			std::string &H_generators_data,
			int selected_orbit_length,
			int verbose_level);
	void do_canonical_form(
			canonical_form_classification::classification_of_objects_description
				*Canonical_form_Descr,
			int verbose_level);
	void do_export_inc(
			design_create *DC,
			int verbose_level);
	void do_export_incidence_matrix_csv(
			design_create *DC,
			int verbose_level);
	void do_export_incidence_matrix_latex(
			design_create *DC,
			int verbose_level);
	void do_intersection_matrix(
			design_create *DC,
			int f_save,
			int verbose_level);
	void do_export_blocks(
			design_create *DC,
			int verbose_level);
	void do_row_sums(
			design_create *DC,
			int verbose_level);
	void do_tactical_decomposition(
			design_create *DC,
			int verbose_level);

};



// #############################################################################
// design_create_description.cpp
// #############################################################################

//! to describe the construction of a known design from the command line



class design_create_description {

public:

	int f_label;
	std::string label_txt;
	std::string label_tex;

	int f_field;
	std::string field_label;

	int f_catalogue;
	int iso;

	int f_family;
	std::string family_name;

	int f_list_of_base_blocks;
	std::string list_of_base_blocks_group_label;
	std::string list_of_base_blocks_fname;
	std::string list_of_base_blocks_col;
	std::string list_of_base_blocks_selection_fname;
	std::string list_of_base_blocks_selection_col;
	int list_of_base_blocks_selection_idx;

	int f_list_of_blocks_coded;
	int list_of_blocks_coded_v;
	int list_of_blocks_coded_k;
	std::string list_of_blocks_coded_label;

	int f_list_of_sets_coded;
	int list_of_sets_coded_v;
	std::string list_of_sets_coded_label;

	int f_list_of_blocks_coded_from_file;
	std::string list_of_blocks_coded_from_file_fname;

	int f_list_of_blocks_from_file;
	int list_of_blocks_from_file_v;
	std::string list_of_blocks_from_file_fname;

	int f_wreath_product_designs;
	int wreath_product_designs_n;
	int wreath_product_designs_k;

	int f_linear_space_from_latin_square;
	std::string linear_space_from_latin_square_name;

	int f_no_group;


	design_create_description();
	~design_create_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// design_create.cpp
// #############################################################################

//! to create a known design using a description from class design_create_description



class design_create {

public:
	design_create_description *Descr;

	std::string prefix;
	std::string label_txt;
	std::string label_tex;

	int q;
	field_theory::finite_field *F;

	int k;

	actions::action *A;
		// Sym(degree)
	actions::action *A2;
		// Sym(degree), in the action on k-subsets


	actions::action *Aut;
	// PGGL(3,q) in case of PG_2_q with q not prime
	// PGL(3,q) in case of PG_2_q with q prime
	actions::action *Aut_on_lines; // Aut induced on lines

	int degree;

	int f_has_set;
	long int *set; // [sz]
		// The subsets are coded as ranks of k-subsets.
	int sz; // = b, the number of blocks

	int f_has_group;
	groups::strong_generators *Sg;


	projective_geometry::projective_space_with_action *PA;
	geometry::projective_space *P;

	int *block; // [k]

	int v;
	int b;
	int nb_inc;
	int f_has_incma;
	int *incma; // [v * b]

	design_create();
	~design_create();
	void init(
			apps_combinatorics::design_create_description *Descr,
			int verbose_level);
	void create_design_PG_2_q(
			field_theory::finite_field *F,
			long int *&set, int &sz, int &k,
			int verbose_level);
	// creates a projective_space_with_action object
	void unrank_block_in_PG_2_q(
			int *block,
			int rk, int verbose_level);
	int rank_block_in_PG_2_q(
			int *block,
			int verbose_level);
	int get_nb_colors_as_two_design(
			int verbose_level);
	int get_color_as_two_design_assume_sorted(
			long int *design, int verbose_level);
	void compute_incidence_matrix_from_set_of_codes(
			int verbose_level);
	void compute_incidence_matrix_from_blocks(
			int *blocks, int nb_blocks, int k, int verbose_level);

};


// #############################################################################
// design_tables.cpp
// #############################################################################

//! a set of designs to be used for a large set


class design_tables {

public:

	actions::action *A;
	actions::action *A2;
	long int *initial_set;
	int design_size;
	std::string label;
	std::string fname_design_table;
	groups::strong_generators *Strong_generators;

	int nb_designs;
	long int *the_table; // [nb_designs * design_size]


	design_tables();
	~design_tables();
	void init(
			actions::action *A,
			actions::action *A2,
			long int *initial_set, int design_size,
			std::string &label,
			groups::strong_generators *Strong_generators,
			int verbose_level);
	void create_table(
			int verbose_level);
	void create_action(
			actions::action *&A_on_designs,
			int verbose_level);
	void extract_solutions_by_index(
			int nb_sol, int Index_width, int *Index,
			std::string &output_fname_csv,
			int verbose_level);
	void make_reduced_design_table(
			long int *set, int set_sz,
			long int *&reduced_table,
			long int *&reduced_table_idx,
			int &nb_reduced_designs,
			int verbose_level);
	void init_from_file(
			actions::action *A,
			actions::action *A2,
			long int *initial_set, int design_size,
			std::string &label,
			groups::strong_generators *Strong_generators,
			int verbose_level);
	int test_if_table_exists(
			std::string &label,
			int verbose_level);
	void save(
			int verbose_level);
	void load(
			int verbose_level);
	int test_if_designs_are_disjoint(
			int i, int j);
	int test_set_within_itself(
			long int *set_of_designs_by_index,
			int set_size);
	int test_between_two_sets(
			long int *set_of_designs_by_index1, int set_size1,
			long int *set_of_designs_by_index2, int set_size2);

};




// #############################################################################
// difference_set_in_heisenberg_group.cpp
// #############################################################################


//! to find difference sets in Heisenberg groups following Tao


class difference_set_in_heisenberg_group {

public:
	std::string fname_base;

	int n;
	int q;
	field_theory::finite_field *F;
	algebra::heisenberg *H;
	int *Table;
	int *Table_abv;
	int *gens;
	int nb_gens;

	actions::action *A;
	groups::strong_generators *Aut_gens;
	ring_theory::longinteger_object Aut_order;

	int given_base_length; // = nb_gens
	long int *given_base; // = gens
	int *base_image;
	int *base_image_elts;
	int *E1;
	int rk_E1;

	std::string prefix;
	std::string fname_magma_out;
	groups::sims *Aut;
	groups::sims *U;
	ring_theory::longinteger_object U_go;
	data_structures_groups::vector_ge *U_gens;
	groups::schreier *Sch;


	// N = normalizer of U in Aut
	int *N_gens;
	int N_nb_gens, N_go;
	actions::action *N;
	ring_theory::longinteger_object N_order;

	actions::action *N_on_orbits;
	int *Paired_with;
	int nb_paired_orbits;
	long int *Pairs;
	int *Pair_orbit_length;


	int *Pairs_of_type1;
	int nb_pairs_of_type1;
	int *Pairs_of_type2;
	int nb_pairs_of_type2;
	int *Sets1;
	int *Sets2;


	long int *Short_pairs;
	long int *Long_pairs;

	int *f_orbit_select;
	int *Short_orbit_inverse;

	actions::action *A_on_short_orbits;
	int nb_short_orbits;
	int nb_long_orbits;

	poset_classification::poset_classification *gen;




	void init(
			int n,
			field_theory::finite_field *F, int verbose_level);
	void do_n2q3(
			int verbose_level);
	void check_overgroups_of_order_nine(
			int verbose_level);
	void create_minimal_overgroups(
			int verbose_level);
	void early_test_func(
			long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);

};




// #############################################################################
// hadamard_classify.cpp
// #############################################################################

//! classification of Hadamard matrices




class hadamard_classify {

public:
	int n;
	int N, N2;
	data_structures::bitvector *Bitvec;
	graph_theory::colored_graph *CG;

	actions::action *A;

	int *v;

	poset_classification::poset_classification *gen;
	int nb_orbits;

	void init(
			int n,
			int f_draw, int verbose_level,
			int verbose_level_clique);
	int clique_test(
			long int *set, int sz);
	void early_test_func(
			long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	int dot_product(
			int a, int b, int n);
};





// #############################################################################
// hall_system_classify.cpp
// #############################################################################

//! classification of Hall systems


class hall_system_classify {
public:
	//int e;
	int n; // 3^e
	int nm1; // n-1
		// number of points different from the reflection point
	int nb_pairs;
		// nm1 / 2
		// = number of lines (=triples) through the reflection point
		// = number of lines (=triples) through any point
	int nb_pairs2; // = nm1 choose 2
		// number of pairs of points
		// different from the reflection point.
		// these are the pairs of points that are covered by the
		// triples that we will choose.
		// The other pairs have been covered by the lines through
		// the reflection point,
		// so they are fine because we assume that
		// these lines exist.
	int nb_blocks_overall; // {n \choose 2} / 6
	int nb_blocks_needed; // nb_blocks_overall - (n - 1) / 2
	int nb_orbits_needed; // nb_blocks_needed / 2
	int depth;
	int N;
		// {nb_pairs choose 3} * 8
		// {nb_pairs choose 3}
		// counts the number of ways to choose three lines
		// through the reflection point.
		// the times 8 is because every triple of lines through the
		// reflection point has 2^3 ways
		// of choosing one point on each line.
	int N0; // {nb_pairs choose 3} * 4
	int *row_sum; // [nm1]
		// this is where we test whether each of the
		// points different from the reflection point lies on
		// the right number of triples.
		// The right number is nb_pairs
	int *pair_covering; // [nb_pairs2]
		// this is where we test whether each of the
		// pairs of points not including the reflection point
		// is covered once


	long int *triples; // [N0 * 6]
		// a table of all triples so that
		// we can induce the group action on to them.


	actions::action *A;
		// The symmetric group on nm1 points.
	actions::action *A_on_triples;
		// the induced action on unordered triples
		// as stored in triples[].
	groups::strong_generators *Strong_gens_Hall_reflection;
		// the involution which switches the
		// points on every line through
		// the center (other than the center).
	groups::strong_generators *Strong_gens_normalizer;
		// Strong generators for the normalizer
		// of the involution.
	groups::sims *S;
		// The normalizer of the involution

	std::string prefix;
	std::string fname_orbits_on_triples;
	groups::schreier *Orbits_on_triples;
		// Orbits of the reflection group on triples.
	actions::action *A_on_orbits;
		// Induced action of A_on_triples
		// on the orbit of the reflection group.
	int f_play_it_safe;

	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Poset;
		// subset lattice for action A_on_orbits
	poset_classification::poset_classification *PC;
		// Classification of subsets in the action A_on_orbits


	hall_system_classify();
	~hall_system_classify();
	void init(
			int argc, const char **argv,
			int n, int depth,
			int verbose_level);
	void orbits_on_triples(
			int verbose_level);
	void print(
			std::ostream &ost, long int *S, int len);
	void unrank_triple(
			long int *T, int rk);
	void unrank_triple_pair(
			long int *T1, long int *T2, int rk);
	void early_test_func(
			long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
};







// #############################################################################
// large_set_activity_description.cpp
// #############################################################################

//! description of an activity for a spread table


class large_set_activity_description {
public:



	large_set_activity_description();
	~large_set_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);

};


// #############################################################################
// large_set_activity.cpp
// #############################################################################

//! an activity for a spread table


class large_set_activity {
public:

	large_set_activity_description *Descr;
	large_set_was *LSW;



	large_set_activity();
	~large_set_activity();
	void perform_activity(
			large_set_activity_description *Descr,
			large_set_was *LSW, int verbose_level);

};



// #############################################################################
// large_set_classify.cpp
// #############################################################################

//! classification of large sets of designs

class large_set_classify {
public:
	design_create *DC;
	int design_size;
		// = DC->sz = b,
		// the number of blocks in the design
	int nb_points; // = DC->A->degree
	int nb_lines; // = DC->A2->degree
	int search_depth;

	std::string problem_label;

	int f_lexorder_test;
	int size_of_large_set; // = nb_lines / design_size


	design_tables *Design_table;

	int nb_colors;
		// = DC->get_nb_colors_as_two_design
	int *design_color_table; // [nb_designs]

	actions::action *A_on_designs;


	data_structures::bitvector *Bitvec;
	int *degree;

	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification *gen;

	int nb_needed;




	large_set_classify();
	~large_set_classify();
	void init(
			design_create *DC,
			design_tables *T,
			int verbose_level);
	void create_action_and_poset(
			int verbose_level);
	void compute(
			int verbose_level);
	void read_classification(
			data_structures_groups::orbit_transversal *&T,
			int level, int verbose_level);
	void read_classification_single_case(
			data_structures_groups::set_and_stabilizer *&Rep,
			int level, int case_nr, int verbose_level);
	void compute_colors(
			design_tables *Design_table, int *&design_color_table,
			int verbose_level);
	int test_if_designs_are_disjoint(
			int i, int j);

};








// #############################################################################
// large_set_was_activity_description.cpp
// #############################################################################

//! description of an activity for a large set search with assumed symmetry


class large_set_was_activity_description {
public:

	int f_normalizer_on_orbits_of_a_given_length;
	int normalizer_on_orbits_of_a_given_length_length;
	int normalizer_on_orbits_of_a_given_length_nb_orbits;
	poset_classification::poset_classification_control
		*normalizer_on_orbits_of_a_given_length_control;

	int f_create_graph_on_orbits_of_length;
	std::string create_graph_on_orbits_of_length_fname;
	int create_graph_on_orbits_of_length_length;

	int f_create_graph_on_orbits_of_length_based_on_N_orbits;
	std::string create_graph_on_orbits_of_length_based_on_N_orbits_fname_mask;
	int create_graph_on_orbits_of_length_based_on_N_orbits_length;
	int create_graph_on_orbits_of_length_based_on_N_nb_N_orbits_preselected;
	int create_graph_on_orbits_of_length_based_on_N_orbits_r;
	int create_graph_on_orbits_of_length_based_on_N_orbits_m;

	int f_read_solution_file;
	int read_solution_file_orbit_length;
	std::string read_solution_file_name;



	large_set_was_activity_description();
	~large_set_was_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// large_set_was_activity.cpp
// #############################################################################

//! an activity for a large set search with assumed symmetry


class large_set_was_activity {
public:

	large_set_was_activity_description *Descr;
	large_set_was *LSW;


	large_set_was_activity();
	~large_set_was_activity();
	void perform_activity(
			large_set_was_activity_description *Descr,
			large_set_was *LSW, int verbose_level);
	void do_normalizer_on_orbits_of_a_given_length(
			int select_orbits_of_length_length, int verbose_level);

};





// #############################################################################
// large_set_was_description.cpp
// #############################################################################

//! command line description of tasks for large sets with assumed symmetry

class large_set_was_description {
public:


	int f_H;
	std::string H_go;
	std::string H_generators_text;

	int f_N;
	std::string N_go;
	std::string N_generators_text;

	int f_report;

	int f_prefix;
	std::string prefix;

	int f_selected_orbit_length;
	int selected_orbit_length;

	large_set_was_description();
	~large_set_was_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// large_set_was.cpp
// #############################################################################

//! classification of large sets of designs with assumed symmetry

class large_set_was {
public:

	large_set_was_description *Descr;

	large_set_classify *LS;

	groups::strong_generators *H_gens;

	groups::orbits_on_something *H_orbits;


	groups::strong_generators *N_gens;

	groups::orbits_on_something *N_orbits;


	// used in do_normalizer_on_orbits_of_a_given_length:
	int orbit_length;
	int nb_of_orbits_to_choose;
	int type_idx;
		// orbits of length orbit_length
		// in H_orbits->Orbits_classified
	long int *Orbit1;
	long int *Orbit2;

	actions::action *A_on_orbits;
		// action on H_orbits->Sch
	actions::action *A_on_orbits_restricted;
		// action A_on_orbits restricted to
		// H_orbits->Orbits_classified->Sets[type_idx]


	// used in do_normalizer_on_orbits_of_
	// a_given_length_multiple_orbits
	poset_classification::poset_classification *PC;
	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Poset;


	int orbit_length2;
	int type_idx2;
		// orbits of length orbit_length2
		// in H_orbits->Orbits_classified

	int selected_type_idx;


	large_set_was();
	~large_set_was();
	void init(
			large_set_was_description *Descr,
			large_set_classify *LS,
			int verbose_level);
	void do_normalizer_on_orbits_of_a_given_length(
			int orbit_length,
			int nb_of_orbits_to_choose,
			poset_classification::poset_classification_control *Control,
			int verbose_level);
	void do_normalizer_on_orbits_of_a_given_length_single_orbit(
			int orbit_length,
			int verbose_level);
	void do_normalizer_on_orbits_of_a_given_length_multiple_orbits(
			int orbit_length,
			int nb_of_orbits_to_choose,
			poset_classification::poset_classification_control *Control,
			int verbose_level);
	void create_graph_on_orbits_of_length(
			std::string &fname, int orbit_length,
			int verbose_level);
	void create_graph_on_orbits_of_length_based_on_N_orbits(
			std::string &fname_mask,
			int orbit_length2, int nb_N_orbits_preselected,
			int orbit_r, int orbit_m,
			int verbose_level);
	void read_solution_file(
			std::string &solution_file_name,
			long int *starter_set,
			int starter_set_sz,
			int orbit_length,
			int verbose_level);
	void normalizer_orbits_early_test_func(
			long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	int normalizer_orbits_check_conditions(
			long int *S, int len, int verbose_level);

};

void large_set_was_normalizer_orbits_early_test_func_callback(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
int large_set_was_design_test_orbit(
		long int *orbit, int orbit_length,
		void *extra_data);
int large_set_was_classify_test_pair_of_orbits(
		long int *orbit1, int orbit_length1,
		long int *orbit2, int orbit_length2,
		void *extra_data);







}}}


#endif /* ORBITER_SRC_LIB_TOP_LEVEL_COMBINATORICS_TL_COMBINATORICS_H_ */
