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
	data_structures_groups::vector_ge *nice_gens;

	induced_actions::action_on_homogeneous_polynomials *AonHPD;
	groups::strong_generators *SG;
	ring_theory::longinteger_object go;

	actions::action *A_affine; // restricted action on affine points

	boolean_function_classify();
	~boolean_function_classify();

	void init_group(combinatorics::boolean_function_domain *BF, int verbose_level);
	void search_for_bent_functions(int verbose_level);

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

	int f_line_type;
	std::string line_type_projective_space_label;
	std::string line_type_prefix;

	int f_conic_type;
	int conic_type_threshold;

	int f_non_conical_type;

	int f_ideal;
	std::string ideal_ring_label;


	// options that apply to IS = data_input_stream

	int f_canonical_form_PG;
	std::string canonical_form_PG_PG_label;
	int f_canonical_form_PG_has_PA;
	projective_geometry::projective_space_with_action *Canonical_form_PG_PA;
	combinatorics::classification_of_objects_description *Canonical_form_PG_Descr;

	int f_canonical_form;
	combinatorics::classification_of_objects_description *Canonical_form_Descr;

	int f_report;
	combinatorics::classification_of_objects_report_options *Classification_of_objects_report_options;

	int f_draw_incidence_matrices;
	std::string draw_incidence_matrices_prefix;

	int f_test_distinguishing_property;
	std::string test_distinguishing_property_graph;

	int f_unpack_from_restricted_action;
	std::string unpack_from_restricted_action_prefix;
	std::string unpack_from_restricted_action_group_label;

	int f_line_covering_type;
	std::string line_covering_type_prefix;
	std::string line_covering_type_projective_space;
	std::string line_covering_type_lines;


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

	int f_has_input_stream;
	data_structures::data_input_stream *IS;


	combinatorial_object_activity();
	~combinatorial_object_activity();
	void init(combinatorial_object_activity_description *Descr,
			geometry::geometric_object_create *GOC,
			int verbose_level);
	void init_input_stream(combinatorial_object_activity_description *Descr,
			data_structures::data_input_stream *IS,
			int verbose_level);
	void perform_activity(int verbose_level);
	void perform_activity_geometric_object(int verbose_level);
	void perform_activity_input_stream(int verbose_level);
	void do_save(std::string &save_as_fname,
			int f_extract, long int *extract_idx_set, int extract_size,
			int verbose_level);
	void post_process_classification(
			combinatorics::classification_of_objects *CO,
			object_with_properties *&OwP,
			int f_projective_space,
			projective_geometry::projective_space_with_action *PA,
			std::string &prefix,
			int verbose_level);
	void classification_report(combinatorics::classification_of_objects *CO,
			object_with_properties *OwP, int verbose_level);
	void latex_report(
			combinatorics::classification_of_objects_report_options *Report_options,
			combinatorics::classification_of_objects *CO,
			object_with_properties *OwP,
			int verbose_level);
	void report_all_isomorphism_types(
			std::ostream &fp,
			combinatorics::classification_of_objects_report_options *Report_options,
			combinatorics::classification_of_objects *CO,
			object_with_properties *OwP,
			int verbose_level);
	void report_isomorphism_type(
			std::ostream &fp,
			combinatorics::classification_of_objects_report_options *Report_options,
			combinatorics::classification_of_objects *CO,
			object_with_properties *OwP,
			int i, int verbose_level);
	void report_object(std::ostream &fp,
			combinatorics::classification_of_objects_report_options *Report_options,
			combinatorics::classification_of_objects *CO,
			object_with_properties *OwP,
			int object_idx,
			int verbose_level);
	void draw_incidence_matrices(
			std::string &prefix,
			data_structures::data_input_stream *IS,
			int verbose_level);
	void unpack_from_restricted_action(
			std::string &prefix,
			std::string &group_label,
			data_structures::data_input_stream *IS,
			int verbose_level);
	void line_covering_type(
			std::string &prefix,
			std::string &projective_space_label,
			std::string &lines,
			data_structures::data_input_stream *IS,
			int verbose_level);
	void line_type(
			std::string &prefix,
			std::string &projective_space_label,
			data_structures::data_input_stream *IS,
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
	void create_design_table(design_create *DC,
			std::string &problem_label,
			design_tables *&T,
			groups::strong_generators *Gens,
			int verbose_level);
	void load_design_table(design_create *DC,
			std::string &problem_label,
			design_tables *&T,
			groups::strong_generators *Gens,
			int verbose_level);

	void Hill_cap56(
		char *fname, int &nb_Pts, long int *&Pts,
		int verbose_level);
	void append_orbit_and_adjust_size(groups::schreier *Orb, int idx, int *set, int &sz);

};


// #############################################################################
// delandtsheer_doyen_description.cpp
// #############################################################################

#define MAX_MASK_TESTS 1000


//! description of the problem for delandtsheer_doyen


class delandtsheer_doyen_description {
public:

	int f_depth;
	int depth;

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
	poset_classification::poset_classification_control *Pair_search_control;

	int f_search_control;
	poset_classification::poset_classification_control *Search_control;

	// row intersection type
	int f_R;
	int nb_row_types;
	int *row_type;     		// [nb_row_types + 1]

	// col intersection type
	int f_C;
	int nb_col_types;
	int *col_type;     		// [nb_col_types + 1]


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

	int f_singletons;
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

	int Xsize; // = D = q1 = # of rows
	int Ysize; // = C = q2 = # of cols

	int V; // = Xsize * Ysize
	int b;
	long int *line;        // [K];
	int *row_sum; // [Xsize]
	int *col_sum; // [Ysize]


	groups::matrix_group *M1;
	groups::matrix_group *M2;
	actions::action *A1;
	actions::action *A2;

	actions::action *A;
	actions::action *A0;

	groups::strong_generators *SG;
	ring_theory::longinteger_object go;
	groups::direct_product *P;
	poset_classification::poset_with_group_action *Poset_pairs;
	poset_classification::poset_with_group_action *Poset_search;
	poset_classification::poset_classification *Pairs;
	poset_classification::poset_classification *Gen;

	// orbits on pairs:
	int *pair_orbit; // [V * V]
	int nb_orbits;
	int *transporter;
	int *tmp_Elt;
	int *orbit_length; 		// [nb_orbits]
	int *orbit_covered; 		// [nb_orbits]
	int *orbit_covered_max; 	// [nb_orbits]
		// orbit_covered_max[i] = orbit_length[i] / b;
	int *orbits_covered; 		// [K * K]


	// intersection type tests:

	int inner_pairs_in_rows;
	int inner_pairs_in_cols;

	// row intersection type
	int *row_type_cur; 		// [nb_row_types + 1]
	int *row_type_this_or_bigger; 	// [nb_row_types + 1]

	// col intersection type
	int *col_type_cur; 		// [nb_col_types + 1]
	int *col_type_this_or_bigger; 	// [nb_col_types + 1]



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
	void init(delandtsheer_doyen_description *Descr, int verbose_level);
	void show_generators(int verbose_level);
	void search_singletons(int verbose_level);
	void search_starter(int verbose_level);
	void compute_orbits_on_pairs(groups::strong_generators *Strong_gens, int verbose_level);
	groups::strong_generators *scan_subgroup_generators(int verbose_level);
	void create_monomial_group(int verbose_level);
	void create_action(int verbose_level);
	void create_graph(long int *line0, int len, int verbose_level);
	int find_pair_orbit(int i, int j, int verbose_level);
	int find_pair_orbit_by_tracing(int i, int j, int verbose_level);
	void compute_pair_orbit_table(int verbose_level);
	void write_pair_orbit_file(int verbose_level);
	void print_mask_test_i(std::ostream &ost, int i);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	int check_conditions(long int *S, int len, int verbose_level);
	int check_orbit_covering(long int *line, int len, int verbose_level);
	int check_row_sums(long int *line, int len, int verbose_level);
	int check_col_sums(long int *line, int len, int verbose_level);
	int check_mask(long int *line, int len, int verbose_level);
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
	combinatorics::classification_of_objects_description *Canonical_form_Descr;

	int f_extract_solutions_by_index_csv;
	int f_extract_solutions_by_index_txt;
	std::string extract_solutions_by_index_label;
	std::string extract_solutions_by_index_group;
	std::string extract_solutions_by_index_fname_solutions_in;
	std::string extract_solutions_by_index_fname_solutions_out;
	std::string extract_solutions_by_index_prefix;

	int f_export_inc;
	int f_export_blocks;
	int f_row_sums;
	int f_tactical_decomposition;

	design_activity_description();
	~design_activity_description();
	int read_arguments(int argc, std::string *argv,
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
	void perform_activity(design_activity_description *Descr,
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
	void do_canonical_form(combinatorics::classification_of_objects_description *Canonical_form_Descr,
			int verbose_level);
	void do_export_inc(
			design_create *DC,
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

	int f_q;
	int q;
	int f_catalogue;
	int iso;
	int f_family;
	std::string family_name;
	int f_list_of_blocks;
	int list_of_blocks_v;
	int list_of_blocks_k;
	std::string list_of_blocks_text;

	int f_list_of_blocks_from_file;
	std::string list_of_blocks_from_file_fname;

	int f_wreath_product_designs;
	int wreath_product_designs_n;
	int wreath_product_designs_k;

	int f_no_group;


	design_create_description();
	~design_create_description();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	int get_q();
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

	//int f_semilinear;

	actions::action *A; // Sym(degree)
	actions::action *A2; // Sym(degree), in the action on k-subsets


	actions::action *Aut;
	// PGGL(3,q) in case of PG_2_q with q not prime
	// PGL(3,q) in case of PG_2_q with q prime
	actions::action *Aut_on_lines; // Aut induced on lines

	int degree;

	long int *set;
	int sz; // = b, the number of blocks

	int f_has_group;
	groups::strong_generators *Sg;


	projective_geometry::projective_space_with_action *PA;
	geometry::projective_space *P;

	int *block; // [k]


	design_create();
	~design_create();
	void init(apps_combinatorics::design_create_description *Descr, int verbose_level);
	void create_design_PG_2_q(field_theory::finite_field *F,
			long int *&set, int &sz, int &k, int verbose_level);
	void unrank_block_in_PG_2_q(int *block,
			int rk, int verbose_level);
	int rank_block_in_PG_2_q(int *block,
			int verbose_level);
	int get_nb_colors_as_two_design(int verbose_level);
	int get_color_as_two_design_assume_sorted(long int *design, int verbose_level);
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
	void init(actions::action *A, actions::action *A2, long int *initial_set, int design_size,
			std::string &label,
			groups::strong_generators *Strong_generators, int verbose_level);
	void create_table(int verbose_level);
	void create_action(actions::action *&A_on_designs, int verbose_level);
	void extract_solutions_by_index(
			int nb_sol, int Index_width, int *Index,
			std::string &ouput_fname_csv,
			int verbose_level);
	void make_reduced_design_table(
			long int *set, int set_sz,
			long int *&reduced_table, long int *&reduced_table_idx, int &nb_reduced_designs,
			int verbose_level);
	void init_from_file(actions::action *A, actions::action *A2,
			long int *initial_set, int design_size,
			std::string &label,
			groups::strong_generators *Strong_generators, int verbose_level);
	int test_if_table_exists(
			std::string &label,
			int verbose_level);
	void save(int verbose_level);
	void load(int verbose_level);
	int test_if_designs_are_disjoint(int i, int j);
	int test_set_within_itself(long int *set_of_designs_by_index, int set_size);
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

#if 0
	int *N_gens;
	int N_nb_gens;
	int N_go;
#endif
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




	void init(int n, field_theory::finite_field *F, int verbose_level);
	void do_n2q3(int verbose_level);
	void check_overgroups_of_order_nine(int verbose_level);
	void create_minimal_overgroups(int verbose_level);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);

};



// #############################################################################
// flag_orbits_incidence_structure.cpp
// #############################################################################

//! classification of flag orbits of an incidence structure




class flag_orbits_incidence_structure {

public:

	object_with_properties *OwP;

	int nb_rows;
	int nb_cols;

	int f_flag_orbits_have_been_computed;
	int nb_flags;
	int *Flags; // [nb_flags]
	long int *Flag_table; // [nb_flags * 2]

	actions::action *A_on_flags;

	groups::orbits_on_something *Orb;

	flag_orbits_incidence_structure();
	~flag_orbits_incidence_structure();
	void init(object_with_properties *OwP,
			int f_anti_flags, actions::action *A_perm,
			groups::strong_generators *SG, int verbose_level);
	int find_flag(int i, int j);
	void report(std::ostream &ost, int verbose_level);

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

	void init(int n, int f_draw, int verbose_level, int verbose_level_clique);
	int clique_test(long int *set, int sz);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	int dot_product(int a, int b, int n);
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
		// number of pairs of points different from the reflection point.
		// these are the pairs of points that are covered by the
		// triples that we will choose.
		// The other pairs have been covered by the lines through
		// the reflection point, so they are fine because we assume that
		// these lines exist.
	int nb_blocks_overall; // {n \choose 2} / 6
	int nb_blocks_needed; // nb_blocks_overall - (n - 1) / 2
	int nb_orbits_needed; // nb_blocks_needed / 2
	int depth;
	int N;
		// {nb_pairs choose 3} * 8
		// {nb_pairs choose 3} counts the number of ways to choose three lines
		// through the reflection point.
		// the times 8 is because every triple of lines through the
		// reflection point has 2^3 ways of choosing one point on each line.
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
		// the induced action on unordered triples as stored in triples[].
	groups::strong_generators *Strong_gens_Hall_reflection;
		// the involution which switches the
		// points on every line through the center (other than the center).
	groups::strong_generators *Strong_gens_normalizer;
		// Strong generators for the normalizer of the involution.
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
	void init(int argc, const char **argv,
			int n, int depth,
			int verbose_level);
	void orbits_on_triples(int verbose_level);
	void print(std::ostream &ost, long int *S, int len);
	void unrank_triple(long int *T, int rk);
	void unrank_triple_pair(long int *T1, long int *T2, int rk);
	void early_test_func(long int *S, int len,
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
	void perform_activity(large_set_activity_description *Descr,
			large_set_was *LSW, int verbose_level);

};



// #############################################################################
// large_set_classify.cpp
// #############################################################################

//! classification of large sets of designs

class large_set_classify {
public:
	design_create *DC;
	int design_size; // = DC->sz = b, the number of blocks in the design
	int nb_points; // = DC->A->degree
	int nb_lines; // = DC->A2->degree
	int search_depth;

	std::string problem_label;

	int f_lexorder_test;
	int size_of_large_set; // = nb_lines / design_size


	design_tables *Design_table;

	int nb_colors; // = DC->get_nb_colors_as_two_design(0 /* verbose_level */);
	int *design_color_table; // [nb_designs]

	actions::action *A_on_designs; // action on designs in Design_table
		//DC->A2->create_induced_action_on_sets(
		//		Design_table->nb_designs, Design_table->design_size,
		//		Design_table->the_table,
		//		0 /* verbose_level */);


	data_structures::bitvector *Bitvec;
	int *degree;

	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification *gen;

	int nb_needed;




	large_set_classify();
	~large_set_classify();
	void init(design_create *DC,
			design_tables *T,
			int verbose_level);
	void create_action_and_poset(int verbose_level);
	void compute(int verbose_level);
	void read_classification(
			data_structures_groups::orbit_transversal *&T,
			int level, int verbose_level);
	void read_classification_single_case(
			data_structures_groups::set_and_stabilizer *&Rep,
			int level, int case_nr, int verbose_level);
	void compute_colors(
			design_tables *Design_table, int *&design_color_table,
			int verbose_level);
	int test_if_designs_are_disjoint(int i, int j);

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
	poset_classification::poset_classification_control *normalizer_on_orbits_of_a_given_length_control;

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
	void perform_activity(large_set_was_activity_description *Descr,
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
	int read_arguments(int argc, std::string *argv,
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

		//H_orbits->init(LS->A_on_designs,
		//		H_gens,
		//			FALSE /* f_load_save */,
		//			Descr->prefix,
		//			verbose_level);

	groups::strong_generators *N_gens;

	groups::orbits_on_something *N_orbits;


	// used in do_normalizer_on_orbits_of_a_given_length:
	int orbit_length;
	int nb_of_orbits_to_choose;
	int type_idx; // orbits of length orbit_length in H_orbits->Orbits_classified
	long int *Orbit1;
	long int *Orbit2;

	actions::action *A_on_orbits;
		// action on H_orbits->Sch
	actions::action *A_on_orbits_restricted;
		// action A_on_orbits restricted to H_orbits->Orbits_classified->Sets[type_idx]


	// used in do_normalizer_on_orbits_of_a_given_length_multiple_orbits::
	poset_classification::poset_classification *PC;
	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Poset;


	int orbit_length2;
	int type_idx2; // orbits of length orbit_length2 in H_orbits->Orbits_classified

#if 0
	// reduced designs are those which are compatible
	// with all the designs in the chosen set

	design_tables *Design_table_reduced;

	//long int *Design_table_reduced; // [nb_reduced * design_size]
	long int *Design_table_reduced_idx; // [nb_reduced], index into Design_table[]
	//int nb_reduced;


	int nb_remaining_colors; // = nb_colors - set_sz; // we assume that k = 4
	int *reduced_design_color_table; // [nb_reduced]
		// colors of the reduced designs after throwing away
		// the colors covered by the designs in the chosen set.
		// The remaining colors are relabeled consecutively.

	action *A_reduced;
		// reduced action A_on_designs based on Design_table_reduced_idx[]
	schreier *Orbits_on_reduced;
	int *color_of_reduced_orbits;
#endif

	int selected_type_idx;


	large_set_was();
	~large_set_was();
	void init(large_set_was_description *Descr,
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
			std::string &fname_mask, int orbit_length2, int nb_N_orbits_preselected,
			int orbit_r, int orbit_m,
			int verbose_level);
	void read_solution_file(
			std::string &solution_file_name,
			long int *starter_set,
			int starter_set_sz,
			int orbit_length,
			int verbose_level);
	void normalizer_orbits_early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	int normalizer_orbits_check_conditions(long int *S, int len, int verbose_level);

};

void large_set_was_normalizer_orbits_early_test_func_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
int large_set_was_design_test_orbit(long int *orbit, int orbit_length,
		void *extra_data);
int large_set_was_classify_test_pair_of_orbits(long int *orbit1, int orbit_length1,
		long int *orbit2, int orbit_length2, void *extra_data);



// #############################################################################
// object_with_properties.cpp
// #############################################################################

//! object properties which are derived from nauty canonical form


class object_with_properties {
public:

	geometry::object_with_canonical_form *OwCF;

	std::string label;

	data_structures::nauty_output *NO;

	int f_projective_space;
	projective_geometry::projective_space_with_action *PA;
	groups::strong_generators *SG; // only used if f_projective_space

	actions::action *A_perm;

	combinatorics::tdo_scheme_compute *TDO;

	flag_orbits_incidence_structure *Flags; // if !f_projective_space
	flag_orbits_incidence_structure *Anti_Flags; // if !f_projective_space

	object_with_properties();
	~object_with_properties();
	void init(
			geometry::object_with_canonical_form *OwCF,
			data_structures::nauty_output *NO,
			int f_projective_space, projective_geometry::projective_space_with_action *PA,
			int max_TDO_depth,
			std::string &label,
			int verbose_level);
	void compute_flag_orbits(int verbose_level);
	void lift_generators_to_matrix_group(int verbose_level);
	void init_object_in_projective_space(
			geometry::object_with_canonical_form *OwCF,
			data_structures::nauty_output *NO,
			projective_geometry::projective_space_with_action *PA,
			std::string &label,
			int verbose_level);
	void latex_report(std::ostream &ost,
			combinatorics::classification_of_objects_report_options *Report_options,
			int verbose_level);
	void compute_TDO(int max_TDO_depth, int verbose_level);
	void print_TDO(std::ostream &ost,
			combinatorics::classification_of_objects_report_options *Report_options);
	void export_TDA_with_flag_orbits(std::ostream &ost,
			groups::schreier *Sch,
			int verbose_level);
	void export_INP_with_flag_orbits(std::ostream &ost,
			groups::schreier *Sch,
			int verbose_level);

};





// #############################################################################
// regular_linear_space_description.cpp
// #############################################################################


//! a description of a class of regular linear spaces from the command line


class regular_linear_space_description {
public:

	int f_m;
	int m;
	int f_n;
	int n;
	int f_k;
	int k;
	int f_r;
	int r;
	int f_target_size;
	int target_size;

	int starter_size;
	int *initial_pair_covering;

	int f_has_control;
	poset_classification::poset_classification_control *Control;



	regular_linear_space_description();
	~regular_linear_space_description();
	void read_arguments_from_string(
			const char *str, int verbose_level);
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);


};



// #############################################################################
// regular_ls_classify.cpp
// #############################################################################


//! classification of regular linear spaces




class regular_ls_classify {

public:

	regular_linear_space_description *Descr;

	int m2;
	int *v1; // [k]

	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification *gen;
	actions::action *A;
	actions::action *A2;
	induced_actions::action_on_k_subsets *Aonk; // only a pointer, do not free

	int *row_sum; // [m]
	int *pairs; // [m2]
	int *open_rows; // [m]
	int *open_row_idx; // [m]
	int *open_pairs; // [m2]
	int *open_pair_idx; // [m2]

	regular_ls_classify();
	~regular_ls_classify();
	void init_and_run(
			regular_linear_space_description *Descr,
			int verbose_level);
	void init_group(int verbose_level);
	void init_action_on_k_subsets(int onk, int verbose_level);
	void init_generator(
			poset_classification::poset_classification_control *Control,
			groups::strong_generators *Strong_gens,
			int verbose_level);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void print(std::ostream &ost, long int *S, int len);
	void lifting_prepare_function_new(
			solvers_package::exact_cover *E, int starter_case,
		long int *candidates, int nb_candidates, groups::strong_generators *Strong_gens,
		solvers::diophant *&Dio, long int *&col_labels,
		int &f_ruled_out,
		int verbose_level);
};







// #############################################################################
// tactical_decomposition.cpp
// #############################################################################

//! tactical decomposition of an incidence structure with respect to a given group



class tactical_decomposition {
public:

	int set_size;
	int nb_blocks;
	geometry::incidence_structure *Inc;
	int f_combined_action;
	actions::action *A;
	actions::action *A_on_points;
	actions::action *A_on_lines;
	groups::strong_generators * gens;
	data_structures::partitionstack *Stack;
	groups::schreier *Sch;
	groups::schreier *Sch_points;
	groups::schreier *Sch_lines;

	tactical_decomposition();
	~tactical_decomposition();
	void init(int nb_rows, int nb_cols,
			geometry::incidence_structure *Inc,
			int f_combined_action,
			actions::action *Aut,
			actions::action *A_on_points,
			actions::action *A_on_lines,
			groups::strong_generators * gens,
			int verbose_level);
	void report(int f_enter_math, std::ostream &ost);

};





}}}


#endif /* ORBITER_SRC_LIB_TOP_LEVEL_COMBINATORICS_TL_COMBINATORICS_H_ */
