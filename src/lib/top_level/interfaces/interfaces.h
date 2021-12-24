/*
 * interfaces.h
 *
 *  Created on: Apr 3, 2020
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_INTERFACES_INTERFACES_H_
#define SRC_LIB_TOP_LEVEL_INTERFACES_INTERFACES_H_




namespace orbiter {
namespace top_level {


// #############################################################################
// activity_description.cpp
// #############################################################################

//! description of an activity for an orbiter symbol


class activity_description {

	interface_symbol_table *Sym;


	int f_finite_field_activity;
	finite_field_activity_description *Finite_field_activity_description;

	int f_projective_space_activity;
	projective_space_activity_description *Projective_space_activity_description;

	int f_orthogonal_space_activity;
	orthogonal_space_activity_description *Orthogonal_space_activity_description;

	int f_group_theoretic_activity;
	group_theoretic_activity_description *Group_theoretic_activity_description;

	int f_cubic_surface_activity;
	cubic_surface_activity_description *Cubic_surface_activity_description;

	int f_quartic_curve_activity;
	quartic_curve_activity_description *Quartic_curve_activity_description;

	int f_combinatorial_object_activity;
	combinatorial_object_activity_description *Combinatorial_object_activity_description;

	int f_graph_theoretic_activity;
	graph_theoretic_activity_description * Graph_theoretic_activity_description;

	int f_classification_of_cubic_surfaces_with_double_sixes_activity;
	classification_of_cubic_surfaces_with_double_sixes_activity_description *Classification_of_cubic_surfaces_with_double_sixes_activity_description;

	int f_spread_table_activity;
	spread_table_activity_description * Spread_table_activity_description;

	int f_packing_with_symmetry_assumption_activity;
	packing_was_activity_description *Packing_was_activity_description;

	int f_packing_fixed_points_activity;
	packing_was_fixpoints_activity_description *Packing_was_fixpoints_activity_description;

	int f_graph_classification_activity;
	graph_classification_activity_description *Graph_classification_activity_description;

	int f_diophant_activity;
	diophant_activity_description *Diophant_activity_description;

	int f_design_activity;
	design_activity_description *Design_activity_description;

	int f_large_set_was_activity;
	large_set_was_activity_description *Large_set_was_activity_description;

public:
	activity_description();
	~activity_description();
	void read_arguments(
			interface_symbol_table *Sym,
			int argc, std::string *argv, int &i, int verbose_level);
	void worker(int verbose_level);
	void print();
	void do_finite_field_activity(int verbose_level);
	void print_finite_field_activity();
	void do_projective_space_activity(int verbose_level);
	void do_orthogonal_space_activity(int verbose_level);
	void do_group_theoretic_activity(int verbose_level);
	void do_cubic_surface_activity(int verbose_level);
	void do_quartic_curve_activity(int verbose_level);
	void do_combinatorial_object_activity(int verbose_level);
	void do_graph_theoretic_activity(int verbose_level);
	void do_classification_of_cubic_surfaces_with_double_sixes_activity(int verbose_level);
	void do_spread_table_activity(int verbose_level);
	void do_packing_was_activity(int verbose_level);
	void do_packing_fixed_points_activity(int verbose_level);
	void do_graph_classification_activity(int verbose_level);
	void do_diophant_activity(int verbose_level);
	void do_design_activity(int verbose_level);
	void do_large_set_was_activity(int verbose_level);

};



// #############################################################################
// interface_algebra.cpp
// #############################################################################

//! interface to the algebra module


class interface_algebra {

	int f_count_subprimitive;
	int count_subprimitive_Q_max;
	int count_subprimitive_H_max;

	int f_equivalence_class_of_fractions;
	int equivalence_class_of_fractions_N;

	int f_character_table_symmetric_group;
	int deg;

	int f_make_A5_in_PSL_2_q;
	int q;

	int f_search_for_primitive_polynomial_in_range;
	int p_min, p_max, deg_min, deg_max;

	int f_order_of_q_mod_n;
	int order_of_q_mod_n_q;
	int order_of_q_mod_n_n_min;
	int order_of_q_mod_n_n_max;

	int f_young_symmetrizer;
	int young_symmetrizer_n;
	int f_young_symmetrizer_sym_4;

	int f_draw_mod_n;
	draw_mod_n_description *Draw_mod_n_description;

	int f_power_mod_n;
	int power_mod_n_a;
	int power_mod_n_n;

	int f_all_rational_normal_forms;
	std::string all_rational_normal_forms_finite_field_label;
	int all_rational_normal_forms_d;

	int f_eigenstuff;
	int f_eigenstuff_from_file;
	std::string eigenstuff_finite_field_label;
	int eigenstuff_n;
	std::string eigenstuff_coeffs;
	std::string eigenstuff_fname;


public:
	interface_algebra();
	void print_help(int argc, std::string *argv, int i, int verbose_level);
	int recognize_keyword(int argc, std::string *argv, int i, int verbose_level);
	void read_arguments(int argc,
			std::string *argv, int &i, int verbose_level);
	void print();
	void worker(int verbose_level);
	void do_character_table_symmetric_group(int deg, int verbose_level);

};



// #############################################################################
// interface_coding_theory.cpp
// #############################################################################

//! interface to the coding theory module


class interface_coding_theory {

	int f_make_macwilliams_system;
	int q;
	int n;
	int k;

	int f_table_of_bounds;
	int table_of_bounds_n_max;
	int table_of_bounds_q;

	int f_make_bounds_for_d_given_n_and_k_and_q;

	int f_BCH;
	int f_BCH_dual;
	int BCH_t;
	//int BCH_b;

	int f_Hamming_space_distance_matrix;

	int f_NTT;
	std::string ntt_fname_code;

	int f_general_code_binary;
	int general_code_binary_n;
	std::string general_code_binary_text;

	int f_code_diagram;
	std::string code_diagram_label;
	std::string code_diagram_codewords_text;
	int code_diagram_n;

	int f_code_diagram_from_file;
	std::string code_diagram_from_file_codewords_fname;

	int f_enhance;
	int enhance_radius;

	int f_metric_balls;
	int radius_of_metric_ball;


	int f_linear_code_through_basis;
	int linear_code_through_basis_n;
	std::string linear_code_through_basis_text;

	int f_linear_code_through_columns_of_parity_check_projectively;
	int f_linear_code_through_columns_of_parity_check;

	int linear_code_through_columns_of_parity_check_k;
	std::string linear_code_through_columns_of_parity_check_text;

	int f_long_code;
	int long_code_n;
	std::vector<std::string> long_code_generators;

	int f_encode_text_5bits;
	std::string encode_text_5bits_input;
	std::string encode_text_5bits_fname;

	int f_field_induction;
	std::string field_induction_fname_in;
	std::string field_induction_fname_out;
	int field_induction_nb_bits;



public:
	interface_coding_theory();
	void print_help(int argc, std::string *argv, int i, int verbose_level);
	int recognize_keyword(int argc, std::string *argv, int i, int verbose_level);
	void read_arguments(int argc,
			std::string *argv, int &i, int verbose_level);
	void print();
	void worker(int verbose_level);
};


// #############################################################################
// interface_combinatorics.cpp
// #############################################################################

//! interface to the coding theory module


class interface_combinatorics {

	//int f_create_combinatorial_object;
	//combinatorial_object_description *Combinatorial_object_description;

	int f_diophant;
	diophant_description *Diophant_description;

	int f_diophant_activity;
	diophant_activity_description *Diophant_activity_description;

	//int f_process_combinatorial_objects;
	//projective_space_job_description *Job_description;

	int f_bent;
	int bent_n;

	int f_random_permutation;
	int random_permutation_degree;
	std::string random_permutation_fname_csv;

	int f_read_poset_file;
	std::string read_poset_file_fname;

	int f_grouping;
	double x_stretch;

	int f_list_parameters_of_SRG;
	int v_max;

	int f_conjugacy_classes_Sym_n;
	int n;

	int f_tree_of_all_k_subsets;
	int tree_n, tree_k;

	int f_Delandtsheer_Doyen;
	delandtsheer_doyen_description *Delandtsheer_Doyen_description;

	int f_tdo_refinement;
	tdo_refinement_description *Tdo_refinement_descr;

	int f_tdo_print;
	std::string tdo_print_fname;

	int f_create_design;
	design_create_description *Design_create_description;

	int f_convert_stack_to_tdo;
	std::string stack_fname;

	int f_maximal_arc_parameters;
	int maximal_arc_parameters_q, maximal_arc_parameters_r;

	int f_arc_parameters;
	int arc_parameters_q, arc_parameters_s, arc_parameters_r;


	int f_pentomino_puzzle;

	int f_regular_linear_space_classify;
	regular_linear_space_description *Rls_descr;

	int f_create_files;
	create_file_description *Create_file_description;

	int f_draw_layered_graph;
	std::string draw_layered_graph_fname;
	layered_graph_draw_options *Layered_graph_draw_options;

	int f_read_solutions_and_tally;
	std::string read_solutions_and_tally_fname;
	int read_solutions_and_tally_sz;

	int f_make_elementary_symmetric_functions;
	int make_elementary_symmetric_functions_n;
	int make_elementary_symmetric_functions_k_max;

	int f_Dedekind_numbers;
	int Dedekind_n_min;
	int Dedekind_n_max;
	int Dedekind_q_min;
	int Dedekind_q_max;

	int f_canonical_form_nauty;
	classification_of_objects_description *Canonical_form_nauty_Descr;

	int f_rank_k_subset;
	int rank_k_subset_n;
	int rank_k_subset_k;
	std::string rank_k_subset_text;

	int f_geometry_builder;
	geometry_builder_description *Geometry_builder_description;


public:
	interface_combinatorics();
	void print_help(int argc, std::string *argv, int i, int verbose_level);
	int recognize_keyword(int argc, std::string *argv, int i, int verbose_level);
	void read_arguments(int argc,
			std::string *argv, int &i, int verbose_level);
	void print();
	void worker(int verbose_level);
	//void do_create_combinatorial_object(int verbose_level);
	void do_diophant(diophant_description *Descr, int verbose_level);
	void do_diophant_activity(diophant_activity_description *Descr, int verbose_level);
	void do_bent(int n, int verbose_level);
	void do_conjugacy_classes_Sym_n(int n, int verbose_level);
	void do_Delandtsheer_Doyen(delandtsheer_doyen_description *Descr, int verbose_level);
	void do_create_design(design_create_description *Descr, int verbose_level);

};


// #############################################################################
// interface_cryptography.cpp
// #############################################################################


enum cipher_type { no_cipher_type, substitution, vigenere, affine };

typedef enum cipher_type cipher_type;

//! interface to the cryptography module

class interface_cryptography {

	int f_cipher;
	cipher_type t;
	int f_decipher;
	int f_analyze;
	int f_kasiski;
	int f_avk;
	int key_length, threshold;
	int affine_a;
	int affine_b;

	std::string ptext;
	std::string ctext;
	std::string guess;
	std::string key;

	int f_RSA;
	long int RSA_d;
	long int RSA_m;
	std::string RSA_text;

	int f_primitive_root;
	std::string primitive_root_p;

	int f_smallest_primitive_root;
	int smallest_primitive_root_p;
	int f_smallest_primitive_root_interval;
	int smallest_primitive_root_interval_min;
	int smallest_primitive_root_interval_max;
	int f_number_of_primitive_roots_interval;
	int f_inverse_mod;
	int inverse_mod_a;
	int inverse_mod_n;
	int f_extended_gcd;
	int extended_gcd_a;
	int extended_gcd_b;

	int f_power_mod;
	std::string power_mod_a;
	std::string power_mod_k;
	std::string power_mod_n;

	int f_discrete_log;
	long int discrete_log_y;
	long int discrete_log_a;
	long int discrete_log_m;
	int f_RSA_setup;
	int RSA_setup_nb_bits;
	int RSA_setup_nb_tests_solovay_strassen;
	int RSA_setup_f_miller_rabin_test;
	int f_RSA_encrypt_text;
	int RSA_block_size;
	std::string RSA_encrypt_text;
	int f_sift_smooth;
	int sift_smooth_from;
	int sift_smooth_len;
	std::string sift_smooth_factor_base;
	int f_square_root;
	std::string square_root_number;
	int f_square_root_mod;
	std::string square_root_mod_a;
	std::string square_root_mod_m;

	int f_all_square_roots_mod_n;
	std::string all_square_roots_mod_n_a;
	std::string all_square_roots_mod_n_n;

	int f_quadratic_sieve;
	int quadratic_sieve_n;
	int quadratic_sieve_factorbase;
	int quadratic_sieve_x0;
	int f_jacobi;
	int jacobi_top;
	int jacobi_bottom;
	int f_solovay_strassen;
	int solovay_strassen_p;
	int solovay_strassen_a;
	int f_miller_rabin;
	int miller_rabin_p;
	int miller_rabin_nb_times;
	int f_fermat_test;
	int fermat_test_p;
	int fermat_test_nb_times;
	int f_find_pseudoprime;
	int find_pseudoprime_nb_digits;
	int find_pseudoprime_nb_fermat;
	int find_pseudoprime_nb_miller_rabin;
	int find_pseudoprime_nb_solovay_strassen;
	int f_find_strong_pseudoprime;
	int f_miller_rabin_text;
	int miller_rabin_text_nb_times;
	std::string miller_rabin_number_text;
	int f_random;
	int random_nb;
	std::string random_fname_csv;
	int f_random_last;
	int random_last_nb;
	int f_affine_sequence;
	int affine_sequence_a;
	int affine_sequence_c;
	int affine_sequence_m;

public:
	interface_cryptography();
	void print_help(int argc, std::string *argv, int i, int verbose_level);
	int recognize_keyword(int argc, std::string *argv, int i, int verbose_level);
	void read_arguments(int argc, std::string *argv, int &i, int verbose_level);
	void print();
	void worker(int verbose_level);


};

// #############################################################################
// interface_povray.cpp
// #############################################################################

//! interface to the povray rendering module


class interface_povray {

	int f_povray;
	int f_output_mask;
	std::string output_mask;
	int f_nb_frames_default;
	int nb_frames_default;
	int f_round;
	int round;
	int f_rounds;
	std::string rounds_as_string;
	video_draw_options *Opt;

	// for povray_worker:
	scene *S;
	animate *A;

	int f_prepare_frames;
	prepare_frames *Prepare_frames;

public:
	interface_povray();
	void print_help(int argc, std::string *argv, int i, int verbose_level);
	int recognize_keyword(int argc, std::string *argv, int i, int verbose_level);
	void read_arguments(int argc, std::string *argv, int &i, int verbose_level);
	void print();
	void worker(int verbose_level);
};

void interface_povray_draw_frame(
	animate *Anim, int h, int nb_frames, int round,
	double clipping_radius,
	std::ostream &fp,
	int verbose_level);



// #############################################################################
// interface_projective.cpp
// #############################################################################

//! interface to the projective geometry module


class interface_projective {


	int f_create_points_on_quartic;
	double desired_distance;

	int f_create_points_on_parabola;
	int parabola_N;
	double parabola_a;
	double parabola_b;
	double parabola_c;

	int f_smooth_curve;
	std::string smooth_curve_label;
	int smooth_curve_N;
	double smooth_curve_boundary;
	double smooth_curve_t_min;
	double smooth_curve_t_max;
	function_polish_description *FP_descr;



	int f_create_spread;
	spread_create_description *Spread_create_description;

	int f_make_table_of_surfaces;

	int f_create_surface_reports;
	std::string create_surface_reports_field_orders_text;

	int f_create_surface_atlas;
	int create_surface_atlas_q_max;

	int f_create_dickson_atlas;

	std::vector<std::string> transform_coeffs;
	std::vector<int> f_inverse_transform;


public:


	interface_projective();
	void print_help(int argc, std::string *argv, int i, int verbose_level);
	int recognize_keyword(int argc, std::string *argv, int i, int verbose_level);
	void read_arguments(int argc,
			std::string *argv, int &i, int verbose_level);
	void print();
	void worker(int verbose_level);
	void do_cheat_sheet_PG(orbiter_session *Session,
			int n, int q,
			int f_decomposition_by_element, int decomposition_by_element_power,
			std::string &decomposition_by_element_data, std::string &fname_base,
			int verbose_level);
	void do_canonical_form_PG(orbiter_session *Session,
			int n, int q, int verbose_level);
	void do_create_BLT_set(BLT_set_create_description *Descr, int verbose_level);
	void do_create_spread(spread_create_description *Descr, int verbose_level);
	void do_create_surface(surface_create_description *Descr, int verbose_level);

};

// #############################################################################
// interface_symbol_table.cpp
// #############################################################################

//! interface to the orbiter symbol table


class interface_symbol_table {

public:
	orbiter_top_level_session *Orbiter_top_level_session;


	int f_define;
	symbol_definition *Symbol_definition;



	int f_print_symbols;

	int f_with;
	std::vector<std::string> with_labels;

	int f_activity;
	activity_description *Activity_description;





	interface_symbol_table();
	void init(orbiter_top_level_session *Orbiter_top_level_session,
			int verbose_level);
	void print_help(int argc, std::string *argv, int i, int verbose_level);
	int recognize_keyword(int argc, std::string *argv, int i, int verbose_level);
	void read_arguments(
			int argc, std::string *argv, int &i, int verbose_level);
	void read_with(
			int argc, std::string *argv, int &i, int verbose_level);
	void worker(int verbose_level);
	void print();
	void print_with();

};



// #############################################################################
// interface_toolkit.cpp
// #############################################################################

//! interface to the general toolkit


class interface_toolkit {

	int f_csv_file_select_rows;
	std::string csv_file_select_rows_fname;
	std::string csv_file_select_rows_text;

	int f_csv_file_split_rows_modulo;
	std::string csv_file_split_rows_modulo_fname;
	int csv_file_split_rows_modulo_n;

	int f_csv_file_select_cols;
	std::string csv_file_select_cols_fname;
	std::string csv_file_select_cols_text;

	int f_csv_file_select_rows_and_cols;
	std::string csv_file_select_rows_and_cols_fname;
	std::string csv_file_select_rows_and_cols_R_text;
	std::string csv_file_select_rows_and_cols_C_text;

	int f_csv_file_sort_each_row;
	std::string csv_file_sort_each_row_fname;


	int f_csv_file_join;
	std::vector<std::string> csv_file_join_fname;
	std::vector<std::string> csv_file_join_identifier;

	int f_csv_file_concatenate;
	std::string csv_file_concatenate_fname_out;
	std::vector<std::string> csv_file_concatenate_fname_in;

	int f_csv_file_extract_column_to_txt;
	std::string csv_file_extract_column_to_txt_fname;
	std::string csv_file_extract_column_to_txt_col_label;


	int f_csv_file_latex;
	int f_produce_latex_header;
	std::string csv_file_latex_fname;

	int f_draw_matrix;
	draw_bitmap_control *Draw_bitmap_control;

	int f_reformat;
	std::string reformat_fname_in;
	std::string reformat_fname_out;
	int reformat_nb_cols;

	int f_split_by_values;
	std::string split_by_values_fname_in;

	int f_store_as_csv_file;
	std::string store_as_csv_file_fname;
	int store_as_csv_file_m;
	int store_as_csv_file_n;
	std::string store_as_csv_file_data;

	int f_mv;
	std::string mv_a;
	std::string mv_b;


	int f_loop;
	int loop_start_idx;
	int loop_end_idx;
	std::string loop_variable;
	int loop_from;
	int loop_to;
	int loop_step;
	std::string *loop_argv;

	int f_plot_function;
	std::string plot_function_fname;

	int f_draw_projective_curve;
	draw_projective_curve_description *Draw_projective_curve_description;

	int f_tree_draw;
	std::string tree_draw_fname;

public:


	interface_toolkit();
	void print_help(int argc, std::string *argv, int i, int verbose_level);
	int recognize_keyword(int argc, std::string *argv, int i, int verbose_level);
	void read_arguments(int argc,
			std::string *argv, int &i, int verbose_level);
	void print();
	void worker(int verbose_level);

};

// #############################################################################
// orbiter_command.cpp
// #############################################################################



//! a command unit


class orbiter_command {

public:

	orbiter_top_level_session *Orbiter_top_level_session;


	int f_algebra;
	interface_algebra *Algebra;

	int f_coding_theory;
	interface_coding_theory *Coding_theory;

	int f_combinatorics;
	interface_combinatorics *Combinatorics;

	int f_cryptography;
	interface_cryptography *Cryptography;

	int f_povray;
	interface_povray *Povray;

	int f_projective;
	interface_projective *Projective;

	int f_symbol_table;
	interface_symbol_table *Symbol_table;

	int f_toolkit;
	interface_toolkit *Toolkit;

	orbiter_command();
	~orbiter_command();
	void parse(orbiter_top_level_session *Orbiter_top_level_session,
			int argc, std::string *Argv, int &i, int verbose_level);
	void execute(int verbose_level);
	void print();

};


// #############################################################################
// orbiter_top_level_session.cpp
// #############################################################################



//! The top level orbiter session is responsible for the command line interface and the program execution and for the orbiter_session


class orbiter_top_level_session {

public:

	orbiter_session *Orbiter_session;

	orbiter_top_level_session();
	~orbiter_top_level_session();
	int startup_and_read_arguments(int argc,
			std::string *argv, int i0);
	void handle_everything(int argc, std::string *Argv, int i, int verbose_level);
	void parse_and_execute(int argc, std::string *Argv, int i, int verbose_level);
	void parse(int argc, std::string *Argv, int &i, std::vector<void * > &program, int verbose_level);
	void *get_object(int idx);
	symbol_table_object_type get_object_type(int idx);
	int find_symbol(std::string &label);
	void find_symbols(std::vector<std::string> &Labels, int *&Idx);
	void print_symbol_table();
	void add_symbol_table_entry(std::string &label,
			orbiter_symbol_table_entry *Symb, int verbose_level);

};



// #############################################################################
// symbol_definition.cpp
// #############################################################################

//! definition of an orbiter symbol


class symbol_definition {

public:
	interface_symbol_table *Sym;


	std::string define_label;

	int f_finite_field;
	finite_field_description *Finite_field_description;

	int f_projective_space;
	projective_space_with_action_description *Projective_space_with_action_description;

	int f_orthogonal_space;
	orthogonal_space_with_action_description *Orthogonal_space_with_action_description;

	int f_linear_group;
	linear_group_description *Linear_group_description;

	int f_permutation_group;
	permutation_group_description *Permutation_group_description;

	int f_group_modification;
	group_modification_description *Group_modification_description;

	int f_formula;
	formula *F;
	std::string label;
	std::string label_tex;
	std::string managed_variables;
	std::string formula_text;

	int f_collection;
	std::string list_of_objects;

	//int f_combinatorial_object;
	//combinatorial_object_description *Combinatorial_object_description;

	int f_graph;
	create_graph_description *Create_graph_description;

	int f_spread_table;
	std::string spread_table_label_PA;
	int dimension_of_spread_elements;
	std::string spread_selection_text;
	std::string spread_tables_prefix;

	int f_packing_was;
	std::string packing_was_label_spread_table;
	packing_was_description * packing_was_descr;

	int f_packing_was_choose_fixed_points;
	std::string packing_with_assumed_symmetry_label;
	int packing_with_assumed_symmetry_choose_fixed_points_clique_size;
	poset_classification_control *packing_with_assumed_symmetry_choose_fixed_points_control;


	int f_packing_long_orbits;
	std::string packing_long_orbits_choose_fixed_points_label;
	packing_long_orbits_description * Packing_long_orbits_description;

	int f_graph_classification;
	graph_classify_description * Graph_classify_description;

	int f_diophant;
	diophant_description *Diophant_description;

	int f_design;
	design_create_description *Design_create_description;

	int f_design_table;
	std::string design_table_label_design;
	std::string design_table_label;
	std::string design_table_group;


	int f_large_set_was;
	std::string  large_set_was_label_design_table;
	large_set_was_description *large_set_was_descr;

	int f_set;
	set_builder_description *Set_builder_description;

	int f_vector;
	vector_builder_description *Vector_builder_description;

	int f_combinatorial_objects;
	data_input_stream_description *Data_input_stream_description;

	int f_geometry_builder;
	geometry_builder_description *Geometry_builder_description;


	symbol_definition();
	~symbol_definition();
	void read_definition(
			interface_symbol_table *Sym,
			int argc, std::string *argv, int &i, int verbose_level);
	void perform_definition(int verbose_level);
	void print();
	void definition_of_finite_field(int verbose_level);
	void definition_of_projective_space(int verbose_level);
	void print_definition_of_projective_space(int verbose_level);
	void definition_of_orthogonal_space(int verbose_level);
	void definition_of_linear_group(int verbose_level);
	void definition_of_permutation_group(int verbose_level);
	void definition_of_modified_group(int verbose_level);
	void definition_of_formula(formula *F,
			int verbose_level);
	void definition_of_collection(std::string &list_of_objects,
			int verbose_level);
	//void definition_of_combinatorial_object(int verbose_level);
	void definition_of_graph(int verbose_level);
	void definition_of_spread_table(int verbose_level);
	void definition_of_packing_was(int verbose_level);
	void definition_of_packing_was_choose_fixed_points(int verbose_level);
	void definition_of_packing_long_orbits(int verbose_level);
	void definition_of_graph_classification(int verbose_level);
	void definition_of_diophant(int verbose_level);
	void definition_of_design(int verbose_level);
	void definition_of_design_table(int verbose_level);
	void definition_of_large_set_was(int verbose_level);
	void definition_of_set(int verbose_level);
	void definition_of_vector(int verbose_level);
	void definition_of_combinatorial_object(int verbose_level);
	void do_geometry_builder(int verbose_level);

};






}}


#endif /* SRC_LIB_TOP_LEVEL_INTERFACES_INTERFACES_H_ */
