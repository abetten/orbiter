/*
 * interfaces.h
 *
 *  Created on: Apr 3, 2020
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_INTERFACES_INTERFACES_H_
#define SRC_LIB_TOP_LEVEL_INTERFACES_INTERFACES_H_




namespace orbiter {
namespace layer5_applications {
namespace user_interface {


// #############################################################################
// activity_description.cpp
// #############################################################################

//! description of an activity for an orbiter symbol


class activity_description {

public:

	interface_symbol_table *Sym;


	int f_finite_field_activity;
	algebra::field_theory::finite_field_activity_description
		*Finite_field_activity_description;

	int f_polynomial_ring_activity;
	algebra::ring_theory::polynomial_ring_activity_description
		*Polynomial_ring_activity_description;

	int f_projective_space_activity;
	projective_geometry::projective_space_activity_description
		*Projective_space_activity_description;

	int f_orthogonal_space_activity;
	orthogonal_geometry_applications::orthogonal_space_activity_description
		*Orthogonal_space_activity_description;

	int f_group_theoretic_activity;
	apps_algebra::group_theoretic_activity_description
		*Group_theoretic_activity_description;

	int f_coding_theoretic_activity;
	apps_coding_theory::coding_theoretic_activity_description
		*Coding_theoretic_activity_description;

	int f_cubic_surface_activity;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::cubic_surface_activity_description
		*Cubic_surface_activity_description;

	int f_quartic_curve_activity;
	applications_in_algebraic_geometry::quartic_curves::quartic_curve_activity_description
		*Quartic_curve_activity_description;

	int f_blt_set_activity;
	orthogonal_geometry_applications::blt_set_activity_description
		*Blt_set_activity_description;

	int f_combinatorial_object_activity;
	apps_combinatorics::combinatorial_object_activity_description
		*Combinatorial_object_activity_description;

	int f_graph_theoretic_activity;
	apps_graph_theory::graph_theoretic_activity_description
		*Graph_theoretic_activity_description;

	int f_classification_of_cubic_surfaces_with_double_sixes_activity;
	applications_in_algebraic_geometry::cubic_surfaces_and_double_sixes::classification_of_cubic_surfaces_with_double_sixes_activity_description
		*Classification_of_cubic_surfaces_with_double_sixes_activity_description;

	int f_spread_table_activity;
	spreads::spread_table_activity_description
		*Spread_table_activity_description;

	int f_packing_classify_activity_description;
	packings::packing_classify_activity_description *Packing_classify_activity_description;

	int f_packing_with_symmetry_assumption_activity;
	packings::packing_was_activity_description
		*Packing_was_activity_description;

	int f_packing_fixed_points_activity;
	packings::packing_was_fixpoints_activity_description
		*Packing_was_fixpoints_activity_description;

	int f_graph_classification_activity;
	apps_graph_theory::graph_classification_activity_description
		*Graph_classification_activity_description;

	int f_diophant_activity;
	combinatorics::solvers::diophant_activity_description
		*Diophant_activity_description;

	int f_design_activity;
	apps_combinatorics::design_activity_description
		*Design_activity_description;

	int f_large_set_was_activity;
	apps_combinatorics::large_set_was_activity_description
		*Large_set_was_activity_description;

	int f_symbolic_object_activity;
	algebra::expression_parser::symbolic_object_activity_description
		*Symbolic_object_activity_description;

	int f_BLT_set_classify_activity;
	orthogonal_geometry_applications::blt_set_classify_activity_description
		*Blt_set_classify_activity_description;

	int f_spread_classify_activity;
	spreads::spread_classify_activity_description
		*Spread_classify_activity_description;

	int f_spread_activity;
	spreads::spread_activity_description
		*Spread_activity_description;

	int f_translation_plane_activity;
	spreads::translation_plane_activity_description
		*Translation_plane_activity_description;

	int f_action_on_forms_activity;
	apps_algebra::action_on_forms_activity_description
		*Action_on_forms_activity_description;

	int f_orbits_activity;
	orbits::orbits_activity_description
		*Orbits_activity_description;

	int f_variety_activity;
	canonical_form::variety_activity_description
		*Variety_activity_description;

	int f_vector_ge_activity;
	apps_algebra::vector_ge_activity_description
		*Vector_ge_activity_description;

	int f_combo_activity;
	apps_combinatorics::combo_activity_description *Combo_activity_description;



	activity_description();
	~activity_description();
	void read_arguments(
			//interface_symbol_table *Sym,
			int argc, std::string *argv, int &i,
			int verbose_level);
	void worker(
			std::vector<std::string> &with_labels,
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);
	void print();
	void do_finite_field_activity(
			other::orbiter_kernel_system::activity_output *&AO,
			int verbose_level);
	void do_ring_theoretic_activity(
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);
	void do_projective_space_activity(
			int verbose_level);
	void do_orthogonal_space_activity(
			int verbose_level);
	void do_group_theoretic_activity(
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);
	void do_coding_theoretic_activity(
			int verbose_level);
	void do_cubic_surface_activity(
			int verbose_level);
	void do_quartic_curve_activity(
			int verbose_level);
	void do_blt_set_activity(
			int verbose_level);
	void do_combinatorial_object_activity(
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			other::orbiter_kernel_system::activity_output *&AO,
			int verbose_level);
	void do_graph_theoretic_activity(
			int verbose_level);
	void do_classification_of_cubic_surfaces_with_double_sixes_activity(
			int verbose_level);
	void do_spread_table_activity(
			int verbose_level);
	void do_packing_classify_activity(
			int verbose_level);
	void do_packing_was_activity(
			int verbose_level);
	void do_packing_fixed_points_activity(
			int verbose_level);
	void do_graph_classification_activity(
			int verbose_level);
	void do_diophant_activity(
			int verbose_level);
	void do_design_activity(
			int verbose_level);
	void do_large_set_was_activity(
			int verbose_level);
	void do_symbolic_object_activity(
			int verbose_level);
	void do_BLT_set_classify_activity(
			int verbose_level);
	void do_spread_classify_activity(
			int verbose_level);
	void do_spread_activity(
			int verbose_level);
	void do_translation_plane_activity(
			int verbose_level);
	void do_action_on_forms_activity(
			int verbose_level);
	void do_orbits_activity(
			int verbose_level);
	void do_variety_activity(
			int verbose_level);
	void do_vector_ge_activity(
			std::vector<std::string> &with_labels,
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);
	void do_combo_activity(
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);

};



// #############################################################################
// interface_algebra.cpp
// #############################################################################

//! interface to the algebra module


class interface_algebra {







	// Section 10.1
	// TABLES/global_basic_number_theory.csv

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

	int f_square_root;
	std::string square_root_number;

	int f_square_root_mod;
	std::string square_root_mod_a;
	std::string square_root_mod_m;

	int f_all_square_roots_mod_n;
	std::string all_square_roots_mod_n_a;
	std::string all_square_roots_mod_n_n;

	int f_count_subprimitive;
	int count_subprimitive_Q_max;
	int count_subprimitive_H_max;

	int f_order_of_q_mod_n;
	int order_of_q_mod_n_q;
	int order_of_q_mod_n_n_min;
	int order_of_q_mod_n_n_max;

	int f_eulerfunction_interval;
	int eulerfunction_interval_n_min;
	int eulerfunction_interval_n_max;

	int f_jacobi;
	long int jacobi_top;
	long int jacobi_bottom;

	int f_Chinese_remainders;
	std::string Chinese_remainders_R;
	std::string Chinese_remainders_M;


	int f_draw_mod_n;
	other::graphics::draw_mod_n_description *Draw_mod_n_description;


	// Section 10.1
	// TABLES/interface_algebra.csv



	int f_power_function_mod_n;
	int power_function_mod_n_k;
	int power_function_mod_n_n;


	// the following two cannot be finite field activities because
	// finite field activities are at layer 1 and these functions require level 5.

	// perhaps they should be projective space activities,
	// because they need a general linear group

	int f_all_rational_normal_forms;
	std::string all_rational_normal_forms_finite_field_label;
	int all_rational_normal_forms_d;

	int f_compute_rational_normal_form;
	std::string compute_rational_normal_form_field_label;
	int compute_rational_normal_form_d;
	std::string compute_rational_normal_form_data;

	int f_eigenstuff;
	std::string eigenstuff_finite_field_label;
	int eigenstuff_n;
	std::string eigenstuff_coeffs;
	std::string eigenstuff_fname;

	int f_smith_normal_form;
	std::string smith_normal_form_matrix;



	// representation theory:
	int f_character_table_symmetric_group;
	int character_table_symmetric_group_n;

	int f_young_symmetrizer;
	int young_symmetrizer_n;

	int f_young_symmetrizer_sym_4;



	// global group theory:

	int f_make_A5_in_PSL_2_q;
	int make_A5_in_PSL_2_q_q;


	int f_order_of_group_Anq;
	int order_of_group_Anq_n;
	int order_of_group_Anq_q;

	int f_order_of_group_Bnq;
	int order_of_group_Bnq_n;
	int order_of_group_Bnq_q;

	int f_order_of_group_Dnq;
	int order_of_group_Dnq_n;
	int order_of_group_Dnq_q;


public:
	interface_algebra();
	~interface_algebra();
	void print_help(
			int argc,
			std::string *argv, int i, int verbose_level);
	int recognize_keyword(
			int argc,
			std::string *argv, int i, int verbose_level);
	void read_arguments(
			int argc,
			std::string *argv, int &i, int verbose_level);
	void print();
	void worker(
			int verbose_level);

};



// #############################################################################
// interface_coding_theory.cpp
// #############################################################################

//! interface to the coding theory module


class interface_coding_theory {

	// Section 11.1
	// TABLES/global_coding_theory.csv

	int f_make_macwilliams_system;
	int make_macwilliams_system_q;
	int make_macwilliams_system_n;
	int make_macwilliams_system_k;

	int f_table_of_bounds;
	int table_of_bounds_n_max;
	int table_of_bounds_q;

	int f_make_bounds_for_d_given_n_and_k_and_q;
	int make_bounds_n;
	int make_bounds_k;
	int make_bounds_q;

	int f_introduce_errors;
	combinatorics::coding_theory::crc_options_description
		*introduce_errors_crc_options_description;

	int f_check_errors;
	combinatorics::coding_theory::crc_options_description
		*check_errors_crc_options_description;

	int f_extract_block;
	combinatorics::coding_theory::crc_options_description
		*extract_block_crc_options_description;

	int f_random_noise_in_bitmap_file;
	std::string random_noise_in_bitmap_file_input;
	std::string random_noise_in_bitmap_file_output;
	int random_noise_in_bitmap_file_numerator;
	int random_noise_in_bitmap_file_denominator;

	int f_random_noise_of_burst_type_in_bitmap_file;
	std::string random_noise_of_burst_type_in_bitmap_file_input;
	std::string random_noise_of_burst_type_in_bitmap_file_output;
	int random_noise_of_burst_type_in_bitmap_file_numerator;
	int random_noise_of_burst_type_in_bitmap_file_denominator;
	int random_noise_of_burst_type_in_bitmap_file_burst_length;

#if 0
	int f_crc_test;
	std::string crc_test_type;
	long int crc_test_block_length;
	long int crc_test_N;
	int crc_test_k;
#endif


public:
	interface_coding_theory();
	~interface_coding_theory();
	void print_help(
			int argc, std::string *argv, int i, int verbose_level);
	int recognize_keyword(
			int argc, std::string *argv, int i, int verbose_level);
	void read_arguments(
			int argc,
			std::string *argv, int &i, int verbose_level);
	void print();
	void worker(
			int verbose_level);
};


// #############################################################################
// interface_combinatorics.cpp
// #############################################################################

//! interface to the coding theory module


class interface_combinatorics {

	// Section 12.1
	// TABLES/combinatorics_1.csv


	int f_random_permutation;
	int random_permutation_degree;
	std::string random_permutation_fname_csv;

	int f_create_random_k_subsets;
	int create_random_k_subsets_n;
	int create_random_k_subsets_k;
	int create_random_k_subsets_nb;

	int f_read_poset_file;
	std::string read_poset_file_fname;

	int f_grouping;
	double grouping_x_stretch;

	int f_list_parameters_of_SRG;
	int list_parameters_of_SRG_v_max;

	int f_conjugacy_classes_Sym_n;
	int conjugacy_classes_Sym_n_n;

	int f_tree_of_all_k_subsets;
	int tree_of_all_k_subsets_n;
	int tree_of_all_k_subsets_k;

	int f_Delandtsheer_Doyen;
	apps_combinatorics::delandtsheer_doyen_description
		*Delandtsheer_Doyen_description;

	int f_tdo_refinement;
	combinatorics::tactical_decompositions::tdo_refinement_description
		*Tdo_refinement_descr;

	int f_tdo_print;
	std::string tdo_print_fname;

	int f_convert_stack_to_tdo;
	std::string stack_fname;

	int f_maximal_arc_parameters;
	int maximal_arc_parameters_q, maximal_arc_parameters_r;

	int f_arc_parameters;
	int arc_parameters_q, arc_parameters_s, arc_parameters_r;

	int f_pentomino_puzzle;


	// Section 12.1
	// TABLES/combinatorics_2.csv

#if 0
	int f_regular_linear_space_classify;
	apps_combinatorics::regular_linear_space_description *Rls_descr;
#endif


	// undocumented:
	int f_read_solutions_and_tally;
	std::string read_solutions_and_tally_fname;
	int read_solutions_and_tally_sz;


	int f_make_elementary_symmetric_functions;
	int make_elementary_symmetric_functions_n;
	int make_elementary_symmetric_functions_k_max;

	int f_make_elementary_symmetric_function;
	int make_elementary_symmetric_function_n;
	int make_elementary_symmetric_function_k;

	int f_Dedekind_numbers;
	int Dedekind_n_min;
	int Dedekind_n_max;
	int Dedekind_q_min;
	int Dedekind_q_max;

	int f_rank_k_subset;
	int rank_k_subset_n;
	int rank_k_subset_k;
	std::string rank_k_subset_text;

	int f_geometry_builder;
	combinatorics::geometry_builder::geometry_builder_description
		*Geometry_builder_description;


	int f_union;
	std::string union_set_of_sets_fname;
	std::string union_input_fname;
	std::string union_output_fname;

	int f_dot_product_of_columns;
	std::string dot_product_of_columns_fname;

	int f_dot_product_of_rows;
	std::string dot_product_of_rows_fname;

	int f_matrix_multiply_over_Z;
	std::string matrix_multiply_over_Z_label1;
	std::string matrix_multiply_over_Z_label2;

	int f_rowspan_over_R;
	std::string rowspan_over_R_label;

	int f_read_widor;
	std::string read_widor_fname;

	int f_Kaempfer;

	int f_domino_portrait;
	int domino_portrait_D;
	int domino_portrait_s;
	std::string domino_portrait_fname;
	other::graphics::layered_graph_draw_options *domino_portrait_draw_options;



public:
	interface_combinatorics();
	~interface_combinatorics();
	void print_help(
			int argc,
			std::string *argv, int i, int verbose_level);
	int recognize_keyword(
			int argc,
			std::string *argv, int i, int verbose_level);
	void read_arguments(
			int argc,
			std::string *argv, int &i, int verbose_level);
	void print();
	void worker(
			int verbose_level);
	void do_conjugacy_classes_Sym_n(
			int n, int verbose_level);
	void do_conjugacy_classes_Sym_n_file(
			int n, int verbose_level);
	void do_Delandtsheer_Doyen(
			apps_combinatorics::delandtsheer_doyen_description *Descr,
			int verbose_level);

};


// #############################################################################
// interface_cryptography.cpp
// #############################################################################


enum cipher_type { no_cipher_type, substitution, vigenere, affine };

typedef enum cipher_type cipher_type;

//! interface to the cryptography module

class interface_cryptography {

	// ToDo: undocumented:

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

	int f_quadratic_sieve;
	int quadratic_sieve_n;
	int quadratic_sieve_factorbase;
	int quadratic_sieve_x0;


	// Section 10.3
	// TABLES/cryptography_1.csv


	int f_solovay_strassen;
	int solovay_strassen_p;
	int solovay_strassen_a;

	int f_miller_rabin;
	int miller_rabin_p;
	int miller_rabin_nb_times;

	// ToDo: undocumented:
	int f_miller_rabin_text;
	int miller_rabin_text_nb_times;
	std::string miller_rabin_number_text;

	int f_fermat_test;
	int fermat_test_p;
	int fermat_test_nb_times;

	int f_find_pseudoprime;
	int find_pseudoprime_nb_digits;
	int find_pseudoprime_nb_fermat;
	int find_pseudoprime_nb_miller_rabin;
	int find_pseudoprime_nb_solovay_strassen;

	int f_find_strong_pseudoprime;


	int f_RSA_encrypt_text;
	int RSA_block_size;
	std::string RSA_encrypt_text;

	int f_RSA;
	long int RSA_d;
	long int RSA_m;
	std::string RSA_text;

	int f_RSA_setup;
	int RSA_setup_nb_bits;
	int RSA_setup_nb_tests_solovay_strassen;
	int RSA_setup_f_miller_rabin_test;





	// Section 10.1
	// TABLES/number_theoretic_commands.csv


	int f_sift_smooth;
	int sift_smooth_from;
	int sift_smooth_len;
	std::string sift_smooth_factor_base;

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
	~interface_cryptography();
	void print_help(
			int argc,
			std::string *argv, int i, int verbose_level);
	int recognize_keyword(
			int argc,
			std::string *argv, int i, int verbose_level);
	void read_arguments(
			int argc,
			std::string *argv, int &i, int verbose_level);
	void print();
	void worker(
			int verbose_level);


};

// #############################################################################
// interface_povray.cpp
// #############################################################################

//! interface to the povray rendering module


class interface_povray {

	int f_povray;
	other::graphics::povray_job_description *Povray_job_description;

	int f_prepare_frames;
	other::orbiter_kernel_system::prepare_frames *Prepare_frames;

public:
	interface_povray();
	~interface_povray();
	void print_help(
			int argc,
			std::string *argv, int i, int verbose_level);
	int recognize_keyword(
			int argc,
			std::string *argv, int i, int verbose_level);
	void read_arguments(
			int argc,
			std::string *argv, int &i, int verbose_level);
	void print();
	void worker(
			int verbose_level);
};




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
	other::polish::function_polish_description *FP_descr;


	int f_make_table_of_surfaces;
	int f_make_table_of_quartic_curves;

	int f_create_surface_reports;
	std::string create_surface_reports_field_orders_text;

	int f_create_surface_atlas;
	int create_surface_atlas_q_max;

	int f_create_dickson_atlas;




public:


	interface_projective();
	~interface_projective();
	void print_help(
			int argc,
			std::string *argv, int i, int verbose_level);
	int recognize_keyword(
			int argc,
			std::string *argv, int i, int verbose_level);
	void read_arguments(
			int argc,
			std::string *argv, int &i, int verbose_level);
	void print();
	void worker(
			int verbose_level);

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

	int f_assign;
	std::vector<std::string> assign_labels;

	int f_print_symbols;

	int f_with;
	std::vector<std::string> with_labels;

	int f_activity;
	activity_description *Activity_description;





	interface_symbol_table();
	~interface_symbol_table();
	void init(
			orbiter_top_level_session *Orbiter_top_level_session,
			int verbose_level);
	void print_help(
			int argc, std::string *argv, int i, int verbose_level);
	int recognize_keyword(
			int argc,
			std::string *argv, int i, int verbose_level);
	void read_arguments(
			int argc, std::string *argv, int &i, int verbose_level);
	void read_with(
			int argc, std::string *argv, int &i, int verbose_level);
	void read_from(
			int argc, std::string *argv, int &i, int verbose_level);
	void worker(
			int verbose_level);
	void do_assignment(
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);
	void print();
	void print_with();

};



// #############################################################################
// interface_toolkit.cpp
// #############################################################################

//! interface to the general toolkit


class interface_toolkit {

	// interface_toolkit_1.csv:

	int f_create_files_direct;
	std::string create_files_direct_fname_mask;
	std::string create_files_direct_content_mask;
	std::vector<std::string> create_files_direct_labels;

	int f_create_files;
	other::orbiter_kernel_system::create_file_description
		*Create_file_description;

	int f_save_matrix_csv;
	std::string save_matrix_csv_label;

	int f_csv_file_tally;
	std::string csv_file_tally_fname;

	int f_collect_stats;
	std::string collect_stats_fname_mask;
	std::string collect_stats_fname_out;
	int collect_stats_first;
	int collect_stats_last;
	int collect_stats_step;

	int f_csv_file_select_rows;
	std::string csv_file_select_rows_fname;
	std::string csv_file_select_rows_text;

	int f_csv_file_select_rows_by_file;
	std::string csv_file_select_rows_by_file_fname;
	std::string csv_file_select_rows_by_file_select;

	int f_csv_file_select_rows_complement;
	std::string csv_file_select_rows_complement_fname;
	std::string csv_file_select_rows_complement_text;

	int f_csv_file_split_rows_modulo;
	std::string csv_file_split_rows_modulo_fname;
	int csv_file_split_rows_modulo_n;

	int f_csv_file_select_cols;
	std::string csv_file_select_cols_fname;
	std::string csv_file_select_cols_fname_append;
	std::string csv_file_select_cols_text;

	int f_csv_file_select_cols_by_label;
	std::string csv_file_select_cols_by_label_fname;
	std::string csv_file_select_cols_by_label_fname_append;
	std::string csv_file_select_cols_by_label_text;


	int f_csv_file_select_rows_and_cols;
	std::string csv_file_select_rows_and_cols_fname;
	std::string csv_file_select_rows_and_cols_R_text;
	std::string csv_file_select_rows_and_cols_C_text;

	int f_csv_file_sort_each_row;
	std::string csv_file_sort_each_row_fname;

	int f_csv_file_sort_rows;
	std::string csv_file_sort_rows_fname;

	int f_csv_file_sort_rows_and_remove_duplicates;
	std::string csv_file_sort_rows_and_remove_duplicates_fname;


	// interface_toolkit_2.csv:

	int f_csv_file_join;
	std::vector<std::string> csv_file_join_fname;
	std::vector<std::string> csv_file_join_identifier;

	int f_csv_file_concatenate;
	std::string csv_file_concatenate_fname_out;
	std::vector<std::string> csv_file_concatenate_fname_in;

	int f_csv_file_concatenate_from_mask;
	int csv_file_concatenate_from_mask_N_min;
	int csv_file_concatenate_from_mask_N_max;
	std::string csv_file_concatenate_from_mask_mask;
	std::string csv_file_concatenate_from_mask_fname_out;

	int f_csv_file_extract_column_to_txt;
	std::string csv_file_extract_column_to_txt_fname;
	std::string csv_file_extract_column_to_txt_col_label;


	int f_csv_file_latex;
	int f_produce_latex_header;
	std::string csv_file_latex_fname;

	// undocumented:
	int f_prepare_tables_for_users_guide;
	std::vector<std::string> prepare_tables_for_users_guide_fname;

	// undocumented:
	int f_prepare_general_tables_for_users_guide;
	std::vector<std::string> prepare_general_tables_for_users_guide_fname;

	// undocumented:
	int f_grade_statistic_from_csv;
	std::string grade_statistic_from_csv_fname;
	std::string grade_statistic_from_csv_m1_label;
	std::string grade_statistic_from_csv_m2_label;
	std::string grade_statistic_from_csv_final_label;
	std::string grade_statistic_from_csv_oracle_grade_label;

	int f_draw_matrix;
	other::graphics::draw_bitmap_control *Draw_bitmap_control;

	int f_reformat;
	std::string reformat_fname_in;
	std::string reformat_fname_out;
	int reformat_nb_cols;

	int f_split_by_values;
	std::string split_by_values_fname_in;

	int f_change_values;
	std::string change_values_fname_in;
	std::string change_values_fname_out;
	std::string change_values_function_input;
	std::string change_values_function_output;

	int f_store_as_csv_file;
	std::string store_as_csv_file_fname;
	int store_as_csv_file_m;
	int store_as_csv_file_n;
	std::string store_as_csv_file_data;

	int f_mv;
	std::string mv_a;
	std::string mv_b;

	int f_system;
	std::string system_command;


	int f_loop;
	int loop_start_idx;
	int loop_end_idx;
	std::string loop_variable;
	int loop_from;
	int loop_to;
	int loop_step;
	std::string *loop_argv;

	int f_loop_over;
	int loop_over_start_idx;
	int loop_over_end_idx;
	std::string loop_over_variable;
	std::string loop_over_domain;
	std::string *loop_over_argv;


	int f_plot_function;
	std::string plot_function_fname;

	int f_draw_projective_curve;
	other::graphics::draw_projective_curve_description
		*Draw_projective_curve_description;

	int f_tree_draw;
	other::graphics::tree_draw_options *Tree_draw_options;

	int f_extract_from_file;
	std::string extract_from_file_fname;
	std::string extract_from_file_label;
	std::string extract_from_file_target_fname;

	int f_extract_from_file_with_tail;
	std::string extract_from_file_with_tail_fname;
	std::string extract_from_file_with_tail_label;
	std::string extract_from_file_with_tail_tail;
	std::string extract_from_file_with_tail_target_fname;

	int f_serialize_file_names;
	std::string serialize_file_names_fname;
	std::string serialize_file_names_output_mask;

	int f_save_4_bit_data_file;
	std::string save_4_bit_data_file_fname;
	std::string save_4_bit_data_file_vector_data;

	int f_gnuplot;
	std::string gnuplot_file_fname;
	std::string gnuplot_title;
	std::string gnuplot_label_x;
	std::string gnuplot_label_y;

	int f_compare_columns;
	std::string compare_columns_fname;
	std::string compare_columns_column1;
	std::string compare_columns_column2;

	int f_gcd_worksheet;
	int gcd_worksheet_nb_problems;
	int gcd_worksheet_N;
	int gcd_worksheet_key;

	int f_draw_layered_graph;
	std::string draw_layered_graph_fname;
	other::graphics::layered_graph_draw_options *Layered_graph_draw_options;

	// undocumented:
	int f_read_gedcom;
	std::string read_gedcom_fname;

	int f_read_xml;
	std::string read_xml_fname;
	std::string read_xml_crossref_fname;

	int f_read_column_and_tally;
	std::string read_column_and_tally_fname;
	std::string read_column_and_tally_col_header;

	int f_intersect_with;
	std::string intersect_with_vector;
	std::string intersect_with_data;


public:


	interface_toolkit();
	~interface_toolkit();
	void print_help(
			int argc,
			std::string *argv, int i, int verbose_level);
	int recognize_keyword(
			int argc,
			std::string *argv,
			int i, int verbose_level);
	void read_arguments(
			int argc,
			std::string *argv, int &i, int verbose_level);
	void print();
	void worker(
			int verbose_level);

};

// #############################################################################
// orbiter_command.cpp
// #############################################################################



//! a single command in the Orbiter dash code language


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
	void parse(
			orbiter_top_level_session *Orbiter_top_level_session,
			int argc, std::string *Argv, int &i, int verbose_level);
	void execute(
			int verbose_level);
	void print();

};


// #############################################################################
// orbiter_top_level_session.cpp
// #############################################################################



//! The top level orbiter session is responsible for the command line interface and the program execution and for the orbiter_session


class orbiter_top_level_session {

public:

	other::orbiter_kernel_system::orbiter_session *Orbiter_session;

	orbiter_top_level_session();
	~orbiter_top_level_session();
	void execute_command_line(
			int argc, const char **argv, int verbose_level);
	int startup_and_read_arguments(
			int argc,
			std::string *argv, int i0, int verbose_level);
	void handle_everything(
			int argc, std::string *Argv, int i, int verbose_level);
	void parse_and_execute(
			int argc, std::string *Argv, int i, int verbose_level);
	void parse(
			int argc, std::string *Argv, int &i,
			std::vector<void * > &program, int verbose_level);
	void *get_object(
			int idx);
	layer1_foundations::other::orbiter_kernel_system::symbol_table_object_type get_object_type(
			int idx);
	int find_symbol(
			std::string &label);
	void find_symbols(
			std::vector<std::string> &Labels, int *&Idx);
	void print_symbol_table();
	void add_symbol_table_entry(
			std::string &label,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb,
			int verbose_level);
	groups::any_group *get_any_group(
			std::string &label);
	projective_geometry::projective_space_with_action
		*get_object_of_type_projective_space(
			std::string &label);
	spreads::spread_table_with_selection
		*get_object_of_type_spread_table(
				std::string &label);
	packings::packing_classify
		*get_packing_classify(
				std::string &label);
	poset_classification::poset_classification_control
		*get_poset_classification_control(
				std::string &label);
	poset_classification::poset_classification_report_options
		*get_poset_classification_report_options(
				std::string &label);
	apps_geometry::arc_generator_description
		*get_object_of_type_arc_generator_control(
				std::string &label);
	poset_classification::poset_classification_activity_description
		*get_object_of_type_poset_classification_activity(
				std::string &label);
	void get_vector_or_set(std::string &label,
			long int *&Pts, int &nb_pts, int verbose_level);
	apps_algebra::vector_ge_builder
		*get_object_of_type_vector_ge(
				std::string &label);
	orthogonal_geometry_applications::orthogonal_space_with_action
		*get_object_of_type_orthogonal_space_with_action(
				std::string &label);
	spreads::spread_create
		*get_object_of_type_spread(
				std::string &label);
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create
		*get_object_of_type_cubic_surface(
				std::string &label);
	apps_coding_theory::create_code
		*get_object_of_type_code(
				std::string &label);
	combinatorics::graph_theory::colored_graph
		*get_object_of_type_graph(
				std::string &label);
	orthogonal_geometry_applications::orthogonal_space_with_action
		*get_orthogonal_space(
				std::string &label);
	orbits::orbits_create
		*get_orbits(
			std::string &label);
	canonical_form::variety_object_with_action
		*get_variety(
				std::string &label);
	canonical_form::combinatorial_object_with_properties
		*get_combo_with_group(
				std::string &label);
	isomorph::isomorph_arguments
		*get_isomorph_arguments(
				std::string &label);

};


void free_symbol_table_entry_callback(
		other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb, int verbose_level);
geometry::projective_geometry::projective_space *get_projective_space_low_level_function(
		void *ptr);



// #############################################################################
// symbol_definition.cpp
// #############################################################################

//! definition of an orbiter symbol


class symbol_definition {

public:
	interface_symbol_table *Sym;


	std::string define_label;

	// TABLES/orbiter_objects1.csv



	int f_finite_field;
	algebra::field_theory::finite_field_description
		*Finite_field_description;

	int f_polynomial_ring;
	algebra::ring_theory::polynomial_ring_description
		*Polynomial_ring_description;

	int f_projective_space;
	projective_geometry::projective_space_with_action_description
		*Projective_space_with_action_description;

	int f_orthogonal_space;
	orthogonal_geometry_applications::orthogonal_space_with_action_description
		*Orthogonal_space_with_action_description;

	int f_BLT_set_classifier;
	orthogonal_geometry_applications::blt_set_classify_description
		*Blt_set_classify_description;

	int f_spread_classifier;
	spreads::spread_classify_description
		*Spread_classify_description;

	int f_linear_group;
	group_constructions::linear_group_description
		*Linear_group_description;

	int f_permutation_group;
	group_constructions::permutation_group_description
		*Permutation_group_description;

	int f_group_modification;
	group_constructions::group_modification_description
		*Group_modification_description;

	int f_collection;
	std::string list_of_objects;

	int f_geometric_object;
	std::string geometric_object_projective_space_label;
	geometry::other_geometry::geometric_object_description
		*Geometric_object_description;

	int f_graph;
	apps_graph_theory::create_graph_description
		*Create_graph_description;

	int f_code;
	apps_coding_theory::create_code_description
		*Create_code_description;

	int f_spread;
	spreads::spread_create_description
		*Spread_create_description;

	int f_cubic_surface;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description
		*Surface_Descr;

	int f_quartic_curve;
	applications_in_algebraic_geometry::quartic_curves::quartic_curve_create_description
		*Quartic_curve_descr;

	int f_BLT_set;
	orthogonal_geometry_applications::BLT_set_create_description
		*BLT_Set_create_description;

	int f_translation_plane;
	std::string translation_plane_spread_label;
	std::string translation_plane_group_n_label;
	std::string translation_plane_group_np1_label;


	int f_spread_table;
	std::string spread_table_label_PA;
	int dimension_of_spread_elements;
	std::string spread_selection_text;
	std::string spread_table_prefix;
	std::string spread_table_control;

	int f_packing_classify;
	std::string packing_classify_label_PA3;
	std::string packing_classify_label_PA5;
	std::string packing_classify_label_spread_table;


	int f_packing_was;
	std::string packing_was_label_packing_classify;
	packings::packing_was_description * packing_was_descr;

	int f_packing_was_choose_fixed_points;
	std::string packing_with_assumed_symmetry_label;
	int packing_with_assumed_symmetry_choose_fixed_points_clique_size;
	std::string packing_was_choose_fixed_points_control_label;
	//poset_classification::poset_classification_control
	//	*packing_with_assumed_symmetry_choose_fixed_points_control;


	// TABLES/orbiter_objects2.csv

	int f_packing_long_orbits;
	std::string packing_long_orbits_choose_fixed_points_label;
	packings::packing_long_orbits_description
		* Packing_long_orbits_description;

	int f_graph_classification;
	apps_graph_theory::graph_classify_description
		* Graph_classify_description;

	int f_diophant;
	combinatorics::solvers::diophant_description
		*Diophant_description;

	int f_design;
	apps_combinatorics::design_create_description
		*Design_create_description;

	int f_design_table;
	std::string design_table_label_design;
	std::string design_table_label;
	std::string design_table_group;


	int f_large_set_was;
	std::string  large_set_was_label_design_table;
	apps_combinatorics::large_set_was_description
		*large_set_was_descr;

	int f_set;
	other::data_structures::set_builder_description
		*Set_builder_description;

	int f_vector;
	other::data_structures::vector_builder_description
		*Vector_builder_description;

	int f_symbolic_object;
	algebra::expression_parser::symbolic_object_builder_description
		*Symbolic_object_builder_description;

	int f_combinatorial_object;
	combinatorics::canonical_form_classification::data_input_stream_description
		*Data_input_stream_description;

	int f_geometry_builder;
	combinatorics::geometry_builder::geometry_builder_description
		*Geometry_builder_description;

	int f_vector_ge;
	data_structures_groups::vector_ge_description
		*Vector_ge_description;

	int f_action_on_forms;
	apps_algebra::action_on_forms_description
		*Action_on_forms_descr;

	int f_orbits;
	orbits::orbits_create_description
		*Orbits_create_description;

	int f_poset_classification_control;
	poset_classification::poset_classification_control
		*Poset_classification_control;

	int f_poset_classification_report_options;
	poset_classification::poset_classification_report_options
		*Poset_classification_report_options;

	// new:
	int f_draw_options;
	other::graphics::layered_graph_draw_options *Draw_options;

	// new:
	int f_draw_incidence_structure_options;
	other::graphics::draw_incidence_structure_description
		*Draw_incidence_structure_description;


	int f_arc_generator_control;
	apps_geometry::arc_generator_description
		*Arc_generator_control;

	int f_poset_classification_activity;
	poset_classification::poset_classification_activity_description
		*Poset_classification_activity;

	int f_crc_code;
	combinatorics::coding_theory::crc_code_description *Crc_code_description;

	int f_mapping;
	apps_geometry::mapping_description *Mapping_description;

	int f_variety;
	geometry::algebraic_geometry::variety_description *Variety_description;

	int f_isomorph_arguments;
	isomorph::isomorph_arguments *Isomorph_arguments;



	symbol_definition();
	~symbol_definition();
	void read_definition(
			interface_symbol_table *Sym,
			int argc, std::string *argv, int &i, int verbose_level);
	void perform_definition(
			int verbose_level);
	void print();
	void definition_of_finite_field(
			int verbose_level);
	void definition_of_polynomial_ring(
			int verbose_level);
	void definition_of_projective_space(
			int verbose_level);
	void print_definition_of_projective_space(
			int verbose_level);
	void definition_of_orthogonal_space(
			int verbose_level);
	void definition_of_BLT_set_classifier(
			int verbose_level);
	void definition_of_spread_classifier(
			int verbose_level);
	void definition_of_linear_group(
			int verbose_level);
	void definition_of_permutation_group(
			int verbose_level);
	void definition_of_modified_group(
			int verbose_level);
	void definition_of_geometric_object(
			int verbose_level);
	void definition_of_collection(
			std::string &list_of_objects,
			int verbose_level);
	void definition_of_graph(
			int verbose_level);
	void definition_of_code(
			int verbose_level);
	void definition_of_spread(
			int verbose_level);
	void definition_of_cubic_surface(
			int verbose_level);
	void definition_of_quartic_curve(
			int verbose_level);
	void definition_of_BLT_set(
			int verbose_level);
	void definition_of_translation_plane(
			int verbose_level);
	void definition_of_spread_table(
			int verbose_level);
	void definition_of_packing_classify(
			int verbose_level);
	void definition_of_packing_was(
			int verbose_level);
	void definition_of_packing_was_choose_fixed_points(
			int verbose_level);
	void definition_of_packing_long_orbits(
			int verbose_level);
	void definition_of_graph_classification(
			int verbose_level);
	void definition_of_diophant(
			int verbose_level);
	void definition_of_design(
			int verbose_level);
	void definition_of_design_table(
			int verbose_level);
	void definition_of_large_set_was(
			int verbose_level);
	void definition_of_set(
			int verbose_level);
	void definition_of_vector(
			std::string &label,
			other::data_structures::vector_builder_description *Descr,
			int verbose_level);
	void definition_of_symbolic_object(
			std::string &label,
			algebra::expression_parser::symbolic_object_builder_description *Descr,
			int verbose_level);
	void definition_of_combinatorial_object(
			int verbose_level);
	void do_geometry_builder(
			int verbose_level);
	void load_finite_field_PG(
			int verbose_level);
	algebra::field_theory::finite_field *get_or_create_finite_field(
			std::string &input_q,
			int verbose_level);
	void definition_of_vector_ge(
			int verbose_level);
	void definition_of_action_on_forms(
			int verbose_level);
	void definition_of_orbits(
			int verbose_level);
	void definition_of_poset_classification_control(
			int verbose_level);
	void definition_of_poset_classification_report_options(
			int verbose_level);
	void definition_of_draw_options(
			int verbose_level);
	void definition_of_draw_incidence_structure_options(
			int verbose_level);
	void definition_of_arc_generator_control(
			int verbose_level);
	void definition_of_poset_classification_activity(
			int verbose_level);
	void definition_of_crc_code(
			int verbose_level);
	void definition_of_mapping(
			int verbose_level);
	void definition_of_variety(
			int verbose_level);
	void definition_of_isomorph_arguments(
			int verbose_level);


};



// #############################################################################
// global variable:
// #############################################################################



extern user_interface::orbiter_top_level_session
	*The_Orbiter_top_level_session;
	// global top level Orbiter session



}}}


#endif /* SRC_LIB_TOP_LEVEL_INTERFACES_INTERFACES_H_ */
