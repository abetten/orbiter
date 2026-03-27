/*
 * activities.h
 *
 *  Created on: Mar 22, 2026
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER6_USER_INTERFACE_ACTIVITIES_ACTIVITIES_H_
#define SRC_LIB_LAYER6_USER_INTERFACE_ACTIVITIES_ACTIVITIES_H_



namespace orbiter {
namespace layer6_user_interface {
namespace activities {


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
	std::string Draw_mod_n_options;
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

	int f_smith_normal_form_from_the_left_only;
	std::string smith_normal_form_from_the_left_only_matrix;



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
	layer5_applications::apps_combinatorics::delandtsheer_doyen_description
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

	int f_q_binomial;
	int q_binomial_n;
	int q_binomial_k;
	int q_binomial_q;

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
	other::graphics::draw_options *domino_portrait_draw_options;

	int f_test_if_distance_regular_graph;
	std::string test_if_distance_regular_graph_fname;


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
			layer5_applications::apps_combinatorics::delandtsheer_doyen_description *Descr,
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

	// ToDo: undocumented:


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


	// ToDo undocumented (all)


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
	layer5_applications::user_interface::core_system::orbiter_top_level_session *Orbiter_top_level_session;

	int f_define;
	layer5_applications::user_interface::core_system::symbol_definition *Symbol_definition;

	int f_assign;
	std::vector<std::string> assign_labels;

	int f_print_symbols;

	int f_with;
	std::vector<std::string> with_labels;

	int f_activity;
	layer6_user_interface::control_everything::activity_description *Activity_description;





	interface_symbol_table();
	~interface_symbol_table();
	void init(
			layer5_applications::user_interface::core_system::orbiter_top_level_session *Orbiter_top_level_session,
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

	int f_tally_column;
	std::string tally_column_fname;
	std::string tally_column_column;

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
	int csv_file_concatenate_from_mask_N_length;
	std::string csv_file_concatenate_from_mask_mask;
	std::string csv_file_concatenate_from_mask_fname_out;

	int f_csv_file_extract_column_to_txt;
	std::string csv_file_extract_column_to_txt_fname;
	std::string csv_file_extract_column_to_txt_col_label;

	int f_csv_file_filter;
	std::string csv_file_filter_fname;
	std::string csv_file_filter_col;
	std::string csv_file_filter_value;

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

	int f_cp;
	std::string cp_a;
	std::string cp_b;

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


	// interface_toolkit_3.csv:

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
	other::graphics::draw_options *Layered_graph_draw_options;

	int f_draw_factor_poset;
	std::string draw_factor_poset_graph_fname;
	other::graphics::draw_options *Factor_poset_draw_options;

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

	int f_make_set_of_sets;
	std::string make_set_of_sets_fname_in;
	std::string make_set_of_sets_new_col_label;
	std::string make_set_of_sets_list;

	int f_copy_and_edit;
	std::string copy_and_edit_input_file;
	std::string copy_and_edit_output_mask;
	std::string copy_and_edit_parameter_values;
	std::string copy_and_edit_search_and_replace;


	int f_join_columns;
	std::string join_columns_file_in;
	std::string join_columns_file_out;
	std::string join_columns_column1;
	std::string join_columns_column2;

	int f_decomposition_matrix;
	std::string decomposition_matrix_fname;
	std::string decomposition_matrix_po_label;
	std::string decomposition_matrix_f_fst_label;
	std::string decomposition_matrix_iso_idx_label;
	std::string decomposition_matrix_n1_label;
	std::string decomposition_matrix_n2_label;
	std::string decomposition_matrix_n3_label;

	int f_stats;
	std::string stats_fname_base;

	int f_intersect_set_of_sets;
	std::string intersect_set_of_sets_fname;
	std::string intersect_set_of_sets_column;

	int f_eliminate_duplicate_columns;
	std::string eliminate_duplicate_columns_fname;


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



}}}



#endif /* SRC_LIB_LAYER6_USER_INTERFACE_ACTIVITIES_ACTIVITIES_H_ */
