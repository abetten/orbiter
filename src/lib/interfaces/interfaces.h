/*
 * interfaces.h
 *
 *  Created on: Apr 3, 2020
 *      Author: betten
 */

#ifndef SRC_LIB_INTERFACES_INTERFACES_H_
#define SRC_LIB_INTERFACES_INTERFACES_H_


using namespace orbiter::foundations;
using namespace orbiter::group_actions;
using namespace orbiter::classification;
using namespace orbiter::discreta;
using namespace orbiter::top_level;


namespace orbiter {

//! classes to interface user input through the command line

namespace interfaces {

class interface_algebra;
class interface_coding_theory;
class interface_combinatorics;
class interface_cryptography;
class interface_povray;
class interface_projective;
class interface_toolkit;
class orbiter_top_level_session;


// #############################################################################
// interface_coding_theory.cpp
// #############################################################################

//! interface to the algebra module


class interface_algebra {

	int f_linear_group;
	linear_group_description *Linear_group_description;
	finite_field *F;
	linear_group *LG;

	int f_group_theoretic_activity;
	group_theoretic_activity_description *Group_theoretic_activity_description;

	int f_finite_field_activity;
	finite_field_activity_description *Finite_field_activity_description;

	int f_poset_classification_control;
	poset_classification_control *Control;

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


	int f_young_symmetrizer;
	int young_symmetrizer_n;
	int f_young_symmetrizer_sym_4;

	int f_draw_mod_n;
	int draw_mod_n;
	std::string draw_mod_n_fname;
	int f_draw_mod_n_inverse;
	int f_draw_mod_n_additive_inverse;
	int f_draw_mod_n_power_cycle;
	int f_draw_mod_n_power_cycle_base;


public:
	interface_algebra();
	void print_help(int argc, std::string *argv, int i, int verbose_level);
	int recognize_keyword(int argc, std::string *argv, int i, int verbose_level);
	int read_arguments(int argc, std::string *argv, int i0, int verbose_level);
	void worker(int verbose_level);
	void do_linear_group(
			linear_group_description *Descr, int verbose_level);
	void perform_group_theoretic_activity(finite_field *F, linear_group *LG,
			group_theoretic_activity_description *Group_theoretic_activity_description,
			int verbose_level);
	void do_character_table_symmetric_group(int deg, int verbose_level);
	void do_make_A5_in_PSL_2_q(int q, int verbose_level);

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
	int f_BCH;
	int f_BCH_dual;
	int BCH_t;
	//int BCH_b;
	int f_Hamming_graph;
	int f_NTT;
	std::string ntt_fname_code;
	int f_draw_matrix;
	std::string fname;
	int box_width;
	int bit_depth; // 8 or 24
	int f_draw_matrix_partition;
	int draw_matrix_partition_width;
	std::string draw_matrix_partition_rows;
	std::string draw_matrix_partition_cols;

	int f_general_code_binary;
	int general_code_binary_n;
	std::string general_code_binary_text;

	int f_linear_code_through_basis;
	int linear_code_through_basis_n;
	std::string linear_code_through_basis_text;

	int f_long_code;
	int long_code_n;
	std::vector<std::string> long_code_generators;

public:
	interface_coding_theory();
	void print_help(int argc, std::string *argv, int i, int verbose_level);
	int recognize_keyword(int argc, std::string *argv, int i, int verbose_level);
	int read_arguments(int argc, std::string *argv, int i0, int verbose_level);
	void worker(int verbose_level);
};


// #############################################################################
// interface_combinatorics.cpp
// #############################################################################

//! interface to the coding theory module


class interface_combinatorics {

	int f_create_combinatorial_object;
	combinatorial_object_description *Combinatorial_object_description;

	int f_diophant;
	diophant_description *Diophant_description;

	int f_diophant_activity;
	diophant_activity_description *Diophant_activity_description;

	int f_save;
	std::string fname_prefix;

	int f_process_combinatorial_objects;
	projective_space_job_description *Job_description;

	int f_bent;
	int bent_n;

	int f_random_permutation;
	int random_permutation_degree;
	std::string random_permutation_fname_csv;

	int f_create_graph;
	colored_graph *CG;
	std::string fname_graph;
	create_graph_description *Create_graph_description;

	int f_read_poset_file;
	std::string read_poset_file_fname;

	int f_grouping;
	double x_stretch;

	int f_graph_theoretic_activity_description;
	graph_theoretic_activity_description *Graph_theoretic_activity_description;

	int f_list_parameters_of_SRG;
	int v_max;

	int f_conjugacy_classes_Sym_n;
	int n;

	int f_tree_of_all_k_subsets;
	int tree_n, tree_k;

	int f_Delandtsheer_Doyen;
	delandtsheer_doyen_description *Delandtsheer_Doyen_description;

	int f_graph_classify;
	graph_classify_description *Graph_classify_description;

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


public:
	interface_combinatorics();
	void print_help(int argc, std::string *argv, int i, int verbose_level);
	int recognize_keyword(int argc, std::string *argv, int i, int verbose_level);
	int read_arguments(int argc, std::string *argv, int i0, int verbose_level);
	void worker(int verbose_level);
	void do_graph_theoretic_activity(
			graph_theoretic_activity_description *Descr, int verbose_level);
	void do_create_graph(
			create_graph_description *Create_graph_description, int verbose_level);
	void do_read_poset_file(std::string &fname,
			int f_grouping, double x_stretch, int verbose_level);
	void do_create_combinatorial_object(int verbose_level);
	void do_diophant(diophant_description *Descr, int verbose_level);
	void do_diophant_activity(diophant_activity_description *Descr, int verbose_level);
	void do_process_combinatorial_object(int verbose_level);
	void do_bent(int n, int verbose_level);
	void do_random_permutation(int deg, std::string &fname_csv, int verbose_level);
	void do_conjugacy_classes_Sym_n(int n, int verbose_level);
	void do_make_tree_of_all_k_subsets(int n, int k, int verbose_level);
	void do_Delandtsheer_Doyen(delandtsheer_doyen_description *Descr, int verbose_level);
	void do_graph_classify(graph_classify_description *Descr, int verbose_level);
	void do_create_design(design_create_description *Descr, int verbose_level);
	void convert_stack_to_tdo(std::string &stack_fname, int verbose_level);
	void do_parameters_maximal_arc(int q, int r, int verbose_level);
	void do_parameters_arc(int q, int s, int r, int verbose_level);
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
	char ptext[1000];
	char ctext[1000];
	char guess[1000];
	char key[1000];
	int f_RSA;
	long int RSA_d;
	long int RSA_m;
	std::string RSA_text;
	int f_primitive_root;
	int primitive_root_p;
	int f_inverse_mod;
	int inverse_mod_a;
	int inverse_mod_n;
	int f_extended_gcd;
	int extended_gcd_a;
	int extended_gcd_b;
	int f_power_mod;
	int power_mod_a;
	int power_mod_k;
	int power_mod_n;
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
	int read_arguments(int argc, std::string *argv, int i0, int verbose_level);
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
	int read_arguments(int argc, std::string *argv, int i0, int verbose_level);
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
	int f_create_surface_atlas;
	int create_surface_atlas_q_max;

	int f_create_dickson_atlas;

#if 0
	int f_create_BLT_set;
	BLT_set_create_description *BLT_set_descr;
#endif

	std::vector<std::string> transform_coeffs;
	std::vector<int> f_inverse_transform;


public:


	interface_projective();
	void print_help(int argc, std::string *argv, int i, int verbose_level);
	int recognize_keyword(int argc, std::string *argv, int i, int verbose_level);
	int read_arguments(int argc, std::string *argv, int i0, int verbose_level);
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
	void do_create_surface_reports(int q_max, int verbose_level);
	void do_create_surface_atlas(int q_max, int verbose_level);
	void do_create_surface_atlas_q_e(int q_max,
			struct table_surfaces_field_order *T, int nb_e, int *Idx, int nb,
			std::string &fname_report_tex,
			int verbose_level);
	void do_create_dickson_atlas(int verbose_level);
	void make_fname_surface_report_tex(std::string &fname, int q, int ocn);
	void make_fname_surface_report_pdf(std::string &fname, int q, int ocn);

};

// #############################################################################
// interface_symbol_table.cpp
// #############################################################################

//! interface to the orbiter symbol table


class interface_symbol_table {

	int f_define;
	std::string define_label;

	int f_finite_field;
	finite_field_description *Finite_field_description;

	int f_projective_space;
	projective_space_with_action_description *Projective_space_with_action_description;

	int f_orthogonal_space;
	orthogonal_space_with_action_description *Orthogonal_space_with_action_description;

	int f_linear_group;
	linear_group_description *Linear_group_description;


	int f_print_symbols;
	int f_with;
	std::vector<std::string> with_labels;

	int f_finite_field_activity;
	finite_field_activity_description *Finite_field_activity_description;

	int f_projective_space_activity;
	projective_space_activity_description *Projective_space_activity_description;

	int f_orthogonal_space_activity;
	orthogonal_space_activity_description *Orthogonal_space_activity_description;

	int f_group_theoretic_activity;
	group_theoretic_activity_description *Group_theoretic_activity_description;


public:


	interface_symbol_table();
	void print_help(int argc, std::string *argv, int i, int verbose_level);
	int recognize_keyword(int argc, std::string *argv, int i, int verbose_level);
	int read_arguments(int argc, std::string *argv, int i0, int verbose_level);
	void read_activity_arguments(int argc,
			std::string *argv, int &i, int verbose_level);
	void worker(orbiter_top_level_session *Orbiter_top_level_session, int verbose_level);
	void definition(orbiter_top_level_session *Orbiter_top_level_session,
			int verbose_level);
	void definition_of_projective_space(orbiter_top_level_session *Orbiter_top_level_session,
			int verbose_level);
	void definition_of_orthogonal_space(orbiter_top_level_session *Orbiter_top_level_session,
			int verbose_level);
	void do_finite_field_activity(orbiter_top_level_session *Orbiter_top_level_session,
			int verbose_level);
	void do_projective_space_activity(
			orbiter_top_level_session *Orbiter_top_level_session,
			int verbose_level);
	void do_orthogonal_space_activity(
			orbiter_top_level_session *Orbiter_top_level_session,
			int verbose_level);
	void do_group_theoretic_activity(orbiter_top_level_session *Orbiter_top_level_session,
			int verbose_level);

};



// #############################################################################
// interface_toolkit.cpp
// #############################################################################

//! interface to the general toolkit


class interface_toolkit {

	int f_csv_file_select_rows;
	std::string csv_file_select_rows_fname;
	std::string csv_file_select_rows_text;

	int f_csv_file_select_cols;
	std::string csv_file_select_cols_fname;
	std::string csv_file_select_cols_text;

	int f_csv_file_select_rows_and_cols;
	std::string csv_file_select_rows_and_cols_fname;
	std::string csv_file_select_rows_and_cols_R_text;
	std::string csv_file_select_rows_and_cols_C_text;


	int f_csv_file_join;
	std::vector<std::string> csv_file_join_fname;
	std::vector<std::string> csv_file_join_identifier;

	int f_csv_file_latex;
	std::string csv_file_latex_fname;


public:


	interface_toolkit();
	void print_help(int argc, std::string *argv, int i, int verbose_level);
	int recognize_keyword(int argc, std::string *argv, int i, int verbose_level);
	int read_arguments(int argc, std::string *argv, int i0, int verbose_level);
	void worker(int verbose_level);

};


// #############################################################################
// orbiter_top_level_session.cpp
// #############################################################################


extern orbiter_top_level_session *The_Orbiter_top_level_session; // global top level Orbiter session

//! The top level orbiter session is reponsible for the command line interface and the program execution and for the orbiter_session


class orbiter_top_level_session {

public:


	orbiter_session *Orbiter_session;

	orbiter_top_level_session();
	~orbiter_top_level_session();
	int startup_and_read_arguments(int argc,
			std::string *argv, int i0);
	void handle_everything(int argc, std::string *Argv, int i, int verbose_level);
	void parse_and_execute(int argc, std::string *Argv, int i, int verbose_level);
	void *get_object(int idx);
	int find_symbol(std::string &label);
	void find_symbols(std::vector<std::string> &Labels, int *&Idx);
	void print_symbol_table();
	void add_symbol_table_entry(std::string &label,
			orbiter_symbol_table_entry *Symb, int verbose_level);

};








}}


#endif /* SRC_LIB_INTERFACES_INTERFACES_H_ */
