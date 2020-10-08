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
	int f_cheat_sheet_GF;
	int q;
	int f_classes_GL;
	int d;
	int f_search_for_primitive_polynomial_in_range;
	int p_min, p_max, deg_min, deg_max;
	int f_make_table_of_irreducible_polynomials;
	int deg;
	int f_make_character_table_symmetric_group;
	int f_make_A5_in_PSL_2_q;
	int f_eigenstuff_matrix_direct;
	int f_eigenstuff_matrix_from_file;
	int eigenstuff_n;
	int eigenstuff_q;
	const char *eigenstuff_coeffs;
	std::string eigenstuff_fname;
	int f_young_symmetrizer;
	int young_symmetrizer_n;
	int f_young_symmetrizer_sym_4;
	int f_poset_classification_control;
	poset_classification_control *Control;

public:
	interface_algebra();
	void print_help(int argc, const char **argv, int i, int verbose_level);
	int recognize_keyword(int argc, const char **argv, int i, int verbose_level);
	void read_arguments(int argc, const char **argv, int i0, int verbose_level);
	void worker(orbiter_session *Session, int verbose_level);
	void do_eigenstuff_matrix_direct(
			int n, int q, const char *coeffs_text, int verbose_level);
	void do_eigenstuff_matrix_from_file(
			int n, int q, std::string &fname, int verbose_level);
	void do_linear_group(
			linear_group_description *Descr, int verbose_level);
	void perform_group_theoretic_activity(finite_field *F, linear_group *LG,
			group_theoretic_activity_description *Group_theoretic_activity_description,
			int verbose_level);
	void do_cheat_sheet_GF(int q, int f_poly, std::string &poly, int verbose_level);
	void do_classes_GL(int d, int q, int f_poly, std::string &poly, int verbose_level);
	void do_search_for_primitive_polynomial_in_range(int p_min, int p_max,
			int deg_min, int deg_max, int verbose_level);
	void do_make_table_of_irreducible_polynomials(int deg, int q, int verbose_level);
	void do_make_character_table_symmetric_group(int deg, int verbose_level);
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

public:
	interface_coding_theory();
	void print_help(int argc, const char **argv, int i, int verbose_level);
	int recognize_keyword(int argc, const char **argv, int i, int verbose_level);
	void read_arguments(int argc, const char **argv, int i0, int verbose_level);
	void worker(int verbose_level);
	void do_make_macwilliams_system(int q, int n, int k, int verbose_level);
	void make_BCH_codes(int n, int q, int t, int b, int f_dual, int verbose_level);
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
	const char *fname_prefix;
	int f_process_combinatorial_objects;
	projective_space_job_description *Job;
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
	const char *read_poset_file_fname;
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
	const char *tdo_print_fname;
	int f_create_design;
	design_create_description *Design_create_description;
	int f_convert_stack_to_tdo;
	const char *stack_fname;
	int f_maximal_arc_parameters;
	int maximal_arc_parameters_q, maximal_arc_parameters_r;
	int f_pentomino_puzzle;

	int f_regular_linear_space_classify;
	regular_linear_space_description *Rls_descr;

	int f_create_files;
	create_file_description *Create_file_description;


public:
	interface_combinatorics();
	void print_help(int argc, const char **argv, int i, int verbose_level);
	int recognize_keyword(int argc, const char **argv, int i, int verbose_level);
	void read_arguments(int argc, const char **argv, int i0, int verbose_level);
	void worker(int verbose_level);
	void do_graph_theoretic_activity(
			graph_theoretic_activity_description *Descr, int verbose_level);
	void do_create_graph(
			create_graph_description *Create_graph_description, int verbose_level);
	void do_read_poset_file(const char *fname,
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
	void do_tdo_refinement(tdo_refinement_description *Descr, int verbose_level);
	void do_tdo_print(const char *fname, int verbose_level);
	void do_create_design(design_create_description *Descr, int verbose_level);
	void convert_stack_to_tdo(const char *stack_fname, int verbose_level);
	void do_parameters_maximal_arc(int q, int r, int verbose_level);
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
	const char *RSA_text;
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
	const char *RSA_encrypt_text;
	int f_sift_smooth;
	int sift_smooth_from;
	int sift_smooth_len;
	const char *sift_smooth_factor_base;
	int f_square_root;
	const char *square_root_number;
	int f_square_root_mod;
	const char *square_root_mod_a;
	const char *square_root_mod_m;
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
	const char *miller_rabin_number_text;
	int f_random;
	int random_nb;
	std::string random_fname_csv;
	int f_random_last;
	int random_last_nb;
	int f_affine_sequence;
	int affine_sequence_a;
	int affine_sequence_c;
	int affine_sequence_m;
	int f_EC_Koblitz_encoding;
	const char *EC_message;
	int EC_s;
	int f_EC_points;
	int f_EC_add;
	const char *EC_pt1_text;
	const char *EC_pt2_text;
	int f_EC_cyclic_subgroup;
	int EC_q;
	int EC_b;
	int EC_c;
	const char *EC_pt_text;
	int f_EC_multiple_of;
	int EC_multiple_of_n;
	int f_EC_discrete_log;
	const char *EC_discrete_log_pt_text;
	int f_EC_baby_step_giant_step;
	const char *EC_bsgs_G;
	int EC_bsgs_N;
	const char *EC_bsgs_cipher_text;
	int f_EC_baby_step_giant_step_decode;
	const char *EC_bsgs_A;
	const char *EC_bsgs_keys;
	int f_nullspace;
	int nullspace_q;
	int nullspace_m;
	int nullspace_n;
	const char *nullspace_text;
	int f_RREF;
	int RREF_q;
	int RREF_m;
	int RREF_n;
	const char *RREF_text;
	int f_weight_enumerator;
	int f_normalize_from_the_right;
	int f_normalize_from_the_left;
	int f_transversal;
	int transversal_q;
	const char *transversal_line_1_basis;
	const char *transversal_line_2_basis;
	const char *transversal_point;
	int f_intersection_of_two_lines;
	int intersection_of_two_lines_q;
	const char *line_1_basis;
	const char *line_2_basis;
	int f_trace;
	int trace_q;
	int f_norm;
	int norm_q;
	int f_count_subprimitive;
	int count_subprimitive_Q_max;
	int count_subprimitive_H_max;

public:
	interface_cryptography();
	void print_help(int argc, const char **argv, int i, int verbose_level);
	int recognize_keyword(int argc, const char **argv, int i, int verbose_level);
	void read_arguments(int argc, const char **argv, int i0, int verbose_level);
	void worker(int verbose_level);

	void do_trace(int q, int verbose_level);
	void do_norm(int q, int verbose_level);
	void do_intersection_of_two_lines(int q,
			const char *line_1_basis,
			const char *line_2_basis,
			int f_normalize_from_the_left, int f_normalize_from_the_right,
			int verbose_level);
	void do_transversal(int q,
			const char *line_1_basis,
			const char *line_2_basis,
			const char *point,
			int f_normalize_from_the_left, int f_normalize_from_the_right,
			int verbose_level);
	void do_nullspace(int q, int m, int n, const char *text,
			int f_normalize_from_the_left, int f_normalize_from_the_right,
			int verbose_level);
	void do_RREF(int q, int m, int n, const char *text,
			int f_normalize_from_the_left, int f_normalize_from_the_right,
			int verbose_level);
	void do_weight_enumerator(int q, int m, int n, const char *text,
			int f_normalize_from_the_left, int f_normalize_from_the_right,
			int verbose_level);
	void do_EC_Koblitz_encoding(int q,
			int EC_b, int EC_c, int EC_s,
			const char *pt_text, const char *EC_message,
			int verbose_level);
	void do_EC_points(int q, int EC_b, int EC_c, int verbose_level);
	int EC_evaluate_RHS(finite_field *F, int EC_b, int EC_c, int x);
	// evaluates x^3 + bx + c
	void do_EC_add(int q, int EC_b, int EC_c,
			const char *pt1_text, const char *pt2_text, int verbose_level);
	void do_EC_cyclic_subgroup(int q, int EC_b, int EC_c,
			const char *pt_text, int verbose_level);
	void do_EC_multiple_of(int q, int EC_b, int EC_c,
			const char *pt_text, int n, int verbose_level);
	void do_EC_discrete_log(int q, int EC_b, int EC_c,
			const char *base_pt_text, const char *pt_text, int verbose_level);
	void do_EC_baby_step_giant_step(int EC_q, int EC_b, int EC_c,
			const char *EC_bsgs_G, int EC_bsgs_N, const char *EC_bsgs_cipher_text,
			int verbose_level);
	void do_EC_baby_step_giant_step_decode(int EC_q, int EC_b, int EC_c,
			const char *EC_bsgs_A, int EC_bsgs_N,
			const char *EC_bsgs_cipher_text_T, const char *EC_bsgs_keys,
			int verbose_level);
	void make_affine_sequence(int a, int c, int m, int verbose_level);
	void make_2D_plot(int *orbit, int orbit_len, int cnt,
			int m, int a, int c, int verbose_level);
	void do_random_last(int random_nb, int verbose_level);
	void do_random(int random_nb, std::string &fname_csv, int verbose_level);
	void do_jacobi(int jacobi_top, int jacobi_bottom, int verbose_level);
	void do_solovay_strassen(int p, int a, int verbose_level);
	void do_miller_rabin(int p, int nb_times, int verbose_level);
	void do_fermat_test(int p, int nb_times, int verbose_level);
	void do_find_pseudoprime(int nb_digits, int nb_fermat,
			int nb_miller_rabin, int nb_solovay_strassen, int verbose_level);
	void do_find_strong_pseudoprime(int nb_digits,
			int nb_fermat, int nb_miller_rabin, int verbose_level);
	void do_miller_rabin_text(const char *number_text,
			int nb_miller_rabin, int verbose_level);
	void quadratic_sieve(int n, int factorbase,
			int x0, int verbose_level);
	void calc_log2(std::vector<int> &primes,
			std::vector<int> &primes_log2, int verbose_level);
	void square_root(const char *square_root_number, int verbose_level);
	void square_root_mod(const char *square_root_number,
			const char *mod_number, int verbose_level);
	void reduce_primes(std::vector<int> &primes,
			longinteger_object &M,
			int &f_found_small_factor, int &small_factor,
			int verbose_level);
	void do_sift_smooth(int sift_smooth_from,
			int sift_smooth_len,
			const char *sift_smooth_factor_base, int verbose_level);
	void do_discrete_log(long int y, long int a, long int p, int verbose_level);
	void do_primitive_root(long int p, int verbose_level);
	void do_inverse_mod(long int a, long int n, int verbose_level);
	void do_extended_gcd(int a, int b, int verbose_level);
	void do_power_mod(long int a, long int k, long int n, int verbose_level);
	void do_RSA_encrypt_text(long int RSA_d, long int RSA_m,
			int RSA_block_size, const char * RSA_encrypt_text, int verbose_level);
	void do_RSA(long int RSA_d, long int RSA_m, const char *RSA_text, int verbose_level);
	void affine_cipher(char *ptext, char *ctext, int a, int b);
	void affine_decipher(char *ctext, char *ptext, char *guess);
	void vigenere_cipher(char *ptext, char *ctext, char *key);
	void vigenere_decipher(char *ctext, char *ptext, char *key);
	void vigenere_analysis(char *ctext);
	void vigenere_analysis2(char *ctext, int key_length);
	int kasiski_test(char *ctext, int threshold);
	void print_candidates(char *ctext,
			int i, int h, int nb_candidates, int *candidates);
	void print_set(int l, int *s);
	void print_on_top(char *text1, char *text2);
	void decipher(char *ctext, char *ptext, char *guess);
	void analyze(char *text);
	double friedman_index(int *mult, int n);
	double friedman_index_shifted(int *mult, int n, int shift);
	void print_frequencies(int *mult);
	void single_frequencies(char *text, int *mult);
	void single_frequencies2(char *text, int stride, int n, int *mult);
	void double_frequencies(char *text, int *mult);
	void substition_cipher(char *ptext, char *ctext, char *key);
	char lower_case(char c);
	char upper_case(char c);
	char is_alnum(char c);
	void get_random_permutation(char *p);

};

// #############################################################################
// interface_povray.cpp
// #############################################################################

//! interface to the povray rendering module


class interface_povray {

	int f_povray;
	int f_output_mask;
	const char *output_mask;
	int f_nb_frames_default;
	int nb_frames_default;
	int f_round;
	int round;
	int f_rounds;
	const char *rounds_as_string;
	video_draw_options *Opt;

	// for povray_worker:
	scene *S;
	animate *A;

	int f_prepare_frames;
	prepare_frames *Prepare_frames;

public:
	interface_povray();
	void print_help(int argc, const char **argv, int i, int verbose_level);
	int recognize_keyword(int argc, const char **argv, int i, int verbose_level);
	void read_arguments(int argc, const char **argv, int i0, int verbose_level);
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

	int f_cheat_sheet_PG;
	int n;
	int q;
	int f_decomposition_by_element;
	int decomposition_by_element_power;
	std::string decomposition_by_element_data;
	std::string decomposition_by_element_fname_base;
	int f_canonical_form_PG;
	projective_space_object_classifier_description *Canonical_form_PG_Descr;

	int f_classify_cubic_curves;
	int f_control_arcs;
	poset_classification_control *Control_arcs;

	int f_create_points_on_quartic;
	double desired_distance;

	int f_create_points_on_parabola;

	int f_smooth_curve;

	int f_create_spread;
	spread_create_description *Spread_create_description;

	int f_study_surface;
	int study_surface_q;
	int study_surface_nb;

	int f_move_two_lines_in_hyperplane_stabilizer;
	long int line1_from;
	long int line2_from;
	long int line1_to;
	long int line2_to;

	int f_move_two_lines_in_hyperplane_stabilizer_text;
	std::string line1_from_text;
	std::string line2_from_text;
	std::string line1_to_text;
	std::string line2_to_text;

	int f_make_table_of_surfaces;

	int f_inverse_isomorphism_klein_quadric;
	std::string inverse_isomorphism_klein_quadric_matrix_A6;

public:

	int parabola_N;
	double parabola_a;
	double parabola_b;
	double parabola_c;

	int smooth_curve_N;
	function_polish_description *FP_descr;
	double smooth_curve_t_min;
	double smooth_curve_t_max;
	double smooth_curve_boundary;
	function_polish *smooth_curve_Polish;
	const char *smooth_curve_label;
	int f_create_BLT_set;
	BLT_set_create_description *BLT_set_descr;
	int nb_transform;
	const char *transform_coeffs[1000];
	int f_inverse_transform[1000];




	interface_projective();
	void print_help(int argc, const char **argv, int i, int verbose_level);
	int recognize_keyword(int argc, const char **argv, int i, int verbose_level);
	void read_arguments(int argc, const char **argv, int i0, int verbose_level);
	//int read_canonical_form_arguments(int argc, const char **argv, int i0, int verbose_level);
	void worker(orbiter_session *Session, int verbose_level);
	void do_cheat_sheet_PG(orbiter_session *Session,
			int n, int q,
			int f_decomposition_by_element, int decomposition_by_element_power,
			std::string &decomposition_by_element_data, std::string &fname_base,
			int verbose_level);
	void do_canonical_form_PG(orbiter_session *Session,
			int n, int q, int verbose_level);
	void do_classify_cubic_curves(int q,
			poset_classification_control *Control_six_arcs, int verbose_level);
	void do_create_points_on_quartic(double desired_distance, int verbose_level);
	void do_create_points_on_parabola(double desired_distance, int N,
			double a, double b, double c, int verbose_level);
	void do_smooth_curve(const char *curve_label,
			double desired_distance, int N,
			double t_min, double t_max, double boundary,
			function_polish_description *FP_descr, int verbose_level);
	void do_create_BLT_set(BLT_set_create_description *Descr, int verbose_level);
	void do_create_spread(spread_create_description *Descr, int verbose_level);
	void do_create_surface(surface_create_description *Descr, int verbose_level);
	void do_study_surface(int q, int nb, int verbose_level);
	void do_move_two_lines_in_hyperplane_stabilizer(
			int q,
			long int line1_from, long int line2_from,
			long int line1_to, long int line2_to, int verbose_level);
	void do_move_two_lines_in_hyperplane_stabilizer_text(
			int q,
			std::string line1_from_text, std::string line2_from_text,
			std::string line1_to_text, std::string line2_to_text,
			int verbose_level);
};




}}


#endif /* SRC_LIB_INTERFACES_INTERFACES_H_ */
