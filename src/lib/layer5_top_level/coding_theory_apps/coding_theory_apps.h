/*
 * coding_theory_apps.h
 *
 *  Created on: Jul 30, 2022
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_CODING_THEORY_APPS_CODING_THEORY_APPS_H_
#define SRC_LIB_TOP_LEVEL_CODING_THEORY_APPS_CODING_THEORY_APPS_H_



namespace orbiter {
namespace layer5_applications {
namespace apps_coding_theory {


// #############################################################################
// coding_theoretic_activity_description.cpp
// #############################################################################

//! description of an activity in coding theory


class coding_theoretic_activity_description {

public:

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

	int f_BCH;
	int f_BCH_dual;
	int BCH_n;
	int BCH_q;
	int BCH_t;
	//int BCH_b;

	int f_Hamming_space_distance_matrix;
	int Hamming_space_n;
	int Hamming_space_q;

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
	int metric_ball_radius;


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

	int f_crc32;
	std::string crc32_text;

	int f_crc32_hexdata;
	std::string crc32_hexdata_text;

	int f_crc32_test;
	int crc32_test_block_length;

	int f_crc256_test;
	int crc256_test_message_length;
	int crc256_test_R;
	int crc256_test_k;

	int f_crc32_remainders;
	int crc32_remainders_message_length;

	int f_crc32_file_based;
	std::string crc32_file_based_fname;
	int crc32_file_based_block_length;

	int f_crc_new_file_based;
	std::string crc_new_file_based_fname;

	int f_weight_enumerator;
	std::string weight_enumerator_input_matrix;

	int f_make_gilbert_varshamov_code;
	int make_gilbert_varshamov_code_n;
	int make_gilbert_varshamov_code_k;
	int make_gilbert_varshamov_code_d;


	int f_generator_matrix_cyclic_code;
	int generator_matrix_cyclic_code_n;
	std::string generator_matrix_cyclic_code_poly;

	int f_nth_roots;
	int nth_roots_n;

	int f_make_BCH_code;
	int make_BCH_code_n;
	int make_BCH_code_d;

	int f_make_BCH_code_and_encode;
	std::string make_BCH_code_and_encode_text;
	std::string make_BCH_code_and_encode_fname;

	int f_NTT;
	int NTT_n;
	int NTT_q;

	int f_find_CRC_polynomials;
	int find_CRC_polynomials_nb_errors;
	int find_CRC_polynomials_information_bits;
	int find_CRC_polynomials_check_bits;

	int f_write_code_for_division;
	std::string write_code_for_division_fname;
	std::string write_code_for_division_A;
	std::string write_code_for_division_B;

	int f_polynomial_division_from_file;
	std::string polynomial_division_from_file_fname;
	long int polynomial_division_from_file_r1;

	int f_polynomial_division_from_file_all_k_bit_error_patterns;
	std::string polynomial_division_from_file_all_k_bit_error_patterns_fname;
	int polynomial_division_from_file_all_k_bit_error_patterns_r1;
	int polynomial_division_from_file_all_k_bit_error_patterns_k;



	coding_theoretic_activity_description();
	~coding_theoretic_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};


// #############################################################################
// coding_theoretic_activity.cpp
// #############################################################################

//! an activity for graphs


class coding_theoretic_activity {

public:

	coding_theoretic_activity_description *Descr;
	field_theory::finite_field *F;


	coding_theoretic_activity();
	~coding_theoretic_activity();
	void init(coding_theoretic_activity_description *Descr,
			field_theory::finite_field *F,
			int verbose_level);
	void perform_activity(int verbose_level);


};






}}}



#endif /* SRC_LIB_TOP_LEVEL_CODING_THEORY_APPS_CODING_THEORY_APPS_H_ */
