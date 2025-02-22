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
// code_modification_description.cpp
// #############################################################################

//! unary operators to modify codes


class code_modification_description {

public:

	// TABLES/code_modification.tex

	int f_dual;

	code_modification_description();
	~code_modification_description();
	int check_and_parse_argument(
		int argc, int &i, std::string *argv,
		int verbose_level);
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();
	void apply(
			apps_coding_theory::create_code *Code, int verbose_level);

};


// #############################################################################
// coding_theoretic_activity_description.cpp
// #############################################################################

//! description of an activity in coding theory


class coding_theoretic_activity_description {

public:

	// TABLSES/coding_theoretic_activity_1.tex

	int f_report;

	int f_general_code_binary;
	int general_code_binary_n;
	std::string general_code_binary_label;
	std::string general_code_binary_text;

	int f_encode_text_5bits;
	std::string encode_text_5bits_input;
	std::string encode_text_5bits_fname;

	int f_field_induction;
	std::string field_induction_fname_in;
	std::string field_induction_fname_out;
	int field_induction_nb_bits;



	int f_weight_enumerator;
	//std::string weight_enumerator_input_matrix;

	int f_minimum_distance;
	std::string minimum_distance_code_label;

	int f_generator_matrix_cyclic_code;
	int generator_matrix_cyclic_code_n;
	std::string generator_matrix_cyclic_code_poly;

	int f_Sylvester_Hadamard_code;
	int Sylvester_Hadamard_code_n;

	int f_NTT;
	int NTT_n;
	int NTT_q;

	int f_fixed_code;
	std::string fixed_code_perm;

	int f_export_magma;
	std::string export_magma_fname;

	int f_export_codewords;
	std::string export_codewords_fname;

	int f_export_codewords_long;
	std::string export_codewords_long_fname;

	int f_export_codewords_by_weight;
	std::string export_codewords_by_weight_fname;

	int f_export_genma;
	std::string export_genma_fname;

	int f_export_checkma;
	std::string export_checkma_fname;

	int f_export_checkma_as_projective_set;
	std::string export_checkma_as_projective_set_fname;


	// TABLSES/coding_theoretic_activity_2.tex


	int f_make_diagram;

	int f_boolean_function_of_code;

	int f_embellish;
	int embellish_radius;

	int f_metric_balls;
	int radius_of_metric_ball;

	int f_Hamming_space_distance_matrix;
	int Hamming_space_distance_matrix_n;

	// TABLSES/coding_theoretic_activity_2.tex


	// CRC stuff:
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

	int f_crc_encode_file_based;
	std::string crc_encode_file_based_fname_in;
	std::string crc_encode_file_based_fname_out;
	std::string crc_encode_file_based_crc_code;

	int f_crc_compare;
	std::string crc_compare_fname_in;
	std::string crc_compare_code1;
	std::string crc_compare_code2;
	int crc_compare_error_weight;
	int crc_compare_nb_tests_per_block;

	int f_crc_compare_read_output_file;
	std::string crc_compare_read_output_file_fname_in;
	int crc_compare_read_output_file_nb_lines;
	std::string crc_compare_read_output_file_crc_code1;
	std::string crc_compare_read_output_file_crc_code2;


	int f_all_errors_of_a_given_weight;
	std::string all_errors_of_a_given_weight_fname_in;
	int all_errors_of_a_given_weight_block_number;
	std::string all_errors_of_a_given_weight_crc_code1;
	std::string all_errors_of_a_given_weight_crc_code2;
	int all_errors_of_a_given_weight_max_weight;


	int f_weight_enumerator_bottom_up;
	std::string weight_enumerator_bottom_up_crc_code;
	int weight_enumerator_bottom_up_max_weight;


	int f_convert_data_to_polynomials;
	std::string convert_data_to_polynomials_fname_in;
	std::string convert_data_to_polynomials_fname_out;
	int convert_data_to_polynomials_block_length;
	int convert_data_to_polynomials_symbol_size;

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

//! an activity for codes or finite fields


class coding_theoretic_activity {

public:

	coding_theoretic_activity_description *Descr;

	int f_has_finite_field;
	algebra::field_theory::finite_field *F;

	int f_has_code;
	apps_coding_theory::create_code *Code;


	coding_theoretic_activity();
	~coding_theoretic_activity();
	void init_field(
			coding_theoretic_activity_description *Descr,
			algebra::field_theory::finite_field *F,
			int verbose_level);
	void init_code(
			coding_theoretic_activity_description *Descr,
			create_code *Code,
			int verbose_level);
	void perform_activity(
			int verbose_level);
	void do_diagram(
			combinatorics::coding_theory::code_diagram *Diagram,
			int verbose_level);


};


// #############################################################################
// crc_process_description.cpp
// #############################################################################

//! a description of a crc process


class crc_process_description {

public:

	// TABLSE/crc_process_description.tex

	int f_code;
	std::string code_label;

	int f_crc_options;
	combinatorics::coding_theory::crc_options_description *Crc_options;



	crc_process_description();
	~crc_process_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// crc_process.cpp
// #############################################################################

//! a crc process


class crc_process {

public:

	crc_process_description *Descr;

	create_code *Code;


	int block_length;
	long int information_length;
	long int check_size;

	long int N;
	long int nb_blocks;
	char *buffer;
	char *check_data;

	crc_process();
	~crc_process();
	void init(
			crc_process_description *Descr,
			int verbose_level);
	void encode_file(
			std::string &fname_in, std::string &fname_out,
			int verbose_level);
	void encode_block(
			long int L,
			int verbose_level);

};



// #############################################################################
// create_code_description.cpp
// #############################################################################

//! a description of a code using command line arguments


class create_code_description {

public:

	// TABLES/create_code.tex

	int f_field;
	std::string field_label;

	int f_generator_matrix;
	std::string generator_matrix_label_genma;

	int f_basis;
	int basis_n;
	std::string basis_label;

	int f_long_code;
	int long_code_n;
	std::vector<std::string> long_code_generators;

	int f_projective_set;
	int projective_set_nmk;
	std::string projective_set_set;

	int f_columns_of_generator_matrix;
	int columns_of_generator_matrix_k;
	std::string columns_of_generator_matrix_set;

	int f_Reed_Muller;
	int Reed_Muller_m;

	int f_BCH;
	int BCH_n;
	int BCH_d;

	int f_Reed_Solomon;
	int Reed_Solomon_n;
	int Reed_Solomon_d;

	int f_Gilbert_Varshamov;
	int Gilbert_Varshamov_n;
	int Gilbert_Varshamov_k;
	int Gilbert_Varshamov_d;

	int f_ttpA;
	std::string ttpA_field_label;

	int f_ttpB;
	std::string ttpB_field_label;

	std::vector<code_modification_description> Modifications;


	create_code_description();
	~create_code_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};

// #############################################################################
// create_code.cpp
// #############################################################################


//! creates a code from a description with create_code_description


class create_code {
public:

	create_code_description *description;

	std::string label_txt;
	std::string label_tex;

	int f_field;
	algebra::field_theory::finite_field *F;

	int f_has_generator_matrix;
	int *genma; // [k * n]

	int f_has_check_matrix;
	int *checkma; // [nmk * n]
	int n;
	int k;
	int nmk;
	int d;

	combinatorics::coding_theory::create_BCH_code *Create_BCH_code; // if BCH code
	combinatorics::coding_theory::create_RS_code *Create_RS_code; // if RS code


	create_code();
	~create_code();
	void init(
			create_code_description *description,
			int verbose_level);
	void dual_code(int verbose_level);
	void export_magma(
			std::string &fname, int verbose_level);
	void create_genma_from_checkma(
			int verbose_level);
	void create_checkma_from_genma(
			int verbose_level);
	void export_codewords(
			std::string &fname, int verbose_level);
	void export_codewords_long(
			std::string &fname, int verbose_level);
	void export_codewords_by_weight(
			std::string &fname_base, int verbose_level);
	void export_genma(
			std::string &fname, int verbose_level);
	void export_checkma(
			std::string &fname, int verbose_level);
	void export_checkma_as_projective_set(
			std::string &fname, int verbose_level);
	void weight_enumerator(
			int verbose_level);
	void fixed_code(
		long int *perm, int n,
		int verbose_level);
	void make_diagram(
			int f_embellish, int embellish_radius,
			int f_metric_balls, int radius_of_metric_ball,
			combinatorics::coding_theory::code_diagram *&Diagram,
			int verbose_level);
	void polynomial_representation_of_boolean_function(
			int verbose_level);
	void report(
			int verbose_level);
	void report2(
			std::ofstream &ost, int verbose_level);

};



}}}



#endif /* SRC_LIB_TOP_LEVEL_CODING_THEORY_APPS_CODING_THEORY_APPS_H_ */
