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
	void apply(apps_coding_theory::create_code *Code, int verbose_level);

};


// #############################################################################
// coding_theoretic_activity_description.cpp
// #############################################################################

//! description of an activity in coding theory


class coding_theoretic_activity_description {

public:

	int f_general_code_binary;
	int general_code_binary_n;
	std::string general_code_binary_label;
	std::string general_code_binary_text;

#if 0
	int f_code_diagram;
	std::string code_diagram_label;
	std::string code_diagram_codewords_text;
	int code_diagram_n;
#endif

#if 0
	int f_code_diagram_from_file;
	std::string code_diagram_from_file_label;
	std::string code_diagram_from_file_codewords_fname;
	int code_diagram_from_file_n;
#endif

#if 0
	int f_long_code;
	int long_code_n;
	std::vector<std::string> long_code_generators;
#endif

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

	int f_make_diagram;

	int f_boolean_function_of_code;

	int f_embellish;
	int embellish_radius;

	int f_metric_balls;
	int radius_of_metric_ball;


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
	std::string crc_encode_file_based_crc_type;
	int crc_encode_file_based_block_length;

#if 0
	int f_crc_new_file_based;
	std::string crc_new_file_based_fname;
#endif

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
	field_theory::finite_field *F;

	int f_has_code;
	apps_coding_theory::create_code *Code;


	coding_theoretic_activity();
	~coding_theoretic_activity();
	void init_field(coding_theoretic_activity_description *Descr,
			field_theory::finite_field *F,
			int verbose_level);
	void init_code(coding_theoretic_activity_description *Descr,
			create_code *Code,
			int verbose_level);
	void perform_activity(int verbose_level);
	void do_diagram(
			coding_theory::code_diagram *Diagram, int verbose_level);


};


// #############################################################################
// crc_process_description.cpp
// #############################################################################

//! a description of a crc process


class crc_process_description {

public:

	int f_code;
	std::string code_label;

	int f_crc_options;
	coding_theory::crc_options_description *Crc_options;



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
	void init(crc_process_description *Descr,
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
	field_theory::finite_field *F;

	int f_has_generator_matrix;
	int *genma; // [k * n]

	int f_has_check_matrix;
	int *checkma; // [nmk * n]
	int n;
	int k;
	int nmk;
	int d;

	coding_theory::create_BCH_code *Create_BCH_code; // if BCH code
	coding_theory::create_RS_code *Create_RS_code; // if RS code


	create_code();
	~create_code();
	void init(
			create_code_description *description,
			int verbose_level);
	void dual_code(int verbose_level);
	void export_magma(std::string &fname, int verbose_level);
	void create_genma_from_checkma(int verbose_level);
	void create_checkma_from_genma(int verbose_level);
	void export_codewords(std::string &fname, int verbose_level);
	void export_codewords_long(std::string &fname, int verbose_level);
	void export_codewords_by_weight(std::string &fname_base, int verbose_level);
	void export_genma(std::string &fname, int verbose_level);
	void export_checkma(std::string &fname, int verbose_level);
	void weight_enumerator(int verbose_level);
	void fixed_code(
		long int *perm, int n,
		int verbose_level);
	void make_diagram(int f_embellish, int embellish_radius,
			int f_metric_balls, int radius_of_metric_ball,
			coding_theory::code_diagram *&Diagram,
			int verbose_level);
	void polynomial_representation_of_boolean_function(int verbose_level);

};



}}}



#endif /* SRC_LIB_TOP_LEVEL_CODING_THEORY_APPS_CODING_THEORY_APPS_H_ */
