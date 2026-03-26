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

//! a description of a linear code using command line arguments


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

	int f_nonlinear_code;
	int nonlinear_code_n;
	std::string nonlinear_code_generators;

	int f_nonlinear_code_long;
	int nonlinear_code_long_n;
	std::vector<std::string> nonlinear_code_long_generators;

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


//! creates a linear code from a description with create_code_description


class create_code {
public:

	create_code_description *description;

	std::string label_txt;
	std::string label_tex;

	int f_field;
	algebra::field_theory::finite_field *F;

	int f_nonlinear;

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
	void all_pairwise_distances(
			std::string &fname, int verbose_level);
	void all_external_distances(
			std::string &fname, int verbose_level);
	void make_codewords(
			long int &N,
			long int *&codewords,
			int verbose_level);
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
