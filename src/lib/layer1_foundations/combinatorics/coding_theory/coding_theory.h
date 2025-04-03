// coding_theory.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005


#ifndef ORBITER_SRC_LIB_FOUNDATIONS_CODING_THEORY_CODING_THEORY_H_
#define ORBITER_SRC_LIB_FOUNDATIONS_CODING_THEORY_CODING_THEORY_H_


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace coding_theory {


// #############################################################################
// code_diagram.cpp:
// #############################################################################

//! diagram of a code in Hamming space


class code_diagram {
public:

	long int *Words;
	int nb_words;

	std::string label;

	int n;
	long int N;

	int nb_rows, nb_cols;
	int *v;
	int *Index_of_codeword;
	int *Place_values;
	int *Characteristic_function;
	int *Distance;
	int *Distance_H;
		// the distance of a word to the code.
		// Can be used to detect deep holes.

	code_diagram();
	~code_diagram();

	void init(
			std::string &label,
			long int *Words, int nb_words,
			int n, int verbose_level);
	void place_codewords(
			int verbose_level);
	void place_metric_balls(
			int radius_of_metric_ball,
			int verbose_level);
	void compute_distances(
			int verbose_level);
	void dimensions(
			int n, int &nb_rows, int &nb_cols);
	void place_binary(
			long int h, int &i, int &j);
	void place_binary(
			int *v, int n, int &i, int &j);
	void convert_to_binary(
			int n, long int h, int *v);
	void print_binary(
			int n, int *v);
	void save_distance(
			int verbose_level);
	void save_distance_H(
			int verbose_level);
	void save_diagram(
			int verbose_level);
	void save_char_func(
			int verbose_level);
	void report(
			int verbose_level);

};


// #############################################################################
// coding_theory_domain.cpp:
// #############################################################################

//! various functions related to coding theory


class coding_theory_domain {
public:

	coding_theory_domain();
	~coding_theory_domain();



	void make_mac_williams_equations(
			algebra::ring_theory::longinteger_object *&M,
			int n, int k, int q, int verbose_level);
	void report_macwilliams_system(
			int q, int n, int k, int verbose_level);

	void make_table_of_bounds(
			int n_max, int q, int verbose_level);
	void make_gilbert_varshamov_code(
			int n, int k, int d,
			algebra::field_theory::finite_field *F,
			int *&genma, int *&checkma,
			int verbose_level);
	void make_gilbert_varshamov_code_recursion(
			algebra::field_theory::finite_field *F,
			int n, int k, int d, long int N_points,
			long int *set, int *f_forbidden, int level,
			int verbose_level);

	int gilbert_varshamov_lower_bound_for_d(
			int n, int k, int q, int verbose_level);
	int singleton_bound_for_d(
			int n, int k, int q, int verbose_level);
	int hamming_bound_for_d(
			int n, int k, int q, int verbose_level);
	int plotkin_bound_for_d(
			int n, int k, int q, int verbose_level);
	int griesmer_bound_for_d(
			int n, int k, int q, int verbose_level);
	int griesmer_bound_for_n(
			int k, int d, int q, int verbose_level);

	void make_Hamming_space_distance_matrix(
			int n, algebra::field_theory::finite_field *F,
			int f_projective, int verbose_level);
	void compute_and_print_projective_weights(
			std::ostream &ost, algebra::field_theory::finite_field *F,
			int *M, int n, int k, int verbose_level);
	int code_minimum_distance(
			algebra::field_theory::finite_field *F, int n, int k,
		int *code, int verbose_level);
		// code[k * n]
	void make_codewords_sorted(
			algebra::field_theory::finite_field *F,
			int n, int k,
			int *genma, // [k * n]
			long int *&codewords, // q^k
			long int &N,
			int verbose_level);
	void make_codewords(
			algebra::field_theory::finite_field *F,
			int n, int k,
			int *genma, // [k * n]
			long int *&codewords, // q^k
			long int &N,
			int verbose_level);
	void codewords_affine(
			algebra::field_theory::finite_field *F, int n, int k,
		int *code, // [k * n]
		long int *codewords, // q^k
		int verbose_level);
	void codewords_table(
			algebra::field_theory::finite_field *F,
			int n, int k,
		int *code, // [k * n]
		int *&codewords, // [q^k * n]
		long int &N, // q^k
		int verbose_level);
	void code_projective_weight_enumerator(
			algebra::field_theory::finite_field *F, int n, int k,
		int *code, // [k * n]
		int *weight_enumerator, // [n + 1]
		int verbose_level);
	void code_weight_enumerator(
			algebra::field_theory::finite_field *F, int n, int k,
		int *code, // [k * n]
		int *weight_enumerator, // [n + 1]
		int verbose_level);
	void code_weight_enumerator_fast(
			algebra::field_theory::finite_field *F, int n, int k,
		int *code, // [k * n]
		int *weight_enumerator, // [n + 1]
		int verbose_level);
	void code_projective_weights(
			algebra::field_theory::finite_field *F, int n, int k,
		int *code, // [k * n]
		int *&weights,
			// will be allocated [N]
			// where N = theta_{k-1}
		int verbose_level);
	void mac_williams_equations(
			algebra::ring_theory::longinteger_object *&M,
			int n, int k, int q);
	void determine_weight_enumerator();
	void do_weight_enumerator(
			algebra::field_theory::finite_field *F,
			int *M, int m, int n,
			int f_normalize_from_the_left,
			int f_normalize_from_the_right,
			int verbose_level);
	void do_minimum_distance_Brouwer_Zimmermann(
			algebra::field_theory::finite_field *F,
			int *M, int m, int n,
			int verbose_level);
	void matrix_from_projective_set(
			algebra::field_theory::finite_field *F,
			int n, int k, long int *columns_set_of_size_n,
			int *genma,
			int verbose_level);
	void do_linear_code_through_columns_of_generator_matrix(
			algebra::field_theory::finite_field *F,
			int n,
			long int *columns_set, int k,
			int *&genma,
			int verbose_level);
	void do_polynomial(
			int n,
			int polynomial_degree,
			int polynomial_nb_vars,
			std::string &polynomial_text,
			int verbose_level);
	void do_sylvester_hadamard(
			algebra::field_theory::finite_field *F3,
			int n,
			int verbose_level);
	void field_reduction(
			algebra::field_theory::finite_field *FQ,
			algebra::field_theory::finite_field *Fq,
			std::string &label,
			int m, int n, std::string &genma_text,
			int verbose_level);
	// creates a field_theory::subfield_structure object
	void field_induction(
			std::string &fname_in,
			std::string &fname_out, int nb_bits,
			int verbose_level);
	void encode_text_5bits(
			std::string &text,
			std::string &fname, int verbose_level);
	int Hamming_distance(
			int *v1, int *v2, int n);
	int Hamming_distance_binary(
			int a, int b, int n);
	void fixed_code(
			algebra::field_theory::finite_field *F,
			int n, int k, int *genma,
			long int *perm,
			int *&subcode_genma, int &subcode_k,
			int verbose_level);
	void polynomial_representation_of_boolean_function(
			algebra::field_theory::finite_field *F,
			std::string &label_txt,
			long int *Words,
			int nb_words, int n,
			int verbose_level);
	// creates a combinatorics::boolean_function_domain object


	// coding_theory_domain_crc.cpp
	void crc_encode_file_based(
			std::string &fname_in,
			std::string &fname_out,
			crc_object *Crc_object,
			int verbose_level);
	void crc_simulate_errors(
			std::string &fname_in,
			crc_object *Crc_object1,
			crc_object *Crc_object2,
			int error_pattern_weight,
			int nb_tests_per_block,
			int verbose_level);
	void crc_all_errors_of_a_given_weight(
			std::string &fname_in,
			int block_number,
			crc_object *Crc_object1,
			crc_object *Crc_object2,
			int error_pattern_max_weight,
			int verbose_level);
	void crc_weight_enumerator_bottom_up(
			crc_object *Crc_object,
			int error_pattern_max_weight,
			int f_collect_words,
			int verbose_level);
	void read_error_pattern_from_output_file(
			std::string &fname_in,
			int nb_lines,
			crc_object *Crc_object1,
			crc_object *Crc_object2,
			int verbose_level);


	// mindist.cpp:
	int mindist(
			int n, int k, int q, int *G,
		int f_verbose_level, int idx_zero, int idx_one,
		int *add_table, int *mult_table);
	//Main routine for the code minimum distance computation.
	//The tables are only needed if $q = p^f$ with $f > 1$.
	//In the GF(p) case, just pass a NULL pointer.

};


// #############################################################################
// crc_code_description.cpp:
// #############################################################################

//! a description of a CRC code


class crc_code_description {
public:

	// TABLES/crc_code_description.tex

	int f_type;
	std::string type;

	int f_block_length;
	int block_length;

	crc_code_description();
	~crc_code_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};




// #############################################################################
// crc_codes.cpp:
// #############################################################################

//! algorithms for CRC codes


class crc_codes {
public:

	crc_codes();
	~crc_codes();

	// crc_codes_search.cpp:
	void find_CRC_polynomials(
			algebra::field_theory::finite_field *F,
			int t, int da, int dc,
			int verbose_level);
	void search_for_CRC_polynomials(int t,
			int da, int *A, int dc, int *C,
			int i, algebra::field_theory::finite_field *F,
			long int &nb_sol,
			std::vector<std::vector<int> > &Solutions,
			int verbose_level);
	void search_for_CRC_polynomials_binary(int t,
			int da, int *A, int dc, int *C, int i,
			long int &nb_sol,
			std::vector<std::vector<int> > &Solutions,
			int verbose_level);
	int test_all_two_bit_patterns(
			int da, int *A, int dc, int *C,
			algebra::field_theory::finite_field *F,
			int verbose_level);
	int test_all_three_bit_patterns(
			int da, int *A, int dc, int *C,
			algebra::field_theory::finite_field *F,
			int verbose_level);
	int test_all_two_bit_patterns_binary(
			int da, int *A, int dc, int *C,
			int verbose_level);
	int test_all_three_bit_patterns_binary(
			int da, int *A, int dc, int *C,
			int verbose_level);
	int remainder_is_nonzero(
			int da, int *A, int db, int *B,
			algebra::field_theory::finite_field *F);
	int remainder_is_nonzero_binary(
			int da, int *A, int db, int *B);

	// crc_codes.cpp:
	//uint16_t crc16(const uint8_t *data, size_t size);
	uint32_t crc32(
			const uint8_t *s, size_t n);
	void crc32_test(
			int block_length, int verbose_level);
	void test_crc_object(
			crc_object *Crc, long int Nb_test, int k, int verbose_level);
	void char_vec_zero(
			unsigned char *p, int len);
#if 0
	void crc256_test_k_subsets(
			int message_length, int R, int k, int verbose_level);
#endif
#if 0
	void crc32_remainders(
			int message_length, int verbose_level);
	void crc32_remainders_compute(
			int message_length, int R,
			uint32_t *&Crc, int verbose_level);
#endif
	void introduce_errors(
			crc_options_description *Crc_options_description,
			int verbose_level);
#if 0
	void crc_encode_file_based(
			std::string &fname_in,
			std::string &fname_out,
			std::string &crc_type,
			int block_length, int verbose_level);
	void crc_general_file_based(
			std::string &fname_in, std::string &fname_out,
			CRC_type type,
			int block_length, int verbose_level);
#endif
	void split_binary_file_to_ascii_polynomials_256(
			std::string &fname_in, std::string &fname_out,
			int block_length, int verbose_level);
	enum CRC_type detect_type_of_CRC(
			std::string &crc_type, int verbose_level);
	int get_check_size_in_bytes(
			enum CRC_type type);
	void check_errors(
			crc_options_description *Crc_options_description,
			int verbose_level);
	void extract_block(
			crc_options_description *Crc_options_description,
			int verbose_level);
	uint16_t NetIpChecksum(
			uint16_t const *ipHeader, int nWords);
	void CRC_encode_text(
			algebra::field_theory::nth_roots *Nth,
			algebra::ring_theory::unipoly_object &CRC_poly,
		std::string &text, std::string &fname,
		int verbose_level);

};

// #############################################################################
// crc_object.cpp:
// #############################################################################

enum crc_object_type {
	t_crc_unknown,
	t_crc_alfa,
	t_crc_bravo,
	t_crc_charlie,
	t_crc_Delta,
	t_crc_Echo,
	t_crc_Foxtrot,
	t_crc_crc32,
	t_crc_crc16,
	t_crc_SuperFastHash32,

};

//! a specific CRC code


class crc_object {
public:

	crc_code_description *Descr;

	std::string label_txt;
	std::string label_tex;

	crc_object_type Crc_object_type;

	int Len_total_in_symbols; // code length, often called n
	int Len_total_in_bits; // = Len_total_in_symbols * symbol_set_size_log;
	int Len_total_in_bytes; // = (Len_total_in_bits + 7) / 8;

	int Len_check_in_symbols;
	int Len_check_in_bits;
	int Len_check_in_bytes;

	int Len_info_in_symbols;
	int Len_info_in_bits;
	int Len_info_in_bytes;

	int block_length_in_bytes; // = Len_total_in_bits / 8;
	int info_length_in_bytes;
	int info_length_in_symbols;

	int number_of_symbols_per_byte;
	int symbol_set_size_log;
	int symbol_set_size;
	//int code_length_in_bits; // = Len_total * symbol_set_size_log

	unsigned char *Data; // [Len_total_in_bytes]
	unsigned char *Check; // [Len_total_in_bytes]

	other::data_structures::bitvector *Bitvector;

	crc_object();
	~crc_object();
	void init(
			crc_code_description *Descr, int verbose_level);
	// block_length is needed for crc32
	//void encode_as_bitvector();
	void print();
	long int get_nb_blocks(
			long int N);
	long int get_this_block_size(
			long int N, long int cnt);
	void expand(
			const unsigned char *in, unsigned char *out);
	void compress(
			const unsigned char *in, unsigned char *out);
	void compress_check(
			const unsigned char *in, unsigned char *out);
	void divide(
			const unsigned char *in, unsigned char *out);
	void divide_alfa(
			const unsigned char *in771, unsigned char *out2);
	void divide_bravo(
			const unsigned char *in771, unsigned char *out4);
	void divide_charlie(
			const unsigned char *in771, unsigned char *out12);
	void divide_Delta(
			const unsigned char *in51, unsigned char *out4);
	void divide_Echo(
			const unsigned char *in51, unsigned char *out8);
	void divide_Foxtrot(
			const unsigned char *in15, unsigned char *out4);
	void divide_crc32(
			const uint8_t *s, size_t n, unsigned char *out4);
		// polynomial x^32 + x^26 + x^23 + x^22 + x^16 + x^12 + x^11
		// + x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1
	void SuperFastHash32(
			const unsigned char * data, int len, unsigned char *out4);
	void divide_crc16(
			const uint8_t *data, size_t size, unsigned char *out2);

};


// #############################################################################
// crc_options_description.cpp:
// #############################################################################

//! options for activities involving CRC codes


class crc_options_description {
public:

	int f_input;
	std::string input_fname;

	int f_output;
	std::string output_fname;

	int f_crc_type;
	std::string crc_type;

	int f_block_length;
	int block_length;

	int f_block_based_error_generator;

	int f_file_based_error_generator;
	int file_based_error_generator_threshold;

	int f_nb_repeats;
	int nb_repeats;

	int f_threshold;
	int threshold;

	int f_error_log;
	std::string error_log_fname;

	int f_selected_block;
	int selected_block;


	crc_options_description();
	~crc_options_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// create_BCH_code.cpp:
// #############################################################################

//! to create a BCH code


class create_BCH_code {
public:

	int n;
	int d;
	algebra::field_theory::finite_field *F;

	algebra::field_theory::nth_roots *Nth;

	algebra::ring_theory::unipoly_object *P;

	int *Selection; // [Nth->Cyc->S->nb_sets]
	int *Sel;  // [nb_sel]
	int nb_sel;

	int degree;
	int k;
	int *Genma; // [k * n]
	int *generator_polynomial; // [degree + 1]



	create_BCH_code();
	~create_BCH_code();
	void init(
			algebra::field_theory::finite_field *F,
			int n, int d, int verbose_level);
	void do_report(
			int verbose_level);
	void report(
			std::ostream &ost, int verbose_level);


};

// #############################################################################
// create_RS_code.cpp:
// #############################################################################

//! to create a RS code


class create_RS_code {
public:

	int n;
	int d;
	algebra::field_theory::finite_field *F;

	algebra::ring_theory::unipoly_domain *FX; // polynomial ring over F

	algebra::ring_theory::unipoly_object *P;

	int degree;
	int k;
	int *Genma; // [k * n]
	int *generator_polynomial; // [degree + 1]



	create_RS_code();
	~create_RS_code();
	void init(
			algebra::field_theory::finite_field *F,
			int n, int d, int verbose_level);
	void do_report(
			int verbose_level);
	void report(
			std::ostream &ost, int verbose_level);


};


// #############################################################################
// cyclic_codes.cpp:
// #############################################################################

//! algorithms for cyclic codes


class cyclic_codes {
public:


	cyclic_codes();
	~cyclic_codes();

	// cyclic_codes.cpp:
	void make_cyclic_code(
			int n, int q, int t,
			int *roots, int nb_roots,
			int f_poly, std::string &poly,
			int f_dual,
			std::string &fname_txt,
			std::string &fname_csv,
			int verbose_level);
	// this function creates a finite field,
	// using the given polynomial if necessary
	void generator_matrix_cyclic_code(
			int n,
			int degree, int *generator_polynomial, int *&M);
	void print_polynomial(
			algebra::ring_theory::unipoly_domain &Fq,
			int degree, algebra::ring_theory::unipoly_object *coeffs);
	void print_polynomial_tight(
			std::ostream &ost, algebra::ring_theory::unipoly_domain &Fq,
			int degree, algebra::ring_theory::unipoly_object *coeffs);
	void field_reduction(
			int n, int q, int p, int e, int m,
			algebra::field_theory::finite_field &Fp,
			algebra::ring_theory::unipoly_domain &Fq,
		int degree, algebra::ring_theory::unipoly_object *generator,
		int *&generator_subfield,
		int f_poly, std::string &poly,
		int verbose_level);
	void BCH_generator_polynomial(
			algebra::field_theory::finite_field *F,
			algebra::ring_theory::unipoly_object &g, int n,
			int designed_distance, int &bose_distance,
			int &transversal_length, int *&transversal,
			algebra::ring_theory::longinteger_object *&rank_of_irreducibles,
			int verbose_level);
	void compute_generator_matrix(
			algebra::ring_theory::unipoly_object a, int *&genma,
		int n, int &k, int verbose_level);
	void generator_matrix_cyclic_code(
			algebra::field_theory::finite_field *F,
			int n,
			std::string &poly_coeffs,
			int verbose_level);


};


// #############################################################################
// error_pattern_generator.cpp:
// #############################################################################

//! to create an error pattern for testing a code


class error_pattern_generator {

public:

	crc_object *Crc_object;

	int k;
	int *f_used; // [Crc->Len_total]
	int *A; // [k]
	int *V; // [k]
	unsigned char *Error_in_symbols;
	unsigned char *Error_in_bytes;

	error_pattern_generator();
	~error_pattern_generator();
	void init(
			crc_object *Crc_object,
			int k, int verbose_level);
	long int number_of_bit_error_patters(
			int wt,
			int verbose_level);
	void first_bit_error_pattern_of_given_weight(
			combinatorics::other_combinatorics::combinatorics_domain &Combi,
			other::data_structures::algorithms &Algo,
			other::data_structures::data_structures_global &DataStructures,
			int wt,
			int verbose_level);
	int next_bit_error_pattern_of_given_weight(
			combinatorics::other_combinatorics::combinatorics_domain &Combi,
			other::data_structures::algorithms &Algo,
			other::data_structures::data_structures_global &DataStructures,
			int wt,
			int verbose_level);
	void create_error_pattern(
			int verbose_level);



};


// #############################################################################
// error_repository.cpp:
// #############################################################################

//! to remember the errors in a block

class error_repository {

public:

	int nb_errors;
	int allocated_length;
	int * Error_storage;
		// [nb_errors * 2]
		// offset and XOR pattern

	error_repository();
	~error_repository();
	void init(
			int verbose_level);
	void add_error(
			int offset,
			int error_pattern, int verbose_level);
	int search(
			int offset, int error_pattern,
		int &idx, int verbose_level);

};



// #############################################################################
// ttp_codes.cpp:
// #############################################################################

//! twisted tensor product codes


class ttp_codes {
public:

	ttp_codes();
	~ttp_codes();

	void twisted_tensor_product_codes(
			algebra::field_theory::finite_field *FQ,
			algebra::field_theory::finite_field *Fq,
		int f_construction_A, int f_hyperoval,
		int f_construction_B,
		int *&H_subfield, int &m, int &n,
		int verbose_level);
	void create_matrix_M(
		int *&M,
		algebra::field_theory::finite_field *FQ,
		algebra::field_theory::finite_field *Fq,
		int &m, int &n, int &beta, int &r, int *exponents,
		int f_construction_A, int f_hyperoval, int f_construction_B,
		int f_elements_exponential, std::string &symbol_for_print,
		int verbose_level);
		// int exponents[9]
	void create_matrix_H_subfield(
			algebra::field_theory::finite_field *FQ,
			algebra::field_theory::finite_field *Fq,
		int *H_subfield, int *C, int *C_inv, int *M, int m, int n,
		int beta, int beta_q,
		int f_elements_exponential,
		std::string &symbol_for_print,
		std::string &symbol_for_print_subfield,
		int f_construction_A, int f_construction_B,
		int verbose_level);
	void tt_field_reduction(
			algebra::field_theory::finite_field &F,
			algebra::field_theory::finite_field &f,
		int m, int n, int *M, int *MM, int verbose_level);


	void make_tensor_code_9dimensional_as_point_set(
			algebra::field_theory::finite_field *F,
		int *&the_set, int &length,
		int verbose_level);
	void make_tensor_code_9_dimensional(
			int q,
			std::string &override_poly_Q,
			std::string &override_poly,
			int f_hyperoval,
			int *&code, int &length,
			int verbose_level);


	void do_tensor(
			int q,
			std::string &override_poly_Q,
			std::string &override_poly_q,
		int f_construction_A, int f_hyperoval,
		int f_construction_B, int verbose_level);
	void action_on_code(
			algebra::field_theory::finite_field &F,
			algebra::field_theory::finite_field &f, int m, int n,
		int *M, int *H_subfield,
		int *C, int *C_inv, int *A, int *U,
		int *perm, int verbose_level);
	void test_cyclic(
			algebra::field_theory::finite_field &F,
			algebra::field_theory::finite_field &f,
		int *Aut, int *M, int *H_subfield,
		int *C, int *C_inv, int *U,
		int q, int Q, int m, int n,
		int beta, int verbose_level);
	void is_cyclic(
			algebra::field_theory::finite_field &FQQ,
			algebra::field_theory::finite_field &F,
			algebra::field_theory::finite_field &f,
		int *Aut, int *M, int *H_subfield,
		int *C, int *C_inv, int *U,
		int q, int Q, int m, int n,
		int beta, int a, int b, int c, int d,
		int verbose_level);
	void test_representation(
			algebra::field_theory::finite_field &F,
			algebra::field_theory::finite_field &f, int Q,
		int beta, int m, int n, int *H_subfield);
	void multiply_abcd(
			algebra::field_theory::finite_field &F,
		int a1, int b1, int c1, int d1,
		int a2, int b2, int c2, int d2,
		int &a3, int &b3, int &c3, int &d3);
	int choose_abcd_first(
			algebra::field_theory::finite_field &F,
			int Q, int &a, int &b, int &c, int &d);
	int choose_abcd_next(
			algebra::field_theory::finite_field &F,
			int Q, int &a, int &b, int &c, int &d);
	int choose_abcd_next2(
			algebra::field_theory::finite_field &F,
			int Q, int &a, int &b, int &c, int &d);
	void choose_abcd_at_random(
			algebra::field_theory::finite_field &F,
			int Q, int &a, int &b, int &c, int &d);
	int compute_mindist(
			algebra::field_theory::finite_field &f,
			int m, int n,
			int *generator_matrix, int verbose_level);
	int abcd_term(
			algebra::field_theory::finite_field &f,
			int a, int b, int c, int d,
			int e1, int e2, int e3, int e4);
	void do_other_stuff(
			algebra::field_theory::finite_field *F,
			algebra::field_theory::finite_field *f,
			int beta, int beta_q,
		int *M, int *C, int *C_inv, int *H_subfield,
		int m, int n, int r,
		int f_elements_exponential,
		std::string &symbol_for_print,
		std::string &symbol_for_print_subfield,
		int f_construction_A, int f_hyperoval,
		int f_construction_B,
		int verbose_level);
	void int_submatrix_all_rows(
			int *A, int m, int n,
		int nb_cols, int *cols, int *B);


};



}}}}



#endif /* ORBITER_SRC_LIB_FOUNDATIONS_CODING_THEORY_CODING_THEORY_H_ */



