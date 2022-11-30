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
namespace coding_theory {


// #############################################################################
// coding_theory_domain.cpp:
// #############################################################################

//! various functions related to coding theory


class coding_theory_domain {
public:

	coding_theory_domain();
	~coding_theory_domain();



	void make_mac_williams_equations(ring_theory::longinteger_object *&M,
			int n, int k, int q, int verbose_level);
	void make_table_of_bounds(
			int n_max, int q, int verbose_level);
	void make_gilbert_varshamov_code(
			int n, int k, int d,
			field_theory::finite_field *F,
			int *&genma, int *&checkma,
			int verbose_level);
	void make_gilbert_varshamov_code_recursion(
			field_theory::finite_field *F,
			int n, int k, int d, long int N_points,
			long int *set, int *f_forbidden, int level, int verbose_level);

	int gilbert_varshamov_lower_bound_for_d(int n, int k, int q, int verbose_level);
	int singleton_bound_for_d(int n, int k, int q, int verbose_level);
	int hamming_bound_for_d(int n, int k, int q, int verbose_level);
	int plotkin_bound_for_d(int n, int k, int q, int verbose_level);
	int griesmer_bound_for_d(int n, int k, int q, int verbose_level);
	int griesmer_bound_for_n(int k, int d, int q, int verbose_level);

	void do_make_macwilliams_system(int q, int n, int k, int verbose_level);
	void make_Hamming_graph_and_write_file(int n, int q,
			int f_projective, int verbose_level);
	void compute_and_print_projective_weights(
			std::ostream &ost, field_theory::finite_field *F, int *M, int n, int k);
	int code_minimum_distance(field_theory::finite_field *F, int n, int k,
		int *code, int verbose_level);
		// code[k * n]
	void codewords_affine(field_theory::finite_field *F, int n, int k,
		int *code, // [k * n]
		long int *codewords, // q^k
		int verbose_level);
	void code_projective_weight_enumerator(field_theory::finite_field *F, int n, int k,
		int *code, // [k * n]
		int *weight_enumerator, // [n + 1]
		int verbose_level);
	void code_weight_enumerator(field_theory::finite_field *F, int n, int k,
		int *code, // [k * n]
		int *weight_enumerator, // [n + 1]
		int verbose_level);
	void code_weight_enumerator_fast(field_theory::finite_field *F, int n, int k,
		int *code, // [k * n]
		int *weight_enumerator, // [n + 1]
		int verbose_level);
	void code_projective_weights(field_theory::finite_field *F, int n, int k,
		int *code, // [k * n]
		int *&weights,
			// will be allocated [N]
			// where N = theta_{k-1}
		int verbose_level);
	void mac_williams_equations(ring_theory::longinteger_object *&M, int n, int k, int q);
	void determine_weight_enumerator();
	void do_weight_enumerator(field_theory::finite_field *F,
			int *M, int m, int n,
			int f_normalize_from_the_left, int f_normalize_from_the_right,
			int verbose_level);
	void do_minimum_distance(field_theory::finite_field *F,
			int *M, int m, int n,
			int verbose_level);

	void do_linear_code_through_basis(
			field_theory::finite_field *F,
			int n,
			long int *basis_set, int k,
			int f_embellish,
			int verbose_level);
	void matrix_from_projective_set(
			field_theory::finite_field *F,
			int n, int k, long int *columns_set_of_size_n,
			int *genma,
			int verbose_level);
	void do_linear_code_through_columns_of_parity_check_projectively(
			field_theory::finite_field *F,
			int n,
			long int *columns_set, int k,
			int verbose_level);
	void do_linear_code_through_columns_of_parity_check(
			field_theory::finite_field *F,
			int n,
			long int *columns_set, int k,
			int *&genma,
			int verbose_level);
	void do_polynomial(
			int n,
			int polynomial_degree,
			int polynomial_nb_vars,
			std::string &polynomial_text,
			int f_embellish,
			int verbose_level);
	void do_sylvester_hadamard(int n,
			int f_embellish,
			int verbose_level);
	void do_long_code(
			int n,
			std::vector<std::string> &long_code_generators_text,
			int f_nearest_codeword,
			std::string &nearest_codeword_text,
			int verbose_level);
	// creates a combinatorics::boolean_function_domain object
	void code_diagram(
			std::string &label,
			long int *Words,
			int nb_words, int n, int f_metric_balls, int radius_of_metric_ball,
			int f_enhance, int radius,
			int verbose_level);
	void investigate_code(long int *Words,
			int nb_words, int n, int f_embellish, int verbose_level);
	// creates a combinatorics::boolean_function_domain object
	void embellish(int *M, int nb_rows, int nb_cols, int i0, int j0, int a, int rad);
	void place_entry(int *M, int nb_rows, int nb_cols, int i, int j, int a);
	void do_it(int n, int r, int a, int c, int seed, int verbose_level);
	void dimensions(int n, int &nb_rows, int &nb_cols);
	void dimensions_N(int N, int &nb_rows, int &nb_cols);
	void print_binary(int n, int *v);
	void convert_to_binary(int n, long int h, int *v);
	int distance(int n, int a, int b);
	void place_binary(long int h, int &i, int &j);
	void place_binary(int *v, int n, int &i, int &j);
	void field_reduction(field_theory::finite_field *FQ, field_theory::finite_field *Fq,
			std::string &label,
			int m, int n, std::string &genma_text,
			int verbose_level);
	// creates a field_theory::subfield_structure object
	void encode_text_5bits(std::string &text,
			std::string &fname, int verbose_level);
	void field_induction(std::string &fname_in,
			std::string &fname_out, int nb_bits, int verbose_level);
	int Hamming_distance(int *v1, int *v2, int n);
	int Hamming_distance_binary(int a, int b, int n);

	// mindist.cpp:
	int mindist(int n, int k, int q, int *G,
		int f_verbose_level, int idx_zero, int idx_one,
		int *add_table, int *mult_table);
	//Main routine for the code minimum distance computation.
	//The tables are only needed if $q = p^f$ with $f > 1$.
	//In the GF(p) case, just pass a NULL pointer.

};

// #############################################################################
// crc_codes.cpp:
// #############################################################################

//! algorithms for CRC codes


class crc_codes {
public:

	crc_codes();
	~crc_codes();

	void find_CRC_polynomials(field_theory::finite_field *F,
			int t, int da, int dc,
			int verbose_level);
	void search_for_CRC_polynomials(int t,
			int da, int *A, int dc, int *C,
			int i, field_theory::finite_field *F,
			long int &nb_sol, std::vector<std::vector<int> > &Solutions,
			int verbose_level);
	void search_for_CRC_polynomials_binary(int t,
			int da, int *A, int dc, int *C, int i,
			long int &nb_sol, std::vector<std::vector<int> > &Solutions,
			int verbose_level);
	int test_all_two_bit_patterns(int da, int *A, int dc, int *C,
			field_theory::finite_field *F, int verbose_level);
	int test_all_three_bit_patterns(int da, int *A, int dc, int *C,
			field_theory::finite_field *F, int verbose_level);
	int test_all_two_bit_patterns_binary(int da, int *A, int dc, int *C,
			int verbose_level);
	int test_all_three_bit_patterns_binary(int da, int *A, int dc, int *C,
			int verbose_level);
	int remainder_is_nonzero(int da, int *A, int db, int *B, field_theory::finite_field *F);
	int remainder_is_nonzero_binary(int da, int *A, int db, int *B);

	uint16_t crc16(const uint8_t *data, size_t size);
	uint32_t crc32(const char *s, size_t n);
	void crc32_test(int block_length, int verbose_level);
	void crc256_test_k_subsets(int message_length, int R, int k, int verbose_level);
	void crc32_remainders(int message_length, int verbose_level);
	void crc32_remainders_compute(int message_length, int R, uint32_t *&Crc, int verbose_level);
	void introduce_errors(
			crc_options_description *Crc_options_description,
			int verbose_level);
	void crc_encode_file_based(std::string &fname_in, std::string &fname_out,
			std::string &crc_type,
			int block_length, int verbose_level);
	void crc16_file_based(std::string &fname_in, std::string &fname_out,
			int block_length, int verbose_level);
	void crc32_file_based(std::string &fname_in, std::string &fname_out,
			int block_length, int verbose_level);
	void crc771_file_based(
			std::string &fname_in,
			std::string &fname_out,
			int verbose_level);
	void check_errors(
			crc_options_description *Crc_options_description,
			int verbose_level);
	void extract_block(
			crc_options_description *Crc_options_description,
			int verbose_level);
	uint16_t NetIpChecksum(uint16_t const *ipHeader, int nWords);
	void CRC_encode_text(field_theory::nth_roots *Nth,
			ring_theory::unipoly_object &CRC_poly,
		std::string &text, std::string &fname,
		int verbose_level);

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
	field_theory::finite_field *F;

	field_theory::nth_roots *Nth;

	ring_theory::unipoly_object *P;

	int *Selection; // [Nth->Cyc->S->nb_sets]
	int *Sel;  // [nb_sel]
	int nb_sel;

	int degree;
	int k;
	int *Genma; // [k * n]
	int *generator_polynomial; // [degree + 1]



	create_BCH_code();
	~create_BCH_code();
	void init(field_theory::finite_field *F, int n, int d, int verbose_level);
	void do_report(int verbose_level);
	void report(std::ostream &ost, int verbose_level);


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
	void make_BCH_code(int n, field_theory::finite_field *F, int d,
			field_theory::nth_roots *&Nth, ring_theory::unipoly_object &P,
			int verbose_level);
	void make_cyclic_code(int n, int q, int t,
			int *roots, int nb_roots, int f_poly, std::string &poly,
			int f_dual, std::string &fname_txt, std::string &fname_csv,
			int verbose_level);
	// this function creates a finite field, using the given polynomial if necessary
	void generator_matrix_cyclic_code(int n,
			int degree, int *generator_polynomial, int *&M);
	void print_polynomial(ring_theory::unipoly_domain &Fq,
			int degree, ring_theory::unipoly_object *coeffs);
	void print_polynomial_tight(std::ostream &ost, ring_theory::unipoly_domain &Fq,
			int degree, ring_theory::unipoly_object *coeffs);
	void field_reduction(int n, int q, int p, int e, int m,
			field_theory::finite_field &Fp, ring_theory::unipoly_domain &Fq,
		int degree, ring_theory::unipoly_object *generator, int *&generator_subfield,
		int f_poly, std::string &poly,
		int verbose_level);
	void BCH_generator_polynomial(
			field_theory::finite_field *F,
			ring_theory::unipoly_object &g, int n,
			int designed_distance, int &bose_distance,
			int &transversal_length, int *&transversal,
			ring_theory::longinteger_object *&rank_of_irreducibles,
			int verbose_level);
	void compute_generator_matrix(ring_theory::unipoly_object a, int *&genma,
		int n, int &k, int verbose_level);
#if 0
	void make_BCH_codes(int n, int q, int t, int b, int f_dual, int verbose_level);
	// this function creates a finite field.
#endif
	void generator_matrix_cyclic_code(field_theory::finite_field *F,
			int n,
			std::string &poly_coeffs,
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
	void init(int verbose_level);
	void add_error(int offset, int error_pattern, int verbose_level);
	int search(int offset, int error_pattern,
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
		int *&H_subfield, int &m, int &n,
		field_theory::finite_field *FQ, field_theory::finite_field *Fq,
		int f_construction_A, int f_hyperoval,
		int f_construction_B, int verbose_level);
	void create_matrix_M(
		int *&M,
		field_theory::finite_field *FQ, field_theory::finite_field *Fq,
		int &m, int &n, int &beta, int &r, int *exponents,
		int f_construction_A, int f_hyperoval, int f_construction_B,
		int f_elements_exponential, std::string &symbol_for_print,
		int verbose_level);
		// int exponents[9]
	void create_matrix_H_subfield(field_theory::finite_field *FQ, field_theory::finite_field *Fq,
		int *H_subfield, int *C, int *C_inv, int *M, int m, int n,
		int beta, int beta_q,
		int f_elements_exponential, std::string &symbol_for_print,
		std::string &symbol_for_print_subfield,
		int f_construction_A, int f_hyperoval, int f_construction_B,
		int verbose_level);
	void tt_field_reduction(field_theory::finite_field &F, field_theory::finite_field &f,
		int m, int n, int *M, int *MM, int verbose_level);


	void make_tensor_code_9dimensional_as_point_set(field_theory::finite_field *F,
		int *&the_set, int &length,
		int verbose_level);
	void make_tensor_code_9_dimensional(int q,
			std::string &override_poly_Q, std::string &override_poly,
			int f_hyperoval,
			int *&code, int &length,
			int verbose_level);

};



}}}



#endif /* ORBITER_SRC_LIB_FOUNDATIONS_CODING_THEORY_CODING_THEORY_H_ */



