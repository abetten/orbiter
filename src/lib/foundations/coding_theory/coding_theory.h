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
namespace foundations {


// #############################################################################
// coding_theory_domain.cpp:
// #############################################################################

//! various functions related to coding theory


class coding_theory_domain {
public:

	coding_theory_domain();
	~coding_theory_domain();

	void twisted_tensor_product_codes(
		int *&H_subfield, int &m, int &n,
		finite_field *F, finite_field *f,
		int f_construction_A, int f_hyperoval,
		int f_construction_B, int verbose_level);
	void create_matrix_M(
		int *&M,
		finite_field *F, finite_field *f,
		int &m, int &n, int &beta, int &r, int *exponents,
		int f_construction_A, int f_hyperoval, int f_construction_B,
		int f_elements_exponential, std::string &symbol_for_print,
		int verbose_level);
		// int exponents[9]
	void create_matrix_H_subfield(finite_field *F, finite_field*f,
		int *H_subfield, int *C, int *C_inv, int *M, int m, int n,
		int beta, int beta_q,
		int f_elements_exponential, std::string &symbol_for_print,
		std::string &symbol_for_print_subfield,
		int f_construction_A, int f_hyperoval, int f_construction_B,
		int verbose_level);
	void tt_field_reduction(finite_field &F, finite_field &f,
		int m, int n, int *M, int *MM, int verbose_level);


	void make_tensor_code_9dimensional_as_point_set(finite_field *F,
		int *&the_set, int &length,
		int verbose_level);
	void make_tensor_code_9_dimensional(int q,
			std::string &override_poly_Q, std::string &override_poly,
			int f_hyperoval,
			int *&code, int &length,
			int verbose_level);
	void make_cyclic_code(int n, int q, int t,
			int *roots, int nb_roots, int f_poly, std::string &poly,
			int f_dual, char *fname, int verbose_level);
	void generator_matrix_cyclic_code(int n,
			int degree, int *generator_polynomial, int *&M);
	void print_polynomial(unipoly_domain &Fq,
			int degree, unipoly_object *coeffs);
	void field_reduction(int n, int q, int p, int e, int m,
		finite_field &Fp, unipoly_domain &Fq,
		int degree, unipoly_object *generator, int *&generator_subfield,
		int f_poly, std::string &poly,
		int verbose_level);


	void make_mac_williams_equations(longinteger_object *&M,
			int n, int k, int q, int verbose_level);
	int singleton_bound_for_d(int n, int k, int q, int verbose_level);
	int hamming_bound_for_d(int n, int k, int q, int verbose_level);
	int plotkin_bound_for_d(int n, int k, int q, int verbose_level);
	int griesmer_bound_for_d(int n, int k, int q, int verbose_level);
	int griesmer_bound_for_n(int k, int d, int q, int verbose_level);
	void BCH_generator_polynomial(
			finite_field *F,
			unipoly_object &g, int n,
			int designed_distance, int &bose_distance,
			int &transversal_length, int *&transversal,
			longinteger_object *&rank_of_irreducibles,
			int verbose_level);
	void compute_generator_matrix(unipoly_object a, int *&genma,
		int n, int &k, int verbose_level);

	void do_make_macwilliams_system(int q, int n, int k, int verbose_level);
	void make_BCH_codes(int n, int q, int t, int b, int f_dual, int verbose_level);
	void make_Hamming_graph_and_write_file(int n, int q,
			int f_projective, int verbose_level);
	void compute_and_print_projective_weights(
			std::ostream &ost, finite_field *F, int *M, int n, int k);
	int code_minimum_distance(finite_field *F, int n, int k,
		int *code, int verbose_level);
		// code[k * n]
	void codewords_affine(finite_field *F, int n, int k,
		int *code, // [k * n]
		int *codewords, // q^k
		int verbose_level);
	void code_projective_weight_enumerator(finite_field *F, int n, int k,
		int *code, // [k * n]
		int *weight_enumerator, // [n + 1]
		int verbose_level);
	void code_weight_enumerator(finite_field *F, int n, int k,
		int *code, // [k * n]
		int *weight_enumerator, // [n + 1]
		int verbose_level);
	void code_weight_enumerator_fast(finite_field *F, int n, int k,
		int *code, // [k * n]
		int *weight_enumerator, // [n + 1]
		int verbose_level);
	void code_projective_weights(finite_field *F, int n, int k,
		int *code, // [k * n]
		int *&weights,
			// will be allocated [N]
			// where N = theta_{k-1}
		int verbose_level);
	void mac_williams_equations(longinteger_object *&M, int n, int k, int q);
	void determine_weight_enumerator();
	void do_weight_enumerator(finite_field *F, int m, int n, std::string &text,
			int f_normalize_from_the_left, int f_normalize_from_the_right,
			int verbose_level);


	// mindist.cpp:
	int mindist(int n, int k, int q, int *G,
		int f_verbose_level, int idx_zero, int idx_one,
		int *add_table, int *mult_table);
	//Main routine for the code minimum distance computation.
	//The tables are only needed if $q = p^f$ with $f > 1$.
	//In the GF(p) case, just pass a NULL pointer.

};



}}


#endif /* ORBITER_SRC_LIB_FOUNDATIONS_CODING_THEORY_CODING_THEORY_H_ */



