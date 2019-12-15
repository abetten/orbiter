// coding_theory.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005


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
		int f_elements_exponential, const char *symbol_for_print,
		int verbose_level);
		// int exponents[9]
	void create_matrix_H_subfield(finite_field *F, finite_field*f,
		int *H_subfield, int *C, int *C_inv, int *M, int m, int n,
		int beta, int beta_q,
		int f_elements_exponential, const char *symbol_for_print,
		const char *symbol_for_print_subfield,
		int f_construction_A, int f_hyperoval, int f_construction_B,
		int verbose_level);
	void tt_field_reduction(finite_field &F, finite_field &f,
		int m, int n, int *M, int *MM, int verbose_level);


	void make_tensor_code_9dimensional_as_point_set(finite_field *F,
		int *&the_set, int &length,
		int verbose_level);
	void make_tensor_code_9_dimensional(int q,
		const char *override_poly_Q, const char *override_poly,
		int f_hyperoval,
		int *&code, int &length,
		int verbose_level);

	// mindist.cpp:
	int mindist(int n, int k, int q, int *G,
		int f_verbose_level, int idx_zero, int idx_one,
		int *add_table, int *mult_table);
	//Main routine for the code minimum distance computation.
	//The tables are only needed if $q = p^f$ with $f > 1$.
	//In the GF(p) case, just pass a NULL pointer.


};



}
}
