/*
 * special_functions.h
 *
 *  Created on: Dec 1, 2024
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER1_FOUNDATIONS_COMBINATORICS_SPECIAL_FUNCTIONS_SPECIAL_FUNCTIONS_H_
#define SRC_LIB_LAYER1_FOUNDATIONS_COMBINATORICS_SPECIAL_FUNCTIONS_SPECIAL_FUNCTIONS_H_




namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace special_functions {



// #############################################################################
// apn_functions.cpp
// #############################################################################

//! boolean functions

class apn_functions {

public:

	algebra::field_theory::finite_field *F;

	apn_functions();
	~apn_functions();
	void init(
			algebra::field_theory::finite_field *F,
			int verbose_level);
	void search_APN(
			int delta_max, int verbose_level);
	void search_APN_recursion(
			int *f, int depth, int f_normalize,
			int &delta_max, int &nb_times,
			std::vector<std::vector<int> > &Solutions,
			int *A_matrix, int *B_matrix,
			int *Count_ab, int *nb_times_ab,
			int verbose_level);
	int search_APN_perform_checks(
			int *f, int depth,
			int delta_max,
			int *A_matrix, int *B_matrix, int *Count_ab,
			int verbose_level);
	void search_APN_undo_checks(
			int *f, int depth,
			int delta_max,
			int *A_matrix, int *B_matrix, int *Count_ab,
			int verbose_level);
	int perform_single_check(
			int *f, int depth, int i, int delta_max,
			int *A_matrix, int *B_matrix, int *Count_ab,
			int verbose_level);
	void undo_single_check(
			int *f, int depth, int i, int delta_max,
			int *A_matrix, int *B_matrix, int *Count_ab,
			int verbose_level);
	void search_APN_old(
			int verbose_level);
	void search_APN_recursion_old(
			int *f, int depth, int f_normalize,
			int &delta_min, int &nb_times,
			std::vector<std::vector<int> > &Solutions,
			int *nb_times_ab,
			int verbose_level);
	int differential_uniformity(
			int *f, int *nb_times_ab, int verbose_level);
	int differential_uniformity_with_fibre(
			int *f, int *nb_times_ab, int *&Fibre,
			int verbose_level);

};



// #############################################################################
// boolean_function_domain.cpp
// #############################################################################

//! boolean functions

class boolean_function_domain {

public:
	int n;
	int n2; // n / 2
	int Q; // 2^n
	int bent; // 2^{n/2}
	int near_bent; // 2^{(n+1)/2}
	//int NN;
	algebra::ring_theory::longinteger_object *NN; // 2^Q
	int N; // size of PG(n,2)

	algebra::field_theory::finite_field *Fq; // the field F2
	//finite_field *FQ; // the field of order 2^n

	algebra::ring_theory::homogeneous_polynomial_domain *Poly;
		// Poly[i] = polynomial of degree i in n + 1 variables.
		// i = 1,..,n
	int **A_poly; // [1..n][Poly[i].get_nb_monomials()]
	int **B_poly; // [1..n][Poly[i].get_nb_monomials()]
	int *Kernel;
	int dim_kernel;


	long int *affine_points; // [Q]
		// affine_points[i] = PG_rank of affine point[i]



	int *v; // [n]
	int *v1; // [n + 1]
	int *w; // [n]
	int *f; // [Q]
	int *f2; // [Q]
	int *F; // [Q]
	int *T; // [Q]
	int *W; // [Q * Q] = Walsh matrix
	int *f_proj;
	int *f_proj2;



	boolean_function_domain();
	~boolean_function_domain();
	void init(
			algebra::field_theory::finite_field *F2, int n,
			int verbose_level);
	void setup_polynomial_rings(
			int verbose_level);
	void compute_polynomial_representation(
			int *func, int *coeff, int verbose_level);
	void evaluate_projectively(
			int *coeff, int *f);
	void evaluate(
			int *coeff, int *f);
	void raise(
			int *in, int *out);
	void apply_Walsh_transform(
			int *in, int *out);
	int is_bent(
			int *T);
	int is_near_bent(
			int *T);
};

// #############################################################################
// permutations.cpp
// #############################################################################


//! permutations given in vector form


class permutations {

public:

	permutations();
	~permutations();
	void random_permutation(
			int *random_permutation, long int n);
	void perm_move(
			int *from, int *to, long int n);
	void perm_identity(
			int *a, long int n);
	int perm_is_identity(
			int *a, long int n);
	void perm_elementary_transposition(
			int *a, long int n, int f);
	void perm_cycle(
			int *perm, long int n);
	void perm_mult(
			int *a, int *b, int *c, long int n);
	void perm_conjugate(
			int *a, int *b, int *c, long int n);
	// c := a^b = b^-1 * a * b
	void perm_inverse(
			int *a, int *b, long int n);
	// b := a^-1
	void perm_raise(
			int *a, int *b, int e, long int n);
	// b := a^e (e >= 0)
	void perm_direct_product(
			long int n1, long int n2,
			int *perm1, int *perm2, int *perm3);
	void perm_print_list(
			std::ostream &ost, int *a, int n);
	void perm_print_list_offset(
			std::ostream &ost, int *a, int n, int offset);
	void perm_print_product_action(
			std::ostream &ost, int *a, int m_plus_n, int m,
		int offset, int f_cycle_length);
	void perm_print(
			std::ostream &ost, int *a, int n);
	void perm_print_with_point_labels(
			std::ostream &ost,
			int *a, int n,
			std::string *Point_labels, void *data);
	void perm_print_with_cycle_length(
			std::ostream &ost, int *a, int n);
	void perm_print_counting_from_one(
			std::ostream &ost, int *a, int n);
	void perm_print_offset(
			std::ostream &ost,
		int *a, int n,
		int offset,
		int f_print_cycles_of_length_one,
		int f_cycle_length,
		int f_max_cycle_length,
		int max_cycle_length,
		int f_orbit_structure,
		std::string *Point_labels, void *data);
	void perm_cycle_type(
			int *perm, long int degree, int *cycles, int &nb_cycles);
	int perm_order(
			int *a, long int n);
	int perm_signum(
			int *perm, long int n);
	int is_permutation(
			int *perm, long int n);
	int is_permutation_lint(
			long int *perm, long int n);
	void first_lehmercode(
			int n, int *v);
	int next_lehmercode(
			int n, int *v);
	int sign_based_on_lehmercode(
			int n, int *v);
	void lehmercode_to_permutation(
			int n, int *code, int *perm);

};


// #############################################################################
// polynomial_function_domain.cpp
// #############################################################################

//! polynomial expressions for functions from a finite field to itself

class polynomial_function_domain {

public:
	algebra::field_theory::finite_field *Fq; // the field Fq
	int q;

	int n;
	int max_degree; // n * (q - 1)

	int Q; // q^n = number of inputs to the function.

	algebra::ring_theory::homogeneous_polynomial_domain *Poly;
		// Poly[i] = polynomial of degree i in n + 1 variables.
		// i = 1,..,max_degree
	int **A_poly; // [1..max_degree][Poly[i].get_nb_monomials()]
	int **B_poly; // [1..max_degree][Poly[i].get_nb_monomials()]
	int **C_poly; // [1..max_degree][Poly[i].get_nb_monomials()]
	int *Kernel;
	int dim_kernel;


	long int *affine_points; // [Q]
		// affine_points[i] = PG_rank of affine point[i]



	int *v; // [n]
	int *v1; // [n + 1]
	int *w; // [n]
	int *f; // [Q]
	int *f2; // [Q]


	polynomial_function_domain();
	~polynomial_function_domain();
	void init(
			algebra::field_theory::finite_field *Fq,
			int n, int verbose_level);
	void setup_polynomial_rings(
			int verbose_level);
	void compute_polynomial_representation(
			int *func, int *coeff, int verbose_level);
	void evaluate_projectively(
			int *coeff, int *f);
	void evaluate(
			int *coeff, int *f);
	void raise(
			int *in, int *out);
	void multiply_i_times_j(
			int i, int j,
			int *A_eqn, int *B_eqn, int *C_eqn,
		int verbose_level);
	void algebraic_normal_form(
			int *func, int len,
			int *&coeff, int &nb_coeff,
			int verbose_level);
};




// #############################################################################
// special_functions_domain.cpp
// #############################################################################

//! polynomial expressions for functions from a projective space to the field

class special_functions_domain {

public:
	algebra::field_theory::finite_field *Fq; // the field Fq
	int q;

	geometry::projective_geometry::projective_space *P;

	int nb_vars;
	int max_degree; // nb_vars * (q - 1)

	special_functions_domain();
	~special_functions_domain();
	void init(
			geometry::projective_geometry::projective_space *P,
			int verbose_level);
	void make_polynomial_representation(
			long int *Pts, int nb_pts,
			std::string &poly_rep,
			int verbose_level);


};



}}}}


#endif /* SRC_LIB_LAYER1_FOUNDATIONS_COMBINATORICS_SPECIAL_FUNCTIONS_SPECIAL_FUNCTIONS_H_ */
