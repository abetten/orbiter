/*
 * ring_theory.h
 *
 *  Created on: Nov 1, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_FOUNDATIONS_RING_THEORY_RING_THEORY_H_
#define SRC_LIB_FOUNDATIONS_RING_THEORY_RING_THEORY_H_

namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace ring_theory {


// #############################################################################
// finite_ring.cpp
// #############################################################################


//! ring of integers modulo q



class finite_ring {

	int *add_table; // [q * q]
	int *mult_table; // [q * q]

	int *f_is_unit_table; // [q]
	int *negate_table; // [q]
	int *inv_table; // [q]

	// only defined if we are a chain ring:
	int p;
	int e;
	algebra::field_theory::finite_field *Fp;

public:
	int q;

	int f_chain_ring; // true if q is a prime power



	finite_ring();
	~finite_ring();
	void init(
			int q, int verbose_level);
	int get_e();
	int get_p();
	algebra::field_theory::finite_field *get_Fp();
	int zero();
	int one();
	int is_zero(
			int i);
	int is_one(
			int i);
	int is_unit(
			int i);
	int add(
			int i, int j);
	int mult(
			int i, int j);
	int negate(
			int i);
	int inverse(
			int i);
	int Gauss_int(
			int *A, int f_special,
		int f_complete, int *base_cols,
		int f_P, int *P, int m, int n, int Pn,
		int verbose_level);
		// returns the rank which is the number
		// of entries in base_cols
		// A is a m x n matrix,
		// P is a m x Pn matrix (if f_P is true)
	int PHG_element_normalize(
			int *v, int stride, int len);
	// last unit element made one
	int PHG_element_normalize_from_front(
			int *v,
		int stride, int len);
	// first non unit element made one
	int PHG_element_rank(
			int *v, int stride, int len);
	void PHG_element_unrank(
			int *v, int stride, int len, int rk);
	int nb_PHG_elements(
			int n);
};

// #############################################################################
// homogeneous_polynomial_domain.cpp
// #############################################################################

//! homogeneous polynomials of a given degree in a given number of variables over a finite field GF(q)


class homogeneous_polynomial_domain {

private:
	algebra::field_theory::finite_field *F;
	int nb_monomials;
		// = Combi.int_n_choose_k(
		// nb_variables + degree - 1, nb_variables - 1);
	int *Monomials; // [nb_monomials * nb_variables]


	std::vector<std::string> symbols; // nb_variables
	std::vector<std::string> symbols_latex; // nb_variables

	std::vector<std::string> monomial_symbols;
	std::vector<std::string> monomial_symbols_latex;
	std::vector<std::string> monomial_symbols_easy;

	int *Variables; // [nb_monomials * degree]
		// Variables contains the monomials written out
		// as a sequence of length degree
		// with entries in 0,..,nb_variables-1.
		// the entries are listed in increasing order.
		// For instance, the monomial x_0^2x_1x_3
		// is recorded as 0,0,1,3
	int nb_affine; // nb_variables^degree

	// Affine could get too big:
	int *Affine; // [nb_affine * degree]
		// the affine elements are used for foiling
		// when doing a linear substitution
	int *v; // [nb_variables]

	// Affine_to_monomial could get too big:
	int *Affine_to_monomial; // [nb_affine]
		// for each vector in the affine space,
		// record the monomial associated with it.

	int *coeff2; // [nb_monomials], used in substitute_linear
	int *coeff3; // [nb_monomials], used in substitute_linear
	int *coeff4; // [nb_monomials], used in substitute_linear
	int *factors; // [degree]
	int *my_affine; // [degree], used in substitute_linear
	int *base_cols; // [nb_monomials]
	int *type1; // [degree + 1]
	int *type2; // [degree + 1]


public:
	int q;
	int nb_variables; // number of variables
	int degree;
	enum monomial_ordering_type Monomial_ordering_type;

	homogeneous_polynomial_domain();
	~homogeneous_polynomial_domain();
	void init(
			polynomial_ring_description *Descr,
			int verbose_level);
	void init(
			algebra::field_theory::finite_field *F,
			int nb_vars, int degree,
			monomial_ordering_type Monomial_ordering_type,
			int verbose_level);
	void init_with_or_without_variables(
			algebra::field_theory::finite_field *F,
			int nb_vars, int degree,
			monomial_ordering_type Monomial_ordering_type,
			int f_has_variables,
			std::vector<std::string> *variables_txt,
			std::vector<std::string> *variables_tex,
			int verbose_level);
	void print();
	void print_latex(
			std::ostream &ost);
	void print_monomials(
			int verbose_level);
	int get_nb_monomials();
	int get_nb_variables();
	algebra::field_theory::finite_field *get_F();
	std::string get_symbol(
			int i);
	std::string list_of_variables();
	int variable_index(
			std::string &s);
	int get_monomial(
			int i, int j);
	std::string get_monomial_symbol_easy(
			int i);
	std::string get_monomial_symbols_latex(
			int i);
	std::string get_monomial_symbols(
			int i);
	int *get_monomial_pointer(
			int i);
	int evaluate_monomial(
			int idx_of_monomial, int *coords);
	void remake_symbols(
			int symbol_offset,
			std::string &symbol_mask, std::string &symbol_mask_latex,
			int verbose_level);
	void remake_symbols_interval(
			int symbol_offset,
			int from, int len,
			std::string &symbol_mask, std::string &symbol_mask_latex,
			int verbose_level);
	void make_monomials(
			monomial_ordering_type Monomial_ordering_type,
			int f_has_variables,
			std::vector<std::string> *variables_txt,
			std::vector<std::string> *variables_tex,
			int verbose_level);
	void rearrange_monomials_by_partition_type(
			int verbose_level);
	int index_of_monomial(
			int *v);
	void affine_evaluation_kernel(
			int *&Kernel, int &dim_kernel, int verbose_level);
	void get_quadratic_form_matrix(
			int *eqn, int *M);
	void print_symbols(
			std::ostream &ost);
	std::string stringify_monomial(
			int i);
	void print_monomial(
			std::ostream &ost, int i);
	void print_monomial(
			std::ostream &ost, int *mon);
	void print_monomial_latex(
			std::ostream &ost, int *mon);
	void print_monomial_latex(
			std::ostream &ost, int i);
	void print_monomial_relaxed(
			std::ostream &ost, int i);
	void print_monomial_latex(
			std::string &s, int *mon);
	void print_monomial_relaxed(
			std::string &s, int *mon);
	void print_monomial_latex(
			std::string &s, int i);
	void print_monomial_str(
			std::stringstream &ost, int i);
	void print_monomial_for_gap_str(
			std::stringstream &ost, int i);
	void print_monomial_latex_str(
			std::stringstream &ost, int i);
	void print_equation(
			std::ostream &ost, int *coeffs);
	std::string stringify_equation(
			int *coeffs);
	void print_equation_simple(
			std::ostream &ost, int *coeffs);
	void print_equation_tex(
			std::ostream &ost, int *coeffs);
	void print_equation_relaxed(
			std::ostream &ost, int *coeffs);
	void print_equation_numerical(
			std::ostream &ost, int *coeffs);
	void print_equation_lint(
			std::ostream &ost, long int *coeffs);
	void print_equation_lint_tex(
			std::ostream &ost, long int *coeffs);
	void print_equation_str(
			std::stringstream &ost, int *coeffs);
	void print_equation_for_gap_str(
			std::stringstream &ost, int *coeffs);
	void print_equation_with_line_breaks_tex(
			std::ostream &ost,
		int *coeffs, int nb_terms_per_line,
		const char *new_line_text);
	void print_equation_with_line_breaks_tex_lint(
		std::ostream &ost, long int *coeffs, int nb_terms_per_line,
		const char *new_line_text);
	void algebraic_set(
			int *Eqns, int nb_eqns,
			long int *Pts, int &nb_pts, int verbose_level);
	void polynomial_function(
			int *coeff, int *f, int verbose_level);
	void polynomial_function_affine(
			int *coeff, int *f, int verbose_level);
	void enumerate_points(int *coeff,
			std::vector<long int> &Pts,
			int verbose_level);
	void enumerate_points_in_intersection(
			int *coeff1,
			int *coeff2,
			std::vector<long int> &Pts,
			int verbose_level);
	void enumerate_points_lint(
			int *coeff,
			long int *&Pts, int &nb_pts, int verbose_level);
	void enumerate_points_in_intersection_lint(
			int *coeff1, int *coeff2,
			long int *&Pts, int &nb_pts, int verbose_level);
	void enumerate_points_zariski_open_set(
			int *coeff,
			std::vector<long int> &Pts,
			int verbose_level);
	int evaluate_at_a_point_by_rank(
			int *coeff, int pt);
	int evaluate_at_a_point(
			int *coeff, int *pt_vec);
	void substitute_linear(
			int *coeff_in, int *coeff_out,
		int *Mtx_inv, int verbose_level);
	void substitute_semilinear(
			int *coeff_in, int *coeff_out,
		int f_semilinear, int frob_power, int *Mtx_inv,
		int verbose_level);
	void substitute_line(
		int *coeff_in, int *coeff_out,
		int *Pt1_coeff, int *Pt2_coeff,
		int verbose_level);
	void multiply_by_scalar(
		int *coeff_in, int scalar, int *coeff_out,
		int verbose_level);
	void multiply_mod(
		int *coeff1, int *coeff2, int *coeff3,
		int verbose_level);
	void multiply_mod_negatively_wrapped(
		int *coeff1, int *coeff2, int *coeff3,
		int verbose_level);
	int is_zero(
			int *coeff);
	void unrank_point(
			int *v, long int rk);
	long int rank_point(
			int *v);
	void unrank_coeff_vector(
			int *v, long int rk);
	long int rank_coeff_vector(
			int *v);
	int test_weierstrass_form(
			int rk,
		int &a1, int &a2, int &a3, int &a4, int &a6,
		int verbose_level);
	int test_potential_algebraic_degree(
			int *eqn, int eqn_size,
			long int *Pts, int nb_pts,
			int d,
			other::data_structures::int_matrix *Subspace_wgr,
			int *eqn_reduced,
			int *eqn_kernel,
			int verbose_level);
	int dimension_of_ideal(
			long int *Pts,
			int nb_pts, int verbose_level);
	void explore_vanishing_ideal(
			long int *Pts, int nb_pts, int verbose_level);
	void evaluate_point_on_all_monomials(
			int *pt_coords,
			int *evaluation,
			int verbose_level);
	void make_system(
			int *Pt_coords, int nb_pts,
			int *&System, int &nb_cols,
			int verbose_level);
	void vanishing_ideal(
			long int *Pts, int nb_pts, int &r,
			other::data_structures::int_matrix *&Kernel,
		int verbose_level);
	void subspace_with_good_reduction(
			int degree, int modulus,
			other::data_structures::int_matrix *&Subspace_wgr,
			int verbose_level);
	int monomial_has_good_reduction(
			int mon_idx, int degree, int modulus,
			int verbose_level);
	void monomial_reduction(
			int mon_idx, int modulus,
			int *reduced_monomial,
			int verbose_level);
	void equation_reduce(
			int modulus,
			homogeneous_polynomial_domain *HPD,
			int *eqn_in,
			int *eqn_out,
			int verbose_level);
	int compare_monomials(
			int *M1, int *M2);
	int compare_monomials_PART(
			int *M1, int *M2);
	void print_monomial_ordering_latex(
			std::ostream &ost);
	int *read_from_string_coefficient_pairs(
			std::string &str, int verbose_level);
	int *read_from_string_coefficient_vector(
			std::string &str, int verbose_level);
	void number_of_conditions_satisfied(
			std::string &variety_label_txt,
			std::string &variety_label_tex,
			std::vector<std::string> &Variety_coeffs,
			std::string &number_of_conditions_satisfied_fname,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
			int verbose_level);
	void create_intersection_of_zariski_open_sets(
			std::string &variety_label_txt,
			std::string &variety_label_tex,
			std::vector<std::string> &Variety_coeffs,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
			int verbose_level);
	void create_projective_variety(
			std::string &variety_label,
			std::string &variety_label_tex,
			int *coeff, int sz,
			//std::string &variety_coeffs,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
			int verbose_level);
	void create_ideal(
			std::string &ideal_label,
			std::string &ideal_label_tex,
			std::string &ideal_point_set_label,
			int &dim_kernel, int &nb_monomials,
			other::data_structures::int_matrix *&Kernel,
			//int *&Kernel,
			int verbose_level);
	void create_projective_curve(
			std::string &variety_label_txt,
			std::string &variety_label_tex,
			std::string &curve_coeffs,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
			int verbose_level);
	void get_coefficient_vector(
			algebra::expression_parser::formula *Formula,
			std::string &evaluate_text,
			int *Coefficient_vector,
			int verbose_level);
	void evaluate_regular_map(
			int *Coefficient_vector,
			int nb_eqns,
			geometry::projective_geometry::projective_space *P,
			long int *&Image_pts, int &N_points,
			int verbose_level);
	std::string stringify(
			int *eqn);
	std::string stringify_algebraic_notation(
			int *eqn);
	void parse_equation_wo_parameters(
			std::string &name_of_formula,
			std::string &name_of_formula_tex,
			std::string &equation_text,
			int *&eqn, int &eqn_size,
			int verbose_level);
	void parse_equation_and_substitute_parameters(
			std::string &name_of_formula,
			std::string &name_of_formula_tex,
			std::string &equation_text,
			std::string &equation_parameters,
			std::string &equation_parameter_values,
			int *&eqn, int &eqn_size,
			int verbose_level);
	void compute_singular_points_projectively(
			geometry::projective_geometry::projective_space *P,
			int *equation,
			std::vector<long int> &Singular_points,
			int verbose_level);
	void compute_partials(
			homogeneous_polynomial_domain *Poly_reduced_degree,
			ring_theory::partial_derivative *&Partials,
			int verbose_level);
	// Partials[nb_variables]
	void compute_and_export_partials(
			homogeneous_polynomial_domain *Poly_reduced_degree,
			int verbose_level);
	void compute_gradient(
			homogeneous_polynomial_domain *Poly_reduced_degree,
			int *equation, int *&gradient, int verbose_level);
	// gradient[nb_variables]


};


// #############################################################################
// longinteger_domain.cpp:
// #############################################################################

//! domain to compute with integers of arbitrary size, using class longinteger_object

class longinteger_domain {

public:
	longinteger_domain();
	~longinteger_domain();
	int compare(
			longinteger_object &a, longinteger_object &b);
	int compare_unsigned(
			longinteger_object &a, longinteger_object &b);
		// returns -1 if a < b, 0 if a = b,
		// and 1 if a > b, treating a and b as unsigned.
	int is_less_than(
			longinteger_object &a, longinteger_object &b);
	void subtract_signless(
			longinteger_object &a,
		longinteger_object &b, longinteger_object &c);
		// c = a - b, assuming a > b
	void subtract_signless_in_place(
			longinteger_object &a,
		longinteger_object &b);
		// a := a - b, assuming a > b
	void add(
			longinteger_object &a,
		longinteger_object &b, longinteger_object &c);
	void add_mod(
			longinteger_object &a,
		longinteger_object &b, longinteger_object &c,
		longinteger_object &m, int verbose_level);
	void add_in_place(
			longinteger_object &a,
			longinteger_object &b);
		// a := a + b
	void subtract_in_place(
			longinteger_object &a, longinteger_object &b);
	// a := a - b
	void add_int_in_place(
			longinteger_object &a, long int b);
	void mult(
			longinteger_object &a,
		longinteger_object &b, longinteger_object &c);
	void mult_in_place(
			longinteger_object &a, longinteger_object &b);
	void mult_integer_in_place(
			longinteger_object &a, int b);
	void mult_mod(
			longinteger_object &a,
		longinteger_object &b, longinteger_object &c,
		longinteger_object &m, int verbose_level);
	void multiply_up(
			longinteger_object &a, int *x, int len,
			int verbose_level);
	void multiply_up_lint(
			longinteger_object &a, long int *x, int len,
			int verbose_level);
	int quotient_as_int(
			longinteger_object &a, longinteger_object &b);
	long int quotient_as_lint(
			longinteger_object &a, longinteger_object &b);
	void integral_division_exact(
			longinteger_object &a,
		longinteger_object &b, longinteger_object &a_over_b);
	void integral_division(
		longinteger_object &a, longinteger_object &b,
		longinteger_object &q, longinteger_object &r,
		int verbose_level);
	void integral_division_by_int(
			longinteger_object &a,
		int b, longinteger_object &q, int &r);
	void integral_division_by_lint(
		longinteger_object &a,
		long int b, longinteger_object &q, long int &r);
	void inverse_mod(
		longinteger_object &a,
		longinteger_object &m, longinteger_object &av,
		int verbose_level);
	void extended_gcd(
			longinteger_object &a,
			longinteger_object &b,
		longinteger_object &g, longinteger_object &u,
		longinteger_object &v, int verbose_level);
	int logarithm_base_b(
			longinteger_object &a, int b);
	void base_b_representation(
			longinteger_object &a,
		int b, int *&rep, int &len);
	void power_int(
			longinteger_object &a, int n);
	void power_int_mod(
			longinteger_object &a, int n,
		longinteger_object &m);
	void power_longint_mod(
			longinteger_object &a,
		longinteger_object &n, longinteger_object &m,
		int verbose_level);
	void square_root(
			longinteger_object &a,
			longinteger_object &sqrt_a,
			int verbose_level);
	int square_root_mod(
			int a, int p, int verbose_level);
		// solves x^2 = a mod p. Returns x


	void create_order_of_group_Anq(
			longinteger_object &order, int n, int q, int verbose_level);
	void create_order_of_group_Bnq(
			longinteger_object &order, int n, int q, int verbose_level);
	void create_order_of_group_Dnq(
			longinteger_object &order, int n, int q, int verbose_level);
	void create_q_to_the_n(
			longinteger_object &a, int q, int n);
	void create_qnm1(
			longinteger_object &a, int q, int n);
	void create_Mersenne(
			longinteger_object &M, int n);
	// $M_n = 2^n - 1$
	void create_Fermat(
			longinteger_object &F, int n);
	// $F_n = 2^{2^n} + 1$
	void Dedekind_number(
			longinteger_object &Dnq,
			int n, int q, int verbose_level);

	int is_even(
			longinteger_object &a);
	int is_odd(
			longinteger_object &a);
	int remainder_mod_int(
			longinteger_object &a, int p);
	int multiplicity_of_p(
			longinteger_object &a,
		longinteger_object &residue, int p);
	long int smallest_primedivisor(
			longinteger_object &a, int p_min,
		int verbose_level);
	void factor_into_longintegers(
			longinteger_object &a,
		int &nb_primes, longinteger_object *&primes,
		int *&exponents, int verbose_level);
	void factor(
			longinteger_object &a, int &nb_primes,
		int *&primes, int *&exponents,
		int verbose_level);
	int jacobi(
			longinteger_object &a, longinteger_object &m,
		int verbose_level);

	void random_number_less_than_n(
			longinteger_object &n,
		longinteger_object &r);
	void random_number_with_n_decimals(
		longinteger_object &R, int n, int verbose_level);
	void matrix_product(
			longinteger_object *A,
			longinteger_object *B,
		longinteger_object *&C, int Am, int An, int Bn);
	void matrix_entries_integral_division_exact(
			longinteger_object *A,
		longinteger_object &b, int Am, int An);
	void matrix_print_GAP(
			std::ostream &ost,
			longinteger_object *A,
		int Am, int An);
	void matrix_print_tex(
			std::ostream &ost,
			longinteger_object *A,
		int Am, int An);
	void power_mod(
			char *aa, char *bb, char *nn,
		longinteger_object &result, int verbose_level);
	void factorial(
			longinteger_object &result, int n);
	void group_order_PGL(
			longinteger_object &result,
		int n, int q, int f_semilinear);
	void square_root_floor(
			longinteger_object &a,
			longinteger_object &x, int verbose_level);
	void print_digits(
			char *rep, int len);
	void Chinese_Remainders(
			std::vector<long int> &Remainders,
			std::vector<long int> &Moduli,
			longinteger_object &x, longinteger_object &M,
			int verbose_level);
	void check_for_int_overflow_given_string_and_convert(
			std::string &number_to_test,
			ring_theory::longinteger_object *&number_to_test_longinteger,
			long int &number_to_test_lint, int &number_to_test_int,
			int verbose_level);
	void check_for_int_overflow_given_string(
			std::string &number_to_test,
			long int &number_to_test_lint, int &number_to_test_int,
			int verbose_level);
	void check_for_int_overflow(
			ring_theory::longinteger_object *number_to_test,
			int verbose_level);
	void check_for_lint_overflow(
			ring_theory::longinteger_object *number_to_test,
			int verbose_level);

};




// #############################################################################
// longinteger_object.cpp:
// #############################################################################


//! a class to represent integers of arbitrary size


class longinteger_object {

private:
	char sgn; // true if negative
	int l;
	char *r;

public:
	longinteger_object();
	~longinteger_object();
	void freeself();

	char &ith(
			int i) { return r[i]; };
	char &sign() { return sgn; };
	int &len() { return l; };
	char *&rep() { return r; };
	void create(
			long int i);
	void create_product(
			int nb_factors, int *factors);
	void create_power(
			int a, int e);
		// creates a^e
	void create_power_minus_one(
			int a, int e);
		// creates a^e  - 1
	void create_from_base_b_representation(
			int b, int *rep, int len);
	void create_from_base_10_string(
			const char *str, int verbose_level);
	void create_from_base_10_string(
			const char *str);
	void create_from_base_10_string(
			std::string &str);
	int as_int();
	long int as_lint();
	//void as_longinteger(longinteger_object &a);
	void assign_to(
			longinteger_object &b);
	void swap_with(
			longinteger_object &b);
	std::ostream& print(
			std::ostream& ost);
	std::ostream& print_not_scientific(
			std::ostream& ost);
	int log10();
	int output_width();
	void print_width(
			std::ostream& ost, int width);
	void print_to_string(
			std::string &s);
	void normalize();
	void negate();
	int is_zero();
	void zero();
	int is_one();
	int is_mone();
	int is_one_or_minus_one();
	void one();
	void increment();
	void decrement();
	void add_int(
			int a);
	void create_i_power_j(
			int i, int j);
	int compare_with_int(
			int a);
	std::string stringify();

};

std::ostream& operator<<(
		std::ostream& ost, longinteger_object& p);






// #############################################################################
// partial_derivative.cpp
// #############################################################################

//! partial derivative with respect to a given variable considered as a linear map between two homogeneous polynomial domains


class partial_derivative {

public:
	homogeneous_polynomial_domain *H;
	homogeneous_polynomial_domain *Hd; // degree one less than H
	int *v; // [H->get_nb_monomials()]
	int variable_idx;
	int *mapping; // [H->get_nb_monomials() * Hd->get_nb_monomials()]


	partial_derivative();
	~partial_derivative();
	void init(
			homogeneous_polynomial_domain *H,
			homogeneous_polynomial_domain *Hd,
			int variable_idx,
			int verbose_level);
	void apply(
			int *eqn_in,
			int *eqn_out,
			int verbose_level);
	void do_export(
			std::string &fname_base,
			int verbose_level);

};


// #############################################################################
// polynomial_double_domain.cpp:
// #############################################################################


//! domain for polynomials with coefficients of type double



class polynomial_double_domain {
public:
	int alloc_length;

	polynomial_double_domain();
	~polynomial_double_domain();
	void init(
			int alloc_length);
	ring_theory::polynomial_double *create_object();
	void mult(
			polynomial_double *A,
			polynomial_double *B, polynomial_double *C);
	void add(
			polynomial_double *A,
			polynomial_double *B, polynomial_double *C);
	void mult_by_scalar_in_place(
			polynomial_double *A,
			double lambda);
	void copy(
			polynomial_double *A,
			polynomial_double *B);
	void determinant_over_polynomial_ring(
			polynomial_double *P,
			polynomial_double *det, int n,
			int verbose_level);
	void find_all_roots(
			polynomial_double *p,
			double *lambda, int verbose_level);
	double divide_linear_factor(
			polynomial_double *p,
			polynomial_double *q,
			double lambda, int verbose_level);
};


// #############################################################################
// polynomial_double.cpp:
// #############################################################################


//! polynomials with double coefficients, related to class polynomial_double_domain


class polynomial_double {
public:
	int alloc_length;
	int degree;
	double *coeff; // [alloc_length]

	polynomial_double();
	~polynomial_double();
	void init(
			int alloc_length);
	void print(
			std::ostream &ost);
	double root_finder(
			int verbose_level);
	double evaluate_at(
			double t);
};





// #############################################################################
// polynomial_ring_activity_description.cpp
// #############################################################################


//! description of a polynomial ring activity

class polynomial_ring_activity_description {
public:


	// TABLES/polynomial_ring_activity.tex


	int f_cheat_sheet;

	int f_export_partials;

	int f_ideal;
	std::string ideal_label_txt;
	std::string ideal_label_tex;
	std::string ideal_point_set_label;

	int f_apply_transformation;
	std::string apply_transformation_Eqn_in_label;
	std::string apply_transformation_vector_ge_label;

	int f_set_variable_names;
	std::string set_variable_names_txt;
	std::string set_variable_names_tex;

	int f_print_equation;
	std::string print_equation_input;

	int f_parse_equation_wo_parameters;
	std::string parse_equation_wo_parameters_name_of_formula;
	std::string parse_equation_wo_parameters_name_of_formula_tex;
	std::string parse_equation_wo_parameters_equation_text;

	int f_parse_equation;
	std::string parse_equation_name_of_formula;
	std::string parse_equation_name_of_formula_tex;
	std::string parse_equation_equation_text;
	std::string parse_equation_equation_parameters;
	std::string parse_equation_equation_parameter_values;

	int f_table_of_monomials_write_csv;
	std::string table_of_monomials_write_csv_label;


	polynomial_ring_activity_description();
	~polynomial_ring_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};

// #############################################################################
// polynomial_ring_description.cpp
// #############################################################################


//! description of a polynomial ring

class polynomial_ring_description {
public:

	// TABLES/polynomial_ring.tex

	int f_field;
	std::string finite_field_label;

	int f_homogeneous;
	int homogeneous_of_degree;

	int f_number_of_variables;
	int number_of_variables;

	monomial_ordering_type Monomial_ordering_type;

	int f_variables;
	std::string variables_txt;
	std::string variables_tex;

	polynomial_ring_description();
	~polynomial_ring_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};




// #############################################################################
// ring_theory_global.cpp
// #############################################################################

//! global functions related to ring theory


class ring_theory_global {

public:
	ring_theory_global();
	~ring_theory_global();
	void Monomial_ordering_type_as_string(
			monomial_ordering_type Monomial_ordering_type,
			std::string &s);
	void write_code_for_division(
			algebra::field_theory::finite_field *F,
			std::string &label_code,
			std::string &A_coeffs,
			std::string &B_coeffs,
			int verbose_level);
	void polynomial_division(
			algebra::field_theory::finite_field *F,
			std::string &A_coeffs,
			std::string &B_coeffs,
			int verbose_level);
	void extended_gcd_for_polynomials(
			algebra::field_theory::finite_field *F,
			std::string &A_coeffs,
			std::string &B_coeffs,
			int verbose_level);
	void polynomial_mult_mod(
			algebra::field_theory::finite_field *F,
			std::string &A_coeffs,
			std::string &B_coeffs,
			std::string &M_coeffs,
			int verbose_level);
	void polynomial_power_mod(
			algebra::field_theory::finite_field *F,
			std::string &A_coeffs,
			std::string &power_text,
			std::string &M_coeffs,
			int verbose_level);
	void polynomial_find_roots(
			algebra::field_theory::finite_field *F,
			std::string &A_coeffs,
			int verbose_level);
	void sift_polynomials(
			algebra::field_theory::finite_field *F,
			long int rk0, long int rk1, int verbose_level);
	void mult_polynomials(
			algebra::field_theory::finite_field *F,
			long int rk0, long int rk1, int verbose_level);
	void polynomial_division_coefficient_table_with_report(
			std::string &prefix,
			algebra::field_theory::finite_field *F,
			int *coeff_table0, int coeff_table0_len,
			int *coeff_table1, int coeff_table1_len,
			int *&coeff_table_q, int &coeff_table_q_len,
			int *&coeff_table_r, int &coeff_table_r_len,
			int verbose_level);
	void polynomial_division_with_report(
			algebra::field_theory::finite_field *F,
			long int rk0, long int rk1, int verbose_level);
	void assemble_monopoly(
			algebra::field_theory::finite_field *F,
			int length,
			std::string &coefficient_vector_text,
			std::string &exponent_vector_text,
			int verbose_level);
	void polynomial_division_from_file_with_report(
			algebra::field_theory::finite_field *F,
			std::string &input_file, long int rk1,
			int verbose_level);
	void polynomial_division_from_file_all_k_error_patterns_with_report(
			algebra::field_theory::finite_field *F,
			std::string &input_file, long int rk1,
			int k, int verbose_level);
	void create_irreducible_polynomial(
			algebra::field_theory::finite_field *F,
			unipoly_domain *Fq,
			unipoly_object *&Beta, int n,
			long int *cyclotomic_set, int cylotomic_set_size,
			unipoly_object *&min_poly,
			int verbose_level);
	void compute_nth_roots_as_polynomials(
			algebra::field_theory::finite_field *F,
			unipoly_domain *FpX,
			unipoly_domain *Fq, unipoly_object *&Beta,
			int n1, int n2, int verbose_level);
	void compute_powers(
			algebra::field_theory::finite_field *F,
			unipoly_domain *Fq,
			int n, int start_idx,
			unipoly_object *&Beta, int verbose_level);
	void make_all_irreducible_polynomials_of_degree_d(
			algebra::field_theory::finite_field *F,
			int d,
			std::vector<std::vector<int> > &Table,
			int verbose_level);
	int count_all_irreducible_polynomials_of_degree_d(
			algebra::field_theory::finite_field *F,
			int d, int verbose_level);
	void do_make_table_of_irreducible_polynomials(
			algebra::field_theory::finite_field *F,
			int deg, int verbose_level);
	char *search_for_primitive_polynomial_of_given_degree(
			int p,
		int degree, int verbose_level);
	void search_for_primitive_polynomials(
			int p_min, int p_max,
		int n_min, int n_max, int verbose_level);
	void factor_cyclotomic(
			int n, int q, int d,
		int *coeffs, int f_poly,
		std::string &poly, int verbose_level);
	void oval_polynomial(
			algebra::field_theory::finite_field *F,
		int *S, unipoly_domain &D, unipoly_object &poly,
		int verbose_level);
	void print_longinteger_after_multiplying(
			std::ostream &ost,
			int *factors, int len);
	void parse_equation_easy(
			ring_theory::homogeneous_polynomial_domain *Poly,
			std::string &equation_text,
			int *&coeffs,
			int verbose_level);
	void parse_equation_with_parameters(
			ring_theory::homogeneous_polynomial_domain *Poly,
			std::string &equation_text,
			std::string &equation_parameters,
			std::string &equation_parameters_tex,
			std::string &equation_parameter_values,
			int *&coeffs,
			int verbose_level);
	void parse_equation(
			ring_theory::homogeneous_polynomial_domain *Poly,
			std::string &name_of_formula,
			std::string &name_of_formula_tex,
			int f_has_managed_variables,
			std::string &managed_variables,
			std::string &equation_text,
			std::string &equation_parameters,
			std::string &equation_parameters_tex,
			std::string &equation_parameter_values,
			int *&coeffs, int &nb_coeffs,
			int verbose_level);
	void simplify_and_expand(
			ring_theory::homogeneous_polynomial_domain *Poly,
			std::string &name_of_formula,
			std::string &name_of_formula_tex,
			int f_has_managed_variables,
			std::string &managed_variables,
			algebra::expression_parser::formula_vector *Formula_vector_after_sub,
			algebra::expression_parser::formula_vector *&Formula_vector_after_expand,
			int verbose_level);
	void perform_substitution(
			ring_theory::homogeneous_polynomial_domain *Poly,
			std::string &name_of_formula,
			std::string &name_of_formula_tex,
			int f_has_managed_variables,
			std::string &managed_variables,
			std::string &equation_parameters,
			std::string &equation_parameter_values,
			algebra::expression_parser::symbolic_object_builder *SB1,
			algebra::expression_parser::formula_vector *&Formula_vector_after_sub,
			int verbose_level);
	void test_unipoly(
			algebra::field_theory::finite_field *F);
	void test_unipoly2(
			algebra::field_theory::finite_field *F);
	void test_longinteger();
	void test_longinteger2();
	void test_longinteger3();
	void test_longinteger4();
	void test_longinteger5();
	void test_longinteger6();
	void test_longinteger7();
	void test_longinteger8();
	void longinteger_collect_setup(
			int &nb_agos,
			ring_theory::longinteger_object *&agos,
			int *&multiplicities);
	void longinteger_collect_free(
			int &nb_agos,
			ring_theory::longinteger_object *&agos,
			int *&multiplicities);
	void longinteger_collect_add(
			int &nb_agos,
			ring_theory::longinteger_object *&agos,
			int *&multiplicities,
			ring_theory::longinteger_object &ago);
	void longinteger_collect_print(
			std::ostream &ost,
			int &nb_agos, ring_theory::longinteger_object *&agos,
			int *&multiplicities);
	void make_table_of_monomials(
			ring_theory::homogeneous_polynomial_domain *Poly,
			std::string &name_of_formula,
			int verbose_level);
	void do_export_partials(
			ring_theory::homogeneous_polynomial_domain *Poly,
			int verbose_level);

};


// #############################################################################
// table_of_irreducible_polynomials.cpp
// #############################################################################

//! a table of all irreducible polynomials over a finite field of bounded degree

class table_of_irreducible_polynomials {
public:
	int degree_bound;
	int q;
	algebra::field_theory::finite_field *F;
	int nb_irred;
	int *Nb_irred;
	int *First_irred;
	int **Tables;
	int *Degree;

	table_of_irreducible_polynomials();
	~table_of_irreducible_polynomials();
	void init(
			int degree_bound,
			algebra::field_theory::finite_field *F,
			int verbose_level);
	void print(
			std::ostream &ost);
	void print_polynomials(
			std::ostream &ost);
	int select_polynomial_first(
			int *Select, int verbose_level);
	int select_polynomial_next(
			int *Select, int verbose_level);
	int is_irreducible(
			unipoly_object &poly, int verbose_level);
	void factorize_polynomial(
			unipoly_object &char_poly, int *Mult,
		int verbose_level);
};


// #############################################################################
// unipoly_domain.cpp:
// #############################################################################

//! domain of polynomials in one variable over a finite field

class unipoly_domain {

private:
	algebra::field_theory::finite_field *F;
	int f_factorring;
	int factor_degree;
	int *factor_coeffs; // [factor_degree + 1]
	unipoly_object factor_poly;
		// the coefficients of factor_poly are negated
		// so that mult_mod is easier

	int f_print_sub;
	//int f_use_variable_name;
	std::string variable_name;

public:

	unipoly_domain();
	unipoly_domain(
			algebra::field_theory::finite_field *GFq);
	void init_basic(
			algebra::field_theory::finite_field *F,
			int verbose_level);
	unipoly_domain(
			algebra::field_theory::finite_field *GFq,
			unipoly_object m,
			int verbose_level);
	~unipoly_domain();
	void init_variable_name(
			std::string &label);
	void init_factorring(
			algebra::field_theory::finite_field *F,
			unipoly_object m,
			int verbose_level);
	algebra::field_theory::finite_field *get_F();
	int &s_i(
			unipoly_object p, int i)
		{ int *rep = (int *) p; return rep[i + 1]; };
	void create_object_of_degree(
			unipoly_object &p, int d);
	void create_object_of_degree_no_test(
			unipoly_object &p, int d);
	void create_object_of_degree_with_coefficients(
			unipoly_object &p,
		int d, int *coeff);
	void create_object_by_rank(
			unipoly_object &p, long int rk,
			int verbose_level);
	void create_object_from_table_of_coefficients(
		unipoly_object &p, int *coeff_table, int coeff_table_len,
		int verbose_level);
	void create_object_from_csv_file(
		unipoly_object &p, std::string &fname,
		int verbose_level);
	void create_object_by_rank_longinteger(
			unipoly_object &p,
		longinteger_object &rank,
		int verbose_level);
	void create_object_by_rank_string(
		unipoly_object &p, std::string &rk, int verbose_level);
	void create_Dickson_polynomial(
			unipoly_object &p, int *map);
	void delete_object(
			unipoly_object &p);
	void unrank(
			unipoly_object p, int rk);
	void unrank_longinteger(
			unipoly_object p, longinteger_object &rank);
	int rank(
			unipoly_object p);
	void rank_longinteger(
			unipoly_object p, longinteger_object &rank);
	int degree(
			unipoly_object p);
	void print_object_latex(
			unipoly_object p, std::ostream &ost);
	void print_object(
			unipoly_object p, std::ostream &ost);
	std::string object_stringify(
			unipoly_object p);
	void print_object_sstr_latex(
			unipoly_object p, std::stringstream &ost);
	void make_companion_matrix(
			unipoly_object p, int *mtx, int verbose_level);
	// mtx[d * d]
	std::string stringify_object(
			unipoly_object p);
	void print_object_sstr(
			unipoly_object p, std::stringstream &ost);
	void print_object_tight(
			unipoly_object p, std::ostream &ost);
	void print_object_sparse(
			unipoly_object p, std::ostream &ost);
	void print_object_dense(
			unipoly_object p, std::ostream &ost);
	void print_factorization_based_off_Mult(
			table_of_irreducible_polynomials *T, int *Mult,
			std::ostream &ost, int verbose_level);
	void assign(
			unipoly_object a, unipoly_object &b,
			int verbose_level);
	void one(
			unipoly_object p);
	void m_one(
			unipoly_object p);
	void zero(
			unipoly_object p);
	int is_one(
			unipoly_object p);
	int is_zero(
			unipoly_object p);
	void negate(
			unipoly_object a);
	void make_monic(
			unipoly_object &a);
	void add(
			unipoly_object a,
			unipoly_object b,
			unipoly_object &c);
	void mult(
			unipoly_object a,
			unipoly_object b,
			unipoly_object &c,
			int verbose_level);
	void mult_mod(
			unipoly_object a,
		unipoly_object b,
		unipoly_object &c,
		unipoly_object m,
		int verbose_level);
	void mult_mod_negated(
			unipoly_object a,
			unipoly_object b,
			unipoly_object &c,
		int factor_polynomial_degree,
		int *factor_polynomial_coefficients_negated,
		int verbose_level);
	void Frobenius_matrix_by_rows(
			int *&Frob,
		unipoly_object factor_polynomial,
		int verbose_level);
		// the j-th row of Frob is x^{j*q} mod m
	void Frobenius_matrix(
			int *&Frob,
			unipoly_object factor_polynomial,
		int verbose_level);
	void Berlekamp_matrix(
			int *&B,
			unipoly_object factor_polynomial,
		int verbose_level);
	void exact_division(
			unipoly_object a,
		unipoly_object b, unipoly_object &q,
		int verbose_level);
	void division_with_remainder(
			unipoly_object a, unipoly_object b,
		unipoly_object &q, unipoly_object &r,
		int verbose_level);
	void derivative(
			unipoly_object a, unipoly_object &b);
	int compare_euclidean(
			unipoly_object m, unipoly_object n);
	void greatest_common_divisor(
			unipoly_object m, unipoly_object n,
		unipoly_object &g, int verbose_level);
	void extended_gcd(
			unipoly_object m, unipoly_object n,
		unipoly_object &u, unipoly_object &v,
		unipoly_object &g, int verbose_level);
	int is_squarefree(
			unipoly_object p,
			int verbose_level);
	void compute_normal_basis(int d,
			int *Normal_basis,
		int *Frobenius, int verbose_level);
	void order_ideal_generator(
			int d,
			int idx, unipoly_object &mue,
		int *A, int *Frobenius,
		int verbose_level);
		// Lueneburg~\cite{Lueneburg87a} p. 105.
		// Frobenius is a matrix of size d x d
		// A is a matrix of size (d + 1) x d
	void matrix_apply(
			unipoly_object &p, int *Mtx, int n,
		int verbose_level);
		// The matrix is applied on the left
	void substitute_matrix_in_polynomial(
			unipoly_object &p,
		int *Mtx_in, int *Mtx_out, int k, int verbose_level);
		// The matrix is substituted into the polynomial
	int substitute_scalar_in_polynomial(
			unipoly_object &p,
		int scalar, int verbose_level);
		// The scalar 'scalar' is substituted into the polynomial
	void module_structure_apply(
			int *v, int *Mtx, int n,
		unipoly_object p, int verbose_level);
	void take_away_all_factors_from_b(
			unipoly_object a,
		unipoly_object b, unipoly_object &a_without_b,
		int verbose_level);
		// Computes the polynomial $r$ with
		//\begin{enumerate}
		//\item
		//$r$ divides $a$
		//\item
		//$gcd(r,b) = 1$ and
		//\item
		//each irreducible polynomial dividing $a/r$ divides $b$.
		//Lueneburg~\cite{Lueneburg87a}, p. 37.
		//\end{enumerate}
	int is_irreducible(
			unipoly_object a, int verbose_level);
	void singer_candidate(
			unipoly_object &m,
		int p, int d, int b, int a);
	int is_primitive(
			unipoly_object &m,
		longinteger_object &qm1,
		int nb_primes, longinteger_object *primes,
		int verbose_level);
	void get_a_primitive_polynomial(
			unipoly_object &m,
		int f, int verbose_level);
	void get_an_irreducible_polynomial(
			unipoly_object &m,
		int f, int verbose_level);
	void power_int(
			unipoly_object &a,
			long int n, int verbose_level);
	void power_longinteger(
			unipoly_object &a,
			longinteger_object &n,
			int verbose_level);
	void power_mod(
			unipoly_object &a, unipoly_object &m,
			long int n, int verbose_level);
	void power_coefficients(
			unipoly_object &a, int n);
	void minimum_polynomial(
			unipoly_object &a,
		int alpha, int p, int verbose_level);
	int minimum_polynomial_factorring(
			int alpha, int p,
		int verbose_level);
	void minimum_polynomial_factorring_longinteger(
		longinteger_object &alpha,
		longinteger_object &rk_minpoly,
		int p, int verbose_level);
	void print_vector_of_polynomials(
			unipoly_object *sigma, int deg);
	void minimum_polynomial_extension_field(
			unipoly_object &g,
		unipoly_object m,
		unipoly_object &minpol, int d, int *Frobenius,
		int verbose_level);
		// Lueneburg~\cite{Lueneburg87a}, p. 112.
	void characteristic_polynomial(
			int *Mtx, int k,
		unipoly_object &char_poly, int verbose_level);
	void print_matrix(
			unipoly_object *M, int k);
	void determinant(
			unipoly_object *M,
			int k, unipoly_object &p,
		int verbose_level);
	void deletion_matrix(
			unipoly_object *M,
			int k, int delete_row,
		int delete_column, unipoly_object *&N,
		int verbose_level);
	void center_lift_coordinates(
			unipoly_object a, int q);
	void reduce_modulo_p(
			unipoly_object a, int p);

	//unipoly_domain2.cpp:
	void mult_easy(
			unipoly_object a,
			unipoly_object b,
			unipoly_object &c);
	void print_coeffs_top_down_assuming_one_character_per_digit(
			unipoly_object a, std::ostream &ost);
	void print_coeffs_top_down_assuming_one_character_per_digit_multiplied(
			unipoly_object a, int c, std::ostream &ost);
	void print_coeffs_top_down_assuming_one_character_per_digit_with_degree_given(
			unipoly_object a, int m, std::ostream &ost);
	void mult_easy_with_report(
			long int rk_a, long int rk_b, long int &rk_c,
			std::ostream &ost, int verbose_level);
	void division_with_remainder_from_file_with_report(
			std::string &input_fname, long int rk_b,
			long int &rk_q, long int &rk_r,
			std::ostream &ost, int verbose_level);
	void division_with_remainder_from_file_all_k_bit_error_patterns(
			std::string &input_fname, long int rk_b, int k,
			long int *&rk_q, long int *&rk_r, int &n, int &N,
			std::ostream &ost, int verbose_level);
	void division_with_remainder_based_on_tables_with_report(
			int *coeff_a, int len_a,
			int *coeff_b, int len_b,
			int *&coeff_q, int &len_q,
			int *&coeff_r, int &len_r,
			std::ostream &ost, int f_report, int verbose_level);
	void division_with_remainder_numerically_with_report(
			long int rk_a, long int rk_b,
			long int &rk_q, long int &rk_r,
			std::ostream &ost, int verbose_level);
	void division_with_remainder_with_report(
			unipoly_object &a, unipoly_object &b,
			unipoly_object &q, unipoly_object &r,
			int f_report, std::ostream &ost,
			int verbose_level);


};

}}}}


#endif /* SRC_LIB_FOUNDATIONS_RING_THEORY_RING_THEORY_H_ */
