/*
 * ring_theory.h
 *
 *  Created on: Nov 1, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_FOUNDATIONS_RING_THEORY_RING_THEORY_H_
#define SRC_LIB_FOUNDATIONS_RING_THEORY_RING_THEORY_H_

namespace orbiter {
namespace foundations {
namespace ring_theory {


// #############################################################################
// finite_ring.cpp
// #############################################################################


//! finite chain rings



class finite_ring {

	int *add_table; // [q * q]
	int *mult_table; // [q * q]

	int *f_is_unit_table; // [q]
	int *negate_table; // [q]
	int *inv_table; // [q]

	// only defined if we are a chain ring:
	int p;
	int e;
	finite_field *Fp;

public:
	int q;

	int f_chain_ring;



	finite_ring();
	~finite_ring();
	void null();
	void freeself();
	void init(int q, int verbose_level);
	int get_e();
	int get_p();
	finite_field *get_Fp();
	int zero();
	int one();
	int is_zero(int i);
	int is_one(int i);
	int is_unit(int i);
	int add(int i, int j);
	int mult(int i, int j);
	int negate(int i);
	int inverse(int i);
	int Gauss_int(int *A, int f_special,
		int f_complete, int *base_cols,
		int f_P, int *P, int m, int n, int Pn,
		int verbose_level);
		// returns the rank which is the number
		// of entries in base_cols
		// A is a m x n matrix,
		// P is a m x Pn matrix (if f_P is TRUE)
	int PHG_element_normalize(int *v, int stride, int len);
	// last unit element made one
	int PHG_element_normalize_from_front(int *v,
		int stride, int len);
	// first non unit element made one
	int PHG_element_rank(int *v, int stride, int len);
	void PHG_element_unrank(int *v, int stride, int len, int rk);
	int nb_PHG_elements(int n);
};

// #############################################################################
// homogeneous_polynomial_domain.cpp
// #############################################################################

//! homogeneous polynomials of a given degree in a given number of variables over a finite field GF(q)


class homogeneous_polynomial_domain {

private:
	enum monomial_ordering_type Monomial_ordering_type;
	finite_field *F;
	int nb_monomials;
	int *Monomials; // [nb_monomials * nb_variables]

	std::string *symbols;
	std::string *symbols_latex;

	char **monomial_symbols;
	char **monomial_symbols_latex;
	char **monomial_symbols_easy;
	int *Variables; // [nb_monomials * degree]
		// Variables contains the monomials written out
		// as a sequence of length degree
		// with entries in 0,..,nb_variables-1.
		// the entries are listed in increasing order.
		// For instance, the monomial x_0^2x_1x_3
		// is recorded as 0,0,1,3
	int nb_affine; // nb_variables^degree
	int *Affine; // [nb_affine * degree]
		// the affine elements are used for foiling
		// when doing a linear substitution
	int *v; // [nb_variables]
	int *Affine_to_monomial; // [nb_affine]
		// for each vector in the affine space,
		// record the monomial associated with it.
	projective_space *P;

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

	homogeneous_polynomial_domain();
	~homogeneous_polynomial_domain();
	void freeself();
	void null();
	void init(finite_field *F, int nb_vars, int degree,
		int f_init_incidence_structure,
		monomial_ordering_type Monomial_ordering_type,
		int verbose_level);
	int get_nb_monomials();
	projective_space *get_P();
	finite_field *get_F();
	int get_monomial(int i, int j);
	char *get_monomial_symbol_easy(int i);
	int *get_monomial_pointer(int i);
	int evaluate_monomial(int idx_of_monomial, int *coords);
	void remake_symbols(int symbol_offset,
			const char *symbol_mask, const char *symbol_mask_latex,
			int verbose_level);
	void remake_symbols_interval(int symbol_offset,
			int from, int len,
			const char *symbol_mask, const char *symbol_mask_latex,
			int verbose_level);
	void make_monomials(
			monomial_ordering_type Monomial_ordering_type,
			int verbose_level);
	void rearrange_monomials_by_partition_type(int verbose_level);
	int index_of_monomial(int *v);
	void affine_evaluation_kernel(
			int *&Kernel, int &dim_kernel, int verbose_level);
	void print_monomial(std::ostream &ost, int i);
	void print_monomial(std::ostream &ost, int *mon);
	void print_monomial_latex(std::ostream &ost, int *mon);
	void print_monomial_latex(std::ostream &ost, int i);
	void print_monomial_latex(std::string &s, int *mon);
	void print_monomial_latex(std::string &s, int i);
	void print_monomial_str(std::stringstream &ost, int i);
	void print_monomial_latex_str(std::stringstream &ost, int i);
	void print_equation(std::ostream &ost, int *coeffs);
	void print_equation_simple(std::ostream &ost, int *coeffs);
	void print_equation_tex(std::ostream &ost, int *coeffs);
	void print_equation_numerical(std::ostream &ost, int *coeffs);
	void print_equation_lint(std::ostream &ost, long int *coeffs);
	void print_equation_lint_tex(std::ostream &ost, long int *coeffs);
	void print_equation_str(std::stringstream &ost, int *coeffs);
	void print_equation_with_line_breaks_tex(std::ostream &ost,
		int *coeffs, int nb_terms_per_line,
		const char *new_line_text);
	void print_equation_with_line_breaks_tex_lint(
		std::ostream &ost, long int *coeffs, int nb_terms_per_line,
		const char *new_line_text);
	void algebraic_set(int *Eqns, int nb_eqns,
			long int *Pts, int &nb_pts, int verbose_level);
	void polynomial_function(int *coeff, int *f, int verbose_level);
	void enumerate_points(int *coeff,
			std::vector<long int> &Pts,
			int verbose_level);
	void enumerate_points_lint(int *coeff,
			long int *&Pts, int &nb_pts, int verbose_level);
	void enumerate_points_zariski_open_set(int *coeff,
			std::vector<long int> &Pts,
			int verbose_level);
	int evaluate_at_a_point_by_rank(int *coeff, int pt);
	int evaluate_at_a_point(int *coeff, int *pt_vec);
	void substitute_linear(int *coeff_in, int *coeff_out,
		int *Mtx_inv, int verbose_level);
	void substitute_semilinear(int *coeff_in, int *coeff_out,
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
	int is_zero(int *coeff);
	void unrank_point(int *v, int rk);
	int rank_point(int *v);
	void unrank_coeff_vector(int *v, long int rk);
	long int rank_coeff_vector(int *v);
	int test_weierstrass_form(int rk,
		int &a1, int &a2, int &a3, int &a4, int &a6,
		int verbose_level);
	void vanishing_ideal(long int *Pts, int nb_pts, int &r, int *Kernel,
		int verbose_level);
	int compare_monomials(int *M1, int *M2);
	int compare_monomials_PART(int *M1, int *M2);
	void print_monomial_ordering(std::ostream &ost);
	int *read_from_string_coefficient_pairs(std::string &str, int verbose_level);
	int *read_from_string_coefficient_vector(std::string &str, int verbose_level);


};


// #############################################################################
// longinteger_domain.cpp:
// #############################################################################

//! domain to compute with objects of type longinteger

class longinteger_domain {

public:
	int compare(longinteger_object &a, longinteger_object &b);
	int compare_unsigned(longinteger_object &a, longinteger_object &b);
		// returns -1 if a < b, 0 if a = b,
		// and 1 if a > b, treating a and b as unsigned.
	int is_less_than(longinteger_object &a, longinteger_object &b);
	void subtract_signless(longinteger_object &a,
		longinteger_object &b, longinteger_object &c);
		// c = a - b, assuming a > b
	void subtract_signless_in_place(longinteger_object &a,
		longinteger_object &b);
		// a := a - b, assuming a > b
	void add(longinteger_object &a,
		longinteger_object &b, longinteger_object &c);
	void add_mod(longinteger_object &a,
		longinteger_object &b, longinteger_object &c,
		longinteger_object &m, int verbose_level);
	void add_in_place(longinteger_object &a, longinteger_object &b);
		// a := a + b
	void subtract_in_place(
			longinteger_object &a, longinteger_object &b);
	// a := a - b
	void add_int_in_place(
			longinteger_object &a, long int b);
	void mult(longinteger_object &a,
		longinteger_object &b, longinteger_object &c);
	void mult_in_place(longinteger_object &a, longinteger_object &b);
	void mult_integer_in_place(longinteger_object &a, int b);
	void mult_mod(longinteger_object &a,
		longinteger_object &b, longinteger_object &c,
		longinteger_object &m, int verbose_level);
	void multiply_up(longinteger_object &a, int *x, int len, int verbose_level);
	void multiply_up_lint(
			longinteger_object &a, long int *x, int len, int verbose_level);
	int quotient_as_int(longinteger_object &a, longinteger_object &b);
	long int quotient_as_lint(longinteger_object &a, longinteger_object &b);
	void integral_division_exact(longinteger_object &a,
		longinteger_object &b, longinteger_object &a_over_b);
	void integral_division(
		longinteger_object &a, longinteger_object &b,
		longinteger_object &q, longinteger_object &r,
		int verbose_level);
	void integral_division_by_int(longinteger_object &a,
		int b, longinteger_object &q, int &r);
	void integral_division_by_lint(
		longinteger_object &a,
		long int b, longinteger_object &q, long int &r);
	void extended_gcd(longinteger_object &a, longinteger_object &b,
		longinteger_object &g, longinteger_object &u,
		longinteger_object &v, int verbose_level);
	int logarithm_base_b(longinteger_object &a, int b);
	void base_b_representation(longinteger_object &a,
		int b, int *&rep, int &len);
	void power_int(longinteger_object &a, int n);
	void power_int_mod(longinteger_object &a, int n,
		longinteger_object &m);
	void power_longint_mod(longinteger_object &a,
		longinteger_object &n, longinteger_object &m,
		int verbose_level);
	void square_root(
			longinteger_object &a, longinteger_object &sqrt_a,
			int verbose_level);
	int square_root_mod(int a, int p, int verbose_level);
		// solves x^2 = a mod p. Returns x


	void create_q_to_the_n(longinteger_object &a, int q, int n);
	void create_qnm1(longinteger_object &a, int q, int n);
	void create_Mersenne(longinteger_object &M, int n);
	// $M_n = 2^n - 1$
	void create_Fermat(longinteger_object &F, int n);
	// $F_n = 2^{2^n} + 1$
	void Dedekind_number(longinteger_object &Dnq, int n, int q, int verbose_level);

	int is_even(longinteger_object &a);
	int is_odd(longinteger_object &a);
	int remainder_mod_int(longinteger_object &a, int p);
	int multiplicity_of_p(longinteger_object &a,
		longinteger_object &residue, int p);
	long int smallest_primedivisor(longinteger_object &a, int p_min,
		int verbose_level);
	void factor_into_longintegers(longinteger_object &a,
		int &nb_primes, longinteger_object *&primes,
		int *&exponents, int verbose_level);
	void factor(longinteger_object &a, int &nb_primes,
		int *&primes, int *&exponents,
		int verbose_level);
	int jacobi(longinteger_object &a, longinteger_object &m,
		int verbose_level);

	void random_number_less_than_n(longinteger_object &n,
		longinteger_object &r);
	void random_number_with_n_decimals(
		longinteger_object &R, int n, int verbose_level);
	void matrix_product(longinteger_object *A, longinteger_object *B,
		longinteger_object *&C, int Am, int An, int Bn);
	void matrix_entries_integral_division_exact(longinteger_object *A,
		longinteger_object &b, int Am, int An);
	void matrix_print_GAP(std::ostream &ost, longinteger_object *A,
		int Am, int An);
	void matrix_print_tex(std::ostream &ost, longinteger_object *A,
		int Am, int An);
	void power_mod(char *aa, char *bb, char *nn,
		longinteger_object &result, int verbose_level);
	void factorial(longinteger_object &result, int n);
	void group_order_PGL(longinteger_object &result,
		int n, int q, int f_semilinear);
	void square_root_floor(longinteger_object &a,
			longinteger_object &x, int verbose_level);
	void print_digits(char *rep, int len);

};




// #############################################################################
// longinteger_object.cpp:
// #############################################################################


//! a class to represent arbitrary precision integers


class longinteger_object {

private:
	char sgn; // TRUE if negative
	int l;
	char *r;

public:
	longinteger_object();
	~longinteger_object();
	void freeself();

	char &ith(int i) { return r[i]; };
	char &sign() { return sgn; };
	int &len() { return l; };
	char *&rep() { return r; };
	void create(long int i, const char *file, int line);
	void create_product(int nb_factors, int *factors);
	void create_power(int a, int e);
		// creates a^e
	void create_power_minus_one(int a, int e);
		// creates a^e  - 1
	void create_from_base_b_representation(int b, int *rep, int len);
	void create_from_base_10_string(const char *str, int verbose_level);
	void create_from_base_10_string(const char *str);
	void create_from_base_10_string(std::string &str);
	int as_int();
	long int as_lint();
	void as_longinteger(longinteger_object &a);
	void assign_to(longinteger_object &b);
	void swap_with(longinteger_object &b);
	std::ostream& print(std::ostream& ost);
	std::ostream& print_not_scientific(std::ostream& ost);
	int log10();
	int output_width();
	void print_width(std::ostream& ost, int width);
	void print_to_string(char *str);
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
	void add_int(int a);
	void create_i_power_j(int i, int j);
	int compare_with_int(int a);
};

std::ostream& operator<<(std::ostream& ost, longinteger_object& p);






// #############################################################################
// partial_derivative.cpp
// #############################################################################

//! partial derivative connects two homogeneous polynomial domains


class partial_derivative {

public:
	homogeneous_polynomial_domain *H;
	homogeneous_polynomial_domain *Hd;
	int *v; // [H->get_nb_monomials()]
	int variable_idx;
	int *mapping; // [H->get_nb_monomials() * H->get_nb_monomials()]


	partial_derivative();
	~partial_derivative();
	void freeself();
	void null();
	void init(homogeneous_polynomial_domain *H,
			homogeneous_polynomial_domain *Hd,
			int variable_idx,
			int verbose_level);
	void apply(int *eqn_in,
			int *eqn_out,
			int verbose_level);
};


// #############################################################################
// ring_theory_global.cpp
// #############################################################################

//! global functions related to ring theory


class ring_theory_global {

public:
	ring_theory_global();
	~ring_theory_global();
	void write_code_for_division(
			finite_field *F,
			std::string &fname_code,
			std::string &A_coeffs, std::string &B_coeffs,
			int verbose_level);
	void polynomial_division(
			finite_field *F,
			std::string &A_coeffs, std::string &B_coeffs, int verbose_level);
	void extended_gcd_for_polynomials(
			finite_field *F,
			std::string &A_coeffs, std::string &B_coeffs, int verbose_level);
	void polynomial_mult_mod(
			finite_field *F,
			std::string &A_coeffs, std::string &B_coeffs, std::string &M_coeffs,
			int verbose_level);
	void polynomial_find_roots(
			finite_field *F,
			std::string &A_coeffs,
			int verbose_level);
	void sift_polynomials(
			finite_field *F,
			long int rk0, long int rk1, int verbose_level);
	void mult_polynomials(
			finite_field *F,
			long int rk0, long int rk1, int verbose_level);
	void polynomial_division_with_report(
			finite_field *F,
			long int rk0, long int rk1, int verbose_level);
	void polynomial_division_from_file_with_report(
			finite_field *F,
			std::string &input_file, long int rk1, int verbose_level);
	void polynomial_division_from_file_all_k_error_patterns_with_report(
			finite_field *F,
			std::string &input_file, long int rk1, int k, int verbose_level);
	void number_of_conditions_satisfied(
			finite_field *F,
			std::string &variety_label_txt,
			std::string &variety_label_tex,
			int variety_nb_vars, int variety_degree,
			std::vector<std::string> &Variety_coeffs,
			monomial_ordering_type Monomial_ordering_type,
			std::string &number_of_conditions_satisfied_fname,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
			int verbose_level);
	// creates homogeneous_polynomial_domain
	void create_intersection_of_zariski_open_sets(
			finite_field *F,
			std::string &variety_label_txt,
			std::string &variety_label_tex,
			int variety_nb_vars, int variety_degree,
			std::vector<std::string> &Variety_coeffs,
			monomial_ordering_type Monomial_ordering_type,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
			int verbose_level);
	// creates homogeneous_polynomial_domain
	void create_projective_variety(
			finite_field *F,
			std::string &variety_label,
			std::string &variety_label_tex,
			int variety_nb_vars, int variety_degree,
			std::string &variety_coeffs,
			monomial_ordering_type Monomial_ordering_type,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
			int verbose_level);
	// creates homogeneous_polynomial_domain
	void create_projective_curve(
			finite_field *F,
			std::string &variety_label_txt,
			std::string &variety_label_tex,
			int curve_nb_vars, int curve_degree,
			std::string &curve_coeffs,
			monomial_ordering_type Monomial_ordering_type,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
			int verbose_level);
	// creates homogeneous_polynomial_domain
	void create_irreducible_polynomial(
			finite_field *F,
			unipoly_domain *Fq,
			unipoly_object *&Beta, int n,
			long int *cyclotomic_set, int cylotomic_set_size,
			unipoly_object *&generator,
			int verbose_level);
	void compute_nth_roots_as_polynomials(
			finite_field *F,
			unipoly_domain *FpX,
			unipoly_domain *Fq, unipoly_object *&Beta, int n1, int n2, int verbose_level);
	void compute_powers(
			finite_field *F,
			unipoly_domain *Fq,
			int n, int start_idx,
			unipoly_object *&Beta, int verbose_level);
	void make_all_irreducible_polynomials_of_degree_d(finite_field *F,
			int d, std::vector<std::vector<int> > &Table,
			int verbose_level);
	int count_all_irreducible_polynomials_of_degree_d(finite_field *F,
			int d, int verbose_level);
	void do_make_table_of_irreducible_polynomials(finite_field *F,
			int deg, int verbose_level);
	void do_search_for_primitive_polynomial_in_range(
			int p_min, int p_max,
			int deg_min, int deg_max,
			int verbose_level);
	char *search_for_primitive_polynomial_of_given_degree(int p,
		int degree, int verbose_level);
	void search_for_primitive_polynomials(int p_min, int p_max,
		int n_min, int n_max, int verbose_level);
	void factor_cyclotomic(int n, int q, int d,
		int *coeffs, int f_poly, std::string &poly, int verbose_level);
	void oval_polynomial(
			finite_field *F,
		int *S, unipoly_domain &D, unipoly_object &poly,
		int verbose_level);
	void print_longinteger_after_multiplying(std::ostream &ost,
			int *factors, int len);

};


// #############################################################################
// table_of_irreducible_polynomials.cpp
// #############################################################################

//! a table of all irreducible polynomials over GF(q) of degree less than a certain value

class table_of_irreducible_polynomials {
public:
	int k;
	int q;
	finite_field *F;
	int nb_irred;
	int *Nb_irred;
	int *First_irred;
	int **Tables;
	int *Degree;

	table_of_irreducible_polynomials();
	~table_of_irreducible_polynomials();
	void init(int k, finite_field *F, int verbose_level);
	void print(std::ostream &ost);
	void print_polynomials(std::ostream &ost);
	int select_polynomial_first(
			int *Select, int verbose_level);
	int select_polynomial_next(
			int *Select, int verbose_level);
	int is_irreducible(unipoly_object &poly, int verbose_level);
	void factorize_polynomial(unipoly_object &char_poly, int *Mult,
		int verbose_level);
};


// #############################################################################
// unipoly_domain.cpp:
// #############################################################################

//! domain of polynomials in one variable over a finite field

class unipoly_domain {

private:
	finite_field *F;
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
	unipoly_domain(finite_field *GFq);
	void init_basic(finite_field *F, int verbose_level);
	unipoly_domain(finite_field *GFq, unipoly_object m, int verbose_level);
	~unipoly_domain();
	void init_variable_name(std::string &label);
	void init_factorring(finite_field *F, unipoly_object m, int verbose_level);
	finite_field *get_F();
	int &s_i(unipoly_object p, int i)
		{ int *rep = (int *) p; return rep[i + 1]; };
	void create_object_of_degree(unipoly_object &p, int d);
	void create_object_of_degree_no_test(unipoly_object &p, int d);
	void create_object_of_degree_with_coefficients(unipoly_object &p,
		int d, int *coeff);
	void create_object_by_rank(unipoly_object &p, long int rk,
			const char *file, int line, int verbose_level);
	void create_object_from_csv_file(
		unipoly_object &p, std::string &fname,
		const char *file, int line,
		int verbose_level);
	void create_object_by_rank_longinteger(unipoly_object &p,
		longinteger_object &rank,
		const char *file, int line,
		int verbose_level);
	void create_object_by_rank_string(
		unipoly_object &p, std::string &rk, int verbose_level);
	void create_Dickson_polynomial(unipoly_object &p, int *map);
	void delete_object(unipoly_object &p);
	void unrank(unipoly_object p, int rk);
	void unrank_longinteger(unipoly_object p, longinteger_object &rank);
	int rank(unipoly_object p);
	void rank_longinteger(unipoly_object p, longinteger_object &rank);
	int degree(unipoly_object p);
	void print_object(unipoly_object p, std::ostream &ost);
	void print_object_tight(unipoly_object p, std::ostream &ost);
	void print_object_sparse(unipoly_object p, std::ostream &ost);
	void print_object_dense(unipoly_object p, std::ostream &ost);
	void assign(unipoly_object a, unipoly_object &b, int verbose_level);
	void one(unipoly_object p);
	void m_one(unipoly_object p);
	void zero(unipoly_object p);
	int is_one(unipoly_object p);
	int is_zero(unipoly_object p);
	void negate(unipoly_object a);
	void make_monic(unipoly_object &a);
	void add(unipoly_object a, unipoly_object b, unipoly_object &c);
	void mult(unipoly_object a, unipoly_object b, unipoly_object &c, int verbose_level);
	void mult_mod(unipoly_object a,
		unipoly_object b, unipoly_object &c, unipoly_object m,
		int verbose_level);
	void mult_mod_negated(unipoly_object a, unipoly_object b, unipoly_object &c,
		int factor_polynomial_degree,
		int *factor_polynomial_coefficients_negated,
		int verbose_level);
	void Frobenius_matrix_by_rows(int *&Frob,
		unipoly_object factor_polynomial, int verbose_level);
		// the j-th row of Frob is x^{j*q} mod m
	void Frobenius_matrix(int *&Frob, unipoly_object factor_polynomial,
		int verbose_level);
	void Berlekamp_matrix(int *&B, unipoly_object factor_polynomial,
		int verbose_level);
	void exact_division(unipoly_object a,
		unipoly_object b, unipoly_object &q, int verbose_level);
	void division_with_remainder(unipoly_object a, unipoly_object b,
		unipoly_object &q, unipoly_object &r, int verbose_level);
	void derivative(unipoly_object a, unipoly_object &b);
	int compare_euclidean(unipoly_object m, unipoly_object n);
	void greatest_common_divisor(unipoly_object m, unipoly_object n,
		unipoly_object &g, int verbose_level);
	void extended_gcd(unipoly_object m, unipoly_object n,
		unipoly_object &u, unipoly_object &v,
		unipoly_object &g, int verbose_level);
	int is_squarefree(unipoly_object p, int verbose_level);
	void compute_normal_basis(int d, int *Normal_basis,
		int *Frobenius, int verbose_level);
	void order_ideal_generator(int d, int idx, unipoly_object &mue,
		int *A, int *Frobenius,
		int verbose_level);
		// Lueneburg~\cite{Lueneburg87a} p. 105.
		// Frobenius is a matrix of size d x d
		// A is a matrix of size (d + 1) x d
	void matrix_apply(unipoly_object &p, int *Mtx, int n,
		int verbose_level);
		// The matrix is applied on the left
	void substitute_matrix_in_polynomial(unipoly_object &p,
		int *Mtx_in, int *Mtx_out, int k, int verbose_level);
		// The matrix is substituted into the polynomial
	int substitute_scalar_in_polynomial(unipoly_object &p,
		int scalar, int verbose_level);
		// The scalar 'scalar' is substituted into the polynomial
	void module_structure_apply(int *v, int *Mtx, int n,
		unipoly_object p, int verbose_level);
	void take_away_all_factors_from_b(unipoly_object a,
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
	int is_irreducible(unipoly_object a, int verbose_level);
	void singer_candidate(unipoly_object &m,
		int p, int d, int b, int a);
	int is_primitive(unipoly_object &m,
		longinteger_object &qm1,
		int nb_primes, longinteger_object *primes,
		int verbose_level);
	void get_a_primitive_polynomial(unipoly_object &m,
		int f, int verbose_level);
	void get_an_irreducible_polynomial(unipoly_object &m,
		int f, int verbose_level);
	void power_int(unipoly_object &a, int n, int verbose_level);
	void power_longinteger(unipoly_object &a, longinteger_object &n, int verbose_level);
	void power_coefficients(unipoly_object &a, int n);
	void minimum_polynomial(unipoly_object &a,
		int alpha, int p, int verbose_level);
	int minimum_polynomial_factorring(int alpha, int p,
		int verbose_level);
	void minimum_polynomial_factorring_longinteger(
		longinteger_object &alpha,
		longinteger_object &rk_minpoly,
		int p, int verbose_level);
	void print_vector_of_polynomials(unipoly_object *sigma, int deg);
	void minimum_polynomial_extension_field(unipoly_object &g,
		unipoly_object m,
		unipoly_object &minpol, int d, int *Frobenius,
		int verbose_level);
		// Lueneburg~\cite{Lueneburg87a}, p. 112.
	void characteristic_polynomial(int *Mtx, int k,
		unipoly_object &char_poly, int verbose_level);
	void print_matrix(unipoly_object *M, int k);
	void determinant(unipoly_object *M, int k, unipoly_object &p,
		int verbose_level);
	void deletion_matrix(unipoly_object *M, int k, int delete_row,
		int delete_column, unipoly_object *&N, int verbose_level);
	void center_lift_coordinates(unipoly_object a, int q);
	void reduce_modulo_p(unipoly_object a, int p);

	//unipoly_domain2.cpp:
	void mult_easy(unipoly_object a, unipoly_object b, unipoly_object &c);
	void print_coeffs_top_down_assuming_one_character_per_digit(unipoly_object a, std::ostream &ost);
	void print_coeffs_top_down_assuming_one_character_per_digit_with_degree_given(
			unipoly_object a, int m, std::ostream &ost);
	void mult_easy_with_report(long int rk_a, long int rk_b, long int &rk_c,
			std::ostream &ost, int verbose_level);
	void division_with_remainder_from_file_with_report(std::string &input_fname, long int rk_b,
			long int &rk_q, long int &rk_r, std::ostream &ost, int verbose_level);
	void division_with_remainder_from_file_all_k_bit_error_patterns(
			std::string &input_fname, long int rk_b, int k,
			long int *&rk_q, long int *&rk_r, int &n, int &N, std::ostream &ost, int verbose_level);
	void division_with_remainder_numerically_with_report(long int rk_a, long int rk_b,
			long int &rk_q, long int &rk_r, std::ostream &ost, int verbose_level);
	void division_with_remainder_with_report(unipoly_object &a, unipoly_object &b,
			unipoly_object &q, unipoly_object &r,
			int f_report, std::ostream &ost, int verbose_level);


};

}}}


#endif /* SRC_LIB_FOUNDATIONS_RING_THEORY_RING_THEORY_H_ */
