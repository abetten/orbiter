/*
 * number_theory.h
 *
 *  Created on: Nov 2, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_FOUNDATIONS_NUMBER_THEORY_NUMBER_THEORY_H_
#define SRC_LIB_FOUNDATIONS_NUMBER_THEORY_NUMBER_THEORY_H_



namespace orbiter {
namespace layer1_foundations {
namespace number_theory {



// #############################################################################
// cyclotomic_sets.cpp
// #############################################################################


//! cyclotomic sets for cyclic codes



class cyclotomic_sets {
public:
	int n;
	int q;
	int m;
	int qm;

	int *Index;
	data_structures::set_of_sets *S;

	cyclotomic_sets();
	~cyclotomic_sets();
	void init(
			field_theory::finite_field *F,
			int n, int verbose_level);
	void print();
	void print_latex(std::ostream &ost);
	void print_latex_with_selection(
			std::ostream &ost, int *Selection, int nb_sel);

};


// #############################################################################
// elliptic_curve.cpp
// #############################################################################



//! a fixed elliptic curve in Weierstrass form



class elliptic_curve {

public:
	int q;
	int p;
	int e;
	int b, c;
		// the equation of the curve is
		// Y^2 = X^3 + bX + c mod p
	int nb; // number of points
	int *T; // [nb * 3] point coordinates
		// the point at infinity is last
	int *A; // [nb * nb] addition table
	field_theory::finite_field *F;


	elliptic_curve();
	~elliptic_curve();
	void init(
			field_theory::finite_field *F,
			int b, int c, int verbose_level);
	void compute_points(int verbose_level);
	void add_point_to_table(
			int x, int y, int z);
	int evaluate_RHS(int x);
		// evaluates x^3 + bx + c
	void print_points();
	void print_points_affine();
	void addition(
		int x1, int y1, int z1,
		int x2, int y2, int z2,
		int &x3, int &y3, int &z3, int verbose_level);
	void save_incidence_matrix(
			std::string &fname, int verbose_level);
	void draw_grid(
			std::string &fname,
			graphics::layered_graph_draw_options *Draw_options,
			int f_with_grid, int f_with_points, int point_density,
			int f_path, int start_idx, int nb_steps,
			int verbose_level);
	void draw_grid2(
			graphics::mp_graphics &G,
			int f_with_grid, int f_with_points, int point_density,
			int f_path, int start_idx, int nb_steps,
			int verbose_level);
	void make_affine_point(
			int x1, int x2, int x3,
		int &a, int &b, int verbose_level);
	void compute_addition_table(int verbose_level);
	void print_addition_table();
	int index_of_point(
			int x1, int x2, int x3);
	void latex_points_with_order(
			std::ostream &ost);
	void latex_order_of_all_points(
			std::ostream &ost);
	void order_of_all_points(
			std::vector<int> &Ord);
	int order_of_point(int i);
	void print_all_powers(int i);
};



// #############################################################################
// number_theoretic_transform.cpp:
// #############################################################################

//! Fourier transform over finite fields

class number_theoretic_transform {
public:

	int k;
	int q;

	std::string fname_code;

	field_theory::finite_field *F; // no ownership, do not destroy

	int *N;

	int alpha, omega;
	int gamma, minus_gamma, minus_one;
	int **A; // Fourier matrices for the positively wrapped convolution
	int **Av;
	int **A_log;
	int *Omega;


	int **G;
	int **D;
	int **Dv;
	int **T;
	int **Tv;
	int **P;

	int *X, *Y, *Z;
	int *X1, *X2;
	int *Y1, *Y2;

	int **Gr;
	int **Dr;
	int **Dvr;
	int **Tr;
	int **Tvr;
	int **Pr;

	int *Tmp1;
	int *Tmp2;

	int *bit_reversal;

	int Q;
	field_theory::finite_field *FQ;
	int alphaQ;
	int psi;
	int *Psi_powers; // powers of psi


	number_theoretic_transform();
	~number_theoretic_transform();
	void init(
			field_theory::finite_field *F,
			int k, int q, int verbose_level);
	void write_code(
			std::string &fname_code,
			int verbose_level);
	void write_code2(
			std::ostream &ost,
			int f_forward,
			int &nb_add, int &nb_negate, int &nb_mult,
			int verbose_level);
	void write_code_header(
			std::ostream &ost,
			std::string &fname_code, int verbose_level);
	void make_level(
			int s, int verbose_level);
	void paste(
			int **Xr, int **X, int s, int verbose_level);
	void make_G_matrix(
			int s, int verbose_level);
	void make_D_matrix(
			int s, int verbose_level);
	void make_T_matrix(
			int s, int verbose_level);
	void make_P_matrix(
			int s, int verbose_level);
	void multiply_matrix_stack(
			field_theory::finite_field *F, int **S,
			int nb, int sz, int *Result, int verbose_level);
};

// #############################################################################
// number_theory_domain.cpp:
// #############################################################################

//! basic number theoretic functions


class number_theory_domain {

public:
	number_theory_domain();
	~number_theory_domain();
	long int mod(long int a, long int p);
	long int int_negate(long int a, long int p);
	long int power_mod(long int a, long int n, long int p);
	long int inverse_mod(long int a, long int p);
	long int mult_mod(long int a, long int b, long int p);
	long int add_mod(long int a, long int b, long int p);
	long int ab_over_c(long int a, long int b, long int c);
	long int int_abs(long int a);
	long int gcd_lint(long int m, long int n);
	void extended_gcd_int(int m, int n, int &g, int &u, int &v);
	void extended_gcd_lint(long int m, long int n,
			long int &g, long int &u, long int &v);
	long int gcd_with_key_in_latex(std::ostream &ost,
			long int a, long int b, int f_key, int verbose_level);
	int i_power_j_safe(int i, int j);
	long int i_power_j_lint_safe(int i, int j, int verbose_level);
	long int i_power_j_lint(long int i, long int j);
	int i_power_j(int i, int j);
	void do_eulerfunction_interval(
			long int n_min, long int n_max, int verbose_level);
	long int euler_function(long int n);
	long int moebius_function(long int n);
	long int order_mod_p(long int a, long int p);
	int int_log2(int n);
	int int_log10(int n);
	int lint_log10(long int n);
	int int_logq(int n, int q);
	// returns the number of digits in base q representation
	int lint_logq(long int n, int q);
	int is_strict_prime_power(int q);
	// assuming that q is a prime power, this function tests
	// if q is a strict prime power
	int is_prime(int p);
	int is_prime_power(int q);
	int is_prime_power(int q, int &p, int &h);
	int smallest_primedivisor(int n);
	//Computes the smallest prime dividing $n$.
	//The algorithm is based on Lueneburg~\cite{Lueneburg87a}.
	int sp_ge(int n, int p_min);
	int factor_int(int a, int *&primes, int *&exponents);
	int nb_prime_factors_counting_multiplicities(long int a);
	int nb_distinct_prime_factors(long int a);
	void factor_lint(
			long int a,
			std::vector<long int> &primes,
			std::vector<int> &exponents);
	void factor_prime_power(int q, int &p, int &e);
	long int primitive_root_randomized(
			long int p, int verbose_level);
	long int primitive_root(
			long int p, int verbose_level);
	int Legendre(
			long int a, long int p, int verbose_level);
	int Jacobi(
			long int a, long int m, int verbose_level);
	int Jacobi_with_key_in_latex(
			std::ostream &ost,
			long int a, long int m, int verbose_level);
	int Legendre_with_key_in_latex(
			std::ostream &ost,
			long int a, long int m, int verbose_level);
	//Computes the Legendre symbol $\left( \frac{a}{m} \right)$.
	int ny2(long int x, long int &x1);
	int ny_p(long int n, long int p);
	//long int sqrt_mod_simple(long int a, long int p);
	void print_factorization(
			int nb_primes, int *primes, int *exponents);
	void print_longfactorization(
			int nb_primes,
			ring_theory::longinteger_object *primes,
			int *exponents);
	void int_add_fractions(int at, int ab, int bt, int bb,
		int &ct, int &cb, int verbose_level);
	void int_mult_fractions(int at, int ab, int bt, int bb,
		int &ct, int &cb, int verbose_level);
	int choose_a_prime_greater_than(int lower_bound);
	int choose_a_prime_in_interval(int lower_bound, int upper_bound);
	int random_integer_greater_than(int n, int lower_bound);
	int random_integer_in_interval(int lower_bound, int upper_bound);
	int nb_primes_available();
	int get_prime_from_table(int idx);
	long int Chinese_Remainders(
			std::vector<long int> &Remainders,
			std::vector<long int> &Moduli,
			long int &M, int verbose_level);
	long int ChineseRemainder2(long int a1, long int a2,
			long int p1, long int p2, int verbose_level);
	void sieve(std::vector<int> &primes,
			int factorbase, int verbose_level);
	void sieve_primes(std::vector<int> &v,
			int from, int to, int limit, int verbose_level);
	int nb_primes(int n);
	void cyclotomic_set(
			std::vector<int> &cyclotomic_set,
			int a, int q, int n, int verbose_level);
	void elliptic_curve_addition(
			field_theory::finite_field *F,
			int b, int c,
		int x1, int x2, int x3,
		int y1, int y2, int y3,
		int &z1, int &z2, int &z3, int verbose_level);
	void elliptic_curve_point_multiple(
			field_theory::finite_field *F,
			int b, int c, int n,
		int x1, int y1, int z1,
		int &x3, int &y3, int &z3,
		int verbose_level);
	void elliptic_curve_point_multiple_with_log(
			field_theory::finite_field *F,
			int b, int c, int n,
		int x1, int y1, int z1,
		int &x3, int &y3, int &z3,
		int verbose_level);
	int elliptic_curve_evaluate_RHS(
			field_theory::finite_field *F,
			int x, int b, int c);
	void elliptic_curve_points(
			field_theory::finite_field *F,
			int b, int c, int &nb, int *&T,
			int verbose_level);
	void elliptic_curve_all_point_multiples(
			field_theory::finite_field *F,
			int b, int c, int &order,
		int x1, int y1, int z1,
		std::vector<std::vector<int> > &Pts,
		int verbose_level);
	int elliptic_curve_discrete_log(
			field_theory::finite_field *F,
			int b, int c,
		int x1, int y1, int z1,
		int x3, int y3, int z3,
		int verbose_level);
	int eulers_totient_function(
			int n, int verbose_level);
	void do_jacobi(
			long int jacobi_top,
			long int jacobi_bottom, int verbose_level);
	void elliptic_curve_addition_table(
			geometry::projective_space *P2,
		int *A6, int *Pts, int nb_pts, int *&Table,
		int verbose_level);
	int elliptic_curve_addition(
			geometry::projective_space *P2,
		int *A6, int p1_rk, int p2_rk,
		int verbose_level);

};




}}}



#endif /* SRC_LIB_FOUNDATIONS_NUMBER_THEORY_NUMBER_THEORY_H_ */
