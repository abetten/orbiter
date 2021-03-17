// algebra_and_number_theory.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005


#ifndef ORBITER_SRC_LIB_FOUNDATIONS_ALGEBRA_AND_NUMBER_THEORY_H_
#define ORBITER_SRC_LIB_FOUNDATIONS_ALGEBRA_AND_NUMBER_THEORY_H_



namespace orbiter {
namespace foundations {

// #############################################################################
// a_domain.cpp
// #############################################################################

enum domain_kind {
	not_applicable, domain_the_integers, domain_integer_fractions
};


//! related to the computation of Young representations


class a_domain {
public:
	domain_kind kind;
	int size_of_instance_in_int;
	
	a_domain();
	~a_domain();
	void null();
	void freeself();
	
	void init_integers(int verbose_level);
	void init_integer_fractions(int verbose_level);
	int as_int(int *elt, int verbose_level);
	void make_integer(int *elt, int n, int verbose_level);
	void make_zero(int *elt, int verbose_level);
	void make_zero_vector(int *elt, int len, int verbose_level);
	int is_zero_vector(int *elt, int len, int verbose_level);
	int is_zero(int *elt, int verbose_level);
	void make_one(int *elt, int verbose_level);
	int is_one(int *elt, int verbose_level);
	void copy(int *elt_from, int *elt_to, int verbose_level);
	void copy_vector(int *elt_from, int *elt_to, 
		int len, int verbose_level);
	void swap_vector(int *elt1, int *elt2, int n, int verbose_level);
	void swap(int *elt1, int *elt2, int verbose_level);
	void add(int *elt_a, int *elt_b, int *elt_c, int verbose_level);
	void add_apply(int *elt_a, int *elt_b, int verbose_level);
	void subtract(int *elt_a, int *elt_b, int *elt_c, int verbose_level);
	void negate(int *elt, int verbose_level);
	void negate_vector(int *elt, int len, int verbose_level);
	void mult(int *elt_a, int *elt_b, int *elt_c, int verbose_level);
	void mult_apply(int *elt_a, int *elt_b, int verbose_level);
	void power(int *elt_a, int *elt_b, int n, int verbose_level);
	void divide(int *elt_a, int *elt_b, int *elt_c, int verbose_level);
	void inverse(int *elt_a, int *elt_b, int verbose_level);
	void print(int *elt);
	void print_vector(int *elt, int n);
	void print_matrix(int *A, int m, int n);
	void print_matrix_for_maple(int *A, int m, int n);
	void make_element_from_integer(int *elt, int n, int verbose_level);
	void mult_by_integer(int *elt, int n, int verbose_level);
	void divide_by_integer(int *elt, int n, int verbose_level);
	int *offset(int *A, int i);
	int Gauss_echelon_form(int *A, int f_special, int f_complete, 
		int *base_cols, 
		int f_P, int *P, int m, int n, int Pn, int verbose_level);
		// returns the rank which is the number 
		// of entries in base_cols
		// A is a m x n matrix,
		// P is a m x Pn matrix (if f_P is TRUE)
	void Gauss_step(int *v1, int *v2, int len, int idx, int verbose_level);
		// afterwards: v2[idx] = 0 and v1,v2 span the same space as before
		// v1 is not changed if v1[idx] is nonzero
	void matrix_get_kernel(int *M, int m, int n, int *base_cols, 
		int nb_base_cols, 
		int &kernel_m, int &kernel_n, int *kernel, int verbose_level);
		// kernel must point to the appropriate amount of memory! 
		// (at least n * (n - nb_base_cols) int's)
		// kernel is stored as column vectors, 
		// i.e. kernel_m = n and kernel_n = n - nb_base_cols.
	void matrix_get_kernel_as_row_vectors(int *M, int m, int n, 
		int *base_cols, int nb_base_cols, 
		int &kernel_m, int &kernel_n, int *kernel, int verbose_level);
		// kernel must point to the appropriate amount of memory! 
		// (at least n * (n - nb_base_cols) int's)
		// kernel is stored as row vectors, 
		// i.e. kernel_m = n - nb_base_cols and kernel_n = n.
	void get_image_and_kernel(int *M, int n, int &rk, int verbose_level);
	void complete_basis(int *M, int m, int n, int verbose_level);
	void mult_matrix(int *A, int *B, int *C, int ma, int na, int nb, 
		int verbose_level);
	void mult_matrix3(int *A, int *B, int *C, int *D, int n, 
		int verbose_level);
	void add_apply_matrix(int *A, int *B, int m, int n, 
		int verbose_level);
	void matrix_mult_apply_scalar(int *A, int *s, int m, int n, 
		int verbose_level);
	void make_block_matrix_2x2(int *Mtx, int n, int k, 
		int *A, int *B, int *C, int *D, int verbose_level);
		// A is k x k, 
		// B is k x (n - k), 
		// C is (n - k) x k, 
		// D is (n - k) x (n - k), 
		// Mtx is n x n
	void make_identity_matrix(int *A, int n, int verbose_level);
	void matrix_inverse(int *A, int *Ainv, int n, int verbose_level);
	void matrix_invert(int *A, int *T, int *basecols, int *Ainv, int n, 
		int verbose_level);

};


// #############################################################################
// algebra_global.cpp
// #############################################################################

//! global functions related to finite fields, irreducible polynomials and such

class algebra_global {
public:
	char *search_for_primitive_polynomial_of_given_degree(int p,
		int degree, int verbose_level);
	void search_for_primitive_polynomials(int p_min, int p_max,
		int n_min, int n_max, int verbose_level);
	void factor_cyclotomic(int n, int q, int d,
		int *coeffs, int f_poly, std::string &poly, int verbose_level);
	void count_subprimitive(int Q_max, int H_max);
	int eulers_totient_function(int n, int verbose_level);
	void formula_subprimitive(int d, int q,
		longinteger_object &Rdq, int &g, int verbose_level);
	void formula(int d, int q, longinteger_object &Rdq, int verbose_level);
	int subprimitive(int q, int h);
	int period_of_sequence(int *v, int l);
	void subexponent(int q, int Q, int h, int f, int j, int k, int &s, int &c);
	void gl_random_matrix(int k, int q, int verbose_level);
	const char *plus_minus_string(int epsilon);
	const char *plus_minus_letter(int epsilon);
	int PHG_element_normalize(finite_ring &R, int *v, int stride, int len);
	// last unit element made one
	int PHG_element_normalize_from_front(finite_ring &R, int *v,
		int stride, int len);
	// first non unit element made one
	int PHG_element_rank(finite_ring &R, int *v, int stride, int len);
	void PHG_element_unrank(finite_ring &R, int *v, int stride, int len, int rk);
	int nb_PHG_elements(int n, finite_ring &R);
	void display_all_PHG_elements(int n, int q);
	void test_unipoly();
	void test_unipoly2();
	int is_diagonal_matrix(int *A, int n);
	const char *get_primitive_polynomial(int p, int e, int verbose_level);
	void test_longinteger();
	void test_longinteger2();
	void test_longinteger3();
	void test_longinteger4();
	void test_longinteger5();
	void test_longinteger6();
	void test_longinteger7();
	void test_longinteger8();
	void longinteger_collect_setup(int &nb_agos,
			longinteger_object *&agos, int *&multiplicities);
	void longinteger_collect_free(int &nb_agos,
			longinteger_object *&agos, int *&multiplicities);
	void longinteger_collect_add(int &nb_agos,
			longinteger_object *&agos, int *&multiplicities,
			longinteger_object &ago);
	void longinteger_collect_print(std::ostream &ost,
			int &nb_agos, longinteger_object *&agos, int *&multiplicities);

	void make_all_irreducible_polynomials_of_degree_d(
			finite_field *F, int d, std::vector<std::vector<int> > &Table,
			int verbose_level);
	int count_all_irreducible_polynomials_of_degree_d(finite_field *F,
		int d, int verbose_level);
	void polynomial_division(finite_field *F,
			std::string &A_coeffs, std::string &B_coeffs, int verbose_level);
	void extended_gcd_for_polynomials(finite_field *F,
			std::string &A_coeffs, std::string &B_coeffs, int verbose_level);
	void polynomial_mult_mod(finite_field *F,
			std::string &A_coeffs, std::string &B_coeffs, std::string &M_coeffs,
			int verbose_level);
	void Berlekamp_matrix(finite_field *F,
			std::string &Berlekamp_matrix_coeffs,
			int verbose_level);
	void compute_normal_basis(finite_field *F, int d, int verbose_level);
	void do_nullspace(finite_field *F, int m, int n, std::string &text,
			int f_normalize_from_the_left, int f_normalize_from_the_right,
			int verbose_level);
	void do_RREF(finite_field *F, int m, int n, std::string &text,
			int f_normalize_from_the_left, int f_normalize_from_the_right,
			int verbose_level);
	void apply_Walsh_Hadamard_transform(finite_field *F,
			std::string &fname_csv_in, int n, int verbose_level);
	void algebraic_normal_form(finite_field *F,
			std::string &fname_csv_in, int n, int verbose_level);
	void apply_trace_function(finite_field *F,
			std::string &fname_csv_in, int verbose_level);
	void apply_power_function(finite_field *F,
			std::string &fname_csv_in, long int d, int verbose_level);
	void identity_function(finite_field *F,
			std::string &fname_csv_out, int verbose_level);
	void do_trace(finite_field *F, int verbose_level);
	void do_norm(finite_field *F, int verbose_level);
	void do_equivalence_class_of_fractions(int N, int verbose_level);
	void do_cheat_sheet_GF(finite_field *F, int verbose_level);
	void do_search_for_primitive_polynomial_in_range(int p_min, int p_max,
			int deg_min, int deg_max, int verbose_level);
	void do_make_table_of_irreducible_polynomials(int deg, finite_field *F, int verbose_level);
	void polynomial_find_roots(finite_field *F,
			std::string &A_coeffs,
			int verbose_level);
	void find_CRC_polynomials(finite_field *F,
			int t, int da, int dc,
			int verbose_level);
	void search_for_CRC_polynomials(int t,
			int da, int *A, int dc, int *C, int i, finite_field *F,
			long int &nb_sol, std::vector<std::vector<int> > &Solutions,
			int verbose_level);
	void search_for_CRC_polynomials_binary(int t,
			int da, int *A, int dc, int *C, int i,
			long int &nb_sol, std::vector<std::vector<int> > &Solutions,
			int verbose_level);
	int test_all_two_bit_patterns(int da, int *A, int dc, int *C,
			finite_field *F, int verbose_level);
	int test_all_three_bit_patterns(int da, int *A, int dc, int *C,
			finite_field *F, int verbose_level);
	int test_all_two_bit_patterns_binary(int da, int *A, int dc, int *C,
			int verbose_level);
	int test_all_three_bit_patterns_binary(int da, int *A, int dc, int *C,
			int verbose_level);
	int remainder_is_nonzero(int da, int *A, int db, int *B, finite_field *F);
	int remainder_is_nonzero_binary(int da, int *A, int db, int *B);
	void sift_polynomials(finite_field *F, long int rk0, long int rk1, int verbose_level);
	void mult_polynomials(finite_field *F, long int rk0, long int rk1, int verbose_level);
	void polynomial_division_from_file_with_report(finite_field *F,
			std::string &input_file, long int rk1, int verbose_level);
	void polynomial_division_from_file_all_k_error_patterns_with_report(finite_field *F,
			std::string &input_file, long int rk1, int k, int verbose_level);
	void polynomial_division_with_report(finite_field *F, long int rk0, long int rk1, int verbose_level);
	void RREF_demo(finite_field *F, int *A, int m, int n, int verbose_level);
	void RREF_demo2(std::ostream &ost, finite_field *F, int *A, int m, int n, int verbose_level);
	void Dedekind_numbers(int n_min, int n_max, int q_min, int q_max, int verbose_level);
	void order_of_q_mod_n(int q, int n_min, int n_max, int verbose_level);

};


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
};


// #############################################################################
// generators_symplectic_group.cpp
// #############################################################################


//! generators of the symplectic group


class generators_symplectic_group {
public:

	finite_field *F; // no ownership, do not destroy
	int n; // must be even
	int n_half; // n / 2
	int q;
	int qn; // = q^n

	int *nb_candidates; // [n + 1]
	int *cur_candidate; // [n]
	int **candidates; // [n + 1][q^n]
	
	int *Mtx; // [n * n]
	int *v; // [n]
	int *v2; // [n]
	int *w; // [n]
	int *Points; // [qn * n]

	int nb_gens;
	int *Data;
	int *transversal_length;

	generators_symplectic_group();
	~generators_symplectic_group();
	void null();
	void freeself();
	void init(finite_field *F, int n, int verbose_level);
	int count_strong_generators(int &nb, int *transversal_length, 
		int &first_moved, int depth, int verbose_level);
	int get_strong_generators(int *Data, int &nb, int &first_moved, 
		int depth, int verbose_level);
	//void backtrack_search(int &nb_sol, int depth, int verbose_level);
	void create_first_candidate_set(int verbose_level);
	void create_next_candidate_set(int level, int verbose_level);
	int dot_product(int *u1, int *u2);
};

// #############################################################################
// gl_classes.cpp
// #############################################################################

//! to list all conjugacy classes in GL(n,q)

class gl_classes {
public:
	int k;
	int q;
	finite_field *F;
	table_of_irreducible_polynomials *Table_of_polynomials;
	int *Nb_part;
	int **Partitions;
	//int *Degree;
	int *v, *w; // [k], used in choose_basis_for_rational_normal_form_block

	gl_classes();
	~gl_classes();
	void null();
	void freeself();
	void init(int k, finite_field *F, int verbose_level);
	int select_partition_first(int *Select, int *Select_partition, 
		int verbose_level);
	int select_partition_next(int *Select, int *Select_partition, 
		int verbose_level);
	int first(int *Select, int *Select_partition, int verbose_level);
	int next(int *Select, int *Select_partition, int verbose_level);
	void make_matrix_from_class_rep(int *Mtx, gl_class_rep *R, 
		int verbose_level);
	void make_matrix_in_rational_normal_form(
			int *Mtx, int *Select, int *Select_Partition,
			int verbose_level);
	void centralizer_order_Kung_basic(int nb_irreds, 
		int *poly_degree, int *poly_mult, int *partition_idx, 
		longinteger_object &co, 
		int verbose_level);
	void centralizer_order_Kung(int *Select_polynomial, 
		int *Select_partition, longinteger_object &co, 
		int verbose_level);
		// Computes the centralizer order of a matrix in GL(k,q) 
		// according to Kung's formula~\cite{Kung81}.
	void make_classes(gl_class_rep *&R, int &nb_classes, 
		int f_no_eigenvalue_one, int verbose_level);
	void identify_matrix(int *Mtx, gl_class_rep *R, int *Basis, 
		int verbose_level);
	void identify2(int *Mtx, unipoly_object &poly, int *Mult, 
		int *Select_partition, int *Basis, int verbose_level);
	void compute_generalized_kernels_for_each_block(
		int *Mtx, int *Irreds, int nb_irreds,
		int *Degree, int *Mult, matrix_block_data *Data,
		int verbose_level);
	void compute_generalized_kernels(matrix_block_data *Data, int *M2, 
		int d, int b0, int m, int *poly_coeffs, int verbose_level);
	int identify_partition(int *part, int m, int verbose_level);
	void choose_basis_for_rational_normal_form(int *Mtx, 
		matrix_block_data *Data, int nb_irreds, 
		int *Basis, 
		int verbose_level);
	void choose_basis_for_rational_normal_form_block(int *Mtx, 
		matrix_block_data *Data, 
		int *Basis, int &b, 
		int verbose_level);
	void generators_for_centralizer(int *Mtx, gl_class_rep *R, 
		int *Basis, int **&Gens, int &nb_gens, int &nb_alloc, 
		int verbose_level);
	void centralizer_generators(int *Mtx, unipoly_object &poly, 
		int *Mult, int *Select_partition, 
		int *Basis, int **&Gens, int &nb_gens, int &nb_alloc,  
		int verbose_level);
	void centralizer_generators_block(int *Mtx, matrix_block_data *Data, 
		int nb_irreds, int h, 
		int **&Gens, int &nb_gens, int &nb_alloc,  
		int verbose_level);
	int choose_basis_for_rational_normal_form_coset(int level1, 
		int level2, int &coset, 
		int *Mtx, matrix_block_data *Data, int &b, int *Basis, 
		int verbose_level);
	int find_class_rep(gl_class_rep *Reps, int nb_reps, 
		gl_class_rep *R, int verbose_level);
	void report(std::ostream &ost, int verbose_level);
	void print_matrix_and_centralizer_order_latex(std::ostream &ost,
		gl_class_rep *R);
};

// #############################################################################
// gl_class_rep.cpp
// #############################################################################

//! conjugacy class in GL(n,q) described using rational normal form

class gl_class_rep {
public:
	int_matrix type_coding;
	longinteger_object centralizer_order;
	longinteger_object class_length;

	gl_class_rep();
	~gl_class_rep();
	void init(int nb_irred, int *Select_polynomial, 
		int *Select_partition, int verbose_level);
	void print(int nb_irred,  int *Select_polynomial,
			int *Select_partition, int verbose_level);
	void compute_vector_coding(gl_classes *C, int &nb_irred, 
		int *&Poly_degree, int *&Poly_mult, int *&Partition_idx, 
		int verbose_level);
	void centralizer_order_Kung(gl_classes *C, longinteger_object &co, 
		int verbose_level);
};

// #############################################################################
// group_generators_domain.cpp
// #############################################################################

//! generators for various classes of groups

class group_generators_domain {
public:
	group_generators_domain();
	~group_generators_domain();
	void generators_symmetric_group(int deg,
		int &nb_perms, int *&perms, int verbose_level);
	void generators_cyclic_group(int deg,
		int &nb_perms, int *&perms, int verbose_level);
	void generators_dihedral_group(int deg,
		int &nb_perms, int *&perms, int verbose_level);
	void generators_dihedral_involution(int deg,
		int &nb_perms, int *&perms, int verbose_level);
	void generators_identity_group(int deg,
		int &nb_perms, int *&perms, int verbose_level);
	void generators_Hall_reflection(int nb_pairs,
			int &nb_perms, int *&perms, int &degree,
			int verbose_level);
	void generators_Hall_reflection_normalizer_group(int nb_pairs,
			int &nb_perms, int *&perms, int &degree,
			int verbose_level);
	void order_Hall_reflection_normalizer_factorized(int nb_pairs,
			int *&factors, int &nb_factors);
	void order_Bn_group_factorized(int n,
		int *&factors, int &nb_factors);
	void generators_Bn_group(int n, int &deg,
		int &nb_perms, int *&perms, int verbose_level);
	void generators_direct_product(int deg1, int nb_perms1, int *perms1,
		int deg2, int nb_perms2, int *perms2,
		int &deg3, int &nb_perms3, int *&perms3,
		int verbose_levels);
	void generators_concatenate(int deg1, int nb_perms1, int *perms1,
		int deg2, int nb_perms2, int *perms2,
		int &deg3, int &nb_perms3, int *&perms3,
		int verbose_level);
	int matrix_group_base_len_projective_group(int n, int q,
		int f_semilinear, int verbose_level);
	int matrix_group_base_len_affine_group(int n, int q,
		int f_semilinear, int verbose_level);
	int matrix_group_base_len_general_linear_group(int n, int q,
		int f_semilinear, int verbose_level);
	void order_POmega_epsilon(int epsilon, int m, int q,
		longinteger_object &o, int verbose_level);
	void order_PO_epsilon(int f_semilinear, int epsilon, int k, int q,
		longinteger_object &go, int verbose_level);
	// k is projective dimension
	void order_PO(int epsilon, int m, int q,
		longinteger_object &o,
		int verbose_level);
	void order_Pomega(int epsilon, int k, int q,
		longinteger_object &go,
		int verbose_level);
	void order_PO_plus(int m, int q,
		longinteger_object &o, int verbose_level);
	void order_PO_minus(int m, int q,
		longinteger_object &o, int verbose_level);
	// m = Witt index, the dimension is n = 2m+2
	void order_PO_parabolic(int m, int q,
		longinteger_object &o, int verbose_level);
	void order_Pomega_plus(int m, int q,
		longinteger_object &o, int verbose_level);
	// m = Witt index, the dimension is n = 2m
	void order_Pomega_minus(int m, int q,
		longinteger_object &o, int verbose_level);
	// m = half the dimension,
	// the dimension is n = 2m, the Witt index is m - 1
	void order_Pomega_parabolic(int m, int q, longinteger_object &o,
		int verbose_level);
	// m = Witt index, the dimension is n = 2m + 1
	int index_POmega_in_PO(int epsilon, int m, int q, int verbose_level);


	void diagonal_orbit_perm(int n, finite_field *F,
			long int *orbit, long int *orbit_inv, int verbose_level);
	void frobenius_orbit_perm(int n, finite_field *F,
		long int *orbit, long int *orbit_inv,
		int verbose_level);
	void projective_matrix_group_base_and_orbits(int n, finite_field *F,
		int f_semilinear,
		int base_len, int degree,
		long int *base, int *transversal_length,
		long int **orbit, long int **orbit_inv,
		int verbose_level);
	void projective_matrix_group_base_and_transversal_length(int n, finite_field *F,
		int f_semilinear,
		int base_len, int degree,
		long int *base, int *transversal_length,
		int verbose_level);
	void affine_matrix_group_base_and_transversal_length(int n, finite_field *F,
		int f_semilinear,
		int base_len, int degree,
		long int *base, int *transversal_length,
		int verbose_level);
	void general_linear_matrix_group_base_and_transversal_length(int n, finite_field *F,
		int f_semilinear,
		int base_len, int degree,
		long int *base, int *transversal_length,
		int verbose_level);
	void strong_generators_for_projective_linear_group(
		int n, finite_field *F,
		int f_semilinear,
		int *&data, int &size, int &nb_gens,
		int verbose_level);
	void strong_generators_for_affine_linear_group(
		int n, finite_field *F,
		int f_semilinear,
		int *&data, int &size, int &nb_gens,
		int verbose_level);
	void strong_generators_for_general_linear_group(
		int n, finite_field *F,
		int f_semilinear,
		int *&data, int &size, int &nb_gens,
		int verbose_level);
	void generators_for_parabolic_subgroup(
		int n, finite_field *F,
		int f_semilinear, int k,
		int *&data, int &size, int &nb_gens,
		int verbose_level);
	void generators_for_stabilizer_of_three_collinear_points_in_PGL4(
		int f_semilinear, finite_field *F,
		int *&data, int &size, int &nb_gens,
		int verbose_level);
	void generators_for_stabilizer_of_triangle_in_PGL4(
		int f_semilinear, finite_field *F,
		int *&data, int &size, int &nb_gens,
		int verbose_level);
	void builtin_transversal_rep_GLnq(int *A, int n, finite_field *F,
		int f_semilinear, int i, int j, int verbose_level);
	void affine_translation(int n, finite_field *F,
			int coordinate_idx,
			int field_base_idx, int *perm, int verbose_level);
		// perm points to q^n int's
		// field_base_idx is the base element whose
		// translation we compute, 0 \le field_base_idx < e
		// coordinate_idx is the coordinate in which we shift,
		// 0 \le coordinate_idx < n
	void affine_multiplication(int n, finite_field *F,
		int multiplication_order, int *perm, int verbose_level);
		// perm points to q^n int's
		// compute the diagonal multiplication by alpha, i.e.
		// the multiplication by alpha of each component
	void affine_frobenius(int n, finite_field *F,
			int k, int *perm, int verbose_level);
		// perm points to q^n int's
		// compute the diagonal action of the Frobenius
		// automorphism to the power k, i.e.,
		// raises each component to the p^k-th power
	int all_affine_translations_nb_gens(int n, finite_field *F);
	void all_affine_translations(int n, finite_field *F, int *gens);
	void affine_generators(int n, finite_field *F,
			int f_translations,
			int f_semilinear, int frobenius_power,
			int f_multiplication, int multiplication_order,
			int &nb_gens, int &degree, int *&gens,
			int &base_len, long int *&the_base, int verbose_level);


};


// #############################################################################
// heisenberg.cpp
// #############################################################################

//! Heisenberg group of n x n matrices


class heisenberg {

public:
	int q;
	finite_field *F;
	int n;
	int len; // 2 * n + 1
	int group_order; // q^len

	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *Elt4;

	heisenberg();
	~heisenberg();
	void null();
	void freeself();
	void init(finite_field *F, int n, int verbose_level);
	void unrank_element(int *Elt, long int rk);
	long int rank_element(int *Elt);
	void element_add(int *Elt1, int *Elt2, int *Elt3, int verbose_level);
	void element_negate(int *Elt1, int *Elt2, int verbose_level);
	int element_add_by_rank(int rk_a, int rk_b, int verbose_level);
	int element_negate_by_rank(int rk_a, int verbose_level);
	void group_table(int *&Table, int verbose_level);
	void group_table_abv(int *&Table_abv, int verbose_level);
	void generating_set(int *&gens, int &nb_gens, int verbose_level);


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
			//long int *Pts, int &nb_pts,
			int verbose_level);
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


};

int homogeneous_polynomial_domain_compare_monomial_with(void *data, 
	int i, void *data2, void *extra_data);
int homogeneous_polynomial_domain_compare_monomial(void *data, 
	int i, int j, void *extra_data);
void homogeneous_polynomial_domain_swap_monomial(void *data, 
	int i, int j, void *extra_data);
void HPD_callback_print_function(
		std::stringstream &ost, void *data, void *callback_data);
void HPD_callback_print_function2(
		std::stringstream &ost, void *data, void *callback_data);



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
};

void longinteger_print_digits(char *rep, int len);


// #############################################################################
// matrix_block_data.cpp
// #############################################################################

//! rational normal form of a matrix in GL(n,q) for gl_class_rep

class matrix_block_data {
public:
	int d;
	int m;
	int *poly_coeffs;
	int b0;
	int b1;

	int_matrix *K;
	int cnt;
	int *dual_part;
	int *part;
	int height;
	int part_idx;

	matrix_block_data();
	~matrix_block_data();
	void null();
	void freeself();
	void allocate(int k);
};

// #############################################################################
// norm_tables.cpp:
// #############################################################################

//! tables for the norm map in a finite field

class norm_tables {
public:
	int *norm_table;
	int *norm_table_sorted;
	int *sorting_perm, *sorting_perm_inv;
	int nb_types;
	int *type_first, *type_len;
	int *the_type;

	norm_tables();
	~norm_tables();
	void init(unusual_model &U, int verbose_level);
	int choose_an_element_of_given_norm(int norm, int verbose_level);
	
};


// #############################################################################
// null_polarity_generator.cpp:
// #############################################################################

//! construct all null polarities

class null_polarity_generator {
public:

	finite_field *F; // no ownership, do not destroy
	int n, q;
	int qn; // = q^n

	int *nb_candidates; // [n + 1]
	int *cur_candidate; // [n]
	int **candidates; // [n + 1][q^n]
	
	int *Mtx; // [n * n]
	int *v; // [n]
	int *w; // [n]
	int *Points; // [qn * n]

	int nb_gens;
	int *Data;
	int *transversal_length;

	null_polarity_generator();
	~null_polarity_generator();
	void null();
	void freeself();
	void init(finite_field *F, int n, int verbose_level);
	int count_strong_generators(int &nb, int *transversal_length, 
		int &first_moved, int depth, int verbose_level);
	int get_strong_generators(int *Data, int &nb, int &first_moved, 
		int depth, int verbose_level);
	void backtrack_search(int &nb_sol, int depth, int verbose_level);
	void create_first_candidate_set(int verbose_level);
	void create_next_candidate_set(int level, int verbose_level);
	int dot_product(int *u1, int *u2);
};

// #############################################################################
// number_theoretic_transform.cpp:
// #############################################################################

//! Fourier transform over finite fields

class number_theoretic_transform {
public:

	int k;
	int q;

	finite_field *F; // no ownership, do not destroy

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
	finite_field *FQ;
	int alphaQ;
	int psi;
	int *Psi_powers; // powers of psi


	number_theoretic_transform();
	~number_theoretic_transform();
	void init(std::string &fname_code, int k, int q, int verbose_level);
	void write_code(std::string &fname_code,
			int verbose_level);
	void write_code2(std::ostream &ost,
			int f_forward,
			int &nb_add, int &nb_negate, int &nb_mult,
			int verbose_level);
	void write_code_header(std::ostream &ost,
			std::string &fname_code, int verbose_level);
	void make_level(int s, int verbose_level);
	void paste(int **Xr, int **X, int s, int verbose_level);
	void make_G_matrix(int s, int verbose_level);
	void make_D_matrix(int s, int verbose_level);
	void make_T_matrix(int s, int verbose_level);
	void make_P_matrix(int s, int verbose_level);
	void multiply_matrix_stack(finite_field *F, int **S,
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
	long int order_mod_p(long int a, long int p);
	int int_log2(int n);
	int int_log10(int n);
	int lint_log10(long int n);
	int int_logq(int n, int q);
	// returns the number of digits in base q representation
	int lint_logq(long int n, int q);
	int is_strict_prime_power(int q);
	// assuming that q is a prime power, this fuction tests
	// whether or not q is a strict prime power
	int is_prime(int p);
	int is_prime_power(int q);
	int is_prime_power(int q, int &p, int &h);
	int smallest_primedivisor(int n);
	//Computes the smallest prime dividing $n$.
	//The algorithm is based on Lueneburg~\cite{Lueneburg87a}.
	int sp_ge(int n, int p_min);
	int factor_int(int a, int *&primes, int *&exponents);
	void factor_lint(long int a, std::vector<long int> &primes, std::vector<int> &exponents);
	void factor_prime_power(int q, int &p, int &e);
	long int primitive_root_randomized(long int p, int verbose_level);
	long int primitive_root(long int p, int verbose_level);
	int Legendre(long int a, long int p, int verbose_level);
	int Jacobi(long int a, long int m, int verbose_level);
	int Jacobi_with_key_in_latex(std::ostream &ost,
			long int a, long int m, int verbose_level);
	int ny2(long int x, long int &x1);
	int ny_p(long int n, long int p);
	//long int sqrt_mod_simple(long int a, long int p);
	void print_factorization(int nb_primes, int *primes, int *exponents);
	void print_longfactorization(int nb_primes,
		longinteger_object *primes, int *exponents);
	long int euler_function(long int n);
	long int moebius_function(long int n);
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
	long int ChineseRemainder2(long int a1, long int a2,
			long int p1, long int p2, int verbose_level);
	void do_babystep_giantstep(long int p, long int g, long int h,
			int f_latex, std::ostream &ost, int verbose_level);
	void sieve(std::vector<int> &primes,
			int factorbase, int verbose_level);
	void sieve_primes(std::vector<int> &v,
			int from, int to, int limit, int verbose_level);
	int nb_primes(int n);
	void cyclotomic_set(std::vector<int> &cyclotomic_set, int a, int q, int n, int verbose_level);

};

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
// rank_checker.cpp:
// #############################################################################


//! checking whether any d - 1 columns are linearly independent


class rank_checker {

public:
	finite_field *GFq;
	int m, n, d;
	
	int *M1; // [m * n]
	int *M2; // [m * n]
	int *base_cols; // [n]
	int *set; // [n] used in check_mindist

	rank_checker();
	~rank_checker();
	void init(finite_field *GFq, int m, int n, int d);
	int check_rank(int len, long int *S, int verbose_level);
	int check_rank_matrix_input(int len, long int *S, int dim_S,
		int verbose_level);
	int check_rank_last_two_are_fixed(int len, long int *S, int verbose_level);
	int compute_rank(int len, long int *S, int f_projective, int verbose_level);
	int compute_rank_row_vectors(
			int len, long int *S, int f_projective, int verbose_level);
};

// #############################################################################
// subfield_structure.cpp:
// #############################################################################

//! a finite field as a vector space over a subfield

class subfield_structure {
public:

	finite_field *FQ;
	finite_field *Fq;
	int Q;
	int q;
	int s; // subfield index: q^s = Q
	int *Basis;
		// [s], entries are elements in FQ

	int *embedding; 
		// [Q], entries are elements in FQ, 
		// indexed by elements in AG(s,q)
	int *embedding_inv;
		// [Q], entries are ranks of elements in AG(s,q), 
		// indexed by elements in FQ
		// the inverse of embedding

	int *components;
		// [Q * s], entries are elements in Fq
		// the vectors corresponding to the AG(s,q) 
		// ranks in embedding_inv[]

	int *FQ_embedding; 
		// [q] entries are elements in FQ corresponding to 
		// the elements in Fq
	int *Fq_element;
		// [Q], entries are the elements in Fq 
		// corresponding to a given FQ element
		// or -1 if the FQ element does not belong to Fq.
	int *v; // [s]
	
	subfield_structure();
	~subfield_structure();
	void null();
	void freeself();
	void init(finite_field *FQ, finite_field *Fq, int verbose_level);
	void init_with_given_basis(finite_field *FQ, finite_field *Fq, 
		int *given_basis, int verbose_level);
	void print_embedding();
	void report(std::ostream &ost);
	int evaluate_over_FQ(int *v);
	int evaluate_over_Fq(int *v);
	void lift_matrix(int *MQ, int m, int *Mq, int verbose_level);
	void retract_matrix(int *Mq, int n, int *MQ, int m, 
		int verbose_level);
	void Adelaide_hyperoval(
			long int *&Pts, int &nb_pts, int verbose_level);
	void create_adelaide_hyperoval(
			std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	void field_reduction(int *input, int sz, int *output,
			int verbose_level);
	// input[sz], output[s * (sz * n)],

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

	unipoly_domain(finite_field *GFq);
	unipoly_domain(finite_field *GFq, unipoly_object m, int verbose_level);
	~unipoly_domain();
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
	void create_object_by_rank_string(unipoly_object &p, 
		const char *rk, int verbose_level);
	void create_Dickson_polynomial(unipoly_object &p, int *map);
	void delete_object(unipoly_object &p);
	void unrank(unipoly_object p, int rk);
	void unrank_longinteger(unipoly_object p, longinteger_object &rank);
	int rank(unipoly_object p);
	void rank_longinteger(unipoly_object p, longinteger_object &rank);
	int degree(unipoly_object p);
	std::ostream& print_object(unipoly_object p, std::ostream& ost);
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
	void print_coeffs_top_down_assuming_one_character_per_digit_with_degree_given(unipoly_object a, int m, std::ostream &ost);
	void mult_easy_with_report(long int rk_a, long int rk_b, long int &rk_c, std::ostream &ost);
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


// #############################################################################
// vector_space.cpp:
// #############################################################################


//! finite dimensional vector space over a finite field


class vector_space {
public:

	int dimension;
	finite_field *F;

	long int (*rank_point_func)(int *v, void *data);
	void (*unrank_point_func)(int *v, long int rk, void *data);
	void *rank_point_data;
	int *v1; // [dimension]
	int *base_cols; // [dimension]
	int *base_cols2; // [dimension]
	int *M1; // [dimension * dimension]
	int *M2; // [dimension * dimension]


	vector_space();
	~vector_space();
	void null();
	void freeself();
	void init(finite_field *F, int dimension,
			int verbose_level);
	void init_rank_functions(
		long int (*rank_point_func)(int *v, void *data),
		void (*unrank_point_func)(int *v, long int rk, void *data),
		void *data,
		int verbose_level);
	void unrank_basis(int *Mtx, long int *set, int len);
	void rank_basis(int *Mtx, long int *set, int len);
	void unrank_point(int *v, long int rk);
	long int rank_point(int *v);
	int RREF_and_rank(int *basis, int k);
	int is_contained_in_subspace(int *v, int *basis, int k);
	int compare_subspaces_ranked(
			long int *set1, long int *set2, int k, int verbose_level);
		// equality test for subspaces given by ranks of basis elements
};


void vector_space_unrank_point_callback(int *v, long int rk, void *data);
long int vector_space_rank_point_callback(int *v, void *data);

}}


#endif /* ORBITER_SRC_LIB_FOUNDATIONS_ALGEBRA_AND_NUMBER_THEORY_H_ */



