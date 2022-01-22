// algebra.h
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
namespace layer1_foundations {
namespace algebra {

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
	void count_subprimitive(int Q_max, int H_max);
	void formula_subprimitive(int d, int q,
			ring_theory::longinteger_object &Rdq, int &g, int verbose_level);
	void formula(int d, int q, ring_theory::longinteger_object &Rdq, int verbose_level);
	int subprimitive(int q, int h);
	int period_of_sequence(int *v, int l);
	void subexponent(int q, int Q, int h, int f, int j, int k, int &s, int &c);
	const char *plus_minus_string(int epsilon);
	const char *plus_minus_letter(int epsilon);
	void display_all_PHG_elements(int n, int q);
	void test_unipoly();
	void test_unipoly2();
	int is_diagonal_matrix(int *A, int n);
	void test_longinteger();
	void test_longinteger2();
	void test_longinteger3();
	void test_longinteger4();
	void test_longinteger5();
	void test_longinteger6();
	void test_longinteger7();
	void test_longinteger8();
	void longinteger_collect_setup(int &nb_agos,
			ring_theory::longinteger_object *&agos, int *&multiplicities);
	void longinteger_collect_free(int &nb_agos,
			ring_theory::longinteger_object *&agos, int *&multiplicities);
	void longinteger_collect_add(int &nb_agos,
			ring_theory::longinteger_object *&agos, int *&multiplicities,
			ring_theory::longinteger_object &ago);
	void longinteger_collect_print(std::ostream &ost,
			int &nb_agos, ring_theory::longinteger_object *&agos, int *&multiplicities);
	void do_equivalence_class_of_fractions(int N, int verbose_level);





	void order_of_q_mod_n(int q, int n_min, int n_max, int verbose_level);
	void power_function_mod_n(int k, int n, int verbose_level);

	void do_trace(field_theory::finite_field *F, int verbose_level);
	void do_norm(field_theory::finite_field *F, int verbose_level);
	void do_cheat_sheet_GF(field_theory::finite_field *F, int verbose_level);
	void gl_random_matrix(field_theory::finite_field *F, int k, int verbose_level);

	// functions with file based input:
	void apply_Walsh_Hadamard_transform(field_theory::finite_field *F, std::string &fname_csv_in, int n, int verbose_level);
	void algebraic_normal_form(field_theory::finite_field *F, std::string &fname_csv_in, int n, int verbose_level);
	void apply_trace_function(field_theory::finite_field *F, std::string &fname_csv_in, int verbose_level);
	void apply_power_function(field_theory::finite_field *F, std::string &fname_csv_in, long int d, int verbose_level);
	void identity_function(field_theory::finite_field *F, std::string &fname_csv_out, int verbose_level);
	void Walsh_matrix(field_theory::finite_field *F, int n, int *&W, int verbose_level);
	void Vandermonde_matrix(field_theory::finite_field *F, int *&W, int *&W_inv, int verbose_level);
	void search_APN(field_theory::finite_field *F, int verbose_level);
	void search_APN_recursion(field_theory::finite_field *F,
			int *f, int depth, int &delta_min, int &nb_times,
			std::vector<std::vector<int> > &Solutions, int verbose_level);
	int non_linearity(field_theory::finite_field *F, int *f, int verbose_level);

	void O4_isomorphism_4to2(field_theory::finite_field *F,
		int *At, int *As, int &f_switch, int *B,
		int verbose_level);
	void O4_isomorphism_2to4(field_theory::finite_field *F,
		int *At, int *As, int f_switch, int *B);
	void O4_grid_coordinates_rank(field_theory::finite_field *F,
		int x1, int x2, int x3, int x4,
		int &grid_x, int &grid_y, int verbose_level);
	void O4_grid_coordinates_unrank(field_theory::finite_field *F,
		int &x1, int &x2, int &x3, int &x4, int grid_x,
		int grid_y, int verbose_level);
	void O4_find_tangent_plane(field_theory::finite_field *F,
		int pt_x1, int pt_x2, int pt_x3, int pt_x4,
		int *tangent_plane, int verbose_level);

};



// #############################################################################
// generators_symplectic_group.cpp
// #############################################################################


//! generators of the symplectic group


class generators_symplectic_group {
public:

	field_theory::finite_field *F; // no ownership, do not destroy
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
	void init(field_theory::finite_field *F, int n, int verbose_level);
	int count_strong_generators(int &nb, int *transversal_length, 
		int &first_moved, int depth, int verbose_level);
	int get_strong_generators(int *Data, int &nb, int &first_moved, 
		int depth, int verbose_level);
	void create_first_candidate_set(int verbose_level);
	void create_next_candidate_set(int level, int verbose_level);
	int dot_product(int *u1, int *u2);
};

// #############################################################################
// gl_class_rep.cpp
// #############################################################################

//! conjugacy class in GL(n,q) described using rational normal form

class gl_class_rep {
public:
	data_structures::int_matrix *type_coding;
	ring_theory::longinteger_object *centralizer_order;
	ring_theory::longinteger_object *class_length;

	gl_class_rep();
	~gl_class_rep();
	void init(int nb_irred, int *Select_polynomial, 
		int *Select_partition, int verbose_level);
	void print(int nb_irred,  int *Select_polynomial,
			int *Select_partition, int verbose_level);
	void compute_vector_coding(gl_classes *C, int &nb_irred, 
		int *&Poly_degree, int *&Poly_mult, int *&Partition_idx, 
		int verbose_level);
	void centralizer_order_Kung(gl_classes *C,
			ring_theory::longinteger_object &co,
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
			ring_theory::longinteger_object &o, int verbose_level);
	void order_PO_epsilon(int f_semilinear, int epsilon, int k, int q,
			ring_theory::longinteger_object &go, int verbose_level);
	// k is projective dimension
	void order_PO(int epsilon, int m, int q,
			ring_theory::longinteger_object &o,
		int verbose_level);
	void order_Pomega(int epsilon, int k, int q,
			ring_theory::longinteger_object &go,
		int verbose_level);
	void order_PO_plus(int m, int q,
			ring_theory::longinteger_object &o, int verbose_level);
	void order_PO_minus(int m, int q,
			ring_theory::longinteger_object &o, int verbose_level);
	// m = Witt index, the dimension is n = 2m+2
	void order_PO_parabolic(int m, int q,
			ring_theory::longinteger_object &o, int verbose_level);
	void order_Pomega_plus(int m, int q,
			ring_theory::longinteger_object &o, int verbose_level);
	// m = Witt index, the dimension is n = 2m
	void order_Pomega_minus(int m, int q,
			ring_theory::longinteger_object &o, int verbose_level);
	// m = half the dimension,
	// the dimension is n = 2m, the Witt index is m - 1
	void order_Pomega_parabolic(int m, int q, ring_theory::longinteger_object &o,
		int verbose_level);
	// m = Witt index, the dimension is n = 2m + 1
	int index_POmega_in_PO(int epsilon, int m, int q, int verbose_level);


	void diagonal_orbit_perm(int n, field_theory::finite_field *F,
			long int *orbit, long int *orbit_inv, int verbose_level);
	void frobenius_orbit_perm(int n, field_theory::finite_field *F,
		long int *orbit, long int *orbit_inv,
		int verbose_level);
	void projective_matrix_group_base_and_orbits(int n, field_theory::finite_field *F,
		int f_semilinear,
		int base_len, int degree,
		long int *base, int *transversal_length,
		long int **orbit, long int **orbit_inv,
		int verbose_level);
	void projective_matrix_group_base_and_transversal_length(int n, field_theory::finite_field *F,
		int f_semilinear,
		int base_len, int degree,
		long int *base, int *transversal_length,
		int verbose_level);
	void affine_matrix_group_base_and_transversal_length(int n, field_theory::finite_field *F,
		int f_semilinear,
		int base_len, int degree,
		long int *base, int *transversal_length,
		int verbose_level);
	void general_linear_matrix_group_base_and_transversal_length(int n, field_theory::finite_field *F,
		int f_semilinear,
		int base_len, int degree,
		long int *base, int *transversal_length,
		int verbose_level);
	void strong_generators_for_projective_linear_group(
		int n, field_theory::finite_field *F,
		int f_semilinear,
		int *&data, int &size, int &nb_gens,
		int verbose_level);
	void strong_generators_for_affine_linear_group(
		int n, field_theory::finite_field *F,
		int f_semilinear,
		int *&data, int &size, int &nb_gens,
		int verbose_level);
	void strong_generators_for_general_linear_group(
		int n, field_theory::finite_field *F,
		int f_semilinear,
		int *&data, int &size, int &nb_gens,
		int verbose_level);
	void generators_for_parabolic_subgroup(
		int n, field_theory::finite_field *F,
		int f_semilinear, int k,
		int *&data, int &size, int &nb_gens,
		int verbose_level);
	void generators_for_stabilizer_of_three_collinear_points_in_PGL4(
		int f_semilinear, field_theory::finite_field *F,
		int *&data, int &size, int &nb_gens,
		int verbose_level);
	void generators_for_stabilizer_of_triangle_in_PGL4(
		int f_semilinear, field_theory::finite_field *F,
		int *&data, int &size, int &nb_gens,
		int verbose_level);
	void builtin_transversal_rep_GLnq(int *A, int n, field_theory::finite_field *F,
		int f_semilinear, int i, int j, int verbose_level);
	void affine_translation(int n, field_theory::finite_field *F,
			int coordinate_idx,
			int field_base_idx, int *perm, int verbose_level);
		// perm points to q^n int's
		// field_base_idx is the base element whose
		// translation we compute, 0 \le field_base_idx < e
		// coordinate_idx is the coordinate in which we shift,
		// 0 \le coordinate_idx < n
	void affine_multiplication(int n, field_theory::finite_field *F,
		int multiplication_order, int *perm, int verbose_level);
		// perm points to q^n int's
		// compute the diagonal multiplication by alpha, i.e.
		// the multiplication by alpha of each component
	void affine_frobenius(int n, field_theory::finite_field *F,
			int k, int *perm, int verbose_level);
		// perm points to q^n int's
		// compute the diagonal action of the Frobenius
		// automorphism to the power k, i.e.,
		// raises each component to the p^k-th power
	int all_affine_translations_nb_gens(int n, field_theory::finite_field *F);
	void all_affine_translations(int n, field_theory::finite_field *F, int *gens);
	void affine_generators(int n, field_theory::finite_field *F,
			int f_translations,
			int f_semilinear, int frobenius_power,
			int f_multiplication, int multiplication_order,
			int &nb_gens, int &degree, int *&gens,
			int &base_len, long int *&the_base, int verbose_level);
	void PG_element_modified_not_in_subspace_perm(field_theory::finite_field *F,
			int n, int m,
		long int *orbit, long int *orbit_inv,
		int verbose_level);


};


// #############################################################################
// gl_classes.cpp
// #############################################################################

//! to list all conjugacy classes in GL(n,q)

class gl_classes {
public:
	int k;
	int q;
	field_theory::finite_field *F;
	ring_theory::table_of_irreducible_polynomials *Table_of_polynomials;
	int *Nb_part;
	int **Partitions;
	int *v, *w; // [k], used in choose_basis_for_rational_normal_form_block

	gl_classes();
	~gl_classes();
	void null();
	void freeself();
	void init(int k, field_theory::finite_field *F, int verbose_level);
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
		ring_theory::longinteger_object &co,
		int verbose_level);
	void centralizer_order_Kung(int *Select_polynomial,
		int *Select_partition, ring_theory::longinteger_object &co,
		int verbose_level);
		// Computes the centralizer order of a matrix in GL(k,q)
		// according to Kung's formula~\cite{Kung81}.
	void make_classes(gl_class_rep *&R, int &nb_classes,
		int f_no_eigenvalue_one, int verbose_level);
	void identify_matrix(int *Mtx, gl_class_rep *R, int *Basis,
		int verbose_level);
	void identify2(int *Mtx, ring_theory::unipoly_object &poly, int *Mult,
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
	void centralizer_generators(int *Mtx, ring_theory::unipoly_object &poly,
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
// heisenberg.cpp
// #############################################################################

//! Heisenberg group of n x n matrices


class heisenberg {

public:
	int q;
	field_theory::finite_field *F;
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
	void init(field_theory::finite_field *F, int n, int verbose_level);
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

	data_structures::int_matrix *K;
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
// null_polarity_generator.cpp:
// #############################################################################

//! construction of all null polarities

class null_polarity_generator {
public:

	field_theory::finite_field *F; // no ownership, do not destroy
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
	void init(field_theory::finite_field *F, int n, int verbose_level);
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
// rank_checker.cpp:
// #############################################################################


//! to check whether any d - 1 elements of a given set are linearly independent


class rank_checker {

public:
	field_theory::finite_field *GFq;
	int m, n, d;
	
	int *M1; // [m * n]
	int *M2; // [m * n]
	int *base_cols; // [n]
	int *set; // [n] used in check_mindist

	rank_checker();
	~rank_checker();
	void init(field_theory::finite_field *GFq, int m, int n, int d);
	int check_rank(int len, long int *S, int verbose_level);
	int check_rank_matrix_input(int len, long int *S, int dim_S,
		int verbose_level);
	int check_rank_last_two_are_fixed(int len, long int *S, int verbose_level);
	int compute_rank(int len, long int *S, int f_projective, int verbose_level);
	int compute_rank_row_vectors(
			int len, long int *S, int f_projective, int verbose_level);
};



// #############################################################################
// vector_space.cpp:
// #############################################################################


//! finite dimensional vector space over a finite field


class vector_space {
public:

	int dimension;
	field_theory::finite_field *F;

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
	void init(field_theory::finite_field *F, int dimension,
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



}}}


#endif /* ORBITER_SRC_LIB_FOUNDATIONS_ALGEBRA_AND_NUMBER_THEORY_H_ */



