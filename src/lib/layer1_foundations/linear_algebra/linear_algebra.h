/*
 * linear_algebra.h
 *
 *  Created on: Jan 10, 2022
 *      Author: betten
 */

#ifndef SRC_LIB_FOUNDATIONS_LINEAR_ALGEBRA_LINEAR_ALGEBRA_H_
#define SRC_LIB_FOUNDATIONS_LINEAR_ALGEBRA_LINEAR_ALGEBRA_H_


namespace orbiter {
namespace layer1_foundations {
namespace linear_algebra {



// #############################################################################
// gl_class_rep.cpp
// #############################################################################

//! description of a conjugacy class in GL(n,q) using the rational normal form

class gl_class_rep {

public:
	data_structures::int_matrix *type_coding;
	ring_theory::longinteger_object *centralizer_order;
	ring_theory::longinteger_object *class_length;

	gl_class_rep();
	~gl_class_rep();
	void init(
			int nb_irred, int *Select_polynomial,
		int *Select_partition, int verbose_level);
	void print(
			int nb_irred,  int *Select_polynomial,
			int *Select_partition, int verbose_level);
	void compute_vector_coding(
			gl_classes *C, int &nb_irred,
		int *&Poly_degree, int *&Poly_mult, int *&Partition_idx,
		int verbose_level);
	void centralizer_order_Kung(
			gl_classes *C,
			ring_theory::longinteger_object &co,
		int verbose_level);
};

// #############################################################################
// gl_classes.cpp
// #############################################################################

//! conjugacy classes in GL(n,q)

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
	void init(
			int k,
			field_theory::finite_field *F, int verbose_level);
	int select_partition_first(
			int *Select, int *Select_partition,
		int verbose_level);
	int select_partition_next(
			int *Select, int *Select_partition,
		int verbose_level);
	int first(
			int *Select, int *Select_partition, int verbose_level);
	int next(
			int *Select, int *Select_partition, int verbose_level);
	void make_matrix_from_class_rep(
			int *Mtx, gl_class_rep *R,
		int verbose_level);
	void make_matrix_in_rational_normal_form(
			int *Mtx, int *Select, int *Select_Partition,
			int verbose_level);
	void centralizer_order_Kung_basic(
			int nb_irreds,
		int *poly_degree, int *poly_mult, int *partition_idx,
		ring_theory::longinteger_object &co,
		int verbose_level);
	void centralizer_order_Kung(
			int *Select_polynomial,
		int *Select_partition,
		ring_theory::longinteger_object &co,
		int verbose_level);
		// Computes the centralizer order of a matrix in GL(k,q)
		// according to Kung's formula~\cite{Kung81}.
	void make_classes(
			gl_class_rep *&R, int &nb_classes,
		int f_no_eigenvalue_one,
		int verbose_level);
	void identify_matrix(
			int *Mtx, gl_class_rep *R, int *Basis,
		int verbose_level);
	void identify2(
			int *Mtx,
			ring_theory::unipoly_object &poly, int *Mult,
		int *Select_partition, int *Basis,
		int verbose_level);
	void compute_generalized_kernels_for_each_block(
		int *Mtx, int *Irreds, int nb_irreds,
		int *Degree, int *Mult, matrix_block_data *Data,
		int verbose_level);
	void compute_generalized_kernels(
			matrix_block_data *Data, int *M2,
		int d, int b0, int m, int *poly_coeffs,
		int verbose_level);
	int identify_partition(
			int *part, int m, int verbose_level);
	void choose_basis_for_rational_normal_form(
			int *Mtx,
		matrix_block_data *Data, int nb_irreds,
		int *Basis,
		int verbose_level);
	void choose_basis_for_rational_normal_form_block(
			int *Mtx,
		matrix_block_data *Data,
		int *Basis, int &b,
		int verbose_level);
	void generators_for_centralizer(
			int *Mtx, gl_class_rep *R,
		int *Basis, int **&Gens, int &nb_gens, int &nb_alloc,
		int verbose_level);
	void centralizer_generators(
			int *Mtx,
			ring_theory::unipoly_object &poly,
		int *Mult, int *Select_partition,
		int *Basis, int **&Gens, int &nb_gens, int &nb_alloc,
		int verbose_level);
	void centralizer_generators_block(
			int *Mtx,
			matrix_block_data *Data,
		int nb_irreds, int h,
		int **&Gens, int &nb_gens, int &nb_alloc,
		int verbose_level);
	int choose_basis_for_rational_normal_form_coset(
			int level1,
		int level2, int &coset,
		int *Mtx, matrix_block_data *Data, int &b, int *Basis,
		int verbose_level);
	int find_class_rep(
			gl_class_rep *Reps, int nb_reps,
		gl_class_rep *R, int verbose_level);
	void report(
			std::ostream &ost, int verbose_level);
	void print_matrix_and_centralizer_order_latex(
			std::ostream &ost,
		gl_class_rep *R);
};



// #############################################################################
// linear_algebra_global.cpp:
// #############################################################################

//! catch all class for functions related to linear algebra

class linear_algebra_global {
public:
	linear_algebra_global();
	~linear_algebra_global();
	void Berlekamp_matrix(
			field_theory::finite_field *F,
			std::string &Berlekamp_matrix_label,
			int verbose_level);
	void compute_normal_basis(
			field_theory::finite_field *F,
			int d, int verbose_level);
	void compute_normal_basis_with_given_polynomial(
			field_theory::finite_field *F,
			std::string &poly_encoded, int d, int verbose_level);
	void nullspace(
			field_theory::finite_field *F,
			int *M, int m, int n,
			int *&Nullspace, int &nullspace_m, int &nullspace_n,
			int *&A,
			int &rk_A,
			int *&base_cols,
			int verbose_level);
	void do_nullspace(
			field_theory::finite_field *F,
			std::string &input_matrix,
			int *&Nullspace, int &nullspace_m, int &nullspace_n,
			int verbose_level);
	void do_RREF(
			field_theory::finite_field *F,
			std::string &input_matrix,
			int verbose_level);
	void RREF_demo(
			field_theory::finite_field *F,
			int *A, int m, int n, int verbose_level);
	void RREF_with_steps_latex(
			field_theory::finite_field *F,
			std::ostream &ost, int *A, int m, int n,
			int verbose_level);
	int reverse_engineer_semilinear_map(
			field_theory::finite_field *F,
			int n,
		int *Elt, int *Mtx, int &frobenius,
		int verbose_level);

};



// #############################################################################
// linear_algebra.cpp:
// #############################################################################

//! linear algebra over a finite field

class linear_algebra {
public:
	field_theory::finite_field *F;


	linear_algebra();
	~linear_algebra();
	void init(
			field_theory::finite_field *F, int verbose_level);


	void copy_matrix(
			int *A, int *B, int ma, int na);
	void reverse_columns_of_matrix(
			int *A, int *B, int m, int n);
	void reverse_matrix(
			int *A, int *B, int ma, int na);
	void identity_matrix(
			int *A, int n);
	int is_identity_matrix(
			int *A, int n);
	int is_diagonal_matrix(
			int *A, int n);
	int is_scalar_multiple_of_identity_matrix(
			int *A,
		int n, int &scalar);
	void diagonal_matrix(
			int *A, int n, int alpha);
	void matrix_minor(
			int f_semilinear, int *A,
		int *B, int n, int f, int l);
		// initializes B as the l x l minor of A
		// (which is n x n) starting from row f.
	int minor_2x2(
			int *Elt, int n, int i, int j, int k, int l,
			int verbose_level);
	void wedge_product(
			int *Elt, int *Mtx2,
			int n, int n2, int verbose_level);
	void mult_vector_from_the_left(
			int *v, int *A,
		int *vA, int m, int n);
		// v[m], A[m][n], vA[n]
	void mult_vector_from_the_right(
			int *A, int *v,
		int *Av, int m, int n);
		// A[m][n], v[n], Av[m]

	void mult_matrix_matrix(
			int *A, int *B,
		int *C, int m, int n, int o, int verbose_level);
		// matrix multiplication C := A * B,
		// where A is m x n and B is n x o, so that C is m by o
	void semilinear_matrix_mult(
			int *A, int *B, int *AB, int n);
		// (A,f1) * (B,f2) = (A*B^{\varphi^{-f1}},f1+f2)
	void semilinear_matrix_mult_memory_given(
			int *A, int *B,
		int *AB, int *tmp_B, int n,
		int verbose_level);
		// (A,f1) * (B,f2) = (A*B^{\varphi^{-f1}},f1+f2)
	void matrix_mult_affine(
			int *A, int *B, int *AB,
		int n, int verbose_level);
	void semilinear_matrix_mult_affine(
			int *A, int *B, int *AB, int n);
	int matrix_determinant(
			int *A, int n, int verbose_level);
	void projective_action_from_the_right(
			int f_semilinear,
		int *v, int *A, int *vA, int n, int verbose_level);
		// vA = (v * A)^{p^f}  if f_semilinear (where f = A[n *  n]),
		// vA = v * A otherwise
	void general_linear_action_from_the_right(
			int f_semilinear,
		int *v, int *A, int *vA, int n, int verbose_level);
	void semilinear_action_from_the_right(
			int *v,
		int *A, int *vA, int n, int verbose_level);
		// vA = (v * A)^{p^f}  (where f = A[n *  n])
	void semilinear_action_from_the_left(
			int *A,
		int *v, int *Av, int n);
		// Av = (A * v)^{p^f}
	void affine_action_from_the_right(
			int f_semilinear,
		int *v, int *A, int *vA, int n);
		// vA = (v * A)^{p^f} + b
	void zero_vector(
			int *A, int m);
	void all_one_vector(
			int *A, int m);
	void support(
			int *A, int m, int *&support, int &size);
	void characteristic_vector(
			int *A, int m, int *set, int size);
	int is_zero_vector(
			int *A, int m);
	void add_vector(
			int *A, int *B, int *C, int m);
	void linear_combination_of_vectors(
			int a, int *A, int b, int *B, int *C, int len);
	void linear_combination_of_three_vectors(
			int a, int *A, int b, int *B,
			int c, int *C, int *D, int len);
	void negate_vector(
			int *A, int *B, int m);
	void negate_vector_in_place(
			int *A, int m);
	void scalar_multiply_vector_in_place(
			int c, int *A, int m);
	void vector_frobenius_power_in_place(
			int *A, int m, int f);
	int dot_product(
			int len, int *v, int *w);
	void transpose_matrix(
			int *A, int *At, int ma, int na);
	void transpose_matrix_in_place(
			int *A, int m);
	void transform_form_matrix(
			int *A, int *Gram,
		int *new_Gram, int d,
		int verbose_level);
		// computes new_Gram = A * Gram * A^\top
	int n_choose_k_mod_p(
			int n, int k, int verbose_level);
	void Dickson_polynomial(
			int *map, int *coeffs);
		// compute the coefficients of a degree q-1 polynomial
		// which interpolates a given map
		// from F_q to F_q
	void projective_action_on_columns_from_the_left(
			int *A,
		int *M, int m, int n, int *perm,
		int verbose_level);
	int evaluate_bilinear_form(
			int n, int *v1, int *v2, int *Gram);
	int evaluate_standard_hyperbolic_bilinear_form(
			int n,
		int *v1, int *v2);
	int evaluate_quadratic_form(
			int n, int nb_terms,
		int *i, int *j, int *coeff, int *x);
	void find_singular_vector_brute_force(
			int n, int form_nb_terms,
		int *form_i, int *form_j, int *form_coeff, int *Gram,
		int *vec, int verbose_level);
	void find_singular_vector(
			int n, int form_nb_terms,
		int *form_i, int *form_j, int *form_coeff, int *Gram,
		int *vec, int verbose_level);
	void complete_hyperbolic_pair(
			int n, int form_nb_terms,
		int *form_i, int *form_j, int *form_coeff, int *Gram,
		int *vec1, int *vec2, int verbose_level);
	void find_hyperbolic_pair(
			int n, int form_nb_terms,
		int *form_i, int *form_j, int *form_coeff, int *Gram,
		int *vec1, int *vec2, int verbose_level);
	void restrict_quadratic_form_list_coding(
			int k,
		int n, int *basis,
		int form_nb_terms,
		int *form_i, int *form_j, int *form_coeff,
		int &restricted_form_nb_terms,
		int *&restricted_form_i, int *&restricted_form_j,
		int *&restricted_form_coeff,
		int verbose_level);
	void restrict_quadratic_form(
			int k, int n, int *basis,
		int *C, int *D, int verbose_level);
	void exterior_square(
			int *An, int *An2, int n, int verbose_level);
	void lift_to_Klein_quadric(
			int *A4, int *A6, int verbose_level);


	// linear_algebra2.cpp

	void Kronecker_product(
			int *A, int *B, int n, int *AB);
	void Kronecker_product_square_but_arbitrary(
			int *A, int *B,
		int na, int nb, int *AB, int &N, int verbose_level);
	int dependency(
			int d, int *v, int *A, int m, int *rho,
		int verbose_level);
		// Lueneburg~\cite{Lueneburg87a} p. 104.
		// A is a matrix of size d + 1 times d
		// v[d]
		// rho is a column permutation of degree d
	void order_ideal_generator(
			int d, int idx, int *mue, int &mue_deg,
		int *A, int *Frobenius,
		int verbose_level);
		// Lueneburg~\cite{Lueneburg87a} p. 105.
		// Frobenius is a matrix of size d x d
		// A is (d + 1) x d
		// mue[d + 1]
	void span_cyclic_module(
			int *A, int *v, int n, int *Mtx,
		int verbose_level);
	void random_invertible_matrix(
			int *M, int k, int verbose_level);
	void vector_add_apply(
			int *v, int *w, int c, int n);
	void vector_add_apply_with_stride(
			int *v, int *w, int stride,
		int c, int n);
	int test_if_commute(
			int *A, int *B, int k, int verbose_level);
	void unrank_point_in_PG(
			int *v, int len, long int rk);
		// len is the length of the vector,
		// not the projective dimension
	long int rank_point_in_PG(
			int *v, int len);
	long int nb_points_in_PG(int n);
		// n is projective dimension
	void Borel_decomposition(
			int n, int *M, int *B1, int *B2,
		int *pivots, int verbose_level);
	void map_to_standard_frame(
			int d, int *A,
		int *Transform, int verbose_level);
		// d = vector space dimension
		// maps d + 1 points to the frame
		// e_1, e_2, ..., e_d, e_1+e_2+..+e_d
		// A is (d + 1) x d
		// Transform is d x d
	void map_frame_to_frame_with_permutation(
			int d, int *A,
		int *perm, int *B, int *Transform, int verbose_level);
	void map_points_to_points_projectively(
			int d, int k,
		int *A, int *B, int *Transform,
		int &nb_maps, int verbose_level);
		// A and B are (d + k + 1) x d
		// Transform is d x d
		// returns true if a map exists
	int BallChowdhury_matrix_entry(
			int *Coord, int *C,
		int *U, int k, int sz_U,
		int *T, int verbose_level);
	int is_unit_vector(
			int *v, int len, int k);
	void make_Fourier_matrices(
			int omega, int k, int *N, int **A, int **Av,
			int *Omega, int verbose_level);

	// linear_algebra3.cpp
	int evaluate_conic_form(
			int *six_coeffs, int *v3);
	int evaluate_quadric_form_in_PG_three(
			int *ten_coeffs, int *v4);
	int Pluecker_12(int *x4, int *y4);
	int Pluecker_21(int *x4, int *y4);
	int Pluecker_13(int *x4, int *y4);
	int Pluecker_31(int *x4, int *y4);
	int Pluecker_14(int *x4, int *y4);
	int Pluecker_41(int *x4, int *y4);
	int Pluecker_23(int *x4, int *y4);
	int Pluecker_32(int *x4, int *y4);
	int Pluecker_24(int *x4, int *y4);
	int Pluecker_42(int *x4, int *y4);
	int Pluecker_34(int *x4, int *y4);
	int Pluecker_43(int *x4, int *y4);
	int Pluecker_ij(
			int i, int j, int *x4, int *y4);
	int evaluate_symplectic_form(
			int len, int *x, int *y);
	int evaluate_symmetric_form(
			int len, int *x, int *y);
	int evaluate_quadratic_form_x0x3mx1x2(
			int *x);
	void solve_y2py(
			int a, int *Y2, int &nb_sol);
	void find_secant_points_wrt_x0x3mx1x2(
			int *Basis_line,
			int *Pts4, int &nb_pts, int verbose_level);
	int is_totally_isotropic_wrt_symplectic_form(
			int k,
		int n, int *Basis);
	int evaluate_monomial(
			int *monomial, int *variables, int nb_vars);



	// linear_algebra_RREF.cpp

	int Gauss_int(
			int *A, int f_special,
		int f_complete, int *base_cols,
		int f_P, int *P, int m, int n,
		int Pn, int verbose_level);
		// returns the rank which is the
		// number of entries in base_cols
		// A is m x n,
		// P is m x Pn (provided f_P is true)
	int Gauss_int_with_pivot_strategy(
		int *A,
		int f_special, int f_complete, int *pivot_perm,
		int m, int n,
		int (*find_pivot_function)(
				int *A, int m, int n, int r,
				int *pivot_perm, void *data),
		void *find_pivot_data,
		int verbose_level);
		// returns the rank which is the number of entries in pivots
		// A is a m x n matrix
	int Gauss_int_with_given_pivots(
			int *A,
		int f_special, int f_complete,
		int *pivots, int nb_pivots,
		int m, int n,
		int verbose_level);
		// A is a m x n matrix
		// returns false if pivot cannot be found at one of the steps
	int RREF_search_pivot(
			int *A, int m, int n,
			int &i, int &j, int *base_cols,
			int verbose_level);
	void RREF_make_pivot_one(
			int *A, int m, int n,
			int &i, int &j, int *base_cols,
			int verbose_level);
	void RREF_elimination_below(
			int *A, int m, int n,
			int &i, int &j, int *base_cols,
			int verbose_level);
	void RREF_elimination_above(
			int *A, int m, int n,
			int i, int *base_cols,
			int verbose_level);
	int rank_of_matrix(
			int *A, int m, int verbose_level);
	int rank_of_matrix_memory_given(
			int *A,
		int m, int *B, int *base_cols, int verbose_level);
	int rank_of_rectangular_matrix(
			int *A,
		int m, int n, int verbose_level);
	int rank_of_rectangular_matrix_memory_given(
			int *A,
		int m, int n, int *B, int *base_cols, int f_complete,
		int verbose_level);
	int rank_and_basecols(
			int *A, int m,
		int *base_cols, int verbose_level);
	void Gauss_step(
			int *v1, int *v2, int len,
		int idx, int verbose_level);
		// afterwards: v2[idx] = 0
		// and v1,v2 span the same space as before
		// v1 is not changed if v1[idx] is nonzero
	void Gauss_step_make_pivot_one(
			int *v1, int *v2,
		int len, int idx, int verbose_level);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
	void extend_basis(
			int m, int n, int *Basis, int verbose_level);
	int base_cols_and_embedding(
			int m, int n, int *A,
		int *base_cols, int *embedding, int verbose_level);
		// returns the rank rk of the matrix.
		// It also computes base_cols[rk] and embedding[m - rk]
		// It leaves A unchanged
	int Gauss_easy(
			int *A, int m, int n);
		// returns the rank
	int Gauss_easy_from_the_back(
			int *A, int m, int n);
	int Gauss_easy_memory_given(
			int *A, int m, int n, int *base_cols);
	int Gauss_simple(
			int *A, int m, int n,
		int *base_cols, int verbose_level);
		// returns the rank which is the
		// number of entries in base_cols
	void kernel_columns(
			int n, int nb_base_cols,
		int *base_cols, int *kernel_cols);
	void matrix_get_kernel_as_int_matrix(
			int *M, int m, int n,
		int *base_cols, int nb_base_cols,
		data_structures::int_matrix *kernel,
		int verbose_level);
	void matrix_get_kernel(
			int *M, int m, int n,
		int *base_cols, int nb_base_cols,
		int &kernel_m, int &kernel_n, int *kernel,
		int verbose_level);
		// kernel[n * (n - nb_base_cols)]
	int perp(
			int n, int k, int *A, int *Gram, int verbose_level);
	int RREF_and_kernel(
			int n, int k, int *A, int verbose_level);
	int perp_standard(
			int n, int k, int *A, int verbose_level);
	int perp_standard_with_temporary_data(
			int n, int k, int *A,
		int *B, int *K, int *base_cols,
		int verbose_level);
	int intersect_subspaces(
			int n, int k1, int *A, int k2, int *B,
		int &k3, int *intersection, int verbose_level);
	int compare_subspaces_ranked(
			int *set1, int *set2, int size,
		int vector_space_dimension, int verbose_level);
		// Compares the span of two sets of vectors.
		// returns 0 if equal, 1 if not
		// (this is so that it matches to the result
		// of a compare function)
	int compare_subspaces_ranked_with_unrank_function(
		int *set1, int *set2, int size,
		int vector_space_dimension,
		void (*unrank_point_func)(int *v, int rk, void *data),
		void *rank_point_data,
		int verbose_level);
	int Gauss_canonical_form_ranked(
			long int *set1, long int *set2, int size,
		int vector_space_dimension, int verbose_level);
		// Computes the Gauss canonical form
		// for the generating set in set1.
		// The result is written to set2.
		// Returns the rank of the span of the elements in set1.
	int lexleast_canonical_form_ranked(
			long int *set1, long int *set2, int size,
		int vector_space_dimension, int verbose_level);
		// Computes the lexleast generating set of the subspace
		// spanned by the elements in set1.
		// The result is written to set2.
		// Returns the rank of the span of the elements in set1.

	void get_coefficients_in_linear_combination(
		int k, int n, int *basis_of_subspace,
		int *input_vector, int *coefficients,
		int verbose_level);
		// basis[k * n]
		// coefficients[k]
		// input_vector[n] is the input vector.
		// At the end, coefficients[k] are
		// the coefficients of the linear combination
		// which expresses input_vector[n] in terms of
		// the given basis of the subspace.
	void reduce_mod_subspace_and_get_coefficient_vector(
		int k, int len, int *basis, int *base_cols,
		int *v, int *coefficients, int verbose_level);
	void reduce_mod_subspace(
			int k,
		int len, int *basis, int *base_cols,
		int *v, int verbose_level);
	int is_contained_in_subspace(
			int k,
		int len, int *basis, int *base_cols,
		int *v, int verbose_level);
	int is_subspace(
			int d, int dim_U, int *Basis_U, int dim_V,
		int *Basis_V, int verbose_level);
	void adjust_basis(
			int *V, int *U, int n, int k, int d,
		int verbose_level);
	void choose_vector_in_here_but_not_in_here_column_spaces(
		data_structures::int_matrix *V,
		data_structures::int_matrix *W,
		int *v, int verbose_level);
	void choose_vector_in_here_but_not_in_here_or_here_column_spaces(
			data_structures::int_matrix *V,
			data_structures::int_matrix *W1,
			data_structures::int_matrix *W2, int *v,
		int verbose_level);
	int
	choose_vector_in_here_but_not_in_here_or_here_column_spaces_coset(
		int &coset,
		data_structures::int_matrix *V,
		data_structures::int_matrix *W1,
		data_structures::int_matrix *W2, int *v,
		int verbose_level);
	void invert_matrix(
			int *A, int *A_inv, int n, int verbose_level);
	void invert_matrix_memory_given(
			int *A, int *A_inv, int n,
			int *tmp_A, int *tmp_basecols, int verbose_level);

	void matrix_inverse(
			int *A, int *Ainv, int n, int verbose_level);
	void matrix_invert(
			int *A, int *Tmp,
		int *Tmp_basecols, int *Ainv, int n, int verbose_level);
		// Tmp points to n * n + 1 int's
		// Tmp_basecols points to n int's
	void semilinear_matrix_invert(
			int *A, int *Tmp,
		int *Tmp_basecols, int *Ainv, int n, int verbose_level);
		// Tmp points to n * n + 1 int's
		// Tmp_basecols points to n int's
	// input: (A,f), output: (A^{-1}^{\Phi^f},-f mod e)
	void semilinear_matrix_invert_affine(
			int *A, int *Tmp,
		int *Tmp_basecols, int *Ainv, int n, int verbose_level);
		// Tmp points to n * n + 1 int's
		// Tmp_basecols points to n int's
	// input: (A,v,f),
	// output: (A^{-1}^{\Phi^f},-(v*A^{-1}^{\Phi^f})^{\Phi^{-f}},-f mod e)
	void matrix_invert_affine(
			int *A, int *Tmp, int *Tmp_basecols,
		int *Ainv, int n, int verbose_level);
		// Tmp points to n * n + 1 int's
		// Tmp_basecols points to n int's
	void intersect_with_subspace(
			int *Pt_coords, int nb_pts,
			int *Basis_save, int *Basis, int m, int n,
			long int *Intersection_idx,
			long int &intersection_sz,
			int verbose_level);
	// Pt_coords[nb_pts * n]
	// Basis_save[m * n]
	// Basis[(m + 1) * n]
	// Intersection_idx[nb_pts]

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
	void allocate(int k);
};



// #############################################################################
// representation_theory_domain.cpp:
// #############################################################################

//! catch all class for representation theory

class representation_theory_domain {
public:

	field_theory::finite_field *F;

	representation_theory_domain();
	~representation_theory_domain();
	void init(
			field_theory::finite_field *F,
			int verbose_level);
	void representing_matrix8_R(
			int *A,
		int q, int a, int b, int c, int d);
	void representing_matrix9_R(
			int *A,
		int q, int a, int b, int c, int d);
	void representing_matrix9_U(
			int *A,
		int a, int b, int c, int d, int beta);
	void representing_matrix8_U(
			int *A,
		int a, int b, int c, int d, int beta);
	void representing_matrix8_V(
			int *A, int beta);
	void representing_matrix9b(
			int *A, int beta);
	void representing_matrix8a(
			int *A,
		int a, int b, int c, int d, int beta);
	void representing_matrix8b(
			int *A, int beta);
	int Term1(
			int a1, int e1);
	int Term2(
			int a1, int a2,
			int e1, int e2);
	int Term3(
			int a1, int a2, int a3,
			int e1, int e2, int e3);
	int Term4(
			int a1, int a2, int a3, int a4,
			int e1, int e2, int e3, int e4);
	int Term5(
			int a1, int a2, int a3, int a4, int a5,
			int e1, int e2, int e3, int e4, int e5);
	int term1(int a1, int e1);
	int term2(
			int a1, int a2, int e1, int e2);
	int term3(
			int a1, int a2, int a3,
			int e1, int e2, int e3);
	int term4(
			int a1, int a2, int a3, int a4,
			int e1, int e2, int e3, int e4);
	int term5(
			int a1, int a2, int a3, int a4, int a5,
			int e1, int e2, int e3, int e4, int e5);
	int m_term(
			int q, int a1, int a2, int a3);
	int beta_trinomial(
			int q, int beta, int a1, int a2, int a3);
	int T3product2(
			int a1, int a2);
	int add(
			int a, int b);
	int add3(
			int a, int b, int c);
	int negate(
			int a);
	int twice(
			int a);
	int mult(
			int a, int b);
	int inverse(
			int a);
	int power(
			int a, int n);
	int T2(
			int a);
	int T3(
			int a);
	int N2(
			int a);
	int N3(
			int a);

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
	void init(
			field_theory::finite_field *F, int dimension,
			int verbose_level);
	void init_rank_functions(
		long int (*rank_point_func)(int *v, void *data),
		void (*unrank_point_func)(int *v, long int rk, void *data),
		void *data,
		int verbose_level);
	void unrank_basis(
			int *Mtx, long int *set, int len);
	void rank_basis(
			int *Mtx, long int *set, int len);
	void unrank_point(
			int *v, long int rk);
	long int rank_point(
			int *v);
	int RREF_and_rank(
			int *basis, int k);
	int is_contained_in_subspace(
			int *v, int *basis, int k);
	int compare_subspaces_ranked(
			long int *set1, long int *set2, int k, int verbose_level);
		// equality test for subspaces given by ranks of basis elements
};




}}}



#endif /* SRC_LIB_FOUNDATIONS_LINEAR_ALGEBRA_LINEAR_ALGEBRA_H_ */
