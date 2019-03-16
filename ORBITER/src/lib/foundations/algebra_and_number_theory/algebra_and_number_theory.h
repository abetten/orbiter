// algebra_and_number_theory.h
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
// finite_field.cpp
// #############################################################################

//! finite field Fq

class finite_field {

private:
	int f_has_table;
	int *add_table; // [q * q]
	int *mult_table; // [q * q]
		// add_table and mult_table are needed in mindist

	int *negate_table;
	int *inv_table;
	int *frobenius_table; // x \mapsto x^p
	int *absolute_trace_table;
	int *log_alpha_table;
	int *alpha_power_table;
	int *v1, *v2, *v3; // vectors of length e.
	char *symbol_for_print;

public:
	const char *override_poly;
	char *polynomial;
		// the actual polynomial we consider 
		// as integer (in text form)
	int q, p, e;
	int alpha; // primitive element
	int log10_of_q; // needed for printing purposes
	int f_print_as_exponentials;
	
	finite_field();
	void null();
	~finite_field();
	void print_call_stats(std::ostream &ost);
	void init(int q);
	void init(int q, int verbose_level);
	void init_symbol_for_print(const char *symbol);
	void init_override_polynomial(int q, const char *poly, 
		int verbose_level);
	int compute_subfield_polynomial(int order_subfield, 
		int verbose_level);
	void compute_subfields(int verbose_level);
	void create_alpha_table(int verbose_level);
	void create_alpha_table_extension_field(int verbose_level);
	void create_alpha_table_prime_field(int verbose_level);
	void create_tables_prime_field(int verbose_level);
	void create_tables_extension_field(int verbose_level);
	int *private_add_table();
	int *private_mult_table();
	int zero();
	int one();
	int minus_one();
	int is_zero(int i);
	int is_one(int i);
	int mult(int i, int j);
	int mult3(int a1, int a2, int a3);
	int product3(int a1, int a2, int a3);
	int mult4(int a1, int a2, int a3, int a4);
	int product4(int a1, int a2, int a3, int a4);
	int product5(int a1, int a2, int a3, int a4, int a5);
	int product_n(int *a, int n);
	int square(int a);
	int twice(int a);
	int four_times(int a);
	int Z_embedding(int k);
	int add(int i, int j);
	int add3(int i1, int i2, int i3);
	int add4(int i1, int i2, int i3, int i4);
	int add5(int i1, int i2, int i3, int i4, int i5);
	int add6(int i1, int i2, int i3, int i4, int i5, int i6);
	int add7(int i1, int i2, int i3, int i4, int i5, int i6, 
		int i7);
	int add8(int i1, int i2, int i3, int i4, int i5, int i6, 
		int i7, int i8);
	int negate(int i);
	int inverse(int i);
	int power(int a, int n); // computes a^n
	int frobenius_power(int a, int i); // computes a^{p^i}
	int absolute_trace(int i);
	int absolute_norm(int i);
	int alpha_power(int i);
	int log_alpha(int i);
	int square_root(int i, int &root);
	int primitive_root();
	int N2(int a);
	int N3(int a);
	int T2(int a);
	int T3(int a);
	int bar(int a);
	void abc2xy(int a, int b, int c, int &x, int &y, 
		int verbose_level);
		// given a, b, c, determine x and y such that 
		// c = a * x^2 + b * y^2
		// such elements x and y exist for any choice of a, b, c.
	int retract(finite_field &subfield, int index, int a, 
		int verbose_level);
	void retract_int_vec(finite_field &subfield, int index, 
		int *v_in, int *v_out, int len, int verbose_level);
	int embed(finite_field &subfield, int index, int b, 
		int verbose_level);
	void subfield_embedding_2dimensional(finite_field &subfield, 
		int *&components, int *&embedding, 
		int *&pair_embedding, int verbose_level);

	// #########################################################################
	// finite_field_linear_algebra.cpp
	// #########################################################################

	void copy_matrix(int *A, int *B, int ma, int na);
	void reverse_matrix(int *A, int *B, int ma, int na);
	void identity_matrix(int *A, int n);
	int is_identity_matrix(int *A, int n);
	int is_diagonal_matrix(int *A, int n);
	int is_scalar_multiple_of_identity_matrix(int *A, 
		int n, int &scalar);
	void diagonal_matrix(int *A, int n, int alpha);
	void matrix_minor(int f_semilinear, int *A, 
		int *B, int n, int f, int l);
		// initializes B as the l x l minor of A 
		// (which is n x n) starting from row f. 
	void mult_vector_from_the_left(int *v, int *A, 
		int *vA, int m, int n);
		// v[m], A[m][n], vA[n]
	void mult_vector_from_the_right(int *A, int *v, 
		int *Av, int m, int n);
		// A[m][n], v[n], Av[m]

	long int nb_calls_to_mult_matrix_matrix;
	void mult_matrix_matrix(int *A, int *B,
		int *C, int m, int n, int o, int verbose_level);
		// matrix multiplication C := A * B,
		// where A is m x n and B is n x o, so that C is m by o
	void semilinear_matrix_mult(int *A, int *B, int *AB, int n);
		// (A,f1) * (B,f2) = (A*B^{\varphi^{-f1}},f1+f2)
	void semilinear_matrix_mult_memory_given(int *A, int *B, 
		int *AB, int *tmp_B, int n);
		// (A,f1) * (B,f2) = (A*B^{\varphi^{-f1}},f1+f2)
	void matrix_mult_affine(int *A, int *B, int *AB, 
		int n, int verbose_level);
	void semilinear_matrix_mult_affine(int *A, int *B, int *AB, int n);
	int matrix_determinant(int *A, int n, int verbose_level);
	void matrix_inverse(int *A, int *Ainv, int n, int verbose_level);
	void matrix_invert(int *A, int *Tmp, 
		int *Tmp_basecols, int *Ainv, int n, int verbose_level);
		// Tmp points to n * n + 1 int's
		// Tmp_basecols points to n int's
	void semilinear_matrix_invert(int *A, int *Tmp, 
		int *Tmp_basecols, int *Ainv, int n, int verbose_level);
		// Tmp points to n * n + 1 int's
		// Tmp_basecols points to n int's
	void semilinear_matrix_invert_affine(int *A, int *Tmp, 
		int *Tmp_basecols, int *Ainv, int n, int verbose_level);
		// Tmp points to n * n + 1 int's
		// Tmp_basecols points to n int's
	void matrix_invert_affine(int *A, int *Tmp, int *Tmp_basecols, 
		int *Ainv, int n, int verbose_level);
		// Tmp points to n * n + 1 int's
		// Tmp_basecols points to n int's
	void projective_action_from_the_right(int f_semilinear, 
		int *v, int *A, int *vA, int n, int verbose_level);
		// vA = (v * A)^{p^f}  if f_semilinear (where f = A[n *  n]), 
		// vA = v * A otherwise
	void general_linear_action_from_the_right(int f_semilinear, 
		int *v, int *A, int *vA, int n, int verbose_level);
	void semilinear_action_from_the_right(int *v, 
		int *A, int *vA, int n);
		// vA = (v * A)^{p^f}  (where f = A[n *  n])
	void semilinear_action_from_the_left(int *A, 
		int *v, int *Av, int n);
		// Av = A * v^{p^f}
	void affine_action_from_the_right(int f_semilinear, 
		int *v, int *A, int *vA, int n);
		// vA = (v * A)^{p^f} + b
	void zero_vector(int *A, int m);
	void all_one_vector(int *A, int m);
	void support(int *A, int m, int *&support, int &size);
	void characteristic_vector(int *A, int m, int *set, int size);
	int is_zero_vector(int *A, int m);
	void add_vector(int *A, int *B, int *C, int m);
	void linear_combination_of_vectors(
			int a, int *A, int b, int *B, int *C, int len);
	void linear_combination_of_three_vectors(
			int a, int *A, int b, int *B, int c, int *C, int *D, int len);
	void negate_vector(int *A, int *B, int m);
	void negate_vector_in_place(int *A, int m);
	void scalar_multiply_vector_in_place(int c, int *A, int m);
	void vector_frobenius_power_in_place(int *A, int m, int f);
	int dot_product(int len, int *v, int *w);
	void transpose_matrix(int *A, int *At, int ma, int na);
	void transpose_matrix_in_place(int *A, int m);
	void invert_matrix(int *A, int *A_inv, int n);
	void invert_matrix_memory_given(int *A, int *A_inv, int n,
			int *tmp_A, int *tmp_basecols);
	void transform_form_matrix(int *A, int *Gram, 
		int *new_Gram, int d);
		// computes new_Gram = A * Gram * A^\top
	int rank_of_matrix(int *A, int m, int verbose_level);
	int rank_of_matrix_memory_given(int *A, 
		int m, int *B, int *base_cols, int verbose_level);
	int rank_of_rectangular_matrix(int *A, 
		int m, int n, int verbose_level);
	int rank_of_rectangular_matrix_memory_given(int *A, 
		int m, int n, int *B, int *base_cols, 
		int verbose_level);
	int rank_and_basecols(int *A, int m, 
		int *base_cols, int verbose_level);
	void Gauss_step(int *v1, int *v2, int len, 
		int idx, int verbose_level);
		// afterwards: v2[idx] = 0 
		// and v1,v2 span the same space as before
		// v1 is not changed if v1[idx] is nonzero
	void Gauss_step_make_pivot_one(int *v1, int *v2, 
		int len, int idx, int verbose_level);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
	void extend_basis(int m, int n, int *Basis, int verbose_level);
	int base_cols_and_embedding(int m, int n, int *A, 
		int *base_cols, int *embedding, int verbose_level);
		// returns the rank rk of the matrix.
		// It also computes base_cols[rk] and embedding[m - rk]
		// It leaves A unchanged
	int Gauss_easy(int *A, int m, int n);
		// returns the rank
	int Gauss_easy_memory_given(int *A, int m, int n, int *base_cols);
	int Gauss_simple(int *A, int m, int n, 
		int *base_cols, int verbose_level);
		// returns the rank which is the 
		// number of entries in base_cols
	int Gauss_int(int *A, int f_special, 
		int f_complete, int *base_cols, 
		int f_P, int *P, int m, int n, 
		int Pn, int verbose_level);
		// returns the rank which is the 
		// number of entries in base_cols
		// A is m x n,
		// P is m x Pn (provided f_P is TRUE)
	int Gauss_int_with_pivot_strategy(int *A, 
		int f_special, int f_complete, int *pivot_perm, 
		int m, int n, 
		int (*find_pivot_function)(int *A, int m, int n, int r, 
		int *pivot_perm, void *data),
		void *find_pivot_data,  
		int verbose_level);
		// returns the rank which is the number of entries in pivots
		// A is a m x n matrix
	void Gauss_int_with_given_pivots(int *A, 
		int f_special, int f_complete, int *pivots, int nb_pivots, 
		int m, int n, 
		int verbose_level);
		// A is a m x n matrix
	void kernel_columns(int n, int nb_base_cols, 
		int *base_cols, int *kernel_cols);
	void matrix_get_kernel_as_int_matrix(int *M, int m, int n, 
		int *base_cols, int nb_base_cols, 
		int_matrix *kernel);
	void matrix_get_kernel(int *M, int m, int n, 
		int *base_cols, int nb_base_cols, 
		int &kernel_m, int &kernel_n, int *kernel);
		// kernel must point to the appropriate amount of memory! 
		// (at least n * (n - nb_base_cols) int's)
	int perp(int n, int k, int *A, int *Gram);
	int RREF_and_kernel(int n, int k, int *A, int verbose_level);
	int perp_standard(int n, int k, int *A, int verbose_level);
	int perp_standard_with_temporary_data(int n, int k, int *A, 
		int *B, int *K, int *base_cols, 
		int verbose_level);
	int intersect_subspaces(int n, int k1, int *A, int k2, int *B, 
		int &k3, int *intersection, int verbose_level);
	int n_choose_k_mod_p(int n, int k, int verbose_level);
	void Dickson_polynomial(int *map, int *coeffs);
		// compute the coefficients of a degree q-1 polynomial 
		// which interpolates a given map
		// from F_q to F_q
	void projective_action_on_columns_from_the_left(int *A, 
		int *M, int m, int n, int *perm, int verbose_level);
	void builtin_transversal_rep_GLnq(int *A, int n, 
		int f_semilinear, int i, int j, int verbose_level);
	void affine_translation(int n, int coordinate_idx, 
		int field_base_idx, int *perm);
		// perm points to q^n int's
		// field_base_idx is the base element whose 
		// translation we compute, 0 \le field_base_idx < e
		// coordinate_idx is the coordinate in which we shift, 
		// 0 \le coordinate_idx < n
	void affine_multiplication(int n, 
		int multiplication_order, int *perm);
		// perm points to q^n int's
		// compute the diagonal multiplication by alpha, i.e. 
		// the multiplication by alpha of each component
	void affine_frobenius(int n, int k, int *perm);
		// perm points to q^n int's
		// compute the diagonal action of the Frobenius 
		// automorphism to the power k, i.e., 
		// raises each component to the p^k-th power
	int all_affine_translations_nb_gens(int n);
	void all_affine_translations(int n, int *gens);
	void affine_generators(int n, int f_translations, 
		int f_semilinear, int frobenius_power, 
		int f_multiplication, int multiplication_order, 
		int &nb_gens, int &degree, int *&gens, 
		int &base_len, int *&the_base);
	int evaluate_bilinear_form(int n, int *v1, int *v2, int *Gram);
	int evaluate_standard_hyperbolic_bilinear_form(int n, 
		int *v1, int *v2);
	int evaluate_quadratic_form(int n, int nb_terms, 
		int *i, int *j, int *coeff, int *x);
	void find_singular_vector_brute_force(int n, int form_nb_terms, 
		int *form_i, int *form_j, int *form_coeff, int *Gram, 
		int *vec, int verbose_level);
	void find_singular_vector(int n, int form_nb_terms, 
		int *form_i, int *form_j, int *form_coeff, int *Gram, 
		int *vec, int verbose_level);
	void complete_hyperbolic_pair(int n, int form_nb_terms, 
		int *form_i, int *form_j, int *form_coeff, int *Gram, 
		int *vec1, int *vec2, int verbose_level);
	void find_hyperbolic_pair(int n, int form_nb_terms, 
		int *form_i, int *form_j, int *form_coeff, int *Gram, 
		int *vec1, int *vec2, int verbose_level);
	void restrict_quadratic_form_list_coding(int k,
		int n, int *basis, 
		int form_nb_terms, 
		int *form_i, int *form_j, int *form_coeff, 
		int &restricted_form_nb_terms, 
		int *&restricted_form_i, int *&restricted_form_j, 
		int *&restricted_form_coeff, 
		int verbose_level);
	void restrict_quadratic_form(int k, int n, int *basis, 
		int *C, int *D, int verbose_level);
	int compare_subspaces_ranked(int *set1, int *set2, int size, 
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
	int Gauss_canonical_form_ranked(int *set1, int *set2, int size, 
		int vector_space_dimension, int verbose_level);
		// Computes the Gauss canonical form 
		// for the generating set in set1.
		// The result is written to set2.
		// Returns the rank of the span of the elements in set1.
	int lexleast_canonical_form_ranked(int *set1, int *set2, int size, 
		int vector_space_dimension, int verbose_level);
		// Computes the lexleast generating set the subspace 
		// spanned by the elements in set1.
		// The result is written to set2.
		// Returns the rank of the span of the elements in set1.
	void reduce_mod_subspace_and_get_coefficient_vector(
		int k, int len, int *basis, int *base_cols, 
		int *v, int *coefficients, int verbose_level);
	void reduce_mod_subspace(int k, 
		int len, int *basis, int *base_cols, 
		int *v, int verbose_level);
	int is_contained_in_subspace(int k, 
		int len, int *basis, int *base_cols, 
		int *v, int verbose_level);
	void compute_and_print_projective_weights(
			std::ostream &ost, int *M, int n, int k);
	int code_minimum_distance(int n, int k, 
		int *code, int verbose_level);
		// code[k * n]
	void codewords_affine(int n, int k, 
		int *code, // [k * n]
		int *codewords, // q^k
		int verbose_level);
	void code_projective_weight_enumerator(int n, int k, 
		int *code, // [k * n]
		int *weight_enumerator, // [n + 1]
		int verbose_level);
	void code_weight_enumerator(int n, int k, 
		int *code, // [k * n]
		int *weight_enumerator, // [n + 1]
		int verbose_level);
	void code_weight_enumerator_fast(int n, int k, 
		int *code, // [k * n]
		int *weight_enumerator, // [n + 1]
		int verbose_level);
	void code_projective_weights(int n, int k, 
		int *code, // [k * n]
		int *&weights,
			// will be allocated [N] 
			// where N = theta_{k-1}
		int verbose_level);
	int is_subspace(int d, int dim_U, int *Basis_U, int dim_V, 
		int *Basis_V, int verbose_level);
	void Kronecker_product(int *A, int *B, 
		int n, int *AB);
	void Kronecker_product_square_but_arbitrary(int *A, int *B, 
		int na, int nb, int *AB, int &N, int verbose_level);
	int dependency(int d, int *v, int *A, int m, int *rho, 
		int verbose_level);
		// Lueneburg~\cite{Lueneburg87a} p. 104.
		// A is a matrix of size d + 1 times d
		// v[d]
		// rho is a column permutation of degree d
	void order_ideal_generator(int d, int idx, int *mue, int &mue_deg, 
		int *A, int *Frobenius, 
		int verbose_level);
		// Lueneburg~\cite{Lueneburg87a} p. 105.
		// Frobenius is a matrix of size d x d
		// A is (d + 1) x d
		// mue[d + 1]
	void span_cyclic_module(int *A, int *v, int n, int *Mtx, 
		int verbose_level);
	void random_invertible_matrix(int *M, int k, int verbose_level);
	void make_all_irreducible_polynomials_of_degree_d(
		int d, int &nb, int *&Table, int verbose_level);
	int count_all_irreducible_polynomials_of_degree_d(
		int d, int verbose_level);
	void adjust_basis(int *V, int *U, int n, int k, int d, 
		int verbose_level);
	void choose_vector_in_here_but_not_in_here_column_spaces(
		int_matrix *V, int_matrix *W, int *v, int verbose_level);
	void choose_vector_in_here_but_not_in_here_or_here_column_spaces(
		int_matrix *V, int_matrix *W1, int_matrix *W2, int *v, 
		int verbose_level);
	int 
	choose_vector_in_here_but_not_in_here_or_here_column_spaces_coset(
		int &coset, 
		int_matrix *V, int_matrix *W1, int_matrix *W2, int *v, 
		int verbose_level);
	void vector_add_apply(int *v, int *w, int c, int n);
	void vector_add_apply_with_stride(int *v, int *w, int stride, 
		int c, int n);
	int test_if_commute(int *A, int *B, int k, int verbose_level);
	void unrank_point_in_PG(int *v, int len, int rk);
		// len is the length of the vector, 
		// not the projective dimension
	int rank_point_in_PG(int *v, int len);
	int nb_points_in_PG(int n);
		// n is projective dimension
	void Borel_decomposition(int n, int *M, int *B1, int *B2, 
		int *pivots, int verbose_level);
	void map_to_standard_frame(int d, int *A, 
		int *Transform, int verbose_level);
		// d = vector space dimension
		// maps d + 1 points to the frame 
		// e_1, e_2, ..., e_d, e_1+e_2+..+e_d 
		// A is (d + 1) x d
		// Transform is d x d
	void map_frame_to_frame_with_permutation(int d, int *A, 
		int *perm, int *B, int *Transform, int verbose_level);
	void map_points_to_points_projectively(int d, int k, 
		int *A, int *B, int *Transform, 
		int &nb_maps, int verbose_level);
		// A and B are (d + k + 1) x d
		// Transform is d x d
		// returns TRUE if a map exists
	int BallChowdhury_matrix_entry(int *Coord, int *C, 
		int *U, int k, int sz_U, 
		int *T, int verbose_level);
	void cubic_surface_family_24_generators(int f_with_normalizer, 
		int f_semilinear, 
		int *&gens, int &nb_gens, int &data_size, 
		int &group_order, int verbose_level);

	// #########################################################################
	// finite_field_representations.cpp
	// #########################################################################

	void representing_matrix8_R(int *A, 
		int q, int a, int b, int c, int d);
	void representing_matrix9_R(int *A, 
		int q, int a, int b, int c, int d);
	void representing_matrix9_U(int *A, 
		int a, int b, int c, int d, int beta);
	void representing_matrix8_U(int *A, 
		int a, int b, int c, int d, int beta);
	void representing_matrix8_V(int *A, int beta);
	void representing_matrix9b(int *A, int beta);
	void representing_matrix8a(int *A, 
		int a, int b, int c, int d, int beta);
	void representing_matrix8b(int *A, int beta);
	int Term1(int a1, int e1);
	int Term2(int a1, int a2, int e1, int e2);
	int Term3(int a1, int a2, int a3, int e1, int e2, int e3);
	int Term4(int a1, int a2, int a3, int a4, int e1, int e2, int e3, 
		int e4);
	int Term5(int a1, int a2, int a3, int a4, int a5, int e1, int e2, 
		int e3, int e4, int e5);
	int term1(int a1, int e1);
	int term2(int a1, int a2, int e1, int e2);
	int term3(int a1, int a2, int a3, int e1, int e2, int e3);
	int term4(int a1, int a2, int a3, int a4, int e1, int e2, int e3, 
		int e4);
	int term5(int a1, int a2, int a3, int a4, int a5, int e1, int e2, 
		int e3, int e4, int e5);
	int m_term(int q, int a1, int a2, int a3);
	int beta_trinomial(int q, int beta, int a1, int a2, int a3);
	int T3product2(int a1, int a2);

	// #########################################################################
	// finite_field_projective.cpp
	// #########################################################################

	void create_projective_variety(
			const char *variety_label,
			int variety_nb_vars, int variety_degree,
			const char *variety_coeffs,
			char *fname, int &nb_pts, int *&Pts,
			int verbose_level);
	void create_projective_curve(
			const char *variety_label,
			int curve_nb_vars, int curve_degree,
			const char *curve_coeffs,
			char *fname, int &nb_pts, int *&Pts,
			int verbose_level);
	void PG_element_normalize(
			int *v, int stride, int len);
	// last non-zero element made one
	void PG_element_normalize_from_front(
			int *v, int stride, int len);
	// first non zero element made one

	long int nb_calls_to_PG_element_rank_modified;
	long int nb_calls_to_PG_element_unrank_modified;

	void PG_elements_embed(
			int *set_in, int *set_out, int sz,
			int old_length, int new_length, int *v);
	int PG_element_embed(
			int rk, int old_length, int new_length, int *v);
	void PG_element_rank_modified(
			int *v, int stride, int len, int &a);
	void PG_element_unrank_fining(
			int *v, int len, int a);
	int PG_element_rank_fining(
			int *v, int len);
	void PG_element_unrank_gary_cook(
			int *v, int len, int a);
	void PG_element_unrank_modified(
			int *v, int stride, int len, int a);
	void PG_element_rank_modified_not_in_subspace(
			int *v, int stride, int len, int m, int &a);
	void PG_element_unrank_modified_not_in_subspace(
			int *v, int stride, int len, int m, int a);

	int evaluate_conic_form(int *six_coeffs, int *v3);
	int evaluate_quadric_form_in_PG_three(int *ten_coeffs, int *v4);
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
	int Pluecker_ij(int i, int j, int *x4, int *y4);
	int evaluate_symplectic_form(int len, int *x, int *y);
	int evaluate_quadratic_form_x0x3mx1x2(int *x);
	int is_totally_isotropic_wrt_symplectic_form(int k,
		int n, int *Basis);
	int evaluate_monomial(int *monomial, int *variables, int nb_vars);
	void projective_point_unrank(int n, int *v, int rk);
	int projective_point_rank(int n, int *v);
	void create_BLT_point(
			int *v5, int a, int b, int c, int verbose_level);
	// creates the point (-b/2,-c,a,-(b^2/4-ac),1)
	// check if it satisfies x_0^2 + x_1x_2 + x_3x_4:
	// b^2/4 + (-c)*a + -(b^2/4-ac)
	// = b^2/4 -ac -b^2/4 + ac = 0
	void Segre_hyperoval(
			int *&Pts, int &nb_pts, int verbose_level);
	void GlynnI_hyperoval(
			int *&Pts, int &nb_pts, int verbose_level);
	void GlynnII_hyperoval(
			int *&Pts, int &nb_pts, int verbose_level);
	void Subiaco_oval(
			int *&Pts, int &nb_pts, int f_short, int verbose_level);
		// following Payne, Penttila, Pinneri:
		// Isomorphisms Between Subiaco q-Clan Geometries,
		// Bull. Belg. Math. Soc. 2 (1995) 197-222.
		// formula (53)
	void Subiaco_hyperoval(
			int *&Pts, int &nb_pts, int verbose_level);
	int OKeefe_Penttila_32(int t);
	int Subiaco64_1(int t);
	int Subiaco64_2(int t);
	int Adelaide64(int t);
	void LunelliSce(int *pts18, int verbose_level);
	int LunelliSce_evaluate_cubic1(int *v);
		// computes X^3 + Y^3 + Z^3 + \eta^3 XYZ
	int LunelliSce_evaluate_cubic2(int *v);
		// computes X^3 + Y^3 + Z^3 + \eta^{12} XYZ
	void O4_isomorphism_4to2(
		int *At, int *As, int &f_switch, int *B,
		int verbose_level);
	void O4_isomorphism_2to4(
		int *At, int *As, int f_switch, int *B);
	void O4_grid_coordinates_rank(
		int x1, int x2, int x3, int x4,
		int &grid_x, int &grid_y, int verbose_level);
	void O4_grid_coordinates_unrank(
		int &x1, int &x2, int &x3, int &x4, int grid_x,
		int grid_y, int verbose_level);
	void O4_find_tangent_plane(
		int pt_x1, int pt_x2, int pt_x3, int pt_x4,
		int *tangent_plane, int verbose_level);
	void do_cone_over(int n,
		int *set_in, int set_size_in, int *&set_out, int &set_size_out,
		int verbose_level);
	void do_blocking_set_family_3(int n,
		int *set_in, int set_size,
		int *&the_set_out, int &set_size_out,
		int verbose_level);
	void create_hyperoval(
		int f_translation, int translation_exponent,
		int f_Segre, int f_Payne, int f_Cherowitzo, int f_OKeefe_Penttila,
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
	void create_subiaco_oval(
		int f_short,
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
	void create_subiaco_hyperoval(
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
	void create_ovoid(
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
	void create_Baer_substructure(int n,
		finite_field *Fq,
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
	// the big field FQ is given
	void create_BLT_from_database(int f_embedded,
		int BLT_k,
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
	void create_orthogonal(int epsilon, int n,
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
	void create_hermitian(int n,
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
	void create_cubic(
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
	void create_twisted_cubic(
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
	void create_elliptic_curve(
		int elliptic_curve_b, int elliptic_curve_c,
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
	void create_ttp_code(finite_field *Fq,
		int f_construction_A, int f_hyperoval, int f_construction_B,
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
	// this is FQ
	void create_unital_XXq_YZq_ZYq(
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
	void create_whole_space(int n,
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
	void create_hyperplane(int n,
		int pt,
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
	void create_segre_variety(int a, int b,
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
	void create_Maruta_Hamada_arc(
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
	void create_desarguesian_line_spread_in_PG_3_q(
		finite_field *Fq,
		int f_embedded_in_PG_4_q,
		char *fname, int &nb_lines, int *&Lines,
		int verbose_level);
	// this is FQ
	void do_Klein_correspondence(int n,
		int *set_in, int set_size,
		int *&the_set_out, int &set_size_out,
		int verbose_level);
	void do_m_subspace_type(int n, int m,
		int *set, int set_size,
		int f_show, int verbose_level);
	void do_m_subspace_type_fast(int n, int m,
		int *set, int set_size,
		int f_show, int verbose_level);
	void do_line_type(int n,
		int *set, int set_size,
		int f_show, int verbose_level);
	void do_plane_type(int n,
		int *set, int set_size,
		int *&intersection_type, int &highest_intersection_number,
		int verbose_level);
	void do_plane_type_failsafe(int n,
		int *set, int set_size,
		int verbose_level);
	void do_conic_type(int n,
		int f_randomized, int nb_times,
		int *set, int set_size,
		int *&intersection_type, int &highest_intersection_number,
		int verbose_level);
	void do_test_diagonal_line(int n,
		int *set_in, int set_size,
		char *fname_orbits_on_quadrangles,
		int verbose_level);
	void do_andre(finite_field *Fq,
		int *the_set_in, int set_size_in,
		int *&the_set_out, int &set_size_out,
		int verbose_level);
	// this is FQ
	void do_print_lines_in_PG(int n,
		int *set_in, int set_size);
	void do_print_points_in_PG(int n,
		int *set_in, int set_size);
	void do_print_points_in_orthogonal_space(
		int epsilon, int n,
		int *set_in, int set_size, int verbose_level);
	void do_print_points_on_grassmannian(
		int n, int k,
		int *set_in, int set_size);
	void do_embed_orthogonal(
		int epsilon, int n,
		int *set_in, int *&set_out, int set_size,
		int verbose_level);
	void do_embed_points(int n,
		int *set_in, int *&set_out, int set_size,
		int verbose_level);
	void do_draw_points_in_plane(
		int *set, int set_size,
		const char *fname_base, int f_point_labels,
		int f_embedded, int f_sideways,
		int verbose_level);
	void do_ideal(int n,
		int *set_in, int set_size, int degree,
		int verbose_level);


	// #########################################################################
	// finite_field_io.cpp
	// #########################################################################

	void cheat_sheet_PG(int n,
			int f_surface, int verbose_level);
	void cheat_sheet_tables(std::ostream &f, int verbose_level);
	void print_minimum_polynomial(int p, const char *polynomial);
	void print();
	void print_detailed(int f_add_mult_table);
	void print_add_mult_tables();
	void print_tables();
	void print_tables_extension_field(const char *poly);
	void display_T2(std::ostream &ost);
	void display_T3(std::ostream &ost);
	void display_N2(std::ostream &ost);
	void display_N3(std::ostream &ost);
	void print_integer_matrix_zech(std::ostream &ost,
		int *p, int m, int n);
	void print_embedding(finite_field &subfield,
		int *components, int *embedding, int *pair_embedding);
		// we think of F as two dimensional vector space
		// over f with basis (1,alpha)
		// for i,j \in f, with x = i + j * alpha \in F, we have
		// pair_embedding[i * q + j] = x;
		// also,
		// components[x * 2 + 0] = i;
		// components[x * 2 + 1] = j;
		// also, for i \in f, embedding[i] is the element
		// in F that corresponds to i
		// components[Q * 2]
		// embedding[q]
		// pair_embedding[q * q]
	void print_embedding_tex(finite_field &subfield,
		int *components, int *embedding, int *pair_embedding);
	void print_indicator_square_nonsquare(int a);
	void print_element(std::ostream &ost, int a);
	void print_element_str(std::stringstream &ost, int a);
	void print_element_with_symbol(std::ostream &ost,
		int a, int f_exponential, int width, const char *symbol);
	void print_element_with_symbol_str(std::stringstream &ost,
			int a, int f_exponential, int width, const char *symbol);
	void int_vec_print(std::ostream &ost, int *v, int len);
	void int_vec_print_elements_exponential(std::ostream &ost,
		int *v, int len, const char *symbol_for_print);
	void latex_addition_table(std::ostream &f,
		int f_elements_exponential, const char *symbol_for_print);
	void latex_multiplication_table(std::ostream &f,
		int f_elements_exponential, const char *symbol_for_print);
	void latex_matrix(std::ostream &f, int f_elements_exponential,
		const char *symbol_for_print, int *M, int m, int n);
	void power_table(int t, int *power_table, int len);
	void cheat_sheet(std::ostream &f, int verbose_level);
	void cheat_sheet_top(std::ostream &f, int nb_cols);
	void cheat_sheet_bottom(std::ostream &f);
	void display_table_of_projective_points(
			std::ostream &ost, int *Pts, int nb_pts, int len);
	void export_magma(int d, int *Pts, int nb_pts, char *fname);
	void export_gap(int d, int *Pts, int nb_pts, char *fname);
	void oval_polynomial(
		int *S, unipoly_domain &D, unipoly_object &poly,
		int verbose_level);
	void all_PG_elements_in_subspace(
			int *genma, int k, int n, int *&point_list, int &nb_points,
			int verbose_level);
	void all_PG_elements_in_subspace_array_is_given(
			int *genma, int k, int n, int *point_list, int &nb_points,
			int verbose_level);
	void display_all_PG_elements(int n);
	void display_all_PG_elements_not_in_subspace(int n, int m);
	void display_all_AG_elements(int n);
};

extern int nb_calls_to_finite_field_init;

// #############################################################################
// finite_field_projective.cpp
// #############################################################################

int nb_PG_elements(int n, int q);
	// $\frac{q^{n+1} - 1}{q-1} = \sum_{i=0}^{n} q^i $
int nb_PG_elements_not_in_subspace(int n, int m, int q);
int nb_AG_elements(int n, int q);
void PG_element_apply_frobenius(int n,
	finite_field &GFq, int *v, int f);
void AG_element_rank(int q, int *v, int stride, int len, int &a);
void AG_element_unrank(int q, int *v, int stride, int len, int a);
void AG_element_rank_longinteger(int q, int *v, int stride, int len,
	longinteger_object &a);
void AG_element_unrank_longinteger(int q, int *v, int stride, int len,
	longinteger_object &a);
int PG_element_modified_is_in_subspace(int n, int m, int *v);
void PG_element_modified_not_in_subspace_perm(int n, int m,
	finite_field &GFq, int *orbit, int *orbit_inv, int verbose_level);
void test_PG(int n, int q);
void line_through_two_points(finite_field &GFq, int len,
	int pt1, int pt2, int *line);
void print_set_in_affine_plane(finite_field &GFq, int len, int *S);
int consecutive_ones_property_in_affine_plane(std::ostream &ost,
	finite_field &GFq, int len, int *S);
int line_intersection_with_oval(finite_field &GFq,
	int *f_oval_point, int line_rk,
	int verbose_level);
int get_base_line(finite_field &GFq, int plane1, int plane2,
	int verbose_level);
void create_Fisher_BLT_set(int *Fisher_BLT, int q,
	const char *poly_q, const char *poly_Q, int verbose_level);
void create_Linear_BLT_set(int *BLT, int q,
	const char *poly_q, const char *poly_Q, int verbose_level);
void create_Mondello_BLT_set(int *BLT, int q,
	const char *poly_q, const char *poly_Q, int verbose_level);
void print_quadratic_form_list_coded(int form_nb_terms,
	int *form_i, int *form_j, int *form_coeff);
void make_Gram_matrix_from_list_coded_quadratic_form(
	int n, finite_field &F,
	int nb_terms, int *form_i, int *form_j,
	int *form_coeff, int *Gram);
void add_term(int n, finite_field &F, int &nb_terms,
	int *form_i, int *form_j, int *form_coeff, int *Gram,
	int i, int j, int coeff);



// #############################################################################
// finite_field_tables.cpp
// #############################################################################

extern int finitefield_primes[];
extern int finitefield_nb_primes;
extern int finitefield_largest_degree_irreducible_polynomial[];
extern const char *finitefield_primitive_polynomial[][100];
const char *get_primitive_polynomial(int p, int e, int verbose_level);

// #############################################################################
// finite_ring.cpp
// #############################################################################


//! finite chain rings



class finite_ring {

	int *add_table; // [q * q]
	int *mult_table; // [q * q]

	int *f_is_unit_table;
	int *negate_table;
	int *inv_table;

public:
	int q;
	int p;
	int e;

	finite_field *Fp;


	finite_ring();
	~finite_ring();
	void null();
	void freeself();
	void init(int q, int verbose_level);
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

int PHG_element_normalize(finite_ring &R, int *v, int stride, int len);
// last unit element made one
int PHG_element_normalize_from_front(finite_ring &R, int *v,
	int stride, int len);
// first non unit element made one
int PHG_element_rank(finite_ring &R, int *v, int stride, int len);
void PHG_element_unrank(finite_ring &R, int *v, int stride, int len, int rk);
int nb_PHG_elements(int n, finite_ring &R);
void display_all_PHG_elements(int n, int q);


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
	int nb_irred;
	int *Nb_irred;
	int *First_irred;
	int *Nb_part;
	int **Tables;
	int **Partitions;
	int *Degree;

	gl_classes();
	~gl_classes();
	void null();
	void freeself();
	void init(int k, finite_field *F, int verbose_level);
	void print_polynomials(std::ostream &ost);
	int select_polynomial_first(int *Select, int verbose_level);
	int select_polynomial_next(int *Select, int verbose_level);
	int select_partition_first(int *Select, int *Select_partition, 
		int verbose_level);
	int select_partition_next(int *Select, int *Select_partition, 
		int verbose_level);
	int first(int *Select, int *Select_partition, int verbose_level);
	int next(int *Select, int *Select_partition, int verbose_level);
	void print_matrix_and_centralizer_order_latex(std::ostream &ost,
		gl_class_rep *R);
	void make_matrix_from_class_rep(int *Mtx, gl_class_rep *R, 
		int verbose_level);
	void make_matrix(int *Mtx, int *Select, int *Select_Partition, 
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
	void compute_data_on_blocks(int *Mtx, int *Irreds, int nb_irreds, 
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
	void factor_polynomial(unipoly_object &char_poly, int *Mult, 
		int verbose_level);
	int find_class_rep(gl_class_rep *Reps, int nb_reps, 
		gl_class_rep *R, int verbose_level);
	void report(const char *fname, int verbose_level);

};

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
// group_generators.cpp
// #############################################################################


void diagonal_orbit_perm(int n, finite_field &GFq, 
	int *orbit, int *orbit_inv, int verbose_level);
void frobenius_orbit_perm(int n, finite_field &GFq, 
	int *orbit, int *orbit_inv, int verbose_level);
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
void projective_matrix_group_base_and_orbits(int n, 
	finite_field *F, int f_semilinear, 
	int base_len, int degree, 
	int *base, int *transversal_length, 
	int **orbit, int **orbit_inv, 
	int verbose_level);
void projective_matrix_group_base_and_transversal_length(int n,
	finite_field *F, int f_semilinear,
	int base_len, int degree,
	int *base, int *transversal_length,
	int verbose_level);
void affine_matrix_group_base_and_transversal_length(int n, 
	finite_field *F, int f_semilinear, 
	int base_len, int degree, 
	int *base, int *transversal_length, 
	int verbose_level);
void general_linear_matrix_group_base_and_transversal_length(int n, 
	finite_field *F, int f_semilinear, 
	int base_len, int degree, 
	int *base, int *transversal_length, 
	int verbose_level);
void strong_generators_for_projective_linear_group(int n, finite_field *F, 
	int f_semilinear, 
	int *&data, int &size, int &nb_gens, 
	int verbose_level);
void strong_generators_for_affine_linear_group(int n, finite_field *F, 
	int f_semilinear, 
	int *&data, int &size, int &nb_gens, 
	int verbose_level);
void strong_generators_for_general_linear_group(int n, finite_field *F, 
	int f_semilinear, 
	int *&data, int &size, int &nb_gens, 
	int verbose_level);
void generators_for_parabolic_subgroup(int n, finite_field *F, 
	int f_semilinear, int k, 
	int *&data, int &size, int &nb_gens, 
	int verbose_level);
void generators_for_stabilizer_of_three_collinear_points_in_PGL4(
	finite_field *F, 
	int f_semilinear, 
	int *&data, int &size, int &nb_gens, 
	int verbose_level);
void generators_for_stabilizer_of_triangle_in_PGL4(finite_field *F, 
	int f_semilinear, 
	int *&data, int &size, int &nb_gens, 
	int verbose_level);


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
	void unrank_element(int *Elt, int rk);
	int rank_element(int *Elt);
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

//! homogeneous polynomials in n variables over a finite field


class homogeneous_polynomial_domain {

public:
	int q;
	int n; // number of variables
	int degree;
	finite_field *F;
	int nb_monomials;
	int *Monomials; // [nb_monomials * n]
	char **symbols;
	char **symbols_latex;
	int *Variables; // [nb_monomials * degree]
		// Variables contains the monomials written out 
		// as a sequence of length degree 
		// with entries in 0,..,n-1.
		// the entries are listed in increasing order.
		// For instance, the monomial x_0^2x_1x_3 
		// is recorded as 0,0,1,3
	int nb_affine; // n^degree
	int *Affine; // [nb_affine * degree]
		// the affine elements are used for foiling 
		// when doing a linear substitution
	int *v; // [n]
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

	homogeneous_polynomial_domain();
	~homogeneous_polynomial_domain();
	void freeself();
	void null();
	void init(finite_field *F, int nb_vars, int degree, 
		int f_init_incidence_structure, int verbose_level);
	void make_monomials(int verbose_level);
	void rearrange_monomials_by_partition_type(int verbose_level);
	int index_of_monomial(int *v);
	void print_monomial(std::ostream &ost, int i);
	void print_monomial(std::ostream &ost, int *mon);
	void print_monomial(char *str, int i);
	void print_monomial_str(std::stringstream &ost, int i);
	void print_equation(std::ostream &ost, int *coeffs);
	void print_equation_str(std::stringstream &ost, int *coeffs);
	void print_equation_with_line_breaks_tex(std::ostream &ost,
		int *coeffs, int nb_terms_per_line, 
		const char *new_line_text);
	void algebraic_set(int *Eqns, int nb_eqns,
			int *Pts, int &nb_pts, int verbose_level);
	void enumerate_points(int *coeff, int *Pts, int &nb_pts, 
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
	int is_zero(int *coeff);
	void unrank_point(int *v, int rk);
	int rank_point(int *v);
	void unrank_coeff_vector(int *v, int rk);
	int rank_coeff_vector(int *v);
	int test_weierstrass_form(int rk, 
		int &a1, int &a2, int &a3, int &a4, int &a6, 
		int verbose_level);
	void vanishing_ideal(int *Pts, int nb_pts, int &r, int *Kernel, 
		int verbose_level);
	int compare_monomials(int *M1, int *M2);
	void print_monomial_ordering(std::ostream &ost);


};

int homogeneous_polynomial_domain_compare_monomial_with(void *data, 
	int i, int *data2, void *extra_data);
int homogeneous_polynomial_domain_compare_monomial(void *data, 
	int i, int j, void *extra_data);
void homogeneous_polynomial_domain_swap_monomial(void *data, 
	int i, int j, void *extra_data);



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
	void subtract_signless(longinteger_object &a, 
		longinteger_object &b, longinteger_object &c);
		// c = a - b, assuming a > b
	void subtract_signless_in_place(longinteger_object &a, 
		longinteger_object &b);
		// a := a - b, assuming a > b
	void add(longinteger_object &a, 
		longinteger_object &b, longinteger_object &c);
	void add_in_place(longinteger_object &a, longinteger_object &b);
		// a := a + b
	void mult(longinteger_object &a, 
		longinteger_object &b, longinteger_object &c);
	void mult_in_place(longinteger_object &a, longinteger_object &b);
	void mult_integer_in_place(longinteger_object &a, int b);
	void mult_mod(longinteger_object &a, 
		longinteger_object &b, longinteger_object &c, 
		longinteger_object &m, int verbose_level);
	void multiply_up(longinteger_object &a, int *x, int len);
	int quotient_as_int(longinteger_object &a, longinteger_object &b);
	void integral_division_exact(longinteger_object &a, 
		longinteger_object &b, longinteger_object &a_over_b);
	void integral_division(
		longinteger_object &a, longinteger_object &b, 
		longinteger_object &q, longinteger_object &r, 
		int verbose_level);
	void integral_division_by_int(longinteger_object &a, 
		int b, longinteger_object &q, int &r);
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
	void create_qnm1(longinteger_object &a, int q, int n);
	void binomial(longinteger_object &a, int n, int k, 
		int verbose_level);
	void size_of_conjugacy_class_in_sym_n(longinteger_object &a, 
		int n, int *part);
	void q_binomial(longinteger_object &a, 
		int n, int k, int q, int verbose_level);
	void q_binomial_no_table(longinteger_object &a, 
		int n, int k, int q, int verbose_level);
	void krawtchouk(longinteger_object &a, int n, int q, int k, int x);
	int is_even(longinteger_object &a);
	int is_odd(longinteger_object &a);
	int remainder_mod_int(longinteger_object &a, int p);
	int multiplicity_of_p(longinteger_object &a, 
		longinteger_object &residue, int p);
	int smallest_primedivisor(longinteger_object &a, int p_min, 
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
	void find_probable_prime_above(
		longinteger_object &a, 
		int nb_solovay_strassen_tests, int f_miller_rabin_test, 
		int verbose_level);
	int solovay_strassen_is_prime(
		longinteger_object &n, int nb_tests, int verbose_level);
	int solovay_strassen_is_prime_single_test(
		longinteger_object &n, int verbose_level);
	int solovay_strassen_test(
		longinteger_object &n, longinteger_object &a, 
		int verbose_level);
	int miller_rabin_test(
		longinteger_object &n, int verbose_level);
	void get_k_bit_random_pseudoprime(
		longinteger_object &n, int k, 
		int nb_tests_solovay_strassen, 
		int f_miller_rabin_test, int verbose_level);
	void RSA_setup(longinteger_object &n, 
		longinteger_object &p, longinteger_object &q, 
		longinteger_object &a, longinteger_object &b, 
		int nb_bits, 
		int nb_tests_solovay_strassen, int f_miller_rabin_test, 
		int verbose_level);
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
	int singleton_bound_for_d(int n, int k, int q, int verbose_level);
	int hamming_bound_for_d(int n, int k, int q, int verbose_level);
	int plotkin_bound_for_d(int n, int k, int q, int verbose_level);
	int griesmer_bound_for_d(int n, int k, int q, int verbose_level);
	int griesmer_bound_for_n(int k, int d, int q, int verbose_level);
};

void test_longinteger();
void test_longinteger2();
void test_longinteger3();
void test_longinteger4();
void test_longinteger5();
void test_longinteger6();
void test_longinteger7();
void test_longinteger8();
void mac_williams_equations(longinteger_object *&M, int n, int k, int q);
void determine_weight_enumerator();
void longinteger_collect_setup(int &nb_agos, 
	longinteger_object *&agos, int *&multiplicities);
void longinteger_collect_free(int &nb_agos, 
	longinteger_object *&agos, int *&multiplicities);
void longinteger_collect_add(int &nb_agos, 
	longinteger_object *&agos, int *&multiplicities, 
	longinteger_object &ago);
void longinteger_collect_print(std::ostream &ost, int &nb_agos,
	longinteger_object *&agos, int *&multiplicities);
void longinteger_free_global_data();
void longinteger_print_digits(char *rep, int len);
void longinteger_domain_free_tab_q_binomials();



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

//! all null polarities

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
// number_theory.cpp:
// #############################################################################


int power_mod(int a, int n, int p);
int inverse_mod(int a, int p);
int mult_mod(int a, int b, int p);
int add_mod(int a, int b, int p);
int int_abs(int a);
int irem(int a, int m);
int gcd_int(int m, int n);
void extended_gcd_int(int m, int n, int &g, int &u, int &v);
int i_power_j_safe(int i, int j);
int i_power_j(int i, int j);
int order_mod_p(int a, int p);
int int_log2(int n);
int int_log10(int n);
int int_logq(int n, int q);
// returns the number of digits in base q representation
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
void factor_prime_power(int q, int &p, int &e);
int primitive_root(int p, int verbose_level);
int Legendre(int a, int p, int verbose_level);
int Jacobi(int a, int m, int verbose_level);
int Jacobi_with_key_in_latex(std::ostream &ost, int a, int m, int verbose_level);
int gcd_with_key_in_latex(std::ostream &ost,
		int a, int b, int f_key, int verbose_level);
int ny2(int x, int &x1);
int ny_p(int n, int p);
int sqrt_mod_simple(int a, int p);
void print_factorization(int nb_primes, int *primes, int *exponents);
void print_longfactorization(int nb_primes, 
	longinteger_object *primes, int *exponents);
int euler_function(int n);
void int_add_fractions(int at, int ab, int bt, int bb, 
	int &ct, int &cb, int verbose_level);
void int_mult_fractions(int at, int ab, int bt, int bb, 
	int &ct, int &cb, int verbose_level);

// #############################################################################
// partial_derivative.cpp
// #############################################################################

//! partial derivative connects two homogeneous polynomial domains


class partial_derivative {

public:
	homogeneous_polynomial_domain *H;
	homogeneous_polynomial_domain *Hd;
	int variable_idx;
	int *mapping; // [H->nb_monomials * H->nb_monomials]


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


//! checking whether any d-1 columns are linearly independent


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
	int check_rank(int len, int *S, int verbose_level);
	int check_rank_matrix_input(int len, int *S, int dim_S, 
		int verbose_level);
	int check_rank_last_two_are_fixed(int len, int *S, int verbose_level);
	int compute_rank(int len, int *S, int f_projective, int verbose_level);
	int compute_rank_row_vectors(int len, int *S, int f_projective, 
		int verbose_level);
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
	int evaluate_over_FQ(int *v);
	int evaluate_over_Fq(int *v);
	void lift_matrix(int *MQ, int m, int *Mq, int verbose_level);
	void retract_matrix(int *Mq, int n, int *MQ, int m, 
		int verbose_level);
	void Adelaide_hyperoval(
			int *&Pts, int &nb_pts, int verbose_level);
	void create_adelaide_hyperoval(
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);

};

// #############################################################################
// unipoly_domain.cpp:
// #############################################################################

//! domain of polynomials in one variable over a finite field

class unipoly_domain {
public:
	finite_field *gfq;
	int f_factorring;
	int factor_degree;
	int *factor_coeffs;
	unipoly_object factor_poly;

	unipoly_domain(finite_field *GFq);
	unipoly_domain(finite_field *GFq, unipoly_object m);
	~unipoly_domain();
	int &s_i(unipoly_object p, int i)
		{ int *rep = (int *) p; return rep[i + 1]; };
	void create_object_of_degree(unipoly_object &p, int d);
	void create_object_of_degree_with_coefficients(unipoly_object &p, 
		int d, int *coeff);
	void create_object_by_rank(unipoly_object &p, int rk);
	void create_object_by_rank_longinteger(unipoly_object &p, 
		longinteger_object &rank, int verbose_level);
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
	void assign(unipoly_object a, unipoly_object &b);
	void one(unipoly_object p);
	void m_one(unipoly_object p);
	void zero(unipoly_object p);
	int is_one(unipoly_object p);
	int is_zero(unipoly_object p);
	void negate(unipoly_object a);
	void make_monic(unipoly_object &a);
	void add(unipoly_object a, unipoly_object b, unipoly_object &c);
	void mult(unipoly_object a, unipoly_object b, unipoly_object &c);
	void mult_easy(unipoly_object a, unipoly_object b, unipoly_object &c);
	void mult_mod(unipoly_object a, unipoly_object b, unipoly_object &c, 
		int factor_polynomial_degree, 
		int *factor_polynomial_coefficents_negated, 
		int verbose_level);
	void Frobenius_matrix_by_rows(int *&Frob,
		unipoly_object factor_polynomial, int verbose_level);
		// the j-th row of Frob is x^{j*q} mod m
	void Frobenius_matrix(int *&Frob, unipoly_object factor_polynomial, 
		int verbose_level);
	void Berlekamp_matrix(int *&B, unipoly_object factor_polynomial, 
		int verbose_level);
	void integral_division_exact(unipoly_object a, 
		unipoly_object b, unipoly_object &q, int verbose_level);
	void integral_division(unipoly_object a, unipoly_object b, 
		unipoly_object &q, unipoly_object &r, int verbose_level);
	void derive(unipoly_object a, unipoly_object &b);
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
	void power_longinteger(unipoly_object &a, longinteger_object &n);
	void power_coefficients(unipoly_object &a, int n);
	void minimum_polynomial(unipoly_object &a, 
		int alpha, int p, int verbose_level);
	int minimum_polynomial_factorring(int alpha, int p, 
		int verbose_level);
	void minimum_polynomial_factorring_longinteger(
		longinteger_object &alpha, 
		longinteger_object &rk_minpoly, 
		int p, int verbose_level);
	void BCH_generator_polynomial(unipoly_object &g, int n, 
		int designed_distance, int &bose_distance, 
		int &transversal_length, int *&transversal, 
		longinteger_object *&rank_of_irreducibles, 
		int verbose_level);
	void compute_generator_matrix(unipoly_object a, int *&genma, 
		int n, int &k, int verbose_level);
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

};


// #############################################################################
// vector_space.cpp:
// #############################################################################


//! finite dimensional vector space over a finite field


class vector_space {
public:

	int dimension;
	finite_field *F;

	int (*rank_point_func)(int *v, void *data);
	void (*unrank_point_func)(int *v, int rk, void *data);
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
		int (*rank_point_func)(int *v, void *data),
		void (*unrank_point_func)(int *v, int rk, void *data),
		void *data,
		int verbose_level);
	void unrank_basis(int *Mtx, int *set, int len);
	void rank_basis(int *Mtx, int *set, int len);
	void unrank_point(int *v, int rk);
	int rank_point(int *v);
	int RREF_and_rank(int *basis, int k);
	int is_contained_in_subspace(int *v, int *basis, int k);
	int compare_subspaces_ranked(
			int *set1, int *set2, int k, int verbose_level);
		// equality test for subspaces given by ranks of basis elements
};


void vector_space_unrank_point_callback(int *v, int rk, void *data);
int vector_space_rank_point_callback(int *v, void *data);

}}


