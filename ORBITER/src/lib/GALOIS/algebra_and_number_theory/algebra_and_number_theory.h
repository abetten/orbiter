// algebra_and_number_theory.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005


// #############################################################################
// a_domain.C:
// #############################################################################

enum domain_kind {
	not_applicable, domain_the_integers, domain_integer_fractions
};

class a_domain {
public:
	domain_kind kind;
	INT size_of_instance_in_INT;
	
	a_domain();
	~a_domain();
	void null();
	void freeself();
	
	void init_integers(INT verbose_level);
	void init_integer_fractions(INT verbose_level);
	INT as_INT(INT *elt, INT verbose_level);
	void make_integer(INT *elt, INT n, INT verbose_level);
	void make_zero(INT *elt, INT verbose_level);
	void make_zero_vector(INT *elt, INT len, INT verbose_level);
	INT is_zero_vector(INT *elt, INT len, INT verbose_level);
	INT is_zero(INT *elt, INT verbose_level);
	void make_one(INT *elt, INT verbose_level);
	INT is_one(INT *elt, INT verbose_level);
	void copy(INT *elt_from, INT *elt_to, INT verbose_level);
	void copy_vector(INT *elt_from, INT *elt_to, 
		INT len, INT verbose_level);
	void swap_vector(INT *elt1, INT *elt2, INT n, INT verbose_level);
	void swap(INT *elt1, INT *elt2, INT verbose_level);
	void add(INT *elt_a, INT *elt_b, INT *elt_c, INT verbose_level);
	void add_apply(INT *elt_a, INT *elt_b, INT verbose_level);
	void subtract(INT *elt_a, INT *elt_b, INT *elt_c, INT verbose_level);
	void negate(INT *elt, INT verbose_level);
	void negate_vector(INT *elt, INT len, INT verbose_level);
	void mult(INT *elt_a, INT *elt_b, INT *elt_c, INT verbose_level);
	void mult_apply(INT *elt_a, INT *elt_b, INT verbose_level);
	void power(INT *elt_a, INT *elt_b, INT n, INT verbose_level);
	void divide(INT *elt_a, INT *elt_b, INT *elt_c, INT verbose_level);
	void inverse(INT *elt_a, INT *elt_b, INT verbose_level);
	void print(INT *elt);
	void print_vector(INT *elt, INT n);
	void print_matrix(INT *A, INT m, INT n);
	void print_matrix_for_maple(INT *A, INT m, INT n);
	void make_element_from_integer(INT *elt, INT n, INT verbose_level);
	void mult_by_integer(INT *elt, INT n, INT verbose_level);
	void divide_by_integer(INT *elt, INT n, INT verbose_level);
	INT *offset(INT *A, INT i);
	INT Gauss_echelon_form(INT *A, INT f_special, INT f_complete, 
		INT *base_cols, 
		INT f_P, INT *P, INT m, INT n, INT Pn, INT verbose_level);
		// returns the rank which is the number 
		// of entries in base_cols
		// A is a m x n matrix,
		// P is a m x Pn matrix (if f_P is TRUE)
	void Gauss_step(INT *v1, INT *v2, INT len, INT idx, INT verbose_level);
		// afterwards: v2[idx] = 0 and v1,v2 span the same space as before
		// v1 is not changed if v1[idx] is nonzero
	void matrix_get_kernel(INT *M, INT m, INT n, INT *base_cols, 
		INT nb_base_cols, 
		INT &kernel_m, INT &kernel_n, INT *kernel, INT verbose_level);
		// kernel must point to the appropriate amount of memory! 
		// (at least n * (n - nb_base_cols) INT's)
		// kernel is stored as column vectors, 
		// i.e. kernel_m = n and kernel_n = n - nb_base_cols.
	void matrix_get_kernel_as_row_vectors(INT *M, INT m, INT n, 
		INT *base_cols, INT nb_base_cols, 
		INT &kernel_m, INT &kernel_n, INT *kernel, INT verbose_level);
		// kernel must point to the appropriate amount of memory! 
		// (at least n * (n - nb_base_cols) INT's)
		// kernel is stored as row vectors, 
		// i.e. kernel_m = n - nb_base_cols and kernel_n = n.
	void get_image_and_kernel(INT *M, INT n, INT &rk, INT verbose_level);
	void complete_basis(INT *M, INT m, INT n, INT verbose_level);
	void mult_matrix(INT *A, INT *B, INT *C, INT ma, INT na, INT nb, 
		INT verbose_level);
	void mult_matrix3(INT *A, INT *B, INT *C, INT *D, INT n, 
		INT verbose_level);
	void add_apply_matrix(INT *A, INT *B, INT m, INT n, 
		INT verbose_level);
	void matrix_mult_apply_scalar(INT *A, INT *s, INT m, INT n, 
		INT verbose_level);
	void make_block_matrix_2x2(INT *Mtx, INT n, INT k, 
		INT *A, INT *B, INT *C, INT *D, INT verbose_level);
		// A is k x k, 
		// B is k x (n - k), 
		// C is (n - k) x k, 
		// D is (n - k) x (n - k), 
		// Mtx is n x n
	void make_identity_matrix(INT *A, INT n, INT verbose_level);
	void matrix_inverse(INT *A, INT *Ainv, INT n, INT verbose_level);
	void matrix_invert(INT *A, INT *T, INT *basecols, INT *Ainv, INT n, 
		INT verbose_level);

};


// #############################################################################
// finite_field.C:
// #############################################################################

class finite_field {

private:
	INT f_has_table;
	INT *add_table; // [q * q]
	INT *mult_table; // [q * q]
		// add_table and mult_table are needed in mindist

	INT *negate_table;
	INT *inv_table;
	INT *frobenius_table; // x \mapsto x^p
	INT *absolute_trace_table;
	INT *log_alpha_table;
	INT *alpha_power_table;
	INT *v1, *v2, *v3; // vectors of length e.
	BYTE *symbol_for_print;

public:
	const BYTE *override_poly;
	BYTE *polynomial;
		// the actual polynomial we consider 
		// as integer (in text form)
	INT q, p, e;
	INT alpha; // primitive element
	INT log10_of_q; // needed for printing purposes
	INT f_print_as_exponentials;
	
	finite_field();
	void null();
	~finite_field();
	void init(INT q);
	void init(INT q, INT verbose_level);
	void init_symbol_for_print(const BYTE *symbol);
	void init_override_polynomial(INT q, const BYTE *poly, 
		INT verbose_level);
	void print_minimum_polynomial(INT p, const BYTE *polynomial);
	INT compute_subfield_polynomial(INT order_subfield, 
		INT verbose_level);
	void compute_subfields(INT verbose_level);
	void create_alpha_table(INT verbose_level);
	void create_alpha_table_extension_field(INT verbose_level);
	void create_alpha_table_prime_field(INT verbose_level);
	void create_tables_prime_field(INT verbose_level);
	void create_tables_extension_field(INT verbose_level);
	void print(INT f_add_mult_table);
	void print_add_mult_tables();
	void print_tables();
	void print_tables_extension_field(const BYTE *poly);
	void display_T2(ostream &ost);
	void display_T3(ostream &ost);
	void display_N2(ostream &ost);
	void display_N3(ostream &ost);
	void print_integer_matrix_zech(ostream &ost, 
		INT *p, INT m, INT n);

	INT *private_add_table();
	INT *private_mult_table();
	INT zero();
	INT one();
	INT minus_one();
	INT is_zero(INT i);
	INT is_one(INT i);
	INT mult(INT i, INT j);
	INT mult3(INT a1, INT a2, INT a3);
	INT product3(INT a1, INT a2, INT a3);
	INT mult4(INT a1, INT a2, INT a3, INT a4);
	INT product4(INT a1, INT a2, INT a3, INT a4);
	INT product5(INT a1, INT a2, INT a3, INT a4, INT a5);
	INT product_n(INT *a, INT n);
	INT square(INT a);
	INT twice(INT a);
	INT four_times(INT a);
	INT Z_embedding(INT k);
	INT add(INT i, INT j);
	INT add3(INT i1, INT i2, INT i3);
	INT add4(INT i1, INT i2, INT i3, INT i4);
	INT add5(INT i1, INT i2, INT i3, INT i4, INT i5);
	INT add6(INT i1, INT i2, INT i3, INT i4, INT i5, INT i6);
	INT add7(INT i1, INT i2, INT i3, INT i4, INT i5, INT i6, 
		INT i7);
	INT add8(INT i1, INT i2, INT i3, INT i4, INT i5, INT i6, 
		INT i7, INT i8);
	INT negate(INT i);
	INT inverse(INT i);
	INT power(INT a, INT n); // computes a^n
	INT frobenius_power(INT a, INT i); // computes a^{p^i}
	INT absolute_trace(INT i);
	INT absolute_norm(INT i);
	INT alpha_power(INT i);
	INT log_alpha(INT i);
	INT square_root(INT i, INT &root);
	INT primitive_root();
	INT N2(INT a);
	INT N3(INT a);
	INT T2(INT a);
	INT T3(INT a);
	INT bar(INT a);
	void abc2xy(INT a, INT b, INT c, INT &x, INT &y, 
		INT verbose_level);
		// given a, b, c, determine x and y such that 
		// c = a * x^2 + b * y^2
		// such elements x and y exist for any choice of a, b, c.
	INT retract(finite_field &subfield, INT index, INT a, 
		INT verbose_level);
	void retract_INT_vec(finite_field &subfield, INT index, 
		INT *v_in, INT *v_out, INT len, INT verbose_level);
	INT embed(finite_field &subfield, INT index, INT b, 
		INT verbose_level);
	void subfield_embedding_2dimensional(finite_field &subfield, 
		INT *&components, INT *&embedding, 
		INT *&pair_embedding, INT verbose_level);
	void print_embedding(finite_field &subfield, 
		INT *components, INT *embedding, INT *pair_embedding);
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
		INT *components, INT *embedding, INT *pair_embedding);
	void print_indicator_square_nonsquare(INT a);
	void print_element(ostream &ost, INT a);
	void print_element_with_symbol(ostream &ost, 
		INT a, INT f_exponential, INT width, const BYTE *symbol);
	void INT_vec_print(ostream &ost, INT *v, INT len);
	void INT_vec_print_elements_exponential(ostream &ost, 
		INT *v, INT len, const BYTE *symbol_for_print);
	void latex_addition_table(ostream &f, 
		INT f_elements_exponential, const BYTE *symbol_for_print);
	void latex_multiplication_table(ostream &f, 
		INT f_elements_exponential, const BYTE *symbol_for_print);
	void latex_matrix(ostream &f, INT f_elements_exponential, 
		const BYTE *symbol_for_print, INT *M, INT m, INT n);
	void power_table(INT t, INT *power_table, INT len);
	INT evaluate_conic_form(INT *six_coeffs, INT *v3);
	INT evaluate_quadric_form_in_PG_three(INT *ten_coeffs, INT *v4);
	INT Pluecker_12(INT *x4, INT *y4);
	INT Pluecker_21(INT *x4, INT *y4);
	INT Pluecker_13(INT *x4, INT *y4);
	INT Pluecker_31(INT *x4, INT *y4);
	INT Pluecker_14(INT *x4, INT *y4);
	INT Pluecker_41(INT *x4, INT *y4);
	INT Pluecker_23(INT *x4, INT *y4);
	INT Pluecker_32(INT *x4, INT *y4);
	INT Pluecker_24(INT *x4, INT *y4);
	INT Pluecker_42(INT *x4, INT *y4);
	INT Pluecker_34(INT *x4, INT *y4);
	INT Pluecker_43(INT *x4, INT *y4);
	INT Pluecker_ij(INT i, INT j, INT *x4, INT *y4);
	INT evaluate_symplectic_form(INT len, INT *x, INT *y);
	INT evaluate_quadratic_form_x0x3mx1x2(INT *x);
	INT is_totally_isotropic_wrt_symplectic_form(INT k, 
		INT n, INT *Basis);
	void cheat_sheet(ostream &f, INT verbose_level);
	void cheat_sheet_top(ostream &f, INT nb_cols);
	void cheat_sheet_bottom(ostream &f);
	INT evaluate_monomial(INT *monomial, INT *variables, INT nb_vars);
	void projective_point_unrank(INT n, INT *v, INT rk);
	INT projective_point_rank(INT n, INT *v);

	// #####################################################################
	// finite_field_linear_algebra.C:
	// #####################################################################

	void copy_matrix(INT *A, INT *B, INT ma, INT na);
	void reverse_matrix(INT *A, INT *B, INT ma, INT na);
	void identity_matrix(INT *A, INT n);
	INT is_identity_matrix(INT *A, INT n);
	INT is_diagonal_matrix(INT *A, INT n);
	INT is_scalar_multiple_of_identity_matrix(INT *A, 
		INT n, INT &scalar);
	void diagonal_matrix(INT *A, INT n, INT alpha);
	void matrix_minor(INT f_semilinear, INT *A, 
		INT *B, INT n, INT f, INT l);
		// initializes B as the l x l minor of A 
		// (which is n x n) starting from row f. 
	void mult_matrix(INT *A, INT *B, INT *C, 
		INT ma, INT na, INT nb);
	void mult_vector_from_the_left(INT *v, INT *A, 
		INT *vA, INT m, INT n);
		// v[m], A[m][n], vA[n]
	void mult_vector_from_the_right(INT *A, INT *v, 
		INT *Av, INT m, INT n);
		// A[m][n], v[n], Av[m]
	void mult_matrix_matrix_verbose(INT *A, INT *B, 
		INT *C, INT m, INT n, INT o, INT verbose_level);
	void mult_matrix_matrix(INT *A, INT *B, INT *C, 
		INT m, INT n, INT o);
		// multiplies C := A * B, 
		// where A is m x n and B is n x o, 
		// so that C is m by o
		// C must already be allocated
	void semilinear_matrix_mult(INT *A, INT *B, INT *AB, INT n);
		// (A,f1) * (B,f2) = (A*B^{\varphi^{-f1}},f1+f2)
	void semilinear_matrix_mult_memory_given(INT *A, INT *B, 
		INT *AB, INT *tmp_B, INT n);
		// (A,f1) * (B,f2) = (A*B^{\varphi^{-f1}},f1+f2)
	void matrix_mult_affine(INT *A, INT *B, INT *AB, 
		INT n, INT verbose_level);
	void semilinear_matrix_mult_affine(INT *A, INT *B, INT *AB, INT n);
	INT matrix_determinant(INT *A, INT n, INT verbose_level);
	void matrix_inverse(INT *A, INT *Ainv, INT n, INT verbose_level);
	void matrix_invert(INT *A, INT *Tmp, 
		INT *Tmp_basecols, INT *Ainv, INT n, INT verbose_level);
		// Tmp points to n * n + 1 INT's
		// Tmp_basecols points to n INT's
	void semilinear_matrix_invert(INT *A, INT *Tmp, 
		INT *Tmp_basecols, INT *Ainv, INT n, INT verbose_level);
		// Tmp points to n * n + 1 INT's
		// Tmp_basecols points to n INT's
	void semilinear_matrix_invert_affine(INT *A, INT *Tmp, 
		INT *Tmp_basecols, INT *Ainv, INT n, INT verbose_level);
		// Tmp points to n * n + 1 INT's
		// Tmp_basecols points to n INT's
	void matrix_invert_affine(INT *A, INT *Tmp, INT *Tmp_basecols, 
		INT *Ainv, INT n, INT verbose_level);
		// Tmp points to n * n + 1 INT's
		// Tmp_basecols points to n INT's
	void projective_action_from_the_right(INT f_semilinear, 
		INT *v, INT *A, INT *vA, INT n, INT verbose_level);
		// vA = (v * A)^{p^f}  if f_semilinear (where f = A[n *  n]), 
		// vA = v * A otherwise
	void general_linear_action_from_the_right(INT f_semilinear, 
		INT *v, INT *A, INT *vA, INT n, INT verbose_level);
	void semilinear_action_from_the_right(INT *v, 
		INT *A, INT *vA, INT n);
		// vA = (v * A)^{p^f}  (where f = A[n *  n])
	void semilinear_action_from_the_left(INT *A, 
		INT *v, INT *Av, INT n);
		// Av = A * v^{p^f}
	void affine_action_from_the_right(INT f_semilinear, 
		INT *v, INT *A, INT *vA, INT n);
		// vA = (v * A)^{p^f} + b
	void zero_vector(INT *A, INT m);
	void all_one_vector(INT *A, INT m);
	void support(INT *A, INT m, INT *&support, INT &size);
	void characteristic_vector(INT *A, INT m, INT *set, INT size);
	INT is_zero_vector(INT *A, INT m);
	void add_vector(INT *A, INT *B, INT *C, INT m);
	void negate_vector(INT *A, INT *B, INT m);
	void negate_vector_in_place(INT *A, INT m);
	void scalar_multiply_vector_in_place(INT c, INT *A, INT m);
	void vector_frobenius_power_in_place(INT *A, INT m, INT f);
	INT dot_product(INT len, INT *v, INT *w);
	void transpose_matrix(INT *A, INT *At, INT ma, INT na);
	void transpose_matrix_in_place(INT *A, INT m);
	void invert_matrix(INT *A, INT *A_inv, INT n);
	void transform_form_matrix(INT *A, INT *Gram, 
		INT *new_Gram, INT d);
		// computes new_Gram = A * Gram * A^\top
	INT rank_of_matrix(INT *A, INT m, INT verbose_level);
	INT rank_of_matrix_memory_given(INT *A, 
		INT m, INT *B, INT *base_cols, INT verbose_level);
	INT rank_of_rectangular_matrix(INT *A, 
		INT m, INT n, INT verbose_level);
	INT rank_of_rectangular_matrix_memory_given(INT *A, 
		INT m, INT n, INT *B, INT *base_cols, 
		INT verbose_level);
	INT rank_and_basecols(INT *A, INT m, 
		INT *base_cols, INT verbose_level);
	void Gauss_step(INT *v1, INT *v2, INT len, 
		INT idx, INT verbose_level);
		// afterwards: v2[idx] = 0 
		// and v1,v2 span the same space as before
		// v1 is not changed if v1[idx] is nonzero
	void Gauss_step_make_pivot_one(INT *v1, INT *v2, 
		INT len, INT idx, INT verbose_level);
		// afterwards: v2[idx] = 0 
		// and v1,v2 span the same space as before
		// v1[idx] is zero
	INT base_cols_and_embedding(INT m, INT n, INT *A, 
		INT *base_cols, INT *embedding, INT verbose_level);
		// returns the rank rk of the matrix.
		// It also computes base_cols[rk] and embedding[m - rk]
		// It leaves A unchanged
	INT Gauss_easy(INT *A, INT m, INT n);
		// returns the rank
	INT Gauss_easy_memory_given(INT *A, INT m, INT n, INT *base_cols);
	INT Gauss_simple(INT *A, INT m, INT n, 
		INT *base_cols, INT verbose_level);
		// returns the rank which is the 
		// number of entries in base_cols
	INT Gauss_INT(INT *A, INT f_special, 
		INT f_complete, INT *base_cols, 
		INT f_P, INT *P, INT m, INT n, 
		INT Pn, INT verbose_level);
		// returns the rank which is the 
		// number of entries in base_cols
		// A is m x n,
		// P is m x Pn (provided f_P is TRUE)
	INT Gauss_INT_with_pivot_strategy(INT *A, 
		INT f_special, INT f_complete, INT *pivot_perm, 
		INT m, INT n, 
		INT (*find_pivot_function)(INT *A, INT m, INT n, INT r, 
		INT *pivot_perm, void *data),
		void *find_pivot_data,  
		INT verbose_level);
		// returns the rank which is the number of entries in pivots
		// A is a m x n matrix
	void Gauss_INT_with_given_pivots(INT *A, 
		INT f_special, INT f_complete, INT *pivots, INT nb_pivots, 
		INT m, INT n, 
		INT verbose_level);
		// A is a m x n matrix
	void kernel_columns(INT n, INT nb_base_cols, 
		INT *base_cols, INT *kernel_cols);
	void matrix_get_kernel_as_INT_matrix(INT *M, INT m, INT n, 
		INT *base_cols, INT nb_base_cols, 
		INT_matrix *kernel);
	void matrix_get_kernel(INT *M, INT m, INT n, 
		INT *base_cols, INT nb_base_cols, 
		INT &kernel_m, INT &kernel_n, INT *kernel);
		// kernel must point to the appropriate amount of memory! 
		// (at least n * (n - nb_base_cols) INT's)
	INT perp(INT n, INT k, INT *A, INT *Gram);
	INT RREF_and_kernel(INT n, INT k, INT *A, INT verbose_level);
	INT perp_standard(INT n, INT k, INT *A, INT verbose_level);
	INT perp_standard_with_temporary_data(INT n, INT k, INT *A, 
		INT *B, INT *K, INT *base_cols, 
		INT verbose_level);
	INT intersect_subspaces(INT n, INT k1, INT *A, INT k2, INT *B, 
		INT &k3, INT *intersection, INT verbose_level);
	INT n_choose_k_mod_p(INT n, INT k, INT verbose_level);
	void Dickson_polynomial(INT *map, INT *coeffs);
		// compute the coefficients of a degree q-1 polynomial 
		// which interpolates a given map
		// from F_q to F_q
	void projective_action_on_columns_from_the_left(INT *A, 
		INT *M, INT m, INT n, INT *perm, INT verbose_level);
	void builtin_transversal_rep_GLnq(INT *A, INT n, 
		INT f_semilinear, INT i, INT j, INT verbose_level);
	void affine_translation(INT n, INT coordinate_idx, 
		INT field_base_idx, INT *perm);
		// perm points to q^n INT's
		// field_base_idx is the base element whose 
		// translation we compute, 0 \le field_base_idx < e
		// coordinate_idx is the coordinate in which we shift, 
		// 0 \le coordinate_idx < n
	void affine_multiplication(INT n, 
		INT multiplication_order, INT *perm);
		// perm points to q^n INT's
		// compute the diagonal multiplication by alpha, i.e. 
		// the multiplication by alpha of each component
	void affine_frobenius(INT n, INT k, INT *perm);
		// perm points to q^n INT's
		// compute the diagonal action of the Frobenius 
		// automorphism to the power k, i.e., 
		// raises each component to the p^k-th power
	INT all_affine_translations_nb_gens(INT n);
	void all_affine_translations(INT n, INT *gens);
	void affine_generators(INT n, INT f_translations, 
		INT f_semilinear, INT frobenius_power, 
		INT f_multiplication, INT multiplication_order, 
		INT &nb_gens, INT &degree, INT *&gens, 
		INT &base_len, INT *&the_base);
	INT evaluate_bilinear_form(INT n, INT *v1, INT *v2, INT *Gram);
	INT evaluate_standard_hyperbolic_bilinear_form(INT n, 
		INT *v1, INT *v2);
	INT evaluate_quadratic_form(INT n, INT nb_terms, 
		INT *i, INT *j, INT *coeff, INT *x);
	void find_singular_vector_brute_force(INT n, INT form_nb_terms, 
		INT *form_i, INT *form_j, INT *form_coeff, INT *Gram, 
		INT *vec, INT verbose_level);
	void find_singular_vector(INT n, INT form_nb_terms, 
		INT *form_i, INT *form_j, INT *form_coeff, INT *Gram, 
		INT *vec, INT verbose_level);
	void complete_hyperbolic_pair(INT n, INT form_nb_terms, 
		INT *form_i, INT *form_j, INT *form_coeff, INT *Gram, 
		INT *vec1, INT *vec2, INT verbose_level);
	void find_hyperbolic_pair(INT n, INT form_nb_terms, 
		INT *form_i, INT *form_j, INT *form_coeff, INT *Gram, 
		INT *vec1, INT *vec2, INT verbose_level);
	void restrict_quadratic_form_list_coding(INT k,
		INT n, INT *basis, 
		INT form_nb_terms, 
		INT *form_i, INT *form_j, INT *form_coeff, 
		INT &restricted_form_nb_terms, 
		INT *&restricted_form_i, INT *&restricted_form_j, 
		INT *&restricted_form_coeff, 
		INT verbose_level);
	void restrict_quadratic_form(INT k, INT n, INT *basis, 
		INT *C, INT *D, INT verbose_level);
	INT compare_subspaces_ranked(INT *set1, INT *set2, INT size, 
		INT vector_space_dimension, INT verbose_level);
		// Compares the span of two sets of vectors.
		// returns 0 if equal, 1 if not
		// (this is so that it matches to the result 
		// of a compare function)
	INT compare_subspaces_ranked_with_unrank_function(
		INT *set1, INT *set2, INT size, 
		INT vector_space_dimension, 
		void (*unrank_point_func)(INT *v, INT rk, void *data), 
		void *rank_point_data, 
		INT verbose_level);
	INT Gauss_canonical_form_ranked(INT *set1, INT *set2, INT size, 
		INT vector_space_dimension, INT verbose_level);
		// Computes the Gauss canonical form 
		// for the generating set in set1.
		// The result is written to set2.
		// Returns the rank of the span of the elements in set1.
	INT lexleast_canonical_form_ranked(INT *set1, INT *set2, INT size, 
		INT vector_space_dimension, INT verbose_level);
		// Computes the lexleast generating set the subspace 
		// spanned by the elements in set1.
		// The result is written to set2.
		// Returns the rank of the span of the elements in set1.
	void reduce_mod_subspace_and_get_coefficient_vector(
		INT k, INT len, INT *basis, INT *base_cols, 
		INT *v, INT *coefficients, INT verbose_level);
	void reduce_mod_subspace(INT k, 
		INT len, INT *basis, INT *base_cols, 
		INT *v, INT verbose_level);
	INT is_contained_in_subspace(INT k, 
		INT len, INT *basis, INT *base_cols, 
		INT *v, INT verbose_level);
	void compute_and_print_projective_weights(INT *M, 
		INT n, INT k);
	INT code_minimum_distance(INT n, INT k, 
		INT *code, INT verbose_level);
		// code[k * n]
	void codewords_affine(INT n, INT k, 
		INT *code, // [k * n]
		INT *codewords, // q^k
		INT verbose_level);
	void code_projective_weight_enumerator(INT n, INT k, 
		INT *code, // [k * n]
		INT *weight_enumerator, // [n + 1]
		INT verbose_level);
	void code_weight_enumerator(INT n, INT k, 
		INT *code, // [k * n]
		INT *weight_enumerator, // [n + 1]
		INT verbose_level);
	void code_weight_enumerator_fast(INT n, INT k, 
		INT *code, // [k * n]
		INT *weight_enumerator, // [n + 1]
		INT verbose_level);
	void code_projective_weights(INT n, INT k, 
		INT *code, // [k * n]
		INT *&weights,
			// will be allocated [N] 
			// where N = theta_{k-1}
		INT verbose_level);
	INT is_subspace(INT d, INT dim_U, INT *Basis_U, INT dim_V, 
		INT *Basis_V, INT verbose_level);
	void Kronecker_product(INT *A, INT *B, 
		INT n, INT *AB);
	void Kronecker_product_square_but_arbitrary(INT *A, INT *B, 
		INT na, INT nb, INT *AB, INT &N, INT verbose_level);
	INT dependency(INT d, INT *v, INT *A, INT m, INT *rho, 
		INT verbose_level);
		// Lueneburg~\cite{Lueneburg87a} p. 104.
		// A is a matrix of size d + 1 times d
		// v[d]
		// rho is a column permutation of degree d
	void order_ideal_generator(INT d, INT idx, INT *mue, INT &mue_deg, 
		INT *A, INT *Frobenius, 
		INT verbose_level);
		// Lueneburg~\cite{Lueneburg87a} p. 105.
		// Frobenius is a matrix of size d x d
		// A is (d + 1) x d
		// mue[d + 1]
	void span_cyclic_module(INT *A, INT *v, INT n, INT *Mtx, 
		INT verbose_level);
	void random_invertible_matrix(INT *M, INT k, INT verbose_level);
	void make_all_irreducible_polynomials_of_degree_d(
		INT d, INT &nb, INT *&Table, INT verbose_level);
	INT count_all_irreducible_polynomials_of_degree_d(
		INT d, INT verbose_level);
	void adjust_basis(INT *V, INT *U, INT n, INT k, INT d, 
		INT verbose_level);
	void choose_vector_in_here_but_not_in_here_column_spaces(
		INT_matrix *V, INT_matrix *W, INT *v, INT verbose_level);
	void choose_vector_in_here_but_not_in_here_or_here_column_spaces(
		INT_matrix *V, INT_matrix *W1, INT_matrix *W2, INT *v, 
		INT verbose_level);
	INT 
	choose_vector_in_here_but_not_in_here_or_here_column_spaces_coset(
		INT &coset, 
		INT_matrix *V, INT_matrix *W1, INT_matrix *W2, INT *v, 
		INT verbose_level);
	void vector_add_apply(INT *v, INT *w, INT c, INT n);
	void vector_add_apply_with_stride(INT *v, INT *w, INT stride, 
		INT c, INT n);
	INT test_if_commute(INT *A, INT *B, INT k, INT verbose_level);
	void unrank_point_in_PG(INT *v, INT len, INT rk);
		// len is the length of the vector, 
		// not the projective dimension
	INT rank_point_in_PG(INT *v, INT len);
	INT nb_points_in_PG(INT n);
		// n is projective dimension
	void Borel_decomposition(INT n, INT *M, INT *B1, INT *B2, 
		INT *pivots, INT verbose_level);
	void map_to_standard_frame(INT d, INT *A, 
		INT *Transform, INT verbose_level);
		// d = vector space dimension
		// maps d + 1 points to the frame 
		// e_1, e_2, ..., e_d, e_1+e_2+..+e_d 
		// A is (d + 1) x d
		// Transform is d x d
	void map_frame_to_frame_with_permutation(INT d, INT *A, 
		INT *perm, INT *B, INT *Transform, INT verbose_level);
	void map_points_to_points_projectively(INT d, INT k, 
		INT *A, INT *B, INT *Transform, 
		INT &nb_maps, INT verbose_level);
		// A and B are (d + k + 1) x d
		// Transform is d x d
		// returns TRUE if a map exists
	INT BallChowdhury_matrix_entry(INT *Coord, INT *C, 
		INT *U, INT k, INT sz_U, 
		INT *T, INT verbose_level);
	void cubic_surface_family_24_generators(INT f_with_normalizer, 
		INT f_semilinear, 
		INT *&gens, INT &nb_gens, INT &data_size, 
		INT &group_order, INT verbose_level);

	// #####################################################################
	// finite_field_representations.C:
	// #####################################################################

	void representing_matrix8_R(INT *A, 
		INT q, INT a, INT b, INT c, INT d);
	void representing_matrix9_R(INT *A, 
		INT q, INT a, INT b, INT c, INT d);
	void representing_matrix9_U(INT *A, 
		INT a, INT b, INT c, INT d, INT beta);
	void representing_matrix8_U(INT *A, 
		INT a, INT b, INT c, INT d, INT beta);
	void representing_matrix8_V(INT *A, INT beta);
	void representing_matrix9b(INT *A, INT beta);
	void representing_matrix8a(INT *A, 
		INT a, INT b, INT c, INT d, INT beta);
	void representing_matrix8b(INT *A, INT beta);
	INT Term1(INT a1, INT e1);
	INT Term2(INT a1, INT a2, INT e1, INT e2);
	INT Term3(INT a1, INT a2, INT a3, INT e1, INT e2, INT e3);
	INT Term4(INT a1, INT a2, INT a3, INT a4, INT e1, INT e2, INT e3, 
		INT e4);
	INT Term5(INT a1, INT a2, INT a3, INT a4, INT a5, INT e1, INT e2, 
		INT e3, INT e4, INT e5);
	INT term1(INT a1, INT e1);
	INT term2(INT a1, INT a2, INT e1, INT e2);
	INT term3(INT a1, INT a2, INT a3, INT e1, INT e2, INT e3);
	INT term4(INT a1, INT a2, INT a3, INT a4, INT e1, INT e2, INT e3, 
		INT e4);
	INT term5(INT a1, INT a2, INT a3, INT a4, INT a5, INT e1, INT e2, 
		INT e3, INT e4, INT e5);
	INT m_term(INT q, INT a1, INT a2, INT a3);
	INT beta_trinomial(INT q, INT beta, INT a1, INT a2, INT a3);
	INT T3product2(INT a1, INT a2);
};

// #############################################################################
// finite_field_tables.C:
// #############################################################################

extern INT finitefield_primes[];
extern INT finitefield_nb_primes;
extern INT finitefield_largest_degree_irreducible_polynomial[];
extern const BYTE *finitefield_primitive_polynomial[][100];
const BYTE *get_primitive_polynomial(INT p, INT e, INT verbose_level);

// #############################################################################
// finite_ring.C:
// #############################################################################

class finite_ring {

	INT *add_table; // [q * q]
	INT *mult_table; // [q * q]

	INT *f_is_unit_table;
	INT *negate_table;
	INT *inv_table;

	public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

	INT q;
	INT p;
	INT e;

	finite_field *Fp;


	finite_ring();
	~finite_ring();
	void null();
	void freeself();
	void init(INT q, INT verbose_level);
	INT zero();
	INT one();
	INT is_zero(INT i);
	INT is_one(INT i);
	INT is_unit(INT i);
	INT add(INT i, INT j);
	INT mult(INT i, INT j);
	INT negate(INT i);
	INT inverse(INT i);
	INT Gauss_INT(INT *A, INT f_special, 
		INT f_complete, INT *base_cols, 
		INT f_P, INT *P, INT m, INT n, INT Pn, 
		INT verbose_level);
		// returns the rank which is the number 
		// of entries in base_cols
		// A is a m x n matrix,
		// P is a m x Pn matrix (if f_P is TRUE)
};

// #############################################################################
// generators_symplectic_group.C:
// #############################################################################


class generators_symplectic_group {
public:

	finite_field *F; // no ownership, do not destroy
	INT n; // must be even
	INT n_half; // n / 2
	INT q;
	INT qn; // = q^n

	INT *nb_candidates; // [n + 1]
	INT *cur_candidate; // [n]
	INT **candidates; // [n + 1][q^n]
	
	INT *Mtx; // [n * n]
	INT *v; // [n]
	INT *v2; // [n]
	INT *w; // [n]
	INT *Points; // [qn * n]

	INT nb_gens;
	INT *Data;
	INT *transversal_length;

	generators_symplectic_group();
	~generators_symplectic_group();
	void null();
	void freeself();
	void init(finite_field *F, INT n, INT verbose_level);
	INT count_strong_generators(INT &nb, INT *transversal_length, 
		INT &first_moved, INT depth, INT verbose_level);
	INT get_strong_generators(INT *Data, INT &nb, INT &first_moved, 
		INT depth, INT verbose_level);
	//void backtrack_search(INT &nb_sol, INT depth, INT verbose_level);
	void create_first_candidate_set(INT verbose_level);
	void create_next_candidate_set(INT level, INT verbose_level);
	INT dot_product(INT *u1, INT *u2);
};

// #############################################################################
// gl_classes.C:
// #############################################################################

class gl_classes {
public:
	INT k;
	INT q;
	finite_field *F;
	INT nb_irred;
	INT *Nb_irred;
	INT *First_irred;
	INT *Nb_part;
	INT **Tables;
	INT **Partitions;
	INT *Degree;

	gl_classes();
	~gl_classes();
	void null();
	void freeself();
	void init(INT k, finite_field *F, INT verbose_level);
	void print_polynomials(ofstream &ost);
	INT select_polynomial_first(INT *Select, INT verbose_level);
	INT select_polynomial_next(INT *Select, INT verbose_level);
	INT select_partition_first(INT *Select, INT *Select_partition, 
		INT verbose_level);
	INT select_partition_next(INT *Select, INT *Select_partition, 
		INT verbose_level);
	INT first(INT *Select, INT *Select_partition, INT verbose_level);
	INT next(INT *Select, INT *Select_partition, INT verbose_level);
	void print_matrix_and_centralizer_order_latex(ofstream &ost, 
		gl_class_rep *R);
	void make_matrix_from_class_rep(INT *Mtx, gl_class_rep *R, 
		INT verbose_level);
	void make_matrix(INT *Mtx, INT *Select, INT *Select_Partition, 
		INT verbose_level);
	void centralizer_order_Kung_basic(INT nb_irreds, 
		INT *poly_degree, INT *poly_mult, INT *partition_idx, 
		longinteger_object &co, 
		INT verbose_level);
	void centralizer_order_Kung(INT *Select_polynomial, 
		INT *Select_partition, longinteger_object &co, 
		INT verbose_level);
		// Computes the centralizer order of a matrix in GL(k,q) 
		// according to Kung's formula~\cite{Kung81}.
	void make_classes(gl_class_rep *&R, INT &nb_classes, 
		INT f_no_eigenvalue_one, INT verbose_level);
	void identify_matrix(INT *Mtx, gl_class_rep *R, INT *Basis, 
		INT verbose_level);
	void identify2(INT *Mtx, unipoly_object &poly, INT *Mult, 
		INT *Select_partition, INT *Basis, INT verbose_level);
	void compute_data_on_blocks(INT *Mtx, INT *Irreds, INT nb_irreds, 
		INT *Degree, INT *Mult, matrix_block_data *Data,
		INT verbose_level);
	void compute_generalized_kernels(matrix_block_data *Data, INT *M2, 
		INT d, INT b0, INT m, INT *poly_coeffs, INT verbose_level);
	INT identify_partition(INT *part, INT m, INT verbose_level);
	void choose_basis_for_rational_normal_form(INT *Mtx, 
		matrix_block_data *Data, INT nb_irreds, 
		INT *Basis, 
		INT verbose_level);
	void choose_basis_for_rational_normal_form_block(INT *Mtx, 
		matrix_block_data *Data, 
		INT *Basis, INT &b, 
		INT verbose_level);
	void generators_for_centralizer(INT *Mtx, gl_class_rep *R, 
		INT *Basis, INT **&Gens, INT &nb_gens, INT &nb_alloc, 
		INT verbose_level);
	void centralizer_generators(INT *Mtx, unipoly_object &poly, 
		INT *Mult, INT *Select_partition, 
		INT *Basis, INT **&Gens, INT &nb_gens, INT &nb_alloc,  
		INT verbose_level);
	void centralizer_generators_block(INT *Mtx, matrix_block_data *Data, 
		INT nb_irreds, INT h, 
		INT **&Gens, INT &nb_gens, INT &nb_alloc,  
		INT verbose_level);
	INT choose_basis_for_rational_normal_form_coset(INT level1, 
		INT level2, INT &coset, 
		INT *Mtx, matrix_block_data *Data, INT &b, INT *Basis, 
		INT verbose_level);
	void factor_polynomial(unipoly_object &char_poly, INT *Mult, 
		INT verbose_level);
	INT find_class_rep(gl_class_rep *Reps, INT nb_reps, 
		gl_class_rep *R, INT verbose_level);

};


class gl_class_rep {
public:
	INT_matrix type_coding;
	longinteger_object centralizer_order;
	longinteger_object class_length;

	gl_class_rep();
	~gl_class_rep();
	void init(INT nb_irred, INT *Select_polynomial, 
		INT *Select_partition, INT verbose_level);
	void compute_vector_coding(gl_classes *C, INT &nb_irred, 
		INT *&Poly_degree, INT *&Poly_mult, INT *&Partition_idx, 
		INT verbose_level);
	void centralizer_order_Kung(gl_classes *C, longinteger_object &co, 
		INT verbose_level);
};


class matrix_block_data {
public:
	INT d;
	INT m;
	INT *poly_coeffs;
	INT b0;
	INT b1;
	
	INT_matrix *K;
	INT cnt;
	INT *dual_part;
	INT *part;
	INT height;
	INT part_idx;
	
	matrix_block_data();
	~matrix_block_data();
	void null();
	void freeself();
	void allocate(INT k);
};

// #############################################################################
// group_generators.C:
// #############################################################################


void diagonal_orbit_perm(INT n, finite_field &GFq, 
	INT *orbit, INT *orbit_inv, INT verbose_level);
void frobenius_orbit_perm(INT n, finite_field &GFq, 
	INT *orbit, INT *orbit_inv, INT verbose_level);
void translation_in_AG(finite_field &GFq, INT n, INT i, 
	INT a, INT *perm, INT *v, INT verbose_level);
	// v[n] needs to be allocated 
	// p[q^n] needs to be allocated
void frobenius_in_AG(finite_field &GFq, INT n, 
	INT *perm, INT *v, INT verbose_level);
	// v[n] needs to be allocated 
	// p[q^n] needs to be allocated
void frobenius_in_PG(finite_field &GFq, INT n, 
	INT *perm, INT *v, INT verbose_level);
	// v[n + 1] needs to be allocated 
	// p[q^n+...+q+1] needs to be allocated
void AG_representation_of_matrix(finite_field &GFq, 
	INT n, INT f_from_the_right, 
	INT *M, INT *v, INT *w, INT *perm, INT verbose_level);
	// perm[q^n] needs to be already allocated
void AG_representation_one_dimensional(finite_field &GFq, 
	INT a, INT *perm, INT verbose_level);
	// perm[q] needs to be already allocated
INT nb_generators_affine_translations(finite_field &GFq, INT n);
void generators_affine_translations(finite_field &GFq, 
	INT n, INT *perms, INT verbose_level);
	// primes[n * d] needs to be allocated, where d = q^n
void generators_AGL1xAGL1_subdirect1(finite_field &GFq1, 
	finite_field &GFq2, INT u, INT v, 
	INT &nb_perms, INT *&perms, INT verbose_level);
void generators_AGL1q(finite_field &GFq, 
	INT &nb_perms, INT *&perms, INT verbose_level);
void generators_AGL1q_subgroup(finite_field &GFq, 
	INT index_in_multiplicative_group, 
	INT &nb_perms, INT *&perms, INT verbose_level);
void generators_AGL1_x_AGL1(finite_field &GFq1, 
	finite_field &GFq2, INT &deg, 
	INT &nb_perms, INT *&perms, INT verbose_level);
void generators_AGL1_x_AGL1_extension(finite_field &GFq1, 
	finite_field &GFq2, INT u, INT v, 
	INT &deg, INT &nb_perms, INT *&perms, INT verbose_level);
void generators_AGL1_x_AGL1_extended_once(finite_field &F1, 
	finite_field &F2, INT u, INT v, 
	INT &deg, INT &nb_perms, INT *&perms, INT verbose_level);
void generators_AGL1_x_AGL1_extended_twice(finite_field &F1, 
	finite_field &F2, INT u1, INT v1, 
	INT u2, INT v2, INT &deg, 
	INT &nb_perms, INT *&perms, INT verbose_level);
void generators_symmetric_group(INT deg, 
	INT &nb_perms, INT *&perms, INT verbose_level);
void generators_cyclic_group(INT deg, 
	INT &nb_perms, INT *&perms, INT verbose_level);
void generators_dihedral_group(INT deg, 
	INT &nb_perms, INT *&perms, INT verbose_level);
void generators_dihedral_involution(INT deg, 
	INT &nb_perms, INT *&perms, INT verbose_level);
void generators_identity_group(INT deg, 
	INT &nb_perms, INT *&perms, INT verbose_level);
void order_Bn_group_factorized(INT n, 
	INT *&factors, INT &nb_factors);
void generators_Bn_group(INT n, INT &deg, 
	INT &nb_perms, INT *&perms, INT verbose_level);
void generators_direct_product(INT deg1, INT nb_perms1, INT *perms1, 
	INT deg2, INT nb_perms2, INT *perms2, 
	INT &deg3, INT &nb_perms3, INT *&perms3, 
	INT verbose_levels);
void generators_concatenate(INT deg1, INT nb_perms1, INT *perms1, 
	INT deg2, INT nb_perms2, INT *perms2, 
	INT &deg3, INT &nb_perms3, INT *&perms3, 
	INT verbose_level);
void O4_isomorphism_4to2(finite_field *F, 
	INT *At, INT *As, INT &f_switch, INT *B, 
	INT verbose_level);
void O4_isomorphism_2to4(finite_field *F, 
	INT *At, INT *As, INT f_switch, INT *B);
void O4_grid_coordinates_rank(finite_field &F, 
	INT x1, INT x2, INT x3, INT x4, 
	INT &grid_x, INT &grid_y, INT verbose_level);
void O4_grid_coordinates_unrank(finite_field &F, 
	INT &x1, INT &x2, INT &x3, INT &x4, INT grid_x, 
	INT grid_y, INT verbose_level);
void O4_find_tangent_plane(finite_field &F, 
	INT pt_x1, INT pt_x2, INT pt_x3, INT pt_x4, 
	INT *tangent_plane, INT verbose_level);
INT matrix_group_base_len_projective_group(INT n, INT q, 
	INT f_semilinear, INT verbose_level);
INT matrix_group_base_len_affine_group(INT n, INT q, 
	INT f_semilinear, INT verbose_level);
INT matrix_group_base_len_general_linear_group(INT n, INT q, 
	INT f_semilinear, INT verbose_level);
void projective_matrix_group_base_and_orbits(INT n, 
	finite_field *F, INT f_semilinear, 
	INT base_len, INT degree, 
	INT *base, INT *transversal_length, 
	INT **orbit, INT **orbit_inv, 
	INT verbose_level);
void affine_matrix_group_base_and_transversal_length(INT n, 
	finite_field *F, INT f_semilinear, 
	INT base_len, INT degree, 
	INT *base, INT *transversal_length, 
	INT verbose_level);
void general_linear_matrix_group_base_and_transversal_length(INT n, 
	finite_field *F, INT f_semilinear, 
	INT base_len, INT degree, 
	INT *base, INT *transversal_length, 
	INT verbose_level);
void strong_generators_for_projective_linear_group(INT n, finite_field *F, 
	INT f_semilinear, 
	INT *&data, INT &size, INT &nb_gens, 
	INT verbose_level);
void strong_generators_for_affine_linear_group(INT n, finite_field *F, 
	INT f_semilinear, 
	INT *&data, INT &size, INT &nb_gens, 
	INT verbose_level);
void strong_generators_for_general_linear_group(INT n, finite_field *F, 
	INT f_semilinear, 
	INT *&data, INT &size, INT &nb_gens, 
	INT verbose_level);
void generators_for_parabolic_subgroup(INT n, finite_field *F, 
	INT f_semilinear, INT k, 
	INT *&data, INT &size, INT &nb_gens, 
	INT verbose_level);
void generators_for_stabilizer_of_three_collinear_points_in_PGL4(
	finite_field *F, 
	INT f_semilinear, 
	INT *&data, INT &size, INT &nb_gens, 
	INT verbose_level);
void generators_for_stabilizer_of_triangle_in_PGL4(finite_field *F, 
	INT f_semilinear, 
	INT *&data, INT &size, INT &nb_gens, 
	INT verbose_level);


// #############################################################################
// heisenberg.C
// #############################################################################

class heisenberg {

public:
	INT q;
	finite_field *F;
	INT n;
	INT len; // 2 * n + 1
	INT group_order; // q^len

	INT *Elt1;
	INT *Elt2;
	INT *Elt3;
	INT *Elt4;

	heisenberg();
	~heisenberg();
	void null();
	void freeself();
	void init(finite_field *F, INT n, INT verbose_level);
	void unrank_element(INT *Elt, INT rk);
	INT rank_element(INT *Elt);
	void element_add(INT *Elt1, INT *Elt2, INT *Elt3, INT verbose_level);
	void element_negate(INT *Elt1, INT *Elt2, INT verbose_level);
	INT element_add_by_rank(INT rk_a, INT rk_b, INT verbose_level);
	INT element_negate_by_rank(INT rk_a, INT verbose_level);
	void group_table(INT *&Table, INT verbose_level);
	void group_table_abv(INT *&Table_abv, INT verbose_level);
	void generating_set(INT *&gens, INT &nb_gens, INT verbose_level);


};

// #############################################################################
// homogeneous_polynomial_domain.C
// #############################################################################


class homogeneous_polynomial_domain {

public:
	INT q;
	INT n; // number of variables
	INT degree;
	finite_field *F;
	INT nb_monomials;
	INT *Monomials; // [nb_monomials * n]
	BYTE **symbols;
	BYTE **symbols_latex;
	INT *Variables; // [nb_monomials * degree]
		// Variables contains the monomials written out 
		// as a sequence of length degree 
		// with entries in 0,..,n-1.
		// the entries are listed in increasing order.
		// For instance, the monomial x_0^2x_1x_3 
		// is recorded as 0,0,1,3
	INT nb_affine; // n^degree
	INT *Affine; // [nb_affine * degree]
		// the affine elements are used for foiling 
		// when doing a linear substitution
	INT *v; // [n]
	INT *Affine_to_monomial; // [nb_affine]
		// for each vector in the affine space, 
		// record the monomial associated with it.
	projective_space *P;

	INT *coeff2; // [nb_monomials], used in substitute_linear
	INT *coeff3; // [nb_monomials], used in substitute_linear
	INT *coeff4; // [nb_monomials], used in substitute_linear
	INT *factors; // [degree]
	INT *my_affine; // [degree], used in substitute_linear
	INT *base_cols; // [nb_monomials]
	INT *type1; // [degree + 1]
	INT *type2; // [degree + 1]

	homogeneous_polynomial_domain();
	~homogeneous_polynomial_domain();
	void freeself();
	void null();
	void init(finite_field *F, INT nb_vars, INT degree, 
		INT f_init_incidence_structure, INT verbose_level);
	void make_monomials(INT verbose_level);
	void rearrange_monomials_by_partition_type(INT verbose_level);
	INT index_of_monomial(INT *v);
	void print_monomial(ostream &ost, INT i);
	void print_monomial(ostream &ost, INT *mon);
	void print_monomial(BYTE *str, INT i);
	void print_equation(ostream &ost, INT *coeffs);
	void print_equation_with_line_breaks_tex(ostream &ost, 
		INT *coeffs, INT nb_terms_per_line, 
		const BYTE *new_line_text);
	void enumerate_points(INT *coeff, INT *Pts, INT &nb_pts, 
		INT verbose_level);
	INT evaluate_at_a_point_by_rank(INT *coeff, INT pt);
	INT evaluate_at_a_point(INT *coeff, INT *pt_vec);
	void substitute_linear(INT *coeff_in, INT *coeff_out, 
		INT *Mtx_inv, INT verbose_level);
	void substitute_semilinear(INT *coeff_in, INT *coeff_out, 
		INT f_semilinear, INT frob_power, INT *Mtx_inv, 
		INT verbose_level);
	INT is_zero(INT *coeff);
	void unrank_point(INT *v, INT rk);
	INT rank_point(INT *v);
	void unrank_coeff_vector(INT *v, INT rk);
	INT rank_coeff_vector(INT *v);
	INT test_weierstrass_form(INT rk, 
		INT &a1, INT &a2, INT &a3, INT &a4, INT &a6, 
		INT verbose_level);
	void vanishing_ideal(INT *Pts, INT nb_pts, INT &r, INT *Kernel, 
		INT verbose_level);
	INT compare_monomials(INT *M1, INT *M2);
	void print_monomial_ordering(ostream &ost);


};

INT homogeneous_polynomial_domain_compare_monomial_with(void *data, 
	INT i, INT *data2, void *extra_data);
INT homogeneous_polynomial_domain_compare_monomial(void *data, 
	INT i, INT j, void *extra_data);
void homogeneous_polynomial_domain_swap_monomial(void *data, 
	INT i, INT j, void *extra_data);



// #############################################################################
// longinteger_domain.C:
// #############################################################################

class longinteger_domain {

public:
	INT compare(longinteger_object &a, longinteger_object &b);
	INT compare_unsigned(longinteger_object &a, longinteger_object &b);
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
	void mult_integer_in_place(longinteger_object &a, INT b);
	void mult_mod(longinteger_object &a, 
		longinteger_object &b, longinteger_object &c, 
		longinteger_object &m, INT verbose_level);
	void multiply_up(longinteger_object &a, INT *x, INT len);
	INT quotient_as_INT(longinteger_object &a, longinteger_object &b);
	void integral_division_exact(longinteger_object &a, 
		longinteger_object &b, longinteger_object &a_over_b);
	void integral_division(
		longinteger_object &a, longinteger_object &b, 
		longinteger_object &q, longinteger_object &r, 
		INT verbose_level);
	void integral_division_by_INT(longinteger_object &a, 
		INT b, longinteger_object &q, INT &r);
	void extended_gcd(longinteger_object &a, longinteger_object &b, 
		longinteger_object &g, longinteger_object &u, 
		longinteger_object &v, INT verbose_level);
	INT logarithm_base_b(longinteger_object &a, INT b);
	void base_b_representation(longinteger_object &a, 
		INT b, INT *&rep, INT &len);
	void power_int(longinteger_object &a, INT n);
	void power_int_mod(longinteger_object &a, INT n, 
		longinteger_object &m);
	void power_longint_mod(longinteger_object &a, 
		longinteger_object &n, longinteger_object &m, 
		INT verbose_level);
	void create_qnm1(longinteger_object &a, INT q, INT n);
	void binomial(longinteger_object &a, INT n, INT k, 
		INT verbose_level);
	void size_of_conjugacy_class_in_sym_n(longinteger_object &a, 
		INT n, INT *part);
	void q_binomial(longinteger_object &a, 
		INT n, INT k, INT q, INT verbose_level);
	void q_binomial_no_table(longinteger_object &a, 
		INT n, INT k, INT q, INT verbose_level);
	void krawtchouk(longinteger_object &a, INT n, INT q, INT k, INT x);
	INT is_even(longinteger_object &a);
	INT is_odd(longinteger_object &a);
	INT remainder_mod_INT(longinteger_object &a, INT p);
	INT multiplicity_of_p(longinteger_object &a, 
		longinteger_object &residue, INT p);
	INT smallest_primedivisor(longinteger_object &a, INT p_min, 
		INT verbose_level);
	void factor_into_longintegers(longinteger_object &a, 
		INT &nb_primes, longinteger_object *&primes, 
		INT *&exponents, INT verbose_level);
	void factor(longinteger_object &a, INT &nb_primes, 
		INT *&primes, INT *&exponents, 
		INT verbose_level);
	INT jacobi(longinteger_object &a, longinteger_object &m, 
		INT verbose_level);
	void random_number_less_than_n(longinteger_object &n, 
		longinteger_object &r);
	void find_probable_prime_above(
		longinteger_object &a, 
		INT nb_solovay_strassen_tests, INT f_miller_rabin_test, 
		INT verbose_level);
	INT solovay_strassen_is_prime(
		longinteger_object &n, INT nb_tests, INT verbose_level);
	INT solovay_strassen_is_prime_single_test(
		longinteger_object &n, INT verbose_level);
	INT solovay_strassen_test(
		longinteger_object &n, longinteger_object &a, 
		INT verbose_level);
	INT miller_rabin_test(
		longinteger_object &n, INT verbose_level);
	void get_k_bit_random_pseudoprime(
		longinteger_object &n, INT k, 
		INT nb_tests_solovay_strassen, 
		INT f_miller_rabin_test, INT verbose_level);
	void RSA_setup(longinteger_object &n, 
		longinteger_object &p, longinteger_object &q, 
		longinteger_object &a, longinteger_object &b, 
		INT nb_bits, 
		INT nb_tests_solovay_strassen, INT f_miller_rabin_test, 
		INT verbose_level);
	void matrix_product(longinteger_object *A, longinteger_object *B, 
		longinteger_object *&C, INT Am, INT An, INT Bn);
	void matrix_entries_integral_division_exact(longinteger_object *A, 
		longinteger_object &b, INT Am, INT An);
	void matrix_print_GAP(ostream &ost, longinteger_object *A, 
		INT Am, INT An);
	void matrix_print_tex(ostream &ost, longinteger_object *A, 
		INT Am, INT An);
	void power_mod(char *aa, char *bb, char *nn, 
		longinteger_object &result, INT verbose_level);
	void factorial(longinteger_object &result, INT n);
	void group_order_PGL(longinteger_object &result, 
		INT n, INT q, INT f_semilinear);
	INT singleton_bound_for_d(INT n, INT k, INT q, INT verbose_level);
	INT hamming_bound_for_d(INT n, INT k, INT q, INT verbose_level);
	INT plotkin_bound_for_d(INT n, INT k, INT q, INT verbose_level);
	INT griesmer_bound_for_d(INT n, INT k, INT q, INT verbose_level);
	INT griesmer_bound_for_n(INT k, INT d, INT q, INT verbose_level);
};

void test_longinteger();
void test_longinteger2();
void test_longinteger3();
void test_longinteger4();
void test_longinteger5();
void test_longinteger6();
void test_longinteger7();
void test_longinteger8();
void mac_williams_equations(longinteger_object *&M, INT n, INT k, INT q);
void determine_weight_enumerator();
void longinteger_collect_setup(INT &nb_agos, 
	longinteger_object *&agos, INT *&multiplicities);
void longinteger_collect_free(INT &nb_agos, 
	longinteger_object *&agos, INT *&multiplicities);
void longinteger_collect_add(INT &nb_agos, 
	longinteger_object *&agos, INT *&multiplicities, 
	longinteger_object &ago);
void longinteger_collect_print(ostream &ost, INT &nb_agos, 
	longinteger_object *&agos, INT *&multiplicities);
void longinteger_free_global_data();
void longinteger_print_digits(BYTE *rep, INT len);



// #############################################################################
// norm_tables.C:
// #############################################################################

class norm_tables {
public:
	INT *norm_table;
	INT *norm_table_sorted;
	INT *sorting_perm, *sorting_perm_inv;
	INT nb_types;
	INT *type_first, *type_len;
	INT *the_type;

	norm_tables();
	~norm_tables();
	void init(unusual_model &U, INT verbose_level);
	INT choose_an_element_of_given_norm(INT norm, INT verbose_level);
	
};


// #############################################################################
// null_polarity_generator.C:
// #############################################################################

class null_polarity_generator {
public:

	finite_field *F; // no ownership, do not destroy
	INT n, q;
	INT qn; // = q^n

	INT *nb_candidates; // [n + 1]
	INT *cur_candidate; // [n]
	INT **candidates; // [n + 1][q^n]
	
	INT *Mtx; // [n * n]
	INT *v; // [n]
	INT *w; // [n]
	INT *Points; // [qn * n]

	INT nb_gens;
	INT *Data;
	INT *transversal_length;

	null_polarity_generator();
	~null_polarity_generator();
	void null();
	void freeself();
	void init(finite_field *F, INT n, INT verbose_level);
	INT count_strong_generators(INT &nb, INT *transversal_length, 
		INT &first_moved, INT depth, INT verbose_level);
	INT get_strong_generators(INT *Data, INT &nb, INT &first_moved, 
		INT depth, INT verbose_level);
	void backtrack_search(INT &nb_sol, INT depth, INT verbose_level);
	void create_first_candidate_set(INT verbose_level);
	void create_next_candidate_set(INT level, INT verbose_level);
	INT dot_product(INT *u1, INT *u2);
};

// #############################################################################
// number_theory.C:
// #############################################################################


INT power_mod(INT a, INT n, INT p);
INT inverse_mod(INT a, INT p);
INT mult_mod(INT a, INT b, INT p);
INT add_mod(INT a, INT b, INT p);
INT INT_abs(INT a);
INT irem(INT a, INT m);
INT gcd_INT(INT m, INT n);
void extended_gcd_INT(INT m, INT n, INT &g, INT &u, INT &v);
INT i_power_j(INT i, INT j);
INT order_mod_p(INT a, INT p);
INT INT_log2(INT n);
INT INT_log10(INT n);
INT INT_logq(INT n, INT q);
// returns the number of digits in base q representation
INT is_strict_prime_power(INT q);
// assuming that q is a prime power, this fuction tests 
// whether or not q is a strict prime power
INT is_prime(INT p);
INT is_prime_power(INT q);
INT is_prime_power(INT q, INT &p, INT &h);
INT smallest_primedivisor(INT n);
//Computes the smallest prime dividing $n$. 
//The algorithm is based on Lueneburg~\cite{Lueneburg87a}.
INT sp_ge(INT n, INT p_min);
INT factor_INT(INT a, INT *&primes, INT *&exponents);
void factor_prime_power(INT q, INT &p, INT &e);
INT primitive_root(INT p, INT verbose_level);
INT Legendre(INT a, INT p, INT verbose_level);
INT Jacobi(INT a, INT m, INT verbose_level);
INT Jacobi_with_key_in_latex(ostream &ost, INT a, INT m, INT verbose_level);
INT ny2(INT x, INT &x1);
INT ny_p(INT n, INT p);
INT sqrt_mod_simple(INT a, INT p);
void print_factorization(INT nb_primes, INT *primes, INT *exponents);
void print_longfactorization(INT nb_primes, 
	longinteger_object *primes, INT *exponents);
INT euler_function(INT n);
void INT_add_fractions(INT at, INT ab, INT bt, INT bb, 
	INT &ct, INT &cb, INT verbose_level);
void INT_mult_fractions(INT at, INT ab, INT bt, INT bb, 
	INT &ct, INT &cb, INT verbose_level);


// #############################################################################
// rank_checker.C:
// #############################################################################



class rank_checker {

public:
	finite_field *GFq;
	INT m, n, d;
	
	INT *M1; // [m * n]
	INT *M2; // [m * n]
	INT *base_cols; // [n]
	INT *set; // [n] used in check_mindist

	rank_checker();
	~rank_checker();
	void init(finite_field *GFq, INT m, INT n, INT d);
	INT check_rank(INT len, INT *S, INT verbose_level);
	INT check_rank_matrix_input(INT len, INT *S, INT dim_S, 
		INT verbose_level);
	INT check_rank_last_two_are_fixed(INT len, INT *S, INT verbose_level);
	INT compute_rank(INT len, INT *S, INT f_projective, INT verbose_level);
	INT compute_rank_row_vectors(INT len, INT *S, INT f_projective, 
		INT verbose_level);
};

// #############################################################################
// subfield_structure.C:
// #############################################################################


class subfield_structure {
public:

	finite_field *FQ;
	finite_field *Fq;
	INT Q;
	INT q;
	INT s; // subfield index: q^s = Q
	INT *Basis;
		// [s], entries are elements in FQ

	INT *embedding; 
		// [Q], entries are elements in FQ, 
		// indexed by elements in AG(s,q)
	INT *embedding_inv;
		// [Q], entries are ranks of elements in AG(s,q), 
		// indexed by elements in FQ
		// the inverse of embedding

	INT *components;
		// [Q * s], entries are elements in Fq
		// the vectors corresponding to the AG(s,q) 
		// ranks in embedding_inv[]

	INT *FQ_embedding; 
		// [q] entries are elements in FQ corresponding to 
		// the elements in Fq
	INT *Fq_element;
		// [Q], entries are the elements in Fq 
		// corresponding to a given FQ element
		// or -1 if the FQ element does not belong to Fq.
	INT *v; // [s]
	
	subfield_structure();
	~subfield_structure();
	void null();
	void freeself();
	void init(finite_field *FQ, finite_field *Fq, INT verbose_level);
	void init_with_given_basis(finite_field *FQ, finite_field *Fq, 
		INT *given_basis, INT verbose_level);
	void print_embedding();
	INT evaluate_over_FQ(INT *v);
	INT evaluate_over_Fq(INT *v);
	void lift_matrix(INT *MQ, INT m, INT *Mq, INT verbose_level);
	void retract_matrix(INT *Mq, INT n, INT *MQ, INT m, 
		INT verbose_level);

};

// #############################################################################
// unipoly_domain.C:
// #############################################################################

class unipoly_domain {
public:
	finite_field *gfq;
	INT f_factorring;
	INT factor_degree;
	INT *factor_coeffs;
	unipoly_object factor_poly;

	unipoly_domain(finite_field *GFq);
	unipoly_domain(finite_field *GFq, unipoly_object m);
	~unipoly_domain();
	INT &s_i(unipoly_object p, INT i)
		{ INT *rep = (INT *) p; return rep[i + 1]; };
	void create_object_of_degree(unipoly_object &p, INT d);
	void create_object_of_degree_with_coefficients(unipoly_object &p, 
		INT d, INT *coeff);
	void create_object_by_rank(unipoly_object &p, INT rk);
	void create_object_by_rank_longinteger(unipoly_object &p, 
		longinteger_object &rank, INT verbose_level);
	void create_object_by_rank_string(unipoly_object &p, 
		const BYTE *rk, INT verbose_level);
	void create_Dickson_polynomial(unipoly_object &p, INT *map);
	void delete_object(unipoly_object &p);
	void unrank(unipoly_object p, INT rk);
	void unrank_longinteger(unipoly_object p, longinteger_object &rank);
	INT rank(unipoly_object p);
	void rank_longinteger(unipoly_object p, longinteger_object &rank);
	INT degree(unipoly_object p);
	ostream& print_object(unipoly_object p, ostream& ost);
	void assign(unipoly_object a, unipoly_object &b);
	void one(unipoly_object p);
	void m_one(unipoly_object p);
	void zero(unipoly_object p);
	INT is_one(unipoly_object p);
	INT is_zero(unipoly_object p);
	void negate(unipoly_object a);
	void make_monic(unipoly_object &a);
	void add(unipoly_object a, unipoly_object b, unipoly_object &c);
	void mult(unipoly_object a, unipoly_object b, unipoly_object &c);
	void mult_easy(unipoly_object a, unipoly_object b, unipoly_object &c);
	void mult_mod(unipoly_object a, unipoly_object b, unipoly_object &c, 
		INT factor_polynomial_degree, 
		INT *factor_polynomial_coefficents_negated, 
		INT verbose_level);
	void Frobenius_matrix(INT *&Frob, unipoly_object factor_polynomial, 
		INT verbose_level);
	void Berlekamp_matrix(INT *&B, unipoly_object factor_polynomial, 
		INT verbose_level);
	void integral_division_exact(unipoly_object a, 
		unipoly_object b, unipoly_object &q, INT verbose_level);
	void integral_division(unipoly_object a, unipoly_object b, 
		unipoly_object &q, unipoly_object &r, INT verbose_level);
	void derive(unipoly_object a, unipoly_object &b);
	INT compare_euclidean(unipoly_object m, unipoly_object n);
	void greatest_common_divisor(unipoly_object m, unipoly_object n, 
		unipoly_object &g, INT verbose_level);
	void extended_gcd(unipoly_object m, unipoly_object n, 
		unipoly_object &u, unipoly_object &v, 
		unipoly_object &g, INT verbose_level);
	INT is_squarefree(unipoly_object p, INT verbose_level);
	void compute_normal_basis(INT d, INT *Normal_basis, 
		INT *Frobenius, INT verbose_level);
	void order_ideal_generator(INT d, INT idx, unipoly_object &mue, 
		INT *A, INT *Frobenius, 
		INT verbose_level);
		// Lueneburg~\cite{Lueneburg87a} p. 105.
		// Frobenius is a matrix of size d x d
		// A is a matrix of size (d + 1) x d
	void matrix_apply(unipoly_object &p, INT *Mtx, INT n, 
		INT verbose_level);
		// The matrix is applied on the left
	void substitute_matrix_in_polynomial(unipoly_object &p, 
		INT *Mtx_in, INT *Mtx_out, INT k, INT verbose_level);
		// The matrix is substituted into the polynomial
	INT substitute_scalar_in_polynomial(unipoly_object &p, 
		INT scalar, INT verbose_level);
		// The scalar 'scalar' is substituted into the polynomial
	void module_structure_apply(INT *v, INT *Mtx, INT n, 
		unipoly_object p, INT verbose_level);
	void take_away_all_factors_from_b(unipoly_object a, 
		unipoly_object b, unipoly_object &a_without_b, 
		INT verbose_level);
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
	INT is_irreducible(unipoly_object a, INT verbose_level);
	void singer_candidate(unipoly_object &m, 
		INT p, INT d, INT b, INT a);
	INT is_primitive(unipoly_object &m, 
		longinteger_object &qm1, 
		INT nb_primes, longinteger_object *primes, 
		INT verbose_level);
	void get_a_primitive_polynomial(unipoly_object &m, 
		INT f, INT verbose_level);
	void get_an_irreducible_polynomial(unipoly_object &m, 
		INT f, INT verbose_level);
	void power_INT(unipoly_object &a, INT n, INT verbose_level);
	void power_longinteger(unipoly_object &a, longinteger_object &n);
	void power_coefficients(unipoly_object &a, INT n);
	void minimum_polynomial(unipoly_object &a, 
		INT alpha, INT p, INT verbose_level);
	INT minimum_polynomial_factorring(INT alpha, INT p, 
		INT verbose_level);
	void minimum_polynomial_factorring_longinteger(
		longinteger_object &alpha, 
		longinteger_object &rk_minpoly, 
		INT p, INT verbose_level);
	void BCH_generator_polynomial(unipoly_object &g, INT n, 
		INT designed_distance, INT &bose_distance, 
		INT &transversal_length, INT *&transversal, 
		longinteger_object *&rank_of_irreducibles, 
		INT verbose_level);
	void compute_generator_matrix(unipoly_object a, INT *&genma, 
		INT n, INT &k, INT verbose_level);
	void print_vector_of_polynomials(unipoly_object *sigma, INT deg);
	void minimum_polynomial_extension_field(unipoly_object &g, 
		unipoly_object m, 
		unipoly_object &minpol, INT d, INT *Frobenius, 
		INT verbose_level);
		// Lueneburg~\cite{Lueneburg87a}, p. 112.
	void characteristic_polynomial(INT *Mtx, INT k, 
		unipoly_object &char_poly, INT verbose_level);
	void print_matrix(unipoly_object *M, INT k);
	void determinant(unipoly_object *M, INT k, unipoly_object &p, 
		INT verbose_level);
	void deletion_matrix(unipoly_object *M, INT k, INT delete_row, 
		INT delete_column, unipoly_object *&N, INT verbose_level);

};



