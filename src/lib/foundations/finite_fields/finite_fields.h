/*
 * finite_fields.h
 *
 *  Created on: Mar 2, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_FOUNDATIONS_FINITE_FIELDS_FINITE_FIELDS_H_
#define SRC_LIB_FOUNDATIONS_FINITE_FIELDS_FINITE_FIELDS_H_

namespace orbiter {
namespace foundations {



// #############################################################################
// finite_field_activity_description.cpp
// #############################################################################


//! description of a finite field activity

class finite_field_activity_description {
public:

	int f_cheat_sheet_GF;

	int f_polynomial_division;
	std::string polynomial_division_A;
	std::string polynomial_division_B;

	int f_extended_gcd_for_polynomials;

	int f_polynomial_mult_mod;
	std::string polynomial_mult_mod_A;
	std::string polynomial_mult_mod_B;
	std::string polynomial_mult_mod_M;

	int f_Berlekamp_matrix;
	std::string Berlekamp_matrix_coeffs;

	int f_normal_basis;
	int normal_basis_d;

	int f_polynomial_find_roots;
	std::string polynomial_find_roots_A;


	int f_normalize_from_the_right;
	int f_normalize_from_the_left;

	int f_nullspace;
	int nullspace_m;
	int nullspace_n;
	std::string nullspace_text;

	int f_RREF;
	int RREF_m;
	int RREF_n;
	std::string RREF_text;

	int f_weight_enumerator;

	int f_Walsh_Hadamard_transform;
	std::string Walsh_Hadamard_transform_fname_csv_in;
	int Walsh_Hadamard_transform_n;

	int f_algebraic_normal_form;
	std::string algebraic_normal_form_fname_csv_in;
	int algebraic_normal_form_n;


	int f_apply_trace_function;
	std::string apply_trace_function_fname_csv_in;

	int f_apply_power_function;
	std::string apply_power_function_fname_csv_in;
	long int apply_power_function_d;

	int f_identity_function;
	std::string identity_function_fname_csv_out;

	int f_trace;

	int f_norm;

	int f_Walsh_matrix;
	int Walsh_matrix_n;

	int f_make_table_of_irreducible_polynomials;
	int make_table_of_irreducible_polynomials_degree;

	int f_EC_Koblitz_encoding;
	std::string EC_message;
	int EC_s;

	int f_EC_points;
	std::string EC_label;

	int f_EC_add;
	std::string EC_pt1_text;
	std::string EC_pt2_text;

	int f_EC_cyclic_subgroup;
	int EC_b;
	int EC_c;
	std::string EC_pt_text;

	int f_EC_multiple_of;
	int EC_multiple_of_n;
	int f_EC_discrete_log;
	std::string EC_discrete_log_pt_text;

	int f_EC_baby_step_giant_step;
	std::string EC_bsgs_G;
	int EC_bsgs_N;
	std::string EC_bsgs_cipher_text;

	int f_EC_baby_step_giant_step_decode;
	std::string EC_bsgs_A;
	std::string EC_bsgs_keys;




	int f_NTRU_encrypt;
	int NTRU_encrypt_N;
	int NTRU_encrypt_p;
	std::string NTRU_encrypt_H;
	std::string NTRU_encrypt_R;
	std::string NTRU_encrypt_Msg;

	int f_polynomial_center_lift;
	std::string polynomial_center_lift_A;

	int f_polynomial_reduce_mod_p;
	std::string polynomial_reduce_mod_p_A;

	int f_cheat_sheet_PG;
	int cheat_sheet_PG_n;

	int f_cheat_sheet_Gr;
	int cheat_sheet_Gr_n;
	int cheat_sheet_Gr_k;

	int f_cheat_sheet_hermitian;
	int cheat_sheet_hermitian_projective_dimension;

	int f_cheat_sheet_desarguesian_spread;
	int cheat_sheet_desarguesian_spread_m;

	int f_find_CRC_polynomials;
	int find_CRC_polynomials_nb_errors;
	int find_CRC_polynomials_information_bits;
	int find_CRC_polynomials_check_bits;

	int f_sift_polynomials;
	long int sift_polynomials_r0;
	long int sift_polynomials_r1;

	int f_mult_polynomials;
	long int mult_polynomials_r0;
	long int mult_polynomials_r1;

	int f_polynomial_division_ranked;
	long int polynomial_division_r0;
	long int polynomial_division_r1;

	int f_polynomial_division_from_file;
	std::string polynomial_division_from_file_fname;
	long int polynomial_division_from_file_r1;

	int f_polynomial_division_from_file_all_k_bit_error_patterns;
	std::string polynomial_division_from_file_all_k_bit_error_patterns_fname;
	int polynomial_division_from_file_all_k_bit_error_patterns_r1;
	int polynomial_division_from_file_all_k_bit_error_patterns_k;

	int f_RREF_random_matrix;
	int RREF_random_matrix_m;
	int RREF_random_matrix_n;



	int f_transversal;
	std::string transversal_line_1_basis;
	std::string transversal_line_2_basis;
	std::string transversal_point;

	int f_intersection_of_two_lines;
	std::string line_1_basis;
	std::string line_2_basis;

	int f_move_two_lines_in_hyperplane_stabilizer;
	long int line1_from;
	long int line2_from;
	long int line1_to;
	long int line2_to;

	int f_move_two_lines_in_hyperplane_stabilizer_text;
	std::string line1_from_text;
	std::string line2_from_text;
	std::string line1_to_text;
	std::string line2_to_text;

	int f_inverse_isomorphism_klein_quadric;
	std::string inverse_isomorphism_klein_quadric_matrix_A6;

	int f_rank_point_in_PG;
	int rank_point_in_PG_n;
	std::string rank_point_in_PG_text;

	int f_rank_point_in_PG_given_as_pairs;
	int rank_point_in_PG_given_as_pairs_n;
	std::string rank_point_in_PG_given_as_pairs_text;


	int f_field_reduction;
	std::string field_reduction_label;
	int field_reduction_q;
	int field_reduction_m;
	int field_reduction_n;
	std::string field_reduction_text;



	int f_parse_and_evaluate;
	std::string parse_name_of_formula;
	std::string parse_managed_variables;
	std::string parse_text;
	std::string parse_parameters;

	int f_evaluate;
	std::string evaluate_formula_label;
	std::string evaluate_parameters;

#if 0
	int f_all_rational_normal_forms;
	int d;
	int f_study_surface;
	int study_surface_nb;

	int f_eigenstuff;
	int f_eigenstuff_from_file;
	int eigenstuff_n;
	std::string eigenstuff_coeffs;
	std::string eigenstuff_fname;

	int f_decomposition_by_element;
	int decomposition_by_element_n;
	int decomposition_by_element_power;
	std::string decomposition_by_element_data;
	std::string decomposition_by_element_fname_base;
#endif

	finite_field_activity_description();
	~finite_field_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);

};



// #############################################################################
// finite_field_activity.cpp
// #############################################################################


//! perform a finite field activity

class finite_field_activity {
public:
	finite_field_activity_description *Descr;
	finite_field *F;
	finite_field *F_secondary;

	finite_field_activity();
	~finite_field_activity();
	void init(finite_field_activity_description *Descr,
			finite_field *F,
			int verbose_level);
	void perform_activity(int verbose_level);

};



// #############################################################################
// finite_field_description.cpp
// #############################################################################


//! description of a finite field

class finite_field_description {
public:

	int f_q;
	int q;

	int f_override_polynomial;
	std::string override_polynomial;

	finite_field_description();
	~finite_field_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

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

	int *negate_table; // [q]
	int *inv_table; // [q]
	int *frobenius_table; // [q], x \mapsto x^p
	int *absolute_trace_table; // [q]
	int *log_alpha_table; // [q]
	// log_alpha_table[i] = the integer k s.t. alpha^k = i (if i > 0)
	// log_alpha_table[0] = -1
	int *alpha_power_table; // [q]
	int *v1, *v2, *v3; // [e]
	std::string symbol_for_print;
	int f_has_quadratic_subfield; // TRUE if e is even.
	int *f_belongs_to_quadratic_subfield; // [q]


	int my_nb_calls_to_elliptic_curve_addition;
	int nb_times_mult;
	int nb_times_add;

public:
	std::string label;
	std::string label_tex;
	std::string override_poly;
	char *polynomial;
		// the actual polynomial we consider
		// as integer (in text form)
	int f_is_prime_field;
	int q, p, e;
	int alpha; // primitive element
	int log10_of_q; // needed for printing purposes
	int f_print_as_exponentials;
	int *reordered_list_of_elements; // [q]
	int *reordered_list_of_elements_inv; // [q]
	long int nb_calls_to_mult_matrix_matrix;
	long int nb_calls_to_PG_element_rank_modified;
	long int nb_calls_to_PG_element_unrank_modified;

	finite_field();
	void null();
	~finite_field();
	void print_call_stats(std::ostream &ost);
	int &nb_calls_to_elliptic_curve_addition();
	void init(finite_field_description *Descr, int verbose_level);
	void finite_field_init(int q, int verbose_level);
	void set_default_symbol_for_print();
	void init_symbol_for_print(const char *symbol);
	void init_override_polynomial(int q, std::string &poly,
		int verbose_level);
	void init_binary_operations(int verbose_level);
	void init_frobenius_table(int verbose_level);
	void init_absolute_trace_table(int verbose_level);
	int has_quadratic_subfield();
	int belongs_to_quadratic_subfield(int a);
	void init_quadratic_subfield(int verbose_level);
	long int compute_subfield_polynomial(int order_subfield,
			int f_latex, std::ostream &ost,
			int verbose_level);
	void compute_subfields(int verbose_level);
	void create_alpha_table(int verbose_level);
	int find_primitive_element(int verbose_level);
	int compute_order_of_element(int elt, int verbose_level);
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
	int mult_verbose(int i, int j, int verbose_level);
	int a_over_b(int a, int b);
	int mult3(int a1, int a2, int a3);
	int product3(int a1, int a2, int a3);
	int mult4(int a1, int a2, int a3, int a4);
	int mult5(int a1, int a2, int a3, int a4, int a5);
	int mult6(int a1, int a2, int a3, int a4, int a5, int a6);
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
	int power(int a, int n);
	int power_verbose(int a, int n, int verbose_level);
		// computes a^n
	void frobenius_power_vec(int *v, int len, int frob_power);
	void frobenius_power_vec_to_vec(int *v_in, int *v_out, int len, int frob_power);
	int frobenius_power(int a, int frob_power);
		// computes a^{p^frob_power}
	int absolute_trace(int i);
	int absolute_norm(int i);
	int alpha_power(int i);
	int log_alpha(int i);
	int multiplicative_order(int a);
	void all_square_roots(int a, int &nb_roots, int *roots2);
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
	int nb_times_mult_called();
	int nb_times_add_called();

	// #########################################################################
	// finite_field_applications.cpp
	// #########################################################################


	void make_all_irreducible_polynomials_of_degree_d(
			int d, std::vector<std::vector<int> > &Table,
			int verbose_level);
	int count_all_irreducible_polynomials_of_degree_d(int d, int verbose_level);
	void polynomial_division(
			std::string &A_coeffs, std::string &B_coeffs, int verbose_level);
	void extended_gcd_for_polynomials(
			std::string &A_coeffs, std::string &B_coeffs, int verbose_level);
	void polynomial_mult_mod(
			std::string &A_coeffs, std::string &B_coeffs, std::string &M_coeffs,
			int verbose_level);
	void Berlekamp_matrix(
			std::string &Berlekamp_matrix_coeffs,
			int verbose_level);
	void compute_normal_basis(int d, int verbose_level);
	void do_nullspace(int m, int n, std::string &text,
			int f_normalize_from_the_left, int f_normalize_from_the_right,
			int verbose_level);
	void do_RREF(int m, int n, std::string &text,
			int f_normalize_from_the_left, int f_normalize_from_the_right,
			int verbose_level);
	void apply_Walsh_Hadamard_transform(
			std::string &fname_csv_in, int n, int verbose_level);
	void algebraic_normal_form(
			std::string &fname_csv_in, int n, int verbose_level);
	void apply_trace_function(
			std::string &fname_csv_in, int verbose_level);
	void apply_power_function(
			std::string &fname_csv_in, long int d, int verbose_level);
	void identity_function(
			std::string &fname_csv_out, int verbose_level);
	void do_trace(int verbose_level);
	void do_norm(int verbose_level);
	void do_cheat_sheet_GF(int verbose_level);
	void do_make_table_of_irreducible_polynomials(int deg, int verbose_level);
	void polynomial_find_roots(
			std::string &A_coeffs,
			int verbose_level);
	void sift_polynomials(long int rk0, long int rk1, int verbose_level);
	void mult_polynomials(long int rk0, long int rk1, int verbose_level);
	void polynomial_division_from_file_with_report(
			std::string &input_file, long int rk1, int verbose_level);
	void polynomial_division_from_file_all_k_error_patterns_with_report(
			std::string &input_file, long int rk1, int k, int verbose_level);
	void polynomial_division_with_report(long int rk0, long int rk1, int verbose_level);
	void RREF_demo(int *A, int m, int n, int verbose_level);
	void RREF_demo2(std::ostream &ost, int *A, int m, int n, int verbose_level);
	void gl_random_matrix(int k, int verbose_level);



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

	void mult_matrix_matrix(int *A, int *B,
		int *C, int m, int n, int o, int verbose_level);
		// matrix multiplication C := A * B,
		// where A is m x n and B is n x o, so that C is m by o
	void semilinear_matrix_mult(int *A, int *B, int *AB, int n);
		// (A,f1) * (B,f2) = (A*B^{\varphi^{-f1}},f1+f2)
	void semilinear_matrix_mult_memory_given(int *A, int *B,
		int *AB, int *tmp_B, int n, int verbose_level);
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
	void invert_matrix(int *A, int *A_inv, int n, int verbose_level);
	void invert_matrix_memory_given(int *A, int *A_inv, int n,
			int *tmp_A, int *tmp_basecols, int verbose_level);
	void transform_form_matrix(int *A, int *Gram,
		int *new_Gram, int d, int verbose_level);
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
	void kernel_columns(int n, int nb_base_cols,
		int *base_cols, int *kernel_cols);
	void matrix_get_kernel_as_int_matrix(int *M, int m, int n,
		int *base_cols, int nb_base_cols,
		int_matrix *kernel, int verbose_level);
	void matrix_get_kernel(int *M, int m, int n,
		int *base_cols, int nb_base_cols,
		int &kernel_m, int &kernel_n, int *kernel, int verbose_level);
		// kernel[n * (n - nb_base_cols)]
	int perp(int n, int k, int *A, int *Gram, int verbose_level);
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
	void exterior_square(int *An, int *An2, int n, int verbose_level);
	void lift_to_Klein_quadric(int *A4, int *A6, int verbose_level);


	// #########################################################################
	// finite_field_linear_algebra2.cpp
	// #########################################################################

	void reduce_mod_subspace_and_get_coefficient_vector(
		int k, int len, int *basis, int *base_cols,
		int *v, int *coefficients, int verbose_level);
	void reduce_mod_subspace(int k,
		int len, int *basis, int *base_cols,
		int *v, int verbose_level);
	int is_contained_in_subspace(int k,
		int len, int *basis, int *base_cols,
		int *v, int verbose_level);
	int is_subspace(int d, int dim_U, int *Basis_U, int dim_V,
		int *Basis_V, int verbose_level);
	void Kronecker_product(int *A, int *B, int n, int *AB);
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
	void cubic_surface_family_G13_generators(
			int a,
			int *&gens, int &nb_gens, int &data_size,
			int &group_order, int verbose_level);
	void cubic_surface_family_F13_generators(
		int a,
		int *&gens, int &nb_gens, int &data_size,
		int &group_order, int verbose_level);
	int is_unit_vector(int *v, int len, int k);
	void make_Fourier_matrices(
			int omega, int k, int *N, int **A, int **Av,
			int *Omega, int verbose_level);

	// #########################################################################
	// finite_field_orthogonal.cpp
	// #########################################################################

	void Q_epsilon_unrank(
		int *v, int stride, int epsilon, int k,
		int c1, int c2, int c3, long int a, int verbose_level);
	long int Q_epsilon_rank(
		int *v, int stride, int epsilon, int k,
		int c1, int c2, int c3, int verbose_level);
	//void init_hash_table_parabolic(int k, int verbose_level);
	void Q_unrank(int *v, int stride, int k, long int a, int verbose_level);
	long int Q_rank(int *v, int stride, int k, int verbose_level);
	void Q_unrank_directly(int *v, int stride, int k, long int a, int verbose_level);
		// parabolic quadric
		// k = projective dimension, must be even
	long int Q_rank_directly(int *v, int stride, int k, int verbose_level);
	void Qplus_unrank(int *v, int stride, int k, long int a, int verbose_level);
		// hyperbolic quadric
		// k = projective dimension, must be odd
	long int Qplus_rank(int *v, int stride, int k, int verbose_level);
	void Qminus_unrank(int *v,
			int stride, int k, long int a,
			int c1, int c2, int c3, int verbose_level);
		// elliptic quadric
		// k = projective dimension, must be odd
		// the form is
		// \sum_{i=0}^n x_{2i}x_{2i+1} + c1 x_{2n}^2 +
		// c2 x_{2n} x_{2n+1} + c3 x_{2n+1}^2
	long int Qminus_rank(int *v, int stride,
			int k, int c1, int c2, int c3, int verbose_level);
	void S_unrank(int *v, int stride, int n, long int a);
	void S_rank(int *v, int stride, int n, long int &a);
	void N_unrank(int *v, int stride, int n, long int a);
	void N_rank(int *v, int stride, int n, long int &a);
	void N1_unrank(int *v, int stride, int n, long int a);
	void N1_rank(int *v, int stride, int n, long int &a);
	void Sbar_unrank(int *v, int stride, int n, long int a, int verbose_level);
	void Sbar_rank(int *v, int stride, int n, long int &a, int verbose_level);
	void Nbar_unrank(int *v, int stride, int n, long int a);
	void Nbar_rank(int *v, int stride, int n, long int &a);
	void Gram_matrix(int epsilon, int k,
		int form_c1, int form_c2, int form_c3,
		int *&Gram, int verbose_level);
	int evaluate_bilinear_form(
			int *u, int *v, int d, int *Gram);
	int evaluate_quadratic_form(int *v, int stride,
		int epsilon, int k, int form_c1, int form_c2, int form_c3);
	int evaluate_hyperbolic_quadratic_form(
			int *v, int stride, int n);
	int evaluate_hyperbolic_bilinear_form(
			int *u, int *v, int n);
	int primitive_element();
	void Siegel_map_between_singular_points(int *T,
			long int rk_from, long int rk_to, long int root,
		int epsilon, int algebraic_dimension,
		int form_c1, int form_c2, int form_c3, int *Gram_matrix,
		int verbose_level);
	// root is not perp to from and to.
	void Siegel_Transformation(
		int epsilon, int k,
		int form_c1, int form_c2, int form_c3,
		int *M, int *v, int *u, int verbose_level);
		// if u is singular and v \in \la u \ra^\perp, then
		// \pho_{u,v}(x) := x + \beta(x,v) u - \beta(x,u) v - Q(v) \beta(x,u) u
		// is called the Siegel transform (see Taylor p. 148)
		// Here Q is the quadratic form
		// and \beta is the corresponding bilinear form
	long int orthogonal_find_root(int rk2,
		int epsilon, int algebraic_dimension,
		int form_c1, int form_c2, int form_c3, int *Gram_matrix,
		int verbose_level);
	void choose_anisotropic_form(
			int &c1, int &c2, int &c3, int verbose_level);


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

	void PG_element_apply_frobenius(int n, int *v, int f);
	void number_of_conditions_satisfied(
			std::string &variety_label,
			int variety_nb_vars, int variety_degree,
			std::vector<std::string> &Variety_coeffs,
			monomial_ordering_type Monomial_ordering_type,
			std::string &number_of_conditions_satisfied_fname,
			std::string &fname, int &nb_pts, long int *&Pts,
			int verbose_level);
	void create_intersection_of_zariski_open_sets(
			std::string &variety_label,
			int variety_nb_vars, int variety_degree,
			std::vector<std::string> &Variety_coeffs,
			monomial_ordering_type Monomial_ordering_type,
			std::string &fname, int &nb_pts, long int *&Pts,
			int verbose_level);
	void create_projective_variety(
			std::string &variety_label,
			int variety_nb_vars, int variety_degree,
			std::string &variety_coeffs,
			monomial_ordering_type Monomial_ordering_type,
			std::string &fname, int &nb_pts, long int *&Pts,
			int verbose_level);
	void create_projective_curve(
			std::string &variety_label,
			int curve_nb_vars, int curve_degree,
			std::string &curve_coeffs,
			monomial_ordering_type Monomial_ordering_type,
			std::string &fname, int &nb_pts, long int *&Pts,
			int verbose_level);
	int test_if_vectors_are_projectively_equal(int *v1, int *v2, int len);
	void PG_element_normalize(int *v, int stride, int len);
	// last non-zero element made one
	void PG_element_normalize_from_front(int *v, int stride, int len);
	// first non zero element made one


	void PG_elements_embed(
			long int *set_in, long int *set_out, int sz,
			int old_length, int new_length, int *v);
	long int PG_element_embed(
			long int rk, int old_length, int new_length, int *v);
	void PG_element_unrank_fining(
			int *v, int len, int a);
	int PG_element_rank_fining(
			int *v, int len);
	void PG_element_unrank_gary_cook(
			int *v, int len, int a);
	void PG_element_rank_modified(
			int *v, int stride, int len, int &a);
	void PG_element_unrank_modified(
			int *v, int stride, int len, int a);
	void PG_element_rank_modified_lint(
			int *v, int stride, int len, long int &a);
	void PG_elements_unrank_lint(
			int *M, int k, int n, long int *rank_vec);
	void PG_elements_rank_lint(
			int *M, int k, int n, long int *rank_vec);
	void PG_element_unrank_modified_lint(
			int *v, int stride, int len, long int a);
	void PG_element_rank_modified_not_in_subspace(
			int *v, int stride, int len, int m, long int &a);
	void PG_element_unrank_modified_not_in_subspace(
			int *v, int stride, int len, int m, long int a);

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
	int evaluate_symmetric_form(int len, int *x, int *y);
	int evaluate_quadratic_form_x0x3mx1x2(int *x);
	int is_totally_isotropic_wrt_symplectic_form(int k,
		int n, int *Basis);
	int evaluate_monomial(int *monomial, int *variables, int nb_vars);
	void projective_point_unrank(int n, int *v, int rk);
	long int projective_point_rank(int n, int *v);
	void create_BLT_point(
			int *v5, int a, int b, int c, int verbose_level);
		// creates the point (-b/2,-c,a,-(b^2/4-ac),1)
		// check if it satisfies x_0^2 + x_1x_2 + x_3x_4:
		// b^2/4 + (-c)*a + -(b^2/4-ac)
		// = b^2/4 -ac -b^2/4 + ac = 0
	void Segre_hyperoval(
			long int *&Pts, int &nb_pts, int verbose_level);
	void GlynnI_hyperoval(
			long int *&Pts, int &nb_pts, int verbose_level);
	void GlynnII_hyperoval(
			long int *&Pts, int &nb_pts, int verbose_level);
	void Subiaco_oval(
			long int *&Pts, int &nb_pts, int f_short, int verbose_level);
		// following Payne, Penttila, Pinneri:
		// Isomorphisms Between Subiaco q-Clan Geometries,
		// Bull. Belg. Math. Soc. 2 (1995) 197-222.
		// formula (53)
	void Subiaco_hyperoval(
			long int *&Pts, int &nb_pts, int verbose_level);
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
	void oval_polynomial(
		int *S, unipoly_domain &D, unipoly_object &poly,
		int verbose_level);
	void all_PG_elements_in_subspace(
			int *genma, int k, int n, long int *&point_list, int &nb_points,
			int verbose_level);
	void all_PG_elements_in_subspace_array_is_given(
			int *genma, int k, int n, long int *point_list, int &nb_points,
			int verbose_level);
	void display_all_PG_elements(int n);
	void display_all_PG_elements_not_in_subspace(int n, int m);
	void display_all_AG_elements(int n);
	void do_cone_over(int n,
		long int *set_in, int set_size_in, long int *&set_out, int &set_size_out,
		int verbose_level);
	void do_blocking_set_family_3(int n,
		long int *set_in, int set_size,
		long int *&the_set_out, int &set_size_out,
		int verbose_level);
	void create_hyperoval(
		int f_translation, int translation_exponent,
		int f_Segre, int f_Payne, int f_Cherowitzo, int f_OKeefe_Penttila,
		std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_subiaco_oval(
		int f_short,
		std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_subiaco_hyperoval(
			std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_ovoid(
			std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_Baer_substructure(int n,
		finite_field *Fq,
		std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
		// the big field FQ is given
	void create_BLT_from_database(int f_embedded,
		int BLT_k,
		std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_orthogonal(int epsilon, int n,
			std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_hermitian(int n,
			std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_cuspidal_cubic(
			std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_twisted_cubic(
			std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_elliptic_curve(
		int elliptic_curve_b, int elliptic_curve_c,
		std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_ttp_code(finite_field *Fq,
		int f_construction_A, int f_hyperoval, int f_construction_B,
		std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
		// this is FQ
	void create_unital_XXq_YZq_ZYq(
			std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_whole_space(int n,
			std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_hyperplane(int n,
		int pt,
		std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_segre_variety(int a, int b,
			std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_Maruta_Hamada_arc(
			std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_desarguesian_line_spread_in_PG_3_q(
		finite_field *Fq,
		int f_embedded_in_PG_4_q,
		std::string &fname, int &nb_lines, long int *&Lines,
		int verbose_level);
		// this is FQ
	void do_Klein_correspondence(int n,
			long int *set_in, int set_size,
			long int *&the_set_out, int &set_size_out,
			int verbose_level);
	void do_m_subspace_type(int n, int m,
			long int *set, int set_size,
		int f_show, int verbose_level);
	void do_m_subspace_type_fast(int n, int m,
			long int *set, int set_size,
			int f_show, int verbose_level);
	void do_line_type(int n,
			long int *set, int set_size,
			int f_show, int verbose_level);
	void do_plane_type(int n,
			long int *set, int set_size,
			int *&intersection_type, int &highest_intersection_number,
		int verbose_level);
	void do_plane_type_failsafe(int n,
			long int *set, int set_size,
			int verbose_level);
	void do_conic_type(int n,
			int f_randomized, int nb_times,
			long int *set, int set_size,
			int *&intersection_type, int &highest_intersection_number,
			int verbose_level);
	void do_test_diagonal_line(int n,
			long int *set_in, int set_size,
			std::string &fname_orbits_on_quadrangles,
		int verbose_level);
	void do_andre(finite_field *Fq,
			long int *the_set_in, int set_size_in,
			long int *&the_set_out, int &set_size_out,
			int verbose_level);
		// this is FQ
	void do_print_lines_in_PG(int n, long int *set_in, int set_size);
	void do_print_points_in_PG(int n, long int *set_in, int set_size);
	void do_print_points_in_orthogonal_space(
		int epsilon, int n,
		long int *set_in, int set_size, int verbose_level);
	void do_print_points_on_grassmannian(
		int n, int k,
		long int *set_in, int set_size);
	void do_embed_orthogonal(
		int epsilon, int n,
		long int *set_in, long int *&set_out, int set_size,
		int verbose_level);
	void do_embed_points(int n,
			long int *set_in, long int *&set_out, int set_size,
			int verbose_level);
	void do_draw_points_in_plane(
			layered_graph_draw_options *O,
			long int *set, int set_size,
			std::string &fname_base, int f_point_labels,
			int verbose_level);
	void do_ideal(int n,
			long int *set_in, int set_size, int degree,
			long int *&set_out, int &size_out,
			monomial_ordering_type Monomial_ordering_type,
			int verbose_level);
	void PG_element_modified_not_in_subspace_perm(int n, int m,
		long int *orbit, long int *orbit_inv,
		int verbose_level);
	void print_set_in_affine_plane(int len, long int *S);
	void elliptic_curve_addition(int b, int c,
		int x1, int x2, int x3,
		int y1, int y2, int y3,
		int &z1, int &z2, int &z3, int verbose_level);
	void elliptic_curve_point_multiple(int b, int c, int n,
		int x1, int y1, int z1,
		int &x3, int &y3, int &z3,
		int verbose_level);
	void elliptic_curve_point_multiple_with_log(int b, int c, int n,
		int x1, int y1, int z1,
		int &x3, int &y3, int &z3,
		int verbose_level);
	int elliptic_curve_evaluate_RHS(int x, int b, int c);
	void elliptic_curve_points(
			int b, int c, int &nb, int *&T, int verbose_level);
	void elliptic_curve_all_point_multiples(int b, int c, int &order,
		int x1, int y1, int z1,
		std::vector<std::vector<int> > &Pts,
		int verbose_level);
	int elliptic_curve_discrete_log(int b, int c,
		int x1, int y1, int z1,
		int x3, int y3, int z3,
		int verbose_level);
	void simeon(int n, int len, long int *S, int s, int verbose_level);
	void wedge_to_klein(int *W, int *K);
	void klein_to_wedge(int *K, int *W);
	void isomorphism_to_special_orthogonal(int *A4, int *A6, int verbose_level);
	void minimal_orbit_rep_under_stabilizer_of_frame_characteristic_two(int x, int y,
			int &a, int &b, int verbose_level);
	int evaluate_Fermat_cubic(int *v);



	// #########################################################################
	// finite_field_io.cpp
	// #########################################################################

	void report(std::ostream &ost, int verbose_level);
	void print_minimum_polynomial(int p, const char *polynomial);
	void print();
	void print_detailed(int f_add_mult_table);
	void print_add_mult_tables();
	void print_add_mult_tables_in_C(std::string &fname_base);
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
		int a, int f_exponential, int width, std::string &symbol);
	void print_element_with_symbol_str(std::stringstream &ost,
			int a, int f_exponential, int width, std::string &symbol);
	void int_vec_print_field_elements(std::ostream &ost, int *v, int len);
	void int_vec_print_elements_exponential(std::ostream &ost,
		int *v, int len, std::string &symbol_for_print);
	void make_fname_addition_table_csv(std::string &fname);
	void make_fname_multiplication_table_csv(std::string &fname);
	void make_fname_addition_table_reordered_csv(std::string &fname);
	void make_fname_multiplication_table_reordered_csv(std::string &fname);
	void addition_table_save_csv();
	void multiplication_table_save_csv();
	void addition_table_reordered_save_csv();
	void multiplication_table_reordered_save_csv();
	void latex_addition_table(std::ostream &f,
		int f_elements_exponential, std::string &symbol_for_print);
	void latex_multiplication_table(std::ostream &f,
		int f_elements_exponential, std::string &symbol_for_print);
	void latex_matrix(std::ostream &f, int f_elements_exponential,
			std::string &symbol_for_print, int *M, int m, int n);
	void power_table(int t, int *power_table, int len);
	void cheat_sheet(std::ostream &f, int verbose_level);
	void cheat_sheet_subfields(std::ostream &f, int verbose_level);
	void report_subfields(std::ostream &f, int verbose_level);
	void report_subfields_detailed(std::ostream &ost, int verbose_level);
	void cheat_sheet_addition_table(std::ostream &f, int verbose_level);
	void cheat_sheet_multiplication_table(std::ostream &f, int verbose_level);
	void cheat_sheet_power_table(std::ostream &f, int f_with_polynomials, int verbose_level);
	void cheat_sheet_power_table_top(std::ostream &ost, int f_with_polynomials, int verbose_level);
	void cheat_sheet_power_table_bottom(std::ostream &ost, int f_with_polynomials, int verbose_level);
	void cheat_sheet_table_of_elements(std::ostream &ost, int verbose_level);
	void print_element_as_polynomial(std::ostream &ost, int *v, int verbose_level);
	void cheat_sheet_main_table(std::ostream &f, int verbose_level);
	void cheat_sheet_main_table_top(std::ostream &f, int nb_cols);
	void cheat_sheet_main_table_bottom(std::ostream &f);
	void display_table_of_projective_points(
			std::ostream &ost, long int *Pts, int nb_pts, int len);
	void display_table_of_projective_points2(
		std::ostream &ost, long int *Pts, int nb_pts, int len);
	void display_table_of_projective_points_easy(
		std::ostream &ost, long int *Pts, int nb_pts, int len);
	void export_magma(int d, long int *Pts, int nb_pts, std::string &fname);
	void export_gap(int d, long int *Pts, int nb_pts, std::string &fname);
	void print_matrix_latex(std::ostream &ost, int *A, int m, int n);
	void print_matrix_numerical_latex(std::ostream &ost, int *A, int m, int n);


	// #########################################################################
	// finite_field_RREF.cpp
	// #########################################################################

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
	int Gauss_int_with_given_pivots(int *A,
		int f_special, int f_complete, int *pivots, int nb_pivots,
		int m, int n,
		int verbose_level);
		// A is a m x n matrix
		// returns FALSE if pivot cannot be found at one of the steps
	int RREF_search_pivot(int *A, int m, int n,
			int &i, int &j, int *base_cols, int verbose_level);
	void RREF_make_pivot_one(int *A, int m, int n,
			int &i, int &j, int *base_cols, int verbose_level);
	void RREF_elimination_below(int *A, int m, int n,
			int &i, int &j, int *base_cols, int verbose_level);
	void RREF_elimination_above(int *A, int m, int n,
			int i, int *base_cols, int verbose_level);

};

extern int nb_calls_to_finite_field_init;



// #############################################################################
// finite_field_orthogonal.cpp
// #############################################################################
void orthogonal_points_free_global_data();

// #############################################################################
// finite_field_tables.cpp
// #############################################################################

extern int finitefield_primes[];
extern int finitefield_nb_primes;
extern int finitefield_largest_degree_irreducible_polynomial[];
extern const char *finitefield_primitive_polynomial[][100];

}}


#endif /* SRC_LIB_FOUNDATIONS_FINITE_FIELDS_FINITE_FIELDS_H_ */
