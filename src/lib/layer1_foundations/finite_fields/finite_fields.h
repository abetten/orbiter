/*
 * finite_fields.h
 *
 *  Created on: Mar 2, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_FOUNDATIONS_FINITE_FIELDS_FINITE_FIELDS_H_
#define SRC_LIB_FOUNDATIONS_FINITE_FIELDS_FINITE_FIELDS_H_

namespace orbiter {
namespace layer1_foundations {
namespace field_theory {



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

	int f_polynomial_power_mod;
	std::string polynomial_power_mod_A;
	std::string polynomial_power_mod_n;
	std::string polynomial_power_mod_M;

	int f_Berlekamp_matrix;
	std::string Berlekamp_matrix_label;

	int f_normal_basis;
	int normal_basis_d;

	int f_polynomial_find_roots;
	std::string polynomial_find_roots_label;


	int f_normalize_from_the_right;
	int f_normalize_from_the_left;

	int f_nullspace;
	std::string nullspace_input_matrix;

	int f_RREF;
	std::string RREF_input_matrix;


	int f_Walsh_Hadamard_transform;
	std::string Walsh_Hadamard_transform_fname_csv_in;
	int Walsh_Hadamard_transform_n;

	int f_algebraic_normal_form_of_boolean_function;
	std::string algebraic_normal_form_of_boolean_function_fname_csv_in;
	int algebraic_normal_form_of_boolean_function_n;

	int f_algebraic_normal_form;
	int algebraic_normal_form_n;
	std::string algebraic_normal_form_input;

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

	int f_Vandermonde_matrix;

	int f_search_APN_function;

	int f_make_table_of_irreducible_polynomials;
	int make_table_of_irreducible_polynomials_degree;

	int f_EC_Koblitz_encoding;
	std::string EC_message;
	int EC_s;
	// EC_b, EC_c, EC_s, EC_pt_text, EC_message

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
	// EC_b, EC_c, EC_pt_text, EC_discrete_log_pt_text


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

#if 0
	int f_cheat_sheet_Gr;
	int cheat_sheet_Gr_n;
	int cheat_sheet_Gr_k;
#endif

	int f_cheat_sheet_hermitian;
	int cheat_sheet_hermitian_projective_dimension;

	int f_cheat_sheet_desarguesian_spread;
	int cheat_sheet_desarguesian_spread_m;

	int f_sift_polynomials;
	long int sift_polynomials_r0;
	long int sift_polynomials_r1;

	int f_mult_polynomials;
	long int mult_polynomials_r0;
	long int mult_polynomials_r1;

	int f_polynomial_division_ranked;
	long int polynomial_division_r0;
	long int polynomial_division_r1;

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

	int f_inverse_isomorphism_klein_quadric;
	std::string inverse_isomorphism_klein_quadric_matrix_A6;

	// ranking and unranking points in PG:

	int f_rank_point_in_PG;
	std::string rank_point_in_PG_label;

	int f_unrank_point_in_PG;
	int unrank_point_in_PG_n;
	std::string unrank_point_in_PG_text;



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

	int f_product_of;
	std::string product_of_elements;

	int f_sum_of;
	std::string sum_of_elements;

	int f_negate;
	std::string negate_elements;

	int f_inverse;
	std::string inverse_elements;

	int f_power_map;
	int power_map_k;
	std::string power_map_elements;

	int f_evaluate;
	std::string evaluate_formula_label;
	std::string evaluate_parameters;

	finite_field_activity_description();
	~finite_field_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

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
// finite_field_implementation_by_tables.cpp
// #############################################################################

//! implementation of a finite Galois field Fq using tables

class finite_field_implementation_by_tables {


private:

	finite_field *F;

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

	int *v1; // [e]
	int *v2; // [e]
	int *v3; // [e]

	int f_has_quadratic_subfield; // TRUE if e is even.
	int *f_belongs_to_quadratic_subfield; // [q]

	int *reordered_list_of_elements; // [q]
	int *reordered_list_of_elements_inv; // [q]

public:

	finite_field_implementation_by_tables();
	~finite_field_implementation_by_tables();
	void init(finite_field *F, int verbose_level);
	int *private_add_table();
	int *private_mult_table();
	int has_quadratic_subfield();
	int belongs_to_quadratic_subfield(int a);
	void create_alpha_table(int verbose_level);
	void create_alpha_table_prime_field(int verbose_level);
	void create_alpha_table_extension_field(int verbose_level);
	void init_binary_operations(int verbose_level);
	void create_tables_prime_field(int verbose_level);
	void create_tables_extension_field(int verbose_level);
	void print_add_mult_tables(std::ostream &ost);
	void print_add_mult_tables_in_C(std::string &fname_base);
	void init_quadratic_subfield(int verbose_level);
	void init_frobenius_table(int verbose_level);
	void init_absolute_trace_table(int verbose_level);
	void print_tables_extension_field(std::string &poly);
	int add(int i, int j);
	int add_without_table(int i, int j);
	int mult_verbose(int i, int j, int verbose_level);
	int mult_using_discrete_log(int i, int j, int verbose_level);
	int negate(int i);
	int negate_without_table(int i);
	int inverse(int i);
	int inverse_without_table(int i);
	int frobenius_image(int a);
	// computes a^p
	int frobenius_power(int a, int frob_power);
	// computes a^{p^frob_power}
	int alpha_power(int i);
	int log_alpha(int i);
	void addition_table_reordered_save_csv(std::string &fname, int verbose_level);
	void multiplication_table_reordered_save_csv(std::string &fname, int verbose_level);

};


// #############################################################################
// finite_field_implementation_wo_tables.cpp
// #############################################################################

//! implementation of a finite Galois field Fq without any tables

class finite_field_implementation_wo_tables {


private:

	finite_field *F;

	int *v1; // [e]
	int *v2; // [e]
	int *v3; // [e]

	finite_field *GFp;

	ring_theory::unipoly_domain *FX;

	ring_theory::unipoly_object m;

	int factor_polynomial_degree;
	int *factor_polynomial_coefficients_negated;


	ring_theory::unipoly_domain *Fq;

	ring_theory::unipoly_object Alpha;


public:
	finite_field_implementation_wo_tables();
	~finite_field_implementation_wo_tables();
	void init(finite_field *F, int verbose_level);
	int mult(int i, int j, int verbose_level);
	int inverse(int i, int verbose_level);
	int negate(int i, int verbose_level);
	int add(int i, int j, int verbose_level);

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

	int f_without_tables;

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
	finite_field_implementation_by_tables *T;

	finite_field_implementation_wo_tables *Iwo;


	std::string symbol_for_print;


	int nb_times_mult;
	int nb_times_add;

public:
	int f_has_table;
		// if TRUE, T is available, otherwise Iwo is available.

	std::string label;
	std::string label_tex;
	std::string override_poly;
	std::string my_poly;
	//char *polynomial;
		// the actual polynomial we consider
		// as integer (in text form)
	int f_is_prime_field;
	int q, p, e;
	int alpha; // primitive element
	int log10_of_q; // needed for printing purposes
	int f_print_as_exponentials;
	long int nb_calls_to_mult_matrix_matrix;
	long int nb_calls_to_PG_element_rank_modified;
	long int nb_calls_to_PG_element_unrank_modified;

	linear_algebra::linear_algebra *Linear_algebra;
	orthogonal_geometry::orthogonal_indexing *Orthogonal_indexing;


	finite_field();
	~finite_field();
	void print_call_stats(std::ostream &ost);
	void init(finite_field_description *Descr, int verbose_level);
	void finite_field_init(int q, int f_without_tables, int verbose_level);
	void init_implementation(int f_without_tables, int verbose_level);
	void set_default_symbol_for_print();
	void init_symbol_for_print(const char *symbol);
	void init_override_polynomial(int q, std::string &poly,
		int f_without_tables, int verbose_level);
	int has_quadratic_subfield();
	int belongs_to_quadratic_subfield(int a);
	long int compute_subfield_polynomial(int order_subfield,
			int f_latex, std::ostream &ost,
			int verbose_level);
	void compute_subfields(int verbose_level);
	int find_primitive_element(int verbose_level);
	int compute_order_of_element(int elt, int verbose_level);
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
	int is_square(int i);
	int square_root(int i);
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
	void compute_nth_roots(int *&Nth_roots, int n, int verbose_level);
	int primitive_element();



	// #########################################################################
	// finite_field_projective.cpp
	// #########################################################################

	void PG_element_apply_frobenius(int n, int *v, int f);
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

	void projective_point_unrank(int n, int *v, int rk);
	long int projective_point_rank(int n, int *v);
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
		long int *set_in, int set_size_in,
		long int *&set_out, int &set_size_out,
		int verbose_level);
	void do_blocking_set_family_3(int n,
		long int *set_in, int set_size,
		long int *&the_set_out, int &set_size_out,
		int verbose_level);
	// creates projective_space PG(n,q)
	void create_orthogonal(int epsilon, int n,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_hermitian(int n,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
		int verbose_level);
	// creates hermitian
	void create_ttp_code(finite_field *Fq_subfield,
		int f_construction_A, int f_hyperoval, int f_construction_B,
		std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
		// this is FQ
	void create_segre_variety(int a, int b,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
		int verbose_level);
	// The Segre map goes from PG(a,q) cross PG(b,q) to PG((a+1)*(b+1)-1,q)
	void do_andre(finite_field *Fq,
			long int *the_set_in, int set_size_in,
			long int *&the_set_out, int &set_size_out,
			int verbose_level);
	// creates PG(2,Q) and PG(4,q)
	// this is FQ
	void do_embed_orthogonal(
		int epsilon, int n,
		long int *set_in, long int *&set_out, int set_size,
		int verbose_level);
	void do_embed_points(int n,
			long int *set_in, long int *&set_out, int set_size,
			int verbose_level);
	void print_set_in_affine_plane(int len, long int *S);
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
	void print_minimum_polynomial(int p, std::string &polynomial);
	void print();
	void print_detailed(int f_add_mult_table);
	void print_tables();
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
	void addition_table_save_csv(int verbose_level);
	void multiplication_table_save_csv(int verbose_level);
	void addition_table_reordered_save_csv(int verbose_level);
	void multiplication_table_reordered_save_csv(int verbose_level);
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
	void cheat_sheet_power_table(std::ostream &f,
			int f_with_polynomials, int verbose_level);
	void cheat_sheet_power_table_top(std::ostream &ost,
			int f_with_polynomials, int verbose_level);
	void cheat_sheet_power_table_bottom(std::ostream &ost,
			int f_with_polynomials, int verbose_level);
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
	void read_from_string_coefficient_vector(std::string &str,
			int *&coeff, int &len,
			int verbose_level);



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
	void init(orthogonal_geometry::unusual_model &U, int verbose_level);
	int choose_an_element_of_given_norm(int norm, int verbose_level);

};


// #############################################################################
// nth_roots.cpp:
// #############################################################################

//! the nth roots over Fq using an extension field

class nth_roots {
public:

	int n;
	finite_field *F; // F_q where q = p^e
	ring_theory::unipoly_object *Beta;
	ring_theory::unipoly_object *Fq_Elements;

	ring_theory::unipoly_object Min_poly;
		// Min_poly = irreducible polynomial
		// over F->p of degree field_degree = e * m

	finite_field *Fp; // the prime field F_p

	ring_theory::unipoly_domain *FpX;

	ring_theory::unipoly_domain *FQ;
		// polynomial ring F_p modulo Min_poly

	ring_theory::unipoly_domain *FX;

	int m, r, field_degree;
		// m is the order of q modulo n
		// field_degree = e * m

	ring_theory::longinteger_object *Qm, *Qm1, *Index, *Subfield_Index;
		// Qm = q^m = p^(e*m)
		// Qm1 = q^m - 1
		// Index = Qm1 / n
		// Subfield_Index = Qm1 / (q - 1)

	number_theory::cyclotomic_sets *Cyc;
	ring_theory::unipoly_object **min_poly_beta_FQ;
	// polynomials whose coefficients are again polynomials,
	// representing field elements in FQ
	//ring_theory::unipoly_object **generator;

	ring_theory::unipoly_object *min_poly_beta_Fq;
	// polynomials whose coefficients are integers in Fq
	//ring_theory::unipoly_object *generator_Fq;

	int subfield_degree;
	int *subfield_basis; // [subfield_degree * field_degree]





	nth_roots();
	~nth_roots();
	void init(finite_field *F, int n, int verbose_level);
	void compute_subfield(int subfield_degree,
			int *&field_basis, int verbose_level);
	void report(std::ostream &ost, int verbose_level);

};



// #############################################################################
// square_nonsquare.cpp:
// #############################################################################

//! keeping track of squares and nonsquares

class square_nonsquare {
public:

	finite_field *F;

	int *minus_squares; // [(q-1)/2]
	int *minus_squares_without; // [(q-1)/2 - 1]
	int *minus_nonsquares; // [(q-1)/2]
	int *f_is_minus_square; // [q]
	int *index_minus_square; // [q]
	int *index_minus_square_without; // [q]
	int *index_minus_nonsquare; // [q]

	square_nonsquare();
	~square_nonsquare();
	void init(field_theory::finite_field *F, int verbose_level);
	int is_minus_square(int i);
	void print_minus_square_tables();

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



}}}



#endif /* SRC_LIB_FOUNDATIONS_FINITE_FIELDS_FINITE_FIELDS_H_ */
