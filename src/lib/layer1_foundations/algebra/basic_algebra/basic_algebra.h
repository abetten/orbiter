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
namespace basic_algebra {

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
	
	void init_integers(
			int verbose_level);
	void init_integer_fractions(
			int verbose_level);
	int as_int(
			int *elt, int verbose_level);
	void make_integer(
			int *elt, int n, int verbose_level);
	void make_zero(
			int *elt, int verbose_level);
	void make_zero_vector(
			int *elt, int len, int verbose_level);
	int is_zero_vector(
			int *elt, int len, int verbose_level);
	int is_zero(
			int *elt, int verbose_level);
	void make_one(
			int *elt, int verbose_level);
	int is_one(
			int *elt, int verbose_level);
	void copy(
			int *elt_from, int *elt_to, int verbose_level);
	void copy_vector(
			int *elt_from, int *elt_to,
		int len, int verbose_level);
	void swap_vector(
			int *elt1, int *elt2, int n, int verbose_level);
	void swap(
			int *elt1, int *elt2, int verbose_level);
	void add(
			int *elt_a, int *elt_b, int *elt_c, int verbose_level);
	void add_apply(
			int *elt_a, int *elt_b, int verbose_level);
	void subtract(
			int *elt_a, int *elt_b, int *elt_c, int verbose_level);
	void negate(
			int *elt, int verbose_level);
	void negate_vector(
			int *elt, int len, int verbose_level);
	void mult(
			int *elt_a, int *elt_b, int *elt_c, int verbose_level);
	void mult_apply(
			int *elt_a, int *elt_b, int verbose_level);
	void power(
			int *elt_a, int *elt_b, int n, int verbose_level);
	void divide(
			int *elt_a, int *elt_b, int *elt_c, int verbose_level);
	void inverse(
			int *elt_a, int *elt_b, int verbose_level);
	void print(
			int *elt);
	void print_vector(
			int *elt, int n);
	void print_matrix(
			int *A, int m, int n);
	void print_matrix_for_maple(
			int *A, int m, int n);
	void make_element_from_integer(
			int *elt, int n, int verbose_level);
	void mult_by_integer(
			int *elt, int n, int verbose_level);
	void divide_by_integer(
			int *elt, int n, int verbose_level);
	int *offset(
			int *A, int i);
	int Gauss_echelon_form(
			int *A, int f_special, int f_complete,
		int *base_cols, 
		int f_P, int *P, int m, int n, int Pn, int verbose_level);
		// returns the rank which is the number 
		// of entries in base_cols
		// A is a m x n matrix,
		// P is a m x Pn matrix (if f_P is true)
	void Gauss_step(
			int *v1, int *v2, int len, int idx, int verbose_level);
		// afterwards: v2[idx] = 0 and v1,v2 span the same space as before
		// v1 is not changed if v1[idx] is nonzero
	void matrix_get_kernel(
			int *M, int m, int n, int *base_cols,
		int nb_base_cols, 
		int &kernel_m, int &kernel_n, int *kernel, int verbose_level);
		// kernel must point to the appropriate amount of memory! 
		// (at least n * (n - nb_base_cols) int's)
		// kernel is stored as column vectors, 
		// i.e. kernel_m = n and kernel_n = n - nb_base_cols.
	void matrix_get_kernel_as_row_vectors(
			int *M, int m, int n,
		int *base_cols, int nb_base_cols, 
		int &kernel_m, int &kernel_n, int *kernel, int verbose_level);
		// kernel must point to the appropriate amount of memory! 
		// (at least n * (n - nb_base_cols) int's)
		// kernel is stored as row vectors, 
		// i.e. kernel_m = n - nb_base_cols and kernel_n = n.
	void get_image_and_kernel(
			int *M, int n, int &rk, int verbose_level);
	void complete_basis(
			int *M, int m, int n, int verbose_level);
	void mult_matrix(
			int *A, int *B, int *C, int ma, int na, int nb,
		int verbose_level);
	void mult_matrix3(
			int *A, int *B, int *C, int *D, int n,
		int verbose_level);
	void add_apply_matrix(
			int *A, int *B, int m, int n,
		int verbose_level);
	void matrix_mult_apply_scalar(
			int *A, int *s, int m, int n,
		int verbose_level);
	void make_block_matrix_2x2(
			int *Mtx, int n, int k,
		int *A, int *B, int *C, int *D, int verbose_level);
		// A is k x k, 
		// B is k x (n - k), 
		// C is (n - k) x k, 
		// D is (n - k) x (n - k), 
		// Mtx is n x n
	void make_identity_matrix(
			int *A, int n, int verbose_level);
	void matrix_inverse(
			int *A, int *Ainv, int n, int verbose_level);
	void matrix_invert(
			int *A, int *T, int *basecols, int *Ainv, int n,
		int verbose_level);

};


// #############################################################################
// algebra_global.cpp
// #############################################################################

//! global functions related to finite fields, irreducible polynomials and such

class algebra_global {
public:

	algebra_global();
	~algebra_global();

	void count_subprimitive(
			int Q_max, int H_max);
	void formula_subprimitive(
			int d, int q,
			ring_theory::longinteger_object &Rdq,
			int &g, int verbose_level);
	void formula(
			int d, int q,
			ring_theory::longinteger_object &Rdq,
			int verbose_level);
	int subprimitive(
			int q, int h);
	int period_of_sequence(
			int *v, int l);
	void subexponent(
			int q, int Q, int h, int f, int j, int k, int &s, int &c);
	std::string plus_minus_string(
			int epsilon);
	void display_all_PHG_elements(
			int n, int q);
	int is_diagonal_matrix(
			int *A, int n);
	int is_lc_of_I_and_J(
			int *A, int n, int &coeff_I, int &coeff_J, int verbose_level);



	void order_of_q_mod_n(
			int q, int n_min, int n_max, int verbose_level);
	void power_function_mod_n(
			int k, int n, int verbose_level);

	void do_trace(
			field_theory::finite_field *F, int verbose_level);
	void do_norm(
			field_theory::finite_field *F, int verbose_level);
	void do_cheat_sheet_GF(
			field_theory::finite_field *F, int verbose_level);
	void export_tables(
			field_theory::finite_field *F, int verbose_level);
	void do_cheat_sheet_ring(
			ring_theory::homogeneous_polynomial_domain *HPD,
			int verbose_level);
	void gl_random_matrix(
			field_theory::finite_field *F, int k, int verbose_level);

	// functions with file based input:
	void apply_Walsh_Hadamard_transform(
			field_theory::finite_field *F,
			std::string &fname_csv_in, int n, int verbose_level);
	void algebraic_normal_form(
			field_theory::finite_field *F,
			int n,
			int *func, int len, int verbose_level);
	void algebraic_normal_form_of_boolean_function(
			field_theory::finite_field *F,
			std::string &fname_csv_in, int n, int verbose_level);
	void apply_trace_function(
			field_theory::finite_field *F,
			std::string &fname_csv_in, int verbose_level);
	void apply_power_function(
			field_theory::finite_field *F,
			std::string &fname_csv_in, long int d, int verbose_level);
	void identity_function(
			field_theory::finite_field *F,
			std::string &fname_csv_out, int verbose_level);
	void Walsh_matrix(
			field_theory::finite_field *F,
			int n, int *&W, int verbose_level);
	void Vandermonde_matrix(
			field_theory::finite_field *F,
			int *&W, int *&W_inv, int verbose_level);

	void O4_isomorphism_4to2(
			field_theory::finite_field *F,
		int *At, int *As, int &f_switch, int *B,
		int verbose_level);
	void O4_isomorphism_2to4(
			field_theory::finite_field *F,
		int *At, int *As, int f_switch, int *B);
	void O4_grid_coordinates_rank(
			field_theory::finite_field *F,
		int x1, int x2, int x3, int x4,
		long int &grid_x, long int &grid_y, int verbose_level);
	void O4_grid_coordinates_unrank(
			field_theory::finite_field *F,
		int &x1, int &x2, int &x3, int &x4, int grid_x,
		int grid_y, int verbose_level);
	void O4_find_tangent_plane(
			field_theory::finite_field *F,
		int pt_x1, int pt_x2, int pt_x3, int pt_x4,
		int *tangent_plane, int verbose_level);
	void create_Nth_roots_and_write_report(
			field_theory::finite_field *F,
			int n, int verbose_level);
	void smith_normal_form(
			int *A, int m, int n, std::string &label, int verbose_level);
	void scan_equation_in_pairs_in_characteristic_p(
			int *eqn, int eqn_sz, int characteristic_p,
			std::string &coefficients_text,
			int verbose_level);

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
	void init(
			field_theory::finite_field *F,
			int n, int verbose_level);
	int count_strong_generators(
			int &nb, int *transversal_length,
		int &first_moved, int depth, int verbose_level);
	int get_strong_generators(
			int *Data, int &nb, int &first_moved,
		int depth, int verbose_level);
	void create_first_candidate_set(
			int verbose_level);
	void create_next_candidate_set(
			int level, int verbose_level);
	int dot_product(
			int *u1, int *u2);
};



// #############################################################################
// group_generators_domain.cpp
// #############################################################################

//! generators for various classes of groups

class group_generators_domain {
public:
	group_generators_domain();
	~group_generators_domain();
	void generators_symmetric_group(
			int deg,
		int &nb_perms, int *&perms, int verbose_level);
	void generators_cyclic_group(
			int deg,
		int &nb_perms, int *&perms, int verbose_level);
	void generators_dihedral_group(
			int deg,
		int &nb_perms, int *&perms, int verbose_level);
	void generators_dihedral_involution(
			int deg,
		int &nb_perms, int *&perms, int verbose_level);
	void generators_identity_group(
			int deg,
		int &nb_perms, int *&perms, int verbose_level);
	void generators_Hall_reflection(
			int nb_pairs,
			int &nb_perms, int *&perms, int &degree,
			int verbose_level);
	void generators_Hall_reflection_normalizer_group(
			int nb_pairs,
			int &nb_perms, int *&perms, int &degree,
			int verbose_level);
	void order_Hall_reflection_normalizer_factorized(
			int nb_pairs,
			int *&factors, int &nb_factors);
	void order_Bn_group_factorized(
			int n,
		int *&factors, int &nb_factors);
	void generators_Bn_group(
			int n, int &deg,
		int &nb_perms, int *&perms, int verbose_level);
	void generators_direct_product(
			int deg1, int nb_perms1, int *perms1,
		int deg2, int nb_perms2, int *perms2,
		int &deg3, int &nb_perms3, int *&perms3,
		int verbose_levels);
	void generators_concatenate(
			int deg1, int nb_perms1, int *perms1,
		int deg2, int nb_perms2, int *perms2,
		int &deg3, int &nb_perms3, int *&perms3,
		int verbose_level);
	int matrix_group_base_len_projective_group(
			int n, int q,
		int f_semilinear, int verbose_level);
	int matrix_group_base_len_affine_group(
			int n, int q,
		int f_semilinear, int verbose_level);
	int matrix_group_base_len_general_linear_group(
			int n, int q,
		int f_semilinear, int verbose_level);
	void order_POmega_epsilon(
			int epsilon, int m, int q,
			algebra::ring_theory::longinteger_object &o, int verbose_level);
	void order_PO_epsilon(
			int f_semilinear, int epsilon, int k, int q,
			algebra::ring_theory::longinteger_object &go, int verbose_level);
	// k is projective dimension
	void order_PO(
			int epsilon, int m, int q,
			algebra::ring_theory::longinteger_object &o,
		int verbose_level);
	void order_Pomega(
			int epsilon, int k, int q,
			algebra::ring_theory::longinteger_object &go,
		int verbose_level);
	void order_PO_plus(
			int m, int q,
			algebra::ring_theory::longinteger_object &o, int verbose_level);
	void order_PO_minus(
			int m, int q,
			ring_theory::longinteger_object &o, int verbose_level);
	// m = Witt index, the dimension is n = 2m+2
	void order_PO_parabolic(
			int m, int q,
			algebra::ring_theory::longinteger_object &o, int verbose_level);
	void order_Pomega_plus(
			int m, int q,
			algebra::ring_theory::longinteger_object &o, int verbose_level);
	// m = Witt index, the dimension is n = 2m
	void order_Pomega_minus(
			int m, int q,
			algebra::ring_theory::longinteger_object &o, int verbose_level);
	// m = half the dimension,
	// the dimension is n = 2m, the Witt index is m - 1
	void order_Pomega_parabolic(
			int m, int q, ring_theory::longinteger_object &o,
		int verbose_level);
	// m = Witt index, the dimension is n = 2m + 1
	int index_POmega_in_PO(
			int epsilon, int m, int q, int verbose_level);


	void diagonal_orbit_perm(
			int n, field_theory::finite_field *F,
			long int *orbit, long int *orbit_inv,
			int verbose_level);
	void frobenius_orbit_perm(
			int n, field_theory::finite_field *F,
		long int *orbit, long int *orbit_inv,
		int verbose_level);
	void projective_matrix_group_base_and_orbits(
			int n, field_theory::finite_field *F,
		int f_semilinear,
		int base_len, int degree,
		long int *base, int *transversal_length,
		long int **orbit, long int **orbit_inv,
		int verbose_level);
	void projective_matrix_group_base_and_transversal_length(
			int n, field_theory::finite_field *F,
		int f_semilinear,
		int base_len, int degree,
		long int *base, int *transversal_length,
		int verbose_level);
	void affine_matrix_group_base_and_transversal_length(
			int n, field_theory::finite_field *F,
		int f_semilinear,
		int base_len, int degree,
		long int *base, int *transversal_length,
		int verbose_level);
	void general_linear_matrix_group_base_and_transversal_length(
			int n, field_theory::finite_field *F,
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
	void builtin_transversal_rep_GLnq(
			int *A, int n,
			field_theory::finite_field *F,
		int f_semilinear, int i, int j, int verbose_level);
	void affine_translation(
			int n, field_theory::finite_field *F,
			int coordinate_idx,
			int field_base_idx, int *perm,
			int verbose_level);
		// perm points to q^n ints
		// field_base_idx is the base element whose
		// translation we compute, 0 \le field_base_idx < e
		// coordinate_idx is the coordinate in which we shift,
		// 0 \le coordinate_idx < n
	void affine_multiplication(
			int n, field_theory::finite_field *F,
		int multiplication_order, int *perm,
		int verbose_level);
		// perm points to q^n ints
		// compute the diagonal multiplication by alpha, i.e.
		// the multiplication by alpha of each component
	void affine_frobenius(
			int n, field_theory::finite_field *F,
			int k, int *perm,
			int verbose_level);
		// perm points to q^n ints
		// compute the diagonal action of the Frobenius
		// automorphism to the power k, i.e.,
		// raises each component to the p^k-th power
	int all_affine_translations_nb_gens(
			int n, field_theory::finite_field *F);
	void all_affine_translations(
			int n, field_theory::finite_field *F, int *gens);
	void affine_generators(
			int n, field_theory::finite_field *F,
			int f_translations,
			int f_semilinear, int frobenius_power,
			int f_multiplication, int multiplication_order,
			int &nb_gens, int &degree, int *&gens,
			int &base_len, long int *&the_base,
			int verbose_level);
	void PG_element_modified_not_in_subspace_perm(
			field_theory::finite_field *F,
			int n, int m,
		long int *orbit, long int *orbit_inv,
		int verbose_level);


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
	void init(
			field_theory::finite_field *F,
			int n, int verbose_level);
	void unrank_element(
			int *Elt, long int rk);
	long int rank_element(
			int *Elt);
	void element_add(
			int *Elt1, int *Elt2, int *Elt3, int verbose_level);
	void element_negate(
			int *Elt1, int *Elt2, int verbose_level);
	int element_add_by_rank(
			int rk_a, int rk_b, int verbose_level);
	int element_negate_by_rank(
			int rk_a, int verbose_level);
	void group_table(
			int *&Table, int verbose_level);
	void group_table_abv(
			int *&Table_abv, int verbose_level);
	void generating_set(
			int *&gens, int &nb_gens, int verbose_level);


};



// #############################################################################
// matrix_group_element.cpp
// #############################################################################

//! implementation of a matrix group over a finite field

class matrix_group_element {

public:

	matrix_group *Matrix_group;

private:

	int *Elt1, *Elt2, *Elt3;
		// used for mult, invert
	int *Elt4;
		// used for invert
	int *Elt5;
	int *tmp_M;
		// used for GL_mult_internal
	int *base_cols;
		// used for Gauss during invert
	int *v1, *v2;
		// temporary vectors of length 2n
	int *v3;
		// used in GL_mult_vector_from_the_left_contragredient
	uchar *elt1, *elt2, *elt3;
		// temporary storage, used in element_store()

	other::data_structures::page_storage *Page_storage;

public:

	matrix_group_element();
	~matrix_group_element();
	void init(
			matrix_group *Matrix_group,
			int verbose_level);
	void allocate_data(
			int verbose_level);
	void free_data(
			int verbose_level);
	void setup_page_storage(
			int page_length_log, int verbose_level);
	int GL_element_entry_ij(
			int *Elt, int i, int j);
	int GL_element_entry_frobenius(
			int *Elt);
	long int image_of_element(
			int *Elt, long int a, int verbose_level);
	long int GL_image_of_PG_element(
			int *Elt, long int a, int verbose_level);
	long int GL_image_of_AG_element(
			int *Elt, long int a, int verbose_level);
	void action_from_the_right_all_types(
		int *v, int *A, int *vA, int verbose_level);
	void projective_action_from_the_right(
		int *v, int *A, int *vA, int verbose_level);
	void general_linear_action_from_the_right(
		int *v, int *A, int *vA, int verbose_level);
	void substitute_surface_equation(
			int *Elt,
			int *coeff_in, int *coeff_out,
			geometry::algebraic_geometry::surface_domain *Surf,
			int verbose_level);
	void GL_one(
			int *Elt);
	void GL_one_internal(
			int *Elt);
	void GL_zero(
			int *Elt);
	int GL_is_one(
			int *Elt);
	void GL_mult(
			int *A, int *B, int *AB, int verbose_level);
	void GL_mult_internal(
			int *A, int *B, int *AB, int verbose_level);
	void GL_copy(
			int *A, int *B);
	void GL_copy_internal(
			int *A, int *B);
	void GL_transpose(
			int *A, int *At, int verbose_level);
	void GL_transpose_only(
			int *A, int *At, int verbose_level);
	// transpose only. no invert
	void GL_transpose_internal(
			int *A, int *At, int verbose_level);
	void GL_invert(
			int *A, int *Ainv);
	void GL_invert_transpose(
			int *A, int *Ainv);
	void GL_invert_internal(
			int *A, int *Ainv, int verbose_level);
	void GL_unpack(
			uchar *elt, int *Elt, int verbose_level);
	void GL_pack(
			int *Elt, uchar *elt, int verbose_level);
	void GL_print_easy(
			int *Elt, std::ostream &ost);
	void GL_code_for_make_element(
			int *Elt, int *data);
	void GL_print_for_make_element(
			int *Elt, std::ostream &ost);
	void GL_print_for_make_element_no_commas(
			int *Elt, std::ostream &ost);
	void GL_print_easy_normalized(
			int *Elt, std::ostream &ost);
	void GL_print_latex(
			int *Elt, std::ostream &ost);
	void GL_print_latex_with_print_point_function(
			int *Elt,
			std::ostream &ost,
			void (*point_label)(
					std::stringstream &sstr, int pt, void *data),
			void *point_label_data);
	void GL_print_easy_latex(
			int *Elt, std::ostream &ost);
	void GL_print_easy_latex_with_option_numerical(
			int *Elt, int f_numerical, std::ostream &ost);
	void decode_matrix(
			int *Elt, int n, unsigned char *elt);
	int get_digit(
			unsigned char *elt, int i, int j);
	int decode_frobenius(
			unsigned char *elt);
	void encode_matrix(
			int *Elt, int n,
			unsigned char *elt, int verbose_level);
	void put_digit(
			unsigned char *elt, int i, int j, int d);
	void encode_frobenius(
			unsigned char *elt, int d);
	void make_element(
			int *Elt, int *data, int verbose_level);
	void make_GL_element(
			int *Elt, int *A, int f);
	int has_shape_of_singer_cycle(
			int *Elt);
	void matrix_minor(
			int *Elt, int *Elt1,
		matrix_group *mtx1, int f, int verbose_level);
	void retrieve(
			int hdl, void *elt, int verbose_level);
	int store(
			void *elt, int verbose_level);
	void dispose(
			int hdl, int verbose_level);
	void print_point(
			long int a, std::ostream &ost, int verbose_level);
	void unrank_point(
			long int rk, int *v, int verbose_level);
	long int rank_point(
			int *v, int verbose_level);

};


// #############################################################################
// matrix_group.cpp
// #############################################################################

//! a matrix group over a finite field in action on a projective space, an affine space, or a vector space

class matrix_group {

public:
	int f_projective;
		// n x n matrices (possibly with Frobenius)
		// acting on PG(n - 1, q)
	int f_affine;
		// n x n matrices plus translations
		// (possibly with Frobenius)
		// acting on F_q^n
	int f_general_linear;
		// n x n matrices (possibly with Frobenius)
		// acting on F_q^n

	int n;
		// the size of the matrices

	int degree;
		// the degree of the action:
		// (q^(n-1)-1) / (q - 1) if f_projective
		// q^n if f_affine or f_general_linear

	int f_semilinear;
		// use Frobenius automorphism

	int f_kernel_is_diagonal_matrices;

	int bits_per_digit;
	int bits_per_elt;
	int bits_extension_degree;
	int char_per_elt;
	int elt_size_int;
	int elt_size_int_half;
	int low_level_point_size; // added Jan 26, 2010
		// = n, the size of the vectors on which we act
	int make_element_size;


	std::string label;
	std::string label_tex;

	int f_GFq_is_allocated;
		// if true, GFq will be destroyed in the destructor
		// if false, it is the responsibility
		// of someone else to destroy GFq

	field_theory::finite_field *GFq;
	void *data;

	linear_algebra::gl_classes *C; // added Dec 2, 2013


	matrix_group_element *Element;


	matrix_group();
	~matrix_group();

	void init_projective_group(
			int n,
			field_theory::finite_field *F,
		int f_semilinear,
		int verbose_level);
	void init_affine_group(
			int n,
			field_theory::finite_field *F,
		int f_semilinear,
		int verbose_level);
	void init_general_linear_group(
			int n,
			field_theory::finite_field *F,
		int f_semilinear,
		int verbose_level);

	void compute_elt_size(
			int verbose_level);
	void init_gl_classes(
			int verbose_level);

	int base_len(
			int verbose_level);
	void base_and_transversal_length(
			int base_len,
			long int *base, int *transversal_length,
			int verbose_level);
	void strong_generators_low_level(
			int *&data,
			int &size, int &nb_gens, int verbose_level);
};



// #############################################################################
// module.cpp:
// #############################################################################

//! a Z module

class module {
public:


	module();
	~module();
	void matrix_multiply_over_Z_low_level(
			int *A1, int *A2, int m1, int n1, int m2, int n2,
			int *A3, int verbose_level);
	void multiply_2by2_from_the_left(
			other::data_structures::int_matrix *M,
			int i, int j,
		int aii, int aij,
		int aji, int ajj, int verbose_level);
	void multiply_2by2_from_the_right(
			other::data_structures::int_matrix *M,
			int i, int j,
		int aii, int aij,
		int aji, int ajj, int verbose_level);
	int clean_column(
			other::data_structures::int_matrix *M,
			other::data_structures::int_matrix *P,
			other::data_structures::int_matrix *Pv,
			int i, int verbose_level);
	int clean_row(
			other::data_structures::int_matrix *M,
			other::data_structures::int_matrix *Q,
			other::data_structures::int_matrix *Qv,
			int i, int verbose_level);
	void smith_normal_form(
			other::data_structures::int_matrix *M,
			other::data_structures::int_matrix *&P,
			other::data_structures::int_matrix *&Pv,
			other::data_structures::int_matrix *&Q,
			other::data_structures::int_matrix *&Qv,
			int verbose_level);
	void apply(
			int *input, int *output, int *perm,
			int module_dimension_m, int module_dimension_n,
			int *module_basis,
			int *v1, int *v2, int verbose_level);

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
	int *Data; // [nb_gens * n * n]
	int *transversal_length;

	null_polarity_generator();
	~null_polarity_generator();
	void init(
			field_theory::finite_field *F,
			int n, int verbose_level);
	int count_strong_generators(
			int &nb, int *transversal_length,
		int &first_moved, int depth, int verbose_level);
	int get_strong_generators(
			int *Data, int &nb, int &first_moved,
		int depth, int verbose_level);
	void backtrack_search(
			int &nb_sol, int depth, int verbose_level);
	void create_first_candidate_set(
			int verbose_level);
	void create_next_candidate_set(
			int level, int verbose_level);
	int dot_product(
			int *u1, int *u2);
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
	void init(
			field_theory::finite_field *GFq,
			int m, int n, int d);
	int check_rank(
			int len, long int *S, int verbose_level);
	int check_rank_matrix_input(
			int len, long int *S, int dim_S,
		int verbose_level);
	int check_rank_last_two_are_fixed(
			int len, long int *S, int verbose_level);
	int compute_rank_row_vectors(
			int len, long int *S, int f_projective,
			int verbose_level);
};





}}}}


#endif /* ORBITER_SRC_LIB_FOUNDATIONS_ALGEBRA_AND_NUMBER_THEORY_H_ */



