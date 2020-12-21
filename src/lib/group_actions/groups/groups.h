// group_theory.h
//
// Anton Betten
//
// moved here from action.h: July 28, 2018
// based on action.h which was started:  August 13, 2005



#ifndef ORBITER_SRC_LIB_GROUP_ACTIONS_GROUPS_GROUPS_H_
#define ORBITER_SRC_LIB_GROUP_ACTIONS_GROUPS_GROUPS_H_


namespace orbiter {

namespace group_actions {



// #############################################################################
// direct_product.cpp
// #############################################################################

//! the direct product of two matrix groups in product action

class direct_product {

public:
	matrix_group *M1;
	matrix_group *M2;
	finite_field *F1;
	finite_field *F2;
	int q1;
	int q2;

	std::string label;
	std::string label_tex;


	int degree_of_matrix_group1;
	int dimension_of_matrix_group1;
	int degree_of_matrix_group2;
	int dimension_of_matrix_group2;
	int degree_of_product_action;
	int degree_overall;
	int low_level_point_size;
	int make_element_size;
	int elt_size_int;

	int *perm_offset_i;
	int *tmp_Elt1;

	int bits_per_digit1;
	int bits_per_digit2;

	int bits_per_elt;
	int char_per_elt;

	uchar *elt1;

	int base_len_in_component1;
	long int *base_for_component1;
	int *tl_for_component1;

	int base_len_in_component2;
	long int *base_for_component2;
	int *tl_for_component2;

	int base_length;
	long int *the_base;
	int *the_transversal_length;

	page_storage *Elts;

	direct_product();
	~direct_product();
	void null();
	void freeself();
	void init(matrix_group *M1, matrix_group *M2,
			int verbose_level);
	long int element_image_of(int *Elt, long int a, int verbose_level);
	void element_one(int *Elt);
	int element_is_one(int *Elt);
	void element_mult(int *A, int *B, int *AB, int verbose_level);
	void element_move(int *A, int *B, int verbose_level);
	void element_invert(int *A, int *Av, int verbose_level);
	int offset_i(int i);
	void element_pack(int *Elt, uchar *elt);
	void element_unpack(uchar *elt, int *Elt);
	void put_digit(uchar *elt, int f, int i, int d);
	int get_digit(uchar *elt, int f, int i);
	void make_element(int *Elt, int *data, int verbose_level);
	void element_print_easy(int *Elt, std::ostream &ost);
	void compute_base_and_transversals(int verbose_level);
	void make_strong_generators_data(int *&data,
			int &size, int &nb_gens, int verbose_level);
	void lift_generators(
			strong_generators *SG1,
			strong_generators *SG2,
			action *A, strong_generators *&SG3,
			int verbose_level);
};

// #############################################################################
// exceptional_isomorphism_O4.cpp
// #############################################################################

//! exceptional isomorphism between orthogonal groups: O4, O5 and GL(2,q)

class exceptional_isomorphism_O4 {
public:
	finite_field *Fq;
	action *A2;
	action *A4;
	action *A5;

	int *E5a;
	int *E4a;
	int *E2a;
	int *E2b;

	exceptional_isomorphism_O4();
	~exceptional_isomorphism_O4();
	void null();
	void freeself();
	void init(finite_field *Fq,
			action *A2, action *A4, action *A5,
			int verbose_level);
	void apply_2to4_embedded(
		int f_switch, int *mtx2x2_T, int *mtx2x2_S, int *Elt,
		int verbose_level);
	void apply_5_to_4(
		int *mtx4x4, int *mtx5x5, int verbose_level);
	void apply_4_to_5(
		int *E4, int *E5, int verbose_level);
	void apply_4_to_2(
		int *E4, int &f_switch, int *E2_a, int *E2_b,
		int verbose_level);
	void apply_2_to_4(
		int &f_switch, int *E2_a, int *E2_b, int *E4,
		int verbose_level);
	void print_as_2x2(int *mtx4x4);
};

// #############################################################################
// linear_group_description.cpp
// #############################################################################

//! description of a linear group from the command line


class linear_group_description {
public:
	int f_projective;
	int f_general;
	int f_affine;
	int f_GL_d_q_wr_Sym_n;
	int GL_wreath_Sym_d;
	int GL_wreath_Sym_n;


	int n;

	// change input_q to string so that we can allow symbols:
	//int input_q;
	std::string input_q;

	int f_override_polynomial;
	std::string override_polynomial;
	finite_field *F;
	int f_semilinear;
	int f_special;

	int f_wedge_action;
	int f_wedge_action_detached;
	int f_PGL2OnConic;
	int f_monomial_group;
	int f_diagonal_group;
	int f_null_polarity_group;
	int f_symplectic_group;
	int f_singer_group;
	int f_singer_group_and_frobenius;
	int singer_power;
	int f_subfield_structure_action;
	int s;
	int f_subgroup_from_file;
	int f_borel_subgroup_upper;
	int f_borel_subgroup_lower;
	int f_identity_group;
	std::string subgroup_fname;
	std::string subgroup_label;
	int f_orthogonal_group;
	int orthogonal_group_epsilon;

	int f_on_k_subspaces;
	int on_k_subspaces_k;

	int f_on_tensors;
	int f_on_rank_one_tensors;

	int f_subgroup_by_generators;
	std::string subgroup_order_text;
	int nb_subgroup_generators;
	std::string *subgroup_generators_as_string;

	int f_Janko1;

	int f_restricted_action;
	std::string restricted_action_text;

	int f_export_magma;


	linear_group_description();
	~linear_group_description();
	void null();
	void freeself();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
};



// #############################################################################
// linear_group.cpp
// #############################################################################

//! creates a linear group from command line arguments using linear_group_description

class linear_group {
public:
	linear_group_description *description;
	int n;
	int input_q;
	finite_field *F;
	int f_semilinear;

	std::string label;
	std::string label_tex;

	strong_generators *initial_strong_gens;
	action *A_linear;
	matrix_group *Mtx;

	int f_has_strong_generators;
	strong_generators *Strong_gens;
	action *A2;
	int vector_space_dimension;
	int q;

	int f_has_nice_gens;
	vector_ge *nice_gens;

	linear_group();
	~linear_group();
	void null();
	void freeself();
	void linear_group_init(linear_group_description *description,
		int verbose_level);
	void init_PGL2q_OnConic(int verbose_level);
	void init_wedge_action(int verbose_level);
	void init_wedge_action_detached(int verbose_level);
	void init_monomial_group(int verbose_level);
	void init_diagonal_group(int verbose_level);
	void init_singer_group(int singer_power, int verbose_level);
	void init_singer_group_and_frobenius(int singer_power, int verbose_level);
	void init_null_polarity_group(int verbose_level);
	void init_borel_subgroup_upper(int verbose_level);
	void init_identity_subgroup(int verbose_level);
	void init_symplectic_group(int verbose_level);
	void init_subfield_structure_action(int s, int verbose_level);
	void init_orthogonal_group(int epsilon, int verbose_level);
	void init_subgroup_from_file(
			std::string &subgroup_fname,
			std::string &subgroup_label,
			int verbose_level);
	void init_subgroup_by_generators(
			std::string &subgroup_label,
			std::string &subgroup_order_text,
			int nb_subgroup_generators,
			std::string *subgroup_generators_as_string,
			int verbose_level);
	void init_subgroup_Janko1(int verbose_level);
	void report(std::ostream &fp, int f_sylow, int f_group_table,
			int f_conjugacy_classes_and_normalizers,
			layered_graph_draw_options *LG_Draw_options,
			int verbose_level);
	void create_latex_report(
			layered_graph_draw_options *O,
			int f_sylow, int f_group_table, int f_classes,
			int verbose_level);

};




// #############################################################################
// matrix_group.cpp
// #############################################################################

//! a matrix group over a finite field in projective, vector space or affine action

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
		// if TRUE, GFq will be destroyed in the destructor
		// if FALSE, it is the responsibility
		// of someone else to destroy GFq
	
	finite_field *GFq;
	void *data;

	gl_classes *C; // added Dec 2, 2013

	
	// temporary variables, do not use!
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
	
	page_storage *Elts;
	

	matrix_group();
	~matrix_group();
	void null();
	void freeself();
	
	void init_projective_group(int n, finite_field *F, 
		int f_semilinear, action *A, int verbose_level);
	void init_affine_group(int n, finite_field *F, 
		int f_semilinear, action *A, int verbose_level);
	void init_general_linear_group(int n, finite_field *F, 
		int f_semilinear, action *A, int verbose_level);
	void allocate_data(int verbose_level);
	void free_data(int verbose_level);
	void setup_page_storage(int page_length_log, int verbose_level);
	void compute_elt_size(int verbose_level);
	void init_base(action *A, int verbose_level);
	void init_base_projective(action *A, int verbose_level);
		// initializes base, base_len, degree,
		// transversal_length, orbit, orbit_inv
	void init_base_affine(action *A, int verbose_level);
	void init_base_general_linear(action *A, int verbose_level);
	void init_gl_classes(int verbose_level);

	int GL_element_entry_ij(int *Elt, int i, int j);
	int GL_element_entry_frobenius(int *Elt);
	long int image_of_element(int *Elt, long int a, int verbose_level);
	long int GL_image_of_PG_element(int *Elt, long int a, int verbose_level);
	long int GL_image_of_AG_element(int *Elt, long int a, int verbose_level);
	void action_from_the_right_all_types(
		int *v, int *A, int *vA, int verbose_level);
	void projective_action_from_the_right(
		int *v, int *A, int *vA, int verbose_level);
	void general_linear_action_from_the_right(
		int *v, int *A, int *vA, int verbose_level);
	void substitute_surface_equation(int *Elt,
			int *coeff_in, int *coeff_out, surface_domain *Surf,
			int verbose_level);
	void GL_one(int *Elt);
	void GL_one_internal(int *Elt);
	void GL_zero(int *Elt);
	int GL_is_one(int *Elt);
	void GL_mult(int *A, int *B, int *AB, int verbose_level);
	void GL_mult_internal(int *A, int *B, int *AB, int verbose_level);
	void GL_copy(int *A, int *B);
	void GL_copy_internal(int *A, int *B);
	void GL_transpose(int *A, int *At, int verbose_level);
	void GL_transpose_internal(int *A, int *At, int verbose_level);
	void GL_invert(int *A, int *Ainv);
	void GL_invert_internal(int *A, int *Ainv, int verbose_level);
	void GL_unpack(uchar *elt, int *Elt, int verbose_level);
	void GL_pack(int *Elt, uchar *elt);
	void GL_print_easy(int *Elt, std::ostream &ost);
	void GL_code_for_make_element(int *Elt, int *data);
	void GL_print_for_make_element(int *Elt, std::ostream &ost);
	void GL_print_for_make_element_no_commas(int *Elt, std::ostream &ost);
	void GL_print_easy_normalized(int *Elt, std::ostream &ost);
	void GL_print_latex(int *Elt, std::ostream &ost);
	void GL_print_latex_with_print_point_function(int *Elt,
			std::ostream &ost,
			void (*point_label)(std::stringstream &sstr, int pt, void *data),
			void *point_label_data);
	void GL_print_easy_latex(int *Elt, std::ostream &ost);
	int get_digit(uchar *elt, int i, int j);
	int get_digit_frobenius(uchar *elt);
	void put_digit(uchar *elt, int i, int j, int d);
	void put_digit_frobenius(uchar *elt, int d);
	void make_element(int *Elt, int *data, int verbose_level);
	void make_GL_element(int *Elt, int *A, int f);
	void orthogonal_group_random_generator(
		action *A, orthogonal *O,
		int f_siegel, 
		int f_reflection, 
		int f_similarity,
		int f_semisimilarity, 
		int *Elt, int verbose_level);
	void matrices_without_eigenvector_one(
		sims *S, int *&Sol, int &cnt,
		int f_path_select, int select_value, 
		int verbose_level);
	void matrix_minor(int *Elt, int *Elt1, 
		matrix_group *mtx1, int f, int verbose_level);
	int base_len(int verbose_level);
	void base_and_transversal_length(
			int base_len,
			long int *base, int *transversal_length,
			int verbose_level);
	void strong_generators_low_level(int *&data,
			int &size, int &nb_gens, int verbose_level);
	int has_shape_of_singer_cycle(int *Elt);
};

// #############################################################################
// orbits_on_something.cpp
// #############################################################################

//! compute orbits of a group in a given action; allows file io

class orbits_on_something {

public:

	action *A;
	strong_generators *SG;
	schreier *Sch;

	int f_load_save;
	std::string prefix;
	std::string fname;

	tally *Classify_orbits_by_length;
	set_of_sets *Orbits_classified;

	int *Orbits_classified_length; // [Orbits_classified_nb_types]
	int Orbits_classified_nb_types;

	orbits_on_something();
	~orbits_on_something();
	void null();
	void freeself();
	void init(
			action *A,
			strong_generators *SG,
			int f_load_save,
			std::string &prefix,
			int verbose_level);
	void idx_of_points_in_orbits_of_length_l(
			long int *set, int set_sz, int go, int l,
			std::vector<int> &Idx,
			int verbose_level);
	void orbit_type_of_set(
			long int *set, int set_sz, int go,
			long int *orbit_type,
			int verbose_level);
	// orbit_type[(go + 1) * go] must be allocated beforehand
	void report_type(std::ostream &ost, long int *orbit_type, long int goi);
	void compute_compact_type(long int *orbit_type, long int goi,
			long int *&compact_type, long int *&row_labels, long int *&col_labels, int &m, int &n);
	void report_orbit_lengths(std::ostream &ost);
	void classify_orbits_by_length(int verbose_level);
	void report_classified_orbit_lengths(std::ostream &ost);
	int get_orbit_type_index(int orbit_length);
	int get_orbit_type_index_if_present(int orbit_length);
	void test_orbits_of_a_certain_length(
		int orbit_length,
		int &type_idx,
		int &prev_nb,
		int (*test_function)(long int *orbit, int orbit_length, void *data),
		void *test_function_data,
		int verbose_level);
	void create_graph_on_orbits_of_a_certain_length(
		colored_graph *&CG,
		std::string &fname,
		int orbit_length,
		int &type_idx,
		int f_has_user_data, long int *user_data, int user_data_size,
		int f_has_colors, int number_colors, int *color_table,
		int (*test_function)(long int *orbit1, int orbit_length1, long int *orbit2, int orbit_length2, void *data),
		void *test_function_data,
		int verbose_level);
	void extract_orbits(
		int orbit_length,
		int nb_orbits,
		int *orbits,
		long int *extracted_set,
		set_of_sets *my_orbits_classified,
		int verbose_level);
	void create_graph_on_orbits_of_a_certain_length_override_orbits_classified(
		colored_graph *&CG,
		std::string &fname,
		int orbit_length,
		int &type_idx,
		int f_has_user_data, long int *user_data, int user_data_size,
		int (*test_function)(long int *orbit1, int orbit_length1, long int *orbit2, int orbit_length2, void *data),
		void *test_function_data,
		set_of_sets *my_orbits_classified,
		int verbose_level);
	void compute_orbit_invariant_after_classification(
			set_of_sets *&Orbit_invariant,
			int (*evaluate_orbit_invariant_function)(int a, int i, int j, void *evaluate_data, int verbose_level),
			void *evaluate_data, int verbose_level);
	void create_latex_report(int verbose_level);
	void report(std::ostream &ost, int verbose_level);

};


// #############################################################################
// permutation_representation_domain.cpp
// #############################################################################

//! a domain for permutation groups whose elements are given in the permutation representation

class permutation_representation_domain {

public:
	int degree;
	
	int f_induced_action;
	
	
	int f_product_action;
	int m;
	int n;
	int mn;
	int offset;
	
	int char_per_elt;
	int elt_size_int;
	
	int *Elt1, *Elt2, *Elt3, *Elt4;
	uchar *elt1, *elt2, *elt3;
		// temporary storage, used in element_store()
	int *Eltrk1, *Eltrk2, *Eltrk3;
		// used in store / retrieve
	
	page_storage *Elts;

	permutation_representation_domain();
	~permutation_representation_domain();
	void null();
	void free();
	void allocate();
	void init_product_action(int m, int n, 
		int page_length_log, int verbose_level);
	void init(int degree, int page_length_log, int verbose_level);
	void init_data(int page_length_log, int verbose_level);
	void init_with_base(int degree, 
		int base_length, int *base, int page_length_log, 
		action &A, int verbose_level);
	void transversal_rep(int i, int j, int *Elt, int verbose_level);
	void one(int *Elt);
	int is_one(int *Elt);
	void mult(int *A, int *B, int *AB);
	void copy(int *A, int *B);
	void invert(int *A, int *Ainv);
	void unpack(uchar *elt, int *Elt);
	void pack(int *Elt, uchar *elt);
	void print(int *Elt, std::ostream &ost);
	void print_with_print_point_function(int *Elt,
			std::ostream &ost,
			void (*point_label)(std::stringstream &sstr, long int pt, void *data),
			void *point_label_data);
	void code_for_make_element(int *Elt, int *data);
	void print_for_make_element(int *Elt, std::ostream &ost);
	void print_for_make_element_no_commas(int *Elt, std::ostream &ost);
	void print_with_action(action *A, int *Elt, std::ostream &ost);
	void make_element(int *Elt, int *data, int verbose_level);

};


// #############################################################################
// permutation_representation.cpp
// #############################################################################

//! homomorphism to a permutation group

class permutation_representation {

public:
	action *A_original;
	int f_stay_in_the_old_action;
	int nb_gens;
	vector_ge *gens; // the original generators in action A_original
	int *Perms; // [nb_gens * degree]
	int degree;
	//longinteger_object target_go;

	permutation_representation_domain *P;
	int perm_offset;
	int elt_size_int;
		// A_original->elt_size_int + P->elt_size_int
	int make_element_size;

	int char_per_elt;
	uchar *elt1; // [char_per_elt]


	std::string label;
	std::string label_tex;

	page_storage *PS;

	int *Elts;
		// [nb_gens * elt_size_int], the generators in the induced action


	permutation_representation();
	~permutation_representation();
	void init(action *A_original,
			int f_stay_in_the_old_action,
			vector_ge *gens,
			int *Perms, int degree,
			int verbose_level);
		// Perms is degree x nb_gens
	long int element_image_of(int *Elt, long int a, int verbose_level);
	void element_one(int *Elt);
	int element_is_one(int *Elt);
	void element_mult(int *A, int *B, int *AB, int verbose_level);
	void element_move(int *A, int *B, int verbose_level);
	void element_invert(int *A, int *Av, int verbose_level);
	void element_pack(int *Elt, uchar *elt);
	void element_unpack(uchar *elt, int *Elt);
	void element_print_for_make_element(int *Elt, std::ostream &ost);
	void element_print_easy(int *Elt, std::ostream &ost);
	void element_print_latex(int *Elt, std::ostream &ost);
};



// #############################################################################
// schreier.cpp
// #############################################################################

//! Schreier trees for orbits of groups on points

class schreier {

public:
	action *A;
	int f_images_only;
	long int degree;
	vector_ge gens;
	vector_ge gens_inv;
	int nb_images;
	int **images;
		// [nb_gens][2 * A->degree], 
		// allocated by init_images, 
		// called from init_generators
		// for each generator,
		// stores the generator as permutation in 0..A->degree-1 ,
		// then the inverse generator in A->degree..2*A->degree-1
	
	int *orbit; // [A->degree]
	int *orbit_inv; // [A->degree]

		// prev and label are indexed
		// by the points in the order as listed in orbit.
	int *prev; // [A->degree]
	int *label; // [A->degree]
	//int *orbit_no; // [A->degree]
		// to find out which orbit point a lies in, 
		// use orbit_number(pt).
		// It used to be orbit_no[orbit_inv[a]]

	int *orbit_first;  // [A->degree + 1]
	int *orbit_len;  // [A->degree]
	int nb_orbits;
	
	int *Elt1, *Elt2, *Elt3;
	int *schreier_gen, *schreier_gen1;
		// used in random_schreier_generator
	int *cosetrep, *cosetrep_tmp;
		// used in coset_rep / coset_rep_inv
	
	int f_print_function;
	void (*print_function)(std::ostream &ost, int pt, void *data);
	void *print_function_data;

	schreier();
	schreier(action *A, int verbose_level);
	~schreier();
	void freeself();
	void delete_images();
	void init_images(int nb_images, int verbose_level);
	void init_images_only(int nb_images,
			long int degree, int *images, int verbose_level);
	void images_append(int verbose_level);
	void init(action *A, int verbose_level);
	void allocate_tables();
	void init2();
	void initialize_tables();
	void init_single_generator(int *elt, int verbose_level);
	void init_generators(vector_ge &generators, int verbose_level);
	void init_images_recycle(int nb_images,
			int **old_images,
			int idx_deleted_generator,
			int verbose_level);
	void init_images_recycle(int nb_images,
			int **old_images, int verbose_level);
	void init_generators(int nb, int *elt, int verbose_level);
	void init_generators_recycle_images(
			vector_ge &generators, int **old_images,
			int idx_generator_to_delete, int verbose_level);
	void init_generators_recycle_images(
			vector_ge &generators, int **old_images, int verbose_level);


		// elt must point to nb * A->elt_size_in_int 
		// int's that are 
		// group elements in int format
	void init_generators_recycle_images(int nb,
			int *elt, int **old_images,
			int idx_generator_to_delete, int verbose_level);
	void init_generators_recycle_images(int nb,
			int *elt, int **old_images, int verbose_level);
	void init_generators_by_hdl(int nb_gen, int *gen_hdl, 
		int verbose_level);
	void init_generators_by_handle(std::vector<int> &gen_hdl, int verbose_level);
	long int get_image(long int i, int gen_idx, int verbose_level);
	void swap_points(int i, int j, int verbose_level);
	void move_point_here(int here, int pt);
	int orbit_representative(int pt);
	int depth_in_tree(int j);
		// j is a coset, not a point
	void transporter_from_orbit_rep_to_point(int pt, 
		int &orbit_idx, int *Elt, int verbose_level);
	void transporter_from_point_to_orbit_rep(int pt, 
		int &orbit_idx, int *Elt, int verbose_level);
	void coset_rep(int j, int verbose_level);
		// j is a coset, not a point
		// result is in cosetrep
		// determines an element in the group 
		// that moves the orbit representative 
		// to the j-th point in the orbit.
	void coset_rep_with_verbosity(int j, int verbose_level);
	void coset_rep_inv(int j, int verbose_level);
	void extend_orbit(int *elt, int verbose_level);
	void compute_all_point_orbits(int verbose_level);
	void compute_all_point_orbits_with_prefered_reps(
		int *prefered_reps, int nb_prefered_reps, 
		int verbose_level);
	void compute_all_point_orbits_with_preferred_labels(
		long int *preferred_labels, int verbose_level);
	void compute_all_orbits_on_invariant_subset(int len, 
		long int *subset, int verbose_level);
	void compute_all_orbits_on_invariant_subset_lint(
		int len, long int *subset, int verbose_level);
	void compute_point_orbit(int pt, int verbose_level);
	void compute_point_orbit_with_limited_depth(
			int pt, int max_depth, int verbose_level);
	int sum_up_orbit_lengths();
	void non_trivial_random_schreier_generator(action *A_original, 
			int *Elt, int verbose_level);
		// computes non trivial random Schreier 
		// generator into schreier_gen
		// non-trivial is with respect to A_original
	void random_schreier_generator_ith_orbit(
			int *Elt, int orbit_no,
			int verbose_level);
		// computes random Schreier generator for the orbit orbit_no into Elt
	void random_schreier_generator(int *Elt, int verbose_level);
		// computes random Schreier generator for the first orbit into Elt
	void trace_back(int *path, int i, int &j);
	void intersection_vector(int *set, int len, 
		int *intersection_cnt);
	void orbits_on_invariant_subset_fast(int len, 
		int *subset, int verbose_level);
	void orbits_on_invariant_subset_fast_lint(
		int len, long int *subset, int verbose_level);
	void orbits_on_invariant_subset(int len, int *subset, 
		int &nb_orbits_on_subset, int *&orbit_perm, int *&orbit_perm_inv);
	void get_orbit_partition_of_points_and_lines(
		partitionstack &S, int verbose_level);
	void get_orbit_partition(partitionstack &S, 
		int verbose_level);
	void get_orbit_in_order(std::vector<int> &Orb,
		int orbit_idx, int verbose_level);
	strong_generators *stabilizer_any_point_plus_cosets(
		action *default_action, 
		longinteger_object &full_group_order, 
		int pt, vector_ge *&cosets, int verbose_level);
	strong_generators *stabilizer_any_point(
		action *default_action, 
		longinteger_object &full_group_order, int pt, 
		int verbose_level);
	set_and_stabilizer *get_orbit_rep(action *default_action,
			longinteger_object &full_group_order,
			int orbit_idx, int verbose_level);
	void get_orbit_rep_to(action *default_action,
			longinteger_object &full_group_order,
			int orbit_idx,
			set_and_stabilizer *Rep,
			int verbose_level);
	strong_generators *stabilizer_orbit_rep(
		action *default_action, 
		longinteger_object &full_group_order, 
		int orbit_idx, int verbose_level);
	void point_stabilizer(action *default_action, 
		longinteger_object &go, 
		sims *&Stab, int orbit_no, int verbose_level);
		// this function allocates a sims structure into Stab.
	void get_orbit(int orbit_idx, long int *set, int &len,
		int verbose_level);
	void compute_orbit_statistic(int *set, int set_size, 
		int *orbit_count, int verbose_level);
	void orbits_as_set_of_sets(set_of_sets *&S, int verbose_level);
	void get_orbit_reps(int *&Reps, int &nb_reps, int verbose_level);
	int find_shortest_orbit_if_unique(int &idx);
	void elements_in_orbit_of(int pt, int *orb, int &nb, 
		int verbose_level);
	void get_orbit_lengths_once_each(int *&orbit_lengths, 
		int &nb_orbit_lengths);
	int orbit_number(int pt);
	void get_orbit_decomposition_scheme_of_graph(
		int *Adj, int n, int *&Decomp_scheme, int verbose_level);
	void create_point_list_sorted(
			int *&point_list, int &point_list_length);
	void shallow_tree_generators(int orbit_idx,
			int f_randomized,
			schreier *&shallow_tree,
			int verbose_level);
	schreier_vector *get_schreier_vector(
			int gen_hdl_first, int nb_gen,
			enum shallow_schreier_tree_strategy Shallow_schreier_tree_strategy,
			int verbose_level);
	int get_num_points();
		// This function returns the number of points in the
		// schreier forest
	double get_average_word_length();
		// This function returns the average word length of the forest.
	double get_average_word_length(int orbit_idx);
	void compute_orbit_invariant(int *&orbit_invariant,
			int (*compute_orbit_invariant_callback)(schreier *Sch,
					int orbit_idx, void *data, int verbose_level),
			void *compute_orbit_invariant_data,
			int verbose_level);

	// schreier_io.cpp:
	void latex(std::string &fname);
	void print_orbit_lengths(std::ostream &ost);
	void print_orbit_lengths_tex(std::ostream &ost);
	void print_orbit_length_distribution(std::ostream &ost);
	void print_orbit_reps(std::ostream &ost);
	void print(std::ostream &ost);
	void print_and_list_orbits(std::ostream &ost);
	void print_and_list_orbits_with_original_labels(std::ostream &ost);
	void print_and_list_orbits_tex(std::ostream &ost);
	void print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			std::ostream &ost, action *default_action, strong_generators *gens,
			int verbose_level);
	void make_orbit_trees(std::ostream &ost,
			std::string &fname_mask, layered_graph_draw_options *Opt,
			int verbose_level);
	void print_and_list_orbits_with_original_labels_tex(std::ostream &ost);
	void print_and_list_orbits_of_given_length(std::ostream &ost,
		int len);
	void print_and_list_orbits_and_stabilizer(std::ostream &ost,
		action *default_action, longinteger_object &go,
		void (*print_point)(std::ostream &ost, int pt, void *data),
			void *data);
	void print_and_list_orbits_using_labels(std::ostream &ost,
		long int *labels);
	void print_tables(std::ostream &ost, int f_with_cosetrep);
	void print_tables_latex(std::ostream &ost, int f_with_cosetrep);
	void print_generators();
	void print_generators_latex(std::ostream &ost);
	void print_generators_with_permutations();
	void print_orbit(int orbit_no);
	void print_orbit_using_labels(int orbit_no, long int *labels);
	void print_orbit(std::ostream &ost, int orbit_no);
	void print_orbit_with_original_labels(std::ostream &ost, int orbit_no);
	void print_orbit_tex(std::ostream &ost, int orbit_no);
	void print_orbit_sorted_tex(std::ostream &ost, int orbit_no, int f_truncate, int max_length);
	void print_and_list_orbit_and_stabilizer_tex(int i,
		action *default_action,
		longinteger_object &full_group_order,
		std::ostream &ost);
	void write_orbit_summary(std::string &fname,
			action *default_action,
			longinteger_object &full_group_order,
			int verbose_level);
	void print_and_list_orbit_and_stabilizer_with_list_of_elements_tex(
		int i, action *default_action,
		strong_generators *gens, std::ostream &ost);
	void print_and_list_orbit_tex(int i, std::ostream &ost);
	void print_and_list_orbits_sorted_by_length_tex(std::ostream &ost);
	void print_and_list_orbits_and_stabilizer_sorted_by_length(
			std::ostream &ost, int f_tex,
		action *default_action,
		longinteger_object &full_group_order);
	void print_fancy(
			std::ostream &ost, int f_tex,
		action *default_action, strong_generators *gens_full_group);
	void print_and_list_orbits_sorted_by_length(std::ostream &ost);
	void print_and_list_orbits_sorted_by_length(std::ostream &ost, int f_tex);
	void print_orbit_sorted_with_original_labels_tex(std::ostream &ost,
			int orbit_no, int f_truncate, int max_length);
	void print_orbit_using_labels(std::ostream &ost, int orbit_no, long int *labels);
	void print_orbit_using_callback(std::ostream &ost, int orbit_no,
		void (*print_point)(std::ostream &ost, int pt, void *data),
		void *data);
	void print_orbit_type(int f_backwards);
	void list_all_orbits_tex(std::ostream &ost);
	void print_orbit_through_labels(std::ostream &ost,
		int orbit_no, long int *point_labels);
	void print_orbit_sorted(std::ostream &ost, int orbit_no);
	void print_orbit(int cur, int last);
	void print_tree(int orbit_no);
	void export_tree_as_layered_graph(int orbit_no,
			std::string &fname_mask,
			int verbose_level);
	void draw_forest(std::string &fname_mask,
			layered_graph_draw_options *Opt,
			int f_has_point_labels, long int *point_labels,
			int verbose_level);
	void draw_tree(std::string &fname,
			layered_graph_draw_options *Opt,
			int orbit_no,
			int f_has_point_labels, long int *point_labels,
			int verbose_level);
	void draw_tree2(std::string &fname,
			layered_graph_draw_options *Opt,
			int *weight, int *placement_x, int max_depth,
			int i, int last,
			int f_has_point_labels, long int *point_labels,
			int verbose_level);
	void subtree_draw_lines(mp_graphics &G,
			layered_graph_draw_options *Opt,
			int parent_x, int parent_y, int *weight,
			int *placement_x, int max_depth, int i, int last,
			int y_max,
			int verbose_level);
	void subtree_draw_vertices(mp_graphics &G,
			layered_graph_draw_options *Opt,
			int parent_x, int parent_y, int *weight,
			int *placement_x, int max_depth, int i, int last,
			int f_has_point_labels, long int *point_labels,
			int y_max,
			int verbose_level);
	void subtree_place(int *weight, int *placement_x,
		int left, int right, int i, int last);
	int subtree_calc_weight(int *weight, int &max_depth,
		int i, int last);
	int subtree_depth_first(std::ostream &ost, int *path, int i, int last);
	void print_path(std::ostream &ost, int *path, int l);
	void write_to_file_binary(std::ofstream &fp, int verbose_level);
	void read_from_file_binary(std::ifstream &fp, int verbose_level);
	void write_file_binary(std::string &fname, int verbose_level);
	void read_file_binary(std::string &fname, int verbose_level);
	void list_elements_as_permutations_vertically(std::ostream &ost);
};

// #############################################################################
// schreier_sims.cpp
// #############################################################################


//! Schreier Sims algorithm to create the stabilizer chain of a permutation group

class schreier_sims {

public:
	action *GA;
	sims *G;

	int f_interested_in_kernel;
	action *KA;
	sims *K;

	longinteger_object G_order, K_order, KG_order;
	
	int *Elt1;
	int *Elt2;
	int *Elt3;

	int f_has_target_group_order;
	longinteger_object tgo; // target group order

	
	int f_from_generators;
	vector_ge *external_gens;

	int f_from_random_process;
	void (*callback_choose_random_generator)(int iteration, 
		int *Elt, void *data, int verbose_level);
	void *callback_choose_random_generator_data;
	
	int f_from_old_G;
	sims *old_G;

	int f_has_base_of_choice;
	int base_of_choice_len;
	int *base_of_choice;

	int f_override_choose_next_base_point_method;
	int (*choose_next_base_point_method)(action *A, 
		int *Elt, int verbose_level); 

	int iteration;

	schreier_sims();
	~schreier_sims();
	void null();
	void freeself();
	void init(action *A, int verbose_level);
	void interested_in_kernel(action *KA, int verbose_level);
	void init_target_group_order(longinteger_object &tgo, 
		int verbose_level);
	void init_generators(vector_ge *gens, int verbose_level);
	void init_random_process(
		void (*callback_choose_random_generator)(
		int iteration, int *Elt, void *data, 
		int verbose_level), 
		void *callback_choose_random_generator_data, 
		int verbose_level);
	void init_old_G(sims *old_G, int verbose_level);
	void init_base_of_choice(
		int base_of_choice_len, int *base_of_choice, 
		int verbose_level);
	void init_choose_next_base_point_method(
		int (*choose_next_base_point_method)(action *A, 
		int *Elt, int verbose_level), 
		int verbose_level);
	void compute_group_orders();
	void print_group_orders();
	void get_generator_internal(int *Elt, int verbose_level);
	void get_generator_external(int *Elt, int verbose_level);
	void get_generator_external_from_generators(int *Elt, 
		int verbose_level);
	void get_generator_external_random_process(int *Elt, 
		int verbose_level);
	void get_generator_external_old_G(int *Elt, 
		int verbose_level);
	void get_generator(int *Elt, int verbose_level);
	void closure_group(int verbose_level);
	void create_group(int verbose_level);
};

// #############################################################################
// sims.cpp
// #############################################################################

//! a permutation group represented via a stabilizer chain

class sims {

public:
	action *A;

	int my_base_len;

	vector_ge gens;
	vector_ge gens_inv;
	
	int *gen_depth; // [nb_gen]
	int *gen_perm; // [nb_gen]
	
	int *nb_gen; // [my_base_len + 1]
		// nb_gen[i] is the number of generators 
		// which stabilize the base points 0,...,i-1, 
		// i.e. which belong to G^{(i)}.
		// The actual generator index ("gen_idx") must be obtained
		// from the array gen_perm[].
		// Thus, gen_perm[j] for 0 \le j < nb_gen[i] are the 
		// indices of generators which belong to G^{(i)}
		// the generators for G^{(i)} modulo G^{(i+1)} 
		// those indexed by nb_gen[i + 1], .., nb_gen[i] - 1 (!!!)
		// Observe that the entries in nb_gen[] are *decreasing*.
		// This is because the generators at the bottom of the 
		// stabilizer chain are listed first. 
		// (And nb_gen[0] is the total number of generators).
	

	int transversal_length;
		// an upper bound for the length of every basic orbit

	int *path; // [my_base_len]
	
	int nb_images;
	int **images;
	

private:
	// stabilizer chain:

	int *orbit_len; // [my_base_len]
		// orbit_len[i] is the length of the i-th basic orbit.
	
	int **orbit;
		// [my_base_len][transversal_length]
		// orbit[i][j] is the j-th point in the orbit 
		// of the i-th base point.
		// for 0 \le j < orbit_len[i].
		// for orbit_len[i] \le j < A->deg, 
		// the points not in the orbit are listed.
	int **orbit_inv;
		// [my_base_len][transversal_length]
		// orbit[i] is the inverse of the permutation orbit[i],
		// i.e. given a point j,
		// orbit_inv[i][j] is the coset (or position in the orbit)
		// which contains j.
	
	int **prev; // [my_base_len][transversal_length]
	int **label; // [my_base_len][transversal_length]
	

	// this is wrong, Path and Label describe a path in a schreier tree
	// and hence should be allocated according
	// to the largest degree, not the base length
	//int *Path; // [my_base_len + 1]
	//int *Label; // [my_base_len]

	
	// storage for temporary data and 
	// group elements computed by various routines.
	int *Elt1, *Elt2, *Elt3, *Elt4;
	int *strip1, *strip2;
		// used in strip
	int *eltrk1, *eltrk2, *eltrk3;
		// used in element rank unrank
	int *cosetrep_tmp;
		// used in coset_rep / coset_rep_inv
	int *schreier_gen, *schreier_gen1;
		// used in random_schreier_generator

	int *cosetrep;
public:


	// sims.cpp:
	sims();
	void null();
	sims(action *A, int verbose_level);
	~sims();
	void freeself();

	void delete_images();
	void init_images(int nb_images);
	void images_append();
	void init(action *A, int verbose_level);
		// initializes the trivial group 
		// with the base as given in A
	void init_cyclic_group_from_generator(action *A, int *Elt, int verbose_level);
	// initializes the cyclic group generated by Elt with the base as given in A
	void init_without_base(action *A, int verbose_level);
	void reallocate_base(int old_base_len, int verbose_level);
	void initialize_table(int i, int verbose_level);
	void init_trivial_group(int verbose_level);
		// clears the generators array, 
		// and sets the i-th transversal to contain
		// only the i-th base point (for all i).
	void init_trivial_orbit(int i, int verbose_level);
	void init_generators(vector_ge &generators, int verbose_level);
	void init_generators(int nb, int *elt, int verbose_level);
		// copies the given elements into the generator array, 
		// then computes depth and perm
	void init_generators_by_hdl(int nb_gen, int *gen_hdl, int verbose_level);
	void init_generator_depth_and_perm(int verbose_level);
	void add_generator(int *elt, int verbose_level);
		// adds elt to list of generators, 
		// computes the depth of the element, 
		// updates the arrays gen_depth and gen_perm accordingly
		// does not change the transversals
	int generator_depth(int gen_idx);
		// returns the index of the first base point 
		// which is moved by a given generator. 
	int generator_depth(int *elt);
		// returns the index of the first base point 
		// which is moved by the given element
	void group_order(longinteger_object &go);
	void group_order_verbose(longinteger_object &go, int verbose_level);
	void subgroup_order_verbose(longinteger_object &go, int level, int verbose_level);
	long int group_order_lint();
	int is_trivial_group();
	int last_moved_base_point();
		// j == -1 means the group is trivial
	int get_image(int i, int gen_idx);
		// get the image of a point i under generator gen_idx, 
		// goes through a 
		// table of stored images by default. 
		// Computes the image only if not yet available.
	int get_image(int i, int *elt);
		// get the image of a point i under a given group element, 
		// does not go through a table.
	void swap_points(int lvl, int i, int j);
		// swaps two points given by their cosets
	void path_unrank_lint(long int a);
	long int path_rank_lint();
	
	void element_from_path(int *elt, int verbose_level);
		// given coset representatives in path[], 
		// the corresponding 
		// element is multiplied.
		// uses eltrk1, eltrk2
	void element_from_path_inv(int *elt);
	void element_unrank(longinteger_object &a, int *elt, 
		int verbose_level);
	void element_unrank(longinteger_object &a, int *elt);
		// Returns group element whose rank is a. 
		// the elements represented by the chain are 
		// enumerated 0, ... go - 1
		// with the convention that 0 always stands 
		// for the identity element.
		// The computed group element will be computed into Elt1
	void element_rank(longinteger_object &a, int *elt);
		// Computes the rank of the element in elt into a.
		// uses eltrk1, eltrk2
	void element_unrank_lint(long int rk, int *Elt, int verbose_level);
	void element_unrank_lint(long int rk, int *Elt);
	long int element_rank_lint(int *Elt);
	int is_element_of(int *elt);
	void test_element_rank_unrank();
	void coset_rep(int *Elt, int i, int j, int verbose_level);
		// computes a coset representative in transversal i 
		// which maps
		// the i-th base point to the point which is in 
		// coset j of the i-th basic orbit.
		// j is a coset, not a point
		// result is in cosetrep
	int compute_coset_rep_depth(int i, int j, int verbose_level);
	void compute_coset_rep_path(int i, int j, int &depth,
			int *&Path, int *&Label,
		int verbose_level);
	void coset_rep_inv(int *Elt, int i, int j, int verbose_level_le);
		// computes the inverse element of what coset_rep computes,
		// i.e. an element which maps the 
		// j-th point in the orbit to the 
		// i-th base point.
		// j is a coset, not a point
		// result is in cosetrep
	void extract_strong_generators_in_order(vector_ge &SG, 
		int *tl, int verbose_level);
	void random_schreier_generator(int *Elt, int verbose_level);
		// computes random Schreier generator
	void element_as_permutation(action *A_special, 
		long int elt_rk, int *perm, int verbose_level);
	int least_moved_point_at_level(int lvl, int verbose_level);
	int get_orbit(int i, int j);
	int get_orbit_inv(int i, int j);
	int get_orbit_length(int i);
	void get_orbit(int orbit_idx, std::vector<int> &Orb, int verbose_level);


	// sims_main.cpp:
	void compute_base_orbits(int verbose_level);
	void compute_base_orbits_known_length(int *tl,
		int verbose_level);
	void extend_base_orbit(int new_gen_idx, int lvl,
		int verbose_level);
	void compute_base_orbit(int lvl, int verbose_level);
		// applies all generators at the given level to compute
		// the corresponding basic orbit.
		// the generators are the first nb_gen[lvl]
		// in the generator array
	void compute_base_orbit_known_length(int lvl,
		int target_length, int verbose_level);
	int strip_and_add(int *elt, int *residue, int verbose_level);
		// returns TRUE if something was added,
		// FALSE if element stripped through
	int strip(int *elt, int *residue, int &drop_out_level,
		int &image, int verbose_level);
		// returns TRUE if the element sifts through
	void add_generator_at_level(int *elt, int lvl,
		int verbose_level);
		// add the generator to the array of generators
		// and then extends the
		// basic orbits 0,..,lvl using extend_base_orbit
	void add_generator_at_level_only(int *elt,
		int lvl, int verbose_level);
		// add the generator to the array of generators
		// and then extends the
		// basic orbit lvl using extend_base_orbit
	void build_up_group_random_process_no_kernel(sims *old_G,
		int verbose_level);
	void extend_group_random_process_no_kernel(sims *extending_by_G,
		longinteger_object &target_go,
		int verbose_level);
	void build_up_group_random_process(sims *K, sims *old_G,
		longinteger_object &target_go,
		int f_override_choose_next_base_point,
		int (*choose_next_base_point_method)(action *A,
			int *Elt, int verbose_level),
		int verbose_level);
	void build_up_group_from_generators(sims *K, vector_ge *gens,
		int f_target_go, longinteger_object *target_go,
		int f_override_choose_next_base_point,
		int (*choose_next_base_point_method)(action *A,
			int *Elt, int verbose_level),
		int verbose_level);
	int closure_group(int nb_times, int verbose_level);




	// sims2.cpp
	void build_up_subgroup_random_process(sims *G, 
		void (*choose_random_generator_for_subgroup)(
			sims *G, int *Elt, int verbose_level), 
		int verbose_level);

	// sims3.cpp
	void subgroup_make_characteristic_vector(sims *Sub, 
		int *C, int verbose_level);
	void normalizer_based_on_characteristic_vector(int *C_sub, 
		int *Gen_idx, int nb_gens, int *N, long int &N_go,
		int verbose_level);
	void order_structure_relative_to_subgroup(int *C_sub, 
		int *Order, int *Residue, int verbose_level);



	// sims_group_theory.cpp:
	void random_element(int *elt, int verbose_level);
		// compute a random element among the group elements
		// represented by the chain
		// (chooses random cosets along the stabilizer chain)
	void random_element_of_order(int *elt, int order,
		int verbose_level);
	void random_elements_of_order(vector_ge *elts,
		int *orders, int nb, int verbose_level);
	void transitive_extension(schreier &O, vector_ge &SG,
		int *tl, int verbose_level);
	int transitive_extension_tolerant(schreier &O,
		vector_ge &SG, int *tl, int f_tolerant,
		int verbose_level);
	void transitive_extension_using_coset_representatives_extract_generators(
		int *coset_reps, int nb_cosets,
		vector_ge &SG, int *tl,
		int verbose_level);
	void transitive_extension_using_coset_representatives(
		int *coset_reps, int nb_cosets,
		int verbose_level);
	void transitive_extension_using_generators(
		int *Elt_gens, int nb_gens, int subgroup_index,
		vector_ge &SG, int *tl,
		int verbose_level);
	void point_stabilizer_stabchain_with_action(action *A2,
		sims &S, int pt, int verbose_level);
		// first computes the orbit of the point pt
		// in action A2 under the generators
		// that are stored at present
		// (using a temporary schreier object),
		// then sifts random schreier generators into S
	void point_stabilizer(vector_ge &SG, int *tl,
		int pt, int verbose_level);
		// computes strong generating set
		// for the stabilizer of point pt
	void point_stabilizer_with_action(action *A2,
		vector_ge &SG, int *tl, int pt, int verbose_level);
		// computes strong generating set for
		// the stabilizer of point pt in action A2
	void conjugate(action *A, sims *old_G, int *Elt,
		int f_overshooting_OK, int verbose_level);
		// Elt * g * Elt^{-1} where g is in old_G
	int test_if_in_set_stabilizer(action *A,
		long int *set, int size, int verbose_level);
	int test_if_subgroup(sims *old_G, int verbose_level);
	int find_element_with_exactly_n_fixpoints_in_given_action(
			int *Elt, int nb_fixpoints, action *A_given, int verbose_level);
	void table_of_group_elements_in_data_form(int *&Table,
		int &len, int &sz, int verbose_level);
	void regular_representation(int *Elt, int *perm,
		int verbose_level);
	void center(vector_ge &gens, int *center_element_ranks,
		int &nb_elements, int verbose_level);
	void all_cosets(int *subset, int size,
		long int *all_cosets, int verbose_level);
	void element_ranks_subgroup(sims *subgroup,
		int *element_ranks, int verbose_level);
	void find_standard_generators_int(int ord_a, int ord_b,
		int ord_ab, int &a, int &b, int &nb_trials,
		int verbose_level);
	long int find_element_of_given_order_int(int ord,
		int &nb_trials, int verbose_level);
	int find_element_of_given_order_int(int *Elt,
		int ord, int &nb_trials, int max_trials,
		int verbose_level);
	void find_element_of_prime_power_order(int p,
		int *Elt, int &e, int &nb_trials, int verbose_level);
	void evaluate_word_int(int word_len,
		int *word, int *Elt, int verbose_level);
	void sylow_subgroup(int p, sims *P, int verbose_level);
	int is_normalizing(int *Elt, int verbose_level);
	void create_Cayley_graph(vector_ge *gens, int *&Adj, long int &n,
		int verbose_level);
	void create_group_table(int *&Table, long int &n, int verbose_level);
	void compute_conjugacy_classes(
		action *&Aconj, action_by_conjugation *&ABC, schreier *&Sch,
		strong_generators *&SG, int &nb_classes,
		int *&class_size, int *&class_rep,
		int verbose_level);
	void compute_all_powers(int elt_idx, int n, int *power_elt,
			int verbose_level);
	long int mult_by_rank(long int rk_a, long int rk_b, int verbose_level);
	long int mult_by_rank(long int rk_a, long int rk_b);
	long int invert_by_rank(long int rk_a, int verbose_level);
	long int conjugate_by_rank(long int rk_a, long int rk_b, int verbose_level);
		// comutes b^{-1} * a * b
	long int conjugate_by_rank_b_bv_given(long int rk_a,
		int *Elt_b, int *Elt_bv, int verbose_level);
	void zuppo_list(
			int *Zuppos, int &nb_zuppos, int verbose_level);
	void dimino(
		int *subgroup, int subgroup_sz, int *gens, int &nb_gens,
		int *cosets,
		int new_gen,
		int *group, int &group_sz,
		int verbose_level);
	void Cayley_graph(int *&Adj, int &sz, vector_ge *gens_S,
		int verbose_level);


	// sims_io.cpp:
	void create_group_tree(const char *fname, int f_full,
		int verbose_level);
	void print_transversals();
	void print_transversals_short();
	void print_transversal_lengths();
	void print_orbit_len();
	void print(int verbose_level);
	void print_generators();
	void print_generators_tex(std::ostream &ost);
	void print_generators_as_permutations();
	void print_generators_as_permutations_override_action(
		action *A);
	void print_basic_orbits();
	void print_basic_orbit(int i);
	void print_generator_depth_and_perm();
	void print_group_order(std::ostream &ost);
	void print_group_order_factored(std::ostream &ost);
	void print_generators_at_level_or_below(int lvl);
	void write_all_group_elements(char *fname, int verbose_level);
	void print_all_group_elements_to_file(char *fname,
		int verbose_level);
	void print_all_group_elements();
	void print_all_group_elements_tex(std::ostream &ost);
	void print_all_group_elements_with_permutations_tex(std::ostream &ost);
	void print_all_group_elements_as_permutations();
	void print_all_group_elements_as_permutations_in_special_action(
		action *A_special);
	void print_all_transversal_elements();
	void save_list_of_elements(char *fname,
		int verbose_level);
	void read_list_of_elements(action *A,
		char *fname, int verbose_level);
	void write_as_magma_permutation_group(std::string &fname_base,
		vector_ge *gens, int verbose_level);
	void report(std::ostream &ost,
			std::string &prefix,
			layered_graph_draw_options *LG_Draw_options,
			int verbose_level);


};

// sims2.cpp
void choose_random_generator_derived_group(sims *G, int *Elt, 
	int verbose_level);





// #############################################################################
// strong_generators.cpp
// #############################################################################

//! a strong generating set for a permutation group with respect to a fixed action

class strong_generators {
public:

	action *A;
	int *tl;
	vector_ge *gens;

	strong_generators();
	~strong_generators();
	void null();
	void freeself();
	void swap_with(strong_generators *SG);
	void init(action *A);
	void init(action *A, int verbose_level);
	void init_from_sims(sims *S, int verbose_level);
	void init_from_ascii_coding(action *A, 
		char *ascii_coding, int verbose_level);
	strong_generators *create_copy();
	void init_copy(strong_generators *S, 
		int verbose_level);
	void init_by_hdl_and_with_tl(action *A,
			std::vector<int> &gen_handle,
			std::vector<int> &tl,
			int verbose_level);
	void init_by_hdl(action *A, int *gen_hdl, 
		int nb_gen, int verbose_level);
	void init_from_permutation_representation(action *A, 
			sims *parent_group_S, int *data,
			int nb_elements, long int group_order, vector_ge *&nice_gens,
			int verbose_level);
	void init_from_data(action *A, int *data, 
		int nb_elements, int elt_size, 
		int *transversal_length, 
		vector_ge *&nice_gens,
		int verbose_level);
	void init_from_data_with_target_go_ascii(action *A, 
		int *data, 
		int nb_elements, int elt_size, 
		const char *ascii_target_go,
		vector_ge *&nice_gens,
		int verbose_level);
	void init_from_data_with_target_go(action *A, 
		int *data_gens, 
		int data_gens_size, int nb_gens, 
		longinteger_object &target_go, 
		vector_ge *&nice_gens,
		int verbose_level);
	void init_point_stabilizer_of_arbitrary_point_through_schreier(
		schreier *Sch, 
		int pt, int &orbit_idx, 
		longinteger_object &full_group_order, 
		int verbose_level);
	void init_point_stabilizer_orbit_rep_schreier(schreier *Sch, 
		int orbit_idx, longinteger_object &full_group_order, 
		int verbose_level);
	void init_generators_for_the_conjugate_group_avGa(
		strong_generators *SG, int *Elt_a, int verbose_level);
	void init_generators_for_the_conjugate_group_aGav(
		strong_generators *SG, int *Elt_a, int verbose_level);
	void init_transposed_group(strong_generators *SG, 
		int verbose_level);
	void init_group_extension(strong_generators *subgroup, 
		int *data, int index, 
		int verbose_level);
	void init_group_extension(strong_generators *subgroup, 
		vector_ge *new_gens, int index, 
		int verbose_level);
	void switch_to_subgroup(const char *rank_vector_text, 
		const char *subgroup_order_text, sims *S, 
		int *&subgroup_gens_idx, int &nb_subgroup_gens, 
		int verbose_level);
	void init_subgroup(action *A, int *subgroup_gens_idx, 
		int nb_subgroup_gens, 
		const char *subgroup_order_text, 
		sims *S, 
		int verbose_level);
	void init_subgroup_by_generators(action *A,
		int nb_subgroup_gens,
		std::string *subgroup_gens,
		std::string &subgroup_order_text,
		vector_ge *&nice_gens,
		int verbose_level);
	sims *create_sims(int verbose_level);
	sims *create_sims_in_different_action(action *A_given, 
		int verbose_level);
	void add_generators(vector_ge *coset_reps, 
		int group_index, int verbose_level);
	void add_single_generator(int *Elt, 
		int group_index, int verbose_level);
	void group_order(longinteger_object &go);
	long int group_order_as_lint();
	//void print_generators();
	//void print_generators_ost(std::ostream &ost);
	void print_generators_in_source_code();
	void print_generators_in_source_code_to_file(
	const char *fname);
	void print_generators_even_odd();
	void print_generators_MAGMA(action *A, std::ostream &ost);
	void export_magma(action *A, std::ostream &ost);
	void print_generators_gap(std::ostream &ost);
	void print_generators_compact(std::ostream &ost);
	void print_generators(std::ostream &ost);
	void print_generators_in_latex_individually(std::ostream &ost);
	void print_generators_tex();
	void print_generators_tex(std::ostream &ost);
	void print_generators_tex_with_print_point_function(
			action *A,
			std::ostream &ost,
			void (*point_label)(std::stringstream &sstr, long int pt, void *data),
			void *point_label_data);
	void print_generators_for_make_element(std::ostream &ost);
	void print_generators_as_permutations();
	void print_generators_as_permutations_tex(std::ostream &ost, action *A2);
	void print_with_given_action(std::ostream &ost, action *A2);
	void print_elements_ost(std::ostream &ost);
	void print_elements_with_special_orthogonal_action_ost(std::ostream &ost);
	void print_elements_with_given_action(std::ostream &ost, action *A2);
	void print_elements_latex_ost(std::ostream &ost);
	void print_elements_latex_ost_with_print_point_function(
			action *A,
			std::ostream &ost,
			void (*point_label)(std::stringstream &sstr, long int pt, void *data),
			void *point_label_data);
	void create_group_table(int *&Table, long int &go,
		int verbose_level);
	void list_of_elements_of_subgroup(
		strong_generators *gens_subgroup, 
		long int *&Subgroup_elements_by_index,
		long int &sz_subgroup, int verbose_level);
	void compute_schreier_with_given_action(action *A_given, 
		schreier *&Sch, int verbose_level);
	void compute_schreier_with_given_action_on_a_given_set(
		action *A_given, 
		schreier *&Sch, long int *set, int len, int verbose_level);
	void orbits_on_points(int &nb_orbits, int *&orbit_reps, 
		int verbose_level);
	void orbits_on_points_with_given_action(action *A_given, 
		int &nb_orbits, int *&orbit_reps, int verbose_level);
	schreier *orbits_on_points_schreier(action *A_given, 
		int verbose_level);
	schreier *orbit_of_one_point_schreier(action *A_given, 
		int pt, int verbose_level);
	void orbits_light(action *A_given, 
		int *&Orbit_reps, int *&Orbit_lengths, int &nb_orbits, 
		int **&Pts_per_generator, int *&Nb_per_generator, 
		int verbose_level);
	void write_to_file_binary(std::ofstream &fp, int verbose_level);
	void read_from_file_binary(action *A, std::ifstream &fp,
		int verbose_level);
	void write_file(std::string &fname, int verbose_level);
	void read_file(action *A, std::string &fname, int verbose_level);
	void compute_ascii_coding(char *&ascii_coding, int verbose_level);
	void decode_ascii_coding(char *ascii_coding, int verbose_level);
	void export_permutation_group_to_magma(std::string &fname,
		int verbose_level);
	void export_permutation_group_to_GAP(std::string &fname,
		int verbose_level);
	void compute_and_print_orbits_on_a_given_set(action *A_given,
		long int *set, int len, int verbose_level);
	void compute_and_print_orbits(action *A_given, 
		int verbose_level);
	int test_if_normalizing(sims *S, int verbose_level);
	void test_if_set_is_invariant_under_given_action(action *A_given, 
		long int *set, int set_sz, int verbose_level);
	strong_generators *point_stabilizer(int pt, int verbose_level);
	strong_generators *find_cyclic_subgroup_with_exactly_n_fixpoints(
			int nb_fixpoints, action *A_given, int verbose_level);
	void make_element_which_moves_a_point_from_A_to_B(action *A_given, 
		int pt_A, int pt_B, int *Elt, int verbose_level);
	void export_group_to_magma_and_copy_to_latex(
			std::string &label_txt,
			std::ostream &ost,
			int verbose_level);
	void export_group_to_GAP_and_copy_to_latex(
			std::string &label_txt,
			std::ostream &ost,
			int verbose_level);
	void export_group_and_copy_to_latex(
			std::string &label_txt,
			std::ostream &ost,
			int verbose_level);
	void report_fixed_objects_in_P3(
			std::ostream &ost,
			projective_space *P3,
			int verbose_level);
	void reverse_isomorphism_exterior_square(int verbose_level);

	// strong_generators_groups.cpp
	void init_linear_group_from_scratch(action *&A, 
		finite_field *F, int n, 
		int f_projective, int f_general, int f_affine, 
		int f_semilinear, int f_special, 
		int f_GL_d_wreath_Sym_n,
		int GL_wreath_Sym_d, int GL_wreath_Sym_n,
		vector_ge *&nice_gens,
		int verbose_level);
	void special_subgroup(int verbose_level);
	void projectivity_subgroup(sims *S, int verbose_level);
	void even_subgroup(int verbose_level);
	void Sylow_subgroup(sims *S, int p, int verbose_level);
	void init_single(action *A, int *Elt, int verbose_level);
	void init_single_with_target_go(action *A,
			int *Elt, int target_go, int verbose_level);
	void init_trivial_group(action *A, int verbose_level);
	void generators_for_the_monomial_group(action *A, 
		matrix_group *Mtx, int verbose_level);
	void generators_for_the_diagonal_group(action *A, 
		matrix_group *Mtx, int verbose_level);
	void generators_for_the_singer_cycle(action *A, 
		matrix_group *Mtx, int power_of_singer,
		vector_ge *&nice_gens,
		int verbose_level);
	void generators_for_the_singer_cycle_and_the_Frobenius(
		action *A,
		matrix_group *Mtx, int power_of_singer,
		vector_ge *&nice_gens,
		int verbose_level);
	void generators_for_the_null_polarity_group(action *A, 
		matrix_group *Mtx, int verbose_level);
	void generators_for_symplectic_group(action *A, 
		matrix_group *Mtx, int verbose_level);
	void init_centralizer_of_matrix(action *A, int *Mtx, 
		int verbose_level);
	void init_centralizer_of_matrix_general_linear(action *A_projective, 
		action *A_general_linear, int *Mtx, 
		int verbose_level);
	void field_reduction(action *Aq, int n, int s, 
		finite_field *Fq, int verbose_level);
	void generators_for_translation_plane_in_andre_model(
		action *A_PGL_n1_q, action *A_PGL_n_q, 
		matrix_group *Mtx_n1, matrix_group *Mtx_n, 
		vector_ge *spread_stab_gens, 
		longinteger_object &spread_stab_go, 
		int verbose_level);
	void generators_for_the_stabilizer_of_two_components(
		action *A_PGL_n_q, 
		matrix_group *Mtx, int verbose_level);
	void regulus_stabilizer(action *A_PGL_n_q, 
		matrix_group *Mtx, int verbose_level);
	void generators_for_the_borel_subgroup_upper(action *A_linear, 
		matrix_group *Mtx, int verbose_level);
	void generators_for_the_borel_subgroup_lower(action *A_linear, 
		matrix_group *Mtx, int verbose_level);
	void generators_for_the_identity_subgroup(action *A_linear, 
		matrix_group *Mtx, int verbose_level);
	void generators_for_parabolic_subgroup(action *A_PGL_n_q, 
		matrix_group *Mtx, int k, int verbose_level);
	void generators_for_stabilizer_of_three_collinear_points_in_PGL4(
		action *A_PGL_4_q, 
		matrix_group *Mtx, int verbose_level);
	void generators_for_stabilizer_of_triangle_in_PGL4(
		action *A_PGL_4_q, 
		matrix_group *Mtx, int verbose_level);
	void generators_for_the_orthogonal_group(action *A, 
		finite_field *F, int n, 
		int epsilon, 
		int f_semilinear, 
		int verbose_level);
	void stabilizer_of_cubic_surface_from_catalogue(
		action *A, 
		finite_field *F, int iso, 
		int verbose_level);
	void stabilizer_of_HCV_surface(
		action *A, 
		finite_field *F, int f_with_normalizer, int f_semilinear, 
		vector_ge *&nice_gens,
		int verbose_level);
	void stabilizer_of_G13_surface(
		action *A,
		finite_field *F, int a,
		vector_ge *&nice_gens,
		int verbose_level);
	void stabilizer_of_F13_surface(
		action *A,
		finite_field *F, int a,
		vector_ge *&nice_gens,
		int verbose_level);
	void BLT_set_from_catalogue_stabilizer(action *A, 
		finite_field *F, int iso, 
		int verbose_level);
	void stabilizer_of_spread_from_catalogue(action *A, 
		int q, int k, int iso, 
		int verbose_level);
	void stabilizer_of_pencil_of_conics(
		action *A,
		finite_field *F,
		int verbose_level);
	void Janko1(
		action *A,
		finite_field *F,
		int verbose_level);
	void Hall_reflection(
		int nb_pairs, int &degree, int verbose_level);
	void normalizer_of_a_Hall_reflection(
		int nb_pairs, int &degree, int verbose_level);
	void hyperplane_lifting_with_two_lines_fixed(
		strong_generators *SG_hyperplane,
		projective_space *P, int line1, int line2,
		int verbose_level);
	void exterior_square(
			action *A_detached,
			strong_generators *SG_original,
			vector_ge *&nice_gens,
			int verbose_level);
	void diagonally_repeat(
			action *An,
			strong_generators *Sn,
			int verbose_level);

};

// #############################################################################
// subgroup.cpp:
// #############################################################################

//! a subgroup of a group using a list of elements

class subgroup {
public:
	action *A;
	int *Elements;
	long int group_order;
	int *gens;
	int nb_gens;
	sims *Sub;
	strong_generators *SG;


	subgroup();
	~subgroup();
	void null();
	void freeself();
	void init_from_sims(sims *S, sims *Sub,
			strong_generators *SG, int verbose_level);
	void init(int *Elements, int group_order, int *gens, int nb_gens);
	void print();
	int contains_this_element(int elt);
	void report(std::ostream &ost);
};

// #############################################################################
// sylow_structure.cpp:
// #############################################################################

//! The Sylow structure of a finite group

class sylow_structure {
public:
	longinteger_object go;
	int *primes;
	int *exponents;
	int nb_primes;

	sims *S; // the group
	subgroup *Sub; // [nb_primes]

	sylow_structure();
	~sylow_structure();
	void null();
	void freeself();
	void init(sims *S, int verbose_level);
	void report(std::ostream &ost);
};


// #############################################################################
// wreath_product.cpp
// #############################################################################

//! the wreath product group GL(d,q) wreath Sym(n)

class wreath_product {

public:
	matrix_group *M;
	action *A_mtx;
	finite_field *F;
	int q;
	int nb_factors;

	std::string label;
	std::string label_tex;

	int degree_of_matrix_group;
		// = M->degree;
	int dimension_of_matrix_group;
		// = M->n
	int dimension_of_tensor_action;
		// = i_power_j(dimension_of_matrix_group, nb_factors);
	long int degree_of_tensor_action;
		// = (i_power_j_safe(q, dimension_of_tensor_action) - 1) / (q - 1);
	long int degree_overall;
		// = nb_factors + nb_factors *
		// degree_of_matrix_group + degree_of_tensor_action;
	int low_level_point_size;
		// = dimension_of_tensor_action
	int make_element_size;
		// = nb_factors + nb_factors *
		// dimension_of_matrix_group * dimension_of_matrix_group;
	permutation_representation_domain *P;
	int elt_size_int;
		// = M->elt_size_int * nb_factors + P->elt_size_int;

	int *perm_offset_i; // [nb_factors + 1]
		// perm_offset_i[0] = nb_factors
		// perm_offset_i[nb_factors] = beginning of tensor domain
	int *mtx_size;
	int *index_set1;
	int *index_set2;
	int *u; // [dimension_of_tensor_action]
	int *v; // [dimension_of_tensor_action]
	int *w; // [dimension_of_tensor_action]
	int *A1; // [dimension_of_tensor_action * dimension_of_tensor_action]
	int *A2; // [dimension_of_tensor_action * dimension_of_tensor_action]
	int *A3; // [dimension_of_tensor_action * dimension_of_tensor_action]
	int *tmp_Elt1; // [elt_size_int]
	int *tmp_perm1; // [P->elt_size_int]
	int *tmp_perm2; // [P->elt_size_int]
	int *induced_perm; // [dimension_of_tensor_action]

	int bits_per_digit;
	int bits_per_elt;
	int char_per_elt;

	uchar *elt1;

	int base_len_in_component;
	long int *base_for_component;
	int *tl_for_component;

	int base_length;
	long int *the_base;
	int *the_transversal_length;

	page_storage *Elts;

	uint32_t *rank_one_tensors; // [nb_rank_one_tensors]
	long int *rank_one_tensors_in_PG; // [nb_rank_one_tensors]
		// rank_one_tensors_in_PG[i] = affine_rank_to_PG_rank(rank_one_tensors[i]);
	long int *rank_one_tensors_in_PG_sorted; // [nb_rank_one_tensors]
	int nb_rank_one_tensors;

	char *TR; // [degree_of_tensor_action + 1]
	uint32_t *Prev; // [degree_of_tensor_action + 1]

	wreath_product();
	~wreath_product();
	void null();
	void freeself();
	void init_tensor_wreath_product(matrix_group *M,
			action *A_mtx, int nb_factors,
			int verbose_level);
	void compute_tensor_ranks(int verbose_level);
	void unrank_point(long int a, int *v, int verbose_level);
	long int rank_point(int *v, int verbose_level);
	long int element_image_of(int *Elt, long int a, int verbose_level);
	void element_image_of_low_level(int *Elt,
			int *input, int *output, int verbose_level);
		// we assume that we are in the tensor product domain
	void element_one(int *Elt);
	int element_is_one(int *Elt);
	void element_mult(int *A, int *B, int *AB, int verbose_level);
	void element_move(int *A, int *B, int verbose_level);
	void element_invert(int *A, int *Av, int verbose_level);
	void compute_induced_permutation(int *Elt, int *perm);
	void apply_permutation(int *Elt,
			int *v_in, int *v_out, int verbose_level);
	int offset_i(int i);
	void create_matrix(int *Elt, int *A, int verbose_level);
		// uses A1, A2
	void element_pack(int *Elt, uchar *elt);
	void element_unpack(uchar *elt, int *Elt);
	void put_digit(uchar *elt, int f, int i, int j, int d);
	int get_digit(uchar *elt, int f, int i, int j);
	void make_element_from_one_component(int *Elt,
			int f, int *Elt_component);
	void make_element_from_permutation(int *Elt, int *perm);
	void make_element(int *Elt, int *data, int verbose_level);
	void element_print_for_make_element(int *Elt, std::ostream &ost);
	void element_print_easy(int *Elt, std::ostream &ost);
	void element_print_latex(int *Elt, std::ostream &ost);
	void compute_base_and_transversals(int verbose_level);
	void make_strong_generators_data(int *&data,
			int &size, int &nb_gens, int verbose_level);
	void report_rank_one_tensors(
			std::ostream &ost, int verbose_level);
	void create_all_rank_one_tensors(
			uint32_t *&rank_one_tensors,
			int &nb_rank_one_tensors, int verbose_level);
	uint32_t tensor_affine_rank(int *tensor);
	void tensor_affine_unrank(int *tensor, uint32_t rk);
	long int tensor_PG_rank(int *tensor);
	void tensor_PG_unrank(int *tensor, long int PG_rk);
	long int affine_rank_to_PG_rank(uint32_t affine_rk);
	uint32_t PG_rank_to_affine_rank(long int PG_rk);
	void save_rank_one_tensors(int verbose_level);
	void compute_tensor_ranks(char *&TR, uint32_t *&Prev, int verbose_level);
	void report(std::ostream &ost, int verbose_level);
	void compute_permutations_and_write_to_file(
			strong_generators* SG,
			action* A,
			int*& result,
			int &nb_gens, int &degree,
			int nb_factors,
			int verbose_level);
	void make_fname(char *fname, int nb_factors, int h, int b);
	int test_if_file_exists(int nb_factors, int h, int b);
	void orbits_using_files_and_union_find(
			strong_generators* SG,
			action* A,
			int*& result,
			int &nb_gens, int &degree,
			int nb_factors,
			int verbosity);
	void orbits_restricted(
			strong_generators* SG,
			action* A,
			int*& result,
			int &nb_gens, int &degree,
			int nb_factors,
			std::string &orbits_restricted_fname,
			int verbose_level);
	void orbits_restricted_compute(
			strong_generators* SG,
			action* A,
			int*& result,
			int &nb_gens, int &degree,
			int nb_factors,
			std::string &orbits_restricted_fname,
			int verbose_level);
};

}}


#endif /* ORBITER_SRC_LIB_GROUP_ACTIONS_GROUPS_GROUPS_H_ */



