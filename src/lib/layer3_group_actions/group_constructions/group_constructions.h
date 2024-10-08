/*
 * group_constructions.h
 *
 *  Created on: Jan 21, 2024
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER3_GROUP_ACTIONS_GROUP_CONSTRUCTIONS_GROUP_CONSTRUCTIONS_H_
#define SRC_LIB_LAYER3_GROUP_ACTIONS_GROUP_CONSTRUCTIONS_GROUP_CONSTRUCTIONS_H_


namespace orbiter {
namespace layer3_group_actions {
namespace group_constructions {

// #############################################################################
// direct_product.cpp
// #############################################################################

//! the direct product of two matrix groups in product action

class direct_product {

public:
	algebra::matrix_group *M1;
	algebra::matrix_group *M2;
	field_theory::finite_field *F1;
	field_theory::finite_field *F2;
	int q1;
	int q2;

	std::string label;
	std::string label_tex;

	// The new permutation domain is partitioned into three parts:
	// A) the domain of M1,
	// B) the domain of M2,
	// C) the cartesian product of domain 1 with domain 2

	// a group element is represented as a pair (a,b)
	// with a in M1 and b in M2.
	// a and b are stored consecutively


	int degree_of_matrix_group1; // |A|
	int dimension_of_matrix_group1;

	int degree_of_matrix_group2; // |B|
	int dimension_of_matrix_group2;

	int degree_of_product_action; // = |C| = |A| * |B|

	int degree_overall; // = |A| + |B| + |C|

	int low_level_point_size; // = dimension_of_matrix_group1 + dimension_of_matrix_group2
	int make_element_size; // = M1->make_element_size + M2->make_element_size
	int elt_size_int; // = M1->elt_size_int + M2->elt_size_int

	int *perm_offset_i; // [3]
	// perm_offset_i[] is the start of A, B, C respectively.
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

	int base_length; // = base_len_in_component1 + base_len_in_component2
	long int *the_base; // the union of the two bases, as a subset of A \cup B.
	int *the_transversal_length;

	data_structures::page_storage *Page_storage;

	direct_product();
	~direct_product();
	void init(
			algebra::matrix_group *M1,
			algebra::matrix_group *M2,
			int verbose_level);
	long int element_image_of(
			int *Elt, long int a, int verbose_level);
	void element_one(
			int *Elt);
	int element_is_one(
			int *Elt);
	void element_mult(
			int *A, int *B, int *AB, int verbose_level);
	void element_move(
			int *A, int *B, int verbose_level);
	void element_invert(
			int *A, int *Av, int verbose_level);
	int offset_i(
			int i);
	// offset_i[] is the beginning of the i-th component
	// in the element representation
	void element_pack(
			int *Elt, uchar *elt);
	void element_unpack(
			uchar *elt, int *Elt);
	void put_digit(
			uchar *elt, int f, int i, int d);
	int get_digit(
			uchar *elt, int f, int i);
	void make_element(
			int *Elt, int *data, int verbose_level);
	void element_print_easy(
			int *Elt, std::ostream &ost);
	void element_print_easy_latex(
			int *Elt, std::ostream &ost);
	void compute_base_and_transversals(
			int verbose_level);
	void make_strong_generators_data(
			int *&data,
			int &size, int &nb_gens, int verbose_level);
	void lift_generators(
			groups::strong_generators *SG1,
			groups::strong_generators *SG2,
			actions::action *A,
			groups::strong_generators *&SG3,
			int verbose_level);
};


// #############################################################################
// linear_group_description.cpp
// #############################################################################

//! description of a linear group from the command line


class linear_group_description {
public:

	// TABLES/linear_group_1.tex


	int f_projective;
	int f_general;
	int f_affine;
	int f_GL_d_q_wr_Sym_n;
	int f_orthogonal;
	int f_orthogonal_p;
	int f_orthogonal_m;
	int GL_wreath_Sym_d;
	int GL_wreath_Sym_n;


	int f_n;
	int n;

	std::string input_q;

	field_theory::finite_field *F;


	int f_semilinear;
	int f_special;



	// TABLES/linear_group_2.tex

	// induced actions and subgroups:

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

	int f_on_tensors;
	int f_on_rank_one_tensors;

	int f_subgroup_by_generators;
	std::string subgroup_order_text;
	int nb_subgroup_generators;
	std::string subgroup_generators_label;

	int f_Janko1;

	int f_export_magma;


	int f_import_group_of_plane;
	std::string import_group_of_plane_label;


	int f_lex_least_base;


	linear_group_description();
	~linear_group_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();

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
	field_theory::finite_field *F;
	int f_semilinear;

	std::string label;
	std::string label_tex;

	groups::strong_generators *initial_strong_gens;
	actions::action *A_linear;
	algebra::matrix_group *Mtx;

	int f_has_strong_generators;
	groups::strong_generators *Strong_gens;
	actions::action *A2;
	int vector_space_dimension;
	int q;

	int f_has_nice_gens;
	data_structures_groups::vector_ge *nice_gens;

	linear_group();
	~linear_group();
	void linear_group_init(
			linear_group_description *description,
		int verbose_level);
	void linear_group_import(
			int verbose_level);
	void linear_group_import_group_of_plane(
			int verbose_level);
	void linear_group_create(
			int verbose_level);
	int linear_group_apply_modification(
			linear_group_description *description,
			int verbose_level);

	void init_PGL2q_OnConic(
			int verbose_level);
	void init_wedge_action(
			int verbose_level);
	void init_wedge_action_detached(
			int verbose_level);
	void init_monomial_group(
			int verbose_level);
	void init_diagonal_group(
			int verbose_level);
	void init_singer_group(
			int singer_power, int verbose_level);
	void init_singer_group_and_frobenius(
			int singer_power, int verbose_level);
	void init_null_polarity_group(
			int verbose_level);
	void init_borel_subgroup_upper(
			int verbose_level);
	void init_identity_subgroup(
			int verbose_level);
	void init_symplectic_group(
			int verbose_level);
	void init_subfield_structure_action(
			int s, int verbose_level);
	void init_orthogonal_group(
			int epsilon, int verbose_level);
	void init_subgroup_from_file(
			std::string &subgroup_fname,
			std::string &subgroup_label,
			int verbose_level);
	void init_subgroup_by_generators(
			std::string &subgroup_label,
			std::string &subgroup_order_text,
			int nb_subgroup_generators,
			int *subgroup_generators_data,
			int verbose_level);
	void init_subgroup_Janko1(
			int verbose_level);

};



// #############################################################################
// permutation_group_create.cpp
// #############################################################################

//! a domain for permutation groups whose elements are given in the permutation representation

class permutation_group_create {

public:
	permutation_group_description *Descr;

	std::string label;
	std::string label_tex;

	actions::action *A_initial;

	int f_has_strong_generators;
	groups::strong_generators *Strong_gens;
	actions::action *A2;

	int f_has_nice_gens;
	data_structures_groups::vector_ge *nice_gens;


	permutation_group_create();
	~permutation_group_create();
	void permutation_group_init(
			permutation_group_description *description,
			int verbose_level);
	void init_subgroup_by_generators(
			std::string &subgroup_label,
			std::string &subgroup_order_text,
			int nb_subgroup_generators,
			std::string &subgroup_generators_label,
			int verbose_level);


};


// #############################################################################
// permutation_group_description.cpp
// #############################################################################

//! a domain for permutation groups whose elements are given in the permutation representation

class permutation_group_description {

public:

	// TABLES/permutation_group.tex

	int degree;
	permutation_group_type type;

	int f_bsgs;
	std::string bsgs_label;
	std::string bsgs_label_tex;
	std::string bsgs_order_text;
	std::string bsgs_base;
	int bsgs_nb_generators;
	std::string bsgs_generators;


	int f_subgroup_by_generators;
	std::string subgroup_label;
	std::string subgroup_order_text;
	int nb_subgroup_generators;
	std::string subgroup_generators_label;

	permutation_group_description();
	~permutation_group_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


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

	data_structures::page_storage *Page_storage;

	permutation_representation_domain();
	~permutation_representation_domain();
	void allocate();
	void init_product_action(
			int m, int n,
		int page_length_log, int verbose_level);
	void init(
			int degree, int page_length_log, int verbose_level);
	void init_data(
			int page_length_log, int verbose_level);
	void init_with_base(
			int degree,
		int base_length, int *base, int page_length_log,
		actions::action &A, int verbose_level);
	void transversal_rep(
			int i, int j, int *Elt,
			int verbose_level);
	void one(
			int *Elt);
	int is_one(
			int *Elt);
	void mult(
			int *A, int *B, int *AB);
	void copy(
			int *A, int *B);
	void invert(
			int *A, int *Ainv);
	void unpack(
			uchar *elt, int *Elt);
	void pack(
			int *Elt, uchar *elt);
	void print(
			int *Elt, std::ostream &ost);
	void print_with_print_point_function(
			int *Elt,
			std::ostream &ost,
			void (*point_label)(
					std::stringstream &sstr, long int pt, void *data),
			void *point_label_data);
	void code_for_make_element(
			int *Elt, int *data);
	void print_for_make_element(
			int *Elt, std::ostream &ost);
	void print_for_make_element_no_commas(
			int *Elt, std::ostream &ost);
	void print_with_action(
			actions::action *A, int *Elt, std::ostream &ost);
	void make_element(
			int *Elt, int *data, int verbose_level);

};


// #############################################################################
// permutation_representation.cpp
// #############################################################################

//! homomorphism to a permutation group

class permutation_representation {

public:
	actions::action *A_original;
	int f_stay_in_the_old_action;
	int nb_gens;
	data_structures_groups::vector_ge *gens;
		// the original generators in action A_original
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

	data_structures::page_storage *Page_storage;

	int *Elts;
		// [nb_gens * elt_size_int], the generators in the induced action


	permutation_representation();
	~permutation_representation();
	void init(
			actions::action *A_original,
			int f_stay_in_the_old_action,
			data_structures_groups::vector_ge *gens,
			int *Perms, int degree,
			int verbose_level);
		// Perms is degree x nb_gens
	long int element_image_of(
			int *Elt, long int a, int verbose_level);
	void element_one(
			int *Elt);
	int element_is_one(
			int *Elt);
	void element_mult(
			int *A, int *B, int *AB, int verbose_level);
	void element_move(
			int *A, int *B, int verbose_level);
	void element_invert(
			int *A, int *Av, int verbose_level);
	void element_pack(
			int *Elt, uchar *elt);
	void element_unpack(
			uchar *elt, int *Elt);
	void element_print_for_make_element(
			int *Elt, std::ostream &ost);
	void element_print_easy(
			int *Elt, std::ostream &ost);
	void element_print_latex(
			int *Elt, std::ostream &ost);
};


// #############################################################################
// polarity_extension.cpp
// #############################################################################

//! the extension of a matrix group by a polarity

class polarity_extension {

public:

	algebra::matrix_group *M;
	field_theory::finite_field *F;

	geometry::projective_space *P;
	geometry::polarity *Polarity;

	std::string label;
	std::string label_tex;


	actions::action *A_on_points;
	actions::action *A_on_hyperplanes;


	// The new permutation domain is partitioned into three parts:
	// A) the two element set {0,1},
	// B) the new domain obtained from P,
	// the projective space on which M acts.

	// a group element is represented as a pair (a,b)
	// with a in M1 and b in {0,1}.
	// a and b are stored consecutively


	int degree_of_matrix_group; // |B|
	int dimension_of_matrix_group;


	int degree_overall; // = 2 + |B|

	//int low_level_point_size; // = dimension_of_matrix_group
	int make_element_size; // = M->make_element_size + 1
	int elt_size_int; // = M->elt_size_int + 1

	int *element_coding_offset; // [2]
	int *perm_offset_i; // [2]
	// perm_offset_i[] is the start of A, B respectively.
	int *tmp_Elt1;
	int *tmp_matrix1; // [n * n]
	int *tmp_matrix2; // [n * n]
	int *tmp_vector; // [n]

	int bits_per_digit;

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

	data_structures::page_storage *Page_storage;

	polarity_extension();
	~polarity_extension();
	void init(
			actions::action *A,
			geometry::projective_space *P,
			geometry::polarity *Polarity,
			int verbose_level);
	long int element_image_of(
			int *Elt, long int a, int verbose_level);
	void element_one(
			int *Elt);
	int element_is_one(
			int *Elt);
	void element_mult(
			int *A, int *B, int *AB, int verbose_level);
	void element_move(
			int *A, int *B, int verbose_level);
	void compute_images_rho_A_rho(
			int *Mtx, int nb_rows, int *A_Elt, int verbose_level);
	void create_rho_A_rho(
			int *A_Elt, int *data,
			int verbose_level);
	void element_inverse_conjugate_by_polarity(
			int *A_Elt, int *rho_A_rho, int verbose_level);
	void element_conjugate_by_polarity(
			int *A_Elt, int *rho_A_rho, int verbose_level);
	void element_invert(
			int *A, int *Av, int verbose_level);
	void element_pack(
			int *Elt, uchar *elt);
	void element_unpack(
			uchar *elt, int *Elt);
	void put_digit(
			uchar *elt, int f, int i, int d);
	int get_digit(
			uchar *elt, int f, int i);
	void make_element(
			int *Elt, int *data, int verbose_level);
	void element_print_easy(
			int *Elt, std::ostream &ost);
	void element_print_easy_latex(
			int *Elt, std::ostream &ost);
	void element_code_for_make_element(
			int *Elt, int *data);
	void element_print_for_make_element(
			int *Elt, std::ostream &ost);
	void element_print_for_make_element_no_commas(
			int *Elt, std::ostream &ost);
	void compute_base_and_transversals(
			int verbose_level);
	void make_strong_generators_data(
			int *&data,
			int &size, int &nb_gens, int verbose_level);
	void unrank_point(
			long int rk, int *v, int verbose_level);
	long int rank_point(
			int *v, int verbose_level);

};




// #############################################################################
// wreath_product.cpp
// #############################################################################

//! the wreath product group GL(d,q) wreath Sym(n)

class wreath_product {

public:
	algebra::matrix_group *M;
	actions::action *A_mtx;
	field_theory::finite_field *F;
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

	data_structures::page_storage *Page_storage;

	uint32_t *rank_one_tensors; // [nb_rank_one_tensors]
	long int *rank_one_tensors_in_PG; // [nb_rank_one_tensors]
		// rank_one_tensors_in_PG[i] = affine_rank_to_PG_rank(rank_one_tensors[i]);
	long int *rank_one_tensors_in_PG_sorted; // [nb_rank_one_tensors]
	int nb_rank_one_tensors;

	char *TR; // [degree_of_tensor_action + 1]
	uint32_t *Prev; // [degree_of_tensor_action + 1]

	wreath_product();
	~wreath_product();
	void init_tensor_wreath_product(
			algebra::matrix_group *M,
			actions::action *A_mtx, int nb_factors,
			int verbose_level);
	void compute_tensor_ranks(
			int verbose_level);
	void unrank_point(
			long int a, int *v, int verbose_level);
	long int rank_point(
			int *v, int verbose_level);
	long int element_image_of(
			int *Elt, long int a, int verbose_level);
	void element_image_of_low_level(
			int *Elt,
			int *input, int *output, int verbose_level);
		// we assume that we are in the tensor product domain
	void element_one(
			int *Elt);
	int element_is_one(
			int *Elt);
	void element_mult(
			int *A, int *B, int *AB, int verbose_level);
	void element_move(
			int *A, int *B, int verbose_level);
	void element_invert(
			int *A, int *Av, int verbose_level);
	void compute_induced_permutation(
			int *Elt, int *perm);
	void apply_permutation(
			int *Elt,
			int *v_in, int *v_out, int verbose_level);
	int offset_i(
			int i);
	void create_matrix(
			int *Elt, int *A, int verbose_level);
		// uses A1, A2
	void element_pack(
			int *Elt, uchar *elt);
	void element_unpack(
			uchar *elt, int *Elt);
	void put_digit(
			uchar *elt, int f, int i, int j, int d);
	int get_digit(
			uchar *elt, int f, int i, int j);
	void make_element_from_one_component(
			int *Elt,
			int f, int *Elt_component);
	void make_element_from_permutation(
			int *Elt, int *perm);
	void make_element(
			int *Elt, int *data, int verbose_level);
	void element_print_for_make_element(
			int *Elt, std::ostream &ost);
	void element_print_easy(
			int *Elt, std::ostream &ost);
	void element_print_latex(
			int *Elt, std::ostream &ost);
	void compute_base_and_transversals(
			int verbose_level);
	void make_strong_generators_data(
			int *&data,
			int &size, int &nb_gens, int verbose_level);
	void report_rank_one_tensors(
			std::ostream &ost, int verbose_level);
	void create_all_rank_one_tensors(
			uint32_t *&rank_one_tensors,
			int &nb_rank_one_tensors, int verbose_level);
	uint32_t tensor_affine_rank(
			int *tensor);
	void tensor_affine_unrank(
			int *tensor, uint32_t rk);
	long int tensor_PG_rank(
			int *tensor);
	void tensor_PG_unrank(
			int *tensor, long int PG_rk);
	long int affine_rank_to_PG_rank(
			uint32_t affine_rk);
	uint32_t PG_rank_to_affine_rank(
			long int PG_rk);
	void save_rank_one_tensors(
			int verbose_level);
	void compute_tensor_ranks(
			char *&TR, uint32_t *&Prev, int verbose_level);
	void report(
			std::ostream &ost, int verbose_level);
	void compute_permutations_and_write_to_file(
			groups::strong_generators* SG,
			actions::action* A,
			int*& result,
			int &nb_gens, int &degree,
			int nb_factors,
			int verbose_level);
	void make_fname(
			std::string &fname, int nb_factors, int h, int b);
	int test_if_file_exists(
			int nb_factors, int h, int b);
	void orbits_using_files_and_union_find(
			groups::strong_generators* SG,
			actions::action* A,
			int *&result,
			int &nb_gens, int &degree,
			int nb_factors,
			int verbosity);
	void orbits_restricted(
			groups::strong_generators* SG,
			actions::action* A,
			int *&result,
			int &nb_gens, int &degree,
			int nb_factors,
			std::string &orbits_restricted_fname,
			int verbose_level);
	void orbits_restricted_compute(
			groups::strong_generators* SG,
			actions::action* A,
			int *&result,
			int &nb_gens, int &degree,
			int nb_factors,
			std::string &orbits_restricted_fname,
			int verbose_level);
};




}}}




#endif /* SRC_LIB_LAYER3_GROUP_ACTIONS_GROUP_CONSTRUCTIONS_GROUP_CONSTRUCTIONS_H_ */
