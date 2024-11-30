// induced_actions.h
//
// Anton Betten
//
// moved here from action.h: July 28, 2018
// based on action.h which was started:  August 13, 2005



#ifndef ORBITER_SRC_LIB_GROUP_ACTIONS_INDUCED_ACTIONS_INDUCED_ACTIONS_H_
#define ORBITER_SRC_LIB_GROUP_ACTIONS_INDUCED_ACTIONS_INDUCED_ACTIONS_H_



namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


// #############################################################################
// action_by_conjugation.cpp
// #############################################################################


//! induced action by conjugation on the elements of a given group



class action_by_conjugation {
public:
	groups::sims *Base_group;
	int f_ownership;
	long int goi;
	
	int *Elt1;
	int *Elt2;
	int *Elt3;

	action_by_conjugation();
	~action_by_conjugation();
	void init(
			groups::sims *Base_group,
			int f_ownership, int verbose_level);
	long int compute_image(
			actions::action *A,
			int *Elt, long int i, int verbose_level);
	long int rank(
			int *Elt);
	long int multiply(
			actions::action *A,
			long int i, long int j, int verbose_level);
};

// #############################################################################
// action_by_representation.cpp
// #############################################################################


//! induced  action of PSL(2,q) on a conic (the only type implemented so far)


class action_by_representation {
public:
	enum representation_type type;
	int n;
	int q;
	actions::action *A;
	algebra::matrix_group *M;
	field_theory::finite_field *F;
	int low_level_point_size;
	int degree;
	
	int dimension; // 
	int *v1; // [dimension]
	int *v2; // [dimension]
	int *v3; // [dimension]
	
	action_by_representation();
	~action_by_representation();
	void init_action_on_conic(
			actions::action *A, int verbose_level);
	long int compute_image_int(
			int *Elt,
			long int a, int verbose_level);
	void compute_image_int_low_level(
			int *Elt,
			int *input, int *output,
		int verbose_level);
	void unrank_point(
			long int a, int *v, int verbose_level);
	long int rank_point(
			int *v, int verbose_level);

};

// #############################################################################
// action_by_restriction.cpp
// #############################################################################


//! restricted action on an invariant subset




class action_by_restriction {
public:
	int nb_points;
	long int *points; // [nb_points]
	long int *points_sorted; // [nb_points]
	long int *perm_inv; // [nb_points]
	int f_single_orbit;
	int pt;
	int idx_of_root_node;

	action_by_restriction();
	~action_by_restriction();
	void init_single_orbit_from_schreier_vector(
			data_structures_groups::schreier_vector *Schreier_vector,
			int pt, int verbose_level);
	void init(
			int nb_points,
			long int *points, int verbose_level);
		// the array points must be ordered
	long int original_point(
			long int pt);
	long int restricted_point_idx(
			long int pt);
	long int compute_image(
			actions::action *A,
			int *Elt, long int i, int verbose_level);
};

// #############################################################################
// action_by_right_multiplication.cpp
// #############################################################################

//! induced action on a the set of elements of a group by right multiplication



class action_by_right_multiplication {
public:
	groups::sims *Base_group;
	int f_ownership;
	int goi;
	
	int *Elt1;
	int *Elt2;

	action_by_right_multiplication();
	~action_by_right_multiplication();
	void init(
			groups::sims *Base_group,
			int f_ownership, int verbose_level);
	long int compute_image(
			actions::action *A, int *Elt,
			long int i,
		int verbose_level);
};

// #############################################################################
// action_by_subfield_structure.cpp
// #############################################################################

//! induced action on the vector space arising from a field over a subfield


class action_by_subfield_structure {
public:
	int n;
	int Q;
	const char *poly_q;
	int q;
	int s;
	int m; // n * s
	int *v1; // [m]
	int *v2; // [m]
	int *v3; // [m]
	
	actions::action *AQ;
	actions::action *Aq;

	algebra::matrix_group *MQ;
	field_theory::finite_field *FQ;
	algebra::matrix_group *Mq;
	field_theory::finite_field *Fq;

	field_theory::subfield_structure *S;

	int *Eltq;
	int *Mtx; // [m * m]

	int low_level_point_size; // = m
	int degree;

	action_by_subfield_structure();
	~action_by_subfield_structure();
	void init(
			actions::action &A,
			field_theory::finite_field *Fq,
			int verbose_level);
	long int compute_image_int(
			actions::action &A, int *Elt,
			long int a, int verbose_level);
	void compute_image_int_low_level(
			actions::action &A, int *Elt,
			int *input, int *output,
		int verbose_level);
};

// #############################################################################
// action_on_andre.cpp
// #############################################################################

//! induced action on the elements of a projective plane constructed via Andre / Bruck / Bose


class action_on_andre {
public:

	actions::action *An;
	actions::action *An1;
	geometry::finite_geometries::andre_construction *Andre;
	int k, n, q;
	int k1, n1;
	int N; // number of points in the plane
	int degree;
	int *coords1; // [(k + 1) * (n + 1)];
	int *coords2; // [(k + 1) * (n + 1)];
	int *coords3; // [k * n];

	action_on_andre();
	~action_on_andre();
	void init(
			actions::action *An,
			actions::action *An1,
			geometry::finite_geometries::andre_construction *Andre,
			int verbose_level);
	long int compute_image(
			int *Elt, long int i,
		int verbose_level);
	long int compute_image_of_point(
			int *Elt, long int pt_idx,
		int verbose_level);
	long int compute_image_of_line(
			int *Elt, long int line_idx,
		int verbose_level);
};

// #############################################################################
// action_on_bricks.cpp
// #############################################################################

//! related to a problem of Neil Sloane


class action_on_bricks {
public:

	actions::action *A;
	combinatorics::brick_domain *B;
	int degree;
	int f_linear_action;

	action_on_bricks();
	~action_on_bricks();
	void init(
			actions::action *A,
			combinatorics::brick_domain *B,
			int f_linear_action,
		int verbose_level);
	long int compute_image(
			int *Elt, long int i,
		int verbose_level);
	long int compute_image_linear_action(
			int *Elt, long int i,
		int verbose_level);
	long int compute_image_permutation_action(
			int *Elt, long int i,
		int verbose_level);
};



// #############################################################################
// action_on_cosets_of_subgroup.cpp
// #############################################################################

//! induced action on the right cosets of a subgroup by right multiplication


class action_on_cosets_of_subgroup {
public:

	actions::action *A;

	groups::strong_generators *Subgroup_gens_H;
	groups::strong_generators *Subgroup_gens_G;

	groups::sims *Sims_H;


	long int degree;
	data_structures_groups::vector_ge *coset_reps;
	data_structures_groups::vector_ge *coset_reps_inverse;

	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *Elt4;

	action_on_cosets_of_subgroup();
	~action_on_cosets_of_subgroup();
	void init(
			actions::action *A,
			groups::strong_generators *Subgroup_gens_H,
			groups::strong_generators *Subgroup_gens_G,
			int verbose_level);
	long int compute_image(
			int *Elt, long int i, int verbose_level);

};

// #############################################################################
// action_on_cosets.cpp
// #############################################################################

//! induced action on the cosets of a subspace by right multiplication


class action_on_cosets {
public:
	actions::action *A_linear;
	field_theory::finite_field *F;
	int dimension_of_subspace;
	int n;
	int *subspace_basis; // [dimension_of_subspace * n]
	int *base_cols; // [dimension_of_subspace] 
		// the pivot column for the subspace basis
		// to be used if a vector v[len] 
		// is reduced modulo a subspace
	
	int f_lint;
	int nb_points;
	int *Points; // ordered list of point ranks
	long int *lint_Points; // ordered list of point ranks

	int *v1;
	int *v2;
	
	void (*unrank_point)(int *v, int a, void *data);
	int (*rank_point)(int *v, void *data);
	void (*unrank_point_lint)(int *v, long int a, void *data);
	long int (*rank_point_lint)(int *v, void *data);
	void *rank_unrank_data;

	action_on_cosets();
	~action_on_cosets();
	void init(
			int nb_points, int *Points,
			actions::action *A_linear,
		field_theory::finite_field *F,
		int dimension_of_subspace, 
		int n, 
		int *subspace_basis, 
		int *base_cols, 
		void (*unrank_point)(int *v, int a, void *data), 
		int (*rank_point)(int *v, void *data), 
		void *rank_unrank_data, 
		int verbose_level);
	void init_lint(
			int nb_points, long int *Points,
			actions::action *A_linear,
		field_theory::finite_field *F,
		int dimension_of_subspace,
		int n,
		int *subspace_basis,
		int *base_cols,
		void (*unrank_point)(int *v, long int a, void *data),
		long int (*rank_point)(int *v, void *data),
		void *rank_unrank_data,
		int verbose_level);
	void reduce_mod_subspace(
			int *v, int verbose_level);
	long int compute_image(
			int *Elt, long int i, int verbose_level);

};

// #############################################################################
// action_on_determinant.cpp
// #############################################################################

//! induced action on the determinant of a group of matrices (used to compute the subgroup PSL)


class action_on_determinant {
public:
	algebra::matrix_group *M;
	int f_projective;
	int m;
	int q;
	int degree;
		// gcd(m, q - 1) if f_projective
		// q - 1 otherwise
	
	action_on_determinant();
	~action_on_determinant();
	void init(
			actions::action &A,
			int f_projective, int m, int verbose_level);
	long int compute_image(
			actions::action *A, int *Elt, long int i,
		int verbose_level);
};

// #############################################################################
// action_on_factor_space.cpp
// #############################################################################

//! induced action on the factor space of a vector space modulo a subspace


class action_on_factor_space {
public:
	linear_algebra::vector_space *VS;


	// VS->dimension = length of vectors in large space

	int *subspace_basis; // [subspace_basis_size * VS->dimension]
	int subspace_basis_size;
	int *base_cols; // [subspace_basis_size] 
		// the pivot column for the subspace basis
		// to be used if a vector v[len] is reduced modulo a subspace

	long int degree;
		// the number of projective points in the small space 
		// (i.e., the factor space)
		// (q^factor_space_len - 1) / (q - 1), 
		// as computed by compute_degree();
	long int large_degree;
		// the number of projective points in the large space
		// (q^len - 1) / (q - 1), 
		// as computed by compute_large_degree();
	
	int factor_space_len;
		// = VS->dimension - subspace_basis_size

	int *embedding;
		// [factor_space_len]
		// the list of columns that are not pivot columns, 
		// i.e. not in base_cols[] 
		// this is the set-theoretic complement of base_cols
	long int *projection_table;
		// [nb_points]
		// projection_table[i] = j 
		// means that the Gauss reduced vector in 
		// the coset of point_list[i] 
		// is in coset_reps_Gauss[j]
	long int *preimage_table; // [degree]
	int *tmp; // [factor_space_len] 
	int *Tmp1; // [VS->dimension]
	int *Tmp2; // [VS->dimension]
	int f_tables_have_been_computed;

	int f_table_mode;

	int nb_cosets;
	long int *coset_reps_Gauss;
		// [nb_cosets]
		// ordered list of Gauss reduced coset representatives
		// the entries are ranks of vectors in the large space

	int *tmp_w; // [VS->dimension] temporary vector for use in rank
	int *tmp_w1;
		// [subspace_basis_size] 
		// temporary vector for lexleast_element_in_coset
	int *tmp_v1;
		// [len] temporary vector 
		// for use in lexleast_element_in_coset
	int *tmp_v2;
		// [len] temporary vector 
		// for use in lexleast_element_in_coset
	
	action_on_factor_space();
	~action_on_factor_space();
	void init_light(
			linear_algebra::vector_space *VS,
		actions::action &A_base, actions::action &A,
		long int *subspace_basis_ranks, int subspace_basis_size,
		int verbose_level);
	void init_by_rank_table_mode(
			linear_algebra::vector_space *VS,
			actions::action &A_base, actions::action &A,
		long int *subspace_basis_ranks, int subspace_basis_size,
		long int *point_list, int nb_points,
		int verbose_level);
	void print_coset_table();
	void print_projection_table(
			long int *point_list, int nb_points);
	void init_coset_table(
			long int *point_list, int nb_points,
			int verbose_level);
	void init_by_rank(
			linear_algebra::vector_space *VS,
			actions::action &A_base, actions::action &A,
		long int *subspace_basis_ranks, int subspace_basis_size,
		int f_compute_tables, int verbose_level);
	void init_from_coordinate_vectors(
			linear_algebra::vector_space *VS,
			actions::action &A_base, actions::action &A,
		int *subspace_basis, int subspace_basis_size, 
		int f_compute_tables, int verbose_level);
	void init2(
			actions::action &A_base, actions::action &A,
		int f_compute_tables,
		int verbose_level);
	void compute_projection_table(
			int verbose_level);
	long int compute_degree();
	long int compute_large_degree();
	void list_all_elements();
	void reduce_mod_subspace(
			int *v, int verbose_level);
	long int lexleast_element_in_coset(
			long int rk, int verbose_level);
		// This function computes the lexleast 
		// element in the coset modulo the subspace.
		// It does so by looping over all q^subspace_basis_size 
		// elements in the subspace and ranking the corresponding 
		// vector in the large space using rank_in_large_space(v2).
	long int project_onto_Gauss_reduced_vector(
			long int rk, int verbose_level);
	long int project(
			long int rk, int verbose_level);
		// unranks the vector rk, and reduces it 
		// modulo the subspace basis.
		// The non-pivot components are considered 
		// as a vector in F_q^factor_space_len 
		// and ranked using the rank function for projective space.
		// This rank is returned.
		// If the vector turns out to lie in the 
		// subspace, -1 is returned.
	long int preimage(
			long int rk, int verbose_level);
	void embed(
			int *from, int *to);
	void unrank(
			int *v, long int rk, int verbose_level);
	long int rank(
			int *v, int verbose_level);
	void unrank_in_large_space(
			int *v, long int rk);
	long int rank_in_large_space(
			int *v);
	void unrank_in_small_space(
			int *v, long int rk);
	long int rank_in_small_space(
			int *v);
	long int compute_image(
			actions::action *A,
			int *Elt, long int i, int verbose_level);
};

// #############################################################################
// action_on_flags.cpp
// #############################################################################


//! induced action on flags


class action_on_flags {
public:
	actions::action *A;
	int n;
	field_theory::finite_field *F;
	int *type;
	int type_len;
	geometry::other_geometry::flag *Flag;
	algebra::matrix_group *M;
	int degree;
	int *M1;
	int *M2;
	
	action_on_flags();
	~action_on_flags();
	void init(
			actions::action *A,
			int *type, int type_len,
		int verbose_level);
	long int compute_image(
			int *Elt,
			long int i, int verbose_level);
};

// #############################################################################
// action_on_galois_group.cpp:
// #############################################################################

//! induced action on the galois group (used to compute the projectivity subgroup of a collineation group)


class action_on_galois_group {
public:
	actions::action *A;
	algebra::matrix_group *M;
	int m;
	int q;
	int degree;

	action_on_galois_group();
	~action_on_galois_group();
	void init(
			actions::action *A,
			int m, int verbose_level);
	long int compute_image(
			int *Elt, long int i,
		int verbose_level);
};


// #############################################################################
// action_on_grassmannian.cpp
// #############################################################################

//! induced action on the grassmannian (subspaces of a fixed dimension of a vector space)


class action_on_grassmannian {
public:
	int n;
	int k;
	int q;
	field_theory::finite_field *F;
	int low_level_point_size;
	
	actions::action *A;
	geometry::projective_geometry::grassmann *G;
	int *M1;
	int *M2;

	int f_embedding;
	int big_n;
	geometry::projective_geometry::grassmann_embedded *GE;
	int *subspace_basis; // [n * big_n]
	int *subspace_basis2; // [n * big_n]
	
	ring_theory::longinteger_object degree_as_text;
	long int degree;
	int max_string_length;
	

	int f_has_print_function;
	void (*print_function)(
			std::ostream &ost, long int a, void *data);
	void *print_function_data;

	action_on_grassmannian();
	~action_on_grassmannian();
	void init(
			actions::action &A,
			geometry::projective_geometry::grassmann *G, int verbose_level);
	void add_print_function(
			void (*print_function)(
					std::ostream &ost, long int a, void *data),
			void *print_function_data,
			int verbose_level);
	void init_embedding(
			int big_n, int *ambient_space,
		int verbose_level);
	void unrank(
			long int i, int *v, int verbose_level);
	long int rank(
			int *v, int verbose_level);
	void compute_image_longinteger(
			actions::action *A,
			int *Elt,
			ring_theory::longinteger_object &i,
			ring_theory::longinteger_object &j,
		int verbose_level);
	long int compute_image_int(
			actions::action *A, int *Elt,
		long int i, int verbose_level);
	long int compute_image_int_ordinary(
			actions::action *A, int *Elt,
		long int i, int verbose_level);
	long int compute_image_int_embedded(
			actions::action *A, int *Elt,
		long int i, int verbose_level);
	void print_point(
			long int a, std::ostream &ost);
};

// #############################################################################
// action_on_homogeneous_polynomials.cpp
// #############################################################################


//! induced action on the set of homogeneous polynomials over a finite field


class action_on_homogeneous_polynomials {
public:
	int n; // the dimension M->n
	int q;
	actions::action *A;
	ring_theory::homogeneous_polynomial_domain *HPD;
	algebra::matrix_group *M;
	field_theory::finite_field *F;
	int low_level_point_size;
	int degree;

	// wedge product
	int dimension; // = HPD->nb_monomials
	int *v1; // [dimension]
	int *v2; // [dimension]
	int *v3; // [dimension]
	int *Elt1;
	
	int f_invariant_set;
	int *Equations;
	int nb_equations;

	data_structures::int_matrix *Table_of_equations;

	action_on_homogeneous_polynomials();
	~action_on_homogeneous_polynomials();
	void init(
			actions::action *A,
			ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level);
	void init_invariant_set_of_equations(
			int *Equations,
		int nb_equations, int verbose_level);
	void unrank_point(
			int *v, long int rk);
	long int rank_point(
			int *v);
	long int compute_image_int(
			int *Elt, long int a, int verbose_level);
	void compute_image_int_low_level(
		int *Elt, int *input, int *output, int verbose_level);
	void compute_representation(
		int *Elt, int *M, int verbose_level);
};

// #############################################################################
// action_on_interior_direct_product.cpp
// #############################################################################


//! induced action on the interior direct product


class action_on_interior_direct_product {
public:
	actions::action *A;
	int nb_rows;
	int nb_cols;
	int degree;


	action_on_interior_direct_product();
	~action_on_interior_direct_product();
	void init(
			actions::action *A,
			int nb_rows, int verbose_level);
	long int compute_image(
			int *Elt, long int a, int verbose_level);
};



// #############################################################################
// action_on_k_subsets.cpp
// #############################################################################

//! induced action on k-subsets of a set of size n


class action_on_k_subsets {
public:
	actions::action *A;
	int k;
	int degree;
	int *set1; // [k]
	int *set2; // [k]

	action_on_k_subsets();
	~action_on_k_subsets();
	void init(
			actions::action *A, int k, int verbose_level);
	long int compute_image(
			int *Elt, long int i, int verbose_level);
};


// #############################################################################
// action_on_module.cpp
// #############################################################################


//! induced action on a module


class action_on_module {
public:

	actions::action *A;
	int n;
	int q;
	algebra::matrix_group *M;
	field_theory::finite_field *F;
	int low_level_point_size; // = module_dimension_m
	//long int degree;

	algebraic_geometry::surface_object *SO;
	int *module_basis; // [module_dimension_m * module_dimension_n]
	int module_dimension_m;
	int module_dimension_n;
	double *module_basis_base_transposed; // [module_dimension_n * module_dimension_m]

	int *module_basis_base_cols; // [module_dimension_n]
	int *module_basis_rref; // [module_dimension_m * module_dimension_n]
	int *module_basis_transformation; // [module_dimension_m * module_dimension]

	int *v1; // [module_dimension_n]
	int *v2; // [module_dimension_n]
	int *perm; // [module_dimension_n]

	actions::action *A_on_the_lines;
	actions::action *A_on_module;

	action_on_module();
	~action_on_module();
	void init_action_on_module(
			algebraic_geometry::surface_object *SO,
			actions::action *A_on_the_lines,
			std::string &module_type,
			int *module_basis, int module_dimension_m, int module_dimension_n,
			int verbose_level);
	void compute_image_int_low_level(
			int *Elt, int *input, int *output,
		int verbose_level);
};



// #############################################################################
// action_on_orbits.cpp
// #############################################################################

//! induced action on the set of orbits (usually by the normalizer)



class action_on_orbits {
public:
	actions::action *A;
	groups::schreier *Sch;
	int f_play_it_safe;
	int degree;
	
	action_on_orbits();
	~action_on_orbits();
	void init(
			actions::action *A,
			groups::schreier *Sch,
			int f_play_it_safe,
		int verbose_level);
	long int compute_image(
			int *Elt, long int i, int verbose_level);
};

// #############################################################################
// action_on_orthogonal.cpp
// #############################################################################


//! induced action on the orthogonal geometry


class action_on_orthogonal {
public:
	actions::action *original_action;
		// needs original_action->Group_element->element_image_of_low_level
	orthogonal_geometry::orthogonal *O;
	int *v1;
	int *v2;
	int *w1;
	int *w2;
	int f_on_points;
	int f_on_lines;
	int f_on_points_and_lines;
	int low_level_point_size;
	int degree;
	
	action_on_orthogonal();
	~action_on_orthogonal();
	void init(
			actions::action *original_action,
			orthogonal_geometry::orthogonal *O,
		int f_on_points, int f_on_lines,
		int f_on_points_and_lines,
		int verbose_level);
	void unrank_point(
			int *v, long int rk);
	long int rank_point(
			int *v);
	long int map_a_point(
			int *Elt, long int i, int verbose_level);
	long int map_a_line(
			int *Elt, long int i, int verbose_level);
	long int compute_image_int(
			int *Elt, long int i, int verbose_level);
};

// #############################################################################
// action_on_set_partitions.cpp:
// #############################################################################


//! induced action on a set partitions.



class action_on_set_partitions {
public:
	int nb_set_partitions;
	int universal_set_size;
	int partition_class_size;
	int nb_parts;
	actions::action *A;
	int *v1;
	int *v2;

	action_on_set_partitions();
	~action_on_set_partitions();
	void init(
			int partition_size,
			actions::action *A,
		int verbose_level);
	long int compute_image(
		int *Elt,
		long int a, int verbose_level);
};

// #############################################################################
// action_on_sets.cpp
// #############################################################################


//! induced action on a given set of sets.



class action_on_sets {
public:
	int nb_sets;
	int set_size;
	long int **sets;
	long int *image_set;
	int *perm;
	int *perm_inv;

	action_on_sets();
	~action_on_sets();
	void init(
			int nb_sets, int set_size,
		long int *input_sets, int verbose_level);
	int find_set(
			long int *set, int verbose_level);
	long int compute_image(
			actions::action *A, int *Elt,
		long int i, int verbose_level);
	void print_sets_sorted();
	void print_sets_in_original_ordering();
	void test_sets();
};


// #############################################################################
// action_on_sign.cpp
// #############################################################################

//! induced action on the sign function of a permutation group (to compute the even subgroup)


class action_on_sign {
public:
	actions::action *A;
	int perm_degree;
	int *perm; // [perm_degree]
	int degree; // 2
	
	action_on_sign();
	~action_on_sign();
	void init(
			actions::action *A, int verbose_level);
	long int compute_image(
			int *Elt, long int i, int verbose_level);
};

// #############################################################################
// action_on_spread_set.cpp
// #############################################################################


//! induced action on a spread set via the associated spread


class action_on_spread_set {
public:

	int k;
	int n; // = 2 * k
	int k2; // = k^2
	int q;
	field_theory::finite_field *F;
	int low_level_point_size; // = k * k
	int degree;
	
	actions::action *A_PGL_n_q;
	actions::action *A_PGL_k_q;
	groups::sims *G_PGL_k_q;

	int *Elt1;
	int *Elt2;
	
	int *mtx1; // [k * k]
	int *mtx2; // [k * k]
	int *subspace1; // [k * n]
	int *subspace2; // [k * n]

	action_on_spread_set();
	~action_on_spread_set();
	void init(
			actions::action *A_PGL_n_q,
			actions::action *A_PGL_k_q,
			groups::sims *G_PGL_k_q,
		int k, field_theory::finite_field *F,
		int verbose_level);
	void report(
			std::ostream &ost, int verbose_level);
	long int compute_image_int(
			int *Elt, long int rk, int verbose_level);
	void matrix_to_subspace(
			int *mtx, int *subspace, int verbose_level);
	void subspace_to_matrix(
			int *subspace, int *mtx, int verbose_level);
	void unrank_point(
			long int rk, int *mtx, int verbose_level);
	long int rank_point(
			int *mtx, int verbose_level);
	void compute_image_low_level(
			int *Elt, int *input, int *output,
		int verbose_level);
};

// #############################################################################
// action_on_subgroups.cpp
// #############################################################################

//! induced action on subgroups of a group


class action_on_subgroups {
public:
	
	actions::action *A;

	groups::sims *S;


	data_structures_groups::hash_table_subgroups *Hash_table_subgroups;


	int max_subgroup_order;

	int *image_set; // [max_subgroup_order]

	int *Elt1;

	action_on_subgroups();
	~action_on_subgroups();
	void init(
			actions::action *A,
			groups::sims *S,
			data_structures_groups::hash_table_subgroups *Hash_table_subgroups,
		int verbose_level);
	// Subgroups[nb_subgroups]
	long int compute_image(
			int *Elt, long int a, int verbose_level);

};


// #############################################################################
// action_on_wedge_product.cpp
// #############################################################################


//! induced wedge product action on the exterior square of a vector space


class action_on_wedge_product {
public:

	actions::action *A;
	int n;
	int q;
	algebra::matrix_group *M;
	field_theory::finite_field *F;
	int low_level_point_size;
	long int degree;

	// wedge product
	int wedge_dimension; // {n \choose 2}
	int *wedge_v1; // [wedge_dimension]
	int *wedge_v2; // [wedge_dimension]
	int *wedge_v3; // [wedge_dimension]
	int *Mtx_wedge; // [wedge_dimension * wedge_dimension]
	
	action_on_wedge_product();
	~action_on_wedge_product();
	void init(
			actions::action *A, int verbose_level);
	void unrank_point(
			int *v, long int rk);
	long int rank_point(
			int *v);
	long int compute_image_int(
			int *Elt, long int a, int verbose_level);
	int element_entry_frobenius(
			int *Elt,
		int verbose_level);
	//int element_entry_ij(int *Elt, int I, int J,
	//	int verbose_level);
	//int element_entry_ijkl(int *Elt,
	//	int i, int j, int k, int l, int verbose_level);
	void compute_image_int_low_level(
			int *Elt, int *input, int *output,
		int verbose_level);
	//void create_induced_matrix(
	//		int *Elt, int *Mtx2, int verbose_level);
	void element_print_latex(
			int *A, std::ostream &ost);
};


// #############################################################################
// product_action.cpp
// #############################################################################

//! induced product action of two group actions


class product_action {
public:
	actions::action *A1;
	actions::action *A2;
	int f_use_projections;
	int offset;
	int degree;
	int elt_size_in_int;
	int coded_elt_size_in_char;

	int *Elt1, *Elt2, *Elt3;
		// temporary storage
	uchar *elt1, *elt2, *elt3;
		// temporary storage, used in element_store()

	data_structures::page_storage *Elts;
	
	product_action();
	~product_action();
	void init(
			actions::action *A1,
			actions::action *A2, int f_use_projections,
		int verbose_level);
	long int compute_image(
			actions::action *A,
			int *Elt, long int i, int verbose_level);
	void element_one(
			actions::action *A, int *Elt, int verbose_level);
	int element_is_one(
			actions::action *A, int *Elt, int verbose_level);
	void element_unpack(
			uchar *elt, int *Elt, int verbose_level);
	void element_pack(
			int *Elt, uchar *elt, int verbose_level);
	void element_retrieve(
			actions::action *A, int hdl, int *Elt,
		int verbose_level);
	int element_store(
			actions::action *A, int *Elt, int verbose_level);
	void element_mult(
			int *A, int *B, int *AB, int verbose_level);
	void element_invert(
			int *A, int *Av, int verbose_level);
	void element_transpose(
			int *A, int *At, int verbose_level);
	void element_move(
			int *A, int *B, int verbose_level);
	void element_print(
			int *A, std::ostream &ost);
	void element_print_latex(
			int *A, std::ostream &ost);
	void make_element(
			int *Elt, int *data, int verbose_level);
};


}}}


#endif /* ORBITER_SRC_LIB_GROUP_ACTIONS_INDUCED_ACTIONS_INDUCED_ACTIONS_H_ */




