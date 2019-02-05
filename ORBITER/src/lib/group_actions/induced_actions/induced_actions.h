// induced_actions.h
//
// Anton Betten
//
// moved here from action.h: July 28, 2018
// based on action.h which was started:  August 13, 2005


namespace orbiter {

namespace group_actions {


// #############################################################################
// action_by_conjugation.C:
// #############################################################################


//! action by conjugation on the elements of a given group



class action_by_conjugation {
public:
	sims *Base_group;
	int f_ownership;
	int goi;
	
	int *Elt1;
	int *Elt2;
	int *Elt3;

	action_by_conjugation();
	~action_by_conjugation();
	void null();
	void free();
	void init(sims *Base_group, int f_ownership, int verbose_level);
	int compute_image(action *A, int *Elt, int i, int verbose_level);
	int rank(int *Elt);
	int multiply(action *A, int i, int j, int verbose_level);
};

// #############################################################################
// action_by_representation.C:
// #############################################################################


//! the action of PSL(2,q) on a conic (the only type implemented so far)


class action_by_representation {
public:
	enum representation_type type;
	int n;
	int q;
	matrix_group *M;
	finite_field *F;
	int low_level_point_size;
	int degree;
	
	int dimension; // 
	int *v1; // [dimension]
	int *v2; // [dimension]
	int *v3; // [dimension]
	
	action_by_representation();
	~action_by_representation();
	void null();
	void free();
	void init_action_on_conic(action &A, int verbose_level);
	int compute_image_int(
		action &A, int *Elt, int a, int verbose_level);
	void compute_image_int_low_level(
		action &A, int *Elt, int *input, int *output, 
		int verbose_level);
};

// #############################################################################
// action_by_restriction.C:
// #############################################################################


//! restricted action on an invariant subset




class action_by_restriction {
public:
	int nb_points;
	int *points; // [nb_points]
	int *points_sorted; // [nb_points]
	int *perm_inv; // [nb_points]

	action_by_restriction();
	~action_by_restriction();
	void null();
	void free();
	void init_from_schreier_vector(
			schreier_vector *Schreier_vector,
			int pt, int verbose_level);
	void init(int nb_points, int *points, int verbose_level);
		// the array points must be orderd
	int compute_image(action *A, int *Elt, int i, int verbose_level);
};

// #############################################################################
// action_by_right_multiplication.C:
// #############################################################################

//! action on a the set of elements of a group by right multiplication



class action_by_right_multiplication {
public:
	sims *Base_group;
	int f_ownership;
	int goi;
	
	int *Elt1;
	int *Elt2;

	action_by_right_multiplication();
	~action_by_right_multiplication();
	void null();
	void free();
	void init(sims *Base_group, int f_ownership, int verbose_level);
	void compute_image(action *A, int *Elt, int i, int &j, 
		int verbose_level);
};

// #############################################################################
// action_by_subfield_structure.C:
// #############################################################################

//! action on the vector space arising from a field over a subfield


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
	
	action *AQ;
	action *Aq;

	matrix_group *MQ;
	finite_field *FQ;
	matrix_group *Mq;
	finite_field *Fq;

	subfield_structure *S;

	int *Eltq;
	int *Mtx; // [m * m]

	int low_level_point_size; // = m
	int degree;

	action_by_subfield_structure();
	~action_by_subfield_structure();
	void null();
	void free();
	void init(action &A, finite_field *Fq, int verbose_level);
	int compute_image_int(
		action &A, int *Elt, int a, int verbose_level);
	void compute_image_int_low_level(
		action &A, int *Elt, int *input, int *output, 
		int verbose_level);
};

// #############################################################################
// action_on_andre.C:
// #############################################################################

//! action on the elements of a projective plane constructed via Andre / Bruck / Bose


class action_on_andre {
public:

	action *An;
	action *An1;
	andre_construction *Andre;
	int k, n, q;
	int k1, n1;
	int N; // number of points in the plane
	int degree;
	int *coords1; // [(k + 1) * (n + 1)];
	int *coords2; // [(k + 1) * (n + 1)];
	int *coords3; // [k * n];

	action_on_andre();
	~action_on_andre();
	void null();
	void free();
	void init(action *An, action *An1, 
		andre_construction *Andre, int verbose_level);
	void compute_image(int *Elt, int i, int &j, 
		int verbose_level);
	int compute_image_of_point(int *Elt, int pt_idx, 
		int verbose_level);
	int compute_image_of_line(int *Elt, int line_idx, 
		int verbose_level);
};

// #############################################################################
// action_on_bricks.C:
// #############################################################################

//! related to a problem of Neil Sloane


class action_on_bricks {
public:

	action *A;
	brick_domain *B;
	int degree;
	int f_linear_action;

	action_on_bricks();
	~action_on_bricks();
	void null();
	void free();
	void init(action *A, brick_domain *B, int f_linear_action, 
		int verbose_level);
	void compute_image(int *Elt, int i, int &j, 
		int verbose_level);
	void compute_image_linear_action(int *Elt, int i, int &j, 
		int verbose_level);
	void compute_image_permutation_action(int *Elt, int i, int &j, 
		int verbose_level);
};

// #############################################################################
// action_on_cosets.C:
// #############################################################################

//! action on the cosets of a subgroup by right multiplication


class action_on_cosets {
public:
	action *A_linear;
	finite_field *F;
	int dimension_of_subspace;
	int n;
	int *subspace_basis; // [dimension_of_subspace * n]
	int *base_cols; // [dimension_of_subspace] 
		// the pivot column for the subspace basis
		// to be used if a vector v[len] 
		// is reduced modulo a subspace
	
	int nb_points;
	int *Points; // ordered list of point ranks

	int *v1;
	int *v2;
	
	void (*unrank_point)(int *v, int a, void *data);
	int (*rank_point)(int *v, void *data);
	void *rank_unrank_data;

	action_on_cosets();
	~action_on_cosets();
	void null();
	void freeself();
	void init(int nb_points, int *Points, 
		action *A_linear, 
		finite_field *F, 
		int dimension_of_subspace, 
		int n, 
		int *subspace_basis, 
		int *base_cols, 
		void (*unrank_point)(int *v, int a, void *data), 
		int (*rank_point)(int *v, void *data), 
		void *rank_unrank_data, 
		int verbose_level);
	void reduce_mod_subspace(int *v, int verbose_level);
	int compute_image(int *Elt, int i, int verbose_level);

};

// #############################################################################
// action_on_determinant.C:
// #############################################################################

//! action on the determinant of a group of matrices (used to compute the subgroup PSL)


class action_on_determinant {
public:
	matrix_group *M;
	int f_projective;
	int m;
	int q;
	int degree;
		// gcd(m, q - 1) if f_projective
		// q - 1 otherwise
	
	action_on_determinant();
	~action_on_determinant();
	void null();
	void free();
	void init(action &A, int f_projective, int m, int verbose_level);
	void compute_image(action *A, int *Elt, int i, int &j, 
		int verbose_level);
};

// #############################################################################
// action_on_factor_space.C:
// #############################################################################

//! induced action on the factor space of a vector space modulo a subspace


class action_on_factor_space {
public:
	vector_space *VS;


	// VS->dimension = length of vectors in large space

	int *subspace_basis; // [subspace_basis_size * VS->dimension]
	int subspace_basis_size;
	int *base_cols; // [subspace_basis_size] 
		// the pivot column for the subspace basis
		// to be used if a vector v[len] is reduced modulo a subspace

	int degree; 
		// the number of projective points in the small space 
		// (i.e., the factor space)
		// (q^factor_space_len - 1) / (q - 1), 
		// as computed by compute_degree();
	int large_degree;
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
	int *projection_table; 
		// [nb_points]
		// projection_table[i] = j 
		// means that the Gauss reduced vector in 
		// the coset of point_list[i] 
		// is in coset_reps_Gauss[j]
	int *preimage_table; // [degree]
	int *tmp; // [factor_space_len] 
	int *Tmp1; // [VS->dimension]
	int *Tmp2; // [VS->dimension]
	int f_tables_have_been_computed;

	int f_table_mode;

	int nb_cosets;
	int *coset_reps_Gauss; 
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
	void null();
	void free();
	void init_light(
		vector_space *VS,
		action &A_base, action &A,
		int *subspace_basis_ranks, int subspace_basis_size, 
		int verbose_level);
	void init_by_rank_table_mode(
		vector_space *VS,
		action &A_base, action &A,
		int *subspace_basis_ranks, int subspace_basis_size, 
		int *point_list, int nb_points, 
		int verbose_level);
	void print_coset_table();
	void print_projection_table(
			int *point_list, int nb_points);
	void init_coset_table(
			int *point_list, int nb_points,
			int verbose_level);
	void init_by_rank(
		vector_space *VS,
		action &A_base, action &A,
		int *subspace_basis_ranks, int subspace_basis_size, 
		int f_compute_tables, int verbose_level);
	void init_from_coordinate_vectors(
		vector_space *VS,
		action &A_base, action &A,
		int *subspace_basis, int subspace_basis_size, 
		int f_compute_tables, int verbose_level);
	void init2(action &A_base, action &A, 
		int f_compute_tables, int verbose_level);
	void compute_projection_table(int verbose_level);
	int compute_degree();
	int compute_large_degree();
	void list_all_elements();
	void reduce_mod_subspace(int *v, int verbose_level);
	int lexleast_element_in_coset(int rk, int verbose_level);
		// This function computes the lexleast 
		// element in the coset modulo the subspace.
		// It does so by looping over all q^subspace_basis_size 
		// elements in the subspace and ranking the corresponding 
		// vector in the large space using rank_in_large_space(v2).
	int project_onto_Gauss_reduced_vector(int rk, int verbose_level);
	int project(int rk, int verbose_level);
		// unranks the vector rk, and reduces it 
		// modulo the subspace basis.
		// The non-pivot components are considered 
		// as a vector in F_q^factor_space_len 
		// and ranked using the rank function for projective space.
		// This rank is returned.
		// If the vector turns out to lie in the 
		// subspace, -1 is returned.
	int preimage(int rk, int verbose_level);
	void embed(int *from, int *to);
	void unrank(int *v, int rk, int verbose_level);
	int rank(int *v, int verbose_level);
	void unrank_in_large_space(int *v, int rk);
	int rank_in_large_space(int *v);
	void unrank_in_small_space(int *v, int rk);
	int rank_in_small_space(int *v);
	int compute_image(action *A, int *Elt, int i, int verbose_level);
};

// #############################################################################
// action_on_flags.C:
// #############################################################################


//! action on flags


class action_on_flags {
public:
	action *A;
	int n;
	finite_field *F;
	int *type;
	int type_len;
	flag *Flag;
	matrix_group *M;
	int degree;
	int *M1;
	int *M2;
	
	action_on_flags();
	~action_on_flags();
	void null();
	void free();
	void init(action *A, int *type, int type_len, 
		int verbose_level);
	int compute_image(int *Elt, int i, int verbose_level);
};

// #############################################################################
// action_on_grassmannian.C:
// #############################################################################

//! action on the grassmannian (subspaces of a fixed dimension of a vectors space)


class action_on_grassmannian {
public:
	int n;
	int k;
	int q;
	finite_field *F;
	int low_level_point_size;
	
	action *A;
	grassmann *G;
	//matrix_group *M;
	int *M1;
	int *M2;

	int f_embedding;
	int big_n;
	grassmann_embedded *GE;
	int *subspace_basis; // [n * big_n]
	int *subspace_basis2; // [n * big_n]
	
	longinteger_object degree;
	int max_string_length;
	
	action_on_grassmannian();
	~action_on_grassmannian();
	void null();
	void free();
	void init(action &A, grassmann *G, int verbose_level);
	void init_embedding(int big_n, int *ambient_space, 
		int verbose_level);
	void compute_image_longinteger(action *A, int *Elt, 
		longinteger_object &i, longinteger_object &j, 
		int verbose_level);
	int compute_image_int(action *A, int *Elt, 
		int i, int verbose_level);
	int compute_image_int_ordinary(action *A, int *Elt, 
		int i, int verbose_level);
	int compute_image_int_embedded(action *A, int *Elt, 
		int i, int verbose_level);
	void print_point(int a, ostream &ost);
};

// #############################################################################
// action_on_homogeneous_polynomials.C:
// #############################################################################


//! induced action on the set of homogeneous polynomials over a finite field


class action_on_homogeneous_polynomials {
public:
	int n; // M->n
	int q;
	action *A;
	homogeneous_polynomial_domain *HPD;
	matrix_group *M;
	finite_field *F;
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
	
	action_on_homogeneous_polynomials();
	~action_on_homogeneous_polynomials();
	void null();
	void free();
	void init(action *A, homogeneous_polynomial_domain *HPD, 
		int verbose_level);
	void init_invariant_set_of_equations(int *Equations, 
		int nb_equations, int verbose_level);
	void unrank_point(int *v, int rk);
	int rank_point(int *v);
	int compute_image_int(int *Elt, int a, int verbose_level);
	void compute_image_int_low_level(
		int *Elt, int *input, int *output, int verbose_level);
};

// #############################################################################
// action_on_k_subsets.C:
// #############################################################################

//! induced action on k-subsets of a set of size n


class action_on_k_subsets {
public:
	action *A;
	int k;
	int degree;
	int *set1; // [k]
	int *set2; // [k]

	action_on_k_subsets();
	~action_on_k_subsets();
	void null();
	void free();
	void init(action *A, int k, int verbose_level);
	void compute_image(int *Elt, int i, int &j, int verbose_level);
};

// #############################################################################
// action_on_orbits.C:
// #############################################################################

//! induced action on the set of orbits (usually by the normalizer)



class action_on_orbits {
public:
	action *A;
	schreier *Sch;
	int f_play_it_safe;
	int degree;
	
	action_on_orbits();
	~action_on_orbits();
	void null();
	void free();
	void init(action *A, schreier *Sch, int f_play_it_safe, 
		int verbose_level);
	int compute_image(int *Elt, int i, int verbose_level);
};

// #############################################################################
// action_on_orthogonal.C:
// #############################################################################


//! action on the orthogonal geometry


class action_on_orthogonal {
public:
	action *original_action;
	orthogonal *O;
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
	void null();
	void free();
	void init(action *original_action, orthogonal *O, 
		int f_on_points, int f_on_lines, int f_on_points_and_lines, 
		int verbose_level);
	int map_a_point(int *Elt, int i, int verbose_level);
	int map_a_line(int *Elt, int i, int verbose_level);
	int compute_image_int(int *Elt, int i, int verbose_level);
};

// #############################################################################
// action_on_set_partitions.cpp:
// #############################################################################


//! induced action on a set partitions.



class action_on_set_partitions {
public:
	int nb_set_partitions;
	int universal_set_size;
	int partition_size;
	int nb_parts;
	action *A;
	int *v1;
	int *v2;

	action_on_set_partitions();
	~action_on_set_partitions();
	void null();
	void free();
	void init(int universal_set_size, int partition_size,
		action *A,
		int verbose_level);
	int compute_image(
		int *Elt,
		int a, int verbose_level);
};

// #############################################################################
// action_on_sets.C:
// #############################################################################


//! induced action on a given set of sets.



class action_on_sets {
public:
	int nb_sets;
	int set_size;
	int **sets;
	int *image_set;
	int *perm;
	int *perm_inv;

	action_on_sets();
	~action_on_sets();
	void null();
	void free();
	void init(int nb_sets, int set_size, 
		int *input_sets, int verbose_level);
	void compute_image(action *A, int *Elt, 
		int i, int &j, int verbose_level);
	void print_sets_sorted();
	void print_sets_in_original_ordering();
	void test_sets();
};

int action_on_sets_compare(void *a, void *b, void *data);
int action_on_sets_compare_inverted(void *a, void *b, void *data);

// #############################################################################
// action_on_sign.C:
// #############################################################################

//! action on the sign function of a permutation group (to compute the even subgroup)


class action_on_sign {
public:
	action *A;
	int perm_degree;
	int *perm; // [perm_degree]
	int degree; // 2
	
	action_on_sign();
	~action_on_sign();
	void null();
	void free();
	void init(action *A, int verbose_level);
	void compute_image(int *Elt, int i, int &j, int verbose_level);
};

// #############################################################################
// action_on_spread_set.C:
// #############################################################################


//! induced action on a spread set via the associated spread


class action_on_spread_set {
public:

	int k;
	int n; // = 2 * k
	int k2; // = k^2
	int q;
	finite_field *F;
	int low_level_point_size; // = k * k
	int degree;
	
	action *A_PGL_n_q;
	action *A_PGL_k_q;
	sims *G_PGL_k_q;

	int *Elt1;
	int *Elt2;
	
	int *mtx1; // [k * k]
	int *mtx2; // [k * k]
	int *subspace1; // [k * n]
	int *subspace2; // [k * n]

	action_on_spread_set();
	~action_on_spread_set();
	void null();
	void free();
	void init(action *A_PGL_n_q, action *A_PGL_k_q, sims *G_PGL_k_q, 
		int k, finite_field *F, int verbose_level);
	int compute_image_int(int *Elt, int rk, int verbose_level);
	void matrix_to_subspace(int *mtx, int *subspace, int verbose_level);
	void subspace_to_matrix(int *subspace, int *mtx, int verbose_level);
	void unrank_point(int rk, int *mtx, int verbose_level);
	int rank_point(int *mtx, int verbose_level);
	void compute_image_low_level(int *Elt, int *input, int *output, 
		int verbose_level);
};

// #############################################################################
// action_on_subgroups.C:
// #############################################################################

//! induced action on subgroups of a group


class action_on_subgroups {
public:
	
	action *A;
	sims *S;
	int nb_subgroups;
	int subgroup_order;
	subgroup **Subgroups;
	int **sets;
	int *image_set; // [subgroup_order]
	int *perm;
	int *perm_inv;
	int *Elt1;

	action_on_subgroups();
	~action_on_subgroups();
	void null();
	void free();
	void init(action *A, sims *S, int nb_subgroups, 
		int subgroup_order, subgroup **Subgroups, 
		int verbose_level);
	int compute_image(int *Elt, int a, int verbose_level);

};

int action_on_subgroups_compare(void *a, void *b, void *data);
int action_on_subgroups_compare_inverted(void *a, void *b, void *data);

// #############################################################################
// action_on_wedge_product.C:
// #############################################################################


//! the wedge product action on exterior square of a vector space


class action_on_wedge_product {
public:

	int n;
	int q;
	matrix_group *M;
	finite_field *F;
	int low_level_point_size;
	int degree;

	// wedge product
	int wedge_dimension; // {n \choose 2}
	int *wedge_v1; // [wedge_dimension]
	int *wedge_v2; // [wedge_dimension]
	int *wedge_v3; // [wedge_dimension]
	
	action_on_wedge_product();
	~action_on_wedge_product();
	void null();
	void free();
	void init(action &A, int verbose_level);
	void unrank_point(int *v, int rk);
	int rank_point(int *v);
	int compute_image_int(
		action &A, int *Elt, int a, int verbose_level);
	int element_entry_frobenius(action &A, int *Elt, 
		int verbose_level);
	int element_entry_ij(action &A, int *Elt, int I, int J, 
		int verbose_level);
	int element_entry_ijkl(action &A, int *Elt, 
		int i, int j, int k, int l, int verbose_level);
	void compute_image_int_low_level(
		action &A, int *Elt, int *input, int *output, 
		int verbose_level);
};


// #############################################################################
// product_action.C:
// #############################################################################

//! the product action of two group actions


class product_action {
public:
	action *A1;
	action *A2;
	int f_use_projections;
	int offset;
	int degree;
	int elt_size_in_int;
	int coded_elt_size_in_char;

	int *Elt1, *Elt2, *Elt3;
		// temporary storage
	uchar *elt1, *elt2, *elt3;
		// temporary storage, used in element_store()

	page_storage *Elts;
	
	product_action();
	~product_action();
	void null();
	void free();
	void init(action *A1, action *A2, int f_use_projections, 
		int verbose_level);
	int compute_image(action *A, int *Elt, int i, int verbose_level);
	void element_one(action *A, int *Elt, int verbose_level);
	int element_is_one(action *A, int *Elt, int verbose_level);
	void element_unpack(uchar *elt, int *Elt, int verbose_level);
	void element_pack(int *Elt, uchar *elt, int verbose_level);
	void element_retrieve(action *A, int hdl, int *Elt, 
		int verbose_level);
	int element_store(action *A, int *Elt, int verbose_level);
	void element_mult(int *A, int *B, int *AB, int verbose_level);
	void element_invert(int *A, int *Av, int verbose_level);
	void element_transpose(int *A, int *At, int verbose_level);
	void element_move(int *A, int *B, int verbose_level);
	void element_print(int *A, ostream &ost);
	void element_print_latex(int *A, ostream &ost);
	void make_element(int *Elt, int *data, int verbose_level);
};


}}





