// induced_actions.h
//
// Anton Betten
//
// moved here from action.h: July 28, 2018
// based on action.h which was started:  August 13, 2005

// #############################################################################
// action_by_conjugation.C:
// #############################################################################


//! action by conjugation on the elements of a given group



class action_by_conjugation {
public:
	sims *Base_group;
	INT f_ownership;
	INT goi;
	
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;

	action_by_conjugation();
	~action_by_conjugation();
	void null();
	void free();
	void init(sims *Base_group, INT f_ownership, INT verbose_level);
	INT compute_image(action *A, INT *Elt, INT i, INT verbose_level);
	INT rank(INT *Elt);
	INT multiply(action *A, INT i, INT j, INT verbose_level);
};

// #############################################################################
// action_by_representation.C:
// #############################################################################


//! the action of PSL(2,q) on a conic (the only type implemented so far)


class action_by_representation {
public:
	enum representation_type type;
	INT n;
	INT q;
	matrix_group *M;
	finite_field *F;
	INT low_level_point_size;
	INT degree;
	
	INT dimension; // 
	INT *v1; // [dimension]
	INT *v2; // [dimension]
	INT *v3; // [dimension]
	
	action_by_representation();
	~action_by_representation();
	void null();
	void free();
	void init_action_on_conic(action &A, INT verbose_level);
	INT compute_image_INT(
		action &A, INT *Elt, INT a, INT verbose_level);
	void compute_image_INT_low_level(
		action &A, INT *Elt, INT *input, INT *output, 
		INT verbose_level);
};

// #############################################################################
// action_by_restriction.C:
// #############################################################################


//! restricted action on an invariant subset




class action_by_restriction {
public:
	INT nb_points;
	INT *points; // [nb_points]
	INT *points_sorted; // [nb_points]
	INT *perm_inv; // [nb_points]

	action_by_restriction();
	~action_by_restriction();
	void null();
	void free();
	void init_from_sv(INT *sv, INT pt, INT verbose_level);
	void init(INT nb_points, INT *points, INT verbose_level);
		// the array points must be orderd
	INT compute_image(action *A, INT *Elt, INT i, INT verbose_level);
};

// #############################################################################
// action_by_right_multiplication.C:
// #############################################################################

//! action on a the set of elements of a group by right multiplication



class action_by_right_multiplication {
public:
	sims *Base_group;
	INT f_ownership;
	INT goi;
	
	INT *Elt1;
	INT *Elt2;

	action_by_right_multiplication();
	~action_by_right_multiplication();
	void null();
	void free();
	void init(sims *Base_group, INT f_ownership, INT verbose_level);
	void compute_image(action *A, INT *Elt, INT i, INT &j, 
		INT verbose_level);
};

// #############################################################################
// action_by_subfield_structure.C:
// #############################################################################

//! action on the vector space arising from a field over a subfield


class action_by_subfield_structure {
public:
	INT n;
	INT Q;
	const BYTE *poly_q;
	INT q;
	INT s;
	INT m; // n * s
	INT *v1; // [m]
	INT *v2; // [m]
	INT *v3; // [m]
	
	action *AQ;
	action *Aq;

	matrix_group *MQ;
	finite_field *FQ;
	matrix_group *Mq;
	finite_field *Fq;

	subfield_structure *S;

	INT *Eltq;
	INT *Mtx; // [m * m]

	INT low_level_point_size; // = m
	INT degree;

	action_by_subfield_structure();
	~action_by_subfield_structure();
	void null();
	void free();
	void init(action &A, finite_field *Fq, INT verbose_level);
	INT compute_image_INT(
		action &A, INT *Elt, INT a, INT verbose_level);
	void compute_image_INT_low_level(
		action &A, INT *Elt, INT *input, INT *output, 
		INT verbose_level);
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
	INT k, n, q;
	INT k1, n1;
	INT N; // number of points in the plane
	INT degree;
	INT *coords1; // [(k + 1) * (n + 1)];
	INT *coords2; // [(k + 1) * (n + 1)];
	INT *coords3; // [k * n];

	action_on_andre();
	~action_on_andre();
	void null();
	void free();
	void init(action *An, action *An1, 
		andre_construction *Andre, INT verbose_level);
	void compute_image(INT *Elt, INT i, INT &j, 
		INT verbose_level);
	INT compute_image_of_point(INT *Elt, INT pt_idx, 
		INT verbose_level);
	INT compute_image_of_line(INT *Elt, INT line_idx, 
		INT verbose_level);
};

// #############################################################################
// action_on_bricks.C:
// #############################################################################

//! related to a problem of Neal Sloan


class action_on_bricks {
public:

	action *A;
	brick_domain *B;
	INT degree;
	INT f_linear_action;

	action_on_bricks();
	~action_on_bricks();
	void null();
	void free();
	void init(action *A, brick_domain *B, INT f_linear_action, 
		INT verbose_level);
	void compute_image(INT *Elt, INT i, INT &j, 
		INT verbose_level);
	void compute_image_linear_action(INT *Elt, INT i, INT &j, 
		INT verbose_level);
	void compute_image_permutation_action(INT *Elt, INT i, INT &j, 
		INT verbose_level);
};

// #############################################################################
// action_on_cosets.C:
// #############################################################################

//! action on the cosets of a subgroup by right multiplication


class action_on_cosets {
public:
	action *A_linear;
	finite_field *F;
	INT dimension_of_subspace;
	INT n;
	INT *subspace_basis; // [dimension_of_subspace * n]
	INT *base_cols; // [dimension_of_subspace] 
		// the pivot column for the subspace basis
		// to be used if a vector v[len] 
		// is reduced modulo a subspace
	
	INT nb_points;
	INT *Points; // ordered list of point ranks

	INT *v1;
	INT *v2;
	
	void (*unrank_point)(INT *v, INT a, void *data);
	INT (*rank_point)(INT *v, void *data);
	void *rank_unrank_data;

	action_on_cosets();
	~action_on_cosets();
	void null();
	void freeself();
	void init(INT nb_points, INT *Points, 
		action *A_linear, 
		finite_field *F, 
		INT dimension_of_subspace, 
		INT n, 
		INT *subspace_basis, 
		INT *base_cols, 
		void (*unrank_point)(INT *v, INT a, void *data), 
		INT (*rank_point)(INT *v, void *data), 
		void *rank_unrank_data, 
		INT verbose_level);
	void reduce_mod_subspace(INT *v, INT verbose_level);
	INT compute_image(INT *Elt, INT i, INT verbose_level);

};

// #############################################################################
// action_on_determinant.C:
// #############################################################################

//! action on the determinant of a group of matrices (used to compute the subgroup PSL)


class action_on_determinant {
public:
	matrix_group *M;
	INT f_projective;
	INT m;
	INT q;
	INT degree;
		// gcd(m, q - 1) if f_projective
		// q - 1 otherwise
	
	action_on_determinant();
	~action_on_determinant();
	void null();
	void free();
	void init(action &A, INT f_projective, INT m, INT verbose_level);
	void compute_image(action *A, INT *Elt, INT i, INT &j, 
		INT verbose_level);
};

// #############################################################################
// action_on_factor_space.C:
// #############################################################################

//! induced action on the factor space of a vector space modulo a subspace


class action_on_factor_space {
public:
	INT len; // length of vectors in large space
	finite_field *F;
	INT *subspace_basis; // [subspace_basis_size * len]
	INT subspace_basis_size;
	INT *base_cols; // [subspace_basis_size] 
		// the pivot column for the subspace basis
		// to be used if a vector v[len] is reduced modulo a subspace

	INT degree; 
		// the number of projective points in the small space 
		// (i.e., the factor space)
		// (q^factor_space_len - 1) / (q - 1), 
		// as computed by compute_degree();
	INT large_degree; 
		// the number of projective points in the large space
		// (q^len - 1) / (q - 1), 
		// as computed by compute_large_degree();
	
	INT factor_space_len; 
		// = len - subspace_basis_size

	INT *embedding; 
		// [factor_space_len]
		// the list of columns that are not pivot columns, 
		// i.e. not in base_cols[] 
		// this is the set-theoretic complement of base_cols
	INT *projection_table; 
		// [nb_points]
		// projection_table[i] = j 
		// means that the Gauss reduced vector in 
		// the coset of point_list[i] 
		// is in coset_reps_Gauss[j]
	INT *preimage_table; // [degree]
	INT *tmp; // [factor_space_len] 
	INT *Tmp1; // [len] 
	INT *Tmp2; // [len]
	INT f_tables_have_been_computed;

	INT f_table_mode;
	INT f_has_rank_function;
	INT (*rank_point_func)(INT *v, void *data);
	void (*unrank_point_func)(INT *v, INT rk, void *data);
	void *rank_point_data;
	INT nb_cosets;
	INT *coset_reps_Gauss; 
		// [nb_cosets]
		// ordered list of Gauss reduced coset representatives
		// the entries are ranks of vectors in the large space

	INT *tmp_w; // [len] temporary vector for use in rank
	INT *tmp_w1;
		// [subspace_basis_size] 
		// temporary vector for lexleast_element_in_coset
	INT *tmp_v1;
		// [len] temporary vector 
		// for use in lexleast_element_in_coset
	INT *tmp_v2;
		// [len] temporary vector 
		// for use in lexleast_element_in_coset
	
	action_on_factor_space();
	~action_on_factor_space();
	void null();
	void free();
	void init_light(action &A_base, action &A, INT len, 
		finite_field *F, 
		INT *subspace_basis_ranks, INT subspace_basis_size, 
		INT (*rank_point_func)(INT *v, void *data), 
		void (*unrank_point_func)(INT *v, INT rk, void *data), 
		void *rank_point_data, 
		INT verbose_level);
	void init_by_rank_table_mode(action &A_base, 
		action &A, INT len, finite_field *F, 
		INT *subspace_basis_ranks, INT subspace_basis_size, 
		INT *point_list, INT nb_points, 
		INT (*rank_point_func)(INT *v, void *data), 
		void (*unrank_point_func)(INT *v, INT rk, void *data), 
		void *rank_point_data, 
		INT verbose_level);
	void init_by_rank(action &A_base, action &A, 
		INT len, finite_field *F, 
		INT *subspace_basis_ranks, INT subspace_basis_size, 
		INT f_compute_tables, INT verbose_level);
	void init_from_coordinate_vectors(action &A_base, 
		action &A, INT len, finite_field *F, 
		INT *subspace_basis, INT subspace_basis_size, 
		INT f_compute_tables, INT verbose_level);
	void init2(action &A_base, action &A, 
		INT f_compute_tables, INT verbose_level);
	void compute_projection_table(INT verbose_level);
	INT compute_degree();
	INT compute_large_degree();
	void list_all_elements();
	void reduce_mod_subspace(INT *v, INT verbose_level);
	INT lexleast_element_in_coset(INT rk, INT verbose_level);
		// This function computes the lexleast 
		// element in the coset modulo the subspace.
		// It does so by looping over all q^subspace_basis_size 
		// elements in the subspace and ranking the corresponding 
		// vector in the large space using rank_in_large_space(v2).
	INT project_onto_Gauss_reduced_vector(INT rk, INT verbose_level);
	INT project(INT rk, INT verbose_level);
		// unranks the vector rk, and reduces it 
		// modulo the subspace basis.
		// The non-pivot components are considered 
		// as a vector in F_q^factor_space_len 
		// and ranked using the rank function for projective space.
		// This rank is returned.
		// If the vector turns out to lie in the 
		// subspace, -1 is returned.
	INT preimage(INT rk, INT verbose_level);
	void unrank(INT *v, INT rk, INT verbose_level);
	INT rank(INT *v, INT verbose_level);
	void unrank_in_large_space(INT *v, INT rk);
	INT rank_in_large_space(INT *v);
	INT compute_image(action *A, INT *Elt, INT i, INT verbose_level);
};

// #############################################################################
// action_on_flags.C:
// #############################################################################


//! action on flags


class action_on_flags {
public:
	action *A;
	INT n;
	finite_field *F;
	INT *type;
	INT type_len;
	flag *Flag;
	matrix_group *M;
	INT degree;
	INT *M1;
	INT *M2;
	
	action_on_flags();
	~action_on_flags();
	void null();
	void free();
	void init(action *A, INT *type, INT type_len, 
		INT verbose_level);
	INT compute_image(INT *Elt, INT i, INT verbose_level);
};

// #############################################################################
// action_on_grassmannian.C:
// #############################################################################

//! action on the grassmannian (subspaces of a fixed dimension of a vectors space)


class action_on_grassmannian {
public:
	INT n;
	INT k;
	INT q;
	finite_field *F;
	INT low_level_point_size;
	
	action *A;
	grassmann *G;
	//matrix_group *M;
	INT *M1;
	INT *M2;

	INT f_embedding;
	INT big_n;
	grassmann_embedded *GE;
	INT *subspace_basis; // [n * big_n]
	INT *subspace_basis2; // [n * big_n]
	
	longinteger_object degree;
	INT max_string_length;
	
	action_on_grassmannian();
	~action_on_grassmannian();
	void null();
	void free();
	void init(action &A, grassmann *G, INT verbose_level);
	void init_embedding(INT big_n, INT *ambient_space, 
		INT verbose_level);
	void compute_image_longinteger(action *A, INT *Elt, 
		longinteger_object &i, longinteger_object &j, 
		INT verbose_level);
	INT compute_image_INT(action *A, INT *Elt, 
		INT i, INT verbose_level);
	INT compute_image_INT_ordinary(action *A, INT *Elt, 
		INT i, INT verbose_level);
	INT compute_image_INT_embedded(action *A, INT *Elt, 
		INT i, INT verbose_level);
	void print_point(INT a, ostream &ost);
};

// #############################################################################
// action_on_homogeneous_polynomials.C:
// #############################################################################


//! induced action on the set of homogeneous polynomials over a finite field


class action_on_homogeneous_polynomials {
public:
	INT n; // M->n
	INT q;
	action *A;
	homogeneous_polynomial_domain *HPD;
	matrix_group *M;
	finite_field *F;
	INT low_level_point_size;
	INT degree;

	// wedge product
	INT dimension; // = HPD->nb_monomials
	INT *v1; // [dimension]
	INT *v2; // [dimension]
	INT *v3; // [dimension]
	INT *Elt1;
	
	INT f_invariant_set;
	INT *Equations;
	INT nb_equations;
	
	action_on_homogeneous_polynomials();
	~action_on_homogeneous_polynomials();
	void null();
	void free();
	void init(action *A, homogeneous_polynomial_domain *HPD, 
		INT verbose_level);
	void init_invariant_set_of_equations(INT *Equations, 
		INT nb_equations, INT verbose_level);
	void unrank_point(INT *v, INT rk);
	INT rank_point(INT *v);
	INT compute_image_INT(INT *Elt, INT a, INT verbose_level);
	void compute_image_INT_low_level(
		INT *Elt, INT *input, INT *output, INT verbose_level);
};

// #############################################################################
// action_on_k_subsets.C:
// #############################################################################

//! induced action on k-subsets of a set of size n


class action_on_k_subsets {
public:
	action *A;
	INT k;
	INT degree;
	INT *set1; // [k]
	INT *set2; // [k]

	action_on_k_subsets();
	~action_on_k_subsets();
	void null();
	void free();
	void init(action *A, INT k, INT verbose_level);
	void compute_image(INT *Elt, INT i, INT &j, INT verbose_level);
};

// #############################################################################
// action_on_orbits.C:
// #############################################################################

//! induced action on the set of orbits (usually by the normalizer)



class action_on_orbits {
public:
	action *A;
	schreier *Sch;
	INT f_play_it_safe;
	INT degree;
	
	action_on_orbits();
	~action_on_orbits();
	void null();
	void free();
	void init(action *A, schreier *Sch, INT f_play_it_safe, 
		INT verbose_level);
	INT compute_image(INT *Elt, INT i, INT verbose_level);
};

// #############################################################################
// action_on_orthogonal.C:
// #############################################################################


//! action on the orthogonal geometry


class action_on_orthogonal {
public:
	action *original_action;
	orthogonal *O;
	INT *v1;
	INT *v2;
	INT *w1;
	INT *w2;
	INT f_on_points;
	INT f_on_lines;
	INT f_on_points_and_lines;
	INT low_level_point_size;
	INT degree;
	
	action_on_orthogonal();
	~action_on_orthogonal();
	void null();
	void free();
	void init(action *original_action, orthogonal *O, 
		INT f_on_points, INT f_on_lines, INT f_on_points_and_lines, 
		INT verbose_level);
	INT map_a_point(INT *Elt, INT i, INT verbose_level);
	INT map_a_line(INT *Elt, INT i, INT verbose_level);
	INT compute_image_INT(INT *Elt, INT i, INT verbose_level);
};

// #############################################################################
// action_on_sets.C:
// #############################################################################


//! induced action on a given set of sets.



class action_on_sets {
public:
	INT nb_sets;
	INT set_size;
	INT **sets;
	INT *image_set;
	INT *perm;
	INT *perm_inv;

	action_on_sets();
	~action_on_sets();
	void null();
	void free();
	void init(INT nb_sets, INT set_size, 
		INT *input_sets, INT verbose_level);
	void compute_image(action *A, INT *Elt, 
		INT i, INT &j, INT verbose_level);
	void print_sets_sorted();
	void print_sets_in_original_ordering();
	void test_sets();
};

INT action_on_sets_compare(void *a, void *b, void *data);
INT action_on_sets_compare_inverted(void *a, void *b, void *data);

// #############################################################################
// action_on_sign.C:
// #############################################################################

//! action on the sign function of a permutation group (to compute the even subgroup)


class action_on_sign {
public:
	action *A;
	INT perm_degree;
	INT *perm; // [perm_degree]
	INT degree; // 2
	
	action_on_sign();
	~action_on_sign();
	void null();
	void free();
	void init(action *A, INT verbose_level);
	void compute_image(INT *Elt, INT i, INT &j, INT verbose_level);
};

// #############################################################################
// action_on_spread_set.C:
// #############################################################################


//! induced action on a spread set via the associated spread


class action_on_spread_set {
public:

	INT k;
	INT n; // = 2 * k
	INT k2; // = k^2
	INT q;
	finite_field *F;
	INT low_level_point_size; // = k * k
	INT degree;
	
	action *A_PGL_n_q;
	action *A_PGL_k_q;
	sims *G_PGL_k_q;

	INT *Elt1;
	INT *Elt2;
	
	INT *mtx1; // [k * k]
	INT *mtx2; // [k * k]
	INT *subspace1; // [k * n]
	INT *subspace2; // [k * n]

	action_on_spread_set();
	~action_on_spread_set();
	void null();
	void free();
	void init(action *A_PGL_n_q, action *A_PGL_k_q, sims *G_PGL_k_q, 
		INT k, finite_field *F, INT verbose_level);
	INT compute_image_INT(INT *Elt, INT rk, INT verbose_level);
	void matrix_to_subspace(INT *mtx, INT *subspace, INT verbose_level);
	void subspace_to_matrix(INT *subspace, INT *mtx, INT verbose_level);
	void unrank_point(INT rk, INT *mtx, INT verbose_level);
	INT rank_point(INT *mtx, INT verbose_level);
	void compute_image_low_level(INT *Elt, INT *input, INT *output, 
		INT verbose_level);
};

// #############################################################################
// action_on_subgroups.C:
// #############################################################################

//! induced action on subgroups of a group


class action_on_subgroups {
public:
	
	action *A;
	sims *S;
	INT nb_subgroups;
	INT subgroup_order;
	subgroup **Subgroups;
	INT **sets;
	INT *image_set; // [subgroup_order]
	INT *perm;
	INT *perm_inv;
	INT *Elt1;

	action_on_subgroups();
	~action_on_subgroups();
	void null();
	void free();
	void init(action *A, sims *S, INT nb_subgroups, 
		INT subgroup_order, subgroup **Subgroups, 
		INT verbose_level);
	INT compute_image(INT *Elt, INT a, INT verbose_level);

};

INT action_on_subgroups_compare(void *a, void *b, void *data);
INT action_on_subgroups_compare_inverted(void *a, void *b, void *data);

// #############################################################################
// action_on_wedge_product.C:
// #############################################################################


//! the wedge product action on exterior square of a vector space


class action_on_wedge_product {
public:

	INT n;
	INT q;
	matrix_group *M;
	finite_field *F;
	INT low_level_point_size;
	INT degree;

	// wedge product
	INT wedge_dimension; // {n \choose 2}
	INT *wedge_v1; // [wedge_dimension]
	INT *wedge_v2; // [wedge_dimension]
	INT *wedge_v3; // [wedge_dimension]
	
	action_on_wedge_product();
	~action_on_wedge_product();
	void null();
	void free();
	void init(action &A, INT verbose_level);
	void unrank_point(INT *v, INT rk);
	INT rank_point(INT *v);
	INT compute_image_INT(
		action &A, INT *Elt, INT a, INT verbose_level);
	INT element_entry_frobenius(action &A, INT *Elt, 
		INT verbose_level);
	INT element_entry_ij(action &A, INT *Elt, INT I, INT J, 
		INT verbose_level);
	INT element_entry_ijkl(action &A, INT *Elt, 
		INT i, INT j, INT k, INT l, INT verbose_level);
	void compute_image_INT_low_level(
		action &A, INT *Elt, INT *input, INT *output, 
		INT verbose_level);
};


// #############################################################################
// product_action.C:
// #############################################################################

//! the product action of two group actions


class product_action {
public:
	action *A1;
	action *A2;
	INT f_use_projections;
	INT offset;
	INT degree;
	INT elt_size_in_INT;
	INT coded_elt_size_in_char;

	INT *Elt1, *Elt2, *Elt3;
		// temporary storage
	UBYTE *elt1, *elt2, *elt3;
		// temporary storage, used in element_store()

	page_storage *Elts;
	
	product_action();
	~product_action();
	void null();
	void free();
	void init(action *A1, action *A2, INT f_use_projections, 
		INT verbose_level);
	INT compute_image(action *A, INT *Elt, INT i, INT verbose_level);
	void element_one(action *A, INT *Elt, INT verbose_level);
	INT element_is_one(action *A, INT *Elt, INT verbose_level);
	void element_unpack(UBYTE *elt, INT *Elt, INT verbose_level);
	void element_pack(INT *Elt, UBYTE *elt, INT verbose_level);
	void element_retrieve(action *A, INT hdl, INT *Elt, 
		INT verbose_level);
	INT element_store(action *A, INT *Elt, INT verbose_level);
	void element_mult(INT *A, INT *B, INT *AB, INT verbose_level);
	void element_invert(INT *A, INT *Av, INT verbose_level);
	void element_transpose(INT *A, INT *At, INT verbose_level);
	void element_move(INT *A, INT *B, INT verbose_level);
	void element_print(INT *A, ostream &ost);
	void element_print_latex(INT *A, ostream &ost);
	void make_element(INT *Elt, INT *data, INT verbose_level);
};






