// group_theory.h
//
// Anton Betten
//
// moved here from action.h: July 28, 2018
// based on action.h which was started:  August 13, 2005


// #############################################################################
// direct_product.C:
// #############################################################################

class direct_product {

public:
	matrix_group *M1;
	matrix_group *M2;
	finite_field *F1;
	finite_field *F2;
	INT q1;
	INT q2;

	BYTE label[1000];
	BYTE label_tex[1000];

	INT degree_of_matrix_group1;
	INT dimension_of_matrix_group1;
	INT degree_of_matrix_group2;
	INT dimension_of_matrix_group2;
	INT degree_of_product_action;
	INT degree_overall;
	INT low_level_point_size;
	INT make_element_size;
	INT elt_size_INT;

	INT *perm_offset_i;
	INT *tmp_Elt1;

	INT bits_per_digit1;
	INT bits_per_digit2;

	INT bits_per_elt;
	INT char_per_elt;

	UBYTE *elt1;

	INT base_len_in_component1;
	INT *base_for_component1;
	INT *tl_for_component1;

	INT base_len_in_component2;
	INT *base_for_component2;
	INT *tl_for_component2;

	INT base_length;
	INT *the_base;
	INT *the_transversal_length;

	page_storage *Elts;

	direct_product();
	~direct_product();
	void null();
	void freeself();
	void init(matrix_group *M1, matrix_group *M2,
			INT verbose_level);
	INT element_image_of(INT *Elt, INT a, INT verbose_level);
	void element_one(INT *Elt);
	INT element_is_one(INT *Elt);
	void element_mult(INT *A, INT *B, INT *AB, INT verbose_level);
	void element_move(INT *A, INT *B, INT verbose_level);
	void element_invert(INT *A, INT *Av, INT verbose_level);
	INT offset_i(INT i);
	void element_pack(INT *Elt, UBYTE *elt);
	void element_unpack(UBYTE *elt, INT *Elt);
	void put_digit(UBYTE *elt, INT f, INT i, INT d);
	INT get_digit(UBYTE *elt, INT f, INT i);
	void make_element(INT *Elt, INT *data, INT verbose_level);
	void element_print_easy(INT *Elt, ostream &ost);
	void compute_base_and_transversals(INT verbose_level);
	void make_strong_generators_data(INT *&data,
			INT &size, INT &nb_gens, INT verbose_level);
};


// #############################################################################
// linear_group.C:
// #############################################################################


class linear_group {
public:
	linear_group_description *description;
	INT n;
	INT input_q;
	finite_field *F;
	INT f_semilinear;

	BYTE prefix[1000];
	strong_generators *initial_strong_gens;
	action *A_linear;
	matrix_group *Mtx;

	INT f_has_strong_generators;
	strong_generators *Strong_gens;
	action *A2;
	INT vector_space_dimension;
	INT q;

	linear_group();
	~linear_group();
	void null();
	void freeself();
	void init(linear_group_description *description, 
		INT verbose_level);
	void init_PGL2q_OnConic(BYTE *prefix, INT verbose_level);
	void init_wedge_action(BYTE *prefix, INT verbose_level);
	void init_monomial_group(BYTE *prefix, INT verbose_level);
	void init_diagonal_group(BYTE *prefix, INT verbose_level);
	void init_singer_group(BYTE *prefix, INT singer_power, 
		INT verbose_level);
	void init_null_polarity_group(BYTE *prefix, INT verbose_level);
	void init_borel_subgroup_upper(BYTE *prefix, INT verbose_level);
	void init_identity_subgroup(BYTE *prefix, INT verbose_level);
	void init_symplectic_group(BYTE *prefix, INT verbose_level);
	void init_subfield_structure_action(BYTE *prefix, INT s, 
		INT verbose_level);
	void init_orthogonal_group(BYTE *prefix, 
		INT epsilon, INT verbose_level);
	void init_subgroup_from_file(BYTE *prefix, 
		const BYTE *subgroup_fname, const BYTE *subgroup_label, 
		INT verbose_level);
};

// #############################################################################
// linear_group_description.C:
// #############################################################################



class linear_group_description {
public:
	INT f_projective;
	INT f_general;
	INT f_affine;

	INT n;
	INT input_q;
	finite_field *F;
	INT f_semilinear;
	INT f_special;

	INT f_wedge_action;
	INT f_PGL2OnConic;
	INT f_monomial_group;
	INT f_diagonal_group;
	INT f_null_polarity_group;
	INT f_symplectic_group;
	INT f_singer_group;
	INT singer_power;
	INT f_subfield_structure_action;
	INT s;
	INT f_subgroup_from_file;
	INT f_borel_subgroup_upper;
	INT f_borel_subgroup_lower;
	INT f_identity_group;
	const BYTE *subgroup_fname;
	const BYTE *subgroup_label;
	INT f_orthogonal_group;
	INT orthogonal_group_epsilon;

	INT f_on_k_subspaces;
	INT on_k_subspaces_k;


	linear_group_description();
	~linear_group_description();
	void null();
	void freeself();
	INT read_arguments(int argc, const char **argv, 
		INT verbose_level);
};

// #############################################################################
// matrix_group.C:
// #############################################################################

class matrix_group {

public:
	INT f_projective;
		// n x n matrices (possibly with Frobenius) 
		// acting on PG(n - 1, q)
	INT f_affine;
		// n x n matrices plus translations
		// (possibly with Frobenius) 
		// acting on F_q^n
	INT f_general_linear;
		// n x n matrices (possibly with Frobenius) 
		// acting on F_q^n

	INT n;
		// the size of the matrices

	INT degree;
		// the degree of the action: 
		// (q^(n-1)-1) / (q - 1) if f_projective
		// q^n if f_affine or f_general_linear
		  
	INT f_semilinear;
		// use Frobenius automorphism

	INT f_kernel_is_diagonal_matrices;
	
	INT bits_per_digit;
	INT bits_per_elt;
	INT bits_extension_degree;
	INT char_per_elt;
	INT elt_size_INT;
	INT elt_size_INT_half;
	INT low_level_point_size; // added Jan 26, 2010
		// = n, the size of the vectors on which we act
	INT make_element_size;

	BYTE label[1000];
	BYTE label_tex[1000];
	
	INT f_GFq_is_allocated;
		// if TRUE, GFq will be destroyed in the destructor
		// if FALSE, it is the responsability 
		// of someone else to destroy GFq
	
	finite_field *GFq;
	void *data;

	gl_classes *C; // added Dec 2, 2013

	
	// temporary variables, do not use!
	INT *Elt1, *Elt2, *Elt3;
		// used for mult, invert
	INT *Elt4;
		// used for invert
	INT *Elt5;
	INT *tmp_M;
		// used for GL_mult_internal
	INT *base_cols;
		// used for Gauss during invert
	INT *v1, *v2;
		// temporary vectors of length 2n
	INT *v3;
		// used in GL_mult_vector_from_the_left_contragredient
	UBYTE *elt1, *elt2, *elt3;
		// temporary storage, used in element_store()
	
	page_storage *Elts;
	

	matrix_group();
	~matrix_group();
	void null();
	void freeself();
	
	void init_projective_group(INT n, finite_field *F, 
		INT f_semilinear, action *A, INT verbose_level);
	void init_affine_group(INT n, finite_field *F, 
		INT f_semilinear, action *A, INT verbose_level);
	void init_general_linear_group(INT n, finite_field *F, 
		INT f_semilinear, action *A, INT verbose_level);
	void allocate_data(INT verbose_level);
	void free_data(INT verbose_level);
	void setup_page_storage(INT page_length_log, INT verbose_level);
	void compute_elt_size(INT verbose_level);
	void init_base(action *A, INT verbose_level);
	void init_base_projective(action *A, INT verbose_level);
	// initializes base, base_len, degree, transversal_length, orbit, orbit_inv
	void init_base_affine(action *A, INT verbose_level);
	void init_base_general_linear(action *A, INT verbose_level);
	void init_gl_classes(INT verbose_level);

	INT GL_element_entry_ij(INT *Elt, INT i, INT j);
	INT GL_element_entry_frobenius(INT *Elt);
	INT image_of_element(INT *Elt, INT a, INT verbose_level);
	INT GL_image_of_PG_element(INT *Elt, INT a, INT verbose_level);
	INT GL_image_of_AG_element(INT *Elt, INT a, INT verbose_level);
	void action_from_the_right_all_types(
		INT *v, INT *A, INT *vA, INT verbose_level);
	void projective_action_from_the_right(
		INT *v, INT *A, INT *vA, INT verbose_level);
	void general_linear_action_from_the_right(
		INT *v, INT *A, INT *vA, INT verbose_level);
	void GL_one(INT *Elt);
	void GL_one_internal(INT *Elt);
	void GL_zero(INT *Elt);
	INT GL_is_one(INT *Elt);
	void GL_mult(INT *A, INT *B, INT *AB, INT verbose_level);
	void GL_mult_internal(INT *A, INT *B, INT *AB, INT verbose_level);
	void GL_copy(INT *A, INT *B);
	void GL_copy_internal(INT *A, INT *B);
	void GL_transpose(INT *A, INT *At, INT verbose_level);
	void GL_transpose_internal(INT *A, INT *At, INT verbose_level);
	void GL_invert(INT *A, INT *Ainv);
	void GL_invert_internal(INT *A, INT *Ainv, INT verbose_level);
	void GL_unpack(UBYTE *elt, INT *Elt, INT verbose_level);
	void GL_pack(INT *Elt, UBYTE *elt);
	void GL_print_easy(INT *Elt, ostream &ost);
	void GL_code_for_make_element(INT *Elt, INT *data);
	void GL_print_for_make_element(INT *Elt, ostream &ost);
	void GL_print_for_make_element_no_commas(INT *Elt, ostream &ost);
	void GL_print_easy_normalized(INT *Elt, ostream &ost);
	void GL_print_latex(INT *Elt, ostream &ost);
	void GL_print_easy_latex(INT *Elt, ostream &ost);
	int get_digit(UBYTE *elt, INT i, INT j);
	int get_digit_frobenius(UBYTE *elt);
	void put_digit(UBYTE *elt, INT i, INT j, INT d);
	void put_digit_frobenius(UBYTE *elt, INT d);
	void make_element(INT *Elt, INT *data, INT verbose_level);
	void make_GL_element(INT *Elt, INT *A, INT f);
	void orthogonal_group_random_generator(action *A, orthogonal *O, 
		INT f_siegel, 
		INT f_reflection, 
		INT f_similarity,
		INT f_semisimilarity, 
		INT *Elt, INT verbose_level);
	void matrices_without_eigenvector_one(sims *S, INT *&Sol, INT &cnt, 
		INT f_path_select, INT select_value, 
		INT verbose_level);
	void matrix_minor(INT *Elt, INT *Elt1, 
		matrix_group *mtx1, INT f, INT verbose_level);
};


// #############################################################################
// perm_group.C:
// #############################################################################

class perm_group {

public:
	INT degree;
	
	INT f_induced_action;
	
	
	INT f_product_action;
	INT m;
	INT n;
	INT mn;
	INT offset;
	
	INT char_per_elt;
	INT elt_size_INT;
	
	INT *Elt1, *Elt2, *Elt3, *Elt4;
	UBYTE *elt1, *elt2, *elt3;
		// temporary storage, used in element_store()
	INT *Eltrk1, *Eltrk2, *Eltrk3;
		// used in store / retrieve
	
	page_storage *Elts;

	perm_group();
	~perm_group();
	void null();
	void free();
	void allocate();
	void init_product_action(INT m, INT n, 
		INT page_length_log, INT verbose_level);
	void init(INT degree, INT page_length_log, INT verbose_level);
	void init_data(INT page_length_log, INT verbose_level);
	void init_with_base(INT degree, 
		INT base_length, INT *base, INT page_length_log, 
		action &A, INT verbose_level);
	void transversal_rep(INT i, INT j, INT *Elt, INT verbose_level);
	void one(INT *Elt);
	INT is_one(INT *Elt);
	void mult(INT *A, INT *B, INT *AB);
	void copy(INT *A, INT *B);
	void invert(INT *A, INT *Ainv);
	void unpack(UBYTE *elt, INT *Elt);
	void pack(INT *Elt, UBYTE *elt);
	void print(INT *Elt, ostream &ost);
	void code_for_make_element(INT *Elt, INT *data);
	void print_for_make_element(INT *Elt, ostream &ost);
	void print_for_make_element_no_commas(INT *Elt, ostream &ost);
	void print_with_action(action *A, INT *Elt, ostream &ost);
	void make_element(INT *Elt, INT *data, INT verbose_level);

};



void perm_group_find_strong_generators_at_level(INT level, INT degree, 
	INT given_base_length, INT *given_base,
	INT nb_gens, INT *gens, 
	INT &nb_generators_found, INT *idx_generators_found);
void perm_group_generators_direct_product(INT nb_diagonal_elements,
	INT degree1, INT degree2, INT &degree3, 
	INT nb_gens1, INT nb_gens2, INT &nb_gens3, 
	INT *gens1, INT *gens2, INT *&gens3, 
	INT base_len1, INT base_len2, INT &base_len3, 
	INT *base1, INT *base2, INT *&base3);




// #############################################################################
// schreier.C:
// #############################################################################


class schreier {

public:
	action *A;
	vector_ge gens;
	vector_ge gens_inv;
	INT nb_images;
	INT **images;
		// [nb_gens][2 * A->degree], 
		// allocated by init_images, 
		// called from init_generators
	
	INT *orbit; // [A->degree]
	INT *orbit_inv; // [A->degree]

		// prev, label and orbit_no are indexed 
		// by the points in the order as listed in orbit.
	INT *prev; // [A->degree]
	INT *label; // [A->degree]
	//INT *orbit_no; // [A->degree]
		// to find out which orbit point a lies in, 
		// use orbit_number(pt).
		// It used to be orbit_no[orbit_inv[a]]

	INT *orbit_first;  // [A->degree + 1]
	INT *orbit_len;  // [A->degree]
	INT nb_orbits;
	
	INT *Elt1, *Elt2, *Elt3;
	INT *schreier_gen, *schreier_gen1;
		// used in random_schreier_generator
	INT *cosetrep, *cosetrep_tmp;
		// used in coset_rep / coset_rep_inv
	
	INT f_print_function;
	void (*print_function)(ostream &ost, INT pt, void *data);
	void *print_function_data;

	schreier();
	schreier(action *A);
	~schreier();
	void freeself();
	void delete_images();
	void init_images(INT nb_images, INT verbose_level);
	void images_append();
	void init(action *A);
	void init2();
	void initialize_tables();
	void init_single_generator(INT *elt);
	void init_generators(vector_ge &generators);
	void init_generators(INT nb, INT *elt);
		// elt must point to nb * A->elt_size_in_INT 
		// INT's that are 
		// group elements in INT format
	void init_generators_by_hdl(INT nb_gen, INT *gen_hdl, 
		INT verbose_level);
	INT get_image(INT i, INT gen_idx, INT verbose_level);
	void print_orbit_lengths(ostream &ost);
	void print_orbit_length_distribution(ostream &ost);
	void print_orbit_reps(ostream &ost);
	void print(ostream &ost);
	void print_and_list_orbits(ostream &ost);
	void print_and_list_orbits_tex(ostream &ost);
	void print_and_list_orbits_of_given_length(ostream &ost, 
		INT len);
	void print_and_list_orbits_and_stabilizer(ostream &ost, 
		action *default_action, longinteger_object &go, 
		void (*print_point)(ostream &ost, INT pt, void *data), 
			void *data);
	void print_and_list_orbits_using_labels(ostream &ost, 
		INT *labels);
	void print_tables(ostream &ost, INT f_with_cosetrep);
	void print_tables_latex(ostream &ost, INT f_with_cosetrep);
	void print_generators();
	void print_generators_with_permutations();
	void print_orbit(INT orbit_no);
	void print_orbit_using_labels(INT orbit_no, INT *labels);
	void print_orbit(ostream &ost, INT orbit_no);
	void print_orbit_tex(ostream &ost, INT orbit_no);
	void print_and_list_orbit_and_stabilizer_tex(INT i, 
		action *default_action, 
		longinteger_object &full_group_order, 
		ostream &ost);
	void print_and_list_orbit_and_stabilizer_with_list_of_elements_tex(
		INT i, action *default_action, 
		strong_generators *gens, ostream &ost);
	void print_and_list_orbit_tex(INT i, ostream &ost);
	void print_and_list_orbits_sorted_by_length_tex(ostream &ost);
	void print_and_list_orbits_and_stabilizer_sorted_by_length(
		ostream &ost, INT f_tex, 
		action *default_action, 
		longinteger_object &full_group_order);
	void 
print_and_list_orbits_and_stabilizer_sorted_by_length_and_list_stabilizer_elements(
		ostream &ost, INT f_tex, 
		action *default_action, strong_generators *gens_full_group);
	void print_and_list_orbits_sorted_by_length(ostream &ost);
	void print_and_list_orbits_sorted_by_length(ostream &ost, INT f_tex);
	void print_orbit_using_labels(ostream &ost, INT orbit_no, INT *labels);
	void print_orbit_using_callback(ostream &ost, INT orbit_no, 
		void (*print_point)(ostream &ost, INT pt, void *data), 
		void *data);
	void print_orbit_type(INT f_backwards);
	void list_all_orbits_tex(ostream &ost);
	void print_orbit_through_labels(ostream &ost, 
		INT orbit_no, INT *point_labels);
	void print_orbit_sorted(ostream &ost, INT orbit_no);
	void print_orbit(INT cur, INT last);
	void swap_points(INT i, INT j);
	void move_point_here(INT here, INT pt);
	INT orbit_representative(INT pt);
	INT depth_in_tree(INT j);
		// j is a coset, not a point
	void transporter_from_orbit_rep_to_point(INT pt, 
		INT &orbit_idx, INT *Elt, INT verbose_level);
	void transporter_from_point_to_orbit_rep(INT pt, 
		INT &orbit_idx, INT *Elt, INT verbose_level);
	void coset_rep(INT j);
		// j is a coset, not a point
		// result is in cosetrep
		// determines an element in the group 
		// that moves the orbit representative 
		// to the j-th point in the orbit.
	void coset_rep_with_verbosity(INT j, INT verbose_level);
	void coset_rep_inv(INT j);
	void get_schreier_vector(INT *&sv, 
		INT f_trivial_group, INT f_compact);
	void get_schreier_vector_compact(INT *&sv, 
		INT f_trivial_group);
		// allocates and creates array sv[size] using NEW_INT
		// where size is n + 1 if  f_trivial_group is TRUE
		// and size is 3 * n + 1 otherwise
		// Here, n is the combined size of all orbits 
		// counted by nb_orbits
		// sv[0] is equal to n
		// sv + 1 is the array point_list of size [n], 
		// listing the point in increasing order
		// if f_trivial_group, sv + 1 + n is the array prev[n] and 
		// sv + 1 + 2 * n is the array label[n] 
	void get_schreier_vector_ordinary(INT *&sv);
		// allocates and creates array sv[2 * A->degree] 
		// using NEW_INT
		// sv[i * 2 + 0] is prev[i]
		// sv[i * 2 + 1] is label[i]
	void extend_orbit(INT *elt, INT verbose_level);
	void compute_all_point_orbits(INT verbose_level);
	void compute_all_point_orbits_with_prefered_reps(
		INT *prefered_reps, INT nb_prefered_reps, 
		INT verbose_level);
	void compute_all_point_orbits_with_preferred_labels(
		INT *preferred_labels, INT verbose_level);
	void compute_all_orbits_on_invariant_subset(INT len, 
		INT *subset, INT verbose_level);
	INT sum_up_orbit_lengths();
	void compute_point_orbit(INT pt, INT verbose_level);
	void non_trivial_random_schreier_generator(action *A_original, 
		INT verbose_level);
		// computes non trivial random Schreier 
		// generator into schreier_gen
		// non-trivial is with respect to A_original
	void random_schreier_generator_ith_orbit(INT orbit_no, 
		INT verbose_level);
	void random_schreier_generator(INT verbose_level);
		// computes random Schreier generator into schreier_gen
	void trace_back(INT *path, INT i, INT &j);
	void print_tree(INT orbit_no);
	void draw_forest(const char *fname_mask, 
		INT xmax, INT ymax, 
		INT f_circletext, INT rad, 
		INT f_embedded, INT f_sideways, 
		double scale, double line_width, 
		INT f_has_point_labels, INT *point_labels, 
		INT verbose_level);
	void draw_tree(const char *fname, INT orbit_no, 
		INT xmax, INT ymax, INT f_circletext, INT rad, 
		INT f_embedded, INT f_sideways, 
		double scale, double line_width, 
		INT f_has_point_labels, INT *point_labels, 
		INT verbose_level);
	void draw_tree2(const char *fname, INT xmax, INT ymax, 
		INT f_circletext, 
		INT *weight, INT *placement_x, INT max_depth, 
		INT i, INT last, INT rad, 
		INT f_embedded, INT f_sideways, 
		double scale, double line_width, 
		INT f_has_point_labels, INT *point_labels, 
		INT verbose_level);
	void subtree_draw_lines(mp_graphics &G, INT f_circletext, 
		INT parent_x, INT parent_y, INT *weight, 
		INT *placement_x, INT max_depth, INT i, INT last, 
		INT verbose_level);
	void subtree_draw_vertices(mp_graphics &G, 
		INT f_circletext, INT parent_x, INT parent_y, INT *weight, 
		INT *placement_x, INT max_depth, INT i, INT last, INT rad, 
		INT f_has_point_labels, INT *point_labels, 
		INT verbose_level);
	void subtree_place(INT *weight, INT *placement_x, 
		INT left, INT right, INT i, INT last);
	INT subtree_calc_weight(INT *weight, INT &max_depth, 
		INT i, INT last);
	INT subtree_depth_first(ostream &ost, INT *path, INT i, INT last);
	void print_path(ostream &ost, INT *path, INT l);
	void intersection_vector(INT *set, INT len, 
		INT *intersection_cnt);
	void orbits_on_invariant_subset_fast(INT len, 
		INT *subset, INT verbose_level);
	void orbits_on_invariant_subset(INT len, INT *subset, 
		INT &nb_orbits_on_subset, INT *&orbit_perm, INT *&orbit_perm_inv);
	void get_orbit_partition_of_points_and_lines(
		partitionstack &S, INT verbose_level);
	void get_orbit_partition(partitionstack &S, 
		INT verbose_level);
	strong_generators *generators_for_stabilizer_of_arbitrary_point_and_transversal(
		action *default_action, 
		longinteger_object &full_group_order, 
		INT pt, vector_ge *&cosets, INT verbose_level);
	strong_generators *generators_for_stabilizer_of_arbitrary_point(
		action *default_action, 
		longinteger_object &full_group_order, INT pt, 
		INT verbose_level);
	strong_generators *generators_for_stabilizer_of_orbit_rep(
		action *default_action, 
		longinteger_object &full_group_order, 
		INT orbit_idx, INT verbose_level);
	void point_stabilizer(action *default_action, 
		longinteger_object &go, 
		sims *&Stab, INT orbit_no, INT verbose_level);
		// this function allocates a sims structure into Stab.
	void get_orbit(INT orbit_idx, INT *set, INT &len, 
		INT verbose_level);
	void compute_orbit_statistic(INT *set, INT set_size, 
		INT *orbit_count, INT verbose_level);
	void test_sv(action *A, INT *hdl_strong_generators, INT *sv, 
		INT f_trivial_group, INT f_compact, INT verbose_level);
	void write_to_memory_object(memory_object *m, INT verbose_level);
	void read_from_memory_object(memory_object *m, INT verbose_level);
	void write_file(BYTE *fname, INT verbose_level);
	void read_file(const BYTE *fname, INT verbose_level);
	void write_to_file_binary(ofstream &fp, INT verbose_level);
	void read_from_file_binary(ifstream &fp, INT verbose_level);
	void write_file_binary(BYTE *fname, INT verbose_level);
	void read_file_binary(const BYTE *fname, INT verbose_level);
	void orbits_as_set_of_sets(set_of_sets *&S, INT verbose_level);
	void get_orbit_reps(INT *&Reps, INT &nb_reps, INT verbose_level);
	INT find_shortest_orbit_if_unique(INT &idx);
	void elements_in_orbit_of(INT pt, INT *orb, INT &nb, 
		INT verbose_level);
	void get_orbit_lengths_once_each(INT *&orbit_lengths, 
		INT &nb_orbit_lengths);
	INT orbit_number(INT pt);
	void latex(const BYTE *fname);
	void get_orbit_decomposition_scheme_of_graph(
		INT *Adj, INT n, INT *&Decomp_scheme, INT verbose_level);
	void list_elements_as_permutations_vertically(ostream &ost);

};

// #############################################################################
// schreier_sims.C:
// #############################################################################

class schreier_sims {

public:
	action *GA;
	sims *G;

	INT f_interested_in_kernel;
	action *KA;
	sims *K;

	longinteger_object G_order, K_order, KG_order;
	
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;

	INT f_has_target_group_order;
	longinteger_object tgo; // target group order

	
	INT f_from_generators;
	vector_ge *gens;

	INT f_from_random_process;
	void (*callback_choose_random_generator)(INT iteration, 
		INT *Elt, void *data, INT verbose_level);
	void *callback_choose_random_generator_data;
	
	INT f_from_old_G;
	sims *old_G;

	INT f_has_base_of_choice;
	INT base_of_choice_len;
	INT *base_of_choice;

	INT f_override_choose_next_base_point_method;
	INT (*choose_next_base_point_method)(action *A, 
		INT *Elt, INT verbose_level); 

	INT iteration;

	schreier_sims();
	~schreier_sims();
	void null();
	void freeself();
	void init(action *A, INT verbose_level);
	void interested_in_kernel(action *KA, INT verbose_level);
	void init_target_group_order(longinteger_object &tgo, 
		INT verbose_level);
	void init_generators(vector_ge *gens, INT verbose_level);
	void init_random_process(
		void (*callback_choose_random_generator)(
		INT iteration, INT *Elt, void *data, 
		INT verbose_level), 
		void *callback_choose_random_generator_data, 
		INT verbose_level);
	void init_old_G(sims *old_G, INT verbose_level);
	void init_base_of_choice(
		INT base_of_choice_len, INT *base_of_choice, 
		INT verbose_level);
	void init_choose_next_base_point_method(
		INT (*choose_next_base_point_method)(action *A, 
		INT *Elt, INT verbose_level), 
		INT verbose_level);
	void compute_group_orders();
	void print_group_orders();
	void get_generator_internal(INT *Elt, INT verbose_level);
	void get_generator_external(INT *Elt, INT verbose_level);
	void get_generator_external_from_generators(INT *Elt, 
		INT verbose_level);
	void get_generator_external_random_process(INT *Elt, 
		INT verbose_level);
	void get_generator_external_old_G(INT *Elt, 
		INT verbose_level);
	void get_generator(INT *Elt, INT verbose_level);
	void closure_group(INT verbose_level);
	void create_group(INT verbose_level);
};

// #############################################################################
// sims.C:
// #############################################################################

class sims {

public:
	action *A;

	INT my_base_len;

	vector_ge gens;
	vector_ge gens_inv;
	
	INT *gen_depth; // [nb_gen]
	INT *gen_perm; // [nb_gen]
	
	INT *nb_gen; // [base_len + 1]
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
	
	INT *path; // [base_len]
	
	INT nb_images;
	INT **images;
	
	INT *orbit_len;
		// [base_len]
		// orbit_len[i] is the length of the i-th basic orbit.
	
	INT **orbit;
		// [base_len][A->deg]
		// orbit[i][j] is the j-th point in the orbit 
		// of the i-th base point.
		// for 0 \le j < orbit_len[i].
		// for orbit_len[i] \le j < A->deg, 
		// the points not in the orbit are listed.
	INT **orbit_inv;
		// [base_len][A->deg]
		// orbit[i] is the inverse of the permutation orbit[i],
		// i.e. given a point j,
		// orbit_inv[i][j] is the coset (or position in the orbit)
		// which contains j.
	
	INT **prev; // [base_len][A->deg]
	INT **label; // [base_len][A->deg]
	
	
	// storage for temporary data and 
	// group elements computed by various routines.
	INT *Elt1, *Elt2, *Elt3, *Elt4;
	INT *strip1, *strip2;
		// used in strip
	INT *eltrk1, *eltrk2;
		// used in element rank unrank
	INT *cosetrep, *cosetrep_tmp;
		// used in coset_rep / coset_rep_inv
	INT *schreier_gen, *schreier_gen1;
		// used in random_schreier_generator
	
	sims();
	void null();
	sims(action *A);
	~sims();
	void freeself();

	void delete_images();
	void init_images(INT nb_images);
	void images_append();
	void init(action *A);
		// initializes the trivial group 
		// with the base as given in A
	void init_without_base(action *A);
	void reallocate_base(INT old_base_len, INT verbose_level);
	void initialize_table(INT i);
	void init_trivial_group(INT verbose_level);
		// clears the generators array, 
		// and sets the i-th transversal to contain
		// only the i-th base point (for all i).
	void init_trivial_orbit(INT i);
	void init_generators(vector_ge &generators, INT verbose_level);
	void init_generators(INT nb, INT *elt, INT verbose_level);
		// copies the given elements into the generator array, 
		// then computes depth and perm
	void init_generators_by_hdl(INT nb_gen, INT *gen_hdl);
	void init_generator_depth_and_perm(INT verbose_level);
	void add_generator(INT *elt, INT verbose_level);
		// adds elt to list of generators, 
		// computes the depth of the element, 
		// updates the arrays gen_depth and gen_perm accordingly
		// does not change the transversals
	void create_group_tree(const BYTE *fname, INT f_full, 
		INT verbose_level);
	void print_transversals();
	void print_transversals_short();
	void print_transversal_lengths();
	void print_orbit_len();
	void print(INT verbose_level);
	void print_generators();
	void print_generators_tex(ostream &ost);
	void print_generators_as_permutations();
	void print_generators_as_permutations_override_action(
		action *A);
	void print_basic_orbits();
	void print_basic_orbit(INT i);
	//void sort();
	//void sort_basic_orbit(INT i);
	void print_generator_depth_and_perm();
	INT generator_depth(INT gen_idx);
		// returns the index of the first base point 
		// which is moved by a given generator. 
	INT generator_depth(INT *elt);
		// returns the index of the first base point 
		// which is moved by the given element
	void group_order(longinteger_object &go);
	INT group_order_INT();
	void print_group_order(ostream &ost);
	void print_group_order_factored(ostream &ost);
	INT is_trivial_group();
	INT last_moved_base_point();
		// j == -1 means the group is trivial
	INT get_image(INT i, INT gen_idx);
		// get the image of a point i under generator gen_idx, 
		// goes through a 
		// table of stored images by default. 
		// Computes the image only if not yet available.
	INT get_image(INT i, INT *elt);
		// get the image of a point i under a given group element, 
		// does not go through a table.
	void swap_points(INT lvl, INT i, INT j);
		// swaps two points given by their cosets
	void random_element(INT *elt, INT verbose_level);
		// compute a random element among the group elements 
		// represented by the chain
		// (chooses random cosets along the stabilizer chain)
	void random_element_of_order(INT *elt, INT order, 
		INT verbose_level);
	void random_elements_of_order(vector_ge *elts, 
		INT *orders, INT nb, INT verbose_level);
	void path_unrank_INT(INT a);
	INT path_rank_INT();
	
	void element_from_path(INT *elt, INT verbose_level);
		// given coset representatives in path[], 
		// the corresponding 
		// element is multiplied.
		// uses eltrk1, eltrk2
	void element_from_path_inv(INT *elt);
	void element_unrank(longinteger_object &a, INT *elt, 
		INT verbose_level);
	void element_unrank(longinteger_object &a, INT *elt);
		// Returns group element whose rank is a. 
		// the elements represented by the chain are 
		// enumerated 0, ... go - 1
		// with the convention that 0 always stands 
		// for the identity element.
		// The computed group element will be computed into Elt1
	void element_rank(longinteger_object &a, INT *elt);
		// Computes the rank of the element in elt into a.
		// uses eltrk1, eltrk2
	void element_unrank_INT(INT rk, INT *Elt, INT verbose_level);
	void element_unrank_INT(INT rk, INT *Elt);
	INT element_rank_INT(INT *Elt);
	INT is_element_of(INT *elt);
	void test_element_rank_unrank();
	void coset_rep(INT i, INT j, INT verbose_level);
		// computes a coset representative in transversal i 
		// which maps
		// the i-th base point to the point which is in 
		// coset j of the i-th basic orbit.
		// j is a coset, not a point
		// result is in cosetrep
	INT compute_coset_rep_depth(INT i, INT j, INT verbose_level);
	void compute_coset_rep_path(INT i, INT j, INT *&Path, 
		INT *&Label, INT &depth, INT verbose_level);
	void coset_rep_inv(INT i, INT j, INT verbose_level_le);
		// computes the inverse element of what coset_rep computes,
		// i.e. an element which maps the 
		// j-th point in the orbit to the 
		// i-th base point.
		// j is a coset, not a point
		// result is in cosetrep
	void compute_base_orbits(INT verbose_level);
	void compute_base_orbits_known_length(INT *tl, 
		INT verbose_level);
	void extend_base_orbit(INT new_gen_idx, INT lvl, 
		INT verbose_level);
	void compute_base_orbit(INT lvl, INT verbose_level);
		// applies all generators at the given level to compute
		// the corresponding basic orbit.
		// the generators are the first nb_gen[lvl] 
		// in the generator array
	void compute_base_orbit_known_length(INT lvl, 
		INT target_length, INT verbose_level);
	void extract_strong_generators_in_order(vector_ge &SG, 
		INT *tl, INT verbose_level);
	void transitive_extension(schreier &O, vector_ge &SG, 
		INT *tl, INT verbose_level);
	INT transitive_extension_tolerant(schreier &O, 
		vector_ge &SG, INT *tl, INT f_tolerant, 
		INT verbose_level);
	void transitive_extension_using_coset_representatives_extract_generators(
		INT *coset_reps, INT nb_cosets, 
		vector_ge &SG, INT *tl, 
		INT verbose_level);
	void transitive_extension_using_coset_representatives(
		INT *coset_reps, INT nb_cosets, 
		INT verbose_level);
	void transitive_extension_using_generators(
		INT *Elt_gens, INT nb_gens, INT subgroup_index, 
		vector_ge &SG, INT *tl, 
		INT verbose_level);
	void point_stabilizer_stabchain_with_action(action *A2, 
		sims &S, INT pt, INT verbose_level);
		// first computes the orbit of the point pt 
		// in action A2 under the generators 
		// that are stored at present 
		// (using a temporary schreier object),
		// then sifts random schreier generators into S
	void point_stabilizer(vector_ge &SG, INT *tl, 
		INT pt, INT verbose_level);
		// computes strong generating set 
		// for the stabilizer of point pt
	void point_stabilizer_with_action(action *A2, 
		vector_ge &SG, INT *tl, INT pt, INT verbose_level);
		// computes strong generating set for 
		// the stabilizer of point pt in action A2
	INT strip_and_add(INT *elt, INT *residue, INT verbose_level);
		// returns TRUE if something was added, 
		// FALSE if element stripped through
	INT strip(INT *elt, INT *residue, INT &drop_out_level, 
		INT &image, INT verbose_level);
		// returns TRUE if the element sifts through
	void add_generator_at_level(INT *elt, INT lvl, 
		INT verbose_level);
		// add the generator to the array of generators 
		// and then extends the 
		// basic orbits 0,..,lvl using extend_base_orbit
	void add_generator_at_level_only(INT *elt, 
		INT lvl, INT verbose_level);
		// add the generator to the array of generators 
		// and then extends the 
		// basic orbit lvl using extend_base_orbit
	void random_schreier_generator(INT verbose_level);
		// computes random Schreier generator into schreier_gen
	void build_up_group_random_process_no_kernel(sims *old_G, 
		INT verbose_level);
	void extend_group_random_process_no_kernel(sims *extending_by_G, 
		longinteger_object &target_go, 
		INT verbose_level);
	void conjugate(action *A, sims *old_G, INT *Elt, 
		INT f_overshooting_OK, INT verbose_level);
		// Elt * g * Elt^{-1} where g is in old_G
	INT test_if_in_set_stabilizer(action *A, 
		INT *set, INT size, INT verbose_level);
	INT test_if_subgroup(sims *old_G, INT verbose_level);
	void build_up_group_random_process(sims *K, sims *old_G, 
		longinteger_object &target_go, 
		INT f_override_choose_next_base_point,
		INT (*choose_next_base_point_method)(action *A, 
			INT *Elt, INT verbose_level), 
		INT verbose_level);
	void build_up_group_from_generators(sims *K, vector_ge *gens, 
		INT f_target_go, longinteger_object *target_go, 
		INT f_override_choose_next_base_point,
		INT (*choose_next_base_point_method)(action *A, 
			INT *Elt, INT verbose_level), 
		INT verbose_level);
	INT closure_group(INT nb_times, INT verbose_level);
	void write_all_group_elements(BYTE *fname, INT verbose_level);
	void print_all_group_elements_to_file(BYTE *fname, 
		INT verbose_level);
	void print_all_group_elements();
	void print_all_group_elements_as_permutations();
	void print_all_group_elements_as_permutations_in_special_action(
		action *A_special);
	void print_all_transversal_elements();
	void element_as_permutation(action *A_special, 
		INT elt_rk, INT *perm, INT verbose_level);
	void table_of_group_elements_in_data_form(INT *&Table, 
		INT &len, INT &sz, INT verbose_level);
	void regular_representation(INT *Elt, INT *perm, 
		INT verbose_level);
	void center(vector_ge &gens, INT *center_element_ranks, 
		INT &nb_elements, INT verbose_level);
	void all_cosets(INT *subset, INT size, 
		INT *all_cosets, INT verbose_level);
	void element_ranks_subgroup(sims *subgroup, 
		INT *element_ranks, INT verbose_level);
	void find_standard_generators_INT(INT ord_a, INT ord_b, 
		INT ord_ab, INT &a, INT &b, INT &nb_trials, 
		INT verbose_level);
	INT find_element_of_given_order_INT(INT ord, 
		INT &nb_trials, INT verbose_level);
	INT find_element_of_given_order_INT(INT *Elt, 
		INT ord, INT &nb_trials, INT max_trials, 
		INT verbose_level);
	void find_element_of_prime_power_order(INT p, 
		INT *Elt, INT &e, INT &nb_trials, INT verbose_level);
	void save_list_of_elements(BYTE *fname, 
		INT verbose_level);
	void read_list_of_elements(action *A, 
		BYTE *fname, INT verbose_level);
	void evaluate_word_INT(INT word_len, 
		INT *word, INT *Elt, INT verbose_level);
	void write_sgs(const BYTE *fname, INT verbose_level);
	void read_sgs(const BYTE *fname, vector_ge *SG, 
		INT verbose_level);
	INT least_moved_point_at_level(INT lvl, INT verbose_level);
	INT mult_by_rank(INT rk_a, INT rk_b, INT verbose_level);
	INT mult_by_rank(INT rk_a, INT rk_b);
	INT invert_by_rank(INT rk_a, INT verbose_level);
	INT conjugate_by_rank(INT rk_a, INT rk_b, INT verbose_level);
		// comutes b^{-1} * a * b
	INT conjugate_by_rank_b_bv_given(INT rk_a, 
		INT *Elt_b, INT *Elt_bv, INT verbose_level);
	void sylow_subgroup(INT p, sims *P, INT verbose_level);
	INT is_normalizing(INT *Elt, INT verbose_level);
	void create_Cayley_graph(vector_ge *gens, INT *&Adj, INT &n, 
		INT verbose_level);
	void create_group_table(INT *&Table, INT &n, INT verbose_level);
	void write_as_magma_permutation_group(const BYTE *fname_base, 
		vector_ge *gens, INT verbose_level);

	// sims2.C:
	void build_up_subgroup_random_process(sims *G, 
		void (*choose_random_generator_for_subgroup)(
			sims *G, INT *Elt, INT verbose_level), 
		INT verbose_level);

	// sims3.C:
	void subgroup_make_characteristic_vector(sims *Sub, 
		INT *C, INT verbose_level);
	void normalizer_based_on_characteristic_vector(INT *C_sub, 
		INT *Gen_idx, INT nb_gens, INT *N, INT &N_go, 
		INT verbose_level);
	void order_structure_relative_to_subgroup(INT *C_sub, 
		INT *Order, INT *Residue, INT verbose_level);
};

// sims2.C:
void choose_random_generator_derived_group(sims *G, INT *Elt, 
	INT verbose_level);


// in sims_global.C:
sims *create_sims_from_generators_with_target_group_order_factorized(
		action *A, 
		vector_ge *gens, INT *tl, INT len, INT verbose_level);
sims *create_sims_from_generators_with_target_group_order(action *A, 
		vector_ge *gens, longinteger_object &target_go, 
		INT verbose_level);
sims *create_sims_from_generators_with_target_group_order_INT(action *A, 
	vector_ge *gens, INT target_go, INT verbose_level);
sims *create_sims_from_generators_without_target_group_order(action *A, 
	vector_ge *gens, INT verbose_level);
sims *create_sims_from_single_generator_without_target_group_order(action *A, 
	INT *Elt, INT verbose_level);
sims *create_sims_from_generators_randomized(action *A, 
	vector_ge *gens, INT f_target_go, 
	longinteger_object &target_go, INT verbose_level);
sims *create_sims_for_centralizer_of_matrix(action *A, 
	INT *Mtx, INT verbose_level);








// #############################################################################
// strong_generators.C:
// #############################################################################


class strong_generators {
public:

	action *A;
	INT *tl;
	vector_ge *gens;

	strong_generators();
	~strong_generators();
	void null();
	void freeself();
	void swap_with(strong_generators *SG);
	void init(action *A);
	void init(action *A, INT verbose_level);
	void init_from_sims(sims *S, INT verbose_level);
	void init_from_ascii_coding(action *A, 
		BYTE *ascii_coding, INT verbose_level);
	strong_generators *create_copy();
	void init_copy(strong_generators *S, 
		INT verbose_level);
	void init_by_hdl(action *A, INT *gen_hdl, 
		INT nb_gen, INT verbose_level);
	void init_from_permutation_representation(action *A, 
		INT *data, 
		INT nb_elements, INT group_order, 
		INT verbose_level);
	void init_from_data(action *A, INT *data, 
		INT nb_elements, INT elt_size, 
		INT *transversal_length, 
		INT verbose_level);
	void init_from_data_with_target_go_ascii(action *A, 
		INT *data, 
		INT nb_elements, INT elt_size, 
		const BYTE *ascii_target_go,
		INT verbose_level);
	void init_from_data_with_target_go(action *A, 
		INT *data_gens, 
		INT data_gens_size, INT nb_gens, 
		longinteger_object &target_go, 
		INT verbose_level);
	void init_point_stabilizer_of_arbitrary_point_through_schreier(
		schreier *Sch, 
		INT pt, INT &orbit_idx, 
		longinteger_object &full_group_order, 
		INT verbose_level);
	void init_point_stabilizer_orbit_rep_schreier(schreier *Sch, 
		INT orbit_idx, longinteger_object &full_group_order, 
		INT verbose_level);
	void init_generators_for_the_conjugate_group_avGa(
		strong_generators *SG, INT *Elt_a, INT verbose_level);
	void init_generators_for_the_conjugate_group_aGav(
		strong_generators *SG, INT *Elt_a, INT verbose_level);
	void init_transposed_group(strong_generators *SG, 
		INT verbose_level);
	void init_group_extension(strong_generators *subgroup, 
		INT *data, INT index, 
		INT verbose_level);
	void init_group_extension(strong_generators *subgroup, 
		vector_ge *new_gens, INT index, 
		INT verbose_level);
	void switch_to_subgroup(const BYTE *rank_vector_text, 
		const BYTE *subgroup_order_text, sims *S, 
		INT *&subgroup_gens_idx, INT &nb_subgroup_gens, 
		INT verbose_level);
	void init_subgroup(action *A, INT *subgroup_gens_idx, 
		INT nb_subgroup_gens, 
		const BYTE *subgroup_order_text, 
		sims *S, 
		INT verbose_level);
	sims *create_sims(INT verbose_level);
	sims *create_sims_in_different_action(action *A_given, 
		INT verbose_level);
	void add_generators(vector_ge *coset_reps, 
		INT group_index, INT verbose_level);
	void add_single_generator(INT *Elt, 
		INT group_index, INT verbose_level);
	void group_order(longinteger_object &go);
	INT group_order_as_INT();
	void print_generators();
	void print_generators_ost(ostream &ost);
	void print_generators_in_source_code();
	void print_generators_in_source_code_to_file(
	const BYTE *fname);
	void print_generators_even_odd();
	void print_generators_MAGMA(action *A, ostream &ost);
	void print_generators_tex();
	void print_generators_tex(ostream &ost);
	void print_generators_as_permutations();
	void print_with_given_action(ostream &ost, action *A2);
	void print_elements_ost(ostream &ost);
	void print_elements_latex_ost(ostream &ost);
	void create_group_table(INT *&Table, INT &go, 
		INT verbose_level);
	void list_of_elements_of_subgroup(
		strong_generators *gens_subgroup, 
		INT *&Subgroup_elements_by_index, 
		INT &sz_subgroup, INT verbose_level);
	void compute_schreier_with_given_action(action *A_given, 
		schreier *&Sch, INT verbose_level);
	void compute_schreier_with_given_action_on_a_given_set(
		action *A_given, 
		schreier *&Sch, INT *set, INT len, INT verbose_level);
	void orbits_on_points(INT &nb_orbits, INT *&orbit_reps, 
		INT verbose_level);
	void orbits_on_points_with_given_action(action *A_given, 
		INT &nb_orbits, INT *&orbit_reps, INT verbose_level);
	schreier *orbits_on_points_schreier(action *A_given, 
		INT verbose_level);
	schreier *orbit_of_one_point_schreier(action *A_given, 
		INT pt, INT verbose_level);
	void orbits_light(action *A_given, 
		INT *&Orbit_reps, INT *&Orbit_lengths, INT &nb_orbits, 
		INT **&Pts_per_generator, INT *&Nb_per_generator, 
		INT verbose_level);
	void write_to_memory_object(memory_object *m, INT verbose_level);
	void read_from_memory_object(memory_object *m, INT verbose_level);
	void write_to_file_binary(ofstream &fp, INT verbose_level);
	void read_from_file_binary(action *A, ifstream &fp, 
		INT verbose_level);
	void write_file(const BYTE *fname, INT verbose_level);
	void read_file(action *A, const BYTE *fname, INT verbose_level);
	void generators_for_shallow_schreier_tree(BYTE *label, 
		vector_ge *chosen_gens, INT verbose_level);
	void compute_ascii_coding(BYTE *&ascii_coding, INT verbose_level);
	void decode_ascii_coding(BYTE *ascii_coding, INT verbose_level);
	void export_permutation_group_to_magma(const BYTE *fname, 
		INT verbose_level);
	void compute_and_print_orbits_on_a_given_set(action *A_given,
		INT *set, INT len, INT verbose_level);
	void compute_and_print_orbits(action *A_given, 
		INT verbose_level);
	INT test_if_normalizing(sims *S, INT verbose_level);
	void test_if_set_is_invariant_under_given_action(action *A_given, 
		INT *set, INT set_sz, INT verbose_level);
	strong_generators *point_stabilizer(INT pt, INT verbose_level);
	void make_element_which_moves_a_point_from_A_to_B(action *A_given, 
		INT pt_A, INT pt_B, INT *Elt, INT verbose_level);

	// strong_generators_groups.C:
	void init_linear_group_from_scratch(action *&A, 
		finite_field *F, INT n, 
		INT f_projective, INT f_general, INT f_affine, 
		INT f_semilinear, INT f_special, 
		INT verbose_level);
	void special_subgroup(INT verbose_level);
	void even_subgroup(INT verbose_level);
	void init_single(action *A, INT *Elt, INT verbose_level);
	void init_trivial_group(action *A, INT verbose_level);
	void generators_for_the_monomial_group(action *A, 
		matrix_group *Mtx, INT verbose_level);
	void generators_for_the_diagonal_group(action *A, 
		matrix_group *Mtx, INT verbose_level);
	void generators_for_the_singer_cycle(action *A, 
		matrix_group *Mtx, INT power_of_singer, INT verbose_level);
	void generators_for_the_null_polarity_group(action *A, 
		matrix_group *Mtx, INT verbose_level);
	void generators_for_symplectic_group(action *A, 
		matrix_group *Mtx, INT verbose_level);
	void init_centralizer_of_matrix(action *A, INT *Mtx, 
		INT verbose_level);
	void init_centralizer_of_matrix_general_linear(action *A_projective, 
		action *A_general_linear, INT *Mtx, 
		INT verbose_level);
	void field_reduction(action *Aq, INT n, INT s, 
		finite_field *Fq, INT verbose_level);
	void generators_for_translation_plane_in_andre_model(
		action *A_PGL_n1_q, action *A_PGL_n_q, 
		matrix_group *Mtx_n1, matrix_group *Mtx_n, 
		vector_ge *spread_stab_gens, 
		longinteger_object &spread_stab_go, 
		INT verbose_level);
	void generators_for_the_stabilizer_of_two_components(
		action *A_PGL_n_q, 
		matrix_group *Mtx, INT verbose_level);
	void regulus_stabilizer(action *A_PGL_n_q, 
		matrix_group *Mtx, INT verbose_level);
	void generators_for_the_borel_subgroup_upper(action *A_linear, 
		matrix_group *Mtx, INT verbose_level);
	void generators_for_the_borel_subgroup_lower(action *A_linear, 
		matrix_group *Mtx, INT verbose_level);
	void generators_for_the_identity_subgroup(action *A_linear, 
		matrix_group *Mtx, INT verbose_level);
	void generators_for_parabolic_subgroup(action *A_PGL_n_q, 
		matrix_group *Mtx, INT k, INT verbose_level);
	void generators_for_stabilizer_of_three_collinear_points_in_PGL4(
		action *A_PGL_4_q, 
		matrix_group *Mtx, INT verbose_level);
	void generators_for_stabilizer_of_triangle_in_PGL4(
		action *A_PGL_4_q, 
		matrix_group *Mtx, INT verbose_level);
	void generators_for_the_orthogonal_group(action *A, 
		finite_field *F, INT n, 
		INT epsilon, 
		INT f_semilinear, 
		INT verbose_level);
	void generators_for_the_stabilizer_of_the_cubic_surface(
		action *A, 
		finite_field *F, INT iso, 
		INT verbose_level);
	void generators_for_the_stabilizer_of_the_cubic_surface_family_24(
		action *A, 
		finite_field *F, INT f_with_normalizer, INT f_semilinear, 
		INT verbose_level);
	void BLT_set_from_catalogue_stabilizer(action *A, 
		finite_field *F, INT iso, 
		INT verbose_level);
	void stabilizer_of_spread_from_catalogue(action *A, 
		INT q, INT k, INT iso, 
		INT verbose_level);


};

void strong_generators_array_write_file(const BYTE *fname, 
	strong_generators *p, INT nb, INT verbose_level);
void strong_generators_array_read_from_file(const BYTE *fname, 
	action *A, strong_generators *&p, INT &nb, INT verbose_level);

// #############################################################################
// subgroup.C:
// #############################################################################

class subgroup {
public:
	INT *Elements;
	INT group_order;
	INT *gens;
	INT nb_gens;


	subgroup();
	~subgroup();
	void null();
	void freeself();
	void init(INT *Elements, INT group_order, INT *gens, INT nb_gens);
	void print();
	INT contains_this_element(INT elt);

};


// #############################################################################
// wreath_product.C:
// #############################################################################

class wreath_product {

public:
	matrix_group *M;
	action *A_mtx;
	finite_field *F;
	INT q;
	INT nb_factors;

	BYTE label[1000];
	BYTE label_tex[1000];

	INT degree_of_matrix_group;
	INT dimension_of_matrix_group;
	INT dimension_of_tensor_action;
	INT degree_of_tensor_action;
	INT degree_overall;
	INT low_level_point_size;
	INT make_element_size;
	perm_group *P;
	INT elt_size_INT;

	INT *perm_offset_i;
	INT *mtx_size;
	INT *index_set1;
	INT *index_set2;
	INT *u;
	INT *v;
	INT *w;
	INT *A1;
	INT *A2;
	INT *A3;
	INT *tmp_Elt1;
	INT *tmp_perm1;
	INT *tmp_perm2;
	INT *induced_perm; // [dimension_of_tensor_action]

	INT bits_per_digit;
	INT bits_per_elt;
	INT char_per_elt;

	UBYTE *elt1;

	INT base_len_in_component;
	INT *base_for_component;
	INT *tl_for_component;

	INT base_length;
	INT *the_base;
	INT *the_transversal_length;

	page_storage *Elts;

	wreath_product();
	~wreath_product();
	void null();
	void freeself();
	void init_tensor_wreath_product(matrix_group *M,
			action *A_mtx, INT nb_factors, INT verbose_level);
	INT element_image_of(INT *Elt, INT a, INT verbose_level);
	void element_image_of_low_level(INT *Elt,
			INT *input, INT *output, INT verbose_level);
		// we assume that we are in the tensor product domain
	void element_one(INT *Elt);
	INT element_is_one(INT *Elt);
	void element_mult(INT *A, INT *B, INT *AB, INT verbose_level);
	void element_move(INT *A, INT *B, INT verbose_level);
	void element_invert(INT *A, INT *Av, INT verbose_level);
	void compute_induced_permutation(INT *Elt, INT *perm);
	void apply_permutation(INT *Elt, INT *v_in, INT *v_out, INT verbose_level);
	INT offset_i(INT i);
	void create_matrix(INT *Elt, INT *A, INT verbose_level);
		// uses A1, A2
	void element_pack(INT *Elt, UBYTE *elt);
	void element_unpack(UBYTE *elt, INT *Elt);
	void put_digit(UBYTE *elt, INT f, INT i, INT j, INT d);
	INT get_digit(UBYTE *elt, INT f, INT i, INT j);
	void make_element_from_one_component(INT *Elt, INT f, INT *Elt_component);
	void make_element_from_permutation(INT *Elt, INT *perm);
	void make_element(INT *Elt, INT *data, INT verbose_level);
	void element_print_easy(INT *Elt, ostream &ost);
	void compute_base_and_transversals(INT verbose_level);
	void make_strong_generators_data(INT *&data,
			INT &size, INT &nb_gens, INT verbose_level);
};

