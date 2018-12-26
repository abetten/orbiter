// group_theory.h
//
// Anton Betten
//
// moved here from action.h: July 28, 2018
// based on action.h which was started:  August 13, 2005


// #############################################################################
// direct_product.C:
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

	char label[1000];
	char label_tex[1000];

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
	int *base_for_component1;
	int *tl_for_component1;

	int base_len_in_component2;
	int *base_for_component2;
	int *tl_for_component2;

	int base_length;
	int *the_base;
	int *the_transversal_length;

	page_storage *Elts;

	direct_product();
	~direct_product();
	void null();
	void freeself();
	void init(matrix_group *M1, matrix_group *M2,
			int verbose_level);
	int element_image_of(int *Elt, int a, int verbose_level);
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
	void element_print_easy(int *Elt, ostream &ost);
	void compute_base_and_transversals(int verbose_level);
	void make_strong_generators_data(int *&data,
			int &size, int &nb_gens, int verbose_level);
	void lift_generators(
			strong_generators *SG1,
			strong_generators *SG2,
			action *A, strong_generators *&SG3,
			int verbose_level);
};


//! used to create a linear group from command line arguments

// #############################################################################
// linear_group.C:
// #############################################################################


class linear_group {
public:
	linear_group_description *description;
	int n;
	int input_q;
	finite_field *F;
	int f_semilinear;

	char prefix[1000];
	strong_generators *initial_strong_gens;
	action *A_linear;
	matrix_group *Mtx;

	int f_has_strong_generators;
	strong_generators *Strong_gens;
	action *A2;
	int vector_space_dimension;
	int q;

	linear_group();
	~linear_group();
	void null();
	void freeself();
	void init(linear_group_description *description, 
		int verbose_level);
	void init_PGL2q_OnConic(char *prefix, int verbose_level);
	void init_wedge_action(char *prefix, int verbose_level);
	void init_monomial_group(char *prefix, int verbose_level);
	void init_diagonal_group(char *prefix, int verbose_level);
	void init_singer_group(char *prefix, int singer_power, 
		int verbose_level);
	void init_null_polarity_group(char *prefix, int verbose_level);
	void init_borel_subgroup_upper(char *prefix, int verbose_level);
	void init_identity_subgroup(char *prefix, int verbose_level);
	void init_symplectic_group(char *prefix, int verbose_level);
	void init_subfield_structure_action(char *prefix, int s, 
		int verbose_level);
	void init_orthogonal_group(char *prefix, 
		int epsilon, int verbose_level);
	void init_subgroup_from_file(char *prefix, 
		const char *subgroup_fname, const char *subgroup_label, 
		int verbose_level);
};

//! description of a linear group from the command line


// #############################################################################
// linear_group_description.C:
// #############################################################################



class linear_group_description {
public:
	int f_projective;
	int f_general;
	int f_affine;

	int n;
	int input_q;
	finite_field *F;
	int f_semilinear;
	int f_special;

	int f_wedge_action;
	int f_PGL2OnConic;
	int f_monomial_group;
	int f_diagonal_group;
	int f_null_polarity_group;
	int f_symplectic_group;
	int f_singer_group;
	int singer_power;
	int f_subfield_structure_action;
	int s;
	int f_subgroup_from_file;
	int f_borel_subgroup_upper;
	int f_borel_subgroup_lower;
	int f_identity_group;
	const char *subgroup_fname;
	const char *subgroup_label;
	int f_orthogonal_group;
	int orthogonal_group_epsilon;

	int f_on_k_subspaces;
	int on_k_subspaces_k;


	linear_group_description();
	~linear_group_description();
	void null();
	void freeself();
	void read_arguments_from_string(
			const char *str, int verbose_level);
	int read_arguments(int argc, const char **argv, 
		int verbose_level);
};


// #############################################################################
// matrix_group.C:
// #############################################################################

//! A linear group implemented as matrices

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

	char label[1000];
	char label_tex[1000];
	
	int f_GFq_is_allocated;
		// if TRUE, GFq will be destroyed in the destructor
		// if FALSE, it is the responsability 
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
	int image_of_element(int *Elt, int a, int verbose_level);
	int GL_image_of_PG_element(int *Elt, int a, int verbose_level);
	int GL_image_of_AG_element(int *Elt, int a, int verbose_level);
	void action_from_the_right_all_types(
		int *v, int *A, int *vA, int verbose_level);
	void projective_action_from_the_right(
		int *v, int *A, int *vA, int verbose_level);
	void general_linear_action_from_the_right(
		int *v, int *A, int *vA, int verbose_level);
	void substitute_surface_eqation(int *Elt,
			int *coeff_in, int *coeff_out, surface *Surf,
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
	void GL_print_easy(int *Elt, ostream &ost);
	void GL_code_for_make_element(int *Elt, int *data);
	void GL_print_for_make_element(int *Elt, ostream &ost);
	void GL_print_for_make_element_no_commas(int *Elt, ostream &ost);
	void GL_print_easy_normalized(int *Elt, ostream &ost);
	void GL_print_latex(int *Elt, ostream &ost);
	void GL_print_easy_latex(int *Elt, ostream &ost);
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
			int *base, int *transversal_length,
			int verbose_level);
	void strong_generators_low_level(int *&data,
			int &size, int &nb_gens, int verbose_level);
};


// #############################################################################
// perm_group.C:
// #############################################################################

//! An abstract permutation group

class perm_group {

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

	perm_group();
	~perm_group();
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
	void print(int *Elt, ostream &ost);
	void code_for_make_element(int *Elt, int *data);
	void print_for_make_element(int *Elt, ostream &ost);
	void print_for_make_element_no_commas(int *Elt, ostream &ost);
	void print_with_action(action *A, int *Elt, ostream &ost);
	void make_element(int *Elt, int *data, int verbose_level);

};




// #############################################################################
// schreier.C:
// #############################################################################

//! Schreier trees for orbits on points

class schreier {

public:
	action *A;
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

		// prev, label and orbit_no are indexed 
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
	void (*print_function)(ostream &ost, int pt, void *data);
	void *print_function_data;

	schreier();
	schreier(action *A);
	~schreier();
	void freeself();
	void delete_images();
	void init_images(int nb_images, int verbose_level);
	void images_append();
	void init(action *A);
	void init2();
	void initialize_tables();
	void init_single_generator(int *elt);
	void init_generators(vector_ge &generators);
	void init_generators(int nb, int *elt);
		// elt must point to nb * A->elt_size_in_int 
		// int's that are 
		// group elements in int format
	void init_generators_by_hdl(int nb_gen, int *gen_hdl, 
		int verbose_level);
	int get_image(int i, int gen_idx, int verbose_level);
	void swap_points(int i, int j);
	void move_point_here(int here, int pt);
	int orbit_representative(int pt);
	int depth_in_tree(int j);
		// j is a coset, not a point
	void transporter_from_orbit_rep_to_point(int pt, 
		int &orbit_idx, int *Elt, int verbose_level);
	void transporter_from_point_to_orbit_rep(int pt, 
		int &orbit_idx, int *Elt, int verbose_level);
	void coset_rep(int j);
		// j is a coset, not a point
		// result is in cosetrep
		// determines an element in the group 
		// that moves the orbit representative 
		// to the j-th point in the orbit.
	void coset_rep_with_verbosity(int j, int verbose_level);
	void coset_rep_inv(int j);
	void compute_point_orbit(int pt, int verbose_level);
	void compute_point_orbit_with_limited_depth(
			int pt, int max_depth, int verbose_level);
	void extend_orbit(int *elt, int verbose_level);
	void compute_all_point_orbits(int verbose_level);
	void compute_all_point_orbits_with_prefered_reps(
		int *prefered_reps, int nb_prefered_reps, 
		int verbose_level);
	void compute_all_point_orbits_with_preferred_labels(
		int *preferred_labels, int verbose_level);
	void compute_all_orbits_on_invariant_subset(int len, 
		int *subset, int verbose_level);
	int sum_up_orbit_lengths();
	void non_trivial_random_schreier_generator(action *A_original, 
		int verbose_level);
		// computes non trivial random Schreier 
		// generator into schreier_gen
		// non-trivial is with respect to A_original
	void random_schreier_generator_ith_orbit(int orbit_no, 
		int verbose_level);
	void random_schreier_generator(int verbose_level);
		// computes random Schreier generator into schreier_gen
	void trace_back(int *path, int i, int &j);
	void intersection_vector(int *set, int len, 
		int *intersection_cnt);
	void orbits_on_invariant_subset_fast(int len, 
		int *subset, int verbose_level);
	void orbits_on_invariant_subset(int len, int *subset, 
		int &nb_orbits_on_subset, int *&orbit_perm, int *&orbit_perm_inv);
	void get_orbit_partition_of_points_and_lines(
		partitionstack &S, int verbose_level);
	void get_orbit_partition(partitionstack &S, 
		int verbose_level);
	strong_generators *stabilizer_any_point_plus_cosets(
		action *default_action, 
		longinteger_object &full_group_order, 
		int pt, vector_ge *&cosets, int verbose_level);
	strong_generators *stabilizer_any_point(
		action *default_action, 
		longinteger_object &full_group_order, int pt, 
		int verbose_level);
	strong_generators *stabilizer_orbit_rep(
		action *default_action, 
		longinteger_object &full_group_order, 
		int orbit_idx, int verbose_level);
	void point_stabilizer(action *default_action, 
		longinteger_object &go, 
		sims *&Stab, int orbit_no, int verbose_level);
		// this function allocates a sims structure into Stab.
	void get_orbit(int orbit_idx, int *set, int &len, 
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
	void list_elements_as_permutations_vertically(ostream &ost);
	void create_point_list_sorted(
			int *&point_list, int &point_list_length);
	void shallow_tree_generators(int orbit_idx,
			schreier *&shallow_tree,
			int verbose_level);
	schreier_vector *get_schreier_vector(
			int gen_hdl_first, int nb_gen, int verbose_level);

	// schreier_io.cpp:
	void latex(const char *fname);
	void print_orbit_lengths(ostream &ost);
	void print_orbit_length_distribution(ostream &ost);
	void print_orbit_reps(ostream &ost);
	void print(ostream &ost);
	void print_and_list_orbits(ostream &ost);
	void print_and_list_orbits_tex(ostream &ost);
	void print_and_list_orbits_of_given_length(ostream &ost,
		int len);
	void print_and_list_orbits_and_stabilizer(ostream &ost,
		action *default_action, longinteger_object &go,
		void (*print_point)(ostream &ost, int pt, void *data),
			void *data);
	void print_and_list_orbits_using_labels(ostream &ost,
		int *labels);
	void print_tables(ostream &ost, int f_with_cosetrep);
	void print_tables_latex(ostream &ost, int f_with_cosetrep);
	void print_generators();
	void print_generators_latex(ostream &ost);
	void print_generators_with_permutations();
	void print_orbit(int orbit_no);
	void print_orbit_using_labels(int orbit_no, int *labels);
	void print_orbit(ostream &ost, int orbit_no);
	void print_orbit_tex(ostream &ost, int orbit_no);
	void print_and_list_orbit_and_stabilizer_tex(int i,
		action *default_action,
		longinteger_object &full_group_order,
		ostream &ost);
	void print_and_list_orbit_and_stabilizer_with_list_of_elements_tex(
		int i, action *default_action,
		strong_generators *gens, ostream &ost);
	void print_and_list_orbit_tex(int i, ostream &ost);
	void print_and_list_orbits_sorted_by_length_tex(ostream &ost);
	void print_and_list_orbits_and_stabilizer_sorted_by_length(
		ostream &ost, int f_tex,
		action *default_action,
		longinteger_object &full_group_order);
	void print_fancy(
		ostream &ost, int f_tex,
		action *default_action, strong_generators *gens_full_group);
	void print_and_list_orbits_sorted_by_length(ostream &ost);
	void print_and_list_orbits_sorted_by_length(ostream &ost, int f_tex);
	void print_orbit_using_labels(ostream &ost, int orbit_no, int *labels);
	void print_orbit_using_callback(ostream &ost, int orbit_no,
		void (*print_point)(ostream &ost, int pt, void *data),
		void *data);
	void print_orbit_type(int f_backwards);
	void list_all_orbits_tex(ostream &ost);
	void print_orbit_through_labels(ostream &ost,
		int orbit_no, int *point_labels);
	void print_orbit_sorted(ostream &ost, int orbit_no);
	void print_orbit(int cur, int last);
	void print_tree(int orbit_no);
	void export_tree_as_layered_graph(int orbit_no,
			const char *fname_mask,
			int verbose_level);
	void draw_forest(const char *fname_mask,
		int xmax, int ymax,
		int f_circletext, int rad,
		int f_embedded, int f_sideways,
		double scale, double line_width,
		int f_has_point_labels, int *point_labels,
		int verbose_level);
	void draw_tree(const char *fname, int orbit_no,
		int xmax, int ymax, int f_circletext, int rad,
		int f_embedded, int f_sideways,
		double scale, double line_width,
		int f_has_point_labels, int *point_labels,
		int verbose_level);
	void draw_tree2(const char *fname, int xmax, int ymax,
		int f_circletext,
		int *weight, int *placement_x, int max_depth,
		int i, int last, int rad,
		int f_embedded, int f_sideways,
		double scale, double line_width,
		int f_has_point_labels, int *point_labels,
		int verbose_level);
	void subtree_draw_lines(mp_graphics &G, int f_circletext,
		int parent_x, int parent_y, int *weight,
		int *placement_x, int max_depth, int i, int last,
		int verbose_level);
	void subtree_draw_vertices(mp_graphics &G,
		int f_circletext, int parent_x, int parent_y, int *weight,
		int *placement_x, int max_depth, int i, int last, int rad,
		int f_has_point_labels, int *point_labels,
		int verbose_level);
	void subtree_place(int *weight, int *placement_x,
		int left, int right, int i, int last);
	int subtree_calc_weight(int *weight, int &max_depth,
		int i, int last);
	int subtree_depth_first(ostream &ost, int *path, int i, int last);
	void print_path(ostream &ost, int *path, int l);
	void write_to_memory_object(memory_object *m, int verbose_level);
	void read_from_memory_object(memory_object *m, int verbose_level);
	void write_file(char *fname, int verbose_level);
	void read_file(const char *fname, int verbose_level);
	void write_to_file_binary(ofstream &fp, int verbose_level);
	void read_from_file_binary(ifstream &fp, int verbose_level);
	void write_file_binary(char *fname, int verbose_level);
	void read_file_binary(const char *fname, int verbose_level);
};

// #############################################################################
// schreier_sims.C:
// #############################################################################


//! Schreier Sims algorithm

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
	vector_ge *gens;

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
// sims.C:
// #############################################################################

//! A stabilizer chain for a permutation group

class sims {

public:
	action *A;

	int my_base_len;

	vector_ge gens;
	vector_ge gens_inv;
	
	int *gen_depth; // [nb_gen]
	int *gen_perm; // [nb_gen]
	
	int *nb_gen; // [base_len + 1]
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
	
	int *path; // [base_len]
	
	int nb_images;
	int **images;
	
	int *orbit_len;
		// [base_len]
		// orbit_len[i] is the length of the i-th basic orbit.
	
	int **orbit;
		// [base_len][A->deg]
		// orbit[i][j] is the j-th point in the orbit 
		// of the i-th base point.
		// for 0 \le j < orbit_len[i].
		// for orbit_len[i] \le j < A->deg, 
		// the points not in the orbit are listed.
	int **orbit_inv;
		// [base_len][A->deg]
		// orbit[i] is the inverse of the permutation orbit[i],
		// i.e. given a point j,
		// orbit_inv[i][j] is the coset (or position in the orbit)
		// which contains j.
	
	int **prev; // [base_len][A->deg]
	int **label; // [base_len][A->deg]
	
	int *Path; // [A->deg + 1]
	int *Label; // [A->deg]
	
	// storage for temporary data and 
	// group elements computed by various routines.
	int *Elt1, *Elt2, *Elt3, *Elt4;
	int *strip1, *strip2;
		// used in strip
	int *eltrk1, *eltrk2;
		// used in element rank unrank
	int *cosetrep, *cosetrep_tmp;
		// used in coset_rep / coset_rep_inv
	int *schreier_gen, *schreier_gen1;
		// used in random_schreier_generator
	
	sims();
	void null();
	sims(action *A);
	~sims();
	void freeself();

	void delete_images();
	void init_images(int nb_images);
	void images_append();
	void init(action *A);
		// initializes the trivial group 
		// with the base as given in A
	void init_without_base(action *A);
	void reallocate_base(int old_base_len, int verbose_level);
	void initialize_table(int i);
	void init_trivial_group(int verbose_level);
		// clears the generators array, 
		// and sets the i-th transversal to contain
		// only the i-th base point (for all i).
	void init_trivial_orbit(int i);
	void init_generators(vector_ge &generators, int verbose_level);
	void init_generators(int nb, int *elt, int verbose_level);
		// copies the given elements into the generator array, 
		// then computes depth and perm
	void init_generators_by_hdl(int nb_gen, int *gen_hdl);
	void init_generator_depth_and_perm(int verbose_level);
	void add_generator(int *elt, int verbose_level);
		// adds elt to list of generators, 
		// computes the depth of the element, 
		// updates the arrays gen_depth and gen_perm accordingly
		// does not change the transversals
	void create_group_tree(const char *fname, int f_full, 
		int verbose_level);
	void print_transversals();
	void print_transversals_short();
	void print_transversal_lengths();
	void print_orbit_len();
	void print(int verbose_level);
	void print_generators();
	void print_generators_tex(ostream &ost);
	void print_generators_as_permutations();
	void print_generators_as_permutations_override_action(
		action *A);
	void print_basic_orbits();
	void print_basic_orbit(int i);
	//void sort();
	//void sort_basic_orbit(int i);
	void print_generator_depth_and_perm();
	int generator_depth(int gen_idx);
		// returns the index of the first base point 
		// which is moved by a given generator. 
	int generator_depth(int *elt);
		// returns the index of the first base point 
		// which is moved by the given element
	void group_order(longinteger_object &go);
	int group_order_int();
	void print_group_order(ostream &ost);
	void print_group_order_factored(ostream &ost);
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
	void random_element(int *elt, int verbose_level);
		// compute a random element among the group elements 
		// represented by the chain
		// (chooses random cosets along the stabilizer chain)
	void random_element_of_order(int *elt, int order, 
		int verbose_level);
	void random_elements_of_order(vector_ge *elts, 
		int *orders, int nb, int verbose_level);
	void path_unrank_int(int a);
	int path_rank_int();
	
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
	void element_unrank_int(int rk, int *Elt, int verbose_level);
	void element_unrank_int(int rk, int *Elt);
	int element_rank_int(int *Elt);
	int is_element_of(int *elt);
	void test_element_rank_unrank();
	void coset_rep(int i, int j, int verbose_level);
		// computes a coset representative in transversal i 
		// which maps
		// the i-th base point to the point which is in 
		// coset j of the i-th basic orbit.
		// j is a coset, not a point
		// result is in cosetrep
	int compute_coset_rep_depth(int i, int j, int verbose_level);
	void compute_coset_rep_path(int i, int j, int &depth,
		int verbose_level);
	void coset_rep_inv(int i, int j, int verbose_level_le);
		// computes the inverse element of what coset_rep computes,
		// i.e. an element which maps the 
		// j-th point in the orbit to the 
		// i-th base point.
		// j is a coset, not a point
		// result is in cosetrep
	void compute_base_orbits(int verbose_level);
	void print_generators_at_level_or_below(int lvl);
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
	void extract_strong_generators_in_order(vector_ge &SG, 
		int *tl, int verbose_level);
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
	void random_schreier_generator(int verbose_level);
		// computes random Schreier generator into schreier_gen
	void build_up_group_random_process_no_kernel(sims *old_G, 
		int verbose_level);
	void extend_group_random_process_no_kernel(sims *extending_by_G, 
		longinteger_object &target_go, 
		int verbose_level);
	void conjugate(action *A, sims *old_G, int *Elt, 
		int f_overshooting_OK, int verbose_level);
		// Elt * g * Elt^{-1} where g is in old_G
	int test_if_in_set_stabilizer(action *A, 
		int *set, int size, int verbose_level);
	int test_if_subgroup(sims *old_G, int verbose_level);
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
	void write_all_group_elements(char *fname, int verbose_level);
	void print_all_group_elements_to_file(char *fname, 
		int verbose_level);
	void print_all_group_elements();
	void print_all_group_elements_as_permutations();
	void print_all_group_elements_as_permutations_in_special_action(
		action *A_special);
	void print_all_transversal_elements();
	void element_as_permutation(action *A_special, 
		int elt_rk, int *perm, int verbose_level);
	void table_of_group_elements_in_data_form(int *&Table, 
		int &len, int &sz, int verbose_level);
	void regular_representation(int *Elt, int *perm, 
		int verbose_level);
	void center(vector_ge &gens, int *center_element_ranks, 
		int &nb_elements, int verbose_level);
	void all_cosets(int *subset, int size, 
		int *all_cosets, int verbose_level);
	void element_ranks_subgroup(sims *subgroup, 
		int *element_ranks, int verbose_level);
	void find_standard_generators_int(int ord_a, int ord_b, 
		int ord_ab, int &a, int &b, int &nb_trials, 
		int verbose_level);
	int find_element_of_given_order_int(int ord, 
		int &nb_trials, int verbose_level);
	int find_element_of_given_order_int(int *Elt, 
		int ord, int &nb_trials, int max_trials, 
		int verbose_level);
	void find_element_of_prime_power_order(int p, 
		int *Elt, int &e, int &nb_trials, int verbose_level);
	void save_list_of_elements(char *fname, 
		int verbose_level);
	void read_list_of_elements(action *A, 
		char *fname, int verbose_level);
	void evaluate_word_int(int word_len, 
		int *word, int *Elt, int verbose_level);
	void write_sgs(const char *fname, int verbose_level);
	void read_sgs(const char *fname, vector_ge *SG, 
		int verbose_level);
	int least_moved_point_at_level(int lvl, int verbose_level);
	int mult_by_rank(int rk_a, int rk_b, int verbose_level);
	int mult_by_rank(int rk_a, int rk_b);
	int invert_by_rank(int rk_a, int verbose_level);
	int conjugate_by_rank(int rk_a, int rk_b, int verbose_level);
		// comutes b^{-1} * a * b
	int conjugate_by_rank_b_bv_given(int rk_a, 
		int *Elt_b, int *Elt_bv, int verbose_level);
	void sylow_subgroup(int p, sims *P, int verbose_level);
	int is_normalizing(int *Elt, int verbose_level);
	void create_Cayley_graph(vector_ge *gens, int *&Adj, int &n, 
		int verbose_level);
	void create_group_table(int *&Table, int &n, int verbose_level);
	void write_as_magma_permutation_group(const char *fname_base, 
		vector_ge *gens, int verbose_level);

	// sims2.C:
	void build_up_subgroup_random_process(sims *G, 
		void (*choose_random_generator_for_subgroup)(
			sims *G, int *Elt, int verbose_level), 
		int verbose_level);

	// sims3.C:
	void subgroup_make_characteristic_vector(sims *Sub, 
		int *C, int verbose_level);
	void normalizer_based_on_characteristic_vector(int *C_sub, 
		int *Gen_idx, int nb_gens, int *N, int &N_go, 
		int verbose_level);
	void order_structure_relative_to_subgroup(int *C_sub, 
		int *Order, int *Residue, int verbose_level);
};

// sims2.C:
void choose_random_generator_derived_group(sims *G, int *Elt, 
	int verbose_level);


// in sims_global.C:
sims *create_sims_from_generators_with_target_group_order_factorized(
		action *A, 
		vector_ge *gens, int *tl, int len, int verbose_level);
sims *create_sims_from_generators_with_target_group_order(action *A, 
		vector_ge *gens, longinteger_object &target_go, 
		int verbose_level);
sims *create_sims_from_generators_with_target_group_order_int(action *A, 
	vector_ge *gens, int target_go, int verbose_level);
sims *create_sims_from_generators_without_target_group_order(action *A, 
	vector_ge *gens, int verbose_level);
sims *create_sims_from_single_generator_without_target_group_order(action *A, 
	int *Elt, int verbose_level);
sims *create_sims_from_generators_randomized(action *A, 
	vector_ge *gens, int f_target_go, 
	longinteger_object &target_go, int verbose_level);
sims *create_sims_for_centralizer_of_matrix(action *A, 
	int *Mtx, int verbose_level);








// #############################################################################
// strong_generators.C:
// #############################################################################

//! a strong generating set for a permutation group with group order

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
	void init_by_hdl(action *A, int *gen_hdl, 
		int nb_gen, int verbose_level);
	void init_from_permutation_representation(action *A, 
		int *data, 
		int nb_elements, int group_order, 
		int verbose_level);
	void init_from_data(action *A, int *data, 
		int nb_elements, int elt_size, 
		int *transversal_length, 
		int verbose_level);
	void init_from_data_with_target_go_ascii(action *A, 
		int *data, 
		int nb_elements, int elt_size, 
		const char *ascii_target_go,
		int verbose_level);
	void init_from_data_with_target_go(action *A, 
		int *data_gens, 
		int data_gens_size, int nb_gens, 
		longinteger_object &target_go, 
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
	sims *create_sims(int verbose_level);
	sims *create_sims_in_different_action(action *A_given, 
		int verbose_level);
	void add_generators(vector_ge *coset_reps, 
		int group_index, int verbose_level);
	void add_single_generator(int *Elt, 
		int group_index, int verbose_level);
	void group_order(longinteger_object &go);
	int group_order_as_int();
	void print_generators();
	void print_generators_ost(ostream &ost);
	void print_generators_in_source_code();
	void print_generators_in_source_code_to_file(
	const char *fname);
	void print_generators_even_odd();
	void print_generators_MAGMA(action *A, ostream &ost);
	void print_generators_tex();
	void print_generators_tex(ostream &ost);
	void print_generators_as_permutations();
	void print_with_given_action(ostream &ost, action *A2);
	void print_elements_ost(ostream &ost);
	void print_elements_latex_ost(ostream &ost);
	void create_group_table(int *&Table, int &go, 
		int verbose_level);
	void list_of_elements_of_subgroup(
		strong_generators *gens_subgroup, 
		int *&Subgroup_elements_by_index, 
		int &sz_subgroup, int verbose_level);
	void compute_schreier_with_given_action(action *A_given, 
		schreier *&Sch, int verbose_level);
	void compute_schreier_with_given_action_on_a_given_set(
		action *A_given, 
		schreier *&Sch, int *set, int len, int verbose_level);
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
	void write_to_memory_object(memory_object *m, int verbose_level);
	void read_from_memory_object(memory_object *m, int verbose_level);
	void write_to_file_binary(ofstream &fp, int verbose_level);
	void read_from_file_binary(action *A, ifstream &fp, 
		int verbose_level);
	void write_file(const char *fname, int verbose_level);
	void read_file(action *A, const char *fname, int verbose_level);
	void generators_for_shallow_schreier_tree(char *label, 
		vector_ge *chosen_gens, int verbose_level);
	void compute_ascii_coding(char *&ascii_coding, int verbose_level);
	void decode_ascii_coding(char *ascii_coding, int verbose_level);
	void export_permutation_group_to_magma(const char *fname, 
		int verbose_level);
	void export_permutation_group_to_GAP(const char *fname,
		int verbose_level);
	void compute_and_print_orbits_on_a_given_set(action *A_given,
		int *set, int len, int verbose_level);
	void compute_and_print_orbits(action *A_given, 
		int verbose_level);
	int test_if_normalizing(sims *S, int verbose_level);
	void test_if_set_is_invariant_under_given_action(action *A_given, 
		int *set, int set_sz, int verbose_level);
	strong_generators *point_stabilizer(int pt, int verbose_level);
	void make_element_which_moves_a_point_from_A_to_B(action *A_given, 
		int pt_A, int pt_B, int *Elt, int verbose_level);

	// strong_generators_groups.C:
	void init_linear_group_from_scratch(action *&A, 
		finite_field *F, int n, 
		int f_projective, int f_general, int f_affine, 
		int f_semilinear, int f_special, 
		int verbose_level);
	void special_subgroup(int verbose_level);
	void even_subgroup(int verbose_level);
	void init_single(action *A, int *Elt, int verbose_level);
	void init_trivial_group(action *A, int verbose_level);
	void generators_for_the_monomial_group(action *A, 
		matrix_group *Mtx, int verbose_level);
	void generators_for_the_diagonal_group(action *A, 
		matrix_group *Mtx, int verbose_level);
	void generators_for_the_singer_cycle(action *A, 
		matrix_group *Mtx, int power_of_singer, int verbose_level);
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
	void generators_for_the_stabilizer_of_the_cubic_surface(
		action *A, 
		finite_field *F, int iso, 
		int verbose_level);
	void generators_for_the_stabilizer_of_the_cubic_surface_family_24(
		action *A, 
		finite_field *F, int f_with_normalizer, int f_semilinear, 
		int verbose_level);
	void BLT_set_from_catalogue_stabilizer(action *A, 
		finite_field *F, int iso, 
		int verbose_level);
	void stabilizer_of_spread_from_catalogue(action *A, 
		int q, int k, int iso, 
		int verbose_level);
	void Hall_reflection(
		int nb_pairs, int &degree, int verbose_level);
	void normalizer_of_a_Hall_reflection(
		int nb_pairs, int &degree, int verbose_level);


};

void strong_generators_array_write_file(const char *fname, 
	strong_generators *p, int nb, int verbose_level);
void strong_generators_array_read_from_file(const char *fname, 
	action *A, strong_generators *&p, int &nb, int verbose_level);

// #############################################################################
// subgroup.C:
// #############################################################################

//! list a subgroup of a group by storing the element indices

class subgroup {
public:
	int *Elements;
	int group_order;
	int *gens;
	int nb_gens;


	subgroup();
	~subgroup();
	void null();
	void freeself();
	void init(int *Elements, int group_order, int *gens, int nb_gens);
	void print();
	int contains_this_element(int elt);

};


// #############################################################################
// wreath_product.C:
// #############################################################################

//! the wreath product group  AGL(d,q) wreath Sym(n)

class wreath_product {

public:
	matrix_group *M;
	action *A_mtx;
	finite_field *F;
	int q;
	int nb_factors;

	char label[1000];
	char label_tex[1000];

	int degree_of_matrix_group;
		// = M->degree;
	int dimension_of_matrix_group;
		// = M->n
	int dimension_of_tensor_action;
		// = i_power_j(dimension_of_matrix_group, nb_factors);
	int degree_of_tensor_action;
		// = (i_power_j_safe(q, dimension_of_tensor_action) - 1) / (q - 1);
	int degree_overall;
		// = nb_factors + nb_factors *
		// degree_of_matrix_group + degree_of_tensor_action;
	int low_level_point_size;
		// = dimension_of_tensor_action
	int make_element_size;
		// = nb_factors + nb_factors *
		// dimension_of_matrix_group * dimension_of_matrix_group;
	perm_group *P;
	int elt_size_int;

	int *perm_offset_i;
	int *mtx_size;
	int *index_set1;
	int *index_set2;
	int *u;
	int *v;
	int *w;
	int *A1;
	int *A2;
	int *A3;
	int *tmp_Elt1;
	int *tmp_perm1;
	int *tmp_perm2;
	int *induced_perm; // [dimension_of_tensor_action]

	int bits_per_digit;
	int bits_per_elt;
	int char_per_elt;

	uchar *elt1;

	int base_len_in_component;
	int *base_for_component;
	int *tl_for_component;

	int base_length;
	int *the_base;
	int *the_transversal_length;

	page_storage *Elts;

	wreath_product();
	~wreath_product();
	void null();
	void freeself();
	void init_tensor_wreath_product(matrix_group *M,
			action *A_mtx, int nb_factors, int verbose_level);
	int element_image_of(int *Elt, int a, int verbose_level);
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
	void element_print_easy(int *Elt, ostream &ost);
	void compute_base_and_transversals(int verbose_level);
	void make_strong_generators_data(int *&data,
			int &size, int &nb_gens, int verbose_level);
};

