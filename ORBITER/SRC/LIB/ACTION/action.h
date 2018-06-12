// action.h
//
// Anton Betten
//
// started:  August 13, 2005


typedef class action action;
typedef class matrix_group matrix_group;
typedef class perm_group perm_group;
typedef class vector_ge vector_ge;
typedef class vector_ge *p_vector_ge;
typedef class schreier schreier;
typedef class sims sims;
typedef class sims *p_sims;
typedef class group group;
typedef class page_storage page_storage;
typedef class action_on_sets action_on_sets;
typedef class action_on_k_subsets action_on_k_subsets;
typedef class action_by_right_multiplication 
	action_by_right_multiplication;
typedef class action_by_restriction action_by_restriction;
typedef class action_by_conjugation action_by_conjugation;
typedef class action_on_orbits action_on_orbits;
typedef class action_on_flags action_on_flags;
typedef class action_by_representation action_by_representation;
typedef class action_by_subfield_structure action_by_subfield_structure;
typedef class action_on_grassmannian action_on_grassmannian;
typedef class action_on_spread_set action_on_spread_set;
typedef class action_on_orthogonal action_on_orthogonal;
typedef class action_on_wedge_product action_on_wedge_product;
typedef class action_on_cosets action_on_cosets;
typedef class action_on_factor_space action_on_factor_space;
typedef class action_on_determinant action_on_determinant;
typedef class action_on_sign action_on_sign;
typedef class action_on_homogeneous_polynomials 
	action_on_homogeneous_polynomials;
typedef class product_action product_action;
typedef class union_find union_find;
typedef class union_find_on_k_subsets union_find_on_k_subsets;
typedef class schreier_sims schreier_sims;
typedef sims *psims;
typedef class action_on_bricks action_on_bricks;
typedef class action_on_andre action_on_andre;
typedef class strong_generators strong_generators;
typedef strong_generators *pstrong_generators;
typedef class linear_group_description linear_group_description;
typedef class linear_group linear_group;
typedef class set_and_stabilizer set_and_stabilizer;
typedef class subgroup subgroup;
typedef class subgroup *psubgroup;
typedef class action_on_subgroups action_on_subgroups;
typedef class orbit_transversal orbit_transversal;


// #############################################################################
// action.C:
// #############################################################################

enum symmetry_group_type { 
	unknown_symmetry_group_t, 
	matrix_group_t, 
	perm_group_t, 
	action_on_sets_t,
	action_on_subgroups_t,
	action_on_k_subsets_t,
	action_on_pairs_t,
	action_on_ordered_pairs_t,
	base_change_t,
	product_action_t,
	action_by_right_multiplication_t,
	action_by_restriction_t,
	action_by_conjugation_t,
	action_on_determinant_t, 
	action_on_sign_t, 
	action_on_grassmannian_t, 
	action_on_spread_set_t, 
	action_on_orthogonal_t, 
	action_on_cosets_t, 
	action_on_factor_space_t, 
	action_on_wedge_product_t, 
	action_by_representation_t,
	action_by_subfield_structure_t,
	action_on_bricks_t,
	action_on_andre_t,
	action_on_orbits_t,
	action_on_flags_t,
	action_on_homogeneous_polynomials_t
};

enum representation_type {
	representation_type_nothing, 
	representation_type_PSL2_on_conic
}; 

union symmetry_group {
	matrix_group *matrix_grp;
	perm_group *perm_grp;
	action_on_sets *on_sets;
	action_on_subgroups *on_subgroups;
	action_on_k_subsets *on_k_subsets;
	product_action *product_action_data;
	action_by_right_multiplication *ABRM;
	action_by_restriction *ABR;
	action_by_conjugation *ABC;
	action_on_determinant *AD;
	action_on_sign *OnSign;
	action_on_grassmannian *AG;
	action_on_spread_set *AS;
	action_on_orthogonal *AO;
	action_on_cosets *OnCosets;
	action_on_factor_space *AF;
	action_on_wedge_product *AW;
	action_by_representation *Rep;
	action_by_subfield_structure *SubfieldStructure;
	action_on_bricks *OnBricks;
	action_on_andre *OnAndre;
	action_on_orbits *OnOrbits;
	action_on_flags *OnFlags;
	action_on_homogeneous_polynomials *OnHP;
};

class action {
public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

	
	
	// the symmetry group is a permutation group
	INT f_allocated;
	symmetry_group_type type_G;
	symmetry_group G;
	
	INT f_has_subaction;
	INT f_subaction_is_allocated;
	action *subaction;
	
	INT f_has_strong_generators;
	strong_generators *Strong_gens;

	INT degree; // the size of the set we act on

	INT f_is_linear; // is it a linear action
		// matrix_group_t, 
		// action_on_wedge_product_t, 
		// action_by_representation_t
	INT dimension; // if f_is_linear
	
	INT f_has_base; // set to TRUE in allocate_base_data()
	INT base_len;
		// the length of the base 
		// (b_0,\ldots,b_{l-1})
	INT *base;
		// the base (b_0,\ldots,b_{l-1})
	INT *transversal_length;
		// the length of the orbit 
		// of $G^{(i)}$ on $b_i$
	INT **orbit;
	INT **orbit_inv;
	INT elt_size_in_INT;
		// how many INT's do we need 
		// to store one group element
	INT coded_elt_size_in_char;
		// how many BYTE's (=char's) do we need 
		// to store a group element packed
	
	INT make_element_size;
		// the number of INT's that are needed to
		// make an element of this group 
		// using the make_element function
	INT low_level_point_size;
		// the number of INT's that are needed to 
		// represent a point in low-level format
		// (input and output in element_image_of_low_level 
		// point to that many INT's)
	
	INT f_has_transversal_reps;
	INT **transversal_reps;
		// [base_len][transversal_length * elt_size_in_INT]
	
	INT f_has_sims;
	sims *Sims;
	
	// this is new 1/1/2009:
	INT f_has_kernel;
	sims *Kernel;
	
	INT f_group_order_is_small;
	INT *path;
	
	// function pointers for group actions
	INT (*ptr_element_image_of)(action &A, INT a, void *elt, 
		INT verbose_level);
	void (*ptr_element_image_of_low_level)(action &A, 
		INT *input, INT *output, void *elt, INT verbose_level);
	INT (*ptr_element_linear_entry_ij)(action &A, 
		void *elt, INT i, INT j, INT verbose_level);
	INT (*ptr_element_linear_entry_frobenius)(action &A, 
		void *elt, INT verbose_level);
	void (*ptr_element_one)(action &A, void *elt, INT verbose_level);
	INT (*ptr_element_is_one)(action &A, void *elt, INT verbose_level);
	void (*ptr_element_unpack)(action &A, void *elt, 
		void *Elt, INT verbose_level);
	void (*ptr_element_pack)(action &A, void *Elt, 
		void *elt, INT verbose_level);
	void (*ptr_element_retrieve)(action &A, INT hdl, 
		void *elt, INT verbose_level);
	INT (*ptr_element_store)(action &A, void *elt, 
		INT verbose_level);
	void (*ptr_element_mult)(action &A, 
		void *a, void *b, void *ab, INT verbose_level);
	void (*ptr_element_invert)(action &A, 
		void *a, void *av, INT verbose_level);
	void (*ptr_element_transpose)(action &A, 
		void *a, void *at, INT verbose_level);
	void (*ptr_element_move)(action &A, 
		void *a, void *b, INT verbose_level);
	void (*ptr_element_dispose)(action &A, 
		INT hdl, INT verbose_level);
	void (*ptr_element_print)(action &A, 
		void *elt, ostream &ost);
	void (*ptr_element_print_quick)(action &A, 
		void *elt, ostream &ost);
	void (*ptr_element_print_latex)(action &A, 
		void *elt, ostream &ost);
	void (*ptr_element_print_verbose)(action &A, 
		void *elt, ostream &ost);
	void (*ptr_print_point)(action &A, INT i, ostream &ost);
	void (*ptr_element_code_for_make_element)(action &A, 
		void *elt, INT *data);
	void (*ptr_element_print_for_make_element)(action &A, 
		void *elt, ostream &ost);
	void (*ptr_element_print_for_make_element_no_commas)(action &A, 
		void *elt, ostream &ost);
	
	INT nb_times_image_of_called;
	INT nb_times_image_of_low_level_called;
	INT nb_times_unpack_called;
	INT nb_times_pack_called;
	INT nb_times_retrieve_called;
	INT nb_times_store_called;
	INT nb_times_mult_called;
	INT nb_times_invert_called;



	INT *Elt1, *Elt2, *Elt3, *Elt4, *Elt5;
	INT *eltrk1, *eltrk2, *eltrk3, *elt_mult_apply;
	UBYTE *elt1;
	BYTE *element_rw_memory_object;
		// [coded_elt_size_in_char]
		// for element_write_to_memory_object, 
		// element_read_from_memory_object

	
	BYTE group_prefix[1000];
	// new 1/1/2009:
	BYTE label[1000];
	BYTE label_tex[1000];



	// action.C:
	action();
	~action();
	
	void null_element_data();
	void allocate_element_data();
	void free_element_data();
	void null_base_data();
	void allocate_base_data(INT base_len);
	void reallocate_base(INT new_base_point);
	void free_base_data();
	
	INT find_non_fixed_point(void *elt, INT verbose_level);
	INT find_fixed_points(void *elt, 
		INT *fixed_points, INT verbose_level);
	INT test_if_set_stabilizes(INT *Elt, 
		INT size, INT *set, INT verbose_level);
	void map_a_set(INT *set, INT *image_set, 
		INT n, INT *Elt, INT verbose_level);
	void map_a_set_and_reorder(INT *set, INT *image_set, 
		INT n, INT *Elt, INT verbose_level);
	void print_all_elements();

	void init_sims(sims *G, INT verbose_level);
	void init_base_from_sims(sims *G, INT verbose_level);
	INT element_has_order_two(INT *E1, INT verbose_level);
	INT product_has_order_two(INT *E1, INT *E2, INT verbose_level);
	INT product_has_order_three(INT *E1, INT *E2, INT verbose_level);
	INT element_order(void *elt);
	INT element_order_verbose(void *elt, INT verbose_level);
	INT element_order_if_divisor_of(void *elt, INT o);
	void compute_all_point_orbits(schreier &S, 
		vector_ge &gens, INT verbose_level);
	
	INT depth_in_stab_chain(INT *Elt);
		// the index of the first moved base point
	void strong_generators_at_depth(INT depth, vector_ge &gen);
		// all strong generators that 
		// leave base points 0,..., depth - 1 fix
	void compute_point_stabilizer_chain(vector_ge &gen, 
		sims *S, INT *sequence, INT len, 
		INT verbose_level);
	INT compute_orbit_of_point(vector_ge &strong_generators, 
		INT pt, INT *orbit, 
		INT verbose_level);
	INT compute_orbit_of_point_generators_by_handle(
		INT nb_gen, INT *gen_handle, 
		INT pt, INT *orbit, 
		INT verbose_level);
	INT least_image_of_point(vector_ge &strong_generators, 
		INT pt, INT *transporter, 
		INT verbose_level);
	INT least_image_of_point_generators_by_handle(INT nb_gen, 
		INT *gen_handle, INT pt, INT *transporter, 
		INT verbose_level);
	void all_point_orbits(schreier &Schreier, INT verbose_level);
	void compute_stabilizer_orbits(partitionstack *&Staborbits, 
		INT verbose_level);
	INT check_if_in_set_stabilizer(INT *Elt, 
		INT size, INT *set, 
		INT verbose_level);
	INT check_if_transporter_for_set(INT *Elt, 
		INT size, INT *set1, INT *set2, 
		INT verbose_level);
	void compute_set_orbit(vector_ge &gens, 
		INT size, INT *set, 
		INT &nb_sets, INT **&Sets, 
		INT **&Transporter, 
		INT verbose_level);
	void delete_set_orbit(INT nb_sets, 
		INT **Sets, INT **Transporter);
	void compute_minimal_set(vector_ge &gens, 
		INT size, INT *set, 
		INT *minimal_set, INT *transporter, 
		INT verbose_level);
	void find_strong_generators_at_level(INT base_len, 
		INT *the_base, INT level, 
		vector_ge &gens, vector_ge &subset_of_gens, 
		INT verbose_level);
	void compute_strong_generators_from_sims(INT verbose_level);
	void make_element_from_permutation_representation(INT *Elt, 
		INT *data, INT verbose_level);
	void make_element_from_base_image(INT *Elt, INT *data, 
		INT verbose_level);
	void make_element_2x2(INT *Elt, INT a0, INT a1, INT a2, INT a3);
	void make_element(INT *Elt, INT *data, INT verbose_level);
	void build_up_automorphism_group_from_aut_data(INT nb_auts, 
		INT *aut_data, 
		sims &S, INT verbose_level);
	void element_power_INT_in_place(INT *Elt, 
		INT n, INT verbose_level);
	void word_in_ab(INT *Elt1, INT *Elt2, INT *Elt3, 
		const BYTE *word, INT verbose_level);
	void init_group_from_generators(INT *group_generator_data, 
		INT group_generator_size, 
		INT f_group_order_target, const BYTE *group_order_target, 
		vector_ge *gens, strong_generators *&Strong_gens, 
		INT verbose_level);
	void init_group_from_generators_by_base_images(
		INT *group_generator_data, INT group_generator_size, 
		INT f_group_order_target, const BYTE *group_order_target, 
		vector_ge *gens, strong_generators *&Strong_gens_out, 
		INT verbose_level);
	void print_symmetry_group_type(ostream &ost);
	void print_info();
	void print_base();
	void group_order(longinteger_object &go);
	void print_group_order(ostream &ost);
	void print_group_order_long(ostream &ost);
	void print_vector(vector_ge &v);
	void print_vector_as_permutation(vector_ge &v);
	void element_print_base_images(INT *Elt);
	void element_print_base_images(INT *Elt, ostream &ost);
	void element_print_base_images_verbose(INT *Elt, 
		ostream &ost, INT verbose_level);
	void element_base_images(INT *Elt, INT *base_images);
	void element_base_images_verbose(INT *Elt, 
		INT *base_images, INT verbose_level);
	void minimize_base_images(INT level, sims *S, 
		INT *Elt, INT verbose_level);
	void element_conjugate_bvab(INT *Elt_A, 
		INT *Elt_B, INT *Elt_C, INT verbose_level);
	void element_conjugate_babv(INT *Elt_A, 
		INT *Elt_B, INT *Elt_C, INT verbose_level);
	void element_commutator_abavbv(INT *Elt_A, 
		INT *Elt_B, INT *Elt_C, INT verbose_level);
	void read_representatives(BYTE *fname, 
		INT *&Reps, INT &nb_reps, INT &size, INT verbose_level);
	void read_representatives_and_strong_generators(BYTE *fname, 
		INT *&Reps, 
		BYTE **&Aut_ascii, INT &nb_reps, 
		INT &size, INT verbose_level);
	void read_file_and_print_representatives(BYTE *fname, 
		INT f_print_stabilizer_generators);
	void read_set_and_stabilizer(const BYTE *fname, 
		INT no, INT *&set, INT &set_sz, sims *&stab, 
		strong_generators *&Strong_gens, 
		INT &nb_cases, 
		INT verbose_level);
	void get_generators_from_ascii_coding(BYTE *ascii_coding, 
		vector_ge *&gens, INT *&tl, INT verbose_level);
	void lexorder_test(INT *set, INT set_sz, INT &set_sz_after_test, 
		vector_ge *gens, INT max_starter, 
		INT verbose_level);
	void compute_orbits_on_points(schreier *&Sch, 
		vector_ge *gens, INT verbose_level);
	void stabilizer_of_dual_hyperoval_representative(INT k, 
		INT n, INT no, vector_ge *&gens, 
		const BYTE *&stab_order, INT verbose_level);
	void stabilizer_of_translation_plane_representative(INT q, 
		INT k, INT no, vector_ge *&gens, const BYTE *&stab_order, 
		INT verbose_level);
	void normalizer_using_MAGMA(const BYTE *prefix, 
		sims *G, sims *H, INT verbose_level);
	void conjugacy_classes_using_MAGMA(const BYTE *prefix, 
		sims *G, INT verbose_level);
	void centralizer_using_MAGMA(const BYTE *prefix, 
		sims *G, INT *Elt, INT verbose_level);
	void point_stabilizer_any_point(INT &pt, 
		schreier *&Sch, sims *&Stab, 
		strong_generators *&stab_gens, 
		INT verbose_level);
	void point_stabilizer_any_point_with_given_group(
		strong_generators *input_gens, 
		INT &pt, 
		schreier *&Sch, sims *&Stab, 
		strong_generators *&stab_gens, 
		INT verbose_level);
	void make_element_which_moves_a_line_in_PG3q(grassmann *Gr, 
		INT line_rk, INT *Elt, INT verbose_level);



	// action_indexing_cosets.C:
	void coset_unrank(sims *G, sims *U, INT rank, 
		INT *Elt, INT verbose_level);
	INT coset_rank(sims *G, sims *U, 
		INT *Elt, INT verbose_level);
		// used in generator::coset_unrank and generator::coset_rank
		// which in turn are used by 
		// generator::orbit_element_unrank and 
		// generator::orbit_element_rank

	// action_init.C:
	void init_BLT(finite_field *F, INT f_basis, 
		INT f_init_hash_table, INT verbose_level);
	void init_group_from_strong_generators(vector_ge *gens, sims *K, 
		INT given_base_length, INT *given_base,
		INT verbose_level);
	void init_orthogonal_group(INT epsilon, 
		INT n, finite_field *F, 
		INT f_on_points, INT f_on_lines, 
		INT f_on_points_and_lines, 
		INT f_semilinear, 
		INT f_basis, INT verbose_level);
	void init_projective_special_group(INT n, finite_field *F, 
		INT f_semilinear, INT f_basis, INT verbose_level);
	void init_projective_group(INT n, finite_field *F, 
		INT f_semilinear, INT f_basis, INT verbose_level);
	void init_affine_group(INT n, finite_field *F, 
		INT f_semilinear, 
		INT f_basis, INT verbose_level);
	void init_general_linear_group(INT n, finite_field *F, 
		INT f_semilinear, INT f_basis, INT verbose_level);
	void setup_linear_group_from_strong_generators(matrix_group *M, 
		INT verbose_level);
	void init_matrix_group_strong_generators_builtin(matrix_group *M, 
		INT verbose_level);
	void init_permutation_group(INT degree, INT verbose_level);
	void init_permutation_group_from_generators(INT degree, 
		INT f_target_go, longinteger_object &target_go, 
		INT nb_gens, INT *gens, 
		INT given_base_length, INT *given_base,
		INT verbose_level);
	void init_affine_group(INT n, INT q, INT f_translations, 
		INT f_semilinear, INT frobenius_power, 
		INT f_multiplication, 
		INT multiplication_order, INT verbose_level);
	void init_affine_grid_group(INT q1, INT q2, 
		INT f_translations1, INT f_translations2, 
		INT f_semilinear1, INT frobenius_power1, 
		INT f_semilinear2, INT frobenius_power2, 
		INT f_multiplication1, INT multiplication_order1, 
		INT f_multiplication2, INT multiplication_order2, 
		INT f_diagonal, 
		INT verbose_level);
	void init_symmetric_group(INT degree, INT verbose_level);
	void null_function_pointers();
	void init_function_pointers_matrix_group();
	void init_function_pointers_permutation_group();
	void init_function_pointers_induced_action();
	void create_sims(INT verbose_level);
	void create_orthogonal_group(action *subaction, 
		INT f_has_target_group_order, 
		longinteger_object &target_go, 
		void (* callback_choose_random_generator)(INT iteration, 
			INT *Elt, void *data, INT verbose_level), 
		INT verbose_level);
	
	// action_induce.C:
	void init_action_on_lines(action *A, finite_field *F, 
		INT n, INT verbose_level);
	void induced_action_by_representation_on_conic(action *A_old, 
		INT f_induce_action, sims *old_G, 
		INT verbose_level);
	void induced_action_on_cosets(action_on_cosets *A_on_cosets, 
		INT f_induce_action, sims *old_G, 
		INT verbose_level);
	void induced_action_on_factor_space(action *A_old, 
		action_on_factor_space *AF, 
		INT f_induce_action, sims *old_G, 
		INT verbose_level);
	action *induced_action_on_grassmannian(INT k, 
		INT verbose_level);
	void induced_action_on_grassmannian(action *A_old, 
		action_on_grassmannian *AG, 
		INT f_induce_action, sims *old_G, 
		INT verbose_level);
	void induced_action_on_spread_set(action *A_old, 
		action_on_spread_set *AS, 
		INT f_induce_action, sims *old_G, 
		INT verbose_level);
	void induced_action_on_orthogonal(action *A_old, 
		action_on_orthogonal *AO, 
		INT f_induce_action, sims *old_G, 
		INT verbose_level);
	void induced_action_on_wedge_product(action *A_old, 
		action_on_wedge_product *AW, 
		INT f_induce_action, sims *old_G, 
		INT verbose_level);
	void induced_action_by_subfield_structure(action *A_old, 
		action_by_subfield_structure *SubfieldStructure, 
		INT f_induce_action, sims *old_G, 
		INT verbose_level);
	void induced_action_on_determinant(sims *old_G, 
		INT verbose_level);
	void induced_action_on_sign(sims *old_G, 
		INT verbose_level);
	void induced_action_by_conjugation(sims *old_G, 
		sims *Base_group, INT f_ownership, 
		INT f_basis, INT verbose_level);
	void induced_action_by_right_multiplication(
		INT f_basis, sims *old_G, 
		sims *Base_group, INT f_ownership, 
		INT verbose_level);
	action *create_induced_action_on_sets(INT nb_sets, 
		INT set_size, INT *sets, 
		INT verbose_level);
	void induced_action_on_sets(action &old_action, sims *old_G, 
		INT nb_sets, INT set_size, INT *sets, 
		INT f_induce_action, INT verbose_level);
	action *create_induced_action_on_subgroups(sims *S, 
		INT nb_subgroups, INT group_order, 
		subgroup **Subgroups, INT verbose_level);
	void induced_action_on_subgroups(action *old_action, 
		sims *S, 
		INT nb_subgroups, INT group_order, 
		subgroup **Subgroups, 
		INT verbose_level);
	void induced_action_by_restriction_on_orbit_with_schreier_vector(
		action &old_action, 
		INT f_induce_action, sims *old_G, 
		INT *sv, INT pt, INT verbose_level);
	action *restricted_action(INT *points, INT nb_points, 
		INT verbose_level);
	void induced_action_by_restriction(action &old_action, 
		INT f_induce_action, sims *old_G, 
		INT nb_points, INT *points, INT verbose_level);
		// uses action_by_restriction data type
	void induced_action_on_pairs(action &old_action, sims *old_G, 
		INT verbose_level);
	action *create_induced_action_on_ordered_pairs(INT verbose_level);
	void induced_action_on_ordered_pairs(action &old_action, 
		sims *old_G, 
		INT verbose_level);
	void induced_action_on_k_subsets(action &old_action, INT k, 
		INT verbose_level);
	void induced_action_on_orbits(action *old_action, 
		schreier *Sch, INT f_play_it_safe, 
		INT verbose_level);
	void induced_action_on_flags(action *old_action, 
		INT *type, INT type_len, 
		INT verbose_level);
	void induced_action_on_bricks(action &old_action, 
		brick_domain *B, INT f_linear_action, 
		INT verbose_level);
	void induced_action_on_andre(action *An, 
		action *An1, andre_construction *Andre, 
		INT verbose_level);
	void setup_product_action(action *A1, action *A2, 
		INT f_use_projections, INT verbose_level);
	void induced_action_on_homogeneous_polynomials(action *A_old, 
		homogeneous_polynomial_domain *HPD, 
		INT f_induce_action, sims *old_G, 
		INT verbose_level);
	void induced_action_on_homogeneous_polynomials_given_by_equations(
		action *A_old, 
		homogeneous_polynomial_domain *HPD, 
		INT *Equations, INT nb_equations, 
		INT f_induce_action, sims *old_G, 
		INT verbose_level);
	void induced_action_recycle_sims(action &old_action, 
		INT verbose_level);
	void induced_action_override_sims(action &old_action, 
		sims *old_G, 
		INT verbose_level);
	void induce(action *old_action, sims *old_G, 
		INT base_of_choice_len, 
		INT *base_of_choice, 
		INT verbose_level);
	INT least_moved_point_at_level(INT level, 
		INT verbose_level);
	void lex_least_base_in_place(INT verbose_level);
	void lex_least_base(action *old_action, INT verbose_level);
	INT test_if_lex_least_base(INT verbose_level);
	void base_change_in_place(INT size, INT *set, INT verbose_level);
	void base_change(action *old_action, 
		INT size, INT *set, INT verbose_level);


	// action_cb.C:
	INT image_of(void *elt, INT a);
	void image_of_low_level(void *elt, INT *input, INT *output);
	INT linear_entry_ij(void *elt, INT i, INT j);
	INT linear_entry_frobenius(void *elt);
	void one(void *elt);
	INT is_one(void *elt);
	void unpack(void *elt, void *Elt);
	void pack(void *Elt, void *elt);
	void retrieve(void *elt, INT hdl);
	INT store(void *elt);
	void mult(void *a, void *b, void *ab);
	void mult_apply_from_the_right(void *a, void *b);
		// a := a * b
	void mult_apply_from_the_left(void *a, void *b);
		// b := a * b
	void invert(void *a, void *av);
	void invert_in_place(void *a);
	void transpose(void *a, void *at);
	void move(void *a, void *b);
	void dispose(INT hdl);
	void print(ostream &ost, void *elt);
	void print_quick(ostream &ost, void *elt);
	void print_as_permutation(ostream &ost, void *elt);
	void print_point(INT a, ostream &ost);
	void code_for_make_element(INT *data, void *elt);
	void print_for_make_element(ostream &ost, void *elt);
	void print_for_make_element_no_commas(ostream &ost, void *elt);
	
	INT element_image_of(INT a, void *elt, INT verbose_level);
	void element_image_of_low_level(INT *input, INT *output, 
		void *elt, INT verbose_level);
	INT element_linear_entry_ij(void *elt, INT i, INT j, 
		INT verbose_level);
	INT element_linear_entry_frobenius(void *elt, INT verbose_level);
	void element_one(void *elt, INT verbose_level);
	INT element_is_one(void *elt, INT verbose_level);
	void element_unpack(void *elt, void *Elt, INT verbose_level);
	void element_pack(void *Elt, void *elt, INT verbose_level);
	void element_retrieve(INT hdl, void *elt, INT verbose_level);
	INT element_store(void *elt, INT verbose_level);
	void element_mult(void *a, void *b, void *ab, INT verbose_level);
	void element_invert(void *a, void *av, INT verbose_level);
	void element_transpose(void *a, void *at, INT verbose_level);
	void element_move(void *a, void *b, INT verbose_level);
	void element_dispose(INT hdl, INT verbose_level);
	void element_print(void *elt, ostream &ost);
	void element_print_quick(void *elt, ostream &ost);
	void element_print_latex(void *elt, ostream &ost);
	void element_print_verbose(void *elt, ostream &ost);
	void element_code_for_make_element(void *elt, INT *data);
	void element_print_for_make_element(void *elt, 
		ostream &ost);
	void element_print_for_make_element_no_commas(void *elt, 
		ostream &ost);
	void element_print_as_permutation(void *elt, 
		ostream &ost);
	void element_as_permutation(void *elt, 
		INT *perm, INT verbose_level);
	void element_print_as_permutation_verbose(void *elt, 
		ostream &ost, INT verbose_level);
	void element_print_as_permutation_with_offset(void *elt, 
		ostream &ost, 
		INT offset, INT f_do_it_anyway_even_for_big_degree, 
		INT f_print_cycles_of_length_one, 
		INT verbose_level);
	void element_print_as_permutation_with_offset_and_max_cycle_length(
		void *elt, 
		ostream &ost, INT offset, INT max_cycle_length, 
		INT f_orbit_structure);
	void element_print_image_of_set(void *elt, 
		INT size, INT *set);
	INT element_signum_of_permutation(void *elt);
	void element_write_file_fp(INT *Elt, 
		FILE *fp, INT verbose_level);
	void element_read_file_fp(INT *Elt, 
		FILE *fp, INT verbose_level);
	void element_write_file(INT *Elt, 
		const BYTE *fname, INT verbose_level);
	void element_read_file(INT *Elt, 
		const BYTE *fname, INT verbose_level);
	void element_write_to_memory_object(INT *Elt, 
		memory_object *m, INT verbose_level);
	void element_read_from_memory_object(INT *Elt, 
		memory_object *m, INT verbose_level);
	void element_write_to_file_binary(INT *Elt, 
		ofstream &fp, INT verbose_level);
	void element_read_from_file_binary(INT *Elt, 
		ifstream &fp, INT verbose_level);
	void random_element(sims *S, INT *Elt, 
		INT verbose_level);

	// in backtrack.C:
	INT is_minimal(
		INT size, INT *set, INT &backtrack_level, 
		INT verbose_level);
	void make_canonical(
		INT size, INT *set, 
		INT *canonical_set, INT *transporter, 
		INT &total_backtrack_nodes, 
		INT f_get_automorphism_group, sims *Aut,
		INT verbose_level);
	INT is_minimal_witness(
		INT size, INT *set, 
		INT &backtrack_level, INT *witness, 
		INT *transporter_witness, 
		INT &backtrack_nodes, 
		INT f_get_automorphism_group, sims &Aut,
		INT verbose_level);
};




// #############################################################################
// action_global.C:
// #############################################################################


action *create_automorphism_group_from_group_table(const BYTE *fname_base, 
	INT *Table, INT group_order, INT *gens, INT nb_gens, 
	strong_generators *&Aut_gens, 
	INT verbose_level);
void create_linear_group(sims *&S, action *&A, 
	finite_field *F, INT m, 
	INT f_projective, INT f_general, INT f_affine, 
	INT f_semilinear, INT f_special, 
	INT verbose_level);
action *create_induced_action_by_restriction(action *A, sims *S, 
	INT size, INT *set, INT f_induce, INT verbose_level);
action *create_induced_action_on_sets(action *A, sims *S, 
	INT nb_sets, INT set_size, INT *sets, INT f_induce, 
	INT verbose_level);
void create_orbits_on_subset_using_restricted_action(
	action *&A_by_restriction, schreier *&Orbits, 
	action *A, sims *S, INT size, INT *set, 
	INT verbose_level);
void create_orbits_on_sets_using_action_on_sets(action *&A_on_sets, 
	schreier *&Orbits, action *A, sims *S, 
	INT nb_sets, INT set_size, INT *sets, INT verbose_level);
action *new_action_by_right_multiplication(sims *group_we_act_on, 
	INT f_transfer_ownership, INT verbose_level);
void action_print_symmetry_group_type(ostream &ost, symmetry_group_type a);
INT choose_next_base_point_default_method(action *A, INT *Elt, 
	INT verbose_level);
void make_generators_stabilizer_of_three_components(
	action *A_PGL_n_q, action *A_PGL_k_q, 
	INT k, vector_ge *gens, INT verbose_level);
void make_generators_stabilizer_of_two_components(
	action *A_PGL_n_q, action *A_PGL_k_q, 
	INT k, vector_ge *gens, INT verbose_level);
// used in semifield
void generators_to_strong_generators(action *A, 
	INT f_target_go, longinteger_object &target_go, 
	vector_ge *gens, strong_generators *&Strong_gens, 
	INT verbose_level);
void compute_generators_GL_n_q(INT *&Gens, INT &nb_gens, 
	INT &elt_size, INT n, finite_field *F, INT verbose_level);
void order_of_PGGL_n_q(longinteger_object &go, INT n, INT q, 
	INT f_semilinear);
void set_orthogonal_group_type(INT f_siegel, 
	INT f_reflection, INT f_similarity, INT f_semisimilarity);
INT get_orthogonal_group_type_f_reflection();
void callback_choose_random_generator_orthogonal(INT iteration, 
	INT *Elt, void *data, INT verbose_level);
	// for use in action_init.C
void test_matrix_group(INT k, INT q, INT f_semilinear, 
	INT verbose_level);
void lift_generators(vector_ge *gens_in, vector_ge *&gens_out, 
	action *Aq, subfield_structure *S, INT n, INT verbose_level);
void retract_generators(vector_ge *gens_in, vector_ge *&gens_out, 
	action *AQ, subfield_structure *S, INT n, 
	INT verbose_level);
void lift_generators_to_subfield_structure(
	INT n, INT s, 
	subfield_structure *S, 
	action *Aq, action *AQ, 
	strong_generators *&Strong_gens, 
	INT verbose_level);
#if 0
action *create_automorphism_group_of_graph(
	INT n, INT *Adj, 
	INT verbose_level);
#endif
action *create_automorphism_group_of_colored_graph_object(
	colored_graph *CG, INT verbose_level);
action *create_automorphism_group_of_colored_graph(
	INT n, INT f_bitvec, UBYTE *Adj_bitvec, INT *Adj, 
	INT *vertex_colors, 
	INT verbose_level);
action *create_automorphism_group_of_graph_bitvec(
	INT n, UBYTE *Adj_bitvec, 
	INT verbose_level);
action *create_automorphism_group_of_graph_with_partition_and_labeling(
	INT n, 
	INT f_bitvector, UBYTE *Adj_bitvec, INT *Adj, 
	INT nb_parts, INT *parts, 
	INT *labeling, 
	INT verbose_level);
void create_incidence_matrix_of_graph(INT *Adj, INT n, 
	INT *&M, INT &nb_rows, INT &nb_cols, INT verbose_level);
action *create_automorphism_group_of_graph(INT *Adj, 
	INT n, INT verbose_level);
action *create_automorphism_group_and_canonical_labeling_of_graph(
	INT *Adj, INT n, INT *labeling, INT verbose_level);
// labeling[n]
action *create_automorphism_group_of_block_system(
	INT nb_points, INT nb_blocks, INT block_size, INT *Blocks, 
	INT verbose_level);
action *create_automorphism_group_of_incidence_matrix(
	INT m, INT n, INT *Mtx, 
	INT verbose_level);
action *create_automorphism_group_of_incidence_structure(
	incidence_structure *Inc, 
	INT verbose_level);
action *create_automorphism_group_of_incidence_structure_low_level(
	INT m, INT n, INT nb_inc, INT *X, 
	INT verbose_level);
void test_self_dual_self_polar(INT input_no, 
	INT m, INT n, INT nb_inc, INT *X, 
	INT &f_self_dual, INT &f_self_polar, 
	INT verbose_level);
void do_self_dual_self_polar(INT input_no, 
	INT m, INT n, INT nb_inc, INT *X, 
	INT &f_self_dual, INT &f_self_polar, 
	INT verbose_level);
void add_configuration_graph(ofstream &g, 
	INT m, INT n, INT nb_inc, INT *X, INT f_first, 
	INT verbose_level);
// O4_model:
void O4_isomorphism_2to4_embedded(action *A4, 
	action *A5, finite_field *Fq, 
	INT f_switch, INT *mtx2x2_T, INT *mtx2x2_S, INT *Elt, 
	INT verbose_level);
void O5_to_O4(action *A4, action *A5, 
	finite_field *Fq, 
	INT *mtx4x4, INT *mtx5x5, 
	INT verbose_level);
void O4_to_O5(action *A4, action *A5, 
	finite_field *Fq, 
	INT *mtx4x4, INT *mtx5x5, 
	INT verbose_level);
void print_4x4_as_2x2(action *A2, 
	finite_field *Fq, INT *mtx4x4);

INT reverse_engineer_semilinear_map(action *A, 
	projective_space *P, 
	INT *Elt, INT *Mtx, INT &frobenius, 
	INT verbose_level);
sims *set_stabilizer_in_projective_space(
	action *A_linear, projective_space *P, 
	INT *set, INT set_size, INT &canonical_pt, 
	INT *canonical_set_or_NULL, 
	INT f_save_incma_in_and_out, 
	const BYTE *save_incma_in_and_out_prefix, 
	INT verbose_level);
void projective_space_init_line_action(projective_space *P, 
	action *A_points, action *&A_on_lines, 
	INT verbose_level);
void color_distribution_matrix(action *A, 
	INT *Elt, INT n, UBYTE *Adj, INT *colors, classify *C, 
	INT *&Mtx, INT verbose_level);
void test_color_distribution(action *A, 
	vector_ge *gens, INT n, UBYTE *Adj, INT *colors, 
	INT verbose_level);
void color_preserving_subgroup(action *A, 
	INT n, UBYTE *Adj, INT *colors, sims *&Subgroup, 
	INT verbose_level);
INT test_automorphism_group_of_graph_bitvec(action *A, 
	INT n, UBYTE *Adj, INT verbose_level);
void compute_conjugacy_classes(sims *S, action *&Aconj, 
	action_by_conjugation *&ABC, schreier *&Sch, 
	strong_generators *&SG, INT &nb_classes, 
	INT *&class_size, INT *&class_rep, 
	INT verbose_level);
INT group_ring_element_size(action *A, sims *S);
void group_ring_element_create(action *A, sims *S, INT *&elt);
void group_ring_element_free(action *A, sims *S, INT *elt);
void group_ring_element_print(action *A, sims *S, INT *elt);
void group_ring_element_copy(action *A, sims *S, 
	INT *elt_from, INT *elt_to);
void group_ring_element_zero(action *A, sims *S, 
	INT *elt);
void group_ring_element_mult(action *A, sims *S, 
	INT *elt1, INT *elt2, INT *elt3);
void perm_print_cycles_sorted_by_length(ostream &ost, 
	INT degree, INT *perm, INT verbose_level);
void perm_print_cycles_sorted_by_length_offset(ostream &ost, 
	INT degree, INT *perm, INT offset, 
	INT f_do_it_anyway_even_for_big_degree, 
	INT f_print_cycles_of_length_one, INT verbose_level);
void do_canonical_form(INT n, finite_field *F, 
	INT *set, INT set_size, INT f_semilinear, 
	const BYTE *fname_base, INT verbose_level);
void create_action_and_compute_orbits_on_equations(
	action *A, homogeneous_polynomial_domain *HPD, 
	INT *The_equations, INT nb_equations, strong_generators *gens, 
	action *&A_on_equations, schreier *&Orb, INT verbose_level);




// #############################################################################
// action_by_representation.C:
// #############################################################################

class action_by_representation {
public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

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
// action_by_subfield_structure.C:
// #############################################################################

class action_by_subfield_structure {
public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

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
// action_on_wedge_product.C:
// #############################################################################

class action_on_wedge_product {
public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

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
// action_on_grassmannian.C:
// #############################################################################

class action_on_grassmannian {
public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

	INT n;
	INT k;
	INT q;
	finite_field *F;
	INT low_level_point_size;
	
	action *A;
	grassmann *G;
	matrix_group *M;
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
// action_on_spread_set.C:
// #############################################################################

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
// action_on_orthogonal.C:
// #############################################################################

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
// action_on_cosets.C:
// #############################################################################

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
// action_on_factor_space.C:
// #############################################################################

class action_on_factor_space {
public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

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
// action_on_determinant.C:
// #############################################################################

class action_on_determinant {
public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

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
// action_on_sign.C:
// #############################################################################

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
// action_by_conjugation.C:
// #############################################################################

class action_by_conjugation {
public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

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
// action_by_right_multiplication.C:
// #############################################################################

class action_by_right_multiplication {
public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

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
// action_on_sets.C:
// #############################################################################

class action_on_sets {
public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

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
// action_on_subgroups.C:
// #############################################################################

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
// action_on_k_subsets.C:
// #############################################################################

class action_on_k_subsets {
public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

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
// action_on_bricks.C:
// #############################################################################

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
// action_on_andre.C:
// #############################################################################

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
// action_on_orbits.C:
// #############################################################################

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
// action_on_flags.C:
// #############################################################################

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
// action_by_restriction.C:
// #############################################################################

class action_by_restriction {
public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

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
// product_action.C:
// #############################################################################

class product_action {
public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

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

// #############################################################################
// action_on_homogeneous_polynomials.C:
// #############################################################################

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
// matrix_group.C:
// #############################################################################

class matrix_group {

public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

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
	INT GL_is_one(action &A, INT *Elt);
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
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

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
		// used in strore / retrieve
	
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
// page_storage.C:
// #############################################################################

class page_storage {

public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

	INT overall_length;
	
	INT entry_size; // in BYTE
	INT page_length_log; // number of bits
	INT page_length; // entries per page
	INT page_size; // size in BYTE of one page
	INT allocation_table_length;
		// size in BYTE of one allocation table
	
	INT page_ptr_used;
	INT page_ptr_allocated;
	INT page_ptr_oversize;
	
	UBYTE **pages;
	UBYTE **allocation_tables;
	
	INT next_free_entry;
	INT nb_free_entries;
	
	void init(INT entry_size, INT page_length_log, 
		INT verbose_level);
	void add_elt_print_function(
		void (* elt_print)(void *p, void *data, ostream &ost), 
		void *elt_print_data);
	void print();
	UBYTE *s_i_and_allocate(INT i);
	UBYTE *s_i_and_deallocate(INT i);
	UBYTE *s_i(INT i);
	UBYTE *s_i_and_allocation_bit(INT i, INT &f_allocated);
	void check_allocation_table();
	INT store(UBYTE *elt);
	void dispose(INT hdl);
	void check_free_list();
	page_storage();
	~page_storage();
	void print_storage_used();
	
	INT f_elt_print_function;
	void (* elt_print)(void *p, void *data, ostream &ost);
	void *elt_print_data;
};

void test_page_storage(INT f_v);




// #############################################################################
// vector_ge.C:
// #############################################################################

class vector_ge {

public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;
	static INT allocation_id;
	static void *allocated_objects;

	action *A;
	INT *data;
	INT len;
	
	vector_ge();
	vector_ge(action *A);
	~vector_ge();
	void null();
	void freeself();
	void init(action *A);
	void init_by_hdl(action *A, INT *gen_hdl, INT nb_gen);
	void init_single(action *A, INT *Elt);
	void init_double(action *A, INT *Elt1, INT *Elt2);
	void init_from_permutation_representation(action *A, INT *data, 
		INT nb_elements, INT verbose_level);
		// data[nb_elements * degree]
	void init_from_data(action *A, INT *data, 
		INT nb_elements, INT elt_size, INT verbose_level);
	void init_conjugate_svas_of(vector_ge *v, INT *Elt, 
		INT verbose_level);
	void init_conjugate_sasv_of(vector_ge *v, INT *Elt, 
		INT verbose_level);
	INT *ith(INT i);
	void print(ostream &ost);
	//ostream& print(ostream& ost);
	ostream& print_quick(ostream& ost);
	ostream& print_tex(ostream& ost);
	ostream& print_as_permutation(ostream& ost);
	void allocate(INT length);
	void reallocate(INT new_length);
	void reallocate_and_insert_at(INT position, INT *elt);
	void insert_at(INT length_before, INT position, INT *elt);
		// does not reallocate, but shifts elements up to make space.
		// the last element might be lost if there is no space.
	void append(INT *elt);
	void copy_in(INT i, INT *elt);
	void copy_out(INT i, INT *elt);
	void conjugate_svas(INT *Elt);
	void conjugate_sasv(INT *Elt);
	void print_with_given_action(ostream &ost, action *A2);
	void print(ostream &ost, INT f_print_as_permutation, 
		INT f_offset, INT offset, 
		INT f_do_it_anyway_even_for_big_degree, 
		INT f_print_cycles_of_length_one);
	void write_to_memory_object(memory_object *m, 
		INT verbose_level);
	void read_from_memory_object(memory_object *m, 
		INT verbose_level);
	void write_to_file_binary(ofstream &fp, 
		INT verbose_level);
	void read_from_file_binary(ifstream &fp, 
		INT verbose_level);
	void extract_subset_of_elements_by_rank_text_vector(
		const BYTE *rank_vector_text, sims *S, 
		INT verbose_level);
	void extract_subset_of_elements_by_rank(INT *rank_vector, 
		INT len, sims *S, INT verbose_level);
	INT test_if_all_elements_stabilize_a_point(action *A2, INT pt);
	INT test_if_all_elements_stabilize_a_set(action *A2, 
		INT *set, INT sz, INT verbose_level);
};


// #############################################################################
// schreier.C:
// #############################################################################


class schreier {

public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;


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

};


// #############################################################################
// schreier_vector.C:
// #############################################################################


INT schreier_vector_coset_rep_inv_general(action *A, 
	INT *sv, INT *hdl_gen, INT pt, 
	INT &pt0, INT *cosetrep, INT *Elt1, INT *Elt2, INT *Elt3, 
	INT f_trivial_group, INT f_check_image, 
	INT f_allow_failure, INT verbose_level);
// determines pt0 to be the first point of the orbit containing pt.
// cosetrep will be a group element that maps pt to pt0.
INT schreier_vector_coset_rep_inv_compact_general(action *A, 
	INT *sv, INT *hdl_gen, INT pt, 
	INT &pt0, INT *cosetrep, INT *Elt1, INT *Elt2, INT *Elt3, 
	INT f_trivial_group, INT f_check_image, 
	INT f_allow_failure, INT verbose_level);
void schreier_vector_coset_rep_inv(action *A, INT *sv, INT *hdl_gen, INT pt, 
	INT &pt0, INT *cosetrep, INT *Elt1, INT *Elt2, INT *Elt3, 
	INT f_trivial_group, INT f_compact, INT f_check_image, 
	INT verbose_level);
	// determines pt0 to be the first point of the 
	// orbit containing pt.
	// cosetrep will be a group element that maps pt to pt0.
void schreier_vector_coset_rep_inv_compact(action *A, 
	INT *sv, INT *hdl_gen, INT pt, 
	INT &pt0, INT *cosetrep, INT *Elt1, INT *Elt2, INT *Elt3, 
	INT f_trivial_group, INT f_check_image, 
	INT verbose_level);
void schreier_vector_coset_rep_inv1(action *A, 
	INT *sv, INT *hdl_gen, INT pt, 
	INT &pt0, INT *cosetrep, INT *Elt1, INT *Elt2, INT *Elt3);


void schreier_vector_print(INT *sv);
void schreier_vector_print_tree(INT *sv, INT verbose_level);
INT schreier_vector_compute_depth_recursively(INT n, 
	INT *Depth, INT *pts, INT *prev, INT pt);
INT sv_number_of_orbits(INT *sv);
void analyze_schreier_vector(INT *sv, INT verbose_level);
	// we assume that the group is not trivial
INT schreier_vector_determine_depth_recursion(INT n, INT *pts, INT *prev, 
	INT *depth, INT *ancestor, INT pos);

// #############################################################################
// sims.C:
// #############################################################################

class sims {

public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;


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
		// in the generator arry
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


// sims2.C:
void choose_random_generator_derived_group(sims *G, INT *Elt, 
	INT verbose_level);


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
// group.C:
// #############################################################################

class group {

public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;


	action *A;

	INT f_has_ascii_coding;
	char *ascii_coding;

	INT f_has_strong_generators;
	vector_ge *SG;
	INT *tl;
	
	INT f_has_sims;
	sims *S;
	
	group();
	group(action *A);
	group(action *A, const char *ascii_coding);
	group(action *A, vector_ge &SG, INT *tl);
	~group();
	void freeself();
	void init(action *A);
	void init_ascii_coding_to_sims(const char *ascii_coding);
	void init_ascii_coding(const char *ascii_coding);
	void delete_ascii_coding();
	void delete_sims();
	void init_strong_generators_empty_set();
	void init_strong_generators(vector_ge &SG, INT *tl);
	void init_strong_generators_by_hdl(INT nb_gen, INT *gen_hdl, 
		INT *tl, INT verbose_level);
	void delete_strong_generators();
	void require_ascii_coding();
	void require_strong_generators();
	void require_sims();
	void group_order(longinteger_object &go);
	void print_group_order(ostream &ost);
	void print_tl();
	void code_ascii(INT verbose_level);
	void decode_ascii(INT verbose_level);
	void schreier_sims(INT verbose_level);
	void get_strong_generators(INT verbose_level);
	void point_stabilizer(group &stab, INT pt, INT verbose_level);
	void point_stabilizer_with_action(action *A2, 
		group &stab, INT pt, INT verbose_level);
	void induced_action(action &induced_action, 
		group &H, group &K, INT verbose_level);
	void extension(group &N, group &H, INT verbose_level);
		// N needs to have strong generators, 
		// H needs to have sims
		// N and H may have different actions, 
		// the action of N is taken for the extension.
	void print_strong_generators(ostream &ost, 
		INT f_print_as_permutation);
	void print_strong_generators_with_different_action(
		ostream &ost, action *A2);
	void print_strong_generators_with_different_action_verbose(
		ostream &ost, action *A2, INT verbose_level);

};


// #############################################################################
// interface.C
// #############################################################################

INT induced_action_element_image_of(action &A, INT a, 
	void *elt, INT verbose_level);
void induced_action_element_image_of_low_level(action &A, 
	INT *input, INT *output, void *elt, INT verbose_level);
INT induced_action_element_linear_entry_ij(action &A, 
	void *elt, INT i, INT j, INT verbose_level);
INT induced_action_element_linear_entry_frobenius(action &A, 
	void *elt, INT verbose_level);
void induced_action_element_one(action &A, 
	void *elt, INT verbose_level);
INT induced_action_element_is_one(action &A, 
	void *elt, INT verbose_level);
void induced_action_element_unpack(action &A, 
	void *elt, void *Elt, INT verbose_level);
void induced_action_element_pack(action &A, 
	void *Elt, void *elt, INT verbose_level);
void induced_action_element_retrieve(action &A, 
	INT hdl, void *elt, INT verbose_level);
INT induced_action_element_store(action &A, 
	void *elt, INT verbose_level);
void induced_action_element_mult(action &A, 
	void *a, void *b, void *ab, INT verbose_level);
void induced_action_element_invert(action &A, 
	void *a, void *av, INT verbose_level);
void induced_action_element_transpose(action &A, 
	void *a, void *at, INT verbose_level);
void induced_action_element_move(action &A, 
	void *a, void *b, INT verbose_level);
void induced_action_element_dispose(action &A, 
	INT hdl, INT verbose_level);
void induced_action_element_print(action &A, 
	void *elt, ostream &ost);
void induced_action_element_print_quick(action &A, 
	void *elt, ostream &ost);
void induced_action_element_print_latex(action &A, 
	void *elt, ostream &ost);
void induced_action_element_print_verbose(action &A, 
	void *elt, ostream &ost);
void induced_action_element_code_for_make_element(action &A, 
	void *elt, INT *data);
void induced_action_element_print_for_make_element(action &A, 
	void *elt, ostream &ost);
void induced_action_element_print_for_make_element_no_commas(
	action &A, void *elt, ostream &ost);
void induced_action_print_point(action &A, INT a, ostream &ost);


// #############################################################################
// interface_matrix_group.C
// #############################################################################


INT matrix_group_element_image_of(action &A, INT a, 
	void *elt, INT verbose_level);
void matrix_group_element_image_of_low_level(action &A, 
	INT *input, INT *output, void *elt, INT verbose_level);
INT matrix_group_element_linear_entry_ij(action &A, 
	void *elt, INT i, INT j, INT verbose_level);
INT matrix_group_element_linear_entry_frobenius(action &A, 
	void *elt, INT verbose_level);
void matrix_group_element_one(action &A, 
	void *elt, INT verbose_level);
INT matrix_group_element_is_one(action &A, 
	void *elt, INT verbose_level);
void matrix_group_element_unpack(action &A, 
	void *elt, void *Elt, INT verbose_level);
void matrix_group_element_pack(action &A, 
	void *Elt, void *elt, INT verbose_level);
void matrix_group_element_retrieve(action &A, 
	INT hdl, void *elt, INT verbose_level);
INT matrix_group_element_store(action &A, 
	void *elt, INT verbose_level);
void matrix_group_element_mult(action &A, 
	void *a, void *b, void *ab, INT verbose_level);
void matrix_group_element_invert(action &A, 
	void *a, void *av, INT verbose_level);
void matrix_group_element_transpose(action &A, 
	void *a, void *at, INT verbose_level);
void matrix_group_element_move(action &A, 
	void *a, void *b, INT verbose_level);
void matrix_group_element_dispose(action &A, 
	INT hdl, INT verbose_level);
void matrix_group_element_print(action &A, 
	void *elt, ostream &ost);
void matrix_group_element_code_for_make_element(
	action &A, void *elt, INT *data);
void matrix_group_element_print_for_make_element(
	action &A, void *elt, ostream &ost);
void matrix_group_element_print_for_make_element_no_commas(
	action &A, void *elt, ostream &ost);
void matrix_group_element_print_quick(action &A, 
	void *elt, ostream &ost);
void matrix_group_element_print_latex(action &A, 
	void *elt, ostream &ost);
void matrix_group_element_print_as_permutation(
	action &A, void *elt, ostream &ost);
void matrix_group_element_print_verbose(action &A, 
	void *elt, ostream &ost);
void matrix_group_elt_print(void *elt, 
	void *data, ostream &ost);
void matrix_group_print_point(action &A, 
	INT a, ostream &ost);


// #############################################################################
// interface_perm_group.C
// #############################################################################

INT perm_group_element_image_of(action &A, INT a, 
	void *elt, INT verbose_level);
void perm_group_element_one(action &A, 
	void *elt, INT verbose_level);
INT perm_group_element_is_one(action &A, 
	void *elt, INT verbose_level);
void perm_group_element_unpack(action &A, 
	void *elt, void *Elt, INT verbose_level);
void perm_group_element_pack(action &A, 
	void *Elt, void *elt, INT verbose_level);
void perm_group_element_retrieve(action &A, 
	INT hdl, void *elt, INT verbose_level);
INT perm_group_element_store(action &A, 
	void *elt, INT verbose_level);
void perm_group_element_mult(action &A, 
	void *a, void *b, void *ab, INT verbose_level);
void perm_group_element_invert(action &A, 
	void *a, void *av, INT verbose_level);
void perm_group_element_move(action &A, 
	void *a, void *b, INT verbose_level);
void perm_group_element_dispose(action &A, 
	INT hdl, INT verbose_level);
void perm_group_element_print(action &A, 
	void *elt, ostream &ost);
void perm_group_element_print_latex(action &A, 
	void *elt, ostream &ost);
void perm_group_element_print_verbose(action &A, 
	void *elt, ostream &ost);
void perm_group_element_code_for_make_element(action &A, 
	void *elt, INT *data);
void perm_group_element_print_for_make_element(action &A, 
	void *elt, ostream &ost);
void perm_group_element_print_for_make_element_no_commas(
	action &A, void *elt, ostream &ost);
void perm_group_elt_print(void *elt, void *data, 
	ostream &ost);
void perm_group_print_point(action &A, INT a, ostream &ost);



// #############################################################################
// union_find.C:
// #############################################################################



class union_find {

public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;


	action *A;
	INT *prev;


	union_find();
	~union_find();
	void freeself();
	void null();
	void init(action *A, INT verbose_level);
	INT ancestor(INT i);
	INT count_ancestors();
	INT count_ancestors_above(INT i0);
	void do_union(INT a, INT b);
	void print();
	void add_generators(vector_ge *gens, INT verbose_level);
	void add_generator(INT *Elt, INT verbose_level);
};

// #############################################################################
// union_find_on_k_subsets.C:
// #############################################################################



class union_find_on_k_subsets {

public:

	INT *set;
	INT set_sz;
	INT k;

	sims *S;

	INT *interesting_k_subsets;
	INT nb_interesting_k_subsets;
	
	action *A_original;
	action *Ar; // restricted action on the set
	action *Ar_perm;
	action *Ark; // Ar_perm on k_subsets
	action *Arkr; // Ark restricted to interesting_k_subsets

	vector_ge *gens_perm;

	union_find *UF;


	union_find_on_k_subsets();
	~union_find_on_k_subsets();
	void freeself();
	void null();
	void init(action *A_original, sims *S, 
		INT *set, INT set_sz, INT k, 
		INT *interesting_k_subsets, INT nb_interesting_k_subsets, 
		INT verbose_level);
	INT is_minimal(INT rk, INT verbose_level);
};


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
// linear_group.C:
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
// set_and_stabilizer.C:
// #############################################################################

class set_and_stabilizer {

public:
	action *A;
	action *A2;
	INT *data;
	INT sz;
	longinteger_object target_go;
	strong_generators *Strong_gens;
	sims *Stab;

	set_and_stabilizer();
	~set_and_stabilizer();
	void null();
	void freeself();
	void init(action *A, action *A2, INT verbose_level);
	void init_everything(action *A, action *A2, INT *Set, INT set_sz, 
		strong_generators *gens, INT verbose_level);
	void allocate_data(INT sz, INT verbose_level);
	set_and_stabilizer *create_copy(INT verbose_level);
	void init_data(INT *data, INT sz, INT verbose_level);
	void init_stab_from_data(INT *data_gens, 
		INT data_gens_size, INT nb_gens, const BYTE *ascii_target_go, 
		INT verbose_level);
	void init_stab_from_file(const BYTE *fname_gens, 
		INT verbose_level);
	void print_set_tex(ostream &ost);
	void print_set_tex_for_inline_text(ostream &ost);
	void print_generators_tex(ostream &ost);
	//set_and_stabilizer *apply(INT *Elt, INT verbose_level);
	void apply_to_self(INT *Elt, INT verbose_level);
	void apply_to_self_inverse(INT *Elt, INT verbose_level);
	void apply_to_self_element_raw(INT *Elt_data, INT verbose_level);
	void apply_to_self_inverse_element_raw(INT *Elt_data, 
		INT verbose_level);
	void rearrange_by_orbits(INT *&orbit_first, 
		INT *&orbit_length, INT *&orbit, 
		INT &nb_orbits, INT verbose_level);
	action *create_restricted_action_on_the_set(INT verbose_level);
	void print_restricted_action_on_the_set(INT verbose_level);
	void test_if_group_acts(INT verbose_level);
	
	void init_surface(surface *Surf, action *A, action *A2, 
		INT q, INT no, INT verbose_level);
};

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
// orbit_transversal.C:
// #############################################################################

class orbit_transversal {

public:
	action *A;
	action *A2;
	
	INT nb_orbits;
	set_and_stabilizer *Reps;

	orbit_transversal();
	~orbit_transversal();
	void null();
	void freeself();
	void read_from_file(action *A, action *A2, 
		const BYTE *fname, INT verbose_level);
};

// #############################################################################
// projective_space_with_action.C:
// #############################################################################


class projective_space_with_action {

public:

	INT n;
	INT d; // n + 1
	INT q;
	finite_field *F; // do not free
	INT f_semilinear;
	INT f_init_incidence_structure;

	projective_space *P;
	

	action *A; // linear group PGGL(d,q)
	action *A_on_lines; // linear group PGGL(d,q) acting on lines
	sims *S; // linear group PGGL(d,q)

	INT *Elt1;


	projective_space_with_action();
	~projective_space_with_action();
	void null();
	void freeself();
	void init(finite_field *F, INT n, INT f_semilinear, 
		INT f_init_incidence_structure, INT verbose_level);
	void init_group(INT f_semilinear, INT verbose_level);
	strong_generators *set_stabilizer(
		INT *set, INT set_size, INT &canonical_pt, 
		INT *canonical_set_or_NULL, 
		INT f_save_incma_in_and_out, 
		const BYTE *save_incma_in_and_out_prefix, 
		INT f_compute_canonical_form, 
		UBYTE *&canonical_form, INT &canonical_form_len, 
		INT verbose_level);
	strong_generators *set_stabilizer_of_object(
		object_in_projective_space *OiP, 
		INT f_save_incma_in_and_out, 
		const BYTE *save_incma_in_and_out_prefix, 
		INT f_compute_canonical_form, 
		UBYTE *&canonical_form, 
		INT &canonical_form_len, 
		INT verbose_level);
	void report_fixed_objects_in_PG_3_tex(
		INT *Elt, ostream &ost, 
		INT verbose_level);
};




