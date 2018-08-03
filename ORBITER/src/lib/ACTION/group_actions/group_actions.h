// group_actions.h
//
// Anton Betten
//
// moved here from action.h: July 28, 2018
// based on action.h which was started:  August 13, 2005

// #############################################################################
// action.C:
// #############################################################################

class action {
public:
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


