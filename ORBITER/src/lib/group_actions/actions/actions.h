// group_actions.h
//
// Anton Betten
//
// moved here from action.h: July 28, 2018
// based on action.h which was started:  August 13, 2005


namespace orbiter {

namespace group_actions {


// #############################################################################
// action.C:
// #############################################################################

//! a permutation group in a fixed action.
/*! The class action provides a unified interface to a permutation group.
 * A permutation group in Orbiter always acts on the set {0,...,degree-1},
 * where degree is the degree stored in the action class.
 * The primary goal of this class is to provide the functionality for
 * group actions. Many different types of group actions are supported.
 * The way in which a group action is realized is by means of the following
 * two components:
 *
 * symmetry_group G
 * symmetry_group_type type_G
 *
 * The type is an enumeration type which specifies the type of group action.
 * The component G is a union consisting of all possible pointer types
 * to objects of action type. Depending on the type of group action, type_G
 * is set and G holds a pointer to the specific class implementing this type
 * of action. There are the atomic types of group action and there are the
 * induced action types. At present, there are exactly two atomic types.
 * One is matrix_group, which represents a matrix group over a finite field.
 * The other one is perm_group, which represents a abstract permutation group.
 * An abstract permutation group is a group where the elements are given
 * as list of images in the form of a vector.
 * We distinguish between matrix groups and abstract permutation group because
 * the elements of a matrix group are given by matrices (possibly plus
 * a vector, and possibly plus a field automorphism). The elements
 * of a matrix group are not stored as permutations.
 * Because of this, a matrix group can be more efficient than the corresponding
 * isomorphic abstract permutation group. For instance, group multiplication
 * for matrix group elements is matrix multiplication. Group multiplication
 * for abstract permutation groups is composition of the list of images.
 * Matrix group multiplication seems faster than composing lists of images,
 * at least asymptotically.
 *
 * Optionally, the class action also serves as a means to represent a group by
 * means of a stabilizer chain (sims chain). This proves to be a bit tricky
 * because the class sims (implementing a sims chain) requires an action object
 * to get going. This is a kind of chicken-and-egg problem. The action needs
 * a sims, and the sims needs an action. The solution is to have a bit
 * of replication of code. The action class has its own tiny implementation of
 * a stabilizer chain. This allows a group to be set up in action without the
 * need for a sims object. It is helpful that we know a stabilizer chain
 * for the projective linear groups, so we can set up the tiny version
 * of a sims chain in action without using a sims object. The
 * problem of setting up a group arises within the various init
 * function in the action class. The functionality for known stabilizer chains
 * is pushed down in the foundations library, in order to keep the
 * mathematics for stabilizer chains away from the implementation of the
 * stabilizer chains and action classes. The action of the projective group
 * on the set of points of projective space is called the natural action.
 *
 *
 * In most cases of induced actions, it is not a good idea to
 * replace the natural action by the induced action. The problem with
 * actions for the purposes of representing groups as sims chains
 * is the following. A sims chain is efficient only if the maximal
 * degree of the basic actions (the actions in the stabilizer chain)
 * is small. For the basic action of projective groups, this contition
 * is usually met. For most induced actions, the maximal degree is
 * large because the action has an even larger degree than the
 * original basic action. For this reason, it is not advisable to represent
 * a group in any of the induced actions. This means that we will carry
 * two actions around at the same time. The first action is the basic action
 * which represents the group in the natural action. The second action is
 * the induced action which we desire in our particular application.
 * This action will be use for the purposes of the group action only.
 * It will not be used for a stabilizer chain for the group.
 * The poset classification algorithm takes two actions as input.
 * The first action will be used to represent groups or subgroups.
 * The second action is the action on the poset of interest.
 *
 *
 */
class action {
public:


	/** the symmetry group is a permutation group
	 *
	 */
	int f_allocated;
	
	/** the type of group */
	symmetry_group_type type_G;

	/** a pointer to the implementation of the group.
	 * symmetry_group is a union of pointer types.
	 */
	symmetry_group G;


	/** Whether the group has a subaction.
	 * For instance, induced actions have subactions. */
	int f_has_subaction;
	int f_subaction_is_allocated;
	
	/** the subaction */
	action *subaction;


	/** whether the group has a strong generators */
	int f_has_strong_generators;

	/** strong generating set for the group */
	strong_generators *Strong_gens;

	/** the size of the set we act on */
	int degree;


	/** whether the action is linear (including semilinear) */
	int f_is_linear;
		// matrix_group_t, 
		// action_on_wedge_product_t, 
		// action_by_representation_t
	
	/** the dimension if we are linear */
	int dimension;


	/** whether we have a base (b_0,\ldots,b_{l-1}) */
	int f_has_base;

	/** the length of the base */
	int base_len;



	/** the base (b_0,\ldots,b_{l-1}) */
	int *base;



	/** the length of the orbit of $G^{(i)}$ on $b_i$ */
	int *transversal_length;

	/** the orbit of  b_i */
	int **orbit;

	/** the inverse orbit of  b_i */
	int **orbit_inv;

	/** how many int we need to store one group element */
	int elt_size_in_int;

	/** how many char do we need
	 * to store a group element in the compressed form */
	int coded_elt_size_in_char;

	
	/** the number of int that are needed to
	 * make an element of this group
	 * using the make_element function */
	int make_element_size;


	/** the number of int that are needed to
	 * represent a point in low-level format
	 * (input and output in element_image_of_low_level
	 * point to that many int) */
	int low_level_point_size;
	
	int f_has_sims;
	/** sims chain for the group */
	sims *Sims;
	
	int f_has_kernel;
	/** kernel of the action */
	sims *Kernel;
	
	int f_group_order_is_small;
	int *path;
	

	action_pointer_table *ptr;

	/** temporary elements */
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5;
	int *eltrk1, *eltrk2, *eltrk3, *elt_mult_apply;
	uchar *elt1;
	char *element_rw_memory_object;
		// [coded_elt_size_in_char]
		// for element_write_to_memory_object, 
		// element_read_from_memory_object

	
	/** a label for the group which can be used in filenames */
	char group_prefix[1000];

	/** a fancy label for the group */
	char label[1000];

	/** a fancy label for the group for latex */
	char label_tex[1000];



	// action.C:
	action();
	~action();
	void null();
	void freeself();
	
	void null_element_data();
	void allocate_element_data();
	void free_element_data();
	void null_base_data();
	void allocate_base_data(int base_len);
	void reallocate_base(int new_base_point);
	void free_base_data();
	
	int find_non_fixed_point(void *elt, int verbose_level);
	int find_fixed_points(void *elt, 
		int *fixed_points, int verbose_level);
	int test_if_set_stabilizes(int *Elt, 
		int size, int *set, int verbose_level);
	void map_a_set(int *set, int *image_set, 
		int n, int *Elt, int verbose_level);
	void map_a_set_and_reorder(int *set, int *image_set, 
		int n, int *Elt, int verbose_level);
	void print_all_elements();

	void init_sims(sims *G, int verbose_level);
	void init_base_from_sims(sims *G, int verbose_level);
	int element_has_order_two(int *E1, int verbose_level);
	int product_has_order_two(int *E1, int *E2, int verbose_level);
	int product_has_order_three(int *E1, int *E2, int verbose_level);
	int element_order(void *elt);
	int element_order_verbose(void *elt, int verbose_level);
	int element_order_if_divisor_of(void *elt, int o);
	void compute_all_point_orbits(schreier &S, 
		vector_ge &gens, int verbose_level);
	
	/** the index of the first moved base point */
	int depth_in_stab_chain(int *Elt);


		/** all strong generators that
		 * leave base points 0,..., depth - 1 fix */
	void strong_generators_at_depth(int depth, vector_ge &gen);
	void compute_point_stabilizer_chain(vector_ge &gen, 
		sims *S, int *sequence, int len, 
		int verbose_level);
	int compute_orbit_of_point(vector_ge &strong_generators, 
		int pt, int *orbit, 
		int verbose_level);
	int compute_orbit_of_point_generators_by_handle(
		int nb_gen, int *gen_handle, 
		int pt, int *orbit, 
		int verbose_level);
	int least_image_of_point(vector_ge &strong_generators, 
		int pt, int *transporter, 
		int verbose_level);
	int least_image_of_point_generators_by_handle(int nb_gen, 
		int *gen_handle, int pt, int *transporter, 
		int verbose_level);
	void all_point_orbits(schreier &Schreier, int verbose_level);
	void all_point_orbits_from_generators(schreier &Schreier,
			strong_generators *SG,
			int verbose_level);
	void all_point_orbits_from_single_generator(schreier &Schreier,
			int *Elt,
			int verbose_level);
	void compute_stabilizer_orbits(partitionstack *&Staborbits, 
		int verbose_level);
	int check_if_in_set_stabilizer(int *Elt, 
		int size, int *set, 
		int verbose_level);
	int check_if_transporter_for_set(int *Elt, 
		int size, int *set1, int *set2, 
		int verbose_level);
	void compute_set_orbit(vector_ge &gens, 
		int size, int *set, 
		int &nb_sets, int **&Sets, 
		int **&Transporter, 
		int verbose_level);
	void delete_set_orbit(int nb_sets, 
		int **Sets, int **Transporter);
	void compute_minimal_set(vector_ge &gens, 
		int size, int *set, 
		int *minimal_set, int *transporter, 
		int verbose_level);
	void find_strong_generators_at_level(int base_len, 
		int *the_base, int level, 
		vector_ge &gens, vector_ge &subset_of_gens, 
		int verbose_level);
	void compute_strong_generators_from_sims(int verbose_level);
	void make_element_from_permutation_representation(int *Elt, 
		int *data, int verbose_level);
	void make_element_from_base_image(int *Elt, int *data, 
		int verbose_level);
	void make_element_2x2(int *Elt, int a0, int a1, int a2, int a3);
	void make_element(int *Elt, int *data, int verbose_level);
	void build_up_automorphism_group_from_aut_data(int nb_auts, 
		int *aut_data, 
		sims &S, int verbose_level);
	void element_power_int_in_place(int *Elt, 
		int n, int verbose_level);
	void word_in_ab(int *Elt1, int *Elt2, int *Elt3, 
		const char *word, int verbose_level);
	void init_group_from_generators(int *group_generator_data, 
		int group_generator_size, 
		int f_group_order_target, const char *group_order_target, 
		vector_ge *gens, strong_generators *&Strong_gens, 
		int verbose_level);
	void init_group_from_generators_by_base_images(
		int *group_generator_data, int group_generator_size, 
		int f_group_order_target, const char *group_order_target, 
		vector_ge *gens, strong_generators *&Strong_gens_out, 
		int verbose_level);
	void print_symmetry_group_type(std::ostream &ost);
	void report(std::ostream &ost);
	void print_info();
	void print_base();
	void group_order(longinteger_object &go);
	void print_group_order(std::ostream &ost);
	void print_group_order_long(std::ostream &ost);
	void print_vector(vector_ge &v);
	void print_vector_as_permutation(vector_ge &v);
	void element_print_base_images(int *Elt);
	void element_print_base_images(int *Elt, std::ostream &ost);
	void element_print_base_images_verbose(int *Elt, 
			std::ostream &ost, int verbose_level);
	void element_base_images(int *Elt, int *base_images);
	void element_base_images_verbose(int *Elt, 
		int *base_images, int verbose_level);
	void minimize_base_images(int level, sims *S, 
		int *Elt, int verbose_level);
	void element_conjugate_bvab(int *Elt_A, 
		int *Elt_B, int *Elt_C, int verbose_level);
	void element_conjugate_babv(int *Elt_A, 
		int *Elt_B, int *Elt_C, int verbose_level);
	void element_commutator_abavbv(int *Elt_A, 
		int *Elt_B, int *Elt_C, int verbose_level);
	void read_representatives(char *fname, 
		int *&Reps, int &nb_reps, int &size, int verbose_level);
	void read_representatives_and_strong_generators(char *fname, 
		int *&Reps, 
		char **&Aut_ascii, int &nb_reps, 
		int &size, int verbose_level);
	void read_file_and_print_representatives(char *fname, 
		int f_print_stabilizer_generators);
	void read_set_and_stabilizer(const char *fname, 
		int no, int *&set, int &set_sz, sims *&stab, 
		strong_generators *&Strong_gens, 
		int &nb_cases, 
		int verbose_level);
	void get_generators_from_ascii_coding(char *ascii_coding, 
		vector_ge *&gens, int *&tl, int verbose_level);
	void lexorder_test(int *set, int set_sz, int &set_sz_after_test, 
		vector_ge *gens, int max_starter, 
		int verbose_level);
	void compute_orbits_on_points(schreier *&Sch, 
		vector_ge *gens, int verbose_level);
	void stabilizer_of_dual_hyperoval_representative(int k, 
		int n, int no, vector_ge *&gens, 
		const char *&stab_order, int verbose_level);
	void stabilizer_of_spread_representative(int q,
		int k, int no, vector_ge *&gens, const char *&stab_order, 
		int verbose_level);
	void point_stabilizer_any_point(int &pt,
		schreier *&Sch, sims *&Stab,
		strong_generators *&stab_gens,
		int verbose_level);
	void point_stabilizer_any_point_with_given_group(
		strong_generators *input_gens,
		int &pt,
		schreier *&Sch, sims *&Stab,
		strong_generators *&stab_gens,
		int verbose_level);
	void make_element_which_moves_a_line_in_PG3q(grassmann *Gr,
		int line_rk, int *Elt, int verbose_level);
	void list_elements_as_permutations_vertically(vector_ge *gens,
			std::ostream &ost);
	matrix_group *get_matrix_group();


	// action_group_theory.cpp:
	void normalizer_using_MAGMA(const char *fname_magma_prefix,
		sims *G, sims *H, strong_generators *&gens_N, int verbose_level);
	void conjugacy_classes_using_MAGMA(const char *prefix, 
		sims *G, int verbose_level);
	void read_conjugacy_classes_from_MAGMA(
			char *fname,
			int &nb_classes,
			int *&perms,
			int *&class_size,
			int *&class_order_of_element,
			int verbose_level);
	void conjugacy_classes_and_normalizers_using_MAGMA(
			const char *prefix,
			sims *G, int verbose_level);
	void read_conjugacy_classes_and_normalizers_from_MAGMA(
			char *fname,
			int &nb_classes,
			int *&perms,
			int *&class_size,
			int *&class_order_of_element,
			int *&class_normalizer_order,
			int *&class_normalizer_number_of_generators,
			int **&normalizer_generators_perms,
			int verbose_level);
	void centralizer_using_MAGMA(const char *prefix, 
		sims *G, int *Elt, int verbose_level);
	void conjugacy_classes_and_normalizers(
			int verbose_level);
	void read_conjugacy_classes_and_normalizers(
			char *fname, int verbose_level);
	void report_fixed_objects(int *Elt,
			char *fname_latex, int verbose_level);


	// action_indexing_cosets.C:
	void coset_unrank(sims *G, sims *U, int rank, 
		int *Elt, int verbose_level);
	int coset_rank(sims *G, sims *U, 
		int *Elt, int verbose_level);
		// used in generator::coset_unrank and generator::coset_rank
		// which in turn are used by 
		// generator::orbit_element_unrank and 
		// generator::orbit_element_rank

	// action_init.C:
	/** Create the projective linear (or semilinear) group PGL (or PGGL)*/
	void init_projective_group(int n, finite_field *F, 
		int f_semilinear, int f_basis,
		vector_ge *&nice_gens,
		int verbose_level);


	/** Create the affine group AGL(n,q) */
	void init_affine_group(int n, finite_field *F, 
		int f_semilinear, 
		int f_basis,
		vector_ge *&nice_gens,
		int verbose_level);

	/** Create the general linear group GL(n,q) */
	void init_general_linear_group(int n, finite_field *F, 
		int f_semilinear, int f_basis,
		vector_ge *&nice_gens,
		int verbose_level);

	void setup_linear_group_from_strong_generators(matrix_group *M, 
		vector_ge *&nice_gens,
		int verbose_level);

	/** Create the projective special linear group PSL */
	void init_projective_special_group(int n, finite_field *F,
		int f_semilinear, int f_basis, int verbose_level);

	void init_matrix_group_strong_generators_builtin(matrix_group *M, 
		vector_ge *&nice_gens,
		int verbose_level);
	void init_permutation_group(int degree, int verbose_level);
	void init_permutation_group_from_generators(int degree, 
		int f_target_go, longinteger_object &target_go, 
		int nb_gens, int *gens, 
		int given_base_length, int *given_base,
		int verbose_level);

	/** Create the affine group AGL(n,q) as abstract permutation group,
	 * not as matrix group */
	void init_affine_group(int n, int q, int f_translations, 
		int f_semilinear, int frobenius_power, 
		int f_multiplication, 
		int multiplication_order, int verbose_level);


	/** Create the symmetric group of degree degree
	 * as abstract permutation group */
	void init_symmetric_group(int degree, int verbose_level);


	void create_sims(int verbose_level);
	void create_orthogonal_group(action *subaction, 
		int f_has_target_group_order, 
		longinteger_object &target_go, 
		void (* callback_choose_random_generator)(int iteration, 
			int *Elt, void *data, int verbose_level), 
		int verbose_level);
	/** Create the direct product group M1 x M2 in product action
	 * and restrict the action to the grid. */
	void init_direct_product_group_and_restrict(
			matrix_group *M1, matrix_group *M2, int verbose_level);

	/** Create the direct product group M1 x M2 in product action */
	void init_direct_product_group(
			matrix_group *M1, matrix_group *M2,
			int verbose_level);

	/** Create the wreath product group AGL(n,q) wreath Sym(nb_factors)
	 * in wreath product action
	 * and restrict the action to the tensor space. */
	void init_wreath_product_group_and_restrict(int nb_factors, int n,
			finite_field *F, int verbose_level);

	/** Create the wreath product group AGL(n,q) wreath Sym(nb_factors)
	 * in wreath product action
	 */
	void init_wreath_product_group(int nb_factors, int n, finite_field *F,
		int verbose_level);

	/** Create the orthogonal group O(5,q) */
	void init_BLT(finite_field *F, int f_basis,
		int f_init_hash_table, int verbose_level);


	/** Create a group from generators */
	void init_group_from_strong_generators(vector_ge *gens, sims *K,
		int given_base_length, int *given_base,
		int verbose_level);


	/** Create the orthogonal group O^epsilon(n,q) */
	void init_orthogonal_group(int epsilon,
		int n, finite_field *F,
		int f_on_points, int f_on_lines,
		int f_on_points_and_lines,
		int f_semilinear,
		int f_basis, int verbose_level);

	sims *create_sims_from_generators_with_target_group_order_factorized(
		vector_ge *gens, int *tl, int len, int verbose_level);
	sims *create_sims_from_generators_with_target_group_order_int(
		vector_ge *gens, int target_go, int verbose_level);
	sims *create_sims_from_generators_with_target_group_order(
		vector_ge *gens, longinteger_object &target_go,
		int verbose_level);
	sims *create_sims_from_generators_without_target_group_order(
		vector_ge *gens, int verbose_level);
	sims *create_sims_from_single_generator_without_target_group_order(
		int *Elt, int verbose_level);
	sims *create_sims_from_generators_randomized(
		vector_ge *gens, int f_target_go, longinteger_object &target_go,
		int verbose_level);
	sims *create_sims_for_centralizer_of_matrix(
			int *Mtx, int verbose_level);

	
	// action_induce.C:

	action *induced_action_on_set_partitions(
			int universal_set_size, int partition_size,
			int verbose_level);
	/** Create the induced action on lines in PG(n-1,q)
	 * using an action_on_grassmannian object */
	void init_action_on_lines(action *A, finite_field *F, 
		int n, int verbose_level);

	void induced_action_by_representation_on_conic(action *A_old, 
		int f_induce_action, sims *old_G, 
		int verbose_level);
	void induced_action_on_cosets(action_on_cosets *A_on_cosets, 
		int f_induce_action, sims *old_G, 
		int verbose_level);
	void induced_action_on_factor_space(action *A_old, 
		action_on_factor_space *AF, 
		int f_induce_action, sims *old_G, 
		int verbose_level);
	action *induced_action_on_grassmannian(int k, 
		int verbose_level);
	void induced_action_on_grassmannian(action *A_old, 
		action_on_grassmannian *AG, 
		int f_induce_action, sims *old_G, 
		int verbose_level);
	void induced_action_on_spread_set(action *A_old, 
		action_on_spread_set *AS, 
		int f_induce_action, sims *old_G, 
		int verbose_level);
	void induced_action_on_orthogonal(action *A_old, 
		action_on_orthogonal *AO, 
		int f_induce_action, sims *old_G, 
		int verbose_level);
	void induced_action_on_wedge_product(action *A_old, 
		action_on_wedge_product *AW, 
		int f_induce_action, sims *old_G, 
		int verbose_level);
	void induced_action_by_subfield_structure(action *A_old, 
		action_by_subfield_structure *SubfieldStructure, 
		int f_induce_action, sims *old_G, 
		int verbose_level);
	void induced_action_on_determinant(sims *old_G, 
		int verbose_level);
	void induced_action_on_sign(sims *old_G, 
		int verbose_level);
	void induced_action_by_conjugation(sims *old_G, 
		sims *Base_group, int f_ownership, 
		int f_basis, int verbose_level);
	void induced_action_by_right_multiplication(
		int f_basis, sims *old_G, 
		sims *Base_group, int f_ownership, 
		int verbose_level);
	action *create_induced_action_on_sets(int nb_sets, 
		int set_size, int *sets, 
		int verbose_level);
	void induced_action_on_sets(action &old_action, sims *old_G, 
		int nb_sets, int set_size, int *sets, 
		int f_induce_action, int verbose_level);
	action *create_induced_action_on_subgroups(sims *S, 
		int nb_subgroups, int group_order, 
		subgroup **Subgroups, int verbose_level);
	void induced_action_on_subgroups(action *old_action, 
		sims *S, 
		int nb_subgroups, int group_order, 
		subgroup **Subgroups, 
		int verbose_level);
	void induced_action_by_restriction_on_orbit_with_schreier_vector(
		action &old_action, 
		int f_induce_action, sims *old_G, 
		schreier_vector *Schreier_vector,
		int pt, int verbose_level);
	action *restricted_action(int *points, int nb_points, 
		int verbose_level);
	void induced_action_by_restriction(action &old_action, 
		int f_induce_action, sims *old_G, 
		int nb_points, int *points, int verbose_level);
		// uses action_by_restriction data type
	void induced_action_on_pairs(action &old_action, sims *old_G, 
		int verbose_level);
	action *create_induced_action_on_ordered_pairs(int verbose_level);
	void induced_action_on_ordered_pairs(action &old_action, 
		sims *old_G, 
		int verbose_level);
	void induced_action_on_k_subsets(action &old_action, int k, 
		int verbose_level);
	void induced_action_on_orbits(action *old_action, 
		schreier *Sch, int f_play_it_safe, 
		int verbose_level);
	void induced_action_on_flags(action *old_action, 
		int *type, int type_len, 
		int verbose_level);
	void induced_action_on_bricks(action &old_action, 
		brick_domain *B, int f_linear_action, 
		int verbose_level);
	void induced_action_on_andre(action *An, 
		action *An1, andre_construction *Andre, 
		int verbose_level);
	void setup_product_action(action *A1, action *A2, 
		int f_use_projections, int verbose_level);
	void induced_action_on_homogeneous_polynomials(action *A_old, 
		homogeneous_polynomial_domain *HPD, 
		int f_induce_action, sims *old_G, 
		int verbose_level);
	void induced_action_on_homogeneous_polynomials_given_by_equations(
		action *A_old, 
		homogeneous_polynomial_domain *HPD, 
		int *Equations, int nb_equations, 
		int f_induce_action, sims *old_G, 
		int verbose_level);
	void induced_action_recycle_sims(action &old_action, 
		int verbose_level);
	void induced_action_override_sims(action &old_action, 
		sims *old_G, 
		int verbose_level);
	void induce(action *old_action, sims *old_G, 
		int base_of_choice_len, 
		int *base_of_choice, 
		int verbose_level);
	int least_moved_point_at_level(int level, 
		int verbose_level);
	void lex_least_base_in_place(int verbose_level);
	void lex_least_base(action *old_action, int verbose_level);
	int test_if_lex_least_base(int verbose_level);
	void base_change_in_place(int size, int *set, int verbose_level);
	void base_change(action *old_action, 
		int size, int *set, int verbose_level);


	// action_cb.C:
	int image_of(void *elt, int a);
	void image_of_low_level(void *elt,
			int *input, int *output, int verbose_level);
	int linear_entry_ij(void *elt, int i, int j);
	int linear_entry_frobenius(void *elt);
	void one(void *elt);
	int is_one(void *elt);
	void unpack(void *elt, void *Elt);
	void pack(void *Elt, void *elt);
	void retrieve(void *elt, int hdl);
	int store(void *elt);
	void mult(void *a, void *b, void *ab);
	void mult_apply_from_the_right(void *a, void *b);
		// a := a * b
	void mult_apply_from_the_left(void *a, void *b);
		// b := a * b
	void invert(void *a, void *av);
	void invert_in_place(void *a);
	void transpose(void *a, void *at);
	void move(void *a, void *b);
	void dispose(int hdl);
	void print(std::ostream &ost, void *elt);
	void print_quick(std::ostream &ost, void *elt);
	void print_as_permutation(std::ostream &ost, void *elt);
	void print_point(int a, std::ostream &ost);
	void code_for_make_element(int *data, void *elt);
	void print_for_make_element(std::ostream &ost, void *elt);
	void print_for_make_element_no_commas(std::ostream &ost, void *elt);
	
	int element_image_of(int a, void *elt, int verbose_level);
	void element_image_of_low_level(int *input, int *output, 
		void *elt, int verbose_level);
	int element_linear_entry_ij(void *elt, int i, int j, 
		int verbose_level);
	int element_linear_entry_frobenius(void *elt, int verbose_level);
	void element_one(void *elt, int verbose_level);
	int element_is_one(void *elt, int verbose_level);
	void element_unpack(void *elt, void *Elt, int verbose_level);
	void element_pack(void *Elt, void *elt, int verbose_level);
	void element_retrieve(int hdl, void *elt, int verbose_level);
	int element_store(void *elt, int verbose_level);
	void element_mult(void *a, void *b, void *ab, int verbose_level);
	void element_invert(void *a, void *av, int verbose_level);
	void element_transpose(void *a, void *at, int verbose_level);
	void element_move(void *a, void *b, int verbose_level);
	void element_dispose(int hdl, int verbose_level);
	void element_print(void *elt, std::ostream &ost);
	void element_print_quick(void *elt, std::ostream &ost);
	void element_print_latex(void *elt, std::ostream &ost);
	void element_print_verbose(void *elt, std::ostream &ost);
	void element_code_for_make_element(void *elt, int *data);
	void element_print_for_make_element(void *elt, 
			std::ostream &ost);
	void element_print_for_make_element_no_commas(void *elt, 
			std::ostream &ost);
	void element_print_as_permutation(void *elt, 
			std::ostream &ost);
	void element_as_permutation(void *elt, 
		int *perm, int verbose_level);
	void element_print_as_permutation_verbose(void *elt, 
			std::ostream &ost, int verbose_level);
	void element_print_as_permutation_with_offset(void *elt, 
			std::ostream &ost,
		int offset, int f_do_it_anyway_even_for_big_degree, 
		int f_print_cycles_of_length_one, 
		int verbose_level);
	void element_print_as_permutation_with_offset_and_max_cycle_length(
		void *elt, 
		std::ostream &ost, int offset, int max_cycle_length,
		int f_orbit_structure);
	void element_print_image_of_set(void *elt, 
		int size, int *set);
	int element_signum_of_permutation(void *elt);
	void element_write_file_fp(int *Elt, 
		FILE *fp, int verbose_level);
	void element_read_file_fp(int *Elt, 
		FILE *fp, int verbose_level);
	void element_write_file(int *Elt, 
		const char *fname, int verbose_level);
	void element_read_file(int *Elt, 
		const char *fname, int verbose_level);
	void element_write_to_memory_object(int *Elt, 
		memory_object *m, int verbose_level);
	void element_read_from_memory_object(int *Elt, 
		memory_object *m, int verbose_level);
	void element_write_to_file_binary(int *Elt, 
			std::ofstream &fp, int verbose_level);
	void element_read_from_file_binary(int *Elt, 
			std::ifstream &fp, int verbose_level);
	void random_element(sims *S, int *Elt, 
		int verbose_level);


	// in action_projective.cpp:
	strong_generators *set_stabilizer_in_projective_space(
		projective_space *P,
		int *set, int set_size, int &canonical_pt,
		int *canonical_set_or_NULL,
		int f_save_incma_in_and_out,
		const char *save_incma_in_and_out_prefix,
		int verbose_level);
	int reverse_engineer_semilinear_map(
		projective_space *P,
		int *Elt, int *Mtx, int &frobenius,
		int verbose_level);
	// uses the function A->element_image_of

	// in backtrack.C:
	int is_minimal(
		int size, int *set, int &backtrack_level, 
		int verbose_level);
	void make_canonical(
		int size, int *set, 
		int *canonical_set, int *transporter, 
		int &total_backtrack_nodes, 
		int f_get_automorphism_group, sims *Aut,
		int verbose_level);
	int is_minimal_witness(
		int size, int *set, 
		int &backtrack_level, int *witness, 
		int *transporter_witness, 
		int &backtrack_nodes, 
		int f_get_automorphism_group, sims &Aut,
		int verbose_level);
};


// #############################################################################
// action_global.C:
// #############################################################################


action *create_automorphism_group_from_group_table(const char *fname_base, 
	int *Table, int group_order, int *gens, int nb_gens, 
	strong_generators *&Aut_gens, 
	int verbose_level);
void create_linear_group(sims *&S, action *&A, 
	finite_field *F, int m, 
	int f_projective, int f_general, int f_affine, 
	int f_semilinear, int f_special, 
	vector_ge *&nice_gens,
	int verbose_level);
action *create_induced_action_by_restriction(action *A, sims *S, 
	int size, int *set, int f_induce, int verbose_level);
action *create_induced_action_on_sets(action *A, sims *S, 
	int nb_sets, int set_size, int *sets, int f_induce, 
	int verbose_level);
void create_orbits_on_subset_using_restricted_action(
	action *&A_by_restriction, schreier *&Orbits, 
	action *A, sims *S, int size, int *set, 
	int verbose_level);
void create_orbits_on_sets_using_action_on_sets(action *&A_on_sets, 
	schreier *&Orbits, action *A, sims *S, 
	int nb_sets, int set_size, int *sets, int verbose_level);
action *new_action_by_right_multiplication(sims *group_we_act_on, 
	int f_transfer_ownership, int verbose_level);
void action_print_symmetry_group_type(std::ostream &ost, symmetry_group_type a);
int choose_next_base_point_default_method(action *A, int *Elt, 
	int verbose_level);
void make_generators_stabilizer_of_three_components(
	action *A_PGL_n_q, action *A_PGL_k_q, 
	int k, vector_ge *gens, int verbose_level);
void make_generators_stabilizer_of_two_components(
	action *A_PGL_n_q, action *A_PGL_k_q, 
	int k, vector_ge *gens, int verbose_level);
// used in semifield
void generators_to_strong_generators(action *A, 
	int f_target_go, longinteger_object &target_go, 
	vector_ge *gens, strong_generators *&Strong_gens, 
	int verbose_level);
void compute_generators_GL_n_q(int *&Gens, int &nb_gens, 
	int &elt_size, int n, finite_field *F,
	vector_ge *&nice_gens,
	int verbose_level);
void order_of_PGGL_n_q(longinteger_object &go, int n, int q, 
	int f_semilinear);
void set_orthogonal_group_type(int f_siegel, 
	int f_reflection, int f_similarity, int f_semisimilarity);
int get_orthogonal_group_type_f_reflection();
void callback_choose_random_generator_orthogonal(int iteration, 
	int *Elt, void *data, int verbose_level);
	// for use in action_init.C
void test_matrix_group(int k, int q, int f_semilinear, 
	int verbose_level);
void lift_generators(vector_ge *gens_in, vector_ge *&gens_out, 
	action *Aq, subfield_structure *S, int n, int verbose_level);
void retract_generators(vector_ge *gens_in, vector_ge *&gens_out, 
	action *AQ, subfield_structure *S, int n, 
	int verbose_level);
void lift_generators_to_subfield_structure(
	int n, int s, 
	subfield_structure *S, 
	action *Aq, action *AQ, 
	strong_generators *&Strong_gens, 
	int verbose_level);
// O4_model:
void O4_isomorphism_2to4_embedded(action *A4, 
	action *A5, finite_field *Fq, 
	int f_switch, int *mtx2x2_T, int *mtx2x2_S, int *Elt, 
	int verbose_level);
void O5_to_O4(action *A4, action *A5, 
	finite_field *Fq, 
	int *mtx4x4, int *mtx5x5, 
	int verbose_level);
void O4_to_O5(action *A4, action *A5, 
	finite_field *Fq, 
	int *mtx4x4, int *mtx5x5, 
	int verbose_level);
void print_4x4_as_2x2(action *A2, 
	finite_field *Fq, int *mtx4x4);

void projective_space_init_line_action(projective_space *P, 
	action *A_points, action *&A_on_lines, 
	int verbose_level);
void color_distribution_matrix(action *A, 
	int *Elt, int n, uchar *Adj, int *colors, classify *C, 
	int *&Mtx, int verbose_level);
void test_color_distribution(action *A, 
	vector_ge *gens, int n, uchar *Adj, int *colors, 
	int verbose_level);
void color_preserving_subgroup(action *A, 
	int n, uchar *Adj, int *colors, sims *&Subgroup, 
	int verbose_level);
int test_automorphism_group_of_graph_bitvec(action *A, 
	int n, uchar *Adj, int verbose_level);
void compute_conjugacy_classes(sims *S, action *&Aconj, 
	action_by_conjugation *&ABC, schreier *&Sch, 
	strong_generators *&SG, int &nb_classes, 
	int *&class_size, int *&class_rep, 
	int verbose_level);
int group_ring_element_size(action *A, sims *S);
void group_ring_element_create(action *A, sims *S, int *&elt);
void group_ring_element_free(action *A, sims *S, int *elt);
void group_ring_element_print(action *A, sims *S, int *elt);
void group_ring_element_copy(action *A, sims *S, 
	int *elt_from, int *elt_to);
void group_ring_element_zero(action *A, sims *S, 
	int *elt);
void group_ring_element_mult(action *A, sims *S, 
	int *elt1, int *elt2, int *elt3);
void perm_print_cycles_sorted_by_length(std::ostream &ost,
	int degree, int *perm, int verbose_level);
void perm_print_cycles_sorted_by_length_offset(std::ostream &ost,
	int degree, int *perm, int offset, 
	int f_do_it_anyway_even_for_big_degree, 
	int f_print_cycles_of_length_one, int verbose_level);
void do_canonical_form(int n, finite_field *F, 
	int *set, int set_size, int f_semilinear, 
	const char *fname_base, int verbose_level);
void create_action_and_compute_orbits_on_equations(
	action *A, homogeneous_polynomial_domain *HPD, 
	int *The_equations, int nb_equations, strong_generators *gens, 
	action *&A_on_equations, schreier *&Orb, int verbose_level);

// #############################################################################
// action_pointer_table.cpp
// #############################################################################


//! interface to the implementation functions for group actions

class action_pointer_table {

public:

	/** function pointers for group actions */
	int (*ptr_element_image_of)(action &A, int a, void *elt,
		int verbose_level);
	void (*ptr_element_image_of_low_level)(action &A,
		int *input, int *output, void *elt, int verbose_level);
	int (*ptr_element_linear_entry_ij)(action &A,
		void *elt, int i, int j, int verbose_level);
	int (*ptr_element_linear_entry_frobenius)(action &A,
		void *elt, int verbose_level);
	void (*ptr_element_one)(action &A, void *elt, int verbose_level);
	int (*ptr_element_is_one)(action &A, void *elt, int verbose_level);
	void (*ptr_element_unpack)(action &A, void *elt,
		void *Elt, int verbose_level);
	void (*ptr_element_pack)(action &A, void *Elt,
		void *elt, int verbose_level);
	void (*ptr_element_retrieve)(action &A, int hdl,
		void *elt, int verbose_level);
	int (*ptr_element_store)(action &A, void *elt,
		int verbose_level);
	void (*ptr_element_mult)(action &A,
		void *a, void *b, void *ab, int verbose_level);
	void (*ptr_element_invert)(action &A,
		void *a, void *av, int verbose_level);
	void (*ptr_element_transpose)(action &A,
		void *a, void *at, int verbose_level);
	void (*ptr_element_move)(action &A,
		void *a, void *b, int verbose_level);
	void (*ptr_element_dispose)(action &A,
		int hdl, int verbose_level);
	void (*ptr_element_print)(action &A,
		void *elt, std::ostream &ost);
	void (*ptr_element_print_quick)(action &A,
		void *elt, std::ostream &ost);
	void (*ptr_element_print_latex)(action &A,
		void *elt, std::ostream &ost);
	void (*ptr_element_print_verbose)(action &A,
		void *elt, std::ostream &ost);
	void (*ptr_print_point)(action &A, int i, std::ostream &ost);
	void (*ptr_element_code_for_make_element)(action &A,
		void *elt, int *data);
	void (*ptr_element_print_for_make_element)(action &A,
		void *elt, std::ostream &ost);
	void (*ptr_element_print_for_make_element_no_commas)(action &A,
		void *elt, std::ostream &ost);

	/** counters for how often a function has been called */
	int nb_times_image_of_called;
	int nb_times_image_of_low_level_called;
	int nb_times_unpack_called;
	int nb_times_pack_called;
	int nb_times_retrieve_called;
	int nb_times_store_called;
	int nb_times_mult_called;
	int nb_times_invert_called;

	action_pointer_table();
	~action_pointer_table();
	void null_function_pointers();
	void init_function_pointers_matrix_group();
	void init_function_pointers_wreath_product_group();
	void init_function_pointers_direct_product_group();
	void init_function_pointers_permutation_group();
	void init_function_pointers_induced_action();
};


// #############################################################################
// interface.C
// #############################################################################

int induced_action_element_image_of(action &A, int a, 
	void *elt, int verbose_level);
void induced_action_element_image_of_low_level(action &A, 
	int *input, int *output, void *elt, int verbose_level);
int induced_action_element_linear_entry_ij(action &A, 
	void *elt, int i, int j, int verbose_level);
int induced_action_element_linear_entry_frobenius(action &A, 
	void *elt, int verbose_level);
void induced_action_element_one(action &A, 
	void *elt, int verbose_level);
int induced_action_element_is_one(action &A, 
	void *elt, int verbose_level);
void induced_action_element_unpack(action &A, 
	void *elt, void *Elt, int verbose_level);
void induced_action_element_pack(action &A, 
	void *Elt, void *elt, int verbose_level);
void induced_action_element_retrieve(action &A, 
	int hdl, void *elt, int verbose_level);
int induced_action_element_store(action &A, 
	void *elt, int verbose_level);
void induced_action_element_mult(action &A, 
	void *a, void *b, void *ab, int verbose_level);
void induced_action_element_invert(action &A, 
	void *a, void *av, int verbose_level);
void induced_action_element_transpose(action &A, 
	void *a, void *at, int verbose_level);
void induced_action_element_move(action &A, 
	void *a, void *b, int verbose_level);
void induced_action_element_dispose(action &A, 
	int hdl, int verbose_level);
void induced_action_element_print(action &A, 
	void *elt, std::ostream &ost);
void induced_action_element_print_quick(action &A, 
	void *elt, std::ostream &ost);
void induced_action_element_print_latex(action &A, 
	void *elt, std::ostream &ost);
void induced_action_element_print_verbose(action &A, 
	void *elt, std::ostream &ost);
void induced_action_element_code_for_make_element(action &A, 
	void *elt, int *data);
void induced_action_element_print_for_make_element(action &A, 
	void *elt, std::ostream &ost);
void induced_action_element_print_for_make_element_no_commas(
	action &A, void *elt, std::ostream &ost);
void induced_action_print_point(action &A, int a, std::ostream &ost);


// #############################################################################
// interface_direct_product_group.C
// #############################################################################


int direct_product_group_element_image_of(action &A, int a,
	void *elt, int verbose_level);
void direct_product_group_element_image_of_low_level(action &A,
	int *input, int *output, void *elt, int verbose_level);
int direct_product_group_element_linear_entry_ij(action &A,
	void *elt, int i, int j, int verbose_level);
int direct_product_group_element_linear_entry_frobenius(action &A,
	void *elt, int verbose_level);
void direct_product_group_element_one(action &A,
	void *elt, int verbose_level);
int direct_product_group_element_is_one(action &A,
	void *elt, int verbose_level);
void direct_product_group_element_unpack(action &A,
	void *elt, void *Elt, int verbose_level);
void direct_product_group_element_pack(action &A,
	void *Elt, void *elt, int verbose_level);
void direct_product_group_element_retrieve(action &A,
	int hdl, void *elt, int verbose_level);
int direct_product_group_element_store(action &A,
	void *elt, int verbose_level);
void direct_product_group_element_mult(action &A,
	void *a, void *b, void *ab, int verbose_level);
void direct_product_group_element_invert(action &A,
	void *a, void *av, int verbose_level);
void direct_product_group_element_transpose(action &A,
	void *a, void *at, int verbose_level);
void direct_product_group_element_move(action &A,
	void *a, void *b, int verbose_level);
void direct_product_group_element_dispose(action &A,
	int hdl, int verbose_level);
void direct_product_group_element_print(action &A,
	void *elt, std::ostream &ost);
void direct_product_group_element_code_for_make_element(
	action &A, void *elt, int *data);
void direct_product_group_element_print_for_make_element(
	action &A, void *elt, std::ostream &ost);
void direct_product_group_element_print_for_make_element_no_commas(
	action &A, void *elt, std::ostream &ost);
void direct_product_group_element_print_quick(action &A,
	void *elt, std::ostream &ost);
void direct_product_group_element_print_latex(action &A,
	void *elt, std::ostream &ost);
void direct_product_group_element_print_as_permutation(
	action &A, void *elt, std::ostream &ost);
void direct_product_group_element_print_verbose(action &A,
	void *elt, std::ostream &ost);
void direct_product_group_print_point(action &A,
	int a, std::ostream &ost);


// #############################################################################
// interface_matrix_group.C
// #############################################################################


int matrix_group_element_image_of(action &A, int a, 
	void *elt, int verbose_level);
void matrix_group_element_image_of_low_level(action &A, 
	int *input, int *output, void *elt, int verbose_level);
int matrix_group_element_linear_entry_ij(action &A, 
	void *elt, int i, int j, int verbose_level);
int matrix_group_element_linear_entry_frobenius(action &A, 
	void *elt, int verbose_level);
void matrix_group_element_one(action &A, 
	void *elt, int verbose_level);
int matrix_group_element_is_one(action &A, 
	void *elt, int verbose_level);
void matrix_group_element_unpack(action &A, 
	void *elt, void *Elt, int verbose_level);
void matrix_group_element_pack(action &A, 
	void *Elt, void *elt, int verbose_level);
void matrix_group_element_retrieve(action &A, 
	int hdl, void *elt, int verbose_level);
int matrix_group_element_store(action &A, 
	void *elt, int verbose_level);
void matrix_group_element_mult(action &A, 
	void *a, void *b, void *ab, int verbose_level);
void matrix_group_element_invert(action &A, 
	void *a, void *av, int verbose_level);
void matrix_group_element_transpose(action &A, 
	void *a, void *at, int verbose_level);
void matrix_group_element_move(action &A, 
	void *a, void *b, int verbose_level);
void matrix_group_element_dispose(action &A, 
	int hdl, int verbose_level);
void matrix_group_element_print(action &A, 
	void *elt, std::ostream &ost);
void matrix_group_element_code_for_make_element(
	action &A, void *elt, int *data);
void matrix_group_element_print_for_make_element(
	action &A, void *elt, std::ostream &ost);
void matrix_group_element_print_for_make_element_no_commas(
	action &A, void *elt, std::ostream &ost);
void matrix_group_element_print_quick(action &A, 
	void *elt, std::ostream &ost);
void matrix_group_element_print_latex(action &A, 
	void *elt, std::ostream &ost);
void matrix_group_element_print_as_permutation(
	action &A, void *elt, std::ostream &ost);
void matrix_group_element_print_verbose(action &A, 
	void *elt, std::ostream &ost);
void matrix_group_print_point(action &A, 
	int a, std::ostream &ost);


// #############################################################################
// interface_perm_group.C
// #############################################################################

int perm_group_element_image_of(action &A, int a, 
	void *elt, int verbose_level);
void perm_group_element_one(action &A, 
	void *elt, int verbose_level);
int perm_group_element_is_one(action &A, 
	void *elt, int verbose_level);
void perm_group_element_unpack(action &A, 
	void *elt, void *Elt, int verbose_level);
void perm_group_element_pack(action &A, 
	void *Elt, void *elt, int verbose_level);
void perm_group_element_retrieve(action &A, 
	int hdl, void *elt, int verbose_level);
int perm_group_element_store(action &A, 
	void *elt, int verbose_level);
void perm_group_element_mult(action &A, 
	void *a, void *b, void *ab, int verbose_level);
void perm_group_element_invert(action &A, 
	void *a, void *av, int verbose_level);
void perm_group_element_move(action &A, 
	void *a, void *b, int verbose_level);
void perm_group_element_dispose(action &A, 
	int hdl, int verbose_level);
void perm_group_element_print(action &A, 
	void *elt, std::ostream &ost);
void perm_group_element_print_latex(action &A, 
	void *elt, std::ostream &ost);
void perm_group_element_print_verbose(action &A, 
	void *elt, std::ostream &ost);
void perm_group_element_code_for_make_element(action &A, 
	void *elt, int *data);
void perm_group_element_print_for_make_element(action &A, 
	void *elt, std::ostream &ost);
void perm_group_element_print_for_make_element_no_commas(
	action &A, void *elt, std::ostream &ost);
void perm_group_print_point(action &A, int a, std::ostream &ost);

// #############################################################################
// interface_wreath_product_group.C
// #############################################################################


int wreath_product_group_element_image_of(action &A, int a,
	void *elt, int verbose_level);
void wreath_product_group_element_image_of_low_level(action &A,
	int *input, int *output, void *elt, int verbose_level);
int wreath_product_group_element_linear_entry_ij(action &A,
	void *elt, int i, int j, int verbose_level);
int wreath_product_group_element_linear_entry_frobenius(action &A,
	void *elt, int verbose_level);
void wreath_product_group_element_one(action &A,
	void *elt, int verbose_level);
int wreath_product_group_element_is_one(action &A,
	void *elt, int verbose_level);
void wreath_product_group_element_unpack(action &A,
	void *elt, void *Elt, int verbose_level);
void wreath_product_group_element_pack(action &A,
	void *Elt, void *elt, int verbose_level);
void wreath_product_group_element_retrieve(action &A,
	int hdl, void *elt, int verbose_level);
int wreath_product_group_element_store(action &A,
	void *elt, int verbose_level);
void wreath_product_group_element_mult(action &A,
	void *a, void *b, void *ab, int verbose_level);
void wreath_product_group_element_invert(action &A,
	void *a, void *av, int verbose_level);
void wreath_product_group_element_transpose(action &A,
	void *a, void *at, int verbose_level);
void wreath_product_group_element_move(action &A,
	void *a, void *b, int verbose_level);
void wreath_product_group_element_dispose(action &A,
	int hdl, int verbose_level);
void wreath_product_group_element_print(action &A,
	void *elt, std::ostream &ost);
void wreath_product_group_element_code_for_make_element(
	action &A, void *elt, int *data);
void wreath_product_group_element_print_for_make_element(
	action &A, void *elt, std::ostream &ost);
void wreath_product_group_element_print_for_make_element_no_commas(
	action &A, void *elt, std::ostream &ost);
void wreath_product_group_element_print_quick(action &A,
	void *elt, std::ostream &ost);
void wreath_product_group_element_print_latex(action &A,
	void *elt, std::ostream &ost);
void wreath_product_group_element_print_as_permutation(
	action &A, void *elt, std::ostream &ost);
void wreath_product_group_element_print_verbose(action &A,
	void *elt, std::ostream &ost);
void wreath_product_group_print_point(action &A,
	int a, std::ostream &ost);

// #############################################################################
// nauty_interface.cpp:
// #############################################################################


class nauty_interface {
public:
	nauty_interface();
	~nauty_interface();
	action *create_automorphism_group_of_colored_graph_object(
		colored_graph *CG, int verbose_level);
	action *create_automorphism_group_of_colored_graph(
		int n, int f_bitvec, uchar *Adj_bitvec, int *Adj,
		int *vertex_colors,
		int verbose_level);
	action *create_automorphism_group_of_graph_bitvec(
		int n, uchar *Adj_bitvec,
		int verbose_level);
	action *create_automorphism_group_of_graph_with_partition_and_labeling(
		int n,
		int f_bitvector, uchar *Adj_bitvec, int *Adj,
		int nb_parts, int *parts,
		int *labeling,
		int verbose_level);
	void create_incidence_matrix_of_graph(int *Adj, int n,
		int *&M, int &nb_rows, int &nb_cols, int verbose_level);
	action *create_automorphism_group_of_graph(int *Adj,
		int n, int verbose_level);
	action *create_automorphism_group_and_canonical_labeling_of_graph(
		int *Adj, int n, int *labeling, int verbose_level);
	// labeling[n]
	action *create_automorphism_group_of_block_system(
		int nb_points, int nb_blocks, int block_size, int *Blocks,
		int verbose_level);
	action *create_automorphism_group_of_collection_of_two_block_systems(
		int nb_points,
		int nb_blocks1, int block_size1, int *Blocks1,
		int nb_blocks2, int block_size2, int *Blocks2,
		int verbose_level);
	action *create_automorphism_group_of_incidence_matrix(
		int m, int n, int *Mtx,
		int verbose_level);
	action *create_automorphism_group_of_incidence_structure(
		incidence_structure *Inc,
		int verbose_level);
	action *create_automorphism_group_of_incidence_structure_low_level(
		int m, int n, int nb_inc, int *X,
		int verbose_level);
	action *create_automorphism_group_of_incidence_structure_with_partition(
		int m, int n, int nb_inc, int *X, int *partition,
		int verbose_level);
	void test_self_dual_self_polar(int input_no,
		int m, int n, int nb_inc, int *X,
		int &f_self_dual, int &f_self_polar,
		int verbose_level);
	void do_self_dual_self_polar(int input_no,
		int m, int n, int nb_inc, int *X,
		int &f_self_dual, int &f_self_polar,
		int verbose_level);
	void add_configuration_graph(std::ofstream &g,
		int m, int n, int nb_inc, int *X, int f_first,
		int verbose_level);

};

}}

