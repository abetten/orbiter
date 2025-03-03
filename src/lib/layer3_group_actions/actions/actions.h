// group_actions.h
//
// Anton Betten
//
// moved here from action.h: July 28, 2018
// based on action.h which was started:  August 13, 2005



#ifndef ORBITER_SRC_LIB_GROUP_ACTIONS_ACTIONS_ACTIONS_H_
#define ORBITER_SRC_LIB_GROUP_ACTIONS_ACTIONS_ACTIONS_H_



namespace orbiter {

namespace layer3_group_actions {


namespace actions {


// #############################################################################
// action.cpp
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
 * is small. For the basic action of projective groups, this condition
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


	/** whether the group has strong generators */
	int f_has_strong_generators;

	/** strong generating set for the group */
	groups::strong_generators *Strong_gens;

	/** the size of the set we act on */
	long int degree;


	/** whether the action is linear (including semilinear) */
	int f_is_linear;
		// matrix_group_t, 
		// action_on_wedge_product_t, 
		// action_by_representation_t
	
	/** the dimension if we are linear */
	int dimension;


	/** the number of int needed to store one group element */
	int elt_size_in_int;

	/** the number of char needed
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
	groups::sims *Sims;
	
	int f_has_kernel;
	/** kernel of the action */
	groups::sims *Kernel;
	
	int f_group_order_is_small;
	

	int f_has_stabilizer_chain;

	stabilizer_chain_base_data *Stabilizer_chain;

	known_groups *Known_groups;

	induced_action *Induced_action;

	group_element *Group_element;


	action_pointer_table *ptr;


	

	/** a label for the group */
	std::string label;

	/** a label for the group for latex */
	std::string label_tex;



	// action.cpp
	action();
	~action();
	void null();
	void freeself();
	
	int f_has_base();
	int base_len();
	void set_base_len(
			int base_len);
	long int &base_i(
			int i);
	long int *&get_base();
	int &transversal_length_i(
			int i);
	int *&get_transversal_length();
	long int &orbit_ij(
			int i, int j);
	long int &orbit_inv_ij(
			int i, int j);

	
	void map_a_set_based_on_hdl(
			long int *set,
			long int *image_set,
			int n, action *A_base, int hdl,
			int verbose_level);
	void print_all_elements();

	void init_sims_only(
			groups::sims *G, int verbose_level);
	void compute_strong_generators_from_sims(
			int verbose_level);
	void compute_all_point_orbits(
			groups::schreier &S,
			data_structures_groups::vector_ge &gens,
			int verbose_level);
	
	/** the index of the first base point which is moved */
	int depth_in_stab_chain(
			int *Elt);


		/** all strong generators that
		 * leave base points 0,..., depth - 1 fix */
	void strong_generators_at_depth(
			int depth,
			data_structures_groups::vector_ge &gen,
			int verbose_level);
	void compute_point_stabilizer_chain(
			data_structures_groups::vector_ge &gen,
			groups::sims *S, int *sequence, int len,
		int verbose_level);
	void compute_stabilizer_orbits(
			other::data_structures::partitionstack *&Staborbits,
			int verbose_level);
	void find_strong_generators_at_level(
			int base_len,
		long int *the_base, int level,
		data_structures_groups::vector_ge &gens,
		data_structures_groups::vector_ge &subset_of_gens,
		int verbose_level);
	void group_order(
			algebra::ring_theory::longinteger_object &go);
	long int group_order_lint();
	std::string group_order_as_string();


	int matrix_group_dimension();
	algebra::field_theory::finite_field *matrix_group_finite_field();
	int is_semilinear_matrix_group();
	int is_projective();
	int is_affine();
	int is_general_linear();
	int is_matrix_group();
	algebra::basic_algebra::matrix_group *get_matrix_group();




	// action_indexing_cosets.cpp
	void coset_unrank(
			groups::sims *G,
			groups::sims *U, long int rank,
		int *Elt, int verbose_level);
	long int coset_rank(
			groups::sims *G, groups::sims *U,
		int *Elt, int verbose_level);
		// used in generator::coset_unrank and generator::coset_rank
		// which in turn are used by 
		// generator::orbit_element_unrank and 
		// generator::orbit_element_rank

	// action_init.cpp

	void init_group_from_generators(
			int *group_generator_data,
		int group_generator_size,
		int f_group_order_target,
		const char *group_order_target,
		data_structures_groups::vector_ge *gens,
		groups::strong_generators *&Strong_gens,
		int verbose_level);
	void init_group_from_generators_by_base_images(
			groups::sims *S,
			int *group_generator_data, int group_generator_size,
			int f_group_order_target,
			const char *group_order_target,
			data_structures_groups::vector_ge *gens,
			groups::strong_generators *&Strong_gens_out,
			int verbose_level);
	void build_up_automorphism_group_from_aut_data(
			int nb_auts,
		int *aut_data,
		groups::sims &S, int verbose_level);


	// the following are various entry points
	// for the randomized schreier sims algorithm:

	groups::sims *create_sims_from_generators_with_target_group_order_factorized(
			data_structures_groups::vector_ge *gens,
			int *tl, int len, int verbose_level);
	groups::sims *create_sims_from_generators_with_target_group_order_lint(
			data_structures_groups::vector_ge *gens,
			long int target_go, int verbose_level);
	groups::sims *create_sims_from_generators_with_target_group_order(
			data_structures_groups::vector_ge *gens,
			algebra::ring_theory::longinteger_object &target_go,
		int verbose_level);
	groups::sims *create_sims_from_generators_without_target_group_order(
			data_structures_groups::vector_ge *gens,
			int verbose_level);
	groups::sims *create_sims_from_single_generator_without_target_group_order(
		int *Elt, int verbose_level);

	groups::sims *create_sims_from_generators_randomized(
			data_structures_groups::vector_ge *gens,
			int f_target_go, algebra::ring_theory::longinteger_object &target_go,
		int verbose_level);
	// uses groups::schreier_sims

	groups::sims *create_sims_for_centralizer_of_matrix(
			int *Mtx, int verbose_level);

	
	// action_induce.cpp
	int least_moved_point_at_level(
			groups::sims *old_Sims,
			int level,
		int verbose_level);
	void lex_least_base_in_place(
			groups::sims *old_Sims,
			int verbose_level);
	void lex_least_base(
			action *old_action, int verbose_level);
	int test_if_lex_least_base(
			int verbose_level);
	void base_change_in_place(
			int size, long int *set, groups::sims *old_Sims,
			int verbose_level);
	int choose_next_base_point_default_method(
			int *Elt, int verbose_level);
	void generators_to_strong_generators(
		int f_target_go,
		algebra::ring_theory::longinteger_object &target_go,
		data_structures_groups::vector_ge *gens,
		groups::strong_generators *&Strong_gens,
		int verbose_level);


	// action_io.cpp:
	void report(
			std::ostream &ost, int f_sims, groups::sims *S,
			int f_strong_gens, groups::strong_generators *SG,
			other::graphics::layered_graph_draw_options *LG_Draw_options,
			int verbose_level);
	void report_group_name_and_degree(
			std::ostream &ost,
			//other::graphics::layered_graph_draw_options *LG_Draw_options,
			int verbose_level);
	void report_type_of_action(
			std::ostream &ost,
			//other::graphics::layered_graph_draw_options *O,
			int verbose_level);
	void report_what_we_act_on(
			std::ostream &ost,
			//other::graphics::layered_graph_draw_options *O,
			int verbose_level);


	void list_elements_as_permutations_vertically(
			data_structures_groups::vector_ge *gens,
			std::ostream &ost);
	void print_symmetry_group_type(
			std::ostream &ost);
	std::string stringify_subaction_labels();
	void print_info();
	void report_basic_orbits(
			std::ostream &ost);
	void print_base();
	std::string stringify_base();
	void print_base(
			std::ostream &ost);
	void print_bare_base(
			std::ofstream &ost);
	void latex_all_points(
			std::ostream &ost);
	void latex_point_set(
			std::ostream &ost,
			long int *set, int sz, int verbose_level);
	void print_group_order(
			std::ostream &ost);
	void print_group_order_long(
			std::ostream &ost);
	void print_vector(
			data_structures_groups::vector_ge &v);
	void print_vector_as_permutation(
			data_structures_groups::vector_ge &v);
	void write_set_of_elements_latex_file(
			std::string &fname,
			std::string &title, int *Elt, int nb_elts);
	void export_to_orbiter(
			std::string &fname, std::string &label,
			groups::strong_generators *SG, int verbose_level);
	void export_to_orbiter_as_bsgs(
			std::string &fname,
			std::string &label,
			std::string &label_tex,
			groups::strong_generators *SG, int verbose_level);
	void print_one_element_tex(
			std::ostream &ost,
			int *Elt, int f_with_permutation);



	// in backtrack.cpp
	int is_minimal(
		int size, long int *set, groups::sims *old_Sims,
		int &backtrack_level,
		int verbose_level);
	int is_minimal_witness(
		int size, long int *set, groups::sims *old_Sims,
		int &backtrack_level, long int *witness,
		int *transporter_witness, 
		long int &backtrack_nodes,
		int f_get_automorphism_group, groups::sims &Aut,
		int verbose_level);
};


// #############################################################################
// action_global.cpp
// #############################################################################

//! global functions related to group actions

class action_global {
public:

	action_global();
	~action_global();

	void action_print_symmetry_group_type(
			std::ostream &ost,
			symmetry_group_type a);
	std::string stringify_symmetry_group_type(
			symmetry_group_type a);
	void get_symmetry_group_type_text(
			std::string &txt, std::string &tex,
			symmetry_group_type a);
	void automorphism_group_as_permutation_group(
			other::l1_interfaces::nauty_output *NO,
			actions::action *&A_perm,
			int verbose_level);
	void reverse_engineer_linear_group_from_permutation_group(
			actions::action *A_linear,
			geometry::projective_geometry::projective_space *P,
			groups::strong_generators *&SG,
			actions::action *&A_perm,
			other::l1_interfaces::nauty_output *NO,
			int verbose_level);
	void make_generators_stabilizer_of_three_components(
		action *A_PGL_n_q, action *A_PGL_k_q,
		int k, data_structures_groups::vector_ge *gens,
		int verbose_level);
	void make_generators_stabilizer_of_two_components(
		action *A_PGL_n_q, action *A_PGL_k_q,
		int k, data_structures_groups::vector_ge *gens,
		int verbose_level);
	// used in semifield
	void compute_generators_GL_n_q(
			int *&Gens, int &nb_gens,
		int &elt_size, int n,
		algebra::field_theory::finite_field *F,
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level);
	void set_orthogonal_group_type(
			int f_siegel,
		int f_reflection, int f_similarity,
		int f_semisimilarity);
	int get_orthogonal_group_type_f_reflection();
	void lift_generators(
			data_structures_groups::vector_ge *gens_in,
			data_structures_groups::vector_ge *&gens_out,
		action *Aq,
		algebra::field_theory::subfield_structure *S, int n,
		int verbose_level);
	void retract_generators(
			data_structures_groups::vector_ge *gens_in,
			data_structures_groups::vector_ge *&gens_out,
		action *AQ,
		algebra::field_theory::subfield_structure *S, int n,
		int verbose_level);
	void lift_generators_to_subfield_structure(
		int n, int s,
		algebra::field_theory::subfield_structure *S,
		action *Aq, action *AQ,
		groups::strong_generators *&Strong_gens,
		int verbose_level);
	void perm_print_cycles_sorted_by_length(
			std::ostream &ost,
		int degree, int *perm, int verbose_level);
	void perm_print_cycles_sorted_by_length_offset(
			std::ostream &ost,
		int degree, int *perm, int offset,
		int f_do_it_anyway_even_for_big_degree,
		int f_print_cycles_of_length_one, int verbose_level);
	action *init_direct_product_group_and_restrict(
			algebra::basic_algebra::matrix_group *M1,
			algebra::basic_algebra::matrix_group *M2,
			int verbose_level);
	action *init_direct_product_group(
			algebra::basic_algebra::matrix_group *M1,
			algebra::basic_algebra::matrix_group *M2,
			int verbose_level);
	action *init_polarity_extension_group_and_restrict(
			actions::action *A,
			geometry::projective_geometry::projective_space *P,
			geometry::projective_geometry::polarity *Polarity,
			int f_on_middle_layer_grassmannian,
			int f_on_points_and_hyperplanes,
			int verbose_level);
	action *init_polarity_extension_group(
			actions::action *A,
			geometry::projective_geometry::projective_space *P,
			geometry::projective_geometry::polarity *Polarity,
			int verbose_level);
	action *init_subgroup_from_strong_generators(
			actions::action *A,
			groups::strong_generators *Strong_gens,
			int verbose_level);
	// shortens the base
	void compute_sims(
			action *A,
			int verbose_level);
	void orbits_on_equations(
			action *A,
			algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		int *The_equations,
		int nb_equations, groups::strong_generators *gens,
		actions::action *&A_on_equations,
		groups::schreier *&Orb,
		int verbose_level);
	groups::strong_generators *set_stabilizer_in_projective_space(
			action *A_linear,
			geometry::projective_geometry::projective_space *P,
		long int *set, int set_size,
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		int verbose_level);
	// assuming we are in a linear action.
	void stabilizer_of_dual_hyperoval_representative(
			action *A,
			int k, int n, int no,
			data_structures_groups::vector_ge *&gens,
			std::string &stab_order,
			int verbose_level);
	void stabilizer_of_spread_representative(
			action *A,
			int q, int k, int no,
			data_structures_groups::vector_ge *&gens,
			std::string &stab_order,
			int verbose_level);
	void stabilizer_of_quartic_curve_representative(
			action *A,
			int q, int no,
			data_structures_groups::vector_ge *&gens,
			std::string &stab_order,
			int verbose_level);
	void perform_tests(
			action *A,
			groups::strong_generators *SG,
			int verbose_level);
	void apply_based_on_text(
			action *A,
			std::string &input_text,
			std::string &input_group_element,
			int verbose_level);
	void multiply_based_on_text(
			action *A,
			std::string &data_A,
			std::string &data_B, int verbose_level);
	void inverse_based_on_text(
			action *A,
			std::string &data_A, int verbose_level);
	void consecutive_powers_based_on_text(
			action *A,
			std::string &data_A,
			std::string &exponent_text, int verbose_level);
	void raise_to_the_power_based_on_text(
			action *A,
			std::string &data_A,
			std::string &exponent_text, int verbose_level);
	void compute_orbit_of_point(
			actions::action *A,
			data_structures_groups::vector_ge &strong_generators,
			int pt, int *orbit, int &len, int verbose_level);
	void compute_orbit_of_point_generators_by_handle(
			actions::action *A,
			int nb_gen,
		int *gen_handle, int pt, int *orbit, int &len,
		int verbose_level);
	int least_image_of_point(
			actions::action *A,
			data_structures_groups::vector_ge &strong_generators,
		int pt, int *transporter, int verbose_level);
	int least_image_of_point_generators_by_handle(
			actions::action *A,
		std::vector<int> &gen_handle,
		int pt, int *transporter, int verbose_level);
	int least_image_of_point_generators_by_handle(
			actions::action *A,
		int nb_gen, int *gen_handle,
		int pt, int *transporter, int verbose_level);
	void all_point_orbits(
			actions::action *A,
			groups::schreier &Schreier, int verbose_level);
	void get_orbits_on_points_as_characteristic_vector(
			actions::action *A,
			int *&orbit_no,
			int verbose_level);
	void all_point_orbits_from_generators(
			actions::action *A,
			groups::schreier &Schreier,
			groups::strong_generators *SG,
			int verbose_level);
	void all_point_orbits_from_single_generator(
			actions::action *A,
			groups::schreier &Schreier,
			int *Elt,
			int verbose_level);
	void induce(
			action *old_action,
			action *new_action,
			groups::sims *old_G,
		int base_of_choice_len, long int *base_of_choice,
		int verbose_level);
	// after this procedure, new_action will have
	// a sims for the group and the kernel
	// it will also have strong generators

	// the old_action may not have a stabilizer chain,
	// but it's subaction does.
	void induced_action_override_sims(
		action *old_action, action *new_action, groups::sims *old_G,
		int verbose_level);
	void make_canonical(
			action *A, groups::sims *Sims,
			int size, long int *set,
		long int *canonical_set, int *transporter,
		long int &total_backtrack_nodes,
		int f_get_automorphism_group, groups::sims *Aut,
		int verbose_level);
	void make_element_which_moves_a_line_in_PG3q(
			action *A,
			geometry::projective_geometry::projective_space_of_dimension_three *P3,
			long int line_rk, int *Elt,
			int verbose_level);
	void orthogonal_group_random_generator(
			action *A,
			geometry::orthogonal_geometry::orthogonal *O,
			algebra::basic_algebra::matrix_group *M,
		int f_siegel,
		int f_reflection,
		int f_similarity,
		int f_semisimilarity,
		int *Elt, int verbose_level);
	void init_base(
			actions::action *A,
			algebra::basic_algebra::matrix_group *M,
			int verbose_level);
	void init_base_projective(
			actions::action *A,
			algebra::basic_algebra::matrix_group *M,
			int verbose_level);
	void init_base_affine(
			actions::action *A,
			algebra::basic_algebra::matrix_group *M,
			int verbose_level);
	void init_base_general_linear(
			actions::action *A,
			algebra::basic_algebra::matrix_group *M,
			int verbose_level);
	void substitute_semilinear(
			action *A,
			algebra::ring_theory::homogeneous_polynomial_domain *HPD,
			int *Elt,
			int *input, int *output,
			int verbose_level);
	void test_if_two_actions_agree_vector(
			action *A1, action *A2,
			data_structures_groups::vector_ge *gens1,
			data_structures_groups::vector_ge *gens2,
			int verbose_level);
	void test_if_two_actions_agree(
			action *A1, action *A2, int *Elt1, int *Elt2,
			int verbose_level);
	void reverse_engineer_semilinear_group(
			action *A_perm, action *A_linear,
			geometry::projective_geometry::projective_space *P,
			data_structures_groups::vector_ge *gens_in,
			data_structures_groups::vector_ge *&gens_out,
			int verbose_level);
	groups::strong_generators *scan_generators(
			action *A0,
			std::string &gens_text,
			std::string &group_order,
			int verbose_level);
	void multiply_all_elements_in_lex_order(
			groups::sims *Sims, int *Elt, int verbose_level);
	void get_generators_from_ascii_coding(
			action *A,
			std::string &ascii_coding,
			data_structures_groups::vector_ge *&gens,
			int *&tl, int verbose_level);
	void lexorder_test(
			action *A,
			long int *set, int set_sz, int &set_sz_after_test,
			data_structures_groups::vector_ge *gens,
			int max_starter,
		int verbose_level);
	void compute_orbits_on_points(
			action *A,
			groups::schreier *&Sch,
			data_structures_groups::vector_ge *gens,
			int verbose_level);

	void point_stabilizer_any_point(
			action *A,
			int &pt,
			groups::schreier *&Sch, groups::sims *&Stab,
			groups::strong_generators *&stab_gens,
		int verbose_level);
	void point_stabilizer_any_point_with_given_group(
			action *A,
			groups::strong_generators *input_gens,
		int &pt,
		groups::schreier *&Sch, groups::sims *&Stab,
		groups::strong_generators *&stab_gens,
		int verbose_level);
	void move_a_to_b_and_stabilizer_of_b(
			actions::action *A_base,
			actions::action *A2,
			groups::strong_generators *SG,
			int a, int b,
			int *&transporter_a_b,
			groups::strong_generators *&Stab_b,
			int verbose_level);
	void rational_normal_form(
			actions::action *A,
			std::string &element_given,
			int verbose_level);
	void find_conjugating_element(
			actions::action *A,
			std::string &element_from,
			std::string &element_to,
			int verbose_level);
	void read_orbit_rep_and_candidates_from_files_and_process(
			action *A,
			std::string &prefix,
		int level, int orbit_at_level, int level_of_candidates_file,
		void (*early_test_func_callback)(long int *S, int len,
			long int *candidates, int nb_candidates,
			long int *good_candidates, int &nb_good_candidates,
			void *data, int verbose_level),
		void *early_test_func_callback_data,
		long int *&starter,
		int &starter_sz,
		groups::sims *&Stab,
		groups::strong_generators *&Strong_gens,
		long int *&candidates,
		int &nb_candidates,
		int &nb_cases,
		int verbose_level);
	void read_orbit_rep_and_candidates_from_files(
			action *A,
			std::string &prefix,
		int level, int orbit_at_level, int level_of_candidates_file,
		long int *&starter,
		int &starter_sz,
		groups::sims *&Stab,
		groups::strong_generators *&Strong_gens,
		long int *&candidates,
		int &nb_candidates,
		int &nb_cases,
		int verbose_level);
	void read_representatives(
			std::string &fname,
		int *&Reps, int &nb_reps, int &size, int verbose_level);
	void read_representatives_and_strong_generators(
			std::string &fname,
		int *&Reps,
		char **&Aut_ascii, int &nb_reps,
		int &size, int verbose_level);
	void read_file_and_print_representatives(
			action *A,
			std::string &fname,
		int f_print_stabilizer_generators, int verbose_level);
	void read_set_and_stabilizer(
			action *A,
			std::string &fname,
		int no, long int *&set, int &set_sz, groups::sims *&stab,
		groups::strong_generators *&Strong_gens,
		int &nb_cases,
		int verbose_level);
	// reads an orbiter data file
	other::data_structures::set_of_sets *set_of_sets_copy_and_apply(
			action *A,
			int *Elt,
			other::data_structures::set_of_sets *old_one,
		int verbose_level);
	actions::action *create_action_on_k_subspaces(
			actions::action *A_previous,
			int k,
			int verbose_level);
	void report_strong_generators(
			std::ostream &ost,
			//other::graphics::layered_graph_draw_options *LG_Draw_options,
			groups::strong_generators *SG,
			action *A,
			int verbose_level);
	void report_strong_generators_GAP(
			std::ostream &ost,
			groups::strong_generators *SG,
			action *A,
			int verbose_level);
	void report_strong_generators_fining(
			std::ostream &ost,
			groups::strong_generators *SG,
			action *A,
			int verbose_level);
	void report_strong_generators_magma(
			std::ostream &ost,
			groups::strong_generators *SG,
			action *A,
			int verbose_level);
	void report_strong_generators_orbiter(
			std::ostream &ost,
			groups::strong_generators *SG,
			action *A,
			int verbose_level);
	void report(
			std::ostream &ost,
			std::string &label,
			std::string &label_tex,
			actions::action *A,
			groups::strong_generators *Strong_gens,
			int f_sylow, int f_group_table,
			other::graphics::layered_graph_draw_options *LG_Draw_options,
			int verbose_level);
	void report_groups_and_normalizers(
			action *A,
			std::ostream &ost,
			int nb_subgroups,
			groups::strong_generators *H_gens,
			groups::strong_generators *N_gens,
			int verbose_level);
	void compute_projectivity_subgroup(
			action *A,
			groups::strong_generators *&projectivity_gens,
			groups::strong_generators *Aut_gens,
			int verbose_level);
	void all_elements(
			action *A,
			data_structures_groups::vector_ge *&vec,
			int verbose_level);
	// unused code
	void all_elements_save_csv(
			action *A,
			std::string &fname, int verbose_level);
	// unused code
	void report_induced_action_on_set_and_kernel(
		std::ostream &file,
		actions::action *A_base,
		actions::action *A2,
		groups::sims *Stab, int size, long int *set,
		int verbose_level);
	// called from isomorph

};

void callback_choose_random_generator_orthogonal(
		int iteration,
	int *Elt, void *data, int verbose_level);
	// for use in action_init.cpp


// #############################################################################
// action_pointer_table.cpp
// #############################################################################


//! interface to the implementation functions for group actions

class action_pointer_table {

public:

	std::string label;

	/** function pointers for group actions. there are 26 of them. */
	long int (*ptr_element_image_of)(
			action &A, long int a, void *elt, int verbose_level);
	void (*ptr_element_image_of_low_level)(
			action &A, int *input, int *output, void *elt, int verbose_level);
	int (*ptr_element_linear_entry_ij)(
			action &A, void *elt, int i, int j, int verbose_level);
	int (*ptr_element_linear_entry_frobenius)(
			action &A, void *elt, int verbose_level);
	void (*ptr_element_one)(
			action &A, void *elt, int verbose_level);
	int (*ptr_element_is_one)(
			action &A, void *elt, int verbose_level);
	void (*ptr_element_unpack)(
			action &A, void *elt, void *Elt, int verbose_level);
	void (*ptr_element_pack)(
			action &A, void *Elt, void *elt, int verbose_level);
	void (*ptr_element_retrieve)(
			action &A, int hdl, void *elt, int verbose_level);
	int (*ptr_element_store)(
			action &A, void *elt, int verbose_level);
	void (*ptr_element_mult)(
			action &A, void *a, void *b, void *ab, int verbose_level);
	void (*ptr_element_invert)(
			action &A, void *a, void *av, int verbose_level);
	void (*ptr_element_transpose)(
			action &A, void *a, void *at, int verbose_level);
	void (*ptr_element_move)(
			action &A, void *a, void *b, int verbose_level);
	void (*ptr_element_dispose)(
			action &A, int hdl, int verbose_level);
	void (*ptr_element_print)(
			action &A, void *elt, std::ostream &ost);
	void (*ptr_element_print_quick)(
			action &A, void *elt, std::ostream &ost);
	void (*ptr_element_print_latex)(
			action &A, void *elt, std::ostream &ost);
	void (*ptr_element_print_latex_with_point_labels)(
			action &A,
		void *elt, std::ostream &ost,
		std::string *Point_labels, void *data);
		//void (*point_label)(std::stringstream &sstr, long int pt, void *data),
		//void *point_label_data);
	void (*ptr_element_print_verbose)(
			action &A, void *elt, std::ostream &ost);
	void (*ptr_print_point)(
			action &A, long int i, std::ostream &ost, int verbose_level);
	void (*ptr_element_code_for_make_element)(
			action &A, void *elt, int *data);
	void (*ptr_element_print_for_make_element)(
			action &A, void *elt, std::ostream &ost);
	void (*ptr_element_print_for_make_element_no_commas)(
			action &A,
		void *elt, std::ostream &ost);
	void (*ptr_unrank_point)(
			action &A, long int rk, int *v, int verbose_level);
	long int (*ptr_rank_point)(
			action &A, int *v, int verbose_level);

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
	void reset_counters();
	void save_stats(
			std::string &fname_base);
	void null_function_pointers();
	void copy_from_but_reset_counters(
			action_pointer_table *T);
	void init_function_pointers_matrix_group();
	void init_function_pointers_wreath_product_group();
	void init_function_pointers_direct_product_group();
	void init_function_pointers_polarity_extension();
	void init_function_pointers_permutation_group();
	void init_function_pointers_permutation_representation_group();
	void init_function_pointers_induced_action();
};



// #############################################################################
// known_groups.cpp
// #############################################################################

//! creating a known group with a default action in an action object

class known_groups {
public:

	action *A;

	known_groups();
	~known_groups();
	void init(
			action *A, int verbose_level);

	/** Create a linear group */
	void init_linear_group(
			algebra::field_theory::finite_field *F, int m,
		int f_projective, int f_general, int f_affine,
		int f_semilinear, int f_special,
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level);


	/** Create a projective linear (or semilinear) group PGL (or PGGL)*/
	void init_projective_group(
			int n, algebra::field_theory::finite_field *F,
		int f_semilinear,
		int f_basis, int f_init_sims,
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level);


	/** Create an affine group AGL(n,q) */
	void init_affine_group(
			int n, algebra::field_theory::finite_field *F,
		int f_semilinear,
		int f_basis, int f_init_sims,
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level);

	/** Create the general linear group GL(n,q) */
	void init_general_linear_group(
			int n, algebra::field_theory::finite_field *F,
		int f_semilinear, int f_basis, int f_init_sims,
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level);

	void compute_special_subgroup(
			int verbose_level);
	void setup_linear_group_from_strong_generators(
			algebra::basic_algebra::matrix_group *M,
			data_structures_groups::vector_ge *&nice_gens,
			int f_init_sims,
		int verbose_level);

	void init_sims_from_generators(
			int verbose_level);

	/** Create the projective special linear group PSL */
	void init_projective_special_group(
			int n, algebra::field_theory::finite_field *F,
		int f_semilinear, int f_basis, int verbose_level);

	void init_matrix_group_strong_generators_builtin(
			algebra::basic_algebra::matrix_group *M,
			data_structures_groups::vector_ge *&nice_gens,
		int verbose_level);


	void init_permutation_group(
			int degree, int f_no_base, int verbose_level);
	void init_permutation_group_from_nauty_output(
			other::l1_interfaces::nauty_output *NO,
		int verbose_level);
	void init_permutation_group_from_generators(
			int degree,
		int f_target_go, algebra::ring_theory::longinteger_object &target_go,
		int nb_gens, int *gens,
		int given_base_length, long int *given_base,
		int f_given_base,
		int verbose_level);
	// calls init_base_and_generators is f_given_base is true, otherwise does not initialize group
	void init_base_and_generators(
			int f_target_go, algebra::ring_theory::longinteger_object &target_go,
			int nb_gens, int *gens,
			int given_base_length, long int *given_base,
			int f_given_base,
			int verbose_level);


	/** Create the symmetric group
	 * as abstract permutation group */
	void init_symmetric_group(
			int degree, int verbose_level);
	void init_cyclic_group(
			int degree, int verbose_level);
	void init_elementary_abelian_group(
			int order, int verbose_level);
	void init_identity_group(
			int degree, int verbose_level);


	void create_sims(
			int verbose_level);

#if 0
	/** Create the orthogonal group O(5,q) */
	void init_BLT(
			field_theory::finite_field *F, int f_basis,
		int f_init_hash_table, int verbose_level);

	/** Create the orthogonal group O^epsilon(n,q) */
	void init_orthogonal_group(
			int epsilon,
		int n, field_theory::finite_field *F,
		int f_on_points, int f_on_lines,
		int f_on_points_and_lines,
		int f_semilinear,
		int f_basis, int verbose_level);
	// creates an object of type orthogonal
#endif

	void init_orthogonal_group_with_O(
			geometry::orthogonal_geometry::orthogonal *O,
		int f_on_points, int f_on_lines, int f_on_points_and_lines,
		int f_semilinear,
		int f_basis, int verbose_level);



	/** Create the wreath product group AGL(n,q) wreath Sym(nb_factors)
	 * in wreath product action
	 * and restrict the action to the tensor space. */
	void init_wreath_product_group_and_restrict(
			int nb_factors, int n,
			algebra::field_theory::finite_field *F,
			data_structures_groups::vector_ge *&nice_gens,
			int verbose_level);

	/** Create the wreath product group AGL(n,q) wreath Sym(nb_factors)
	 * in wreath product action
	 */
	void init_wreath_product_group(
			int nb_factors, int n,
			algebra::field_theory::finite_field *F,
			data_structures_groups::vector_ge *&nice_gens,
			int verbose_level);


	/** Create the permutation representation with a given set of generators
	 */
	void init_permutation_representation(
			action *A_original,
			int f_stay_in_the_old_action,
			data_structures_groups::vector_ge *gens,
			int *Perms, int degree,
			int verbose_level);

#if 0
	/** Create a group from generators */
	void init_group_from_strong_generators(
			data_structures_groups::vector_ge *gens,
			groups::sims *K,
		int given_base_length, int *given_base,
		int verbose_level);
	// calls sims::build_up_group_from_generators
#endif

	void create_orthogonal_group(
			action *subaction,
		int f_has_target_group_order,
		algebra::ring_theory::longinteger_object &target_go,
		void (* callback_choose_random_generator)(int iteration,
			int *Elt, void *data, int verbose_level),
		int verbose_level);
	// uses groups::schreier_sims


};


// #############################################################################
// group_element.cpp:
// #############################################################################



//! action related functions that are specific to a group element


class group_element {
public:

	action *A;

	/** temporary elements */
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5;
	int *eltrk1, *eltrk2, *eltrk3, *elt_mult_apply;
	uchar *elt1;
	char *element_rw_memory_object;
		// [coded_elt_size_in_char]
		// for element_write_to_memory_object,
		// element_read_from_memory_object


	group_element();
	~group_element();
	void init(
			action *A, int verbose_level);
	void null_element_data();
	void allocate_element_data();

	int image_of(
			void *elt, int a);
	void image_of_low_level(
			void *elt,
			int *input, int *output, int verbose_level);
	int linear_entry_ij(
			void *elt, int i, int j);
	int linear_entry_frobenius(
			void *elt);
	void one(
			void *elt);
	int is_one(
			void *elt);
	void unpack(
			void *elt, void *Elt);
	void pack(
			void *Elt, void *elt);
	void retrieve(
			void *elt, int hdl);
	int store(
			void *elt);
	void mult(
			void *a, void *b, void *ab);
	void mult_apply_from_the_right(
			void *a, void *b);
		// a := a * b
	void mult_apply_from_the_left(
			void *a, void *b);
		// b := a * b
	void invert(
			void *a, void *av);
	void invert_in_place(
			void *a);
	void transpose(
			void *a, void *at);
	void move(
			void *a, void *b);
	void dispose(
			int hdl);
	void print(
			std::ostream &ost, void *elt);
	void print_quick(
			std::ostream &ost, void *elt);
	void print_as_permutation(
			std::ostream &ost, void *elt);
	void print_point(
			int a, std::ostream &ost);
	void unrank_point(
			long int rk, int *v);
	long int rank_point(
			int *v);
	void code_for_make_element(
			int *data, void *elt);
	void print_for_make_element(
			std::ostream &ost, void *elt);
	void print_for_make_element_no_commas(
			std::ostream &ost, void *elt);

	long int element_image_of(
			long int a, void *elt, int verbose_level);
	void element_image_of_low_level(
			int *input, int *output,
		void *elt, int verbose_level);
	int element_linear_entry_ij(
			void *elt, int i, int j,
		int verbose_level);
	int element_linear_entry_frobenius(
			void *elt, int verbose_level);
	void element_one(
			void *elt, int verbose_level);
	int element_is_one(
			void *elt, int verbose_level);
	void element_unpack(
			void *elt, void *Elt, int verbose_level);
	void element_pack(
			void *Elt, void *elt, int verbose_level);
	void element_retrieve(
			int hdl, void *elt, int verbose_level);
	int element_store(
			void *elt, int verbose_level);
	void element_mult(
			void *a, void *b, void *ab, int verbose_level);
	void element_invert(
			void *a, void *av, int verbose_level);
	void element_transpose(
			void *a, void *at, int verbose_level);
	void element_move(
			void *a, void *b, int verbose_level);
	void element_dispose(
			int hdl, int verbose_level);
	void element_print(
			void *elt, std::ostream &ost);
	void element_print_quick(
			void *elt, std::ostream &ost);
	void element_print_latex(
			void *elt, std::ostream &ost);
	void element_print_latex_with_extras(
			void *elt, std::string &label, std::ostream &ost);
	void element_print_latex_with_point_labels(
		void *elt, std::ostream &ost,
		std::string *Point_labels, void *data);
	void element_print_verbose(
			void *elt, std::ostream &ost);
	void element_code_for_make_element(
			void *elt, int *data);
	std::string element_stringify_code_for_make_element(
			void *elt);
	void element_print_for_make_element(
			void *elt,
			std::ostream &ost);
	void element_print_for_make_element_no_commas(
			void *elt,
			std::ostream &ost);
	void element_print_as_permutation(
			void *elt,
			std::ostream &ost);
	void compute_permutation(
			void *elt,
		int *perm, int verbose_level);
	void element_print_as_permutation_verbose(
			void *elt,
			std::ostream &ost, int verbose_level);
	void cycle_type(
			void *elt,
			int *cycles, int &nb_cycles,
			int verbose_level);
	void element_print_as_permutation_with_offset(
			void *elt,
			std::ostream &ost,
		int offset, int f_do_it_anyway_even_for_big_degree,
		int f_print_cycles_of_length_one,
		int verbose_level);
	void element_print_as_permutation_with_offset_and_max_cycle_length(
		void *elt,
		std::ostream &ost, int offset, int max_cycle_length,
		int f_orbit_structure);
	void element_print_image_of_set(
			void *elt,
		int size, long int *set);
	int element_signum_of_permutation(
			void *elt);
	void element_write_file_fp(
			int *Elt,
			std::ofstream &fp, int verbose_level);
	void element_read_file_fp(
			int *Elt,
			std::ifstream &fp, int verbose_level);
	void element_write_file(
			int *Elt,
			std::string &fname, int verbose_level);
	void element_read_file(
			int *Elt,
			std::string &fname, int verbose_level);
	void element_write_to_memory_object(
			int *Elt,
			other::orbiter_kernel_system::memory_object *m,
			int verbose_level);
	void element_read_from_memory_object(
			int *Elt,
			other::orbiter_kernel_system::memory_object *m,
			int verbose_level);
	void element_write_to_file_binary(
			int *Elt,
			std::ofstream &fp, int verbose_level);
	void element_read_from_file_binary(
			int *Elt,
			std::ifstream &fp, int verbose_level);
	void random_element(
			groups::sims *S, int *Elt,
		int verbose_level);
	int element_has_order_two(
			int *E1, int verbose_level);
	int product_has_order_two(
			int *E1, int *E2, int verbose_level);
	int product_has_order_three(
			int *E1, int *E2, int verbose_level);
	int element_order(
			void *elt);
	int element_order_and_cycle_type(
			void *elt, int *cycle_type);
	int element_order_and_cycle_type_verbose(
			void *elt, int *cycle_type, int verbose_level);
	int element_order_if_divisor_of(
			void *elt, int o);
	void element_print_base_images(
			int *Elt);
	void element_print_base_images(
			int *Elt, std::ostream &ost);
	void element_print_base_images_verbose(
			int *Elt,
			std::ostream &ost, int verbose_level);
	void element_base_images(
			int *Elt, int *base_images);
	void element_base_images_verbose(
			int *Elt,
		int *base_images, int verbose_level);
	void minimize_base_images(
			int level, groups::sims *S,
		int *Elt, int verbose_level);
	void element_conjugate_bvab(
			int *Elt_A,
		int *Elt_B, int *Elt_C, int verbose_level);
	void element_conjugate_babv(
			int *Elt_A,
		int *Elt_B, int *Elt_C, int verbose_level);
	void element_commutator_abavbv(
			int *Elt_A,
		int *Elt_B, int *Elt_C, int verbose_level);
	int find_non_fixed_point(
			void *elt, int verbose_level);
#if 0
	int find_fixed_points(
			void *elt,
			int *fixed_points, int verbose_level);
#endif
	void compute_fixed_points(
			void *elt,
			std::vector<long int> &fixed_points, int verbose_level);
	int count_fixed_points(
			void *elt, int verbose_level);
	int test_if_set_stabilizes(
			int *Elt,
			int size, long int *set, int verbose_level);
	void map_a_set(
			long int *set,
			long int *image_set,
		int n, int *Elt, int verbose_level);
	void map_a_set_and_reorder(
			long int *set, long int *image_set,
		int n, int *Elt, int verbose_level);
	void make_element_from_permutation_representation(
			int *Elt,
			groups::sims *S, int *data, int verbose_level);
	void make_element_from_base_image(
			int *Elt, groups::sims *S,
			int *data, int verbose_level);
	void compute_base_images(
			int *Elt, int *base_images, int verbose_level);
	void make_element_2x2(
			int *Elt, int a0, int a1, int a2, int a3);
	void make_element_from_string(
			int *Elt,
			std::string &data_string, int verbose_level);
	void make_element(
			int *Elt, int *data, int verbose_level);
	void element_power_int_in_place(
			int *Elt,
		int n, int verbose_level);
	void word_in_ab(
			int *Elt1, int *Elt2, int *Elt3,
		const char *word, int verbose_level);
	void evaluate_word(
			int *Elt, int *word, int len,
			data_structures_groups::vector_ge *gens,
			int verbose_level);
	int check_if_in_set_stabilizer(
			int *Elt,
			int size, long int *set, int verbose_level);
	void check_if_in_set_stabilizer_debug(
			int *Elt,
			int size, long int *set, int verbose_level);
	int check_if_transporter_for_set(
			int *Elt,
			int size,
			long int *set1, long int *set2,
			int verbose_level);
#if 0
	void compute_fixed_objects_in_PG(
			int up_to_which_rank,
			geometry::projective_space *P,
		int *Elt,
		std::vector<std::vector<long int> > &Fix,
		int verbose_level);
	void compute_fixed_points_in_induced_action_on_grassmannian(
		int *Elt,
		int dimension,
		std::vector<long int> &fixpoints,
		int verbose_level);
#endif
	void report_fixed_objects_in_PG(
			std::ostream &ost,
			geometry::projective_geometry::projective_space *P,
		int *Elt,
		int verbose_level);
	int test_if_it_fixes_the_polynomial(
		int *Elt,
		int *input,
		algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level);
	void action_on_polynomial(
		int *Elt,
		int *input, int *output,
		algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level);
	std::string stringify(
		int *Elt);
	std::string stringify_base_images(
			int *Elt, int verbose_level);


};



// #############################################################################
// induced_action.cpp:
// #############################################################################



//! create new actions from old


class induced_action {
public:

	action *A_old;

	induced_action();
	~induced_action();
	void init(
			action *A, int verbose_level);

	action *induced_action_on_interior_direct_product(
			int nb_rows,
			int verbose_level);
	action *induced_action_on_set_partitions(
			int partition_class_size,
			int verbose_level);

	action *induced_action_by_representation_on_conic(
		int f_induce_action, groups::sims *old_G,
		int verbose_level);
	action *induced_action_on_cosets(
			induced_actions::action_on_cosets *A_on_cosets,
		int f_induce_action,
		groups::sims *old_G,
		int verbose_level);
	action *induced_action_on_factor_space(
			induced_actions::action_on_factor_space *AF,
		int f_induce_action, groups::sims *old_G,
		int verbose_level);

	action *induced_action_on_cosets_of_subgroup(
			groups::strong_generators *Subgroup_gens_H,
			groups::strong_generators *Subgroup_gens_G,
			int verbose_level);

	action *induced_action_on_grassmannian(
			int k,
		int verbose_level);
	action *induced_action_on_grassmannian_preloaded(
			induced_actions::action_on_grassmannian *AG,
		int f_induce_action, groups::sims *old_G,
		int verbose_level);
	action *induced_action_on_spread_set(
			induced_actions::action_on_spread_set *AS,
		int f_induce_action, groups::sims *old_G,
		int verbose_level);

	action *induced_action_on_wedge_product(
			int verbose_level);
	action *induced_action_on_determinant(
			groups::sims *old_G, int verbose_level);
	action *induced_action_on_Galois_group(
			groups::sims *old_G, int verbose_level);
	action *induced_action_on_sign(
			groups::sims *old_G, int verbose_level);

	action *create_induced_action_by_conjugation(
			groups::sims *Base_group, int f_ownership,
			int f_basis, groups::sims *old_G,
			int verbose_level);
	action *induced_action_by_right_multiplication(
		int f_basis, groups::sims *old_G,
		groups::sims *Base_group, int f_ownership,
		int verbose_level);
	action *create_induced_action_on_sets(
			int nb_sets,
		int set_size, long int *sets,
		int verbose_level);
	action *induced_action_on_sets(
			groups::sims *old_G,
		int nb_sets, int set_size, long int *sets,
		int f_induce_action, int verbose_level);
	action *create_induced_action_on_subgroups(
			groups::sims *S,
			data_structures_groups::hash_table_subgroups *Hash_table_subgroups,
		int verbose_level);
	action *induced_action_on_subgroups(
			action *old_action,
			groups::sims *S,
			data_structures_groups::hash_table_subgroups *Hash_table_subgroups,
		int verbose_level);
	action *induced_action_by_restriction_on_orbit_with_schreier_vector(
		int f_induce_action, groups::sims *old_G,
		data_structures_groups::schreier_vector *Schreier_vector,
		int pt, int verbose_level);
	void original_point_labels(
			long int *points, int nb_points,
			long int *&original_points, int verbose_level);
	action *restricted_action(
			long int *points, int nb_points,
			std::string &label_of_set,
			std::string &label_of_set_tex,
		int verbose_level);
	action *create_induced_action_by_restriction(
			groups::sims *old_G, int size,
			long int *set,
			std::string &label_of_set,
			std::string &label_of_set_tex,
			int f_induce,
			int verbose_level);
	action *induced_action_by_restriction(
		action *old_action,
		int f_induce_action, groups::sims *old_G,
		int nb_points, long int *points,
		std::string &label_of_set,
		std::string &label_of_set_tex,
		int verbose_level);
	action *induced_action_on_pairs(
		int verbose_level);
	action *induced_action_on_ordered_pairs(
			groups::sims *old_G,
		int verbose_level);
	action *induced_action_on_k_subsets(
		int k,
		int verbose_level);
	action *induced_action_on_orbits(
			groups::schreier *Sch, int f_play_it_safe,
		int verbose_level);
	action *induced_action_on_andre(
			action *An,
		action *An1,
		geometry::finite_geometries::andre_construction *Andre,
		int verbose_level);
	action *induced_action_on_homogeneous_polynomials(
			algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		int f_induce_action, groups::sims *old_G,
		int verbose_level);
	action *induced_action_on_homogeneous_polynomials_given_by_equations(
			algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		int *Equations, int nb_equations,
		int f_induce_action, groups::sims *old_G,
		int verbose_level);
	action *base_change(
		int size, long int *set,
		groups::sims *old_Sims,
		int verbose_level);

};



// #############################################################################
// stabilizer_chain_base_data.cpp:
// #############################################################################



#define STABILIZER_CHAIN_DATA_MAX_DEGREE 1L << 29

// 2^29 = 536870912

//! base and transversals in a stabilizer chain for a permutation group


class stabilizer_chain_base_data {
private:
	action *A;

	/** whether we have a base (b_0,\ldots,b_{l-1}) */
	int f_has_base;

	/** the length of the base */
	int base_len;



	/** the base (b_0,\ldots,b_{l-1}) */
	long int *base; // [base_len]



	/** the length of the orbit of $G^{(i)}$ on $b_i$ */
	int *transversal_length; // [base_len]

	/** the orbit of b_i as a permutation of the points of the set we act on */
	long int **orbit; // [base_len][A->degree]

	/** the inverse orbit permutation associated with the orbit of b_i */
	long int **orbit_inv; // [base_len][A->degree]

	int *path;
public:

	stabilizer_chain_base_data();
	~stabilizer_chain_base_data();
	void free_base_data();
	void allocate_base_data(
			action *A,
			int base_len, int verbose_level);
	void reallocate_base(
			int new_base_point, int verbose_level);
	void init_base_from_sims_after_shortening(
			actions::action *A,
			groups::sims *Sims, int verbose_level);
	void init_base_from_sims(
			groups::sims *G, int verbose_level);
	actions::action *get_A();
	void set_A(
			actions::action *A);
	int &get_f_has_base();
	int &get_base_len();
	long int &base_i(
			int i);
	long int *&get_base();
	int &transversal_length_i(
			int i);
	int *&get_transversal_length();
	long int &orbit_ij(
			int i, int j);
	long int &orbit_inv_ij(
			int i, int j);
	int &path_i(
			int i);
	void group_order(
			algebra::ring_theory::longinteger_object &go);
	void init_projective_matrix_group(
			algebra::field_theory::finite_field *F,
			int n, int f_semilinear, int degree,
			int verbose_level);
	void init_affine_matrix_group(
			algebra::field_theory::finite_field *F,
			int n, int f_semilinear, int degree,
			int verbose_level);
	void init_linear_matrix_group(
			algebra::field_theory::finite_field *F,
			int n, int f_semilinear, int degree,
			int verbose_level);
	void report_basic_orbits(
			std::ostream &ost);

};



}}}



#endif /* ORBITER_SRC_LIB_GROUP_ACTIONS_ACTIONS_ACTIONS_H_ */


