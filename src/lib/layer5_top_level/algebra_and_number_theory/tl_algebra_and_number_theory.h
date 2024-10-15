// tl_algebra_and_number_theory.h
//
// Anton Betten
//
// moved here from top_level.h: July 28, 2018
// top_level started:  September 23 2010
// based on global.h, which was taken from reader.h: 3/22/09


#ifndef ORBITER_SRC_LIB_TOP_LEVEL_ALGEBRA_AND_NUMBER_THEORY_TL_ALGEBRA_AND_NUMBER_THEORY_H_
#define ORBITER_SRC_LIB_TOP_LEVEL_ALGEBRA_AND_NUMBER_THEORY_TL_ALGEBRA_AND_NUMBER_THEORY_H_


namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


// #############################################################################
// action_on_forms_activity_description.cpp
// #############################################################################


//! description of an action on forms


class action_on_forms_activity_description {

public:

	// ToDo: undocumented


	int f_algebraic_normal_form;
	std::string algebraic_normal_form_input;

	int f_orbits_on_functions;
	std::string orbits_on_functions_input;

	int f_associated_set_in_plane;
	std::string associated_set_in_plane_input;

	int f_differential_uniformity;
	std::string differential_uniformity_input;


	action_on_forms_activity_description();
	~action_on_forms_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// action_on_forms_activity.cpp
// #############################################################################


//! perform an activity associated with an action on forms

class action_on_forms_activity {
public:
	action_on_forms_activity_description *Descr;

	apps_algebra::action_on_forms *AF;



	action_on_forms_activity();
	~action_on_forms_activity();
	void init(
			action_on_forms_activity_description *Descr,
			apps_algebra::action_on_forms *AF,
			int verbose_level);
	void perform_activity(
			int verbose_level);
	void do_algebraic_normal_form(
			int verbose_level);
	void do_orbits_on_functions(
			int verbose_level);
	void do_associated_set_in_plane(
			int verbose_level);
	void do_differential_uniformity(
			int verbose_level);

};



// #############################################################################
// action_on_forms_description.cpp
// #############################################################################


//! description of an action on forms


class action_on_forms_description {

public:

	// ToDo: undocumented

	int f_space;
	std::string space_label;

	int f_degree;
	int degree;


	action_on_forms_description();
	~action_on_forms_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};




// #############################################################################
// action_on_forms.cpp
// #############################################################################


//! an action on forms


class action_on_forms {

public:
	action_on_forms_description *Descr;

	std::string prefix;
	std::string label_txt;
	std::string label_tex;

	int q;
	field_theory::finite_field *F;

	int f_semilinear;

	projective_geometry::projective_space_with_action *PA;

	combinatorics::polynomial_function_domain *PF;

	actions::action *A_on_poly;

	int f_has_group;
	groups::strong_generators *Sg;




	action_on_forms();
	~action_on_forms();
	void create_action_on_forms(
			action_on_forms_description *Descr,
			int verbose_level);
	void orbits_on_functions(
			int *The_functions, int nb_functions, int len,
			int verbose_level);
	void orbits_on_equations(
			int *The_equations, int nb_equations, int len,
			groups::schreier *&Orb,
			actions::action *&A_on_equations,
			int verbose_level);
	void associated_set_in_plane(
			int *func, int len,
			long int *&Rk, int verbose_level);
	void differential_uniformity(
			int *func, int len, int verbose_level);

};


// #############################################################################
// algebra_global_with_action.cpp
// #############################################################################

//! group theoretic functions which require an action


class algebra_global_with_action {
public:
	void orbits_under_conjugation(
			long int *the_set, int set_size,
			groups::sims *S,
			groups::strong_generators *SG,
			data_structures_groups::vector_ge *Transporter,
			int verbose_level);
	void create_subgroups(
			groups::strong_generators *SG,
			long int *the_set, int set_size,
			groups::sims *S,
			actions::action *A_conj,
			groups::schreier *Classes,
			data_structures_groups::vector_ge *Transporter,
			int verbose_level);
	void conjugacy_classes_based_on_normal_forms(
			actions::action *A,
			groups::sims *override_Sims,
			std::string &label,
			std::string &label_tex,
			int verbose_level);
	void classes_GL(
			field_theory::finite_field *F,
			int d, int f_no_eigenvalue_one, int verbose_level);
	void do_normal_form(
			int q, int d,
			int f_no_eigenvalue_one, int *data, int data_sz,
			int verbose_level);
	void do_identify_one(
			int q, int d,
			int f_no_eigenvalue_one, int elt_idx,
			int verbose_level);
	void do_identify_all(
			int q, int d,
			int f_no_eigenvalue_one, int verbose_level);
	void do_random(
			int q, int d,
			int f_no_eigenvalue_one, int verbose_level);
	void group_table(
			int q, int d, int f_poly, std::string &poly,
			int f_no_eigenvalue_one, int verbose_level);
	void centralizer_in_PGL_d_q_single_element_brute_force(
			int q, int d,
			int elt_idx, int verbose_level);
	void centralizer_in_PGL_d_q_single_element(
			int q, int d,
			int elt_idx, int verbose_level);
	// creates a finite_field, and two actions
	// using init_projective_group and init_general_linear_group
	void compute_centralizer_of_all_elements_in_PGL_d_q(
			int q, int d, int verbose_level);
	void compute_regular_representation(
			actions::action *A, groups::sims *S,
			data_structures_groups::vector_ge *SG,
			int *&perm, int verbose_level);
	// allocates perm[SG->len * goi]
	void presentation(
			actions::action *A,
			groups::sims *S, int goi,
			data_structures_groups::vector_ge *gens,
		int *primes, int verbose_level);

	void do_eigenstuff(
			field_theory::finite_field *F,
			int size, int *Data, int verbose_level);
	void A5_in_PSL_(
			int q, int verbose_level);
	void A5_in_PSL_2_q(
			int q,
			layer2_discreta::typed_objects::discreta_matrix & A,
			layer2_discreta::typed_objects::discreta_matrix & B,
			layer2_discreta::typed_objects::domain *dom_GFq,
			int verbose_level);
	void A5_in_PSL_2_q_easy(
			int q,
			layer2_discreta::typed_objects::discreta_matrix & A,
			layer2_discreta::typed_objects::discreta_matrix & B,
			layer2_discreta::typed_objects::domain *dom_GFq,
			int verbose_level);
	void A5_in_PSL_2_q_hard(
			int q,
			layer2_discreta::typed_objects::discreta_matrix & A,
			layer2_discreta::typed_objects::discreta_matrix & B,
			layer2_discreta::typed_objects::domain *dom_GFq,
			int verbose_level);
	int proj_order(
			layer2_discreta::typed_objects::discreta_matrix &A);
	void trace(
			layer2_discreta::typed_objects::discreta_matrix &A,
			layer2_discreta::typed_objects::discreta_base &tr);
	void elementwise_power_int(
			layer2_discreta::typed_objects::discreta_matrix &A,
			int k, int verbose_level);
	int is_in_center(
			layer2_discreta::typed_objects::discreta_matrix &B);
	void matrix_convert_to_numerical(
			layer2_discreta::typed_objects::discreta_matrix &A,
			int *AA, int q);


	void young_symmetrizer(
			int n, int verbose_level);
	void young_symmetrizer_sym_4(
			int verbose_level);
	void linear_codes_with_bounded_minimum_distance(
			poset_classification::poset_classification_control *Control,
			group_constructions::linear_group *LG,
			int d, int target_depth, int verbose_level);
	void permutation_representation_of_element(
			actions::action *A,
			std::string &element_description,
			int verbose_level);
	void relative_order_vector_of_cosets(
			actions::action *A, groups::strong_generators *SG,
			data_structures_groups::vector_ge *cosets,
			int *&relative_order_table, int verbose_level);
	void representation_on_polynomials(
			group_constructions::linear_group *LG,
			ring_theory::homogeneous_polynomial_domain *HPD,
			int verbose_level);

	void do_eigenstuff_with_coefficients(
			field_theory::finite_field *F,
			int n, std::string &coeffs_text,
			int verbose_level);
	void do_eigenstuff_from_file(
			field_theory::finite_field *F,
			int n, std::string &fname, int verbose_level);

	void find_singer_cycle(
			any_group *Any_group,
			actions::action *A1, actions::action *A2,
			int verbose_level);
	void search_element_of_order(
			any_group *Any_group,
			actions::action *A1, actions::action *A2,
			int order, int verbose_level);
	void find_standard_generators(
			any_group *Any_group,
			actions::action *A1, actions::action *A2,
			int order_a, int order_b, int order_ab, int verbose_level);
	void do_character_table_symmetric_group(
			int deg, int verbose_level);
	void group_of_automorphisms_by_images_of_generators(
			data_structures_groups::vector_ge *Elements_ge,
			int *Images, int m, int n,
			any_group *AG,
			std::string &label,
			int verbose_level);

};


// #############################################################################
// any_group_linear.cpp
// #############################################################################

//! group theoretic activities specifically for linear groups

class any_group_linear {

public:

	any_group *Any_group;

	any_group_linear();
	~any_group_linear();

	void init(
			any_group *Any_group, int verbose_level);
	void classes_based_on_normal_form(
			int verbose_level);
	void find_singer_cycle(
			int verbose_level);
	void isomorphism_Klein_quadric(
			std::string &fname, int verbose_level);
	void do_orbits_on_subspaces(
			poset_classification::poset_classification_control *Control,
			orbits::orbits_on_subspaces *&OoS,
			int depth, int verbose_level);
	void do_tensor_classify(
			std::string &control_label,
			apps_geometry::tensor_classify *&T,
			int depth, int verbose_level);
	void do_tensor_permutations(
			int verbose_level);
	void do_linear_codes(
			std::string &control_label,
			int minimum_distance,
			int target_size, int verbose_level);
	void do_classify_ovoids(
			apps_geometry::ovoid_classify_description
				*Ovoid_classify_description,
			int verbose_level);
	int subspace_orbits_test_set(
			int len, long int *S, int verbose_level);

};



// #############################################################################
// any_group.cpp
// #############################################################################

//! front end for group theoretic activities for three kinds of groups: linear groups, permutation groups, modified groups

class any_group {

public:

	int f_linear_group;
	group_constructions::linear_group *LG;

	int f_permutation_group;
	group_constructions::permutation_group_create *PGC;

	int f_modified_group;
	modified_group_create *MGC;

	actions::action *A_base;
	actions::action *A;

	std::string label;
	std::string label_tex;

	groups::strong_generators *Subgroup_gens;
	groups::sims *Subgroup_sims;

	any_group_linear *Any_group_linear;

	int f_has_subgroup_lattice;
	groups::subgroup_lattice *Subgroup_lattice;

	int f_has_class_data;
	interfaces::conjugacy_classes_of_subgroups *class_data;


	any_group();
	~any_group();
	void init_basic(
			int verbose_level);
	void init_linear_group(
			group_constructions::linear_group *LG, int verbose_level);
	void init_permutation_group(
			group_constructions::permutation_group_create *PGC,
			int verbose_level);
	void init_modified_group(
			modified_group_create *MGC, int verbose_level);
	void create_latex_report(
			graphics::layered_graph_draw_options *O,
			int f_sylow, int f_group_table, //int f_classes,
			int verbose_level);
	void export_group_table(
			int verbose_level);
	void do_export_orbiter(
			actions::action *A2, int verbose_level);
	void do_export_gap(
			int verbose_level);
	void do_export_magma(
			int verbose_level);
	void do_canonical_image_GAP(
			std::string &input_set, int verbose_level);
	void do_canonical_image_orbiter(
			std::string &input_set_text,
			int verbose_level);
	void create_group_table(
			int *&Table, long int &n, int verbose_level);
	void normalizer(
			int verbose_level);
	void centralizer(
			std::string &element_label,
			std::string &element_description_text,
			int verbose_level);
	void permutation_representation_of_element(
			std::string &element_description_text,
			int verbose_level);
	void normalizer_of_cyclic_subgroup(
			std::string &element_label,
			std::string &element_description_text,
			int verbose_level);
	void do_find_subgroups(
			int order_of_subgroup,
			int verbose_level);
	void print_elements(
			int verbose_level);
	void print_elements_tex(
			int f_with_permutation,
			int f_override_action, actions::action *A_special,
			int verbose_level);
	void order_of_products_of_elements_by_rank(
			std::string &Elements_text,
			int verbose_level);
	void save_elements_csv(
			std::string &fname, int verbose_level);
	void export_inversion_graphs(
			std::string &fname, int verbose_level);
#if 0
	void multiply_elements_csv(
			std::string &fname1,
			std::string &fname2,
			std::string &fname3,
			int f_column_major_ordering,
			int verbose_level);
	void apply_elements_to_set_csv(
			std::string &fname1,
			std::string &fname2,
			std::string &set_text,
			int verbose_level);
#endif
	void random_element(
			std::string &elt_label, int verbose_level);
	void element_rank(
			std::string &elt_data, int verbose_level);
	void element_unrank(
			std::string &rank_string, int verbose_level);
	void conjugacy_class_of(
			std::string &label_of_class,
			std::string &rank_string,
			int verbose_level);
	void automorphism_by_generator_images(
			std::string &label,
			data_structures_groups::vector_ge *Elements_ge,
			int *Images, int m, int n,
			int *&Perms, long int &go,
			int verbose_level);
	// uses orbits_schreier::orbit_of_sets
	// needs Subgroup_sims to set up action by right multiplication
	// output: Perms[m * go]
	void automorphism_by_generator_images_save(
			int *Images, int m, int n,
			int *Perms, long int go,
			int verbose_level);
	void do_reverse_isomorphism_exterior_square(
			int verbose_level);

#if 0
	void create_latex_report_for_permutation_group(
			graphics::layered_graph_draw_options *LG_Draw_options,
			int verbose_level);
	void create_latex_report_for_modified_group(
			graphics::layered_graph_draw_options *LG_Draw_options,
			int verbose_level);
#endif
	groups::strong_generators *get_strong_generators();
	int is_subgroup_of(
			any_group *AG_secondary, int verbose_level);
	void set_of_coset_representatives(
			any_group *AG_secondary,
			data_structures_groups::vector_ge *&coset_reps,
			int verbose_level);
	void report_coset_reps(
			data_structures_groups::vector_ge *coset_reps,
			int verbose_level);
	void print_given_elements_tex(
			std::string &label_of_elements,
			int *element_data, int nb_elements,
			int f_with_permutation,
			int f_with_fix_structure,
			int verbose_level);
	void process_given_elements(
			std::string &label_of_elements,
			int *element_data, int nb_elements,
			int verbose_level);
	void apply_isomorphism_wedge_product_4to6(
			std::string &label_of_elements,
			int *element_data, int nb_elements,
			int verbose_level);
	void order_of_products_of_pairs(
			std::string &label_of_elements,
			int *element_data, int nb_elements,
			int verbose_level);
	void conjugate(
			std::string &label_of_elements,
			std::string &conjugate_data,
			int *element_data, int nb_elements,
			int verbose_level);
	void print_action_on_surface(
			std::string &surface_label,
			std::string &label_of_elements,
			int *element_data, int nb_elements,
			int verbose_level);
	void subgroup_lattice_compute(
			int verbose_level);
	void subgroup_lattice_load(
			std::string &fname,
			int verbose_level);
	void subgroup_lattice_draw(
			int verbose_level);
	void subgroup_lattice_draw_by_orbits(
			int verbose_level);
	void subgroup_lattice_intersection_orbit_orbit(
			int orbit1, int orbit2,
			int verbose_level);
	void subgroup_lattice_find_overgroup_in_orbit(
			int orbit_global1, int group1, int orbit_global2,
			int verbose_level);
	void subgroup_lattice_create_flag_transitive_geometry_with_partition(
			int P_orbit_global,
			int Q_orbit_global,
			int R_orbit_global,
			int R_group,
			int intersection_size,
			int verbose_level);
	void subgroup_lattice_create_coset_geometry(
			int P_orb_global, int P_group,
			int Q_orb_global, int Q_group,
			int intersection_size,
			int verbose_level);
	void subgroup_lattice_identify_subgroup(
			std::string &group_label,
			int &go, int &layer_idx, int &orb_idx, int &group_idx,
			int verbose_level);
	void element_processing(
			element_processing_description *element_processing_descr,
			int verbose_level);
	void print();
	void classes(
			int verbose_level);
	void subgroup_lattice_magma(
			int verbose_level);
	void find_standard_generators(
			int order_a,
			int order_b,
			int order_ab,
			int verbose_level);
	void search_element_of_order(
			int order, int verbose_level);




};


// #############################################################################
// character_table_burnside.cpp
// #############################################################################

//! character table of a finite group using the algorithm of Burnside


class character_table_burnside {
public:

	void do_it(
			int n, int verbose_level);
	void create_matrix(
			typed_objects::discreta_matrix &M, int i, int *S, int nb_classes,
		int *character_degree, int *class_size,
		int verbose_level);
	void compute_character_table(
			algebra::a_domain *D, int nb_classes, int *Omega,
		int *character_degree, int *class_size,
		int *&character_table, int verbose_level);
	void compute_character_degrees(
			algebra::a_domain *D,
		int goi, int nb_classes, int *Omega, int *class_size,
		int *&character_degree,
		int verbose_level);
	void compute_omega(
			algebra::a_domain *D, int *N0, int nb_classes,
			int *Mu, int nb_mu, int *&Omega,
			int verbose_level);
	int compute_r0(
			int *N, int nb_classes, int verbose_level);
	void compute_multiplication_constants_center_of_group_ring(
			actions::action *A,
			induced_actions::action_by_conjugation *ABC,
		groups::schreier *Sch, int nb_classes, int *&N,
		int verbose_level);
	void compute_Distribution_table(
			actions::action *A,
			induced_actions::action_by_conjugation *ABC,
			groups::schreier *Sch, int nb_classes,
		int **Gens, int nb_gens, int t_max, int *&Distribution,
		int verbose_level);
	void multiply_word(
			actions::action *A, int **Gens,
			int *Choice, int t, int *Elt1, int *Elt2,
			int verbose_level);
	void create_generators(
			actions::action *A, int n,
			int **&Elt, int &nb_gens, int f_special,
			int verbose_level);
	void integral_eigenvalues(
			int *M, int n,
		int *&Lambda,
		int &nb_lambda,
		int *&Mu,
		int *&Mu_mult,
		int &nb_mu,
		int verbose_level);
	void characteristic_poly(
			int *N, int size,
			typed_objects::unipoly &charpoly,
			int verbose_level);
	void double_swap(
			double &a, double &b);
	int double_Gauss(
			double *A, int m, int n, int *base_cols,
			int verbose_level);
	void double_matrix_print(
			double *A, int m, int n);
	double double_abs(
			double x);
	void kernel_columns(
			int n, int nb_base_cols, int *base_cols,
			int *kernel_cols);
	void matrix_get_kernel(
			double *M, int m, int n,
			int *base_cols, int nb_base_cols,
		int &kernel_m, int &kernel_n, double *kernel);
	int double_as_int(
			double x);

};

// #############################################################################
// element_processing_description.cpp
// #############################################################################

//! describe a process applied to a set of group elements

class element_processing_description {

public:

	// TABLES/element_processing.tex

	int f_input;
	std::string input_label;

	int f_print;

	int f_apply_isomorphism_wedge_product_4to6;

	int f_with_permutation;

	int f_with_fix_structure;

	int f_order_of_products_of_pairs;

	int f_conjugate;
	std::string conjugate_data;

	int f_print_action_on_surface;
	std::string print_action_on_surface_label;



	element_processing_description();
	~element_processing_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};


// #############################################################################
// group_modification_description.cpp
// #############################################################################

//! create a new group or group action from an old

class group_modification_description {

public:

	// TABLES/group_modification.tex

	int f_restricted_action;
	std::string restricted_action_set_text;
	std::string restricted_action_set_text_tex;

	int f_on_k_subspaces;
	int on_k_subspaces_k;

	int f_on_k_subsets;
	int on_k_subsets_k;

	int f_on_wedge_product;

	int f_create_special_subgroup;

	int f_create_even_subgroup;

	int f_point_stabilizer;
	int point_stabilizer_point;

	int f_set_stabilizer;
	std::string set_stabilizer_the_set;
	std::string set_stabilizer_control;


	int f_projectivity_subgroup;

	int f_subfield_subgroup;
	int subfield_subgroup_index;

	int f_action_on_self_by_right_multiplication;

	int f_direct_product;
	std::string direct_product_input;
	std::string direct_product_subgroup_order;
	std::string direct_product_subgroup_gens;

	int f_polarity_extension;
	std::string polarity_extension_input;
	std::string polarity_extension_PA;

	int f_on_middle_layer_grassmannian;

	int f_on_points_and_hyperplanes;

	int f_holomorph;

	int f_automorphism_group;

	int f_subgroup_by_lattice;
	int subgroup_by_lattice_orbit_index;

	int f_stabilizer_of_variety;
	std::string stabilizer_of_variety_label;

	std::vector<std::string> from;

	group_modification_description();
	~group_modification_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};




// #############################################################################
// group_theoretic_activity_description.cpp
// #############################################################################


//! description of an activity associated with a group

class group_theoretic_activity_description {
public:

	// TABLES/group_theoretic_activity_1.tex


	int f_report;

		// flags that apply to report:
		int f_report_sylow;
		int f_report_group_table;
		//int f_report_classes;


	int f_export_group_table;

	int f_random_element;
	std::string random_element_label;


	int f_permutation_representation_of_element;
	std::string permutation_representation_element_text;

	int f_apply;
	std::string apply_input;
	std::string apply_element;

	int f_element_processing;
	element_processing_description *element_processing_descr;

	int f_multiply;
	std::string multiply_a;
	std::string multiply_b;

	int f_inverse;
	std::string inverse_a;

	int f_consecutive_powers;
	std::string consecutive_powers_a_text;
	std::string consecutive_powers_exponent_text;

	int f_raise_to_the_power;
	std::string raise_to_the_power_a_text;
	std::string raise_to_the_power_exponent_text;

	int f_export_orbiter;

	int f_export_gap;

	int f_export_magma;


	// GAP:
	int f_canonical_image_GAP;
	std::string canonical_image_GAP_input_set;

	int f_canonical_image;
	std::string canonical_image_input_set;

	int f_search_element_of_order;
	int search_element_order;

	int f_find_standard_generators;
	int find_standard_generators_order_a;
	int find_standard_generators_order_b;
	int find_standard_generators_order_ab;

	int f_element_rank;
	std::string element_rank_data;

	int f_element_unrank;
	std::string element_unrank_data;




	// TABLES/group_theoretic_activity_2.tex

	int f_find_singer_cycle;


	int f_classes_based_on_normal_form;


	// Magma:
	int f_normalizer;


	// Magma:
	int f_centralizer_of_element;
	std::string centralizer_of_element_label;
	std::string centralizer_of_element_data;


#if 0
	int f_orbits_on_group_elements_under_conjugation;
	std::string orbits_on_group_elements_under_conjugation_fname;
	std::string orbits_on_group_elements_under_conjugation_transporter_fname;
#endif

	// Magma:
	int f_normalizer_of_cyclic_subgroup;
	std::string normalizer_of_cyclic_subgroup_label;
	std::string normalizer_of_cyclic_subgroup_data;

	// Magma:
	int f_classes;

	int f_subgroup_lattice_magma;

	// undocumented:
	int f_find_subgroup;
	int find_subgroup_order;

	//int f_test_if_geometric;
	//int test_if_geometric_depth;

	int f_conjugacy_class_of;
	std::string conjugacy_class_of_label;
	std::string conjugacy_class_of_data;

	int f_isomorphism_Klein_quadric;
	std::string isomorphism_Klein_quadric_fname;

	int f_print_elements;
	int f_print_elements_tex;

	int f_save_elements_csv;
	std::string save_elements_csv_fname;

	int f_export_inversion_graphs;
	std::string export_inversion_graphs_fname;

	int f_evaluate_word;
	std::string evaluate_word_word;
	std::string evaluate_word_gens;

	int f_multiply_all_elements_in_lex_order;


	int f_stats;
	std::string stats_fname_base;

	int f_move_a_to_b;
	int move_a_to_b_a;
	int move_a_to_b_b;


	int f_rational_normal_form;
	std::string rational_normal_form_input;

	int f_find_conjugating_element;
	std::string find_conjugating_element_element_from;
	std::string find_conjugating_element_element_to;


	int f_group_of_automorphisms_by_images_of_generators;
	std::string group_of_automorphisms_by_images_of_generators_label;
	std::string group_of_automorphisms_by_images_of_generators_elements;
	std::string group_of_automorphisms_by_images_of_generators_images;




	// TABLES/group_theoretic_activity_3.tex


#if 0
	int f_multiply_elements_csv_column_major_ordering;
	std::string multiply_elements_csv_column_major_ordering_fname1;
	std::string multiply_elements_csv_column_major_ordering_fname2;
	std::string multiply_elements_csv_column_major_ordering_fname3;

	int f_multiply_elements_csv_row_major_ordering;
	std::string multiply_elements_csv_row_major_ordering_fname1;
	std::string multiply_elements_csv_row_major_ordering_fname2;
	std::string multiply_elements_csv_row_major_ordering_fname3;

	int f_apply_elements_csv_to_set;
	std::string apply_elements_csv_to_set_fname1;
	std::string apply_elements_csv_to_set_fname2;
	std::string apply_elements_csv_to_set_set;
#endif

	int f_order_of_products;
	std::string order_of_products_elements;

	int f_reverse_isomorphism_exterior_square;

	int f_is_subgroup_of;
	int f_coset_reps;


	// orbit stuff:

	int f_subgroup_lattice;

	int f_subgroup_lattice_load;
	std::string subgroup_lattice_load_fname;

	int f_subgroup_lattice_draw_by_orbits;

	int f_subgroup_lattice_draw_by_groups;

	int f_subgroup_lattice_intersection_orbit_orbit;
	int subgroup_lattice_intersection_orbit_orbit_orbit1;
	int subgroup_lattice_intersection_orbit_orbit_orbit2;

	int f_subgroup_lattice_find_overgroup_in_orbit;
	int subgroup_lattice_find_overgroup_in_orbit_orbit_global1;
	int subgroup_lattice_find_overgroup_in_orbit_group1;
	int subgroup_lattice_find_overgroup_in_orbit_orbit_global2;

	int f_subgroup_lattice_create_flag_transitive_geometry_with_partition;
	int subgroup_lattice_create_flag_transitive_geometry_with_partition_P_orbit;
	int subgroup_lattice_create_flag_transitive_geometry_with_partition_Q_orbit;
	int subgroup_lattice_create_flag_transitive_geometry_with_partition_R_orbit;
	int subgroup_lattice_create_flag_transitive_geometry_with_partition_R_group;
	int subgroup_lattice_create_flag_transitive_geometry_with_partition_intersection_size;

	int f_subgroup_lattice_create_coset_geometry;
	int subgroup_lattice_create_coset_geometry_P_orb_global;
	int subgroup_lattice_create_coset_geometry_P_group;
	int subgroup_lattice_create_coset_geometry_Q_orb_global;
	int subgroup_lattice_create_coset_geometry_Q_group;
	int subgroup_lattice_create_coset_geometry_intersection_size;


	int f_subgroup_lattice_identify_subgroup;
	std::string subgroup_lattice_identify_subgroup_subgroup_label;

#if 0
	int f_orbit_of;
	int orbit_of_point_idx;

	int f_orbits_on_set_system_from_file;
	std::string orbits_on_set_system_from_file_fname;
	int orbits_on_set_system_first_column;
	int orbits_on_set_system_number_of_columns;

	int f_orbit_of_set_from_file;
	std::string orbit_of_set_from_file_fname;
#endif

	// classification stuff:


	// classification of optimal linear codes using poset classification
	int f_linear_codes;
	std::string linear_codes_control;
	int linear_codes_minimum_distance;
	int linear_codes_target_size;


	int f_tensor_permutations;

	int f_classify_ovoids;
	apps_geometry::ovoid_classify_description *Ovoid_classify_description;

	//int f_classify_cubic_curves;

	int f_representation_on_polynomials;
	std::string representation_on_polynomials_ring;



	group_theoretic_activity_description();
	~group_theoretic_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// group_theoretic_activity.cpp
// #############################################################################


//! perform an activity associated with a linear group

class group_theoretic_activity {
public:
	group_theoretic_activity_description *Descr;

	any_group *AG;

	any_group *AG_secondary; // used in is_subgroup_of, coset_reps



	group_theoretic_activity();
	~group_theoretic_activity();
	void init_group(
			group_theoretic_activity_description *Descr,
			any_group *AG,
			int verbose_level);
	void init_secondary_group(
			group_theoretic_activity_description *Descr,
			any_group *AG_secondary,
			int verbose_level);
	void perform_activity(
			int verbose_level);

};


// #############################################################################
// modified_group_create.cpp
// #############################################################################

//! to create a new group or group action from old ones, using class group_modification_description

class modified_group_create {

public:
	group_modification_description *Descr;

	std::string label;
	std::string label_tex;


	actions::action *A_base;
	actions::action *A_previous;
	actions::action *A_modified;

	int f_has_strong_generators;
	groups::strong_generators *Strong_gens;

	groups::sims *action_on_self_by_right_multiplication_sims;
	induced_actions::action_by_right_multiplication *Action_by_right_multiplication;



	modified_group_create();
	~modified_group_create();
	void modified_group_init(
			group_modification_description *description,
			int verbose_level);
	void create_restricted_action(
			group_modification_description *description,
			int verbose_level);
	void create_action_on_k_subspaces(
			group_modification_description *description,
			int verbose_level);
	void create_action_on_k_subsets(
			group_modification_description *description,
			int verbose_level);
	void create_action_on_wedge_product(
			group_modification_description *description,
			int verbose_level);
	void create_special_subgroup(
			group_modification_description *description,
			int verbose_level);
	void create_even_subgroup(
			group_modification_description *description,
			int verbose_level);
	void create_point_stabilizer_subgroup(
			group_modification_description *description,
			int verbose_level);
	void create_set_stabilizer_subgroup(
			group_modification_description *description,
			int verbose_level);
	void create_projectivity_subgroup(
			group_modification_description *description,
			int verbose_level);
	void create_subfield_subgroup(
			group_modification_description *description,
			int verbose_level);
	void create_action_on_self_by_right_multiplication(
			group_modification_description *description,
			int verbose_level);
	void create_product_action(
			group_modification_description *description,
			int verbose_level);
	void create_polarity_extension(
			std::string &input_group_label,
			std::string &input_projective_space_label,
			int f_on_middle_layer_grassmannian,
			int f_on_points_and_hyperplanes,
			int verbose_level);
	void create_automorphism_group(
			int verbose_level);
	void create_subgroup_by_lattice(
			int orbit_index,
			int verbose_level);
	void do_stabilizer_of_variety(
			std::string &variety_label,
			int verbose_level);

};



// #############################################################################
// polynomial_ring_activity.cpp
// #############################################################################


//! a polynomial ring activity

class polynomial_ring_activity {
public:

	ring_theory::polynomial_ring_activity_description *Descr;
	ring_theory::homogeneous_polynomial_domain *HPD;



	polynomial_ring_activity();
	~polynomial_ring_activity();
	void init(
			ring_theory::polynomial_ring_activity_description *Descr,
			ring_theory::homogeneous_polynomial_domain *HPD,
			int verbose_level);
	void perform_activity(
			int verbose_level);

};


// #############################################################################
// vector_ge_builder.cpp
// #############################################################################



//! to build a vector of group elements based on class vector_ge_description


class vector_ge_builder {

public:

	data_structures_groups::vector_ge_description *Descr;

	data_structures_groups::vector_ge *V;


	vector_ge_builder();
	~vector_ge_builder();
	void init(
			data_structures_groups::vector_ge_description *Descr,
			int verbose_level);

};








// #############################################################################
// young.cpp
// #############################################################################


//! The Young representations of the symmetric group


class young {
public:
	int n;
	actions::action *A;
	groups::sims *S;
	ring_theory::longinteger_object go;
	int goi;
	int *Elt;
	int *v;

	actions::action *Aconj;
	induced_actions::action_by_conjugation *ABC;
	groups::schreier *Sch;
	groups::strong_generators *SG;
	int nb_classes;
	int *class_size;
	int *class_rep;
	algebra::a_domain *D;

	int l1, l2;
	int *row_parts;
	int *col_parts;
	int *Tableau;

	data_structures::set_of_sets *Row_partition;
	data_structures::set_of_sets *Col_partition;

	data_structures_groups::vector_ge *gens1, *gens2;
	groups::sims *S1, *S2;


	young();
	~young();
	void init(
			int n, int verbose_level);
	void create_module(
			int *h_alpha,
		int *&Base, int *&base_cols, int &rk, 
		int verbose_level);
	void create_representations(
			int *Base, int *Base_inv, int rk,
		int verbose_level);
	void create_representation(
			int *Base, int *base_cols, int rk,
		int group_elt, int *Mtx, int verbose_level);
		// Mtx[rk * rk * D->size_of_instance_in_int]
	void young_symmetrizer(
			int *row_parts, int nb_row_parts,
		int *tableau, 
		int *elt1, int *elt2, int *elt3, 
		int verbose_level);
	void compute_generators(
			int &go1, int &go2, int verbose_level);
	void Maschke(
			int *Rep,
		int dim_of_module, int dim_of_submodule, 
		int *&Mu, 
		int verbose_level);
	long int group_ring_element_size(
			actions::action *A, groups::sims *S);
	void group_ring_element_create(
			actions::action *A, groups::sims *S, int *&elt);
	void group_ring_element_free(
			actions::action *A, groups::sims *S, int *elt);
	void group_ring_element_print(
			actions::action *A, groups::sims *S, int *elt);
	void group_ring_element_copy(
			actions::action *A, groups::sims *S,
		int *elt_from, int *elt_to);
	void group_ring_element_zero(
			actions::action *A, groups::sims *S,
		int *elt);
	void group_ring_element_mult(
			actions::action *A, groups::sims *S,
		int *elt1, int *elt2, int *elt3);
};


}}}


#endif /* ORBITER_SRC_LIB_TOP_LEVEL_ALGEBRA_AND_NUMBER_THEORY_TL_ALGEBRA_AND_NUMBER_THEORY_H_ */


