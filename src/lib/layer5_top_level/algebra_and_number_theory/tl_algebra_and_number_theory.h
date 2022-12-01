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


//! description of an action of forms


class action_on_forms_activity_description {

public:

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

	//any_group *AG_secondary; // used in is_subgroup_of, coset_reps



	action_on_forms_activity();
	~action_on_forms_activity();
	void init(action_on_forms_activity_description *Descr,
			apps_algebra::action_on_forms *AF,
			int verbose_level);
	void perform_activity(int verbose_level);
	void do_algebraic_normal_form(int verbose_level);
	void do_orbits_on_functions(int verbose_level);
	void do_associated_set_in_plane(int verbose_level);
	void do_differential_uniformity(int verbose_level);

};



// #############################################################################
// action_on_forms_description.cpp
// #############################################################################


//! description of an action of forms


class action_on_forms_description {

public:

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


//! an action of forms


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
	void algebraic_normal_form(int *func, int len, int verbose_level);
	void orbits_on_functions(int *The_functions, int nb_functions, int len,
			int verbose_level);
	void orbits_on_equations(int *The_equations, int nb_equations, int len,
			groups::schreier *&Orb,
			actions::action *&A_on_equations,
			int verbose_level);
	void associated_set_in_plane(int *func, int len,
			long int *&Rk, int verbose_level);
	void differential_uniformity(int *func, int len, int verbose_level);

};


// #############################################################################
// algebra_global_with_action.cpp
// #############################################################################

//! group theoretic functions which require an action


class algebra_global_with_action {
public:
	void orbits_under_conjugation(
			long int *the_set, int set_size, groups::sims *S,
			groups::strong_generators *SG,
			data_structures_groups::vector_ge *Transporter,
			int verbose_level);
	void create_subgroups(
			groups::strong_generators *SG,
			long int *the_set, int set_size, groups::sims *S,
			actions::action *A_conj,
			groups::schreier *Classes,
			data_structures_groups::vector_ge *Transporter,
			int verbose_level);
	void compute_orbit_of_set(
			long int *the_set, int set_size,
			actions::action *A1, actions::action *A2,
			data_structures_groups::vector_ge *gens,
			std::string &label_set,
			std::string &label_group,
			long int *&Table,
			int &orbit_length,
			int verbose_level);
	void conjugacy_classes_based_on_normal_forms(actions::action *A,
			groups::sims *override_Sims,
			std::string &label,
			std::string &label_tex,
			int verbose_level);
	void classes_GL(field_theory::finite_field *F, int d, int f_no_eigenvalue_one, int verbose_level);
	void do_normal_form(int q, int d,
			int f_no_eigenvalue_one, int *data, int data_sz,
			int verbose_level);
	void do_identify_one(int q, int d,
			int f_no_eigenvalue_one, int elt_idx,
			int verbose_level);
	void do_identify_all(int q, int d,
			int f_no_eigenvalue_one, int verbose_level);
	void do_random(int q, int d, int f_no_eigenvalue_one, int verbose_level);
	void group_table(int q, int d, int f_poly, std::string &poly,
			int f_no_eigenvalue_one, int verbose_level);
	void centralizer_brute_force(int q, int d,
			int elt_idx, int verbose_level);
	void centralizer(int q, int d,
			int elt_idx, int verbose_level);
	// creates a finite_field, and two actions
	// using init_projective_group and init_general_linear_group
	void centralizer(int q, int d, int verbose_level);
	void compute_regular_representation(actions::action *A, groups::sims *S,
			data_structures_groups::vector_ge *SG,
			int *&perm, int verbose_level);
	void presentation(actions::action *A,
			groups::sims *S, int goi,
			data_structures_groups::vector_ge *gens,
		int *primes, int verbose_level);

	void do_eigenstuff(field_theory::finite_field *F, int size, int *Data, int verbose_level);
	void A5_in_PSL_(int q, int verbose_level);
	void A5_in_PSL_2_q(int q,
			layer2_discreta::discreta_matrix & A,
			layer2_discreta::discreta_matrix & B,
			layer2_discreta::domain *dom_GFq, int verbose_level);
	void A5_in_PSL_2_q_easy(int q,
			layer2_discreta::discreta_matrix & A,
			layer2_discreta::discreta_matrix & B,
			layer2_discreta::domain *dom_GFq,
			int verbose_level);
	void A5_in_PSL_2_q_hard(int q,
			layer2_discreta::discreta_matrix & A,
			layer2_discreta::discreta_matrix & B,
			layer2_discreta::domain *dom_GFq,
			int verbose_level);
	int proj_order(layer2_discreta::discreta_matrix &A);
	void trace(layer2_discreta::discreta_matrix &A, layer2_discreta::discreta_base &tr);
	void elementwise_power_int(layer2_discreta::discreta_matrix &A, int k);
	int is_in_center(layer2_discreta::discreta_matrix &B);
	void matrix_convert_to_numerical(layer2_discreta::discreta_matrix &A, int *AA, int q);


	void young_symmetrizer(int n, int verbose_level);
	void young_symmetrizer_sym_4(int verbose_level);
	void report_tactical_decomposition_by_automorphism_group(
			std::ostream &ost, geometry::projective_space *P,
			actions::action *A_on_points, actions::action *A_on_lines,
			groups::strong_generators *gens, int size_limit_for_printing,
			int verbose_level);
	void linear_codes_with_bounded_minimum_distance(
			poset_classification::poset_classification_control *Control,
			groups::linear_group *LG,
			int d, int target_depth, int verbose_level);
	void centralizer_of_element(
			actions::action *A, groups::sims *S,
			std::string &element_description,
			std::string &label, int verbose_level);
	void permutation_representation_of_element(
			actions::action *A,
			std::string &element_description,
			int verbose_level);
	void normalizer_of_cyclic_subgroup(
			actions::action *A, groups::sims *S,
			std::string &element_description,
			std::string &label, int verbose_level);
	void find_subgroups(
			actions::action *A, groups::sims *S,
			int subgroup_order,
			std::string &label,
			int &nb_subgroups,
			groups::strong_generators *&H_gens,
			groups::strong_generators *&N_gens,
			int verbose_level);
	void relative_order_vector_of_cosets(
			actions::action *A, groups::strong_generators *SG,
			data_structures_groups::vector_ge *cosets,
			int *&relative_order_table, int verbose_level);
#if 0
	void do_orbits_on_polynomials(
			groups::linear_group *LG,
			int degree_of_poly,
			int f_recognize, std::string &recognize_text,
			int f_draw_tree, int draw_tree_idx,
			graphics::layered_graph_draw_options *Opt,
			int verbose_level);
#endif
	void representation_on_polynomials(
			groups::linear_group *LG,
			int degree_of_poly,
			int verbose_level);

	void do_eigenstuff_with_coefficients(
			field_theory::finite_field *F, int n, std::string &coeffs_text, int verbose_level);
	void do_eigenstuff_from_file(
			field_theory::finite_field *F, int n, std::string &fname, int verbose_level);

	void orbits_on_points(
			actions::action *A2,
			groups::strong_generators *Strong_gens,
			int f_load_save,
			std::string &prefix,
			groups::orbits_on_something *&Orb,
			int verbose_level);
	void find_singer_cycle(any_group *Any_group,
			actions::action *A1, actions::action *A2,
			int verbose_level);
	void search_element_of_order(any_group *Any_group,
			actions::action *A1, actions::action *A2,
			int order, int verbose_level);
	void find_standard_generators(any_group *Any_group,
			actions::action *A1, actions::action *A2,
			int order_a, int order_b, int order_ab, int verbose_level);
	void Nth_roots(field_theory::finite_field *F,
			int n, int verbose_level);

};



// #############################################################################
// any_group.cpp
// #############################################################################

//! a wrapper for linear_group and permutation_group_create

class any_group {

public:

	int f_linear_group;
	groups::linear_group *LG;

	int f_permutation_group;
	groups::permutation_group_create *PGC;

	int f_modified_group;
	modified_group_create *MGC;

	actions::action *A_base;
	actions::action *A;

	std::string label;
	std::string label_tex;

	groups::strong_generators *Subgroup_gens;
	groups::sims *Subgroup_sims;

	any_group();
	~any_group();
	void init_linear_group(groups::linear_group *LG, int verbose_level);
	void init_permutation_group(groups::permutation_group_create *PGC, int verbose_level);
	void init_modified_group(modified_group_create *MGC, int verbose_level);
	void create_latex_report(
			graphics::layered_graph_draw_options *O,
			int f_sylow, int f_group_table, int f_classes,
			int verbose_level);
	void export_group_table(int verbose_level);
	void do_export_orbiter(actions::action *A2, int verbose_level);
	void do_export_gap(int verbose_level);
	void do_export_magma(int verbose_level);
	void do_canonical_image_GAP(std::string &input_set, int verbose_level);
	void create_group_table(int *&Table, long int &n, int verbose_level);
	void normalizer(int verbose_level);
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
	void print_elements(int verbose_level);
	void print_elements_tex(int f_with_permutation,
			int f_override_action, actions::action *A_special,
			int verbose_level);
	void order_of_products_of_elements(
			std::string &Elements_text,
			int verbose_level);
	void save_elements_csv(std::string &fname, int verbose_level);
	void export_inversion_graphs(std::string &fname, int verbose_level);
	void multiply_elements_csv(std::string &fname1, std::string &fname2, std::string &fname3,
			int f_column_major_ordering, int verbose_level);
	void apply_elements_to_set_csv(std::string &fname1, std::string &fname2,
			std::string &set_text,
			int verbose_level);
	void element_rank(std::string &elt_data, int verbose_level);
	void element_unrank(std::string &rank_string, int verbose_level);
	void conjugacy_class_of(std::string &rank_string, int verbose_level);
	void do_reverse_isomorphism_exterior_square(int verbose_level);

	void orbits_on_set_system_from_file(std::string &fname_csv,
			int number_of_columns, int first_column, int verbose_level);
	void orbits_on_set_from_file(std::string &fname_csv, int verbose_level);
	void orbit_of(int point_idx, int verbose_level);
	void orbits_on_points(groups::orbits_on_something *&Orb, int verbose_level);

	void create_latex_report_for_permutation_group(
			graphics::layered_graph_draw_options *O,
			int verbose_level);
	void create_latex_report_for_modified_group(
			graphics::layered_graph_draw_options *O,
			int verbose_level);
	groups::strong_generators *get_strong_generators();
	int is_subgroup_of(any_group *AG_secondary, int verbose_level);
	void set_of_coset_representatives(any_group *AG_secondary,
			data_structures_groups::vector_ge *&coset_reps,
			int verbose_level);
	void report_coset_reps(
			data_structures_groups::vector_ge *coset_reps,
			int verbose_level);

	// any_group_linear.cpp:
	void classes_based_on_normal_form(int verbose_level);
	void classes(int verbose_level);
	void find_singer_cycle(int verbose_level);
	void search_element_of_order(int order, int verbose_level);
	void find_standard_generators(int order_a,
			int order_b,
			int order_ab,
			int verbose_level);
	void isomorphism_Klein_quadric(std::string &fname, int verbose_level);
	void do_orbits_on_subspaces(
			poset_classification::poset_classification_control *Control,
			orbits_on_subspaces *&OoS,
			int depth, int verbose_level);
	void do_tensor_classify(
			std::string &control_label,
			apps_geometry::tensor_classify *&T,
			int depth, int verbose_level);
	void do_tensor_permutations(int verbose_level);
	void do_linear_codes(
			std::string &control_label,
			int minimum_distance,
			int target_size, int verbose_level);
	void do_classify_ovoids(
			apps_geometry::ovoid_classify_description *Ovoid_classify_description,
			int verbose_level);
	int subspace_orbits_test_set(
			int len, long int *S, int verbose_level);


	// any_group_orbits.cpp
	void orbits_on_subsets(
			poset_classification::poset_classification_control *Control,
			poset_classification::poset_classification *&PC,
			int subset_size,
			int verbose_level);
	void orbits_on_poset_post_processing(
			poset_classification::poset_classification *PC,
			int depth,
			int verbose_level);
	void do_conjugacy_class_of_element(
			std::string &elt_label, std::string &elt_text, int verbose_level);
	void do_orbits_on_group_elements_under_conjugation(
			std::string &fname_group_elements_coded,
			std::string &fname_transporter,
			int verbose_level);

};


// #############################################################################
// character_table_burnside.cpp
// #############################################################################

//! character table of a finite group using the algorithm of Burnside


class character_table_burnside {
public:

	void do_it(int n, int verbose_level);
	void create_matrix(discreta_matrix &M, int i, int *S, int nb_classes,
		int *character_degree, int *class_size,
		int verbose_level);
	void compute_character_table(
			algebra::a_domain *D, int nb_classes, int *Omega,
		int *character_degree, int *class_size,
		int *&character_table, int verbose_level);
	void compute_character_degrees(algebra::a_domain *D,
		int goi, int nb_classes, int *Omega, int *class_size,
		int *&character_degree, int verbose_level);
	void compute_omega(algebra::a_domain *D, int *N0, int nb_classes,
			int *Mu, int nb_mu, int *&Omega, int verbose_level);
	int compute_r0(int *N, int nb_classes, int verbose_level);
	void compute_multiplication_constants_center_of_group_ring(
			actions::action *A,
			induced_actions::action_by_conjugation *ABC,
		groups::schreier *Sch, int nb_classes, int *&N, int verbose_level);
	void compute_Distribution_table(actions::action *A, induced_actions::action_by_conjugation *ABC,
			groups::schreier *Sch, int nb_classes,
		int **Gens, int nb_gens, int t_max, int *&Distribution, int verbose_level);
	void multiply_word(actions::action *A, int **Gens,
			int *Choice, int t, int *Elt1, int *Elt2, int verbose_level);
	void create_generators(actions::action *A, int n,
			int **&Elt, int &nb_gens, int f_special, int verbose_level);
	void integral_eigenvalues(int *M, int n,
		int *&Lambda,
		int &nb_lambda,
		int *&Mu,
		int *&Mu_mult,
		int &nb_mu,
		int verbose_level);
	void characteristic_poly(int *N, int size, unipoly &charpoly, int verbose_level);
	void double_swap(double &a, double &b);
	int double_Gauss(double *A, int m, int n, int *base_cols, int verbose_level);
	void double_matrix_print(double *A, int m, int n);
	double double_abs(double x);
	void kernel_columns(int n, int nb_base_cols, int *base_cols, int *kernel_cols);
	void matrix_get_kernel(double *M, int m, int n, int *base_cols, int nb_base_cols,
		int &kernel_m, int &kernel_n, double *kernel);
	int double_as_int(double x);

};


// #############################################################################
// group_modification_description.cpp
// #############################################################################

//! create a new group or group action from an old

class group_modification_description {

public:

	int f_restricted_action;
	std::string restricted_action_set_text;

	int f_on_k_subspaces;
	int on_k_subspaces_k;

	int f_on_k_subsets;
	int on_k_subsets_k;

	int f_on_wedge_product;

	int f_create_special_subgroup;

	int f_point_stabilizer;
	int point_stabilizer_index;

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

	int f_apply;
	std::string apply_input;
	std::string apply_element;

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

	int f_find_singer_cycle;


	int f_classes_based_on_normal_form;

	// Magma:
	int f_normalizer;

	// Magma:
	int f_centralizer_of_element;
	std::string element_description_text;
	std::string element_label;

	int f_permutation_representation_of_element;
	std::string permutation_representation_element_text;

	int f_conjugacy_class_of_element;
	// uses element_description_text and element_label

	int f_orbits_on_group_elements_under_conjugation;
	std::string orbits_on_group_elements_under_conjugation_fname;
	std::string orbits_on_group_elements_under_conjugation_transporter_fname;

	// Magma:
	int f_normalizer_of_cyclic_subgroup;

	// Magma:
	int f_classes;

	int f_find_subgroup;
	int find_subgroup_order;

	int f_report;

		// flags that apply to report:
		int f_report_sylow;
		int f_report_group_table;
		int f_report_classes;


	int f_export_group_table;

	int f_test_if_geometric;
	int test_if_geometric_depth;

	int f_conjugacy_class_of;
	std::string conjugacy_class_of_data;

	int f_isomorphism_Klein_quadric;
	std::string isomorphism_Klein_quadric_fname;

	int f_print_elements;
	int f_print_elements_tex;

	int f_save_elements_csv;
	std::string save_elements_csv_fname;

	int f_export_inversion_graphs;
	std::string export_inversion_graphs_fname;


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


	int f_order_of_products;
	std::string order_of_products_elements;

	int f_reverse_isomorphism_exterior_square;

	int f_is_subgroup_of;
	int f_coset_reps;






	int f_orbit_of;
	int orbit_of_point_idx;

	int f_orbits_on_set_system_from_file;
	std::string orbits_on_set_system_from_file_fname;
	int orbits_on_set_system_first_column;
	int orbits_on_set_system_number_of_columns;

	int f_orbit_of_set_from_file;
	std::string orbit_of_set_from_file_fname;

	// classification of optimal linear codes using poset classification
	int f_linear_codes;
	std::string linear_codes_control;
	int linear_codes_minimum_distance;
	int linear_codes_target_size;




	int f_tensor_permutations;

	int f_classify_ovoids;
	apps_geometry::ovoid_classify_description *Ovoid_classify_description;

	int f_classify_cubic_curves;

	int f_representation_on_polynomials;
	int representation_on_polynomials_degree;



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
	void init_group(group_theoretic_activity_description *Descr,
			any_group *AG,
			int verbose_level);
	void init_secondary_group(group_theoretic_activity_description *Descr,
			any_group *AG_secondary,
			int verbose_level);
	void perform_activity(int verbose_level);
	void apply(int verbose_level);
	void multiply(int verbose_level);
	void inverse(int verbose_level);
	void raise_to_the_power(int verbose_level);

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

	//strong_generators *initial_strong_gens;

	actions::action *A_base;
	actions::action *A_previous;
	actions::action *A_modified;

	int f_has_strong_generators;
	groups::strong_generators *Strong_gens;


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
	void create_point_stabilizer_subgroup(
			group_modification_description *description,
			int verbose_level);

};


// #############################################################################
// orbit_cascade.cpp
// #############################################################################

//! a cascade of nested orbit algorithms


class orbit_cascade {
public:

	int N; // total size of the set
	int k; // size of the subsets

	// we assume that N = 3 * k

	any_group *G;

	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Primary_poset;
	poset_classification::poset_classification *Orbits_on_primary_poset;

	int number_primary_orbits;
	groups::strong_generators **stabilizer_gens; // [number_primary_orbits]
	long int *Reps_and_complements; // [number_primary_orbits * degree]

	actions::action **A_restricted;
	poset_classification::poset_with_group_action **Secondary_poset; // [number_primary_orbits]
	poset_classification::poset_classification **orbits_secondary_poset; // [number_primary_orbits]

	int *nb_orbits_secondary; // [number_primary_orbits]
	int *flag_orbit_first; // [number_primary_orbits]
	int nb_orbits_secondary_total;

	invariant_relations::flag_orbits *Flag_orbits;

	int nb_orbits_reduced;

	invariant_relations::classification_step *Partition_orbits; // [nb_orbits_reduced]


	orbit_cascade();
	~orbit_cascade();
	void init(int N, int k, any_group *G,
			std::string &Control_label,
			int verbose_level);
	void downstep(int verbose_level);
	void upstep(std::vector<long int> &Ago, int verbose_level);

};



// #############################################################################
// orbits_activity_description.cpp
// #############################################################################


//! description of an action for orbits


class orbits_activity_description {

public:


	int f_report;

	int f_export_something;
	std::string export_something_what;
	int export_something_data1;

	int f_export_trees;

	int f_draw_tree;
	int draw_tree_idx;

	int f_stabilizer;
	int stabilizer_point;

	int f_stabilizer_of_orbit_rep;
	int stabilizer_of_orbit_rep_orbit_idx;

	int f_Kramer_Mesner_matrix;
	int Kramer_Mesner_t;
	int Kramer_Mesner_k;

	int f_recognize;
	std::vector<std::string> recognize;

	int f_report_options;
	poset_classification::poset_classification_report_options *report_options;


	orbits_activity_description();
	~orbits_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// orbits_activity.cpp
// #############################################################################


//! perform an activity associated with orbits

class orbits_activity {
public:
	orbits_activity_description *Descr;

	apps_algebra::orbits_create *OC;




	orbits_activity();
	~orbits_activity();
	void init(orbits_activity_description *Descr,
			apps_algebra::orbits_create *OC,
			int verbose_level);
	void perform_activity(int verbose_level);
	void do_report(int verbose_level);
	void do_export(int verbose_level);
	void do_export_trees(int verbose_level);
	void do_draw_tree(int verbose_level);
	void do_stabilizer(int verbose_level);
	void do_stabilizer_of_orbit_rep(int verbose_level);
	void do_Kramer_Mesner_matrix(int verbose_level);
	void do_recognize(int verbose_level);

};






// #############################################################################
// orbits_create_description.cpp
// #############################################################################

//! to describe an orbit problem



class orbits_create_description {

public:

	int f_group;
	std::string group_label;

	int f_on_points;

	int f_on_subsets;
	int on_subsets_size;
	std::string on_subsets_poset_classification_control_label;

	int f_on_subspaces;
	int on_subspaces_dimension;
	std::string on_subspaces_poset_classification_control_label;

	int f_on_tensors;
	int on_tensors_dimension;
	std::string on_tensors_poset_classification_control_label;

	int f_on_partition;
	int on_partition_k;
	std::string on_partition_poset_classification_control_label;

	int f_on_polynomials;
	int on_polynomials_degree;

#if 0
	int f_draw_tree;
	int draw_tree_idx;

	int f_recognize;
	std::string recognize_text;
#endif

	orbits_create_description();
	~orbits_create_description();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// orbits_create.cpp
// #############################################################################

//! to create orbits



class orbits_create {

public:
	orbits_create_description *Descr;

	any_group *Group;

	std::string prefix;
	std::string label_txt;
	std::string label_tex;

	int f_has_Orb;
	groups::orbits_on_something *Orb;

	int f_has_On_subsets;
	poset_classification::poset_classification *On_subsets;

	int f_has_On_Subspaces;
	orbits_on_subspaces *On_Subspaces;

	int f_has_On_tensors;
	apps_geometry::tensor_classify *On_tensors;

	int f_has_Cascade;
	apps_algebra::orbit_cascade *Cascade;

	int f_has_On_polynomials;
	orbits_on_polynomials *On_polynomials;

	orbits_create();
	~orbits_create();
	void init(apps_algebra::orbits_create_description *Descr, int verbose_level);

};




// #############################################################################
// orbits_on_polynomials.cpp
// #############################################################################


//! orbits of a group on polynomials using Schreier vectors

class orbits_on_polynomials {
public:

	groups::linear_group *LG;
	int degree_of_poly;

	field_theory::finite_field *F;
	actions::action *A;
	int n;
	ring_theory::longinteger_object go;

	ring_theory::homogeneous_polynomial_domain *HPD;

	geometry::projective_space *P;

	actions::action *A2;

	int *Elt1;
	int *Elt2;
	int *Elt3;

	groups::schreier *Sch;
	ring_theory::longinteger_object full_go;

	std::string fname_base;
	std::string fname_csv;
	std::string fname_report;

	data_structures_groups::orbit_transversal *T;
	int *Nb_pts; // [T->nb_orbits]
	std::vector<std::vector<long int> > Points;


	orbits_on_polynomials();
	~orbits_on_polynomials();
	void init(
			groups::linear_group *LG,
			int degree_of_poly,
			//int f_recognize, std::string &recognize_text,
			int verbose_level);
	void compute_points(int verbose_level);
	void report(int verbose_level);
	void report_detailed_list(std::ostream &ost,
			int verbose_level);
	void export_something(std::string &what, int data1,
			std::string &fname, int verbose_level);
	void export_something_worker(
			std::string &fname_base,
			std::string &what, int data1,
			std::string &fname,
			int verbose_level);


};



// #############################################################################
// orbits_on_subspaces.cpp
// #############################################################################


//! orbits of a group on subspaces of a vector space

class orbits_on_subspaces {
public:
	//group_theoretic_activity *GTA;
	apps_algebra::any_group *Group;

	// local data for orbits on subspaces:
	poset_classification::poset_with_group_action *orbits_on_subspaces_Poset;
	poset_classification::poset_classification *orbits_on_subspaces_PC;
	algebra::vector_space *orbits_on_subspaces_VS;
	int *orbits_on_subspaces_M;
	int *orbits_on_subspaces_base_cols;


	orbits_on_subspaces();
	~orbits_on_subspaces();
	void init(
			apps_algebra::any_group *Group,
			poset_classification::poset_classification_control *Control,
			int depth,
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
	void perform_activity(int verbose_level);

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
	void init(data_structures_groups::vector_ge_description *Descr,
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
	void init(int n, int verbose_level);
	void create_module(int *h_alpha, 
		int *&Base, int *&base_cols, int &rk, 
		int verbose_level);
	void create_representations(int *Base, int *Base_inv, int rk, 
		int verbose_level);
	void create_representation(int *Base, int *base_cols, int rk, 
		int group_elt, int *Mtx, int verbose_level);
		// Mtx[rk * rk * D->size_of_instance_in_int]
	void young_symmetrizer(int *row_parts, int nb_row_parts, 
		int *tableau, 
		int *elt1, int *elt2, int *elt3, 
		int verbose_level);
	void compute_generators(int &go1, int &go2, int verbose_level);
	void Maschke(int *Rep, 
		int dim_of_module, int dim_of_submodule, 
		int *&Mu, 
		int verbose_level);
	long int group_ring_element_size(actions::action *A, groups::sims *S);
	void group_ring_element_create(actions::action *A, groups::sims *S, int *&elt);
	void group_ring_element_free(actions::action *A, groups::sims *S, int *elt);
	void group_ring_element_print(actions::action *A, groups::sims *S, int *elt);
	void group_ring_element_copy(actions::action *A, groups::sims *S,
		int *elt_from, int *elt_to);
	void group_ring_element_zero(actions::action *A, groups::sims *S,
		int *elt);
	void group_ring_element_mult(actions::action *A, groups::sims *S,
		int *elt1, int *elt2, int *elt3);
};


}}}


#endif /* ORBITER_SRC_LIB_TOP_LEVEL_ALGEBRA_AND_NUMBER_THEORY_TL_ALGEBRA_AND_NUMBER_THEORY_H_ */


