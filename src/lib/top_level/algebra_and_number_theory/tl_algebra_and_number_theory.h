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
namespace top_level {

// #############################################################################
// algebra_global_with_action.cpp
// #############################################################################

//! group theoretic functions which require an action


class algebra_global_with_action {
public:
	void orbits_under_conjugation(
			long int *the_set, int set_size, sims *S,
			strong_generators *SG,
			vector_ge *Transporter,
			int verbose_level);
	void create_subgroups(
			strong_generators *SG,
			long int *the_set, int set_size, sims *S, action *A_conj,
			schreier *Classes,
			vector_ge *Transporter,
			int verbose_level);
	void orbits_on_set_from_file(
			long int *the_set, int set_size,
			action *A1, action *A2,
			vector_ge *gens,
			std::string &label_set,
			std::string &label_group,
			long int *&Table,
			int &orbit_length,
			int verbose_level);
	void conjugacy_classes_based_on_normal_forms(action *A,
			sims *override_Sims,
			std::string &label,
			std::string &label_tex,
			int verbose_level);
	void classes_GL(finite_field *F, int d, int f_no_eigenvalue_one, int verbose_level);
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
	void centralizer(int q, int d, int verbose_level);
	void compute_regular_representation(action *A, sims *S,
		vector_ge *SG, int *&perm, int verbose_level);
	void presentation(action *A, sims *S, int goi, vector_ge *gens,
		int *primes, int verbose_level);

	void do_eigenstuff(finite_field *F, int size, int *Data, int verbose_level);
	void A5_in_PSL_(int q, int verbose_level);
	void A5_in_PSL_2_q(int q,
			discreta_matrix & A, discreta_matrix & B, domain *dom_GFq, int verbose_level);
	void A5_in_PSL_2_q_easy(int q,
			discreta_matrix & A, discreta_matrix & B, domain *dom_GFq, int verbose_level);
	void A5_in_PSL_2_q_hard(int q,
			discreta_matrix & A, discreta_matrix & B, domain *dom_GFq, int verbose_level);
	int proj_order(discreta_matrix &A);
	void trace(discreta_matrix &A, discreta_base &tr);
	void elementwise_power_int(discreta_matrix &A, int k);
	int is_in_center(discreta_matrix &B);
	void matrix_convert_to_numerical(discreta_matrix &A, int *AA, int q);


	void young_symmetrizer(int n, int verbose_level);
	void young_symmetrizer_sym_4(int verbose_level);
	void report_tactical_decomposition_by_automorphism_group(
			std::ostream &ost, projective_space *P,
			action *A_on_points, action *A_on_lines,
			strong_generators *gens, int size_limit_for_printing,
			int verbose_level);
	void linear_codes_with_bounded_minimum_distance(
			poset_classification_control *Control, linear_group *LG,
			int d, int target_depth, int verbose_level);
	void centralizer_of_element(
			action *A, sims *S,
			std::string &element_description,
			std::string &label, int verbose_level);
	void normalizer_of_cyclic_subgroup(
			action *A, sims *S,
			std::string &element_description,
			std::string &label, int verbose_level);
	void find_subgroups(
			action *A, sims *S,
			int subgroup_order,
			std::string &label,
			int &nb_subgroups,
			strong_generators *&H_gens,
			strong_generators *&N_gens,
			int verbose_level);
	void relative_order_vector_of_cosets(
			action *A, strong_generators *SG,
			vector_ge *cosets, int *&relative_order_table, int verbose_level);
	void do_orbits_on_polynomials(
			linear_group *LG,
			int degree_of_poly,
			int f_recognize, std::string &recognize_text,
			int f_draw_tree, int draw_tree_idx, layered_graph_draw_options *Opt,
			int verbose_level);
	void representation_on_polynomials(
			linear_group *LG,
			int degree_of_poly,
			int verbose_level);

	void do_eigenstuff_with_coefficients(
			finite_field *F, int n, std::string &coeffs_text, int verbose_level);
	void do_eigenstuff_from_file(
			finite_field *F, int n, std::string &fname, int verbose_level);

	void orbits_on_points(
			//linear_group *LG,
			action *A2,
			strong_generators *Strong_gens,
			int f_load_save,
			std::string &prefix,
			orbits_on_something *&Orb,
			int verbose_level);
	void find_singer_cycle(linear_group *LG,
			action *A1, action *A2,
			int verbose_level);
	void search_element_of_order(linear_group *LG,
			action *A1, action *A2,
			int order, int verbose_level);
	void find_standard_generators(linear_group *LG,
			action *A1, action *A2,
			int order_a, int order_b, int order_ab, int verbose_level);
	void element_rank(linear_group *LG,
			action *A1,
			std::string &elt_data, int verbose_level);
	void element_unrank(linear_group *LG,
			action *A1,
			std::string &rank_string, int verbose_level);

};



// #############################################################################
// any_group.cpp
// #############################################################################

//! a wrapper for linear_group and permutation_group_create

class any_group {

public:

	int f_linear_group;
	linear_group *LG;

	int f_permutation_group;
	permutation_group_create *PGC;

	int f_modified_group;
	modified_group_create *MGC;

	action *A_base;
	action *A;

	std::string label;
	std::string label_tex;

	strong_generators *Subgroup_gens;
	sims *Subgroup_sims;

	any_group();
	~any_group();
	void init_linear_group(linear_group *LG, int verbose_level);
	void init_permutation_group(permutation_group_create *PGC, int verbose_level);
	void init_modified_group(modified_group_create *MGC, int verbose_level);
	void create_latex_report(
			layered_graph_draw_options *O,
			int f_sylow, int f_group_table, int f_classes,
			int verbose_level);
	void do_export_orbiter(action *A2, int verbose_level);
	void do_export_gap(int verbose_level);
	void do_export_magma(int verbose_level);
	void do_canonical_image_GAP(std::string &input_set, int verbose_level);
	void create_group_table(int verbose_level);
	void normalizer(int verbose_level);
	void centralizer(
			std::string &element_label,
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
	void print_elements_tex(int f_order_of_products, std::string &elements, int verbose_level);
	void save_elements_csv(std::string &fname, int verbose_level);
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
	void orbits_on_points(orbits_on_something *&Orb, int verbose_level);
	void orbits_on_subsets(poset_classification_control *Control,
			poset_classification *&PC,
			int subset_size,
			int verbose_level);
	void orbits_on_poset_post_processing(
			poset_classification *PC,
			int depth,
			int verbose_level);
	void do_conjugacy_class_of_element(
			std::string &elt_label, std::string &elt_text, int verbose_level);
	void do_orbits_on_group_elements_under_conjugation(
			std::string &fname_group_elements_coded,
			std::string &fname_transporter,
			int verbose_level);
	void create_latex_report_for_permutation_group(
			layered_graph_draw_options *O,
			int verbose_level);
	void create_latex_report_for_modified_group(
			layered_graph_draw_options *O,
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
	void do_orbits_on_subspaces(group_theoretic_activity *GTA,
			poset_classification_control *Control, int depth, int verbose_level);
	void do_tensor_classify(poset_classification_control *Control, int depth, int verbose_level);
	void do_tensor_permutations(int verbose_level);
	void do_linear_codes(poset_classification_control *Control,
			int minimum_distance,
			int target_size, int verbose_level);
	void do_classify_ovoids(
			poset_classification_control *Control,
			ovoid_classify_description *Ovoid_classify_description,
			int verbose_level);
	int subspace_orbits_test_set(
			int len, long int *S, int verbose_level);


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
	void compute_character_table(a_domain *D, int nb_classes, int *Omega,
		int *character_degree, int *class_size,
		int *&character_table, int verbose_level);
	void compute_character_degrees(a_domain *D,
		int goi, int nb_classes, int *Omega, int *class_size,
		int *&character_degree, int verbose_level);
	void compute_omega(a_domain *D, int *N0, int nb_classes,
			int *Mu, int nb_mu, int *&Omega, int verbose_level);
	int compute_r0(int *N, int nb_classes, int verbose_level);
	void compute_multiplication_constants_center_of_group_ring(action *A,
		action_by_conjugation *ABC,
		schreier *Sch, int nb_classes, int *&N, int verbose_level);
	void compute_Distribution_table(action *A, action_by_conjugation *ABC,
		schreier *Sch, int nb_classes,
		int **Gens, int nb_gens, int t_max, int *&Distribution, int verbose_level);
	void multiply_word(action *A, int **Gens,
			int *Choice, int t, int *Elt1, int *Elt2, int verbose_level);
	void create_generators(action *A, int n,
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


//! description of an activity associated with a linear group

class group_theoretic_activity_description {
public:

	int f_multiply;
	std::string multiply_a;
	std::string multiply_b;

	int f_inverse;
	std::string inverse_a;

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




	int f_poset_classification_control;
	poset_classification_control *Control;


	int f_orbits_on_points;
	int f_export_trees;

	int f_stabilizer;
		// compute stabilizer of orbit 0,
		// must be given with -orbits_on_points


	int f_orbits_on_subsets;
	int orbits_on_subsets_size;
	int f_classes_based_on_normal_form;
	int f_classes;
	int f_group_table;
	int f_normalizer;
	int f_centralizer_of_element;
	std::string element_description_text;
	std::string element_label;

	int f_conjugacy_class_of_element;
	// uses element_description_text and element_label

	int f_orbits_on_group_elements_under_conjugation;
	std::string orbits_on_group_elements_under_conjugation_fname;
	std::string orbits_on_group_elements_under_conjugation_transporter_fname;

	int f_normalizer_of_cyclic_subgroup;
	int f_find_subgroup;
	int find_subgroup_order;
	int f_report;
	int f_sylow;
	int f_test_if_geometric;
	int test_if_geometric_depth;
	int f_draw_tree;

	int f_orbit_of;
	int orbit_of_point_idx;

	int f_orbits_on_set_system_from_file;
	std::string orbits_on_set_system_from_file_fname;
	int orbits_on_set_system_first_column;
	int orbits_on_set_system_number_of_columns;

	int f_orbit_of_set_from_file;
	std::string orbit_of_set_from_file_fname;

	int f_conjugacy_class_of;
	std::string conjugacy_class_of_data;

	int f_isomorphism_Klein_quadric;
	std::string isomorphism_Klein_quadric_fname;

	int f_print_elements;
	int f_print_elements_tex;

	int f_save_elements_csv;
	std::string save_elements_csv_fname;


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

	// classification of optimal linear codes:
	int f_linear_codes;
	int linear_codes_minimum_distance;
	int linear_codes_target_size;


	int f_exact_cover;
	exact_cover_arguments *ECA;
	int f_isomorph_arguments;
	isomorph_arguments *IA;




		// subspace orbits:
		int f_orbits_on_subspaces;
		int orbits_on_subspaces_depth;
		int f_mindist;
		int mindist;
		int f_self_orthogonal;
		int f_doubly_even;

		int f_tensor_classify;
		int tensor_classify_depth;
		int f_tensor_permutations;

		int f_classify_ovoids;
		ovoid_classify_description *Ovoid_classify_description;

		int f_classify_cubic_curves;

		int f_orbits_on_polynomials;
		int orbits_on_polynomials_degree;
		int f_recognize_orbits_on_polynomials;
		std::string recognize_orbits_on_polynomials_text;

		int f_orbits_on_polynomials_draw_tree;
		int orbits_on_polynomials_draw_tree_idx;


		int f_representation_on_polynomials;
		int representation_on_polynomials_degree;

		int f_Andre_Bruck_Bose_construction;
		int Andre_Bruck_Bose_construction_spread_no;
		std::string Andre_Bruck_Bose_construction_label;




	group_theoretic_activity_description();
	~group_theoretic_activity_description();
	void null();
	void freeself();
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
	//finite_field *F;

	any_group *AG;

	action *A1;
	action *A2;



	group_theoretic_activity();
	~group_theoretic_activity();
	void init_group(group_theoretic_activity_description *Descr,
			any_group *AG,
			int verbose_level);
#if 0
	void init_linear_group(group_theoretic_activity_description *Descr,
			finite_field *F, linear_group *LG,
			int verbose_level);
	void init_permutation_group(group_theoretic_activity_description *Descr,
			permutation_group_create *PGC,
			int verbose_level);
#endif
	void perform_activity(int verbose_level);
	void multiply(int verbose_level);
	void inverse(int verbose_level);
	void raise_to_the_power(int verbose_level);
	void do_Andre_Bruck_Bose_construction(int spread_no,
			int f_Fano, int f_arcs, int f_depth, int depth,
			std::string &label,
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

	//strong_generators *initial_strong_gens;

	action *A_base;
	action *A_previous;
	action *A_modified;

	int f_has_strong_generators;
	strong_generators *Strong_gens;


	modified_group_create();
	~modified_group_create();
	void modified_group_init(
			group_modification_description *description,
			int verbose_level);

};




// #############################################################################
// orbits_on_polynomials.cpp
// #############################################################################


//! orbits of a group on polynomials using Schreier orbits

class orbits_on_polynomials {
public:

	linear_group *LG;
	int degree_of_poly;

	finite_field *F;
	action *A;
	int n;
	longinteger_object go;

	homogeneous_polynomial_domain *HPD;

	action *A2;

	int *Elt1;
	int *Elt2;
	int *Elt3;

	schreier *Sch;
	longinteger_object full_go;

	std::string fname_base;
	std::string fname_csv;
	std::string fname_report;

	orbit_transversal *T;
	int *Nb_pts; // [T->nb_orbits]
	std::vector<std::vector<long int> > Points;


	orbits_on_polynomials();
	~orbits_on_polynomials();
	void init(
			linear_group *LG,
			int degree_of_poly,
			int f_recognize, std::string &recognize_text,
			int verbose_level);
	void compute_points(int verbose_level);
	void report(int verbose_level);
	void report_detailed_list(std::ostream &ost,
			int verbose_level);


};



// #############################################################################
// orbits_on_subspaces.cpp
// #############################################################################


//! orbits of a group on subspaces of a vector space

class orbits_on_subspaces {
public:
	group_theoretic_activity *GTA;

	// local data for orbits on subspaces:
	poset_with_group_action *orbits_on_subspaces_Poset;
	poset_classification *orbits_on_subspaces_PC;
	vector_space *orbits_on_subspaces_VS;
	int *orbits_on_subspaces_M;
	int *orbits_on_subspaces_base_cols;


	orbits_on_subspaces();
	~orbits_on_subspaces();
	void init(group_theoretic_activity *GTA,
			poset_classification_control *Control, int depth,
			int verbose_level);


};


long int orbits_on_subspaces_rank_point_func(int *v, void *data);
void orbits_on_subspaces_unrank_point_func(int *v, long int rk, void *data);
void orbits_on_subspaces_early_test_func(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);




// #############################################################################
// young.cpp
// #############################################################################


//! The Young representations of the symmetric group


class young {
public:
	int n;
	action *A;
	sims *S;
	longinteger_object go;
	int goi;
	int *Elt;
	int *v;

	action *Aconj;
	action_by_conjugation *ABC;
	schreier *Sch;
	strong_generators *SG;
	int nb_classes;
	int *class_size;
	int *class_rep;
	a_domain *D;

	int l1, l2;
	int *row_parts;
	int *col_parts;
	int *Tableau;

	set_of_sets *Row_partition;
	set_of_sets *Col_partition;

	vector_ge *gens1, *gens2;
	sims *S1, *S2;


	young();
	~young();
	void null();
	void freeself();
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
	long int group_ring_element_size(action *A, sims *S);
	void group_ring_element_create(action *A, sims *S, int *&elt);
	void group_ring_element_free(action *A, sims *S, int *elt);
	void group_ring_element_print(action *A, sims *S, int *elt);
	void group_ring_element_copy(action *A, sims *S,
		int *elt_from, int *elt_to);
	void group_ring_element_zero(action *A, sims *S,
		int *elt);
	void group_ring_element_mult(action *A, sims *S,
		int *elt1, int *elt2, int *elt3);
};


}}


#endif /* ORBITER_SRC_LIB_TOP_LEVEL_ALGEBRA_AND_NUMBER_THEORY_TL_ALGEBRA_AND_NUMBER_THEORY_H_ */


