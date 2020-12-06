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
	void analyze_group(action *A, sims *S, vector_ge *SG,
		vector_ge *gens2, int verbose_level);
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


	void classify_surfaces(
			finite_field *F, linear_group *LG,
			poset_classification_control *Control,
			surface_domain *&Surf, surface_with_action *&Surf_A,
			surface_classify_wedge *&SCW,
			int verbose_level);
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
	void packing_init(
			poset_classification_control *Control, linear_group *LG,
			int dimension_of_spread_elements,
			int f_select_spread, std::string &select_spread_text,
			std::string &path_to_spread_tables,
			packing_classify *&P,
			int verbose_level);
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

	void do_cheat_sheet_for_decomposition_by_element_PG(finite_field *F,
			int n,
			int decomposition_by_element_power,
			std::string &decomposition_by_element_data, std::string &fname_base,
			int verbose_level);
	void do_canonical_form_PG(finite_field *F,
			projective_space_object_classifier_description *Canonical_form_PG_Descr,
			int n, int verbose_level);
	void do_study_surface(finite_field *F, int nb, int verbose_level);
	void do_cubic_surface_properties(
			linear_group *LG,
			std::string fname_csv, int defining_q,
			int column_offset,
			int verbose_level);
	void do_cubic_surface_properties_analyze(
			linear_group *LG,
			std::string fname_csv, int defining_q,
			int verbose_level);
	void report_singular_surfaces(std::ostream &ost,
			struct cubic_surface_data_set *Data, int nb_orbits,
			int verbose_level);
	void report_non_singular_surfaces(std::ostream &ost,
			struct cubic_surface_data_set *Data, int nb_orbits,
			int verbose_level);
	void report_surfaces_by_lines(std::ostream &ost,
			struct cubic_surface_data_set *Data, tally &T, int verbose_level);

};



// #############################################################################
// character_table_burnside.cpp
// #############################################################################

//! character table using Burnside algorithm


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
// factor_group.cpp
// #############################################################################


//! auxiliary class for create_factor_group, which is used in analyze_group.cpp

struct factor_group {
	long int goi;
	action *A;
	sims *S;
	int size_subgroup;
	int *subgroup;
	long int *all_cosets;
	int nb_cosets;
	action *ByRightMultiplication;
	action *FactorGroup;
	action *FactorGroupConjugated;
	long int goi_factor_group;
};

void create_factor_group(action *A, sims *S, long int goi,
	int size_subgroup, int *subgroup, factor_group *F, int verbose_level);


// #############################################################################
// finite_field_activity_description.cpp
// #############################################################################


//! description of a finite field activity

class finite_field_activity_description {
public:

	int f_q;
	int q;

	int f_override_polynomial;
	std::string override_polynomial;

	int f_cheat_sheet_GF;
	int f_all_rational_normal_forms;
	int d;

	int f_polynomial_division;
	std::string polynomial_division_A;
	std::string polynomial_division_B;

	int f_extended_gcd_for_polynomials;

	int f_polynomial_mult_mod;
	std::string polynomial_mult_mod_A;
	std::string polynomial_mult_mod_B;
	std::string polynomial_mult_mod_M;

	int f_Berlekamp_matrix;
	std::string Berlekamp_matrix_coeffs;

	int f_normal_basis;
	int normal_basis_d;

	int f_polynomial_find_roots;
	std::string polynomial_find_roots_A;


	int f_normalize_from_the_right;
	int f_normalize_from_the_left;

	int f_nullspace;
	int nullspace_m;
	int nullspace_n;
	std::string nullspace_text;

	int f_RREF;
	int RREF_m;
	int RREF_n;
	std::string RREF_text;

	int f_weight_enumerator;

	int f_trace;

	int f_norm;

	int f_make_table_of_irreducible_polynomials;
	int make_table_of_irreducible_polynomials_degree;

	int f_EC_Koblitz_encoding;
	std::string EC_message;
	int EC_s;
	int f_EC_points;
	int f_EC_add;
	std::string EC_pt1_text;
	std::string EC_pt2_text;

	int f_EC_cyclic_subgroup;
	int EC_b;
	int EC_c;
	std::string EC_pt_text;

	int f_EC_multiple_of;
	int EC_multiple_of_n;
	int f_EC_discrete_log;
	std::string EC_discrete_log_pt_text;

	int f_EC_baby_step_giant_step;
	std::string EC_bsgs_G;
	int EC_bsgs_N;
	std::string EC_bsgs_cipher_text;

	int f_EC_baby_step_giant_step_decode;
	std::string EC_bsgs_A;
	std::string EC_bsgs_keys;




	int f_NTRU_encrypt;
	int NTRU_encrypt_N;
	int NTRU_encrypt_p;
	std::string NTRU_encrypt_H;
	std::string NTRU_encrypt_R;
	std::string NTRU_encrypt_Msg;

	int f_polynomial_center_lift;
	std::string polynomial_center_lift_A;

	int f_polynomial_reduce_mod_p;
	std::string polynomial_reduce_mod_p_A;

	int f_cheat_sheet_PG;
	int cheat_sheet_PG_n;

	int f_cheat_sheet_Gr;
	int cheat_sheet_Gr_n;
	int cheat_sheet_Gr_k;

	int f_cheat_sheet_orthogonal;
	int cheat_sheet_orthogonal_epsilon;
	int cheat_sheet_orthogonal_n;

	int f_decomposition_by_element;
	int decomposition_by_element_n;
	int decomposition_by_element_power;
	std::string decomposition_by_element_data;
	std::string decomposition_by_element_fname_base;

	int f_canonical_form_PG;
	int canonical_form_PG_n;
	projective_space_object_classifier_description *Canonical_form_PG_Descr;


	int f_transversal;
	std::string transversal_line_1_basis;
	std::string transversal_line_2_basis;
	std::string transversal_point;

	int f_intersection_of_two_lines;
	std::string line_1_basis;
	std::string line_2_basis;

	int f_move_two_lines_in_hyperplane_stabilizer;
	long int line1_from;
	long int line2_from;
	long int line1_to;
	long int line2_to;

	int f_move_two_lines_in_hyperplane_stabilizer_text;
	std::string line1_from_text;
	std::string line2_from_text;
	std::string line1_to_text;
	std::string line2_to_text;

	int f_study_surface;
	int study_surface_nb;

	int f_inverse_isomorphism_klein_quadric;
	std::string inverse_isomorphism_klein_quadric_matrix_A6;

	int f_rank_point_in_PG;
	int rank_point_in_PG_n;
	std::string rank_point_in_PG_text;

	int f_rank_point_in_PG_given_as_pairs;
	int rank_point_in_PG_given_as_pairs_n;
	std::string rank_point_in_PG_given_as_pairs_text;


	int f_eigenstuff;
	int f_eigenstuff_from_file;
	int eigenstuff_n;
	std::string eigenstuff_coeffs;
	std::string eigenstuff_fname;


	finite_field_activity_description();
	~finite_field_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);

};



// #############################################################################
// finite_field_activity.cpp
// #############################################################################


//! perform a finite field activity

class finite_field_activity {
public:
	finite_field_activity_description *Descr;
	finite_field *F;

	finite_field_activity();
	~finite_field_activity();
	void init(finite_field_activity_description *Descr,
			int verbose_level);
	void perform_activity(int verbose_level);


};


// #############################################################################
// group_theoretic_activity_description.cpp
// #############################################################################


//! description of a group theoretic actvity

class group_theoretic_activity_description {
public:

	int f_poset_classification_control;
	poset_classification_control *Control;

	int f_draw_options;
	layered_graph_draw_options *draw_options;

	int f_orbits_on_points;
	int f_export_trees;
	int f_shallow_tree;
	int f_stabilizer;
	int f_orbits_on_subsets;
	int orbits_on_subsets_size;
	int f_classes_based_on_normal_form;
	int f_classes;
	int f_group_table;
	int f_normalizer;
	int f_centralizer_of_element;
	std::string element_description_text;
	std::string element_label;
	int f_normalizer_of_cyclic_subgroup;
	int f_find_subgroup;
	int find_subgroup_order;
	int f_report;
	int f_sylow;
	int f_test_if_geometric;
	int test_if_geometric_depth;
	int f_draw_tree;
	int f_orbit_of;
	int orbit_of_idx;
	int f_orbits_on_set_system_from_file;
	std::string orbits_on_set_system_from_file_fname;
	int orbits_on_set_system_first_column;
	int orbits_on_set_system_number_of_columns;
	int f_orbit_of_set_from_file;
	std::string orbit_of_set_from_file_fname;
	int f_search_subgroup;
	int f_find_singer_cycle;
	int f_search_element_of_order;
	int search_element_order;
	int f_element_rank;
	std::string element_rank_data;
	int f_element_unrank;
	std::string element_unrank_data;
	int f_conjugacy_class_of;
	std::string conjugacy_class_of_data;
	int f_isomorphism_Klein_quadric;
	std::string isomorphism_Klein_quadric_fname;
	int f_print_elements;
	int f_print_elements_tex;
	int f_multiply;
	std::string multiply_a;
	std::string multiply_b;
	int f_inverse;
	std::string inverse_a;
	int f_export_gap;
	int f_export_magma;
	int f_order_of_products;
	std::string order_of_products_elements;
	int f_reverse_isomorphism_exterior_square;

	// classification of optimal linear codes:
	int f_linear_codes;
	int linear_codes_minimum_distance;
	int linear_codes_target_size;


	// classification of arcs in projective spaces:
	int f_classify_arcs;
	arc_generator_description *Arc_generator_description;


	int f_exact_cover;
	exact_cover_arguments *ECA;
	int f_isomorph_arguments;
	isomorph_arguments *IA;


	// for cubic surfaces:
	int f_surface_classify;
	int f_surface_report;
	int f_surface_identify_HCV;
	int f_surface_identify_F13;
	int f_surface_identify_Bes;
	int f_surface_identify_general_abcd;
	int f_surface_isomorphism_testing;
		surface_create_description *surface_descr_isomorph1;
		surface_create_description *surface_descr_isomorph2;
	int f_surface_recognize;
		surface_create_description *surface_descr;
	int f_classify_surfaces_through_arcs_and_two_lines;
	int f_test_nb_Eckardt_points;
	int nb_E;
	int f_classify_surfaces_through_arcs_and_trihedral_pairs;
		int f_trihedra1_control;
		poset_classification_control *Trihedra1_control;
		int f_trihedra2_control;
		poset_classification_control *Trihedra2_control;
		int f_control_six_arcs;
		poset_classification_control *Control_six_arcs;
	int f_create_surface;
	surface_create_description *surface_description;
	int f_six_arcs;
	int f_filter_by_nb_Eckardt_points;
	int nb_Eckardt_points;
	int f_surface_quartic;
	int f_surface_clebsch;
	int f_surface_codes;

	int f_cubic_surface_properties;
	std::string cubic_surface_properties_fname_csv;
	int cubic_surface_properties_defining_q;
	int cubic_surface_properties_column_offset;
	int f_cubic_surface_properties_analyze;


		// subspace orbits:
		int f_orbits_on_subspaces;
		int orbits_on_subspaces_depth;
		int f_mindist;
		int mindist;
		int f_self_orthogonal;
		int f_doubly_even;

		int f_spread_classify;
		int spread_classify_k;


		int f_packing_classify;
		int dimension_of_spread_elements;
		std::string spread_selection_text;
		std::string spread_tables_prefix;

		int f_packing_with_assumed_symmetry;
		packing_was_description *packing_was_descr;


		int f_tensor_classify;
		int tensor_classify_depth;
		int f_tensor_permutations;

		int f_classify_ovoids;
		ovoid_classify_description *Ovoid_classify_description;

		int f_classify_cubic_curves;

		int f_classify_semifields;
		semifield_classify_description *Semifield_classify_description;

		int f_orbits_on_polynomials;
		int orbits_on_polynomials_degree;
		int f_recognize_orbits_on_polynomials;
		std::string recognize_orbits_on_polynomials_text;

		int f_orbits_on_polynomials_draw_tree;
		int orbits_on_polynomials_draw_tree_idx;


		int f_representation_on_polynomials;
		int representation_on_polynomials_degree;



	group_theoretic_activity_description();
	~group_theoretic_activity_description();
	void null();
	void freeself();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);

};


// #############################################################################
// group_theoretic_activity.cpp
// #############################################################################


//! perform a group theoretic activity

class group_theoretic_activity {
public:
	group_theoretic_activity_description *Descr;
	finite_field *F;
	linear_group *LG;
	action *A1;
	action *A2;

	// local data for orbits on subspaces:
	poset *orbits_on_subspaces_Poset;
	poset_classification *orbits_on_subspaces_PC;
	vector_space *orbits_on_subspaces_VS;
	int *orbits_on_subspaces_M;
	int *orbits_on_subspaces_base_cols;


	group_theoretic_activity();
	~group_theoretic_activity();
	void init(group_theoretic_activity_description *Descr,
			finite_field *F, linear_group *LG,
			int verbose_level);
	void perform_activity(int verbose_level);
	void classes_based_on_normal_form(int verbose_level);
	void classes(int verbose_level);
	void multiply(int verbose_level);
	void inverse(int verbose_level);
	void do_export_gap(int verbose_level);
	void do_export_magma(int verbose_level);
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
	void report(layered_graph_draw_options *draw_option, int verbose_level);
	void print_elements(int verbose_level);
	void print_elements_tex(int verbose_level);
	void search_subgroup(int verbose_level);
	void find_singer_cycle(int verbose_level);
	void search_element_of_order(int order, int verbose_level);
	void element_rank(std::string &elt_data, int verbose_level);
	void element_unrank(std::string &rank_string, int verbose_level);
	void conjugacy_class_of(std::string &rank_string, int verbose_level);
	void do_reverse_isomorphism_exterior_square(int verbose_level);
	void isomorphism_Klein_quadric(std::string &fname, int verbose_level);
	void orbits_on_set_system_from_file(int verbose_level);
	void orbits_on_set_from_file(int verbose_level);
	void orbit_of(int verbose_level);
	void orbits_on_points(int verbose_level);
	void orbits_on_subsets(int verbose_level);
	void orbits_on_subspaces(int verbose_level);
	void orbits_on_poset_post_processing(
			poset_classification *PC,
			int depth,
			int verbose_level);
	void do_classify_arcs(
			arc_generator_description *Arc_generator_description,
			int verbose_level);
	void do_spread_classify(int k, int verbose_level);
	void do_packing_classify(int dimension_of_spread_elements,
			std::string &spread_selection_text,
			std::string &spread_tables_prefix,
			int starter_size,
			packing_classify *&P,
			int verbose_level);
	void do_tensor_classify(int depth, int verbose_level);
	void do_tensor_permutations(int verbose_level);
	void do_linear_codes(int minimum_distance,
			int target_size, int verbose_level);
	void do_classify_ovoids(
			poset_classification_control *Control,
			ovoid_classify_description *Ovoid_classify_description,
			int verbose_level);
	void do_classify_cubic_curves(
			arc_generator_description *Arc_generator_description,
			int verbose_level);
	void do_classify_semifields(
			semifield_classify_description *Semifield_classify_description,
			poset_classification_control *Control,
			int verbose_level);
	int subspace_orbits_test_set(
			int len, long int *S, int verbose_level);


	// group_theoretic_activity_for_surfaces.cpp:

	void do_create_surface(
			surface_create_description *Descr,
			poset_classification_control *Control_six_arcs,
			int verbose_level);
	void do_surface_classify(int verbose_level);
	void do_surface_report(int verbose_level);
	void do_surface_identify_HCV(int verbose_level);
	void do_surface_identify_F13(int verbose_level);
	void do_surface_identify_Bes(int verbose_level);
	void do_surface_identify_general_abcd(int verbose_level);
	void do_surface_isomorphism_testing(
			surface_create_description *surface_descr_isomorph1,
			surface_create_description *surface_descr_isomorph2,
			int verbose_level);
	void do_surface_recognize(
			surface_create_description *surface_descr,
			int verbose_level);
	void do_classify_surfaces_through_arcs_and_two_lines(
			poset_classification_control *Control_six_arcs,
			int f_test_nb_Eckardt_points, int nb_E,
			int verbose_level);
	void do_classify_surfaces_through_arcs_and_trihedral_pairs(
			poset_classification_control *Control1,
			poset_classification_control *Control2,
			poset_classification_control *Control_six_arcs,
			int f_test_nb_Eckardt_points, int nb_E,
			int verbose_level);
	void do_six_arcs(
			poset_classification_control *Control_six_arcs,
			int f_filter_by_nb_Eckardt_points, int nb_Eckardt_points,
			int verbose_level);


};

long int gta_subspace_orbits_rank_point_func(int *v, void *data);
void gta_subspace_orbits_unrank_point_func(int *v, long int rk, void *data);
void gta_subspace_orbits_early_test_func(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);


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
	//matrix_group *M;
	int n;
	//int degree;
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


