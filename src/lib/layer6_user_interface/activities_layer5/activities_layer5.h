/*
 * activities_layer5.h
 *
 *  Created on: Mar 23, 2026
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER6_USER_INTERFACE_ACTIVITIES_LAYER5_ACTIVITIES_LAYER5_H_
#define SRC_LIB_LAYER6_USER_INTERFACE_ACTIVITIES_LAYER5_ACTIVITIES_LAYER5_H_




namespace orbiter {
namespace layer6_user_interface {
namespace activities_layer5 {



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

	layer5_applications::apps_algebra::action_on_forms *AF;



	action_on_forms_activity();
	~action_on_forms_activity();
	void init(
			action_on_forms_activity_description *Descr,
			layer5_applications::apps_algebra::action_on_forms *AF,
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
// blt_set_activity_description.cpp
// #############################################################################

//! description of an activity for a BLT-set



class blt_set_activity_description {

public:

	// TABLES/blt_set_activity.tex

	int f_report;

	int f_export_gap;

	int f_create_flock;
	int create_flock_point_idx;

	int f_BLT_test;

	int f_export_set_in_PG;

	int f_plane_invariant;

	blt_set_activity_description();
	~blt_set_activity_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// blt_set_activity.cpp
// #############################################################################

//! an activity regarding a BLT-sets



class blt_set_activity {

public:

	blt_set_activity_description *Descr;

	layer5_applications::orthogonal_geometry_applications::BLT_set_create *BC;

	blt_set_activity();
	~blt_set_activity();
	void init(
			blt_set_activity_description *Descr,
			layer5_applications::orthogonal_geometry_applications::BLT_set_create *BC,
			int verbose_level);
	void perform_activity(
			int verbose_level);

};



// #############################################################################
// blt_set_classify_activity_description.cpp
// #############################################################################

//! description of an activity regarding the classification of BLT-sets

class blt_set_classify_activity_description {

public:

	// TABLES/blt_set_classify_activity.tex

	int f_compute_starter;
	poset_classification::poset_classification_control *starter_control;

	int f_poset_classification_activity;
	std::string poset_classification_activity_label;

	int f_create_graphs;

	int f_split;
	int split_r;
	int split_m;

	int f_isomorph;
	std::string isomorph_label;

	int f_build_db;
	int f_read_solutions;
	int f_compute_orbits;
	int f_isomorph_testing;
	int f_isomorph_report;


	blt_set_classify_activity_description();
	~blt_set_classify_activity_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// blt_set_classify_activity.cpp
// #############################################################################

//! an activity regarding the classification of BLT-sets



class blt_set_classify_activity {

public:

	blt_set_classify_activity_description *Descr;
	layer5_applications::orthogonal_geometry_applications::blt_set_classify *BLT_classify;
	layer5_applications::orthogonal_geometry_applications::orthogonal_space_with_action *OA;

	blt_set_classify_activity();
	~blt_set_classify_activity();
	void init(
			blt_set_classify_activity_description *Descr,
			layer5_applications::orthogonal_geometry_applications::blt_set_classify *BLT_classify,
			layer5_applications::orthogonal_geometry_applications::orthogonal_space_with_action *OA,
			int verbose_level);
	void perform_activity(
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);

};






// #############################################################################
// classification_of_cubic_surfaces_with_double_sixes_activity_description.cpp
// #############################################################################

//! description of an activity for a classification of cubic surfaces with 27 lines with double sixes


class classification_of_cubic_surfaces_with_double_sixes_activity_description {
public:

	// TABLES/classification_of_cubic_surfaces_with_double_sixes_activity.tex

	int f_report;
	std::string report_options;
	//poset_classification::poset_classification_report_options
	//	*report_options;

	int f_stats;
	std::string stats_prefix;

	int f_identify_Eckardt;

	int f_identify_F13;

	int f_identify_Bes;

	int f_identify_general_abcd;

	int f_isomorphism_testing;
	std::string isomorphism_testing_surface1_label;
	std::string isomorphism_testing_surface2_label;

	int f_recognize;
	std::string recognize_surface_label;

	int f_create_source_code;

	int f_sweep_Cayley;


	classification_of_cubic_surfaces_with_double_sixes_activity_description();
	~classification_of_cubic_surfaces_with_double_sixes_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};

// #############################################################################
// classification_of_cubic_surfaces_with_double_sixes_activity.cpp
// #############################################################################

//! an activity for a classification of cubic surfaces with 27 lines with double sixes


class classification_of_cubic_surfaces_with_double_sixes_activity {
public:

	classification_of_cubic_surfaces_with_double_sixes_activity_description
		*Descr;
	layer5_applications::applications_in_algebraic_geometry::cubic_surfaces_and_double_sixes::surface_classify_wedge *SCW;

	classification_of_cubic_surfaces_with_double_sixes_activity();
	~classification_of_cubic_surfaces_with_double_sixes_activity();
	void init(
			classification_of_cubic_surfaces_with_double_sixes_activity_description
				*Descr,
				layer5_applications::applications_in_algebraic_geometry::cubic_surfaces_and_double_sixes::surface_classify_wedge *SCW,
			int verbose_level);
	void perform_activity(
			int verbose_level);
	void report(
			std::string &options,
			//poset_classification::poset_classification_report_options
			//	*report_options,
			int verbose_level);
	void do_write_source_code(
			int verbose_level);


};



// #############################################################################
// coding_theoretic_activity_description.cpp
// #############################################################################

//! description of an activity in coding theory


class coding_theoretic_activity_description {

public:

	// TABLES/coding_theoretic_activity_1.tex

	int f_report;

	int f_general_code_binary;
	int general_code_binary_n;
	std::string general_code_binary_label;
	std::string general_code_binary_text;

	int f_encode_text_5bits;
	std::string encode_text_5bits_input;
	std::string encode_text_5bits_fname;

	int f_field_induction;
	std::string field_induction_fname_in;
	std::string field_induction_fname_out;
	int field_induction_nb_bits;



	int f_weight_enumerator;
	//std::string weight_enumerator_input_matrix;

	int f_minimum_distance;
	std::string minimum_distance_code_label;

	int f_generator_matrix_cyclic_code;
	int generator_matrix_cyclic_code_n;
	std::string generator_matrix_cyclic_code_poly;

	int f_Sylvester_Hadamard_code;
	int Sylvester_Hadamard_code_n;

	// undocumented:
	int f_NTT;
	int NTT_n;
	int NTT_q;

	int f_fixed_code;
	std::string fixed_code_perm;

	int f_export_magma;
	std::string export_magma_fname;

	int f_export_codewords;
	std::string export_codewords_fname;

	int f_all_distances;

	int f_all_external_distances;

	int f_export_codewords_long;
	std::string export_codewords_long_fname;

	int f_export_codewords_by_weight;
	std::string export_codewords_by_weight_fname;

	int f_export_genma;
	std::string export_genma_fname;

	int f_export_checkma;
	std::string export_checkma_fname;

	int f_export_checkma_as_projective_set;
	std::string export_checkma_as_projective_set_fname;


	// TABLES/coding_theoretic_activity_2.tex


	int f_make_diagram;

	int f_boolean_function_of_code;

	int f_embellish;
	int embellish_radius;

	int f_metric_balls;
	int radius_of_metric_ball;

	int f_Hamming_space_distance_matrix;
	int Hamming_space_distance_matrix_n;

	// TABLES/coding_theoretic_activity_3.tex


	// CRC stuff:
	int f_crc32;
	std::string crc32_text;

	int f_crc32_hexdata;
	std::string crc32_hexdata_text;

	int f_crc32_test;
	int crc32_test_block_length;

	int f_crc256_test;
	int crc256_test_message_length;
	int crc256_test_R;
	int crc256_test_k;

	int f_crc32_remainders;
	int crc32_remainders_message_length;

	int f_crc_encode_file_based;
	std::string crc_encode_file_based_fname_in;
	std::string crc_encode_file_based_fname_out;
	std::string crc_encode_file_based_crc_code;

	int f_crc_compare;
	std::string crc_compare_fname_in;
	std::string crc_compare_code1;
	std::string crc_compare_code2;
	int crc_compare_error_weight;
	int crc_compare_nb_tests_per_block;

	int f_crc_compare_read_output_file;
	std::string crc_compare_read_output_file_fname_in;
	int crc_compare_read_output_file_nb_lines;
	std::string crc_compare_read_output_file_crc_code1;
	std::string crc_compare_read_output_file_crc_code2;


	int f_all_errors_of_a_given_weight;
	std::string all_errors_of_a_given_weight_fname_in;
	int all_errors_of_a_given_weight_block_number;
	std::string all_errors_of_a_given_weight_crc_code1;
	std::string all_errors_of_a_given_weight_crc_code2;
	int all_errors_of_a_given_weight_max_weight;


	int f_weight_enumerator_bottom_up;
	std::string weight_enumerator_bottom_up_crc_code;
	int weight_enumerator_bottom_up_max_weight;


	int f_convert_data_to_polynomials;
	std::string convert_data_to_polynomials_fname_in;
	std::string convert_data_to_polynomials_fname_out;
	int convert_data_to_polynomials_block_length;
	int convert_data_to_polynomials_symbol_size;

	int f_find_CRC_polynomials;
	int find_CRC_polynomials_nb_errors;
	int find_CRC_polynomials_information_bits;
	int find_CRC_polynomials_check_bits;

	int f_write_code_for_division;
	std::string write_code_for_division_fname;
	std::string write_code_for_division_A;
	std::string write_code_for_division_B;

	int f_polynomial_division_from_file;
	std::string polynomial_division_from_file_fname;
	long int polynomial_division_from_file_r1;

	int f_polynomial_division_from_file_all_k_bit_error_patterns;
	std::string polynomial_division_from_file_all_k_bit_error_patterns_fname;
	int polynomial_division_from_file_all_k_bit_error_patterns_r1;
	int polynomial_division_from_file_all_k_bit_error_patterns_k;



	coding_theoretic_activity_description();
	~coding_theoretic_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};


// #############################################################################
// coding_theoretic_activity.cpp
// #############################################################################

//! an activity for codes or finite fields


class coding_theoretic_activity {

public:

	coding_theoretic_activity_description *Descr;

	int f_has_finite_field;
	algebra::field_theory::finite_field *F;

	int f_has_code;
	layer5_applications::apps_coding_theory::create_code *Code;


	coding_theoretic_activity();
	~coding_theoretic_activity();
	void init_field(
			coding_theoretic_activity_description *Descr,
			algebra::field_theory::finite_field *F,
			int verbose_level);
	void init_code(
			coding_theoretic_activity_description *Descr,
			layer5_applications::apps_coding_theory::create_code *Code,
			int verbose_level);
	void perform_activity(
			int verbose_level);
	void do_diagram(
			combinatorics::coding_theory::code_diagram *Diagram,
			int verbose_level);


};




// #############################################################################
// combinatorial_object_activity_description.cpp
// #############################################################################


//! description of an activity for a combinatorial object

class combinatorial_object_activity_description {
public:


	// TABLE/combinatorial_object_activity.csv

	// options that apply to GOC = geometric_object_create

	int f_save;

	int f_save_as;
	std::string save_as_fname;

	int f_extract_subset;
	std::string extract_subset_set;
	std::string extract_subset_fname;

	int f_line_type_old;

	int f_line_type;

	int f_conic_type;
	int conic_type_threshold;

	int f_non_conical_type;

	int f_ideal;
	std::string ideal_ring_label;


	// options that apply to IS = data_input_stream

	int f_canonical_form_PG;
	std::string canonical_form_PG_PG_label;

	int f_canonical_form_PG_has_PA;
	layer5_applications::projective_geometry::projective_space_with_action
		*Canonical_form_PG_PA;

	combinatorics::canonical_form_classification::classification_of_objects_description
		*Canonical_form_PG_Descr;

	int f_canonical_form;
	combinatorics::canonical_form_classification::classification_of_objects_description
		*Canonical_form_Descr;

	int f_post_processing;

	int f_get_combo_with_group;
	int get_combo_with_group_idx;

	int f_report;
	combinatorics::canonical_form_classification::objects_report_options
		*Objects_report_options;

	int f_draw_incidence_matrices;
	std::string draw_incidence_matrices_prefix;
	std::string draw_incidence_matrices_options_label;

	int f_test_distinguishing_property;
	std::string test_distinguishing_property_graph;

	int f_covering_type;
	std::string covering_type_orbits;
	int covering_type_size;

	int f_filter_by_Steiner_property;

	int f_compute_frequency;
	std::string compute_frequency_graph;

	int f_unpack_from_restricted_action;
	std::string unpack_from_restricted_action_prefix;
	std::string unpack_from_restricted_action_group_label;

	int f_line_covering_type;
	std::string line_covering_type_prefix;
	std::string line_covering_type_projective_space;
	std::string line_covering_type_lines;

	int f_activity;
	layer6_user_interface::control_everything::activity_description *Activity_description;

	int f_algebraic_degree;
	std::string algebraic_degree_PG_label;

	// ToDo undocumented:
	int f_polynomial_representation;
	std::string polynomial_representation_PG_label;

	combinatorial_object_activity_description();
	~combinatorial_object_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};

// #############################################################################
// combinatorial_object_activity.cpp
// #############################################################################


//! perform an activity for a combinatorial object

class combinatorial_object_activity {
public:
	combinatorial_object_activity_description *Descr;

	int f_has_geometric_object;
	geometry::other_geometry::geometric_object_create *GOC;

	int f_has_combo;
	layer5_applications::apps_combinatorics::combinatorial_object_stream *Combo;

	int nb_output;
	other::orbiter_kernel_system::orbiter_symbol_table_entry *Output;


	combinatorial_object_activity();
	~combinatorial_object_activity();
	void init_geometric_object_create(
			combinatorial_object_activity_description *Descr,
			geometry::other_geometry::geometric_object_create *GOC,
			int verbose_level);
	void init_combo(
			combinatorial_object_activity_description *Descr,
			layer5_applications::apps_combinatorics::combinatorial_object_stream *Combo,
			int verbose_level);
	void perform_activity(
			other::orbiter_kernel_system::activity_output *&AO,
			int verbose_level);
	void perform_activity_geometric_object(
			other::orbiter_kernel_system::activity_output *&AO,
			int verbose_level);
	void perform_activity_combo(
			other::orbiter_kernel_system::activity_output *&AO,
			int verbose_level);

};




// #############################################################################
// combo_activity_description.cpp
// #############################################################################


//! description of an activity for a combinatorial object with group

class combo_activity_description {
public:


	// TABLE/combo_activity.tex

	int f_report;
	combinatorics::canonical_form_classification::objects_report_options
		*Objects_report_options;

	int f_get_group;

	combo_activity_description();
	~combo_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// combo_activity.cpp
// #############################################################################


//! perform an activity for a combinatorial object with group

class combo_activity {
public:
	combo_activity_description *Descr;

	layer5_applications::canonical_form::combinatorial_object_with_properties **pOwP;
	int nb_objects;

	int nb_output;
	other::orbiter_kernel_system::orbiter_symbol_table_entry *Output;


	combo_activity();
	~combo_activity();
	void init(
			combo_activity_description *Descr,
			layer5_applications::canonical_form::combinatorial_object_with_properties **pOwP,
			int nb_objects,
			int verbose_level);
	void perform_activity(
			other::orbiter_kernel_system::activity_output *&AO,
			int verbose_level);

};





// #############################################################################
// cubic_surface_activity_description.cpp
// #############################################################################

//! description of an activity associated with a cubic surface


class cubic_surface_activity_description {
public:


	// cubic_surface_activity.tex


	int f_report;
	std::string report_draw_options_label;

	int f_report_group_elements;
	std::string report_group_elements_csv_file;
	std::string report_group_elements_heading;

	int f_export_something;
	std::string export_something_what;

	int f_export_gap;

	int f_report_all_flag_orbits;
	std::string report_all_flag_orbits_classification_of_arcs;

	int f_export_all_quartic_curves;
	std::string export_all_quartic_curves_classification_of_arcs;

	int f_export_something_with_group_element;
	std::string export_something_with_group_element_what;
	std::string export_something_with_group_element_label;

	int f_action_on_module;
	std::string action_on_module_type;
	std::string action_on_module_basis;
	std::string action_on_module_gens;

	int f_Clebsch_map_up;
	int Clebsch_map_up_line_1_idx;
	int Clebsch_map_up_line_2_idx;

	int f_Clebsch_map_up_single_point;
	int Clebsch_map_up_single_point_input_point;
	int Clebsch_map_up_single_point_line_1_idx;
	int Clebsch_map_up_single_point_line_2_idx;


	int f_recognize_Fabcd;
	std::string recognize_Fabcd_classification_of_arcs;

	cubic_surface_activity_description();
	~cubic_surface_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};

// #############################################################################
// cubic_surface_activity.cpp
// #############################################################################

//! an activity associated with a cubic surface


class cubic_surface_activity {
public:

	cubic_surface_activity_description *Descr;
	layer5_applications::applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create *SC;


	cubic_surface_activity();
	~cubic_surface_activity();
	void init(
			layer6_user_interface::activities_layer5::cubic_surface_activity_description
				*Cubic_surface_activity_description,
				layer5_applications::applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create *SC, int verbose_level);
	void perform_activity(
			other::orbiter_kernel_system::activity_output *&AO,
			int verbose_level);
	void recognize_Fabcd(
			std::string &classification_of_arcs_label,
			int &a, int &b, int &c, int &d,
			other::orbiter_kernel_system::activity_output *&AO,
			int verbose_level);

};



// #############################################################################
// design_activity_description.cpp
// #############################################################################

//! to describe an activity for a design



class design_activity_description {

public:

	// TABLES/design_activity.tex

	int f_load_table;
	std::string load_table_label;
	std::string load_table_group;


	std::string load_table_H_label;
	std::string load_table_H_group_order;
	std::string load_table_H_gens;
	int load_table_selected_orbit_length;


	int f_canonical_form;
	combinatorics::canonical_form_classification::classification_of_objects_description
		*Canonical_form_Descr;

	int f_extract_solutions_by_index_csv;
	int f_extract_solutions_by_index_txt;
	std::string extract_solutions_by_index_label;
	std::string extract_solutions_by_index_group;
	std::string extract_solutions_by_index_fname_solutions_in;
	std::string extract_solutions_by_index_col_label;
	std::string extract_solutions_by_index_fname_solutions_out;
	std::string extract_solutions_by_index_prefix;

	int f_export_flags;
	int f_export_incidence_matrix;
	int f_export_incidence_matrix_as_flags;

	int f_export_incidence_matrix_latex;
	std::string export_incidence_matrix_latex_draw_options;

	int f_intersection_matrix;
	int f_save;
	int f_export_blocks;
	int f_row_sums;
	int f_tactical_decomposition;

	int f_orbits_on_blocks;
	int orbits_on_blocks_sz;
	std::string orbits_on_blocks_control;

	int f_one_point_extension;
	int one_point_extension_pair_orbit_idx;
	std::string one_point_extension_control;

	design_activity_description();
	~design_activity_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// design_activity.cpp
// #############################################################################

//! an activity for a design



class design_activity {

public:
	design_activity_description *Descr;


	design_activity();
	~design_activity();
	void perform_activity(
			design_activity_description *Descr,
			combinatorics::design_theory::design_object *Design_object,
			int verbose_level);
	void do_extract_solutions_by_index(
			layer5_applications::apps_combinatorics::design_create *DC,
			std::string &label,
			std::string &group_label,
			std::string &fname_in,
			std::string &col_label,
			std::string &fname_out,
			std::string &prefix_text,
			int f_csv_format,
			int verbose_level);
	// does not need DC. This should be an activity for the design_table
	void do_create_table(
			layer5_applications::apps_combinatorics::design_create *DC,
			std::string &label,
			std::string &group_label,
			int verbose_level);
	void do_load_table(
			layer5_applications::apps_combinatorics::design_create *DC,
			std::string &label,
			std::string &group_label,
			std::string &H_label,
			std::string &H_go_text,
			std::string &H_generators_data,
			int selected_orbit_length,
			int verbose_level);
	void do_canonical_form(
			combinatorics::canonical_form_classification::classification_of_objects_description
				*Canonical_form_Descr,
			int verbose_level);
#if 0
	void do_export_inc(
			design_create *DC,
			int verbose_level);
#endif
	void do_pair_orbits_on_blocks(
			layer5_applications::apps_combinatorics::design_create *DC,
			std::string &control_label,
			int *&Pair_orbits, int &degree,
			int verbose_level);

};





// #############################################################################
// group_theoretic_activity_description.cpp
// #############################################################################


//! description of an activity associated with a group

class group_theoretic_activity_description {
public:

	// TABLES/group_theoretic_activity_1.tex


	int f_report;
	std::string report_draw_options;

	int f_order_invariant;

	int f_group_table;
	std::string group_table_draw_options;

	int f_sylow;


	int f_generators;

	int f_elements;

	int f_elements_by_class;
	int elements_by_class_order;
	int elements_by_class_id;


	int f_select_elements;
	std::string select_elements_ranks;

	int f_export_group_table;

	int f_random_element;
	std::string random_element_label;


	int f_permutation_representation_of_element;
	std::string permutation_representation_element_text;

	int f_apply;
	std::string apply_input;
	std::string apply_element;

	// ToDo: undocumented
	int f_apply_to_set;
	std::string apply_to_set_input;
	std::string apply_to_set_element;

	// ToDo: this should become a vector_ge_activity
	int f_element_processing;
	layer5_applications::apps_algebra::element_processing_description *element_processing_descr;

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


	int f_search_element_of_order;
	int search_element_order;

	int f_find_standard_generators;
	int find_standard_generators_order_a;
	int find_standard_generators_order_b;
	int find_standard_generators_order_ab;

	int f_find_standard_generators_M24;

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

	int f_split_by_classes;
	std::string split_by_classes_fname;
	std::string split_by_classes_column;

	int f_identify_elements_by_class;
	std::string identify_elements_by_class_fname;
	std::string identify_elements_by_class_column;
	int identify_elements_by_class_expand_go;
	std::string identify_elements_by_class_supergroup;



	// undocumented (too specialized):
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
	int f_make_element_tree;

	int f_vector_ge_print_elements_tex;
	std::string vector_ge_print_elements_tex_label;

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


	// TABLES/group_theoretic_activity_3.tex


	int f_find_conjugating_element;
	std::string find_conjugating_element_element_from;
	std::string find_conjugating_element_element_to;


	int f_group_of_automorphisms_by_images_of_generators;
	std::string group_of_automorphisms_by_images_of_generators_label;
	std::string group_of_automorphisms_by_images_of_generators_elements;
	std::string group_of_automorphisms_by_images_of_generators_images;



	// this should become a vector_ge_activity:
	int f_order_of_products;
	std::string order_of_products_elements;

	int f_reverse_isomorphism_exterior_square;

	int f_reverse_isomorphism_exterior_square_vector_of_ge;
	std::string reverse_isomorphism_exterior_square_vector_of_ge_label;

	int f_is_subgroup_of;
	int f_coset_reps;

	// ToDo: not yet documented
	int f_diagram;


	// orbit stuff:

	int f_subgroup_lattice;

	int f_subgroup_lattice_load;
	std::string subgroup_lattice_load_fname;

	int f_subgroup_lattice_draw_by_orbits;
	std::string subgroup_lattice_draw_by_orbits_draw_options;

	int f_subgroup_lattice_draw_by_groups;
	std::string subgroup_lattice_draw_by_groups_draw_options;



	// TABLES/group_theoretic_activity_4.tex



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

	int f_create_flag_transitive_geometry;
	std::string create_flag_transitive_geometry_P;
	std::string create_flag_transitive_geometry_Q;

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
	layer5_applications::apps_geometry::ovoid_classify_description *Ovoid_classify_description;

	//int f_classify_cubic_curves;

	int f_representation_on_polynomials;
	std::string representation_on_polynomials_ring;

	// GAP:
	int f_canonical_image_GAP;
	std::string canonical_image_GAP_input_set;

	int f_canonical_image;
	std::string canonical_image_input_set;


	// TABLES/group_theoretic_activity_5.tex


	int f_subgroup_lattice_magma;

	int f_find_overgroup;
	int find_overgroup_order;
	std::string find_overgroup_of;


	int f_identify_subgroups_from_file;
	std::string identify_subgroups_from_file_fname;
	std::string identify_subgroups_from_file_col_label;
	int identify_subgroups_from_expand_go;

	// ToDo undocumented
	int f_permutation_subgroup;


	group_theoretic_activity_description();
	~group_theoretic_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// graph_classification_activity_description.cpp
// #############################################################################

//! an activity for a classification of graphs and tournaments


class graph_classification_activity_description {

public:

	// TABLES/graph_classification_activity.tex

	int f_draw_level_graph;
	int draw_level_graph_level;

	int f_draw_graphs;

	int f_list_graphs_at_level;
	int list_graphs_at_level_level_min;
	int list_graphs_at_level_level_max;

	int f_draw_graphs_at_level;
	int draw_graphs_at_level_level;

	int f_draw_options;
	std::string draw_options_label;
	//other::graphics::layered_graph_draw_options *draw_options;

	int f_recognize_graphs_from_adjacency_matrix_csv;
	std::string recognize_graphs_from_adjacency_matrix_csv_fname;


	graph_classification_activity_description();
	~graph_classification_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// graph_classification_activity.cpp
// #############################################################################

//! an activity for a classification of graphs and tournaments


class graph_classification_activity {

public:

	graph_classification_activity_description *Descr;
	layer5_applications::apps_graph_theory::graph_classify *GC;

	graph_classification_activity();
	~graph_classification_activity();
	void init(
			graph_classification_activity_description *Descr,
			layer5_applications::apps_graph_theory::graph_classify *GC,
			int verbose_level);
	void perform_activity(int verbose_level);


};



// #############################################################################
// graph_theoretic_activity_description.cpp
// #############################################################################

//! description of an activity for graphs


class graph_theoretic_activity_description {

public:

	// TABLES/graph_theoretic_activities_1.tex

	int f_find_cliques;
	combinatorics::graph_theory::clique_finder_control *Clique_finder_control;

	int f_test_SRG_property;

	int f_test_Neumaier_property;
	int test_Neumaier_property_clique_size;

	int f_find_subgraph;
	std::string find_subgraph_label;

	int f_test_automorphism_property_of_group;
	std::string test_automorphism_property_of_group_label;

	int f_common_neighbors;
	std::string common_neighbors_set;


	int f_export_magma;
	int f_export_maple;
	int f_export_csv;
	int f_export_graphviz;
	int f_print;
	int f_sort_by_colors;

	int f_A_powers;
	int f_A_powers_max;

	int f_distance_from;
	int distance_from_vertex;

	int f_split;
	std::string split_input_fname;
	std::string split_by_file;

	int f_split_by_starters;
	std::string split_by_starters_fname_reps;
	std::string split_by_starters_col_label;

	int f_combine_by_starters;
	std::string combine_by_starters_fname_reps;
	std::string combine_by_starters_col_label;
	std::string combine_by_starters_mask_fname_solutions; // added recently

	int f_split_by_clique;
	std::string split_by_clique_label;
	std::string split_by_clique_set;

	int f_save;


	// TABLES/graph_theoretic_activities_2.tex


	int f_automorphism_group;
	int f_automorphism_group_colored_graph;

	int f_properties;
	int f_eigenvalues;
	int f_eigenvalue_report;

	int f_draw;
	std::string draw_options;

	int f_create_distance_poset;
	int create_distance_poset_vertex;


	graph_theoretic_activity_description();
	~graph_theoretic_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};


// #############################################################################
// graph_theoretic_activity.cpp
// #############################################################################

//! an activity for graphs


class graph_theoretic_activity {

public:

	graph_theoretic_activity_description *Descr;
	int nb;
	combinatorics::graph_theory::colored_graph **CG; // [nb]


	graph_theoretic_activity();
	~graph_theoretic_activity();
	void init(
			graph_theoretic_activity_description *Descr,
			int nb,
			combinatorics::graph_theory::colored_graph **CG,
			int verbose_level);
	void feedback_headings(
			graph_theoretic_activity_description *Descr,
			std::string &headings,
			int &nb_cols,
			int verbose_level);
	void get_label(
			graph_theoretic_activity_description *Descr,
			std::string &description_txt,
			int verbose_level);
	void perform_activity(
			std::vector<std::string> &feedback,
			int verbose_level);


};




// #############################################################################
// group_theoretic_activity.cpp
// #############################################################################


//! perform an activity associated with a linear group

class group_theoretic_activity {
public:
	group_theoretic_activity_description *Descr;

	groups::any_group *AG;

	groups::any_group *AG_secondary; // used in is_subgroup_of, coset_reps


	int nb_output;
	other::orbiter_kernel_system::orbiter_symbol_table_entry *Output;


	group_theoretic_activity();
	~group_theoretic_activity();
	void init_group(
			group_theoretic_activity_description *Descr,
			groups::any_group *AG,
			int verbose_level);
	void init_secondary_group(
			group_theoretic_activity_description *Descr,
			groups::any_group *AG_secondary,
			int verbose_level);
	void perform_activity(
			int verbose_level);
	int perform_activity_part1(
			int verbose_level);
	int perform_activity_part2(
			int verbose_level);
	int perform_activity_part3(
			int verbose_level);
	int perform_activity_part4(
			int verbose_level);
	int perform_activity_part5(
			int verbose_level);

};

// #############################################################################
// large_set_activity_description.cpp
// #############################################################################

//! description of an activity for a spread table


class large_set_activity_description {
public:



	large_set_activity_description();
	~large_set_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// large_set_activity.cpp
// #############################################################################

//! an activity for a spread table


class large_set_activity {
public:

	large_set_activity_description *Descr;
	layer5_applications::apps_combinatorics::large_set_classify *Large_set_classify;



	large_set_activity();
	~large_set_activity();
	void perform_activity(
			large_set_activity_description *Descr,
			layer5_applications::apps_combinatorics::large_set_classify *large_set_classify, int verbose_level);

};


// #############################################################################
// large_set_was_activity_description.cpp
// #############################################################################

//! description of an activity for a large set search with assumed symmetry


class large_set_was_activity_description {
public:

	// TABLES/large_set_was_activity.tex

	int f_normalizer_on_orbits_of_a_given_length;
	int normalizer_on_orbits_of_a_given_length_length;
	int normalizer_on_orbits_of_a_given_length_nb_orbits;
	poset_classification::poset_classification_control
		*normalizer_on_orbits_of_a_given_length_control;

	int f_create_graph_on_orbits_of_length;
	std::string create_graph_on_orbits_of_length_fname;
	int create_graph_on_orbits_of_length_length;

	int f_create_graph_on_orbits_of_length_based_on_N_orbits;
	std::string create_graph_on_orbits_of_length_based_on_N_orbits_fname_mask;
	int create_graph_on_orbits_of_length_based_on_N_orbits_length;
	int create_graph_on_orbits_of_length_based_on_N_nb_N_orbits_preselected;
	int create_graph_on_orbits_of_length_based_on_N_orbits_r;
	int create_graph_on_orbits_of_length_based_on_N_orbits_m;

	int f_read_solution_file;
	int read_solution_file_orbit_length;
	std::string read_solution_file_name;



	large_set_was_activity_description();
	~large_set_was_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// large_set_was_activity.cpp
// #############################################################################

//! an activity for a large set search with assumed symmetry


class large_set_was_activity {
public:

	large_set_was_activity_description *Descr;
	layer5_applications::apps_combinatorics::large_set_was *LSW;


	large_set_was_activity();
	~large_set_was_activity();
	void perform_activity(
			large_set_was_activity_description *Descr,
			layer5_applications::apps_combinatorics::large_set_was *LSW,
			int verbose_level);
	void do_normalizer_on_orbits_of_a_given_length(
			int select_orbits_of_length_length, int verbose_level);

};



// #############################################################################
// orbits_activity_description.cpp
// #############################################################################


//! description of an action for orbits


class orbits_activity_description {

public:


	// TABLES/orbits_activity.tex

	int f_report;
	std::string report_options;

	int f_export_something;
	std::string export_something_what;
	int export_something_data1;

	int f_export_trees;

	int f_export_source_code;

	int f_export_levels;
	int export_levels_orbit_idx;

	int f_draw_tree;
	int draw_tree_idx;
	std::string draw_tree_draw_options;

	int f_stabilizer;
	int stabilizer_point;

	int f_stabilizer_of_orbit_rep;
	int stabilizer_of_orbit_rep_orbit_idx;

	int f_Kramer_Mesner_matrix;
	int Kramer_Mesner_t;
	int Kramer_Mesner_k;

	int f_recognize;
	std::vector<std::string> recognize;

	int f_transporter;
	std::string transporter_label_of_set;

	int f_schreier_poset;
	int schreier_poset_orbit_idx;
	std::string schreier_poset_prefix;

	int f_draw_options;
	std::string draw_options_label;

	int f_report_options;
	std::string report_options_label;


	int f_poset_classification_activity;
	std::string poset_classification_activity_label;

	//poset_classification::poset_classification_report_options
	//	*report_options;


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

	layer5_applications::orbits::orbits_create *OC;




	orbits_activity();
	~orbits_activity();
	void init(
			orbits_activity_description *Descr,
			layer5_applications::orbits::orbits_create *OC,
			int verbose_level);
	void perform_activity(
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);
	void do_report(
			std::string &options,
			int verbose_level);
	void do_export(
			int verbose_level);
	void do_export_trees(
			int verbose_level);
	void do_export_source_code(
			int verbose_level);
	void do_export_levels(
			int orbit_idx, int verbose_level);
	void do_draw_tree(
			other::graphics::draw_options *Draw_options,
			int verbose_level);
	void do_stabilizer(
			int verbose_level);
	void do_stabilizer_of_orbit_rep(
			int verbose_level);
	void do_Kramer_Mesner_matrix(
			int verbose_level);
	void do_recognize(
			int verbose_level);
	void do_transporter(
			std::string &label_of_set, int verbose_level);
	void do_schreier_poset(
			std::string &prefix, int orbit_idx, int verbose_level);
	void do_poset_classification_activity(
			std::string &activity_label,
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);

};


// #############################################################################
// orthogonal_space_activity_description.cpp
// #############################################################################

//! description of an activity associated with an orthogonal space


class orthogonal_space_activity_description {
public:

	// TABLES/orthogonal_space_activity.csv

	int f_cheat_sheet_orthogonal;
	std::string cheat_sheet_orthogonal_draw_options_label;

	int f_print_points;
	std::string print_points_label;

	int f_print_lines;
	std::string print_lines_label;

	int f_unrank_line_through_two_points;
	std::string unrank_line_through_two_points_p1;
	std::string unrank_line_through_two_points_p2;

	int f_lines_on_point;
	long int lines_on_point_rank;

	int f_perp;
	std::string perp_text;

	// undocumented:
	int f_set_stabilizer;
	int set_stabilizer_intermediate_set_size;
	std::string set_stabilizer_fname_mask;
	int set_stabilizer_nb;
	std::string set_stabilizer_column_label;
	std::string set_stabilizer_fname_out;

	int f_export_point_line_incidence_matrix;

	int f_intersect_with_subspace;
	std::string intersect_with_subspace_label;

	int f_intersect_with_subspace_given_by_base;
	std::string intersect_with_subspace_given_by_base_label;

	int f_table_of_blt_sets;

	int f_create_orthogonal_reflection;
	std::string create_orthogonal_reflection_points;

	int f_create_perp_of_points;
	std::string create_perp_of_points_points;

	int f_create_Siegel_transformation;
	std::string create_Siegel_transformation_u;
	std::string create_Siegel_transformation_v;

	int f_make_all_Siegel_transformations;

	int f_create_orthogonal_reflection_6_and_4;
	std::string create_orthogonal_reflection_6_and_4_points;
	std::string create_orthogonal_reflection_6_and_4_A4;


	orthogonal_space_activity_description();
	~orthogonal_space_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};

// #############################################################################
// orthogonal_space_activity.cpp
// #############################################################################

//! an activity associated with an orthogonal space


class orthogonal_space_activity {
public:

	orthogonal_space_activity_description *Descr;

	layer5_applications::orthogonal_geometry_applications::orthogonal_space_with_action *OA;

	orthogonal_space_activity();
	~orthogonal_space_activity();
	void init(
			orthogonal_space_activity_description *Descr,
			layer5_applications::orthogonal_geometry_applications::orthogonal_space_with_action *OA,
			int verbose_level);
	void perform_activity(
			int verbose_level);
	void set_stabilizer(
			layer5_applications::orthogonal_geometry_applications::orthogonal_space_with_action *OA,
			int intermediate_subset_size,
			std::string &fname_mask, int nb, std::string &column_label,
			std::string &fname_out,
			int verbose_level);


};


// #############################################################################
// packing_classify_activity_description.cpp
// #############################################################################

//! description of an activity for an object of type classification of packings in PG(3,q)

class packing_classify_activity_description {

public:

	// packing_classify.csv

	int f_report;

	int f_classify;
	std::string classify_control_label;

	int f_make_graph_of_disjoint_spreads;

	int f_export_group_on_spreads;

	packing_classify_activity_description();
	~packing_classify_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};


// #############################################################################
// packing_classify_activity.cpp
// #############################################################################

//! performs an activity for an object of type classification of packings in PG(3,q)

class packing_classify_activity {

public:

	packing_classify_activity_description *Descr;
	layer5_applications::packings::packing_classify *Packing_classify;

	packing_classify_activity();
	~packing_classify_activity();
	void init(
			packing_classify_activity_description *Descr,
			layer5_applications::packings::packing_classify *Packing_classify,
			int verbose_level);
	void perform_activity(
			int verbose_level);

};


// #############################################################################
// packing_was_activity_description.cpp
// #############################################################################

//! description of an activity involving a packing_was

class packing_was_activity_description {
public:

	// TABLES/packing_was_activity.tex

	int f_report;

	int f_export_reduced_spread_orbits;
	std::string export_reduced_spread_orbits_fname_base;

	int f_create_graph_on_mixed_orbits;
	std::string create_graph_on_mixed_orbits_orbit_lengths;


	packing_was_activity_description();
	~packing_was_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// packing_was_activity.cpp
// #############################################################################

//! an activity involving a packing_was

class packing_was_activity {
public:

	packing_was_activity_description *Descr;
	layer5_applications::packings::packing_was *PW;

	packing_was_activity();
	~packing_was_activity();
	void init(
			packing_was_activity_description *Descr,
			layer5_applications::packings::packing_was *PW,
			int verbose_level);
	void perform_activity(
			int verbose_level);
};



// #############################################################################
// packing_was_fixpoints_activity_description.cpp
// #############################################################################

//! description of an activity after the fixed points have been selected in the construction of packings in PG(3,q) with assumed symmetry

class packing_was_fixpoints_activity_description {
public:

	// TABLES/packing_was_fixpoints_activity.tex

	int f_report;

	int f_print_packing;
	std::string print_packing_text;

	int f_compare_files_of_packings;
	std::string compare_files_of_packings_fname1;
	std::string compare_files_of_packings_fname2;

	packing_was_fixpoints_activity_description();
	~packing_was_fixpoints_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};

// #############################################################################
// packing_was_fixpoints_activity.cpp
// #############################################################################

//! an activity after the fixed points have been selected in the construction of packings in PG(3,q) with assumed symmetry

class packing_was_fixpoints_activity {
public:

	packing_was_fixpoints_activity_description *Descr;
	layer5_applications::packings::packing_was_fixpoints *PWF;

	packing_was_fixpoints_activity();
	~packing_was_fixpoints_activity();
	void init(
			packing_was_fixpoints_activity_description *Descr,
			layer5_applications::packings::packing_was_fixpoints *PWF,
			int verbose_level);
	void perform_activity(
			int verbose_level);

};





// #############################################################################
// plesken_ring_activity_description.cpp
// #############################################################################

//! description of an activity for a plesken_ring


class plesken_ring_activity_description {
public:

	int f_report;
	std::string report_draw_options_label;

	int f_evaluate_join;
	std::string evaluate_join_ring_label;
	std::string evaluate_join_formula_label;

	int f_evaluate_meet;
	std::string evaluate_meet_ring_label;
	std::string evaluate_meet_formula_label;

	plesken_ring_activity_description();
	~plesken_ring_activity_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();


};


// #############################################################################
// plesken_ring_activity.cpp
// #############################################################################

//! an activity for a plesken_ring


class plesken_ring_activity {
public:

	plesken_ring_activity_description *Descr;
	layer5_applications::apps_combinatorics::plesken_ring *Plesken_ring;

	plesken_ring_activity();
	~plesken_ring_activity();
	void init(
			plesken_ring_activity_description *Descr,
			layer5_applications::apps_combinatorics::plesken_ring *Plesken_ring,
			int verbose_level);
	void perform_activity(
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);


};



// #############################################################################
// polynomial_ring_activity.cpp
// #############################################################################


//! a polynomial ring activity

class polynomial_ring_activity {
public:


	// used as -ring_theoretic_activity


	layer6_user_interface::activities_layer1::polynomial_ring_activity_description *Descr;

	algebra::ring_theory::homogeneous_polynomial_domain *HPD;


	int nb_output;
	other::orbiter_kernel_system::orbiter_symbol_table_entry *Output;


	polynomial_ring_activity();
	~polynomial_ring_activity();
	void init(
			layer6_user_interface::activities_layer1::polynomial_ring_activity_description *Descr,
			algebra::ring_theory::homogeneous_polynomial_domain *HPD,
			int verbose_level);
	void perform_activity(
			int verbose_level);

};



// #############################################################################
// projective_space_activity_description.cpp
// #############################################################################

//! description of an activity associated with a projective space


class projective_space_activity_description {
public:

	// TABLES/projective_space_activity_1.tex

	int f_cheat_sheet;
	std::string cheat_sheet_draw_options_label;

	int f_print_points;
	std::string print_points_label;


	int f_export_lines;
	int f_export_point_line_incidence_matrix;

	int f_export_restricted_point_line_incidence_matrix;
	std::string export_restricted_point_line_incidence_matrix_rows;
	std::string export_restricted_point_line_incidence_matrix_cols;

	int f_export_cubic_surface_line_vs_line_incidence_matrix;

	int f_export_cubic_surface_line_tritangent_plane_incidence_matrix;

	int f_export_double_sixes;


	int f_table_of_cubic_surfaces_compute_properties;
	std::string table_of_cubic_surfaces_compute_fname_csv;
	int table_of_cubic_surfaces_compute_defining_q;
	int table_of_cubic_surfaces_compute_column_offset;

	int f_cubic_surface_properties_analyze;
	std::string cubic_surface_properties_fname_csv;
	int cubic_surface_properties_defining_q;

	int f_canonical_form_of_code;
	std::string canonical_form_of_code_label;
	std::string canonical_form_of_code_generator_matrix;
	combinatorics::canonical_form_classification::classification_of_objects_description
		*Canonical_form_codes_Descr;

	int f_map;
	std::string map_ring_label;
	std::string map_formula_label;
	std::string map_parameters;

	int f_affine_map;
	std::string affine_map_ring_label;
	std::string affine_map_formula_label;
	std::string affine_map_parameters;

	int f_projective_variety;
	std::string projective_variety_ring_label;
	std::string projective_variety_formula_label;
	std::string projective_variety_parameters;

	int f_analyze_del_Pezzo_surface;
	std::string analyze_del_Pezzo_surface_label;
	std::string analyze_del_Pezzo_surface_parameters;

	int f_decomposition_by_element_PG;
	int decomposition_by_element_power;
	std::string decomposition_by_element_data;
	std::string decomposition_by_element_fname;

	int f_decomposition_by_subgroup;
	std::string decomposition_by_subgroup_label;
	group_constructions::linear_group_description
		* decomposition_by_subgroup_Descr;

	int f_table_of_quartic_curves;
		// based on knowledge_base

	int f_table_of_cubic_surfaces;
		// based on knowledge_base



	// TABLES/projective_space_activity_2.tex




	int f_sweep;
	std::string sweep_options;
	layer5_applications::applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description
		*sweep_surface_description;

	int f_set_stabilizer;
	int set_stabilizer_intermediate_set_size;
	std::string set_stabilizer_fname_mask;
	int set_stabilizer_nb;
	std::string set_stabilizer_column_label;
	std::string set_stabilizer_fname_out;

	int f_conic_type;
	int conic_type_threshold;
	std::string conic_type_set_text;

	// undocumented:
	int f_lift_skew_hexagon;
	std::string lift_skew_hexagon_text;

	// undocumented:
	int f_lift_skew_hexagon_with_polarity;
	std::string lift_skew_hexagon_with_polarity_polarity;

	int f_arc_with_given_set_as_s_lines_after_dualizing;
	int arc_size;
	int arc_d;
	int arc_d_low;
	int arc_s;
	std::string arc_input_set;
	std::string arc_label;

	int f_arc_with_two_given_sets_of_lines_after_dualizing;
	int arc_t;
	std::string t_lines_string;

	int f_arc_with_three_given_sets_of_lines_after_dualizing;
	int arc_u;
	std::string u_lines_string;

	int f_dualize_hyperplanes_to_points;
	int f_dualize_points_to_hyperplanes;
	std::string dualize_input_set;

	int f_dualize_rank_k_subspaces;
	int dualize_rank_k_subspaces_k;


	// TABLES/projective_space_activity_3.tex




	int f_lines_on_point_but_within_a_plane;
	long int lines_on_point_but_within_a_plane_point_rk;
	long int lines_on_point_but_within_a_plane_plane_rk;

	int f_union_of_lines;
	std::string union_of_lines_text;

	int f_rank_lines_in_PG;
	std::string rank_lines_in_PG_label;

	int f_unrank_lines_in_PG;
	std::string unrank_lines_in_PG_text;

	int f_points_on_lines;
	std::string points_on_lines_text;

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

	int f_planes_through_line;
	std::string planes_through_line_rank;

	int f_restricted_incidence_matrix;
	int restricted_incidence_matrix_type_row_objects;
	int restricted_incidence_matrix_type_col_objects;
	std::string restricted_incidence_matrix_row_objects;
	std::string restricted_incidence_matrix_col_objects;
	std::string restricted_incidence_matrix_file_name;

	//int f_make_relation;
	//long int make_relation_plane_rk;

	int f_plane_intersection_type;
	int plane_intersection_type_threshold;
	std::string plane_intersection_type_input;


	int f_plane_intersection_type_of_klein_image;
	int plane_intersection_type_of_klein_image_threshold;
	std::string plane_intersection_type_of_klein_image_input;

	int f_report_Grassmannian;
	int report_Grassmannian_k;

	int f_report_fixed_objects;
	std::string report_fixed_objects_Elt;
	std::string report_fixed_objects_label;


	int f_evaluation_matrix;
	std::string evaluation_matrix_ring;

	int f_polynomial_representation;
	std::string polynomial_representation_set_label;


	// TABLES/projective_space_activity_4.tex



	// classification stuff:


	int f_classify_surfaces_through_arcs_and_two_lines;

	int f_test_nb_Eckardt_points;
	int nb_E;

	int f_classify_surfaces_through_arcs_and_trihedral_pairs;
	std::string classify_surfaces_through_arcs_and_trihedral_pairs_draw_options_label;

	int f_trihedra1_control;
	poset_classification::poset_classification_control
		*Trihedra1_control;
	int f_trihedra2_control;
	poset_classification::poset_classification_control
		*Trihedra2_control;

	int f_control_six_arcs;
	std::string Control_six_arcs_label;

#if 0
	int f_six_arcs_not_on_conic;
	int f_filter_by_nb_Eckardt_points;
	int nb_Eckardt_points;
#endif

	//int f_classify_arcs;
	//apps_geometry::arc_generator_description
	//	*Arc_generator_description;

	//int f_classify_cubic_curves;



#if 0
	int f_classify_semifields;
	semifields::semifield_classify_description
		*Semifield_classify_description;
	poset_classification::poset_classification_control
		*Semifield_classify_Control;

	int f_classify_bent_functions;
	int classify_bent_functions_n;
#endif

	projective_space_activity_description();
	~projective_space_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};


// #############################################################################
// projective_space_activity.cpp
// #############################################################################

//! an activity associated with a projective space


class projective_space_activity {
public:

	projective_space_activity_description *Descr;

	layer5_applications::projective_geometry::projective_space_with_action *PA;


	projective_space_activity();
	~projective_space_activity();
	void perform_activity(
			int verbose_level);
	void do_rank_lines_in_PG(
			std::string &label,
			int verbose_level);

};




// #############################################################################
// quartic_curve_activity_description.cpp
// #############################################################################

//! description of an activity associated with a quartic curve


class quartic_curve_activity_description {
public:


	// quartic_curve_activity.tex

	int f_report;

	int f_export;

	int f_export_something;
	std::string export_something_what;


	int f_create_surface;

	int f_extract_orbit_on_bitangents_by_length;
	int extract_orbit_on_bitangents_by_length_length;

	int f_extract_specific_orbit_on_bitangents_by_length;
	int extract_specific_orbit_on_bitangents_by_length_length;
	int extract_specific_orbit_on_bitangents_by_length_index;

	int f_extract_specific_orbit_on_kovalevski_points_by_length;
	int extract_specific_orbit_on_kovalevski_points_by_length_length;
	int extract_specific_orbit_on_kovalevski_points_by_length_index;

	int f_get_Kovalevski_configuration;

	quartic_curve_activity_description();
	~quartic_curve_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// quartic_curve_activity.cpp
// #############################################################################

//! an activity associated with a quartic curve


class quartic_curve_activity {
public:

	quartic_curve_activity_description *Descr;
	layer5_applications::applications_in_algebraic_geometry::quartic_curves::quartic_curve_create *QC;


	quartic_curve_activity();
	~quartic_curve_activity();
	void init(
			quartic_curve_activity_description
				*Quartic_curve_activity_description,
				layer5_applications::applications_in_algebraic_geometry::quartic_curves::quartic_curve_create *QC, int verbose_level);
	void perform_activity(
			other::orbiter_kernel_system::activity_output *&AO,
			int verbose_level);

};



// #############################################################################
// spread_activity_description.cpp
// #############################################################################

//! description of an activity for an object of type spread



class spread_activity_description {

public:

	// TABLES/spread_activity.tex

	int f_report;

	spread_activity_description();
	~spread_activity_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();


};



// #############################################################################
// spread_activity.cpp
// #############################################################################

//! an activity for an object of type spread



class spread_activity {

public:

	spread_activity_description *Descr;
	layer5_applications::spreads::spread_create *Spread_create;
	geometry::finite_geometries::spread_domain *SD;

	actions::action *A;
		// PGGL(n,q)
	actions::action *A2;
		// action of A on grassmannian
		// of k-subspaces of V(n,q)
	induced_actions::action_on_grassmannian *AG;

	actions::action *AGr;


	spread_activity();
	~spread_activity();
	void init(
			spread_activity_description *Descr,
			layer5_applications::spreads::spread_create *Spread_create,
			int verbose_level);
	void perform_activity(
			int verbose_level);
	void report(
			int verbose_level);
	void report2(
			std::ostream &ost, int verbose_level);

};







// #############################################################################
// spread_classify_activity_description.cpp
// #############################################################################

//! description of an activity regarding the classification of spreads



class spread_classify_activity_description {

public:


	// TABLES/spread_classify_activity.tex

	int f_compute_starter;

	int f_prepare_lifting_single_case;
	int prepare_lifting_single_case_case_number;

	int f_prepare_lifting_all_cases;

	int f_split;
	int split_r;
	int split_m;

	int f_isomorph;
	std::string isomorph_label;

	int f_build_db;
	int f_read_solutions;
	int f_compute_orbits;
	int f_isomorph_testing;
	int f_isomorph_report;


	spread_classify_activity_description();
	~spread_classify_activity_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// spread_classify_activity.cpp
// #############################################################################

//! an activity regarding the classification of spreads



class spread_classify_activity {

public:

	spread_classify_activity_description *Descr;
	layer5_applications::spreads::spread_classify *Spread_classify;

	spread_classify_activity();
	~spread_classify_activity();
	void init(
			spread_classify_activity_description *Descr,
			layer5_applications::spreads::spread_classify *Spread_classify,
			int verbose_level);
	void perform_activity(
			int verbose_level);
	void create_context(
			std::string &isomorph_label,
			layer4_classification::isomorph::isomorph_context *&Isomorph_context,
			int verbose_level);

};






// #############################################################################
// spread_table_activity_description.cpp
// #############################################################################

//! description of an activity for a spread table


class spread_table_activity_description {
public:

	// TABLES/spread_table_activity.csv

	int f_find_spread;
	std::string find_spread_text;

	int f_find_spread_and_dualize;
	std::string find_spread_and_dualize_text;

	int f_dualize_packing;
	std::string dualize_packing_text;

	int f_print_spreads;
	std::string print_spreads_idx_text;

	int f_export_spreads_to_csv;
	std::string export_spreads_to_csv_fname;
	std::string export_spreads_to_csv_idx_text;


	int f_find_spreads_containing_two_lines;
	int find_spreads_containing_two_lines_line1;
	int find_spreads_containing_two_lines_line2;

	int f_find_spreads_containing_one_line;
	int find_spreads_containing_one_line_line_idx;

	int f_isomorphism_type_of_spreads;
	std::string isomorphism_type_of_spreads_list;


	int f_dualize_packings;
	std::string dualize_packings_fname_in;
	std::string dualize_packings_fname_out;
	std::string dualize_packings_col_label;


	spread_table_activity_description();
	~spread_table_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// spread_table_activity.cpp
// #############################################################################

//! an activity for a spread table


class spread_table_activity {
public:

	spread_table_activity_description *Descr;

	layer5_applications::spreads::spread_table_with_selection *Spread_table_with_selection;

	//packings::packing_classify *P;
	// why is this here?


	spread_table_activity();
	~spread_table_activity();
	void init(
			spread_table_activity_description *Descr,
			layer5_applications::spreads::spread_table_with_selection *Spread_table_with_selection,
			int verbose_level);
	void perform_activity(
			int verbose_level);
	void export_spreads_to_csv(
			std::string &fname,
			int *spread_idx, int nb, int verbose_level);
	void packings_dualize(
			std::string &fname_in, std::string &fname_out, std::string &col_label,
			int verbose_level);
	void dualize_packings(
			long int *Packing_in, int nb_packings, int packing_sz,
			long int *&Dual_packings,
			int verbose_level);
	void report_spreads(
			int *spread_idx, int nb, int verbose_level);
	void report_spread2(
			std::ostream &ost, int spread_idx, int verbose_level);

};




// #############################################################################
// translation_plane_activity_description.cpp
// #############################################################################

//! description of an activity regarding a translation plane



class translation_plane_activity_description {

public:

	// TABLES/translation_plane_activity.tex

	int f_export_incma;

	int f_p_rank;
	int p_rank_p;

	int f_report;

	translation_plane_activity_description();
	~translation_plane_activity_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// translation_plane_activity.cpp
// #############################################################################

//! an activity regarding a translation plane



class translation_plane_activity {

public:

	translation_plane_activity_description *Descr;
	combinatorics_with_groups::translation_plane_via_andre_model *TP;


	translation_plane_activity();
	~translation_plane_activity();
	void init(
			translation_plane_activity_description *Descr,
			combinatorics_with_groups::translation_plane_via_andre_model *TP,
			int verbose_level);
	void perform_activity(
			int verbose_level);

};






// #############################################################################
// vector_ge_activity_description.cpp
// #############################################################################



//! description of an activity associated with a vector of group elements


class vector_ge_activity_description {

public:

	// TABLES/vector_ge_activity.csv

	int f_report;

	int f_report_with_options;
	std::string report_options;

	int f_report_elements_coded;

	int f_export_GAP;

	int f_transform_variety;
	std::string transform_variety_label;

	int f_multiply;

	int f_conjugate;

	int f_conjugate_by;
	std::string conjugate_by_data;

	int f_conjugate_inverse;

	int f_select_subset;
	std::string select_subset_vector_label;

	int f_field_reduction;
	int field_reduction_subfield_index;

	int f_rational_canonical_form;
	// returns two vectors:
	// the rational canonical forms and the base change matrices

	int f_products_of_pairs;

	int f_order_of_products_of_pairs;

	int f_apply_isomorphism_wedge_product_4to6;

	// ToDo not yet documented
	int f_filter_subfield;
	int subfield_index;


	vector_ge_activity_description();
	~vector_ge_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// variety_activity_description.cpp
// #############################################################################




//! description of an activity for a variety object



class variety_activity_description {

public:

	// TABLES/variety_activity.tex

	int f_compute_group;

	int f_test_isomorphism;

	int f_compute_set_stabilizer;


	int f_nauty_control;
	other::l1_interfaces::nauty_interface_control *Nauty_interface_control;


	int f_report;

	int f_export;

	// ToDo: undocumented
	int f_classify; // not yet implemented

	int f_apply_transformation_to_self;
	std::string apply_transformation_to_self_group_element;

	int f_singular_points;

	int f_output_fname_base;
	std::string output_fname_base;


	variety_activity_description();
	~variety_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};


// #############################################################################
// variety_activity.cpp
// #############################################################################


//! performs an activity associated with a variety

class variety_activity {
public:

	variety_activity_description *Descr;

	int nb_input_Vo;
	layer5_applications::canonical_form::variety_object_with_action **Input_Vo; // [nb_input_Vo]



	variety_activity();
	~variety_activity();
	void init(
			variety_activity_description *Descr,
			int nb_input_Vo,
			layer5_applications::canonical_form::variety_object_with_action **Input_Vo,
			int verbose_level);
	void perform_activity(
			int verbose_level);
	void do_compute_group(
			int f_has_output_fname_base,
			std::string &output_fname_base,
			int f_nauty_control,
			other::l1_interfaces::nauty_interface_control *Nauty_interface_control,
			int verbose_level);
	void do_test_isomorphism(
			int f_has_output_fname_base,
			std::string &output_fname_base,
			int f_nauty_control,
			other::l1_interfaces::nauty_interface_control *Nauty_interface_control,
			int verbose_level);
	void do_compute_set_stabilizer(
			int f_has_output_fname_base,
			std::string &output_fname_base,
			int f_nauty_control,
			other::l1_interfaces::nauty_interface_control *Nauty_interface_control,
			int verbose_level);
	void do_apply_transformation_to_self(
			int f_inverse,
			std::string &transformation_coded,
			int verbose_level);
	void do_singular_points(
			int verbose_level);

};





// #############################################################################
// vector_ge_activity.cpp
// #############################################################################



//! an activity associated with a vector of group elements


class vector_ge_activity {

public:

	vector_ge_activity_description *Descr;

	int nb_objects;

	std::vector<std::string> *with_labels;

	layer5_applications::apps_algebra::vector_ge_builder **VB; // [nb_objects]

	data_structures_groups::vector_ge **vec; // [nb_objects]

	int nb_output;
	other::orbiter_kernel_system::orbiter_symbol_table_entry *Output; // [nb_output]

	vector_ge_activity();
	~vector_ge_activity();
	void init(
			vector_ge_activity_description *Descr,
			layer5_applications::apps_algebra::vector_ge_builder **VB,
			int nb_objects,
			std::vector<std::string> &with_labels,
			int verbose_level);
	void perform_activity(
			int verbose_level);


};






}}}





#endif /* SRC_LIB_LAYER6_USER_INTERFACE_ACTIVITIES_LAYER5_ACTIVITIES_LAYER5_H_ */
