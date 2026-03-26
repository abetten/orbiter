/*
 * core_system.h
 *
 *  Created on: Mar 22, 2026
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER5_TOP_LEVEL_USER_INTERFACE_CORE_SYSTEM_CORE_SYSTEM_H_
#define SRC_LIB_LAYER5_TOP_LEVEL_USER_INTERFACE_CORE_SYSTEM_CORE_SYSTEM_H_


namespace orbiter {
namespace layer5_applications {
namespace user_interface {
namespace core_system {



// #############################################################################
// activity_description.cpp
// #############################################################################

//! description of an activity for an orbiter symbol


class activity_description {

public:

	user_interface::activities::interface_symbol_table *Sym;


	int f_finite_field_activity;
	user_interface::activities_layer1::finite_field_activity_description
		*Finite_field_activity_description;

	int f_polynomial_ring_activity;
	user_interface::activities_layer1::polynomial_ring_activity_description
		*Polynomial_ring_activity_description;

	int f_projective_space_activity;
	user_interface::activities_layer5::projective_space_activity_description
		*Projective_space_activity_description;

	int f_orthogonal_space_activity;
	user_interface::activities_layer5::orthogonal_space_activity_description
		*Orthogonal_space_activity_description;

	int f_group_theoretic_activity;
	user_interface::activities_layer5::group_theoretic_activity_description
		*Group_theoretic_activity_description;

	int f_coding_theoretic_activity;
	user_interface::activities_layer5::coding_theoretic_activity_description
		*Coding_theoretic_activity_description;

	int f_cubic_surface_activity;
	user_interface::activities_layer5::cubic_surface_activity_description
		*Cubic_surface_activity_description;

	int f_quartic_curve_activity;
	user_interface::activities_layer5::quartic_curve_activity_description
		*Quartic_curve_activity_description;

	int f_blt_set_activity;
	user_interface::activities_layer5::blt_set_activity_description
		*Blt_set_activity_description;

	int f_combinatorial_object_activity;
	user_interface::activities_layer5::combinatorial_object_activity_description
		*Combinatorial_object_activity_description;

	int f_graph_theoretic_activity;
	user_interface::activities_layer5::graph_theoretic_activity_description
		*Graph_theoretic_activity_description;

	int f_classification_of_cubic_surfaces_with_double_sixes_activity;
	user_interface::activities_layer5::classification_of_cubic_surfaces_with_double_sixes_activity_description
		*Classification_of_cubic_surfaces_with_double_sixes_activity_description;

	int f_spread_table_activity;
	user_interface::activities_layer5::spread_table_activity_description
		*Spread_table_activity_description;

	int f_packing_classify_activity_description;
	user_interface::activities_layer5::packing_classify_activity_description *Packing_classify_activity_description;

	int f_packing_with_symmetry_assumption_activity;
	user_interface::activities_layer5::packing_was_activity_description
		*Packing_was_activity_description;

	int f_packing_fixed_points_activity;
	user_interface::activities_layer5::packing_was_fixpoints_activity_description
		*Packing_was_fixpoints_activity_description;

	int f_graph_classification_activity;
	user_interface::activities_layer5::graph_classification_activity_description
		*Graph_classification_activity_description;

	int f_diophant_activity;
	user_interface::activities_layer1::diophant_activity_description
		*Diophant_activity_description;

	int f_design_activity;
	user_interface::activities_layer5::design_activity_description
		*Design_activity_description;

	// ToDo: not documented
	int f_large_set_activity;
	user_interface::activities_layer5::large_set_activity_description
		*Large_set_activity_description;

	int f_large_set_was_activity;
	user_interface::activities_layer5::large_set_was_activity_description
		*Large_set_was_activity_description;

	int f_symbolic_object_activity;
	user_interface::activities_layer1::symbolic_object_activity_description
		*Symbolic_object_activity_description;

	int f_BLT_set_classify_activity;
	user_interface::activities_layer5::blt_set_classify_activity_description
		*Blt_set_classify_activity_description;

	int f_spread_classify_activity;
	user_interface::activities_layer5::spread_classify_activity_description
		*Spread_classify_activity_description;

	int f_spread_activity;
	user_interface::activities_layer5::spread_activity_description
		*Spread_activity_description;

	int f_translation_plane_activity;
	user_interface::activities_layer5::translation_plane_activity_description
		*Translation_plane_activity_description;

	int f_action_on_forms_activity;
	user_interface::activities_layer5::action_on_forms_activity_description
		*Action_on_forms_activity_description;

	int f_orbits_activity;
	user_interface::activities_layer5::orbits_activity_description
		*Orbits_activity_description;

	int f_variety_activity;
	user_interface::activities_layer5::variety_activity_description
		*Variety_activity_description;

	int f_vector_ge_activity;
	user_interface::activities_layer5::vector_ge_activity_description
		*Vector_ge_activity_description;

	int f_combo_activity;
	user_interface::activities_layer5::combo_activity_description *Combo_activity_description;


	// ToDo: not documented
	int f_plesken_ring_activity;
	user_interface::activities_layer5::plesken_ring_activity_description *Plesken_ring_activity_description;


	activity_description();
	~activity_description();
	void read_arguments(
			int argc, std::string *argv, int &i,
			int verbose_level);
	void worker(
			std::vector<std::string> &with_labels,
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);
	void print();
	void do_finite_field_activity(
			other::orbiter_kernel_system::activity_output *&AO,
			int verbose_level);
	void do_ring_theoretic_activity(
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);
	void do_projective_space_activity(
			int verbose_level);
	void do_orthogonal_space_activity(
			int verbose_level);
	void do_group_theoretic_activity(
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);
	void do_coding_theoretic_activity(
			int verbose_level);
	void do_cubic_surface_activity(
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			other::orbiter_kernel_system::activity_output *&AO,
			int verbose_level);
	void do_quartic_curve_activity(
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			other::orbiter_kernel_system::activity_output *&AO,
			int verbose_level);
	void do_blt_set_activity(
			int verbose_level);
	void do_combinatorial_object_activity(
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			other::orbiter_kernel_system::activity_output *&AO,
			int verbose_level);
	void do_graph_theoretic_activity(
			int verbose_level);
	void do_classification_of_cubic_surfaces_with_double_sixes_activity(
			int verbose_level);
	void do_spread_table_activity(
			int verbose_level);
	void do_packing_classify_activity(
			int verbose_level);
	void do_packing_was_activity(
			int verbose_level);
	void do_packing_fixed_points_activity(
			int verbose_level);
	void do_graph_classification_activity(
			int verbose_level);
	void do_diophant_activity(
			int verbose_level);
	void do_design_activity(
			int verbose_level);
	void do_large_set_activity(
			int verbose_level);
	void do_large_set_was_activity(
			int verbose_level);
	void do_symbolic_object_activity(
			int verbose_level);
	void do_BLT_set_classify_activity(
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);
	void do_spread_classify_activity(
			int verbose_level);
	void do_spread_activity(
			int verbose_level);
	void do_translation_plane_activity(
			int verbose_level);
	void do_action_on_forms_activity(
			int verbose_level);
	void do_orbits_activity(
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);
	void do_variety_activity(
			int verbose_level);
	void do_vector_ge_activity(
			std::vector<std::string> &with_labels,
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);
	void do_combo_activity(
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);
	void do_plesken_ring_activity(
			int &nb_output,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
			int verbose_level);

};



// #############################################################################
// orbiter_command.cpp
// #############################################################################



//! a single command in the Orbiter dash code language


class orbiter_command {

public:

	orbiter_top_level_session *Orbiter_top_level_session;


	int f_algebra;
	user_interface::activities::interface_algebra *Algebra;

	int f_coding_theory;
	user_interface::activities::interface_coding_theory *Coding_theory;

	int f_combinatorics;
	user_interface::activities::interface_combinatorics *Combinatorics;

	int f_cryptography;
	user_interface::activities::interface_cryptography *Cryptography;

	int f_povray;
	user_interface::activities::interface_povray *Povray;

	int f_projective;
	user_interface::activities::interface_projective *Projective;

	int f_symbol_table;
	user_interface::activities::interface_symbol_table *Symbol_table;

	int f_toolkit;
	user_interface::activities::interface_toolkit *Toolkit;

	orbiter_command();
	~orbiter_command();
	void parse(
			orbiter_top_level_session *Orbiter_top_level_session,
			int argc, std::string *Argv, int &i, int verbose_level);
	void execute(
			int verbose_level);
	void print();

};


// #############################################################################
// orbiter_top_level_session.cpp
// #############################################################################



//! The top level orbiter session is responsible for the command line interface and the program execution and for the orbiter_session


class orbiter_top_level_session {

public:

	other::orbiter_kernel_system::orbiter_session *Orbiter_session;

	orbiter_top_level_session();
	~orbiter_top_level_session();
	void execute_command_line(
			int argc, const char **argv, int verbose_level);
	int startup_and_read_arguments(
			int argc,
			std::string *argv, int i0, int verbose_level);
	void handle_everything(
			int argc, std::string *Argv, int i, int verbose_level);
	void parse_and_execute(
			int argc, std::string *Argv, int i, int verbose_level);
	void parse(
			int argc, std::string *Argv, int &i,
			std::vector<void * > &program, int verbose_level);
	void *get_object(
			int idx);
	layer1_foundations::other::orbiter_kernel_system::symbol_table_object_type get_object_type(
			int idx);
	int find_symbol(
			std::string &label);
	void find_symbols(
			std::vector<std::string> &Labels, int *&Idx);
	void print_symbol_table();
	void add_symbol_table_entry(
			std::string &label,
			other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb,
			int verbose_level);
	groups::any_group *get_any_group(
			std::string &label);
	projective_geometry::projective_space_with_action
		*get_object_of_type_projective_space(
			std::string &label);
	spreads::spread_table_with_selection
		*get_object_of_type_spread_table(
				std::string &label);
	packings::packing_classify
		*get_packing_classify(
				std::string &label);
	poset_classification::poset_classification_control
		*get_poset_classification_control(
				std::string &label);
	poset_classification::poset_classification_report_options
		*get_poset_classification_report_options(
				std::string &label);
	apps_geometry::arc_generator_description
		*get_object_of_type_arc_generator_control(
				std::string &label);
	user_interface::activities_layer5::poset_classification_activity_description
		*get_object_of_type_poset_classification_activity(
				std::string &label);
	void get_vector_or_set(std::string &label,
			long int *&Pts, int &nb_pts, int verbose_level);
	apps_algebra::vector_ge_builder
		*get_object_of_type_vector_ge(
				std::string &label);
	orthogonal_geometry_applications::orthogonal_space_with_action
		*get_object_of_type_orthogonal_space_with_action(
				std::string &label);
	spreads::spread_create
		*get_object_of_type_spread(
				std::string &label);
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create
		*get_object_of_type_cubic_surface(
				std::string &label);
	apps_coding_theory::create_code
		*get_object_of_type_code(
				std::string &label);
	combinatorics::graph_theory::colored_graph
		*get_object_of_type_graph(
				std::string &label);
	orthogonal_geometry_applications::orthogonal_space_with_action
		*get_orthogonal_space(
				std::string &label);
	orbits::orbits_create
		*get_orbits(
			std::string &label);
	canonical_form::variety_object_with_action
		*get_variety(
				std::string &label);
	canonical_form::combinatorial_object_with_properties
		*get_combo_with_group(
				std::string &label);
	isomorph::isomorph_arguments
		*get_isomorph_arguments(
				std::string &label);
	orbits::classify_cubic_surfaces_description
		*get_classify_cubic_surfaces(
				std::string &label);

};


void free_symbol_table_entry_callback(
		other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb, int verbose_level);
geometry::projective_geometry::projective_space *get_projective_space_low_level_function(
		void *ptr);



// #############################################################################
// symbol_definition.cpp
// #############################################################################

//! definition of an orbiter symbol


class symbol_definition {

public:
	user_interface::activities::interface_symbol_table *Sym;


	std::string define_label;

	// TABLES/general/orbiter_objects1.csv



	int f_finite_field;
	algebra::field_theory::finite_field_description
		*Finite_field_description;

	int f_polynomial_ring;
	algebra::ring_theory::polynomial_ring_description
		*Polynomial_ring_description;

	int f_projective_space;
	projective_geometry::projective_space_with_action_description
		*Projective_space_with_action_description;

	int f_orthogonal_space;
	orthogonal_geometry_applications::orthogonal_space_with_action_description
		*Orthogonal_space_with_action_description;

	int f_BLT_set_classifier;
	orthogonal_geometry_applications::blt_set_classify_description
		*Blt_set_classify_description;

	int f_spread_classifier;
	spreads::spread_classify_description
		*Spread_classify_description;

	int f_linear_group;
	group_constructions::linear_group_description
		*Linear_group_description;

	int f_permutation_group;
	group_constructions::permutation_group_description
		*Permutation_group_description;

	int f_group_modification;
	group_constructions::group_modification_description
		*Group_modification_description;

	int f_collection;
	std::string list_of_objects;

	int f_geometric_object;
	std::string geometric_object_projective_space_label;
	geometry::other_geometry::geometric_object_description
		*Geometric_object_description;

	int f_graph;
	apps_graph_theory::create_graph_description
		*Create_graph_description;

	int f_code;
	apps_coding_theory::create_code_description
		*Create_code_description;

	int f_spread;
	spreads::spread_create_description
		*Spread_create_description;

	int f_cubic_surface;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description
		*Surface_Descr;

	int f_quartic_curve;
	applications_in_algebraic_geometry::quartic_curves::quartic_curve_create_description
		*Quartic_curve_descr;

	int f_BLT_set;
	orthogonal_geometry_applications::BLT_set_create_description
		*BLT_Set_create_description;

	int f_translation_plane;
	std::string translation_plane_spread_label;
	std::string translation_plane_group_n_label;
	std::string translation_plane_group_np1_label;


	int f_spread_table;
	geometry::finite_geometries::spread_table_description *Spread_table_description;


	int f_packing_classify;
	std::string packing_classify_label_PA3;
	std::string packing_classify_label_PA5;
	std::string packing_classify_label_spread_table;


	int f_packing_was;
	std::string packing_was_label_packing_classify;
	packings::packing_was_description * packing_was_descr;

	int f_packing_was_choose_fixed_points;
	std::string packing_with_assumed_symmetry_label;
	int packing_with_assumed_symmetry_choose_fixed_points_clique_size;
	std::string packing_was_choose_fixed_points_control_label;


	// TABLES/general/orbiter_objects2.csv

	int f_packing_long_orbits;
	std::string packing_long_orbits_choose_fixed_points_label;
	packings::packing_long_orbits_description
		* Packing_long_orbits_description;

	int f_graph_classification;
	apps_graph_theory::graph_classify_description
		* Graph_classify_description;

	int f_diophant;
	combinatorics::solvers::diophant_description
		*Diophant_description;

	int f_design;
	apps_combinatorics::design_create_description
		*Design_create_description;

	int f_design_table;
	std::string design_table_label_design;
	std::string design_table_label;
	std::string design_table_group;


	int f_large_set_was;
	std::string  large_set_was_label_design_table;
	apps_combinatorics::large_set_was_description
		*large_set_was_descr;

	// set and vector:

	int f_set;
	other::data_structures::set_builder_description
		*Set_builder_description;

	int f_vector;
	other::data_structures::vector_builder_description
		*Vector_builder_description;

	int f_text;
	other::data_structures::text_builder_description
		*Text_builder_description;


	int f_symbolic_object;
	algebra::expression_parser::symbolic_object_builder_description
		*Symbolic_object_builder_description;

	int f_combinatorial_object;
	combinatorics::canonical_form_classification::data_input_stream_description
		*Data_input_stream_description;

	int f_geometry_builder;
	combinatorics::geometry_builder::geometry_builder_description
		*Geometry_builder_description;

	int f_vector_ge;
	data_structures_groups::vector_ge_description
		*Vector_ge_description;

	int f_action_on_forms;
	apps_algebra::action_on_forms_description
		*Action_on_forms_descr;

	int f_orbits;
	orbits::orbits_create_description
		*Orbits_create_description;

	int f_poset_classification_control;
	poset_classification::poset_classification_control
		*Poset_classification_control;

	int f_poset_classification_report_options;
	poset_classification::poset_classification_report_options
		*Poset_classification_report_options;

	int f_draw_options;
	other::graphics::draw_options *Draw_options;

	int f_draw_incidence_structure_options;
	other::graphics::draw_incidence_structure_description
		*Draw_incidence_structure_description;


	int f_arc_generator_control;
	apps_geometry::arc_generator_description
		*Arc_generator_control;

	int f_poset_classification_activity;
	user_interface::activities_layer5::poset_classification_activity_description
		*Poset_classification_activity;

	int f_crc_code;
	combinatorics::coding_theory::crc_code_description *Crc_code_description;

	int f_mapping;
	apps_geometry::mapping_description *Mapping_description;

	int f_variety;
	geometry::algebraic_geometry::variety_description *Variety_description;

	int f_isomorph_arguments;
	isomorph::isomorph_arguments *Isomorph_arguments;

	int f_classify_cubic_surfaces;
	orbits::classify_cubic_surfaces_description *Classify_cubic_surfaces_description;


	symbol_definition();
	~symbol_definition();
	void read_definition(
			user_interface::activities::interface_symbol_table *Sym,
			int argc, std::string *argv, int &i,
			int verbose_level);
	void perform_definition(
			int verbose_level);
	void print();
	void definition_of_finite_field(
			int verbose_level);
	void definition_of_polynomial_ring(
			int verbose_level);
	void definition_of_projective_space(
			int verbose_level);
	void print_definition_of_projective_space(
			int verbose_level);
	void definition_of_orthogonal_space(
			int verbose_level);
	void definition_of_BLT_set_classifier(
			int verbose_level);
	void definition_of_spread_classifier(
			int verbose_level);
	void definition_of_linear_group(
			int verbose_level);
	void definition_of_permutation_group(
			int verbose_level);
	void definition_of_modified_group(
			int verbose_level);
	void definition_of_geometric_object(
			int verbose_level);
	void definition_of_collection(
			std::string &list_of_objects,
			int verbose_level);
	void definition_of_graph(
			int verbose_level);
	void definition_of_code(
			int verbose_level);
	void definition_of_spread(
			int verbose_level);
	void definition_of_cubic_surface(
			int verbose_level);
	void definition_of_quartic_curve(
			int verbose_level);
	void definition_of_BLT_set(
			int verbose_level);
	void definition_of_translation_plane(
			int verbose_level);
	void definition_of_spread_table(
			int verbose_level);
	void definition_of_packing_classify(
			int verbose_level);
	void definition_of_packing_was(
			int verbose_level);
	void definition_of_packing_was_choose_fixed_points(
			int verbose_level);
	void definition_of_packing_long_orbits(
			int verbose_level);
	void definition_of_graph_classification(
			int verbose_level);
	void definition_of_diophant(
			int verbose_level);
	void definition_of_design(
			int verbose_level);
	void definition_of_design_table(
			int verbose_level);
	void definition_of_large_set_was(
			int verbose_level);
	void definition_of_set(
			int verbose_level);
	void definition_of_vector(
			std::string &label,
			other::data_structures::vector_builder_description *Descr,
			int verbose_level);
	void definition_of_text(
			std::string &label,
			other::data_structures::text_builder_description *Descr,
			int verbose_level);
	void definition_of_symbolic_object(
			std::string &label,
			algebra::expression_parser::symbolic_object_builder_description *Descr,
			int verbose_level);
	void definition_of_combinatorial_object(
			int verbose_level);
	void do_geometry_builder(
			int verbose_level);
	void load_finite_field_PG(
			int verbose_level);
	algebra::field_theory::finite_field *get_or_create_finite_field(
			std::string &input_q,
			int verbose_level);
	void definition_of_vector_ge(
			int verbose_level);
	void definition_of_action_on_forms(
			int verbose_level);
	void definition_of_orbits(
			int verbose_level);
	void definition_of_poset_classification_control(
			int verbose_level);
	void definition_of_poset_classification_report_options(
			int verbose_level);
	void definition_of_draw_options(
			int verbose_level);
	void definition_of_draw_incidence_structure_options(
			int verbose_level);
	void definition_of_arc_generator_control(
			int verbose_level);
	void definition_of_poset_classification_activity(
			int verbose_level);
	void definition_of_crc_code(
			int verbose_level);
	void definition_of_mapping(
			int verbose_level);
	void definition_of_variety(
			int verbose_level);
	void definition_of_isomorph_arguments(
			int verbose_level);
	void definition_of_classify_cubic_surfaces(
			int verbose_level);


};



// #############################################################################
// global variable:
// #############################################################################



extern user_interface::core_system::orbiter_top_level_session
	*The_Orbiter_top_level_session;
	// global top level Orbiter session



}}}}



#endif /* SRC_LIB_LAYER5_TOP_LEVEL_USER_INTERFACE_CORE_SYSTEM_CORE_SYSTEM_H_ */
