/*
 * control_everything.h
 *
 *  Created on: Mar 26, 2026
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER6_USER_INTERFACE_CONTROL_EVERYTHING_CONTROL_EVERYTHING_H_
#define SRC_LIB_LAYER6_USER_INTERFACE_CONTROL_EVERYTHING_CONTROL_EVERYTHING_H_


namespace orbiter {
namespace layer6_user_interface {
namespace control_everything {



// #############################################################################
// activity_description.cpp
// #############################################################################

//! description of an activity for an orbiter symbol


class activity_description {

public:

	layer6_user_interface::activities::interface_symbol_table *Sym;


	int f_finite_field_activity;
	layer6_user_interface::activities_layer1::finite_field_activity_description
		*Finite_field_activity_description;

	int f_polynomial_ring_activity;
	layer6_user_interface::activities_layer1::polynomial_ring_activity_description
		*Polynomial_ring_activity_description;

	int f_projective_space_activity;
	layer6_user_interface::activities_layer5::projective_space_activity_description
		*Projective_space_activity_description;

	int f_orthogonal_space_activity;
	layer6_user_interface::activities_layer5::orthogonal_space_activity_description
		*Orthogonal_space_activity_description;

	int f_group_theoretic_activity;
	layer6_user_interface::activities_layer5::group_theoretic_activity_description
		*Group_theoretic_activity_description;

	int f_coding_theoretic_activity;
	layer6_user_interface::activities_layer5::coding_theoretic_activity_description
		*Coding_theoretic_activity_description;

	int f_cubic_surface_activity;
	layer6_user_interface::activities_layer5::cubic_surface_activity_description
		*Cubic_surface_activity_description;

	int f_quartic_curve_activity;
	layer6_user_interface::activities_layer5::quartic_curve_activity_description
		*Quartic_curve_activity_description;

	int f_blt_set_activity;
	layer6_user_interface::activities_layer5::blt_set_activity_description
		*Blt_set_activity_description;

	int f_combinatorial_object_activity;
	layer6_user_interface::activities_layer5::combinatorial_object_activity_description
		*Combinatorial_object_activity_description;

	int f_graph_theoretic_activity;
	layer6_user_interface::activities_layer5::graph_theoretic_activity_description
		*Graph_theoretic_activity_description;

	int f_classification_of_cubic_surfaces_with_double_sixes_activity;
	layer6_user_interface::activities_layer5::classification_of_cubic_surfaces_with_double_sixes_activity_description
		*Classification_of_cubic_surfaces_with_double_sixes_activity_description;

	int f_spread_table_activity;
	layer6_user_interface::activities_layer5::spread_table_activity_description
		*Spread_table_activity_description;

	int f_packing_classify_activity_description;
	layer6_user_interface::activities_layer5::packing_classify_activity_description *Packing_classify_activity_description;

	int f_packing_with_symmetry_assumption_activity;
	layer6_user_interface::activities_layer5::packing_was_activity_description
		*Packing_was_activity_description;

	int f_packing_fixed_points_activity;
	layer6_user_interface::activities_layer5::packing_was_fixpoints_activity_description
		*Packing_was_fixpoints_activity_description;

	int f_graph_classification_activity;
	layer6_user_interface::activities_layer5::graph_classification_activity_description
		*Graph_classification_activity_description;

	int f_diophant_activity;
	layer6_user_interface::activities_layer1::diophant_activity_description
		*Diophant_activity_description;

	int f_design_activity;
	layer6_user_interface::activities_layer5::design_activity_description
		*Design_activity_description;

	// ToDo: not documented
	int f_large_set_activity;
	layer6_user_interface::activities_layer5::large_set_activity_description
		*Large_set_activity_description;

	int f_large_set_was_activity;
	layer6_user_interface::activities_layer5::large_set_was_activity_description
		*Large_set_was_activity_description;

	int f_symbolic_object_activity;
	layer6_user_interface::activities_layer1::symbolic_object_activity_description
		*Symbolic_object_activity_description;

	int f_BLT_set_classify_activity;
	layer6_user_interface::activities_layer5::blt_set_classify_activity_description
		*Blt_set_classify_activity_description;

	int f_spread_classify_activity;
	layer6_user_interface::activities_layer5::spread_classify_activity_description
		*Spread_classify_activity_description;

	int f_spread_activity;
	layer6_user_interface::activities_layer5::spread_activity_description
		*Spread_activity_description;

	int f_translation_plane_activity;
	layer6_user_interface::activities_layer5::translation_plane_activity_description
		*Translation_plane_activity_description;

	int f_action_on_forms_activity;
	layer6_user_interface::activities_layer5::action_on_forms_activity_description
		*Action_on_forms_activity_description;

	int f_orbits_activity;
	layer6_user_interface::activities_layer5::orbits_activity_description
		*Orbits_activity_description;

	int f_variety_activity;
	layer6_user_interface::activities_layer5::variety_activity_description
		*Variety_activity_description;

	int f_vector_ge_activity;
	layer6_user_interface::activities_layer5::vector_ge_activity_description
		*Vector_ge_activity_description;

	int f_combo_activity;
	layer6_user_interface::activities_layer5::combo_activity_description *Combo_activity_description;


	// ToDo: not documented
	int f_plesken_ring_activity;
	layer6_user_interface::activities_layer5::plesken_ring_activity_description *Plesken_ring_activity_description;


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



//! a single command in the Orbiter dash code language, used by orbiter_top_level_session::parse


class orbiter_command {

public:

	layer5_applications::user_interface::core_system::orbiter_top_level_session *Orbiter_top_level_session;


	int f_algebra;
	layer6_user_interface::activities::interface_algebra *Algebra;

	int f_coding_theory;
	layer6_user_interface::activities::interface_coding_theory *Coding_theory;

	int f_combinatorics;
	activities::interface_combinatorics *Combinatorics;

	int f_cryptography;
	layer6_user_interface::activities::interface_cryptography *Cryptography;

	int f_povray;
	layer6_user_interface::activities::interface_povray *Povray;

	int f_projective;
	layer6_user_interface::activities::interface_projective *Projective;

	int f_symbol_table;
	layer6_user_interface::activities::interface_symbol_table *Symbol_table;

	int f_toolkit;
	layer6_user_interface::activities::interface_toolkit *Toolkit;

	orbiter_command();
	~orbiter_command();
	void parse(
			layer5_applications::user_interface::core_system::orbiter_top_level_session *Orbiter_top_level_session,
			int argc, std::string *Argv, int &i, int verbose_level);
	void execute(
			int verbose_level);
	void print();

};


// #############################################################################
// control_everything.cpp
// #############################################################################


void orbiter_execute_command_line(
		layer5_applications::user_interface::core_system::orbiter_top_level_session
			*The_Orbiter_top_level_session,
		int argc, const char **argv, int verbose_level);
// called from do_orbiter_session in the front-end orbiter.cpp

int orbiter_startup_and_read_arguments(
		layer5_applications::user_interface::core_system::orbiter_top_level_session
			*The_Orbiter_top_level_session,
		int argc,
		std::string *argv, int i0, int verbose_level);
// called from execute_command_line

void orbiter_handle_everything(
		layer5_applications::user_interface::core_system::orbiter_top_level_session
			*The_Orbiter_top_level_session,
		int argc, std::string *Argv, int i, int verbose_level);
// called from execute_command_line

void orbiter_parse_and_execute(
		layer5_applications::user_interface::core_system::orbiter_top_level_session
			*The_Orbiter_top_level_session,
		int argc, std::string *Argv, int i, int verbose_level);
// called from handle_everything

void orbiter_parse(
		layer5_applications::user_interface::core_system::orbiter_top_level_session
			*The_Orbiter_top_level_session,
		int argc, std::string *Argv,
		int &i, std::vector<void *> &program, int verbose_level);
// called from parse_and_execute


}}}




#endif /* SRC_LIB_LAYER6_USER_INTERFACE_CONTROL_EVERYTHING_CONTROL_EVERYTHING_H_ */
