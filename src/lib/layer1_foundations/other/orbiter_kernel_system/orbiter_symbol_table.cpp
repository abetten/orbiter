/*
 * orbiter_symbol_table.cpp
 *
 *  Created on: Dec 2, 2020
 *      Author: betten
 */




#include "foundations.h"


using namespace std;

namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace orbiter_kernel_system {




orbiter_symbol_table::orbiter_symbol_table()
{
	Record_birth();
	// Table;

	f_has_free_entry_callback = false;
	free_entry_callback = NULL;
}

orbiter_symbol_table::~orbiter_symbol_table()
{
	Record_death();

}

int orbiter_symbol_table::find_symbol(
		std::string &str)
{
	int i;
	data_structures::string_tools ST;

	for (i = 0; i < Table.size(); i++) {
		if (ST.stringcmp(str, Table[i].label.c_str()) == 0) {
			return i;
		}
	}
	//cout << "orbiter_symbol_table::find_symbol " << str << " not found" << endl;
	return -1;
}

void orbiter_symbol_table::add_symbol_table_entry(
		std::string &str,
		orbiter_symbol_table_entry *Symb,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx;

	if (f_v) {
		cout << "orbiter_symbol_table::add_symbol_table_entry" << endl;
	}
	idx = find_symbol(str);

	if (idx >= 0) {

		if (f_v) {
			cout << "orbiter_symbol_table::add_symbol_table_entry "
				"overriding symbol " << str << " at position " << idx << endl;
		}

		if (f_v) {
			cout << "orbiter_symbol_table::add_symbol_table_entry before free_table_entry" << endl;
		}
		free_table_entry(idx, verbose_level - 1);
		if (f_v) {
			cout << "orbiter_symbol_table::add_symbol_table_entry after free_table_entry" << endl;
		}
		Table[idx] = *Symb;

#if 0
		cout << "orbiter_symbol_table::add_symbol_table_entry Overriding "
				"symbol " << str << " in symbol table at position " << idx << endl;
		Table[idx].freeself();
		Table[idx] = *Symb;
#endif

	}
	else {
		Table.push_back(*Symb);
		//Symb->freeself();
	}
	if (f_v) {
		cout << "orbiter_symbol_table::add_symbol_table_entry done" << endl;
	}
}

void orbiter_symbol_table::free_table_entry(
		int idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table::free_table_entry idx = " << idx << endl;
	}
	if (f_has_free_entry_callback) {
		if (f_v) {
			cout << "orbiter_symbol_table::free_table_entry before (*free_entry_callback)()" << endl;
		}
		(*free_entry_callback)(&Table[idx], verbose_level - 1);
		if (f_v) {
			cout << "orbiter_symbol_table::free_table_entry after (*free_entry_callback)()" << endl;
		}
	}
	if (f_v) {
		cout << "orbiter_symbol_table::free_table_entry done" << endl;
	}
}


void orbiter_symbol_table::print_symbol_table()
{
	int i;

	if (Table.size()) {
		for (i = 0; i < Table.size(); i++) {
			cout << i << " : " << Table[i].label << " : ";
			Table[i].print();
			cout << endl;
		}
	}
	else {
		cout << "orbiter_symbol_table::print_symbol_table symbol table is empty" << endl;
	}
}

void *orbiter_symbol_table::get_object(
		int idx)
{
	if (idx >= Table.size()) {
		cout << "orbiter_symbol_table::get_object out of bounds" << endl;
		exit(1);
	}
	return Table[idx].ptr;
}

symbol_table_object_type orbiter_symbol_table::get_object_type(
		int idx)
{
	if (idx >= Table.size()) {
		cout << "orbiter_symbol_table::get_object_type out of bounds" << endl;
		cout << "orbiter_symbol_table::get_object_type idx = " << idx << endl;
		cout << "orbiter_symbol_table::get_object_type Table.size() = " << Table.size() << endl;
		exit(1);
	}
	return Table[idx].object_type;
}

void orbiter_symbol_table::print_type(
		symbol_table_object_type t)
{
	string s;

	s = stringify_type(t);
	cout << s << endl;
}

std::string orbiter_symbol_table::stringify_type(
		symbol_table_object_type t)
{
	string s;

	// group of ten:

	// t_nothing_object,
	// t_finite_field,
	// t_polynomial_ring,
	// t_any_group,
	// t_linear_group,
	// t_permutation_group,
	// t_modified_group,
	// t_projective_space,
	// t_orthogonal_space,
	// t_BLT_set_classify,


	if (t == t_nothing_object) {
		s = "t_nothing_object";
	}
	else if (t == t_finite_field) {
		s = "t_finite_field";
	}
	else if (t == t_polynomial_ring) {
		s = "t_polynomial_ring";
	}
	else if (t == t_any_group) {
		s = "t_any_group";
	}
	else if (t == t_linear_group) {
		s = "t_linear_group";
	}
	else if (t == t_permutation_group) {
		s = "t_permutation_group";
	}
	else if (t == t_modified_group) {
		s = "t_modified_group";
	}
	else if (t == t_projective_space) {
		s = "t_projective_space";
	}
	else if (t == t_orthogonal_space) {
		s = "t_orthogonal_space";
	}
	else if (t == t_BLT_set_classify) {
		s = "t_BLT_set_classify";
	}

	// group of ten:
	// t_spread_classify,
	// t_cubic_surface,
	// t_quartic_curve,
	// t_BLT_set,
	// t_classification_of_cubic_surfaces_with_double_sixes,
	// t_collection,
	// t_geometric_object,
	// t_graph,
	// t_code,
	// t_spread,

	else if (t == t_spread_classify) {
		s = "t_spread_classify";
	}
	else if (t == t_cubic_surface) {
		s = "t_cubic_surface";
	}
	else if (t == t_quartic_curve) {
		s = "t_quartic_curve";
	}
	else if (t == t_BLT_set) {
		s = "t_BLT_set";
	}
	else if (t == t_classification_of_cubic_surfaces_with_double_sixes) {
		s = "t_classification_of_cubic_surfaces_with_double_sixes";
	}
	else if (t == t_collection) {
		s = "t_collection";
	}
	else if (t == t_geometric_object) {
		s = "t_geometric_object";
	}
	else if (t == t_graph) {
		s = "t_graph";
	}
	else if (t == t_code) {
		s = "t_code";
	}
	else if (t == t_spread) {
		s = "t_spread";
	}


	// group of ten:
	// t_translation_plane,
	// t_spread_table,
	// t_packing_classify,
	// t_packing_was,
	// t_packing_was_choose_fixed_points,
	// t_packing_long_orbits,
	// t_graph_classify,
	// t_diophant,
	// t_design,
	// t_design_table,

	else if (t == t_translation_plane) {
		s = "t_translation_plane";
	}
	else if (t == t_spread_table) {
		s = "t_spread_table";
	}
	else if (t == t_packing_classify) {
		s = "t_packing_classify";
	}
	else if (t == t_packing_was) {
		s ="t_packing_was";
	}
	else if (t == t_packing_was_choose_fixed_points) {
		s = "t_packing_was_choose_fixed_points";
	}
	else if (t == t_packing_long_orbits) {
		s = "t_packing_long_orbits";
	}
	else if (t == t_graph_classify) {
		s = "t_graph_classify";
	}
	else if (t == t_diophant) {
		s = "t_diophant";
	}
	else if (t == t_design) {
		s = "t_design";
	}
	else if (t == t_design_table) {
		s = "t_design_table";
	}



	// group of ten:
	// t_large_set_was,
	// t_set,
	// t_vector,
	// t_symbolic_object,
	// t_combinatorial_object,
	// t_geometry_builder,
	// t_vector_ge,
	// t_action_on_forms,
	// t_orbits,
	// t_poset_classification_control,

	else if (t == t_large_set_was) {
		s = "t_large_set_was";
	}
	else if (t == t_set) {
		s = "t_set";
	}
	else if (t == t_vector) {
		s = "t_vector";
	}
	else if (t == t_text) {
		s = "t_text";
	}
	else if (t == t_symbolic_object) {
		s = "t_symbolic_object";
	}
	else if (t == t_combinatorial_object) {
		s = "t_combinatorial_object";
	}
	else if (t == t_geometry_builder) {
		s = "t_geometry_builder";
	}
	else if (t == t_vector_ge) {
		s = "t_vector_ge";
	}
	else if (t == t_action_on_forms) {
		s = "t_action_on_forms";
	}
	else if (t == t_orbits) {
		s = "t_orbits";
	}


	// group of 10:
	// t_poset_classification_report_options,
	// t_draw_options,
	// t_draw_incidence_structure_options,
	// t_arc_generator_control,
	// t_poset_classification_activity,
	// t_crc_code,
	// t_mapping,
	// t_variety,
	// t_combo_with_group,

	else if (t == t_poset_classification_control) {
		s = "t_poset_classification_control";
	}
	else if (t == t_poset_classification_report_options) {
		s = "t_poset_classification_report_options";
	}
	else if (t == t_draw_options) {
		s = "t_draw_options";
	}
	else if (t == t_draw_incidence_structure_options) {
		s = "t_draw_incidence_structure_options";
	}
	else if (t == t_arc_generator_control) {
		s = "t_arc_generator_control";
	}
	else if (t == t_poset_classification_activity) {
		s = "t_poset_classification_activity";
	}
	else if (t == t_crc_code) {
		s = "t_crc_code";
	}
	else if (t == t_mapping) {
		s = "t_mapping";
	}
	else if (t == t_variety) {
		s = "t_variety";
	}
	else if (t == t_combo_with_group) {
		s = "t_combo_with_group";
	}


	// group of 2:
	// t_isomorph_arguments,
	// t_classify_cubic_surfaces,

	else if (t == t_isomorph_arguments) {
		s = "t_isomorph_arguments";
	}
	else if (t == t_classify_cubic_surfaces) {
		s = "t_classify_cubic_surfaces";
	}

	else {
		cout << "orbiter_symbol_table::stringify_type "
				"type is unknown" << endl;
		exit(1);
	}
	return s;
}




}}}}



