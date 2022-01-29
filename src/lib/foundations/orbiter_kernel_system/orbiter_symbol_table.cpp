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
namespace orbiter_kernel_system {




orbiter_symbol_table::orbiter_symbol_table()
{
	// Table;
}

orbiter_symbol_table::~orbiter_symbol_table()
{

}

int orbiter_symbol_table::find_symbol(std::string &str)
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

void orbiter_symbol_table::add_symbol_table_entry(std::string &str,
		orbiter_symbol_table_entry *Symb, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx;

	if (f_v) {
		cout << "orbiter_symbol_table::add_symbol_table_entry" << endl;
	}
	idx = find_symbol(str);

	if (idx >= 0) {
		cout << "orbiter_symbol_table::add_symbol_table_entry Overriding "
				"symbol " << str << " in symbol table at position " << idx << endl;
		Table[idx].freeself();
		Table[idx] = *Symb;
	}
	else {
		Table.push_back(*Symb);
		Symb->freeself();
	}
	if (f_v) {
		cout << "orbiter_symbol_table::add_symbol_table_entry done" << endl;
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

void *orbiter_symbol_table::get_object(int idx)
{
	if (idx >= Table.size()) {
		cout << "orbiter_symbol_table::get_object out of bounds" << endl;
		exit(1);
	}
	return Table[idx].ptr;
}

symbol_table_object_type orbiter_symbol_table::get_object_type(int idx)
{
	if (idx >= Table.size()) {
		cout << "orbiter_symbol_table::get_object_type out of bounds" << endl;
		exit(1);
	}
	return Table[idx].object_type;
}

void orbiter_symbol_table::print_type(symbol_table_object_type t)
{
	if (t == t_nothing_object) {
		cout << "t_nothing_object" << endl;
	}
	else if (t == t_finite_field) {
		cout << "t_finite_field" << endl;
	}
	else if (t == t_any_group) {
		cout << "t_any_group" << endl;
	}
	else if (t == t_linear_group) {
		cout << "t_linear_group" << endl;
	}
	else if (t == t_permutation_group) {
		cout << "t_permutation_group" << endl;
	}
	else if (t == t_modified_group) {
		cout << "t_modified_group" << endl;
	}
	else if (t == t_projective_space) {
		cout << "t_projective_space" << endl;
	}
	else if (t == t_orthogonal_space) {
		cout << "t_orthogonal_space" << endl;
	}
	else if (t == t_formula) {
		cout << "t_formula" << endl;
	}
	else if (t == t_cubic_surface) {
		cout << "t_cubic_surface" << endl;
	}
	else if (t == t_quartic_curve) {
		cout << "t_quartic_curve" << endl;
	}
	else if (t == t_classification_of_cubic_surfaces_with_double_sixes) {
		cout << "t_classification_of_cubic_surfaces_with_double_sixes" << endl;
	}
	else if (t == t_collection) {
		cout << "t_collection" << endl;
	}
	else if (t == t_geometric_object) {
		cout << "t_geometric_object" << endl;
	}
	else if (t == t_graph) {
		cout << "t_graph" << endl;
	}
	else if (t == t_spread_table) {
		cout << "t_spread_table" << endl;
	}
	else if (t == t_packing_was) {
		cout << "t_packing_was" << endl;
	}
	else if (t == t_packing_was_choose_fixed_points) {
		cout << "t_packing_was_choose_fixed_points" << endl;
	}
	else if (t == t_packing_long_orbits) {
		cout << "t_packing_long_orbits" << endl;
	}
	else if (t == t_graph_classify) {
		cout << "t_graph_classify" << endl;
	}
	else if (t == t_diophant) {
		cout << "t_diophant" << endl;
	}
	else if (t == t_design) {
		cout << "t_design" << endl;
	}
	else if (t == t_design_table) {
		cout << "t_design_table" << endl;
	}
	else if (t == t_large_set_was) {
		cout << "t_large_set_was" << endl;
	}
	else if (t == t_set) {
		cout << "t_set" << endl;
	}
	else if (t == t_vector) {
		cout << "t_vector" << endl;
	}
	else if (t == t_combinatorial_objects) {
		cout << "t_combinatorial_objects" << endl;
	}
	else if (t == t_action) {
		cout << "t_action" << endl;
	}
	else if (t == t_poset) {
		cout << "t_poset" << endl;
	}
	else if (t == t_poset_classification) {
		cout << "t_poset_classification" << endl;
	}
	else {
		cout << "type is unknown" << endl;
		exit(1);
	}

}



}}}


