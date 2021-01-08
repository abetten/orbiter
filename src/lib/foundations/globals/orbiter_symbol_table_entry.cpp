/*
 * orbiter_symbol_table_entry.cpp
 *
 *  Created on: Dec 2, 2020
 *      Author: betten
 */




#include "foundations.h"


using namespace std;

namespace orbiter {
namespace foundations {



orbiter_symbol_table_entry::orbiter_symbol_table_entry()
{

	//std::string label;
	type = t_nothing;
	object_type = t_nothing_object;
	vec = NULL;
	vec_len = 0;
	//std::string str;
	ptr = NULL;
}

orbiter_symbol_table_entry::~orbiter_symbol_table_entry()
{
	freeself();
}

void orbiter_symbol_table_entry::freeself()
{
	if (type == t_intvec && vec) {
		FREE_int(vec);
		vec = 0;
	}
	type = t_nothing;
	object_type = t_nothing_object;
}

void orbiter_symbol_table_entry::init(std::string &str_label)
{
	label.assign(str_label);
}

void orbiter_symbol_table_entry::init_finite_field(std::string &label,
		finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_finite_field" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_finite_field;
	ptr = F;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_finite_field done" << endl;
	}
}

void orbiter_symbol_table_entry::init_linear_group(std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_linear_group" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_linear_group;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_linear_group done" << endl;
	}
}

void orbiter_symbol_table_entry::init_projective_space(std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_projective_space" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_projective_space;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_projective_space done" << endl;
	}
}

void orbiter_symbol_table_entry::print()
{
	if (type == t_intvec) {
		int_vec_print(cout, vec, vec_len);
		cout << endl;
	}
	else if (type == t_object) {
		if (object_type == t_finite_field) {
			finite_field *F;

			F = (finite_field *) ptr;
			F->print();
		}
		else if (object_type == t_linear_group) {
			cout << "linear group" << endl;
		}
		else if (object_type == t_projective_space) {
			cout << "projective space" << endl;
		}
#if 0
		else if (object_type == t_action) {
			action *A;

			A = (action *) ptr;
			A->print_info();
		}
		else if (object_type == t_poset) {
			poset *P;

			P = (poset *) ptr;
			P->print();
		}
		else if (object_type == t_poset_classification) {
			poset_classification *PC;

			PC = (poset_classification *) ptr;
			PC->print();
		}
#endif
	}
}


}}
