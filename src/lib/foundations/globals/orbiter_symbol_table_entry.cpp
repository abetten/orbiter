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

void orbiter_symbol_table_entry::init_orthogonal_space(std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_orthogonal_space" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_orthogonal_space;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_orthogonal_space done" << endl;
	}
}

void orbiter_symbol_table_entry::init_formula(std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_formula" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_formula;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_formula done" << endl;
	}
}

void orbiter_symbol_table_entry::init_cubic_surface(std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_cubic_surface" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_cubic_surface;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_cubic_surface done" << endl;
	}
}

void orbiter_symbol_table_entry::init_classification_of_cubic_surfaces_with_double_sixes(
		std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_classification_of_cubic_surfaces_with_double_sixes" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_classification_of_cubic_surfaces_with_double_sixes;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_classification_of_cubic_surfaces_with_double_sixes done" << endl;
	}

}

void orbiter_symbol_table_entry::init_collection(std::string &label,
		std::string &list_of_objects, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_collection" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_collection;

	const char *p = list_of_objects.c_str();
	char str[1000];

	std::vector<std::string> *the_list;
	the_list = new std::vector<std::string>;

	while (TRUE) {
		if (!s_scan_token_comma_separated(&p, str)) {
			break;
		}
		string var;

		var.assign(str);
		if (f_v) {
			cout << "adding object " << var << " to the collection" << endl;
		}

		the_list->push_back(var);

	}


	ptr = the_list;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_collection done" << endl;
	}
}

void orbiter_symbol_table_entry::init_combinatorial_object(std::string &label,
		combinatorial_object_create *COC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_combinatorial_object" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_combinatorial_object;
	ptr = COC;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_combinatorial_object done" << endl;
	}
}

void orbiter_symbol_table_entry::init_graph(std::string &label,
		void *Gr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_graph" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_graph;
	ptr = Gr;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_graph done" << endl;
	}
}

void orbiter_symbol_table_entry::print()
{
	if (type == t_intvec) {
		Orbiter->Int_vec.print(cout, vec, vec_len);
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
		else if (object_type == t_orthogonal_space) {
			cout << "orthogonal space" << endl;
		}
		else if (object_type == t_formula) {
			cout << "formula" << endl;
			formula *F;

			F = (formula *) ptr;
			F->print();
		}
		else if (object_type == t_cubic_surface) {
			cout << "cubic surface" << endl;
		}
		else if (object_type == t_classification_of_cubic_surfaces_with_double_sixes) {
			cout << "classification_of_cubic_surfaces_with_double_sixes" << endl;
		}
		else if (object_type == t_collection) {
			cout << "collection" << endl;
			std::vector<std::string> *the_list;
			int i;

			the_list = (std::vector<std::string> *) ptr;
			for (i = 0; i < the_list->size(); i++) {
				cout << i << " : " << (*the_list)[i] << endl;
			}
		}
		else if (object_type == t_combinatorial_object) {
			cout << "combinatorial object" << endl;
		}
		else if (object_type == t_graph) {
			cout << "graph" << endl;
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
