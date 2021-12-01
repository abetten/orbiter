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

void orbiter_symbol_table_entry::init_any_group(std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_any_group" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_any_group;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_any_group done" << endl;
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

void orbiter_symbol_table_entry::init_permutation_group(std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_permutation_group" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_permutation_group;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_permutation_group done" << endl;
	}
}

void orbiter_symbol_table_entry::init_modified_group(std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_modified_group" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_modified_group;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_modified_group done" << endl;
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

void orbiter_symbol_table_entry::init_quartic_curve(std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_quartic_curve" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_quartic_curve;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_quartic_curve done" << endl;
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

	string_tools ST;
	const char *p = list_of_objects.c_str();
	char str[1000];

	std::vector<std::string> *the_list;
	the_list = new std::vector<std::string>;

	while (TRUE) {
		if (!ST.s_scan_token_comma_separated(&p, str)) {
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

void orbiter_symbol_table_entry::init_spread_table(std::string &label,
		void *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_spread_table" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_spread_table;
	ptr = P;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_spread_table done" << endl;
	}
}

void orbiter_symbol_table_entry::init_packing_was(std::string &label,
		void *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_packing_was" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_packing_was;
	ptr = P;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_packing_was done" << endl;
	}
}


void orbiter_symbol_table_entry::init_packing_was_choose_fixed_points(std::string &label,
		void *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_packing_was_choose_fixed_points" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_packing_was_choose_fixed_points;
	ptr = P;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_packing_was_choose_fixed_points done" << endl;
	}
}


void orbiter_symbol_table_entry::init_packing_long_orbits(std::string &label,
		void *PL, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_packing_long_orbits" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_packing_long_orbits;
	ptr = PL;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_packing_long_orbits done" << endl;
	}
}

void orbiter_symbol_table_entry::init_graph_classify(std::string &label,
		void *GC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_graph_classify" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_graph_classify;
	ptr = GC;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_graph_classify done" << endl;
	}
}

void orbiter_symbol_table_entry::init_diophant(std::string &label,
		void *Dio, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_diophant" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_diophant;
	ptr = Dio;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_diophant done" << endl;
	}
}

void orbiter_symbol_table_entry::init_design(std::string &label,
		void *DC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_design" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_design;
	ptr = DC;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_design done" << endl;
	}
}

void orbiter_symbol_table_entry::init_design_table(std::string &label,
		void *DT, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_design_table" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_design_table;
	ptr = DT;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_design_table done" << endl;
	}
}

void orbiter_symbol_table_entry::init_large_set_was(std::string &label,
		void *LSW, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_large_set_was" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_large_set_was;
	ptr = LSW;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_large_set_was done" << endl;
	}
}

void orbiter_symbol_table_entry::init_set(std::string &label,
		void *SB, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_set" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_set;
	ptr = SB;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_set done" << endl;
	}
}

void orbiter_symbol_table_entry::init_vector(std::string &label,
		void *VB, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_set" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_vector;
	ptr = VB;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_set done" << endl;
	}
}

void orbiter_symbol_table_entry::init_combinatorial_objects(std::string &label,
		data_input_stream *IS, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_combinatorial_objects" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_combinatorial_objects;
	ptr = IS;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_combinatorial_objects done" << endl;
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
		else if (object_type == t_permutation_group) {
			cout << "permutation group" << endl;
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
		else if (object_type == t_quartic_curve) {
			cout << "quartic curve" << endl;
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
		else if (object_type == t_spread_table) {
			cout << "spread table" << endl;
		}
		else if (object_type == t_packing_was) {
			cout << "packing with symmetry assumption" << endl;
		}
		else if (object_type == t_packing_was_choose_fixed_points) {
			cout << "packing with symmetry assumption, choice of fixed points" << endl;
		}
		else if (object_type == t_packing_long_orbits) {
			cout << "packing with symmetry assumption, choosing long orbits" << endl;
		}
		else if (object_type == t_graph_classify) {
			cout << "graph_classification" << endl;
		}
		else if (object_type == t_diophant) {
			cout << "diophant" << endl;
		}
		else if (object_type == t_design) {
			cout << "design" << endl;
		}
		else if (object_type == t_design_table) {
			cout << "design_table" << endl;
		}
		else if (object_type == t_large_set_was) {
			cout << "large_set_was" << endl;
		}
		else if (object_type == t_set) {
			cout << "set" << endl;
		}
		else if (object_type == t_vector) {
			cout << "vector" << endl;
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
