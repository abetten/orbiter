/*
 * modified_group_create.cpp
 *
 *  Created on: Dec 1, 2021
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;
using namespace orbiter::layer1_foundations;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


modified_group_create::modified_group_create()
{
		Descr = NULL;

		//std::string label;
		//std::string label_tex;

		//initial_strong_gens = NULL;

		A_base = NULL;
		A_previous = NULL;
		A_modified = NULL;

		f_has_strong_generators = FALSE;
		Strong_gens = NULL;
}


modified_group_create::~modified_group_create()
{
		Descr = NULL;
}


void modified_group_create::modified_group_init(
		group_modification_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::modified_group_init" << endl;
	}
	modified_group_create::Descr = description;

	if (f_v) {
		cout << "modified_group_create::modified_group_init initializing group" << endl;
	}


	if (Descr->f_restricted_action) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init initializing restricted action" << endl;
		}

		if (Descr->from.size() != 1) {
			cout << "modified_group_create::modified_group_init need exactly one argument of type -from" << endl;
			exit(1);
		}

		int idx;


		idx = orbiter_kernel_system::Orbiter->find_symbol(Descr->from[0]);


		symbol_table_object_type t = orbiter_kernel_system::Orbiter->get_object_type(idx);



		if (t != t_any_group) {
			cout << "-from must give object of type t_any_group" << endl;
			cout << "type given: ";
			orbiter_kernel_system::Orbiter->print_type(t);
			exit(1);
		}

		any_group *AG;

		AG = (any_group *) orbiter_kernel_system::Orbiter->get_object(idx);

		A_base = AG->A_base;
		A_previous = AG->A;

		long int *points;
		int nb_points;


		orbiter_kernel_system::Orbiter->get_lint_vector_from_label(Descr->restricted_action_set_text,
				points, nb_points, verbose_level);

		//Orbiter->Lint_vec.scan(Descr->restricted_action_set_text, points, nb_points);

		if (f_v) {
			cout << "modified_group_create::modified_group_init before A_previous->restricted_action" << endl;
		}
		A_modified = A_previous->restricted_action(points, nb_points,
				verbose_level);
		if (f_v) {
			cout << "modified_group_create::modified_group_init after A_previous->restricted_action" << endl;
		}
		A_modified->f_is_linear = A_previous->f_is_linear;

		f_has_strong_generators = TRUE;
		if (f_v) {
			cout << "modified_group_create::modified_group_init before Strong_gens = AG->Subgroup_gens" << endl;
		}
		Strong_gens = AG->Subgroup_gens;

#if 0
		A_modified->Strong_gens->print_generators_in_latex_individually(cout);
		A_modified->Strong_gens->print_generators_in_source_code();
		A_modified->print_base();
#endif
		A_modified->print_info();

		if (f_v) {
			cout << "modified_group_create::modified_group_init before assigning label" << endl;
		}
		label.assign(A_previous->label);
		label_tex.assign(A_previous->label_tex);

		if (f_v) {
			cout << "modified_group_create::modified_group_init initializing restricted action done" << endl;
		}
	}

	else {
		cout << "modified_group_create::modified_group_init unknown operation" << endl;

	}




	if (f_v) {
		cout << "modified_group_create::modified_group_init done" << endl;
	}
}



}}}




