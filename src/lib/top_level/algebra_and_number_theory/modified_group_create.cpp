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
			cout << "modified_group_create::modified_group_init before create_restricted_action" << endl;
		}

		create_restricted_action(description, verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init after create_restricted_action" << endl;
		}
	}

	else if (Descr->f_on_k_subspaces) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init before create_action_on_k_subspaces" << endl;
		}

		create_action_on_k_subspaces(description, verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init after create_action_on_k_subspaces" << endl;
		}
	}

	else {
		cout << "modified_group_create::modified_group_init unknown operation" << endl;

	}




	if (f_v) {
		cout << "modified_group_create::modified_group_init done" << endl;
	}
}


void modified_group_create::create_restricted_action(
		group_modification_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_restricted_action" << endl;
	}
	if (Descr->from.size() != 1) {
		cout << "modified_group_create::create_restricted_action need exactly one argument of type -from" << endl;
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
		cout << "modified_group_create::create_restricted_action before A_previous->restricted_action" << endl;
	}
	A_modified = A_previous->restricted_action(points, nb_points,
			verbose_level);
	if (f_v) {
		cout << "modified_group_create::create_restricted_action after A_previous->restricted_action" << endl;
	}
	A_modified->f_is_linear = A_previous->f_is_linear;

	f_has_strong_generators = TRUE;
	if (f_v) {
		cout << "modified_group_create::create_restricted_action before Strong_gens = AG->Subgroup_gens" << endl;
	}
	Strong_gens = AG->Subgroup_gens;

#if 0
	A_modified->Strong_gens->print_generators_in_latex_individually(cout);
	A_modified->Strong_gens->print_generators_in_source_code();
	A_modified->print_base();
#endif
	A_modified->print_info();

	if (f_v) {
		cout << "modified_group_create::create_restricted_action before assigning label" << endl;
	}
	label.assign(A_previous->label);
	label_tex.assign(A_previous->label_tex);

	if (f_v) {
		cout << "modified_group_create::create_restricted_action done" << endl;
	}
}


void modified_group_create::create_action_on_k_subspaces(
		group_modification_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces" << endl;
	}
	if (Descr->from.size() != 1) {
		cout << "modified_group_create::create_action_on_k_subspaces need exactly one argument of type -from" << endl;
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



#if 0
	long int *points;
	int nb_points;

	orbiter_kernel_system::Orbiter->get_lint_vector_from_label(Descr->restricted_action_set_text,
			points, nb_points, verbose_level);

	//Orbiter->Lint_vec.scan(Descr->restricted_action_set_text, points, nb_points);

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces before A_previous->restricted_action" << endl;
	}
	A_modified = A_previous->restricted_action(points, nb_points,
			verbose_level);
	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces after A_previous->restricted_action" << endl;
	}
	A_modified->f_is_linear = A_previous->f_is_linear;

	f_has_strong_generators = TRUE;
	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces before Strong_gens = AG->Subgroup_gens" << endl;
	}
	Strong_gens = AG->Subgroup_gens;

#if 0
	A_modified->Strong_gens->print_generators_in_latex_individually(cout);
	A_modified->Strong_gens->print_generators_in_source_code();
	A_modified->print_base();
#endif
	A_modified->print_info();

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces before assigning label" << endl;
	}
	label.assign(A_previous->label);
	label_tex.assign(A_previous->label_tex);
#else

	if (!A_previous->f_is_linear) {
		cout << "modified_group_create::create_action_on_k_subspaces previous action is not linear" << endl;
		exit(1);
	}


	groups::matrix_group *M;
	field_theory::finite_field *Fq;
	int n;

	M = A_previous->get_matrix_group();

	n = M->n;
	Fq = M->GFq;

	induced_actions::action_on_grassmannian *AonG;
	geometry::grassmann *Grass;

	AonG = NEW_OBJECT(induced_actions::action_on_grassmannian);

	Grass = NEW_OBJECT(geometry::grassmann);


	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces before Grass->init" << endl;
	}

	Grass->init(n,
			description->on_k_subspaces_k,
			Fq, 0 /* verbose_level */);

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces after Grass->init" << endl;
	}


	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces before AonG->init" << endl;
	}

	AonG->init(*A_previous, Grass, verbose_level - 2);

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces after AonG->init" << endl;
	}


	A_modified = NEW_OBJECT(actions::action);

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces before A_modified->induced_action_on_grassmannian" << endl;
	}

	A_modified->induced_action_on_grassmannian(A_previous, AonG,
		FALSE /* f_induce_action */, NULL /*sims *old_G */,
		verbose_level - 2);

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces after A_modified->induced_action_on_grassmannian" << endl;
	}


	A_modified->f_is_linear = TRUE;

	f_has_strong_generators = TRUE;

	A_modified->f_is_linear = A_previous->f_is_linear;
	A_modified->dimension = A_previous->dimension;

	f_has_strong_generators = TRUE;
	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces before Strong_gens = AG->Subgroup_gens" << endl;
	}
	Strong_gens = AG->Subgroup_gens;

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces action A_modified created: ";
		A_modified->print_info();
		}

	//A2 = A3;
	//f_OK = TRUE;

	char str1[1000];
	char str2[1000];

	sprintf(str1, "_OnGr_%d", description->on_k_subspaces_k);
	sprintf(str2, " {\\rm Gr}_{%d,%d}(%d)",
			n, description->on_k_subspaces_k, Fq->q);
	label.append(str1);
	label_tex.append(str2);


#endif

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces done" << endl;
	}
}



}}}




