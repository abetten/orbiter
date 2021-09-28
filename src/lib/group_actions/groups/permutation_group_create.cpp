/*
 * permutation_group_create.cpp
 *
 *  Created on: Sep 26, 2021
 *      Author: betten
 */





#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;



namespace orbiter {
namespace group_actions {

permutation_group_create::permutation_group_create()
{
		Descr = NULL;

		//std::string label;
		//std::string label_tex;

		//initial_strong_gens = NULL;
		A_initial = NULL;

		f_has_strong_generators = FALSE;
		Strong_gens = NULL;
		A2 = NULL;

		f_has_nice_gens = FALSE;
		nice_gens = NULL;
}

permutation_group_create::~permutation_group_create()
{
		Descr = NULL;
}


void permutation_group_create::permutation_group_init(
		permutation_group_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "permutation_group_create::permutation_group_init" << endl;
	}
	permutation_group_create::Descr = description;
	int f_OK = FALSE;

	if (f_v) {
		cout << "permutation_group_create::permutation_group_init initializing group" << endl;
	}


	if (Descr->type == symmetric_group_t) {

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init initializing symmetric_group_t" << endl;
		}

		A_initial = NEW_OBJECT(action);

		A_initial->init_symmetric_group(Descr->degree, verbose_level);

		A_initial->Strong_gens->print_generators_in_latex_individually(cout);
		A_initial->Strong_gens->print_generators_in_source_code();
		A_initial->print_base();
		A_initial->print_info();

		label.assign(A_initial->label);
		label_tex.assign(A_initial->label_tex);

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init initializing symmetric_group_t done" << endl;
		}
	}

	if (Descr->f_subgroup_by_generators) {
		if (f_v) {
			cout << "permutation_group_create::permutation_group_init "
					"before init_subgroup_by_generators" << endl;
		}
		init_subgroup_by_generators(
			description->subgroup_label,
			description->subgroup_order_text,
			description->nb_subgroup_generators,
			description->subgroup_generators_as_string,
			verbose_level);
		if (f_v) {
			cout << "permutation_group_create::permutation_group_init "
					"after init_subgroup_by_generators" << endl;
		}
		f_OK = TRUE;

	}

#if 0
	else {
		cout << "permutation_group_create::permutation_group_init unknown group type" << endl;

	}
#endif

	if (!f_OK) {
		if (f_v) {
			cout << "permutation_group_create::permutation_group_init !f_OK, A2 = A_initial" << endl;
		}
		A2 = A_initial;
		f_has_strong_generators = TRUE;
		Strong_gens = A_initial->Strong_gens;

	}


	if (f_v) {
		cout << "permutation_group_create::permutation_group_init done" << endl;
	}
}

void permutation_group_create::init_subgroup_by_generators(
		std::string &subgroup_label,
		std::string &subgroup_order_text,
		int nb_subgroup_generators,
		std::string *subgroup_generators_as_string,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "permutation_group_create::init_subgroup_by_generators" << endl;
		cout << "label=" << subgroup_label << endl;
	}

	Strong_gens = NEW_OBJECT(strong_generators);
	if (f_v) {
		cout << "permutation_group_create::init_subgroup_by_generators before "
				"Strong_gens->init_subgroup_by_generators" << endl;
	}

	Strong_gens->init_subgroup_by_generators(A_initial,
			nb_subgroup_generators, subgroup_generators_as_string,
			subgroup_order_text,
			nice_gens,
			verbose_level - 1);
	if (f_v) {
		cout << "permutation_group_create::init_subgroup_by_generators after "
				"Strong_gens->init_subgroup_by_generators" << endl;
	}

	f_has_nice_gens = TRUE;

	if (f_v) {
		cout << "permutation_group_create::init_subgroup_by_generators after "
				"Strong_gens->init_subgroup_by_generators" << endl;
	}

	f_has_strong_generators = TRUE;

	//A2 = A_initial;

	stringstream str;
	latex_interface L;
	int max_len = 80;
	int line_skip = 0;


	L.latexable_string(str, subgroup_label.c_str(), max_len, line_skip);



	label.append("_Subgroup_");
	label.append(subgroup_label);
	label.append("_");
	label.append(subgroup_order_text);


	label_tex.append("{\\rm Subgroup ");
	label_tex.append(str.str());
	label_tex.append(" order ");
	label_tex.append(subgroup_order_text);
	label_tex.append("}");

	if (f_v) {
		cout << "permutation_group_create::init_subgroup_by_generators "
				"created group " << label << endl;
	}
}



}}


