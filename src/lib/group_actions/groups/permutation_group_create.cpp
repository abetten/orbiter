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
namespace layer3_group_actions {
namespace groups {

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

		A_initial = NEW_OBJECT(actions::action);
		int f_no_base = FALSE;

		A_initial->init_symmetric_group(Descr->degree, f_no_base, verbose_level);

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init generators:" << endl;
			A_initial->Strong_gens->print_generators_in_latex_individually(cout);
			A_initial->Strong_gens->print_generators_in_source_code();
			A_initial->print_base();
			A_initial->print_info();
		}

		label.assign(A_initial->label);
		label_tex.assign(A_initial->label_tex);

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init initializing symmetric_group_t done" << endl;
		}
	}

	else if (Descr->type == cyclic_group_t) {

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init initializing cyclic_group_t" << endl;
		}

		A_initial = NEW_OBJECT(actions::action);
		int f_no_base = FALSE;

		A_initial->init_cyclic_group(Descr->degree, f_no_base, verbose_level);

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init generators:" << endl;
			A_initial->Strong_gens->print_generators_in_latex_individually(cout);
			A_initial->Strong_gens->print_generators_in_source_code();
			A_initial->print_base();
			A_initial->print_info();
		}

		label.assign(A_initial->label);
		label_tex.assign(A_initial->label_tex);

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init initializing cyclic_group_t done" << endl;
		}
	}

	else if (Descr->type == bsgs_t) {

		A_initial = NEW_OBJECT(actions::action);

		ring_theory::longinteger_object target_go;
		long int *given_base;
		int given_base_length;

#if 0
		int *gens;
		int *gens_i;
		int sz;
		int h;

		gens = NEW_int(Descr->bsgs_nb_generators * Descr->degree);
		for (h = 0; h < Descr->bsgs_nb_generators; h++) {

			Orbiter->Int_vec.scan(Descr->bsgs_generators[h], gens_i, sz);
			if (sz != Descr->degree) {
				cout << "permutation_group_create::permutation_group_init generator "
						<< h << " does not have the right length" << endl;
				exit(1);
			}
			Orbiter->Int_vec.copy(gens_i, gens + h * Descr->degree, Descr->degree);

			FREE_int(gens_i);

		}
#else
		int *gens;
		int sz;

		orbiter_kernel_system::Orbiter->get_vector_from_label(Descr->bsgs_generators, gens, sz, verbose_level);
#endif

		int f_no_base = FALSE;

		target_go.create_from_base_10_string(Descr->bsgs_order_text);

		Lint_vec_scan(Descr->bsgs_base, given_base, given_base_length);


		A_initial->init_permutation_group_from_generators(Descr->degree,
			TRUE /* f_target_go */, target_go,
			Descr->bsgs_nb_generators, gens,
			given_base_length, given_base,
			f_no_base,
			verbose_level);

		A_initial->Strong_gens->print_generators_in_latex_individually(cout);
		A_initial->Strong_gens->print_generators_in_source_code();
		A_initial->print_base();
		A_initial->print_info();

		label.assign(Descr->bsgs_label);
		label_tex.assign(Descr->bsgs_label_tex);

		FREE_int(gens);

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
			description->subgroup_generators_label,
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
		std::string &subgroup_generators_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "permutation_group_create::init_subgroup_by_generators" << endl;
		cout << "label=" << subgroup_label << endl;
	}

	Strong_gens = NEW_OBJECT(strong_generators);

	int *gens;
	int sz;

	orbiter_kernel_system::Orbiter->get_vector_from_label(subgroup_generators_label, gens, sz, verbose_level);

	if (sz != nb_subgroup_generators * A_initial->degree) {
		cout << "permutation_group_create::init_subgroup_by_generators "
				"sz != nb_subgroup_generators * A_initial->degree" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "permutation_group_create::init_subgroup_by_generators before "
				"Strong_gens->init_subgroup_by_generators" << endl;
	}
	Strong_gens->init_subgroup_by_generators(A_initial,
			nb_subgroup_generators, gens,
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

	A2 = A_initial;

	stringstream str;
	orbiter_kernel_system::latex_interface L;
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



}}}


