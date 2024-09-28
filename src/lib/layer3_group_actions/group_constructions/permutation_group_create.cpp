/*
 * permutation_group_create.cpp
 *
 *  Created on: Sep 26, 2021
 *      Author: betten
 */





#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;



namespace orbiter {
namespace layer3_group_actions {
namespace group_constructions {

permutation_group_create::permutation_group_create()
{
		Descr = NULL;

		//std::string label;
		//std::string label_tex;

		//initial_strong_gens = NULL;
		A_initial = NULL;

		f_has_strong_generators = false;
		Strong_gens = NULL;
		A2 = NULL;

		f_has_nice_gens = false;
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
	int f_OK = false;

	if (f_v) {
		cout << "permutation_group_create::permutation_group_init "
				"initializing group" << endl;
	}


	if (Descr->type == symmetric_group_t) {

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init "
					"initializing symmetric_group_t" << endl;
		}

		A_initial = NEW_OBJECT(actions::action);

		A_initial->Known_groups->init_symmetric_group(
				Descr->degree, verbose_level);

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init generators:" << endl;
			A_initial->Strong_gens->print_generators_in_latex_individually(cout, verbose_level - 1);
			A_initial->Strong_gens->print_generators_in_source_code(verbose_level - 1);
			A_initial->print_base();
			A_initial->print_info();
		}

		label.assign(A_initial->label);
		label_tex.assign(A_initial->label_tex);

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init "
					"initializing symmetric_group_t done" << endl;
		}
	}

	else if (Descr->type == cyclic_group_t) {

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init "
					"initializing cyclic_group_t" << endl;
		}

		A_initial = NEW_OBJECT(actions::action);

		A_initial->Known_groups->init_cyclic_group(
				Descr->degree, verbose_level);

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init generators:" << endl;
			A_initial->Strong_gens->print_generators_in_latex_individually(cout, verbose_level - 1);
			A_initial->Strong_gens->print_generators_in_source_code(verbose_level - 1);
			A_initial->print_base();
			A_initial->print_info();
		}

		label.assign(A_initial->label);
		label_tex.assign(A_initial->label_tex);

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init "
					"initializing cyclic_group_t done" << endl;
		}
	}

	else if (Descr->type == elementary_abelian_group_t) {

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init "
					"initializing elementary_abelian_group_t" << endl;
		}

		A_initial = NEW_OBJECT(actions::action);

		A_initial->Known_groups->init_elementary_abelian_group(
				Descr->degree, verbose_level);

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init generators:" << endl;
			A_initial->Strong_gens->print_generators_in_latex_individually(cout, verbose_level - 1);
			A_initial->Strong_gens->print_generators_in_source_code(verbose_level - 1);
			A_initial->print_base();
			A_initial->print_info();
		}

		label.assign(A_initial->label);
		label_tex.assign(A_initial->label_tex);

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init "
					"initializing elementary_abelian_group_t done" << endl;
		}
	}


	else if (Descr->type == identity_group_t) {

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init "
					"initializing identity_group_t" << endl;
		}

		A_initial = NEW_OBJECT(actions::action);

		A_initial->Known_groups->init_identity_group(
				Descr->degree, verbose_level);

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init generators:" << endl;
			A_initial->Strong_gens->print_generators_in_latex_individually(cout, verbose_level - 1);
			A_initial->Strong_gens->print_generators_in_source_code(verbose_level - 1);
			A_initial->print_base();
			A_initial->print_info();
		}

		label.assign(A_initial->label);
		label_tex.assign(A_initial->label_tex);

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init "
					"initializing identity_group_t done" << endl;
		}
	}

	else if (Descr->type == bsgs_t) {

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init "
					"bsgs_t" << endl;
		}
		A_initial = NEW_OBJECT(actions::action);

		ring_theory::longinteger_object target_go;
		long int *given_base;
		int given_base_length;

		int *gens;
		int sz;

		Get_int_vector_from_label(Descr->bsgs_generators, gens, sz, verbose_level);

		//int f_no_base = false;

		target_go.create_from_base_10_string(Descr->bsgs_order_text);

		if (f_v) {
			cout << "permutation_group_create::permutation_group_init "
					"Descr->bsgs_base=" << Descr->bsgs_base << endl;
		}

		Lint_vec_scan(Descr->bsgs_base, given_base, given_base_length);


		if (f_v) {
			cout << "permutation_group_create::permutation_group_init "
					"before init_permutation_group_from_generators" << endl;
		}
		A_initial->Known_groups->init_permutation_group_from_generators(
				Descr->degree,
			true /* f_target_go */, target_go,
			Descr->bsgs_nb_generators, gens,
			given_base_length, given_base,
			true /* f_given_base */,
			verbose_level);
		if (f_v) {
			cout << "permutation_group_create::permutation_group_init "
					"after init_permutation_group_from_generators" << endl;
		}

		A_initial->Strong_gens->print_generators_in_latex_individually(cout, verbose_level - 1);
		A_initial->Strong_gens->print_generators_in_source_code(verbose_level - 1);
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
		f_OK = true;

	}

#if 0
	else {
		cout << "permutation_group_create::permutation_group_init unknown group type" << endl;

	}
#endif

	if (!f_OK) {
		if (f_v) {
			cout << "permutation_group_create::permutation_group_init "
					"!f_OK, A2 = A_initial" << endl;
		}
		A2 = A_initial;
		f_has_strong_generators = true;
		Strong_gens = A_initial->Strong_gens;

	}
	if (f_v) {
		cout << "permutation_group_create::permutation_group_init label = " << label << endl;
		cout << "permutation_group_create::permutation_group_init label_tex = " << label_tex << endl;

		ring_theory::longinteger_object go;

		Strong_gens->group_order(go);
		cout << "permutation_group_create::permutation_group_init go = " << go << endl;
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
// group will be in Strong_gens
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "permutation_group_create::init_subgroup_by_generators" << endl;
		cout << "label=" << subgroup_label << endl;
	}

	Strong_gens = NEW_OBJECT(groups::strong_generators);

	int *gens;
	int sz;

	Get_int_vector_from_label(subgroup_generators_label, gens, sz, verbose_level);

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

	f_has_nice_gens = true;

	if (f_v) {
		cout << "permutation_group_create::init_subgroup_by_generators after "
				"Strong_gens->init_subgroup_by_generators" << endl;
	}

	f_has_strong_generators = true;

	A2 = A_initial;

	stringstream str;
	l1_interfaces::latex_interface L;
	int max_len = 80;
	int line_skip = 0;


	L.latexable_string(str, subgroup_label.c_str(), max_len, line_skip);



	label += "_Subgroup_" + subgroup_label + "_" + subgroup_order_text;


	label_tex += "{\\rm Subgroup " + str.str() + " order " + subgroup_order_text + "}";

	if (f_v) {
		cout << "permutation_group_create::init_subgroup_by_generators "
				"created group " << label << endl;
	}
}



}}}


