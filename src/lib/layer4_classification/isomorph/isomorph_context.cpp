/*
 * isomorph_context.cpp
 *
 *  Created on: Jan 8, 2025
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace isomorph {


isomorph_context::isomorph_context()
{
	Record_birth();

	Descr = NULL;

	A = NULL;
	A2 = NULL;
	gen = NULL;
	target_size = 0;
	Control = NULL;

	ECA = NULL;

	callback_report = NULL;
	callback_subset_orbits = NULL;
	callback_data = NULL;

	f_has_final_test_function = false;
	final_test_function = NULL;
	final_test_data = NULL;

}

isomorph_context::~isomorph_context()
{
	Record_death();
}

void isomorph_context::init(
		isomorph_arguments *Descr,
		actions::action *A,
		actions::action *A2,
		poset_classification::poset_classification *gen,
	int target_size,
	poset_classification::poset_classification_control *Control,
	solvers_package::exact_cover_arguments *ECA,
	void (*callback_report)(
			isomorph *Iso, void *data, int verbose_level),
	void (*callback_subset_orbits)(
			isomorph *Iso, void *data, int verbose_level),
	void *callback_data,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph_context::init" << endl;
	}
	isomorph_context::Descr = Descr;
	isomorph_context::A = A;
	isomorph_context::A2 = A2;
	isomorph_context::gen = gen;
	isomorph_context::target_size = target_size;
	isomorph_context::Control = Control;
	isomorph_context::ECA = ECA;
	isomorph_context::callback_report = callback_report;
	isomorph_context::callback_subset_orbits = callback_subset_orbits;
	isomorph_context::callback_data = callback_data;

	if (!Descr->f_solution_prefix) {
		cout << "isomorph_context::init please "
				"use -solution_prefix <solution_prefix>" << endl;
		exit(1);
	}
	if (!Descr->f_base_fname) {
		cout << "isomorph_context::init please "
				"use -base_fname <base_fname>" << endl;
		exit(1);
	}

	//f_init_has_been_called = true;

	if (f_v) {
		cout << "isomorph_context::init done" << endl;
	}
}



}}}





