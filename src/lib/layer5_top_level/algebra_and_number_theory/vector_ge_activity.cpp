/*
 * vector_ge_activity.cpp
 *
 *  Created on: Dec 25, 2024
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {



vector_ge_activity::vector_ge_activity()
{
	Record_birth();
	Descr = NULL;
	VB = NULL;
	vec = NULL;

}


vector_ge_activity::~vector_ge_activity()
{
	Record_death();

}

void vector_ge_activity::init(
		vector_ge_activity_description *Descr,
		apps_algebra::vector_ge_builder *VB,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge_activity::init" << endl;
	}


	vector_ge_activity::Descr = Descr;
	vector_ge_activity::VB = VB;
	vec = VB->V;

	if (f_v) {
		cout << "vector_ge_activity::init done" << endl;
	}
}

void vector_ge_activity::perform_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge_activity::perform_activity" << endl;
	}


	if (Descr->f_report) {

		if (f_v) {
			cout << "vector_ge_activity::perform_activity f_report" << endl;
		}

		int f_with_permutation = true;
		int f_override_action = true;
		actions::action *A_special;

		A_special = vec->A;
		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"before vec->report_elements" << endl;
		}
		vec->report_elements(
				A_special->label,
				f_with_permutation,
				f_override_action,
				A_special,
				verbose_level);

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"after vec->report_elements" << endl;
		}


	}


	else if (Descr->f_export_GAP) {

		if (f_v) {
			cout << "vector_ge_activity::perform_activity f_export_GAP" << endl;
		}

		actions::action *A_special;

		A_special = vec->A;

		string fname;

		fname = A_special->label + "_elements.gap";

		{
			std::ofstream ost(fname);

			vec->print_generators_gap(
					ost, verbose_level);

			other::orbiter_kernel_system::file_io Fio;

			if (f_v) {
				cout << "vector_ge_activity::perform_activity "
						"Written file " << fname << " of size "
						<< Fio.file_size(fname) << endl;
			}

		}

	}

	else if (Descr->f_transform_variety) {

		if (f_v) {
			cout << "vector_ge_activity::perform_activity f_transform_variety" << endl;
		}

	}

	canonical_form::variety_object_with_action *Variety;


	Variety = Get_variety(Descr->transform_variety_label);

	int i;
	int *Elt;

	for (i = 0; i < vec->len; i++) {

		Elt = vec->ith(i);



	}


	if (f_v) {
		cout << "vector_ge_activity::perform_activity done" << endl;
	}

}



}}}

