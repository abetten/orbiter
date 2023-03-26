/*
 * blt_set_activity.cpp
 *
 *  Created on: Jan 13, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {


blt_set_activity::blt_set_activity()
{
	Descr = NULL;

	BC = NULL;
}

blt_set_activity::~blt_set_activity()
{
}

void blt_set_activity::init(
		blt_set_activity_description *Descr,
		orthogonal_geometry_applications::BLT_set_create *BC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_activity::init" << endl;
	}

	blt_set_activity::Descr = Descr;
	blt_set_activity::BC = BC;



	if (f_v) {
		cout << "blt_set_activity::init done" << endl;
	}
}


void blt_set_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_activity::perform_activity" << endl;
	}

	if (Descr->f_report) {

		if (f_v) {
			cout << "blt_set_activity::perform_activity f_report" << endl;
		}
		if (f_v) {
			cout << "blt_set_activity::perform_activity before BC->report" << endl;
		}
		BC->report(verbose_level);
		if (f_v) {
			cout << "blt_set_activity::perform_activity after BC->report" << endl;
		}

	}

	else if (Descr->f_export_gap) {

		if (f_v) {
			cout << "blt_set_activity::perform_activity f_export_gap" << endl;
		}
		if (f_v) {
			cout << "blt_set_activity::perform_activity before BC->export_gap" << endl;
		}
		BC->export_gap(verbose_level);
		if (f_v) {
			cout << "blt_set_activity::perform_activity after BC->export_gap" << endl;
		}

	}


	else if (Descr->f_create_flock) {

		if (f_v) {
			cout << "blt_set_activity::perform_activity f_create_flock" << endl;
		}
		if (f_v) {
			cout << "blt_set_activity::perform_activity before BC->create_flock" << endl;
		}
		BC->create_flock(Descr->create_flock_point_idx, verbose_level);
		if (f_v) {
			cout << "blt_set_activity::perform_activity after BC->create_flock" << endl;
		}

	}


	else if (Descr->f_BLT_test) {

		if (f_v) {
			cout << "blt_set_activity::perform_activity f_BLT_test" << endl;
		}

		if (f_v) {
			cout << "blt_set_activity::perform_activity before BC->BLT_test" << endl;
		}
		BC->BLT_test(verbose_level);
		if (f_v) {
			cout << "blt_set_activity::perform_activity after BC->BLT_test" << endl;
		}

	}


	if (f_v) {
		cout << "blt_set_activity::perform_activity done" << endl;
	}

}




}}}



