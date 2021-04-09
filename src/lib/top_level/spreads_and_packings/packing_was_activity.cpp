/*
 * packing_was_activity.cpp
 *
 *  Created on: Apr 3, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


packing_was_activity::packing_was_activity()
{
	Descr = NULL;
	PW = NULL;

}

packing_was_activity::~packing_was_activity()
{

}



void packing_was_activity::init(packing_was_activity_description *Descr,
		packing_was *PW,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was_activity::init" << endl;
	}

	packing_was_activity::Descr = Descr;
	packing_was_activity::PW = PW;

	if (f_v) {
		cout << "packing_was_activity::init done" << endl;
	}
}

void packing_was_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (Descr->f_report) {


		if (f_v) {
			cout << "packing_was_activity::perform_activity before PW->report" << endl;
		}

		PW->report(0 /* verbose_level */);

		if (f_v) {
			cout << "packing_was_activity::perform_activity after PW->report" << endl;
		}


	}
	else if (Descr->f_export_reduced_spread_orbits) {


		if (f_v) {
			cout << "packing_was_activity::perform_activity before PW->export_reduced_spread_orbits_csv" << endl;
		}

		int f_original_spread_numbers = TRUE;

		PW->export_reduced_spread_orbits_csv(Descr->export_reduced_spread_orbits_fname_base,
				f_original_spread_numbers, verbose_level);

		if (f_v) {
			cout << "packing_was_activity::perform_activity after PW->export_reduced_spread_orbits_csv" << endl;
		}

	}



	if (f_v) {
		cout << "packing_was_activity::perform_activity" << endl;
	}

}

}}


