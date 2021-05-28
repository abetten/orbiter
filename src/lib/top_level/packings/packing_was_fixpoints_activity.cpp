/*
 * packing_was_fixpoints_activity.cpp
 *
 *  Created on: Apr 3, 2021
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


packing_was_fixpoints_activity::packing_was_fixpoints_activity()
{
	Descr = NULL;
	PWF = NULL;

}

packing_was_fixpoints_activity::~packing_was_fixpoints_activity()
{

}



void packing_was_fixpoints_activity::init(packing_was_fixpoints_activity_description *Descr,
		packing_was_fixpoints *PWF,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was_fixpoints_activity::init" << endl;
	}

	packing_was_fixpoints_activity::Descr = Descr;
	packing_was_fixpoints_activity::PWF = PWF;

	if (f_v) {
		cout << "packing_was_fixpoints_activity::init done" << endl;
	}
}

void packing_was_fixpoints_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (Descr->f_report) {


		if (f_v) {
			cout << "packing_was_fixpoints_activity::perform_activity before PW->report" << endl;
		}

		PWF->report(verbose_level);

		if (f_v) {
			cout << "packing_was_fixpoints_activity::perform_activity after PW->report" << endl;
		}


	}



	if (f_v) {
		cout << "packing_was_fixpoints_activity::perform_activity" << endl;
	}

}

}}



