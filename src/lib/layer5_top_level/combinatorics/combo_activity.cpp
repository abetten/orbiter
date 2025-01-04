/*
 * combo_activity.cpp
 *
 *  Created on: Jan 3, 2025
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {



combo_activity::combo_activity()
{
	Record_birth();
	Descr = NULL;

	pOwP = NULL;
	nb_objects = 0;

	nb_output = 0;
	Output = NULL;

}

combo_activity::~combo_activity()
{
	Record_death();
}


void combo_activity::init(
		combo_activity_description *Descr,
		canonical_form::combinatorial_object_with_properties **pOwP,
		int nb_objects,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combo_activity::init" << endl;
	}

	combo_activity::Descr = Descr;
	combo_activity::pOwP = pOwP;
	combo_activity::nb_objects = nb_objects;

	if (f_v) {
		cout << "combo_activity::init done" << endl;
	}
}



void combo_activity::perform_activity(
		other::orbiter_kernel_system::activity_output *&AO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combo_activity::perform_activity" << endl;
		cout << "combo_activity::perform_activity verbose_level = " << verbose_level << endl;
	}


	if (Descr->f_report) {
		if (f_v) {
			cout << "combo_activity::perform_activity_combo "
					"f_report" << endl;
		}

		int i;

		for (i = 0; i < nb_objects; i++) {

			if (f_v) {
				cout << "combo_activity::perform_activity_combo "
						"before OwP->latex_report_wrapper" << endl;
			}
			pOwP[i]->latex_report_wrapper(
					pOwP[i]->label,
					Descr->Objects_report_options,
					verbose_level);


			if (f_v) {
				cout << "combo_activity::perform_activity_combo "
						"after OwP->latex_report_wrapper" << endl;
			}
		}
	}

	if (f_v) {
		cout << "combo_activity::perform_activity done" << endl;
	}
}


#if 0
void compute_TDO(
		int max_TDO_depth, int verbose_level);
#endif






}}}




