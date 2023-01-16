/*
 * large_set_activity.cpp
 *
 *  Created on: May 26, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {


large_set_activity::large_set_activity()
{
	Descr = NULL;
	LSW = NULL;
}


large_set_activity::~large_set_activity()
{
}


void large_set_activity::perform_activity(
		large_set_activity_description *Descr,
		large_set_was *LSW, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_activity::perform_activity" << endl;
	}

	large_set_activity::Descr = Descr;
	large_set_activity::LSW = LSW;

#if 0
	if (Descr->f_create_table) {
	}
#endif


	if (f_v) {
		cout << "large_set_activity::perform_activity done" << endl;
	}

}




}}}
