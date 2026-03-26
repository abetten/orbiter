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
namespace user_interface {
namespace activities_layer5 {


large_set_activity::large_set_activity()
{
	Record_birth();
	Descr = NULL;
	Large_set_classify = NULL;
}


large_set_activity::~large_set_activity()
{
	Record_death();
}


void large_set_activity::perform_activity(
		large_set_activity_description *Descr,
		apps_combinatorics::large_set_classify *Large_set_classify, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_activity::perform_activity" << endl;
	}

	large_set_activity::Descr = Descr;
	large_set_activity::Large_set_classify = Large_set_classify;

#if 0
	if (Descr->f_create_table) {
	}
#endif


	if (f_v) {
		cout << "large_set_activity::perform_activity done" << endl;
	}

}




}}}}


