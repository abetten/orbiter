/*
 * combinatorial_object_activity.cpp
 *
 *  Created on: Mar 20, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


combinatorial_object_activity::combinatorial_object_activity()
{
	Descr = NULL;
	COC = NULL;

}

combinatorial_object_activity::~combinatorial_object_activity()
{
}


void combinatorial_object_activity::init(combinatorial_object_activity_description *Descr,
		combinatorial_object_create *COC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::init" << endl;
	}

	combinatorial_object_activity::Descr = Descr;
	combinatorial_object_activity::COC = COC;


	if (f_v) {
		cout << "combinatorial_object_activity::init done" << endl;
	}
}

void combinatorial_object_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity" << endl;
	}

#if 0
	if (Descr->f_conic_type) {
		COC->Pts, COC->nb_pts

	}
#endif
	if (Descr->f_save) {

		file_io Fio;
		string fname;

		fname.assign(COC->fname);

		if (f_v) {
			cout << "We will write to the file " << fname << endl;
		}
		Fio.write_set_to_file(fname, COC->Pts, COC->nb_pts, verbose_level);
		if (f_v) {
			cout << "Written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
	}

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity done" << endl;
	}
}

}}


