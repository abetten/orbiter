/*
 * classify_cubic_curves.cpp
 *
 *  Created on: Mar 7, 2019
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


classify_cubic_curves::classify_cubic_curves()
{
	q = 0;
	F = NULL;
	A = NULL; // do not free

	CCA = NULL; // do not free
	CC = NULL; // do not free

	Arc_gen = NULL;



	Flag_orbits = NULL;

	nb_orbits_on_curves = 0;

	Curves = NULL;
	//null();
}

classify_cubic_curves::~classify_cubic_curves()
{
	freeself();
}

void classify_cubic_curves::null()
{
}

void classify_cubic_curves::freeself()
{
	if (Arc_gen) {
		FREE_OBJECT(Arc_gen);
	}
	if (Flag_orbits) {
		FREE_OBJECT(Flag_orbits);
	}
	if (Curves) {
		FREE_OBJECT(Curves);
	}
	null();
}

void classify_cubic_curves::init(cubic_curve_with_action *CCA,
		const char *starter_directory_name,
		const char *base_fname,
		int argc, const char **argv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_cubic_curves::init" << endl;
		}
	classify_cubic_curves::CCA = CCA;
	F = CCA->F;
	q = F->q;
	A = CCA->A;
	CC = CCA->CC;

	Arc_gen = NEW_OBJECT(arc_generator);


	if (f_v) {
		cout << "classify_cubic_curves::init before Arc_gen->init" << endl;
		}

	Arc_gen->read_arguments(argc, argv);


	Arc_gen->init(F,
			starter_directory_name,
			base_fname,
			9 /* starter_size */,
			argc, argv,
			verbose_level);


	if (f_v) {
		cout << "classify_cubic_curves::init after Arc_gen->init" << endl;
		}


	if (f_v) {
		cout << "classify_cubic_curves::init done" << endl;
		}
}

void classify_cubic_curves::compute_starter(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "classify_cubic_curves::compute_starter" << endl;
		}
	Arc_gen->compute_starter(verbose_level);
	if (f_v) {
		cout << "classify_cubic_curves::compute_starter done" << endl;
		}
}

}}


