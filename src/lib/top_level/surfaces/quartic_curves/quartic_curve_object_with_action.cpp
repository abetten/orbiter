/*
 * quartic_curve_object_with_action.cpp
 *
 *  Created on: May 22, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {




quartic_curve_object_with_action::quartic_curve_object_with_action()
{
	F = NULL;
	DomA = NULL;
	QO = NULL;
	Aut_gens = NULL;
	f_has_nice_gens = FALSE;
	nice_gens = NULL;
	projectivity_group_gens = NULL;
	Syl = NULL;
	A_on_points = NULL;
	Orbits_on_points = NULL;
}

quartic_curve_object_with_action::~quartic_curve_object_with_action()
{
}

void quartic_curve_object_with_action::init(quartic_curve_domain_with_action *DomA,
		quartic_curve_object *QO,
		strong_generators *Aut_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_with_action::init" << endl;
	}
	quartic_curve_object_with_action::DomA = DomA;
	quartic_curve_object_with_action::QO = QO;
	quartic_curve_object_with_action::Aut_gens = Aut_gens;
	F = DomA->Dom->F;

	if (f_v) {
		cout << "quartic_curve_object_with_action::init done" << endl;
	}
}


}}
