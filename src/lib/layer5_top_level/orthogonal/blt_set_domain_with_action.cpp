/*
 * blt_set_domain_with_action.cpp
 *
 *  Created on: Mar 28, 2023
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {


blt_set_domain_with_action::blt_set_domain_with_action()
{
	A = NULL;
	P = NULL;
	O = NULL;
	Blt_set_domain = NULL;

	PF = NULL;

}

blt_set_domain_with_action::~blt_set_domain_with_action()
{
	if (PF) {
		FREE_OBJECT(PF);
	}
}

void blt_set_domain_with_action::init(
		actions::action *A,
		geometry::projective_space *P,
		layer1_foundations::orthogonal_geometry::orthogonal *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_domain_with_action::init" << endl;
	}
	blt_set_domain_with_action::A = A;
	blt_set_domain_with_action::P = P;
	blt_set_domain_with_action::O = O;


	Blt_set_domain = NEW_OBJECT(orthogonal_geometry::blt_set_domain);

	if (f_v) {
		cout << "blt_set_domain_with_action::init "
				"before Blt_set_domain->init_blt_set_domain" << endl;
	}
	Blt_set_domain->init_blt_set_domain(O, P, verbose_level);
	if (f_v) {
		cout << "blt_set_domain_with_action::init "
				"after Blt_set_domain->init" << endl;
	}



	PF = NEW_OBJECT(combinatorics::polynomial_function_domain);

	if (f_v) {
		cout << "blt_set_domain_with_action::init "
				"before PF->init" << endl;
	}
	PF->init(Blt_set_domain->F, 1 /*n*/, verbose_level);
	if (f_v) {
		cout << "blt_set_domain_with_action::init "
				"after PF->init" << endl;
	}


	if (f_v) {
		cout << "blt_set_domain_with_action::init done" << endl;
	}
}

}}}



