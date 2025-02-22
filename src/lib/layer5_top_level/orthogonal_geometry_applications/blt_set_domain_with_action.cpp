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
	Record_birth();
	A = NULL;
	P = NULL;
	O = NULL;
	Blt_set_domain = NULL;

	PF = NULL;

}

blt_set_domain_with_action::~blt_set_domain_with_action()
{
	Record_death();
	if (PF) {
		FREE_OBJECT(PF);
	}
}

void blt_set_domain_with_action::init(
		actions::action *A,
		geometry::projective_geometry::projective_space *P,
		layer1_foundations::geometry::orthogonal_geometry::orthogonal *O,
		int f_create_extension_fields,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_domain_with_action::init" << endl;
	}
	blt_set_domain_with_action::A = A;
	blt_set_domain_with_action::P = P;
	blt_set_domain_with_action::O = O;


	Blt_set_domain = NEW_OBJECT(geometry::orthogonal_geometry::blt_set_domain);

	if (f_v) {
		cout << "blt_set_domain_with_action::init "
				"before Blt_set_domain->init_blt_set_domain" << endl;
	}
	Blt_set_domain->init_blt_set_domain(O, P, f_create_extension_fields, verbose_level);
	if (f_v) {
		cout << "blt_set_domain_with_action::init "
				"after Blt_set_domain->init" << endl;
	}


	if (f_create_extension_fields) {
		PF = NEW_OBJECT(combinatorics::special_functions::polynomial_function_domain);

		if (f_v) {
			cout << "blt_set_domain_with_action::init "
					"before PF->init" << endl;
		}
		PF->init(Blt_set_domain->F, 1 /*n*/, verbose_level);
		if (f_v) {
			cout << "blt_set_domain_with_action::init "
					"after PF->init" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "blt_set_domain_with_action::init "
					"not setting up polynomial_function_domain" << endl;
		}

	}


	if (f_v) {
		cout << "blt_set_domain_with_action::init done" << endl;
	}
}

}}}



