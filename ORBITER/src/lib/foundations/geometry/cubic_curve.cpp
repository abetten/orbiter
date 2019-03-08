/*
 * cubic_curve.cpp
 *
 *  Created on: Mar 7, 2019
 *      Author: betten
 */

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


cubic_curve::cubic_curve()
{
	q = 0;
	F = NULL;
	P = NULL;


	nb_monomials = 0;


	Poly = NULL;

}

cubic_curve::~cubic_curve()
{
	freeself();
}


void cubic_curve::freeself()
{
	int f_v = FALSE;

	if (f_v) {
		cout << "cubic_curve::freeself" << endl;
		}
	if (P) {
		FREE_OBJECT(P);
	}
	if (Poly) {
		FREE_OBJECT(Poly);
	}
}

void cubic_curve::init(finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cubic_curve::init" << endl;
		}

	cubic_curve::F = F;
	q = F->q;
	if (f_v) {
		cout << "cubic_curve::init q = "
				<< q << endl;
		}

	P = NEW_OBJECT(projective_space);
	if (f_v) {
		cout << "cubic_curve::init before P->init" << endl;
		}
	P->init(2, F,
		TRUE /*f_init_incidence_structure */,
		verbose_level - 2);
	if (f_v) {
		cout << "cubic_curve::init after P->init" << endl;
		}

	Poly = NEW_OBJECT(homogeneous_polynomial_domain);

	Poly->init(F,
			3 /* nb_vars */, 3 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);

	nb_monomials = Poly->nb_monomials;
	if (f_v) {
		cout << "cubic_curve::init nb_monomials=" << nb_monomials << endl;
		}


	if (f_v) {
		cout << "cubic_curve::init done" << endl;
		}
}


}}

