/*
 * cubic_curve_with_action.cpp
 *
 *  Created on: Mar 7, 2019
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace top_level {


cubic_curve_with_action::cubic_curve_with_action()
{
	null();
}

cubic_curve_with_action::~cubic_curve_with_action()
{
	freeself();
}

void cubic_curve_with_action::null()
{
	q = 0;
	F = NULL;
	CC = NULL;
	A = NULL;
	A2 = NULL;
	Elt1 = NULL;
	AonHPD_3_3 = NULL;

}

void cubic_curve_with_action::freeself()
{
	if (A) {
		FREE_OBJECT(A);
	}
	if (A2) {
		FREE_OBJECT(A2);
	}
	if (Elt1) {
		FREE_int(Elt1);
	}
	if (AonHPD_3_3) {
		FREE_OBJECT(AonHPD_3_3);
	}
	null();
}

void cubic_curve_with_action::init(cubic_curve *CC,
		int f_semilinear, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cubic_curve_with_action::init" << endl;
	}
	cubic_curve_with_action::CC = CC;
	cubic_curve_with_action::f_semilinear = f_semilinear;
	F = CC->F;
	q = F->q;

	init_group(f_semilinear, verbose_level);

	Elt1 = NEW_int(A->elt_size_in_int);

	AonHPD_3_3 = NEW_OBJECT(action_on_homogeneous_polynomials);
	if (f_v) {
		cout << "cubic_curve_with_action::init "
				"before AonHPD_3_3->init" << endl;
	}
	AonHPD_3_3->init(A, CC->Poly, verbose_level);


	if (f_v) {
		cout << "cubic_curve_with_action::init done" << endl;
	}
}

void cubic_curve_with_action::init_group(int f_semilinear,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cubic_curve_with_action::init_group" << endl;
	}
	if (f_v) {
		cout << "cubic_curve_with_action::init_group "
				"creating linear group" << endl;
	}

	vector_ge *nice_gens;
	//sims *S;

	A = NEW_OBJECT(action);

	A->init_linear_group(//S,
		F, 3,
		TRUE /*f_projective*/,
		FALSE /* f_general*/,
		FALSE /* f_affine */,
		f_semilinear, FALSE /* f_special */,
		nice_gens,
		0 /* verbose_level*/);

	FREE_OBJECT(nice_gens);
	//FREE_OBJECT(S);

	if (f_v) {
		cout << "cubic_curve_with_action::init_group "
				"creating linear group done" << endl;
	}


	if (f_v) {
		cout << "cubic_curve_with_action::init_group "
				"creating action on lines" << endl;
	}
	A2 = A->induced_action_on_grassmannian(2, verbose_level);
	if (f_v) {
		cout << "cubic_curve_with_action::init_group "
				"creating action on lines done" << endl;
	}


	if (f_v) {
		cout << "cubic_curve_with_action::init_group done" << endl;
	}
}


}}

