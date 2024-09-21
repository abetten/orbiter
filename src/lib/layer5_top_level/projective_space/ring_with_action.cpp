/*
 * ring_with_action.cpp
 *
 *  Created on: Sep 20, 2024
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace projective_geometry {



ring_with_action::ring_with_action()
{
	PA = NULL;

	Poly_ring = NULL;

	AonHPD = NULL;
}


ring_with_action::~ring_with_action()
{
	if (AonHPD) {
		FREE_OBJECT(AonHPD);
	}
}


void ring_with_action::ring_with_action_init(
		projective_geometry::projective_space_with_action *PA,
		ring_theory::homogeneous_polynomial_domain *Poly_ring,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_with_action::ring_with_action_init "
				"verbose_level=" << verbose_level << endl;
	}

	ring_with_action::PA = PA;
	ring_with_action::Poly_ring = Poly_ring;
	//ring_with_action::AonHPD = AonHPD;

	AonHPD = NEW_OBJECT(induced_actions::action_on_homogeneous_polynomials);
	if (f_v) {
		cout << "ring_with_action::ring_with_action_init "
				"before AonHPD->init" << endl;
	}
	AonHPD->init(
			PA->A, Poly_ring, verbose_level - 3);
	if (f_v) {
		cout << "ring_with_action::ring_with_action_init "
				"after AonHPD->init" << endl;
	}


	if (f_v) {
		cout << "ring_with_action::ring_with_action_init done" << endl;
	}
}

void ring_with_action::lift_mapping(
		int *gamma, int *Elt, int verbose_level)
// turn the permutation gamma into a semilinear mapping
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_with_action::lift_mapping" << endl;
	}



	int d;
	int *Mtx; // [d * d + 1]


	d = PA->P->Subspaces->n + 1;
	Mtx = NEW_int(d * d + 1);


	int frobenius;
	linear_algebra::linear_algebra_global LA;

	if (f_v) {
		cout << "ring_with_action::lift_mapping "
				"before LA.reverse_engineer_semilinear_map" << endl;
	}

	// works for any dimension:

	LA.reverse_engineer_semilinear_map(
			PA->P->Subspaces->F,
			PA->P->Subspaces->n,
			gamma, Mtx, frobenius,
		0 /*verbose_level*/);


	if (f_v) {
		cout << "ring_with_action::lift_mapping "
				"after LA.reverse_engineer_semilinear_map" << endl;
	}

	Mtx[d * d] = frobenius;

	PA->A->Group_element->make_element(
			Elt, Mtx,
			0 /* verbose_level*/);


	FREE_int(Mtx);

	if (f_v) {
		cout << "ring_with_action::lift_mapping done" << endl;
	}
}

void ring_with_action::apply(
		int *Elt, int *eqn_in, int *eqn_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_with_action::apply" << endl;
	}
	if (f_v) {
		cout << "ring_with_action::apply "
				"before substitute_semilinear" << endl;
	}

	int d, frobenius;
	int *Mtx; // [d * d + 1]


	d = PA->P->Subspaces->n + 1;
	Mtx = Elt;
	frobenius = Mtx[d * d];

	Poly_ring->substitute_semilinear(
			eqn_in,
			eqn_out,
			PA->A->is_semilinear_matrix_group(),
			frobenius, Mtx,
			0/*verbose_level*/);

	if (f_v) {
		cout << "ring_with_action::apply "
				"after substitute_semilinear" << endl;
	}

	if (f_v) {
		cout << "ring_with_action::apply done" << endl;
	}
}

}}}

