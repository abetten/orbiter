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


int cubic_curve::compute_system_in_RREF(
		int nb_pts, int *pt_list, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int i, j, r;
	int *Pts;
	int *System;
	int *base_cols;

	if (f_v) {
		cout << "cubic_curve::compute_system_in_RREF" << endl;
		}
	Pts = NEW_int(nb_pts * 3);
	System = NEW_int(nb_pts * nb_monomials);
	base_cols = NEW_int(nb_monomials);

	if (FALSE) {
		cout << "cubic_curve::compute_system_in_RREF list of "
				"covered points by lines:" << endl;
		int_matrix_print(pt_list, nb_pts, P->k);
		}
	for (i = 0; i < nb_pts; i++) {
		P->unrank_point(Pts + i * 3, pt_list[i]);
		}
	if (f_v && FALSE) {
		cout << "cubic_curve::compute_system_in_RREF list of "
				"covered points in coordinates:" << endl;
		int_matrix_print(Pts, nb_pts, 3);
		}

	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < nb_monomials; j++) {
			System[i * nb_monomials + j] =
				F->evaluate_monomial(
					Poly->Monomials + j * 3,
					Pts + i * 3, 3);
			}
		}
	if (f_v && FALSE) {
		cout << "cubic_curve::compute_system_in_RREF "
				"The system:" << endl;
		int_matrix_print(System, nb_pts, nb_monomials);
		}
	r = F->Gauss_simple(System, nb_pts, nb_monomials,
		base_cols, 0 /* verbose_level */);
	if (FALSE) {
		cout << "cubic_curve::compute_system_in_RREF "
				"The system in RREF:" << endl;
		int_matrix_print(System, nb_pts, nb_monomials);
		}
	if (f_v) {
		cout << "cubic_curve::compute_system_in_RREF "
				"The system has rank " << r << endl;
		}
	FREE_int(Pts);
	FREE_int(System);
	FREE_int(base_cols);
	return r;
}




}}

