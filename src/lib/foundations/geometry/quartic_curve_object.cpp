/*
 * quartic_curve_object.cpp
 *
 *  Created on: May 20, 2021
 *      Author: betten
 */




#include "foundations.h"


using namespace std;



namespace orbiter {
namespace foundations {



quartic_curve_object::quartic_curve_object()
{
	q = 0;
	F = NULL;
	Dom = NULL;


	Pts = NULL;
	nb_pts = 0;

	Lines = NULL;
	nb_lines = 0;

	//eqn15[15]

	f_has_bitangents = FALSE;
	//bitangents28[28]

	QP = NULL;

	//null();
}

quartic_curve_object::~quartic_curve_object()
{
	freeself();
}

void quartic_curve_object::freeself()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::freeself" << endl;
	}
	if (Pts) {
		FREE_lint(Pts);
	}
	if (Lines) {
		FREE_lint(Lines);
	}
	if (QP) {
		FREE_OBJECT(QP);
	}



	if (f_v) {
		cout << "quartic_curve_object::freeself done" << endl;
	}
}

void quartic_curve_object::null()
{
}

void quartic_curve_object::init_equation_but_no_bitangents(quartic_curve_domain *Dom,
		int *eqn15,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::init_equation_but_no_bitangents" << endl;
		Orbiter->Int_vec.print(cout, eqn15, 15);
		cout << endl;
	}

	quartic_curve_object::Dom = Dom;
	F = Dom->P->F;
	q = F->q;

	f_has_bitangents = FALSE;
	Orbiter->Int_vec.copy(eqn15, quartic_curve_object::eqn15, 15);



	if (f_v) {
		cout << "quartic_curve_object::init_equation_but_no_bitangents before enumerate_points" << endl;
	}
	enumerate_points(0/*verbose_level - 1*/);
	if (f_v) {
		cout << "quartic_curve_object::init_equation_but_no_bitangents after enumerate_points" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object::init_equation_but_no_bitangents done" << endl;
	}
}

void quartic_curve_object::init_equation_and_bitangents(quartic_curve_domain *Dom,
		int *eqn15, long int *bitangents28,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents" << endl;
		cout << "eqn15:";
		Orbiter->Int_vec.print(cout, eqn15, 15);
		cout << endl;
		cout << "bitangents28:";
		Orbiter->Lint_vec.print(cout, bitangents28, 28);
		cout << endl;
	}

	quartic_curve_object::Dom = Dom;
	F = Dom->P->F;
	q = F->q;

	f_has_bitangents = TRUE;
	Orbiter->Int_vec.copy(eqn15, quartic_curve_object::eqn15, 15);
	Orbiter->Lint_vec.copy(bitangents28, quartic_curve_object::bitangents28, 28);



	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents before enumerate_points" << endl;
	}
	enumerate_points(0/*verbose_level - 1*/);
	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents after enumerate_points" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents done" << endl;
	}
}


void quartic_curve_object::init_equation_and_bitangents_and_compute_properties(quartic_curve_domain *Dom,
		int *eqn15, long int *bitangents28,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties before init_equation_and_bitangents" << endl;
	}
	init_equation_and_bitangents(Dom, eqn15, bitangents28, verbose_level);
	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties after init_equation_and_bitangents" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties before "
				"compute_properties" << endl;
	}
	compute_properties(verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties after "
				"compute_properties" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties after "
				"enumerate_points" << endl;
	}
}



void quartic_curve_object::enumerate_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::enumerate_points" << endl;
	}

	vector<long int> Points;

	if (f_v) {
		cout << "quartic_curve_object::enumerate_points before "
				"Surf->enumerate_points" << endl;
	}
	Dom->Poly4_3->enumerate_points(eqn15, Points, 0 /*verbose_level - 1*/);

	if (f_v) {
		cout << "quartic_curve_object::enumerate_points after "
				"Surf->enumerate_points" << endl;
	}
	if (f_v) {
		cout << "quartic_curve_object::enumerate_points The surface "
				"has " << Points.size() << " points" << endl;
	}
	int i;

	nb_pts = Points.size();
	Pts = NEW_lint(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		Pts[i] = Points[i];
	}


	if (f_v) {
		cout << "quartic_curve_object::enumerate_points done" << endl;
	}
}



void quartic_curve_object::compute_properties(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::compute_properties" << endl;
	}

	QP = NEW_OBJECT(quartic_curve_object_properties);

	QP->init(this, verbose_level);

	if (f_v) {
		cout << "quartic_curve_object::compute_properties done" << endl;
	}
}

void quartic_curve_object::recompute_properties(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::recompute_properties" << endl;
	}


	if (QP) {
		FREE_OBJECT(QP);
		QP = NULL;
	}

	QP = NEW_OBJECT(quartic_curve_object_properties);

	QP->init(this, verbose_level);


	if (f_v) {
		cout << "quartic_curve_object::recompute_properties done" << endl;
	}
}










void quartic_curve_object::identify_lines(long int *lines, int nb_lines,
	int *line_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, idx;
	sorting Sorting;

	if (f_v) {
		cout << "quartic_curve_object::identify_lines" << endl;
		}
	for (i = 0; i < nb_lines; i++) {
		if (!Sorting.lint_vec_search_linear(bitangents28, 28, lines[i], idx)) {
			cout << "quartic_curve_object::identify_lines could "
					"not find lines[" << i << "]=" << lines[i]
					<< " in bitangents28[]" << endl;
			exit(1);
			}
		line_idx[i] = idx;
		}
	if (f_v) {
		cout << "quartic_curve_object::identify_lines done" << endl;
		}
}



int quartic_curve_object::find_point(long int P, int &idx)
{
	sorting Sorting;

	if (Sorting.lint_vec_search(Pts, nb_pts, P,
			idx, 0 /* verbose_level */)) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

void quartic_curve_object::create_surface(int *eqn20, int verbose_level)
// Given a quartic Q in X1,X2,X3, compute an associated cubic surface
// whose projection from (1,0,0,0) gives back the quartic Q.
// Pick 4 bitangents L0,L1,L2,L3 so that the 8 points of tangency lie on a conic C.
// Then, create the cubic surface with equation
// (- lambda * mu) / 4 * X0^2 * L0 (the equation of the first of the four bitangents)
// + X0 * lambda * C (the conic equation)
// + L1 * L2 * L3 (the product of the equations of the last three bitangents)
// Here 1, lambda, mu are the coefficients of a linear dependency between
// Q (the quartic), C^2, L0*L1*L2*L3, so
// Q + lambda * C^2 + mu * L0*L1*L2*L3 = 0.
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "quartic_curve_object::create_surface" << endl;
	}

	if (QP == NULL) {
		cout << "quartic_curve_object::create_surface, QP == NULL" << endl;
		exit(1);
	}

	int *Bitangents;
	int nb_bitangents;
	int set[4];
	int Idx[4];
	int pt_idx[8];
	long int Points[8];
	long int Bitangents4[4];
	int Bitangents_coeffs[16];
	int six_coeffs_conic[6];
	int i, r;
	long int nCk, h;
	combinatorics_domain Combi;
	int conic_squared_15[15];
	int four_lines_15[15];
	int M1[3 * 15];
	int M2[15 * 3];

	QP->Bitangent_line_type->get_class_by_value(Bitangents, nb_bitangents, 2 /*value */,
			verbose_level);

	if (nb_bitangents < 4) {
		cout << "quartic_curve_object::create_surface, nb_bitangents < 4" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "quartic_curve_object::create_surface we found " << nb_bitangents << " bitangents" << endl;
		Orbiter->Int_vec.print(cout, Bitangents, nb_bitangents);
		cout << endl;
	}

	nCk = Combi.binomial_lint(nb_bitangents, 4);
	for (h = 0; h < nCk; h++) {
		Combi.unrank_k_subset(h, set, nb_bitangents, 4);
		if (f_v) {
			cout << "quartic_curve_object::create_surface trying subset " << h << " / " << nCk << " which is ";
			Orbiter->Int_vec.print(cout, set, 4);
			cout << endl;
		}
		for (i = 0; i < 4; i++) {
			Idx[i] = Bitangents[set[i]];
			Bitangents4[i] = bitangents28[Idx[i]];
		}

		for (i = 0; i < 4; i++) {

			if (QP->pts_on_lines->Set_size[Idx[i]] != 2) {
				cout << "quartic_curve_object::create_surface QP->pts_on_lines->Set_size[Idx[i]] != 2" << endl;
				exit(1);
			}
			pt_idx[i * 2 + 0] = QP->pts_on_lines->Sets[Idx[i]][0];
			pt_idx[i * 2 + 1] = QP->pts_on_lines->Sets[Idx[i]][1];

		}
		for (i = 0; i < 8; i++) {
			Points[i] = Pts[pt_idx[i]];
		}
		if (f_v) {
			cout << "quartic_curve_object::create_surface trying subset " << h << " / " << nCk << " Points = ";
			Orbiter->Lint_vec.print(cout, Points, 8);
			cout << endl;
		}

		if (Dom->P->determine_conic_in_plane(
				Points, 8,
				six_coeffs_conic,
				verbose_level)) {
			cout << "quartic_curve_object::create_surface The four bitangents are syzygetic" << endl;
			break;
		}
	}
	if (h == nCk) {
		cout << "quartic_curve_object::create_surface, could not find a syzygetic set of bitangents" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "quartic_curve_object::create_surface trying subset " << h << " / " << nCk << " Bitangents4 = ";
		Orbiter->Lint_vec.print(cout, Bitangents4, 4);
		cout << endl;
	}

	Dom->multiply_conic_times_conic(six_coeffs_conic,
			six_coeffs_conic, conic_squared_15,
			verbose_level);

	if (f_v) {
		cout << "quartic_curve_object::create_surface conic squared = ";
		Orbiter->Int_vec.print(cout, conic_squared_15, 15);
		cout << endl;
	}

	for (i = 0; i < 4; i++) {
		Dom->unrank_line_in_dual_coordinates(Bitangents_coeffs + i * 3, Bitangents4[i]);
	}

	if (f_v) {
		cout << "quartic_curve_object::create_surface chosen bitangents in dual coordinates = ";
		Orbiter->Int_vec.matrix_print(Bitangents_coeffs, 4, 3);
	}


	Dom->multiply_four_lines(Bitangents_coeffs + 0 * 3,
			Bitangents_coeffs + 1 * 3,
			Bitangents_coeffs + 2 * 3,
			Bitangents_coeffs + 3 * 3,
			four_lines_15,
			verbose_level);

	if (f_v) {
		cout << "quartic_curve_object::create_surface product of 4 bitangents = ";
		Orbiter->Int_vec.print(cout, four_lines_15, 15);
		cout << endl;
	}

	Orbiter->Int_vec.copy(eqn15, M1, 15);
	Orbiter->Int_vec.copy(conic_squared_15, M1 + 15, 15);
	Orbiter->Int_vec.copy(four_lines_15, M1 + 30, 15);

	Orbiter->Int_vec.transpose(M1, 3, 15, M2);

	r = F->RREF_and_kernel(3, 15, M2, 0 /* verbose_level*/);

	if (r != 2) {
		cout << "quartic_curve_object::create_surface r != 2" << endl;
		exit(1);
	}

	F->PG_element_normalize_from_front(M2 + 6, 1, 3);
	if (f_v) {
		cout << "quartic_curve_object::create_surface kernel = ";
		Orbiter->Int_vec.print(cout, M2 + 6, 3);
		cout << endl;
	}
	int lambda, mu;

	lambda = M2[7];
	mu = M2[8];
	if (f_v) {
		cout << "quartic_curve_object::create_surface lambda = " << lambda << " mu = " << mu << endl;
	}

	int f1_three_coeff[3]; // - lambda * mu / 4 * the equation of the first of the four bitangents
	int f2_six_coeff[6]; // lambda * conic equation
	int f3_ten_coeff[10]; // the product of the last three bitangents

	Dom->multiply_three_lines(
			Bitangents_coeffs + 1 * 3,
			Bitangents_coeffs + 2 * 3,
			Bitangents_coeffs + 3 * 3,
			f3_ten_coeff,
			verbose_level);

#if 0
	int sqrt_lambda;

	if (f_v) {
		cout << "quartic_curve_object::create_surface computing square root of lambda" << endl;
	}
	F->square_root(lambda, sqrt_lambda);
	if (f_v) {
		cout << "quartic_curve_object::create_surface sqrt_lambda = " << sqrt_lambda << endl;
	}
#endif

	Dom->Poly2_3->multiply_by_scalar(
			six_coeffs_conic, lambda, f2_six_coeff,
			verbose_level);

	int half, fourth, a;

	half = F->inverse(2);
	fourth = F->mult(half, half);
	a = F->mult(F->negate(F->mult(lambda, mu)), fourth);

	Dom->Poly1_3->multiply_by_scalar(
			Bitangents_coeffs + 0 * 3, a, f1_three_coeff,
			verbose_level);


	// and now, create the cubic with equation
	// (- lambda * mu) / 4 * X0^2 * L0 (the equation of the first of the four bitangents)
	// + X0 * lambda * conic equation
	// + L1 * L2 * L3 (the product of the equations of the last three bitangents)

	Dom->assemble_cubic_surface(f1_three_coeff, f2_six_coeff, f3_ten_coeff, eqn20,
		verbose_level);

	if (f_v) {
		cout << "quartic_curve_object::create_surface eqn20 = ";
		Orbiter->Int_vec.print(cout, eqn20, 20);
		cout << endl;
	}

	FREE_int(Bitangents);

	if (f_v) {
		cout << "quartic_curve_object::create_surface done" << endl;
	}
}



}}

