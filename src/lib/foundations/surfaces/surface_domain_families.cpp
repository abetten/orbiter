/*
 * surface_domain_families.cpp
 *
 *  Created on: Jul 22, 2020
 *      Author: betten
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


void surface_domain::create_equation_bes(int a, int c, int *coeff, int verbose_level)
// bes means five in Turkish
{
	int f_v = (verbose_level >= 1);
	int ap1c, apc, acp1;
	int a2, a3, a4;
	int w1, w2;
	int alpha, beta, gamma, delta, epsilon;

	if (f_v) {
		cout << "surface_domain::create_equation_bes" << endl;
	}

	a2 = F->mult(a, a);
	a3 = F->mult(a2, a);
	a4 = F->mult(a3, a);

	w1 = F->add6(a4, F->mult(a3, c), F->mult(a2, c), F->mult(a, c), c, 1);

	w2 = F->add6(F->mult(a4, c), a4, a3, a2, a, c);


	ap1c = F->power(F->add(a, 1), 3);
	apc = F->add(a, c);
	acp1 = F->add(F->mult(a, c), 1);

	alpha = F->mult(ap1c, apc);
	beta = w1;
	gamma = F->mult(ap1c, acp1);
	delta = w2;
	epsilon = F->mult(ap1c, F->mult(a, c));

	int_vec_zero(coeff, nb_monomials);

	coeff[4] = coeff[7] = coeff[8] = coeff[11] = coeff[12] = alpha;
	coeff[17] = beta;
	coeff[18] = gamma;
	coeff[19] = delta;
	coeff[15] = epsilon;


	if (f_v) {
		cout << "surface_domain::create_equation_bes done" << endl;
	}
}



void surface_domain::create_equation_F13(int a, int *coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int b, c;

	if (f_v) {
		cout << "surface_domain::create_equation_F13" << endl;
	}

	b = F->power(F->add(a, 1), 5);
	c = F->add(F->power(a, 3), 1);
	int_vec_zero(coeff, nb_monomials);

	coeff[6] = b;
	coeff[13] = b;
	coeff[8] = a;
	coeff[11] = a;
	coeff[19] = c;
	if (f_v) {
		cout << "surface_domain::create_equation_F13 done" << endl;
	}
}

void surface_domain::create_equation_G13(int a, int *coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int b, c;

	if (f_v) {
		cout << "surface_domain::create_equation_G13" << endl;
	}

	b = F->mult(a, F->add(a, 1));
	c = F->add(F->mult(a, a), F->add(a, 1));
	int_vec_zero(coeff, nb_monomials);

	coeff[5] = coeff[8] = coeff[9] = coeff[10] = coeff[11] = coeff[12] = 1;
	coeff[14] = coeff[15] = b;
	coeff[18] = coeff[19] = c;
	if (f_v) {
		cout << "surface_domain::create_equation_G13 done" << endl;
	}
}

int surface_domain::create_surface_bes(int a, int c,
	int *coeff20,
	long int *Lines27,
	int &nb_E,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::create_surface_bes" << endl;
	}

	if (f_v) {
		cout << "surface_domain::create_surface_bes before create_equation_bes" << endl;
	}
	create_equation_bes(a, c, coeff20, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_bes after create_equation_bes" << endl;
	}

	surface_object *SO;

	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_domain::create_surface_bes before SO->init_equation" << endl;
	}
	if (!SO->init_equation(this, coeff20, verbose_level)) {
		if (f_v) {
			cout << "surface_domain::create_surface_bes SO->init_equation returns FALSE, returning" << endl;
		}
		FREE_OBJECT(SO);
		return FALSE;
	}

	if (f_v) {
		cout << "surface_domain::create_surface_bes after SO->init_equation" << endl;
	}

	nb_E = SO->nb_Eckardt_points;

	lint_vec_copy(SO->Lines, Lines27, 27);

	FREE_OBJECT(SO);


	if (f_v) {
		cout << "surface_domain::create_surface_bes done" << endl;
	}
	return TRUE;
}


int surface_domain::create_surface_F13(int a,
	int *coeff20,
	long int *Lines27,
	int &nb_E,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::create_surface_F13" << endl;
	}

	if (f_v) {
		cout << "surface_domain::create_surface_F13 before create_equation_F13" << endl;
	}
	create_equation_F13(a, coeff20, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_F13 after create_equation_F13" << endl;
	}

	surface_object *SO;

	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_domain::create_surface_F13 before SO->init_equation" << endl;
	}
	if (!SO->init_equation(this, coeff20, verbose_level)) {
		if (f_v) {
			cout << "surface_domain::create_surface_F13 SO->init_equation returns FALSE, returning" << endl;
		}
		FREE_OBJECT(SO);
		return FALSE;
	}

	if (f_v) {
		cout << "surface_domain::create_surface_F13 after SO->init_equation" << endl;
	}

	nb_E = SO->nb_Eckardt_points;

	lint_vec_copy(SO->Lines, Lines27, 27);

	FREE_OBJECT(SO);


	if (f_v) {
		cout << "surface_domain::create_surface_F13 done" << endl;
	}
	return TRUE;
}

int surface_domain::create_surface_G13(int a,
	int *coeff20,
	long int *Lines27,
	int &nb_E,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::create_surface_G13" << endl;
	}

	if (f_v) {
		cout << "surface_domain::create_surface_G13 before create_equation_G13" << endl;
	}
	create_equation_G13(a, coeff20, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_G13 after create_equation_G13" << endl;
	}

	surface_object *SO;

	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_domain::create_surface_G13 before SO->init_equation" << endl;
	}
	if (!SO->init_equation(this, coeff20, verbose_level)) {
		if (f_v) {
			cout << "surface_domain::create_surface_G13 SO->init_equation returns FALSE, returning" << endl;
		}
		FREE_OBJECT(SO);
		return FALSE;
	}

	if (f_v) {
		cout << "surface_domain::create_surface_G13 after SO->init_equation" << endl;
	}

	nb_E = SO->nb_Eckardt_points;

	lint_vec_copy(SO->Lines, Lines27, 27);

	FREE_OBJECT(SO);


	if (f_v) {
		cout << "surface_domain::create_surface_G13 done" << endl;
	}
	return TRUE;
}

void surface_domain::create_equation_HCV(int a, int b,
		int *coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int alpha, beta;

	if (f_v) {
		cout << "surface_domain::create_equation_HCV" << endl;
	}
	alpha = F->negate(F->mult(b, b));
	beta = F->mult(F->mult(F->power(b, 3),
		F->add(1, F->mult(a, a))), F->inverse(a));
	int_vec_zero(coeff, nb_monomials);

	coeff[3] = 1;
	coeff[6] = alpha;
	coeff[9] = alpha;
	coeff[12] = alpha;
	coeff[16] = beta;
	//coeff[19] = beta;
	if (f_v) {
		cout << "surface_domain::create_equation_HCV done" << endl;
	}
}

int surface_domain::test_HCV_form_alpha_beta(int *coeff,
	int &alpha, int &beta, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = TRUE;
	int zeroes[] = {0,1,2,4,5,7,8,10,11,13,14,15,17,18,19};
	int alphas[] = {6,9,12};
	int betas[] = {16};
	int a;

	if (f_v) {
		cout << "surface_domain::test_HCV_form_alpha_beta" << endl;
	}
	if (!int_vec_is_constant_on_subset(coeff,
		zeroes, sizeof(zeroes) / sizeof(int), a)) {
		cout << "surface_domain::test_HCV_form_alpha_beta "
				"not constant on zero set" << endl;
		return FALSE;
	}
	if (a != 0) {
		cout << "surface_domain::test_HCV_form_alpha_beta "
				"not zero on zero set" << endl;
		return FALSE;
	}
	if (coeff[3] != 1) {
		cout << "surface_domain::test_special_form_alpha_beta "
				"not normalized" << endl;
		exit(1);
	}
	if (!int_vec_is_constant_on_subset(coeff,
		alphas, sizeof(alphas) / sizeof(int), a)) {
		cout << "surface_domain::test_HCV_form_alpha_beta "
				"not constant on alpha set" << endl;
		return FALSE;
	}
	alpha = a;
	if (!int_vec_is_constant_on_subset(coeff,
		betas, sizeof(betas) / sizeof(int), a)) {
		cout << "surface_domain::test_HCV_form_alpha_beta "
				"not constant on beta set" << endl;
		return FALSE;
	}
	beta = a;

	if (f_v) {
		cout << "surface_domain::test_HCV_form_alpha_beta done" << endl;
	}
	return ret;
}

void surface_domain::create_HCV_double_six(long int *double_six,
	int a, int b, int verbose_level)
// create double-six for the Hilbert, Cohn-Vossen surface
{
	int f_v = (verbose_level >= 1);
	int Basis[12 * 8] = {
		1,2,0,0,0,0,1,6,
		1,3,0,0,0,0,1,7,
		1,0,5,0,0,1,0,7,
		1,0,4,0,0,1,0,6,
		1,0,0,7,0,1,3,0,
		1,0,0,6,0,1,2,0,
		1,5,0,0,0,0,1,7,
		1,4,0,0,0,0,1,6,
		1,0,2,0,0,1,0,6,
		1,0,3,0,0,1,0,7,
		1,0,0,6,0,1,4,0,
		1,0,0,7,0,1,5,0
	};
	int i, c, ma, mb, av, mav;

	if (f_v) {
		cout << "surface_domain::create_HCV_double_six "
				"a=" << a << " b=" << b << endl;
	}
	ma = F->negate(a);
	mb = F->negate(b);
	av = F->inverse(a);
	mav = F->negate(av);
	for (i = 0; i < 12 * 8; i++) {
		c = Basis[i];
		if (c == 2) {
			c = a;
		}
		else if (c == 3) {
			c = ma;
		}
		else if (c == 4) {
			c = av;
		}
		else if (c == 5) {
			c = mav;
		}
		else if (c == 6) {
			c = b;
		}
		else if (c == 7) {
			c = mb;
		}
		Basis[i] = c;
	}
	for (i = 0; i < 12; i++) {
		double_six[i] = Gr->rank_lint_here(Basis + i * 8,
				0 /* verbose_level */);
	}
	if (f_v) {
		cout << "surface_domain::create_HCV_double_six done" << endl;
	}
}

void surface_domain::create_HCV_fifteen_lines(long int *fifteen_lines,
	int a, int b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis[15 * 8] = {
		1,0,0,0,0,1,0,0, // 0 = c_12
		0,1,-1,0,2,3,0,-1, // 1 = c_13
		0,-1,-1,0,-2,-3,0,-1, // 2 = c_14
		1,0,-1,0,-3,2,0,-1, // 3 = c_15
		-1,0,-1,0,3,-2,0,-1, // 4 = c_16
		0,-1,-1,0,-2,3,0,-1, // 5 = c_23
		0,1,-1,0,2,-3,0,-1, // 6 = c_24
		-1,0,-1,0,-3,-2,0,-1, // 7 = c_25
		1,0,-1,0,3,2,0,-1, // 8 = c_26
		1,0,0,0,0,0,1,0, // 9 = c_34
		-1,0,4,5,0,-1,-4,-5, // 10 = c_35
		-1,0,4,-5,0,-1,4,-5, // 11 = c_36
		-1,0,-4,5,0,-1,-4,5, // 12 = c_45
		-1,0,-4,-5,0,-1,4,5, // 13 = c_46
		0,1,0,0,0,0,1,0 // 14 = c_56
	};
	int i, m1, a2, a2p1, a2m1, ba2p1, /*ba2m1,*/ twoa;
	int c, c2, cm2, c3, cm3, c4, cm4, c5, cm5;

	// 2 stands for (2a)/(b(a^2+1))
	// -2 stands for -(2a)/(b(b^2+1))
	// 3 stands for (a^2-1)/(b(a^2+1))
	// -3 stands for -(a^2-1)/(b(a^2+1))
	// 4 stands for (2a)/(a^2-1)
	// -4 stands for -(2a)/(a^2-1)
	// 5 stands for 3 inverse
	// -5 stands for -3 inverse

	if (f_v) {
		cout << "surface_domain::create_HCV_fifteen_lines "
				"a=" << a << " b=" << b << endl;
	}
	m1 = F->negate(1);
	a2 = F->mult(a, a);
	a2p1 = F->add(a2, 1);
	a2m1 = F->add(a2, m1);
	twoa = F->add(a, a);
	ba2p1 = F->mult(b, a2p1);
	//ba2m1 = F->mult(b, a2m1);

	if (ba2p1 == 0) {
		cout << "surface_domain::create_HCV_fifteen_lines "
				"ba2p1 = 0, cannot invert" << endl;
		exit(1);
	}
	c2 = F->mult(twoa, F->inverse(ba2p1));
	cm2 = F->negate(c2);
	c3 = F->mult(a2m1, F->inverse(ba2p1));
	cm3 = F->negate(c3);
	if (a2m1 == 0) {
		cout << "surface_domain::create_HCV_fifteen_lines "
				"a2m1 = 0, cannot invert" << endl;
		exit(1);
	}
	c4 = F->mult(twoa, F->inverse(a2m1));
	cm4 = F->negate(c4);

	if (c3 == 0) {
		cout << "surface_domain::create_HCV_fifteen_lines "
				"c3 = 0, cannot invert" << endl;
		exit(1);
	}
	c5 = F->inverse(c3);
	if (cm3 == 0) {
		cout << "surface_domain::create_HCV_fifteen_lines "
				"cm3 = 0, cannot invert" << endl;
		exit(1);
	}
	cm5 = F->inverse(cm3);


	for (i = 0; i < 15 * 8; i++) {
		c = Basis[i];
		if (c == 0) {
			c = 0;
		}
		else if (c == 1) {
			c = 1;
		}
		else if (c == -1) {
			c = m1;
		}
		else if (c == 2) {
			c = c2;
		}
		else if (c == -2) {
			c = cm2;
		}
		else if (c == 3) {
			c = c3;
		}
		else if (c == -3) {
			c = cm3;
		}
		else if (c == 4) {
			c = c4;
		}
		else if (c == -4) {
			c = cm4;
		}
		else if (c == 5) {
			c = c5;
		}
		else if (c == -5) {
			c = cm5;
		}
		else {
			cout << "surface_domain::create_HCV_fifteen_lines "
					"unknown value" << c << endl;
			exit(1);
		}
		Basis[i] = c;
	}
	for (i = 0; i < 15; i++) {
		fifteen_lines[i] = Gr->rank_lint_here(
			Basis + i * 8, 0 /* verbose_level */);
	}
	if (f_v) {
		cout << "surface_domain::create_HCV_fifteen_lines done" << endl;
	}
}


void surface_domain::create_surface_family_HCV(int a,
	long int *Lines27,
	int *equation20, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::create_surface_family_HCV" << endl;
	}

	int nb_E = 0;
	int b = 1;
	int alpha, beta;

	if (f_v) {
		cout << "surface_domain::create_surface_family_HCV "
				"before create_surface_HCV for a=" << a << ":" << endl;
	}

	create_surface_HCV(a, b,
		equation20,
		Lines27,
		alpha, beta, nb_E,
		0 /* verbose_level */);

	if (f_v) {
		cout << "surface_domain::create_surface_family_HCV "
				"The double six is:" << endl;
		lint_matrix_print(Lines27, 2, 6);
		cout << "The lines are : ";
		lint_vec_print(cout, Lines27, 27);
		cout << endl;
	}

	if (f_v) {
		cout << "surface_domain::create_surface_family_HCV "
				"done" << endl;
	}
}
int surface_domain::create_surface_HCV(int a, int b,
	int *coeff20,
	long int *Lines27,
	int &alpha, int &beta, int &nb_E,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int alpha0, beta0;
	//int line_rk;
	//int Basis[8];
	//int Lines[27];
	int nb, i, e, ee, nb_lines, rk, nb_pts;
	//int *coeff;
	long int *Pts;
	int v[4];
	geometry_global Gg;
	sorting Sorting;

	if (f_v) {
		cout << "surface_domain::create_surface_HCV" << endl;
	}
	alpha = -1;
	beta = -1;
	nb_E = -1;

	int a2, a2p1, a2m1;

	a2 = F->mult(a, a);
	a2p1 = F->add(a2, 1);
	a2m1 = F->add(a2, F->negate(1));
	if (a2p1 == 0 || a2m1 == 0) {
		cout << "surface_domain::create_surface_HCV "
				"a2p1 == 0 || a2m1 == 0" << endl;
		return FALSE;
	}


	Pts = NEW_lint(Gg.nb_PG_elements(3, F->q));

	//coeff = NEW_int(20);
	alpha0 = F->negate(F->mult(b, b));
	beta0 = F->mult(F->mult(F->power(b, 3),
		F->add(1, F->mult(a, a))), F->inverse(a));
	if (f_v) {
		cout << "surface_domain::create_surface_HCV a="
			<< a << " b=" << b << " alpha0=" << alpha0
			<< " beta0=" << beta0 << endl;
	}

#if 0
	int_vec_zero(Basis, 8);
	Basis[0 * 4 + 0] = 1;
	Basis[0 * 4 + 1] = a;
	Basis[1 * 4 + 2] = 1;
	Basis[1 * 4 + 3] = b;
	line_rk = Gr->rank_int_here(Basis, 0);
#endif


#if 0
	//int_vec_copy(desired_lines, Lines, 3);
	//nb = 3;

	cout << "The triangle lines are:" << endl;
	Gr->print_set(desired_lines, 3);
#endif


	long int *Oab;

	Oab = NEW_lint(12);
	create_HCV_double_six(Oab, a, b, 0 /* verbose_level */);

#if 0
	if (!test_if_sets_are_equal(Oab, Lines, 12)) {
		cout << "the sets are not equal" << endl;
		exit(1);
	}
#endif

	if (f_v) {
		cout << "surface_domain::create_surface_HCV The double six is:" << endl;
		Gr->print_set(Oab, 12);
	}


	lint_vec_copy(Oab, Lines27, 12);
	FREE_lint(Oab);


	nb = 12;

	if (f_v) {
		cout << "surface_domain::create_surface_HCV We have a set of "
				"lines of size " << nb << ":";
		lint_vec_print(cout, Lines27, nb);
		cout << endl;
	}

	create_remaining_fifteen_lines(Lines27,
		Lines27 + 12, 0 /* verbose_level */);

	if (f_v) {
		cout << "surface_domain::create_surface_HCV The remaining 15 lines are:";
		lint_vec_print(cout, Lines27 + 12, 15);
		cout << endl;
		Gr->print_set(Lines27 + 12, 15);
	}


	if (f_v) {
		cout << "surface_domain::create_surface_HCV before create_HCV_"
				"fifteen_lines" << endl;
	}

	long int special_lines[15];

	create_HCV_fifteen_lines(special_lines, a, b, verbose_level);
	for (i = 0; i < 15; i++) {
		if (special_lines[i] != Lines27[12 + i]) {
			cout << "surface_domain::create_surface_HCV something is wrong "
					"with the special line " << i << " / 15 " << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "surface_domain::create_surface_HCV after create_special_"
				"fifteen_lines" << endl;
	}

	rk = compute_system_in_RREF(27, Lines27, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::create_surface_HCV a=" << a
			<< " b=" << b << " rk=" << rk << endl;
	}

	if (rk != 19) {
		cout << "surface_domain::create_surface_HCV rk != 19" << endl;
		FREE_lint(Pts);
		//FREE_int(coeff);
		exit(1);
	}
	build_cubic_surface_from_lines(27, Lines27, coeff20,
			0 /* verbose_level */);
	F->PG_element_normalize_from_front(coeff20, 1, 20);



	enumerate_points(coeff20, Pts, nb_pts, 0 /* verbose_level */);
	Sorting.lint_vec_heapsort(Pts, nb_pts);


	if (f_v) {
		cout << "surface_domain::create_surface_HCV "
				"a=" << a << " b=" << b << " equation: ";
		print_equation(cout, coeff20);
		cout << endl;
	}

	if (nb_pts != nb_pts_on_surface) {
		cout << "surface_domain::create_surface_HCV degenerate surface" << endl;
		cout << "nb_pts=" << nb_pts << endl;
		cout << "should be =" << nb_pts_on_surface << endl;
		alpha = -1;
		beta = -1;
		nb_E = -1;
		return FALSE;
	}

	if (f_v) {
		cout << "surface_domain::create_surface_HCV Pts: " << endl;
		lint_vec_print_as_table(cout, Pts, nb_pts, 10);
	}


	int *Adj;
	int *Intersection_pt;
	int *Intersection_pt_idx;

	compute_adjacency_matrix_of_line_intersection_graph(
		Adj, Lines27, 27, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_HCV "
				"The adjacency matrix is:" << endl;
		int_matrix_print(Adj, 27, 27);
	}



	compute_intersection_points_and_indices(
		Adj, Pts, nb_pts, Lines27, 27,
		Intersection_pt, Intersection_pt_idx,
		verbose_level);

	if (f_v) {
		cout << "surface_domain::create_surface_HCV "
				"The intersection points are:" << endl;
		int_matrix_print(Intersection_pt_idx, 27, 27);
	}


	tally C;

	C.init(Intersection_pt_idx, 27 * 27, FALSE, 0);
	if (f_v) {
		cout << "surface_domain::create_surface_HCV "
				"classification of points by multiplicity:" << endl;
		C.print_naked(TRUE);
		cout << endl;
	}




	if (!test_HCV_form_alpha_beta(coeff20, alpha, beta,
		0 /* verbose_level */)) {
		cout << "surface_domain::create_surface_HCV not of special form" << endl;
		exit(1);
	}


	if (alpha != alpha0) {
		cout << "surface_domain::create_surface_HCV alpha != alpha0" << endl;
		exit(1);
	}
	if (beta != beta0) {
		cout << "surface_domain::create_surface_HCV beta != beta0" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "surface_domain::create_surface_HCV "
				"determining all lines on the surface:" << endl;
	}
	{
		long int Lines2[27];
		P->find_lines_which_are_contained(Pts, nb_pts,
			Lines2, nb_lines, 27 /* max_lines */,
			0 /* verbose_level */);
	}

	if (f_v) {
		cout << "surface_domain::create_surface_HCV "
				"nb_lines = " << nb_lines << endl;
	}
	if (nb_lines != 27) {
		cout << "surface_domain::create_surface_HCV "
				"nb_lines != 27, something is wrong "
				"with the surface" << endl;
		exit(1);
	}
	set_of_sets *pts_on_lines;
	set_of_sets *lines_on_pt;

	compute_points_on_lines(Pts, nb_pts,
		Lines27, nb_lines,
		pts_on_lines,
		verbose_level);


	if (f_v) {
		cout << "surface_domain::create_surface_HCV pts_on_lines: " << endl;
		pts_on_lines->print_table();
	}

	int *E;

	pts_on_lines->get_eckardt_points(E, nb_E, 0 /* verbose_level */);
	//nb_E = pts_on_lines->number_of_eckardt_points(verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_HCV The surface contains "
			<< nb_E << " Eckardt points" << endl;
	}

#if 0
	if (a == 2 && b == 1) {
		exit(1);
	}
#endif


	pts_on_lines->dualize(lines_on_pt, 0 /* verbose_level */);

#if 0
	cout << "lines_on_pt: " << endl;
	lines_on_pt->print_table();
#endif

	if (f_v) {
		cout << "surface_domain::create_surface_HCV "
				"The Eckardt points are:" << endl;
		for (i = 0; i < nb_E; i++) {
			e = E[i];
			ee = Pts[e];
			unrank_point(v, ee);
			cout << i << " : " << ee << " : ";
			int_vec_print(cout, v, 4);
			cout << " on lines: ";
			lint_vec_print(cout, lines_on_pt->Sets[e],
				lines_on_pt->Set_size[e]);
			cout << endl;
		}
	}


	FREE_int(E);
	//FREE_int(coeff);
	FREE_lint(Pts);
	FREE_int(Intersection_pt);
	FREE_int(Intersection_pt_idx);
	FREE_OBJECT(pts_on_lines);
	FREE_OBJECT(lines_on_pt);
	if (f_v) {
		cout << "surface_domain::create_surface_HCV done" << endl;
	}
	return TRUE;
}


}}
