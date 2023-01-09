/*
 * surface_domain_families.cpp
 *
 *  Created on: Jul 22, 2020
 *      Author: betten
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {


void surface_domain::create_equation_general_abcd(int a, int b, int c, int d,
		int *coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surface_domain::create_equation_general_abcd" << endl;
	}


	int m1 = F->negate(1);
	int two = F->add(1, 1);
	int four = F->add(two, two);

	int amc = F->add(a, F->negate(c));
	int bmd = F->add(b, F->negate(d));

	int abc = F->mult3(a, b, c);
	int abd = F->mult3(a, b, d);
	int mabd = F->mult4(m1, a, b, d);
	int macd = F->mult4(m1, a, c, d);
	int bcd = F->mult3(b, c, d);
	int admcb = F->add(F->mult(a, d), F->mult3(m1, c, b));
	int A = F->add5(abc, mabd, macd, bcd, admcb);

	int aac = F->mult3(a, a, c);
	int aad = F->mult3(a, a, d);
	int acc = F->mult3(a, c, c);
	int bcc = F->mult3(b, c, c);
	int B = F->add4(aac, bcc, admcb, F->negate(F->add(aad, acc)));

	int aacd = F->mult4(a, a, c, d);
	int abcc = F->mult4(a, b, c, c);

	int C = F->add4(aacd, abd, bcc, F->negate(F->add3(abcc, aad, bcd)));

	int D = F->add4(admcb, b, c, F->negate(F->add(a, d)));

	int two_aabcd = F->mult6(two, a, a, b, c, d);
	int abbcd = F->mult5(a, b, b, c, d);
	int two_abccd = F->mult6(two, a, b, c, c, d);
	int abcdd = F->mult5(a, b, c, d, d);

	int aadd = F->mult4(a, a, d, d);
	int abbc = F->mult4(a, b, b, c);
	int acdd = F->mult4(a, c, d, d);
	int bbcc = F->mult4(b, b, c, c);

	int aabdd = F->mult5(a, a, b, d, d);
	int two_aacdd = F->mult6(two, a, a, c, d, d);
	int two_abbcc = F->mult6(two, a, b, b, c, c);
	int bbccd = F->mult5(b, b, c, c, d);
	int aabc = F->mult4(a, a, b, c);
	int four_abcd = F->mult5(four, a, b, c, d);
	int accd = F->mult4(a, c, c, d);

	int E_plus1 = F->add5(two_aabcd, abbcd, two_abccd, abcdd, aacd);
	int E_plus2 = F->add5(aadd, abbc, abcc, acdd, bbcc);
	int E_plus = F->add(E_plus1, E_plus2);

	//cout << "E_plus1 = " << E_plus1 << endl;
	//cout << "E_plus2 = " << E_plus2 << endl;
	//cout << "E_plus = " << E_plus << endl;

	int E_minus1 = F->add4(aabdd, two_aacdd, two_abbcc, bbccd);
	int E_minus2 = F->add3(aabc, four_abcd, accd);
	int E_minus = F->add(E_minus1, E_minus2);

	int E = F->add(E_plus, F->negate(E_minus));

	Int_vec_zero(coeff, PolynomialDomains->nb_monomials);

	coeff[5] = F->mult3(m1, A, bmd);
	coeff[16] = F->mult(A, F->add4(a, b, F->negate(c), F->negate(d)));
	coeff[17] = F->mult(B, bmd);
	coeff[10] = F->mult3(m1, A, admcb);
	coeff[18] = F->mult3(m1, C, bmd);
	coeff[8] = coeff[9] = F->mult3(m1, amc, A);
	coeff[11] = F->mult(A, admcb);
	coeff[14] = F->mult4(c, a, D, bmd);
	coeff[19] = E;



	if (f_v) {
		cout << "surface_domain::create_equation_general_abcd done" << endl;
	}
}

void surface_domain::create_equation_Cayley_klmn(int k, int l, int m, int n,
		int *coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surface_domain::create_equation_Cayley_klmn" << endl;
	}
	Int_vec_zero(coeff, PolynomialDomains->nb_monomials);

	coeff[6] = coeff[9] = coeff[12] = coeff[3] = 1;

	coeff[16] = k;

	int mn, nl, lm;

	mn = F->mult(m, n);
	nl = F->mult(n, l);
	lm = F->mult(l, m);

	coeff[19] = F->add(mn, F->inverse(mn));
	coeff[18] = F->add(nl, F->inverse(nl));
	coeff[17] = F->add(lm, F->inverse(lm));
	coeff[13] = F->add(l, F->inverse(l));
	coeff[14] = F->add(m, F->inverse(m));
	coeff[15] = F->add(n, F->inverse(n));

	if (f_v) {
		cout << "surface_domain::create_equation_Cayley_klmn done" << endl;
	}
}


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

	Int_vec_zero(coeff, PolynomialDomains->nb_monomials);

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
	Int_vec_zero(coeff, PolynomialDomains->nb_monomials);

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
	Int_vec_zero(coeff, PolynomialDomains->nb_monomials);

	coeff[5] = coeff[8] = coeff[9] = coeff[10] = coeff[11] = coeff[12] = 1;
	coeff[14] = coeff[15] = b;
	coeff[18] = coeff[19] = c;
	if (f_v) {
		cout << "surface_domain::create_equation_G13 done" << endl;
	}
}

surface_object *surface_domain::create_surface_general_abcd(int a, int b, int c, int d,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::create_surface_general_abcd" << endl;
	}
	int coeff20[20];

	if (f_v) {
		cout << "surface_domain::create_surface_general_abcd "
				"before create_equation_general_abcd" << endl;
	}
	create_equation_general_abcd(a, b, c, d, coeff20, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_general_abcd "
				"after create_equation_general_abcd" << endl;
	}

	surface_object *SO;

	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_domain::create_surface_general_abcd "
				"before SO->init_equation" << endl;
	}
	SO->init_equation(this, coeff20, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_general_abcd "
				"after SO->init_equation" << endl;
	}


	if (f_v) {
		cout << "surface_domain::create_surface_general_abcd done" << endl;
	}
	return SO;
}



surface_object *surface_domain::create_surface_bes(int a, int c,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::create_surface_bes" << endl;
	}
	int coeff20[20];

	if (f_v) {
		cout << "surface_domain::create_surface_bes "
				"before create_equation_bes" << endl;
	}
	create_equation_bes(a, c, coeff20, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_bes "
				"after create_equation_bes" << endl;
	}

	surface_object *SO;

	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_domain::create_surface_bes "
				"before SO->init_equation" << endl;
	}
	SO->init_equation(this, coeff20, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_bes "
				"after SO->init_equation" << endl;
	}

	if (f_v) {
		cout << "surface_domain::create_surface_bes done" << endl;
	}
	return SO;
}


surface_object *surface_domain::create_surface_F13(int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::create_surface_F13" << endl;
	}
	int coeff20[20];

	if (f_v) {
		cout << "surface_domain::create_surface_F13 "
				"before create_equation_F13" << endl;
	}
	create_equation_F13(a, coeff20, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_F13 "
				"after create_equation_F13" << endl;
	}

	surface_object *SO;

	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_domain::create_surface_F13 "
				"before SO->init_equation" << endl;
	}
	SO->init_equation(this, coeff20, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_F13 "
				"after SO->init_equation" << endl;
	}

	if (f_v) {
		cout << "surface_domain::create_surface_F13 done" << endl;
	}
	return SO;
}

surface_object *surface_domain::create_surface_G13(int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::create_surface_G13" << endl;
	}
	int coeff20[20];

	if (f_v) {
		cout << "surface_domain::create_surface_G13 "
				"before create_equation_G13" << endl;
	}
	create_equation_G13(a, coeff20, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_G13 "
				"after create_equation_G13" << endl;
	}

	surface_object *SO;

	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_domain::create_surface_G13 "
				"before SO->init_equation" << endl;
	}
	SO->init_equation(this, coeff20, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_G13 "
				"after SO->init_equation" << endl;
	}

	if (f_v) {
		cout << "surface_domain::create_surface_G13 done" << endl;
	}
	return SO;
}

surface_object *surface_domain::create_Eckardt_surface(int a, int b,
	int &alpha, int &beta,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int alpha0, beta0;
	long int Lines27[27];
	int i, rk, nb;
	geometry::geometry_global Gg;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "surface_domain::create_Eckardt_surface" << endl;
	}
	alpha = -1;
	beta = -1;
	//nb_E = -1;

	int a2, a2p1, a2m1;

	a2 = F->mult(a, a);
	a2p1 = F->add(a2, 1);
	a2m1 = F->add(a2, F->negate(1));
	if (a2p1 == 0 || a2m1 == 0) {
		cout << "surface_domain::create_Eckardt_surface "
				"a2p1 == 0 || a2m1 == 0" << endl;
		return FALSE;
	}


	//Pts = NEW_lint(Gg.nb_PG_elements(3, F->q));

	//coeff = NEW_int(20);
	alpha0 = F->negate(F->mult(b, b));
	beta0 = F->mult(F->mult(F->power(b, 3),
		F->add(1, F->mult(a, a))), F->inverse(a));
	if (f_v) {
		cout << "surface_domain::create_Eckardt_surface a="
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
	create_Eckardt_double_six(Oab, a, b, 0 /* verbose_level */);

#if 0
	if (!test_if_sets_are_equal(Oab, Lines, 12)) {
		cout << "the sets are not equal" << endl;
		exit(1);
	}
#endif

	if (f_v) {
		cout << "surface_domain::create_Eckardt_surface "
				"The double six is:" << endl;
		Gr->print_set(Oab, 12);
	}


	Lint_vec_copy(Oab, Lines27, 12);
	FREE_lint(Oab);


	nb = 12;

	if (f_v) {
		cout << "surface_domain::create_Eckardt_surface "
				"We have a set of "
				"lines of size " << nb << ":";
		Lint_vec_print(cout, Lines27, nb);
		cout << endl;
	}

	create_remaining_fifteen_lines(Lines27,
		Lines27 + 12, 0 /* verbose_level */);

	if (f_v) {
		cout << "surface_domain::create_Eckardt_surface "
				"The remaining 15 lines are:";
		Lint_vec_print(cout, Lines27 + 12, 15);
		cout << endl;
		Gr->print_set(Lines27 + 12, 15);
	}


	if (f_v) {
		cout << "surface_domain::create_Eckardt_surface "
				"before create_HCV_fifteen_lines" << endl;
	}

	long int special_lines[15];

	create_Eckardt_fifteen_lines(special_lines, a, b, verbose_level);
	for (i = 0; i < 15; i++) {
		if (special_lines[i] != Lines27[12 + i]) {
			cout << "surface_domain::create_Eckardt_surface something is wrong "
					"with the special line " << i << " / 15 " << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "surface_domain::create_Eckardt_surface "
				"after create_special_fifteen_lines" << endl;
	}

	if (f_v) {
		cout << "surface_domain::create_Eckardt_surface "
				"before rank_of_system" << endl;
	}
	rk = rank_of_system(27, Lines27, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::create_Eckardt_surface "
				"a=" << a << " b=" << b << " rk=" << rk << endl;
	}

	if (rk != 19) {
		cout << "surface_domain::create_Eckardt_surface rk != 19" << endl;
		exit(1);
	}

	int coeff20[20];

	build_cubic_surface_from_lines(27, Lines27, coeff20,
			0 /* verbose_level */);
	F->PG_element_normalize_from_front(coeff20, 1, 20);





	if (!test_Eckardt_form_alpha_beta(coeff20, alpha, beta,
		0 /* verbose_level */)) {
		cout << "surface_domain::create_Eckardt_surface "
				"not of special form" << endl;
		exit(1);
	}


	if (alpha != alpha0) {
		cout << "surface_domain::create_Eckardt_surface "
				"alpha != alpha0" << endl;
		exit(1);
	}
	if (beta != beta0) {
		cout << "surface_domain::create_Eckardt_surface "
				"beta != beta0" << endl;
		exit(1);
	}



	surface_object *SO;

	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_domain::create_Eckardt_surface "
				"before SO->init_with_27_lines" << endl;
	}
	SO->init_with_27_lines(this,
		Lines27, coeff20,
		FALSE /* f_find_double_six_and_rearrange_lines */,
		verbose_level);
	if (f_v) {
		cout << "surface_domain::create_Eckardt_surface "
				"after SO->init_with_27_lines" << endl;
	}

	if (f_v) {
		cout << "surface_domain::create_Eckardt_surface done" << endl;
	}
	return SO;
}


void surface_domain::create_equation_Eckardt_surface(int a, int b,
		int *coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int alpha, beta;

	if (f_v) {
		cout << "surface_domain::create_equation_Eckardt_surface" << endl;
	}
	alpha = F->negate(F->mult(b, b));
	beta = F->mult(F->mult(F->power(b, 3),
		F->add(1, F->mult(a, a))), F->inverse(a));
	Int_vec_zero(coeff, PolynomialDomains->nb_monomials);

	coeff[3] = 1;
	coeff[6] = alpha;
	coeff[9] = alpha;
	coeff[12] = alpha;
	coeff[16] = beta;
	//coeff[19] = beta;
	if (f_v) {
		cout << "surface_domain::create_equation_Eckardt_surface done" << endl;
	}
}

int surface_domain::test_Eckardt_form_alpha_beta(int *coeff,
	int &alpha, int &beta, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = TRUE;
	int zeroes[] = {0,1,2,4,5,7,8,10,11,13,14,15,17,18,19};
	int alphas[] = {6,9,12};
	int betas[] = {16};
	int a;

	if (f_v) {
		cout << "surface_domain::test_Eckardt_form_alpha_beta" << endl;
	}
	if (!orbiter_kernel_system::Orbiter->Int_vec->is_constant_on_subset(coeff,
		zeroes, sizeof(zeroes) / sizeof(int), a)) {
		cout << "surface_domain::test_Eckardt_form_alpha_beta "
				"not constant on zero set" << endl;
		return FALSE;
	}
	if (a != 0) {
		cout << "surface_domain::test_Eckardt_form_alpha_beta "
				"not zero on zero set" << endl;
		return FALSE;
	}
	if (coeff[3] != 1) {
		cout << "surface_domain::test_Eckardt_form_alpha_beta "
				"not normalized" << endl;
		exit(1);
	}
	if (!orbiter_kernel_system::Orbiter->Int_vec->is_constant_on_subset(coeff,
		alphas, sizeof(alphas) / sizeof(int), a)) {
		cout << "surface_domain::test_Eckardt_form_alpha_beta "
				"not constant on alpha set" << endl;
		return FALSE;
	}
	alpha = a;
	if (!orbiter_kernel_system::Orbiter->Int_vec->is_constant_on_subset(coeff,
		betas, sizeof(betas) / sizeof(int), a)) {
		cout << "surface_domain::test_Eckardt_form_alpha_beta "
				"not constant on beta set" << endl;
		return FALSE;
	}
	beta = a;

	if (f_v) {
		cout << "surface_domain::test_Eckardt_form_alpha_beta done" << endl;
	}
	return ret;
}

void surface_domain::create_Eckardt_double_six(long int *double_six,
	int a, int b, int verbose_level)
// create double-six for the Eckardt surface
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
		cout << "surface_domain::create_Eckardt_double_six "
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
		cout << "surface_domain::create_Eckardt_double_six done" << endl;
	}
}

void surface_domain::create_Eckardt_fifteen_lines(long int *fifteen_lines,
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
		cout << "surface_domain::create_Eckardt_fifteen_lines "
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
		cout << "surface_domain::create_Eckardt_fifteen_lines "
				"ba2p1 = 0, cannot invert" << endl;
		exit(1);
	}
	c2 = F->mult(twoa, F->inverse(ba2p1));
	cm2 = F->negate(c2);
	c3 = F->mult(a2m1, F->inverse(ba2p1));
	cm3 = F->negate(c3);
	if (a2m1 == 0) {
		cout << "surface_domain::create_Eckardt_fifteen_lines "
				"a2m1 = 0, cannot invert" << endl;
		exit(1);
	}
	c4 = F->mult(twoa, F->inverse(a2m1));
	cm4 = F->negate(c4);

	if (c3 == 0) {
		cout << "surface_domain::create_Eckardt_fifteen_lines "
				"c3 = 0, cannot invert" << endl;
		exit(1);
	}
	c5 = F->inverse(c3);
	if (cm3 == 0) {
		cout << "surface_domain::create_Eckardt_fifteen_lines "
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
			cout << "surface_domain::create_Eckardt_fifteen_lines "
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
		cout << "surface_domain::create_Eckardt_fifteen_lines done" << endl;
	}
}





}}}

