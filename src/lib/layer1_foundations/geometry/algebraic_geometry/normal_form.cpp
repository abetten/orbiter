/*
 * normal_form.cpp
 *
 *  Created on: Oct 7, 2025
 *      Author: betten
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace algebraic_geometry {



normal_form::normal_form()
{
	F = NULL;
	//Gr = NULL;

	a = b = c = d = 0;

	m1 = 0;
	two = 0;
	four = 0;

	s1 = 0;
	s2 = 0;
	s3 = 0;
	s4 = 0;
	s5 = 0;
	s6 = 0;
	s7 = 0;
	s8 = 0;
	ab = 0;
	ad = 0;
	bc = 0;
	cd = 0;
	abc = 0;
	abd = 0;
	acd = 0;
	bcd = 0;
	delta = 0;
	epsilon = 0;
	gamma = 0;
	lambda = 0;
	nu = 0;
	zeta = 0;
	xi = 0;
	theta = 0;


	c002 = 0;
	c012 = 0;
	c013 = 0;
	c022 = 0;
	c023 = 0;
	c112 = 0;
	c113 = 0;
	c122 = 0;
	c133 = 0;
	c123 = 0;

	//int a1[8];
	//int a2[8];
	//int a3[8];
	//int a4[8];
	//int a5[8];
	//int a6[8];

	//int b1[8];
	//int b2[8];
	//int b3[8];
	//int b4[8];
	//int b5[8];
	//int b6[8];

	//long int Double_six[12];


}

normal_form::~normal_form()
{

}

void normal_form::init(
		algebra::field_theory::finite_field *F,
		//geometry::projective_geometry::grassmann *Gr,
		int a, int b, int c, int d,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "normal_form::init" << endl;
	}

	normal_form::F = F;
	//normal_form::Gr = Gr;

	normal_form::a = a;
	normal_form::b = b;
	normal_form::c = c;
	normal_form::d = d;


	if (f_v) {
		cout << "normal_form::init before init_constants" << endl;
	}
	init_constants(verbose_level);
	if (f_v) {
		cout << "normal_form::init after init_constants" << endl;
	}

	if (f_v) {
		cout << "normal_form::init "
				"before create_coefficients_for_cubic_surface" << endl;
	}

	create_coefficients_for_cubic_surface(
			verbose_level);

	if (f_v) {
		cout << "normal_form::init "
				"after create_coefficients_for_cubic_surface" << endl;
	}

	if (f_v) {
		cout << "normal_form::init "
				"before create_lines_of_cubic_surface" << endl;
	}

	create_lines_of_cubic_surface(verbose_level);

	if (f_v) {
		cout << "normal_form::init "
				"after create_lines_of_cubic_surface" << endl;
	}



	if (f_v) {
		cout << "normal_form::init done" << endl;
	}
}

void normal_form::init_constants(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "normal_form::init_constants" << endl;
	}

	m1 = F->negate(1);
	two = F->add(1, 1);
	four = F->add(two, two);

	s1 = F->add(a, m1);
	s2 = F->add(b, m1);
	s3 = F->add(c, m1);
	s4 = F->add(d, m1);
	if (f_v) {
		cout << "normal_form::init_constants "
				"before s5" << endl;
	}
	s5 = F->add(a, F->negate(b));
	s6 = F->add(a, F->negate(c));
	s7 = F->add(b, F->negate(d));
	s8 = F->add(c, F->negate(d));
	ab = F->mult(a, b);
	ad = F->mult(a, d);
	bc = F->mult(b, c);
	cd = F->mult(c, d);
	if (f_v) {
		cout << "normal_form::init_constants "
				"before abc" << endl;
	}
	abc = F->mult3(a, b, c);
	abd = F->mult3(a, b, d);
	acd = F->mult3(a, c, d);
	bcd = F->mult3(b, c, d);
	if (f_v) {
		cout << "normal_form::init_constants "
				"before delta" << endl;
	}
	delta = F->add(ad, F->negate(bc));
	epsilon = F->add5(abc, F->negate(abd), F->negate(acd), bcd, delta);
	gamma = F->add5(delta, F->negate(a), b, c, F->negate(d));


	lambda = F->add3(F->mult3(b, b, s8), F->negate(F->mult3(d, d, s5)), delta);
	// !!! mistake fixed A. Betten 8/24/2025
	// it was s7, but it should be s5

	//int mu = F->add3(F->negate(abd), bcd, delta);
	if (f_v) {
		cout << "normal_form::init_constants "
				"before nu" << endl;
	}
	nu = F->add(F->mult3(a, c, s7), F->negate(F->mult3(b, d, s6)));
	//int eta = F->add6(F->negate(F->mult(a, acd)), F->mult(abc, c), F->mult(a, ad), F->negate(abd), F->negate(F->mult(bc, c)), bcd);
	zeta = F->mult3(s1, s3, s7);
	if (f_v) {
		cout << "normal_form::init_constants "
				"before xi" << endl;
	}
	xi = F->add5(F->mult3(a, a, c), F->negate(F->mult(a, ad)), F->negate(F->mult3(a, c, c)), F->mult3(b, c, c), delta);
	theta = F->add6(abc, F->negate(acd), F->negate(ab), cd, a, F->negate(c));

	if (f_v) {
		cout << "normal_form::init_constants done" << endl;
	}

}


void normal_form::create_coefficients_for_cubic_surface(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "normal_form::create_coefficients_for_cubic_surface" << endl;
	}

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

	c002 = F->mult3(m1, A, bmd);
	c012 = F->mult(A, F->add4(a, b, F->negate(c), F->negate(d)));
	c013 = F->mult(B, bmd);
	c022 = F->mult3(m1, A, admcb);
	c023 = F->mult3(m1, C, bmd);
	c112 = c113 = F->mult3(m1, amc, A);
	c122 = F->mult(A, admcb);
	c133 = F->mult4(c, a, D, bmd);
	c123 = E;
	if (f_v) {
		cout << "normal_form::create_coefficients_for_cubic_surface c002=" << c002 << endl;
		cout << "normal_form::create_coefficients_for_cubic_surface c012=" << c012 << endl;
		cout << "normal_form::create_coefficients_for_cubic_surface c013=" << c013 << endl;
		cout << "normal_form::create_coefficients_for_cubic_surface c022=" << c022 << endl;
		cout << "normal_form::create_coefficients_for_cubic_surface c023=" << c023 << endl;
		cout << "normal_form::create_coefficients_for_cubic_surface c112=" << c112 << endl;
		cout << "normal_form::create_coefficients_for_cubic_surface c113=" << c113 << endl;
		cout << "normal_form::create_coefficients_for_cubic_surface c122=" << c122 << endl;
		cout << "normal_form::create_coefficients_for_cubic_surface c133=" << c133 << endl;
		cout << "normal_form::create_coefficients_for_cubic_surface c123=" << c123 << endl;
	}


	if (f_v) {
		cout << "normal_form::create_coefficients_for_cubic_surface done" << endl;
	}
}


void normal_form::create_lines_of_cubic_surface(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface" << endl;
	}


	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing a1" << endl;
	}

	a1[0] = F->mult3(s6, lambda, F->inverse(F->mult(s7, epsilon)));
	a1[1] = 1;
	a1[2] = 0;
	a1[3] = 0;
	a1[4] = F->negate(F->mult5(s6, b, d, gamma, F->inverse(F->mult(s7, epsilon))));
	a1[5] = 0;
	a1[6] = m1;
	a1[7] = 1;

	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing a2" << endl;
	}

	if (xi == 0) {
		a2[0] = 1;
		a2[1] = 0;
		a2[2] = 0;
		a2[3] = 0;
		a2[4] = 0;
		a2[5] = c133;
		a2[6] = 0;
		a2[7] = F->negate(c112);
	}
	else {
		a2[0] = F->mult3(s6, epsilon, F->inverse(F->mult(xi, s7)));
		a2[1] = 1;
		a2[2] = 0;
		a2[3] = 0;
		a2[4] = F->negate(F->mult4(a, c, gamma, F->inverse(xi)));
		a2[5] = 0;
		a2[6] = 0;
		a2[7] = 1;
	}


	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing a3" << endl;
	}

	a3[0] = 0;
	a3[1] = 0;
	a3[2] = 1;
	a3[3] = 0;
	a3[4] = 0;
	a3[5] = 0;
	a3[6] = 0;
	a3[7] = 1;


	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing a4" << endl;
	}

	a4[0] = 1;
	a4[1] = 1;
	a4[2] = 1;
	a4[3] = 0;
	a4[4] = 1;
	a4[5] = 0;
	a4[6] = 0;
	a4[7] = 1;


	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing a5" << endl;
	}

	a5[0] = a;
	a5[1] = b;
	a5[2] = 1;
	a5[3] = 0;
	a5[4] = a;
	a5[5] = 0;
	a5[6] = 0;
	a5[7] = 1;


	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing a6" << endl;
	}

	a6[0] = c;
	a6[1] = d;
	a6[2] = 1;
	a6[3] = 0;
	a6[4] = c;
	a6[5] = 0;
	a6[6] = 0;
	a6[7] = 1;


	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing b1" << endl;
	}

	b1[0] = 1;
	b1[1] = 0;
	b1[2] = 0;
	b1[3] = 0;
	b1[4] = 0;
	b1[5] = 0;
	b1[6] = 0;
	b1[7] = 1;

	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing b2" << endl;
	}

	b2[0] = 0;
	b2[1] = 1;
	b2[2] = 0;
	b2[3] = 0;
	b2[4] = 0;
	b2[5] = 0;
	b2[6] = m1;
	b2[7] = 1;

	int gamma_inv = F->inverse(gamma);
	int delta_inv = F->inverse(delta);

	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing b3" << endl;
	}

	b3[0] = F->negate(F->mult(theta, gamma_inv));
	b3[1] = F->negate(F->mult(zeta, gamma_inv));
	b3[2] = 0;
	b3[3] = 1;
	b3[4] = F->negate(F->mult(epsilon, gamma_inv));
	b3[5] = F->negate(F->mult(epsilon, gamma_inv));
	b3[6] = 1;
	b3[7] = 0;

	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing b4" << endl;
	}

	b4[0] = F->negate(F->mult(nu, delta_inv));
	b4[1] = F->negate(F->mult(nu, delta_inv));
	b4[2] = 1;
	b4[3] = 0;
	b4[4] = F->negate(F->mult4(a, c, s7, delta_inv));
	b4[5] = F->negate(F->mult4(a, c, s7, delta_inv));
	b4[6] = 0;
	b4[7] = 1;

	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing b5" << endl;
	}

	b5[0] = F->mult3(s4, delta, F->inverse(F->mult(s8, s7)));
	b5[1] = F->mult3(s3, delta, F->inverse(F->mult(s8, s6)));
	b5[2] = 1;
	b5[3] = 0;
	b5[4] = F->negate(F->mult3(c, s4, F->inverse(s8)));
	b5[5] = F->negate(F->mult4(c, s3, s7, F->inverse(F->mult(s8, s6))));
	b5[6] = 0;
	b5[7] = 1;

	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing b6" << endl;
	}

	b6[0] = F->mult3(s2, delta, F->inverse(F->mult(s5, s7)));
	b6[1] = F->mult3(s1, delta, F->inverse(F->mult(s6, s5)));
	b6[2] = 1;
	b6[3] = 0;
	b6[4] = F->negate(F->mult3(s2, a, F->inverse(s5)));
	b6[5] = F->negate(F->mult4(s7, s1, a, F->inverse(F->mult(s6, s5))));
	b6[6] = 0;
	b6[7] = 1;

	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"a1=";
		Int_vec_print(cout, a1, 8);
		cout << endl;
		cout << "normal_form::create_lines_of_cubic_surface "
				"a2=";
		Int_vec_print(cout, a2, 8);
		cout << endl;
		cout << "normal_form::create_lines_of_cubic_surface "
				"a3=";
		Int_vec_print(cout, a3, 8);
		cout << endl;
		cout << "normal_form::create_lines_of_cubic_surface "
				"a4=";
		Int_vec_print(cout, a4, 8);
		cout << endl;
		cout << "normal_form::create_lines_of_cubic_surface "
				"a5=";
		Int_vec_print(cout, a5, 8);
		cout << endl;
		cout << "normal_form::create_lines_of_cubic_surface "
				"a6=";
		Int_vec_print(cout, a6, 8);
		cout << endl;
	}

	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"b1=";
		Int_vec_print(cout, b1, 8);
		cout << endl;
		cout << "normal_form::create_lines_of_cubic_surface "
				"b2=";
		Int_vec_print(cout, b2, 8);
		cout << endl;
		cout << "normal_form::create_lines_of_cubic_surface "
				"b3=";
		Int_vec_print(cout, b3, 8);
		cout << endl;
		cout << "normal_form::create_lines_of_cubic_surface "
				"b4=";
		Int_vec_print(cout, b4, 8);
		cout << endl;
		cout << "normal_form::create_lines_of_cubic_surface "
				"b5=";
		Int_vec_print(cout, b5, 8);
		cout << endl;
		cout << "normal_form::create_lines_of_cubic_surface "
				"b6=";
		Int_vec_print(cout, b6, 8);
		cout << endl;
	}


#if 0
	int Basis[8];

	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing Double_six[0]" << endl;
	}
	Double_six[0] = Gr->rank_lint_here(a1, 0 /* verbose_level */);
	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing a1 = Double_six[0]=" << Double_six[0] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[0], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}



	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing Double_six[1]" << endl;
	}
	Double_six[1] = Gr->rank_lint_here(a2, 0 /* verbose_level */);
	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing a2 = Double_six[1]=" << Double_six[1] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[1], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}

	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing Double_six[2]" << endl;
	}
	Double_six[2] = Gr->rank_lint_here(a3, 0 /* verbose_level */);
	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing a3 = Double_six[2]=" << Double_six[2] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[2], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}


	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing Double_six[3]" << endl;
	}
	Double_six[3] = Gr->rank_lint_here(a4, 0 /* verbose_level */);
	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing a4 = Double_six[3]=" << Double_six[3] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[3], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}

	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing Double_six[4]" << endl;
	}
	Double_six[4] = Gr->rank_lint_here(a5, 0 /* verbose_level */);
	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing a5 = Double_six[4]=" << Double_six[4] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[4], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}

	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing Double_six[5]" << endl;
	}
	Double_six[5] = Gr->rank_lint_here(a6, 0 /* verbose_level */);
	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing a6 = Double_six[5]=" << Double_six[5] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[5], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}

	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing Double_six[6]" << endl;
	}
	Double_six[6] = Gr->rank_lint_here(b1, 0 /* verbose_level */);
	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing b1 = Double_six[6]=" << Double_six[6] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[6], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}


	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing Double_six[7]" << endl;
	}
	Double_six[7] = Gr->rank_lint_here(b2, 0 /* verbose_level */);
	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing b2 = Double_six[7]=" << Double_six[7] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[7], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}

	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing Double_six[8]" << endl;
	}
	Double_six[8] = Gr->rank_lint_here(b3, 0 /* verbose_level */);
	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing b3 = Double_six[8]=" << Double_six[8] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[8], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}

	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing Double_six[9]" << endl;
	}
	Double_six[9] = Gr->rank_lint_here(b4, 0 /* verbose_level */);
	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing b4 = Double_six[9]=" << Double_six[9] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[9], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}

	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing Double_six[10]" << endl;
	}
	Double_six[10] = Gr->rank_lint_here(b5, 0 /* verbose_level */);
	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing b5 = Double_six[10]=" << Double_six[10] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[10], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}

	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing Double_six[11]" << endl;
	}
	Double_six[11] = Gr->rank_lint_here(b6, 0 /* verbose_level */);
	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface "
				"computing b6 = Double_six[11]=" << Double_six[11] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[11], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}
#endif







	if (f_v) {
		cout << "normal_form::create_lines_of_cubic_surface done" << endl;
	}
}






}}}}



