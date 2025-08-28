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
namespace geometry {
namespace algebraic_geometry {


void surface_domain::create_equation_general_abcd(
		int a, int b, int c, int d,
		int *coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surface_domain::create_equation_general_abcd" << endl;
	}



	Int_vec_zero(coeff, PolynomialDomains->nb_monomials);

	int c002;
	int c012;
	int c013;
	int c022;
	int c023;
	int c112;
	int c113;
	int c122;
	int c133;
	int c123;

	create_coefficients_for_general_abcd(
			a, b, c, d,
			c002,
			c012,
			c013,
			c022,
			c023,
			c112,
			c113,
			c122,
			c133,
			c123,
			verbose_level);

	coeff[5] = c002;
	coeff[16] = c012;
	coeff[17] = c013;
	coeff[10] = c022;
	coeff[18] = c023;
	coeff[8] = c112;
	coeff[9] = c113;
	coeff[11] = c122;
	coeff[14] = c133;
	coeff[19] = c123;



	if (f_v) {
		cout << "surface_domain::create_equation_general_abcd done" << endl;
	}
}

void surface_domain::create_coefficients_for_general_abcd(
		int a, int b, int c, int d,
		int &c002,
		int &c012,
		int &c013,
		int &c022,
		int &c023,
		int &c112,
		int &c113,
		int &c122,
		int &c133,
		int &c123,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surface_domain::create_coefficients_for_general_abcd" << endl;
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
		cout << "surface_domain::create_coefficients_for_general_abcd c002=" << c002 << endl;
		cout << "surface_domain::create_coefficients_for_general_abcd c012=" << c012 << endl;
		cout << "surface_domain::create_coefficients_for_general_abcd c013=" << c013 << endl;
		cout << "surface_domain::create_coefficients_for_general_abcd c022=" << c022 << endl;
		cout << "surface_domain::create_coefficients_for_general_abcd c023=" << c023 << endl;
		cout << "surface_domain::create_coefficients_for_general_abcd c112=" << c112 << endl;
		cout << "surface_domain::create_coefficients_for_general_abcd c113=" << c113 << endl;
		cout << "surface_domain::create_coefficients_for_general_abcd c122=" << c122 << endl;
		cout << "surface_domain::create_coefficients_for_general_abcd c133=" << c133 << endl;
		cout << "surface_domain::create_coefficients_for_general_abcd c123=" << c123 << endl;
	}


	if (f_v) {
		cout << "surface_domain::create_coefficients_for_general_abcd done" << endl;
	}
}

void surface_domain::create_lines_for_general_abcd(
		int a, int b, int c, int d,
		long int *Lines27, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd" << endl;
		cout << "surface_domain::create_lines_for_general_abcd a = " << a << endl;
		cout << "surface_domain::create_lines_for_general_abcd b = " << b << endl;
		cout << "surface_domain::create_lines_for_general_abcd c = " << c << endl;
		cout << "surface_domain::create_lines_for_general_abcd d = " << d << endl;
	}

	int m1 = F->negate(1);
	int s1 = F->add(a, m1);
	int s2 = F->add(b, m1);
	int s3 = F->add(c, m1);
	int s4 = F->add(d, m1);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"before s5" << endl;
	}
	int s5 = F->add(a, F->negate(b));
	int s6 = F->add(a, F->negate(c));
	int s7 = F->add(b, F->negate(d));
	int s8 = F->add(c, F->negate(d));
	int ab = F->mult(a, b);
	int ad = F->mult(a, d);
	int bc = F->mult(b, c);
	int cd = F->mult(c, d);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"before abc" << endl;
	}
	int abc = F->mult3(a, b, c);
	int abd = F->mult3(a, b, d);
	int acd = F->mult3(a, c, d);
	int bcd = F->mult3(b, c, d);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"before delta" << endl;
	}
	int delta = F->add(ad, F->negate(bc));
	int epsilon = F->add5(abc, F->negate(abd), F->negate(acd), bcd, delta);
	int gamma = F->add5(delta, F->negate(a), b, c, F->negate(d));


	int lambda = F->add3(F->mult3(b, b, s8), F->negate(F->mult3(d, d, s5)), delta);
	// !!! mistake fixed A. Betten 8/24/2025
	// it was s7, but it should be s5

	//int mu = F->add3(F->negate(abd), bcd, delta);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"before nu" << endl;
	}
	int nu = F->add(F->mult3(a, c, s7), F->negate(F->mult3(b, d, s6)));
	//int eta = F->add6(F->negate(F->mult(a, acd)), F->mult(abc, c), F->mult(a, ad), F->negate(abd), F->negate(F->mult(bc, c)), bcd);
	int zeta = F->mult3(s1, s3, s7);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"before xi" << endl;
	}
	int xi = F->add5(F->mult3(a, a, c), F->negate(F->mult(a, ad)), F->negate(F->mult3(a, c, c)), F->mult3(b, c, c), delta);
	int theta = F->add6(abc, F->negate(acd), F->negate(ab), cd, a, F->negate(c));


	int c002;
	int c012;
	int c013;
	int c022;
	int c023;
	int c112;
	int c113;
	int c122;
	int c133;
	int c123;

	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"before create_coefficients_for_general_abcd" << endl;
	}

	create_coefficients_for_general_abcd(
			a, b, c, d,
			c002,
			c012,
			c013,
			c022,
			c023,
			c112,
			c113,
			c122,
			c133,
			c123,
			verbose_level);

	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"after create_coefficients_for_general_abcd" << endl;
	}




	int a1[8];
	int a2[8];
	int a3[8];
	int a4[8];
	int a5[8];
	int a6[8];

	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
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
		cout << "surface_domain::create_lines_for_general_abcd "
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
		cout << "surface_domain::create_lines_for_general_abcd "
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
		cout << "surface_domain::create_lines_for_general_abcd "
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
		cout << "surface_domain::create_lines_for_general_abcd "
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
		cout << "surface_domain::create_lines_for_general_abcd "
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


	int b1[8];
	int b2[8];
	int b3[8];
	int b4[8];
	int b5[8];
	int b6[8];

	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
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
		cout << "surface_domain::create_lines_for_general_abcd "
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
		cout << "surface_domain::create_lines_for_general_abcd "
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
		cout << "surface_domain::create_lines_for_general_abcd "
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
		cout << "surface_domain::create_lines_for_general_abcd "
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
		cout << "surface_domain::create_lines_for_general_abcd "
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
		cout << "surface_domain::create_lines_for_general_abcd "
				"a1=";
		Int_vec_print(cout, a1, 8);
		cout << endl;
		cout << "surface_domain::create_lines_for_general_abcd "
				"a2=";
		Int_vec_print(cout, a2, 8);
		cout << endl;
		cout << "surface_domain::create_lines_for_general_abcd "
				"a3=";
		Int_vec_print(cout, a3, 8);
		cout << endl;
		cout << "surface_domain::create_lines_for_general_abcd "
				"a4=";
		Int_vec_print(cout, a4, 8);
		cout << endl;
		cout << "surface_domain::create_lines_for_general_abcd "
				"a5=";
		Int_vec_print(cout, a5, 8);
		cout << endl;
		cout << "surface_domain::create_lines_for_general_abcd "
				"a6=";
		Int_vec_print(cout, a6, 8);
		cout << endl;
	}

	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"b1=";
		Int_vec_print(cout, b1, 8);
		cout << endl;
		cout << "surface_domain::create_lines_for_general_abcd "
				"b2=";
		Int_vec_print(cout, b2, 8);
		cout << endl;
		cout << "surface_domain::create_lines_for_general_abcd "
				"b3=";
		Int_vec_print(cout, b3, 8);
		cout << endl;
		cout << "surface_domain::create_lines_for_general_abcd "
				"b4=";
		Int_vec_print(cout, b4, 8);
		cout << endl;
		cout << "surface_domain::create_lines_for_general_abcd "
				"b5=";
		Int_vec_print(cout, b5, 8);
		cout << endl;
		cout << "surface_domain::create_lines_for_general_abcd "
				"b6=";
		Int_vec_print(cout, b6, 8);
		cout << endl;
	}


	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing Double_six" << endl;
	}

	long int Double_six[12];
	int Basis[8];

	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing Double_six[0]" << endl;
	}
	Double_six[0] = Gr->rank_lint_here(a1, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing a1 = Double_six[0]=" << Double_six[0] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[0], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}



	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing Double_six[1]" << endl;
	}
	Double_six[1] = Gr->rank_lint_here(a2, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing a2 = Double_six[1]=" << Double_six[1] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[1], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}

	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing Double_six[2]" << endl;
	}
	Double_six[2] = Gr->rank_lint_here(a3, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing a3 = Double_six[2]=" << Double_six[2] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[2], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}


	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing Double_six[3]" << endl;
	}
	Double_six[3] = Gr->rank_lint_here(a4, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing a4 = Double_six[3]=" << Double_six[3] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[3], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}

	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing Double_six[4]" << endl;
	}
	Double_six[4] = Gr->rank_lint_here(a5, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing a5 = Double_six[4]=" << Double_six[4] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[4], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}

	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing Double_six[5]" << endl;
	}
	Double_six[5] = Gr->rank_lint_here(a6, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing a6 = Double_six[5]=" << Double_six[5] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[5], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}

	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing Double_six[6]" << endl;
	}
	Double_six[6] = Gr->rank_lint_here(b1, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing b1 = Double_six[6]=" << Double_six[6] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[6], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}


	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing Double_six[7]" << endl;
	}
	Double_six[7] = Gr->rank_lint_here(b2, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing b2 = Double_six[7]=" << Double_six[7] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[7], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}

	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing Double_six[8]" << endl;
	}
	Double_six[8] = Gr->rank_lint_here(b3, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing b3 = Double_six[8]=" << Double_six[8] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[8], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}

	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing Double_six[9]" << endl;
	}
	Double_six[9] = Gr->rank_lint_here(b4, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing b4 = Double_six[9]=" << Double_six[9] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[9], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}

	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing Double_six[10]" << endl;
	}
	Double_six[10] = Gr->rank_lint_here(b5, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing b5 = Double_six[10]=" << Double_six[10] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[10], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}

	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing Double_six[11]" << endl;
	}
	Double_six[11] = Gr->rank_lint_here(b6, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"computing b6 = Double_six[11]=" << Double_six[11] << endl;
	}
	Gr->unrank_lint_here(Basis, Double_six[11], 0 /* verbose_level */);
	if (f_v) {
		Int_matrix_print(Basis, 2, 4);
	}



	Lint_vec_copy(Double_six, Lines27, 12);


	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"before create_remaining_fifteen_lines" << endl;
	}
	create_remaining_fifteen_lines(
			Double_six, Lines27 + 12,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd "
				"after create_remaining_fifteen_lines" << endl;
	}




	if (f_v) {
		cout << "surface_domain::create_lines_for_general_abcd done" << endl;
	}
}


void surface_domain::create_equation_Cayley_klmn(
		int k, int l, int m, int n,
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


void surface_domain::create_equation_bes(
		int a, int c, int *coeff, int verbose_level)
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



void surface_domain::create_equation_F13(
		int a, int *coeff, int verbose_level)
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

void surface_domain::create_equation_G13(
		int a, int *coeff, int verbose_level)
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

surface_object *surface_domain::create_surface_general_abcd(
		int a, int b, int c, int d,
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
	create_equation_general_abcd(a, b, c, d, coeff20, verbose_level - 2);
	if (f_v) {
		cout << "surface_domain::create_surface_general_abcd "
				"after create_equation_general_abcd" << endl;
	}

#if 1
	long int Lines27[27];

	if (f_v) {
		cout << "surface_domain::create_surface_general_abcd "
				"before create_lines_for_general_abcd" << endl;
	}
	create_lines_for_general_abcd(
			a, b, c, d,
			Lines27, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_general_abcd "
				"after create_lines_for_general_abcd" << endl;
	}

#endif


	std::string label_txt;
	std::string label_tex;

	std::string s_a;
	std::string s_b;
	std::string s_c;
	std::string s_d;

	s_a = std::to_string(a);
	s_b = std::to_string(b);
	s_c = std::to_string(c);
	s_d = std::to_string(d);

	label_txt = "F_a" + s_a + "_b" + s_b + "_c" + s_c + "_d" + s_d;
	label_tex = "F\\_a" + s_a + "\\_b" + s_b + "\\_c" + s_c + "\\_d" + s_d;
	surface_object *SO;

	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_domain::create_surface_general_abcd "
				"before SO->init_equation" << endl;
	}
#if 0
	SO->init_equation(
			this, coeff20,
			label_txt, label_tex,
			verbose_level - 2);
#else
	SO->init_equation_with_27_lines(
			this, coeff20,
			Lines27,
			label_txt, label_tex,
			verbose_level - 2);
#endif
	if (f_v) {
		cout << "surface_domain::create_surface_general_abcd "
				"after SO->init_equation" << endl;
	}


	if (f_v) {
		cout << "surface_domain::create_surface_general_abcd done" << endl;
	}
	return SO;
}



surface_object *surface_domain::create_surface_bes(
		int a, int c,
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

	std::string label_txt;
	std::string label_tex;

	label_txt = "Bes_a" + std::to_string(a) + "_c" + std::to_string(c);
	label_tex = "Bes\\_a" + std::to_string(a) + "\\_c" + std::to_string(c);

	surface_object *SO;

	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_domain::create_surface_bes "
				"before SO->init_equation" << endl;
	}
	SO->init_equation(this, coeff20, label_txt, label_tex, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_bes "
				"after SO->init_equation" << endl;
	}

	if (f_v) {
		cout << "surface_domain::create_surface_bes done" << endl;
	}
	return SO;
}


surface_object *surface_domain::create_surface_F13(
		int a, int verbose_level)
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

	std::string label_txt;
	std::string label_tex;

	label_txt = "F13_a" + std::to_string(a);
	label_tex = "F13\\_a" + std::to_string(a);

	surface_object *SO;

	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_domain::create_surface_F13 "
				"before SO->init_equation" << endl;
	}
	SO->init_equation(this, coeff20, label_txt, label_tex, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_F13 "
				"after SO->init_equation" << endl;
	}

	if (f_v) {
		cout << "surface_domain::create_surface_F13 done" << endl;
	}
	return SO;
}

surface_object *surface_domain::create_surface_G13(
		int a, int verbose_level)
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

	std::string label_txt;
	std::string label_tex;

	label_txt = "G13_a" + std::to_string(a);
	label_tex = "G13\\_a" + std::to_string(a);

	surface_object *SO;

	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_domain::create_surface_G13 "
				"before SO->init_equation" << endl;
	}
	SO->init_equation(this, coeff20, label_txt, label_tex, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_G13 "
				"after SO->init_equation" << endl;
	}

	if (f_v) {
		cout << "surface_domain::create_surface_G13 done" << endl;
	}
	return SO;
}

surface_object *surface_domain::create_Eckardt_surface(
		int a, int b,
	int &alpha, int &beta,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int alpha0, beta0;
	long int Lines27[27];
	int i, rk, nb;
	geometry::other_geometry::geometry_global Gg;
	other::data_structures::sorting Sorting;

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
		return false;
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

	create_remaining_fifteen_lines(
			Lines27,
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

	create_Eckardt_fifteen_lines(
			special_lines, a, b, verbose_level);
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

	build_cubic_surface_from_lines(
			27, Lines27, coeff20,
			0 /* verbose_level */);
	F->Projective_space_basic->PG_element_normalize_from_front(coeff20, 1, 20);





	if (!test_Eckardt_form_alpha_beta(
			coeff20, alpha, beta,
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

	std::string label_txt;
	std::string label_tex;

	label_txt = "Eckardt_alpha" + std::to_string(alpha) + "_beta" + std::to_string(beta);
	label_tex = "Eckardt\\_alpha" + std::to_string(alpha) + "\\_beta" + std::to_string(beta);


	surface_object *SO;

	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_domain::create_Eckardt_surface "
				"before SO->init_with_27_lines" << endl;
	}
	SO->init_with_27_lines(this,
		Lines27, coeff20,
		label_txt, label_tex,
		false /* f_find_double_six_and_rearrange_lines */,
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


void surface_domain::create_equation_Eckardt_surface(
		int a, int b,
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

int surface_domain::test_Eckardt_form_alpha_beta(
		int *coeff,
	int &alpha, int &beta, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = true;
	int zeroes[] = {0,1,2,4,5,7,8,10,11,13,14,15,17,18,19};
	int alphas[] = {6,9,12};
	int betas[] = {16};
	int a;

	if (f_v) {
		cout << "surface_domain::test_Eckardt_form_alpha_beta" << endl;
	}
	if (!other::orbiter_kernel_system::Orbiter->Int_vec->is_constant_on_subset(coeff,
		zeroes, sizeof(zeroes) / sizeof(int), a)) {
		cout << "surface_domain::test_Eckardt_form_alpha_beta "
				"not constant on zero set" << endl;
		return false;
	}
	if (a != 0) {
		cout << "surface_domain::test_Eckardt_form_alpha_beta "
				"not zero on zero set" << endl;
		return false;
	}
	if (coeff[3] != 1) {
		cout << "surface_domain::test_Eckardt_form_alpha_beta "
				"not normalized" << endl;
		exit(1);
	}
	if (!other::orbiter_kernel_system::Orbiter->Int_vec->is_constant_on_subset(coeff,
		alphas, sizeof(alphas) / sizeof(int), a)) {
		cout << "surface_domain::test_Eckardt_form_alpha_beta "
				"not constant on alpha set" << endl;
		return false;
	}
	alpha = a;
	if (!other::orbiter_kernel_system::Orbiter->Int_vec->is_constant_on_subset(coeff,
		betas, sizeof(betas) / sizeof(int), a)) {
		cout << "surface_domain::test_Eckardt_form_alpha_beta "
				"not constant on beta set" << endl;
		return false;
	}
	beta = a;

	if (f_v) {
		cout << "surface_domain::test_Eckardt_form_alpha_beta done" << endl;
	}
	return ret;
}

void surface_domain::create_Eckardt_double_six(
		long int *double_six,
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

void surface_domain::create_Eckardt_fifteen_lines(
		long int *fifteen_lines,
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





}}}}

