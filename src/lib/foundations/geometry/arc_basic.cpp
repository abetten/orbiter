/*
 * arc_basic.cpp
 *
 *  Created on: Nov 26, 2021
 *      Author: betten
 */




#include "foundations.h"


using namespace std;



namespace orbiter {
namespace foundations {


arc_basic::arc_basic()
{
	F = NULL;

}

arc_basic::~arc_basic()
{

}

void arc_basic::init(finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_basic::init" << endl;
	}

	arc_basic::F = F;

	if (f_v) {
		cout << "arc_basic::init done" << endl;
	}
}

void arc_basic::Segre_hyperoval(
		long int *&Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int N = F->q + 2;
	int i, t, a, t6;
	int *Mtx;

	if (f_v) {
		cout << "arc_basic::Segre_hyperoval q=" << F->q << endl;
	}
	if (EVEN(F->e)) {
		cout << "arc_basic::Segre_hyperoval needs e odd" << endl;
		exit(1);
	}

	nb_pts = N;

	Pts = NEW_lint(N);
	Mtx = NEW_int(N * 3);
	Orbiter->Int_vec->zero(Mtx, N * 3);
	for (t = 0; t < F->q; t++) {
		t6 = F->power(t, 6);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = t6;
	}
	t = F->q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = F->q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		F->PG_element_rank_modified(Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
	}

	FREE_int(Mtx);
	if (f_v) {
		cout << "arc_basic::Segre_hyperoval "
				"q=" << F->q << " done" << endl;
	}
}


void arc_basic::GlynnI_hyperoval(
		long int *&Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int N = F->q + 2;
	int i, t, te, a;
	int sigma, gamma = 0, Sigma, /*Gamma,*/ exponent;
	int *Mtx;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "arc_basic::GlynnI_hyperoval q=" << F->q << endl;
	}
	if (EVEN(F->e)) {
		cout << "arc_basic::GlynnI_hyperoval needs e odd" << endl;
		exit(1);
	}

	sigma = F->e - 1;
	for (i = 0; i < F->e; i++) {
		if (((i * i) % F->e) == sigma) {
			gamma = i;
			break;
		}
	}
	if (i == F->e) {
		cout << "arc_basic::GlynnI_hyperoval "
				"did not find gamma" << endl;
		exit(1);
	}

	cout << "arc_basic::GlynnI_hyperoval sigma = " << sigma
			<< " gamma = " << gamma << endl;
	//Gamma = i_power_j(2, gamma);
	Sigma = NT.i_power_j(2, sigma);

	exponent = 3 * Sigma + 4;

	nb_pts = N;

	Pts = NEW_lint(N);
	Mtx = NEW_int(N * 3);
	Orbiter->Int_vec->zero(Mtx, N * 3);
	for (t = 0; t < F->q; t++) {
		te = F->power(t, exponent);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = te;
	}
	t = F->q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = F->q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		F->PG_element_rank_modified(Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
	}

	FREE_int(Mtx);
	if (f_v) {
		cout << "arc_basic::GlynnI_hyperoval "
				"q=" << F->q << " done" << endl;
	}
}

void arc_basic::GlynnII_hyperoval(
		long int *&Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int N = F->q + 2;
	int i, t, te, a;
	int sigma, gamma = 0, Sigma, Gamma, exponent;
	int *Mtx;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "arc_basic::GlynnII_hyperoval q=" << F->q << endl;
	}
	if (EVEN(F->e)) {
		cout << "arc_basic::GlynnII_hyperoval "
				"needs e odd" << endl;
		exit(1);
	}

	sigma = F->e - 1;
	for (i = 0; i < F->e; i++) {
		if (((i * i) % F->e) == sigma) {
			gamma = i;
			break;
		}
	}
	if (i == F->e) {
		cout << "arc_basic::GlynnII_hyperoval "
				"did not find gamma" << endl;
		exit(1);
	}

	cout << "arc_basic::GlynnII_hyperoval "
			"sigma = " << sigma << " gamma = " << i << endl;
	Gamma = NT.i_power_j(2, gamma);
	Sigma = NT.i_power_j(2, sigma);

	exponent = Sigma + Gamma;

	nb_pts = N;

	Pts = NEW_lint(N);
	Mtx = NEW_int(N * 3);
	Orbiter->Int_vec->zero(Mtx, N * 3);
	for (t = 0; t < F->q; t++) {
		te = F->power(t, exponent);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = te;
	}
	t = F->q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = F->q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		F->PG_element_rank_modified(Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
	}

	FREE_int(Mtx);
	if (f_v) {
		cout << "arc_basic::GlynnII_hyperoval "
				"q=" << F->q << " done" << endl;
	}
}


void arc_basic::Subiaco_oval(
		long int *&Pts, int &nb_pts, int f_short, int verbose_level)
// following Payne, Penttila, Pinneri:
// Isomorphisms Between Subiaco q-Clan Geometries,
// Bull. Belg. Math. Soc. 2 (1995) 197-222.
// formula (53)
{
	int f_v = (verbose_level >= 1);
	int N = F->q + 1;
	int i, t, a, b, h, k, top, bottom;
	int omega, omega2;
	int t2, t3, t4, sqrt_t;
	int *Mtx;

	if (f_v) {
		cout << "arc_basic::Subiaco_oval "
				"q=" << F->q << " f_short=" << f_short << endl;
	}

	nb_pts = N;
	k = (F->q - 1) / 3;
	if (k * 3 != F->q - 1) {
		cout << "arc_basic::Subiaco_oval k * 3 != q - 1" << endl;
		exit(1);
	}
	omega = F->power(F->alpha, k);
	omega2 = F->mult(omega, omega);
	if (F->add3(omega2, omega, 1) != 0) {
		cout << "arc_basic::Subiaco_oval "
				"add3(omega2, omega, 1) != 0" << endl;
		exit(1);
	}
	Pts = NEW_lint(N);
	Mtx = NEW_int(N * 3);
	Orbiter->Int_vec->zero(Mtx, N * 3);
	for (t = 0; t < F->q; t++) {
		t2 = F->mult(t, t);
		t3 = F->mult(t2, t);
		t4 = F->mult(t2, t2);
		sqrt_t = F->frobenius_power(t, F->e - 1);
		if (F->mult(sqrt_t, sqrt_t) != t) {
			cout << "arc_basic::Subiaco_oval "
					"mult(sqrt_t, sqrt_t) != t" << endl;
			exit(1);
		}
		bottom = F->add3(t4, F->mult(omega2, t2), 1);
		if (f_short) {
			top = F->mult(omega2, F->add(t4, t));
		}
		else {
			top = F->add3(t3, t2, F->mult(omega2, t));
		}
		if (FALSE) {
			cout << "t=" << t << " top=" << top
					<< " bottom=" << bottom << endl;
		}
		a = F->mult(top, F->inverse(bottom));
		if (f_short) {
			b = sqrt_t;
		}
		else {
			b = F->mult(omega, sqrt_t);
		}
		h = F->add(a, b);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = h;
	}
	t = F->q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	for (i = 0; i < N; i++) {
		F->PG_element_rank_modified(Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
	}

	FREE_int(Mtx);
	if (f_v) {
		cout << "arc_basic::Subiaco_oval "
				"q=" << F->q << " done" << endl;
	}
}




void arc_basic::Subiaco_hyperoval(
		long int *&Pts, int &nb_pts, int verbose_level)
// email 12/27/2014
//The o-polynomial of the Subiaco hyperoval is

//t^{1/2}+(d^2t^4 + d^2(1+d+d^2)t^3
// + d^2(1+d+d^2)t^2 + d^2t)/(t^4+d^2t^2+1)

//where d has absolute trace 1.

//Best,
//Tim

//absolute trace of 1/d is 1 not d...

{
	int f_v = (verbose_level >= 1);
	int N = F->q + 2;
	int i, t, d, dv, d2, one_d_d2, a, h;
	int t2, t3, t4, sqrt_t;
	int top1, top2, top3, top4, top, bottom;
	int *Mtx;

	if (f_v) {
		cout << "arc_basic::Subiaco_hyperoval q=" << F->q << endl;
	}

	nb_pts = N;
	for (d = 1; d < F->q; d++) {
		dv = F->inverse(d);
		if (F->absolute_trace(dv) == 1) {
			break;
		}
	}
	if (d == F->q) {
		cout << "arc_basic::Subiaco_hyperoval "
				"cannot find element d" << endl;
		exit(1);
	}
	d2 = F->mult(d, d);
	one_d_d2 = F->add3(1, d, d2);

	Pts = NEW_lint(N);
	Mtx = NEW_int(N * 3);
	Orbiter->Int_vec->zero(Mtx, N * 3);
	for (t = 0; t < F->q; t++) {
		t2 = F->mult(t, t);
		t3 = F->mult(t2, t);
		t4 = F->mult(t2, t2);
		sqrt_t = F->frobenius_power(t, F->e - 1);
		if (F->mult(sqrt_t, sqrt_t) != t) {
			cout << "arc_basic::Subiaco_hyperoval "
					"mult(sqrt_t, sqrt_t) != t" << endl;
			exit(1);
		}


		bottom = F->add3(t4, F->mult(d2, t2), 1);

		//t^{1/2}+(d^2t^4 + d^2(1+d+d^2)t^3 +
		// d^2(1+d+d^2)t^2 + d^2t)/(t^4+d^2t^2+1)

		top1 = F->mult(d2,t4);
		top2 = F->mult3(d2, one_d_d2, t3);
		top3 = F->mult3(d2, one_d_d2, t2);
		top4 = F->mult(d2, t);
		top = F->add4(top1, top2, top3, top4);

		if (f_v) {
			cout << "t=" << t << " top=" << top
					<< " bottom=" << bottom << endl;
		}
		a = F->mult(top, F->inverse(bottom));
		h = F->add(a, sqrt_t);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = h;
	}
	t = F->q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = F->q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		F->PG_element_rank_modified(Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
	}

	FREE_int(Mtx);
	if (f_v) {
		cout << "arc_basic::Subiaco_hyperoval "
				"q=" << F->q << " done" << endl;
	}
}




// From Bill Cherowitzo's web page:
// In 1991, O'Keefe and Penttila [OKPe92]
// by means of a detailed investigation
// of the divisibility properties of the orders
// of automorphism groups
// of hypothetical hyperovals in this plane,
// discovered a new hyperoval.
// Its o-polynomial is given by:

//f(x) = x4 + x16 + x28 + beta*11(x6 + x10 + x14 + x18 + x22 + x26)
// + beta*20(x8 + x20) + beta*6(x12 + x24),
//where ß is a primitive root of GF(32) satisfying beta^5 = beta^2 + 1.
//The full automorphism group of this hyperoval has order 3.

int arc_basic::OKeefe_Penttila_32(int t)
// needs the field generated by beta with beta^5 = beta^2+1
// From Bill Cherowitzo's hyperoval page
{
	int *t_powers;
	int a, b, c, d, e, beta6, beta11, beta20;

	t_powers = NEW_int(31);

	F->power_table(t, t_powers, 31);
	a = F->add3(t_powers[4], t_powers[16], t_powers[28]);
	b = F->add6(t_powers[6], t_powers[10], t_powers[14],
			t_powers[18], t_powers[22], t_powers[26]);
	c = F->add(t_powers[8], t_powers[20]);
	d = F->add(t_powers[12], t_powers[24]);

	beta6 = F->power(2, 6);
	beta11 = F->power(2, 11);
	beta20 = F->power(2, 20);

	b = F->mult(b, beta11);
	c = F->mult(c, beta20);
	d = F->mult(d, beta6);

	e = F->add4(a, b, c, d);

	FREE_int(t_powers);
	return e;
}



int arc_basic::Subiaco64_1(int t)
// needs the field generated by beta with beta^6 = beta+1
// The first one from Bill Cherowitzo's hyperoval page
{
	int *t_powers;
	int a, b, c, d, beta21, beta42;

	t_powers = NEW_int(65);

	F->power_table(t, t_powers, 65);
	a = F->add6(t_powers[8], t_powers[12], t_powers[20],
			t_powers[22], t_powers[42], t_powers[52]);
	b = F->add6(t_powers[4], t_powers[10], t_powers[14],
			t_powers[16], t_powers[30], t_powers[38]);
	c = F->add6(t_powers[44], t_powers[48], t_powers[54],
			t_powers[56], t_powers[58], t_powers[60]);
	b = F->add3(b, c, t_powers[62]);
	c = F->add7(t_powers[2], t_powers[6], t_powers[26],
			t_powers[28], t_powers[32], t_powers[36], t_powers[40]);
	beta21 = F->power(2, 21);
	beta42 = F->mult(beta21, beta21);
	d = F->add3(a, F->mult(beta21, b), F->mult(beta42, c));
	FREE_int(t_powers);
	return d;
}

int arc_basic::Subiaco64_2(int t)
// needs the field generated by beta with beta^6 = beta+1
// The second one from Bill Cherowitzo's hyperoval page
{
	int *t_powers;
	int a, b, c, d, beta21, beta42;

	t_powers = NEW_int(65);

	F->power_table(t, t_powers, 65);
	a = F->add3(t_powers[24], t_powers[30], t_powers[62]);
	b = F->add6(t_powers[4], t_powers[8], t_powers[10],
			t_powers[14], t_powers[16], t_powers[34]);
	c = F->add6(t_powers[38], t_powers[40], t_powers[44],
			t_powers[46], t_powers[52], t_powers[54]);
	b = F->add4(b, c, t_powers[58], t_powers[60]);
	c = F->add5(t_powers[6], t_powers[12], t_powers[18],
			t_powers[20], t_powers[26]);
	d = F->add5(t_powers[32], t_powers[36], t_powers[42],
			t_powers[48], t_powers[50]);
	c = F->add(c, d);
	beta21 = F->power(2, 21);
	beta42 = F->mult(beta21, beta21);
	d = F->add3(a, F->mult(beta21, b), F->mult(beta42, c));
	FREE_int(t_powers);
	return d;
}

int arc_basic::Adelaide64(int t)
// needs the field generated by beta with beta^6 = beta+1
{
	int *t_powers;
	int a, b, c, d, beta21, beta42;

	t_powers = NEW_int(65);

	F->power_table(t, t_powers, 65);
	a = F->add7(t_powers[4], t_powers[8], t_powers[14], t_powers[34],
			t_powers[42], t_powers[48], t_powers[62]);
	b = F->add8(t_powers[6], t_powers[16], t_powers[26], t_powers[28],
			t_powers[30], t_powers[32], t_powers[40], t_powers[58]);
	c = F->add8(t_powers[10], t_powers[18], t_powers[24], t_powers[36],
			t_powers[44], t_powers[50], t_powers[52], t_powers[60]);
	beta21 = F->power(2, 21);
	beta42 = F->mult(beta21, beta21);
	d = F->add3(a, F->mult(beta21, b), F->mult(beta42, c));
	FREE_int(t_powers);
	return d;
}



void arc_basic::LunelliSce(int *pts18, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//const char *override_poly = "19";
	//finite_field F;
	//int n = 3;
	//int q = 16;
	int v[3];
	//int w[3];
	geometry_global Gg;

	if (f_v) {
		cout << "arc_basic::LunelliSce" << endl;
	}
	//F.init(q), verbose_level - 2);
	//F.init_override_polynomial(q, override_poly, verbose_level);

#if 0
	int cubic1[100];
	int cubic1_size = 0;
	int cubic2[100];
	int cubic2_size = 0;
	int hoval[100];
	int hoval_size = 0;
#endif

	int a, b, i, sz, N;

	if (F->q != 16) {
		cout << "arc_basic::LunelliSce "
				"field order must be 16" << endl;
		exit(1);
	}
	N = Gg.nb_PG_elements(2, 16);
	sz = 0;
	for (i = 0; i < N; i++) {
		F->PG_element_unrank_modified(v, 1, 3, i);
		//cout << "i=" << i << " v=";
		//int_vec_print(cout, v, 3);
		//cout << endl;

		a = LunelliSce_evaluate_cubic1(v);
		b = LunelliSce_evaluate_cubic2(v);

		// form the symmetric difference of the two cubics:
		if ((a == 0 && b) || (b == 0 && a)) {
			pts18[sz++] = i;
		}
	}
	if (sz != 18) {
		cout << "sz != 18" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "the size of the LinelliSce hyperoval is " << sz << endl;
		cout << "the LinelliSce hyperoval is:" << endl;
		Orbiter->Int_vec->print(cout, pts18, sz);
		cout << endl;
	}

#if 0
	cout << "the size of cubic1 is " << cubic1_size << endl;
	cout << "the cubic1 is:" << endl;
	int_vec_print(cout, cubic1, cubic1_size);
	cout << endl;
	cout << "the size of cubic2 is " << cubic2_size << endl;
	cout << "the cubic2 is:" << endl;
	int_vec_print(cout, cubic2, cubic2_size);
	cout << endl;
#endif

}

int arc_basic::LunelliSce_evaluate_cubic1(int *v)
// computes X^3 + Y^3 + Z^3 + \eta^3 XYZ
{
	int a, b, c, d, e, eta3;

	eta3 = F->power(2, 3);
	//eta12 = power(2, 12);
	a = F->power(v[0], 3);
	b = F->power(v[1], 3);
	c = F->power(v[2], 3);
	d = F->product4(eta3, v[0], v[1], v[2]);
	e = F->add4(a, b, c, d);
	return e;
}

int arc_basic::LunelliSce_evaluate_cubic2(int *v)
// computes X^3 + Y^3 + Z^3 + \eta^{12} XYZ
{
	int a, b, c, d, e, eta12;

	//eta3 = power(2, 3);
	eta12 = F->power(2, 12);
	a = F->power(v[0], 3);
	b = F->power(v[1], 3);
	c = F->power(v[2], 3);
	d = F->product4(eta12, v[0], v[1], v[2]);
	e = F->add4(a, b, c, d);
	return e;
}

}}



