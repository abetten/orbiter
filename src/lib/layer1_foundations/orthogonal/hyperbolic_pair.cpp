/*
 * hyperbolic_pair.cpp
 *
 *  Created on: Sep 30, 2022
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace orthogonal_geometry {


hyperbolic_pair::hyperbolic_pair()
{

	O = NULL;
	F = NULL;

	q = 0;
	epsilon = 0;
	m = 0;
	n = 0;

	pt_P = pt_Q = 0;
	nb_points = 0;
	nb_lines = 0;

	T1_m = 0;
	T1_mm1 = 0;
	T1_mm2 = 0;
	T2_m = 0;
	T2_mm1 = 0;
	T2_mm2 = 0;
	N1_m = 0;
	N1_mm1 = 0;
	N1_mm2 = 0;
	S_m = 0;
	S_mm1 = 0;
	S_mm2 = 0;
	Sbar_m = 0;
	Sbar_mm1 = 0;
	Sbar_mm2 = 0;

	alpha = beta = gamma = 0;
	subspace_point_type = 0;
	subspace_line_type = 0;

	nb_point_classes = nb_line_classes = 0;
	A = B = P = L = NULL;

	p1 = p2 = p3 = p4 = p5 = p6 = 0;
	l1 = l2 = l3 = l4 = l5 = l6 = l7 = 0;
	a11 = a12 = a22 = a23 = a26 = a32 = a34 = a37 = 0;
	a41 = a43 = a44 = a45 = a46 = a47 = a56 = a67 = 0;
	b11 = b12 = b22 = b23 = b26 = b32 = b34 = b37 = 0;
	b41 = b43 = b44 = b45 = b46 = b47 = b56 = b67 = 0;

	p7 = l8 = 0;
	a21 = a36 = a57 = a22a = a33 = a22b = 0;
	a32b = a42b = a51 = a53 = a54 = a55 = a66 = a77 = 0;
	b21 = b36 = b57 = b22a = b33 = b22b = 0;
	b32b = b42b = b51 = b53 = b54 = b55 = b66 = b77 = 0;
	a12b = a52a = 0;
	b12b = b52a = 0;
	delta = omega = lambda = mu = nu = zeta = 0;

	v1 = NULL;
	v2 = NULL;
	v3 = NULL;
	v4 = NULL;
	v5 = NULL;
	v_tmp = NULL;
	v_tmp2 = NULL;
	v_neighbor5 = NULL;

	rk_pt_v = NULL;

}

hyperbolic_pair::~hyperbolic_pair()
{

	//cout << "hyperbolic_pair::~hyperbolic_pair freeing A" << endl;
	if (A) {
		FREE_lint(A);
	}
	//cout << "hyperbolic_pair::~hyperbolic_pair freeing B" << endl;
	if (B) {
		FREE_lint(B);
	}
	//cout << "hyperbolic_pair::~hyperbolic_pair freeing P" << endl;
	if (P) {
		FREE_lint(P);
	}
	//cout << "hyperbolic_pair::~hyperbolic_pair freeing L" << endl;
	if (L) {
		FREE_lint(L);
	}

	//cout << "hyperbolic_pair::~hyperbolic_pair freeing v1" << endl;
	if (v1) {
		FREE_int(v1);
	}
	//cout << "hyperbolic_pair::~hyperbolic_pair freeing v2" << endl;
	if (v2) {
		FREE_int(v2);
	}
	//cout << "hyperbolic_pair::~hyperbolic_pair freeing v3" << endl;
	if (v3) {
		FREE_int(v3);
	}
	if (v4) {
		FREE_int(v4);
	}
	if (v5) {
		FREE_int(v5);
	}
	if (v_tmp) {
		FREE_int(v_tmp);
	}
	if (v_tmp2) {
		FREE_int(v_tmp2);
	}
	if (v_neighbor5) {
		FREE_int(v_neighbor5);
	}


	if (rk_pt_v) {
		FREE_int(rk_pt_v);
	}


}

void hyperbolic_pair::init(orthogonal *O, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hyperbolic_pair::init" << endl;
	}

	hyperbolic_pair::O = O;

	F = O->F;
	q = F->q;
	epsilon = O->epsilon;
	m = O->m;
	n = O->n;


	if (f_v) {
		cout << "hyperbolic_pair::init before init_counting_functions" << endl;
	}
	init_counting_functions(verbose_level - 2);
	if (f_v) {
		cout << "hyperbolic_pair::init after init_counting_functions" << endl;
	}

	if (f_v) {
		cout << "hyperbolic_pair::init before init_decomposition" << endl;
	}
	init_decomposition(verbose_level - 2);
	if (f_v) {
		cout << "hyperbolic_pair::init after init_decomposition" << endl;
	}

	v1 = NEW_int(n);
	v2 = NEW_int(n);
	v3 = NEW_int(n);
	v4 = NEW_int(n);
	v5 = NEW_int(n);
	v_tmp = NEW_int(n);
	v_tmp2 = NEW_int(n);
	v_neighbor5 = NEW_int(n);
	rk_pt_v = NEW_int(n);



	if (f_v) {
		cout << "hyperbolic_pair::init done" << endl;
	}

}

void hyperbolic_pair::init_counting_functions(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "hyperbolic_pair::init_counting_functions" << endl;
	}

	int epsilon, m, q;

	epsilon = O->epsilon;
	m = O->m;
	q = O->q;

	T1_m = Gg.count_T1(epsilon, m, q);
	if (f_v) {
		cout << "T1_m(" << epsilon << ","
				<< m << "," << q << ") = " << T1_m << endl;
	}
	T1_mm1 = Gg.count_T1(epsilon, m - 1, q);
	if (f_v) {
		cout << "T1_mm1(" << epsilon << ","
				<< m - 1 << "," << q << ") = " << T1_mm1 << endl;
	}
	if (m > 1) {
		T1_mm2 = Gg.count_T1(epsilon, m - 2, q);
		if (f_v) {
			cout << "T1_mm2(" << epsilon << ","
					<< m - 2 << "," << q << ") = " << T1_mm2 << endl;
		}
	}
	else {
		T1_mm2 = 0;
	}
	T2_m = Gg.count_T2(m, q);
	T2_mm1 = Gg.count_T2(m - 1, q);
	if (m > 1) {
		T2_mm2 = Gg.count_T2(m - 2, q);
	}
	else {
		T2_mm2 = 0;
	}
	N1_m = Gg.count_N1(m, q);
	N1_mm1 = Gg.count_N1(m - 1, q);
	if (m > 1) {
		N1_mm2 = Gg.count_N1(m - 2, q);
	}
	else {
		N1_mm2 = 0;
	}
	S_m = Gg.count_S(m, q);
	S_mm1 = Gg.count_S(m - 1, q);
	if (m > 1) {
		S_mm2 = Gg.count_S(m - 2, q);
	}
	else {
		S_mm2 = 0;
	}
	Sbar_m = Gg.count_Sbar(m, q);
	Sbar_mm1 = Gg.count_Sbar(m - 1, q);
	if (m > 1) {
		Sbar_mm2 = Gg.count_Sbar(m - 2, q);
	}
	else {
		Sbar_mm2 = 0;
	}

	if (f_v) {
		cout << "T1(" << m << "," << q << ") = " << T1_m << endl;
		if (m >= 1) {
			cout << "T1(" << m - 1 << "," << q << ") = " << T1_mm1 << endl;
		}
		if (m >= 2) {
			cout << "T1(" << m - 2 << "," << q << ") = " << T1_mm2 << endl;
		}
		cout << "T2(" << m << "," << q << ") = " << T2_m << endl;
		if (m >= 1) {
			cout << "T2(" << m - 1 << "," << q << ") = " << T2_mm1 << endl;
		}
		if (m >= 2) {
			cout << "T2(" << m - 2 << "," << q << ") = " << T2_mm2 << endl;
		}
		cout << "nb_pts_N1(" << m << "," << q << ") = " << N1_m << endl;
		if (m >= 1) {
			cout << "nb_pts_N1(" << m - 1 << "," << q << ") = "
			<< N1_mm1 << endl;
		}
		if (m >= 2) {
			cout << "nb_pts_N1(" << m - 2 << "," << q << ") = "
			<< N1_mm2 << endl;
		}
		cout << "S_m=" << S_m << endl;
		cout << "S_mm1=" << S_mm1 << endl;
		cout << "S_mm2=" << S_mm2 << endl;
		cout << "Sbar_m=" << Sbar_m << endl;
		cout << "Sbar_mm1=" << Sbar_mm1 << endl;
		cout << "Sbar_mm2=" << Sbar_mm2 << endl;
		cout << "N1_m=" << N1_m << endl;
		cout << "N1_mm1=" << N1_mm1 << endl;
		cout << "N1_mm2=" << N1_mm2 << endl;
	}
	if (f_v) {
		cout << "hyperbolic_pair::init_counting_functions done" << endl;
	}
}

void hyperbolic_pair::init_decomposition(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	geometry::geometry_global Gg;
	int i;

	if (f_v) {
		cout << "hyperbolic_pair::init_decomposition" << endl;
	}

	if (O->epsilon == 1) {
#if 1
		long int u;

		u = Gg.nb_pts_Qepsilon(O->epsilon, 2 * O->m - 1, O->q);
		if (T1_m != u) {
			cout << "T1_m != nb_pts_Qepsilon" << endl;
			cout << "T1_m=" << T1_m << endl;
			cout << "u=" << u << endl;
			exit(1);
		}
#endif
		if (f_v) {
			cout << "hyperbolic_pair::init_decomposition before init_hyperbolic" << endl;
		}
		init_hyperbolic(verbose_level /*- 3*/);
		if (f_v) {
			cout << "hyperbolic_pair::init_decomposition after init_hyperbolic" << endl;
		}
	}
	else if (O->epsilon == 0) {
		if (f_v) {
			cout << "hyperbolic_pair::init_decomposition before init_parabolic" << endl;
		}
		init_parabolic(verbose_level /*- 3*/);
		if (f_v) {
			cout << "hyperbolic_pair::init_decomposition after init_parabolic" << endl;
		}
	}
	else if (O->epsilon == -1) {
		nb_points = Gg.nb_pts_Qepsilon(O->epsilon, O->n - 1, O->q);
		nb_lines = 0;
		if (f_v) {
			cout << "nb_points=" << nb_points << endl;
		}
		//cout << "elliptic type not yet implemented" << endl;
		return;
		//exit(1);
	}
	else {
		cout << "hyperbolic_pair::init_decomposition epsilon = " << O->epsilon << " is illegal" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "hyperbolic_pair::init_decomposition Point partition:" << endl;
		for (i = 0; i < nb_point_classes; i++) {
			cout << P[i] << endl;
		}
		cout << "hyperbolic_pair::init_decomposition Line partition:" << endl;
		for (i = 0; i < nb_line_classes; i++) {
			cout << L[i] << endl;
		}
	}
	nb_points = 0;
	for (i = 0; i < nb_point_classes; i++) {
		nb_points += P[i];
	}
	nb_lines = 0;
	for (i = 0; i < nb_line_classes; i++) {
		nb_lines += L[i];
	}
	if (f_v) {
		cout << "hyperbolic_pair::init_decomposition nb_points = " << nb_points << endl;
		cout << "hyperbolic_pair::init_decomposition nb_lines = " << nb_lines << endl;
	}
	if (f_v) {
		cout << "hyperbolic_pair::init_decomposition done" << endl;
	}

}
void hyperbolic_pair::init_parabolic(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	geometry::geometry_global Gg;

	//int a, b, c;

	if (f_v) {
		cout << "hyperbolic_pair::init_parabolic m=" << O->m << " q=" << O->q << endl;
	}

	nb_point_classes = 7;
	nb_line_classes = 8;
	subspace_point_type = 5;
	subspace_line_type = 6;

	A = NEW_lint(nb_point_classes * nb_line_classes);
	B = NEW_lint(nb_point_classes * nb_line_classes);
	P = NEW_lint(nb_point_classes);
	L = NEW_lint(nb_line_classes);

	for (i = 0; i < nb_point_classes * nb_line_classes; i++) {
		A[i] = B[i] = 0;
	}

	if (O->f_even) {
		init_parabolic_even(verbose_level);
	}
	else {
		init_parabolic_odd(verbose_level);
	}


	P[0] = p1;
	P[1] = p2;
	P[2] = p3;
	P[3] = p4;
	P[4] = p5;
	P[5] = p6;
	P[6] = p7;
	L[0] = l1;
	L[1] = l2;
	L[2] = l3;
	L[3] = l4;
	L[4] = l5;
	L[5] = l6;
	L[6] = l7;
	L[7] = l8;

	pt_P = Gg.count_T1(1, O->m - 1, O->q);
	pt_Q = pt_P + Gg.count_S(O->m - 1, O->q);

	for (j = 0; j < nb_line_classes; j++) {
		if (L[j] == 0) {
			for (i = 0; i < nb_point_classes; i++) {
				B[i * nb_line_classes + j] = 0;
			}
		}
	}
	if (f_v) {
		cout << "hyperbolic_pair::init_parabolic done" << endl;
	}
}

void hyperbolic_pair::init_parabolic_even(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "hyperbolic_pair::init_parabolic_even" << endl;
	}
	if (O->m >= 2) {
		beta = Gg.count_T1(0, O->m - 2, O->q);
	}
	else {
		beta = 0;
	}
	if (O->m >= 1) {
		alpha = Gg.count_T1(0, O->m - 1, O->q);
		gamma = alpha * beta / (O->q + 1);
	}
	else {
		alpha = 0;
		gamma = 0;
	}
	delta = alpha - 1 - O->q * beta;
	zeta = alpha - beta - 2 * (O->q - 1) * beta - O->q - 1;
	//cout << "alpha = " << alpha << endl;
	//cout << "beta = " << beta << endl;
	//cout << "gamma = " << gamma << endl;
	//cout << "delta = " << delta << endl;
	//cout << "zeta = " << zeta << endl;

	p1 = O->q - 1;
	p2 = alpha * (O->q - 1) * (O->q - 1);
	p3 = p4 = (O->q - 1) * alpha;
	p5 = alpha;
	p6 = p7 = 1;

	l1 = alpha * (O->q - 1);
	l2 = (O->q - 1) * (O->q - 1) * alpha * beta;
	l3 = (O->q - 1) * alpha * delta;
	l4 = l5 = alpha * beta * (O->q - 1);
	l6 = gamma;
	l7 = l8 = alpha;

	a11 = alpha;
	a21 = a36 = a47 = a56 = a57 = 1;
	a22a = a33 = a44 = O->q * beta;
	a22b = a32b = a42b = delta;
	a51 = O->q - 1;
	a52a = zeta;
	a53 = a54 = (O->q - 1) * beta;
	a55 = beta;
	a66 = a77 = alpha;

	b11 = b51 = b52a = b32b = b42b = b53 = b54 = b56 = b57 = b66 = b77 = 1;
	b21 = b22b = b36 = b47 = O->q - 1;
	b22a = b33 = b44 = O->q;
	b55 = O->q + 1;


	fill(A, 1, 1, a11);
	fill(A, 2, 1, a21);
	fill(A, 5, 1, a51);

	fill(A, 2, 2, a22a);
	fill(A, 5, 2, a52a);

	fill(A, 2, 3, a22b);
	fill(A, 3, 3, a32b);
	fill(A, 4, 3, a42b);

	fill(A, 3, 4, a33);
	fill(A, 5, 4, a53);

	fill(A, 4, 5, a44);
	fill(A, 5, 5, a54);

	fill(A, 5, 6, a55);

	fill(A, 3, 7, a36);
	fill(A, 5, 7, a56);
	fill(A, 6, 7, a66);

	fill(A, 4, 8, a47);
	fill(A, 5, 8, a57);
	fill(A, 7, 8, a77);

	fill(B, 1, 1, b11);
	fill(B, 2, 1, b21);
	fill(B, 5, 1, b51);

	fill(B, 2, 2, b22a);
	fill(B, 5, 2, b52a);

	fill(B, 2, 3, b22b);
	fill(B, 3, 3, b32b);
	fill(B, 4, 3, b42b);

	fill(B, 3, 4, b33);
	fill(B, 5, 4, b53);

	fill(B, 4, 5, b44);
	fill(B, 5, 5, b54);

	fill(B, 5, 6, b55);

	fill(B, 3, 7, b36);
	fill(B, 5, 7, b56);
	fill(B, 6, 7, b66);

	fill(B, 4, 8, b47);
	fill(B, 5, 8, b57);
	fill(B, 7, 8, b77);
	if (f_v) {
		cout << "hyperbolic_pair::init_parabolic_even done" << endl;
	}
}

void hyperbolic_pair::init_parabolic_odd(int verbose_level)
{
	long int a, b, c;
	int f_v = (verbose_level >= 1);
	geometry::geometry_global Gg;


	int q, m;

	m = O->m;
	q = O->q;

	if (f_v) {
		cout << "hyperbolic_pair::init_parabolic_odd" << endl;
		cout << "count_N1(" << m - 1 << "," << q << ")=";
		cout << Gg.count_N1(m - 1, q) << endl;
		cout << "count_S(" << m - 1 << "," << q << ")=";
		cout << Gg.count_S(m - 1, q) << endl;
	}
	a = Gg.count_N1(m - 1, q) * (q - 1) / 2;
	b = Gg.count_S(m - 1, q) * (q - 1);
	c = (((q - 1) / 2) - 1) * (q - 1) * Gg.count_N1(O->m - 1, q);
	p1 = a + b + c;
	p2 = a + ((q - 1) / 2) * (q - 1) * Gg.count_N1(O->m - 1, q);
	if (f_v) {
		cout << "a=" << a << endl;
		cout << "b=" << b << endl;
		cout << "c=" << c << endl;
		cout << "p1=" << p1 << endl;
		cout << "p2=" << p2 << endl;
	}

	if (m >= 2) {
		beta = Gg.count_T1(0, m - 2, q);
	}
	else {
		beta = 0;
	}
	if (m >= 1) {
		alpha = Gg.count_T1(0, m - 1, q);
		gamma = alpha * beta / (q + 1);
	}
	else {
		alpha = 0;
		gamma = 0;
	}
	if (f_v) {
		cout << "alpha=" << alpha << endl;
		cout << "beta=" << beta << endl;
		cout << "gamma=" << gamma << endl;
	}
	p3 = p4 = (q - 1) * alpha;
	p5 = alpha;
	p6 = p7 = 1;
	if (f_v) {
		cout << "p3=" << p3 << endl;
		cout << "p5=" << p5 << endl;
		cout << "p6=" << p6 << endl;
	}

	omega = (q - 1) * Gg.count_S(m - 2, q) +
		Gg.count_N1(m - 2, q) * (q - 1) / 2 +
		Gg.count_N1(m - 2, q) * ((q - 1) / 2 - 1) * (q - 1);
	if (f_v) {
		cout << "omega=" << omega << endl;
	}
	zeta = alpha - omega - 2 * (q - 1) * beta - beta - 2;
	if (f_v) {
		cout << "zeta=" << zeta << endl;
	}


	a66 = a77 = alpha;
	a56 = a57 = a36 = a47 = 1;
	a55 = beta;
	a53 = a54 = (q - 1) * beta;
	a33 = a44 = q * beta;
	a32b = a42b = alpha - 1 - q * beta;
	a51 = omega;
	a52a = zeta;

	l1 = p5 * omega;
	l2 = p5 * zeta;
	l3 = (q - 1) * alpha * (alpha - 1 - q * beta);
	l4 = l5 = (q - 1) * alpha * beta;
	l6 = gamma;
	l7 = l8 = alpha;

	if (f_v) {
		cout << "l1=" << l1 << endl;
		cout << "l2=" << l2 << endl;
		cout << "l3=" << l3 << endl;
		cout << "l4=" << l4 << endl;
		cout << "l5=" << l5 << endl;
		cout << "l6=" << l6 << endl;
		cout << "l7=" << l7 << endl;
		cout << "l8=" << l8 << endl;
	}

	if (p1) {
		lambda = l1 * q / p1;
	}
	else {
		lambda = 0;
	}
	if (p2) {
		delta = l2 * q / p2;
	}
	else {
		delta = 0;
	}
	a11 = lambda;
	a22a = delta;
	a12b = alpha - lambda;
	a22b = alpha - delta;
	mu = alpha - lambda;
	nu = alpha - delta;
	a12b = mu;
	a22b = nu;

	b51 = b52a = b32b = b42b = b53 = b54 = b56 = b57 = b66 = b77 = 1;
	b11 = b22a = b33 = b44 = q;
	b55 = q + 1;
	b36 = b47 = q - 1;
	if (l3) {
		b12b = p1 * mu / l3;
		b22b = p2 * nu / l3;
	}
	else {
		b12b = 0;
		b22b = 0;
	}


	fill(A, 1, 1, a11);
	fill(A, 5, 1, a51);

	fill(A, 2, 2, a22a);
	fill(A, 5, 2, a52a);

	fill(A, 1, 3, a12b);
	fill(A, 2, 3, a22b);
	fill(A, 3, 3, a32b);
	fill(A, 4, 3, a42b);

	fill(A, 3, 4, a33);
	fill(A, 5, 4, a53);

	fill(A, 4, 5, a44);
	fill(A, 5, 5, a54);

	fill(A, 5, 6, a55);

	fill(A, 3, 7, a36);
	fill(A, 5, 7, a56);
	fill(A, 6, 7, a66);

	fill(A, 4, 8, a47);
	fill(A, 5, 8, a57);
	fill(A, 7, 8, a77);

	fill(B, 1, 1, b11);
	fill(B, 5, 1, b51);

	fill(B, 2, 2, b22a);
	fill(B, 5, 2, b52a);

	fill(B, 1, 3, b12b);
	fill(B, 2, 3, b22b);
	fill(B, 3, 3, b32b);
	fill(B, 4, 3, b42b);

	fill(B, 3, 4, b33);
	fill(B, 5, 4, b53);

	fill(B, 4, 5, b44);
	fill(B, 5, 5, b54);

	fill(B, 5, 6, b55);

	fill(B, 3, 7, b36);
	fill(B, 5, 7, b56);
	fill(B, 6, 7, b66);

	fill(B, 4, 8, b47);
	fill(B, 5, 8, b57);
	fill(B, 7, 8, b77);


	O->SN = NEW_OBJECT(field_theory::square_nonsquare);

	if (f_v) {
		cout << "hyperbolic_pair::init_parabolic_odd before SN->init" << endl;
	}
	O->SN->init(O->F, verbose_level - 2);
	if (f_v) {
		cout << "hyperbolic_pair::init_parabolic_odd after SN->init" << endl;
	}

	if (f_v) {
		cout << "hyperbolic_pair::init_parabolic_odd done" << endl;
	}
}


void hyperbolic_pair::init_hyperbolic(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "hyperbolic_pair::init_hyperbolic" << endl;
	}

	int q, m;

	m = O->m;
	q = O->q;

	nb_point_classes = 6;
	nb_line_classes = 7;
	subspace_point_type = 4;
	subspace_line_type = 5;

	p5 = p6 = 1;
	p4 = T1_mm1;
	p2 = p3 = (q - 1) * T1_mm1;
	p1 = NT.i_power_j_lint(q, 2 * m - 2) - 1 - p2;
	l6 = l7 = T1_mm1;
	l5 = T2_mm1;
	l3 = l4 = (q - 1) * T1_mm2 * T1_mm1;

	alpha = T1_mm1;
	beta = T1_mm2;
	gamma = alpha * beta / (q + 1);

	a47 = a46 = a37 = a26 = 1;
	b67 = b56 = b47 = b46 = b44 = b43 = b41 = b32 = b22 = 1;
	b45 = q + 1;
	b37 = b26 = b12 = q - 1;
	b34 = b23 = b11 = q;
	a67 = a56 = T1_mm1;
	a45 = T1_mm2;
	a44 = a43 = T1_mm2 * (q - 1);

	a41 = (q - 1) * N1_mm2;

	a34 = q * T1_mm2;
	a23 = q * T1_mm2;
	a32 = a22 = T1_mm1 - 1 - a23;

	l2 = p2 * a22;
	if (p1 == 0) {
		//cout << "orthogonal::init_hyperbolic p1 == 0" << endl;
		a12 = 0;
	}
	else {
		//a12 = l2 * (q - 1) / p1;
		a12 = NT.number_theory_domain::ab_over_c(l2, q - 1, p1);
	}
	a11 = T1_mm1 - a12;



	//l1 = a11 * p1 / q;

	l1 = NT.number_theory_domain::ab_over_c(a11, p1, q);
	if (f_v) {
		cout << "hyperbolic_pair::init_hyperbolic a11 = " << a11 << endl;
		cout << "hyperbolic_pair::init_hyperbolic p1 = " << p1 << endl;
		cout << "hyperbolic_pair::init_hyperbolic l1 = " << l1 << endl;
	}


#if 0
	if (l1 * q != a11 * p1) {
		cout << "orthogonal::init_hyperbolic l1 * q != a11 * p1, overflow" << endl;
		exit(1);
	}
#endif

	//a41 = l1 / T1_mm1;


	A = NEW_lint(6 * 7);
	B = NEW_lint(6 * 7);
	P = NEW_lint(6);
	L = NEW_lint(7);

	for (i = 0; i < 6 * 7; i++) {
		A[i] = B[i] = 0;
	}
	P[0] = p1;
	P[1] = p2;
	P[2] = p3;
	P[3] = p4;
	P[4] = p5;
	P[5] = p6;
	L[0] = l1;
	L[1] = l2;
	L[2] = l3;
	L[3] = l4;
	L[4] = l5;
	L[5] = l6;
	L[6] = l7;
	fill(A, 1, 1, a11);
	fill(A, 1, 2, a12);
	fill(A, 2, 2, a22);
	fill(A, 2, 3, a23);
	fill(A, 2, 6, a26);
	fill(A, 3, 2, a32);
	fill(A, 3, 4, a34);
	fill(A, 3, 7, a37);
	fill(A, 4, 1, a41);
	fill(A, 4, 3, a43);
	fill(A, 4, 4, a44);
	fill(A, 4, 5, a45);
	fill(A, 4, 6, a46);
	fill(A, 4, 7, a47);
	fill(A, 5, 6, a56);
	fill(A, 6, 7, a67);

	fill(B, 1, 1, b11);
	fill(B, 1, 2, b12);
	fill(B, 2, 2, b22);
	fill(B, 2, 3, b23);
	fill(B, 2, 6, b26);
	fill(B, 3, 2, b32);
	fill(B, 3, 4, b34);
	fill(B, 3, 7, b37);
	fill(B, 4, 1, b41);
	fill(B, 4, 3, b43);
	fill(B, 4, 4, b44);
	fill(B, 4, 5, b45);
	fill(B, 4, 6, b46);
	fill(B, 4, 7, b47);
	fill(B, 5, 6, b56);
	fill(B, 6, 7, b67);

	pt_P = p4;
	pt_Q = p4 + 1 + p3;

	if (f_v) {
		cout << "hyperbolic_pair::init_hyperbolic done" << endl;
	}
}

void hyperbolic_pair::fill(long int *M, int i, int j, long int a)
{
	M[(i - 1) * nb_line_classes + j - 1] = a;
}

void hyperbolic_pair::print_schemes()
{
	int i, j;


	cout << "       ";
	for (j = 0; j < nb_line_classes; j++) {
		cout << setw(7) << L[j];
	}
	cout << endl;
	for (i = 0; i < nb_point_classes; i++) {
		cout << setw(7) << P[i];
		for (j = 0; j < nb_line_classes; j++) {
			cout << setw(7) << A[i * nb_line_classes + j];
		}
		cout << endl;
	}
	cout << endl;
	cout << "       ";
	for (j = 0; j < nb_line_classes; j++) {
		cout << setw(7) << L[j];
	}
	cout << endl;
	for (i = 0; i < nb_point_classes; i++) {
		cout << setw(7) << P[i];
		for (j = 0; j < nb_line_classes; j++) {
			cout << setw(7) << B[i * nb_line_classes + j];
		}
		cout << endl;
	}
	cout << endl;

}


}}}

