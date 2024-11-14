/*
 * group_constructions_global.cpp
 *
 *  Created on: Nov 2, 2024
 *      Author: betten
 */





#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace group_constructions {


group_constructions_global::group_constructions_global()
{
}

group_constructions_global::~group_constructions_global()
{
}


// a5_in_PSL.cpp
//
// Anton Betten, Evi Haberberger
// 10.06.2000
//
// moved here from D2: 3/18/2010

void group_constructions_global::A5_in_PSL_(
		int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int p, f;
	typed_objects::discreta_matrix A, B, D; //, B1, B2, C, D, A2, A3, A4;
	number_theory::number_theory_domain NT;


	NT.factor_prime_power(q, p, f);
	typed_objects::domain *dom;

	if (f_v) {
		cout << "group_constructions_global::A5_in_PSL_ "
				"q=" << q << ", p=" << p << ", f=" << f << endl;
	}
	dom = typed_objects::allocate_finite_field_domain(q, verbose_level);

	A5_in_PSL_2_q(q, A, B, dom, verbose_level);

	{
		typed_objects::with w(dom);
		D.mult(A, B, verbose_level);

		if (f_v) {
			cout << "A5_in_PSL_2_q done" << endl;
			cout << "A=\n" << A << endl;
			cout << "B=\n" << B << endl;
			cout << "AB=\n" << D << endl;
			int AA[4], BB[4], DD[4];
			matrix_convert_to_numerical(A, AA, q);
			matrix_convert_to_numerical(B, BB, q);
			matrix_convert_to_numerical(D, DD, q);
			cout << "A=" << endl;
			Int_vec_print_integer_matrix_width(cout, AA, 2, 2, 2, 7);
			cout << "B=" << endl;
			Int_vec_print_integer_matrix_width(cout, BB, 2, 2, 2, 7);
			cout << "AB=" << endl;
			Int_vec_print_integer_matrix_width(cout, DD, 2, 2, 2, 7);
		}

		int oA, oB, oD;

		oA = proj_order(A);
		oB = proj_order(B);
		oD = proj_order(D);
		if (f_v) {
			cout << "projective order of A = " << oA << endl;
			cout << "projective order of B = " << oB << endl;
			cout << "projective order of AB = " << oD << endl;
		}


	}
	free_finite_field_domain(dom);
	if (f_v) {
		cout << "group_constructions_global::A5_in_PSL_ done" << endl;
	}
}

void group_constructions_global::A5_in_PSL_2_q(
		int q,
		orbiter::layer2_discreta::typed_objects::discreta_matrix & A,
		orbiter::layer2_discreta::typed_objects::discreta_matrix & B,
		orbiter::layer2_discreta::typed_objects::domain *dom_GFq,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_constructions_global::A5_in_PSL_2_q" << endl;
	}
	if (((q - 1) % 5) == 0) {
		A5_in_PSL_2_q_easy(q, A, B, dom_GFq, verbose_level);
	}
	else if (((q + 1) % 5) == 0) {
		A5_in_PSL_2_q_hard(q, A, B, dom_GFq, verbose_level);
	}
	else {
		cout << "either q + 1 or q - 1 must be divisible by 5!" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "group_constructions_global::A5_in_PSL_2_q done" << endl;
	}
}

void group_constructions_global::A5_in_PSL_2_q_easy(
		int q,
		orbiter::layer2_discreta::typed_objects::discreta_matrix & A,
		orbiter::layer2_discreta::typed_objects::discreta_matrix & B,
		orbiter::layer2_discreta::typed_objects::domain *dom_GFq,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, r;
	typed_objects::integer zeta5, zeta5v, b, c, d, b2, e;

	if (f_v) {
		cout << "group_constructions_global::A5_in_PSL_2_q_easy "
				"verbose_level=" << verbose_level << endl;
	}
	typed_objects::with w(dom_GFq);

	i = (q - 1) / 5;
	r = typed_objects::finite_field_domain_primitive_root();
	zeta5.m_i(r);
	zeta5.power_int(i, verbose_level);
	zeta5v = zeta5;
	zeta5v.power_int(4, verbose_level);

	if (f_v) {
		cout << "zeta5=" << zeta5 << endl;
		cout << "zeta5v=" << zeta5v << endl;
	}

	A.m_mn_n(2, 2);
	B.m_mn_n(2, 2);
	A[0][0] = zeta5;
	A[0][1].zero();
	A[1][0].zero();
	A[1][1] = zeta5v;

	if (f_v) {
		cout << "A=\n" << A << endl;
	}

	// b := (zeta5 - zeta5^{-1})^{-1}:
	b = zeta5v;
	b.negate();
	b += zeta5;
	b.invert(verbose_level);

	// determine c, d such that $-b^2 -cd = 1$:
	b2 = b;
	b2 *= b;
	b2.negate();
	e.m_one();
	e += b2;
	c.one();
	d = e;
	B[0][0] = b;
	B[0][1] = c;
	B[1][0] = d;
	B[1][1] = b;
	B[1][1].negate();

	if (f_v) {
		cout << "B=\n" << B << endl;
	}
	if (f_v) {
		cout << "group_constructions_global::A5_in_PSL_2_q_easy done" << endl;
	}
}


void group_constructions_global::A5_in_PSL_2_q_hard(
		int q,
		orbiter::layer2_discreta::typed_objects::discreta_matrix & A,
		orbiter::layer2_discreta::typed_objects::discreta_matrix & B,
		orbiter::layer2_discreta::typed_objects::domain *dom_GFq,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	typed_objects::with w(dom_GFq);
	typed_objects::unipoly m;
	int i, q2;
	typed_objects::discreta_matrix S, Sv, E, /*Sbart, SSbart,*/ AA, BB;
	typed_objects::integer a, b, m1;
	int norm_alpha, l;

	if (f_v) {
		cout << "group_constructions_global::A5_in_PSL_2_q_hard" << endl;
	}
#if 0
	m.get_an_irreducible_polynomial(2, verbose_level);
#else
	m.Singer(q, 2, verbose_level);
#endif
	cout << "m=" << m << endl;
	norm_alpha = m.s_ii(0);
	cout << "norm_alpha=" << norm_alpha << endl;

	typed_objects::domain GFq2(&m, dom_GFq);
	typed_objects::with ww(&GFq2);
	q2 = q * q;

	if (f_v) {
		cout << "searching for element of norm -1:" << endl;
	}
	S.m_mn_n(2, 2);
	m1.m_one();
	if (f_v) {
		cout << "-1=" << m1 << endl;
	}
#if 0
	for (i = q; i < q2; i++) {
		// cout << "i=" << i;
		a.m_i(i);
		b = a;
		b.power_int(q + 1);
		cout << i << ": (" << a << ")^" << q + 1 << " = " << b << endl;
		if (b.is_m_one())
			break;
		}
	if (i == q2) {
		cout << "A5_in_PSL_2_q_hard() couldn't find element of norm -1" << endl;
		exit(1);
		}
#else
	a.m_i(q); // alpha
	a.power_int((q - 1) >> 1, verbose_level);
	b = a;
	b.power_int(q + 1, verbose_level);
	cout << "(" << a << ")^" << q + 1 << " = " << b << endl;
	if (!b.is_m_one()) {
		cout << "fatal: element a does not have norm -1" << endl;
		exit(1);
	}
#endif
	if (f_v) {
		cout << "element of norm -1:" << a << endl;
	}
#if 1
	S[0][0] = a;
	S[0][1].one();
	S[1][0].one();
	S[1][0].negate();
	S[1][1] = a;
#else
	// Huppert I page 105 (does not work!)
	S[0][0].one();
	S[0][1] = a;
	S[1][0].one();
	S[1][1] = a;
	S[1][1].negate();
#endif
	if (f_v) {
		cout << "S=\n" << S << endl;
	}
	Sv = S;
	Sv.invert(verbose_level);
	E.mult(S, Sv, verbose_level);
	if (f_v) {
		cout << "S^{-1}=\n" << Sv << endl;
		cout << "S \\cdot S^{-1}=\n" << E << endl;
	}

#if 0
	Sbart = S;
	elementwise_power_int(Sbart, q);
	Sbart.transpose();
	SSbart.mult(S, Sbart);
	if (f_v) {
		cout << "\\bar{S}^\\top=\n" << Sbart << endl;
		cout << "S \\cdot \\bar{S}^\\top=\n" << SSbart << endl;
		}
#endif

	int r;
	typed_objects::integer zeta5, zeta5v;

	i = (q2 - 1) / 5;
	r = typed_objects::finite_field_domain_primitive_root();
	zeta5.m_i(r);
	zeta5.power_int(i, verbose_level);
	zeta5v = zeta5;
	zeta5v.power_int(4, verbose_level);

	if (f_v) {
		cout << "zeta5=" << zeta5 << endl;
		cout << "zeta5v=" << zeta5v << endl;
	}

	AA.m_mn_n(2, 2);
	BB.m_mn_n(2, 2);
	AA[0][0] = zeta5;
	AA[0][1].zero();
	AA[1][0].zero();
	AA[1][1] = zeta5v;

	if (f_v) {
		cout << "AA=\n" << AA << endl;
	}

	typed_objects::integer bb, c, d, e, f, c1, b1;

	// b := (zeta5 - zeta5^{-1})^{-1}:
	b = zeta5v;
	b.negate();
	b += zeta5;
	b.invert(verbose_level);

	if (f_v) {
		cout << "b=" << b << endl;
	}

	// compute $c$ with $N(c) = c \cdot \bar{c} = 1 - N(b) = 1 - b \cdot \bar{b}$:
	b1 = b;
	b1.power_int(q, verbose_level);

	bb.mult(b, b1, verbose_level);
	bb.negate();
	e.one();
	e += bb;
	if (f_v) {
		cout << "1 - b \\cdot \\bar{b}=" << e << endl;
	}
#if 1
	for (l = 0; l < q; l++) {
		c.m_i(norm_alpha);
		f = c;
		f.power_int(l, verbose_level);
		if (f.compare_with(e) == 0) {
			break;
		}
	}
	if (f_v) {
		cout << "the discrete log with respect to " << norm_alpha << " is " << l << endl;
	}
	c.m_i(q);
	c.power_int(l, verbose_level);

	f = c;
	f.power_int(q + 1, verbose_level);
	if (f.compare_with(e) != 0) {
		cout << "fatal: norm of " << c << " is not " << e << endl;
		exit(1);
	}
#else
	for (i = q; i < q2; i++) {
		c.m_i(i);
		f = c;
		f.power_int(q + 1);
		if (f.compare_with(e) == 0) {
			break;
		}
	}
	if (i == q2) {
		cout << "A5_in_PSL_2_q_hard couldn't find element c" << endl;
		exit(1);
	}
#endif
	if (f_v) {
		cout << "element c=" << c << endl;
	}
	c1 = c;
	c1.power_int(q, verbose_level);

	BB[0][0] = b;
	BB[0][1] = c;
	BB[1][0] = c1;
	BB[1][0].negate();
	BB[1][1] = b1;
	if (f_v) {
		cout << "BB=\n" << BB << endl;
	}
	A.mult(S, AA, verbose_level);
	A *= Sv;
	B.mult(S, BB, verbose_level);
	B *= Sv;

	if (f_v) {
		cout << "A=\n" << A << endl;
		cout << "B=\n" << B << endl;
	}
	if (f_v) {
		cout << "group_constructions_global::A5_in_PSL_2_q_hard done" << endl;
	}
}

int group_constructions_global::proj_order(
		orbiter::layer2_discreta::typed_objects::discreta_matrix &A)
{
	typed_objects::discreta_matrix B;
	int m, n;
	int ord;

	m = A.s_m();
	n = A.s_n();
	if (m != n) {
		cout << "group_constructions_global::proj_order m != n" << endl;
		exit(1);
	}
	if (A.is_zero()) {
		ord = 0;
		cout << "is zero matrix!" << endl;
	}
	else {
		B = A;
		ord = 1;
		while (is_in_center(B) == false) {
			ord++;
			B *= A;
		}
	}
	return ord;
}

void group_constructions_global::trace(
		orbiter::layer2_discreta::typed_objects::discreta_matrix &A,
		orbiter::layer2_discreta::typed_objects::discreta_base &tr)
{
	int i, m, n;

	m = A.s_m();
	n = A.s_n();
	if (m != n) {
		cout << "ERROR: matrix::trace not a square matrix!" << endl;
		exit(1);
	}
	tr = A[0][0];
	for (i = 1; i < m; i++) {
		tr += A[i][i];
	}
}

void group_constructions_global::elementwise_power_int(
		orbiter::layer2_discreta::typed_objects::discreta_matrix &A,
		int k,
		int verbose_level)
{
	int i, j, m, n;

	m = A.s_m();
	n = A.s_n();

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			A[i][j].power_int(k, verbose_level);
		}
	}
}

int group_constructions_global::is_in_center(
		orbiter::layer2_discreta::typed_objects::discreta_matrix &B)
{
	int m, n, i, j;
	typed_objects::discreta_matrix A;
	typed_objects::integer c;

	m = B.s_m();
	n = B.s_n();
	A = B;
	c = A[0][0];
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			typed_objects::integer e;

			e = A[i][j];
			if (i != j && !e.is_zero()) {
				return false;
			}
			if (i == j && e.s_i() != c.s_i()) {
				return false;
			}
		}
	}
	return true;
}


void group_constructions_global::matrix_convert_to_numerical(
		orbiter::layer2_discreta::typed_objects::discreta_matrix &A,
		int *AA, int q)
{
	int m, n, i, j, val;

	m = A.s_m();
	n = A.s_n();
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {

			//cout << "i=" << i << " j=" << j << endl;
			typed_objects::discreta_base a;

			A[i][j].copyobject_to(a);

			//cout << "a=" << a << endl;
			//a.printobjectkindln(cout);

			val = a.s_i_i();
#if 0
			l = a.as_unipoly().s_l();
			cout << "degree=" << l << endl;
			for (h = l - 1; h >= 0; h--) {
				val *= q;
				cout << "coeff=" << a.as_unipoly().s_ii(h) << endl;
				val += a.as_unipoly().s_ii(h);
				}
#endif
			//cout << "val=" << val << endl;
			AA[i * n + j] = val;
		}
	}
}




}}}


