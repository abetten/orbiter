// a5_in_PSL.cpp
//
// Anton Betten, Evi Haberberger
// 10.06.2000
//
// moved here from D2: 3/18/2010

#include "orbiter.h"

using namespace std;


using namespace orbiter;




void A5_in_PSL_(int q, int verbose_level);
void A5_in_PSL_2_q(int q,
		discreta_matrix & A, discreta_matrix & B, domain *dom_GFq, int verbose_level);
void A5_in_PSL_2_q_easy(int q,
		discreta_matrix & A, discreta_matrix & B, domain *dom_GFq, int verbose_level);
void A5_in_PSL_2_q_hard(int q,
		discreta_matrix & A, discreta_matrix & B, domain *dom_GFq, int verbose_level);
int proj_order(discreta_matrix &A);
void trace(discreta_matrix &A, discreta_base &tr);
void elementwise_power_int(discreta_matrix &A, int k);
int is_in_center(discreta_matrix &B);
void matrix_convert_to_numerical(discreta_matrix &A, int *AA, int q);


void print_usage()
{
	cout << "usage: a5_in_PSL.out [options] -q <q>" << endl;
	cout << "where options can be:" << endl;
	cout << "-v <verbose_level" << endl;
	cout << "   : verbose level" << endl;
}


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_q = FALSE;
	int q = 0;
	
	if (argc <= 1) {
		print_usage();
		exit(1);
		}
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		}

	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}
	
	os_interface Os;

	int t0 = Os.os_ticks();
	
		
	A5_in_PSL_(q, verbose_level);
	
	
	Os.time_check(cout, t0);
}


void A5_in_PSL_(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int p, f;
	discreta_matrix A, B, D; //, B1, B2, C, D, A2, A3, A4;
	number_theory_domain NT;
	
	
	NT.factor_prime_power(q, p, f);
	domain *dom;

	if (f_v) {
		cout << "a5_in_psl.out: "
				"q=" << q << ", p=" << p << ", f=" << f << endl;
		}
	dom = allocate_finite_field_domain(q, verbose_level);
	
	A5_in_PSL_2_q(q, A, B, dom, verbose_level);
	
	{
	with w(dom);
	D.mult(A, B);
	
	if (f_v) {
		cout << "finished with A5_in_PSL_2_q()" << endl;
		cout << "A=\n" << A << endl;
		cout << "B=\n" << B << endl;
		cout << "AB=\n" << D << endl;
		int AA[4], BB[4], DD[4];
		matrix_convert_to_numerical(A, AA, q);
		matrix_convert_to_numerical(B, BB, q);
		matrix_convert_to_numerical(D, DD, q);
		cout << "A=" << endl;
		print_integer_matrix_width(cout, AA, 2, 2, 2, 7);
		cout << "B=" << endl;
		print_integer_matrix_width(cout, BB, 2, 2, 2, 7);
		cout << "AB=" << endl;
		print_integer_matrix_width(cout, DD, 2, 2, 2, 7);
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
}

void A5_in_PSL_2_q(int q,
		discreta_matrix & A, discreta_matrix & B, domain *dom_GFq, int verbose_level)
{
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
}

void A5_in_PSL_2_q_easy(int q,
		discreta_matrix & A, discreta_matrix & B, domain *dom_GFq, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, r;
	integer zeta5, zeta5v, b, c, d, b2, e;
	
	if (f_v) {
		cout << "A5_in_PSL_2_q_easy verbose_level=" << verbose_level << endl;
		}
	with w(dom_GFq);
	
	i = (q - 1) / 5;
	r = finite_field_domain_primitive_root();
	zeta5.m_i(r);
	zeta5.power_int(i);
	zeta5v = zeta5;
	zeta5v.power_int(4);
	
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
	b.invert();
	
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
}


void A5_in_PSL_2_q_hard(int q,
		discreta_matrix & A, discreta_matrix & B, domain *dom_GFq, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	with w(dom_GFq);
	unipoly m;
	int i, q2;
	discreta_matrix S, Sv, E, /*Sbart, SSbart,*/ AA, BB;
	integer a, b, m1;
	int norm_alpha, l;
	
#if 0
	m.get_an_irreducible_polynomial(2, verbose_level);
#else
	m.Singer(q, 2, verbose_level);
#endif
	cout << "m=" << m << endl;
	norm_alpha = m.s_ii(0);
	cout << "norm_alpha=" << norm_alpha << endl;
	
	domain GFq2(&m, dom_GFq);
	with ww(&GFq2);
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
	a.power_int((q - 1) >> 1);
	b = a;
	b.power_int(q + 1);
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
	Sv.invert();
	E.mult(S, Sv);
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
	integer zeta5, zeta5v;
	
	i = (q2 - 1) / 5;
	r = finite_field_domain_primitive_root();
	zeta5.m_i(r);
	zeta5.power_int(i);
	zeta5v = zeta5;
	zeta5v.power_int(4);
	
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

	integer bb, c, d, e, f, c1, b1;
	
	// b := (zeta5 - zeta5^{-1})^{-1}:
	b = zeta5v;
	b.negate();
	b += zeta5;
	b.invert();
	
	if (f_v) {
		cout << "b=" << b << endl;
		}

	// compute $c$ with $N(c) = c \cdot \bar{c} = 1 - N(b) = 1 - b \cdot \bar{b}$:
	b1 = b;
	b1.power_int(q);

	bb.mult(b, b1);
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
		f.power_int(l);
		if (f.compare_with(e) == 0)
			break;
		}
	if (f_v) {
		cout << "the discrete log with respect to " << norm_alpha << " is " << l << endl;
		}
	c.m_i(q);
	c.power_int(l);
	
	f = c;
	f.power_int(q + 1);
	if (f.compare_with(e) != 0) {
		cout << "fatal: norm of " << c << " is not " << e << endl;
		exit(1);
		}	
#else
	for (i = q; i < q2; i++) {
		c.m_i(i);
		f = c;
		f.power_int(q + 1);
		if (f.compare_with(e) == 0)
			break;
		}
	if (i == q2) {
		cout << "A5_in_PSL_2_q_hard() couldn't find element c" << endl;
		exit(1);
		}
#endif
	if (f_v) {
		cout << "element c=" << c << endl;
		}
	c1 = c;
	c1.power_int(q);
	
	BB[0][0] = b;
	BB[0][1] = c;
	BB[1][0] = c1;
	BB[1][0].negate();
	BB[1][1] = b1;
	if (f_v) {
		cout << "BB=\n" << BB << endl;
		}
	A.mult(S, AA);
	A *= Sv;
	B.mult(S, BB);
	B *= Sv;
	
	if (f_v) {
		cout << "A=\n" << A << endl;
		cout << "B=\n" << B << endl;
		}
}

int proj_order(discreta_matrix &A)
{
	discreta_matrix B;
	int m, n;
	int ord;
	
	m = A.s_m();
	n = A.s_n();
	if (m != n)
	{
		cout << "matrix::proj_order_mod() m != n" << endl;
		exit(1);
	}
	if (A.is_zero())
	{
		ord = 0;
		cout << "is zero matrix!" << endl;
	}
	else
	{
		B = A;
		ord = 1;
		while (is_in_center(B) == FALSE)
		{
			ord++;
			B *= A;
		}
	}
	return ord;
}

void trace(discreta_matrix &A, discreta_base &tr)
{
	int i, m, n;
	
	m = A.s_m();
	n = A.s_n();
	if (m != n)
	{
		cout << "ERROR: matrix::trace(): no square matrix!" << endl;
		exit(1);
	}
	tr = A[0][0];
	for (i = 1; i < m; i++)
	{
		tr += A[i][i];
	}
}

void elementwise_power_int(discreta_matrix &A, int k)
{
	int i, j, m, n;
	
	m = A.s_m();
	n = A.s_n();
	
	for (i=0; i < m; i++)
	{
		for (j=0; j < n; j++)
		{
			A[i][j].power_int(k);
		}
	}
}

int is_in_center(discreta_matrix &B)
{
	int m, n, i, j;
	discreta_matrix A;
	integer c;
	
	m = B.s_m();
	n = B.s_n();
	A = B;
	c = A[0][0];
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			integer e;
			
			e = A[i][j];
			if (i != j && !e.is_zero())
			{
				return FALSE;
			}
			if (i == j && e.s_i() != c.s_i())
			{
				return FALSE;
			}
		}
	}
	return TRUE;
}


void matrix_convert_to_numerical(discreta_matrix &A, int *AA, int q)
{
	int m, n, i, j, /*h, l,*/ val;
	
	m = A.s_m();
	n = A.s_n();
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {

			//cout << "i=" << i << " j=" << j << endl;
			discreta_base a;

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


