/*
 * finite_field_orthogonal.cpp
 *
 *  Created on: Apr 12, 2019
 *      Author: betten
 */





#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {



void finite_field::Q_epsilon_unrank(
	int *v, int stride, int epsilon, int k,
	int c1, int c2, int c3, long int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::Q_epsilon_unrank" << endl;
	}
	if (epsilon == 0) {
		if (f_v) {
			cout << "finite_field::Q_epsilon_unrank before Q_unrank" << endl;
		}
		Q_unrank(v, stride, k, a, verbose_level);
		if (f_v) {
			cout << "finite_field::Q_epsilon_unrank after Q_unrank" << endl;
		}
	}
	else if (epsilon == 1) {
		if (f_v) {
			cout << "finite_field::Q_epsilon_unrank before Qplus_unrank" << endl;
		}
		Qplus_unrank(v, stride, k, a, verbose_level);
		if (f_v) {
			cout << "finite_field::Q_epsilon_unrank after Qplus_unrank" << endl;
		}
	}
	else if (epsilon == -1) {
		if (f_v) {
			cout << "finite_field::Q_epsilon_unrank before Qminus_unrank" << endl;
		}
		Qminus_unrank(v, stride, k, a, c1, c2, c3, verbose_level);
		if (f_v) {
			cout << "finite_field::Q_epsilon_unrank after Qminus_unrank" << endl;
		}
	}
	else {
		cout << "Q_epsilon_unrank epsilon is wrong" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "finite_field::Q_epsilon_unrank done" << endl;
	}
}

long int finite_field::Q_epsilon_rank(
	int *v, int stride, int epsilon, int k,
	int c1, int c2, int c3, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int a;

	if (f_v) {
		cout << "finite_field::Q_epsilon_rank" << endl;
	}
	if (epsilon == 0) {
		if (f_v) {
			cout << "finite_field::Q_epsilon_rank before Q_rank" << endl;
		}
		a = Q_rank(v, stride, k, verbose_level);
		if (f_v) {
			cout << "finite_field::Q_epsilon_rank after Q_rank" << endl;
		}
	}
	else if (epsilon == 1) {
		if (f_v) {
			cout << "finite_field::Q_epsilon_rank before Qplus_rank" << endl;
		}
		a = Qplus_rank(v, stride, k, verbose_level);
		if (f_v) {
			cout << "finite_field::Q_epsilon_rank after Qplus_rank" << endl;
		}
	}
	else if (epsilon == -1) {
		if (f_v) {
			cout << "finite_field::Q_epsilon_rank before Qminus_rank" << endl;
		}
		a = Qminus_rank(v, stride, k, c1, c2, c3, verbose_level);
		if (f_v) {
			cout << "finite_field::Q_epsilon_rank after Qminus_rank" << endl;
		}
	}
	else {
		cout << "Q_epsilon_unrank epsilon is wrong" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "finite_field::Q_epsilon_rank done" << endl;
	}
	return a;
}

#if 0
// old version
void Q_unrank(finite_field &GFq, int *v, int stride, int k, int a)
// k = projective dimension, must be even
{
	int n, x, i, minusone;

	n = Witt_index(0, k);
	x = nb_pts_Sbar(n, GFq.q);
	if (a < x) {
		v[2 * n * stride] = 0;
		Sbar_unrank(GFq, v, stride, n, a);
		return;
		}
	a -= x;
	v[2 * n * stride] = 1;
	N1_unrank(GFq, v, stride, n, a);
	minusone = GFq.negate(1);
	if (minusone != 1) {
		for (i = 0; i < n; i++) {
			v[2 * i * stride] = GFq.mult(v[2 * i * stride], minusone);
			}
		}
}

int Q_rank(finite_field &GFq, int *v, int stride, int k)
// k = projective dimension, must be even
{
	int n, x, a, b, i, minusone;

	n = Witt_index(0, k);
	x = nb_pts_Sbar(n, GFq.q);
	if (v[2 * n * stride] == 0) {
		Sbar_rank(GFq, v, stride, n, a);
		return a;
		}
	a = x;
	if (v[2 * n * stride] != 1) {
		PG_element_normalize(GFq, v, stride, k + 1);
		}
	minusone = GFq.negate(1);
	if (minusone != 1) {
		for (i = 0; i < n; i++) {
			v[2 * i * stride] = GFq.mult(v[2 * i * stride], minusone);
			}
		}
	N1_rank(GFq, v, stride, n, b);
	return a + b;
}
#endif


#if 0
vector_hashing *Hash_table_parabolic = NULL;
int Hash_table_parabolic_q = 0;
int Hash_table_parabolic_k = 0;

void finite_field::init_hash_table_parabolic(int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int n, ln2q, N, i, j;
	int *v;
	number_theory_domain NT;
	geometry_global Gg;

	if (f_v) {
		cout << "finite_field::init_hash_table_parabolic" << endl;
		cout << "q=" << q << endl;
		cout << "k=" << k << endl;
		}

	ln2q = NT.int_log2(q);
	Hash_table_parabolic = NEW_OBJECT(vector_hashing);
	Hash_table_parabolic_q = q;
	Hash_table_parabolic_k = k;
	n = k + 1;
	N = Gg.nb_pts_Q(k, q);
	if (f_v) {
		cout << "N=" << N << endl;
		}
	Hash_table_parabolic->allocate(n, N, ln2q);
	for (i = 0; i < N; i++) {
		v = Hash_table_parabolic->vector_data + i * n;
		Q_unrank_directly(v, 1 /* stride */, k, i, verbose_level);
		for (j = 0; j < k + 1; j++) {
			if (v[j])
				break;
			}
		if (v[j] != 1) {
			cout << "finite_field::init_hash_table_parabolic vector "
					"is not normalized" << endl;
			cout << "i=" << i << endl;
			int_vec_print(cout, v, k + 1);
			cout << endl;
			exit(1);
			}
		}
	Hash_table_parabolic->compute_tables(verbose_level - 1);

}
#endif

void finite_field::Q_unrank(int *v, int stride, int k, long int a, int verbose_level)
{

#if 0
	if (Hash_table_parabolic) {
		if (Hash_table_parabolic_q == q &&
				Hash_table_parabolic_k == k) {
			if (stride != 1) {
				cout << "Q_unrank with Hash table "
						"needs stride == 1" << endl;
				exit(1);
				}
			Hash_table_parabolic->unrank(a, v);
			return;
			}
		}
#endif
	Q_unrank_directly(v, stride, k, a, verbose_level);
}

long int finite_field::Q_rank(int *v, int stride, int k, int verbose_level)
{
#if 0
	if (Hash_table_parabolic) {
		PG_element_normalize_from_front(v, stride, k + 1);
		if (Hash_table_parabolic_q == q &&
				Hash_table_parabolic_k == k) {
			if (stride != 1) {
				cout << "Q_unrank with Hash table "
						"needs stride == 1" << endl;
				exit(1);
				}
			return Hash_table_parabolic->rank(v);
			}
		}
#endif
	return Q_rank_directly(v, stride, k, verbose_level);
}

void finite_field::Q_unrank_directly(int *v, int stride, int k, long int a, int verbose_level)
// parabolic quadric
// k = projective dimension, must be even
{
	int n, i, minusone;
	long int x;
	geometry_global Gg;

	n = Gg.Witt_index(0, k);
	x = Gg.nb_pts_Sbar(n, q);
	if (a < x) {
		v[0] = 0;
		Sbar_unrank(v + stride, stride, n, a, verbose_level);
		PG_element_normalize_from_front(v + stride, stride, k);
		return;
		}
	a -= x;
	v[0] = 1;
	N1_unrank(v + stride, stride, n, a);
	minusone = negate(1);
	if (minusone != 1) {
		for (i = 0; i < n; i++) {
			v[(1 + 2 * i) * stride] =
				mult(v[(1 + 2 * i) * stride], minusone);
			}
		}
}

long int finite_field::Q_rank_directly(int *v, int stride, int k, int verbose_level)
// parabolic quadric
// k = projective dimension, must be even
{
	int n, i;
	long int x, a, b;
	int minusone;
	geometry_global Gg;

	n = Gg.Witt_index(0, k);
	x = Gg.nb_pts_Sbar(n, q);
	if (v[0] == 0) {
		Sbar_rank(v + stride, stride, n, a, verbose_level);
		return a;
		}
	a = x;
	if (v[0] != 1) {
		PG_element_normalize_from_front(v, stride, k + 1);
		}
	minusone = negate(1);
	if (minusone != 1) {
		for (i = 0; i < n; i++) {
			v[(1 + 2 * i) * stride] =
				mult(v[(1 + 2 * i) * stride], minusone);
			}
		}
	N1_rank(v + stride, stride, n, b);
	return a + b;
}

void finite_field::Qplus_unrank(int *v, int stride, int k, long int a, int verbose_level)
// hyperbolic quadric
// k = projective dimension, must be odd
{
	int n;
	geometry_global Gg;

	n = Gg.Witt_index(1, k);
	Sbar_unrank(v, stride, n, a, verbose_level);
}

long int finite_field::Qplus_rank(int *v, int stride, int k, int verbose_level)
// hyperbolic quadric
// k = projective dimension, must be odd
{
	int f_v = (verbose_level >= 1);
	int n;
	long int a;
	geometry_global Gg;

	if (f_v) {
		cout << "finite_field::Qplus_rank" << endl;
	}
	n = Gg.Witt_index(1, k);
	Sbar_rank(v, stride, n, a, verbose_level);
	return a;
}

void finite_field::Qminus_unrank(int *v,
		int stride, int k, long int a,
		int c1, int c2, int c3, int verbose_level)
// elliptic quadric
// k = projective dimension, must be odd
// the form is
// \sum_{i=0}^n x_{2i}x_{2i+1} + c1 x_{2n}^2 +
// c2 x_{2n} x_{2n+1} + c3 x_{2n+1}^2
{
	int n, z, minusz, u, vv, w, i;
	long int x, b, c, x1, x2;
	geometry_global Gg;

	n = Gg.Witt_index(-1, k);
	x = Gg.nb_pts_Sbar(n, q);
	if (a < x) {
		v[2 * n * stride] = 0;
		v[(2 * n + 1) * stride] = 0;
		Sbar_unrank(v, stride, n, a, verbose_level);
		return;
	}
	a -= x;
	x = Gg.nb_pts_N1(n, q);
	b = a / x;
	c = a % x;
	// b determines an element on the projective line
	if (b == 0) {
		x1 = 1;
		x2 = 0;
	}
	else {
		b--;
		x1 = b;
		x2 = 1;
		if (b >= q) {
			cout << "finite_field::Qminus_unrank "
					"b >= q, the rank was too big" << endl;
			exit(1);
		}
	}
	v[2 * n * stride] = x1;
	v[(2 * n + 1) * stride] = x2;
	u = product3(c1, x1, x1);
	vv = product3(c2, x1, x2);
	w = product3(c3, x2, x2);
	z = add3(u, vv, w);
	if (z == 0) {
		cout << "Qminus_unrank z = 0" << endl;
		cout << "b=" << b << endl;
		cout << "c1=" << c1 << endl;
		cout << "c2=" << c2 << endl;
		cout << "c3=" << c3 << endl;
		cout << "x1=" << x1 << endl;
		cout << "x2=" << x2 << endl;
		cout << "u=c1*x1*x1=" << u << endl;
		cout << "vv=c2*x1*x2=" << vv << endl;
		cout << "w=c3*x2*x2=" << w << endl;
		exit(1);
	}
	N1_unrank(v, stride, n, c);
	minusz = negate(z);
	if (minusz != 1) {
		for (i = 0; i < n; i++) {
			v[2 * i * stride] = mult(v[2 * i * stride], minusz);
		}
	}
}

long int finite_field::Qminus_rank(int *v,
		int stride, int k, int c1, int c2, int c3, int verbose_level)
// elliptic quadric
// k = projective dimension, must be odd
// the form is
// \sum_{i=0}^n x_{2i}x_{2i+1} + c1 x_{2n}^2 +
// c2 x_{2n} x_{2n+1} + c3 x_{2n+1}^2
{
	int n, minusz, minuszv;
	long int a, b, c, x, x1, x2, u, vv, w, z, i;
	geometry_global Gg;

	n = Gg.Witt_index(-1, k);

	{
		int aa;
		aa = evaluate_quadratic_form(v, stride, -1, k, c1, c2, c3);
		if (aa) {
			cout << "Qminus_rank fatal: the vector "
					"is not zero under the quadratic form" << endl;
			cout << "value=" << aa << endl;
			cout << "stride=" << stride << endl;
			cout << "k=" << k << endl;
			cout << "c1=" << c1 << endl;
			cout << "c2=" << c2 << endl;
			cout << "c3=" << c3 << endl;
			Orbiter->Int_vec.print(cout, v, k + 1);
			cout << endl;
			exit(1);
		}
	}
	PG_element_normalize(v, stride, k + 1);
	x1 = v[2 * n * stride];
	x2 = v[(2 * n + 1) * stride];
	if (x1 == 0 && x2 == 0) {
		Sbar_rank(v, stride, n, a, verbose_level);
		return a;
		}
	a = Gg.nb_pts_Sbar(n, q);
	// determine b from an element on the projective line
	if (x1 == 1 && x2 == 0) {
		b = 0;
		}
	else {
		if (x2 != 1) {
			cout << "Qminus_rank x2 != 1" << endl;
			exit(1);
			}
		b = x1 + 1;
		}

	x = Gg.nb_pts_N1(n, q);
	//b = a / x;
	//c = a % x;
	u = product3(c1, x1, x1);
	vv = product3(c2, x1, x2);
	w = product3(c3, x2, x2);
	z = add3(u, vv, w);
	if (z == 0) {
		cout << "Qminus_rank z = 0" << endl;
		cout << "b=" << b << endl;
		exit(1);
		}

	minusz = negate(z);
	minuszv = inverse(minusz);
	if (minuszv != 1) {
		for (i = 0; i < n; i++) {
			v[2 * i * stride] = mult(v[2 * i * stride], minuszv);
			}
		}

	N1_rank(v, stride, n, c);
	a += b * x + c;
	return a;
}





// #############################################################################
// unrank functions for the hyperbolic quadric:
// #############################################################################



void finite_field::S_unrank(int *v, int stride, int n, long int a)
{
	long int l, i, j, x, y, u;
	int alpha, beta;
	geometry_global Gg;

	if (n == 1) {
		if (a < q) {
			v[0 * stride] = a;
			v[1 * stride] = 0;
			return;
			}
		a -= (q - 1);
		if (a < q) {
			v[0 * stride] = 0;
			v[1 * stride] = a;
			return;
			}
		else {
			cout << "finite_field::S_unrank "
					"error in S_unrank n = 1 a = " << a << endl;
			exit(1);
			}
		}
	else {
		x = Gg.nb_pts_S(1, q);
		y = Gg.nb_pts_S(n - 1, q);
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			S_unrank(v + (n - 1) * 2 * stride, stride, 1, i);
			S_unrank(v, stride, n - 1, j);
			return;
			}
		a -= l;
		//cout << "S_unrank subtracting " << l
		//<< " to bring a down to " << a << endl;
		x = Gg.nb_pts_N(1, q);
		y = Gg.nb_pts_N1(n - 1, q);
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			N_unrank(v + (n - 1) * 2 * stride, stride, 1, i);
			N1_unrank(v, stride, n - 1, j);

			alpha = mult(v[2 * (n - 1) * stride],
					v[(2 * (n - 1) + 1) * stride]);
			beta = negate(alpha);
			for (u = 0; u < n - 1; u++) {
				v[2 * u * stride] = mult(v[2 * u * stride], beta);
				}
			return;
			}
		else {
			cout << "finite_field::S_unrank "
					"error in S_unrank n = " << n << ", a = " << a << endl;
			exit(1);
			}
		}
}

void finite_field::S_rank(int *v, int stride, int n, long int &a)
{
	long int l, i, j, x, y, u;
	int alpha, beta, gamma, delta, epsilon;
	geometry_global Gg;

	if (n == 1) {
		if (v[1 * stride] == 0) {
			a = v[0 * stride];
			return;
			}
		if (v[0 * stride]) {
			cout << "finite_field::S_rank "
					"error in S_rank v[0] not null" << endl;
			exit(1);
			}
		a = q - 1;
		a += v[1 * stride];
		}
	else {
		x = Gg.nb_pts_S(1, q);
		y = Gg.nb_pts_S(n - 1, q);
		l = x * y;
		alpha = mult(v[2 * (n - 1) * stride],
				v[(2 * (n - 1) + 1) * stride]);
		if (alpha == 0) {
			S_rank(v + (n - 1) * 2 * stride, stride, 1, i);
			S_rank(v, stride, n - 1, j);
			a = i * y + j;
			return;
			}
		a = l;
		x = Gg.nb_pts_N(1, q);
		y = Gg.nb_pts_N1(n - 1, q);

		N_rank(v + (n - 1) * 2 * stride, stride, 1, i);


		beta = negate(alpha);
		gamma = evaluate_hyperbolic_quadratic_form(
				v, stride, n - 1);
		if (gamma != beta) {
			cout << "finite_field::S_rank "
					"error in S_rank gamma != beta" << endl;
			exit(1);
			}
		delta = inverse(beta);
		for (u = 0; u < n - 1; u++) {
			v[2 * u * stride] = mult(v[2 * u * stride], delta);
			}
		epsilon = evaluate_hyperbolic_quadratic_form(
				v, stride, n - 1);
		if (epsilon != 1) {
			cout << "finite_field::S_rank "
					"error in S_rank epsilon != 1" << endl;
			exit(1);
			}
		N1_rank(v, stride, n - 1, j);
		a += i * y + j;
		}
}

void finite_field::N_unrank(int *v, int stride, int n, long int a)
{
	long int l, i, j, k, j1, x, y, z, yz, u;
	int alpha, beta, gamma, delta, epsilon;
	geometry_global Gg;

	if (n == 1) {
		x = q - 1;
		y = q - 1;
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			v[0 * stride] = 1 + j;
			v[1 * stride] = 1 + i;
			return;
			}
		else {
			cout << "finite_field::N_unrank "
					"error in N_unrank n = 1 a = " << a << endl;
			exit(1);
			}
		}
	else {
		x = Gg.nb_pts_S(1, q);
		y = Gg.nb_pts_N(n - 1, q);
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			S_unrank(v + (n - 1) * 2 * stride, stride, 1, i);
			N_unrank(v, stride, n - 1, j);
			return;
			}
		a -= l;
		x = Gg.nb_pts_N(1, q);
		y = Gg.nb_pts_S(n - 1, q);
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			N_unrank(v + (n - 1) * 2 * stride, stride, 1, i);
			S_unrank(v, stride, n - 1, j);
			return;
			}
		a -= l;
		x = Gg.nb_pts_N(1, q);
		y = (q - 2);
		z = Gg.nb_pts_N1(n - 1, q);
		yz = y * z;
		l = x * yz;
		if (a < l) {
			i = a / yz;
			j1 = a % yz;
			j = j1 / z;
			k = j1 % z;
			N_unrank(v + (n - 1) * 2 * stride, stride, 1, i);
			N1_unrank(v, stride, n - 1, k);
			alpha = primitive_element();

			beta = power(alpha, j + 1);
			gamma = mult(v[(n - 1) * 2 * stride],
					v[((n - 1) * 2 + 1) * stride]);
			delta = negate(gamma);
			epsilon = mult(delta, beta);
			for (u = 0; u < n - 1; u++) {
				v[2 * u * stride] = mult(v[2 * u * stride], epsilon);
				}
			return;
			}
		else {
			cout << "finite_field::N_unrank "
					"error in N_unrank n = " << n << ", a = " << a << endl;
			exit(1);
			}
		}
}

void finite_field::N_rank(int *v, int stride, int n, long int &a)
{
	long int l, i, j, k, x, y, z, yz, u;
	int alpha, beta, gamma, delta;
	int epsilon, gamma2, epsilon_inv;
	geometry_global Gg;

	if (n == 1) {
		x = q - 1;
		y = q - 1;
		if (v[0 * stride] == 0 || v[1 * stride] == 0) {
			cout << "finite_field::N_rank "
					"v[0 * stride] == 0 || "
					"v[1 * stride] == 0" << endl;
			exit(1);
			}
		j = v[0 * stride] - 1;
		i = v[1 * stride] - 1;
		a = i * y + j;
		}
	else {
		gamma = mult(v[(n - 1) * 2 * stride],
				v[((n - 1) * 2 + 1) * stride]);
		x = Gg.nb_pts_S(1, q);
		y = Gg.nb_pts_N(n - 1, q);
		l = x * y;
		if (gamma == 0) {
			S_rank(v + (n - 1) * 2 * stride, stride, 1, i);
			N_rank(v, stride, n - 1, j);
			a = i * y + j;
			return;
			}
		a = l;
		x = Gg.nb_pts_N(1, q);
		y = Gg.nb_pts_S(n - 1, q);
		l = x * y;
		gamma2 = evaluate_hyperbolic_quadratic_form(
				v, stride, n - 1);
		if (gamma2 == 0) {
			N_rank(v + (n - 1) * 2, stride, 1, i);
			S_rank(v, stride, n - 1, j);
			a += i * y + j;
			}
		a += l;

		x = Gg.nb_pts_N(1, q);
		y = (q - 2);
		z = Gg.nb_pts_N1(n - 1, q);
		yz = y * z;
		l = x * yz;

		N_rank(v + (n - 1) * 2 * stride, stride, 1, i);
		alpha = primitive_element();
		delta = negate(gamma);
		for (j = 0; j < q - 2; j++) {
			beta = power(alpha, j + 1);
			epsilon = mult(delta, beta);
			if (epsilon == gamma2) {
				epsilon_inv = inverse(epsilon);
				for (u = 0; u < n - 1; u++) {
					v[2 * u * stride] = mult(
							v[2 * u * stride], epsilon_inv);
					}
				N1_rank(v, stride, n - 1, k);
				a += i * yz + j * z + k;
				return;
				}
			}
		cout << "finite_field::N_rank "
				"error, gamma2 not found" << endl;
		exit(1);
		}
}

void finite_field::N1_unrank(int *v, int stride, int n, long int a)
{
	long int l, i, j, k, j1, x, y, z, yz, u;
	int alpha, beta, gamma;
	geometry_global Gg;

	if (n == 1) {
		l = q - 1;
		if (a < l) {
			alpha = a + 1;
			beta = inverse(alpha);
			//cout << "finite_field::N1_unrank n == 1, a = " << a
			// << " alpha = " << alpha << " beta = " << beta << endl;
			v[0 * stride] = alpha;
			v[1 * stride] = beta;
			return;
			}
		else {
			cout << "finite_field::N1_unrank "
					"error in N1_unrank n = 1 a = " << a << endl;
			exit(1);
			}
		}
	else {
		x = Gg.nb_pts_S(1, q);
		y = Gg.nb_pts_N1(n - 1, q);
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			S_unrank(v + (n - 1) * 2 * stride, stride, 1, i);
			N1_unrank(v, stride, n - 1, j);
			return;
			}
		a -= l;
		//cout << "finite_field::N1_unrank subtracting " << l
		// << " to bring a down to " << a << endl;
		x = Gg.nb_pts_N1(1, q);
		y = Gg.nb_pts_S(n - 1, q);
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			N1_unrank(v + (n - 1) * 2 * stride, stride, 1, i);
			S_unrank(v, stride, n - 1, j);
			return;
			}
		a -= l;
		//cout << "N1_unrank subtracting " << l
		// << " to bring a down to " << a << endl;
		x = Gg.nb_pts_N1(1, q);
		y = (q - 2); // zero for q = 2
		z = Gg.nb_pts_N1(n - 1, q);
		yz = y * z;
		l = x * yz; // zero for q = 2
		if (a < l) {
			// the case q = 2 does not appear here any more
			i = a / yz;
			j1 = a % yz;
			j = j1 / z;
			k = j1 % z;

			//cout << "a = " << a << endl;
			//cout << "y = " << y << endl;
			//cout << "z = " << z << endl;
			//cout << "i = a / yz = " << i << endl;
			//cout << "j1 = a % yz = " << j1 << endl;
			//cout << "j = j1 / z = " << j << endl;
			//cout << "k = j1 % z = " << k << endl;

			N1_unrank(v + (n - 1) * 2 * stride, stride, 1, i);

			//cout << "(" << v[2 * (n - 1) * stride] << ","
			// << v[(2 * (n - 1) + 1) * stride] << ")" << endl;

			alpha = 2 + j;
			v[2 * (n - 1) * stride] = mult(
					v[2 * (n - 1) * stride], alpha);

			N1_unrank(v, stride, n - 1, k);

			//int_set_print(v, 2 * (n - 1));
			//cout << endl;

			beta = negate(alpha);
			gamma = add(beta, 1);

			//cout << "alpha = j + 2 = " << alpha << endl;
			//cout << "beta = - alpha = " << beta << endl;
			//cout << "gamma = beta + 1 = " << gamma << endl;

			for (u = 0; u < n - 1; u++) {
				v[2 * u * stride] = mult(v[2 * u * stride], gamma);
				}

			//int_set_print(v, 2 * n);
			//cout << endl;
			return;
			}
		else {
			cout << "finite_field::N1_unrank "
					"error in N1_unrank n = " << n << ", a = " << a << endl;
			exit(1);
			}
		}
}

void finite_field::N1_rank(int *v, int stride, int n, long int &a)
{
	long int l, i, j, k, x, y, z, yz, u;
	int alpha, alpha_inv, beta, gamma, gamma2, gamma_inv;
	geometry_global Gg;

	if (n == 1) {
		alpha = v[0 * stride];
		beta = v[1 * stride];
		if (alpha == 0 || beta == 0) {
			cout << "finite_field::N1_rank "
					"alpha == 0 || beta == 0" << endl;
			exit(1);
			}
		gamma = inverse(alpha);
		if (gamma != beta) {
			cout << "finite_field::N1_rank "
					"error in N1_rank gamma = " << gamma
					<< " != beta = " << beta << endl;
			exit(1);
			}
		a = alpha - 1;
		}
	else {
		a = 0;
		alpha = mult(v[2 * (n - 1) * stride],
				v[(2 * (n - 1) + 1) * stride]);
		x = Gg.nb_pts_S(1, q);
		y = Gg.nb_pts_N1(n - 1, q);
		l = x * y;
		if (alpha == 0) {
			S_rank(v + (n - 1) * 2 * stride, stride, 1, i);
			N1_rank(v, stride, n - 1, j);
			a = i * y + j;
			return;
			}
		a += l;
		gamma2 = evaluate_hyperbolic_quadratic_form(
				v, stride, n - 1);
		x = Gg.nb_pts_N1(1, q);
		y = Gg.nb_pts_S(n - 1, q);
		l = x * y;
		if (gamma2 == 0) {
			N1_rank(v + (n - 1) * 2 * stride, stride, 1, i);
			S_rank(v, stride, n - 1, j);
			a += i * y + j;
			return;
			}
		a += l;
		// the case q = 2 does not appear here any more
		if (q == 2) {
			cout << "finite_field::N1_rank "
					"the case q=2 should not appear here" << endl;
			exit(1);
			}


		x = Gg.nb_pts_N1(1, q);
		y = (q - 2); // zero for q = 2
		z = Gg.nb_pts_N1(n - 1, q);
		yz = y * z;
		l = x * yz; // zero for q = 2

		alpha = mult(v[2 * (n - 1) * stride],
				v[(2 * (n - 1) + 1) * stride]);
		if (alpha == 0) {
			cout << "N1_rank alpha == 0" << endl;
			exit(1);
			}
		if (alpha == 1) {
			cout << "N1_rank alpha == 1" << endl;
			exit(1);
			}
		j = alpha - 2;
		alpha_inv = inverse(alpha);
		v[2 * (n - 1) * stride] = mult(
				v[2 * (n - 1) * stride], alpha_inv);

		N1_rank(v + (n - 1) * 2 * stride, stride, 1, i);

		gamma2 = evaluate_hyperbolic_quadratic_form(
				v, stride, n - 1);
		if (gamma2 == 0) {
			cout << "finite_field::N1_rank "
					"gamma2 == 0" << endl;
			exit(1);
			}
		if (gamma2 == 1) {
			cout << "finite_field::N1_rank "
					"gamma2 == 1" << endl;
			exit(1);
			}
		gamma_inv = inverse(gamma2);
		for (u = 0; u < n - 1; u++) {
			v[2 * u * stride] = mult(v[2 * u * stride], gamma_inv);
			}
		N1_rank(v, stride, n - 1, k);

		a += i * yz + j * z + k;

		}
}

void finite_field::Sbar_unrank(int *v, int stride, int n, long int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int l, i, j, x, y, u;
	int alpha, beta;
	geometry_global Gg;

	if (f_v) {
		cout << "finite_field::Sbar_unrank" << endl;
	}
	if (n == 1) {
		if (a == 0) {
			v[0 * stride] = 1;
			v[1 * stride] = 0;
			return;
			}
		if (a == 1) {
			v[0 * stride] = 0;
			v[1 * stride] = 1;
			return;
			}
		else {
			cout << "finite_field::Sbar_unrank "
					"error in Sbar_unrank n = 1 a = " << a << endl;
			exit(1);
			}
		}
	else {
		y = Gg.nb_pts_Sbar(n - 1, q);
		l = y;
		if (a < l) {
			u = n - 1;
			v[2 * u * stride] = 0;
			v[(2 * u + 1) * stride] = 0;
			Sbar_unrank(v, stride, n - 1, a, verbose_level);
			return;
			}
		if (f_v) {
			cout << "finite_field::Sbar_unrank a = " << a << endl;
			cout << "finite_field::Sbar_unrank l = " << l << endl;
		}
		a -= l;
		if (f_v) {
			cout << "finite_field::Sbar_unrank a = " << a << endl;
			}
		//cout << "subtracting " << l << " to bring a to " << a << endl;
		x = Gg.nb_pts_Sbar(1, q);
		y = Gg.nb_pts_S(n - 1, q);
		//cout << "nb_pts_S(" << n - 1 << ") = " << y << endl;
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			Sbar_unrank(v + (n - 1) * 2 * stride, stride, 1, i, verbose_level);
			S_unrank(v, stride, n - 1, j);
			return;
			}
		if (f_v) {
			cout << "finite_field::Sbar_unrank a = " << a << endl;
			cout << "finite_field::Sbar_unrank l = " << l << endl;
		}
		a -= l;
		if (f_v) {
			cout << "finite_field::Sbar_unrank a = " << a << endl;
			}
		//cout << "subtracting " << l << " to bring a to " << a << endl;
		x = Gg.nb_pts_Nbar(1, q);
		y = Gg.nb_pts_N1(n - 1, q);
		//cout << "nb_pts_N1(" << n - 1 << ") = " << y << endl;
		l = x * y;
		if (f_v) {
			cout << "finite_field::Sbar_unrank x = " << x << endl;
			cout << "finite_field::Sbar_unrank y = " << y << endl;
			cout << "finite_field::Sbar_unrank l = " << l << endl;
		}
		if (a < l) {
			i = a / y;
			j = a % y;
			//cout << "i=" << i << " j=" << j << endl;
			Nbar_unrank(v + (n - 1) * 2 * stride, stride, 1, i);
			//cout << "(" << v[2 * (n - 1) * stride] << ","
			//<< v[(2 * (n - 1) + 1) * stride] << ")" << endl;
			N1_unrank(v, stride, n - 1, j);

			alpha = mult(v[2 * (n - 1) * stride],
					v[(2 * (n - 1) + 1) * stride]);
			beta = negate(alpha);
			for (u = 0; u < n - 1; u++) {
				v[2 * u * stride] = mult(v[2 * u * stride], beta);
			}
			//int_set_print(v, 2 * n);
			//cout << endl;
			return;
			}
		else {
			cout << "finite_field::Sbar_unrank "
					"error in Sbar_unrank n = " << n
					<< ", a = " << a << endl;
			exit(1);
			}
		}
}

void finite_field::Sbar_rank(int *v, int stride, int n, long int &a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int l, i, j, x, y, u;
	int alpha, beta, beta2, beta_inv;
	geometry_global Gg;

	if (f_v) {
		cout << "finite_field::Sbar_rank" << endl;
	}
	PG_element_normalize(v, stride, 2 * n);
	if (f_v) {
		cout << "finite_field::Sbar_rank: ";
		if (stride == 1) {
			Orbiter->Int_vec.print(cout, v, 2 * n);
			cout << endl;
		}
	}
	if (n == 1) {
		// test for (1,0) or (0,1):
		if (v[0 * stride] == 1 && v[1 * stride] == 0) {
			a = 0;
			return;
			}
		if (v[0 * stride] == 0 && v[1 * stride] == 1) {
			a = 1;
			return;
			}
		else {
			cout << "finite_field::Sbar_rank "
					"error in Sbar_rank n = 1 bad vector" << endl;
			if (stride == 1) {
				Orbiter->Int_vec.print(cout, v, 2);
				}
			exit(1);
			}
		}
	else {
		a = 0;
		// test for leading (0,0):
		if (v[2 * (n - 1) * stride] == 0 &&
				v[(2 * (n - 1) + 1) * stride] == 0) {
			// rank Sbar for the rest:
			Sbar_rank(v, stride, n - 1, a, verbose_level);
			return;
			}
		l = Gg.nb_pts_Sbar(n - 1, q);
		if (f_v) {
			cout << "finite_field::Sbar_rank not a leading zero, l = " << l << endl;
		}
		a += l;

		// alpha = form value for the top two coefficients:
		alpha = mult(v[2 * (n - 1) * stride],
				v[(2 * (n - 1) + 1) * stride]);
		x = Gg.nb_pts_Sbar(1, q); // = 2
		y = Gg.nb_pts_S(n - 1, q);

		if (f_v) {
			cout << "finite_field::Sbar_rank alpha = " << alpha << endl;
			cout << "finite_field::Sbar_rank x = " << x << endl;
			cout << "finite_field::Sbar_rank y = " << y << endl;
		}

		// test for 0 + 0
		// (i.e. 0 = alpha = value of the form on
		// the top two coefficients
		// and 0 for value of the form on the rest):
		if (alpha == 0) {
			Sbar_rank(v + (n - 1) * 2 * stride, stride, 1, i, verbose_level);
			S_rank(v, stride, n - 1, j);
			a += i * y + j;
			//cout << "i*y+j=" << i << "*" << y << "+" << j << endl;
			return;
			}

		l = x * y;
		a += l;
		if (f_v) {
			cout << "finite_field::Sbar_rank l = " << l << endl;
			cout << "finite_field::Sbar_rank a = " << a << endl;
		}

		// now it must be n + (-n) for some n \neq 0
		// (i.e. n = alpha = value of the form on
		// the top two coefficients and
		// -n for the value of the form on the rest):
		x = Gg.nb_pts_Nbar(1, q);
		y = Gg.nb_pts_N1(n - 1, q);
		Nbar_rank(v + (n - 1) * 2 * stride, stride, 1, i);
		if (f_v) {
			cout << "finite_field::Sbar_rank x = " << x << endl;
			cout << "finite_field::Sbar_rank y = " << y << endl;
			cout << "finite_field::Sbar_rank i = " << i << endl;
		}

		beta = negate(alpha);
		// beta = - alpha
		beta2 = evaluate_hyperbolic_quadratic_form(
				v, stride, n - 1);
		// beta2 = value of the quadratic form on the rest
		// must be - alpha (otherwise the vector does
		// not represent a point in Sbar)
		if (beta2 != beta) {
			cout << "finite_field::Sbar_rank "
					"error in Sbar_rank beta2 != beta" << endl;
			exit(1);
			}
		beta_inv = inverse(beta);
		// divide by beta so that the quadratic form
		// on the rest is equal to 1.
		for (u = 0; u < n - 1; u++) {
			v[2 * u * stride] = mult(
					v[2 * u * stride], beta_inv);
			}
		// rank the N1 part:
		N1_rank(v, stride, n - 1, j);
		if (f_v) {
			cout << "finite_field::Sbar_rank j = " << j << endl;
		}
		a += i * y + j;
		if (f_v) {
			cout << "finite_field::Sbar_rank a = " << a << endl;
		}
	}
}

void finite_field::Nbar_unrank(int *v, int stride, int n, long int a)
{
	int y, l;

	if (n == 1) {
		y = q - 1;
		l = y;
		if (a < l) {
			v[0 * stride] = 1 + a;
			v[1 * stride] = 1;
			return;
			}
		else {
			cout << "finite_field::Nbar_unrank "
					"error in Nbar_unrank n = 1 a = " << a << endl;
			exit(1);
			}
		}
	else {
		cout << "finite_field::Nbar_unrank "
				"only defined for n = 1" << endl;
		exit(1);
		}
}

void finite_field::Nbar_rank(int *v, int stride, int n, long int &a)
{
	if (n == 1) {
		if (v[1 * stride] != 1) {
			cout << "error in Nbar_rank n = 1 v[1 * stride] != 1" << endl;
			exit(1);
			}
		if (v[0 * stride] == 0) {
			cout << "error in Nbar_rank n = 1 v[0 * stride] == 0" << endl;
			exit(1);
			}
		a = v[0 * stride] - 1;
		return;
		}
	else {
		cout << "finite_field::Nbar_rank only defined for n = 1" << endl;
		exit(1);
		}
}







void finite_field::Gram_matrix(int epsilon, int k,
	int form_c1, int form_c2, int form_c3,
	int *&Gram, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int d = k + 1;
	int n, i, j, u, offset = 0;
	geometry_global Gg;

	if (f_v) {
		cout << "finite_field::Gram_matrix" << endl;
	}
	Gram = NEW_int(d * d);
	Orbiter->Int_vec.zero(Gram, d * d);
	n = Gg.Witt_index(epsilon, k);
	if (epsilon == 0) {
		Gram[0 * d + 0] = add(form_c1, form_c1);
		offset = 1;
	}
	else if (epsilon == 1) {
	}
	else if (epsilon == -1) {
		Gram[(d - 2) * d + d - 2] = add(form_c1, form_c1);
		Gram[(d - 2) * d + d - 1] = form_c2;
		Gram[(d - 1) * d + d - 2] = form_c2;
		Gram[(d - 1) * d + d - 1] = add(form_c3, form_c3);
	}
	for (i = 0; i < n; i++) {
		j = 2 * i;
		u = offset + j;
		Gram[u * d + u + 1] = 1;
			// X_u * Y_{u+1}
		Gram[(u + 1) * d + u] = 1;
			// X_{u+1} * Y_u
	}
	if (f_v) {
		cout << "finite_field::Gram_matrix done" << endl;
	}
}

int finite_field::evaluate_bilinear_form(
		int *u, int *v, int d, int *Gram)
{
	int i, j, a, b, c, e, A;

	A = 0;
	for (i = 0; i < d; i++) {
		a = u[i];
		for (j = 0; j < d; j++) {
			b = Gram[i * d + j];
			c = v[j];
			e = mult(a, b);
			e = mult(e, c);
			A = add(A, e);
			}
		}
	return A;
}

int finite_field::evaluate_quadratic_form(int *v, int stride,
	int epsilon, int k, int form_c1, int form_c2, int form_c3)
{
	int n, a, b, c = 0, d, x, x1, x2;
	geometry_global Gg;

	n = Gg.Witt_index(epsilon, k);
	if (epsilon == 0) {
		a = evaluate_hyperbolic_quadratic_form(v + stride, stride, n);
		x = v[0];
		b = product3(form_c1, x, x);
		c = add(a, b);
		}
	else if (epsilon == 1) {
		c = evaluate_hyperbolic_quadratic_form(v, stride, n);
		}
	else if (epsilon == -1) {
		a = evaluate_hyperbolic_quadratic_form(v, stride, n);
		x1 = v[2 * n * stride];
		x2 = v[(2 * n + 1) * stride];
		b = product3(form_c1, x1, x1);
		c = product3(form_c2, x1, x2);
		d = product3(form_c3, x2, x2);
		c = add4(a, b, c, d);
		}
	return c;
}


int finite_field::evaluate_hyperbolic_quadratic_form(
		int *v, int stride, int n)
{
	int alpha = 0, beta, u;

	for (u = 0; u < n; u++) {
		beta = mult(v[2 * u * stride], v[(2 * u + 1) * stride]);
		alpha = add(alpha, beta);
		}
	return alpha;
}

int finite_field::evaluate_hyperbolic_bilinear_form(
		int *u, int *v, int n)
{
	int alpha = 0, beta1, beta2, i;

	for (i = 0; i < n; i++) {
		beta1 = mult(u[2 * i], v[2 * i + 1]);
		beta2 = mult(u[2 * i + 1], v[2 * i]);
		alpha = add(alpha, beta1);
		alpha = add(alpha, beta2);
		}
	return alpha;
}

int finite_field::primitive_element()
{
	number_theory_domain NT;

	if (e == 1) {
		return NT.primitive_root(p, FALSE);
		}
	return p;
}


void finite_field::Siegel_map_between_singular_points(int *T,
		long int rk_from, long int rk_to, long int root,
	int epsilon, int algebraic_dimension,
	int form_c1, int form_c2, int form_c3, int *Gram_matrix,
	int verbose_level)
// root is not perp to from and to.
{
	int *B, *Bv, *w, *z, *x;
	int i, j, a, b, av, bv, minus_one;
	int d, k; //, epsilon, form_c1, form_c2, form_c3;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "finite_field::Siegel_map_between_singular_points "
				"rk_from=" << rk_from
				<< " rk_to=" << rk_to
				<< " root=" << root << endl;
		}
	d = algebraic_dimension;
	k = d - 1;

	B = NEW_int(d * d);
	Bv = NEW_int(d * d);
	w = NEW_int(d);
	z = NEW_int(d);
	x = NEW_int(d);
	Q_epsilon_unrank(B, 1, epsilon, k,
			form_c1, form_c2, form_c3, root, 0 /* verbose_level */);
	Q_epsilon_unrank(B + d, 1, epsilon, k,
			form_c1, form_c2, form_c3, rk_from, 0 /* verbose_level */);
	Q_epsilon_unrank(w, 1, epsilon, k,
			form_c1, form_c2, form_c3, rk_to, 0 /* verbose_level */);
	if (f_vv) {
		cout << "    root=";
		Orbiter->Int_vec.print(cout, B, d);
		cout << endl;
		cout << " rk_from=";
		Orbiter->Int_vec.print(cout, B + d, d);
		cout << endl;
		cout << "   rk_to=";
		Orbiter->Int_vec.print(cout, w, d);
		cout << endl;
	}

	a = evaluate_bilinear_form(B, B + d, d, Gram_matrix);
	b = evaluate_bilinear_form(B, w, d, Gram_matrix);
	av = inverse(a);
	bv = inverse(b);
	for (i = 0; i < d; i++) {
		B[d + i] = mult(B[d + i], av);
		w[i] = mult(w[i], bv);
	}
	if (f_vv) {
		cout << "after scaling:" << endl;
		cout << " rk_from=";
		Orbiter->Int_vec.print(cout, B + d, d);
		cout << endl;
		cout << "   rk_to=";
		Orbiter->Int_vec.print(cout, w, d);
		cout << endl;
	}
	for (i = 2; i < d; i++) {
		for (j = 0; j < d; j++) {
			B[i * d + j] = 0;
		}
	}

	if (f_vv) {
		cout << "before perp, the matrix B is:" << endl;
		Orbiter->Int_vec.print_integer_matrix(cout, B, d, d);
	}
	Linear_algebra->perp(d, 2, B, Gram_matrix, 0 /* verbose_level */);
	if (f_vv) {
		cout << "after perp, the matrix B is:" << endl;
		Orbiter->Int_vec.print_integer_matrix(cout, B, d, d);
	}
	Linear_algebra->invert_matrix(B, Bv, d, 0 /* verbose_level */);
	if (f_vv) {
		cout << "the matrix Bv = B^{-1} is:" << endl;
		Orbiter->Int_vec.print_integer_matrix(cout, B, d, d);
	}
	Linear_algebra->mult_matrix_matrix(w, Bv, z, 1, d, d, 0 /* verbose_level */);
	if (f_vv) {
		cout << "the coefficient vector z = w * Bv is:" << endl;
		Orbiter->Int_vec.print(cout, z, d);
		cout << endl;
	}
	z[0] = 0;
	z[1] = 0;
	if (f_vv) {
		cout << "we zero out the first two coordinates:" << endl;
		Orbiter->Int_vec.print(cout, z, d);
		cout << endl;
	}
	Linear_algebra->mult_matrix_matrix(z, B, x, 1, d, d, 0 /* verbose_level */);
	if (f_vv) {
		cout << "the vector x = z * B is:" << endl;
		Orbiter->Int_vec.print(cout, x, d);
		cout << endl;
	}
	minus_one = negate(1);
	for (i = 0; i < d; i++) {
		x[i] = mult(x[i], minus_one);
	}
	if (f_vv) {
		cout << "the vector -x is:" << endl;
		Orbiter->Int_vec.print(cout, x, d);
		cout << endl;
	}
	Siegel_Transformation(epsilon, d - 1,
		form_c1, form_c2, form_c3, T, x, B,
		verbose_level - 2);
	if (f_v) {
		cout << "finite_field::Siegel_map_between_singular_points "
				"the Siegel transformation is:" << endl;
		Orbiter->Int_vec.print_integer_matrix(cout, T, d, d);
	}
	FREE_int(B);
	FREE_int(Bv);
	FREE_int(w);
	FREE_int(z);
	FREE_int(x);
}

void finite_field::Siegel_Transformation(
	int epsilon, int k,
	int form_c1, int form_c2, int form_c3,
	int *M, int *v, int *u, int verbose_level)
// if u is singular and v \in \la u \ra^\perp, then
// \pho_{u,v}(x) := x + \beta(x,v) u - \beta(x,u) v - Q(v) \beta(x,u) u
// is called the Siegel transform (see Taylor p. 148)
// Here Q is the quadratic form
// and \beta is the corresponding bilinear form
{
	int f_v = (verbose_level >= 1);
	int d = k + 1;
	int i, j, Qv, a, b, c, e;
	int *Gram;
	int *new_Gram;
	int *N1;
	int *N2;
	int *w;

	if (f_v) {
		cout << "finite_field::Siegel_Transformation "
				"v=";
		Orbiter->Int_vec.print(cout, v, d);
		cout << " u=";
		Orbiter->Int_vec.print(cout, u, d);
		cout << endl;
	}
	Gram_matrix(epsilon, k,
			form_c1, form_c2, form_c3, Gram, verbose_level);
	Qv = evaluate_quadratic_form(v, 1 /*stride*/,
			epsilon, k, form_c1, form_c2, form_c3);
	if (f_v) {
		cout << "Qv=" << Qv << endl;
	}
	N1 = NEW_int(d * d);
	N2 = NEW_int(d * d);
	new_Gram = NEW_int(d * d);
	w = NEW_int(d);
	for (i = 0; i < d; i++) {
		for (j = 0; j < d; j++) {
			if (i == j) {
				M[i * d + j] = 1;
			}
			else {
				M[i * d + j] = 0;
			}
		}
	}
	// compute w^T := Gram * v^T
	for (i = 0; i < d; i++) {
		a = 0;
		for (j = 0; j < d; j++) {
			b = Gram[i * d + j];
			c = v[j];
			e = mult(b, c);
			a = add(a, e);
		}
		w[i] = a;
	}
	// M := M + w^T * u
	for (i = 0; i < d; i++) {
		b = w[i];
		for (j = 0; j < d; j++) {
			c = u[j];
			e = mult(b, c);
			M[i * d + j] = add(M[i * d + j], e);
		}
	}
	// compute w^T := Gram * u^T
	for (i = 0; i < d; i++) {
		a = 0;
		for (j = 0; j < d; j++) {
			b = Gram[i * d + j];
			c = u[j];
			e = mult(b, c);
			a = add(a, e);
		}
		w[i] = a;
	}
	// M := M - w^T * v
	for (i = 0; i < d; i++) {
		b = w[i];
		for (j = 0; j < d; j++) {
			c = v[j];
			e = mult(b, c);
			M[i * d + j] = add(M[i * d + j], negate(e));
		}
	}
	// M := M - Q(v) * w^T * u
	for (i = 0; i < d; i++) {
		b = w[i];
		for (j = 0; j < d; j++) {
			c = u[j];
			e = mult(b, c);
			M[i * d + j] = add(M[i * d + j], mult(negate(e), Qv));
		}
	}
	if (f_v) {
		cout << "finite_field::Siegel_Transformation "
				"Siegel matrix:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout, M, d, d, d, 2);
		//GFq.transform_form_matrix(M, Gram, new_Gram, N1, N2, d);
		//cout << "transformed Gram matrix:" << endl;
		//print_integer_matrix_width(cout, new_Gram, d, d, d, 2);
		//cout << endl;
	}

	FREE_int(Gram);
	FREE_int(new_Gram);
	FREE_int(N1);
	FREE_int(N2);
	FREE_int(w);
}


long int finite_field::orthogonal_find_root(int rk2,
	int epsilon, int algebraic_dimension,
	int form_c1, int form_c2, int form_c3, int *Gram_matrix,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *x, *y, *z;
	int d, k, i;
	//int epsilon, d, k, form_c1, form_c2, form_c3, i;
	int y2_minus_y3, minus_y1, y3_minus_y2, a, a2, u, v;
	long int root;

	d = algebraic_dimension;
	k = d - 1;
	if (f_v) {
		cout << "finite_field::orthogonal_find_root "
				"rk2=" << rk2 << endl;
	}
	if (rk2 == 0) {
		cout << "finite_field::orthogonal_find_root: "
				"rk2 must not be 0" << endl;
		exit(1);
	}
	//epsilon = orthogonal_epsilon;
	//d = orthogonal_d;
	//k = d - 1;
	//form_c1 = orthogonal_form_c1;
	//form_c2 = orthogonal_form_c2;
	//form_c3 = orthogonal_form_c3;
	x = NEW_int(d);
	y = NEW_int(d);
	z = NEW_int(d);
	for (i = 0; i < d; i++) {
		x[i] = 0;
		z[i] = 0;
	}
	x[0] = 1;

	Q_epsilon_unrank(y, 1, epsilon, k,
			form_c1, form_c2, form_c3, rk2, 0 /* verbose_level */);
	if (y[0]) {
		z[1] = 1;
		goto finish;
	}
	if (y[1] == 0) {
		for (i = 2; i < d; i++) {
			if (y[i]) {
				if (EVEN(i)) {
					z[1] = 1;
					z[i + 1] = 1;
					goto finish;
				}
				else {
					z[1] = 1;
					z[i - 1] = 1;
					goto finish;
				}
			}
		}
		cout << "finite_field::orthogonal_find_root "
				"error: y is zero vector" << endl;
	}
	y2_minus_y3 = add(y[2], negate(y[3]));
	minus_y1 = negate(y[1]);
	if (minus_y1 != y2_minus_y3) {
		z[0] = 1;
		z[1] = 1;
		z[2] = negate(1);
		z[3] = 1;
		goto finish;
	}
	y3_minus_y2 = add(y[3], negate(y[2]));
	if (minus_y1 != y3_minus_y2) {
		z[0] = 1;
		z[1] = 1;
		z[2] = 1;
		z[3] = negate(1);
		goto finish;
	}
	// now we are in characteristic 2
	if (q == 2) {
		if (y[2] == 0) {
			z[1] = 1;
			z[2] = 1;
			goto finish;
		}
		else if (y[3] == 0) {
			z[1] = 1;
			z[3] = 1;
			goto finish;
		}
		cout << "finite_field::orthogonal_find_root "
				"error neither y2 nor y3 is zero" << endl;
		exit(1);
	}
	// now the field has at least 4 elements
	a = 3;
	a2 = mult(a, a);
	z[0] = a2;
	z[1] = 1;
	z[2] = a;
	z[3] = a;
finish:

	u = evaluate_bilinear_form(z, x, d, Gram_matrix);
	if (u == 0) {
		cout << "u=" << u << endl;
		exit(1);
	}
	v = evaluate_bilinear_form(z, y, d, Gram_matrix);
	if (v == 0) {
		cout << "v=" << v << endl;
		exit(1);
	}
	root = Q_epsilon_rank(z, 1, epsilon, k,
			form_c1, form_c2, form_c3, 0 /* verbose_level */);
	if (f_v) {
		cout << "finite_field::orthogonal_find_root "
				"root=" << root << endl;
	}

	FREE_int(x);
	FREE_int(y);
	FREE_int(z);

	return root;
}

void finite_field::choose_anisotropic_form(
		int &c1, int &c2, int &c3, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	unipoly_domain FX(this);
	unipoly_object m;

	if (f_v) {
		cout << "finite_field::choose_anisotropic_form "
				"over GF(" << q << ")" << endl;
		}

	if (ODD(q)) {
		c1 = 1;
		c2 = 0;
		c3 = negate(primitive_element());
	}
	else {
		algebra_global Algebra;

		FX.create_object_by_rank_string(m,
				Algebra.get_primitive_polynomial(q, 2, 0),
				verbose_level);

		//FX.create_object_by_rank_string(m,
		//get_primitive_polynomial(GFq.p, 2 * GFq.e, 0), verbose_level);

		if (f_v) {
			cout << "finite_field::choose_anisotropic_form "
					"choosing the following primitive polynomial:" << endl;
			FX.print_object(m, cout); cout << endl;
		}

		int *rep = (int *) m;
		int *coeff = rep + 1;
		c1 = coeff[2];
		c2 = coeff[1];
		c3 = coeff[0];
	}

#if 0
	finite_field GFQ;

	GFQ.init(GFq.q * GFq.q, 0);
	cout << "finite_field::choose_anisotropic_form "
			"choose_anisotropic_form created field GF("
			<< GFQ.q << ")" << endl;

	c1 = 1;
	c2 = GFQ.negate(GFQ.T2(GFQ.p));
	c3 = GFQ.N2(GFQ.p);
	if (f_v) {
		cout << "c1=" << c1 << " c2=" << c2 << " c3=" << c3 << endl;
		}

	c2 = GFQ.retract(GFq, 2, c2, verbose_level);
	c3 = GFQ.retract(GFq, 2, c3, verbose_level);
	if (f_v) {
		cout << "after retract:" << endl;
		cout << "c1=" << c1 << " c2=" << c2 << " c3=" << c3 << endl;
		}
#endif

	if (f_v) {
		cout << "finite_field::choose_anisotropic_form "
				"over GF(" << q << "): choosing c1=" << c1 << ", c2=" << c2
				<< ", c3=" << c3 << endl;
	}
}



void orthogonal_points_free_global_data()
{
	//cout << "orthogonal_points_free_global_data" << endl;
#if 0
	if (Hash_table_parabolic) {
		delete Hash_table_parabolic;
		Hash_table_parabolic = NULL;
		}
#endif
	//cout << "orthogonal_points_free_global_data done" << endl;
}


}}

