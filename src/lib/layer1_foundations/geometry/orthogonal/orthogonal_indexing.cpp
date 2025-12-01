/*
 * orthogonal_indexing.cpp
 *
 *  Created on: Jan 12, 2022
 *      Author: betten
 */





#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace orthogonal_geometry {


orthogonal_indexing::orthogonal_indexing()
{
	Record_birth();
	Quadratic_form = NULL;
	F = NULL;
}

orthogonal_indexing::~orthogonal_indexing()
{
	Record_death();
}

void orthogonal_indexing::init(
		quadratic_form *Quadratic_form,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_indexing::init" << endl;
	}
	orthogonal_indexing::Quadratic_form = Quadratic_form;
	F = Quadratic_form->F;
	if (f_v) {
		cout << "orthogonal_indexing::init done" << endl;
	}
}

void orthogonal_indexing::Q_epsilon_unrank_private(
	int *v, int stride, int epsilon, int k,
	int c1, int c2, int c3, long int a,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_indexing::Q_epsilon_unrank" << endl;
	}
	if (epsilon == 0) {
		if (f_v) {
			cout << "orthogonal_indexing::Q_epsilon_unrank "
					"before Q_unrank" << endl;
		}
		Q_unrank(v, stride, k, a, verbose_level);
		if (f_v) {
			cout << "orthogonal_indexing::Q_epsilon_unrank "
					"after Q_unrank" << endl;
		}
	}
	else if (epsilon == 1) {
		if (f_v) {
			cout << "orthogonal_indexing::Q_epsilon_unrank "
					"before Qplus_unrank" << endl;
		}
		Qplus_unrank(v, stride, k, a, verbose_level);
		if (f_v) {
			cout << "orthogonal_indexing::Q_epsilon_unrank "
					"after Qplus_unrank" << endl;
		}
	}
	else if (epsilon == -1) {
		if (f_v) {
			cout << "orthogonal_indexing::Q_epsilon_unrank "
					"before Qminus_unrank" << endl;
		}
		Qminus_unrank(v, stride, k, a, c1, c2, c3, verbose_level);
		if (f_v) {
			cout << "finite_field::Q_epsilon_unrank "
					"after Qminus_unrank" << endl;
		}
	}
	else {
		cout << "orthogonal_indexing epsilon is wrong" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "orthogonal_indexing::Q_epsilon_unrank done" << endl;
	}
}

long int orthogonal_indexing::Q_epsilon_rank_private(
	int *v, int stride, int epsilon, int k,
	int c1, int c2, int c3,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int a;

	if (f_v) {
		cout << "orthogonal_indexing::Q_epsilon_rank" << endl;
	}
	if (epsilon == 0) {
		if (f_v) {
			cout << "orthogonal_indexing::Q_epsilon_rank "
					"before Q_rank" << endl;
		}
		a = Q_rank(v, stride, k, verbose_level);
		if (f_v) {
			cout << "orthogonal_indexing::Q_epsilon_rank "
					"after Q_rank" << endl;
		}
	}
	else if (epsilon == 1) {
		if (f_v) {
			cout << "orthogonal_indexing::Q_epsilon_rank "
					"before Qplus_rank" << endl;
		}
		a = Qplus_rank(v, stride, k, verbose_level);
		if (f_v) {
			cout << "orthogonal_indexing::Q_epsilon_rank "
					"after Qplus_rank" << endl;
		}
	}
	else if (epsilon == -1) {
		if (f_v) {
			cout << "orthogonal_indexing::Q_epsilon_rank "
					"before Qminus_rank" << endl;
		}
		a = Qminus_rank(v, stride, k, c1, c2, c3, verbose_level);
		if (f_v) {
			cout << "orthogonal_indexing::Q_epsilon_rank "
					"after Qminus_rank" << endl;
		}
	}
	else {
		cout << "orthogonal_indexing::Q_epsilon_unrank "
				"epsilon is wrong" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "orthogonal_indexing::Q_epsilon_rank done" << endl;
	}
	return a;
}




void orthogonal_indexing::Q_unrank(
		int *v,
		int stride, int k, long int a, int verbose_level)
{

	Q_unrank_directly(v, stride, k, a, verbose_level);
}

long int orthogonal_indexing::Q_rank(
		int *v,
		int stride, int k, int verbose_level)
{
	return Q_rank_directly(v, stride, k, verbose_level);
}

void orthogonal_indexing::Q_unrank_directly(
		int *v,
		int stride, int k, long int a,
		int verbose_level)
// parabolic quadric
// k = projective dimension, must be even
// quadratic form: x_0^2 + x_1x_2 + x_3x_4 + ...
{
	int n, i, minusone;
	long int x;
	geometry::other_geometry::geometry_global Gg;

	n = Gg.Witt_index(0, k);
	x = Gg.nb_pts_Sbar(n, F->q);
	if (a < x) {
		v[0] = 0;
		Sbar_unrank(v + stride, stride, n, a, verbose_level);
		F->Projective_space_basic->PG_element_normalize_from_front(
				v + stride, stride, k);
		return;
	}
	a -= x;
	v[0] = 1;
	N1_unrank(v + stride, stride, n, a);
	minusone = F->negate(1);
	if (minusone != 1) {
		for (i = 0; i < n; i++) {
			v[(1 + 2 * i) * stride] =
				F->mult(v[(1 + 2 * i) * stride], minusone);
		}
	}
}

long int orthogonal_indexing::Q_rank_directly(
		int *v,
		int stride, int k,
		int verbose_level)
// parabolic quadric
// k = projective dimension, must be even
{
	int n, i;
	long int x, a, b;
	int minusone;
	geometry::other_geometry::geometry_global Gg;

	n = Gg.Witt_index(0, k);
	x = Gg.nb_pts_Sbar(n, F->q);
	if (v[0] == 0) {
		Sbar_rank(v + stride, stride, n, a, verbose_level);
		return a;
	}
	a = x;
	if (v[0] != 1) {
		F->Projective_space_basic->PG_element_normalize_from_front(
				v, stride, k + 1);
	}
	minusone = F->negate(1);
	if (minusone != 1) {
		for (i = 0; i < n; i++) {
			v[(1 + 2 * i) * stride] =
					F->mult(v[(1 + 2 * i) * stride], minusone);
		}
	}
	N1_rank(v + stride, stride, n, b);
	return a + b;
}

void orthogonal_indexing::Qplus_unrank(
		int *v,
		int stride, int k, long int a,
		int verbose_level)
// hyperbolic quadric
// k = projective dimension, must be odd
{
	int n;
	geometry::other_geometry::geometry_global Gg;

	n = Gg.Witt_index(1, k);
	Sbar_unrank(v, stride, n, a, verbose_level);
}

long int orthogonal_indexing::Qplus_rank(
		int *v,
		int stride, int k,
		int verbose_level)
// hyperbolic quadric
// k = projective dimension, must be odd
{
	int f_v = (verbose_level >= 1);
	int n;
	long int a;
	geometry::other_geometry::geometry_global Gg;

	if (f_v) {
		cout << "orthogonal_indexing::Qplus_rank" << endl;
	}
	n = Gg.Witt_index(1, k);
	Sbar_rank(v, stride, n, a, verbose_level);
	return a;
}

void orthogonal_indexing::Qminus_unrank(
		int *v,
		int stride, int k, long int a,
		int c1, int c2, int c3,
		int verbose_level)
// elliptic quadric
// k = projective dimension, must be odd
// the form is
// \sum_{i=0}^n x_{2i}x_{2i+1} + c1 x_{2n}^2 +
// c2 x_{2n} x_{2n+1} + c3 x_{2n+1}^2
{
	int n, z, minusz, u, vv, w, i;
	long int x, b, c, x1, x2;
	geometry::other_geometry::geometry_global Gg;

	n = Gg.Witt_index(-1, k);
	x = Gg.nb_pts_Sbar(n, F->q);
	if (a < x) {
		v[2 * n * stride] = 0;
		v[(2 * n + 1) * stride] = 0;
		Sbar_unrank(v, stride, n, a, verbose_level);
		return;
	}
	a -= x;
	x = Gg.nb_pts_N1(n, F->q);
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
		if (b >= F->q) {
			cout << "orthogonal_indexing::Qminus_unrank "
					"b >= q, the rank was too big" << endl;
			exit(1);
		}
	}
	v[2 * n * stride] = x1;
	v[(2 * n + 1) * stride] = x2;
	u = F->product3(c1, x1, x1);
	vv = F->product3(c2, x1, x2);
	w = F->product3(c3, x2, x2);
	z = F->add3(u, vv, w);
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
	minusz = F->negate(z);
	if (minusz != 1) {
		for (i = 0; i < n; i++) {
			v[2 * i * stride] = F->mult(v[2 * i * stride], minusz);
		}
	}
}

long int orthogonal_indexing::Qminus_rank(
		int *v,
		int stride, int k, int c1, int c2, int c3,
		int verbose_level)
// elliptic quadric
// k = projective dimension, must be odd
// the form is
// \sum_{i=0}^n x_{2i}x_{2i+1} + c1 x_{2n}^2 +
// c2 x_{2n} x_{2n+1} + c3 x_{2n+1}^2
{
	int n, minusz, minuszv;
	long int a, b, c, x, x1, x2, u, vv, w, z, i;
	geometry::other_geometry::geometry_global Gg;

	n = Gg.Witt_index(-1, k);

	{
		int aa;
		aa = Quadratic_form->evaluate_quadratic_form(v, stride);
		if (aa) {
			cout << "Qminus_rank fatal: the vector "
					"is not zero under the quadratic form" << endl;
			cout << "value=" << aa << endl;
			cout << "stride=" << stride << endl;
			cout << "k=" << k << endl;
			cout << "c1=" << c1 << endl;
			cout << "c2=" << c2 << endl;
			cout << "c3=" << c3 << endl;
			Int_vec_print(cout, v, k + 1);
			cout << endl;
			exit(1);
		}
	}
	F->Projective_space_basic->PG_element_normalize(
			v, stride, k + 1, 0 /* verbose_level */);
	x1 = v[2 * n * stride];
	x2 = v[(2 * n + 1) * stride];
	if (x1 == 0 && x2 == 0) {
		Sbar_rank(v, stride, n, a, verbose_level);
		return a;
	}
	a = Gg.nb_pts_Sbar(n, F->q);
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

	x = Gg.nb_pts_N1(n, F->q);
	//b = a / x;
	//c = a % x;
	u = F->product3(c1, x1, x1);
	vv = F->product3(c2, x1, x2);
	w = F->product3(c3, x2, x2);
	z = F->add3(u, vv, w);
	if (z == 0) {
		cout << "Qminus_rank z = 0" << endl;
		cout << "b=" << b << endl;
		exit(1);
	}

	minusz = F->negate(z);
	minuszv = F->inverse(minusz);
	if (minuszv != 1) {
		for (i = 0; i < n; i++) {
			v[2 * i * stride] = F->mult(v[2 * i * stride], minuszv);
		}
	}

	N1_rank(v, stride, n, c);
	a += b * x + c;
	return a;
}





// #############################################################################
// unrank functions for the hyperbolic quadric:
// #############################################################################



void orthogonal_indexing::S_unrank(
		int *v, int stride, int n, long int a)
{
	long int l, i, j, x, y, u;
	int alpha, beta;
	geometry::other_geometry::geometry_global Gg;

	if (n == 1) {
		if (a < F->q) {
			v[0 * stride] = a;
			v[1 * stride] = 0;
			return;
		}
		a -= (F->q - 1);
		if (a < F->q) {
			v[0 * stride] = 0;
			v[1 * stride] = a;
			return;
		}
		else {
			cout << "orthogonal_indexing::S_unrank "
					"error in S_unrank n = 1 a = " << a << endl;
			exit(1);
		}
	}
	else {
		x = Gg.nb_pts_S(1, F->q);
		y = Gg.nb_pts_S(n - 1, F->q);
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
		x = Gg.nb_pts_N(1, F->q);
		y = Gg.nb_pts_N1(n - 1, F->q);
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			N_unrank(v + (n - 1) * 2 * stride, stride, 1, i);
			N1_unrank(v, stride, n - 1, j);

			alpha = F->mult(v[2 * (n - 1) * stride],
					v[(2 * (n - 1) + 1) * stride]);
			beta = F->negate(alpha);
			for (u = 0; u < n - 1; u++) {
				v[2 * u * stride] = F->mult(v[2 * u * stride], beta);
			}
			return;
		}
		else {
			cout << "orthogonal_indexing::S_unrank "
					"error in S_unrank n = " << n << ", a = " << a << endl;
			exit(1);
		}
	}
}

void orthogonal_indexing::S_rank(
		int *v, int stride, int n, long int &a)
{
	long int l, i, j, x, y, u;
	int alpha, beta, gamma, delta, epsilon;
	geometry::other_geometry::geometry_global Gg;

	if (n == 1) {
		if (v[1 * stride] == 0) {
			a = v[0 * stride];
			return;
		}
		if (v[0 * stride]) {
			cout << "orthogonal_indexing::S_rank "
					"error in S_rank v[0] not null" << endl;
			exit(1);
		}
		a = F->q - 1;
		a += v[1 * stride];
	}
	else {
		x = Gg.nb_pts_S(1, F->q);
		y = Gg.nb_pts_S(n - 1, F->q);
		l = x * y;
		alpha = F->mult(v[2 * (n - 1) * stride],
				v[(2 * (n - 1) + 1) * stride]);
		if (alpha == 0) {
			S_rank(v + (n - 1) * 2 * stride, stride, 1, i);
			S_rank(v, stride, n - 1, j);
			a = i * y + j;
			return;
		}
		a = l;
		x = Gg.nb_pts_N(1, F->q);
		y = Gg.nb_pts_N1(n - 1, F->q);

		N_rank(v + (n - 1) * 2 * stride, stride, 1, i);


		beta = F->negate(alpha);
		gamma = Quadratic_form->evaluate_hyperbolic_quadratic_form_with_m(
				v, stride, n - 1);
		if (gamma != beta) {
			cout << "orthogonal_indexing::S_rank "
					"error in S_rank gamma != beta" << endl;
			exit(1);
		}
		delta = F->inverse(beta);
		for (u = 0; u < n - 1; u++) {
			v[2 * u * stride] = F->mult(v[2 * u * stride], delta);
		}
		epsilon = Quadratic_form->evaluate_hyperbolic_quadratic_form_with_m(
				v, stride, n - 1);
		if (epsilon != 1) {
			cout << "orthogonal_indexing::S_rank "
					"error in S_rank epsilon != 1" << endl;
			exit(1);
		}
		N1_rank(v, stride, n - 1, j);
		a += i * y + j;
	}
}

void orthogonal_indexing::N_unrank(
		int *v, int stride, int n, long int a)
{
	long int l, i, j, k, j1, x, y, z, yz, u;
	int alpha, beta, gamma, delta, epsilon;
	geometry::other_geometry::geometry_global Gg;

	if (n == 1) {
		x = F->q - 1;
		y = F->q - 1;
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			v[0 * stride] = 1 + j;
			v[1 * stride] = 1 + i;
			return;
		}
		else {
			cout << "orthogonal_indexing::N_unrank "
					"error in N_unrank n = 1 a = " << a << endl;
			exit(1);
		}
	}
	else {
		x = Gg.nb_pts_S(1, F->q);
		y = Gg.nb_pts_N(n - 1, F->q);
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			S_unrank(v + (n - 1) * 2 * stride, stride, 1, i);
			N_unrank(v, stride, n - 1, j);
			return;
		}
		a -= l;
		x = Gg.nb_pts_N(1, F->q);
		y = Gg.nb_pts_S(n - 1, F->q);
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			N_unrank(v + (n - 1) * 2 * stride, stride, 1, i);
			S_unrank(v, stride, n - 1, j);
			return;
		}
		a -= l;
		x = Gg.nb_pts_N(1, F->q);
		y = (F->q - 2);
		z = Gg.nb_pts_N1(n - 1, F->q);
		yz = y * z;
		l = x * yz;
		if (a < l) {
			i = a / yz;
			j1 = a % yz;
			j = j1 / z;
			k = j1 % z;
			N_unrank(v + (n - 1) * 2 * stride, stride, 1, i);
			N1_unrank(v, stride, n - 1, k);
			alpha = F->primitive_element();

			beta = F->power(alpha, j + 1);
			gamma = F->mult(v[(n - 1) * 2 * stride],
					v[((n - 1) * 2 + 1) * stride]);
			delta = F->negate(gamma);
			epsilon = F->mult(delta, beta);
			for (u = 0; u < n - 1; u++) {
				v[2 * u * stride] = F->mult(v[2 * u * stride], epsilon);
			}
			return;
		}
		else {
			cout << "orthogonal_indexing::N_unrank "
					"error in N_unrank n = " << n << ", a = " << a << endl;
			exit(1);
		}
	}
}

void orthogonal_indexing::N_rank(
		int *v, int stride, int n, long int &a)
{
	long int l, i, j, k, x, y, z, yz, u;
	int alpha, beta, gamma, delta;
	int epsilon, gamma2, epsilon_inv;
	geometry::other_geometry::geometry_global Gg;

	if (n == 1) {
		x = F->q - 1;
		y = F->q - 1;
		if (v[0 * stride] == 0 || v[1 * stride] == 0) {
			cout << "orthogonal_indexing::N_rank "
					"v[0 * stride] == 0 || "
					"v[1 * stride] == 0" << endl;
			exit(1);
		}
		j = v[0 * stride] - 1;
		i = v[1 * stride] - 1;
		a = i * y + j;
	}
	else {
		gamma = F->mult(v[(n - 1) * 2 * stride],
				v[((n - 1) * 2 + 1) * stride]);
		x = Gg.nb_pts_S(1, F->q);
		y = Gg.nb_pts_N(n - 1, F->q);
		l = x * y;
		if (gamma == 0) {
			S_rank(v + (n - 1) * 2 * stride, stride, 1, i);
			N_rank(v, stride, n - 1, j);
			a = i * y + j;
			return;
		}
		a = l;
		x = Gg.nb_pts_N(1, F->q);
		y = Gg.nb_pts_S(n - 1, F->q);
		l = x * y;
		gamma2 = Quadratic_form->evaluate_hyperbolic_quadratic_form_with_m(
				v, stride, n - 1);
		if (gamma2 == 0) {
			N_rank(v + (n - 1) * 2, stride, 1, i);
			S_rank(v, stride, n - 1, j);
			a += i * y + j;
		}
		a += l;

		x = Gg.nb_pts_N(1, F->q);
		y = (F->q - 2);
		z = Gg.nb_pts_N1(n - 1, F->q);
		yz = y * z;
		l = x * yz;

		N_rank(v + (n - 1) * 2 * stride, stride, 1, i);
		alpha = F->primitive_element();
		delta = F->negate(gamma);
		for (j = 0; j < F->q - 2; j++) {
			beta = F->power(alpha, j + 1);
			epsilon = F->mult(delta, beta);
			if (epsilon == gamma2) {
				epsilon_inv = F->inverse(epsilon);
				for (u = 0; u < n - 1; u++) {
					v[2 * u * stride] = F->mult(
							v[2 * u * stride], epsilon_inv);
				}
				N1_rank(v, stride, n - 1, k);
				a += i * yz + j * z + k;
				return;
			}
		}
		cout << "orthogonal_indexing::N_rank "
				"error, gamma2 not found" << endl;
		exit(1);
	}
}

void orthogonal_indexing::N1_unrank(
		int *v, int stride, int n, long int a)
{
	long int l, i, j, k, j1, x, y, z, yz, u;
	int alpha, beta, gamma;
	geometry::other_geometry::geometry_global Gg;

	if (n == 1) {
		l = F->q - 1;
		if (a < l) {
			alpha = a + 1;
			beta = F->inverse(alpha);
			//cout << "finite_field::N1_unrank n == 1, a = " << a
			// << " alpha = " << alpha << " beta = " << beta << endl;
			v[0 * stride] = alpha;
			v[1 * stride] = beta;
			return;
		}
		else {
			cout << "orthogonal_indexing::N1_unrank "
					"error in N1_unrank n = 1 a = " << a << endl;
			exit(1);
		}
	}
	else {
		x = Gg.nb_pts_S(1, F->q);
		y = Gg.nb_pts_N1(n - 1, F->q);
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
		x = Gg.nb_pts_N1(1, F->q);
		y = Gg.nb_pts_S(n - 1, F->q);
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
		x = Gg.nb_pts_N1(1, F->q);
		y = (F->q - 2); // zero for q = 2
		z = Gg.nb_pts_N1(n - 1, F->q);
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
			v[2 * (n - 1) * stride] = F->mult(
					v[2 * (n - 1) * stride], alpha);

			N1_unrank(v, stride, n - 1, k);

			//int_set_print(v, 2 * (n - 1));
			//cout << endl;

			beta = F->negate(alpha);
			gamma = F->add(beta, 1);

			//cout << "alpha = j + 2 = " << alpha << endl;
			//cout << "beta = - alpha = " << beta << endl;
			//cout << "gamma = beta + 1 = " << gamma << endl;

			for (u = 0; u < n - 1; u++) {
				v[2 * u * stride] = F->mult(v[2 * u * stride], gamma);
			}

			//int_set_print(v, 2 * n);
			//cout << endl;
			return;
		}
		else {
			cout << "orthogonal_indexing::N1_unrank "
					"error in N1_unrank n = " << n << ", a = " << a << endl;
			exit(1);
		}
	}
}

void orthogonal_indexing::N1_rank(
		int *v, int stride, int n, long int &a)
{
	long int l, i, j, k, x, y, z, yz, u;
	int alpha, alpha_inv, beta, gamma, gamma2, gamma_inv;
	geometry::other_geometry::geometry_global Gg;

	if (n == 1) {
		alpha = v[0 * stride];
		beta = v[1 * stride];
		if (alpha == 0 || beta == 0) {
			cout << "orthogonal_indexing::N1_rank "
					"alpha == 0 || beta == 0" << endl;
			exit(1);
		}
		gamma = F->inverse(alpha);
		if (gamma != beta) {
			cout << "orthogonal_indexing::N1_rank "
					"error in N1_rank gamma = " << gamma
					<< " != beta = " << beta << endl;
			exit(1);
		}
		a = alpha - 1;
	}
	else {
		a = 0;
		alpha = F->mult(v[2 * (n - 1) * stride],
				v[(2 * (n - 1) + 1) * stride]);
		x = Gg.nb_pts_S(1, F->q);
		y = Gg.nb_pts_N1(n - 1, F->q);
		l = x * y;
		if (alpha == 0) {
			S_rank(v + (n - 1) * 2 * stride, stride, 1, i);
			N1_rank(v, stride, n - 1, j);
			a = i * y + j;
			return;
		}
		a += l;
		gamma2 = Quadratic_form->evaluate_hyperbolic_quadratic_form_with_m(
				v, stride, n - 1);
		x = Gg.nb_pts_N1(1, F->q);
		y = Gg.nb_pts_S(n - 1, F->q);
		l = x * y;
		if (gamma2 == 0) {
			N1_rank(v + (n - 1) * 2 * stride, stride, 1, i);
			S_rank(v, stride, n - 1, j);
			a += i * y + j;
			return;
		}
		a += l;
		// the case q = 2 does not appear here any more
		if (F->q == 2) {
			cout << "orthogonal_indexing::N1_rank "
					"the case q=2 should not appear here" << endl;
			exit(1);
		}


		x = Gg.nb_pts_N1(1, F->q);
		y = (F->q - 2); // zero for q = 2
		z = Gg.nb_pts_N1(n - 1, F->q);
		yz = y * z;
		l = x * yz; // zero for q = 2

		alpha = F->mult(v[2 * (n - 1) * stride],
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
		alpha_inv = F->inverse(alpha);
		v[2 * (n - 1) * stride] = F->mult(
				v[2 * (n - 1) * stride], alpha_inv);

		N1_rank(v + (n - 1) * 2 * stride, stride, 1, i);

		gamma2 = Quadratic_form->evaluate_hyperbolic_quadratic_form_with_m(
				v, stride, n - 1);
		if (gamma2 == 0) {
			cout << "orthogonal_indexing::N1_rank "
					"gamma2 == 0" << endl;
			exit(1);
		}
		if (gamma2 == 1) {
			cout << "orthogonal_indexing::N1_rank "
					"gamma2 == 1" << endl;
			exit(1);
		}
		gamma_inv = F->inverse(gamma2);
		for (u = 0; u < n - 1; u++) {
			v[2 * u * stride] = F->mult(v[2 * u * stride], gamma_inv);
		}
		N1_rank(v, stride, n - 1, k);

		a += i * yz + j * z + k;

	}
}

void orthogonal_indexing::Sbar_unrank(
		int *v,
		int stride, int n, long int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int l, i, j, x, y, u;
	int alpha, beta;
	geometry::other_geometry::geometry_global Gg;

	if (f_v) {
		cout << "orthogonal_indexing::Sbar_unrank" << endl;
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
			cout << "orthogonal_indexing::Sbar_unrank "
					"error in Sbar_unrank n = 1 a = " << a << endl;
			exit(1);
		}
	}
	else {
		y = Gg.nb_pts_Sbar(n - 1, F->q);
		l = y;
		if (a < l) {
			u = n - 1;
			v[2 * u * stride] = 0;
			v[(2 * u + 1) * stride] = 0;
			Sbar_unrank(v, stride, n - 1, a, verbose_level);
			return;
		}
		if (f_v) {
			cout << "orthogonal_indexing::Sbar_unrank a = " << a << endl;
			cout << "orthogonal_indexing::Sbar_unrank l = " << l << endl;
		}
		a -= l;
		if (f_v) {
			cout << "orthogonal_indexing::Sbar_unrank a = " << a << endl;
		}
		//cout << "subtracting " << l << " to bring a to " << a << endl;
		x = Gg.nb_pts_Sbar(1, F->q);
		y = Gg.nb_pts_S(n - 1, F->q);
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
			cout << "orthogonal_indexing::Sbar_unrank a = " << a << endl;
			cout << "orthogonal_indexing::Sbar_unrank l = " << l << endl;
		}
		a -= l;
		if (f_v) {
			cout << "orthogonal_indexing::Sbar_unrank a = " << a << endl;
		}
		//cout << "subtracting " << l << " to bring a to " << a << endl;
		x = Gg.nb_pts_Nbar(1, F->q);
		y = Gg.nb_pts_N1(n - 1, F->q);
		//cout << "nb_pts_N1(" << n - 1 << ") = " << y << endl;
		l = x * y;
		if (f_v) {
			cout << "orthogonal_indexing::Sbar_unrank x = " << x << endl;
			cout << "orthogonal_indexing::Sbar_unrank y = " << y << endl;
			cout << "orthogonal_indexing::Sbar_unrank l = " << l << endl;
		}
		if (a < l) {
			i = a / y;
			j = a % y;
			//cout << "i=" << i << " j=" << j << endl;
			Nbar_unrank(v + (n - 1) * 2 * stride, stride, 1, i);
			//cout << "(" << v[2 * (n - 1) * stride] << ","
			//<< v[(2 * (n - 1) + 1) * stride] << ")" << endl;
			N1_unrank(v, stride, n - 1, j);

			alpha = F->mult(v[2 * (n - 1) * stride],
					v[(2 * (n - 1) + 1) * stride]);
			beta = F->negate(alpha);
			for (u = 0; u < n - 1; u++) {
				v[2 * u * stride] = F->mult(v[2 * u * stride], beta);
			}
			//int_set_print(v, 2 * n);
			//cout << endl;
			return;
		}
		else {
			cout << "orthogonal_indexing::Sbar_unrank "
					"error in Sbar_unrank n = " << n
					<< ", a = " << a << endl;
			exit(1);
		}
	}
}

void orthogonal_indexing::Sbar_rank(
		int *v,
		int stride, int n, long int &a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int l, i, j, x, y, u;
	int alpha, beta, beta2, beta_inv;
	geometry::other_geometry::geometry_global Gg;

	if (f_v) {
		cout << "orthogonal_indexing::Sbar_rank" << endl;
	}
	F->Projective_space_basic->PG_element_normalize(
			v, stride, 2 * n, 0 /* verbose_level */);
	if (f_v) {
		cout << "orthogonal_indexing::Sbar_rank: ";
		if (stride == 1) {
			Int_vec_print(cout, v, 2 * n);
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
			cout << "orthogonal_indexing::Sbar_rank "
					"error in Sbar_rank n = 1 bad vector" << endl;
			if (stride == 1) {
				Int_vec_print(cout, v, 2);
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
		l = Gg.nb_pts_Sbar(n - 1, F->q);
		if (f_v) {
			cout << "orthogonal_indexing::Sbar_rank not a leading zero, l = " << l << endl;
		}
		a += l;

		// alpha = form value for the top two coefficients:
		alpha = F->mult(v[2 * (n - 1) * stride],
				v[(2 * (n - 1) + 1) * stride]);
		x = Gg.nb_pts_Sbar(1, F->q); // = 2
		y = Gg.nb_pts_S(n - 1, F->q);

		if (f_v) {
			cout << "orthogonal_indexing::Sbar_rank alpha = " << alpha << endl;
			cout << "orthogonal_indexing::Sbar_rank x = " << x << endl;
			cout << "orthogonal_indexing::Sbar_rank y = " << y << endl;
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
			cout << "orthogonal_indexing::Sbar_rank l = " << l << endl;
			cout << "orthogonal_indexing::Sbar_rank a = " << a << endl;
		}

		// now it must be n + (-n) for some n \neq 0
		// (i.e. n = alpha = value of the form on
		// the top two coefficients and
		// -n for the value of the form on the rest):
		x = Gg.nb_pts_Nbar(1, F->q);
		y = Gg.nb_pts_N1(n - 1, F->q);
		Nbar_rank(v + (n - 1) * 2 * stride, stride, 1, i);
		if (f_v) {
			cout << "orthogonal_indexing::Sbar_rank x = " << x << endl;
			cout << "orthogonal_indexing::Sbar_rank y = " << y << endl;
			cout << "orthogonal_indexing::Sbar_rank i = " << i << endl;
		}

		beta = F->negate(alpha);
		// beta = - alpha
		beta2 = Quadratic_form->evaluate_hyperbolic_quadratic_form_with_m(
				v, stride, n - 1);
		// beta2 = value of the quadratic form on the rest
		// must be - alpha (otherwise the vector does
		// not represent a point in Sbar)
		if (beta2 != beta) {
			cout << "orthogonal_indexing::Sbar_rank "
					"error in Sbar_rank beta2 != beta" << endl;
			exit(1);
		}
		beta_inv = F->inverse(beta);
		// divide by beta so that the quadratic form
		// on the rest is equal to 1.
		for (u = 0; u < n - 1; u++) {
			v[2 * u * stride] = F->mult(
					v[2 * u * stride], beta_inv);
		}
		// rank the N1 part:
		N1_rank(v, stride, n - 1, j);
		if (f_v) {
			cout << "orthogonal_indexing::Sbar_rank j = " << j << endl;
		}
		a += i * y + j;
		if (f_v) {
			cout << "orthogonal_indexing::Sbar_rank a = " << a << endl;
		}
	}
}

void orthogonal_indexing::Nbar_unrank(
		int *v, int stride, int n, long int a)
{
	int y, l;

	if (n == 1) {
		y = F->q - 1;
		l = y;
		if (a < l) {
			v[0 * stride] = 1 + a;
			v[1 * stride] = 1;
			return;
		}
		else {
			cout << "orthogonal_indexing::Nbar_unrank "
					"error in Nbar_unrank n = 1 a = " << a << endl;
			exit(1);
		}
	}
	else {
		cout << "orthogonal_indexing::Nbar_unrank "
				"only defined for n = 1" << endl;
		exit(1);
	}
}

void orthogonal_indexing::Nbar_rank(
		int *v, int stride, int n, long int &a)
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
		cout << "orthogonal_indexing::Nbar_rank only defined for n = 1" << endl;
		exit(1);
	}
}





}}}}





