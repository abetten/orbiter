/*
 * linear_algebra3.cpp
 *
 *  Created on: Jan 10, 2022
 *      Author: betten
 */


#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {





void linear_algebra::Gram_matrix(int epsilon, int k,
	int form_c1, int form_c2, int form_c3,
	int *&Gram, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int d = k + 1;
	int n, i, j, u, offset = 0;
	geometry_global Gg;

	if (f_v) {
		cout << "linear_algebra::Gram_matrix" << endl;
	}
	Gram = NEW_int(d * d);
	Orbiter->Int_vec.zero(Gram, d * d);
	n = Gg.Witt_index(epsilon, k);
	if (epsilon == 0) {
		Gram[0 * d + 0] = F->add(form_c1, form_c1);
		offset = 1;
	}
	else if (epsilon == 1) {
	}
	else if (epsilon == -1) {
		Gram[(d - 2) * d + d - 2] = F->add(form_c1, form_c1);
		Gram[(d - 2) * d + d - 1] = form_c2;
		Gram[(d - 1) * d + d - 2] = form_c2;
		Gram[(d - 1) * d + d - 1] = F->add(form_c3, form_c3);
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
		cout << "linear_algebra::Gram_matrix done" << endl;
	}
}

int linear_algebra::evaluate_bilinear_form(
		int *u, int *v, int d, int *Gram)
{
	int i, j, a, b, c, e, A;

	A = 0;
	for (i = 0; i < d; i++) {
		a = u[i];
		for (j = 0; j < d; j++) {
			b = Gram[i * d + j];
			c = v[j];
			e = F->mult(a, b);
			e = F->mult(e, c);
			A = F->add(A, e);
			}
		}
	return A;
}

int linear_algebra::evaluate_quadratic_form(int *v, int stride,
	int epsilon, int k, int form_c1, int form_c2, int form_c3)
{
	int n, a, b, c = 0, d, x, x1, x2;
	geometry_global Gg;

	n = Gg.Witt_index(epsilon, k);
	if (epsilon == 0) {
		a = evaluate_hyperbolic_quadratic_form(v + stride, stride, n);
		x = v[0];
		b = F->product3(form_c1, x, x);
		c = F->add(a, b);
		}
	else if (epsilon == 1) {
		c = evaluate_hyperbolic_quadratic_form(v, stride, n);
		}
	else if (epsilon == -1) {
		a = evaluate_hyperbolic_quadratic_form(v, stride, n);
		x1 = v[2 * n * stride];
		x2 = v[(2 * n + 1) * stride];
		b = F->product3(form_c1, x1, x1);
		c = F->product3(form_c2, x1, x2);
		d = F->product3(form_c3, x2, x2);
		c = F->add4(a, b, c, d);
		}
	return c;
}


int linear_algebra::evaluate_hyperbolic_quadratic_form(
		int *v, int stride, int n)
{
	int alpha = 0, beta, u;

	for (u = 0; u < n; u++) {
		beta = F->mult(v[2 * u * stride], v[(2 * u + 1) * stride]);
		alpha = F->add(alpha, beta);
		}
	return alpha;
}

int linear_algebra::evaluate_hyperbolic_bilinear_form(
		int *u, int *v, int n)
{
	int alpha = 0, beta1, beta2, i;

	for (i = 0; i < n; i++) {
		beta1 = F->mult(u[2 * i], v[2 * i + 1]);
		beta2 = F->mult(u[2 * i + 1], v[2 * i]);
		alpha = F->add(alpha, beta1);
		alpha = F->add(alpha, beta2);
		}
	return alpha;
}



void linear_algebra::Siegel_map_between_singular_points(int *T,
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
		cout << "linear_algebra::Siegel_map_between_singular_points "
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
	F->Orthogonal_indexing->Q_epsilon_unrank(B, 1, epsilon, k,
			form_c1, form_c2, form_c3, root, 0 /* verbose_level */);
	F->Orthogonal_indexing->Q_epsilon_unrank(B + d, 1, epsilon, k,
			form_c1, form_c2, form_c3, rk_from, 0 /* verbose_level */);
	F->Orthogonal_indexing->Q_epsilon_unrank(w, 1, epsilon, k,
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
	av = F->inverse(a);
	bv = F->inverse(b);
	for (i = 0; i < d; i++) {
		B[d + i] = F->mult(B[d + i], av);
		w[i] = F->mult(w[i], bv);
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
	perp(d, 2, B, Gram_matrix, 0 /* verbose_level */);
	if (f_vv) {
		cout << "after perp, the matrix B is:" << endl;
		Orbiter->Int_vec.print_integer_matrix(cout, B, d, d);
	}
	invert_matrix(B, Bv, d, 0 /* verbose_level */);
	if (f_vv) {
		cout << "the matrix Bv = B^{-1} is:" << endl;
		Orbiter->Int_vec.print_integer_matrix(cout, B, d, d);
	}
	mult_matrix_matrix(w, Bv, z, 1, d, d, 0 /* verbose_level */);
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
	mult_matrix_matrix(z, B, x, 1, d, d, 0 /* verbose_level */);
	if (f_vv) {
		cout << "the vector x = z * B is:" << endl;
		Orbiter->Int_vec.print(cout, x, d);
		cout << endl;
	}
	minus_one = F->negate(1);
	for (i = 0; i < d; i++) {
		x[i] = F->mult(x[i], minus_one);
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
		cout << "linear_algebra::Siegel_map_between_singular_points "
				"the Siegel transformation is:" << endl;
		Orbiter->Int_vec.print_integer_matrix(cout, T, d, d);
	}
	FREE_int(B);
	FREE_int(Bv);
	FREE_int(w);
	FREE_int(z);
	FREE_int(x);
}

void linear_algebra::Siegel_Transformation(
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
		cout << "linear_algebra::Siegel_Transformation "
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
			e = F->mult(b, c);
			a = F->add(a, e);
		}
		w[i] = a;
	}
	// M := M + w^T * u
	for (i = 0; i < d; i++) {
		b = w[i];
		for (j = 0; j < d; j++) {
			c = u[j];
			e = F->mult(b, c);
			M[i * d + j] = F->add(M[i * d + j], e);
		}
	}
	// compute w^T := Gram * u^T
	for (i = 0; i < d; i++) {
		a = 0;
		for (j = 0; j < d; j++) {
			b = Gram[i * d + j];
			c = u[j];
			e = F->mult(b, c);
			a = F->add(a, e);
		}
		w[i] = a;
	}
	// M := M - w^T * v
	for (i = 0; i < d; i++) {
		b = w[i];
		for (j = 0; j < d; j++) {
			c = v[j];
			e = F->mult(b, c);
			M[i * d + j] = F->add(M[i * d + j], F->negate(e));
		}
	}
	// M := M - Q(v) * w^T * u
	for (i = 0; i < d; i++) {
		b = w[i];
		for (j = 0; j < d; j++) {
			c = u[j];
			e = F->mult(b, c);
			M[i * d + j] = F->add(M[i * d + j], F->mult(F->negate(e), Qv));
		}
	}
	if (f_v) {
		cout << "linear_algebra::Siegel_Transformation "
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


long int linear_algebra::orthogonal_find_root(int rk2,
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
		cout << "linear_algebra::orthogonal_find_root "
				"rk2=" << rk2 << endl;
	}
	if (rk2 == 0) {
		cout << "linear_algebra::orthogonal_find_root: "
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

	F->Orthogonal_indexing->Q_epsilon_unrank(y, 1, epsilon, k,
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
		cout << "linear_algebra::orthogonal_find_root "
				"error: y is zero vector" << endl;
	}
	y2_minus_y3 = F->add(y[2], F->negate(y[3]));
	minus_y1 = F->negate(y[1]);
	if (minus_y1 != y2_minus_y3) {
		z[0] = 1;
		z[1] = 1;
		z[2] = F->negate(1);
		z[3] = 1;
		goto finish;
	}
	y3_minus_y2 = F->add(y[3], F->negate(y[2]));
	if (minus_y1 != y3_minus_y2) {
		z[0] = 1;
		z[1] = 1;
		z[2] = 1;
		z[3] = F->negate(1);
		goto finish;
	}
	// now we are in characteristic 2
	if (F->q == 2) {
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
		cout << "linear_algebra::orthogonal_find_root "
				"error neither y2 nor y3 is zero" << endl;
		exit(1);
	}
	// now the field has at least 4 elements
	a = 3;
	a2 = F->mult(a, a);
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
	root = F->Orthogonal_indexing->Q_epsilon_rank(z, 1, epsilon, k,
			form_c1, form_c2, form_c3, 0 /* verbose_level */);
	if (f_v) {
		cout << "linear_algebra::orthogonal_find_root "
				"root=" << root << endl;
	}

	FREE_int(x);
	FREE_int(y);
	FREE_int(z);

	return root;
}

void linear_algebra::choose_anisotropic_form(
		int &c1, int &c2, int &c3, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	unipoly_domain FX(F);
	unipoly_object m;

	if (f_v) {
		cout << "linear_algebra::choose_anisotropic_form "
				"over GF(" << F->q << ")" << endl;
		}

	if (ODD(F->q)) {
		c1 = 1;
		c2 = 0;
		c3 = F->negate(F->primitive_element());
	}
	else {
		knowledge_base K;
		string poly;

		K.get_primitive_polynomial(poly, F->q, 2, 0);

		FX.create_object_by_rank_string(m,
				poly,
				verbose_level);

		//FX.create_object_by_rank_string(m,
		//get_primitive_polynomial(GFq.p, 2 * GFq.e, 0), verbose_level);

		if (f_v) {
			cout << "linear_algebra::choose_anisotropic_form "
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
	cout << "linear_algebra::choose_anisotropic_form "
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
		cout << "linear_algebra::choose_anisotropic_form "
				"over GF(" << F->q << "): choosing c1=" << c1 << ", c2=" << c2
				<< ", c3=" << c3 << endl;
	}
}


int linear_algebra::evaluate_conic_form(int *six_coeffs, int *v3)
{
	//int a = 2, b = 0, c = 0, d = 4, e = 4, f = 4, val, val1;
	//int a = 3, b = 1, c = 2, d = 4, e = 1, f = 4, val, val1;
	int val, val1;

	val = 0;
	val1 = F->product3(six_coeffs[0], v3[0], v3[0]);
	val = F->add(val, val1);
	val1 = F->product3(six_coeffs[1], v3[1], v3[1]);
	val = F->add(val, val1);
	val1 = F->product3(six_coeffs[2], v3[2], v3[2]);
	val = F->add(val, val1);
	val1 = F->product3(six_coeffs[3], v3[0], v3[1]);
	val = F->add(val, val1);
	val1 = F->product3(six_coeffs[4], v3[0], v3[2]);
	val = F->add(val, val1);
	val1 = F->product3(six_coeffs[5], v3[1], v3[2]);
	val = F->add(val, val1);
	return val;
}

int linear_algebra::evaluate_quadric_form_in_PG_three(
		int *ten_coeffs, int *v4)
{
	int val, val1;

	val = 0;
	val1 = F->product3(ten_coeffs[0], v4[0], v4[0]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[1], v4[1], v4[1]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[2], v4[2], v4[2]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[3], v4[3], v4[3]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[4], v4[0], v4[1]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[5], v4[0], v4[2]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[6], v4[0], v4[3]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[7], v4[1], v4[2]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[8], v4[1], v4[3]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[9], v4[2], v4[3]);
	val = F->add(val, val1);
	return val;
}

int linear_algebra::Pluecker_12(int *x4, int *y4)
{
	return Pluecker_ij(0, 1, x4, y4);
}

int linear_algebra::Pluecker_21(int *x4, int *y4)
{
	return Pluecker_ij(1, 0, x4, y4);
}

int linear_algebra::Pluecker_13(int *x4, int *y4)
{
	return Pluecker_ij(0, 2, x4, y4);
}

int linear_algebra::Pluecker_31(int *x4, int *y4)
{
	return Pluecker_ij(2, 0, x4, y4);
}

int linear_algebra::Pluecker_14(int *x4, int *y4)
{
	return Pluecker_ij(0, 3, x4, y4);
}

int linear_algebra::Pluecker_41(int *x4, int *y4)
{
	return Pluecker_ij(3, 0, x4, y4);
}

int linear_algebra::Pluecker_23(int *x4, int *y4)
{
	return Pluecker_ij(1, 2, x4, y4);
}

int linear_algebra::Pluecker_32(int *x4, int *y4)
{
	return Pluecker_ij(2, 1, x4, y4);
}

int linear_algebra::Pluecker_24(int *x4, int *y4)
{
	return Pluecker_ij(1, 3, x4, y4);
}

int linear_algebra::Pluecker_42(int *x4, int *y4)
{
	return Pluecker_ij(3, 1, x4, y4);
}

int linear_algebra::Pluecker_34(int *x4, int *y4)
{
	return Pluecker_ij(2, 3, x4, y4);
}

int linear_algebra::Pluecker_43(int *x4, int *y4)
{
	return Pluecker_ij(3, 2, x4, y4);
}

int linear_algebra::Pluecker_ij(int i, int j, int *x4, int *y4)
{
	return F->add(F->mult(x4[i], y4[j]), F->negate(F->mult(x4[j], y4[i])));
}


int linear_algebra::evaluate_symplectic_form(int len, int *x, int *y)
{
	int i, n, c;

	if (ODD(len)) {
		cout << "linear_algebra::evaluate_symplectic_form "
				"len must be even" << endl;
		cout << "len=" << len << endl;
		exit(1);
	}
	c = 0;
	n = len >> 1;
	for (i = 0; i < n; i++) {
		c = F->add(c, F->add(
				F->mult(x[2 * i + 0], y[2 * i + 1]),
				F->negate(F->mult(x[2 * i + 1], y[2 * i + 0]))
				));
	}
	return c;
}

int linear_algebra::evaluate_symmetric_form(int len, int *x, int *y)
{
	int i, n, c;

	if (ODD(len)) {
		cout << "linear_algebra::evaluate_symmetric_form "
				"len must be even" << endl;
		cout << "len=" << len << endl;
		exit(1);
	}
	c = 0;
	n = len >> 1;
	for (i = 0; i < n; i++) {
		c = F->add(c, F->add(
				F->mult(x[2 * i + 0], y[2 * i + 1]),
				F->mult(x[2 * i + 1], y[2 * i + 0])
				));
	}
	return c;
}

int linear_algebra::evaluate_quadratic_form_x0x3mx1x2(int *x)
{
	int a;

	a = F->add(F->mult(x[0], x[3]), F->negate(F->mult(x[1], x[2])));
	return a;
}

void linear_algebra::solve_y2py(int a, int *Y2, int &nb_sol)
{
	int y, y2py;

	nb_sol = 0;
	for (y = 0; y < F->q; y++) {
		y2py = F->add(F->mult(y, y), y);
		if (y2py == a) {
			Y2[nb_sol++] = y;
		}
	}
	if (nb_sol > 2) {
		cout << "linear_algebra::solve_y2py nb_sol > 2" << endl;
		exit(1);
	}
}

void linear_algebra::find_secant_points_wrt_x0x3mx1x2(int *Basis_line, int *Pts4, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int u;
	int b0, b1, b2, b3, b4, b5, b6, b7;
	int a, av, b, c, bv, acbv2, cav, t, r, i;

	if (f_v) {
		cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2" << endl;
	}
	nb_pts = 0;

#if 0
	u = evaluate_quadratic_form_x0x3mx1x2(Basis_line);
	if (u == 0) {
		Pts4[nb_pts * 2 + 0] = 1;
		Pts4[nb_pts * 2 + 1] = 0;
		nb_pts++;
	}
#endif

	u = evaluate_quadratic_form_x0x3mx1x2(Basis_line + 4);
	if (u == 0) {
		Pts4[nb_pts * 2 + 0] = 0;
		Pts4[nb_pts * 2 + 1] = 1;
		nb_pts++;
	}

	b0 = Basis_line[0];
	b1 = Basis_line[1];
	b2 = Basis_line[2];
	b3 = Basis_line[3];
	b4 = Basis_line[4];
	b5 = Basis_line[5];
	b6 = Basis_line[6];
	b7 = Basis_line[7];
	a = F->add(F->mult(b4, b7), F->negate(F->mult(b5, b6)));
	c = F->add(F->mult(b0, b3), F->negate(F->mult(b1, b2)));
	b = F->add4(F->mult(b0, b7), F->mult(b3, b4), F->negate(F->mult(b1, b6)), F->negate(F->mult(b2, b5)));
	if (f_v) {
		cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 a=" << a << " b=" << b << " c=" << c << endl;
	}
	if (a == 0) {
		cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 a == 0" << endl;
		exit(1);
	}
	av = F->inverse(a);
	if (EVEN(F->p)) {
		if (b == 0) {
			cav = F->mult(c, av);
			if (f_v) {
				cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 cav=" << cav << endl;
			}
			r = F->frobenius_power(cav, F->e - 1);
			if (f_v) {
				cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 r=" << r << endl;
			}
			Pts4[nb_pts * 2 + 0] = 1;
			Pts4[nb_pts * 2 + 1] = r;
			nb_pts++;
		}
		else {
			bv = F->inverse(b);
			acbv2 = F->mult4(a, c, bv, bv);
			if (f_v) {
				cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 acbv2=" << acbv2 << endl;
			}
			t = F->absolute_trace(acbv2);
			if (f_v) {
				cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 t=" << t << endl;
			}
			if (t == 0) {
				int Y2[2];
				int nb_sol;

				if (f_v) {
					cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 before solve_y2py" << endl;
				}
				solve_y2py(acbv2, Y2, nb_sol);
				if (f_v) {
					cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 after solve_y2py nb_sol= " << nb_sol << endl;
					Orbiter->Int_vec.print(cout, Y2, nb_sol);
					cout << endl;
				}
				if (nb_sol + nb_pts > 2) {
					cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 nb_sol + nb_pts > 2" << endl;
					exit(1);
				}
				for (i = 0; i < nb_sol; i++) {
					r = F->mult3(b, Y2[i], av);
					if (f_v) {
						cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 solution " << i << " r=" << r << endl;
					}
					Pts4[nb_pts * 2 + 0] = 1;
					Pts4[nb_pts * 2 + 1] = r;
					nb_pts++;
				}
			}
			else {
				// no solution
				if (f_v) {
					cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 no solution" << endl;
				}
				nb_pts = 0;
			}
		}
	}
	else {
		cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 odd characteristic not yet implemented" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 done" << endl;
	}
}

int linear_algebra::is_totally_isotropic_wrt_symplectic_form(
		int k, int n, int *Basis)
{
	int i, j;

	for (i = 0; i < k; i++) {
		for (j = i + 1; j < k; j++) {
			if (evaluate_symplectic_form(n, Basis + i * n, Basis + j * n)) {
				return FALSE;
			}
		}
	}
	return TRUE;
}

int linear_algebra::evaluate_monomial(int *monomial,
		int *variables, int nb_vars)
{
	int i, j, a, b, x;

	a = 1;
	for (i = 0; i < nb_vars; i++) {
		b = monomial[i];
		x = variables[i];
		for (j = 0; j < b; j++) {
			a = F->mult(a, x);
		}
	}
	return a;
}





}}
