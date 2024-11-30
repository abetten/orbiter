/*
 * quadratic_form.cpp
 *
 *  Created on: Jan 9, 2023
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace orthogonal_geometry {



quadratic_form::quadratic_form()
{
	epsilon = n = m = q = 0;
	f_even = false;
	form_c1 = form_c2 = form_c3 = 0;

	nb_points = 0;

	//std::string label_txt;
	//std::string label_tex;

	Poly = NULL;
	the_quadratic_form = NULL;
	the_monomial = NULL;

	Gram_matrix = NULL;

	F = NULL;

	Orthogonal_indexing = NULL;
}

quadratic_form::~quadratic_form()
{
	if (Poly) {
		FREE_OBJECT(Poly);
	}
	if (the_quadratic_form) {
		FREE_int(the_quadratic_form);
	}
	if (the_monomial) {
		FREE_int(the_monomial);
	}
	if (Gram_matrix) {
		FREE_int(Gram_matrix);
	}

	if (Orthogonal_indexing) {
		FREE_OBJECT(Orthogonal_indexing);
	}


}


void quadratic_form::init(
		int epsilon, int n,
		field_theory::finite_field *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	geometry::other_geometry::geometry_global Gg;


	if (f_v) {
		cout << "quadratic_form::init" << endl;
	}

	quadratic_form::epsilon = epsilon;
	quadratic_form::F = F;
	quadratic_form::n = n;

	q = F->q;
	m = Gg.Witt_index(epsilon, n - 1);

	if (epsilon == 1) {
		label_txt = "Op_" + std::to_string(n) + "_" + std::to_string(q);
	}
	else if (epsilon == -1) {
		label_txt = "Om_" + std::to_string(n) + "_" + std::to_string(q);
	}
	else if (epsilon == 0) {
		label_txt = "O_" + std::to_string(n) + "_" + std::to_string(q);
	}


	if (epsilon == 1) {
		label_tex = "O^+(" + std::to_string(n) + "," + std::to_string(q) + ")";
	}
	else if (epsilon == -1) {
		label_tex = "O^-(" + std::to_string(n) + "," + std::to_string(q) + ")";
	}
	else if (epsilon == 0) {
		label_tex = "O(" + std::to_string(n) + "," + std::to_string(q) + ")";
	}


	if (f_v) {
		cout << "quadratic_form::init: epsilon=" << epsilon
			<< " n=" << n << " (= vector space dimension)"
			<< " m=" << m << " (= Witt index)"
			<< " q=" << q
			<< " label_txt=" << label_txt
			<< " label_tex=" << label_tex
			<< " verbose_level=" << verbose_level
			<< endl;
	}

	if (EVEN(q)) {
		f_even = true;
	}
	else {
		f_even = false;
	}

	if (f_v) {
		cout << "quadratic_form::init "
				"before init_form_and_Gram_matrix" << endl;
	}
	init_form_and_Gram_matrix(verbose_level);
	if (f_v) {
		cout << "quadratic_form::init "
				"after init_form_and_Gram_matrix" << endl;
	}


	Orthogonal_indexing = NEW_OBJECT(orthogonal_indexing);

	if (f_v) {
		cout << "quadratic_form::init "
				"before Orthogonal_indexing->init" << endl;
	}
	Orthogonal_indexing->init(this, verbose_level);
	if (f_v) {
		cout << "quadratic_form::init "
				"after Orthogonal_indexing->init" << endl;
	}


	nb_points = Gg.nb_pts_Qepsilon(epsilon, n - 1, q);
	if (f_v) {
		cout << "quadratic_form::init "
				"nb_points = " << nb_points << endl;
	}

	if (f_v) {
		cout << "quadratic_form::init done" << endl;
	}
}


void quadratic_form::init_form_and_Gram_matrix(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx;

	if (f_v) {
		cout << "quadratic_form::init_form_and_Gram_matrix" << endl;
	}
	form_c1 = 1;
	form_c2 = 0;
	form_c3 = 0;

	Poly = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "quadratic_form::init_form_and_Gram_matrix "
				"before Poly->init" << endl;
	}
	Poly->init(F,
			n /* nb_vars */, 2 /* degree */,
			t_LEX,
			verbose_level);
	if (f_v) {
		cout << "quadratic_form::init_form_and_Gram_matrix "
				"after Poly->init" << endl;
	}
	the_quadratic_form = NEW_int(Poly->get_nb_monomials());
	Int_vec_zero(the_quadratic_form, Poly->get_nb_monomials());

	the_monomial = NEW_int(n);
	Int_vec_zero(the_monomial, n);

	if (epsilon == -1) {
		choose_anisotropic_form(verbose_level);

		Int_vec_zero(the_monomial, n);
		the_monomial[n - 2] = 2;
		idx = Poly->index_of_monomial(the_monomial);
		the_quadratic_form[idx] =
				F->add(the_quadratic_form[idx], form_c1);

		Int_vec_zero(the_monomial, n);
		the_monomial[n - 2] = 1;
		the_monomial[n - 1] = 1;
		idx = Poly->index_of_monomial(the_monomial);
		the_quadratic_form[idx] =
				F->add(the_quadratic_form[idx], form_c2);

		Int_vec_zero(the_monomial, n);
		the_monomial[n - 1] = 2;
		idx = Poly->index_of_monomial(the_monomial);
		the_quadratic_form[idx] =
				F->add(the_quadratic_form[idx], form_c3);

	}
	else if (epsilon == 0) {

		Int_vec_zero(the_monomial, n);
		the_monomial[0] = 2;
		idx = Poly->index_of_monomial(the_monomial);
		the_quadratic_form[idx] =
				F->add(the_quadratic_form[idx], form_c1);

	}

	int i, j, u;
	int offset;

	if (epsilon == 0) {
		offset = 1;
	}
	else {
		offset = 0;
	}

	for (i = 0; i < m; i++) {
		j = 2 * i;
		u = offset + j;

		Int_vec_zero(the_monomial, n);
		the_monomial[u] = 1;
		the_monomial[u + 1] = 1;
		idx = Poly->index_of_monomial(the_monomial);
		the_quadratic_form[idx] =
				F->add(the_quadratic_form[idx], 1);

			// X_u * X_{u+1}
	}

	if (f_v) {
		cout << "quadratic_form::init_form_and_Gram_matrix "
				"the quadratic form is: ";
		Poly->print_equation_tex(cout, the_quadratic_form);
		cout << endl;
	}


	if (f_v) {
		cout << "quadratic_form::init_form_and_Gram_matrix "
				"before make_Gram_matrix" << endl;
	}
	make_Gram_matrix(verbose_level);
	if (f_v) {
		cout << "quadratic_form::init_form_and_Gram_matrix "
				"after make_Gram_matrix" << endl;
	}
	if (f_v) {
		cout << "quadratic_form::init_form_and_Gram_matrix done" << endl;
	}
}

void quadratic_form::make_Gram_matrix(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, u, offset = 0;
	geometry::other_geometry::geometry_global Gg;

	if (f_v) {
		cout << "quadratic_form::make_Gram_matrix" << endl;
	}
	Gram_matrix = NEW_int(n * n);
	Int_vec_zero(Gram_matrix, n * n);

	//m = Gg.Witt_index(epsilon, n - 1);

	if (epsilon == 0) {
		Gram_matrix[0 * n + 0] = F->add(form_c1, form_c1);
		offset = 1;
	}
	else if (epsilon == 1) {
	}
	else if (epsilon == -1) {
		Gram_matrix[(n - 2) * n + n - 2] = F->add(form_c1, form_c1);
		Gram_matrix[(n - 2) * n + n - 1] = form_c2;
		Gram_matrix[(n - 1) * n + n - 2] = form_c2;
		Gram_matrix[(n - 1) * n + n - 1] = F->add(form_c3, form_c3);
	}
	for (i = 0; i < m; i++) {
		j = 2 * i;
		u = offset + j;
		Gram_matrix[u * n + u + 1] = 1;
			// X_u * Y_{u+1}
		Gram_matrix[(u + 1) * n + u] = 1;
			// X_{u+1} * Y_u
	}
	if (f_v) {
		cout << "quadratic_form::make_Gram_matrix done" << endl;
	}
}



int quadratic_form::evaluate_quadratic_form(
		int *v, int stride)
{
	int f;

	if (epsilon == 1) {
		f = evaluate_hyperbolic_quadratic_form(v, stride);
	}
	else if (epsilon == 0) {
		f = evaluate_parabolic_quadratic_form(v, stride);
	}
	else if (epsilon == -1) {
		f = evaluate_elliptic_quadratic_form(v, stride);
	}
	else {
		cout << "quadratic_form::evaluate_quadratic_form "
				"epsilon = " << epsilon << endl;
		exit(1);
	}
	return f;
}

int quadratic_form::evaluate_hyperbolic_quadratic_form(
		int *v, int stride)
{
	int alpha = 0, beta, i;

	for (i = 0; i < m; i++) {
		beta = F->mult(v[2 * i * stride], v[(2 * i + 1) * stride]);
		alpha = F->add(alpha, beta);
	}
	return alpha;
}

int quadratic_form::evaluate_hyperbolic_quadratic_form_with_m(
		int *v, int stride, int m)
{
	int alpha = 0, beta, i;

	for (i = 0; i < m; i++) {
		beta = F->mult(v[2 * i * stride], v[(2 * i + 1) * stride]);
		alpha = F->add(alpha, beta);
	}
	return alpha;
}

int quadratic_form::evaluate_parabolic_quadratic_form(
		int *v, int stride)
{
	int a, b, c;

	a = evaluate_hyperbolic_quadratic_form(v + stride, stride);
	b = F->mult(v[0], v[0]);
	c = F->add(a, b);
	return c;
}

int quadratic_form::evaluate_elliptic_quadratic_form(
		int *v, int stride)
{
	int a, x1, x2, b, c, d;

	a = evaluate_hyperbolic_quadratic_form(v, stride);
	x1 = v[2 * m * stride];
	x2 = v[(2 * m + 1) * stride];
	b = F->mult(x1, x1);
	b = F->mult(form_c1, b);
	c = F->mult(x1, x2);
	c = F->mult(form_c2, c);
	d = F->mult(x2, x2);
	d = F->mult(form_c3, d);
	a = F->add(a, b);
	c = F->add(a, c);
	c = F->add(d, c);
	return c;

}



int quadratic_form::evaluate_bilinear_form(
		int *u, int *v, int stride)
{
	int f;

	if (epsilon == 1) {
		f = evaluate_hyperbolic_bilinear_form(u, v, stride, m);
	}
	else if (epsilon == 0) {
		f = evaluate_parabolic_bilinear_form(u, v, stride, m);
	}
	else if (epsilon == -1) {
		if (stride != 1) {
			cout << "quadratic_form::evaluate_bilinear_form "
					"stride != 1 and epsilon == -1" << endl;
			exit(1);
		}
		f = evaluate_bilinear_form_Gram_matrix(u, v);
	}
	else {
		cout << "quadratic_form::evaluate_bilinear_form "
				"epsilon = " << epsilon << endl;
		exit(1);
	}
	return f;
}


int quadratic_form::evaluate_hyperbolic_bilinear_form(
		int *u, int *v, int stride, int m)
{
	int alpha = 0, beta1, beta2, i;

	for (i = 0; i < m; i++) {
		beta1 = F->mult(u[2 * i * stride], v[(2 * i + 1) * stride]);
		beta2 = F->mult(u[(2 * i + 1) * stride], v[2 * i * stride]);
		alpha = F->add(alpha, beta1);
		alpha = F->add(alpha, beta2);
	}
	return alpha;
}

int quadratic_form::evaluate_parabolic_bilinear_form(
		int *u, int *v, int stride, int m)
{
	int a, b, c;

	a = evaluate_hyperbolic_bilinear_form(
			u + stride, v + stride, stride, m);
	if (f_even) {
		return a;
	}
	b = F->mult(2, u[0]);
	b = F->mult(b, v[0]);
	c = F->add(a, b);
	return c;
}

int quadratic_form::evaluate_bilinear_form_Gram_matrix(
		int *u, int *v)
{
	int i, j, a, b, c, e, f;

	f = 0;
	for (i = 0; i < n; i++) {
		a = u[i];
		for (j = 0; j < n; j++) {
			b = Gram_matrix[i * n + j];
			c = v[j];
			e = F->mult(a, b);
			e = F->mult(e, c);
			f = F->add(f, e);
			}
		}
	return f;
}


void quadratic_form::report_quadratic_form(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quadratic_form::report_quadratic_form" << endl;
	}
	l1_interfaces::latex_interface Li;

	ost << "The quadratic form is: " << endl;
	ost << "$$" << endl;
	Poly->print_equation_tex(ost, the_quadratic_form);
	ost << " = 0";
	ost << "$$" << endl;
	ost << "The associated Gram matrix is: " << endl;
	ost << "$$" << endl;

	ost << "\\left[" << endl;

	Li.print_integer_matrix_tex(ost, Gram_matrix, n, n);

	ost << "\\right]" << endl;

	ost << "$$" << endl;

	if (f_v) {
		cout << "quadratic_form::report_quadratic_form done" << endl;
	}

}


long int quadratic_form::find_root(
		int rk2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *x, *y, *z;
	int d, i;
	int y2_minus_y3, minus_y1, y3_minus_y2, a, a2, u, v;
	long int root;

	d = n;
	//k = d - 1;
	if (f_v) {
		cout << "quadratic_form::find_root "
				"rk2=" << rk2 << endl;
	}
	if (rk2 == 0) {
		cout << "quadratic_form::find_root: "
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

	//Orthogonal_indexing->Q_epsilon_unrank(y, 1, epsilon, k,
	//		form_c1, form_c2, form_c3, rk2, 0 /* verbose_level */);
	unrank_point(y, rk2, 0 /* verbose_level */);
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
		cout << "quadratic_form::find_root "
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
		cout << "quadratic_form::find_root "
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

	u = evaluate_bilinear_form(z, x, 1);
	if (u == 0) {
		cout << "u=" << u << endl;
		exit(1);
	}
	v = evaluate_bilinear_form(z, y, 1);
	if (v == 0) {
		cout << "v=" << v << endl;
		exit(1);
	}
	//root = Orthogonal_indexing->Q_epsilon_rank(z, 1, epsilon, k,
	//		form_c1, form_c2, form_c3, 0 /* verbose_level */);
	root = rank_point(z, 0 /* verbose_level */);
	if (f_v) {
		cout << "quadratic_form::find_root "
				"root=" << root << endl;
	}

	FREE_int(x);
	FREE_int(y);
	FREE_int(z);

	return root;
}

void quadratic_form::Siegel_Transformation(
	int *M, int *v, int *u, int verbose_level)
// if u is singular and v \in \la u \ra^\perp, then
// \pho_{u,v}(x) := x + \beta(x,v) u - \beta(x,u) v - Q(v) \beta(x,u) u
// is called the Siegel transform (see Taylor p. 148)
// Here Q is the quadratic form
// and \beta is the corresponding bilinear form
{
	int f_v = (verbose_level >= 1);
	int d = n; // k + 1;
	int i, j, Qv, a, b, c, e;
	//int *Gram;
	//int *new_Gram;
	int *N1;
	int *N2;
	int *w;

	if (f_v) {
		cout << "quadratic_form::Siegel_Transformation "
				"v=";
		Int_vec_print(cout, v, d);
		cout << " u=";
		Int_vec_print(cout, u, d);
		cout << endl;
	}
	//Gram_matrix(epsilon, k,
	//		form_c1, form_c2, form_c3, Gram, verbose_level);
	Qv = evaluate_quadratic_form(v, 1 /*stride*/);
			//epsilon, k, form_c1, form_c2, form_c3);
	if (f_v) {
		cout << "Qv=" << Qv << endl;
	}
	N1 = NEW_int(d * d);
	N2 = NEW_int(d * d);
	//new_Gram = NEW_int(d * d);
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
			b = Gram_matrix[i * d + j];
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
			b = Gram_matrix[i * d + j];
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
		cout << "quadratic_form::Siegel_Transformation "
				"Siegel matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, M, d, d, d, 2);
		//GFq.transform_form_matrix(M, Gram, new_Gram, N1, N2, d);
		//cout << "transformed Gram matrix:" << endl;
		//print_integer_matrix_width(cout, new_Gram, d, d, d, 2);
		//cout << endl;
	}

	//FREE_int(Gram);
	//FREE_int(new_Gram);
	FREE_int(N1);
	FREE_int(N2);
	FREE_int(w);
}


void quadratic_form::Siegel_map_between_singular_points(
		int *T,
		long int rk_from, long int rk_to, long int root,
	//int epsilon, int algebraic_dimension,
	//int form_c1, int form_c2, int form_c3, int *Gram_matrix,
	int verbose_level)
// root is not perp to from and to.
{
	int *B, *Bv, *w, *z, *x;
	int i, j, a, b, av, bv, minus_one;
	int d; //, epsilon, form_c1, form_c2, form_c3;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "quadratic_form::Siegel_map_between_singular_points "
				"rk_from=" << rk_from
				<< " rk_to=" << rk_to
				<< " root=" << root << endl;
		}
	d = n; // algebraic_dimension;
	//k = d - 1;

	B = NEW_int(d * d);
	Bv = NEW_int(d * d);
	w = NEW_int(d);
	z = NEW_int(d);
	x = NEW_int(d);
	//Orthogonal_indexing->Q_epsilon_unrank(B, 1, epsilon, k,
	//		form_c1,
	//		form_c2,
	//		form_c3,
	//		root, 0 /* verbose_level */);
	unrank_point(B, root, 0 /* verbose_level */);

	//Orthogonal_indexing->Q_epsilon_unrank(B + d, 1, epsilon, k,
	//		form_c1,
	//		form_c2,
	//		form_c3,
	//		rk_from, 0 /* verbose_level */);
	unrank_point(B + d, rk_from, 0 /* verbose_level */);

	//Orthogonal_indexing->Q_epsilon_unrank(w, 1, epsilon, k,
	//		form_c1,
	//		form_c2,
	//		form_c3,
	//		rk_to, 0 /* verbose_level */);
	unrank_point(w, rk_to, 0 /* verbose_level */);

	if (f_vv) {
		cout << "    root=";
		Int_vec_print(cout, B, d);
		cout << endl;
		cout << " rk_from=";
		Int_vec_print(cout, B + d, d);
		cout << endl;
		cout << "   rk_to=";
		Int_vec_print(cout, w, d);
		cout << endl;
	}

	a = evaluate_bilinear_form(B, B + d, 1);
	b = evaluate_bilinear_form(B, w, 1);
	av = F->inverse(a);
	bv = F->inverse(b);
	for (i = 0; i < d; i++) {
		B[d + i] = F->mult(B[d + i], av);
		w[i] = F->mult(w[i], bv);
	}
	if (f_vv) {
		cout << "after scaling:" << endl;
		cout << " rk_from=";
		Int_vec_print(cout, B + d, d);
		cout << endl;
		cout << "   rk_to=";
		Int_vec_print(cout, w, d);
		cout << endl;
	}
	for (i = 2; i < d; i++) {
		for (j = 0; j < d; j++) {
			B[i * d + j] = 0;
		}
	}

	if (f_vv) {
		cout << "before perp, the matrix B is:" << endl;
		Int_vec_print_integer_matrix(cout, B, d, d);
	}
	F->Linear_algebra->perp(d, 2, B, Gram_matrix, 0 /* verbose_level */);
	if (f_vv) {
		cout << "after perp, the matrix B is:" << endl;
		Int_vec_print_integer_matrix(cout, B, d, d);
	}
	F->Linear_algebra->invert_matrix(B, Bv, d, 0 /* verbose_level */);
	if (f_vv) {
		cout << "the matrix Bv = B^{-1} is:" << endl;
		Int_vec_print_integer_matrix(cout, B, d, d);
	}
	F->Linear_algebra->mult_matrix_matrix(
			w, Bv, z, 1, d, d, 0 /* verbose_level */);
	if (f_vv) {
		cout << "the coefficient vector z = w * Bv is:" << endl;
		Int_vec_print(cout, z, d);
		cout << endl;
	}
	z[0] = 0;
	z[1] = 0;
	if (f_vv) {
		cout << "we zero out the first two coordinates:" << endl;
		Int_vec_print(cout, z, d);
		cout << endl;
	}
	F->Linear_algebra->mult_matrix_matrix(
			z, B, x, 1, d, d, 0 /* verbose_level */);
	if (f_vv) {
		cout << "the vector x = z * B is:" << endl;
		Int_vec_print(cout, x, d);
		cout << endl;
	}
	minus_one = F->negate(1);
	for (i = 0; i < d; i++) {
		x[i] = F->mult(x[i], minus_one);
	}
	if (f_vv) {
		cout << "the vector -x is:" << endl;
		Int_vec_print(cout, x, d);
		cout << endl;
	}
	Siegel_Transformation(T, x, B, verbose_level - 2);
	if (f_v) {
		cout << "quadratic_form::Siegel_map_between_singular_points "
				"the Siegel transformation is:" << endl;
		Int_vec_print_integer_matrix(cout, T, d, d);
	}
	FREE_int(B);
	FREE_int(Bv);
	FREE_int(w);
	FREE_int(z);
	FREE_int(x);
}

void quadratic_form::choose_anisotropic_form(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::unipoly_domain FX(F);
	ring_theory::unipoly_object m;

	if (f_v) {
		cout << "quadratic_form::choose_anisotropic_form "
				"over GF(" << F->q << ")" << endl;
		}

	if (ODD(F->q)) {
		form_c1 = 1;
		form_c2 = 0;
		form_c3 = F->negate(F->primitive_element());
	}
	else {
		knowledge_base::knowledge_base K;
		string poly;

		K.get_primitive_polynomial(poly, F->q, 2, 0);

		FX.create_object_by_rank_string(m,
				poly,
				verbose_level);

		//FX.create_object_by_rank_string(m,
		//get_primitive_polynomial(GFq.p, 2 * GFq.e, 0), verbose_level);

		if (f_v) {
			cout << "quadratic_form::choose_anisotropic_form "
					"choosing the following primitive polynomial:" << endl;
			FX.print_object(m, cout); cout << endl;
		}

		int *rep = (int *) m;
		int *coeff = rep + 1;
		form_c1 = coeff[2];
		form_c2 = coeff[1];
		form_c3 = coeff[0];
	}

#if 0
	finite_field GFQ;

	GFQ.init(GFq.q * GFq.q, 0);
	cout << "quadratic_form::choose_anisotropic_form "
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
		cout << "quadratic_form::choose_anisotropic_form "
				"over GF(" << F->q << "): choosing c1=" << form_c1 << ", c2=" << form_c2
				<< ", c3=" << form_c3 << endl;
	}
}

void quadratic_form::unrank_point(
		int *v, long int a, int verbose_level)
{
	Orthogonal_indexing->Q_epsilon_unrank_private(
			v, 1 /* stride */, epsilon, n - 1 /* proj dimension */,
			form_c1, form_c2, form_c3, a, verbose_level);
}

long int quadratic_form::rank_point(
		int *v, int verbose_level)
{
	long int a;

	a = Orthogonal_indexing->Q_epsilon_rank_private(
			v, 1 /* stride */, epsilon, n - 1 /* proj dimension */,
			form_c1, form_c2, form_c3, verbose_level);
	return a;
}

void quadratic_form::make_collinearity_graph(
		int *&Adj, int &N,
		long int *Set, int sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quadratic_form::make_collinearity_graph" << endl;
	}

	int i, j;
	int d; //, nb_e, nb_inc;
	int *v1, *v2;
	long int Nb_points;


	d = n; // algebraic dimension

	v1 = NEW_int(d);
	v2 = NEW_int(d);

	if (f_v) {
		cout << "quadratic_form::make_collinearity_graph" << endl;
	}


	Nb_points = nb_points;

	if (f_v) {
		cout << "quadratic_form::make_collinearity_graph "
				"number of points = " << Nb_points << endl;
	}

	N = sz;

	if (f_v) {
		cout << "quadratic_form::make_collinearity_graph field:" << endl;
		F->Io->print();
	}




#if 0
	if (f_list_points) {
		for (i = 0; i < N; i++) {
			F->Q_epsilon_unrank(v, 1, epsilon, n, c1, c2, c3, i, 0 /* verbose_level */);
			cout << i << " : ";
			int_vec_print(cout, v, n + 1);
			j = F->Q_epsilon_rank(v, 1, epsilon, n, c1, c2, c3, 0 /* verbose_level */);
			cout << " : " << j << endl;

			}
		}
#endif


	if (f_v) {
		cout << "quadratic_form::make_collinearity_graph "
				"allocating adjacency matrix" << endl;
	}
	Adj = NEW_int(N * N);
	if (f_v) {
		cout << "quadratic_form::make_collinearity_graph "
				"allocating adjacency matrix was successful" << endl;
	}

	long int a, b;
	int val;


	for (i = 0; i < sz; i++) {

		a = Set[i];

		if (a < 0 || a >= Nb_points) {
			cout << "quadratic_form::make_collinearity_graph out of range" << endl;
			exit(1);
		}
	}

	//nb_e = 0;
	//nb_inc = 0;
	for (i = 0; i < sz; i++) {

		a = Set[i];


		unrank_point(v1, a, 0 /* verbose_level */);

		for (j = i + 1; j < sz; j++) {

			b = Set[j];

			unrank_point(v2, b, 0 /* verbose_level */);

			val = evaluate_bilinear_form(v1, v2, 1);

			if (val == 0) {
				//nb_e++;
				Adj[i * N + j] = 1;
				Adj[j * N + i] = 1;
			}
			else {
				Adj[i * N + j] = 0;
				Adj[j * N + i] = 0;
				//nb_inc++;
			}
		}
		Adj[i * N + i] = 0;
	}
	if (f_v) {
		cout << "quadratic_form::make_collinearity_graph "
				"The adjacency matrix of the collinearity graph has been computed" << endl;
	}


	FREE_int(v1);
	FREE_int(v2);

	if (f_v) {
		cout << "quadratic_form::make_collinearity_graph done" << endl;
	}
}


void quadratic_form::make_affine_polar_graph(
		int *&Adj, int &N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quadratic_form::make_affine_polar_graph" << endl;
	}

	int i, j;
	int d;
	int *v1, *v2, *v3;
	long int Nb_points;


	d = n; // algebraic dimension

	v1 = NEW_int(d);
	v2 = NEW_int(d);
	v3 = NEW_int(d);

	if (f_v) {
		cout << "quadratic_form::make_affine_polar_graph" << endl;
	}


	Nb_points = nb_points;

	if (f_v) {
		cout << "quadratic_form::make_affine_polar_graph "
				"number of points = " << Nb_points << endl;
	}

	geometry::other_geometry::geometry_global Geometry;

	N = Geometry.nb_AG_elements(
			d, q);


	if (f_v) {
		cout << "quadratic_form::make_affine_polar_graph field:" << endl;
		F->Io->print();
	}



	if (f_v) {
		cout << "quadratic_form::make_affine_polar_graph "
				"allocating adjacency matrix" << endl;
	}
	Adj = NEW_int(N * N);
	if (f_v) {
		cout << "quadratic_form::make_affine_polar_graph "
				"allocating adjacency matrix was successful" << endl;
	}
	Int_vec_zero(Adj, N * N);


	int h, u;


	for (i = 0; i < N; i++) {

		Geometry.AG_element_unrank(
				q, v1, 1, d, i);

		if (f_v) {
			cout << "vertex " << setw(3) << i << " = ";
			Int_vec_print(cout, v1, d);
			cout << " is adjacent to " << endl;
		}

		for (h = 0; h < nb_points; h++) {

			unrank_point(v2, h, 0 /* verbose_level */);

			for (u = 0; u < d; u++) {
				v3[u] = F->add(v1[u], v2[u]);
			}

			j = Geometry.AG_element_rank(
					q, v3, 1, d);

			if (f_v) {
				cout << "  vertex " << setw(3) << j << " = ";
				Int_vec_print(cout, v3, d);
				cout << endl;
			}


			Adj[i * N + j] = 1;
			Adj[j * N + i] = 1;
		}
	}
	if (f_v) {
		cout << "quadratic_form::make_affine_polar_graph "
				"The adjacency matrix of the polar graph has been computed" << endl;
	}


	FREE_int(v1);
	FREE_int(v2);
	FREE_int(v3);

	if (f_v) {
		cout << "quadratic_form::make_affine_polar_graph done" << endl;
	}
}

}}}

