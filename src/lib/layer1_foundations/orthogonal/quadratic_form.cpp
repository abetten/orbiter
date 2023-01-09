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
	f_even = FALSE;
	form_c1 = form_c2 = form_c3 = 0;

	//std::string label_txt;
	//std::string label_tex;

	Poly = NULL;
	the_quadratic_form = NULL;
	the_monomial = NULL;

	Gram_matrix = NULL;

	F = NULL;
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


}


void quadratic_form::init(int epsilon, int n,
		field_theory::finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	geometry::geometry_global Gg;


	if (f_v) {
		cout << "quadratic_form::init" << endl;
	}

	quadratic_form::epsilon = epsilon;
	quadratic_form::F = F;
	quadratic_form::n = n;

	q = F->q;
	m = Gg.Witt_index(epsilon, n - 1);

	char str[1000];

	if (epsilon == 1) {
		snprintf(str, sizeof(str), "Op_%d_%d", n, q);
	}
	else if (epsilon == -1) {
		snprintf(str, sizeof(str), "Om_%d_%d", n, q);
	}
	else if (epsilon == 0) {
		snprintf(str, sizeof(str), "O_%d_%d", n, q);
	}

	label_txt.assign(str);

	if (epsilon == 1) {
		snprintf(str, sizeof(str), "O^+(%d,%d)", n, q);
	}
	else if (epsilon == -1) {
		snprintf(str, sizeof(str), "O^-(%d,%d)", n, q);
	}
	else if (epsilon == 0) {
		snprintf(str, sizeof(str), "O(%d,%d)", n, q);
	}


	label_tex.assign(str);

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
		f_even = TRUE;
	}
	else {
		f_even = FALSE;
	}

	if (f_v) {
		cout << "quadratic_form::init before init_form_and_Gram_matrix" << endl;
	}
	init_form_and_Gram_matrix(verbose_level);
	if (f_v) {
		cout << "quadratic_form::init after init_form_and_Gram_matrix" << endl;
	}

	if (f_v) {
		cout << "quadratic_form::init done" << endl;
	}
}


void quadratic_form::init_form_and_Gram_matrix(int verbose_level)
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
		cout << "quadratic_form::init_form_and_Gram_matrix before Poly->init" << endl;
	}
	Poly->init(F,
			n /* nb_vars */, 2 /* degree */,
			t_LEX,
			verbose_level);
	if (f_v) {
		cout << "quadratic_form::init_form_and_Gram_matrix after Poly->init" << endl;
	}
	the_quadratic_form = NEW_int(Poly->get_nb_monomials());
	Int_vec_zero(the_quadratic_form, Poly->get_nb_monomials());

	the_monomial = NEW_int(n);
	Int_vec_zero(the_monomial, n);

	if (epsilon == -1) {
		F->Linear_algebra->choose_anisotropic_form(
				form_c1, form_c2, form_c3, verbose_level);

		Int_vec_zero(the_monomial, n);
		the_monomial[n - 2] = 2;
		idx = Poly->index_of_monomial(the_monomial);
		the_quadratic_form[idx] = F->add(the_quadratic_form[idx], form_c1);

		Int_vec_zero(the_monomial, n);
		the_monomial[n - 2] = 1;
		the_monomial[n - 1] = 1;
		idx = Poly->index_of_monomial(the_monomial);
		the_quadratic_form[idx] = F->add(the_quadratic_form[idx], form_c2);

		Int_vec_zero(the_monomial, n);
		the_monomial[n - 1] = 2;
		idx = Poly->index_of_monomial(the_monomial);
		the_quadratic_form[idx] = F->add(the_quadratic_form[idx], form_c3);

	}
	else if (epsilon == 0) {

		Int_vec_zero(the_monomial, n);
		the_monomial[0] = 2;
		idx = Poly->index_of_monomial(the_monomial);
		the_quadratic_form[idx] = F->add(the_quadratic_form[idx], form_c1);

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
		the_quadratic_form[idx] = F->add(the_quadratic_form[idx], 1);

			// X_u * X_{u+1}
	}

	if (f_v) {
		cout << "quadratic_form::init_form_and_Gram_matrix the quadratic form is: ";
		Poly->print_equation_tex(cout, the_quadratic_form);
		cout << endl;
	}


	if (f_v) {
		cout << "quadratic_form::init_form_and_Gram_matrix computing Gram matrix" << endl;
	}
	F->Linear_algebra->Gram_matrix(
			epsilon, n - 1,
			form_c1, form_c2, form_c3, Gram_matrix,
			verbose_level);
	if (f_v) {
		cout << "quadratic_form::init_form_and_Gram_matrix "
				"computing Gram matrix done" << endl;
	}
	if (f_v) {
		cout << "quadratic_form::init_form_and_Gram_matrix done" << endl;
	}
}


int quadratic_form::evaluate_quadratic_form(int *v, int stride)
{
	if (epsilon == 1) {
		return evaluate_hyperbolic_quadratic_form(v, stride, m);
	}
	else if (epsilon == 0) {
		int a, b, c;

		a = evaluate_hyperbolic_quadratic_form(v + stride, stride, m);
		//if (f_even)
			//return a;
		b = F->mult(v[0], v[0]);
		c = F->add(a, b);
		return c;
	}
	else if (epsilon == -1) {
		int a, x1, x2, b, c, d;

		a = evaluate_hyperbolic_quadratic_form(v, stride, m);
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
	else {
		cout << "quadratic_form::evaluate_quadratic_form epsilon = " << epsilon << endl;
		exit(1);
	}
}

int quadratic_form::evaluate_bilinear_form(int *u, int *v, int stride)
{
	if (epsilon == 1) {
		return evaluate_hyperbolic_bilinear_form(u, v, stride, m);
	}
	else if (epsilon == 0) {
		return evaluate_parabolic_bilinear_form(u, v, stride, m);
	}
	else if (epsilon == -1) {
		return F->Linear_algebra->evaluate_bilinear_form(
				u, v, n, Gram_matrix);
	}
	else {
		cout << "quadratic_form::evaluate_bilinear_form epsilon = " << epsilon << endl;
		exit(1);
	}
}

int quadratic_form::evaluate_hyperbolic_quadratic_form(
		int *v, int stride, int m)
{
	int alpha = 0, beta, i;

	for (i = 0; i < m; i++) {
		beta = F->mult(v[2 * i * stride], v[(2 * i + 1) * stride]);
		alpha = F->add(alpha, beta);
	}
	return alpha;
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

void quadratic_form::report_quadratic_form(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quadratic_form::report_quadratic_form" << endl;
	}

	ost << "The quadratic form is: " << endl;
	ost << "$$" << endl;
	Poly->print_equation_tex(ost, the_quadratic_form);
	ost << " = 0";
	ost << "$$" << endl;

	if (f_v) {
		cout << "quadratic_form::report_quadratic_form done" << endl;
	}

}


}}}

