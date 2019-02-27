/*
 * polynomial_double_domain.cpp
 *
 *  Created on: Feb 17, 2019
 *      Author: betten
 */


#include "foundations.h"

using namespace std;



#define EPSILON 0.01

namespace orbiter {
namespace foundations {


polynomial_double_domain::polynomial_double_domain()
{
	alloc_length = 0;
}

polynomial_double_domain::~polynomial_double_domain()
{
	alloc_length = 0;
}

void polynomial_double_domain::init(int alloc_length)
{
	polynomial_double_domain::alloc_length = alloc_length;
}

polynomial_double *polynomial_double_domain::create_object()
{
	polynomial_double *p;

	p = NEW_OBJECT(polynomial_double);
	p->init(alloc_length);
	return p;
}

void polynomial_double_domain::mult(polynomial_double *A,
		polynomial_double *B, polynomial_double *C)
{
	int i, j;

	C->degree = A->degree + B->degree;
	for (i = 0; i <= C->degree; i++) {
		C->coeff[i] = 0;
	}

	for (i = 0; i <= A->degree; i++) {
		for (j = 0; j <= B->degree; j++) {
			C->coeff[i + j] += A->coeff[i] * B->coeff[j];
		}
	}
}

void polynomial_double_domain::add(polynomial_double *A,
		polynomial_double *B, polynomial_double *C)
{
	int i;
	double a, b, c;

	C->degree = MAXIMUM(A->degree, B->degree);
	for (i = 0; i <= C->degree; i++) {
		C->coeff[i] = 0;
	}

	for (i = 0; i <= C->degree; i++) {
		if (i <= A->degree) {
			a = A->coeff[i];
		}
		else {
			a = 0.;
		}
		if (i <= B->degree) {
			b = B->coeff[i];
		}
		else {
			b = 0.;
		}
		c = a + b;
		//cout << "add i=" << i << " a=" << a << " b=" << b << " c=" << c << endl;
		C->coeff[i] = c;
	}
}

void polynomial_double_domain::mult_by_scalar_in_place(
		polynomial_double *A,
		double lambda)
{
	int i;

	for (i = 0; i <= A->degree; i++) {
		A->coeff[i] *= lambda;
		//cout << "scalar multiply: " << A->coeff[i] << endl;
	}
}

void polynomial_double_domain::copy(polynomial_double *A,
		polynomial_double *B)
{
	int i;

	B->degree = A->degree;
	for (i = 0; i <= A->degree; i++) {
		B->coeff[i] = A->coeff[i];
	}
}

void polynomial_double_domain::determinant_over_polynomial_ring(
		polynomial_double *P,
		polynomial_double *det, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	polynomial_double *Q;
	polynomial_double *a, *b, *c, *d;
	int i, j, h, u;

	if (f_v) {
		cout << "polynomial_double_domain::determinant_over_polynomial_ring" << endl;
#if 0
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				P[i * n + j].print(cout);
				cout << "; ";
			}
		cout << endl;
		}
#endif
	}
	if (n == 1) {
		copy(P, det);
	}
	else {
		a = NEW_OBJECT(polynomial_double);
		b = NEW_OBJECT(polynomial_double);
		c = NEW_OBJECT(polynomial_double);
		d = NEW_OBJECT(polynomial_double);

		Q = NEW_OBJECTS(polynomial_double, (n - 1) * (n - 1));

		a->init(n + 1);
		b->init(n + 1);
		c->init(n + 1);
		d->init(n + 1);
		for (i = 0; i < n - 1; i++) {
			for (j = 0; j < n - 1; j++) {
				Q[i * (n - 1) + j].init(n + 1);
			}
		}

		c->degree = 0;
		c->coeff[0] = 0;

		for (h = 0; h < n; h++) {
			//cout << "h=" << h << " / " << n << ":" << endl;

			u = 0;
			for (i = 0; i < n; i++) {
				if (i == h) {
					continue;
				}
				for (j = 1; j < n; j++) {
					copy(&P[i * n + j], &Q[u * (n - 1) + j - 1]);
				}
				u++;
			}
			determinant_over_polynomial_ring(Q, a, n - 1, verbose_level);


			mult(a, &P[h * n + 0], b);

			if (h % 2) {
				mult_by_scalar_in_place(b, -1.);
			}


			add(b, c, d);


			copy(d, c);

		}
		copy(c, det);

		FREE_OBJECTS(Q);
		FREE_OBJECT(a);
		FREE_OBJECT(b);
		FREE_OBJECT(c);
		FREE_OBJECT(d);
	}
#if 0
	cout << "polynomial_double_domain::determinant_over_polynomial_ring "
			"the determinant of " << endl;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			P[i * n + j].print(cout);
			cout << "; ";
		}
	cout << endl;
	}
	cout << "is: ";
	det->print(cout);
	cout << endl;
#endif
	if (f_v) {
		cout << "polynomial_double_domain::determinant_over_"
				"polynomial_ring done" << endl;
	}

}

void polynomial_double_domain::find_all_roots(polynomial_double *p,
		double *lambda, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, d;
	double rem;

	if (f_v) {
		cout << "polynomial_double_domain::find_all_roots" << endl;
	}

	polynomial_double *q;
	polynomial_double *r;

	q = create_object();
	r = create_object();
	copy(p, q);

	d = q->degree;
	for (i = 0; i < d; i++) {
		cout << "polynomial_double_domain::find_all_roots i=" << i
				<< " / " << d << ":" << endl;
		lambda[i] = q->root_finder(0 /*verbose_level*/);
		cout << "polynomial_double_domain::find_all_roots i=" << i
				<< " / " << d << ": lambda=" << lambda[i] << endl;
		rem = divide_linear_factor(q,
				r,
				lambda[i], verbose_level);
		cout << "quotient: ";
		r->print(cout);
		cout << endl;
		cout << "remainder=" << rem << endl;
		copy(r, q);
	}
	FREE_OBJECT(q);
	FREE_OBJECT(r);
	if (f_v) {
		cout << "polynomial_double_domain::find_all_roots done" << endl;
	}
}

double polynomial_double_domain::divide_linear_factor(
		polynomial_double *p,
		polynomial_double *q,
		double lambda, int verbose_level)
// divides p(X) by X-lambda, puts the result into q(X),
// returns the remainder
{
	int f_v = (verbose_level >= 1);
	int i, d;
	double a, b;

	if (f_v) {
		cout << "polynomial_double_domain::divide_linear_factor" << endl;
	}
	d = p->degree;

	a = p->coeff[d];
	q->degree = d - 1;
	q->coeff[d - 1] = a;
	for (i = 1; i <= d - 1; i++) {
		a = a * lambda + p->coeff[d - 1 - i + 1];
		q->coeff[d - 1 - i] = a;
	}
	b = q->coeff[0] * lambda + p->coeff[0];
	if (f_v) {
		cout << "polynomial_double_domain::divide_linear_factor done" << endl;
	}
	return b;
}



}}

