/*
 * cperm.cpp
 *
 *  Created on: Aug 14, 2021
 *      Author: betten
 */

#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace geometry_builder {


cperm::cperm()
{
	Record_birth();
	l = 0;
	data = NULL;
}

cperm::~cperm()
{
	Record_death();

}

void cperm::init_and_identity(
		int l)
{
	//cp_free(p);
	data = new int[l];
	cperm::l = l;
	identity();
}

void cperm::free()
{
	if (data) {
		delete [] data;
		data = NULL;
	}
	l = 0;
}

void cperm::move_to(
		cperm *q)
{
	int i;
	
	if (l != q->l) {
		cout << "cp_mv p->l != q->l" << endl;
		exit(1);
	}
	for (i = 0; i < l; i++) {
		q->data[i] = data[i];
	}
}

void cperm::identity()
{
	int i;
	
	for (i = 0; i < l; i++) {
		data[i] = i;
	}
}

void cperm::mult(
		cperm *b, cperm *c)
/* erst a, dann b; Ergebnis nach c */
{
	int i, j;
	
	if (l != b->l) {
		cout << "cp_mult l != b->l" << endl;
		exit(1);
	}
	c->init_and_identity(l);
	for (i = 0; i < l; i++) {
		j = data[i];
		c->data[i] = b->data[j];
	}
}

void cperm::inverse(
		cperm *b)
/* b:= a^-1 */
{
	int i, j;
	
	b->init_and_identity(l);
	for (i = 0; i < l; i++) {
		j = data[i];
		b->data[j] = i;
	}
}

void cperm::power(
		cperm *res, int exp)
{
	cperm *b = NULL;
	cperm *c = NULL;
	cperm *d = NULL;
	int len;
	
	len = l;
	res->init_and_identity(len);
	b = new cperm;
	c = new cperm;
	d = new cperm;
	b->init_and_identity(len);
	c->init_and_identity(len);
	d->init_and_identity(len);
	
	move_to(b);
	while (exp > 0) {
		/* res = b^exp * c */
		if (ODD(exp)) {
			b->mult(c, d);
			d->move_to(c);
			exp--;
			continue; /* exp == 0 possible */
		}
		if (EVEN(exp)) {
			b->mult(b, d);
			d->move_to(b);
			exp >>= 1;
		}
	}
	c->move_to(res);

	if (b) {
		delete b;
	}
	if (c) {
		delete c;
	}
	if (d) {
		delete d;
	}
}

void cperm::print()
{
	int i;

	cout << "[";
	for (i = 0; i < l; i++) {
		cout << data[i];
		if (i < l - 1) {
			cout << ", ";
		}
	}
	cout << "]";
}

void cperm::mult_apply_forwc_r(
		int i, int l)
/* a := a (i i+1 ... i+l-1). */
{
	int m, j, k;
	int *t;

	t = new int[cperm::l];

	for (m = 0; m < cperm::l; m++) {
		j = data[m];
		if (j >= i && (k = j - i) < l) {
			t[k] = m;
		}
	}
	/* now: t[k] -> i+k -> i+k+1 for 0 < k < l-1
	 *      t[l-1] -> i+l-1 -> i */
	for (k = 0; k < l - 1; k++) {
		data[t[k]] = (i + k + 1);
	}
	data[t[l - 1]] = i;

	delete [] t;
}


void cperm::mult_apply_tau_r(
		int i, int j)
/* a := a (i j). */
{
	int k, i1, j1;
	
	for (k = 0; k < l; k++) {
		if (data[k] == i) {
			i1 = k;
		}
		if (data[k] == j) {
			j1 = k;
		}
	}
	/* now: i1 a == i, j1 a == j */
	data[i1] = j;
	data[j1] = i;
}

void cperm::mult_apply_tau_l(
		int i, int j)
/* a := (i j) a. */
{
	int i1, j1;
	
	i1 = data[i];
	j1 = data[j];
	/* now: i -> j -> j1; j -> i -> i1 */
	data[i] = j1;
	data[j] = i1;
}

void cperm::mult_apply_backwc_l(
		int i, int l)
/* a := (i+l-1 i+l-2 ... i+1 i) a. */
{
	int t, m;

	/* i+m -> i+m-1 -> a[i+m-1]  for 1 < m < l
	 * i -> i+l-1 -> a[i+l-1] */
	t = data[i + l - 1];
	for (m = l - 1; m > 0; m--) {
		data[i + m] = data[i + m - 1];
	}
	data[i] = t;
}





}}}}


