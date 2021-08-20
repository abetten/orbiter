/*
 * cperm.cpp
 *
 *  Created on: Aug 14, 2021
 *      Author: betten
 */

#include "geo.h"

using namespace std;

cperm::cperm()
{
	l = 0;
	data = NULL;
}

cperm::~cperm()
{

}

void cperm::init_and_identity(int l)
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
		data = NIL;
		}
	l = 0;
}

void cperm::move_to(cperm *q)
{
	int i;
	
	if (l != q->l) {
		cout << "cp_mv p->l != q->l" << endl;
		exit(1);
		}
	for (i = 0; i < l; i++)
		q->data[i] = data[i];
}

void cperm::identity()
{
	int i;
	
	for (i = 0; i < l; i++) {
		data[i] = i;
	}
}

void cperm::mult(cperm *b, cperm *c)
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

void cperm::inverse(cperm *b)
/* b:= a^-1 */
{
	int i, j;
	
	b->init_and_identity(l);
	for (i = 0; i < l; i++) {
		j = data[i];
		b->data[j] = i;
	}
}

void cperm::power(cperm *res, int exp)
{
	cperm *b = NIL;
	cperm *c = NIL;
	cperm *d = NIL;
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

l_exit:
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
#if 0
	char s[256];

	s[0] = 0;
	cp_sprint(p, s);
	printf("%s\n", s);
#else
	int i;

	cout << "[";
	for (i = 0; i < l; i++) {
		cout << data[i];
		if (i < l - 1) {
			cout << ", ";
		}
	}
	cout << "]";
#endif
}

void cperm::mult_apply_forwc_r(int i, int l)
/* a := a (i i+1 ... i+l-1). */
{
	int m, j, k;
	int *t;

	t = new int[cperm::l];

	for (m = 0; m < cperm::l; m++) {
		j = data[m];
		if (j >= i && (k = j - i) < l)
			t[k] = m;
		}
	/* now: t[k] -> i+k -> i+k+1 for 0 < k < l-1
	 *      t[l-1] -> i+l-1 -> i */
	for (k = 0; k < l - 1; k++)
		data[t[k]] = (i + k + 1);
	data[t[l - 1]] = i;

	delete [] t;
}


void cperm::mult_apply_tau_r(int i, int j)
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

void cperm::mult_apply_tau_l(int i, int j)
/* a := (i j) a. */
{
	int i1, j1;
	
	i1 = data[i];
	j1 = data[j];
	/* now: i -> j -> j1; j -> i -> i1 */
	data[i] = j1;
	data[j] = i1;
}

void cperm::mult_apply_backwc_l(int i, int l)
/* a := (i+l-1 i+l-2 ... i+1 i) a. */
{
	int t, m;

	/* i+m -> i+m-1 -> a[i+m-1]  for 1 < m < l
	 * i -> i+l-1 -> a[i+l-1] */
	t = data[i + l - 1];
	for (m = l - 1; m > 0; m--)
		data[i + m] = data[i + m - 1];
	data[i] = t;
}



#if 0


int cp_mult_apply_backwc_r(
	CPERM *a, int i, int l)
/* a := a (i+l-1 i+l-2 ... i+1 i). */
{
	int t[256], m, j, k;
	
	if (l > 256) {
		Srfs("cp_mult_apply_backwc_r", "l > 256");
		return ERROR;
		}
	for (m = 0; m < a->l; m++) {
		j = a->a[m];
		if (j >= i && (k = j - i) < l)
			t[k] = m;
		}
	/* now: t[k] -> i+k -> i+k-1 for 1 < k < l
	 *      t[0] -> i -> i+l-1 */
	for (k = 1; k < l; k++)
		a->a[t[k]] = (i + k - 1);
	a->a[t[0]] = (i + l - 1);
	return OK;
}

int cp_mult_apply_forwc_l(
	CPERM *a, int i, int l)
/* a := (i i+1 ... i+l-1) a. */
{
	int t, m;
	
	/* i+m -> i+m+1 -> a[i+m+1]  for m < l - 1 
	 * i+l-1 -> i -> a[i] */
	t = a->a[i];
	for (m = 0; m < l - 1; m++)
		a->a[i + m] = a->a[i + m + 1];
	a->a[i+l-1] = t;
	return OK;
}

int cp_onep(CPERM *p)
{
	int i;
	
	for (i = 0; i < p->l; i++)
		if (p->a[i] != i)
			return FALSE;
	return TRUE;
}

int cp_cmp(CPERM *a, CPERM *b)
{
	int i;
	
	if (a->l != b->l) {
		Srfs("cp_cmp", "a->l != b->l");
		return 0;
		}
	for (i = 0; i < a->l; i++) {
		if (a->a[i] < b->a[i])
			return -1;
		if (a->a[i] > b->a[i])
			return 1;
		}
	return 0;
}


int cp_sprint(CPERM *p, char *s)
/* haengt an s an. */
{
	int *have_seen = NIL;
	int l, l1, first, next, len;
	int f_nothing_printed_at_all = TRUE;
	char str1[256];
	char str2[256];
	
	if (p == NIL || s == NIL) {
		Srfs("cp_sprint", "args NIL");
		return ERROR;
		}
	str1[0] = 0;
	have_seen = (int *) my_malloc(p->l * sizeof(int), "cp_sprint");
	if (have_seen == NIL) {
		Srfs("cp_sprint", "no memory");
		return ERROR;
		}
	for (l = 0; l < p->l; l++) {
		have_seen[l] = FALSE;
		}
	l = 0;
	while (TRUE) {
		if (l >= p->l) {
			if (f_nothing_printed_at_all) {
				strcat(s, "id");
				}
			else {
				strcat(s, str1);
				}
			if (have_seen)
				my_free(have_seen, "cp_sprint");
			return OK;
			}
		if (have_seen[l]) {
			l++;
			continue;
			}
		/* cycle starting at l: */
		first = l;
		l1 = l;
		len = 1;
		while (TRUE) {
			have_seen[l1] = TRUE;
			next = p->a[l1];
			if (next == first) {
				break;
				}
			l1 = next;
			len ++;
			}
		if (len == 1) {
			l++;
			continue;
			}
		f_nothing_printed_at_all = FALSE;
		/* print cycle, starting at first: */
		l1 = first;
		strcat(str1, "(");
		while (TRUE) {
			sprintf(str2, "%d", l1);
			strcat(str1, str2);
			next = p->a[l1];
			if (next == first) {
				break;
				}
			strcat(str1, " ");
			l1 = next;
			}
		strcat(str1, ")");
		}
}

int cp_latex(CPERM *p, FILE *fp)
/* haengt an s an. */
{
	int *have_seen = NIL;
	int l, l1, first, next, len;
	int f_nothing_printed_at_all = TRUE;
	char str1[256];
	char str2[256];
	
	str1[0] = 0;
	have_seen = (int *) my_malloc(p->l * sizeof(int), "cp_latex");
	if (have_seen == NIL) {
		Srfs("cp_latex", "no memory");
		return ERROR;
		}
	for (l = 0; l < p->l; l++) {
		have_seen[l] = FALSE;
		}
	l = 0;
	while (TRUE) {
		if (l >= p->l) {
			if (f_nothing_printed_at_all) {
				fprintf(fp, "id");
				}
			else {
				fprintf(fp, "%s", str1);
				}
			if (have_seen)
				my_free(have_seen, "cp_latex");
			return OK;
			}
		if (have_seen[l]) {
			l++;
			continue;
			}
		/* cycle starting at l: */
		first = l;
		l1 = l;
		len = 1;
		while (TRUE) {
			have_seen[l1] = TRUE;
			next = p->a[l1];
			if (next == first) {
				break;
				}
			l1 = next;
			len ++;
			}
		if (len == 1) {
			l++;
			continue;
			}
		f_nothing_printed_at_all = FALSE;
		/* print cycle, starting at first: */
		l1 = first;
		strcat(str1, "(");
		while (TRUE) {
			sprintf(str2, "%d", l1);
			strcat(str1, str2);
			next = p->a[l1];
			if (next == first) {
				break;
				}
			strcat(str1, " \\, ");
			l1 = next;
			}
		strcat(str1, ")");
		}
}
#endif



