/*
 * globals.cpp
 *
 *  Created on: Aug 15, 2021
 *      Author: betten
 */

#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {


void inc_transpose(int *R,
	int *theX, int f_full, int max_r,
	int v, int b,
	int **theY, int *theYdim_n, int **R_new)
/* theX vom format v x inc->max_r
 * bzw. v x MAX_R falls f_full;
 * theY wird allociert auf b x theYdim_n.
 * R[] bestimmt die Ausmasse von theX.
 * R_new wird allociert b.
 * theYdim_n wird auf die max. vorkommende
 * Anzahl Kreuze pro Spalte gesetzt. */
{
	int dim_n;
	int *theY1, *R1, i, r, j;

	*theY = NULL;
	*R_new = NULL;
	if (f_full)
		dim_n = MAX_R;
	else
		dim_n = max_r;
	R1 = new int [b];

	for (i = 0; i < b; i++) {
		R1[i] = 0;
	}
	*theYdim_n = 0;
	for (i = 0; i < v; i++) {
		if (R[i] >= MAX_R) {
			cout << "inc_transpose R[i] >= MAX_R" << endl;
			exit(1);
		}
		for (r = 0; r < R[i]; r++) {
			j = theX[i * dim_n + r];
			R1[j]++;
			if (R1[j] >= MAX_R) {
				cout << "inc_transpose R1[j] >= MAX_R" << endl;
				exit(1);
			}
			*theYdim_n = MAX(*theYdim_n, R1[j]);
		}
	}
	for (i = 0; i < b; i++) {
		R1[i] = 0;
	}

	theY1 = new int [b * *theYdim_n];

	for (i = 0; i < v; i++) {
		for (r = 0; r < R[i]; r++) {
			j = theX[i * dim_n + r];
			theY1[j * *theYdim_n + R1[j] ] = i;
			R1[j]++;
		}
	}

	*theY = theY1;
	*R_new = R1;
}



int tuple_cmp(int *a, int *b, int l)
{
	int i;

	for (i = 0; i < l; i++) {
		if (a[i] > b[i]) {
			return -1;
		}
		if (a[i] < b[i]) {
			return 1;
		}
	}
	return 0;
}


void print_theX(int *theX, int dim_n, int v, int b, int *R)
{
	int i, j, o;

	for (i = 0; i < v; i++) {
		for (j = 0; j < b; j++) {
			for (o = 0; o < R[i]; o++) {
				/* before: r */
				if (theX[i * dim_n + o] == j) {
					break;
				}
			}
			if (o < R[i])
				cout << "X";
			else
				cout << ".";
			}
		cout << endl;
		}
	cout << endl;
}

void print_theX_pq(
	int *theX, int dim_n, int v, int b, int *R, cperm *pv, cperm *qv)
{
	inc_encoding E;

	E.theX = theX;
	E.v = v;
	E.b = b;
	E.R = R;
	E.dim_n = dim_n;
	E.print_permuted(pv, qv);
	E.theX = NULL;
	E.R = NULL;

}


void cperm_test(void)
{
	cperm p, q, r;
	char s[256];

	s[0] = 0;
	p.init_and_identity(5);

#if 0
	cp_mult_apply_backwc_r(&p, 1, 3);
	cp_mult_apply_backwc_l(&p, 2, 3);
	/* cp_mult_apply_tau_r(&p, 0, 1);
	cp_mult_apply_tau_r(&p, 1, 2); */
	cp_inv(&p, &q);
	cp_mult(&p, &q, &r);
	strcat(s, "p = ");
	cp_sprint(&p, s);
	strcat(s, "; q = ");
	cp_sprint(&q, s);
	strcat(s, "; r = ");
	cp_sprint(&r, s);
	printf("\n%s\n", s);
#endif
}

void frame2grid(geo_frame *frame, grid *G)
/* kopiert alles ausser type_idx[],
 * type[][], f_points, m, n. */
{
	int i, j, first, len;

	G->G_max = frame->G_max;
	for (i = 0; i < frame->G_max; i++) {
		first = frame->first[i];
		len = frame->len[i];
		G->first[i] = first;
		G->len[i] = len;
		for (j = 0; j < len; j++) {
			G->grid_entry[first + j] = i;
		}
	}
	G->first[i] = frame->first[i];
}

int tdos_cmp(tdo_scheme *t1, tdo_scheme *t2, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tdos_cmp" << endl;
		cout << "tdos_cmp t1=" << endl;
		t1->print();
		cout << "tdos_cmp t2=" << endl;
		t2->print();
	}
	int m, n, i, l;

	if (t1->m < t2->m) {
		return -1;
	}
	if (t1->m > t2->m) {
		return 1;
	}
	if (t1->n < t2->n) {
		return -1;
	}
	if (t1->n > t2->n) {
		return 1;
	}

	m = t1->m + 1;
	n = t1->n + 1;

	l = m * n;
	for (i = 0; i < l; i++) {
		if (t1->a[i] < t2->a[i]) {
			return -1;
		}
		if (t1->a[i] > t2->a[i]) {
			return +1;
		}
	}
	return 0;
}

int true_false_string_numeric(const char *p)
{
	int zweipot = 8, j;
	int flag_numeric;

	flag_numeric = 0;
	// Die Schleife wandelt den True-False-String in eine Zahl JS 120100 */
	for (j = 0; j < strlen(p); j++) {
		if (p[j] == 'T') {
			flag_numeric += zweipot;
		}
		zweipot /= 2;
	}
	return flag_numeric;
}

}}

