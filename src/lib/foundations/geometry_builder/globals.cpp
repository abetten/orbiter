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

