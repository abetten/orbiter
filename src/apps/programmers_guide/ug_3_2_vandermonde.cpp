/*
 * ug_3_2_vandermonde.cpp
 *
 *  Created on: Jan 15, 2023
 *      Author: betten
 */

#include "orbiter.h"

using namespace std;
using namespace orbiter;



int main()
{

	orbiter::layer5_applications::user_interface::orbiter_top_level_session Orbiter;

	int q = 7;
	int verbose_level = 2;
	int f_without_tables = FALSE;
	field_theory::finite_field Fq;

	Fq.finite_field_init_small_order(q,
			f_without_tables,
			TRUE /* f_compute_related_fields */,
			verbose_level);

	int a;
	int i, j;
	int *V;
	int *W;

	V = NEW_int(q * q);
	W = NEW_int(q * q);

	for (i = 0; i < q; i++) {
		a = 1;
		V[i * q + 0] = 1;
		for (j = 1; j < q; j++) {
			a = Fq.mult(i, a);
			V[i * q + j] = a;
		}
	}

	cout << "Vandermonde matrix over F_" << q << endl;
	Int_matrix_print(V, q, q);

	Fq.Linear_algebra->invert_matrix(V, W, q, verbose_level);
	cout << endl;

	cout << "Inverse matrix:" << endl;
	Int_matrix_print(W, q, q);

	FREE_int(V);
	FREE_int(W);

}



