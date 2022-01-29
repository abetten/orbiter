/*
 * girth_test.cpp
 *
 *  Created on: Nov 18, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace geometry_builder {



girth_test::girth_test()
{
	gg = NULL;
	girth = 0;
	V = 0;
	S = NULL;
	D = NULL;
}

girth_test::~girth_test()
{
	int i;

	if (S) {
		for (i = 0; i < V; i++) {
			FREE_int(S[i]);
			FREE_int(D[i]);
		}
		FREE_pint(S);
		FREE_pint(D);
	}
}

void girth_test::init(gen_geo *gg, int girth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "girth_test::init" << endl;
	}

	girth_test::gg = gg;
	girth_test::girth = girth;
	V = gg->GB->V;

	S = NEW_pint(V);
	D = NEW_pint(V);
	for (i = 0; i < V; i++) {
		S[i] = NEW_int(V * V);
		D[i] = NEW_int(V * V);
	}


	if (f_v) {
		cout << "girth_test::init done" << endl;
	}
}

void girth_test::Floyd(int row, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, a;

	if (f_v) {
		cout << "girth_test::Floyd" << endl;
	}
	if (row == 0) {
		Int_vec_zero(D[0], V * V);
	}
	else {
		Int_vec_copy(D[row - 1], D[row], V * V);
	}

	// set all connected positions to 1
	// and all unconnected positions to infinity:

	for (i = 0; i < V; i++) {
		for (j = 0; j < V; j++) {
			if (i == j) {
				S[row][i * V + j] = 0;
			}
			if (D[row][i * V + j] == 0) {
				S[row][i * V + j] = 9999;    // Does this look like infinity?
			}
			else {
				S[row][i * V + j] = 1;
			}
		}
	}

	// for each route via k from i to j pick any better routes and
	// replace i-j path with sum of paths i-k and j-k

	for (k = 0; k < V; k++) {
		for (i = 0; i < V; i++) {
			for (j = 0; j < V; j++) {

				a = S[row][i * V + k] + S[row][k * V + j];

				if (a < S[row][i * V + j] ) {
						S[row][i * V + j] = a;
					}
				}
			}
		}


	if (f_v) {
		cout << "girth_test::Floyd done" << endl;
	}

}

void girth_test::add_incidence(int i, int j_idx, int j)
{
	int h, a;

	for (h = 0; h < gg->inc->K[j]; h++) {
		a = gg->inc->theY[j][h];
		//cout << "girth_test::add_incidence a=" << a << " i=" << i << endl;
		D[i][a * V + i] = 1;
		D[i][i * V + a] = 1;
	}
}

void girth_test::delete_incidence(int i, int j_idx, int j)
{
	int h, a;

	for (h = 0; h < gg->inc->K[j]; h++) {
		a = gg->inc->theY[j][h];
		D[i][a * V + i] = 0;
		D[i][i * V + a] = 0;
	}
}

int girth_test::check_girth_condition(int i, int j_idx, int j, int verbose_level)
{
	int h, dim_n, j1, u1, u2, a1, a2;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "girth_test::check_girth_condition i = " << i << ", j = " << j << endl;
	}
	dim_n = gg->inc->Encoding->dim_n;
	for (h = 0; h < j_idx; h++) {
		j1 = gg->inc->Encoding->theX_ir(i, h);
		for (u1 = 0; u1 < gg->inc->K[j1]; u1++) {
			a1 = gg->inc->theY[j1][u1];
			if (a1 == i) {
				continue;
			}
			for (u2 = 0; u2 < gg->inc->K[j]; u2++) {
				a2 = gg->inc->theY[j][u2];
				if (a2 == a1) {
					continue;
				}
				if (a1 == i) {
					continue;
				}
				if (S[i][a1 * V + a2] + 2 < girth) {
					if (f_v) {
						cout << "girth_test::check_girth_condition reject:" << endl;
						cout << "a1 = " << a1 << ", a2 = " << a2 << ", and nb_completed_rows = " << i << endl;
						cout << "path from a1 to a2 = " << S[i][a1 * V + a2] << ", and girth = " << girth << endl;
					}
					return FALSE;
				}
			}
		}
	}
	if (f_v) {
		cout << "girth_test::check_girth_condition OK" << endl;
	}
	return TRUE;
}

void girth_test::print_Si(int i)
{
	cout << "S[" << i << "]:" << endl;
	Int_matrix_print(S[i], V, V);
}

void girth_test::print_Di(int i)
{
	cout << "D[" << i << "]:" << endl;
	Int_matrix_print(D[i], V, V);
}

}}}


