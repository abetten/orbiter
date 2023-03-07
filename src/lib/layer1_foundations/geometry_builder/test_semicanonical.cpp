/*
 * test_semicanonical.cpp
 *
 *  Created on: Dec 24, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace geometry_builder {



test_semicanonical::test_semicanonical()
{
	gg = NULL;
	MAX_V = 0;

	// initial vbar / hbar
	nb_i_vbar = 0;
	i_vbar = NULL;
	nb_i_hbar = 0;
	i_hbar = NULL;


	f_vbar = NULL;
	vbar = NULL;
	hbar = NULL;
}

test_semicanonical::~test_semicanonical()
{

	if (i_vbar) {
		FREE_int(i_vbar);
	}
	if (i_hbar) {
		FREE_int(i_hbar);
	}


	if (f_vbar) {
		FREE_int(f_vbar);
	}
	if (vbar) {
		FREE_int(vbar);
	}
	if (hbar) {
		FREE_int(hbar);
	}
}

void test_semicanonical::init(
		gen_geo *gg, int MAX_V, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "test_semicanonical::init" << endl;
	}

	test_semicanonical::gg = gg;
	test_semicanonical::MAX_V = MAX_V;


	f_vbar = NEW_int(gg->GB->V * gg->inc->Encoding->dim_n);

	for (i = 0; i < gg->GB->V * gg->inc->Encoding->dim_n; i++) {
		f_vbar[i] = FALSE;
	}

	hbar = NEW_int(gg->GB->V + 1);
	for (i = 0; i <= gg->GB->V; i++) {
		hbar[i] = INT_MAX;
	}

	vbar = NEW_int(gg->GB->B + 1);
	for (i = 0; i <= gg->GB->B; i++) {
		vbar[i] = INT_MAX;
	}

	if (f_v) {
		cout << "gen_geo::init_bars_and_partition before init_bars" << endl;
	}
	init_bars(verbose_level);
	if (f_v) {
		cout << "gen_geo::init_bars_and_partition after init_bars" << endl;
	}




	if (f_v) {
		cout << "girth_test::test_semicanonical done" << endl;
	}
}

void test_semicanonical::init_bars(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j;

	if (f_v) {
		cout << "test_semicanonical::init_bars" << endl;
	}
	if (f_v) {
		cout << "test_semicanonical::init_bars before i_hbar" << endl;
	}
	i_vbar = NEW_int(gg->GB->b_len + 1);
	i_hbar = NEW_int(gg->GB->v_len + 1);



	nb_i_vbar = 0;

	for (j = 0; j < gg->GB->b_len; j++) {
		if (f_v) {
			cout << "j=" << j << endl;
		}
		gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(0, j);

		vbar[C->j0] = -1;

		i_vbar[nb_i_vbar++] = C->j0;

	}


	nb_i_hbar = 0;
	i_hbar[nb_i_hbar++] = 0;


	if (f_v) {
		cout << "test_semicanonical::init_bars done" << endl;
	}

}

void test_semicanonical::print()
{
	int i;

	//cout << "V = " << Encoding->v << ", B = " << Encoding->b << endl;

	cout << "vbar: ";
	for (i = 0; i < nb_i_vbar; i++) {
		cout << i_vbar[i];
		if (i < nb_i_vbar - 1) {
			cout << ", ";
		}
	}
	cout << endl;

	cout << "hbar: ";
	for (i = 0; i < nb_i_hbar; i++) {
		cout << i_hbar[i];
		if (i < nb_i_hbar - 1) {
			cout << ", ";
		}
	}
	cout << endl;

}

void test_semicanonical::markers_update(
		int I, int m, int J, int n, int j,
		int i1, int j0, int r,
		int verbose_level)
{

	if (gg->GB->Descr->f_orderly) {
		return;
	}

	// manage vbar:

	if (vbar[j0 + j] == i1) {
		if (n == 0) {
			cout << "test_semicanonical::markers_update n == 0" << endl;
			exit(1);
		}
		if (gg->inc->Encoding->theX[i1 * gg->inc->Encoding->dim_n + r - 1] != j0 + j - 1) {

			// previous incidence:
			cout << "test_semicanonical::markers_update theX[i1 * inc.max_r + r - 1] != j0 + j - 1" << endl;
			exit(1);
		}
		if (!f_vbar[i1 * gg->inc->Encoding->dim_n + r - 1]) {
			cout << "test_semicanonical::markers_update !f_vbar[i1 * inc->Encoding->dim_n + r - 1]" << endl;
			exit(1);
		}
		f_vbar[i1 * gg->inc->Encoding->dim_n + r - 1] = FALSE;
		vbar[j0 + j] = MAX_V;
		// the value MAX_V indicates that there is no vbar
	}

	// create new vbar to the right:

	if (vbar[j0 + j + 1] == i1) {
		cout << "test_semicanonical::markers_update vbar[j0 + j + 1] == i1" << endl;
		exit(1);
	}
	if (vbar[j0 + j + 1] > i1) {
		f_vbar[i1 * gg->inc->Encoding->dim_n + r] = TRUE;
		vbar[j0 + j + 1] = i1;
	}


	// ToDo: row_marker_test_and_update:
	if (hbar[i1] > J) {
		if (m == 0) {
			cout << "test_semicanonical::markers_update no hbar && m == 0" << endl;
			exit(1);
		}
		if (j0 + j != gg->inc->Encoding->theX[(i1 - 1) * gg->inc->Encoding->dim_n + r]) {
			// create new hbar:
			hbar[i1] = J;
		}
	}

}

void test_semicanonical::marker_move_on(
		int I, int m, int J, int n, int j,
		int i1, int j0, int r,
		int verbose_level)
{

	if (gg->GB->Descr->f_orderly) {
		return;
	}

	// generate new vbar to the left of this incidence:
	if (vbar[j0 + j + 1] == i1) {
		cout << "test_semicanonical::marker_move_on vbar[j0 + j + 1] == i1" << endl;
		exit(1);
	}
	if (vbar[j0 + j + 1] > i1) {
		f_vbar[i1 * gg->inc->Encoding->dim_n + r] = TRUE;
		vbar[j0 + j + 1] = i1;
	}

	if (hbar[i1] > J) {
		if (m == 0) {
			cout << "test_semicanonical::marker_move_on no hbar && m == 0" << endl;
			exit(1);
		}
		if (j0 + j != gg->inc->Encoding->theX[(i1 - 1) * gg->inc->Encoding->dim_n + r]) {
			// generate new hbar:
			hbar[i1] = J;
		}
	}
}

int test_semicanonical::row_starter(
		int I, int m, int J, int n,
		int i1, int j0, int r,
		int verbose_level)
{
	int j;

	if (gg->GB->Descr->f_orderly) {
		if (n == 0) {
			// first incidence inside the block?
			j = 0;
		}
		else {

			// start out one to the right of the previous incidence:

			j = gg->inc->Encoding->theX[i1 * gg->inc->Encoding->dim_n + r - 1] - j0 + 1;
		}
	}
	else {
		if (hbar[i1] <= J) {
			// hbar exists, which means that the left part of the row differs from the row above.
			// The next incidence must be tried starting from the leftmost position.
			// We cannot copy over from the previous row.
			if (n == 0) {
				// first incidence inside the block?
				j = 0;
			}
			else {

				// start out one to the right of the previous incidence:

				j = gg->inc->Encoding->theX[i1 * gg->inc->Encoding->dim_n + r - 1] - j0 + 1;
			}
		}
		else {
			if (m == 0) {
				cout << "gen_geo::GeoXFst hbar[i1] > J && m == 0" << endl;
				exit(1);
			}
			// no hbar means that the left parts agree.

			// pick the incidence according to the previous row:
			j = gg->inc->Encoding->theX[(i1 - 1) * gg->inc->Encoding->dim_n + r] - j0;
		}
	}
	return j;
}

void test_semicanonical::row_init(int I, int m, int J,
		int i1,
		int verbose_level)
{
	if (gg->GB->Descr->f_orderly) {
		return;
	}

	if (m == 0) {
		hbar[i1] = -1;
		// initial hbar
	}
	else {
		hbar[i1] = gg->GB->b_len;
		// no hbar
	}

}

int test_semicanonical::col_marker_test(
		int j0, int j, int i1)
{
	if (gg->GB->Descr->f_orderly) {
		return FALSE;
	}

	if (vbar[j0 + j] > i1) {
		// no vbar, skip
		if (FALSE) {
			//cout << "gen_geo::X_Fst I=" << I << " m=" << m << " J=" << J << " n=" << n << " j=" << j << " skipped because of vbar" << endl;
		}
		return TRUE;
	}
	return FALSE;

}

void test_semicanonical::col_marker_remove(
		int I, int m, int J, int n,
		int i1, int j0, int r, int old_x)
{
	if (gg->GB->Descr->f_orderly) {
		return;
	}


	// remove vbar:
	if (f_vbar[i1 * gg->inc->Encoding->dim_n + r]) {
		vbar[old_x + 1] = MAX_V;
		f_vbar[i1 * gg->inc->Encoding->dim_n + r] = FALSE;
	}

	// possibly create new vbar on the left:
	if (n > 0) {
		if (vbar[old_x] > i1 &&
			gg->inc->Encoding->theX[i1 * gg->inc->Encoding->dim_n + r - 1] == old_x - 1) {
			vbar[old_x] = i1;
			f_vbar[i1 * gg->inc->Encoding->dim_n + r - 1] = TRUE;
		}
	}

}

void test_semicanonical::row_test_continue(
		int I, int m, int J, int i1)
{
	if (gg->GB->Descr->f_orderly) {
		return;
	}


	if (hbar[i1] > J) {
		hbar[i1] = J;
	}
}


}}}


