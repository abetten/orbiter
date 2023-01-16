/*
 * inc_encoding.cpp
 *
 *  Created on: Aug 17, 2021
 *      Author: betten
 */



#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace geometry_builder {


inc_encoding::inc_encoding()
{
	theX = NULL;
	dim_n = 0;
	v = 0;
	b = 0;
	R = NULL;

}

inc_encoding::~inc_encoding()
{
	if (theX) {
		delete [] theX;
	}
}

int &inc_encoding::theX_ir(int i, int r)
{
	return theX[i * dim_n + r];
}

void inc_encoding::init(int v, int b, int *R, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "inc_encoding::init v=" << v << " b=" << b << endl;
		cout << "inc_encoding::init R=" << endl;
		for (i = 0; i < v; i++) {
			cout << i << " : " << R[i] << endl;
		}
	}
	inc_encoding::v = v;
	inc_encoding::b = b;
	inc_encoding::R = R;

	dim_n = 0;
	for (i = 0; i < v; i++) {
		dim_n = MAXIMUM(dim_n, R[i]);
	}
	if (f_v) {
		cout << "inc_encoding::init dim_n=" << dim_n << endl;
	}
	theX = new int[v * dim_n];
	if (f_v) {
		cout << "inc_encoding::init done" << endl;
	}
}

long int inc_encoding::rank_row(int row)
{
	int *S;
	int k, i;
	long int rk;
	combinatorics::combinatorics_domain Combi;

	S = NEW_int(dim_n);

	k = R[row];
	for (i = 0; i < k; i++) {
		S[i] = theX[row * dim_n + i];
	}

	rk = Combi.rank_k_subset(S, b, k);

	FREE_int(S);

	return rk;
}

void inc_encoding::get_flags(int row, std::vector<int> &flags)
{
	int i, h, r, a;


	for (i = 0; i < row; i++) {
		r = R[i];
		for (h = 0; h < r; h++) {
			a = i * b + theX[i * dim_n + h];
			flags.push_back(a);
		}
	}
}


int inc_encoding::find_square(int m, int n)
{
	int i, j, l, u, v;
	int f_found_u;

	if (m == 0) {
		return FALSE;
	}
	if (n == 0) {
		return FALSE;
	}

	v = theX[m * dim_n + n];
	for (l = 0; l < n; l++) {
		u = theX[m * dim_n + l];
		for (i = 0; i < m; i++) {
			// loop over all previous rows

			// search for u:

			f_found_u = FALSE;
			for (j = 0; j < R[i] - 1; j++) {

				// < R[i] - 1,
				// since one incidence is reserved for v

				if (theX[i * dim_n + j] == u) {
					f_found_u = TRUE;
					break;
				}
			}
			if (!f_found_u) {
				continue; /* next i */
			}

			// look for v
			for (j++; j < R[i]; j++) {
				if (theX[i * dim_n + j] == v) {
					return TRUE;
				}
			}
			// v is not found, square test is negative

		} // next i
	} // next l
	return FALSE;
}

void inc_encoding::print_horizontal_bar(
	std::ostream &ost,
	gen_geo *gg, int f_print_isot, iso_type *it)
{
	int J, j;

	//cout << "inc_encoding::print_horizontal_bar" << endl;
	J = 0;
	for (j = 0; j <= b; j++) {
		if ((j == b) ||
				(J < gg->Test_semicanonical->nb_i_vbar &&
						j == gg->Test_semicanonical->i_vbar[J])) {
			ost << "+";
			J++;
		}
		if (j == b) {
			break;
		}
		ost << "-";
	}
	if (f_print_isot && it) {
		it->print_status(ost, TRUE /* f_with_flags */);
	}
	ost << endl;

}

void inc_encoding::print_partitioned(
		std::ostream &ost, int v_cur, int v_cut,
		gen_geo *gg, int f_print_isot)
{
	int *the_X;

	the_X = theX;
	print_partitioned_override_theX(ost, v_cur, v_cut, gg, the_X, f_print_isot);
}


void inc_encoding::print_partitioned_override_theX(
		std::ostream &ost, int v_cur, int v_cut,
		gen_geo *gg, int *the_X, int f_print_isot)
{
	int i, j, r, J, f_kreuz;
	int f_h_bar;
	iso_type *it;


	ost << endl;
	//I = 0;

	print_horizontal_bar(ost, gg, FALSE /* f_print_isot */, NULL);

	for (i = 0; i <= v; i++) {

		//cout << "inc_encoding::print_partitioned_override_theX i=" << i << endl;


#if 0
		if ((i == v) ||
				(I < gg->Test_semicanonical->nb_i_hbar &&
						i == gg->Test_semicanonical->i_hbar[I])) {
			print_horizontal_bar(ost, gg, FALSE /* f_print_isot */, NULL);
			I++;
		}
#endif

		if (i == v) {
			break;
		}

		if (i == v_cur) {
			break;
		}

		// output a line of the geometry:
		J = 0;
		r = 0;
		for (j = 0; j <= b; j++) {
			if ((j == b) ||
					(J < gg->Test_semicanonical->nb_i_vbar &&
					j == gg->Test_semicanonical->i_vbar[J])) {
				ost << "|";
				J++;
			}
			if (j == b) {
				break;
			}

			if (i >= v_cut || r >= R[i]) {
				f_kreuz = FALSE;
			}
			else if (the_X[r] == j) {
				f_kreuz = TRUE;
			}
			else {
				f_kreuz = FALSE;
			}

			if (f_kreuz) {
				ost << "X";
				r++;
			}
			else {
				ost << ".";
			}

		}

		{
			combinatorics::combinatorics_domain Combi;
			int *S;
			int row, k, ii;
			long int rk;

			S = NEW_int(dim_n);

			row = i;
			k = R[row];
			for (ii = 0; ii < k; ii++) {
				S[ii] = theX[row * dim_n + ii];
			}

			rk = Combi.rank_k_subset(S, b, k);
			ost << setw(3) << i << " : ";
			ost << setw(4) << rk << " : ";

			FREE_int(S);
		}

		if (gg->GB->Descr->f_orderly) {
			iso_type *It;

			It = gg->Geometric_backtrack_search->Row_stabilizer_orbits[i];

			if (It) {
				ost << gg->Geometric_backtrack_search->Row_stabilizer_orbit_idx[i]
					<< " / " << It->Canonical_forms->B.size();
			}
		}
		else {
			if (gg->inc->iso_type_at_line[i] && f_print_isot) {

				it = gg->inc->iso_type_at_line[i];

				it->print_status(ost, TRUE /* f_with_flags */);
			}
		}
		ost << endl;

		the_X += dim_n;

		if (gg->GB->V_partition[i]) {
			f_h_bar = TRUE;
		}
		else {
			f_h_bar = FALSE;

		}

		if (f_h_bar) {
			print_horizontal_bar(ost, gg, FALSE /* f_print_isot */, NULL);
		}


		//print_horizontal_bar(ost, gg, TRUE /* f_print_isot */, gg->inc->iso_type_no_vhbars);

	} // next i

	if (i == v && gg->inc->iso_type_no_vhbars && f_print_isot) {
		print_horizontal_bar(ost, gg, TRUE /* f_print_isot */, gg->inc->iso_type_no_vhbars);

		ost << endl;
	}

	ost << endl;
}

void inc_encoding::print_permuted(cperm *pv, cperm *qv)
{
	int i, j, i1, j1, o;

	cout << "   ";
	for (j = 0; j < b; j++) {
		cout << setw(2) << qv->data[j] << " ";
	}
	cout << endl;
	for (i = 0; i < v; i++) {
		i1 = pv->data[i];
		cout << setw(2) << i1 << " ";
		for (j = 0; j < b; j++) {
			j1 = qv->data[j];
			for (o = 0; o < R[i1]; o++) {
				/* before: r */
				if (theX[i1 * dim_n + o] == j1) {
					break;
				}
			}
			if (o < R[i1]) {
				cout << " X ";
			}
			else {
				cout << " . ";
			}
		}
		cout << endl;
	}
	cout << endl;
}

#if 0
tactical_decomposition *inc_encoding::calc_tdo_without_vhbar(
	int f_second_tactical_decomposition, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "inc_encoding::calc_tdo_without_vhbar" << endl;
	}
	tactical_decomposition *tdo = NULL;

	tdo = new tactical_decomposition;

	tdo->init(this, verbose_level);

	tdo->make_point_and_block_partition(tdo->G_current, tdo->G_last);

	tdo->calc2(v, verbose_level);

	tdo->tdos = tdo->get_tdos(tdo->G_current, tdo->G_next, FALSE, verbose_level);

	if (f_second_tactical_decomposition) {

		tdo->second_order_tdo(v, verbose_level);

		delete tdo->tdos;
		tdo->tdos = tdo->tdos2;
		tdo->tdos2 = NULL;
	}


	if (f_v) {
		cout << "inc_encoding::calc_tdo_without_vhbar done" << endl;
	}
	return tdo;
}
#endif

void inc_encoding::apply_permutation(incidence *inc, int v,
	int *theY, cperm *p, cperm *q, int verbose_level)
/* p vertauscht nur innerhalb
 * der Bereiche gleicher R[] Laenge.
 * int theY[v * MAX_R]. */
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "inc_encoding::apply_permutation v=" << v << " b=" << b << endl;
		cout << "inc_encoding::apply_permutation p=";
		p->print();
		cout << endl;
		cout << "inc_encoding::apply_permutation q=";
		q->print();
		cout << endl;
	}
	int i, j, r, i1, j1;
	int *theZ;

	theZ = NEW_int(v * b);
	for (i = 0; i < v; i++) {
		for (j = 0; j < b; j++) {
			theZ[i * b + j] = 0;
		}
	}
	for (i = 0; i < v; i++) {
		for (r = 0; r < inc->Encoding->R[i]; r++) {
			j = theX[i * dim_n + r];
			i1 = p->data[i];
			j1 = q->data[j];
			if (i1 >= v) {
				cout << "inc_encoding::apply_permutation i1 >= v" << endl;
				exit(1);
			}
			if (j1 >= b) {
				cout << "inc_encoding::apply_permutation j1 >= b" << endl;
				exit(1);
			}
			//  (i, j) is mapped to (i1, j1)
			theZ[i1 * b + j1] = 1;
		}
	}
	if (f_v) {
		cout << "theZ:" << endl;
		for (i = 0; i < v; i++) {
			for (j = 0; j < b; j++) {
				cout << theZ[i * b + j];
			}
			cout << endl;
		}

	}
	for (i = 0; i < v; i++) {
		if (f_v) {
			cout << "inc_encoding::apply_permutation i=" << i << endl;
		}
		r = 0;
		for (j = 0; j < b; j++) {
			if (theZ[i * b + j]) {
				if (r == inc->Encoding->R[i]) {
					cout << "inc_theX_apply_pq r == inc->Encoding->R[i]" << endl;
					inc->print_R(v, p, q);
					exit(1);
				}
				theY[i * dim_n + r] = j;
				r++;
			}
		}
		if (r != inc->Encoding->R[i]) {
			cout << "inc_theX_apply_pq r != inc->R[i]" << endl;
			inc->print_R(v, p, q);
			exit(1);
		}
	}
	FREE_int(theZ);
}



}}}


