/*
 * schlaefli.cpp
 *
 *  Created on: Oct 13, 2020
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace algebraic_geometry {


schlaefli::schlaefli()
{
	Record_birth();
	Surf = NULL;

	Labels = NULL;

	Schlaefli_double_six = NULL;

	Schlaefli_tritangent_planes = NULL;

	Schlaefli_trihedral_pairs = NULL;

	adjacency_matrix_of_lines = NULL;


}

schlaefli::~schlaefli()
{
	Record_death();
	int f_v = false;

	if (Labels) {
		FREE_OBJECT(Labels);
	}


	if (Schlaefli_double_six) {
		FREE_OBJECT(Schlaefli_double_six);
	}

	if (Schlaefli_tritangent_planes) {
		FREE_OBJECT(Schlaefli_tritangent_planes);
	}

	if (Schlaefli_trihedral_pairs) {
		FREE_OBJECT(Schlaefli_trihedral_pairs);
	}


	if (f_v) {
		cout << "before FREE_int(Trihedral_pairs);" << endl;
	}


	if (adjacency_matrix_of_lines) {
		FREE_int(adjacency_matrix_of_lines);
	}

}

void schlaefli::init(
		surface_domain *Surf, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schlaefli::init" << endl;
	}

	schlaefli::Surf = Surf;


	Labels = NEW_OBJECT(schlaefli_labels);
	if (f_v) {
		cout << "schlaefli::init "
				"before Labels->init" << endl;
	}
	Labels->init(verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after Labels->init" << endl;
	}


	Schlaefli_double_six = NEW_OBJECT(schlaefli_double_six);

	if (f_v) {
		cout << "schlaefli::init "
				"before Schlaefli_double_six->init" << endl;
	}
	Schlaefli_double_six->init(this, verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after Schlaefli_double_six->init" << endl;
	}


	Schlaefli_tritangent_planes = NEW_OBJECT(schlaefli_tritangent_planes);

	if (f_v) {
		cout << "schlaefli::init "
				"before Schlaefli_tritangent_planes->init" << endl;
	}
	Schlaefli_tritangent_planes->init(this, verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after Schlaefli_tritangent_planes->init" << endl;
	}



	Schlaefli_trihedral_pairs = NEW_OBJECT(schlaefli_trihedral_pairs);

	if (f_v) {
		cout << "schlaefli::init "
				"before Schlaefli_trihedral_pairs->init" << endl;
	}
	Schlaefli_trihedral_pairs->init(this, verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after Schlaefli_trihedral_pairs->init" << endl;
	}




	if (f_v) {
		cout << "schlaefli::init "
				"before init_adjacency_matrix_of_lines" << endl;
	}
	init_adjacency_matrix_of_lines(verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after init_adjacency_matrix_of_lines" << endl;
	}



	if (f_v) {
		cout << "schlaefli::init done" << endl;
	}
}







int schlaefli::line_ai(
		int i)
{
	if (i >= 6) {
		cout << "schlaefli::line_ai i >= 6" << endl;
		exit(1);
		}
	return i;
}

int schlaefli::line_bi(
		int i)
{
	if (i >= 6) {
		cout << "schlaefli::line_bi i >= 6" << endl;
		exit(1);
		}
	return 6 + i;
}

int schlaefli::line_cij(
		int i, int j)
{
	int a;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (i > j) {
		return line_cij(j, i);
		}
	if (i == j) {
		cout << "schlaefli::line_cij i==j" << endl;
		exit(1);
		}
	if (i >= 6) {
		cout << "schlaefli::line_cij i >= 6" << endl;
		exit(1);
		}
	if (j >= 6) {
		cout << "schlaefli::line_cij j >= 6" << endl;
		exit(1);
		}
	a = Combi.ij2k(i, j, 6);
	return 12 + a;
}

int schlaefli::type_of_line(
		int line)
// 0 = a_i, 1 = b_i, 2 = c_ij
{
	if (line < 6) {
		return 0;
		}
	else if (line < 12) {
		return 1;
		}
	else if (line < 27) {
		return 2;
		}
	else {
		cout << "schlaefli::type_of_line error" << endl;
		exit(1);
		}
}

void schlaefli::index_of_line(
		int line, int &i, int &j)
// returns i for a_i, i for b_i and (i,j) for c_ij
{
	int a;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (line < 6) { // ai
		i = line;
		}
	else if (line < 12) { // bj
		i = line - 6;
		}
	else if (line < 27) { // c_ij
		a = line - 12;
		Combi.k2ij(a, i, j, 6);
		}
	else {
		cout << "schlaefli::index_of_line error" << endl;
		exit(1);
		}
}




void schlaefli::ijklm2n(
		int i, int j, int k, int l, int m, int &n)
{
	int v[6];
	int size_complement;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	v[0] = i;
	v[1] = j;
	v[2] = k;
	v[3] = l;
	v[4] = m;
	Combi.set_complement_safe(v, 5, v + 5, size_complement, 6);
	if (size_complement != 1) {
		cout << "schlaefli::ijklm2n size_complement != 1" << endl;
		exit(1);
	}
	n = v[5];
}

void schlaefli::ijkl2mn(
		int i, int j, int k, int l, int &m, int &n)
{
	int v[6];
	int size_complement;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	v[0] = i;
	v[1] = j;
	v[2] = k;
	v[3] = l;
	Combi.set_complement_safe(v, 4, v + 4, size_complement, 6);
	if (size_complement != 2) {
		cout << "schlaefli::ijkl2mn size_complement != 2" << endl;
		exit(1);
	}
	m = v[4];
	n = v[5];
}

void schlaefli::ijk2lmn(
		int i, int j, int k, int &l, int &m, int &n)
{
	int v[6];
	int size_complement;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	v[0] = i;
	v[1] = j;
	v[2] = k;
	cout << "schlaefli::ijk2lmn v=";
	Int_vec_print(cout, v, 3);
	cout << endl;
	Combi.set_complement_safe(v, 3, v + 3, size_complement, 6);
	if (size_complement != 3) {
		cout << "schlaefli::ijk2lmn size_complement != 3" << endl;
		cout << "size_complement=" << size_complement << endl;
		exit(1);
	}
	l = v[3];
	m = v[4];
	n = v[5];
}

void schlaefli::ij2klmn(
		int i, int j, int &k, int &l, int &m, int &n)
{
	int v[6];
	int size_complement;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	v[0] = i;
	v[1] = j;
	Combi.set_complement_safe(v, 2, v + 2, size_complement, 6);
	if (size_complement != 4) {
		cout << "schlaefli::ij2klmn size_complement != 4" << endl;
		exit(1);
	}
	k = v[2];
	l = v[3];
	m = v[4];
	n = v[5];
}

void schlaefli::get_half_double_six_associated_with_Clebsch_map(
	int line1, int line2, int transversal,
	int hds[6],
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t1, t2, t3;
	int i, j, k, l, m, n;
	int i1, j1;
	int null;

	if (f_v) {
		cout << "schlaefli::get_half_double_six_associated_with_Clebsch_map" << endl;
	}

	if (line1 > line2) {
		cout << "schlaefli::get_half_double_six_associated_"
				"with_Clebsch_map line1 > line2" << endl;
		exit(1);
	}
	t1 = type_of_line(line1);
	t2 = type_of_line(line2);
	t3 = type_of_line(transversal);

	if (f_v) {
		cout << "t1=" << t1 << " t2=" << t2 << " t3=" << t3 << endl;
	}
	if (t1 == 0 && t2 == 0) { // ai and aj:
		index_of_line(line1, i, null);
		index_of_line(line2, j, null);
		if (t3 == 1) { // bk
			index_of_line(transversal, k, null);
			//cout << "i=" << i << " j=" << j << " k=" << k <<< endl;
			ijk2lmn(i, j, k, l, m, n);
			// bl, bm, bn, cij, cik, cjk
			hds[0] = line_bi(l);
			hds[1] = line_bi(m);
			hds[2] = line_bi(n);
			hds[3] = line_cij(i, j);
			hds[4] = line_cij(i, k);
			hds[5] = line_cij(j, k);
		}
		else if (t3 == 2) { // cij
			index_of_line(transversal, i1, j1);
				// test whether {i1,j1} =  {i,j}
			if ((i == i1 && j == j1) || (i == j1 && j == i1)) {
				ij2klmn(i, j, k, l, m, n);
				// bi, bj, bk, bl, bm, bn
				hds[0] = line_bi(i);
				hds[1] = line_bi(j);
				hds[2] = line_bi(k);
				hds[3] = line_bi(l);
				hds[4] = line_bi(m);
				hds[5] = line_bi(n);
			}
			else {
				cout << "schlaefli::get_half_doble_six_associated_"
						"with_Clebsch_map not {i,j} = {i1,j1}" << endl;
				exit(1);
			}
		}
	}
	else if (t1 == 1 && t2 == 1) { // bi and bj:
		index_of_line(line1, i, null);
		index_of_line(line2, j, null);
		if (t3 == 0) { // ak
			index_of_line(transversal, k, null);
			ijk2lmn(i, j, k, l, m, n);
			// al, am, an, cij, cik, cjk
			hds[0] = line_ai(l);
			hds[1] = line_ai(m);
			hds[2] = line_ai(n);
			hds[3] = line_cij(i, j);
			hds[4] = line_cij(i, k);
			hds[5] = line_cij(j, k);
		}
		else if (t3 == 2) { // cij
			index_of_line(transversal, i1, j1);
			if ((i == i1 && j == j1) || (i == j1 && j == i1)) {
				ij2klmn(i, j, k, l, m, n);
				// ai, aj, ak, al, am, an
				hds[0] = line_ai(i);
				hds[1] = line_ai(j);
				hds[2] = line_ai(k);
				hds[3] = line_ai(l);
				hds[4] = line_ai(m);
				hds[5] = line_ai(n);
			}
			else {
				cout << "schlaefli::get_half_doble_six_associated_"
						"with_Clebsch_map not {i,j} = {i1,j1}" << endl;
				exit(1);
			}
		}
	}
	else if (t1 == 0 && t2 == 1) { // ai and bi:
		index_of_line(line1, i, null);
		index_of_line(line2, j, null);
		if (j != i) {
			cout << "schlaefli::get_half_double_six_associated_"
					"with_Clebsch_map j != i" << endl;
			exit(1);
		}
		if (t3 != 2) {
			cout << "schlaefli::get_half_double_six_associated_"
					"with_Clebsch_map t3 != 2" << endl;
			exit(1);
		}
		index_of_line(transversal, i1, j1);
		if (i1 == i) {
			j = j1;
		}
		else {
			j = i1;
		}
		ij2klmn(i, j, k, l, m, n);
		// cik, cil, cim, cin, aj, bj
		hds[0] = line_cij(i, k);
		hds[1] = line_cij(i, l);
		hds[2] = line_cij(i, m);
		hds[3] = line_cij(i, n);
		hds[4] = line_ai(j);
		hds[5] = line_bi(j);
	}
	else if (t1 == 1 && t2 == 2) { // bi and cjk:
		index_of_line(line1, i, null);
		index_of_line(line2, j, k);
		if (t3 == 2) { // cil
			index_of_line(transversal, i1, j1);
			if (i1 == i) {
				l = j1;
			}
			else if (j1 == i) {
				l = i1;
			}
			else {
				cout << "schlaefli::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
			}
			ijkl2mn(i, j, k, l, m, n);
			// cin, cim, aj, ak, al, cnm
			hds[0] = line_cij(i, n);
			hds[1] = line_cij(i, m);
			hds[2] = line_ai(j);
			hds[3] = line_ai(k);
			hds[4] = line_ai(l);
			hds[5] = line_cij(n, m);
		}
		else if (t3 == 0) { // aj
			index_of_line(transversal, j1, null);
			if (j1 == k) {
				// swap k and j
				int tmp = k;
				k = j;
				j = tmp;
			}
			if (j1 != j) {
				cout << "schlaefli::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
			}
			ijk2lmn(i, j, k, l, m, n);
			// ak, cil, cim, cin, bk, cij
			hds[0] = line_ai(k);
			hds[1] = line_cij(i, l);
			hds[2] = line_cij(i, m);
			hds[3] = line_cij(i, n);
			hds[4] = line_bi(k);
			hds[5] = line_cij(i, j);
		}
	}
	else if (t1 == 0 && t2 == 2) { // ai and cjk:
		index_of_line(line1, i, null);
		index_of_line(line2, j, k);
		if (t3 == 2) { // cil
			index_of_line(transversal, i1, j1);
			if (i1 == i) {
				l = j1;
			}
			else if (j1 == i) {
				l = i1;
			}
			else {
				cout << "schlaefli::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
			}
			ijkl2mn(i, j, k, l, m, n);
			// cin, cim, bj, bk, bl, cnm
			hds[0] = line_cij(i, n);
			hds[1] = line_cij(i, m);
			hds[2] = line_bi(j);
			hds[3] = line_bi(k);
			hds[4] = line_bi(l);
			hds[5] = line_cij(n, m);
		}
		else if (t3 == 1) { // bj
			index_of_line(transversal, j1, null);
			if (j1 == k) {
				// swap k and j
				int tmp = k;
				k = j;
				j = tmp;
			}
			if (j1 != j) {
				cout << "schlaefli::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
			}
			ijk2lmn(i, j, k, l, m, n);
			// bk, cil, cim, cin, ak, cij
			hds[0] = line_bi(k);
			hds[1] = line_cij(i, l);
			hds[2] = line_cij(i, m);
			hds[3] = line_cij(i, n);
			hds[4] = line_ai(k);
			hds[5] = line_cij(i, j);
		}
	}
	else if (t1 == 2 && t2 == 2) { // cij and cik:
		index_of_line(line1, i, j);
		index_of_line(line2, i1, j1);
		if (i == i1) {
			k = j1;
		}
		else if (i == j1) {
			k = i1;
		}
		else if (j == i1) {
			j = i;
			i = i1;
			k = j1;
		}
		else if (j == j1) {
			j = i;
			i = j1;
			k = i1;
		}
		else {
			cout << "schlaefli::get_half_double_six_associated_"
					"with_Clebsch_map error" << endl;
			exit(1);
		}
		if (t3 == 0) { // ai
			index_of_line(transversal, i1, null);
			if (i1 != i) {
				cout << "schlaefli::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
			}
			ijk2lmn(i, j, k, l, m, n);
			// bi, clm, cnm, cln, bj, bk
			hds[0] = line_bi(i);
			hds[1] = line_cij(l, m);
			hds[2] = line_cij(n, m);
			hds[3] = line_cij(l, n);
			hds[4] = line_bi(j);
			hds[5] = line_bi(k);
		}
		else if (t3 == 1) { // bi
			index_of_line(transversal, i1, null);
			if (i1 != i) {
				cout << "schlaefli::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
			}
			ijk2lmn(i, j, k, l, m, n);
			// ai, clm, cnm, cln, aj, ak
			hds[0] = line_ai(i);
			hds[1] = line_cij(l, m);
			hds[2] = line_cij(n, m);
			hds[3] = line_cij(l, n);
			hds[4] = line_ai(j);
			hds[5] = line_ai(k);
		}
		else if (t3 == 2) { // clm
			index_of_line(transversal, l, m);
			ijklm2n(i, j, k, l, m, n);
			// ai, bi, cmn, cln, ckn, cjn
			hds[0] = line_ai(i);
			hds[1] = line_bi(i);
			hds[2] = line_cij(m, n);
			hds[3] = line_cij(l, n);
			hds[4] = line_cij(k, n);
			hds[5] = line_cij(j, n);
		}
	}
	else {
		cout << "schlaefli::get_half_double_six_associated_"
				"with_Clebsch_map error" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "schlaefli::get_half_double_six_associated_with_Clebsch_map done" << endl;
	}
}

void schlaefli::prepare_clebsch_map(
		int ds, int ds_row,
	int &line1, int &line2, int &transversal,
	int verbose_level)
{
	int ij, i, j, k, l, m, n, size_complement;
	int set[6];
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (ds == 0) {
		if (ds_row == 0) {
			line1 = line_bi(0);
			line2 = line_bi(1);
			transversal = line_cij(0, 1);
			return;
		}
		else {
			line1 = line_ai(0);
			line2 = line_ai(1);
			transversal = line_cij(0, 1);
			return;
		}
	}
	ds--;
	if (ds < 15) {
		ij = ds;
		Combi.k2ij(ij, i, j, 6);

		if (ds_row == 0) {
			line1 = line_ai(j);
			line2 = line_bi(j);
			transversal = line_cij(i, j);
			return;
		}
		else {
			line1 = line_ai(i);
			line2 = line_bi(i);
			transversal = line_cij(i, j);
			return;
		}
	}
	ds -= 15;
	Combi.unrank_k_subset(ds, set, 6, 3);
	Combi.set_complement(set, 3 /* subset_size */, set + 3,
		size_complement, 6 /* universal_set_size */);
	i = set[0];
	j = set[1];
	k = set[2];
	l = set[3];
	m = set[4];
	n = set[5];
	if (ds_row == 0) {
		line1 = line_bi(l);
		line2 = line_bi(m);
		transversal = line_ai(n);
		return;
	}
	else {
		line1 = line_ai(i);
		line2 = line_ai(j);
		transversal = line_bi(k);
		return;
	}
}

void schlaefli::init_adjacency_matrix_of_lines(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, l;

	if (f_v) {
		cout << "schlaefli::init_adjacency_matrix_of_lines" << endl;
	}

	adjacency_matrix_of_lines = NEW_int(27 * 27);
	Int_vec_zero(adjacency_matrix_of_lines, 27 * 27);

	// the ai lines:
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 6; j++) {
			if (j == i) {
				continue;
			}
			set_adjacency_matrix_of_lines(line_ai(i), line_bi(j));
		}
		for (k = 0; k < 6; k++) {
			if (k == i) {
				continue;
			}
			set_adjacency_matrix_of_lines(line_ai(i), line_cij(i, k));
		}
	}


	// the bi lines:
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 6; j++) {
			if (j == i) {
				continue;
			}
			set_adjacency_matrix_of_lines(line_bi(i), line_ai(j));
		}
		for (k = 0; k < 6; k++) {
			if (k == i) {
				continue;
			}
			set_adjacency_matrix_of_lines(line_bi(i), line_cij(i, k));
		}
	}




	// the cij lines:
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 6; j++) {
			if (j == i) {
				continue;
			}
			for (k = 0; k < 6; k++) {
				if (k == i) {
					continue;
				}
				if (k == j) {
					continue;
				}
				for (l = 0; l < 6; l++) {
					if (l == i) {
						continue;
					}
					if (l == j) {
						continue;
					}
					if (k == l) {
						continue;
					}
					set_adjacency_matrix_of_lines(
							line_cij(i, j), line_cij(k, l));
				} // next l
			} // next k
		} // next j
	} // next i

	int r, c;

	for (i = 0; i < 27; i++) {
		r = 0;
		for (j = 0; j < 27; j++) {
			if (get_adjacency_matrix_of_lines(i, j)) {
				r++;
			}
		}
		if (r != 10) {
			cout << "schlaefli::init_adjacency_matrix_of_lines "
					"row sum r != 10, r = " << r << " in row " << i << endl;
		}
	}

	for (j = 0; j < 27; j++) {
		c = 0;
		for (i = 0; i < 27; i++) {
			if (get_adjacency_matrix_of_lines(i, j)) {
				c++;
			}
		}
		if (c != 10) {
			cout << "schlaefli::init_adjacency_matrix_of_lines "
					"col sum c != 10, c = " << c << " in col " << j << endl;
		}
	}

	if (f_v) {
		cout << "schlaefli::init_adjacency_matrix_of_lines done" << endl;
	}
}


void schlaefli::set_adjacency_matrix_of_lines(
		int i, int j)
{
	adjacency_matrix_of_lines[i * 27 + j] = 1;
	adjacency_matrix_of_lines[j * 27 + i] = 1;
}

int schlaefli::get_adjacency_matrix_of_lines(
		int i, int j)
{
	return adjacency_matrix_of_lines[i * 27 + j];
}


void schlaefli::print_Steiner_and_Eckardt(
		std::ostream &ost)
{
	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Eckardt Points}" << endl;
	Schlaefli_tritangent_planes->latex_table_of_Eckardt_points(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Double Sixes}" << endl;
	Schlaefli_double_six->latex_table_of_double_sixes(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Half Double Sixes}" << endl;
	Schlaefli_double_six->latex_table_of_half_double_sixes(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Tritangent Planes}" << endl;
	Schlaefli_tritangent_planes->latex_table_of_tritangent_planes(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Steiner Trihedral Pairs}" << endl;
	Schlaefli_trihedral_pairs->latex_table_of_trihedral_pairs(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Triads}" << endl;
	Schlaefli_trihedral_pairs->latex_triads(ost);

}


void schlaefli::latex_table_of_Schlaefli_labeling_of_lines(
		std::ostream &ost)
{
	int i;

	ost << "\\begin{multicols}{5}" << endl;
	ost << "\\noindent";
	for (i = 0; i < 27; i++) {
		ost << "$" << i << " = ";
		print_line(ost, i);
		ost << "$\\\\" << endl;
	}
	ost << "\\end{multicols}" << endl;
}





void schlaefli::print_line(
		std::ostream &ost, int rk)
{
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (rk < 6) {
		ost << "a_" << rk + 1 << endl;
	}
	else if (rk < 12) {
		ost << "b_" << rk - 6 + 1 << endl;
	}
	else {
		int i, j;

		rk -= 12;
		Combi.k2ij(rk, i, j, 6);
		ost << "c_{" << i + 1 << j + 1 << "}";
	}
}

void schlaefli::print_Schlaefli_labelling(
		std::ostream &ost)
{
	int j, h;

	ost << "The Schlaefli labeling of lines:\\\\" << endl;
	ost << "$$" << endl;
	for (j = 0; j < 3; j++) {
		ost << "\\begin{array}{|r|r|}" << endl;
		ost << "\\hline" << endl;
		ost << "h &  \\mbox{line} \\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\hline" << endl;
		for (h = 0; h < 9; h++) {
			ost << j * 9 + h << " & "
				<< Labels->Line_label_tex[j * 9 + h] << "\\\\" << endl;
		}
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		if (j < 3 - 1) {
			ost << "\\qquad" << endl;
		}
	}
	ost << "$$" << endl;
}

void schlaefli::print_set_of_lines_tex(
		std::ostream &ost, long int *v, int len)
{
	int i;

	ost << "\\{";
	for (i = 0; i < len; i++) {
		ost << Labels->Line_label_tex[v[i]];
		if (i < len - 1) {
			ost << ", ";
		}
	}
	ost << "\\}";
}

void schlaefli::latex_table_of_clebsch_maps(
		std::ostream &ost)
{
	int e, line, j, l1, l2, t1, t2, t3, t4, c1, c2, cnt;
	int three_lines[3];
	int transversal_line;
	//int intersecting_lines[10];

	cnt = 0;
	//cout << "schlaefli::latex_table_of_clebsch_maps" << endl;
	//ost << "\\begin{multicols}{2}" << endl;
	for (e = 0; e < Schlaefli_tritangent_planes->nb_Eckardt_points; e++) {

		Schlaefli_tritangent_planes->Eckardt_points[e].three_lines(Surf, three_lines);

		for (line = 0; line < 3; line++) {

			transversal_line = three_lines[line];
			if (line == 0) {
				c1 = three_lines[1];
				c2 = three_lines[2];
			}
			else if (line == 1) {
				c1 = three_lines[0];
				c2 = three_lines[2];
			}
			else if (line == 2) {
				c1 = three_lines[0];
				c2 = three_lines[1];
			}

			for (l1 = 0; l1 < 27; l1++) {
				if (l1 == c1 || l1 == c2) {
					continue;
				}
				if (get_adjacency_matrix_of_lines(
						transversal_line, l1) == 0) {
					continue;
				}
				for (l2 = l1 + 1; l2 < 27; l2++) {
					if (l2 == c1 || l2 == c2) {
						continue;
					}
					if (get_adjacency_matrix_of_lines(
							transversal_line, l2) == 0) {
						continue;
					}



					cout << "e=" << e << endl;
					cout << "transversal_line=" << transversal_line << endl;
					cout << "c1=" << c1 << endl;
					cout << "c2=" << c2 << endl;
					cout << "l1=" << l1 << endl;
					cout << "l2=" << l2 << endl;

					for (t1 = 0; t1 < 27; t1++) {
						if (t1 == three_lines[0] ||
								t1 == three_lines[1] ||
								t1 == three_lines[2]) {
							continue;
						}
						if (t1 == l1 || t1 == l2) {
							continue;
						}
						if (get_adjacency_matrix_of_lines(l1, t1) == 0 ||
								get_adjacency_matrix_of_lines(l2, t1) == 0) {
							continue;
						}
						cout << "t1=" << t1 << endl;

						for (t2 = t1 + 1; t2 < 27; t2++) {
							if (t2 == three_lines[0] ||
									t2 == three_lines[1] ||
									t2 == three_lines[2]) {
								continue;
							}
							if (t2 == l1 || t2 == l2) {
								continue;
							}
							if (get_adjacency_matrix_of_lines(l1, t2) == 0 ||
									get_adjacency_matrix_of_lines(l2, t2) == 0) {
								continue;
							}
							cout << "t2=" << t2 << endl;

							for (t3 = t2 + 1; t3 < 27; t3++) {
								if (t3 == three_lines[0] ||
										t3 == three_lines[1] ||
										t3 == three_lines[2]) {
									continue;
								}
								if (t3 == l1 || t3 == l2) {
									continue;
								}
								if (get_adjacency_matrix_of_lines(l1, t3) == 0 ||
										get_adjacency_matrix_of_lines(l2, t3) == 0) {
									continue;
								}
								cout << "t3=" << t3 << endl;

								for (t4 = t3 + 1; t4 < 27; t4++) {
									if (t4 == three_lines[0] ||
											t4 == three_lines[1] ||
											t4 == three_lines[2]) {
										continue;
									}
									if (t4 == l1 || t4 == l2) {
										continue;
									}
									if (get_adjacency_matrix_of_lines(l1, t4) == 0 ||
											get_adjacency_matrix_of_lines(l2, t4) == 0) {
										continue;
									}
									cout << "t4=" << t4 << endl;


									int tc1[4], tc2[4];
									int n1 = 0, n2 = 0;

									if (get_adjacency_matrix_of_lines(t1, c1)) {
										tc1[n1++] = t1;
									}
									if (get_adjacency_matrix_of_lines(t1, c2)) {
										tc2[n2++] = t1;
									}
									if (get_adjacency_matrix_of_lines(t2, c1)) {
										tc1[n1++] = t2;
									}
									if (get_adjacency_matrix_of_lines(t2, c2)) {
										tc2[n2++] = t2;
									}
									if (get_adjacency_matrix_of_lines(t3, c1)) {
										tc1[n1++] = t3;
									}
									if (get_adjacency_matrix_of_lines(t3, c2)) {
										tc2[n2++] = t3;
									}
									if (get_adjacency_matrix_of_lines(t4, c1)) {
										tc1[n1++] = t4;
									}
									if (get_adjacency_matrix_of_lines(t4, c2)) {
										tc2[n2++] = t4;
									}
									cout << "n1=" << n1 << endl;
									cout << "n2=" << n2 << endl;

									ost << cnt << " : $\\pi_{" << e << "} = \\pi_{";
									Schlaefli_tritangent_planes->Eckardt_points[e].latex_index_only(ost);
									ost << "}$, $\\;$ ";

#if 0
									ost << " = ";
									for (j = 0; j < 3; j++) {
										ost << Line_label_tex[three_lines[j]];
										}
									ost << "$, $\\;$ " << endl;
#endif

									ost << "$" << Labels->Line_label_tex[transversal_line] << "$, $\\;$ ";
									//ost << "$(" << Line_label_tex[c1] << ", " << Line_label_tex[c2];
									//ost << ")$, $\\;$ ";

									ost << "$(" << Labels->Line_label_tex[l1] << "," << Labels->Line_label_tex[l2] << ")$, $\\;$ ";
#if 0
									ost << "$(" << Line_label_tex[t1]
										<< "," << Line_label_tex[t2]
										<< "," << Line_label_tex[t3]
										<< "," << Line_label_tex[t4]
										<< ")$, $\\;$ ";
#endif
									ost << "$"
											<< Labels->Line_label_tex[c1] << " \\cap \\{";
									for (j = 0; j < n1; j++) {
										ost << Labels->Line_label_tex[tc1[j]];
										if (j < n1 - 1) {
											ost << ", ";
										}
									}
									ost << "\\}$ ";
									ost << "$"
											<< Labels->Line_label_tex[c2] << " \\cap \\{";
									for (j = 0; j < n2; j++) {
										ost << Labels->Line_label_tex[tc2[j]];
										if (j < n2 - 1) {
											ost << ", ";
										}
									}
									ost << "\\}$ ";
									ost << "\\\\" << endl;
									cnt++;

								} // next t4
							} // next t3
						} // next t2
					} // next t1
					//ost << "\\hline" << endl;
				} // next l2
			} // next l1

		} // line
		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;
	} // e
	//ost << "\\end{multicols}" << endl;
	//cout << "schlaefli::latex_table_of_clebsch_maps done" << endl;
}


int schlaefli::identify_Eckardt_point(
		int line1, int line2, int line3, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int lines[3];
	other::data_structures::sorting Sorting;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	int idx;

	if (f_v) {
		cout << "schlaefli::identify_Eckardt_point" << endl;
	}
	lines[0] = line1;
	lines[1] = line2;
	lines[2] = line3;
	Sorting.int_vec_heapsort(lines, 3);
	line1 = lines[0];
	line2 = lines[1];
	line3 = lines[2];
	if (line1 < 6) {
		if (line2 < 6) {
			cout << "schlaefli::identify_Eckardt_point "
					"line1 < 6 and line2 < 6" << endl;
			exit(1);
		}
		idx = Combi.ordered_pair_rank(line1, line2 - 6, 6);
	}
	else {
		int i, j, k, l, m, n;

		if (line1 < 12) {
			cout << "schlaefli::identify_Eckardt_point "
					"second case, line1 < 12" << endl;
			exit(1);
		}
		if (line2 < 12) {
			cout << "schlaefli::identify_Eckardt_point "
					"second case, line2 < 12" << endl;
			exit(1);
		}
		if (line3 < 12) {
			cout << "schlaefli::identify_Eckardt_point "
					"second case, line3 < 12" << endl;
			exit(1);
		}
		Combi.k2ij(line1 - 12, i, j, 6);
		Combi.k2ij(line2 - 12, k, l, 6);
		Combi.k2ij(line3 - 12, m, n, 6);
		idx = 30 + Combi.unordered_triple_pair_rank(i, j, k, l, m, n);
	}
	if (f_v) {
		cout << "schlaefli::identify_Eckardt_point done" << endl;
	}
	return idx;
}

void schlaefli::write_lines_vs_line(
		std::string &prefix, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schlaefli::write_lines_vs_line" << endl;
	}

	other::orbiter_kernel_system::file_io Fio;
	string fname;

	fname = prefix + "_lines_vs_lines_incma.csv";

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, adjacency_matrix_of_lines,
			27, 27);


	if (f_v) {
		cout << "schlaefli::write_lines_vs_line done" << endl;
	}

}






}}}}

