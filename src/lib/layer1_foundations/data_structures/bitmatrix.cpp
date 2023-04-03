/*
 * bitmatrix.cpp
 *
 *  Created on: Aug 15, 2019
 *      Author: betten
 */



#include "foundations.h"

using namespace std;

namespace orbiter {
namespace layer1_foundations {
namespace data_structures {


bitmatrix::bitmatrix()
{
	m = 0;
	n = 0;
	N = 0;
	data = NULL;
}

bitmatrix::~bitmatrix()
{
	if (data) {
		FREE_int((int *) data);
	}
}

void bitmatrix::init(int m, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int n1, sz, i;

	if (f_v) {
		cout << "bitmatrix::init" << endl;
	}
	bitmatrix::m = m;
	bitmatrix::n = n;
	N = n >> 5; // 4 bytes = 32 bits = 2^5
	n1 = N << 5;
	if (n > n1) {
		N++;
	}
	sz = m * N;
	data = (uint32_t *) NEW_int(sz);
	for (i = 0; i < m * N; i++) {
		data[i] = 0;
	}
	if (f_v) {
		cout << "bitmatrix::init done" << endl;
	}
}

void bitmatrix::unrank_PG_elements_in_columns_consecutively(
		field_theory::finite_field *F,
		long int start_value, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *v;
	int i, j, n100;

	if (f_v) {
		cout << "bitmatrix::unrank_PG_elements_in_columns_consecutively" << endl;
	}
	if (F->q != 2) {
		cout << "bitmatrix::unrank_PG_elements_in_columns_consecutively F->q != 2" << endl;
		exit(1);
	}
	zero_out();
	v = NEW_int(m);
	n100 = n / 100 + 1;
	for (j = 0; j < n; j++) {
		if ((j % n100) == 0) {
			cout << "bitmatrix::unrank_PG_elements_in_columns_consecutively " << j / n100 << " % done unranking" << endl;
		}
		F->Projective_space_basic->PG_element_unrank_modified_lint(
				v, 1, m, start_value + j);
		for (i = 0; i < m; i++) {
			if (v[i]) {
				m_ij(i, j, 1);
			}
		}
	}
	if (f_v) {
		cout << "bitmatrix::unrank_PG_elements_in_columns_consecutively done" << endl;
	}
}

void bitmatrix::rank_PG_elements_in_columns(
		field_theory::finite_field *F,
		int *perms, unsigned int *PG_ranks,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false;
	int *v;
	int i;
	int j;
	int n100;
	long int b;

	if (f_v) {
		cout << "bitmatrix::rank_PG_elements_in_columns" << endl;
	}
	if (F->q != 2) {
		cout << "bitmatrix::rank_PG_elements_in_columns F->q != 2" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "perm=";
		Int_vec_print(cout, perms, m);
		cout << endl;
	}
	v = NEW_int(m);
	n100 = n / 100 + 1;
	for (j = 0; j < n; j++) {
		if ((j % n100) == 0) {
			cout << "bitmatrix::rank_PG_elements_in_columns "
					<< j / n100 << " % done ranking" << endl;
		}
		for (i = 0; i < m; i++) {
			int a = perms[i];
			//cout << "i=" << i << " j=" << j << " s_ij(i, j)=" << s_ij(i, j) << endl;
			if (s_ij(i, j)) {
				v[a] = 1;
			}
			else {
				v[a] = 0;
			}
		}
		//cout << "j=" << j << " v=";
		//int_vec_print(cout, v, m);
		//cout << endl;
		F->Projective_space_basic->PG_element_rank_modified_lint(
				v, 1, m, b);
		PG_ranks[j] = (unsigned int) b;
	}
	FREE_int(v);
	if (f_v) {
		cout << "bitmatrix::rank_PG_elements_in_columns done" << endl;
	}
}

void bitmatrix::print()
{
	int i, j, x;

	for (i = 0; i < m; i++) {
		for (j = 0; j < MIN(n, 10); j++) {
			x = s_ij(i, j);
			if (x)
				cout << "1";
			else
				cout << "0";
			}
		cout << endl;
		}
	cout << endl;
}

void bitmatrix::zero_out()
{
	int i;

	for (i = 0; i < m * N; i++) {
		data[i] = 0;
	}
}

int bitmatrix::s_ij(int i, int j)
{
	int jj, bit;
	uint32_t mask;

	if (i < 0 || i >= m) {
		cout << "bitmatrix::s_ij addressing error, i = " << i << ", m = " << m << endl;
		exit(1);
		}
	if ( j < 0 || j >= n ) {
		cout << "bitmatrix::s_ij addressing error, j = " << j << ", n = " << n << endl;
		exit(1);
		}
	jj = j >> 5;
	bit = j & 31;
	mask = ((uint32_t) 1) << bit;
	uint32_t &x = data[i * N + jj];
	if (x & mask)
		return 1;
	else
		return 0;
}

void bitmatrix::m_ij(int i, int j, int a)
{
	int jj, bit;
	uint32_t mask;

	if (i < 0 || i >= m) {
		cout << "bitmatrix::m_ij addressing error, i = " << i << ", m = " << m << endl;
		exit(1);
		}
	if ( j < 0 || j >= n ) {
		cout << "bitmatrix::m_ij addressing error, j = " << j << ", n = " << n << endl;
		exit(1);
		}
	jj = j >> 5;
	bit = j & 31;
	mask = ((uint32_t) 1) << bit;
	uint32_t &x = data[i * N + jj];
	if (a == 0) {
		uint32_t not_mask = ~mask;
		x &= not_mask;
		}
	else {
		x |= mask;
		}
}

void bitmatrix::mult_int_matrix_from_the_left(
		int *A, int Am, int An,
		bitmatrix *Out, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false;
	int i, j, h;

	if (f_v) {
		cout << "bitmatrix::mult_int_matrix_from_the_left" << endl;
	}
	if (An != m) {
		cout << "bitmatrix::mult_int_matrix_from_the_left An != m" << endl;
		cout << "An=" << An << endl;
		cout << "m=" << m << endl;
		exit(1);
	}
	if (Out->m != Am) {
		cout << "bitmatrix::mult_int_matrix_from_the_left Out->m != Am" << endl;
		cout << "Am=" << Am << endl;
		cout << "Out->m=" << Out->m << endl;
		exit(1);
	}
	if (Out->n != n) {
		cout << "bitmatrix::mult_int_matrix_from_the_left Out->n != n" << endl;
		cout << "n=" << n << endl;
		cout << "Out->n=" << Out->n << endl;
		exit(1);
	}
	Out->zero_out();
	for (i = 0; i < Am; i++) {
		if (f_vv) {
			cout << "bitmatrix::mult_int_matrix_from_the_left row " << i << " : ";
		}
		for (j = 0; j < An; j++) {
			if (A[i * An + j]) {
				if (f_vv) {
					cout << j << ", ";
				}
				for (h = 0; h < N; h++) {
					Out->data[i * N + h] ^= data[j * N + h];
				}
			}
		}
		if (f_vv) {
			cout << endl;
		}
	}


	if (f_v) {
		cout << "bitmatrix::mult_int_matrix_from_the_left done" << endl;
	}
}



}}}

