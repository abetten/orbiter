// INT_matrix.C
//
// Anton Betten
//
// Oct 23, 2013




#include "galois.h"

INT_matrix::INT_matrix()
{
	null();
}

INT_matrix::~INT_matrix()
{
	freeself();
}

void INT_matrix::null()
{
	M = NULL;
	m = 0;
	n = 0;
}

void INT_matrix::freeself()
{
	if (M) {
		FREE_INT(M);
		}
	null();
}

void INT_matrix::allocate(INT m, INT n)
{
	freeself();
	M = NEW_INT(m * n);
	INT_matrix::m = m;
	INT_matrix::n = n;
}

void INT_matrix::allocate_and_init(INT m, INT n, INT *Mtx)
{
	freeself();
	M = NEW_INT(m * n);
	INT_matrix::m = m;
	INT_matrix::n = n;
	INT_vec_copy(Mtx, M, m * n);
}

INT &INT_matrix::s_ij(INT i, INT j)
{
	return M[i * n + j];
}

INT &INT_matrix::s_m()
{
	return m;
}

INT &INT_matrix::s_n()
{
	return n;
}

void INT_matrix::print()
{
	INT_matrix_print(M, m, n);
}



