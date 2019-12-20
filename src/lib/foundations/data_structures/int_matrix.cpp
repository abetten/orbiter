// int_matrix.cpp
//
// Anton Betten
//
// Oct 23, 2013




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


int_matrix::int_matrix()
{
	null();
}

int_matrix::~int_matrix()
{
	freeself();
}

void int_matrix::null()
{
	M = NULL;
	m = 0;
	n = 0;
}

void int_matrix::freeself()
{
	if (M) {
		FREE_int(M);
		}
	null();
}

void int_matrix::allocate(int m, int n)
{
	freeself();
	M = NEW_int(m * n);
	int_matrix::m = m;
	int_matrix::n = n;
}

void int_matrix::allocate_and_init(int m, int n, int *Mtx)
{
	freeself();
	M = NEW_int(m * n);
	int_matrix::m = m;
	int_matrix::n = n;
	int_vec_copy(Mtx, M, m * n);
}

int &int_matrix::s_ij(int i, int j)
{
	return M[i * n + j];
}

int &int_matrix::s_m()
{
	return m;
}

int &int_matrix::s_n()
{
	return n;
}

void int_matrix::print()
{
	int_matrix_print(M, m, n);
}

}
}

