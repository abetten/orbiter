// int_matrix.cpp
//
// Anton Betten
//
// Oct 23, 2013




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace data_structures {


int_matrix::int_matrix()
{
	M = NULL;
	m = 0;
	n = 0;
}

int_matrix::~int_matrix()
{
	if (M) {
		FREE_int(M);
		}
}

void int_matrix::allocate(int m, int n)
{
	if (M) {
		FREE_int(M);
		}
	M = NEW_int(m * n);
	int_matrix::m = m;
	int_matrix::n = n;
}

void int_matrix::allocate_and_init(int m, int n, int *Mtx)
{
	if (M) {
		FREE_int(M);
		}
	M = NEW_int(m * n);
	int_matrix::m = m;
	int_matrix::n = n;
	Int_vec_copy(Mtx, M, m * n);
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
	Int_matrix_print(M, m, n);
}

}}}

