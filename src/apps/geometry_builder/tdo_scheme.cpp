/*
 * tdo_scheme.cpp
 *
 *  Created on: Aug 16, 2021
 *      Author: betten
 */




#include "geo.h"

using namespace std;


tdo_scheme::tdo_scheme()
{
	m = 0;
	n = 0;
	a = NULL;
}

tdo_scheme::~tdo_scheme()
{
	delete [] a;
}

void tdo_scheme::allocate(int nb_rows, int nb_cols)
{
	int sz, i;

	m = nb_rows;
	n = nb_cols;
	sz = (m + 1) * (n + 1);

	a = new int [sz];

	for (i = 0; i < sz; i++) {
		a[i] = 0;
	}
	a[0] = sz;
}

int &tdo_scheme::nb_rows()
{
	return m;
}

int &tdo_scheme::nb_cols()
{
	return n;
}

int &tdo_scheme::Vi(int i)
{
	return a[(i + 1) * (n + 1) + 0];
}

int &tdo_scheme::Bj(int j)
{
	return a[0 * (n + 1) + j + 1];
}

int &tdo_scheme::aij(int i, int j)
{
	return a[(i + 1) * (n + 1) + j + 1];
}



void tdo_scheme::print()
{
	int nb_rows, nb_cols, i, j;

	nb_rows = m;
	nb_cols = n;
	cout << "      ";
	for (j = 0; j < nb_cols; j++)
		cout << setw(3) << Bj(j) << " ";
	cout << endl;
	for (i = 0; i < nb_rows; i++) {
		cout << setw(3) << Vi(i) << " | ";
		for (j = 0; j < nb_cols; j++) {
			cout << setw(3) << aij(i, j) << " ";
			}
		cout << endl;
		}
}

#if 0
void tdos_copy(tdo_scheme *t1, tdo_scheme *t2)
{
	t2->m = t1->m;
	t2->n = t1->n;
	t2->a = t1->a;
}
#endif

