/*
 * data_structures_global.cpp
 *
 *  Created on: Sep 29, 2020
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace data_structures {

data_structures_global::data_structures_global()
{
	//null();
}

data_structures_global::~data_structures_global()
{
	//null();
}

void data_structures_global::bitvector_m_ii(uchar *bitvec, long int i, int a)
{
	long int ii, bit;
	uchar mask;

	ii = i >> 3;
	bit = i & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = bitvec[ii];
	if (a == 0) {
		uchar not_mask = ~mask;
		x &= not_mask;
	}
	else {
		x |= mask;
	}
}

void data_structures_global::bitvector_set_bit(uchar *bitvec, long int i)
{
	long int ii, bit;
	uchar mask;

	ii = i >> 3;
	bit = i & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = bitvec[ii];
	x |= mask;
}

int data_structures_global::bitvector_s_i(uchar *bitvec, long int i)
// returns 0 or 1
{
	long int ii, bit;
	uchar mask;

	ii = i >> 3;
	bit = i & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = bitvec[ii];
	if (x & mask) {
		return 1;
	}
	else {
		return 0;
	}
}



uint32_t data_structures_global::int_vec_hash(int *data, int len)
{
	uint32_t h;
	algorithms Algo;

	h = Algo.SuperFastHash ((const char *) data, (uint32_t) len * sizeof(int));
	return h;
}

uint32_t data_structures_global::lint_vec_hash(long int *data, int len)
{
	uint32_t h;
	algorithms Algo;

	h = Algo.SuperFastHash ((const char *) data, (uint32_t) len * sizeof(long int));
	return h;
}

uint32_t data_structures_global::char_vec_hash(char *data, int len)
{
	uint32_t h;
	algorithms Algo;

	h = Algo.SuperFastHash ((const char *) data, (uint32_t) len);
	return h;
}

int data_structures_global::int_vec_hash_after_sorting(int *data, int len)
{
	int *data2;
	int i, h;
	sorting Sorting;

	data2 = NEW_int(len);
	for (i = 0; i < len; i++) {
		data2[i] = data[i];
	}
	Sorting.int_vec_heapsort(data2, len);
	h = int_vec_hash(data2, len);
	FREE_int(data2);
	return h;
}

int data_structures_global::lint_vec_hash_after_sorting(long int *data, int len)
{
	long int *data2;
	int i, h;
	sorting Sorting;

	data2 = NEW_lint(len);
	for (i = 0; i < len; i++) {
		data2[i] = data[i];
	}
	Sorting.lint_vec_heapsort(data2, len);
	h = lint_vec_hash(data2, len);
	FREE_lint(data2);
	return h;
}



}}}


