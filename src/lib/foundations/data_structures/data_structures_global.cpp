/*
 * data_structures_global.cpp
 *
 *  Created on: Sep 29, 2020
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {

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


}}

