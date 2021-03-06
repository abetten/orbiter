/*
 * bitvector.cpp
 *
 *  Created on: Sep 28, 2020
 *      Author: betten
 */



#include "foundations.h"

using namespace std;

namespace orbiter {
namespace foundations {


bitvector::bitvector()
{
		data = NULL;
		length = 0;
		allocated_length = 0;
}

bitvector::~bitvector()
{
	if (data) {
		FREE_uchar(data);
		data = NULL;
	}
}

void bitvector::allocate(long int length)
{
	long int i;

	bitvector::length = length;
	allocated_length = (length + 7) >> 3;
	data = NEW_uchar(allocated_length);
	for (i = 0; i < allocated_length; i++) {
		data[i] = 0;
	}
}

long int bitvector::get_length()
{
	return length;
}

long int bitvector::get_allocated_length()
{
	return allocated_length;
}

uchar *bitvector::get_data()
{
	return data;
}

void bitvector::m_i(long int i, int a)
{
	long int ii, bit;
	uchar mask;

	ii = i >> 3;
	bit = i & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = data[ii];
	if (a == 0) {
		uchar not_mask = ~mask;
		x &= not_mask;
	}
	else {
		x |= mask;
	}
}

void bitvector::set_bit(long int i)
{
	long int ii, bit;
	uchar mask;

	ii = i >> 3;
	bit = i & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = data[ii];
	x |= mask;
}

int bitvector::s_i(long int i)
// returns 0 or 1
{
	long int ii, bit;
	uchar mask;

	ii = i >> 3;
	bit = i & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = data[ii];
	if (x & mask) {
		return 1;
	}
	else {
		return 0;
	}
}

void bitvector::save(ofstream &fp)
{
	fp.write((char*) &length, sizeof(long int));
	fp.write((char*) &allocated_length, sizeof(long int));
	fp.write((char*) data, allocated_length);
}

void bitvector::load(ifstream &fp)
{
	fp.read((char*) &length, sizeof(long int));
	fp.read((char*) &allocated_length, sizeof(long int));
	data = NEW_uchar(allocated_length);
	fp.read((char*) data, allocated_length);
}

}}

