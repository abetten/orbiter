/*
 * bitvector.cpp
 *
 *  Created on: Sep 28, 2020
 *      Author: betten
 */



#include "foundations.h"

using namespace std;

namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace data_structures {


bitvector::bitvector()
{
	Record_birth();
		data = NULL;
		length = 0;
		allocated_length = 0;
}

bitvector::~bitvector()
{
	Record_death();
	if (data) {
		FREE_uchar(data);
		data = NULL;
	}
}

void bitvector::allocate(
		long int length)
{
	long int i;

	bitvector::length = length;
	allocated_length = (length + 7) >> 3;
	data = NEW_uchar(allocated_length);
	for (i = 0; i < allocated_length; i++) {
		data[i] = 0;
	}
}

void bitvector::zero()
{
	int i;

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

void bitvector::m_i(
		long int i, int a)
{
	long int ii, bit;
	uchar mask;

	ii = i >> 3;
	bit = i & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = data[ii];

	//cout << "before: x = " << (int) x << endl;
	if (a == 0) {
		uchar not_mask = ~mask;
		x &= not_mask;
	}
	else {
		x |= mask;
	}
	//cout << "after: x = " << (int) x << endl;
}

void bitvector::set_bit(
		long int i)
{
	long int ii, bit;
	uchar mask;

	ii = i >> 3;
	bit = i & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = data[ii];
	x |= mask;
}

int bitvector::s_i(
		long int i)
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

void bitvector::save(
		std::ofstream &fp)
{
	fp.write((char*) &length, sizeof(long int));
	fp.write((char*) &allocated_length, sizeof(long int));
	fp.write((char*) data, allocated_length);
}

void bitvector::load(
		std::ifstream &fp)
{
	fp.read((char*) &length, sizeof(long int));
	fp.read((char*) &allocated_length, sizeof(long int));
	data = NEW_uchar(allocated_length);
	fp.read((char*) data, allocated_length);
}

uint32_t bitvector::compute_hash()
{
	data_structures_global Data;
	uint32_t h;

	h = Data.char_vec_hash((char*) data, allocated_length);
	return h;
}

void bitvector::print_bitwise()
{
	long int i;
	int a;

	for (i = 0; i < length; i++) {
		a = s_i(i);
		cout << a;
	}
	cout << endl;
}

void bitvector::print()
{
	long int i;
	//int a;

	uint32_t h;

	h = compute_hash();
	cout << "hash = " << setw(5) << (int) h << " : allocated_length=" << allocated_length << " : ";
	if (allocated_length < 50) {
		for (i = 0; i < allocated_length; i++) {
			cout << (int) data[i];
			if (i < allocated_length - 1) {
				cout << ", ";
			}
		}
	}
	else {
		cout << "too long to print";
	}
#if 0
	for (i = 0; i < length; i++) {
		a = s_i(i);
		cout << a;
	}
#endif
	cout << endl;
}


}}}}


