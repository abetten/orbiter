/*
 * error_pattern.cpp
 *
 *  Created on: Jun 22, 2023
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace coding_theory {


error_pattern::error_pattern()
{
	Crc_object = NULL;
	k = 0;
	f_used = NULL;
	A = NULL;
	V = NULL;
	Error_in_symbols = NULL;
	Error_in_bytes = NULL;
}

error_pattern::~error_pattern()
{
	k = 0;
	if (f_used) {
		FREE_int(f_used);
	}
	if (A) {
		FREE_int(A);
	}
	if (V) {
		FREE_int(V);
	}
	if (Error_in_symbols) {
		FREE_char((char *) Error_in_symbols);
	}
	if (Error_in_bytes) {
		FREE_char((char *) Error_in_bytes);
	}
}

void error_pattern::init(crc_object *Crc_object,
		int k, int verbose_level)
{
	error_pattern::Crc_object = Crc_object;
	error_pattern::k = k;

	f_used = NEW_int(Crc_object->Len_total_in_symbols);
	A = NEW_int(k);
	V = NEW_int(k);
	Error_in_symbols = (unsigned char *) NEW_char(Crc_object->Len_total_in_symbols);
	Error_in_bytes = (unsigned char *) NEW_char(Crc_object->Len_total_in_bytes);
}


void error_pattern::create_error_pattern(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "error_pattern::create_error_pattern" << endl;
	}

	orbiter_kernel_system::os_interface Os;
	data_structures::algorithms Algo;

	Algo.uchar_zero(Error_in_symbols, Crc_object->Len_total_in_symbols);
	Algo.uchar_zero(Error_in_bytes, Crc_object->Len_total_in_bytes);

	Int_vec_zero(f_used, Crc_object->Len_total_in_symbols);

	int j, a, v;
	unsigned char c;

	for (j = 0; j < k; j++) {

		while (true) {
			a = Crc_object->Len_check_in_symbols +
					Os.random_integer(Crc_object->info_length_in_bytes);
			if (!f_used[a]) {
				break;
			}
		}

		f_used[a] = true;

		A[j] = a;

		v = 1 + Os.random_integer(Crc_object->symbol_set_size - 1);

		V[j] = v;

		c = (unsigned char) v;

		Error_in_symbols[a] = c;
	}

	Crc_object->compress(Error_in_symbols, Error_in_bytes);


}




}}}


