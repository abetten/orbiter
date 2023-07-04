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

long int error_pattern::number_of_bit_error_patters(
		int wt,
		int verbose_level)
// binomial(Len_total_in_bits, wt)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "error_pattern::number_of_bit_error_patters" << endl;
	}

	combinatorics::combinatorics_domain Combi;
	ring_theory::longinteger_object N;
	long int nb;

	Combi.binomial(
			N,
			Crc_object->Len_total_in_bits, wt, 0 /*verbose_level */);

	nb = N.as_lint();

	if (f_v) {
		cout << "error_pattern::number_of_bit_error_patters: n = Crc_object->Len_total_in_bits = " << Crc_object->Len_total_in_bits << endl;
		cout << "error_pattern::number_of_bit_error_patters: k = wt = " << wt << endl;
		cout << "error_pattern::number_of_bit_error_patters: n_choose_k = " << N << endl;
		cout << "error_pattern::number_of_bit_error_patters: n_choose_k lint = " << nb << endl;
	}



	if (f_v) {
		cout << "error_pattern::number_of_bit_error_patters done" << endl;
	}
	return nb;
}

void error_pattern::first_bit_error_pattern_of_given_weight(
		combinatorics::combinatorics_domain &Combi,
		data_structures::algorithms &Algo,
		data_structures::data_structures_global &DataStructures,
		int wt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "error_pattern::first_bit_error_pattern_of_given_weight" << endl;
	}
	Combi.first_k_subset(A, Crc_object->Len_total_in_bits, wt);
	Algo.uchar_zero(Error_in_bytes, Crc_object->Len_total_in_bytes);

	int i, bit_pos;

	for (i = 0; i < wt; i++) {
		bit_pos = A[i];
		DataStructures.bitvector_set_bit_reversed(Error_in_bytes, bit_pos);
	}
}

int error_pattern::next_bit_error_pattern_of_given_weight(
		combinatorics::combinatorics_domain &Combi,
		data_structures::algorithms &Algo,
		data_structures::data_structures_global &DataStructures,
		int wt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "error_pattern::first_bit_error_pattern_of_given_weight" << endl;
	}
	if (!Combi.next_k_subset(A, Crc_object->Len_total_in_bits, wt)) {
		return false;
	}
	else {
		Algo.uchar_zero(Error_in_bytes, Crc_object->Len_total_in_bytes);

		int i, bit_pos;
		data_structures::data_structures_global DataStructures;

		for (i = 0; i < wt; i++) {
			bit_pos = A[i];
			DataStructures.bitvector_set_bit_reversed(Error_in_bytes, bit_pos);
		}

		return true;
	}
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
					Os.random_integer(Crc_object->info_length_in_symbols);
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


