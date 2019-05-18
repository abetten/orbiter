/*
 * classify_vector_data.cpp
 *
 *  Created on: May 17, 2019
 *      Author: betten
 */





#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {

classify_vector_data::classify_vector_data()
{
	data_set_sz = 0;
	data_length = 0;
	data = NULL;
	data_2_unique_data = NULL;
	data_unique_length = 0;
	Data_unique = NULL;
	Data_multiplicity = NULL;
	sorting_perm = NULL;
	sorting_perm_inv = NULL;
	nb_types = 0;
	type_first = NULL;
	type_len = NULL;

	f_second = FALSE;
	second_data_sorted = NULL;
	second_sorting_perm = NULL;
	second_sorting_perm_inv = NULL;
	second_nb_types = 0;
	second_type_first = NULL;
	second_type_len = NULL;

}


classify_vector_data::~classify_vector_data()
{

}

void classify_vector_data::init(int *data, int data_length, int data_set_sz,
		int f_second, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "classify_vector_data::init" << endl;
		}
	classify_vector_data::data = data;
	classify_vector_data::data_length = data_length;
	classify_vector_data::data_set_sz = data_set_sz;
	classify_vector_data::f_second = f_second;

	data_2_unique_data = NEW_int(data_length);
	Data_unique = NEW_int(data_length * data_set_sz);
	Data_multiplicity = NEW_int(data_length);
	data_unique_length = 0;
	sorting_perm = NEW_int(data_length);
	sorting_perm_inv = NEW_int(data_length);
	type_first = NEW_int(data_length);
	type_len = NEW_int(data_length);

	uint32_t h;
	int idx, a;

	for (i = 0; i < data_length; i++) {
		if (!hash_and_find(data + i * data_set_sz,
				idx, h, verbose_level)) {
			Hashing.insert(pair<uint32_t, int>(h, data_unique_length));
			int_vec_copy(data + i * data_set_sz,
					Data_unique + data_unique_length * data_set_sz,
					data_set_sz);
			Data_multiplicity[data_unique_length] = 1;
			data_2_unique_data[i] = data_unique_length;
			data_unique_length++;
		}
		else {
			data_2_unique_data[i] = idx;
			Data_multiplicity[idx]++;
		}
	}

	nb_types = data_unique_length;
	for (i = 0; i < data_unique_length; i++) {
		if (i == 0) {
			type_first[i] = 0;
		}
		else {
			type_first[i] = type_first[i - 1] + type_len[i - 1];
		}
		type_len[i] = Data_multiplicity[i];
	}

	int_vec_zero(sorting_perm_inv, data_length);
	int_vec_zero(type_len, data_unique_length);

	for (i = 0; i < data_length; i++) {
		a = data_2_unique_data[i];
		sorting_perm_inv[type_first[a] + type_len[a]++] = i;
	}
	for (i = 0; i < data_unique_length; i++) {
		if (type_len[i] != Data_multiplicity[i]) {
			cout << "type_len[i] != Data_multiplicity[i]" << endl;
			exit(1);
		}
	}
	combinatorics_domain Combi;

	Combi.perm_inverse(sorting_perm_inv, sorting_perm, data_length);

}

int classify_vector_data::hash_and_find(int *data,
		int &idx, uint32_t &h, int verbose_level)
{
	int f_found;



	h = int_vec_hash(data, data_set_sz);

    map<uint32_t, int>::iterator itr, itr1, itr2;

    itr1 = Hashing.lower_bound(h);
    itr2 = Hashing.upper_bound(h);
    f_found = FALSE;
	for (itr = itr1; itr != itr2; ++itr) {
    	idx = itr->second;
		if (int_vec_compare(data,
				Data_unique + idx * data_set_sz,
				data_set_sz) == 0) {
			f_found = TRUE;
			break;
        }
    }
    return f_found;
}


void classify_vector_data::print()
{
	uint32_t h;
	int i;

	for (i = 0; i < data_unique_length; i++) {

		h = int_vec_hash(Data_unique + i * data_set_sz, data_set_sz);

		cout << Data_multiplicity[i] << " x ";
		int_vec_print(cout, Data_unique + i * data_set_sz, data_set_sz);
		cout << endl;
		cout << "for elements ";
		int_vec_print(cout, sorting_perm_inv + type_first[i], type_len[i]);
		cout << endl;
		}
}


}}

