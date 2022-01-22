/*
 * tally_vector_data.cpp
 *
 *  Created on: May 17, 2019
 *      Author: betten
 */





#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {


static int classify_int_vec_compare_function(void *a, void *b, void *data);

tally_vector_data::tally_vector_data()
{
	data_set_sz = 0;
	data_length = 0;
	data = NULL;
	rep_idx = NULL;
	Reps = NULL;
	Frequency = NULL;
	sorting_perm = NULL;
	sorting_perm_inv = NULL;
	nb_types = 0;
	type_first = NULL;
	Reps_in_lex_order = NULL;
	Frequency_in_lex_order = NULL;
}


tally_vector_data::~tally_vector_data()
{
	if (rep_idx) {
		FREE_int(rep_idx);
	}
	if (Reps) {
		FREE_int(Reps);
	}
	if (Frequency) {
		FREE_int(Frequency);
	}
	if (sorting_perm) {
		FREE_int(sorting_perm);
	}
	if (sorting_perm_inv) {
		FREE_int(sorting_perm_inv);
	}
	if (type_first) {
		FREE_int(type_first);
	}
	if (Reps_in_lex_order) {
		int i;

		for (i = 0; i < nb_types; i++) {
			FREE_int(Reps_in_lex_order[i]);
		}
		FREE_pint(Reps_in_lex_order);
	}
	if (Frequency_in_lex_order) {
		FREE_int(Frequency_in_lex_order);
	}
}


void tally_vector_data::init(int *data, int data_length, int data_set_sz,
		int verbose_level)
// data[data_length * data_set_sz]
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::sorting Sorting;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "tally_vector_data::init" << endl;
	}
	tally_vector_data::data = data;
	tally_vector_data::data_length = data_length;
	tally_vector_data::data_set_sz = data_set_sz;

	rep_idx = NEW_int(data_length);
	Reps = NEW_int(data_length * data_set_sz);
	Frequency = NEW_int(data_length);
	nb_types = 0;
	type_first = NEW_int(data_length);

	uint32_t h;
	int idx, a;

	if (f_v) {
		cout << "tally_vector_data::init starting to collect data" << endl;
	}
	for (i = 0; i < data_length; i++) {
		if (FALSE) {
			cout << "tally_vector_data::init starting to collect data i=" << i << " / " << data_length << endl;
		}
		if (!hash_and_find(data + i * data_set_sz,
				idx, h, verbose_level)) {
			Hashing.insert(pair<uint32_t, int>(h, nb_types));
			Orbiter->Int_vec->copy(data + i * data_set_sz,
					Reps + nb_types * data_set_sz,
					data_set_sz);
			Frequency[nb_types] = 1;
			rep_idx[i] = nb_types;
			nb_types++;
		}
		else {
			rep_idx[i] = idx;
			Frequency[idx]++;
		}
	}
	if (f_v) {
		cout << "tally_vector_data::init finished collecting data" << endl;
	}

	for (i = 0; i < nb_types; i++) {
		if (i == 0) {
			type_first[i] = 0;
		}
		else {
			type_first[i] = type_first[i - 1] + Frequency[i - 1];
		}
	}

	int *Frequency2;

	sorting_perm = NEW_int(data_length);
	sorting_perm_inv = NEW_int(data_length);

	Frequency2 = NEW_int(nb_types);
	Orbiter->Int_vec->zero(Frequency2, nb_types);
	Orbiter->Int_vec->zero(sorting_perm_inv, data_length);
	for (i = 0; i < data_length; i++) {
		a = rep_idx[i];
		sorting_perm_inv[type_first[a] + Frequency2[a]++] = i;
	}
	Combi.perm_inverse(sorting_perm_inv, sorting_perm, data_length);

	FREE_int(Frequency2);


	if (f_v) {
		cout << "tally_vector_data::init computing Reps_in_lex_order" << endl;
	}

	Reps_in_lex_order = NEW_pint(nb_types);
	Frequency_in_lex_order = NEW_int(nb_types);

	int nb_types2, j;

	nb_types2 = 0;
	for (i = 0; i < nb_types; i++) {
		if (Sorting.vec_search((void **)Reps_in_lex_order,
				classify_int_vec_compare_function, this,
				nb_types2,
				Reps + i * data_set_sz,
				idx,
				0 /* verbose_level */)) {
			cout << "tally_vector_data::init error!" << endl;
			exit(1);
		}
		else {
			for (j = nb_types2; j > idx; j--) {
				Reps_in_lex_order[j] = Reps_in_lex_order[j - 1];
				Frequency_in_lex_order[j] = Frequency_in_lex_order[j - 1];
			}
			Reps_in_lex_order[idx] = NEW_int(data_set_sz);
			Frequency_in_lex_order[idx] = Frequency[i];
			Orbiter->Int_vec->copy(Reps + i * data_set_sz, Reps_in_lex_order[idx],
					data_set_sz);
			nb_types2++;
		}
	}


	if (f_v) {
		cout << "tally_vector_data::init done" << endl;
	}

}

int tally_vector_data::hash_and_find(int *data,
		int &idx, uint32_t &h, int verbose_level)
{
	int f_found;
	data_structures::data_structures_global D;
	data_structures::sorting Sorting;


	h = D.int_vec_hash(data, data_set_sz);

    map<uint32_t, int>::iterator itr, itr1, itr2;

    itr1 = Hashing.lower_bound(h);
    itr2 = Hashing.upper_bound(h);
    f_found = FALSE;
	for (itr = itr1; itr != itr2; ++itr) {
    	idx = itr->second;
		if (Sorting.int_vec_compare(data,
				Reps + idx * data_set_sz,
				data_set_sz) == 0) {
			f_found = TRUE;
			break;
        }
    }
    return f_found;
}


void tally_vector_data::print()
{
	//uint32_t h;
	int i;

	for (i = 0; i < nb_types; i++) {

		//h = int_vec_hash(Reps + i * data_set_sz, data_set_sz);

		cout << Frequency[i] << " x ";
		Orbiter->Int_vec->print(cout, Reps + i * data_set_sz, data_set_sz);
		cout << endl;
#if 0
		cout << "for elements ";
		int_vec_print(cout, sorting_perm_inv + type_first[i], Frequency[i]);
		cout << endl;
#endif
	}
}

void tally_vector_data::save_classes_individually(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//uint32_t h;
	int i, j;
	file_io Fio;

	if (f_v) {
		cout << "tally_vector_data::save_classes_individually fname = " << fname << endl;
	}
	if (f_v) {
		cout << "tally_vector_data::save_classes_individually nb_types = " << nb_types << endl;
	}
	for (i = 0; i < nb_types; i++) {

		string fname2;
		char str[10000];

		fname2.assign(fname);
		for (j = 0; j < data_set_sz; j++) {
			sprintf(str, "%d", Reps[i * data_set_sz + j]);
			fname2.append(str);
		}
		fname2.append(".csv");

		//h = int_vec_hash(Reps + i * data_set_sz, data_set_sz);

		if (f_v) {
			cout << "tally_vector_data::save_classes_individually saving file = " << fname2 << endl;
		}
		Fio.int_vec_write_csv(sorting_perm_inv + type_first[i], Frequency[i], fname2, "case");
		cout << "Written file " << fname2 << " of size " << Fio.file_size(fname2) << endl;
	}

	string fname2;

	fname2.assign(fname);
	fname2.append("_all_in_one.csv");
	Fio.int_vec_write_csv(sorting_perm_inv, data_length, fname2, "case");
	cout << "Written file " << fname2 << " of size " << Fio.file_size(fname2) << endl;


	if (f_v) {
		cout << "tally_vector_data::save_classes_individually" << endl;
	}
}

void tally_vector_data::get_transversal(
		int *&transversal, int *&frequency, int &nb_types, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, f, l;

	if (f_v) {
		cout << "tally_vector_data::get_transversal" << endl;
		}

	nb_types = tally_vector_data::nb_types;
	transversal = NEW_int(nb_types);
	frequency = NEW_int(nb_types);
	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		l = Frequency[i];
		transversal[i] = sorting_perm_inv[f + 0];
		frequency[i] = l;
	}

	if (f_v) {
		cout << "tally_vector_data::get_transversal done" << endl;
	}
}

void tally_vector_data::print_classes_bigger_than_one(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, f, l;

	if (f_v) {
		cout << "tally_vector_data::print_classes_bigger_than_one" << endl;
		}

	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		l = Frequency[i];
		if (l > 1) {
			cout << "class " << i << " has size " << l << " : ";
			for (j = 0; j < l; j++) {
				a = sorting_perm_inv[f + j];
				cout << a;
				if (j < l - 1) {
					cout << ", ";
				}
			}
			cout << endl;
		}
	}

	if (f_v) {
		cout << "tally_vector_data::print_classes_bigger_than_one done" << endl;
	}
}


static int classify_int_vec_compare_function(void *a, void *b, void *data)
{
	tally_vector_data *C = (tally_vector_data *) data;
	int *A = (int *) a;
	int *B = (int *) b;
	int i;

	for (i = 0; i < C->data_set_sz; i++) {
		if (A[i] > B[i]) {
			return 1;
		}
		if (A[i] < B[i]) {
			return -1;
		}
	}
	return 0;

}

}}

