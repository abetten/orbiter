// vector_hashing.cpp
//
// Anton Betten
//
// started:  October 14, 2008


#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace data_structures {


vector_hashing::vector_hashing()
{
	data_size = 0;
	N = 0;
	bit_length = 0;
	vector_data = NULL;
	H = NULL;
	H_sorted = NULL;
	perm = NULL;
	perm_inv = NULL;
	nb_types = 0;
	type_first = NULL;
	type_len = NULL;
	type_value = NULL;

}

vector_hashing::~vector_hashing()
{
	if (vector_data) {
		FREE_int(vector_data);
		vector_data = NULL;
		}
	if (H) {
		FREE_int(H);
		H = NULL;
		}
	if (H_sorted) {
		FREE_int(H_sorted);
		H_sorted = NULL;
		}
	if (perm) {
		FREE_int(perm);
		perm = NULL;
		}
	if (perm_inv) {
		FREE_int(perm_inv);
		perm_inv = NULL;
		}
	if (type_first) {
		FREE_int(type_first);
		type_first = NULL;
		}
	if (type_len) {
		FREE_int(type_len);
		type_len = NULL;
		}
	if (type_value) {
		FREE_int(type_value);
		type_value = NULL;
		}
}

void vector_hashing::allocate(int data_size, int N, int bit_length)
{
	vector_hashing::data_size = data_size;
	vector_hashing::N = N;
	vector_hashing::bit_length = bit_length;
	vector_data = NEW_int(N * data_size);
	H = NEW_int(N);
}

void vector_hashing::compute_tables(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, idx;
	sorting Sorting;

	
	if (f_v) {
		cout << "vector_hashing::compute_tables" << endl;
		}
	for (i = 0; i < N; i++) {
		H[i] = Orbiter->Int_vec->hash(
				vector_data + i * data_size,
				data_size, bit_length);
		}
#if 0
	cout << "H:" << endl;
	int_vec_print(cout, H, N);
	cout << endl;
#endif

	Sorting.int_vec_classify(N, H, H_sorted, perm, perm_inv,
		nb_types, type_first, type_len);
	
	if (f_v) {
		cout << "N       =" << N << endl;
		cout << "nb_types=" << nb_types << endl;
		}
	type_value = NEW_int(nb_types);
	
	for (i = 0; i < nb_types; i++) {
		idx = type_first[i] + 0;
		type_value[i] = H_sorted[idx];
		}
	
	//int_vec_sorting_permutation(H, N, perm, perm_inv,
	//TRUE /* f_increasingly */);
	
	
#if 0
	for (i = 0; i < N; i++) {
		H_sorted[perm[i]] = H[i];
		}
#endif
	
	
#if 0
	cout << "H sorted:" << endl;
	int_vec_print(cout, H_sorted, N);
	cout << endl;
#endif
	if (f_vv) {
		cout << "vector_hashing::compute_tables() N=" << N
				<< " nb_types=" << nb_types << endl;
		for (i = 0; i < nb_types; i++) {
			if (type_len[i] == 1)
				continue;
			cout << i << " : " 
				<< type_first[i] << " : " 
				<< type_len[i] 
				<< " : " << H_sorted[type_first[i]] << " : " << endl;
			for (j = 0; j < type_len[i]; j++) {
				idx = perm_inv[type_first[i] + j];
				cout << "j=" << j << " index " << idx << endl;
				cout << idx << " : ";
				Orbiter->Int_vec->print(cout, vector_data + idx * data_size, data_size);
				cout << " : " << H[idx] << endl;
				}
			}
		}
}

void vector_hashing::print()
{
	int i, j, idx;
	
	cout << "vector_hashing  N=" << N << " nb_types=" << nb_types << endl;
	cout << "data:" << endl;
	for (i = 0; i < N; i++) {
		cout << i << " : ";
		Orbiter->Int_vec->print(cout, vector_data + i * data_size, data_size);
		cout << " : " << H[i] << endl;
		}

	cout << "H sorted:" << endl;
	Orbiter->Int_vec->print(cout, H_sorted, N);
	cout << endl;

	cout << "types:" << endl;
	for (i = 0; i < nb_types; i++) {
		//if (type_len[i] == 1)
			//continue;
		cout << i << " : " 
			<< type_first[i] << " : " 
			<< type_len[i] 
			<< " : " << H_sorted[type_first[i]] << " : " << endl;
		for (j = 0; j < type_len[i]; j++) {
			idx = perm_inv[type_first[i] + j];
			cout << "j=" << j << " index " << idx << endl;
			cout << idx << " : ";
			Orbiter->Int_vec->print(cout, vector_data + idx * data_size, data_size);
			cout << " : " << H[idx] << endl;
			}
		}
	cout << "type_value:" << endl;
	for (i = 0; i < nb_types; i++) {
		cout << setw(4) << i << " : " << setw(10) << type_value[i] << endl;
		}
}

int vector_hashing::rank(int *data)
{
	int h, idx, f, l, i, I;
	sorting Sorting;
	
	h = Orbiter->Int_vec->hash(data, data_size, bit_length);
	if (!Sorting.int_vec_search(type_value, nb_types, h, idx)) {
		cout << "vector_hashing::rank did not "
				"find hash value h=" << h << endl;
		exit(1);
		}
	f = type_first[idx];
	l = type_len[idx];
	for (i = 0; i < l; i++) {
		I = f + i;
		idx = perm_inv[I];
		if (Sorting.int_vec_compare(vector_data + idx * data_size,
				data, data_size) == 0) {
			return idx;
			}
		}
	cout << "vector_hashing::rank did not find "
			"data f=" << f << " l=" << l << endl;
	cout << "data:" << endl;
	Orbiter->Int_vec->print(cout, data, data_size);
	cout << endl;
	cout << "hash h=" << h << endl;
	cout << "idx=" << idx << endl;
	for (i = 0; i < l; i++) {
		I = f + i;
		idx = perm_inv[I];
		cout << I << " : " << idx << " : ";
		Orbiter->Int_vec->print(cout, vector_data + idx * data_size, data_size);
		cout << endl;
		}
	cout << endl;
	
	print();
	
	exit(1);
	
}

void vector_hashing::unrank(int rk, int *data)
{
	int i;
	
	for (i = 0; i < data_size; i++) {
		data[i] = vector_data[rk * data_size + i];
		}
}

}}}

