// vector_hashing.C
//
// Anton Betten
//
// started:  October 14, 2008


#include "galois.h"

vector_hashing::vector_hashing()
{
	vector_data = NULL;
	H = NULL;
	H_sorted = NULL;
	perm = NULL;
	perm_inv = NULL;
	type_first = NULL;
	type_len = NULL;
	type_value = NULL;
}

vector_hashing::~vector_hashing()
{
	if (vector_data) {
		FREE_INT(vector_data);
		vector_data = NULL;
		}
	if (H) {
		FREE_INT(H);
		H = NULL;
		}
	if (H_sorted) {
		FREE_INT(H_sorted);
		H_sorted = NULL;
		}
	if (perm) {
		FREE_INT(perm);
		perm = NULL;
		}
	if (perm_inv) {
		FREE_INT(perm_inv);
		perm_inv = NULL;
		}
	if (type_first) {
		FREE_INT(type_first);
		type_first = NULL;
		}
	if (type_len) {
		FREE_INT(type_len);
		type_len = NULL;
		}
	if (type_value) {
		FREE_INT(type_value);
		type_value = NULL;
		}
}

void vector_hashing::allocate(INT data_size, INT N, INT bit_length)
{
	vector_hashing::data_size = data_size;
	vector_hashing::N = N;
	vector_hashing::bit_length = bit_length;
	vector_data = NEW_INT(N * data_size);
	H = NEW_INT(N);
}

void vector_hashing::compute_tables(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, idx;
	
	if (f_v) {
		cout << "vector_hashing::compute_tables" << endl;
		}
	for (i = 0; i < N; i++) {
		H[i] = INT_vec_hash(vector_data + i * data_size, data_size, bit_length);
		}
#if 0
	cout << "H:" << endl;
	INT_vec_print(cout, H, N);
	cout << endl;
#endif

	INT_vec_classify(N, H, H_sorted, perm, perm_inv, 
		nb_types, type_first, type_len);
	
	if (f_v) {
		cout << "N       =" << N << endl;
		cout << "nb_types=" << nb_types << endl;
		}
	type_value = NEW_INT(nb_types);
	
	for (i = 0; i < nb_types; i++) {
		idx = type_first[i] + 0;
		type_value[i] = H_sorted[idx];
		}
	
	//INT_vec_sorting_permutation(H, N, perm, perm_inv, TRUE /* f_increasingly */);
	
	
#if 0
	for (i = 0; i < N; i++) {
		H_sorted[perm[i]] = H[i];
		}
#endif
	
	
#if 0
	cout << "H sorted:" << endl;
	INT_vec_print(cout, H_sorted, N);
	cout << endl;
#endif
	if (f_vv) {
		cout << "vector_hashing::compute_tables() N=" << N << " nb_types=" << nb_types << endl;
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
				INT_vec_print(cout, vector_data + idx * data_size, data_size);
				cout << " : " << H[idx] << endl;
				}
			}
		}
}

void vector_hashing::print()
{
	INT i, j, idx;
	
	cout << "vector_hashing  N=" << N << " nb_types=" << nb_types << endl;
	cout << "data:" << endl;
	for (i = 0; i < N; i++) {
		cout << i << " : ";
		INT_vec_print(cout, vector_data + i * data_size, data_size);
		cout << " : " << H[i] << endl;
		}

	cout << "H sorted:" << endl;
	INT_vec_print(cout, H_sorted, N);
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
			INT_vec_print(cout, vector_data + idx * data_size, data_size);
			cout << " : " << H[idx] << endl;
			}
		}
	cout << "type_value:" << endl;
	for (i = 0; i < nb_types; i++) {
		cout << setw(4) << i << " : " << setw(10) << type_value[i] << endl;
		}
}

INT vector_hashing::rank(INT *data)
{
	INT h, idx, f, l, i, I;
	
	h = INT_vec_hash(data, data_size, bit_length);
	if (!INT_vec_search(type_value, nb_types, h, idx)) {
		cout << "vector_hashing::rank did not find hash value h=" << h << endl;
		exit(1);
		}
	f = type_first[idx];
	l = type_len[idx];
	for (i = 0; i < l; i++) {
		I = f + i;
		idx = perm_inv[I];
		if (INT_vec_compare(vector_data + idx * data_size, data, data_size) == 0) {
			return idx;
			}
		}
	cout << "vector_hashing::rank did not find data f=" << f << " l=" << l << endl;
	cout << "data:" << endl;
	INT_vec_print(cout, data, data_size);
	cout << endl;
	cout << "hash h=" << h << endl;
	cout << "idx=" << idx << endl;
	for (i = 0; i < l; i++) {
		I = f + i;
		idx = perm_inv[I];
		cout << I << " : " << idx << " : ";
		INT_vec_print(cout, vector_data + idx * data_size, data_size);
		cout << endl;
		}
	cout << endl;
	
	print();
	
	exit(1);
	
}

void vector_hashing::unrank(INT rk, INT *data)
{
	INT i;
	
	for (i = 0; i < data_size; i++) {
		data[i] = vector_data[rk * data_size + i];
		}
}

