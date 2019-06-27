// classify_bitvectors.cpp
//
// Anton Betten
//
// December 23, 2017




#include "foundations.h"

using namespace std;

namespace orbiter {
namespace foundations {


classify_bitvectors::classify_bitvectors()
{
	null();
}

classify_bitvectors::~classify_bitvectors()
{
	freeself();
}

void classify_bitvectors::null()
{
	nb_types = 0;
	Type_data = NULL;
	Type_rep = NULL;
	Type_mult = NULL;
	Type_extra_data = NULL;
	rep_len = 0;
	n = 0;
	N = 0;
	type_of = NULL;
	C_type_of = NULL;
	perm = NULL;
}

void classify_bitvectors::freeself()
{
	int i;

	if (Type_data) {
		for (i = 0; i < nb_types; i++) {
			FREE_uchar(Type_data[i]);
			}
		FREE_puchar(Type_data);
		}
	if (Type_extra_data) {
		for (i = 0; i < nb_types; i++) {
			//FREE_uchar(Type_data[i]);
			}
		FREE_pvoid(Type_extra_data);
		}
	if (Type_rep) {
		FREE_int(Type_rep);
		}
	if (Type_mult) {
		FREE_int(Type_mult);
		}
	if (type_of) {
		FREE_int(type_of);
		}
	if (C_type_of) {
		FREE_OBJECT(C_type_of);
		}
	if (perm) {
		FREE_int(perm);
		}
	null();
}


void classify_bitvectors::init(int N, int rep_len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "classify_bitvectors::init" << endl;
		}
	classify_bitvectors::N = N;
	classify_bitvectors::rep_len = rep_len;
	Type_data = NEW_puchar(N);
	Type_extra_data = NEW_pvoid(N);
	Type_rep = NEW_int(N);
	Type_mult = NEW_int(N);
	type_of = NEW_int(N);
	for (i = 0; i < N; i++) {
		type_of[i] = -1;
		}
	nb_types = 0;
	n = 0;
	C_type_of = NULL;
	perm = NULL;
	
	if (f_v) {
		cout << "classify_bitvectors::init done" << endl;
		}
}

int classify_bitvectors::search(uchar *data,
		int &idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;
	sorting Sorting;

	if (f_v) {
		cout << "classify_bitvectors::search" << endl;
		}
	if (Sorting.vec_search((void **) Type_data,
			compare_func_for_bitvectors, (void *) this,
		nb_types, data, idx, 0 /*verbose_level - 1*/)) {
		ret = TRUE;
	}
	else {
		ret = FALSE;
	}
	if (f_v) {
		cout << "classify_bitvectors::search done ret=" << ret << endl;
		}
	return ret;
}

int classify_bitvectors::add(uchar *data,
		void *extra_data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx, i, ret;
	sorting Sorting;
	
	if (f_v) {
		cout << "classify_bitvectors::add" << endl;
		}

	if (n >= N) {
		cout << "classify_bitvectors::add n >= N" << endl;
		cout << "n=" << n << endl;
		cout << "N=" << N << endl;
		exit(1);
		}
	if (Sorting.vec_search((void **) Type_data,
			compare_func_for_bitvectors, (void *) this,
		nb_types, data, idx, 0 /*verbose_level */)) {
		type_of[n] = idx;
		Type_mult[idx]++;
		ret = FALSE;
		}
	else {
		for (i = nb_types; i > idx; i--) {
			Type_data[i] = Type_data[i - 1];
			Type_extra_data[i] = Type_extra_data[i - 1];
			Type_rep[i] = Type_rep[i - 1];
			Type_mult[i] = Type_mult[i - 1];
			}
		Type_data[idx] = data;
		Type_extra_data[idx] = extra_data;
		Type_rep[idx] = n;
		Type_mult[idx] = 1;
		nb_types++;
		for (i = 0; i < n; i++) {
			if (type_of[i] >= idx) {
				type_of[i]++;
				}
			}
		type_of[n] = idx;
		ret = TRUE;
		}
	n++;


	if (f_v) {
		cout << "classify_bitvectors::add done, nb_types="
				<< nb_types << " ret=" << ret << endl;
		}
	return ret;
}

void classify_bitvectors::finalize(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sorting Sorting;

	if (f_v) {
		cout << "classify_bitvectors::finalize" << endl;
		}
	C_type_of = NEW_OBJECT(classify);

	C_type_of->init(type_of, N, FALSE, 0);


	int *v;
	int i;

	perm = NEW_int(nb_types);
	v = NEW_int(nb_types);
	for (i = 0; i < nb_types; i++) {
		perm[i] = i;
		v[i] = Type_rep[i];
		}
	Sorting.int_vec_heapsort_with_log(v, perm, nb_types);

	FREE_int(v);
	
	if (f_v) {
		cout << "classify_bitvectors::finalize done" << endl;
		}
}

void classify_bitvectors::print_reps()
{
	int i;
	
	cout << "We found " << nb_types << " types:" << endl;
	for (i = 0; i < nb_types; i++) {
		cout << i << " : " << Type_rep[i]
				<< " : " << Type_mult[i] << " : ";

#if 0
		for (j = 0; j < rep_len; j++) {
			cout << (int) Type_data[i][j];
			if (j < rep_len - 1) {
				cout << ", ";
				}
			}
#endif
		cout << endl;
		}
}

void classify_bitvectors::print_table()
{
	int i, j;

	cout << "We found " << nb_types << " types:" << endl;
	for (i = 0; i < nb_types; i++) {
		cout << i << " : " << Type_rep[i]
				<< " : " << Type_mult[i] << " : ";

		for (j = 0; j < rep_len; j++) {
			cout << (int) Type_data[i][j];
			if (j < rep_len - 1) {
				cout << ", ";
				}
			}
		cout << endl;
		}
}

void classify_bitvectors::save(const char *prefix, 
	void (*encode_function)(void *extra_data,
			int *&encoding, int &encoding_sz, void *global_data),
	void (*get_group_order_or_NULL)(void *extra_data,
			longinteger_object &go, void *global_data),
	void *global_data, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname_txt[1000];
	char fname_csv[1000];
	int i, j;
	
	if (f_v) {
		cout << "classify_bitvectors::save" << endl;
		}

	sprintf(fname_txt, "%s_iso.txt", prefix);
	sprintf(fname_csv, "%s_iso.csv", prefix);
	

	if (perm == NULL) {
		cout << "classify_bitvectors::save perm == NULL" << endl;
		exit(1);
		}
	int *Reps = NULL; // [nb_types * sz]
	int sz = 0;


	if (f_v) {
		cout << "classify_bitvectors::save writing file "
				<< fname_txt << endl;
		}
	{
	ofstream fp(fname_txt);
	int h;

	for (i = 0; i < nb_types; i++) {
		j = perm[i];

		int *encoding;
		int encoding_sz;

		if (f_v) {
			cout << "classify_bitvectors::save " << i << " / "
					<< nb_types << " j=" << j
					<< " before encode_function" << endl;
			}
		(*encode_function)(Type_extra_data[j],
				encoding, encoding_sz, global_data);
		if (f_v) {
			cout << "classify_bitvectors::save " << i
					<< " / " << nb_types
					<< " encoding_sz=" << encoding_sz << endl;
			}
		fp << encoding_sz;
		for (h = 0; h < encoding_sz; h++) {
			fp << " " << encoding[h];
			}
		if (get_group_order_or_NULL) {
			longinteger_object go;
			
			(*get_group_order_or_NULL)(Type_extra_data[j],
					go, global_data);
			fp << " ";
			go.print_not_scientific(fp);
			}
		fp << endl;
		if (i == 0) {
			sz = encoding_sz;
			Reps = NEW_int(nb_types * sz);
			int_vec_copy(encoding, Reps, sz);
			}
		else {
			if (encoding_sz != sz) {
				cout << "encoding_sz != sz" << endl;
				exit(1);
				}
			int_vec_copy(encoding, Reps + i * sz, sz);
			}
		FREE_int(encoding);
		}
	fp << "-1 " << nb_types << " " << N << endl;
	}
	file_io Fio;

	if (f_v) {
		cout << "classify_bitvectors::save Written "
				"file " << fname_txt
			<< " of size " << Fio.file_size(fname_txt)
			<< " with " << nb_types
			<< " orbit representatives obtained from "
			<< N << " candidates, encoding size = " << sz << endl;
		}

	if (f_v) {
		cout << "classify_bitvectors::save writing "
				"file " << fname_csv << endl;
		}
	Fio.int_matrix_write_csv(fname_csv, Reps, nb_types, sz);

	if (f_v) {
		cout << "classify_bitvectors::save "
				"Written file " << fname_csv
			<< " of size " << Fio.file_size(fname_csv)
			<< " with " << nb_types
			<< " orbit representatives obtained from "
			<< N << " candidates, encoding size = " << sz << endl;
		}

	if (Reps) {
		FREE_int(Reps);
		}
	
	if (f_v) {
		cout << "classify_bitvectors::save done" << endl;
		}
}

int compare_func_for_bitvectors(void *a, void *b, void *data)
{
	classify_bitvectors *CB = (classify_bitvectors *) data;
	uchar *A = (uchar *) a;
	uchar *B = (uchar *) b;
	int i;
	
	for (i = 0; i < CB->rep_len; i++) {
		if (A[i] < B[i]) {
			return -1;
			}
		if (A[i] > B[i]) {
			return 1;
			}
		}
	return 0;
}

}}


