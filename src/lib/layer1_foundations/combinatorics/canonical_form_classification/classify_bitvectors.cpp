// classify_bitvectors.cpp
//
// Anton Betten
//
// December 23, 2017




#include "foundations.h"

using namespace std;

namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace canonical_form_classification {


static int compare_func_for_bitvectors(
		void *a, void *b, void *data);


classify_bitvectors::classify_bitvectors()
{
	Record_birth();
	nb_types = 0;
	rep_len = 0;
	Type_data = NULL;
	Type_rep = NULL;
	Type_mult = NULL;
	Type_extra_data = NULL;

	N = 0;
	n = 0;

	type_of = NULL;
	C_type_of = NULL;

	perm = NULL;
}



classify_bitvectors::~classify_bitvectors()
{
	int i;

	Record_death();
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
}


void classify_bitvectors::init(
		int N, int rep_len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "classify_bitvectors::init, N=" << N << endl;
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

int classify_bitvectors::search(
		uchar *data,
		int &idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "classify_bitvectors::search" << endl;
	}
	if (Sorting.vec_search((void **) Type_data,
			compare_func_for_bitvectors, (void *) this,
		nb_types, data, idx, 0 /*verbose_level - 1*/)) {
		ret = true;
	}
	else {
		ret = false;
	}
	if (f_v) {
		cout << "classify_bitvectors::search done ret=" << ret << endl;
	}
	return ret;
}



void classify_bitvectors::canonical_form_search_and_add_if_new(
		other::data_structures::bitvector *Canonical_form,
		void *extra_data, int &f_found, int &idx_of_canonical_form,
		int verbose_level)
// if f_found is true: idx_of_canonical_form is where the canonical form was found.
// if f_found is false: idx_of_canonical_form is where the new canonical form was added.
{
	int f_v = (verbose_level >= 1);
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "classify_bitvectors::canonical_form_search_and_add_if_new" << endl;
		cout << "classify_bitvectors::canonical_form_search_and_add_if_new verbose_level=" << verbose_level << endl;
		cout << "classify_bitvectors::canonical_form_search_and_add_if_new rep_len=" << rep_len << endl;
	}

	if (n >= N) {
		cout << "classify_bitvectors::canonical_form_search_and_add_if_new n >= N" << endl;
		cout << "n=" << n << endl;
		cout << "N=" << N << endl;
		exit(1);
	}

	if (Canonical_form->get_allocated_length() != rep_len) {
		cout << "classify_bitvectors::canonical_form_search_and_add_if_new Canonical_form->allocated_length != rep_len" << endl;
		cout << "classify_bitvectors::canonical_form_search_and_add_if_new Canonical_form->allocated_length = " << Canonical_form->get_allocated_length() << endl;
		cout << "classify_bitvectors::canonical_form_search_and_add_if_new rep_len = " << rep_len << endl;
		exit(1);
	}
	if (Sorting.vec_search(
			(void **) Type_data,
			compare_func_for_bitvectors, (void *) this,
			nb_types, Canonical_form->get_data(), idx_of_canonical_form,
			0 /*verbose_level - 2*/)) {
		if (f_v) {
			cout << "classify_bitvectors::canonical_form_search_and_add_if_new vec_search "
					"returns true, idx_of_canonical_form=" << idx_of_canonical_form << endl;
		}
		type_of[n] = idx_of_canonical_form;
		Type_mult[idx_of_canonical_form]++;
		f_found = true;
	}
	else {
		if (f_v) {
			cout << "classify_bitvectors::canonical_form_search_and_add_if_new vec_search "
					"returns false, new bitvector, before add_at_idx" << endl;
		}
		add_at_idx(
				Canonical_form->get_data(), extra_data, idx_of_canonical_form,
				0/*verbose_level*/);
		if (f_v) {
			cout << "classify_bitvectors::canonical_form_search_and_add_if_new vec_search "
					"after add_at_idx" << endl;
		}
		f_found = false;
	}
	n++;


	if (f_v) {
		cout << "classify_bitvectors::canonical_form_search_and_add_if_new done, nb_types="
				<< nb_types << endl;
	}
}


void classify_bitvectors::search_and_add_if_new(
		uchar *data,
		void *extra_data, int &f_found, int &idx,
		int verbose_level)
// if f_found is true: idx is where the canonical form was found.
// if f_found is false: idx is where the new canonical form was added.
{
	int f_v = (verbose_level >= 1);
	other::data_structures::sorting Sorting;
	
	if (f_v) {
		cout << "classify_bitvectors::search_and_add_if_new" << endl;
		cout << "classify_bitvectors::search_and_add_if_new verbose_level=" << verbose_level << endl;
		cout << "classify_bitvectors::search_and_add_if_new rep_len=" << rep_len << endl;
	}

	if (n >= N) {
		cout << "classify_bitvectors::search_and_add_if_new n >= N" << endl;
		cout << "n=" << n << endl;
		cout << "N=" << N << endl;
		exit(1);
	}
	if (Sorting.vec_search(
			(void **) Type_data,
			compare_func_for_bitvectors, (void *) this,
			nb_types, data, idx,
			0 /*verbose_level - 2*/)) {
		if (f_v) {
			cout << "classify_bitvectors::search_and_add_if_new vec_search "
					"returns true, idx=" << idx << endl;
		}
		type_of[n] = idx;
		Type_mult[idx]++;
		f_found = true;
	}
	else {
		if (f_v) {
			cout << "classify_bitvectors::search_and_add_if_new vec_search "
					"returns false, new bitvector, before add_at_idx" << endl;
		}
		add_at_idx(data, extra_data, idx, 0/*verbose_level*/);
		if (f_v) {
			cout << "classify_bitvectors::search_and_add_if_new vec_search "
					"after add_at_idx" << endl;
		}
		f_found = false;
	}
	n++;


	if (f_v) {
		cout << "classify_bitvectors::search_and_add_if_new done, nb_types="
				<< nb_types << endl;
	}
}

int classify_bitvectors::compare_at(
		uchar *data, int idx)
{
	int ret;

	ret = compare_func_for_bitvectors(
			(void **) Type_data[idx], data, (void *) this);
	return ret;
}

void classify_bitvectors::add_at_idx(
		uchar *data,
		void *extra_data, int idx, int verbose_level)
// stores extra_data in Type_extra_data[idx]
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "classify_bitvectors::add_at_idx" << endl;
	}
	for (i = nb_types; i > idx; i--) {
		Type_data[i] = Type_data[i - 1];
		Type_extra_data[i] = Type_extra_data[i - 1];
		Type_rep[i] = Type_rep[i - 1];
		Type_mult[i] = Type_mult[i - 1];
	}

	Type_data[idx] = NEW_uchar(rep_len);
	for (i = 0; i < rep_len; i++) {
		Type_data[idx][i] = data[i];
	}

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
}


void classify_bitvectors::finalize(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "classify_bitvectors::finalize" << endl;
	}
	C_type_of = NEW_OBJECT(other::data_structures::tally);

	if (f_v) {
		cout << "classify_bitvectors::finalize type_of=";
		Int_vec_print(cout, type_of, N);
		cout << endl;
	}
	C_type_of->init(type_of, N, false, 0);
	if (f_v) {
		cout << "classify_bitvectors::finalize classification:" << endl;
		C_type_of->print(true /* f_backwards*/);
		cout << endl;
	}

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

void classify_bitvectors::print_canonical_forms()
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

void classify_bitvectors::save(
	std::string &prefix,
	void (*encode_function)(void *extra_data,
			long int *&encoding, int &encoding_sz, void *global_data),
	void (*get_group_order_or_NULL)(void *extra_data,
			algebra::ring_theory::longinteger_object &go, void *global_data),
	void *global_data, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname_txt;
	string fname_csv;
	int i, j;
	
	if (f_v) {
		cout << "classify_bitvectors::save" << endl;
	}

	fname_txt = prefix + "_iso.txt";
	fname_csv = prefix + "_iso.csv";
	

	if (perm == NULL) {
		cout << "classify_bitvectors::save perm == NULL" << endl;
		exit(1);
	}
	long int *Reps = NULL; // [nb_types * sz]
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

			long int *encoding;
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
				algebra::ring_theory::longinteger_object go;

				(*get_group_order_or_NULL)(Type_extra_data[j], go, global_data);
				fp << " ";
				go.print_not_scientific(fp);
			}
			fp << endl;
			if (i == 0) {
				sz = encoding_sz;
				Reps = NEW_lint(nb_types * sz);
				Lint_vec_copy(encoding, Reps, sz);
			}
			else {
				if (encoding_sz != sz) {
					cout << "encoding_sz != sz" << endl;
					exit(1);
				}
				Lint_vec_copy(encoding, Reps + i * sz, sz);
			}
			FREE_lint(encoding);
		}
		fp << "-1 " << nb_types << " " << N << endl;
	}
	other::orbiter_kernel_system::file_io Fio;

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
	Fio.Csv_file_support->lint_matrix_write_csv(
			fname_csv, Reps, nb_types, sz);

	if (f_v) {
		cout << "classify_bitvectors::save "
				"Written file " << fname_csv
			<< " of size " << Fio.file_size(fname_csv)
			<< " with " << nb_types
			<< " orbit representatives obtained from "
			<< N << " candidates, encoding size = " << sz << endl;
	}

	if (Reps) {
		FREE_lint(Reps);
	}
	
	if (f_v) {
		cout << "classify_bitvectors::save done" << endl;
	}
}

static int compare_func_for_bitvectors(
		void *a, void *b, void *data)
{
	classify_bitvectors *CB = (classify_bitvectors *) data;
	uchar *A = (uchar *) a;
	uchar *B = (uchar *) b;
	int i;
	
	//cout << "compare_func_for_bitvectors CB->rep_len=" << CB->rep_len << endl;
	for (i = 0; i < CB->rep_len; i++) {
		//cout << "i = " << i << " A[i]=" << (int) A[i] << " B[i]=" << (int) B[i] << endl;
		if (A[i] < B[i]) {
			return -1;
		}
		if (A[i] > B[i]) {
			return 1;
		}
	}
	return 0;
}

}}}}



