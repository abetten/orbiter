// int_matrix.cpp
//
// Anton Betten
//
// Oct 23, 2013




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace data_structures {


static int int_matrix_compare_with(
		void *data, void *entry, int j, void *extra_data);
static int int_matrix_compare_with_restricted(
		void *data, void *entry, int j, void *extra_data);
static int int_matrix_compare_rows(
		void *data, int i, int j, void *extra_data);
static int int_matrix_compare_rows_in_reverse(
		void *data, int i, int j, void *extra_data);
static void int_matrix_swap_rows(
		void *data, int i, int j, void *extra_data);


int_matrix::int_matrix()
{
	Record_birth();
	M = NULL;
	m = 0;
	n = 0;
	perm_inv = NULL;
	perm = NULL;
}

int_matrix::~int_matrix()
{
	Record_death();
	if (M) {
		FREE_int(M);
	}
	if (perm) {
		FREE_int(perm);
	}
	if (perm_inv) {
		FREE_int(perm_inv);
	}
}

void int_matrix::allocate(
		int m, int n)
{
	if (M) {
		FREE_int(M);
	}
	M = NEW_int(m * n);
	int_matrix::m = m;
	int_matrix::n = n;
}

void int_matrix::allocate_and_initialize_with_zero(
		int m, int n)
{
	if (M) {
		FREE_int(M);
	}
	M = NEW_int(m * n);
	Int_vec_zero(M, m * n);
	int_matrix::m = m;
	int_matrix::n = n;
}

void int_matrix::allocate_and_init(
		int m, int n, int *Mtx)
{
	//if (M) {
	//	FREE_int(M);
	//	}
	M = NEW_int(m * n);
	int_matrix::m = m;
	int_matrix::n = n;
	Int_vec_copy(Mtx, M, m * n);
}

int &int_matrix::s_ij(
		int i, int j)
{
	return M[i * n + j];
}

int &int_matrix::s_m()
{
	return m;
}

int &int_matrix::s_n()
{
	return n;
}

void int_matrix::print()
{
	Int_matrix_print(M, m, n);
}

void int_matrix::sort_rows(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "int_matrix::sort_rows" << endl;
	}
	int i;

	perm = NEW_int(m);
	perm_inv = NEW_int(m);

	for (i = 0; i < m; i++) {
		perm[i] = i;
		perm_inv[i] = i;
	}

	if (f_v) {
		cout << "int_matrix::sort_rows before Sorting.Heapsort_general" << endl;
	}
	Sorting.Heapsort_general_with_log(M, perm_inv, m,
			int_matrix_compare_rows,
			int_matrix_swap_rows,
		this);


#if 0
	orbiter_kernel_system::file_io Fio;

	string fname;

	fname.assign("int_matrix_sorted.csv");
	Fio.int_matrix_write_csv(fname, M, m, n);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
#endif

	if (f_v) {
		cout << "int_matrix::sort_rows after Sorting.Heapsort_general" << endl;
	}

	Combi.Permutations->perm_inverse(perm_inv, perm, m);


	if (f_v) {
		cout << "int_matrix::sort_rows done" << endl;
	}
}


void int_matrix::sort_rows_in_reverse(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "int_matrix::sort_rows_in_reverse" << endl;
	}
	int i;

	perm = NEW_int(m);
	perm_inv = NEW_int(m);

	for (i = 0; i < m; i++) {
		perm[i] = i;
		perm_inv[i] = i;
	}

	if (f_v) {
		cout << "int_matrix::sort_rows_in_reverse before Sorting.Heapsort_general" << endl;
	}
	Sorting.Heapsort_general_with_log(M, perm_inv, m,
			int_matrix_compare_rows_in_reverse,
			int_matrix_swap_rows,
		this);


#if 0
	orbiter_kernel_system::file_io Fio;

	string fname;

	fname.assign("int_matrix_sorted.csv");
	Fio.int_matrix_write_csv(fname, M, m, n);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
#endif

	if (f_v) {
		cout << "int_matrix::sort_rows_in_reverse after Sorting.Heapsort_general" << endl;
	}

	Combi.Permutations->perm_inverse(perm_inv, perm, m);


	if (f_v) {
		cout << "int_matrix::sort_rows_in_reverse done" << endl;
	}
}


void int_matrix::remove_duplicates(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "int_matrix::remove_duplicates" << endl;
	}
	int i, j;

	if (m) {
		i = 1;
		for (j = 1; j < m; j++) {
			if (int_matrix_compare_rows(
					M, i - 1, j, this) == 0) {
				if (f_v) {
					cout << "entry " << i - 1 << " and " << j << " are the same" << endl;
				}
			}
			else {
				Int_vec_copy(M + j * n, M + i * n, n);
				i++;
			}
		}
		m = i;
	}

	if (f_v) {
		cout << "int_matrix::remove_duplicates done" << endl;
	}
}

void int_matrix::check_that_entries_are_distinct(
		int verbose_level)
{
	int i;

	for (i = 1; i < m; i++) {
		if (int_matrix_compare_rows(
				M, i - 1, i, this) == 0) {
			cout << "entry " << i - 1 << " and " << i << " are the same" << endl;
			exit(1);
		}
	}
}

int int_matrix::search(
		int *entry, int &idx, int verbose_level)
{
	int ret;

	data_structures::sorting Sorting;

	ret = Sorting.vec_search_general(M,
			int_matrix_compare_with,
			this,
			m, entry, idx,
			0 /* verbose_level*/);

#if 0
	int sorting::vec_search_general(void *vec,
		int (*compare_func)(void *vec, void *a, int b, void *data_for_compare),
		void *data_for_compare,
		int len, void *a, int &idx, int verbose_level)
#endif

	return ret;
}

int int_matrix::search_first_column_only(
		int value, int &idx, int verbose_level)
{
	int ret;
	int data[1];

	data_structures::sorting Sorting;

	data[0] = value;

	ret = Sorting.vec_search_general(M,
			int_matrix_compare_with_restricted,
			this,
			m, data, idx,
			0 /* verbose_level*/);

#if 0
	int sorting::vec_search_general(void *vec,
		int (*compare_func)(void *vec, void *a, int b, void *data_for_compare),
		void *data_for_compare,
		int len, void *a, int &idx, int verbose_level)
#endif

	return ret;
}


void int_matrix::write_csv(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "int_matrix::write_csv" << endl;
	}
	orbiter_kernel_system::file_io Fio;

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, M, m, n);

	if (f_v) {
		cout << "int_matrix::write_csv Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}

}

// callbacks:

static int int_matrix_compare_with(
		void *data, void *entry, int j, void *extra_data)
{
	int_matrix *IM = (int_matrix *) extra_data;
	int *Data;
	int n;
	int ret;
	data_structures::sorting Sorting;

	Data = (int *) IM->M;
	n = IM->n;

	ret = Sorting.int_vec_compare((int *)entry, Data + j * n, n) * -1;

	return ret;

}

static int int_matrix_compare_with_restricted(
		void *data, void *entry, int j, void *extra_data)
{
	int_matrix *IM = (int_matrix *) extra_data;
	int *Data;
	int n;
	int ret;
	data_structures::sorting Sorting;

	Data = (int *) IM->M;
	n = IM->n;

	ret = Sorting.int_vec_compare((int *)entry, Data + j * n, 1) * -1;

	return ret;

}


static int int_matrix_compare_rows(
		void *data, int i, int j, void *extra_data)
{
	int_matrix *IM = (int_matrix *) extra_data;
	int *Data;
	int n;
	int ret;
	data_structures::sorting Sorting;

	Data = (int *) IM->M;
	n = IM->n;

	ret = Sorting.int_vec_compare(Data + i * n, Data + j * n, n); // * -1

	return ret;
}

static int int_matrix_compare_rows_in_reverse(
		void *data, int i, int j, void *extra_data)
{
	int_matrix *IM = (int_matrix *) extra_data;
	int *Data;
	int n;
	int ret;
	data_structures::sorting Sorting;

	Data = (int *) IM->M;
	n = IM->n;

	ret = Sorting.int_vec_compare(Data + i * n, Data + j * n, n); // * -1

	ret = -1 * ret;

	return ret;
}

static void int_matrix_swap_rows(
		void *data, int i, int j, void *extra_data)
{
	int_matrix *IM = (int_matrix *) extra_data;
	int *Data;
	int h, a, n;

	Data = (int *) IM->M;
	n = IM->n;

	for (h = 0; h < n; h++) {
		a = Data[i * n + h];
		Data[i * n + h] = Data[j * n + h];
		Data[j * n + h] = a;
	}

}



}}}}

