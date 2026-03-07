/*
 * lint_matrix.cpp
 *
 *  Created on: Mar 6, 2026
 *      Author: betten
 */







#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace data_structures {




lint_matrix::lint_matrix()
{
	Record_birth();
	M = NULL;
	m = 0;
	n = 0;
	perm_inv = NULL;
	perm = NULL;
}

lint_matrix::~lint_matrix()
{
	Record_death();
	if (M) {
		FREE_lint(M);
	}
	if (perm) {
		FREE_int(perm);
	}
	if (perm_inv) {
		FREE_int(perm_inv);
	}
}

void lint_matrix::allocate(
		int m, int n)
{
	if (M) {
		FREE_lint(M);
	}
	M = NEW_lint(m * n);
	lint_matrix::m = m;
	lint_matrix::n = n;
}

void lint_matrix::allocate_and_initialize_with_zero(
		int m, int n)
{
	if (M) {
		FREE_lint(M);
	}
	M = NEW_lint(m * n);
	Lint_vec_zero(M, m * n);
	lint_matrix::m = m;
	lint_matrix::n = n;
}

void lint_matrix::allocate_and_init(
		int m, int n, long int *Mtx)
{
	//if (M) {
	//	FREE_int(M);
	//	}
	M = NEW_lint(m * n);
	lint_matrix::m = m;
	lint_matrix::n = n;
	Lint_vec_copy(Mtx, M, m * n);
}

long int &lint_matrix::s_ij(
		int i, int j)
{
	return M[i * n + j];
}

int &lint_matrix::s_m()
{
	return m;
}

int &lint_matrix::s_n()
{
	return n;
}

void lint_matrix::print()
{
	Lint_matrix_print(M, m, n);
}


void lint_matrix::write_csv(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "lint_matrix::write_csv" << endl;
	}
	other::orbiter_kernel_system::file_io Fio;


	Fio.Csv_file_support->lint_matrix_write_csv(
			fname, M, m, n);

	if (f_v) {
		cout << "lint_matrix::write_csv Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "lint_matrix::write_csv done" << endl;
	}
}

void lint_matrix::write_csv_vectorized(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "lint_matrix::write_csv_vectorized" << endl;
	}
	orbiter_kernel_system::file_io Fio;


	string *Table;
	int nb_cols;
	int nb_rows;
	int i;

	nb_rows = m;
	nb_cols = 2;

	Table = new string[nb_rows * nb_cols];
	for (i = 0; i < nb_rows; i++) {

		Table[i * nb_cols + 0] =
				std::to_string(i);


		Table[i * nb_cols + 1] =
				"\"" + Lint_vec_stringify(M + i * n, n) + "\"";
	}

	std::string Col_headings[2];

	Col_headings[0] = "Idx";
	Col_headings[1] = "Vector";




	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);

	delete [] Table;


	if (f_v) {
		cout << "lint_matrix::write_csv_vectorized Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}

}


void lint_matrix::write_index_set_csv(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "lint_matrix::write_index_set_csv" << endl;
	}
	orbiter_kernel_system::file_io Fio;

#if 0
	Fio.Csv_file_support->int_matrix_write_csv(
			fname, M, m, n);
#endif

	string *Table;
	int nb_cols;
	int nb_rows;
	int i;
	//int f_latex = false;
	int *index_set;
	int sz, j;

	nb_rows = m;
	nb_cols = 3;

	index_set = NEW_int(n);
	Table = new string[nb_rows * nb_cols];
	for (i = 0; i < nb_rows; i++) {

		Table[i * nb_cols + 0] =
				std::to_string(i);

		sz = 0;
		for (j = 0; j < n; j++) {
			if (M[i * n + j]) {
				index_set[sz++] = j;
			}
		}

		Table[i * nb_cols + 1] =
				std::to_string(sz);

		Table[i * nb_cols + 2] =
				"\"" + Int_vec_stringify(index_set, sz) + "\"";
	}

	std::string Col_headings[3];

	Col_headings[0] = "Idx";
	Col_headings[1] = "Sz";
	Col_headings[2] = "Vector";


	//other::orbiter_kernel_system::file_io Fio;


	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);

	delete [] Table;


	if (f_v) {
		cout << "lint_matrix::write_index_set_csv Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}

}





}}}}

