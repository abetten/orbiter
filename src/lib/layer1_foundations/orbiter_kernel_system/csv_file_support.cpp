/*
 * csv_file_support.cpp
 *
 *  Created on: Jul 18, 2023
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace orbiter_kernel_system {


csv_file_support::csv_file_support()
{
	Fio = NULL;
}

csv_file_support::~csv_file_support()
{

}

void csv_file_support::init(
		file_io *Fio)
{
	csv_file_support::Fio = Fio;
}

void csv_file_support::int_vec_write_csv(
		int *v, int len,
	std::string &fname, std::string &label)
{
	int i;

	{
		ofstream f(fname);

		f << "Case," << label << endl;
		for (i = 0; i < len; i++) {
			f << i << "," << v[i] << endl;
		}
		f << "END" << endl;
	}
}

void csv_file_support::lint_vec_write_csv(
		long int *v, int len,
		std::string &fname, std::string &label)
{
	int i;

	{
		ofstream f(fname);

		f << "Case," << label << endl;
		for (i = 0; i < len; i++) {
			f << i << "," << v[i] << endl;
		}
		f << "END" << endl;
	}
}

void csv_file_support::int_vecs_write_csv(
		int *v1, int *v2, int len,
		std::string &fname, std::string &label1, std::string &label2)
{
	int i;

	{
		ofstream f(fname);

		f << "Case," << label1 << "," << label2 << endl;
		for (i = 0; i < len; i++) {
			f << i << "," << v1[i] << "," << v2[i] << endl;
		}
		f << "END" << endl;
	}
}

void csv_file_support::int_vecs3_write_csv(
		int *v1, int *v2, int *v3, int len,
		std::string &fname,
	std::string &label1, std::string &label2, std::string &label3)
{
	int i;

	{
		ofstream f(fname);

		f << "Case," << label1 << "," << label2 << "," << label3 << endl;
		for (i = 0; i < len; i++) {
			f << i << "," << v1[i] << "," << v2[i] << "," << v3[i] << endl;
		}
		f << "END" << endl;
	}
}

void csv_file_support::int_vec_array_write_csv(
		int nb_vecs, int **Vec, int len,
		std::string &fname, std::string *column_label)
{
	int i, j;

	cout << "csv_file_support::int_vec_array_write_csv nb_vecs=" << nb_vecs << endl;
	cout << "column labels:" << endl;
	for (j = 0; j < nb_vecs; j++) {
		cout << j << " : " << column_label[j] << endl;
		}

	{
		ofstream f(fname);

		f << "Row";
		for (j = 0; j < nb_vecs; j++) {
			f << "," << column_label[j];
		}
		f << endl;
		for (i = 0; i < len; i++) {
			f << i;
			for (j = 0; j < nb_vecs; j++) {
				f << "," << Vec[j][i];
			}
			f << endl;
		}
		f << "END" << endl;
	}
}

void csv_file_support::lint_vec_array_write_csv(
		int nb_vecs, long int **Vec, int len,
		std::string &fname, std::string *column_label)
{
	int i, j;

	cout << "csv_file_support::lint_vec_array_write_csv nb_vecs=" << nb_vecs << endl;
	cout << "column labels:" << endl;
	for (j = 0; j < nb_vecs; j++) {
		cout << j << " : " << column_label[j] << endl;
	}

	{
		ofstream f(fname);

		f << "Row";
		for (j = 0; j < nb_vecs; j++) {
			f << "," << column_label[j];
		}
		f << endl;
		for (i = 0; i < len; i++) {
			f << i;
			for (j = 0; j < nb_vecs; j++) {
				f << "," << Vec[j][i];
			}
			f << endl;
		}
		f << "END" << endl;
	}
}

void csv_file_support::int_matrix_write_csv(
		std::string &fname, int *M, int m, int n)
{
	int i, j;

	{
		ofstream f(fname);

		f << "Row";
		for (j = 0; j < n; j++) {
			f << ",C" << j;
		}
		f << endl;
		for (i = 0; i < m; i++) {
			f << i;
			for (j = 0; j < n; j++) {
				f << "," << M[i * n + j];
			}
			f << endl;
		}
		f << "END" << endl;
	}
}

void csv_file_support::lint_matrix_write_csv(
		std::string &fname,
		long int *M, int m, int n)
{
	int i, j;

	{
		ofstream f(fname);

		f << "Row";
		for (j = 0; j < n; j++) {
			f << ",C" << j;
		}
		f << endl;
		for (i = 0; i < m; i++) {
			f << i;
			for (j = 0; j < n; j++) {
				f << "," << M[i * n + j];
			}
			f << endl;
		}
		f << "END" << endl;
	}
}

void csv_file_support::lint_matrix_write_csv_override_headers(
		std::string &fname,
		std::string *headers, long int *M, int m, int n)
{
	int i, j;

	{
		ofstream f(fname);

		f << "Row";
		for (j = 0; j < n; j++) {
			f << "," << headers[j];
		}
		f << endl;
		for (i = 0; i < m; i++) {
			f << i;
			for (j = 0; j < n; j++) {
				f << "," << M[i * n + j];
			}
			f << endl;
		}
		f << "END" << endl;
	}
}

void csv_file_support::vector_write_csv(
		std::string &fname,
		std::vector<int > &V)
{
	int i, j;
	int m, n;

	m = V.size();
	n = 1;


	{
		ofstream f(fname);

		f << "Row";
		for (j = 0; j < n; j++) {
			f << ",C" << j;
		}
		f << endl;
		for (i = 0; i < m; i++) {
			f << i;
			f << "," << V[i];
			f << endl;
		}
		f << "END" << endl;
	}
}

void csv_file_support::vector_lint_write_csv(
		std::string &fname,
		std::vector<long int > &V)
{
	int i, j;
	int m, n;

	m = V.size();
	n = 1;


	{
		ofstream f(fname);

		f << "Row";
		for (j = 0; j < n; j++) {
			f << ",C" << j;
		}
		f << endl;
		for (i = 0; i < m; i++) {
			f << i;
			f << "," << V[i];
			f << endl;
		}
		f << "END" << endl;
	}
}


void csv_file_support::vector_matrix_write_csv(
		std::string &fname,
		std::vector<std::vector<int> > &V)
{
	int i, j;
	int m, n;

	m = V.size();
	n = V[0].size();
	for (i = 0; i < m; i++) {
		if (V[i].size() != n) {
			cout << "csv_file_support::int_matrix_write_csv "
					"the vectors must have the same length" << endl;
			exit(1);
		}
	}


	{
		ofstream f(fname);

		f << "Row";
		for (j = 0; j < n; j++) {
			f << ",C" << j;
		}
		f << endl;
		for (i = 0; i < m; i++) {
			f << i;
			for (j = 0; j < n; j++) {
				f << "," << V[i][j];
			}
			f << endl;
		}
		f << "END" << endl;
	}
}

void csv_file_support::vector_matrix_write_csv_compact(
		std::string &fname,
		std::vector<std::vector<int> > &V)
{
	int i, j;
	int m, n;

	m = V.size();
	n = V[0].size();
	for (i = 0; i < m; i++) {
		if (V[i].size() != n) {
			cout << "csv_file_support::int_matrix_write_csv "
					"the vectors must have the same length" << endl;
			exit(1);
		}
	}

	int *T;

	T = NEW_int(m * n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			T[i * n + j] = V[i][j];
		}
	}


	{
		ofstream f(fname);

		f << "Row,C0" << endl;
		for (i = 0; i < m; i++) {
			f << i;
			string str;

			str = Int_vec_stringify(T + i * n, n);
			f << ",\"" << str << "\"" << endl;
		}
		f << "END" << endl;
	}

	FREE_int(T);
}



void csv_file_support::double_matrix_write_csv(
		std::string &fname, double *M, int m, int n)
{
	int i, j;

	{
		ofstream f(fname);

		f << "Row";
		for (j = 0; j < n; j++) {
			f << ",C" << j;
		}
		f << endl;
		for (i = 0; i < m; i++) {
			f << i;
			for (j = 0; j < n; j++) {
				f << "," << M[i * n + j];
			}
			f << endl;
		}
		f << "END" << endl;
	}
}

void csv_file_support::int_matrix_write_csv_with_labels(
		std::string &fname,
	int *M, int m, int n, const char **column_label)
{
	int i, j;

	{
		ofstream f(fname);

		f << "Row";
		for (j = 0; j < n; j++) {
			f << "," << column_label[j];
		}
		f << endl;
		for (i = 0; i < m; i++) {
			f << i;
			for (j = 0; j < n; j++) {
				f << "," << M[i * n + j];
			}
			f << endl;
		}
		f << "END" << endl;
	}
}

void csv_file_support::lint_matrix_write_csv_with_labels(
		std::string &fname,
	long int *M, int m, int n, const char **column_label)
{
	int i, j;

	{
		ofstream f(fname);

		f << "Row";
		for (j = 0; j < n; j++) {
			f << "," << column_label[j];
		}
		f << endl;
		for (i = 0; i < m; i++) {
			f << i;
			for (j = 0; j < n; j++) {
				f << "," << M[i * n + j];
			}
			f << endl;
		}
		f << "END" << endl;
	}
}

void csv_file_support::int_matrix_read_csv(
		std::string &fname,
	int *&M, int &m, int &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int a;

	if (f_v) {
		cout << "csv_file_support::int_matrix_read_csv "
				"reading file " << fname << endl;
	}
	if (Fio->file_size(fname) <= 0) {
		cout << "csv_file_support::int_matrix_read_csv file " << fname
			<< " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << Fio->file_size(fname) << endl;
		exit(1);
	}
	{
		data_structures::spreadsheet S;

		if (f_v) {
			cout << "csv_file_support::int_matrix_read_csv "
					"before S.read_spreadsheet" << endl;
		}
		S.read_spreadsheet(fname, 0 /* verbose_level - 1*/);
		if (f_v) {
			cout << "csv_file_support::int_matrix_read_csv "
					"after S.read_spreadsheet" << endl;
		}

		m = S.nb_rows - 1;
		n = S.nb_cols - 1;
		if (f_v) {
			cout << "The spreadsheet has " << S.nb_cols << " columns" << endl;
		}
		M = NEW_int(m * n);
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {

				a = S.get_lint(i + 1, j + 1);
				M[i * n + j] = a;
			}
		}
	}
	if (f_v) {
		cout << "csv_file_support::int_matrix_read_csv done" << endl;
	}
}

void csv_file_support::read_column_as_table_of_int(
		std::string &fname, std::string &col_label,
	int *&M, int &m, int &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::read_column_as_table_of_int "
				"reading file " << fname << endl;
	}
	if (Fio->file_size(fname) <= 0) {
		cout << "csv_file_support::read_column_as_table_of_int file " << fname
			<< " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << Fio->file_size(fname) << endl;
		exit(1);
	}

	orbiter_kernel_system::file_io Fio;
	data_structures::set_of_sets *SoS;


	if (f_v) {
		cout << "csv_file_support::read_column_as_table_of_int "
				"reading file " << fname << ", column " << col_label << endl;
	}

	Fio.Csv_file_support->read_column_and_parse(
			fname, col_label,
				SoS,
				verbose_level);

	if (SoS->nb_sets == 0) {
		cout << "csv_file_support::read_column_as_table_of_int the file seems to be empty" << endl;
		exit(1);
	}

	m = SoS->nb_sets;
	n = SoS->Set_size[0];
	M = NEW_int(m * n);

	int i;

	for (i = 0; i < m; i++) {
		Lint_vec_copy_to_int(SoS->Sets[i], M + i * n, n);
	}

	FREE_OBJECT(SoS);

	if (f_v) {
		cout << "csv_file_support::read_column_as_table_of_int done" << endl;
	}
}



void csv_file_support::read_column_as_table_of_lint(
		std::string &fname, std::string &col_label,
	long int *&M, int &m, int &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::read_column_as_table_of_lint "
				"reading file " << fname << endl;
	}
	if (Fio->file_size(fname) <= 0) {
		cout << "csv_file_support::read_column_as_table_of_lint file " << fname
			<< " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << Fio->file_size(fname) << endl;
		exit(1);
	}

	orbiter_kernel_system::file_io Fio;
	data_structures::set_of_sets *SoS;


	if (f_v) {
		cout << "csv_file_support::read_column_as_table_of_lint "
				"reading file " << fname << ", column " << col_label << endl;
	}

	Fio.Csv_file_support->read_column_and_parse(
			fname, col_label,
				SoS,
				verbose_level);

	if (SoS->nb_sets == 0) {
		cout << "csv_file_support::read_column_as_table_of_lint the file seems to be empty" << endl;
		exit(1);
	}

	m = SoS->nb_sets;
	n = SoS->Set_size[0];
	M = NEW_lint(m * n);

	int i;

	for (i = 0; i < m; i++) {
		Lint_vec_copy(SoS->Sets[i], M + i * n, n);
	}

	FREE_OBJECT(SoS);

	if (f_v) {
		cout << "csv_file_support::read_column_as_table_of_lint done" << endl;
	}
}


void csv_file_support::int_matrix_read_csv_no_border(
		std::string &fname,
	int *&M, int &m, int &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a;

	if (f_v) {
		cout << "csv_file_support::int_matrix_read_csv_no_border "
				"reading file " << fname << endl;
	}
	if (Fio->file_size(fname) <= 0) {
		cout << "csv_file_support::int_matrix_read_csv_no_border "
				"file " << fname
			<< " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << Fio->file_size(fname) << endl;
		exit(1);
	}
	{
		data_structures::spreadsheet S;

		if (f_v) {
			cout << "csv_file_support::int_matrix_read_csv_no_border "
					"before S.read_spreadsheet" << endl;
		}
		S.read_spreadsheet(fname, verbose_level - 1);
		if (f_v) {
			cout << "csv_file_support::int_matrix_read_csv_no_border "
					"after S.read_spreadsheet" << endl;
		}

		m = S.nb_rows;
		n = S.nb_cols;
		if (f_v) {
			cout << "The spreadsheet has " << S.nb_cols << " columns" << endl;
		}
		M = NEW_int(m * n);
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				a = S.get_lint(i, j);
				M[i * n + j] = a;
			}
		}
	}
	if (f_v) {
		cout << "csv_file_support::int_matrix_read_csv_no_border done" << endl;
	}
}

void csv_file_support::lint_matrix_read_csv_no_border(
		std::string &fname,
	long int *&M, int &m, int &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "csv_file_support::lint_matrix_read_csv_no_border "
				"reading file " << fname << endl;
	}
	if (Fio->file_size(fname) <= 0) {
		cout << "csv_file_support::int_matrix_read_csv_no_border "
				"file " << fname
			<< " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << Fio->file_size(fname) << endl;
		exit(1);
	}
	{
		data_structures::spreadsheet S;

		if (f_v) {
			cout << "csv_file_support::lint_matrix_read_csv_no_border "
					"before S.read_spreadsheet" << endl;
		}
		S.read_spreadsheet(fname, verbose_level - 1);
		if (f_v) {
			cout << "csv_file_support::lint_matrix_read_csv_no_border "
					"after S.read_spreadsheet" << endl;
		}

		m = S.nb_rows;
		n = S.nb_cols;
		if (f_v) {
			cout << "The spreadsheet has "
					<< S.nb_cols << " columns" << endl;
		}
		M = NEW_lint(m * n);

		long int a;

		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				a = S.get_lint(i, j);
				M[i * n + j] = a;
			}
		}
	}
	if (f_v) {
		cout << "csv_file_support::lint_matrix_read_csv_no_border done" << endl;
	}
}

void csv_file_support::int_matrix_read_csv_data_column(
		std::string &fname,
	int *&M, int &m, int &n, int col_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a;

	if (f_v) {
		cout << "csv_file_support::int_matrix_read_csv_data_column "
				"reading file " << fname << " column=" << col_idx << endl;
	}
	if (Fio->file_size(fname) <= 0) {
		cout << "csv_file_support::int_matrix_read_csv_data_column "
				"file " << fname
			<< " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << Fio->file_size(fname) << endl;
		exit(1);
	}
	{
		data_structures::spreadsheet S;

		if (f_v) {
			cout << "csv_file_support::int_matrix_read_csv_data_column "
					"before S.read_spreadsheet" << endl;
		}
		S.read_spreadsheet(fname, verbose_level - 1);
		if (f_v) {
			cout << "csv_file_support::int_matrix_read_csv_data_column "
					"after S.read_spreadsheet" << endl;
		}
		{
			int *v;
			int sz;
			string str, str2;
			data_structures::string_tools ST;

			S.get_string(str, 1, col_idx);

			cout << "csv_file_support::int_matrix_read_csv_data_column "
					"read " << str << endl;

			ST.drop_quotes(str, str2);
			Int_vec_scan(str2, v, sz);

			FREE_int(v);
			n = sz;
		}
		m = S.nb_rows - 1;
		if (f_v) {
			cout << "The spreadsheet has " << m << " rows" << endl;
			cout << "The spreadsheet has " << S.nb_cols << " columns" << endl;
		}
		M = NEW_int(m * n);
		for (i = 0; i < m; i++) {
			string str, str2;
			int *v;
			int sz;
			data_structures::string_tools ST;

			S.get_string(str, i + 1, col_idx);
			cout << "csv_file_support::int_matrix_read_csv_data_column "
					"row " << i << " read " << str << endl;
			ST.drop_quotes(str, str2);
			Int_vec_scan(str2, v, sz);

			if (sz != n) {
				cout << "sz != n" << endl;
				cout << "sz=" << sz << endl;
				cout << "n=" << n << endl;
				exit(1);
			}
			for (j = 0; j < sz; j++) {
				a = v[j];
				M[i * n + j] = a;
			}
			FREE_int(v);
		}
	}
	if (f_v) {
		cout << "csv_file_support::int_matrix_read_csv_data_column done" << endl;
	}
}


void csv_file_support::lint_matrix_read_csv_data_column(
		std::string &fname,
	long int *&M, int &m, int &n, int col_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "csv_file_support::lint_matrix_read_csv_data_column "
				"reading file " << fname << " column=" << col_idx << endl;
	}
	if (Fio->file_size(fname) <= 0) {
		cout << "csv_file_support::lint_matrix_read_csv_data_column "
				"file " << fname
			<< " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << Fio->file_size(fname) << endl;
		exit(1);
	}
	{
		data_structures::spreadsheet S;

		if (f_v) {
			cout << "csv_file_support::lint_matrix_read_csv_data_column "
					"before S.read_spreadsheet" << endl;
		}
		S.read_spreadsheet(fname, verbose_level - 1);
		if (f_v) {
			cout << "csv_file_support::lint_matrix_read_csv_data_column "
					"after S.read_spreadsheet" << endl;
		}
		{
			long int *v;
			int sz;
			string str, str2;
			data_structures::string_tools ST;

			S.get_string(str, 1, col_idx);

			cout << "csv_file_support::lint_matrix_read_csv_data_column "
					"read " << str << endl;

			ST.drop_quotes(str, str2);
			Lint_vec_scan(str2, v, sz);

			FREE_lint(v);
			n = sz;
		}
		m = S.nb_rows - 1;
		if (f_v) {
			cout << "csv_file_support::lint_matrix_read_csv_data_column "
					"The spreadsheet has " << m << " rows" << endl;
			cout << "csv_file_support::lint_matrix_read_csv_data_column "
					"The spreadsheet has " << S.nb_cols << " columns" << endl;
		}
		M = NEW_lint(m * n);
		for (i = 0; i < m; i++) {
			string str, str2;
			long int *v;
			int sz;
			data_structures::string_tools ST;

			S.get_string(str, i + 1, col_idx);
			cout << "csv_file_support::lint_matrix_read_csv_data_column "
					"row " << i << " read " << str << endl;
			ST.drop_quotes(str, str2);
			Lint_vec_scan(str2, v, sz);

			long int a;
			if (sz != n) {
				cout << "sz != n" << endl;
				cout << "sz=" << sz << endl;
				cout << "n=" << n << endl;
				exit(1);
			}
			for (j = 0; j < sz; j++) {
				a = v[j];
				M[i * n + j] = a;
			}
			FREE_lint(v);
		}
	}
	if (f_v) {
		cout << "csv_file_support::lint_matrix_read_csv_data_column done" << endl;
	}
}


void csv_file_support::lint_matrix_read_csv(
		std::string &fname,
	long int *&M, int &m, int &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int a;

	if (f_v) {
		cout << "csv_file_support::lint_matrix_read_csv "
				"reading file " << fname << endl;
	}
	if (Fio->file_size(fname) <= 0) {
		cout << "csv_file_support::lint_matrix_read_csv file " << fname
			<< " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << Fio->file_size(fname) << endl;
		exit(1);
	}
	{
		data_structures::spreadsheet S;

		S.read_spreadsheet(fname, verbose_level - 1);

		m = S.nb_rows - 1;
		n = S.nb_cols - 1;
		M = NEW_lint(m * n);
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				a = S.get_lint(i + 1, j + 1);
				M[i * n + j] = a;
			}
		}
	}
	if (f_v) {
		cout << "csv_file_support::lint_matrix_read_csv done" << endl;
	}

}

void csv_file_support::double_matrix_read_csv(
		std::string &fname,
	double *&M, int &m, int &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "csv_file_support::double_matrix_read_csv reading file "
			<< fname << endl;
	}
	if (Fio->file_size(fname) <= 0) {
		cout << "csv_file_support::double_matrix_read_csv file " << fname
			<< " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << Fio->file_size(fname) << endl;
		exit(1);
	}
	{
		data_structures::spreadsheet S;
		double d;

		S.read_spreadsheet(fname, 0/*verbose_level - 1*/);

		m = S.nb_rows - 1;
		n = S.nb_cols - 1;
		M = new double [m * n];
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				d = S.get_double(i + 1, j + 1);
				M[i * n + j] = d;
			}
		}
	}
	if (f_v) {
		cout << "csv_file_support::double_matrix_read_csv done" << endl;
	}
}


void csv_file_support::do_csv_file_select_rows(
		std::string &fname,
		std::string &rows_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_rows" << endl;
	}

	do_csv_file_select_rows_worker(fname, rows_text, false, verbose_level);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_rows done" << endl;
	}
}

void csv_file_support::do_csv_file_select_rows_by_file(
		std::string &fname,
		std::string &rows_fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_rows_by_file" << endl;
	}

	int *M;
	int m, n;

	Fio->Csv_file_support->int_matrix_read_csv(
			rows_fname,
			M, m, n, verbose_level);

	int *select;
	int nb_select;
	int i;

	nb_select = m;
	select = NEW_int(nb_select);
	for (i = 0; i < nb_select; i++) {
		select[i] = M[i * n + 1];
	}


	data_structures::string_tools ST;
	string fname_out;

	fname_out = fname;
	ST.chop_off_extension(fname_out);
	fname_out += "_transversal.csv";


	data_structures::spreadsheet S;

	S.read_spreadsheet(fname, verbose_level);
	int nb_rows_in_file;

	nb_rows_in_file = S.nb_rows - 1;

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_rows_by_file "
				"nb_rows_in_file = " << nb_rows_in_file << endl;
	}


	{
		ofstream ost(fname_out);
		//ost << "Row,";
		S.print_table_row(0, false, ost);
		for (i = 0; i < nb_select; i++) {
			//ost << i << ",";
			S.print_table_row(select[i] + 1, false, ost);
		}
		ost << "END" << endl;
	}
	cout << "Written file " << fname_out
			<< " of size " << Fio->file_size(fname_out) << endl;

	//do_csv_file_select_rows_worker(fname, rows_text, false, verbose_level);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_rows_by_file done" << endl;
	}
}

void csv_file_support::do_csv_file_select_rows_complement(
		std::string &fname,
		std::string &rows_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_rows" << endl;
	}

	do_csv_file_select_rows_worker(fname, rows_text, true, verbose_level);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_rows done" << endl;
	}
}

void csv_file_support::do_csv_file_select_rows_worker(
		std::string &fname,
		std::string &rows_text,
		int f_complement,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_rows_worker" << endl;
	}

	data_structures::spreadsheet S;

	S.read_spreadsheet(fname, verbose_level);
	int nb_rows_in_file;

	nb_rows_in_file = S.nb_rows - 1;

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_rows_worker "
				"nb_rows_in_file = " << nb_rows_in_file << endl;
	}


	int *Rows0;
	int nb_rows0;
	int *Rows;
	int nb_rows;
	data_structures::string_tools ST;

	Int_vec_scan(rows_text, Rows0, nb_rows0);

	Rows = NEW_int(nb_rows_in_file);

	if (f_complement) {
		Int_vec_complement_to(Rows0, Rows, nb_rows_in_file, nb_rows0);
		nb_rows = nb_rows_in_file - nb_rows0;
	}
	else {
		Int_vec_copy(Rows0, Rows, nb_rows0);
		nb_rows = nb_rows0;
	}

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_rows_worker "
				"restricting to the rows:" << endl;
		Int_vec_print(cout, Rows, nb_rows);
		cout << endl;
	}

	int i;


#if 1
	string fname_out;

	fname_out = fname;
	ST.chop_off_extension(fname_out);
	fname_out += "_select.csv";

	{
		ofstream ost(fname_out);
		//ost << "Row,";
		S.print_table_row(0, false, ost);
		for (i = 0; i < nb_rows; i++) {
			//ost << i << ",";
			S.print_table_row(Rows[i] + 1, false, ost);
		}
		ost << "END" << endl;
	}
	cout << "Written file " << fname_out
			<< " of size " << Fio->file_size(fname_out) << endl;
#endif

	FREE_int(Rows0);
	FREE_int(Rows);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_rows_worker done" << endl;
	}
}



void csv_file_support::do_csv_file_split_rows_modulo(
		std::string &fname,
		int split_modulo,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_split_rows_modulo" << endl;
	}
	data_structures::string_tools ST;
	data_structures::spreadsheet S;

	S.read_spreadsheet(fname, verbose_level);


	int i, I;

	for (I = 0; I < split_modulo; I++) {

		string fname_out;

		fname_out = fname;
		ST.chop_off_extension(fname_out);
		fname_out += "_split_" + std::to_string(I)
				+ "_mod_" + std::to_string(split_modulo) + ".csv";

		{
			ofstream ost(fname_out);
			S.print_table_row(0, false, ost);
			for (i = 0; i < S.nb_rows - 1; i++) {
				if ((i % split_modulo) != I) {
					continue;
				}
				//ost << i << ",";
				S.print_table_row(i + 1, false, ost);
				}
			ost << "END" << endl;
		}
		cout << "Written file " << fname_out
				<< " of size " << Fio->file_size(fname_out) << endl;
	}

	if (f_v) {
		cout << "csv_file_support::do_csv_file_split_rows_modulo done" << endl;
	}
}

void csv_file_support::do_csv_file_select_cols(
		std::string &fname,
		std::string &fname_append,
		std::string &cols_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_cols" << endl;
	}
	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_cols fname = " << fname << endl;
		cout << "csv_file_support::do_csv_file_select_cols fname_append = " << fname_append << endl;
		cout << "csv_file_support::do_csv_file_select_cols cols_text = " << cols_text << endl;
	}
	long int *Cols;
	int nb_cols;

	//Int_vec_scan(cols_text, Cols, nb_cols);


	data_structures::vector_builder *vb;

	vb = Get_vector(cols_text);

	nb_cols = vb->len;
	Cols = vb->v;
	//long int *v;
	//int len;


	data_structures::spreadsheet S;

	S.read_spreadsheet(fname, verbose_level);


	int nb_rows;

	nb_rows = S.nb_rows;
	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_cols "
				"nb_rows=" << nb_rows << endl;
		cout << "csv_file_support::do_csv_file_select_cols "
				"nb_cols=" << nb_cols << endl;
		cout << "csv_file_support::do_csv_file_select_cols "
				"Cols=";
		Lint_vec_print(cout, Cols, nb_cols);
		cout << endl;
	}


	std::string *Header_rows;
	std::string *Header_cols;
	std::string *T;
	int nb_r, nb_c;

	S.stringify(
			Header_rows, Header_cols, T,
			nb_r, nb_c,
			verbose_level - 1);

	int i, j;

	if (false) {
		cout << "Header_cols" << endl;
		for (j = 0; j < nb_c; j++) {
			cout << j << " : " << Header_cols[j] << endl;
		}
		cout << "Header_rows" << endl;
		for (i = 0; i < nb_r; i++) {
			cout << i << " : " << Header_rows[i] << endl;
		}
		cout << "T" << endl;
		for (i = 0; i < nb_r; i++) {
			for (j = 0; j < nb_c; j++) {
				cout << i << "," << j << " : " << T[i * nb_c + j] << endl;

			}
		}
	}

	std::string *Header_cols2;
	std::string *T2;

	Header_cols2 = new string [nb_cols];
	for (j = 0; j < nb_cols; j++) {
		Header_cols2[j] = Header_cols[Cols[j]];
	}

	T2 = new string [nb_r * nb_cols];
	for (i = 0; i < nb_r; i++) {
		for (j = 0; j < nb_cols; j++) {
			T2[i * nb_cols + j] = T[i * nb_c + Cols[j]];

		}
	}


	data_structures::string_tools ST;
	string fname_out;

	fname_out = fname;
	ST.chop_off_extension(fname_out);
	fname_out += fname_append;
	fname_out += ".csv";


	string headings;


	for (j = 0; j < nb_cols; j++) {
		headings += Header_cols2[j];
		if (j < nb_cols - 1) {
			headings += ",";
		}
	}

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_cols "
				"before Fio.Csv_file_support->write_table_of_strings" << endl;
	}
	Fio->Csv_file_support->write_table_of_strings(
			fname_out,
			nb_r, nb_cols, T2,
			headings,
			verbose_level);

	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::export_csv "
				"after Fio.Csv_file_support->write_table_of_strings" << endl;
	}




	cout << "Written file " << fname_out
			<< " of size " << Fio->file_size(fname_out) << endl;


	//FREE_int(Cols);
	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_cols done" << endl;
	}
}



void csv_file_support::do_csv_file_select_cols_by_label(
		std::string &fname,
		std::string &fname_append,
		std::string &cols_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_cols_by_label" << endl;
	}
	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_cols_by_label fname = " << fname << endl;
		cout << "csv_file_support::do_csv_file_select_cols_by_label fname_append = " << fname_append << endl;
		cout << "csv_file_support::do_csv_file_select_cols_by_label cols_label = " << cols_label << endl;
	}


	data_structures::string_tools String;
	std::vector<std::string> Headings;

	String.parse_comma_separated_list(
			cols_label, Headings,
			verbose_level);
	if (f_v) {
		int j;
		cout << "Headings" << endl;
		for (j = 0; j < Headings.size(); j++) {
			cout << j << " : " << Headings[j] << endl;
		}
	}


	data_structures::spreadsheet S;

	S.read_spreadsheet(fname, verbose_level);


	int nb_rows;

	nb_rows = S.nb_rows;
	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_cols_by_label "
				"nb_rows=" << nb_rows << endl;
	}




	std::string *Header_rows;
	std::string *Header_cols;
	std::string *T;
	int nb_r, nb_c;

	S.stringify(
			Header_rows, Header_cols, T,
			nb_r, nb_c,
			verbose_level - 1);

	int i, j;

	if (false) {
		cout << "Header_cols" << endl;
		for (j = 0; j < nb_c; j++) {
			cout << j << " : " << Header_cols[j] << endl;
		}
		cout << "Header_rows" << endl;
		for (i = 0; i < nb_r; i++) {
			cout << i << " : " << Header_rows[i] << endl;
		}
		cout << "T" << endl;
		for (i = 0; i < nb_r; i++) {
			for (j = 0; j < nb_c; j++) {
				cout << i << "," << j << " : " << T[i * nb_c + j] << endl;

			}
		}
	}

	std::string *T2;
	int nb_selected_cols;
	int *Col_idx;

	nb_selected_cols = Headings.size();

	Col_idx = NEW_int(nb_selected_cols);
	for (i = 0; i < nb_selected_cols; i++) {
		Col_idx[i] = -1;
		for (j = 0; j < nb_c; j++) {
			if (String.compare_string_string(Header_cols[j], Headings[i]) == 0) {
				Col_idx[i] = j;
				break;
			}
		}
		if (Col_idx[i] == -1) {
			cout << "Cannot find column with label " << Headings[i] << endl;
			exit(1);
		}
	}



	T2 = new string [nb_r * nb_selected_cols];
	for (i = 0; i < nb_r; i++) {
		for (j = 0; j < nb_selected_cols; j++) {
			T2[i * nb_selected_cols + j] = T[i * nb_c + Col_idx[j]];

		}
	}


	data_structures::string_tools ST;
	string fname_out;

	fname_out = fname;
	ST.chop_off_extension(fname_out);
	fname_out += fname_append;
	fname_out += ".csv";


	string headings;


	for (j = 0; j < nb_selected_cols; j++) {
		headings += Headings[j];
		if (j < nb_selected_cols - 1) {
			headings += ",";
		}
	}

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_cols_by_label "
				"before Fio.Csv_file_support->write_table_of_strings" << endl;
	}
	Fio->Csv_file_support->write_table_of_strings(
			fname_out,
			nb_r, nb_selected_cols, T2,
			headings,
			verbose_level);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_cols_by_label "
				"after Fio.Csv_file_support->write_table_of_strings" << endl;
	}




	cout << "Written file " << fname_out
			<< " of size " << Fio->file_size(fname_out) << endl;

	delete [] T;
	delete [] T2;

	FREE_int(Col_idx);
	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_cols_by_label done" << endl;
	}
}




void csv_file_support::do_csv_file_select_rows_and_cols(
		std::string &fname,
		std::string &rows_text, std::string &cols_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_rows_and_cols" << endl;
	}
	int *Rows;
	int nb_rows;
	int *Cols;
	int nb_cols;
	data_structures::string_tools ST;

	Int_vec_scan(rows_text, Rows, nb_rows);
	cout << "Rows: ";
	Int_vec_print(cout, Rows, nb_rows);
	cout << endl;

	Int_vec_scan(cols_text, Cols, nb_cols);
	cout << "Cols: ";
	Int_vec_print(cout, Cols, nb_cols);
	cout << endl;

	data_structures::spreadsheet S;

	S.read_spreadsheet(fname, verbose_level);


	int i;



	string fname_out;

	fname_out = fname;
	ST.chop_off_extension(fname_out);
	fname_out += "_select.csv";

	{
		ofstream ost(fname_out);




		ost << "Row,";
		S.print_table_row_with_column_selection(
				0, false, Cols, nb_cols, ost, verbose_level);

		for (i = 0; i < nb_rows; i++) {


			S.print_table_row(Rows[i] + 1, false, cout);
			cout << endl;

			ost << i << ",";
			S.print_table_row_with_column_selection(Rows[i] + 1, false,
					Cols, nb_cols, ost, verbose_level);
			}
		ost << "END" << endl;
	}
	cout << "Written file " << fname_out
			<< " of size " << Fio->file_size(fname_out) << endl;


	FREE_int(Rows);
	FREE_int(Cols);
	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_rows done" << endl;
	}
}

void csv_file_support::do_csv_file_extract_column_to_txt(
		std::string &csv_fname, std::string &col_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_extract_column_to_txt" << endl;
	}
	string fname;
	data_structures::string_tools ST;

	data_structures::spreadsheet *S;
	int identifier_column;

	S = NEW_OBJECT(data_structures::spreadsheet);

	S->read_spreadsheet(csv_fname, 0 /*verbose_level*/);
	cout << "Table " << csv_fname << " has been read" << endl;


	identifier_column = S->find_column(col_label);


	fname = csv_fname;
	ST.replace_extension_with(fname, "_");
	fname += col_label + ".txt";




	{
		ofstream ost(fname);

		int i, j;

		for (i = 1; i < S->nb_rows; i++) {
			string entry;
			long int *v;
			int sz;

			S->get_string(entry, i, identifier_column);
			Lint_vec_scan(entry, v, sz);
			ost << sz;
			for (j = 0; j < sz; j++) {
				ost << " " << v[j];
			}
			ost << endl;
			FREE_lint(v);
		}
		ost << -1 << endl;
	}
	if (f_v) {
		cout << "Written file " << fname
				<< " of size " << Fio->file_size(fname) << endl;
	}

	if (f_v) {
		cout << "csv_file_support::do_csv_file_extract_column_to_txt done" << endl;
	}
}



void csv_file_support::do_csv_file_sort_each_row(
		std::string &csv_fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_sort_each_row" << endl;
	}
	int *M;
	int m, n;
	data_structures::sorting Sorting;
	int i;
	string fname;
	data_structures::string_tools ST;

	int_matrix_read_csv(csv_fname, M, m, n, verbose_level);
	for (i = 0; i < m; i++) {
		Sorting.int_vec_heapsort(M + i * n, n);
	}
	fname = csv_fname;
	ST.replace_extension_with(fname, "_sorted.csv");

	int_matrix_write_csv(fname, M, m, n);

	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio->file_size(fname) << endl;
	}

	if (f_v) {
		cout << "csv_file_support::do_csv_file_sort_each_row done" << endl;
	}
}

void csv_file_support::do_csv_file_join(
		std::vector<std::string> &csv_file_join_fname,
		std::vector<std::string> &csv_file_join_identifier,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_join" << endl;
	}

	int nb_files;
	int i;

	nb_files = csv_file_join_fname.size();

	data_structures::spreadsheet *S;
	int *identifier_column;

	S = new data_structures::spreadsheet[nb_files];
	identifier_column = NEW_int(nb_files);

	for (i = 0; i < nb_files; i++) {
		cout << "Reading table " << csv_file_join_fname[i] << endl;
		S[i].read_spreadsheet(csv_file_join_fname[i], 0 /*verbose_level*/);
		cout << "Table " << csv_file_join_fname[i] << " has been read" << endl;
#if 0
		if (i == 0) {
			cout << "The first table is:" << endl;
			S[0].print_table(cout, false);
		}
#endif
		if (false) {
			cout << "The " << i << "th table is:" << endl;
			S[i].print_table(cout, false);
		}


	}

#if 0
	cout << "adding " << nb_with << " -with entries" << endl;
	for (i = 0; i < nb_with; i++) {
		S[with_table[i]].add_column_with_constant_value(with_label[i], with_value[i]);
		}
#endif

	for (i = 0; i < nb_files; i++) {
		identifier_column[i] = S[i].find_column(csv_file_join_identifier[i]);
		cout << "Table " << csv_file_join_fname[i]
			<< ", identifier " << identifier_column[i] << endl;
	}

#if 0
	for (i = 0; i < nb_files; i++) {
		by_column[i] = S[i].find_by_column(join_by);
		cout << "File " << fname[i] << " by_column[" << i << "]=" << by_column[i] << endl;
	}
#endif

	cout << "joining " << nb_files << " files" << endl;
	for (i = 1; i < nb_files; i++) {
		cout << "Joining table " << 0 << " = " << csv_file_join_fname[0]
			<< " with table " << i << " = " << csv_file_join_fname[i] << endl;
		S[0].join_with(
				S + i,
				identifier_column[0], identifier_column[i],
				verbose_level - 2);
		cout << "joining " << csv_file_join_fname[0]
			<< " with table " << csv_file_join_fname[i] << " done" << endl;
#if 0
		cout << "After join, the table is:" << endl;
		S[0].print_table(cout, false);
#endif
	}






#if 0
	if (f_drop) {
		S[0].remove_rows(drop_column, drop_label, verbose_level);
	}
#endif


	string save_fname;
	data_structures::string_tools ST;

	save_fname = csv_file_join_fname[0];
	ST.chop_off_extension(save_fname);
	save_fname += "_joined.csv";

	{
		ofstream f(save_fname);
		S[0].print_table(f, false);
		f << "END" << endl;
	}
	cout << "Written file " << save_fname
			<< " of size " << Fio->file_size(save_fname) << endl;


	if (f_v) {
		cout << "csv_file_support::do_csv_file_join done" << endl;
	}
}

void csv_file_support::do_csv_file_concatenate(
		std::vector<std::string> &fname_in, std::string &fname_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_concatenate" << endl;
	}

	int nb_files;
	int i;

	nb_files = fname_in.size();

	data_structures::spreadsheet *S;
	//int *identifier_column;

	S = new data_structures::spreadsheet[nb_files];
	//identifier_column = NEW_int(nb_files);

	std::string *Header_cols;
	std::string **T;
	int *Nb_r;
	int *Nb_c;
	int nb_r = 0;
	int nb_c = 0;

	Nb_r = NEW_int(nb_files);
	Nb_c = NEW_int(nb_files);
	T = (std::string **) NEW_pvoid(nb_files);

	for (i = 0; i < nb_files; i++) {
		cout << "Reading table " << fname_in[i] << endl;
		S[i].read_spreadsheet(fname_in[i], 0 /*verbose_level*/);
		cout << "Table " << fname_in[i] << " has been read" << endl;

		if (false) {
			cout << "The " << i << "-th table is:" << endl;
			S[i].print_table(cout, false);
		}

		std::string *Header_rows1;
		std::string *Header_cols1;
		std::string *T1;

		S[i].stringify(
				Header_rows1, Header_cols1, T1,
				Nb_r[i], Nb_c[i],
				verbose_level);

		if (i == 0) {
			Header_cols = Header_cols1;
			nb_c = Nb_c[0];
		}
		else {
			delete [] Header_cols1;

			if (Nb_c[i] != Nb_c[0]) {
				cout << "The number of columns is not constant, cannot merge" << endl;
				exit(1);
			}
		}

		delete [] Header_rows1;

		T[i] = T1;
		nb_r += Nb_r[i];

	}

	std::string *Table;
	int r, j, h;


	Table = new std::string [nb_r * nb_c];
	r = 0;
	for (h = 0; h < nb_files; h++) {
		for (i = 0; i < Nb_r[h]; i++, r++) {
			for (j = 0; j < nb_c; j++) {
				Table[r * nb_c + j] = T[h][i * nb_c + j];
			}
		}
	}

	write_table_of_strings_with_col_headings(
			fname_out,
			nb_r, nb_c, Table,
			Header_cols,
			verbose_level);

#if 0
	{
		ofstream ost(fname_out);
		int j;
		int f_enclose_in_parentheses = false;

		S[0].print_table_row(
				0, f_enclose_in_parentheses, ost);
		for (i = 0; i < nb_files; i++) {
			//S[i].print_table(ost, false);
			for (j = 1; j < S[i].nb_rows; j++) {
				S[i].print_table_row(
						j, f_enclose_in_parentheses, ost);
			}
		}
		ost << "END" << endl;
	}
#endif
	cout << "Written file " << fname_out
			<< " of size " << Fio->file_size(fname_out) << endl;


	delete [] Table;
	for (h = 0; h < nb_files; h++) {
		delete [] T[h];
	}
	delete [] Header_cols;


	FREE_int(Nb_r);
	FREE_int(Nb_c);
	FREE_pvoid((void **) T);


	if (f_v) {
		cout << "csv_file_support::do_csv_file_concatenate done" << endl;
	}
}

void csv_file_support::do_csv_file_concatenate_from_mask(
		std::string &fname_in_mask, int N, std::string &fname_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_concatenate_from_mask" << endl;
	}

	int nb_files;
	long int i;

	nb_files = N;

	data_structures::spreadsheet *S;
	//int *identifier_column;

	S = new data_structures::spreadsheet[nb_files];
	//identifier_column = NEW_int(nb_files);

	data_structures::string_tools ST;



	for (i = 0; i < nb_files; i++) {


		std::string fname_in;

		fname_in = ST.printf_d(fname_in_mask, i);

		cout << "Reading table " << fname_in << endl;
		S[i].read_spreadsheet(fname_in, 0 /*verbose_level*/);
		cout << "Table " << fname_in << " has been read" << endl;

		if (false) {
			cout << "The " << i << "-th table is:" << endl;
			S[i].print_table(cout, false);
		}


	}

	{
		ofstream ost(fname_out);
		int j;
		int cnt = 0;
		int f_enclose_in_parentheses = false;

		ost << "Line,PO,";
		S[0].print_table_row(0, f_enclose_in_parentheses, ost);
		for (i = 0; i < nb_files; i++) {
			//S[i].print_table(ost, false);
			for (j = 1; j < S[i].nb_rows; j++) {
				ost << cnt << ",";
				ost << i << ",";
				S[i].print_table_row(j, f_enclose_in_parentheses, ost);
				cnt++;
			}
		}
		ost << "END" << endl;
	}
	cout << "Written file " << fname_out << " of size " << Fio->file_size(fname_out) << endl;


	if (f_v) {
		cout << "csv_file_support::do_csv_file_concatenate_from_mask done" << endl;
	}
}


void csv_file_support::do_csv_file_latex(
		std::string &fname,
		int f_produce_latex_header,
		int nb_lines_per_table,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::do_csv_file_latex" << endl;
	}

	data_structures::spreadsheet S;

	S.read_spreadsheet(fname, verbose_level);


	if (f_v) {
		cout << "csv_file_support::do_csv_file_latex "
				"S.nb_rows = " << S.nb_rows << endl;
		cout << "csv_file_support::do_csv_file_latex "
				"S.nb_cols = " << S.nb_cols << endl;
	}


	string author;
	string title;
	string extra_praeamble;


	title = "File";
	author = "Orbiter";




	string fname_out;
	data_structures::string_tools ST;

	fname_out = fname;
	ST.chop_off_extension(fname_out);
	fname_out += ".tex";

	{
		ofstream ost(fname_out);
		l1_interfaces::latex_interface L;

		//S.print_table_latex_all_columns(ost, false /* f_enclose_in_parentheses */);

		int *f_column_select;
		int j;

		f_column_select = NEW_int(S.nb_cols);
		for (j = 0; j < S.nb_cols; j++) {
			f_column_select[j] = true;
		}
		f_column_select[0] = false;


		if (f_produce_latex_header) {
			//L.head_easy(ost);
			L.head(ost,
				false /* f_book */,
				true /* f_title */,
				title, author,
				false /*f_toc */,
				false /* f_landscape */,
				false /* f_12pt */,
				false /* f_enlarged_page */,
				true /* f_pagenumbers */,
				extra_praeamble /* extras_for_preamble */);
		}

		S.print_table_latex(ost,
				f_column_select,
				false /* f_enclose_in_parentheses */,
				nb_lines_per_table);

		FREE_int(f_column_select);

		if (f_produce_latex_header) {
			L.foot(ost);
		}

	}
	cout << "Written file " << fname_out
			<< " of size " << Fio->file_size(fname_out) << endl;


	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_rows done" << endl;
	}
}

void csv_file_support::read_csv_file_and_tally(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::read_csv_file_and_tally" << endl;
	}

	long int *M;
	int m, n;

	lint_matrix_read_csv(fname, M, m, n, verbose_level);

	cout << "The matrix has size " << m << " x " << n << endl;

	data_structures::tally T;

	T.init_lint(M, m * n, true, 0);
	cout << "tally:" << endl;
	T.print(true);
	cout << endl;


	data_structures::set_of_sets *SoS;
	int *types;
	int nb_types;
	int i;

	SoS = T.get_set_partition_and_types(
			types, nb_types, verbose_level);

	cout << "fibers:" << endl;
	for (i = 0; i < nb_types; i++) {
		cout << i << " : " << types[i] << " : " << SoS->Set_size[i] << " : ";
		Lint_vec_print(cout, SoS->Sets[i], SoS->Set_size[i]);
		cout << endl;
	}

	//cout << "set partition:" << endl;
	//SoS->print_table();

	FREE_lint(M);

	if (f_v) {
		cout << "csv_file_support::read_csv_file_and_tally done" << endl;
	}
}




void csv_file_support::grade_statistic_from_csv(
		std::string &fname_csv,
		int f_midterm1, std::string &midterm1_label,
		int f_midterm2, std::string &midterm2_label,
		int f_final, std::string &final_label,
		int f_oracle_grade, std::string &oracle_grade_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::grade_statistic_from_csv" << endl;
	}

	data_structures::spreadsheet S;

	S.read_spreadsheet(fname_csv, verbose_level);


	if (f_v) {
		cout << "csv_file_support::grade_statistic_from_csv "
				"S.nb_rows = " << S.nb_rows << endl;
		cout << "csv_file_support::grade_statistic_from_csv "
				"S.nb_cols = " << S.nb_cols << endl;
	}

	int m1_idx, m2_idx, f_idx, o_idx;

	m1_idx = S.find_column(midterm1_label);
	m2_idx = S.find_column(midterm2_label);
	f_idx = S.find_column(final_label);
	o_idx = S.find_column(oracle_grade_label);

	int *M1_score;
	int *M2_score;
	int *F_score;
	std::string *O_grade;
	int i, j, a;

	M1_score = NEW_int(S.nb_rows);
	M2_score = NEW_int(S.nb_rows);
	F_score = NEW_int(S.nb_rows);
	O_grade = new string[S.nb_rows];

	for (i = 0; i < S.nb_rows - 1; i++) {
		M1_score[i] = S.get_lint(i + 1, m1_idx);
		M2_score[i] = S.get_lint(i + 1, m2_idx);
		F_score[i] = S.get_lint(i + 1, f_idx);
		S.get_string(O_grade[i], i + 1, o_idx);
	}

	int m1_count_dec[10];
	int m2_count_dec[10];
	int f_count_dec[10];

	Int_vec_zero(m1_count_dec, 10);
	Int_vec_zero(m2_count_dec, 10);
	Int_vec_zero(f_count_dec, 10);

	for (i = 0; i < S.nb_rows - 1; i++) {
		a = M1_score[i];
		j = a / 10;
		if (j >= 10) {
			j = 0;
		}
		if (j < 0) {
			j = 0;
		}
		m1_count_dec[j]++;
	}

	for (i = 0; i < S.nb_rows - 1; i++) {
		a = M2_score[i];
		j = a / 10;
		if (j >= 10) {
			j = 9;
		}
		if (j < 0) {
			j = 0;
		}
		m2_count_dec[j]++;
	}

	for (i = 0; i < S.nb_rows - 1; i++) {
		a = F_score[i];
		j = a / 10;
		if (j >= 10) {
			j = 9;
		}
		if (j < 0) {
			j = 0;
		}
		f_count_dec[j]++;
	}

	int *T;

	T = NEW_int(10 * 3);
	for (i = 0; i < 10; i++) {
		T[i * 3 + 0] = m1_count_dec[i];
		T[i * 3 + 1] = m2_count_dec[i];
		T[i * 3 + 2] = f_count_dec[i];
	}

	string fname_summary;

	data_structures::string_tools ST;


	fname_summary = fname_csv;
	ST.chop_off_extension(fname_summary);
	fname_summary += "_summary.csv";

	int_matrix_write_csv(fname_summary, T, 10, 3);

	if (f_v) {
		cout << "Written file " << fname_summary
				<< " of size " << Fio->file_size(fname_summary) << endl;
	}




	if (f_v) {
		cout << "csv_file_support::grade_statistic_from_csv done" << endl;
	}
}


void csv_file_support::csv_file_sort_rows(
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::csv_file_sort_rows "
				"fname = "<< fname << endl;
	}

	int *M;
	int m, n;


	int_matrix_read_csv(
				fname, M,
			m, n, verbose_level);

	data_structures::int_matrix *I;

	I = NEW_OBJECT(data_structures::int_matrix);
	I->allocate(m, n);

	int i, j, a;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = M[i * n + j];
			I->M[i * n + j] = a;
		}
	}


	I->sort_rows(verbose_level);

	string fname2;
	data_structures::string_tools ST;

	fname2 = fname;

	ST.chop_off_extension_and_path(fname2);

	fname2 += "_sorted.csv";

	int_matrix_write_csv(fname2, I->M, m, n);

	cout << "written file "
		<< fname << " of size " << Fio->file_size(fname) << endl;


	if (f_v) {
		cout << "csv_file_support::csv_file_sort_rows done" << endl;
	}
}

void csv_file_support::csv_file_sort_rows_and_remove_duplicates(
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::csv_file_sort_rows_and_remove_duplicates "
				"fname = "<< fname << endl;
	}

	int *M;
	int m, n;


	int_matrix_read_csv(
				fname, M,
			m, n, verbose_level);

	data_structures::int_matrix *I;

	I = NEW_OBJECT(data_structures::int_matrix);
	I->allocate(m, n);

	int i, j, a;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = M[i * n + j];
			I->M[i * n + j] = a;
		}
	}


	I->sort_rows(verbose_level);

	I->remove_duplicates(verbose_level);

	string fname2;
	data_structures::string_tools ST;

	fname2 = fname;

	ST.chop_off_extension_and_path(fname2);

	fname2 += "_sorted.csv";

	int_matrix_write_csv(fname2, I->M, I->m, n);

	cout << "written file "
		<< fname << " of size " << Fio->file_size(fname) << endl;


	if (f_v) {
		cout << "csv_file_support::csv_file_sort_rows_and_remove_duplicates done" << endl;
	}
}

void csv_file_support::write_table_of_strings(
		std::string &fname,
		int nb_rows, int nb_cols, std::string *Table,
		std::string &headings,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::write_table_of_strings "
				"fname = "<< fname << endl;
	}


	{
		ofstream f(fname);
		int i, j;

		f << "Row," << headings;
		f << endl;
		for (i = 0; i < nb_rows; i++) {
			f << i;

			for (j = 0; j < nb_cols; j++) {
				f << "," << Table[i * nb_cols + j];
			}
			f << endl;
		}
		f << "END" << endl;
	}


	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio->file_size(fname) << endl;
	}

	if (f_v) {
		cout << "csv_file_support::write_table_of_strings done" << endl;
	}
}

void csv_file_support::write_table_of_strings_with_headings(
		std::string &fname,
		int nb_rows, int nb_cols, std::string *Table,
		std::string *Row_headings,
		std::string *Col_headings,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::write_table_of_strings_with_headings "
				"fname = "<< fname << endl;
	}


	{
		ofstream f(fname);
		int i, j;

		f << "Row";
		for (j = 0; j < nb_cols; j++) {
			f << "," << Col_headings[j];
		}
		f << endl;


		for (i = 0; i < nb_rows; i++) {
			f << i << "," << Row_headings[i];

			for (j = 0; j < nb_cols; j++) {
				f << "," << Table[i * nb_cols + j];
			}
			f << endl;
		}
		f << "END" << endl;
	}


	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio->file_size(fname) << endl;
	}

	if (f_v) {
		cout << "csv_file_support::write_table_of_strings_with_headings done" << endl;
	}
}

void csv_file_support::write_table_of_strings_with_col_headings(
		std::string &fname,
		int nb_rows, int nb_cols, std::string *Table,
		std::string *Col_headings,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::write_table_of_strings_with_col_headings "
				"fname = "<< fname << endl;
	}


	{
		ofstream f(fname);
		int i, j;

		f << "Row";
		for (j = 0; j < nb_cols; j++) {
			f << "," << Col_headings[j];
		}
		f << endl;


		for (i = 0; i < nb_rows; i++) {
			f << i;

			for (j = 0; j < nb_cols; j++) {
				f << "," << Table[i * nb_cols + j];
			}
			f << endl;
		}
		f << "END" << endl;
	}


	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio->file_size(fname) << endl;
	}

	if (f_v) {
		cout << "csv_file_support::write_table_of_strings_with_col_headings done" << endl;
	}
}

int csv_file_support::read_column_and_count_nb_sets(
		std::string &fname, std::string &col_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::count_nb_sets "
				"reading file " << fname << endl;
	}
	if (Fio->file_size(fname) <= 0) {
		cout << "csv_file_support::count_nb_sets file " << fname
			<< " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << Fio->file_size(fname) << endl;
		exit(1);
	}
	int nb_sets;
	{
		data_structures::spreadsheet S;
		int idx;

		S.read_spreadsheet(fname, 0/*verbose_level - 1*/);

		idx = S.find_column(col_label);
		if (idx == -1) {
			cout << "csv_file_support::count_nb_sets "
					"cannot find column " << col_label << endl;
			exit(1);
		}
		nb_sets = S.nb_rows - 1;

	}

	return nb_sets;
}


void csv_file_support::read_column_and_parse(
		std::string &fname, std::string &col_label,
		data_structures::set_of_sets *&SoS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::read_column_and_parse "
				"reading file " << fname << endl;
	}
	if (Fio->file_size(fname) <= 0) {
		cout << "csv_file_support::read_column_and_parse file " << fname
			<< " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << Fio->file_size(fname) << endl;
		exit(1);
	}
	{
		data_structures::spreadsheet S;
		data_structures::string_tools ST;
		int idx;
		int nb_sets;
		int i;

		S.read_spreadsheet(fname, 0/*verbose_level - 1*/);

		idx = S.find_column(col_label);
		if (idx == -1) {
			cout << "csv_file_support::read_column_and_parse "
					"cannot find column " << col_label << endl;
			exit(1);
		}
		nb_sets = S.nb_rows - 1;

		int underlying_set_size = INT_MAX;

		SoS = NEW_OBJECT(data_structures::set_of_sets);

		SoS->init_simple(underlying_set_size,
				nb_sets, verbose_level);

		for (i = 0; i < nb_sets; i++) {

			if (f_vv) {
				cout << "csv_file_support::read_column_and_parse "
						"i= " << i << " / " << nb_sets << endl;
			}


			string str1, str2;
			long int *set;
			int sz;

			S.get_string(str1, i + 1, idx);

			if (f_vv) {
				cout << "csv_file_support::read_column_and_parse "
						"str1 = " << str1 << endl;
			}

			ST.drop_quotes(
				str1, str2);

			if (f_vv) {
				cout << "csv_file_support::read_column_and_parse "
						"str2 = " << str2 << endl;
			}

			Lint_vec_scan(str2, set, sz);

			if (f_vv) {
				cout << "csv_file_support::read_column_and_parse "
						"str = ";
				Lint_vec_print(cout, set, sz);
				cout << endl;
			}
			SoS->Sets[i] = set;
			SoS->Set_size[i] = sz;

		}
	}
	if (f_v) {
		cout << "csv_file_support::read_column_and_parse done" << endl;
	}

}

#if 0
void csv_file_support::save_fibration(
		std::vector<std::vector<std::pair<int, int> > > &Fibration,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::save_fibration" << endl;
	}

	data_structures::string_tools ST;
	string data_fname1;
	string data_fname2;
	data_structures::set_of_sets *File_idx;
	data_structures::set_of_sets *Obj_idx;
	int nb_sets;
	int *Sz;
	int i, j, l, a, b;

	nb_sets = Fibration.size();
	Sz = NEW_int(nb_sets);
	for (i = 0; i < nb_sets; i++) {
		Sz[i] = Fibration[i].size();
	}

	File_idx = NEW_OBJECT(data_structures::set_of_sets);
	Obj_idx = NEW_OBJECT(data_structures::set_of_sets);

	File_idx->init_basic_with_Sz_in_int(
			INT_MAX /* underlying_set_size */,
				nb_sets, Sz, verbose_level);

	Obj_idx->init_basic_with_Sz_in_int(
			INT_MAX /* underlying_set_size */,
				nb_sets, Sz, verbose_level);

	for (i = 0; i < nb_sets; i++) {
		l = Fibration[i].size();
		for (j = 0; j < l; j++) {
			a = Fibration[i][j].first;
			b = Fibration[i][j].second;
			File_idx->Sets[i][j] = a;
			Obj_idx->Sets[i][j] = b;
		}
	}


	data_fname1.assign(fname);
	ST.replace_extension_with(data_fname1, "1.csv");

	data_fname2.assign(fname);
	ST.replace_extension_with(data_fname2, "2.csv");

	if (f_v) {
		cout << "csv_file_support::save_fibration "
				"before File_idx->save_csv" << endl;
	}
	File_idx->save_csv(data_fname1, true, verbose_level);

	if (f_v) {
		cout << "csv_file_support::save_fibration "
				"before Obj_idx->save_csv" << endl;
	}
	Obj_idx->save_csv(data_fname2, true, verbose_level);
	if (f_v) {
		cout << "csv_file_support::save_fibration "
				"after Obj_idx->save_csv" << endl;
	}


	if (f_v) {
		cout << "Written file " << data_fname1
				<< " of size " << Fio->file_size(data_fname1) << endl;
		cout << "Written file " << data_fname2
				<< " of size " << Fio->file_size(data_fname2) << endl;
	}

	FREE_int(Sz);
	FREE_OBJECT(File_idx);
	FREE_OBJECT(Obj_idx);

	if (f_v) {
		cout << "csv_file_support::save_fibration done" << endl;
	}
}


void csv_file_support::save_cumulative_canonical_labeling(
		std::vector<std::vector<int> > &Cumulative_canonical_labeling,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::save_cumulative_canonical_labeling" << endl;
	}
	//string_tools ST;
	string canonical_labeling_fname;
	int canonical_labeling_len;
	int u, v;
	long int *M;

	if (Cumulative_canonical_labeling.size()) {
		canonical_labeling_len = Cumulative_canonical_labeling[0].size();
	}
	else {
		canonical_labeling_len = 0;
	}

	canonical_labeling_fname.assign(fname);
	//ST.replace_extension_with(canonical_labeling_fname, "_can_lab.csv");


	M = NEW_lint(Cumulative_canonical_labeling.size() * canonical_labeling_len);
	for (u = 0; u < Cumulative_canonical_labeling.size(); u++) {
		for (v = 0; v < canonical_labeling_len; v++) {
			M[u * canonical_labeling_len + v] = Cumulative_canonical_labeling[u][v];
		}
	}
	lint_matrix_write_csv(
			canonical_labeling_fname,
			M, Cumulative_canonical_labeling.size(), canonical_labeling_len);

	if (f_v) {
		cout << "Written file " << canonical_labeling_fname
				<< " of size " << Fio->file_size(canonical_labeling_fname) << endl;
	}
	FREE_lint(M);

}

void csv_file_support::save_cumulative_ago(
		std::vector<long int> &Cumulative_Ago,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::save_cumulative_ago" << endl;
	}
	string ago_fname;
	int u;
	long int *M;
	data_structures::string_tools ST;

	ago_fname.assign(fname);

	M = NEW_lint(Cumulative_Ago.size());
	for (u = 0; u < Cumulative_Ago.size(); u++) {
		M[u] = Cumulative_Ago[u];
	}

	string label;

	label.assign("Ago");

	lint_vec_write_csv(
			M, Cumulative_Ago.size(), ago_fname, label);

	data_structures::tally T;

	T.init_lint(M, Cumulative_Ago.size(), false, 0);
	if (f_v) {
		cout << "Written file " << ago_fname
				<< " of size " << Fio->file_size(ago_fname) << endl;
	}

	if (f_v) {
		cout << "Ago distribution: ";
		T.print(true);
		cout << endl;
	}

	string ago_fname1;

	ago_fname1.assign(ago_fname);
	ST.replace_extension_with(ago_fname1, "_ago_class_");
	T.save_classes_individually(ago_fname1);

	FREE_lint(M);

}

void csv_file_support::save_cumulative_data(
		std::vector<std::vector<int> > &Cumulative_data,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::save_cumulative_data" << endl;
	}

	string data_fname;
	int data_len;
	int u, v;
	long int *M;

	if (Cumulative_data.size()) {
		data_len = Cumulative_data[0].size();
	}
	else {
		data_len = 0;
	}

	data_fname.assign(fname);

	M = NEW_lint(Cumulative_data.size() * data_len);
	for (u = 0; u < Cumulative_data.size(); u++) {
		for (v = 0; v < data_len; v++) {
			M[u * data_len + v] = Cumulative_data[u][v];
		}
	}
	lint_matrix_write_csv(
			data_fname,
			M, Cumulative_data.size(), data_len);

	if (f_v) {
		cout << "Written file " << data_fname
				<< " of size " << Fio->file_size(data_fname) << endl;
	}
	FREE_lint(M);

}
#endif

void csv_file_support::write_characteristic_matrix(
		std::string &fname,
		long int *data, int nb_rows, int data_sz, int nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *T;
	int i, j, h;

	T = NEW_int(nb_rows * nb_cols);

	Int_vec_zero(T, nb_rows * nb_cols);
	for (i = 0; i < nb_rows; i++) {
		for (h = 0; h < data_sz; h++) {
			j = data[i * data_sz + h];
			T[i * nb_cols + j] = 1;
		}

	}
	int_matrix_write_csv(
			fname, T,
			nb_rows,
			nb_cols);

	if (f_v) {
		cout << "csv_file_support::write_characteristic_matrix "
				"Written file " << fname << " of size " << Fio->file_size(fname) << endl;
	}
	FREE_int(T);


}

void csv_file_support::split_by_values(
		std::string &fname_in, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::split_by_values" << endl;
	}
	int *M;
	int *M2;
	int m, n, len, t, h, a;

	int_matrix_read_csv(
			fname_in, M, m, n, verbose_level);
	len = m * n;
	data_structures::tally T;

	T.init(M, m * n, false, 0);
	cout << "values in the file : ";
	T.print(false);
	cout << endl;

	M2 = NEW_int(len);
	for (t = 0; t < T.nb_types; t++) {
		Int_vec_zero(M2, len);
		a = T.data_sorted[T.type_first[t]];
		string fname;
		data_structures::string_tools ST;

		fname = fname_in;
		ST.chop_off_extension(fname);
		fname += "_value" + std::to_string(a) + ".csv";

		for (h = 0; h < len; h++) {
			if (M[h] == a) {
				M2[h] = 1;
			}
		}
		int_matrix_write_csv(fname, M2, m, n);
		if (f_v) {
			cout << "Written file " << fname
					<< " of size " << Fio->file_size(fname) << endl;
		}
	}
	FREE_int(M2);

	if (f_v) {
		cout << "csv_file_support::split_by_values done" << endl;
	}
}

void csv_file_support::change_values(
		std::string &fname_in, std::string &fname_out,
		std::string &input_values_label,
		std::string &output_values_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::change_values" << endl;
	}

	long int *input_values;
	int sz_input;
	long int *output_values;
	int sz_output;


	Get_lint_vector_from_label(
			input_values_label,
			input_values, sz_input, 0 /* verbose_level */);
	Get_lint_vector_from_label(
			output_values_label,
			output_values, sz_output, 0 /* verbose_level */);

	if (sz_input != sz_output) {
		cout << "csv_file_support::change_values sz_input != sz_output" << endl;
		exit(1);
	}

	int *M;
	long int *orig_pos;
	int m, n, len, h, a, b, idx;

	int_matrix_read_csv(fname_in, M, m, n, verbose_level);
	len = m * n;

	orig_pos = NEW_lint(sz_input);

	for (h = 0; h < sz_input; h++) {
		orig_pos[h] = h;
	}

	data_structures::sorting Sorting;

	Sorting.lint_vec_heapsort_with_log(input_values, orig_pos, sz_input);

	for (h = 0; h < len; h++) {
		a = M[h];

		if (Sorting.lint_vec_search(input_values, sz_input, a,
			idx, 0 /*verbose_level*/)) {
			b = output_values[orig_pos[idx]];
		}
		else {
			b = a;
		}
		M[h] = b;
	}

	int_matrix_write_csv(fname_out, M, m, n);
	if (f_v) {
		cout << "Written file " << fname_out
				<< " of size " << Fio->file_size(fname_out) << endl;
	}

	FREE_int(M);
	FREE_lint(orig_pos);
	FREE_lint(input_values);
	FREE_lint(output_values);

	if (f_v) {
		cout << "csv_file_support::change_values done" << endl;
	}
}

void csv_file_support::write_gedcom_file_as_csv(
		std::string &fname,
		std::vector<std::vector<std::string> > &Data,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::write_gedcom_file_as_csv" << endl;
	}
	{
		ofstream f(fname);
		int len;
		int i;

		len = Data.size();
		f << "Row,C1,C2,C3" << endl;
		for (i = 0; i < len; i++) {
			f << i << "," << Data[i][0] << "," << "\"" << Data[i][1] << "\"" << "," << "\"" << Data[i][2] << "\"" << endl;
		}
		f << "END" << endl;
	}
	if (f_v) {
		cout << "Written file " << fname
				<< " of size " << Fio->file_size(fname) << endl;
	}
	if (f_v) {
		cout << "csv_file_support::write_gedcom_file_as_csv done" << endl;
	}

}

void csv_file_support::write_ancestry_indi(
		std::string &fname,
		std::vector<std::vector<std::string> > &Data,
		int nb_indi,
		data_structures::ancestry_indi **Individual,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::write_ancestry_indi" << endl;
	}
	{
		int i;

		ofstream f(fname);

		f << "Row,ID,SEX,NAME,FAMC,FAMS,BIRTHDATE,DEATHDATE" << endl;
		for (i = 0; i < nb_indi; i++) {
			f << i
					<< "," << "\"" << Individual[i]->id << "\""
					<< "," << "\"" << Individual[i]->sex << "\""
					<< "," << "\"" << Individual[i]->name << "\""
					<< "," << "\"" << Individual[i]->famc << "\""
					<< "," << "\"" << Individual[i]->fams << "\""
					<< "," << "\"" << Individual[i]->birth_date << "\""
					<< "," << "\"" << Individual[i]->death_date << "\""
				<< endl;
		}
		f << "END" << endl;
	}

	if (f_v) {
		cout << "csv_file_support::write_ancestry_indi done" << endl;
	}
}


void csv_file_support::write_ancestry_family(
		std::string &fname,
		std::vector<std::vector<std::string> > &Data,
		int nb_indi,
		int nb_fam,
		data_structures::ancestry_indi **Individual,
		data_structures::ancestry_family **Family,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::write_ancestry_family" << endl;
	}
	{
		int i;

		ofstream f(fname);

		f << "Row,IDX,ID,HUSB,HUSBIDX,HUSBFAMIDX,WIFE,WIFEIDX,WIFEFAMIDX,HUSBNAME,WIFENAME" << endl;
		for (i = 0; i < nb_fam; i++) {
			f << i
					<< "," << "\"" << Family[i]->idx << "\""
					<< "," << "\"" << Family[i]->id << "\""
					<< "," << "\"" << Family[i]->husband << "\""
					<< "," << "\"" << Family[i]->husband_index << "\""
					<< "," << "\"" << Family[i]->husband_family_index << "\""
					<< "," << "\"" << Family[i]->wife << "\""
					<< "," << "\"" << Family[i]->wife_index << "\""
					<< "," << "\"" << Family[i]->wife_family_index << "\"";
					if (Family[i]->husband_index >= 0) {
						f << "," << "\"" << Individual[Family[i]->husband_index]->name << "\"";
					}
					else {
						f << "," << "\"\"";

					}
					if (Family[i]->wife_index >= 0) {
						f << "," << "\"" << Individual[Family[i]->wife_index]->name << "\"";
					}
					else {
						f << "," << "\"\"";

					}
				f << endl;
		}
		f << "END" << endl;
	}

	if (f_v) {
		cout << "csv_file_support::write_ancestry_family done" << endl;
	}
}


#if 0
std::string id;
std::string name;
std::string given_name;
std::string sur_name;
std::string sex;
std::string famc;
std::string fams;
std::string birth_date;
std::string death_date;


int idx;

int start;
int length;
std::string id;

std::string husband;
int husband_index;
int husband_family_index;

std::string wife;
int wife_index;
int wife_family_index;

std::vector<std::string> child;
std::vector<int> child_index;
std::vector<std::vector<int> > child_family_index;

std::vector<int> topo_downlink;
#endif



void csv_file_support::read_table_of_strings(
		std::string &fname, std::string *&col_label,
		std::string *&Table, int &m, int &n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::read_table_of_strings "
				"reading file " << fname << endl;
	}
	if (Fio->file_size(fname) <= 0) {
		cout << "csv_file_support::read_table_of_strings file " << fname
			<< " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << Fio->file_size(fname) << endl;
		exit(1);
	}
	{
		data_structures::spreadsheet S;
		data_structures::string_tools ST;
		int i, j;

		S.read_spreadsheet(fname, 0/*verbose_level - 1*/);

		m = S.nb_rows - 1;
		n = S.nb_cols - 1;
		col_label = new std::string[n];
		Table = new std::string [m * n];
		for (j = 0; j < n; j++) {
			col_label[j] = S.get_entry_ij(
						0, 1 + j);
		}
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				Table[i * n + j] = S.get_entry_ij(
							1 + i, 1 + j);
			}
		}
	}
}


void csv_file_support::read_column_of_strings(
		std::string &fname, std::string &col_label,
		std::string *&Column, int &len,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::read_table_of_strings "
				"reading file " << fname << endl;
	}

	std::string *Col_label;
	std::string *Table;
	int m, n;

	read_table_of_strings(
			fname, Col_label,
			Table, m, n,
			verbose_level);

	int j;

	for (j = 0; j < n; j++) {
		if (Col_label[j] == col_label) {
			break;
		}
	}
	if (j == n) {
		cout << "csv_file_support::read_column_of_strings "
				"did not find column with label " << col_label << endl;
		exit(1);
	}

	int i;

	len = m;
	Column = new string[len];
	for (i = 0; i < len; i++) {
		Column[i] = Table[i * n + j];
	}

	delete [] Table;
	delete [] Col_label;

	if (f_v) {
		cout << "csv_file_support::read_table_of_strings done" << endl;
	}
}



void csv_file_support::read_csv_file_and_get_column(
		std::string &fname, std::string &col_header,
		long int *&Data, int &data_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::read_csv_file_and_get_column" << endl;
		cout << "csv_file_support::read_csv_file_and_get_column fname = " << fname << endl;
		cout << "csv_file_support::read_csv_file_and_get_column col_header = " << col_header << endl;
	}

	data_structures::string_tools ST;

	std::string *col_labels;
	std::string *Table;
	int m, n;

	read_table_of_strings(
			fname, col_labels,
			Table, m, n,
			verbose_level);

	int i, j;

	for (j = 0; j < n; j++) {
		if (col_labels[j] == col_header) {
			break;
		}
	}
	if (j == n) {
		cout << "csv_file_support::read_csv_file_and_get_column did not find column heading " << col_header << endl;
		exit(1);
	}

	data_size = m;
	Data = NEW_lint(m);
	for (i = 0; i < m; i++) {
		cout << "row << " << i << " / " << m << " reading " << Table[i * n + j] << endl;

		std::string s;

		ST.drop_quotes(
				Table[i * n + j], s);


		Data[i] = std::stol(s, NULL, 10);
	}

#if 0
	data_structures::tally T;

	T.init_lint(Data, m, true, 0);
	cout << "tally:" << endl;
	T.print(true);
	cout << endl;


	data_structures::set_of_sets *SoS;
	int *types;
	int nb_types;
	int i;

	SoS = T.get_set_partition_and_types(
			types, nb_types, verbose_level);

	cout << "fibers:" << endl;
	for (i = 0; i < nb_types; i++) {
		cout << i << " : " << types[i] << " : ";
		Lint_vec_print(cout, SoS->Sets[i], SoS->Set_size[i]);
		cout << endl;
	}

	//cout << "set partition:" << endl;
	//SoS->print_table();
#endif

	delete [] Table;

	if (f_v) {
		cout << "csv_file_support::read_csv_file_and_get_column done" << endl;
	}
}





}}}


