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

void csv_file_support::init(file_io *Fio)
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
		std::string &fname, long int *M, int m, int n)
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
	int *Rows;
	int nb_rows;
	data_structures::string_tools ST;

	Int_vec_scan(rows_text, Rows, nb_rows);

	data_structures::spreadsheet S;

	S.read_spreadsheet(fname, verbose_level);


	int i;



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

	FREE_int(Rows);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_rows done" << endl;
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
		std::string &cols_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_cols" << endl;
	}
	int *Cols;
	int nb_cols;
	data_structures::string_tools ST;

	Int_vec_scan(cols_text, Cols, nb_cols);

	data_structures::spreadsheet S;

	S.read_spreadsheet(fname, verbose_level);


	int i;
	int nb_rows;

	nb_rows = S.nb_rows;
	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_cols "
				"nb_rows=" << nb_rows << endl;
	}


	string fname_out;

	fname_out = fname;
	ST.chop_off_extension(fname_out);
	fname_out += "_select.csv";

	{
		ofstream ost(fname_out);
		ost << "Row,";
		S.print_table_row_with_column_selection(
				0, false, Cols, nb_cols, ost, verbose_level);
		for (i = 0; i < nb_rows - 1; i++) {
			ost << i << ",";
			S.print_table_row_with_column_selection(i + 1, false,
					Cols, nb_cols, ost, verbose_level);
			}
		ost << "END" << endl;
	}
	cout << "Written file " << fname_out
			<< " of size " << Fio->file_size(fname_out) << endl;

	fname_out = fname;
	ST.chop_off_extension(fname_out);
	fname_out += "_special.csv";

	{
		ofstream ost(fname_out);
		//ost << "Row,";
		//S.print_table_row_with_column_selection(0, false, Cols, nb_cols, ost);
		for (i = 0; i < nb_rows - 1; i++) {
			ost << "Orb" << i << "=";
			S.print_table_row_with_column_selection(i + 1, false,
					Cols, nb_cols, ost, verbose_level);
			}
		ost << "END" << endl;
	}
	cout << "Written file " << fname_out
			<< " of size " << Fio->file_size(fname_out) << endl;

	FREE_int(Cols);
	if (f_v) {
		cout << "csv_file_support::do_csv_file_select_cols done" << endl;
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
		identifier_column[i] = S[i].find_by_column(csv_file_join_identifier[i].c_str());
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
		S[0].join_with(S + i, identifier_column[0], identifier_column[i], verbose_level - 2);
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

	for (i = 0; i < nb_files; i++) {
		cout << "Reading table " << fname_in[i] << endl;
		S[i].read_spreadsheet(fname_in[i], 0 /*verbose_level*/);
		cout << "Table " << fname_in[i] << " has been read" << endl;

		if (false) {
			cout << "The " << i << "-th table is:" << endl;
			S[i].print_table(cout, false);
		}


	}

	{
		ofstream ost(fname_out);
		int j;
		int f_enclose_in_parentheses = false;

		S[0].print_table_row(0, f_enclose_in_parentheses, ost);
		for (i = 0; i < nb_files; i++) {
			//S[i].print_table(ost, false);
			for (j = 1; j < S[i].nb_rows; j++) {
				S[i].print_table_row(j, f_enclose_in_parentheses, ost);
			}
		}
		ost << "END" << endl;
	}
	cout << "Written file " << fname_out
			<< " of size " << Fio->file_size(fname_out) << endl;


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

	for (i = 0; i < nb_files; i++) {

		char str[1000];
		std::string fname_in;

		snprintf(str, sizeof(str), fname_in_mask.c_str(), i);
		fname_in.assign(str);

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
		cout << "csv_file_support::do_csv_file_latex S.nb_rows = " << S.nb_rows << endl;
		cout << "csv_file_support::do_csv_file_latex S.nb_cols = " << S.nb_cols << endl;
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
		cout << i << " : " << types[i] << " : ";
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
		cout << "csv_file_support::csv_file_sort_rows fname = "<< fname << endl;
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

void csv_file_support::write_table_of_strings(std::string &fname,
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



}}}


