/*
 * interface_toolkit.cpp
 *
 *  Created on: Nov 29, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {




interface_toolkit::interface_toolkit()
{
	f_csv_file_select_rows = FALSE;
	//std::string csv_file_select_rows_fname;
	//std::string csv_file_select_rows_text;

	f_csv_file_select_cols = FALSE;
	//std::string csv_file_select_cols_fname;
	//std::string csv_file_select_cols_text;

	f_csv_file_select_rows_and_cols = FALSE;
	//csv_file_select_rows_and_cols_fname;
	//std::string csv_file_select_rows_and_cols_R_text;
	//std::string csv_file_select_rows_and_cols_C_text;

	f_csv_file_join = FALSE;
	//csv_file_join_fname
	//csv_file_join_identifier

	f_csv_file_latex = FALSE;
	//std::vector<std::string> csv_file_latex_fname;

	f_draw_matrix = FALSE;
	bit_depth = 8;
	//fname = NULL;
	box_width = 0;


	f_reformat = FALSE;
	//std::string reformat_fname_in;
	//std::string reformat_fname_out;
	reformat_nb_cols = 0;

	f_split_by_values = FALSE;
	//std::string split_by_values_fname_in;

	f_draw_matrix_partition = FALSE;
	draw_matrix_partition_width = 0;
	//std::string draw_matrix_partition_rows;
	//std::string draw_matrix_partition_cols;

	f_store_as_csv_file = FALSE;
	//std::string> store_as_csv_file_fname;
	store_as_csv_file_m = 0;
	store_as_csv_file_n = 0;
	//std::string store_as_csv_file_data;
}


void interface_toolkit::print_help(int argc,
		std::string *argv, int i, int verbose_level)
{
	if (stringcmp(argv[i], "-csv_file_select_rows") == 0) {
		cout << "-cvs_file_select_rows <string : csv_file_name> <string : list of rows>" << endl;
	}
	else if (stringcmp(argv[i], "-csv_file_select_cols") == 0) {
		cout << "-cvs_file_select_cols <string : csv_file_name> <string : list of cols>" << endl;
	}
	else if (stringcmp(argv[i], "-csv_file_select_rows_and_cols") == 0) {
		cout << "-csv_file_select_rows_and_cols <string : csv_file_name> <string : list of rows> <string : list of cols>" << endl;
	}
	else if (stringcmp(argv[i], "-csv_file_join") == 0) {
		cout << "-cvs_file_join <string : file_name> <string : column label by which we join>" << endl;
	}
	else if (stringcmp(argv[i], "-csv_file_latex") == 0) {
		cout << "-cvs_file_latex <string : file_name>" << endl;
	}
	else if (stringcmp(argv[i], "-draw_matrix") == 0) {
		cout << "-draw_matrix <string : fname> <int : box_width> <int : bit_depth>" << endl;
	}
	else if (stringcmp(argv[i], "-reformat") == 0) {
		cout << "-reformat <string : fname_in> <string : fname_out> <int : new_width>" << endl;
	}
	else if (stringcmp(argv[i], "-split_by_values") == 0) {
		cout << "-split_by_values <string : fname_in>" << endl;
	}
	else if (stringcmp(argv[i], "-draw_matrix_partition") == 0) {
		cout << "-draw_matrix_partition <int : width> "
				"<string : row partition> <string : col partition> " << endl;
	}
	else if (stringcmp(argv[i], "-store_as_csv_file") == 0) {
		cout << "-store_as_csv_file <string : fname> <int : m> "
				"<int : n> <string : data> " << endl;
	}
}

int interface_toolkit::recognize_keyword(int argc,
		std::string *argv, int i, int verbose_level)
{
	if (i >= argc) {
		return false;
	}
	if (stringcmp(argv[i], "-csv_file_select_rows") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-csv_file_select_cols") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-csv_file_select_rows_and_cols") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-csv_file_join") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-csv_file_latex") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-draw_matrix") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-reformat") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-split_by_values") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-draw_matrix_partition") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-store_as_csv_file") == 0) {
		return true;
	}
	return false;
}

void interface_toolkit::read_arguments(int argc,
		std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_toolkit::read_arguments" << endl;
	}

	if (f_v) {
		cout << "interface_toolkit::read_arguments the next argument is " << argv[i] << endl;
	}
	if (stringcmp(argv[i], "-csv_file_select_rows") == 0) {
		f_csv_file_select_rows = TRUE;
		csv_file_select_rows_fname.assign(argv[++i]);
		csv_file_select_rows_text.assign(argv[++i]);
		cout << "-csv_file_select_rows " << csv_file_select_rows_fname << " " << csv_file_select_rows_text << endl;
	}
	else if (stringcmp(argv[i], "-csv_file_select_cols") == 0) {
		f_csv_file_select_cols = TRUE;
		csv_file_select_cols_fname.assign(argv[++i]);
		csv_file_select_cols_text.assign(argv[++i]);
		cout << "-csv_file_select_cols " << csv_file_select_cols_fname << " " << csv_file_select_cols_text << endl;
	}
	else if (stringcmp(argv[i], "-csv_file_select_rows_and_cols") == 0) {
		f_csv_file_select_rows_and_cols = TRUE;
		csv_file_select_rows_and_cols_fname.assign(argv[++i]);
		csv_file_select_rows_and_cols_R_text.assign(argv[++i]);
		csv_file_select_rows_and_cols_C_text.assign(argv[++i]);
		cout << "-csv_file_select_rows_and_cols "
				<< csv_file_select_rows_and_cols_fname
				<< " " << csv_file_select_rows_and_cols_R_text
				<< " " << csv_file_select_rows_and_cols_C_text
				<< endl;
	}
	else if (stringcmp(argv[i], "-csv_file_join") == 0) {
		string s;

		f_csv_file_join = TRUE;
		s.assign(argv[++i]);
		csv_file_join_fname.push_back(s);
		s.assign(argv[++i]);
		csv_file_join_identifier.push_back(s);
		cout << "-join " << csv_file_join_fname[csv_file_join_fname.size() - 1] << " "
				<< csv_file_join_identifier[csv_file_join_identifier.size() - 1] << endl;
	}
	else if (stringcmp(argv[i], "-csv_file_latex") == 0) {
		f_csv_file_latex = TRUE;
		csv_file_latex_fname.assign(argv[++i]);
		cout << "-csv_file_latex " << csv_file_latex_fname << endl;
	}
	else if (stringcmp(argv[i], "-draw_matrix") == 0) {
		f_draw_matrix = TRUE;
		fname.assign(argv[++i]);
		box_width = strtoi(argv[++i]);
		bit_depth = strtoi(argv[++i]);
		cout << "-draw_matrix " << fname << " " << box_width << " " << bit_depth << endl;
	}
	else if (stringcmp(argv[i], "-reformat") == 0) {
		f_reformat = TRUE;
		reformat_fname_in.assign(argv[++i]);
		reformat_fname_out.assign(argv[++i]);
		reformat_nb_cols = strtoi(argv[++i]);
		cout << "-reformat " << reformat_fname_in << " " << reformat_fname_out << " " << reformat_nb_cols << endl;
	}
	else if (stringcmp(argv[i], "-split_by_values") == 0) {
		f_split_by_values = TRUE;
		split_by_values_fname_in.assign(argv[++i]);
		cout << "-split_by_values " << split_by_values_fname_in << endl;
	}
	else if (stringcmp(argv[i], "-draw_matrix_partition") == 0) {
		f_draw_matrix_partition = TRUE;
		draw_matrix_partition_width = strtoi(argv[++i]);
		draw_matrix_partition_rows.assign(argv[++i]);
		draw_matrix_partition_cols.assign(argv[++i]);
		cout << "-draw_matrix_partition " << draw_matrix_partition_rows
				<< " " << draw_matrix_partition_cols << endl;
	}
	else if (stringcmp(argv[i], "-store_as_csv_file") == 0) {
		f_store_as_csv_file = TRUE;
		store_as_csv_file_fname.assign(argv[++i]);
		store_as_csv_file_m = strtoi(argv[++i]);
		store_as_csv_file_n = strtoi(argv[++i]);
		store_as_csv_file_data.assign(argv[++i]);
		cout << "-store_as_csv_file " << store_as_csv_file_fname
				<< " " << store_as_csv_file_m << " " << store_as_csv_file_n << " " << store_as_csv_file_data << endl;
	}
	if (f_v) {
		cout << "interface_toolkit::read_arguments done" << endl;
	}
}

void interface_toolkit::worker(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_toolkit::worker" << endl;
	}

	if (f_csv_file_select_rows) {

		file_io Fio;

		Fio.do_csv_file_select_rows(csv_file_select_rows_fname,
				csv_file_select_rows_text, verbose_level);
	}
	else if (f_csv_file_select_cols) {

		file_io Fio;

		Fio.do_csv_file_select_cols(csv_file_select_cols_fname,
				csv_file_select_cols_text, verbose_level);
	}
	else if (f_csv_file_select_rows_and_cols) {

		file_io Fio;

		Fio.do_csv_file_select_rows_and_cols(
				csv_file_select_rows_and_cols_fname,
				csv_file_select_rows_and_cols_R_text, csv_file_select_rows_and_cols_C_text,
				verbose_level);
	}
	else if (f_csv_file_join) {

		file_io Fio;

		Fio.do_csv_file_join(csv_file_join_fname,
				csv_file_join_identifier, verbose_level);
	}
	else if (f_csv_file_latex) {

		file_io Fio;

		Fio.do_csv_file_latex(csv_file_latex_fname, verbose_level);
	}
	else if (f_draw_matrix) {
		file_io Fio;
		int *M;
		int m, n;

		Fio.int_matrix_read_csv(fname, M, m, n, verbose_level);

		if (f_draw_matrix_partition) {
			int *row_parts;
			int *col_parts;
			int nb_row_parts;
			int nb_col_parts;

			Orbiter->Int_vec.scan(draw_matrix_partition_rows, row_parts, nb_row_parts);
			Orbiter->Int_vec.scan(draw_matrix_partition_cols, col_parts, nb_col_parts);
			draw_bitmap(fname, M, m, n,
					TRUE, draw_matrix_partition_width, // int f_partition, int part_width,
					nb_row_parts, row_parts, nb_col_parts, col_parts, // int nb_row_parts, int *Row_part, int nb_col_parts, int *Col_part,
					TRUE /* f_box_width */, box_width,
					FALSE /* f_invert_colors */, bit_depth,
					verbose_level);
			FREE_int(row_parts);
			FREE_int(col_parts);
		}
		else {
			draw_bitmap(fname, M, m, n,
					FALSE, 0, // int f_partition, int part_width,
					0, NULL, 0, NULL, // int nb_row_parts, int *Row_part, int nb_col_parts, int *Col_part,
					TRUE /* f_box_width */, box_width,
					FALSE /* f_invert_colors */, bit_depth,
					verbose_level);
		}
		FREE_int(M);
	}
	else if (f_reformat) {
		file_io Fio;
		int *M;
		int *M2;
		int m, n;
		int len;
		int m2;

		Fio.int_matrix_read_csv(reformat_fname_in, M, m, n, verbose_level);
		len = m * n;
		m2 = (len + reformat_nb_cols - 1) / reformat_nb_cols;
		M2 = NEW_int(m2 * reformat_nb_cols);
		Orbiter->Int_vec.zero(M2, m2 * reformat_nb_cols);
		Orbiter->Int_vec.copy(M, M2, len);
		Fio.int_matrix_write_csv(reformat_fname_out, M2, m2, reformat_nb_cols);
		cout << "Written file " << reformat_fname_out << " of size " << Fio.file_size(reformat_fname_out) << endl;
	}
	else if (f_split_by_values) {
		file_io Fio;
		int *M;
		int *M2;
		int m, n, len, t, h, a;

		Fio.int_matrix_read_csv(split_by_values_fname_in, M, m, n, verbose_level);
		len = m * n;
		tally T;

		T.init(M, m * n, FALSE, 0);
		cout << "values in the file : ";
		T.print(FALSE);
		cout << endl;

		M2 = NEW_int(len);
		for (t = 0; t < T.nb_types; t++) {
			Orbiter->Int_vec.zero(M2, len);
			a = T.data_sorted[T.type_first[t]];
			string fname;
			char str[1000];

			fname.assign(split_by_values_fname_in);
			chop_off_extension(fname);
			sprintf(str, "_value%d.csv", a);
			fname.append(str);
			for (h = 0; h < len; h++) {
				if (M[h] == a) {
					M2[h] = 1;
				}
			}
			Fio.int_matrix_write_csv(fname, M2, m, n);
			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}
		FREE_int(M2);
	}
	else if (f_store_as_csv_file) {
		long int *D;
		int sz;

		cout << "f_store_as_csv_file" << endl;
		cout << "data=" << store_as_csv_file_data << endl;

		lint_vec_scan(store_as_csv_file_data, D, sz);
		if (sz != store_as_csv_file_m * store_as_csv_file_n) {
			cout << "sz != store_as_csv_file_m * store_as_csv_file_n" << endl;
			cout << "sz = " << sz << endl;
			cout << "store_as_csv_file_m = " << store_as_csv_file_m << endl;
			cout << "store_as_csv_file_n = " << store_as_csv_file_n << endl;
			exit(1);
		}
		file_io Fio;

		Fio.lint_matrix_write_csv(store_as_csv_file_fname, D, store_as_csv_file_m, store_as_csv_file_n);
		cout << "Written file " << store_as_csv_file_fname << " of size " << Fio.file_size(store_as_csv_file_fname) << endl;
	}

	if (f_v) {
		cout << "interface_toolkit::worker done" << endl;
	}
}




}}
