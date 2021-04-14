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
	Draw_bitmap_control = NULL;


	f_reformat = FALSE;
	//std::string reformat_fname_in;
	//std::string reformat_fname_out;
	reformat_nb_cols = 0;

	f_split_by_values = FALSE;
	//std::string split_by_values_fname_in;

	f_store_as_csv_file = FALSE;
	//std::string> store_as_csv_file_fname;
	store_as_csv_file_m = 0;
	store_as_csv_file_n = 0;
	//std::string store_as_csv_file_data;

	f_mv = FALSE;
	//std::string mv_a;
	//std::string mv_b;

	f_loop = FALSE;
	loop_start_idx = 0;
	loop_end_idx = 0;
	//std::string loop_variable;
	loop_from = 0;
	loop_to = 0;
	loop_step = 0;
	loop_argv = NULL;


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
		cout << "-draw_matrix options -end" << endl;
	}
	else if (stringcmp(argv[i], "-reformat") == 0) {
		cout << "-reformat <string : fname_in> <string : fname_out> <int : new_width>" << endl;
	}
	else if (stringcmp(argv[i], "-split_by_values") == 0) {
		cout << "-split_by_values <string : fname_in>" << endl;
	}
	else if (stringcmp(argv[i], "-store_as_csv_file") == 0) {
		cout << "-store_as_csv_file <string : fname> <int : m> "
				"<int : n> <string : data> " << endl;
	}
	else if (stringcmp(argv[i], "-mv") == 0) {
		cout << "-mv <string : from> <string : to> " << endl;
	}
	else if (stringcmp(argv[i], "-loop") == 0) {
		cout << "-loop <string : variable> <string : logfile_mask> <int : from> <int : to> <int : step> <arguments> -loop_end" << endl;
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
	else if (stringcmp(argv[i], "-store_as_csv_file") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-mv") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-loop") == 0) {
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
		Draw_bitmap_control = NEW_OBJECT(draw_bitmap_control);
		cout << "reading -draw_matrix" << endl;
		i += Draw_bitmap_control->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		cout << "-draw_matrix " << endl;
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
	else if (stringcmp(argv[i], "-store_as_csv_file") == 0) {
		f_store_as_csv_file = TRUE;
		store_as_csv_file_fname.assign(argv[++i]);
		store_as_csv_file_m = strtoi(argv[++i]);
		store_as_csv_file_n = strtoi(argv[++i]);
		store_as_csv_file_data.assign(argv[++i]);
		cout << "-store_as_csv_file " << store_as_csv_file_fname
				<< " " << store_as_csv_file_m << " " << store_as_csv_file_n << " " << store_as_csv_file_data << endl;
	}
	else if (stringcmp(argv[i], "-mv") == 0) {
		f_mv = TRUE;
		mv_a.assign(argv[++i]);
		mv_b.assign(argv[++i]);
		cout << "-mv " << mv_a
				<< " " << mv_b << endl;
	}
	else if (stringcmp(argv[i], "-loop") == 0) {
		f_loop = TRUE;
		loop_start_idx = i + 5;
		loop_variable.assign(argv[++i]);
		loop_from = strtoi(argv[++i]);
		loop_to = strtoi(argv[++i]);
		loop_step = strtoi(argv[++i]);
		loop_argv = argv;

		for (++i; i < argc; i++) {
			if (stringcmp(argv[i], "-end_loop") == 0) {
				loop_end_idx = i;
				break;
			}
		}
		if (i == argc) {
			cout << "-loop cannot find -end_loop" << endl;
			exit(1);
		}
		cout << "-loop " << loop_variable
				<< " " << loop_from
				<< " " << loop_to
				<< " " << loop_step
				<< " " << loop_start_idx
				<< " " << loop_end_idx;

		for (int j = loop_start_idx; j < loop_end_idx; j++) {
			cout << " " << argv[j];
		}
		cout << endl;

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
		graphical_output GO;

		GO.draw_bitmap(Draw_bitmap_control, verbose_level);

		FREE_int(Draw_bitmap_control->M);
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

		Orbiter->Lint_vec.scan(store_as_csv_file_data, D, sz);
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
	else if (f_mv) {
		string cmd;

		cmd.assign("mv ");
		cmd.append(mv_a);
		cmd.append(" ");
		cmd.append(mv_b);
		cout << "executing " << cmd << endl;
		system(cmd.c_str());
	}
	else if (f_loop) {
		std::string *argv2;
		int argc2;
		int j;

		argc2 = loop_end_idx - loop_start_idx;
		int h, s;

		for (h = loop_from; h < loop_to; h += loop_step) {
			cout << "loop h=" << h << ":" << endl;
			argv2 = new string[argc2];
			for (j = loop_start_idx, s = 0; j < loop_end_idx; j++, s++) {

				char str[1000];
				string arg;
				string value_h;
				string variable;

				arg.assign(loop_argv[j]);
				sprintf(str, "%d", h);
				value_h.assign(str);
				variable.assign("%");
				variable.append(loop_variable);

				while (arg.find(variable) != std::string::npos) {
					arg.replace(arg.find(variable), variable.length(), value_h);
				}
				argv2[s].assign(arg);
			}
			cout << "loop iteration " << h << ", executing sequence of length " << argc2 << " : ";
			for (s = 0; s < argc2; s++) {
				cout << " " << argv2[s];
			}
			cout << endl;


			The_Orbiter_top_level_session->parse_and_execute(argc2 - 1, argv2, 0, verbose_level);

			cout << "loop iteration " << h << "done" << endl;

			delete [] argv2;
		}

	}

	if (f_v) {
		cout << "interface_toolkit::worker done" << endl;
	}
}




}}
