/*
 * interface_toolkit.cpp
 *
 *  Created on: Nov 29, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace user_interface {




interface_toolkit::interface_toolkit()
{
	f_create_files = FALSE;
	Create_file_description = NULL;


	f_csv_file_select_rows = FALSE;
	//std::string csv_file_select_rows_fname;
	//std::string csv_file_select_rows_text;

	f_csv_file_split_rows_modulo = FALSE;
	//std::string csv_file_split_rows_modulo_fname;
	csv_file_split_rows_modulo_n = 0;

	f_csv_file_select_cols = FALSE;
	//std::string csv_file_select_cols_fname;
	//std::string csv_file_select_cols_text;

	f_csv_file_select_rows_and_cols = FALSE;
	//csv_file_select_rows_and_cols_fname;
	//std::string csv_file_select_rows_and_cols_R_text;
	//std::string csv_file_select_rows_and_cols_C_text;

	f_csv_file_sort_each_row = FALSE;
	//std::string csv_file_sort_each_row_fname;

	f_csv_file_join = FALSE;
	//csv_file_join_fname
	//csv_file_join_identifier


	f_csv_file_concatenate = FALSE;
	//std::string csv_file_concatenate_fname_out;
	//std::vector<std::string> csv_file_concatenate_fname_in;

	f_csv_file_extract_column_to_txt = FALSE;
	//std::string csv_file_extract_column_to_txt_fname;
	//std::string csv_file_extract_column_to_txt_col_label;

	f_csv_file_latex = FALSE;
	f_produce_latex_header = FALSE;
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

	f_plot_function = FALSE;
	//std::string plot_function_fname;

	f_draw_projective_curve = FALSE;
	Draw_projective_curve_description = NULL;

	f_tree_draw = FALSE;
	Tree_draw_options = NULL;
}


void interface_toolkit::print_help(int argc,
		std::string *argv, int i, int verbose_level)
{
	data_structures::string_tools ST;

	if (ST.stringcmp(argv[i], "-create_files") == 0) {
		cout << "-create_files <description>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows") == 0) {
		cout << "-cvs_file_select_rows <string : csv_file_name> <string : list of rows>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_split_rows_modulo") == 0) {
		cout << "-csv_file_split_rows_modulo <string : csv_file_name> <int : n>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_cols") == 0) {
		cout << "-cvs_file_select_cols <string : csv_file_name> <string : list of cols>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows_and_cols") == 0) {
		cout << "-csv_file_select_rows_and_cols <string : csv_file_name> <string : list of rows> <string : list of cols>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_sort_each_row") == 0) {
		cout << "-csv_file_sort_each_row <string : input file>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_join") == 0) {
		cout << "-cvs_file_join <int : number of files> <string : input file1> <string : column label1> ..." << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_concatenate") == 0) {
		cout << "-csv_file_concatenate <string : fname_out> <int : number of input files> <string : inout file1> ..." << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_extract_column_to_txt") == 0) {
		cout << "-csv_file_extract_column_to_txt <string : csv_fname> <string : col_label>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_latex") == 0) {
		cout << "-cvs_file_latex <int : f_produce_header> <string : file_name>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-draw_matrix") == 0) {
		cout << "-draw_matrix options -end" << endl;
	}
	else if (ST.stringcmp(argv[i], "-reformat") == 0) {
		cout << "-reformat <string : fname_in> <string : fname_out> <int : new_width>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-split_by_values") == 0) {
		cout << "-split_by_values <string : fname_in>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-store_as_csv_file") == 0) {
		cout << "-store_as_csv_file <string : fname> <int : m> "
				"<int : n> <string : data> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-mv") == 0) {
		cout << "-mv <string : from> <string : to> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-loop") == 0) {
		cout << "-loop <string : variable> <string : logfile_mask> <int : from> <int : to> <int : step> <arguments> -loop_end" << endl;
	}
	else if (ST.stringcmp(argv[i], "-plot_function") == 0) {
		cout << "-plot_function <string : fname_csv>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-draw_projective_curve") == 0) {
		cout << "-draw_projective_curve ..." << endl;
	}
	else if (ST.stringcmp(argv[i], "-tree_draw") == 0) {
		cout << "-tree_draw <options> -end" << endl;
	}
}

int interface_toolkit::recognize_keyword(int argc,
		std::string *argv, int i, int verbose_level)
{
	data_structures::string_tools ST;

	if (i >= argc) {
		return false;
	}
	if (ST.stringcmp(argv[i], "-create_files") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_split_rows_modulo") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_cols") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows_and_cols") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_sort_each_row") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_join") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_concatenate") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_extract_column_to_txt") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_latex") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-draw_matrix") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-reformat") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-split_by_values") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-store_as_csv_file") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-mv") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-loop") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-plot_function") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-draw_projective_curve") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-tree_draw") == 0) {
		return true;
	}
	return false;
}

void interface_toolkit::read_arguments(int argc,
		std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "interface_toolkit::read_arguments" << endl;
	}

	if (f_v) {
		cout << "interface_toolkit::read_arguments the next argument is " << argv[i] << endl;
	}
	if (ST.stringcmp(argv[i], "-create_files") == 0) {
		f_create_files = TRUE;

		if (f_v) {
			cout << "-create_files " << endl;
		}

		Create_file_description = NEW_OBJECT(orbiter_kernel_system::create_file_description);
		i += Create_file_description->read_arguments(argc - i - 1,
			argv + i + 1, verbose_level);
		if (f_v) {
			cout << "interface_combinatorics::read_arguments finished "
					"reading -create_files" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}

			cout << "-create_files " <<endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows") == 0) {
		f_csv_file_select_rows = TRUE;
		csv_file_select_rows_fname.assign(argv[++i]);
		csv_file_select_rows_text.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_select_rows " << csv_file_select_rows_fname
				<< " " << csv_file_select_rows_text << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_split_rows_modulo") == 0) {
		f_csv_file_split_rows_modulo = TRUE;
		csv_file_split_rows_modulo_fname.assign(argv[++i]);
		csv_file_split_rows_modulo_n = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-csv_file_split_rows_modulo " << csv_file_split_rows_modulo_fname
				<< " " << csv_file_split_rows_modulo_n << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_cols") == 0) {
		f_csv_file_select_cols = TRUE;
		csv_file_select_cols_fname.assign(argv[++i]);
		csv_file_select_cols_text.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_select_cols " << csv_file_select_cols_fname
				<< " " << csv_file_select_cols_text << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows_and_cols") == 0) {
		f_csv_file_select_rows_and_cols = TRUE;
		csv_file_select_rows_and_cols_fname.assign(argv[++i]);
		csv_file_select_rows_and_cols_R_text.assign(argv[++i]);
		csv_file_select_rows_and_cols_C_text.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_select_rows_and_cols "
				<< csv_file_select_rows_and_cols_fname
				<< " " << csv_file_select_rows_and_cols_R_text
				<< " " << csv_file_select_rows_and_cols_C_text
				<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_sort_each_row") == 0) {
		f_csv_file_sort_each_row = TRUE;
		csv_file_sort_each_row_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_sort_each_row "
				<< csv_file_sort_each_row_fname
				<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_join") == 0) {
		string s;
		int nb, j;

		f_csv_file_join = TRUE;
		nb = ST.strtoi(argv[++i]);
		for (j = 0; j < nb; j++) {
			s.assign(argv[++i]);
			csv_file_join_fname.push_back(s);
			s.assign(argv[++i]);
			csv_file_join_identifier.push_back(s);
		}
		if (f_v) {
			cout << "-csv_file_join " << endl;
			for (j = 0; j < nb; j++) {
				cout << j << " : " << csv_file_join_fname[j]
					<< " : " << csv_file_join_identifier[j] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_concatenate") == 0) {
		string s;
		int nb, j;

		f_csv_file_concatenate = TRUE;
		csv_file_concatenate_fname_out.assign(argv[++i]);
		nb = ST.strtoi(argv[++i]);
		for (j = 0; j < nb; j++) {
			s.assign(argv[++i]);
			csv_file_concatenate_fname_in.push_back(s);
		}
		if (f_v) {
			cout << "-csv_file_concatenate " << csv_file_concatenate_fname_out << endl;
			for (j = 0; j < nb; j++) {
				cout << j << " : " << csv_file_concatenate_fname_in[j] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_extract_column_to_txt") == 0) {
		f_csv_file_extract_column_to_txt = TRUE;
		csv_file_extract_column_to_txt_fname.assign(argv[++i]);
		csv_file_extract_column_to_txt_col_label.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_extract_column_to_txt " << csv_file_extract_column_to_txt_fname << " " << csv_file_extract_column_to_txt_col_label << endl;
		}
	}

	else if (ST.stringcmp(argv[i], "-csv_file_latex") == 0) {
		f_csv_file_latex = TRUE;
		f_produce_latex_header = ST.strtoi(argv[++i]);
		csv_file_latex_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_latex " << f_produce_latex_header << " " << csv_file_latex_fname << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-draw_matrix") == 0) {
		f_draw_matrix = TRUE;
		Draw_bitmap_control = NEW_OBJECT(graphics::draw_bitmap_control);
		if (f_v) {
			cout << "reading -draw_matrix" << endl;
		}
		i += Draw_bitmap_control->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);
		if (f_v) {
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-draw_matrix " << endl;
			Draw_bitmap_control->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-reformat") == 0) {
		f_reformat = TRUE;
		reformat_fname_in.assign(argv[++i]);
		reformat_fname_out.assign(argv[++i]);
		reformat_nb_cols = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-reformat " << reformat_fname_in
				<< " " << reformat_fname_out
				<< " " << reformat_nb_cols << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-split_by_values") == 0) {
		f_split_by_values = TRUE;
		split_by_values_fname_in.assign(argv[++i]);
		if (f_v) {
			cout << "-split_by_values " << split_by_values_fname_in << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-store_as_csv_file") == 0) {
		f_store_as_csv_file = TRUE;
		store_as_csv_file_fname.assign(argv[++i]);
		store_as_csv_file_m = ST.strtoi(argv[++i]);
		store_as_csv_file_n = ST.strtoi(argv[++i]);
		store_as_csv_file_data.assign(argv[++i]);
		if (f_v) {
			cout << "-store_as_csv_file " << store_as_csv_file_fname
				<< " " << store_as_csv_file_m
				<< " " << store_as_csv_file_n
				<< " " << store_as_csv_file_data << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-mv") == 0) {
		f_mv = TRUE;
		mv_a.assign(argv[++i]);
		mv_b.assign(argv[++i]);
		if (f_v) {
			cout << "-mv " << mv_a
				<< " " << mv_b << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-loop") == 0) {
		f_loop = TRUE;
		loop_start_idx = i + 5;
		loop_variable.assign(argv[++i]);
		loop_from = ST.strtoi(argv[++i]);
		loop_to = ST.strtoi(argv[++i]);
		loop_step = ST.strtoi(argv[++i]);
		loop_argv = argv;

		for (++i; i < argc; i++) {
			if (ST.stringcmp(argv[i], "-end_loop") == 0) {
				loop_end_idx = i;
				break;
			}
		}
		if (i == argc) {
			cout << "-loop cannot find -end_loop" << endl;
			exit(1);
		}
		if (f_v) {
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
	}
	else if (ST.stringcmp(argv[i], "-plot_function") == 0) {
		f_plot_function = TRUE;
		plot_function_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-plot_function " << plot_function_fname << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-draw_projective_curve") == 0) {
		f_draw_projective_curve = TRUE;
		Draw_projective_curve_description = NEW_OBJECT(graphics::draw_projective_curve_description);
		if (f_v) {
			cout << "reading -draw_projective_curve" << endl;
		}
		i += Draw_projective_curve_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);
		if (f_v) {
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-draw_projective_curve " << endl;
			Draw_projective_curve_description->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-tree_draw") == 0) {
		f_tree_draw = TRUE;
		Tree_draw_options = NEW_OBJECT(graphics::tree_draw_options);
		if (f_v) {
			cout << "reading -tree_draw" << endl;
		}
		i += Tree_draw_options->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);
		if (f_v) {
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-tree_draw " << endl;
			Tree_draw_options->print();
		}
	}



	if (f_v) {
			cout << "interface_toolkit::read_arguments done" << endl;
	}
}

void interface_toolkit::print()
{
	int j;

	if (f_create_files) {
		cout << "-create_files " << endl;
		Create_file_description->print();
	}
	if (f_csv_file_select_rows) {
		cout << "-csv_file_select_rows " << csv_file_select_rows_fname
				<< " " << csv_file_select_rows_text << endl;
	}
	if (f_csv_file_split_rows_modulo) {
		cout << "-csv_file_split_rows_modulo " << csv_file_split_rows_modulo_fname
				<< " " << csv_file_split_rows_modulo_n << endl;
	}
	if (f_csv_file_select_cols) {
		cout << "-csv_file_select_cols " << csv_file_select_cols_fname
				<< " " << csv_file_select_cols_text << endl;
	}
	if (f_csv_file_select_rows_and_cols) {
		cout << "-csv_file_select_rows_and_cols "
				<< csv_file_select_rows_and_cols_fname
				<< " " << csv_file_select_rows_and_cols_R_text
				<< " " << csv_file_select_rows_and_cols_C_text
				<< endl;
	}
	if (f_csv_file_sort_each_row) {
		cout << "-csv_file_sort_each_row "
				<< csv_file_sort_each_row_fname
				<< endl;
	}
	if (f_csv_file_join) {
		cout << "-csv_file_join " << endl;
		for (j = 0; j < csv_file_join_fname.size(); j++) {
			cout << j << " : " << csv_file_join_fname[j] << " : " << csv_file_join_identifier[j] << endl;
		}
	}
	if (f_csv_file_concatenate) {
		cout << "-csv_file_concatenate " << csv_file_concatenate_fname_out << endl;
		for (j = 0; j < csv_file_concatenate_fname_in.size(); j++) {
			cout << j << " : " << csv_file_concatenate_fname_in[j] << endl;
		}
	}
	if (f_csv_file_extract_column_to_txt) {
		cout << "-csv_file_extract_column_to_txt " << csv_file_extract_column_to_txt_fname << " " << csv_file_extract_column_to_txt_col_label << endl;
	}
	if (f_csv_file_latex) {
		cout << "-csv_file_latex " << csv_file_latex_fname << endl;
	}
	if (f_draw_matrix) {
		cout << "-draw_matrix " << endl;
		Draw_bitmap_control->print();
	}
	if (f_reformat) {
		cout << "-reformat " << reformat_fname_in
				<< " " << reformat_fname_out
				<< " " << reformat_nb_cols << endl;
	}
	if (f_split_by_values) {
		cout << "-split_by_values " << split_by_values_fname_in << endl;
	}
	if (f_store_as_csv_file) {
		cout << "-store_as_csv_file " << store_as_csv_file_fname
				<< " " << store_as_csv_file_m
				<< " " << store_as_csv_file_n
				<< " " << store_as_csv_file_data << endl;
	}
	if (f_mv) {
		cout << "-mv " << mv_a
				<< " " << mv_b << endl;
	}
	if (f_loop) {
		cout << "-loop " << loop_variable
				<< " " << loop_from
				<< " " << loop_to
				<< " " << loop_step
				<< " " << loop_start_idx
				<< " " << loop_end_idx;

		cout << endl;

	}
	if (f_plot_function) {
		cout << "-plot_function " << plot_function_fname << endl;
	}
	if (f_draw_projective_curve) {
		cout << "-draw_projective_curve " << endl;
		Draw_projective_curve_description->print();
	}
	if (f_tree_draw) {
		cout << "-tree_draw " << endl;
		Tree_draw_options->print();
	}
}

void interface_toolkit::worker(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_toolkit::worker" << endl;
	}

	if (f_create_files) {
		orbiter_kernel_system::file_io Fio;

		Fio.create_file(Create_file_description, verbose_level);
	}
	else if (f_csv_file_select_rows) {

		orbiter_kernel_system::file_io Fio;

		Fio.do_csv_file_select_rows(csv_file_select_rows_fname,
				csv_file_select_rows_text, verbose_level);
	}
	else if (f_csv_file_split_rows_modulo) {

		orbiter_kernel_system::file_io Fio;

		Fio.do_csv_file_split_rows_modulo(csv_file_split_rows_modulo_fname,
				csv_file_split_rows_modulo_n, verbose_level);
	}
	else if (f_csv_file_select_cols) {

		orbiter_kernel_system::file_io Fio;

		Fio.do_csv_file_select_cols(csv_file_select_cols_fname,
				csv_file_select_cols_text, verbose_level);
	}
	else if (f_csv_file_select_rows_and_cols) {

		orbiter_kernel_system::file_io Fio;

		Fio.do_csv_file_select_rows_and_cols(
				csv_file_select_rows_and_cols_fname,
				csv_file_select_rows_and_cols_R_text, csv_file_select_rows_and_cols_C_text,
				verbose_level);
	}
	else if (f_csv_file_sort_each_row) {
		orbiter_kernel_system::file_io Fio;

		Fio.do_csv_file_sort_each_row(csv_file_sort_each_row_fname, verbose_level);

	}
	else if (f_csv_file_join) {

		orbiter_kernel_system::file_io Fio;

		Fio.do_csv_file_join(csv_file_join_fname,
				csv_file_join_identifier, verbose_level);
	}
	else if (f_csv_file_concatenate) {

		orbiter_kernel_system::file_io Fio;

		Fio.do_csv_file_concatenate(csv_file_concatenate_fname_in,
				csv_file_concatenate_fname_out, verbose_level);
	}
	else if (f_csv_file_extract_column_to_txt) {

		orbiter_kernel_system::file_io Fio;

		Fio.do_csv_file_extract_column_to_txt(csv_file_extract_column_to_txt_fname, csv_file_extract_column_to_txt_col_label, verbose_level);

	}
	else if (f_csv_file_latex) {

		orbiter_kernel_system::file_io Fio;
		int nb_lines_per_table = 25;

		Fio.do_csv_file_latex(csv_file_latex_fname,
				f_produce_latex_header,
				nb_lines_per_table,
				verbose_level);
	}
	else if (f_draw_matrix) {
		graphics::graphical_output GO;

		GO.draw_bitmap(Draw_bitmap_control, verbose_level);

		FREE_int(Draw_bitmap_control->M);
	}
	else if (f_reformat) {
		orbiter_kernel_system::file_io Fio;
		int *M;
		int *M2;
		int m, n;
		int len;
		int m2;

		Fio.int_matrix_read_csv(reformat_fname_in, M, m, n, verbose_level);
		len = m * n;
		m2 = (len + reformat_nb_cols - 1) / reformat_nb_cols;
		M2 = NEW_int(m2 * reformat_nb_cols);
		Int_vec_zero(M2, m2 * reformat_nb_cols);
		Int_vec_copy(M, M2, len);
		Fio.int_matrix_write_csv(reformat_fname_out, M2, m2, reformat_nb_cols);
		cout << "Written file " << reformat_fname_out << " of size " << Fio.file_size(reformat_fname_out) << endl;
	}
	else if (f_split_by_values) {
		orbiter_kernel_system::file_io Fio;
		int *M;
		int *M2;
		int m, n, len, t, h, a;

		Fio.int_matrix_read_csv(split_by_values_fname_in, M, m, n, verbose_level);
		len = m * n;
		data_structures::tally T;

		T.init(M, m * n, FALSE, 0);
		cout << "values in the file : ";
		T.print(FALSE);
		cout << endl;

		M2 = NEW_int(len);
		for (t = 0; t < T.nb_types; t++) {
			Int_vec_zero(M2, len);
			a = T.data_sorted[T.type_first[t]];
			string fname;
			char str[1000];
			data_structures::string_tools ST;

			fname.assign(split_by_values_fname_in);
			ST.chop_off_extension(fname);
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

		Lint_vec_scan(store_as_csv_file_data, D, sz);
		if (sz != store_as_csv_file_m * store_as_csv_file_n) {
			cout << "sz != store_as_csv_file_m * store_as_csv_file_n" << endl;
			cout << "sz = " << sz << endl;
			cout << "store_as_csv_file_m = " << store_as_csv_file_m << endl;
			cout << "store_as_csv_file_n = " << store_as_csv_file_n << endl;
			exit(1);
		}
		orbiter_kernel_system::file_io Fio;

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
	else if (f_plot_function) {
		orbiter_kernel_system::file_io Fio;
		data_structures::string_tools ST;
		int *T;
		int *M;
		int m1, n1, n, x, y;

		Fio.int_matrix_read_csv(plot_function_fname, T, m1, n1, verbose_level);


		n = m1 * n1;

		M = NEW_int(n * n);
		Int_vec_zero(M, n * n);


		for (x = 0; x < n; x++) {

			y = T[x];
			M[(n - 1 - y) * n + x] = 1;
		}
		string fname;

		fname.assign(plot_function_fname);
		ST.chop_off_extension(fname);
		fname.append("_graph.csv");
		Fio.int_matrix_write_csv(fname, M, n, n);
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	}
	else if (f_draw_projective_curve) {
		graphics::graphical_output GO;

		GO.draw_projective_curve(Draw_projective_curve_description,
				orbiter_kernel_system::Orbiter->draw_options, verbose_level);

	}
	else if (f_tree_draw) {
		graphics::graphical_output GO;

		GO.tree_draw(Tree_draw_options, verbose_level);

	}


	if (f_v) {
		cout << "interface_toolkit::worker done" << endl;
	}
}


}}}

