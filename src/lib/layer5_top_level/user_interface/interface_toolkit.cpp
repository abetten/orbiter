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
	Record_birth();
	f_create_files_direct = false;
	//std::string create_files_direct_fname_mask;
	//std::string create_files_direct_text;
	//std::vector<std::string> create_files_direct_labels;

	f_create_files = false;
	Create_file_description = NULL;

	f_save_matrix_csv = false;
	//std::string save_matrix_csv_label;

	f_csv_file_tally = false;
	//std::string csv_file_tally_fname;

	f_tally_column = false;
	std::string tally_column_fname;
	std::string tally_column_column;

	f_collect_stats = false;
	//std::string collect_stats_fname_mask;
	//std::string collect_stats_fname_out;
	collect_stats_first = 0;
	collect_stats_last = 0;
	collect_stats_step = 0;

	f_csv_file_select_rows = false;
	//std::string csv_file_select_rows_fname;
	//std::string csv_file_select_rows_text;

	f_csv_file_select_rows_by_file = false;
	//std::string csv_file_select_rows_by_file_fname;
	//std::string csv_file_select_rows_by_file_select;

	f_csv_file_select_rows_complement = false;
	//std::string csv_file_select_rows_complement_fname;
	//std::string csv_file_select_rows_complement_text;

	f_csv_file_split_rows_modulo = false;
	//std::string csv_file_split_rows_modulo_fname;
	csv_file_split_rows_modulo_n = 0;

	f_csv_file_select_cols = false;
	//std::string csv_file_select_cols_fname;
	//std::string csv_file_select_cols_fname_append;
	//std::string csv_file_select_cols_text;


	f_csv_file_select_cols_by_label = false;
	//std::string csv_file_select_cols_by_label_fname;
	//std::string csv_file_select_cols_by_label_fname_append;
	//std::string csv_file_select_cols_by_label_text;


	f_csv_file_select_rows_and_cols = false;
	//csv_file_select_rows_and_cols_fname;
	//std::string csv_file_select_rows_and_cols_R_text;
	//std::string csv_file_select_rows_and_cols_C_text;

	f_csv_file_sort_each_row = false;
	//std::string csv_file_sort_each_row_fname;

	f_csv_file_sort_rows = false;
	//std::string csv_file_sort_rows_fname;

	f_csv_file_sort_rows_and_remove_duplicates = false;
	//std::string csv_file_sort_rows_and_remove_duplicates_fname;


	f_csv_file_join = false;
	//csv_file_join_fname
	//csv_file_join_identifier


	f_csv_file_concatenate = false;
	//std::string csv_file_concatenate_fname_out;
	//std::vector<std::string> csv_file_concatenate_fname_in;

	f_csv_file_concatenate_from_mask = false;
	csv_file_concatenate_from_mask_N_min = 0;
	csv_file_concatenate_from_mask_N_length = 0;
	//std::string csv_file_concatenate_from_mask_mask;
	//std::string csv_file_concatenate_from_mask_fname_out;

	f_csv_file_extract_column_to_txt = false;
	//std::string csv_file_extract_column_to_txt_fname;
	//std::string csv_file_extract_column_to_txt_col_label;

	f_csv_file_filter = false;
	//std::string csv_file_filter_fname
	//std::string csv_file_filter_col;
	//std::string csv_file_filter_value;

	f_csv_file_latex = false;
	f_produce_latex_header = false;
	//std::vector<std::string> csv_file_latex_fname;

	f_prepare_tables_for_users_guide = false;
	//std::vector<std::string> prepare_tables_for_users_guide_fname;

	f_prepare_general_tables_for_users_guide = false;
	//std::vector<std::string> prepare_general_tables_for_users_guide_fname;

	f_grade_statistic_from_csv = false;
	//std::string grade_statistic_from_csv_fname;
	//std::string grade_statistic_from_csv_m1_label;
	//std::string grade_statistic_from_csv_m2_label;
	//std::string grade_statistic_from_csv_final_label;
	//std::string grade_statistic_from_csv_oracle_grade_label;

	f_draw_matrix = false;
	Draw_bitmap_control = NULL;


	f_reformat = false;
	//std::string reformat_fname_in;
	//std::string reformat_fname_out;
	reformat_nb_cols = 0;

	f_split_by_values = false;
	//std::string split_by_values_fname_in;

	f_change_values = false;
	//std::string change_values_fname_in;
	//std::string change_values_fname_out;
	//std::string change_values_function_input;
	//std::string change_values_function_output;

	f_store_as_csv_file = false;
	//std::string> store_as_csv_file_fname;
	store_as_csv_file_m = 0;
	store_as_csv_file_n = 0;
	//std::string store_as_csv_file_data;

	f_mv = false;
	//std::string mv_a;
	//std::string mv_b;

	f_cp = false;
	//std::string cp_a;
	//std::string cp_b;

	f_system = false;
	//std::string system_command;

	f_loop = false;
	loop_start_idx = 0;
	loop_end_idx = 0;
	//std::string loop_variable;
	loop_from = 0;
	loop_to = 0;
	loop_step = 0;
	loop_argv = NULL;

	f_loop_over = false;
	loop_over_start_idx = 0;
	loop_over_end_idx = 0;
	//std::string loop_over_variable;
	//std::string loop_over_domain;
	loop_over_argv = NULL;

	f_plot_function = false;
	//std::string plot_function_fname;

	f_draw_projective_curve = false;
	Draw_projective_curve_description = NULL;

	f_tree_draw = false;
	Tree_draw_options = NULL;

	f_extract_from_file = false;
	//std::string extract_from_file_fname;
	//std::string extract_from_file_label;
	//std::string extract_from_file_target_fname;

	f_extract_from_file_with_tail = false;
	//std::string extract_from_file_with_tail_fname;
	//std::string extract_from_file_with_tail_label;
	//std::string extract_from_file_with_tail_tail;
	//std::string extract_from_file_with_tail_target_fname;

	f_serialize_file_names = false;
	//std::string serialize_file_names_fname;
	//std::string serialize_file_names_output_mask;

	f_save_4_bit_data_file = false;
	//std::string save_4_bit_data_file_fname;
	//std::string save_4_bit_data_file_vector_data;


	f_gnuplot = false;
	//std::string gnuplot_file_fname;
	//std::string gnuplot_title;
	//std::string gnuplot_label_x;
	//std::string gnuplot_label_y;


	f_compare_columns = false;
	//std::string compare_columns_fname;
	//std::string compare_columns_column1;
	//std::string compare_columns_column2;

	f_gcd_worksheet = false;
	gcd_worksheet_nb_problems = 0;
	gcd_worksheet_N = 0;
	gcd_worksheet_key = false;

	f_draw_layered_graph = false;
	//draw_layered_graph_fname;
	Layered_graph_draw_options = NULL;

	f_read_gedcom = false;
	//std::string read_gedcom_fname;

	f_read_xml = false;
	//std::string read_xml_fname;
	//std::string read_xml_crossref_fname;

	f_read_column_and_tally = false;
	//std::string read_column_and_tally_fname;
	//std::string read_column_and_tally_col_header;

	f_intersect_with = false;
	//std::string intersect_with_vector;
	//std::string intersect_with_data;

	f_make_set_of_sets = false;
	//std::string make_set_of_sets_fname_in;
	//std::string make_set_of_sets_new_col_label;
	//std::string make_set_of_sets_list;

	f_copy_and_edit = false;
	//std::string copy_and_edit_input_file;
	//std::string copy_and_edit_output_mask;
	//std::string copy_and_edit_parameter_values;
	//std::string copy_and_edit_search_and_replace;

	f_join_columns = false;
	//std::string join_columns_file_in;
	//std::string join_columns_file_out;
	//std::string join_columns_column1;
	//std::string join_columns_column2;

	f_decomposition_matrix = false;
	//std::string decomposition_matrix_fname;
}


interface_toolkit::~interface_toolkit()
{
	Record_birth();
}


void interface_toolkit::print_help(
		int argc,
		std::string *argv, int i, int verbose_level)
{
	other::data_structures::string_tools ST;

	if (ST.stringcmp(argv[i], "-create_files_direct") == 0) {
		cout << "-create_files_direct <string : mask> <string : text> <string : label_1> ... <string : label_n> -end" << endl;
	}
	else if (ST.stringcmp(argv[i], "-create_files") == 0) {
		cout << "-create_files <description>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-save_matrix_csv") == 0) {
		cout << "-save_matrix_csv <string : label>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_tally") == 0) {
		cout << "-csv_file_tally <string : label>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-tally_column") == 0) {
		cout << "-tally_column <string : fname> <string : column>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-collect_stats") == 0) {
		cout << "-collect_stats <string : fname_mask> <string : fname_out> <int : first> <int : last> <int : step>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows") == 0) {
		cout << "-cvs_file_select_rows <string : csv_file_name> <string : list of rows>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows_by_file") == 0) {
		cout << "-cvs_file_select_rows_by_file <string : csv_file_name> <string : file with list of rows>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows_complement") == 0) {
		cout << "-csv_file_select_rows_complement <string : csv_file_name> <string : list of rows>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_split_rows_modulo") == 0) {
		cout << "-csv_file_split_rows_modulo <string : csv_file_name> <int : n>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_cols") == 0) {
		cout << "-cvs_file_select_cols <string : csv_file_name> <string : list of cols>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_cols_by_label") == 0) {
		cout << "-cvs_file_select_cols_by_label <string : csv_file_name> <string : fname_append> <string : list of col-labels>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows_and_cols") == 0) {
		cout << "-csv_file_select_rows_and_cols <string : csv_file_name> <string : list of rows> <string : list of cols>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_sort_each_row") == 0) {
		cout << "-csv_file_sort_each_row <string : input file>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_sort_rows") == 0) {
		cout << "-csv_file_sort_rows <string : input file>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_sort_rows_and_remove_duplicates") == 0) {
		cout << "-csv_file_sort_rows_and_remove_duplicates <string : input file>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_join") == 0) {
		cout << "-cvs_file_join <int : number of files> <string : input file1> <string : column label1> ..." << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_concatenate") == 0) {
		cout << "-csv_file_concatenate <string : fname_out> <int : number of input files> <string : input file1> ..." << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_concatenate_from_mask") == 0) {
		cout << "-csv_file_concatenate_from_mask <int : first> <int : nb_files> <string : fname_mask> <string : fname_out> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_extract_column_to_txt") == 0) {
		cout << "-csv_file_extract_column_to_txt <string : csv_fname> <string : col_label>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_filter") == 0) {
		cout << "-csv_file_filter <string : csv_fname> <string : col_label> <string : filter_value>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_latex") == 0) {
		cout << "-cvs_file_latex <int : f_produce_header> <string : file_name>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_latex") == 0) {
		cout << "-cvs_file_latex <int : f_produce_header> <string : file_name>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-prepare_tables_for_users_guide") == 0) {
		cout << "-prepare_tables_for_users_guide <fname> <fname_1> ... <fname_n> -end" << endl;
	}
	else if (ST.stringcmp(argv[i], "-prepare_general_tables_for_users_guide") == 0) {
		cout << "-prepare_general_tables_for_users_guide <fname> <fname_1> ... <fname_n> -end" << endl;
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
	else if (ST.stringcmp(argv[i], "-change_values") == 0) {
		cout << "-change_values <string : fname_in> <string : fname_out> <string : input_values> <string : output_values>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-store_as_csv_file") == 0) {
		cout << "-store_as_csv_file <string : fname> <int : m> "
				"<int : n> <string : data> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-mv") == 0) {
		cout << "-mv <string : from> <string : to> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-cp") == 0) {
		cout << "-cp <string : from> <string : to> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-system") == 0) {
		cout << "-system <string : command> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-loop") == 0) {
		cout << "-loop <string : variable> <int : from> <int : to> <int : step> <arguments> -loop_end" << endl;
	}
	else if (ST.stringcmp(argv[i], "-loop_over") == 0) {
		cout << "-loop_over <string : index_variable> <string : domain>  <loop body> -loop_end <string : index_variable>" << endl;
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
	else if (ST.stringcmp(argv[i], "-extract_from_file") == 0) {
		cout << "-extract_from_file <fname> <label> <extract_from_file_target_fname>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-extract_from_file_with_tail") == 0) {
		cout << "-extract_from_file_with_tail <fname> <label> <tail> <extract_from_file_target_fname>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-serialize_file_names") == 0) {
		cout << "-serialize_file_names <fname> <fname> <mask>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-save_4_bit_data_file") == 0) {
		cout << "-save_4_bit_data_file <fname> <vector>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-gnuplot") == 0) {
		cout << "-gnuplot <fname> <title> <label_x> <label_y> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-compare_columns") == 0) {
		cout << "-compare_columns <col1> <col2> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-gcd_worksheet") == 0) {
		cout << "-gcd_worksheet <nb_problems> <N> <f_key>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-draw_layered_graph") == 0) {
		cout << "-draw_layered_graph <string : fname> <layered_graph_options>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-read_gedcom") == 0) {
		cout << "-read_gedcom <string : fname>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-read_xml") == 0) {
		cout << "-read_xml <string : fname> <string : crossref_fname>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-read_column_and_tally") == 0) {
		cout << "-read_column_and_tally <string : fname> <string : col_header>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-intersect_with") == 0) {
		cout << "-intersect_with <string : vector> <string : data>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-make_set_of_sets") == 0) {
		cout << "-make_set_of_sets <string : fname_in> <string : new_col_label> <string : labels>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-copy_and_edit") == 0) {
		cout << "-copy_and_edit <string : fname_in> <string : fname_out_mask> <string : parameter_values> <string : search_and_replace_patterns>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-join_columns") == 0) {
		cout << "-join_columns <string : fname_in> <string : fname_out> <string : col1> <string : col2>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-decomposition_matrix") == 0) {
		cout << "-decomposition_matrix <string : fname>" << endl;
	}

}



int interface_toolkit::recognize_keyword(
		int argc,
		std::string *argv, int i, int verbose_level)
{
	other::data_structures::string_tools ST;

	if (i >= argc) {
		return false;
	}
	if (ST.stringcmp(argv[i], "-create_files_direct") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-create_files") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-save_matrix_csv") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_tally") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-tally_column") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-collect_stats") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows_by_file") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows_complement") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_split_rows_modulo") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_cols") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_cols_by_label") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows_and_cols") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_sort_each_row") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_sort_rows") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_sort_rows_and_remove_duplicates") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_join") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_concatenate") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_concatenate_from_mask") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_extract_column_to_txt") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_filter") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-csv_file_latex") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-prepare_tables_for_users_guide") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-prepare_general_tables_for_users_guide") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-grade_statistic_from_csv") == 0) {
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
	else if (ST.stringcmp(argv[i], "-change_values") == 0) {
		return true;
	}

	else if (ST.stringcmp(argv[i], "-store_as_csv_file") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-mv") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-cp") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-system") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-loop") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-loop_over") == 0) {
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
	else if (ST.stringcmp(argv[i], "-extract_from_file") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-extract_from_file_with_tail") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-serialize_file_names") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-save_4_bit_data_file") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-gnuplot") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-compare_columns") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-gcd_worksheet") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-draw_layered_graph") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-read_gedcom") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-read_xml") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-read_column_and_tally") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-intersect_with") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-make_set_of_sets") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-copy_and_edit") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-join_columns") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-decomposition_matrix") == 0) {
		return true;
	}
	return false;
}

void interface_toolkit::read_arguments(
		int argc,
		std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "interface_toolkit::read_arguments" << endl;
	}

	if (f_v) {
		cout << "interface_toolkit::read_arguments "
				"the next argument is " << argv[i] << endl;
	}
	if (ST.stringcmp(argv[i], "-create_files_direct") == 0) {
		f_create_files_direct = true;
		create_files_direct_fname_mask.assign(argv[++i]);
		create_files_direct_content_mask.assign(argv[++i]);

		string s;
		int j;

		for (j = 0; ; j++) {
			s.assign(argv[++i]);
			if (s == "-end") {
				break;
			}
			create_files_direct_labels.push_back(s);
		}
		if (f_v) {
			cout << "-create_files_direct " << create_files_direct_fname_mask
					<< " " << create_files_direct_content_mask;
			for (j = 0; j < create_files_direct_labels.size(); j++) {
				cout << " " << create_files_direct_labels[j];
			}
			cout << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-create_files") == 0) {
		f_create_files = true;

		if (f_v) {
			cout << "-create_files " << endl;
		}

		Create_file_description =
				NEW_OBJECT(other::orbiter_kernel_system::create_file_description);
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
	else if (ST.stringcmp(argv[i], "-save_matrix_csv") == 0) {
		f_save_matrix_csv = true;
		save_matrix_csv_label.assign(argv[++i]);
		if (f_v) {
			cout << "-save_matrix_csv " << save_matrix_csv_label << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_tally") == 0) {
		f_csv_file_tally = true;
		csv_file_tally_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_tally " << csv_file_tally_fname << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-tally_column") == 0) {
		f_tally_column = true;
		tally_column_fname.assign(argv[++i]);
		tally_column_column.assign(argv[++i]);
		if (f_v) {
			cout << "-tally_column "
					<< " " << tally_column_fname
					<< " " << tally_column_column
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-collect_stats") == 0) {
		f_collect_stats = true;
		collect_stats_fname_mask.assign(argv[++i]);
		collect_stats_fname_out.assign(argv[++i]);
		collect_stats_first = ST.strtoi(argv[++i]);
		collect_stats_last = ST.strtoi(argv[++i]);
		collect_stats_step = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-collect_stats"
					<< " " << collect_stats_fname_mask
					<< " " << collect_stats_fname_out
					<< " " << collect_stats_first
					<< " " << collect_stats_last
					<< " " << collect_stats_step
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows") == 0) {
		f_csv_file_select_rows = true;
		csv_file_select_rows_fname.assign(argv[++i]);
		csv_file_select_rows_text.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_select_rows "
					<< csv_file_select_rows_fname
				<< " " << csv_file_select_rows_text << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows_by_file") == 0) {
		f_csv_file_select_rows_by_file = true;
		csv_file_select_rows_by_file_fname.assign(argv[++i]);
		csv_file_select_rows_by_file_select.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_select_rows_by_file "
					<< csv_file_select_rows_by_file_fname
				<< " " << csv_file_select_rows_by_file_select << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows_complement") == 0) {
		f_csv_file_select_rows_complement = true;
		csv_file_select_rows_complement_fname.assign(argv[++i]);
		csv_file_select_rows_complement_text.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_select_rows_complement "
					<< csv_file_select_rows_complement_fname
				<< " " << csv_file_select_rows_complement_text << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_split_rows_modulo") == 0) {
		f_csv_file_split_rows_modulo = true;
		csv_file_split_rows_modulo_fname.assign(argv[++i]);
		csv_file_split_rows_modulo_n = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-csv_file_split_rows_modulo "
					<< csv_file_split_rows_modulo_fname
				<< " " << csv_file_split_rows_modulo_n << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_cols") == 0) {
		f_csv_file_select_cols = true;
		csv_file_select_cols_fname.assign(argv[++i]);
		csv_file_select_cols_fname_append.assign(argv[++i]);
		csv_file_select_cols_text.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_select_cols "
					<< csv_file_select_cols_fname
					<< " " << csv_file_select_cols_fname_append
					<< " " << csv_file_select_cols_text
				<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_cols_by_label") == 0) {
		f_csv_file_select_cols_by_label = true;
		csv_file_select_cols_by_label_fname.assign(argv[++i]);
		csv_file_select_cols_by_label_fname_append.assign(argv[++i]);
		csv_file_select_cols_by_label_text.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_select_cols_by_label "
					<< " " << csv_file_select_cols_by_label_fname
					<< " " << csv_file_select_cols_by_label_fname_append
					<< " " << csv_file_select_cols_by_label_text
				<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_select_rows_and_cols") == 0) {
		f_csv_file_select_rows_and_cols = true;
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
		f_csv_file_sort_each_row = true;
		csv_file_sort_each_row_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_sort_each_row "
				<< csv_file_sort_each_row_fname
				<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_sort_rows") == 0) {
		f_csv_file_sort_rows = true;
		csv_file_sort_rows_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_sort_rows "
				<< csv_file_sort_rows_fname
				<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_sort_rows_and_remove_duplicates") == 0) {
		f_csv_file_sort_rows_and_remove_duplicates = true;
		csv_file_sort_rows_and_remove_duplicates_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_sort_rows_and_remove_duplicates "
				<< csv_file_sort_rows_and_remove_duplicates_fname
				<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_join") == 0) {
		string s;
		int nb, j;

		f_csv_file_join = true;
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
				cout << j
						<< " : " << csv_file_join_fname[j]
					<< " : " << csv_file_join_identifier[j] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_concatenate") == 0) {
		string s;
		int nb, j;

		f_csv_file_concatenate = true;
		csv_file_concatenate_fname_out.assign(argv[++i]);
		nb = ST.strtoi(argv[++i]);
		for (j = 0; j < nb; j++) {
			s.assign(argv[++i]);
			csv_file_concatenate_fname_in.push_back(s);
		}
		if (f_v) {
			cout << "-csv_file_concatenate "
					<< csv_file_concatenate_fname_out << endl;
			for (j = 0; j < nb; j++) {
				cout << j << " : " << csv_file_concatenate_fname_in[j] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_concatenate_from_mask") == 0) {
		f_csv_file_concatenate_from_mask = true;
		csv_file_concatenate_from_mask_N_min = ST.strtoi(argv[++i]);
		csv_file_concatenate_from_mask_N_length = ST.strtoi(argv[++i]);
		csv_file_concatenate_from_mask_mask.assign(argv[++i]);
		csv_file_concatenate_from_mask_fname_out.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_concatenate_from_mask "
					<< " " << csv_file_concatenate_from_mask_N_min
					<< " " << csv_file_concatenate_from_mask_N_length
					<< " " << csv_file_concatenate_from_mask_mask
					<< " " << csv_file_concatenate_from_mask_fname_out
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_extract_column_to_txt") == 0) {
		f_csv_file_extract_column_to_txt = true;
		csv_file_extract_column_to_txt_fname.assign(argv[++i]);
		csv_file_extract_column_to_txt_col_label.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_extract_column_to_txt "
					<< csv_file_extract_column_to_txt_fname
					<< " " << csv_file_extract_column_to_txt_col_label << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_filter") == 0) {
		f_csv_file_filter = true;
		csv_file_filter_fname.assign(argv[++i]);
		csv_file_filter_col.assign(argv[++i]);
		csv_file_filter_value.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_filter "
					<< csv_file_filter_fname
					<< " " << csv_file_filter_col
					<< " " << csv_file_filter_value
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-csv_file_latex") == 0) {
		f_csv_file_latex = true;
		f_produce_latex_header = ST.strtoi(argv[++i]);
		csv_file_latex_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-csv_file_latex " << f_produce_latex_header
					<< " " << csv_file_latex_fname << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-prepare_tables_for_users_guide") == 0) {
		f_prepare_tables_for_users_guide = true;

		string s;
		int j;

		for (j = 0; ; j++) {
			s.assign(argv[++i]);
			if (s == "-end") {
				break;
			}
			prepare_tables_for_users_guide_fname.push_back(s);
		}
		if (f_v) {
			cout << "-prepare_tables_for_users_guide ";
			for (j = 0; j < prepare_tables_for_users_guide_fname.size(); j++) {
				cout << " " << prepare_tables_for_users_guide_fname[j];
			}
			cout << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-prepare_general_tables_for_users_guide") == 0) {
		f_prepare_general_tables_for_users_guide = true;

		string s;
		int j;

		for (j = 0; ; j++) {
			s.assign(argv[++i]);
			if (s == "-end") {
				break;
			}
			prepare_general_tables_for_users_guide_fname.push_back(s);
		}
		if (f_v) {
			cout << "-prepare_general_tables_for_users_guide ";
			for (j = 0; j < prepare_general_tables_for_users_guide_fname.size(); j++) {
				cout << " " << prepare_general_tables_for_users_guide_fname[j];
			}
			cout << endl;
		}
	}

	else if (ST.stringcmp(argv[i], "-csv_file_concatenate") == 0) {
		string s;
		int nb, j;

		f_csv_file_concatenate = true;
		csv_file_concatenate_fname_out.assign(argv[++i]);
		nb = ST.strtoi(argv[++i]);
		for (j = 0; j < nb; j++) {
			s.assign(argv[++i]);
			csv_file_concatenate_fname_in.push_back(s);
		}
		if (f_v) {
			cout << "-csv_file_concatenate "
					<< csv_file_concatenate_fname_out << endl;
			for (j = 0; j < nb; j++) {
				cout << j << " : " << csv_file_concatenate_fname_in[j] << endl;
			}
		}
	}

	else if (ST.stringcmp(argv[i], "-grade_statistic_from_csv") == 0) {
		f_grade_statistic_from_csv = true;
		grade_statistic_from_csv_fname.assign(argv[++i]);
		grade_statistic_from_csv_m1_label.assign(argv[++i]);
		grade_statistic_from_csv_m2_label.assign(argv[++i]);
		grade_statistic_from_csv_final_label.assign(argv[++i]);
		grade_statistic_from_csv_oracle_grade_label.assign(argv[++i]);
		if (f_v) {
			cout << "-grade_statistic_from_csv "
					<< " " << grade_statistic_from_csv_fname
					<< " " << grade_statistic_from_csv_m1_label
					<< " " << grade_statistic_from_csv_m2_label
					<< " " << grade_statistic_from_csv_final_label
					<< " " << grade_statistic_from_csv_oracle_grade_label
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-draw_matrix") == 0) {
		f_draw_matrix = true;
		Draw_bitmap_control = NEW_OBJECT(other::graphics::draw_bitmap_control);
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
		f_reformat = true;
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
		f_split_by_values = true;
		split_by_values_fname_in.assign(argv[++i]);
		if (f_v) {
			cout << "-split_by_values "
					<< split_by_values_fname_in << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-change_values") == 0) {
		f_change_values = true;
		change_values_fname_in.assign(argv[++i]);
		change_values_fname_out.assign(argv[++i]);
		change_values_function_input.assign(argv[++i]);
		change_values_function_output.assign(argv[++i]);
		if (f_v) {
			cout << "-split_by_values "
					<< " " << change_values_fname_in
					<< " " << change_values_fname_out
					<< " " << change_values_function_input
					<< " " << change_values_function_output
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-store_as_csv_file") == 0) {
		f_store_as_csv_file = true;
		store_as_csv_file_fname.assign(argv[++i]);
		store_as_csv_file_m = ST.strtoi(argv[++i]);
		store_as_csv_file_n = ST.strtoi(argv[++i]);
		store_as_csv_file_data.assign(argv[++i]);
		if (f_v) {
			cout << "-store_as_csv_file "
					<< store_as_csv_file_fname
				<< " " << store_as_csv_file_m
				<< " " << store_as_csv_file_n
				<< " " << store_as_csv_file_data << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-mv") == 0) {
		f_mv = true;
		mv_a.assign(argv[++i]);
		mv_b.assign(argv[++i]);
		if (f_v) {
			cout << "-mv " << mv_a
				<< " " << mv_b << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-cp") == 0) {
		f_cp = true;
		cp_a.assign(argv[++i]);
		cp_b.assign(argv[++i]);
		if (f_v) {
			cout << "-cp " << cp_a
				<< " " << cp_b << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-system") == 0) {
		f_system = true;
		system_command.assign(argv[++i]);
		if (f_v) {
			cout << "-system " << system_command << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-loop") == 0) {
		f_loop = true;
		loop_start_idx = i + 5;
		loop_variable.assign(argv[++i]);
		loop_from = ST.strtoi(argv[++i]);
		loop_to = ST.strtoi(argv[++i]);
		loop_step = ST.strtoi(argv[++i]);
		loop_argv = argv;

		for (++i; i < argc - 1; i++) {
			if (ST.stringcmp(argv[i], "-end_loop") == 0
					&& ST.stringcmp(argv[i + 1], loop_variable.c_str()) == 0) {
				cout << "found -end_loop " << loop_variable
						<< " at i=" << i << " argc=" << argc << endl;
				loop_end_idx = i + 1;
				break;
			}
		}
		if (i == argc - 1) {
			cout << "-loop cannot find -end_loop " << loop_variable << " in:" << endl;
			for (int j = loop_start_idx; j < argc; j++) {
				cout << " " << argv[j];
			}
			cout << endl;
			exit(1);
		}
		i++;
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
	else if (ST.stringcmp(argv[i], "-loop_over") == 0) {
		f_loop_over = true;
		loop_over_start_idx = i + 3;
		loop_over_variable.assign(argv[++i]);
		//loop_over_index.assign(argv[++i]);
		loop_over_domain.assign(argv[++i]);
		loop_over_argv = argv;

		for (++i; i < argc - 1; i++) {
			if (ST.stringcmp(argv[i], "-end_loop_over") == 0
					&& ST.stringcmp(argv[i + 1], loop_over_variable.c_str()) == 0) {
				loop_over_end_idx = i + 1;
				break;
			}
		}
		if (i == argc - 1) {
			cout << "-loop_over cannot find -end_loop_over <variable>, "
					"looking for variable " << loop_over_variable << endl;
			exit(1);
		}
		i++;
		if (f_v) {
			cout << "-loop_over"
					<< " " << loop_over_variable
					//<< " " << loop_over_index
					<< " " << loop_over_domain
					<< " " << loop_over_start_idx
					<< " " << loop_over_end_idx;

			for (int j = loop_over_start_idx; j < loop_over_end_idx; j++) {
				cout << " " << argv[j];
			}
			cout << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-plot_function") == 0) {
		f_plot_function = true;
		plot_function_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-plot_function " << plot_function_fname << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-draw_projective_curve") == 0) {
		f_draw_projective_curve = true;
		Draw_projective_curve_description =
				NEW_OBJECT(other::graphics::draw_projective_curve_description);
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
		f_tree_draw = true;
		Tree_draw_options = NEW_OBJECT(other::graphics::tree_draw_options);
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
	else if (ST.stringcmp(argv[i], "-extract_from_file") == 0) {
		f_extract_from_file = true;
		extract_from_file_fname.assign(argv[++i]);
		extract_from_file_label.assign(argv[++i]);
		extract_from_file_target_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-extract_from_file "
					<< extract_from_file_fname
					<< " " << extract_from_file_label
					<< " " << extract_from_file_target_fname
						<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-extract_from_file_with_tail") == 0) {
		f_extract_from_file_with_tail = true;
		extract_from_file_with_tail_fname.assign(argv[++i]);
		extract_from_file_with_tail_label.assign(argv[++i]);
		extract_from_file_with_tail_tail.assign(argv[++i]);
		extract_from_file_with_tail_target_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-extract_from_file_with_tail "
					<< extract_from_file_with_tail_fname
					<< " " << extract_from_file_with_tail_label
					<< " " << extract_from_file_with_tail_tail
					<< " " << extract_from_file_with_tail_target_fname
						<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-serialize_file_names") == 0) {
		f_serialize_file_names = true;
		serialize_file_names_fname.assign(argv[++i]);
		serialize_file_names_output_mask.assign(argv[++i]);
		if (f_v) {
			cout << "-serialize_file_names "
					<< serialize_file_names_fname
					<< " " << serialize_file_names_output_mask << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-save_4_bit_data_file") == 0) {
		f_save_4_bit_data_file = true;
		save_4_bit_data_file_fname.assign(argv[++i]);
		save_4_bit_data_file_vector_data.assign(argv[++i]);
		if (f_v) {
			cout << "-save_4_bit_data_file "
					<< save_4_bit_data_file_fname
					<< " " << save_4_bit_data_file_vector_data << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-gnuplot") == 0) {
		f_gnuplot = true;
		gnuplot_file_fname.assign(argv[++i]);
		gnuplot_title.assign(argv[++i]);
		gnuplot_label_x.assign(argv[++i]);
		gnuplot_label_y.assign(argv[++i]);
		if (f_v) {
			cout << "-gnuplot "
					<< " " << gnuplot_file_fname
					<< " " << gnuplot_title
					<< " " << gnuplot_label_x
					<< " " << gnuplot_label_y
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-compare_columns") == 0) {
		f_compare_columns = true;
		compare_columns_fname.assign(argv[++i]);
		compare_columns_column1.assign(argv[++i]);
		compare_columns_column2.assign(argv[++i]);
		if (f_v) {
			cout << "-compare_columns "
					<< " " << compare_columns_fname
					<< " " << compare_columns_column1
					<< " " << compare_columns_column2
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-gcd_worksheet") == 0) {
		f_gcd_worksheet = true;
		gcd_worksheet_nb_problems = ST.strtoi(argv[++i]);
		gcd_worksheet_N = ST.strtoi(argv[++i]);
		gcd_worksheet_key = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-gcd_worksheet "
					<< " " << gcd_worksheet_nb_problems
					<< " " << gcd_worksheet_N
					<< " " << gcd_worksheet_key
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-draw_layered_graph") == 0) {
		f_draw_layered_graph = true;
		draw_layered_graph_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-draw_layered_graph " << endl;
		}
		Layered_graph_draw_options = NEW_OBJECT(other::graphics::layered_graph_draw_options);
		i += Layered_graph_draw_options->read_arguments(argc - i - 1,
				argv + i + 1, verbose_level);
		if (f_v) {
			cout << "interface_combinatorics::read_arguments "
					"finished reading -draw_layered_graph" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-read_gedcom") == 0) {
		f_read_gedcom = true;
		read_gedcom_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-read_gedcom "
					<< " " << read_gedcom_fname
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-read_xml") == 0) {
		f_read_xml = true;
		read_xml_fname.assign(argv[++i]);
		read_xml_crossref_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-read_xml "
					<< " " << read_xml_fname
					<< " " << read_xml_crossref_fname
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-read_column_and_tally") == 0) {
		f_read_column_and_tally = true;
		read_column_and_tally_fname.assign(argv[++i]);
		read_column_and_tally_col_header.assign(argv[++i]);
		if (f_v) {
			cout << "-read_column_and_tally "
					<< " " << read_column_and_tally_fname
					<< " " << read_column_and_tally_col_header
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-intersect_with") == 0) {
		f_intersect_with = true;
		intersect_with_vector.assign(argv[++i]);
		intersect_with_data.assign(argv[++i]);
		if (f_v) {
			cout << "-intersect_with "
					<< intersect_with_vector
				<< " " << intersect_with_data << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-make_set_of_sets") == 0) {
		f_make_set_of_sets = true;
		make_set_of_sets_fname_in.assign(argv[++i]);
		make_set_of_sets_new_col_label.assign(argv[++i]);
		make_set_of_sets_list.assign(argv[++i]);
		if (f_v) {
			cout << "-make_set_of_sets "
					<< make_set_of_sets_fname_in << " "
					<< make_set_of_sets_new_col_label << " "
					<< make_set_of_sets_list<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-copy_and_edit") == 0) {
		f_copy_and_edit = true;
		copy_and_edit_input_file.assign(argv[++i]);
		copy_and_edit_output_mask.assign(argv[++i]);
		copy_and_edit_parameter_values.assign(argv[++i]);
		copy_and_edit_search_and_replace.assign(argv[++i]);
		if (f_v) {
			cout << "-copy_and_edit "
					<< copy_and_edit_input_file << " "
					<< copy_and_edit_output_mask << " "
					<< copy_and_edit_parameter_values << " "
					<< copy_and_edit_search_and_replace << " "
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-join_columns") == 0) {
		f_join_columns = true;
		join_columns_file_in.assign(argv[++i]);
		join_columns_file_out.assign(argv[++i]);
		join_columns_column1.assign(argv[++i]);
		join_columns_column2.assign(argv[++i]);
		if (f_v) {
			cout << "-join_columns "
					<< join_columns_file_in << " "
					<< join_columns_file_out << " "
					<< join_columns_column1 << " "
					<< join_columns_column2 << " "
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-decomposition_matrix") == 0) {
		f_decomposition_matrix = true;
		decomposition_matrix_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-decomposition_matrix "
					<< decomposition_matrix_fname << " "
					<< endl;
		}
	}



	if (f_v) {
			cout << "interface_toolkit::read_arguments done" << endl;
	}
}

void interface_toolkit::print()
{
	int j;

	if (f_create_files_direct) {
		int j;

		cout << "-create_files_direct " << create_files_direct_fname_mask
				<< " " << create_files_direct_content_mask;
		for (j = 0; j < create_files_direct_labels.size(); j++) {
			cout << " " << create_files_direct_labels[j];
		}
		cout << endl;
	}
	if (f_create_files) {
		cout << "-create_files " << endl;
		Create_file_description->print();
	}
	if (f_save_matrix_csv) {
		cout << "-save_csv " << save_matrix_csv_label << endl;
	}
	if (f_csv_file_tally) {
		cout << "-csv_file_tally " << csv_file_tally_fname << endl;
	}
	if (f_tally_column) {
		cout << "-tally_column "
				<< " " << tally_column_fname
				<< " " << tally_column_column
				<< endl;
	}
	if (f_collect_stats) {
		cout << "-collect_stats"
				<< " " << collect_stats_fname_mask
				<< " " << collect_stats_fname_out
				<< " " << collect_stats_first
				<< " " << collect_stats_last
				<< " " << collect_stats_step
				<< endl;
	}
	if (f_csv_file_select_rows) {
		cout << "-csv_file_select_rows " << csv_file_select_rows_fname
				<< " " << csv_file_select_rows_text << endl;
	}
	if (f_csv_file_select_rows_by_file) {
		cout << "-csv_file_select_rows_by_file "
				<< csv_file_select_rows_by_file_fname
			<< " " << csv_file_select_rows_by_file_select << endl;
	}
	if (f_csv_file_split_rows_modulo) {
		cout << "-csv_file_split_rows_modulo "
				<< csv_file_split_rows_modulo_fname
				<< " " << csv_file_split_rows_modulo_n << endl;
	}
	if (f_csv_file_select_cols) {
		cout << "-csv_file_select_cols "
				<< csv_file_select_cols_fname
				<< " " << csv_file_select_cols_fname_append
				<< " " << csv_file_select_cols_text
			<< endl;
	}
	if (f_csv_file_select_cols_by_label) {
		cout << "-csv_file_select_cols_by_label "
				<< " " << csv_file_select_cols_by_label_fname
				<< " " << csv_file_select_cols_by_label_fname_append
				<< " " << csv_file_select_cols_by_label_text
			<< endl;
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
	if (f_csv_file_sort_rows) {
		cout << "-csv_file_sort_rows "
				<< csv_file_sort_rows_fname
				<< endl;
	}
	if (f_csv_file_sort_rows_and_remove_duplicates) {
		cout << "-csv_file_sort_rows_and_remove_duplicates "
				<< csv_file_sort_rows_and_remove_duplicates_fname
				<< endl;
	}
	if (f_csv_file_join) {
		cout << "-csv_file_join " << endl;
		for (j = 0; j < csv_file_join_fname.size(); j++) {
			cout << j << " : " << csv_file_join_fname[j]
				<< " : " << csv_file_join_identifier[j] << endl;
		}
	}
	if (f_csv_file_concatenate) {
		cout << "-csv_file_concatenate " << csv_file_concatenate_fname_out << endl;
		for (j = 0; j < csv_file_concatenate_fname_in.size(); j++) {
			cout << j << " : " << csv_file_concatenate_fname_in[j] << endl;
		}
	}
	if (f_csv_file_concatenate_from_mask) {
		cout << "-csv_file_concatenate_from_mask "
				<< " " << csv_file_concatenate_from_mask_N_min
				<< " " << csv_file_concatenate_from_mask_N_length
				<< " " << csv_file_concatenate_from_mask_mask
				<< " " << csv_file_concatenate_from_mask_fname_out
				<< endl;
	}
	if (f_csv_file_extract_column_to_txt) {
		cout << "-csv_file_extract_column_to_txt "
				<< csv_file_extract_column_to_txt_fname
				<< " " << csv_file_extract_column_to_txt_col_label << endl;
	}
	if (f_csv_file_filter) {
		cout << "-csv_file_filter "
				<< csv_file_filter_fname
				<< " " << csv_file_filter_col
				<< " " << csv_file_filter_value
				<< endl;
	}
	if (f_csv_file_latex) {
		cout << "-csv_file_latex "
				<< f_produce_latex_header
				<< " " << csv_file_latex_fname << endl;
	}
	if (f_prepare_tables_for_users_guide) {
		int j;

		cout << "-prepare_tables_for_users_guide ";
		for (j = 0; j < prepare_tables_for_users_guide_fname.size(); j++) {
			cout << " " << prepare_tables_for_users_guide_fname[j];
		}
		cout << endl;

	}
	if (f_prepare_general_tables_for_users_guide) {
		int j;

		cout << "-prepare_general_tables_for_users_guide ";
		for (j = 0; j < prepare_general_tables_for_users_guide_fname.size(); j++) {
			cout << " " << prepare_general_tables_for_users_guide_fname[j];
		}
		cout << endl;

	}
	if (f_grade_statistic_from_csv) {
		cout << "-grade_statistic_from_csv "
				<< " " << grade_statistic_from_csv_fname
				<< " " << grade_statistic_from_csv_m1_label
				<< " " << grade_statistic_from_csv_m2_label
				<< " " << grade_statistic_from_csv_final_label
				<< " " << grade_statistic_from_csv_oracle_grade_label
				<< endl;
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
	if (f_change_values) {
		cout << "-split_by_values " << change_values_fname_in
				<< " " << change_values_function_input
				<< " " << change_values_function_output
				<< endl;
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
	if (f_cp) {
		cout << "-cp " << cp_a
				<< " " << cp_b << endl;
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
	if (f_loop_over) {
		cout << "-loop_over"
				<< " " << loop_over_variable
				//<< " " << loop_over_index
				<< " " << loop_over_domain
				<< " " << loop_over_start_idx
				<< " " << loop_over_end_idx;

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
	if (f_extract_from_file) {
		cout << "-extract_from_file "
				<< extract_from_file_fname
				<< " " << extract_from_file_label
				<< " " << extract_from_file_target_fname
					<< endl;
	}
	if (f_extract_from_file_with_tail) {
		cout << "-extract_from_file_with_tail "
				<< extract_from_file_with_tail_fname
				<< " " << extract_from_file_with_tail_label
				<< " " << extract_from_file_with_tail_tail
				<< " " << extract_from_file_with_tail_target_fname
					<< endl;
	}
	if (f_serialize_file_names) {
		cout << "-serialize_file_names "
				<< serialize_file_names_fname
				<< " " << serialize_file_names_output_mask << endl;
	}
	if (f_save_4_bit_data_file) {
			cout << "-save_4_bit_data_file "
					<< save_4_bit_data_file_fname
					<< " " << save_4_bit_data_file_vector_data << endl;
	}
	if (f_gnuplot) {
			cout << "-gnuplot "
					<< " " << gnuplot_file_fname
					<< " " << gnuplot_title
					<< " " << gnuplot_label_x
					<< " " << gnuplot_label_y
					<< endl;
	}
	if (f_compare_columns) {
			cout << "-compare_columns "
					<< " " << compare_columns_fname
					<< " " << compare_columns_column1
					<< " " << compare_columns_column2
					<< endl;
	}
	if (f_gcd_worksheet) {
		cout << "-gcd_worksheet "
				<< " " << gcd_worksheet_nb_problems
				<< " " << gcd_worksheet_N
				<< " " << gcd_worksheet_key
				<< endl;
	}
	if (f_draw_layered_graph) {
		cout << "-draw_layered_graph " << endl;
		Layered_graph_draw_options->print();
	}
	if (f_read_gedcom) {
		cout << "-read_gedcom "
				<< " " << read_gedcom_fname
				<< endl;
	}
	if (f_read_xml) {
		cout << "-read_xml "
				<< " " << read_xml_fname
				<< " " << read_xml_crossref_fname
				<< endl;
	}
	if (f_read_column_and_tally) {
		cout << "-read_column_and_tally "
				<< " " << read_column_and_tally_fname
				<< " " << read_column_and_tally_col_header
				<< endl;
	}
	if (f_intersect_with) {
		cout << "-intersect_with "
				<< intersect_with_vector
			<< " " << intersect_with_data << endl;
	}
	if (f_make_set_of_sets) {
		cout << "-make_set_of_sets "
				<< make_set_of_sets_fname_in << " "
				<< make_set_of_sets_new_col_label << " "
				<< make_set_of_sets_list<< endl;
	}
	if (f_join_columns) {
		cout << "-join_columns "
				<< join_columns_file_in << " "
				<< join_columns_file_out << " "
				<< join_columns_column1 << " "
				<< join_columns_column2 << " "
				<< endl;
	}
	if (f_decomposition_matrix) {
		cout << "-decomposition_matrix "
				<< decomposition_matrix_fname << " "
				<< endl;
	}
}

void interface_toolkit::worker(
		int verbose_level)
// called from orbiter_command::execute
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_toolkit::worker" << endl;
	}

	if (f_create_files_direct) {

		if (f_v) {
			cout << "interface_toolkit::worker f_create_files_direct" << endl;
		}


		other::orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "-create_files_direct " << create_files_direct_fname_mask
					<< " " << create_files_direct_content_mask;
		}

		Fio.create_files_direct(
				create_files_direct_fname_mask,
				create_files_direct_content_mask,
				create_files_direct_labels,
				verbose_level);

#if 0
		int j;

		cout << "-create_files_direct " << create_files_direct_fname_mask << " " << create_files_direct_text;
		for (j = 0; j < create_files_direct_labels.size(); j++) {
			cout << " " << create_files_direct_labels[j];
		}
		cout << endl;
#endif

	}
	else if (f_create_files) {

		if (f_v) {
			cout << "interface_toolkit::worker f_create_files" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		Fio.create_file(Create_file_description, verbose_level);
	}
	else if (f_save_matrix_csv) {

		if (f_v) {
			cout << "interface_toolkit::worker f_save_matrix_csv" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;
		int *v;
		int m, n;
		string fname;

		fname = save_matrix_csv_label + "_matrix.csv";

		Get_matrix(save_matrix_csv_label, v, m, n);
		Fio.Csv_file_support->int_matrix_write_csv(
				fname, v, m, n);

		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}
	else if (f_csv_file_tally) {

		if (f_v) {
			cout << "interface_toolkit::worker -csv_file_tally" << endl;
		}

		cout << "-csv_file_tally " << csv_file_tally_fname << endl;

		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->read_csv_file_and_tally(
				csv_file_tally_fname, verbose_level);

	}
	else if (f_tally_column) {

		if (f_v) {
			cout << "interface_toolkit::worker -tally_column" << endl;
		}

		cout << "-tally_column " << tally_column_fname << endl;

		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->tally_column(
				tally_column_fname, tally_column_column, verbose_level);

	}
	else if (f_collect_stats) {

		if (f_v) {
			cout << "interface_toolkit::worker -f_collect_stats" << endl;
		}

		cout << "-f_collect_stats " << collect_stats_fname_mask << endl;

		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->collect_stats(
				collect_stats_fname_mask,
				collect_stats_fname_out,
				collect_stats_first, collect_stats_last, collect_stats_step,
				verbose_level);

	}


	else if (f_csv_file_select_rows) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_csv_file_select_rows" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->do_csv_file_select_rows(
				csv_file_select_rows_fname,
				csv_file_select_rows_text,
				verbose_level);
	}
	else if (f_csv_file_select_rows_by_file) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_csv_file_select_rows_by_file" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->do_csv_file_select_rows_by_file(
				csv_file_select_rows_by_file_fname,
				csv_file_select_rows_by_file_select,
				verbose_level);
	}
	else if (f_csv_file_select_rows_complement) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_csv_file_select_rows" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->do_csv_file_select_rows_complement(
				csv_file_select_rows_complement_fname,
				csv_file_select_rows_complement_text,
				verbose_level);
	}
	else if (f_csv_file_split_rows_modulo) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_csv_file_split_rows_modulo" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->do_csv_file_split_rows_modulo(
				csv_file_split_rows_modulo_fname,
				csv_file_split_rows_modulo_n, verbose_level);
	}
	else if (f_csv_file_select_cols) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_csv_file_select_cols" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->do_csv_file_select_cols(
				csv_file_select_cols_fname,
				csv_file_select_cols_fname_append,
				csv_file_select_cols_text,
				verbose_level);
	}
	else if (f_csv_file_select_cols_by_label) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_csv_file_select_cols_by_label" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->do_csv_file_select_cols_by_label(
				csv_file_select_cols_by_label_fname,
				csv_file_select_cols_by_label_fname_append,
				csv_file_select_cols_by_label_text,
				verbose_level);

	}


	else if (f_csv_file_select_rows_and_cols) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_csv_file_select_rows_and_cols" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->do_csv_file_select_rows_and_cols(
				csv_file_select_rows_and_cols_fname,
				csv_file_select_rows_and_cols_R_text,
				csv_file_select_rows_and_cols_C_text,
				verbose_level);
	}
	else if (f_csv_file_sort_each_row) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_csv_file_sort_each_row" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->do_csv_file_sort_each_row(
				csv_file_sort_each_row_fname, verbose_level);

	}
	else if (f_csv_file_sort_rows) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_csv_file_sort_rows" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->csv_file_sort_rows(
				csv_file_sort_rows_fname, verbose_level);

	}
	else if (f_csv_file_sort_rows_and_remove_duplicates) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_csv_file_sort_rows_and_remove_duplicates" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->csv_file_sort_rows_and_remove_duplicates(
				csv_file_sort_rows_and_remove_duplicates_fname, verbose_level);

	}

	else if (f_csv_file_join) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_csv_file_join" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->do_csv_file_join(
				csv_file_join_fname,
				csv_file_join_identifier, verbose_level);
	}
	else if (f_csv_file_concatenate) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_csv_file_concatenate" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->do_csv_file_concatenate(
				csv_file_concatenate_fname_in,
				csv_file_concatenate_fname_out, verbose_level);
	}
	else if (f_csv_file_concatenate_from_mask) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_csv_file_concatenate_from_mask" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->do_csv_file_concatenate_from_mask(
				csv_file_concatenate_from_mask_mask,
				csv_file_concatenate_from_mask_N_min,
				csv_file_concatenate_from_mask_N_length,
				csv_file_concatenate_from_mask_fname_out,
				verbose_level);
	}
	else if (f_csv_file_extract_column_to_txt) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_csv_file_extract_column_to_txt" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->do_csv_file_extract_column_to_txt(
				csv_file_extract_column_to_txt_fname,
				csv_file_extract_column_to_txt_col_label,
				verbose_level);

	}
	else if (f_csv_file_filter) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_csv_file_filter" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->do_csv_file_filter(
				csv_file_filter_fname,
				csv_file_filter_col,
				csv_file_filter_value,
				verbose_level);

	}
	else if (f_csv_file_latex) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_csv_file_latex" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;
		int nb_lines_per_table = 40;

		Fio.Csv_file_support->do_csv_file_latex(
				csv_file_latex_fname,
				f_produce_latex_header,
				nb_lines_per_table,
				verbose_level);
	}
	else if (f_prepare_tables_for_users_guide) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_prepare_tables_for_users_guide" << endl;
		}

		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->prepare_tables_for_users_guide(
				prepare_tables_for_users_guide_fname,
				verbose_level);
	}

	else if (f_prepare_general_tables_for_users_guide) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_prepare_general_tables_for_users_guide" << endl;
		}

		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->prepare_general_tables_for_users_guide(
				prepare_general_tables_for_users_guide_fname,
				verbose_level);
	}

	else if (f_grade_statistic_from_csv) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_grade_statistic_from_csv" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;
		int f_midterm1 = true;
		int f_midterm2 = true;
		int f_final = true;
		int f_oracle_grade = true;

		Fio.Csv_file_support->grade_statistic_from_csv(
				grade_statistic_from_csv_fname,
				f_midterm1, grade_statistic_from_csv_m1_label,
				f_midterm2, grade_statistic_from_csv_m2_label,
				f_final, grade_statistic_from_csv_final_label,
				f_oracle_grade,
				grade_statistic_from_csv_oracle_grade_label,
				verbose_level);

	}
	else if (f_draw_matrix) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_draw_matrix" << endl;
		}
		other::l1_interfaces::easy_BMP_interface BMP;

		BMP.draw_bitmap(Draw_bitmap_control, verbose_level);

		FREE_int(Draw_bitmap_control->M);
	}
	else if (f_reformat) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_reformat" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;
		int *M;
		int *M2;
		int m, n;
		int len;
		int m2;

		Fio.Csv_file_support->int_matrix_read_csv(
				reformat_fname_in, M, m, n, verbose_level);
		len = m * n;
		m2 = (len + reformat_nb_cols - 1) / reformat_nb_cols;
		M2 = NEW_int(m2 * reformat_nb_cols);
		Int_vec_zero(M2, m2 * reformat_nb_cols);
		Int_vec_copy(M, M2, len);
		Fio.Csv_file_support->int_matrix_write_csv(
				reformat_fname_out, M2, m2, reformat_nb_cols);
		cout << "Written file " << reformat_fname_out
				<< " of size "
				<< Fio.file_size(reformat_fname_out) << endl;
	}
	else if (f_split_by_values) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_split_by_values" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "interface_toolkit::worker "
					"before Fio.Csv_file_support->split_by_values" << endl;
		}
		Fio.Csv_file_support->split_by_values(
				split_by_values_fname_in, verbose_level);
		if (f_v) {
			cout << "interface_toolkit::worker "
					"after Fio.Csv_file_support->split_by_values" << endl;
		}


	}
	else if (f_change_values) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_change_values" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "interface_toolkit::worker "
					"before Fio.Csv_file_support->change_values" << endl;
		}
		Fio.Csv_file_support->change_values(
				change_values_fname_in,
				change_values_fname_out,
				change_values_function_input,
				change_values_function_output,
				verbose_level);
		if (f_v) {
			cout << "interface_toolkit::worker "
					"after Fio.Csv_file_support->change_values" << endl;
		}


	}
	else if (f_store_as_csv_file) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_store_as_csv_file" << endl;
		}
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
		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->lint_matrix_write_csv(
				store_as_csv_file_fname, D,
				store_as_csv_file_m, store_as_csv_file_n);
		cout << "Written file " << store_as_csv_file_fname
				<< " of size "
				<< Fio.file_size(store_as_csv_file_fname) << endl;
	}
	else if (f_mv) {

		if (f_v) {
			cout << "interface_toolkit::worker f_mv" << endl;
		}
		string cmd;

		cmd = "mv " + mv_a + " " + mv_b;
		cout << "executing " << cmd << endl;
		system(cmd.c_str());
	}
	else if (f_cp) {

		if (f_v) {
			cout << "interface_toolkit::worker f_cp" << endl;
		}
		string cmd;

		cmd = "cp " + cp_a + " " + cp_b;
		cout << "executing " << cmd << endl;
		system(cmd.c_str());
	}
	else if (f_system) {

		if (f_v) {
			cout << "interface_toolkit::worker f_system" << endl;
		}
		string cmd;

		cmd = system_command;
		cout << "executing " << cmd << endl;
		system(cmd.c_str());
	}
	else if (f_loop) {

		if (f_v) {
			cout << "interface_toolkit::worker f_loop" << endl;
		}
		std::string *argv2;
		int argc2;
		int j;

		argc2 = loop_end_idx - loop_start_idx;
		int h, s;

		for (h = loop_from; h < loop_to; h += loop_step) {
			cout << "loop h=" << h << ":" << endl;
			argv2 = new string[argc2];
			for (j = loop_start_idx, s = 0; j < loop_end_idx; j++, s++) {

				string arg;
				string value_h;
				string variable;

				arg.assign(loop_argv[j]);
				value_h = std::to_string(h);
				variable = "%" + loop_variable;

				while (arg.find(variable) != std::string::npos) {
					arg.replace(arg.find(variable), variable.length(), value_h);
				}
				argv2[s].assign(arg);
			}
			cout << "loop iteration "
					<< h << ", executing sequence of length " << argc2 << " : ";
			for (s = 0; s < argc2; s++) {
				cout << " " << argv2[s];
			}
			cout << endl;


			The_Orbiter_top_level_session->parse_and_execute(
					argc2 - 1, argv2, 0, verbose_level);

			cout << "loop iteration " << h << "done" << endl;

			delete [] argv2;
		}

	}

	else if (f_loop_over) {

		if (f_v) {
			cout << "interface_toolkit::worker f_loop_over" << endl;
		}
		std::string *argv2;
		int argc2;
		int j;

		argc2 = loop_over_end_idx - loop_over_start_idx;
		int h, s;

		long int *Domain;
		int sz;

		Get_lint_vector_from_label(loop_over_domain, Domain, sz, 0 /* verbose_level */);

		for (h = 0; h < sz; h++) {
			cout << "loop_over iteration "
					"h=" << h << " / " << sz << " value=" << Domain[h] << endl;
			argv2 = new string[argc2];
			for (j = loop_over_start_idx, s = 0; j < loop_over_end_idx; j++, s++) {

				string arg;
				string token;
				string value;

				arg.assign(loop_over_argv[j]);

				value = std::to_string(h);

				token = "%" + loop_over_variable;

				size_t pos, pos1, pos2;
				int f_square_bracket;
				string index_object;

				while ((pos = arg.find(token)) != std::string::npos) {
					f_square_bracket = false;
					if ((pos1 = arg.find('[', pos + token.length())) != std::string::npos) {

						if ((pos2 = arg.find(']', pos1 + 1)) != std::string::npos) {
							f_square_bracket = true;
							index_object = arg.substr(pos + token.length() + 1, pos2 - pos1 - 1);
						}
						else {
							cout << "found opening square bracket "
									"but not a corresponding closing one." << endl;
							exit(1);
						}
					}
					if (f_square_bracket) {
						cout << "square_bracket of " << index_object << endl;


						long int *v;
						int sz_v;

						Get_lint_vector_from_label(
								index_object, v, sz_v, 0 /* verbose_level */);
						cout << "found object of length " << sz_v << endl;

						if (h >= sz_v) {
							cout << "access error: index is out of range" << endl;
							cout << "index = " << h << endl;
							cout << "object size = " << sz_v << endl;
							exit(1);
						}
						value = std::to_string(v[h]);

						arg.replace(pos, pos2 - pos + 1, value);
					}
					else {
						value = std::to_string(h);

						arg.replace(pos, token.length(), value);

					}
				}

				argv2[s].assign(arg);
			}
		cout << "loop_over iteration " << h
				<< ", executing sequence of length " << argc2 << " : ";
		for (s = 0; s < argc2; s++) {
			cout << " " << argv2[s];
		}
		cout << endl;


		The_Orbiter_top_level_session->parse_and_execute(
				argc2 - 1, argv2, 0, verbose_level);

		cout << "loop_over iteration " << h << " / " << sz << "done" << endl;

		delete [] argv2;
		} // next h
	}

	else if (f_plot_function) {

		if (f_v) {
			cout << "interface_toolkit::worker f_plot_function" << endl;
		}
		other::orbiter_kernel_system::file_io Fio;
		other::data_structures::string_tools ST;
		int *T;
		int *M;
		int m1, n1, x, y;

		Fio.Csv_file_support->int_matrix_read_csv(
				plot_function_fname, T, m1, n1, verbose_level);



		M = NEW_int(m1 * m1);
		Int_vec_zero(M, m1 * m1);


		for (x = 0; x < m1; x++) {

			y = T[x];
			M[(m1 - 1 - y) * m1 + x] = 1;
		}
		string fname;

		fname = plot_function_fname;
		ST.chop_off_extension(fname);
		fname += "_graph.csv";

		Fio.Csv_file_support->int_matrix_write_csv(fname, M, m1, m1);
		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;

	}
	else if (f_draw_projective_curve) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_draw_projective_curve" << endl;
		}
		other::graphics::graphical_output GO;

		GO.draw_projective_curve(
				Draw_projective_curve_description,
				verbose_level);

	}
	else if (f_tree_draw) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_tree_draw" << endl;
		}
		other::graphics::graphical_output GO;

		GO.tree_draw(Tree_draw_options, verbose_level);

	}
	else if (f_extract_from_file) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_extract_from_file" << endl;
		}
		cout << "-extract_from_file "
				<< extract_from_file_fname
				<< " " << extract_from_file_label << endl;
		other::orbiter_kernel_system::file_io Fio;
		std::vector<std::string> text;
		int i;
		std::string tail;

		Fio.extract_from_makefile(extract_from_file_fname,
				extract_from_file_label,
				false /*  f_tail */, tail,
				text,
				verbose_level);

		cout << "We have extracted "
				<< text.size() << " lines of text:" << endl;
		for (i = 0; i < text.size(); i++) {
			cout << i << " : " << text[i] << endl;
		}
		{
			std::ofstream fp_out(extract_from_file_target_fname);
			for (i = 0; i < text.size(); i++) {
				fp_out << text[i] << endl;
			}
		}
		cout << "Written file "
				<< extract_from_file_target_fname
				<< " of size "
				<< Fio.file_size(extract_from_file_target_fname) << endl;

	}
	else if (f_extract_from_file_with_tail) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_extract_from_file_with_tail" << endl;
		}
		cout << "-extract_from_file_with_tail "
				<< extract_from_file_with_tail_fname
				<< " " << extract_from_file_with_tail_label << endl;
		other::orbiter_kernel_system::file_io Fio;
		std::vector<std::string> text;
		int i;

		Fio.extract_from_makefile(
				extract_from_file_with_tail_fname,
				extract_from_file_with_tail_label,
				true /*  f_tail */, extract_from_file_with_tail_tail,
				text,
				verbose_level);

		cout << "We have extracted "
				<< text.size() << " lines of text:" << endl;
		for (i = 0; i < text.size(); i++) {
			cout << i << " : " << text[i] << endl;
		}
		{
			std::ofstream fp_out(
					extract_from_file_with_tail_target_fname);
			for (i = 0; i < text.size(); i++) {
				fp_out << text[i] << endl;
			}
		}
		cout << "Written file "
				<< extract_from_file_with_tail_target_fname
				<< " of size "
				<< Fio.file_size(extract_from_file_with_tail_target_fname) << endl;

	}
	else if (f_serialize_file_names) {
		if (f_v) {
			cout << "interface_toolkit::worker "
					"-serialize_file_names "
					<< serialize_file_names_fname << endl;
		}
		other::orbiter_kernel_system::file_io Fio;


		int nb_files;

		Fio.serialize_file_names(
				serialize_file_names_fname,
				serialize_file_names_output_mask,
			nb_files,
			verbose_level);



	}
	else if (f_save_4_bit_data_file) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"-save_4_bit_data_file "
					<< save_4_bit_data_file_fname
					<< " " << save_4_bit_data_file_vector_data << endl;
		}

		long int *Data;
		int sz;

		Get_lint_vector_from_label(
				save_4_bit_data_file_vector_data,
				Data, sz, 0 /* verbose_level */);

		other::data_structures::algorithms Algo;

		unsigned char *data_long;
		unsigned char *data;

		data_long = (unsigned char *) NEW_char(sz);
		data = (unsigned char *) NEW_char(sz);

		int i, sz1;

		for (i = 0; i < sz; i++) {
			data_long[i] = (unsigned char) Data[i];
		}

		if (f_v) {
			cout << "interface_toolkit::worker sz = " << sz << endl;
			for (i = 0; i < sz; i++) {
				cout << i << " : " << (int) data_long[i] << endl;
			}
		}



		if (f_v) {
			cout << "interface_toolkit::worker before compress" << endl;
		}

		Algo.uchar_compress_4(data_long, data, sz);

		if (f_v) {
			cout << "interface_toolkit::worker after compress" << endl;
		}

		sz1 = (sz + 1) / 2;

		if (f_v) {
			cout << "interface_toolkit::worker sz1 = " << sz1 << endl;
			for (i = 0; i < sz; i++) {
				cout << i << " : " << (int) data[i] << endl;
			}
		}



		other::orbiter_kernel_system::file_io Fio;

		{
			ofstream ost(save_4_bit_data_file_fname, ios::binary);

			ost.write((char *)data, sz1);
		}

		cout << "Written file " << save_4_bit_data_file_fname << " of size "
				<< Fio.file_size(save_4_bit_data_file_fname) << endl;

	}
	else if (f_gnuplot) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"-gnuplot "
					<< " " << gnuplot_file_fname
					<< " " << gnuplot_title
					<< " " << gnuplot_label_x
					<< " " << gnuplot_label_y
					<< endl;
		}

		other::l1_interfaces::gnuplot_interface Gnuplot;

		Gnuplot.gnuplot(
				gnuplot_file_fname,
				gnuplot_title,
				gnuplot_label_x,
				gnuplot_label_y,
				verbose_level);

	}

	else if (f_compare_columns) {
		if (f_v) {
			cout << "interface_toolkit::worker "
					"-compare_columns "
					<< " " << compare_columns_fname
					<< " " << compare_columns_column1
					<< " " << compare_columns_column2
					<< endl;
		}

		other::data_structures::spreadsheet S;

		S.read_spreadsheet(
				compare_columns_fname, 0/*verbose_level - 1*/);
		S.compare_columns(
				compare_columns_column1, compare_columns_column2,
				verbose_level);
	}

	else if (f_gcd_worksheet) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"-gcd_worksheet "
					<< " " << gcd_worksheet_nb_problems
					<< " " << gcd_worksheet_N
					<< " " << gcd_worksheet_key
					<< endl;
		}
		algebra::number_theory::number_theory_domain NT;

		NT.create_gcd_worksheet(
				gcd_worksheet_nb_problems, gcd_worksheet_N, gcd_worksheet_key,
				verbose_level);
	}
	else if (f_draw_layered_graph) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					"-draw_layered_graph "
					<< " " << draw_layered_graph_fname
					<< " " << Layered_graph_draw_options
					<< endl;
		}

		other::graphics::graphical_output GO;



		GO.draw_layered_graph_from_file(
				draw_layered_graph_fname,
				Layered_graph_draw_options,
				verbose_level);

	}
	else if (f_read_gedcom) {
		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_read_gedcom " << read_gedcom_fname << endl;
		}


		other::data_structures::ancestry_tree *AT;

		AT = NEW_OBJECT(other::data_structures::ancestry_tree);

		AT->read_gedcom(read_gedcom_fname, verbose_level);

	}
	else if (f_read_xml) {
		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_read_xml " << read_xml_fname << endl;
		}


		other::l1_interfaces::pugixml_interface Pugi;
		std::vector<std::vector<std::string> > Classes_parsed;

		if (f_v) {
			cout << "interface_toolkit::worker "
					"before Pugi.read_file" << endl;
		}
		Pugi.read_file(
				read_xml_fname, Classes_parsed, verbose_level);
		if (f_v) {
			cout << "interface_toolkit::worker "
					"after Pugi.read_file" << endl;
		}


		other::data_structures::algorithms Algo;

		if (f_v) {
			cout << "interface_toolkit::worker "
					"before Algo.process_class_list" << endl;
		}
		Algo.process_class_list(
				Classes_parsed,
				read_xml_crossref_fname,
				verbose_level);
		if (f_v) {
			cout << "interface_toolkit::worker "
					"after Algo.process_class_list" << endl;
		}


	}
	else if (f_read_column_and_tally) {
		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_read_column_and_tally " << read_column_and_tally_fname << endl;
		}

		other::orbiter_kernel_system::file_io Fio;
		long int *Data;
		int data_size;

		Fio.Csv_file_support->read_csv_file_and_get_column(
				read_column_and_tally_fname, read_column_and_tally_col_header,
				Data, data_size, verbose_level);

		if (f_v) {
			cout << "interface_toolkit::worker "
					"read data of size " << data_size << endl;
		}


		other::data_structures::tally T;

		T.init_lint(Data, data_size, false, 0);
		cout << "tally:" << endl;
		T.print_first_tex(true);
		cout << endl;


		other::data_structures::set_of_sets *SoS;
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

		FREE_lint(Data);

	}
	else if (f_intersect_with) {
		if (f_v) {
			cout << "interface_toolkit::worker "
					"f_intersect_with " << intersect_with_vector
					<< " " << intersect_with_data << endl;
		}

		int *M;
		int m, n;
		int *Cnt;

		Get_matrix(intersect_with_vector, M, m, n);

		if (f_v) {
			cout << "matrix of size " << m << " x " << n << endl;
		}

		int *data;
		int sz;

		Get_int_vector_from_label(
				intersect_with_data,
				data, sz,
				verbose_level);

		if (f_v) {
			cout << "data: ";
			Int_vec_print(cout, data, sz);
			cout << endl;
		}

		Cnt = NEW_int(m);
		Int_vec_zero(Cnt, m);
		int i, j, a, idx;

		other::data_structures::sorting Sorting;

		for (i = 0; i < m; i++) {
			for (j = 0; j < sz; j++) {
				a = data[j];

				if (Sorting.int_vec_search_linear(
						M + i * n, n, a, idx)) {
					Cnt[i]++;
				}
			}

		}

		other::data_structures::tally T;

		T.init(
				Cnt,
				m, false /* f_second */, verbose_level);

		T.print_types();

		idx = T.Set_partition->nb_sets - 1;
		cout << "type = " << idx << " has " << T.Set_partition->Set_size[idx] << " elements" << endl;
		Lint_vec_print_fully(cout, T.Set_partition->Sets[idx], T.Set_partition->Set_size[idx]);
		cout << endl;

	}
	else if (f_make_set_of_sets) {

		if (f_v) {
			cout << "interface_toolkit::worker "
					" fname_in = " << make_set_of_sets_fname_in
					<< " new_col_label = " << make_set_of_sets_new_col_label
					<< " list " << make_set_of_sets_list << endl;
		}


		other::data_structures::string_tools ST;
		std::vector<std::string> list;

		ST.parse_comma_separated_list(
				make_set_of_sets_list, list,
				verbose_level);


		other::data_structures::set_of_sets_lint *SoS;

		SoS = NEW_OBJECT(other::data_structures::set_of_sets_lint);

		long int underlying_set_size = 0;

		SoS->init_simple(
				underlying_set_size,
				list.size(),
				0 /* verbose_level */);



		int i;

		for (i = 0; i < list.size(); i++) {

			long int *data;
			int sz;

			Get_lint_vector_from_label(
					list[i],
					data, sz,
					verbose_level);

			if (f_v) {
				cout << "data: ";
				Lint_vec_print(cout, data, sz);
				cout << endl;
			}

			SoS->init_set(
					i /* idx_of_set */,
					data, sz,
					0 /* verbose_level */);


			FREE_lint(data);

		}

		other::orbiter_kernel_system::file_io Fio;
		std::string fname_out;


		Fio.Csv_file_support->append_column_of_int_from_set_of_sets(
				make_set_of_sets_fname_in,
				fname_out,
				SoS,
				make_set_of_sets_new_col_label,
				verbose_level);

		FREE_OBJECT(SoS);

	}

	else if (f_copy_and_edit) {

		if (f_v) {
			cout << "interface_toolkit::worker f_copy_and_edit "
					" fname_in = " << copy_and_edit_input_file << endl;
			cout << "interface_toolkit::worker "
					" copy_and_edit_output_mask = " << copy_and_edit_output_mask << endl;
			cout << "interface_toolkit::worker "
					" copy_and_edit_parameter_values = " << copy_and_edit_parameter_values << endl;
			cout << "interface_toolkit::worker "
					" copy_and_edit_search_and_replace = " << copy_and_edit_search_and_replace << endl;

		}

		int *parameter_values;
		int nb_parameter_values;

		Int_vec_scan(copy_and_edit_parameter_values, parameter_values, nb_parameter_values);



		other::data_structures::string_tools ST;
		//std::vector<std::string> list;
		std::map<std::string, std::string> symbol_table;
		string separator;

		separator = "--->";

		ST.parse_value_pairs_with_separator(
				symbol_table,
				separator,
				copy_and_edit_search_and_replace, verbose_level);

#if 0
		ST.parse_comma_separated_list(
				copy_and_edit_parameter_values, list,
				verbose_level);
#endif

		other::orbiter_kernel_system::file_io Fio;

		Fio.file_edit(
				copy_and_edit_input_file,
				copy_and_edit_output_mask,
				parameter_values,
				nb_parameter_values,
				symbol_table,
				verbose_level);

	}
	else if (f_join_columns) {
		if (f_v) {
			cout << "interface_toolkit::worker join_columns "
					" file_in = " << join_columns_file_in << endl;
			cout << "interface_toolkit::worker join_columns "
					" file_out = " << join_columns_file_out << endl;
			cout << "interface_toolkit::worker "
					" column1 = " << join_columns_column1 << endl;
			cout << "interface_toolkit::worker "
					" column2 = " << join_columns_column2 << endl;

		}


		other::orbiter_kernel_system::file_io Fio;
		other::data_structures::set_of_sets *SoS;

		Fio.Csv_file_support->join_columns(
				join_columns_file_in,
				join_columns_column1, join_columns_column2,
				SoS,
				verbose_level);

		cout << "interface_toolkit::worker after join_columns: data =" << endl;
		SoS->print_table();

		SoS->set_of_sets::save_csv(
				join_columns_file_out,
				verbose_level);

		cout << "Written file " << join_columns_file_out << " of size "
				<< Fio.file_size(join_columns_file_out) << endl;


	}
	else if (f_decomposition_matrix) {
		if (f_v) {
			cout << "interface_toolkit::worker decomposition_matrix "
					" fname = " << decomposition_matrix_fname << endl;
		}


		cout << "reading file " << decomposition_matrix_fname << endl;


		other::data_structures::spreadsheet S;

		S.read_spreadsheet(decomposition_matrix_fname, 0/*verbose_level - 1*/);

		int nb_flag_orbits;

		nb_flag_orbits = S.nb_rows - 1;
		//n = S.nb_cols;

		cout << "nb_flag_orbits = " << nb_flag_orbits << endl;

		//std::string fo_label;
		std::string po_label;
		//std::string so_label;
		std::string f_fst_label;
		std::string iso_idx_label;

		//fo_label = "FO";
		po_label = "PO";
		//so_label = "SO";
		f_fst_label = "F_Fst";
		iso_idx_label = "Iso_idx";

		//int fo_idx;
		int po_idx;
		//int so_idx;
		int f_fst_idx;
		int iso_idx_idx;



		//fo_idx = S.find_column(fo_label);
		po_idx = S.find_column(po_label);
		//so_idx = S.find_column(so_label);
		f_fst_idx = S.find_column(f_fst_label);
		iso_idx_idx = S.find_column(iso_idx_label);

		int f, i, j;

		int *Decomp;
		int nb_rows;
		int nb_cols;


		nb_rows = S.get_lint(1 + nb_flag_orbits - 1, po_idx);
		nb_rows++;

		cout << "nb_rows = " << nb_rows << endl;

		nb_cols = 0;
		for (f = 0; f < nb_flag_orbits; f++) {
			j = S.get_lint(1 + f, f_fst_idx);
			if (j) {
				nb_cols++;
			}
		}

		cout << "nb_cols = " << nb_cols << endl;

		Decomp = NEW_int(nb_rows * nb_cols);
		Int_vec_zero(Decomp, nb_rows * nb_cols);

		for (f = 0; f < nb_flag_orbits; f++) {
			i = S.get_lint(1 + f, po_idx);
			j = S.get_lint(1 + f, iso_idx_idx);
			cout << "f = " << f << " i=" << i << " j=" << j << endl;
			Decomp[i * nb_cols + j]++;
		}

		other::orbiter_kernel_system::file_io Fio;

		int m;

		m = Int_vec_maximum(Decomp, nb_rows * nb_cols);

		cout << "decomposition matrix has size " << nb_rows << " x " << nb_cols << " max entry = " << m << endl;
		Int_matrix_print(Decomp, nb_rows, nb_cols);

		std::string fname_out;

		fname_out = decomposition_matrix_fname;

		other::data_structures::string_tools String;

		String.chop_off_extension(
				fname_out);

		fname_out += "_decomp.csv";

		Fio.Csv_file_support->int_matrix_write_csv(
				fname_out, Decomp, nb_rows, nb_cols);

		cout << "Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;


	}

	if (f_v) {
		cout << "interface_toolkit::worker done" << endl;
	}
}


}}}

