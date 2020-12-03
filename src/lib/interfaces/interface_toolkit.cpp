/*
 * interface_toolkit.cpp
 *
 *  Created on: Nov 29, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace interfaces {




interface_toolkit::interface_toolkit()
{
	f_csv_file_select_rows = FALSE;
	//std::string csv_file_select_rows_fname;
	//std::string csv_file_select_rows_text;

	f_csv_file_join = FALSE;
	//csv_file_join_fname
	//csv_file_join_identifier

}


void interface_toolkit::print_help(int argc,
		std::string *argv, int i, int verbose_level)
{
	if (stringcmp(argv[i], "-csv_file_select_rows") == 0) {
		cout << "-cvs_file_select_rows <string : csv_file_name> <string : list of rows>" << endl;
	}
	else if (stringcmp(argv[i], "-csv_file_join") == 0) {
		cout << "-cvs_file_join <string : file_name> <string : column label by which we join>" << endl;
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
	else if (stringcmp(argv[i], "-csv_file_join") == 0) {
		return true;
	}
	return false;
}

int interface_toolkit::read_arguments(int argc,
		std::string *argv, int i0, int verbose_level)
{
	int i;

	cout << "interface_toolkit::read_arguments" << endl;

	for (i = i0; i < argc; i++) {
		if (stringcmp(argv[i], "-csv_file_select_rows") == 0) {
			f_csv_file_select_rows = TRUE;
			csv_file_select_rows_fname.assign(argv[++i]);
			csv_file_select_rows_text.assign(argv[++i]);
			cout << "-csv_file_select_rows " << csv_file_select_rows_fname << " " << csv_file_select_rows_text << endl;
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
		else {
			break;
		}
	}
	cout << "interface_toolkit::read_arguments done" << endl;
	return i;
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
	else if (f_csv_file_join) {

		file_io Fio;

		Fio.do_csv_file_join(csv_file_join_fname,
				csv_file_join_identifier, verbose_level);
	}

	if (f_v) {
		cout << "interface_toolkit::worker done" << endl;
	}
}




}}
