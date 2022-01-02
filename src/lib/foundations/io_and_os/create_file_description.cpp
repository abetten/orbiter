/*
 * create_file_description.cpp
 *
 *  Created on: Jul 2, 2020
 *      Author: betten
 */




#include "foundations.h"

using namespace std;




namespace orbiter {
namespace foundations {


create_file_description::create_file_description()
{
	f_file_mask = FALSE;
	//file_mask ;
	f_N = FALSE;
	N = 0;
	nb_lines = 0;
	//const char *lines[MAX_LINES];
	nb_final_lines = 0;
	//const char *final_lines[MAX_LINES];
	f_command = FALSE;
	//command;
	f_repeat = FALSE;
	repeat_N = 0;
	repeat_start = 0;
	repeat_increment = 0;
	//repeat_mask;
	f_split = FALSE;
	split_m = 0;
	f_read_cases = FALSE;
	//read_cases_fname = NULL;
	f_read_cases_text = FALSE;
	read_cases_column_of_case = 0;
	read_cases_column_of_fname = 0;
	f_tasks = FALSE;
	nb_tasks = 0;
	//tasks_line;

}

create_file_description::~create_file_description()
{

}


int create_file_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "create_file_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {
		if (stringcmp(argv[i], "-file_mask") == 0) {
			f_file_mask = TRUE;
			file_mask.assign(argv[++i]);
			cout << "-file_mask " << file_mask << endl;
		}
		else if (stringcmp(argv[i], "-N") == 0) {
			f_N = TRUE;
			N = strtoi(argv[++i]);
			cout << "-N " << N << endl;
		}
		else if (stringcmp(argv[i], "-read_cases") == 0) {
			f_read_cases = TRUE;
			read_cases_fname.assign(argv[++i]);
			cout << "-read_cases " << read_cases_fname << endl;
		}
		else if (stringcmp(argv[i], "-read_cases_text") == 0) {
			f_read_cases_text = TRUE;
			read_cases_fname.assign(argv[++i]);
			read_cases_column_of_case = strtoi(argv[++i]);
			read_cases_column_of_fname = strtoi(argv[++i]);
			cout << "-read_cases_text " << read_cases_fname << " "
					<< read_cases_column_of_case << " "
					<< read_cases_column_of_fname << endl;
		}
		else if (stringcmp(argv[i], "-line") == 0) {
			lines[nb_lines].assign(argv[++i]);
			f_line_numeric[nb_lines] = FALSE;
			cout << "-line " << lines[nb_lines] << endl;
			nb_lines++;
		}
		else if (stringcmp(argv[i], "-line_numeric") == 0) {
			lines[nb_lines].assign(argv[++i]);
			f_line_numeric[nb_lines] = TRUE;
			cout << "-line_numeric " << lines[nb_lines] << endl;
			nb_lines++;
		}
		else if (stringcmp(argv[i], "-final_line") == 0) {
			final_lines[nb_final_lines].assign(argv[++i]);
			cout << "-final_line " << final_lines[nb_final_lines] << endl;
			nb_final_lines++;
		}
		else if (stringcmp(argv[i], "-command") == 0) {
			f_command = TRUE;
			command.assign(argv[++i]);
			cout << "-command " << command << endl;
		}
		else if (stringcmp(argv[i], "-repeat") == 0) {
			f_repeat = TRUE;
			repeat_N = strtoi(argv[++i]);
			repeat_start = strtoi(argv[++i]);
			repeat_increment = strtoi(argv[++i]);
			repeat_mask.assign(argv[++i]);
			cout << "-repeat " << repeat_N
					<< " " << repeat_start
					<< " " << repeat_increment
					<< " " << repeat_mask
					<< endl;
		}
		else if (stringcmp(argv[i], "-split") == 0) {
			f_split = TRUE;
			split_m = strtoi(argv[++i]);
			cout << "-split " << split_m << endl;
		}
		else if (stringcmp(argv[i], "-tasks") == 0) {
			f_tasks = TRUE;
			nb_tasks = strtoi(argv[++i]);
			tasks_line.assign(argv[++i]);
			cout << "-tasks " << nb_tasks << " " << tasks_line << endl;
		}
		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "create_file_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "create_file_description::read_arguments done" << endl;
	return i + 1;
}

void create_file_description::print()
{
	if (f_file_mask) {
		cout << "-file_mask " << file_mask << endl;
	}
	if (f_N) {
		cout << "-N " << N << endl;
	}
	if (f_read_cases) {
		cout << "-read_cases " << read_cases_fname << endl;
	}
	if (f_read_cases_text) {
		cout << "-read_cases_text " << read_cases_fname << " "
				<< read_cases_column_of_case << " "
				<< read_cases_column_of_fname << endl;
	}
	if (nb_lines) {
		int i;
		for (i = 0; i < nb_lines; i++) {
			if (f_line_numeric[nb_lines]) {
				cout << "-line_numeric " << lines[i] << endl;
			}
			else {
				cout << "-line " << lines[i] << endl;
			}
		}
	}
	if (nb_final_lines) {
		int i;
		for (i = 0; i < nb_final_lines; i++) {
			cout << "-final_line " << final_lines[i] << endl;
		}

	}
	if (f_command) {
		cout << "-command " << command << endl;
	}
	if (f_repeat) {
		cout << "-repeat " << repeat_N
				<< " " << repeat_start
				<< " " << repeat_increment
				<< " " << repeat_mask
				<< endl;
	}
	if (f_split) {
		cout << "-split " << split_m << endl;
	}
	if (f_tasks) {
		cout << "-tasks " << nb_tasks << " " << tasks_line << endl;
	}
}


}}

