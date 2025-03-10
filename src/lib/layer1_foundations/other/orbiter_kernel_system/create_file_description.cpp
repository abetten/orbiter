/*
 * create_file_description.cpp
 *
 *  Created on: Jul 2, 2020
 *      Author: betten
 */




#include "foundations.h"

using namespace std;




namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace orbiter_kernel_system {


create_file_description::create_file_description()
{
	Record_birth();
	f_file_mask = false;
	//file_mask ;
	f_N = false;
	N = 0;
	nb_lines = 0;
	//const char *lines[MAX_LINES];
	nb_final_lines = 0;
	//const char *final_lines[MAX_LINES];
	f_command = false;
	//command;
	f_repeat = false;
	repeat_N = 0;
	repeat_start = 0;
	repeat_increment = 0;
	//repeat_mask;
	f_split = false;
	split_m = 0;
	f_read_cases = false;
	//read_cases_fname = NULL;
	f_read_cases_text = false;
	read_cases_column_of_case = 0;
	read_cases_column_of_fname = 0;
	f_tasks = false;
	nb_tasks = 0;
	//tasks_line;

}

create_file_description::~create_file_description()
{
	Record_death();

}


int create_file_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;
	data_structures::string_tools ST;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_file_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-file_mask") == 0) {
			f_file_mask = true;
			file_mask.assign(argv[++i]);
			if (f_v) {
				cout << "-file_mask " << file_mask << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-N") == 0) {
			f_N = true;
			N = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-N " << N << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-read_cases") == 0) {
			f_read_cases = true;
			read_cases_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-read_cases " << read_cases_fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-read_cases_text") == 0) {
			f_read_cases_text = true;
			read_cases_fname.assign(argv[++i]);
			read_cases_column_of_case = ST.strtoi(argv[++i]);
			read_cases_column_of_fname = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-read_cases_text " << read_cases_fname << " "
						<< read_cases_column_of_case << " "
						<< read_cases_column_of_fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-line") == 0) {
			lines[nb_lines].assign(argv[++i]);
			f_line_numeric[nb_lines] = false;
			if (f_v) {
				cout << "-line " << lines[nb_lines] << endl;
			}
			nb_lines++;
		}
		else if (ST.stringcmp(argv[i], "-line_numeric") == 0) {
			lines[nb_lines].assign(argv[++i]);
			f_line_numeric[nb_lines] = true;
			if (f_v) {
				cout << "-line_numeric " << lines[nb_lines] << endl;
			}
			nb_lines++;
		}
		else if (ST.stringcmp(argv[i], "-final_line") == 0) {
			final_lines[nb_final_lines].assign(argv[++i]);
			if (f_v) {
				cout << "-final_line " << final_lines[nb_final_lines] << endl;
			}
			nb_final_lines++;
		}
		else if (ST.stringcmp(argv[i], "-command") == 0) {
			f_command = true;
			command.assign(argv[++i]);
			if (f_v) {
				cout << "-command " << command << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-repeat") == 0) {
			f_repeat = true;
			repeat_N = ST.strtoi(argv[++i]);
			repeat_start = ST.strtoi(argv[++i]);
			repeat_increment = ST.strtoi(argv[++i]);
			repeat_mask.assign(argv[++i]);
			if (f_v) {
				cout << "-repeat " << repeat_N
						<< " " << repeat_start
						<< " " << repeat_increment
						<< " " << repeat_mask
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-split") == 0) {
			f_split = true;
			split_m = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-split " << split_m << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-tasks") == 0) {
			f_tasks = true;
			nb_tasks = ST.strtoi(argv[++i]);
			tasks_line.assign(argv[++i]);
			if (f_v) {
				cout << "-tasks " << nb_tasks << " " << tasks_line << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "create_file_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	if (f_v) {
		cout << "create_file_description::read_arguments done" << endl;
	}
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


}}}}



