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
	file_mask = NULL;
	f_N = FALSE;
	N = 0;
	nb_lines = 0;
	//const char *lines[MAX_LINES];
	nb_final_lines = 0;
	//const char *final_lines[MAX_LINES];
	f_command = FALSE;
	command = NULL;
	f_repeat = FALSE;
	repeat_N = 0;
	repeat_start = 0;
	repeat_increment = 0;
	repeat_mask = NULL;
	f_split = FALSE;
	split_m = 0;
	f_read_cases = FALSE;
	read_cases_fname = NULL;
	f_read_cases_text = FALSE;
	read_cases_column_of_case = 0;
	read_cases_column_of_fname = 0;
	f_tasks = FALSE;
	nb_tasks = 0;
	tasks_line = NULL;

}

create_file_description::~create_file_description()
{

}


int create_file_description::read_arguments(
	int argc, const char **argv,
	int verbose_level)
{
	int i;

	cout << "create_file_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {
		if (strcmp(argv[i], "-file_mask") == 0) {
			f_file_mask = TRUE;
			file_mask = argv[++i];
			cout << "-file_mask " << file_mask << endl;
		}
		else if (strcmp(argv[i], "-N") == 0) {
			f_N = TRUE;
			N = atoi(argv[++i]);
			cout << "-N " << N << endl;
		}
		else if (strcmp(argv[i], "-read_cases") == 0) {
			f_read_cases = TRUE;
			read_cases_fname = argv[++i];
			cout << "-read_cases " << read_cases_fname << endl;
		}
		else if (strcmp(argv[i], "-read_cases_text") == 0) {
			f_read_cases_text = TRUE;
			read_cases_fname = argv[++i];
			read_cases_column_of_case = atoi(argv[++i]);
			read_cases_column_of_fname = atoi(argv[++i]);
			cout << "-read_cases_text " << read_cases_fname << " "
					<< read_cases_column_of_case << " "
					<< read_cases_column_of_fname << endl;
		}
		else if (strcmp(argv[i], "-line") == 0) {
			lines[nb_lines] = argv[++i];
			f_line_numeric[nb_lines] = FALSE;
			cout << "-line " << lines[nb_lines] << endl;
			nb_lines++;
		}
		else if (strcmp(argv[i], "-line_numeric") == 0) {
			lines[nb_lines] = argv[++i];
			f_line_numeric[nb_lines] = TRUE;
			cout << "-line_numeric " << lines[nb_lines] << endl;
			nb_lines++;
		}
		else if (strcmp(argv[i], "-final_line") == 0) {
			final_lines[nb_final_lines] = argv[++i];
			cout << "-final_line " << final_lines[nb_final_lines] << endl;
			nb_final_lines++;
		}
		else if (strcmp(argv[i], "-command") == 0) {
			f_command = TRUE;
			command = argv[++i];
			cout << "-command " << command << endl;
		}
		else if (strcmp(argv[i], "-repeat") == 0) {
			f_repeat = TRUE;
			repeat_N = atoi(argv[++i]);
			repeat_start = atoi(argv[++i]);
			repeat_increment = atoi(argv[++i]);
			repeat_mask = argv[++i];
			cout << "-repeat " << repeat_N
					<< " " << repeat_start
					<< " " << repeat_increment
					<< " " << repeat_mask
					<< endl;
		}
		else if (strcmp(argv[i], "-split") == 0) {
			f_split = TRUE;
			split_m = atoi(argv[++i]);
			cout << "-split " << split_m << endl;
		}
		else if (strcmp(argv[i], "-tasks") == 0) {
			f_tasks = TRUE;
			nb_tasks = atoi(argv[++i]);
			tasks_line = argv[++i];
			cout << "-tasks " << nb_tasks << " " << tasks_line << endl;
		}
		else if (strcmp(argv[i], "-end") == 0) {
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



}}

