// create_file.C
// 
// Anton Betten
// February 20, 2018
//
// 
//

#include "orbiter.h"
#include "discreta.h"

#define MAX_LINES 1000


void create_files(INT N, 
	const BYTE **lines, INT nb_lines, 
	const BYTE *file_mask, 
	INT f_repeat, INT repeat_N, const BYTE *repeat_mask, 
	INT f_split, INT split_m, 
	INT verbose_level);
void create_files_list_of_cases(spreadsheet *S, 
	INT read_cases_column_of_case, INT read_cases_column_of_fname, 
	INT N, 
	const BYTE **lines, INT nb_lines, 
	const BYTE **final_lines, INT nb_final_lines, 
	INT f_tasks, INT nb_tasks, const BYTE *tasks_line, 
	const BYTE *command, 
	const BYTE *file_mask, 
	INT verbose_level);



int main(int argc, char **argv)
{
	INT i, j;
	INT verbose_level = 0;
	INT f_file_mask = FALSE;
	const BYTE *file_mask = NULL;
	INT f_N = FALSE;
	INT N = 0;
	INT nb_lines = 0;
	const BYTE *lines[MAX_LINES];
	INT nb_final_lines = 0;
	const BYTE *final_lines[MAX_LINES];
	INT f_command = FALSE;
	const BYTE *command = NULL;
	INT f_repeat = FALSE;
	INT repeat_N = 0;
	const BYTE *repeat_mask;
	INT f_split = FALSE;
	INT split_m = 0;
	INT f_read_cases = FALSE;
	const BYTE *read_cases_fname = NULL;
	INT f_read_cases_text = FALSE;
	INT read_cases_column_of_case = 0;
	INT read_cases_column_of_fname = 0;
	INT f_tasks = FALSE;
	INT nb_tasks = 0;
	const BYTE *tasks_line = NULL;

	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-file_mask") == 0) {
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
			cout << "-read_cases_text " << read_cases_fname << " " << read_cases_column_of_case << " " << read_cases_column_of_fname << endl;
			}
		else if (strcmp(argv[i], "-line") == 0) {
			lines[nb_lines] = argv[++i];
			cout << "-line " << lines[nb_lines] << endl;
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
			repeat_mask = argv[++i];
			cout << "-repeat " << repeat_N << " " << repeat_mask << endl;
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
		}
	if (!f_file_mask) {
		cout << "please use -file_mask <file_mask>" << endl;
		exit(1);
		}
#if 0
	if (!f_N && !f_read_cases) {
		cout << "please use -N <N> or -read_cases <fname>" << endl;
		exit(1);
		}
#endif

	if (f_read_cases) {

		BYTE fname[1000];
		BYTE str[1000];
		INT *Cases;
		INT nb_cases;
		INT n, c;
		
		INT_matrix_read_csv(read_cases_fname, Cases, nb_cases, n, 0 /* verbose_level */);
		
		if (n != 1) {
			cout << "read cases, n != 1" << endl;
			exit(1);
			}
		cout << "We found " << nb_cases << " cases to do:" << endl;
		INT_vec_print(cout, Cases, nb_cases);
		cout << endl;

			const BYTE *log_fname = "log_file.txt";
			const BYTE *log_mask = "\tsbatch job%04ld";
			{
			ofstream fp_log(log_fname);

			for (c = 0; c < nb_cases; c++) {

				i = Cases[c];
				sprintf(fname, file_mask, i);
		

					{
					ofstream fp(fname);
		
					for (j = 0; j < nb_lines; j++) {
						sprintf(str, lines[j], i);
						fp << str << endl;
						}
					}
				cout << "Written file " << fname << " of size " << file_size(fname) << endl;

				BYTE log_entry[1000];
				
				sprintf(log_entry, log_mask, i);
				fp_log << log_entry << endl;
				}
			}
			cout << "Written file " << log_fname << " of size " << file_size(log_fname) << endl;
		}
	else if (f_read_cases_text) {
		cout << "read_cases_text" << endl;
		
		if (!f_N) {
			cout << "please use option -N <N>" << endl;
			exit(1);
			}
		if (!f_command) {
			cout << "please use option -command <command>" << endl;
			exit(1);
			}

		cout << "Reading file " << read_cases_fname << endl;

		spreadsheet *S;
		INT row;

		S = new spreadsheet;
		S->read_spreadsheet(read_cases_fname, 0 /*verbose_level*/);
	
		cout << "Read spreadsheet with " << S->nb_rows << " rows" << endl;

		//S->print_table(cout, FALSE /* f_enclose_in_parentheses */);
		for (row = 0; row < MINIMUM(10, S->nb_rows); row++) {
			cout << "row " << row << " : ";
			S->print_table_row(row, FALSE /* f_enclose_in_parentheses */, cout);
			}
		cout << "..." << endl;
		for (row = MAXIMUM(S->nb_rows - 10, 0); row < S->nb_rows; row++) {
			cout << "row " << row << " : ";
			S->print_table_row(row, FALSE /* f_enclose_in_parentheses */, cout);
			}



		create_files_list_of_cases(S, 
			read_cases_column_of_case, read_cases_column_of_fname, 
			N, 
			lines, nb_lines, 
			final_lines, nb_final_lines, 
			f_tasks, nb_tasks, tasks_line, 
			command, 
			file_mask, 
			verbose_level);
		
		}
	else if (f_N) {
		create_files(N, 
			lines, nb_lines, 
			file_mask, 
			f_repeat, repeat_N, repeat_mask, 
			f_split, split_m, 
			verbose_level);
		}
}

void create_files(INT N, 
	const BYTE **lines, INT nb_lines, 
	const BYTE *file_mask, 
	INT f_repeat, INT repeat_N, const BYTE *repeat_mask, 
	INT f_split, INT split_m, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;

	BYTE fname[1000];
	BYTE str[1000];
	INT r;

	if (f_v) {
		cout << "create_files" << endl;
		}
	
	const BYTE *makefile_fname = "makefile_submit";
	{
	ofstream fp_makefile(makefile_fname);
	
	for (i = 0; i < N; i++) {

		sprintf(fname, file_mask, i);
	
		fp_makefile << "\tsbatch " << fname << endl;
		{
		ofstream fp(fname);
	
		for (j = 0; j < nb_lines; j++) {
			sprintf(str, lines[j], i, i, i, i, i, i, i, i);
			fp << str << endl;
			}
		if (f_repeat) {
			if (f_split) {
				for (r = 0; r < split_m; r++) {
					for (j = 0; j < repeat_N; j++) {
						if ((j % split_m) == r) {
							sprintf(str, repeat_mask, j);
							fp << str << endl;
							}
						}
					fp << endl;
					}
				}
			else {
				}
			}
		}
		cout << "Written file " << fname << " of size " << file_size(fname) << endl;
	
		}

	}
	cout << "Written file " << makefile_fname << " of size " << file_size(makefile_fname) << endl;


	if (f_v) {
		cout << "create_files done" << endl;
		}
}

void create_files_list_of_cases(spreadsheet *S, 
	INT read_cases_column_of_case, INT read_cases_column_of_fname, 
	INT N, 
	const BYTE **lines, INT nb_lines, 
	const BYTE **final_lines, INT nb_final_lines, 
	INT f_tasks, INT nb_tasks, const BYTE *tasks_line, 
	const BYTE *command, 
	const BYTE *file_mask, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;

	BYTE fname[1000];
	BYTE str[1000];

	if (f_v) {
		cout << "create_files_list_of_cases" << endl;
		}
	
	INT nb_cases = S->nb_rows - 1;
	cout << "nb_cases=" << nb_cases << endl;


	const BYTE *makefile_fname = "makefile_submit";
	{
	ofstream fp_makefile(makefile_fname);
	
	for (i = 0; i < N; i++) {

		sprintf(fname, file_mask, i);
	
		fp_makefile << "\tsbatch " << fname << endl;
		{
		ofstream fp(fname);
	
		for (j = 0; j < nb_lines; j++) {
			sprintf(str, lines[j], i, i, i, i, i, i, i, i);
			fp << str << endl;
			}

		if (f_tasks) {
			BYTE str[1000];
			INT t;
			INT NT;

			sprintf(str, tasks_line, nb_tasks);
			fp << str << endl;
			NT = N * nb_tasks;
			for (t = 0; t < nb_tasks; t++) {
				sprintf(str, command, i, t, i, t);
				fp << str << " \\" << endl;
				for (j = 0; j < nb_cases; j++) {
					if ((j % N) != i) {
						continue;
						}
					if (((j - i) / N) % nb_tasks != t) {
						continue;
						}
					BYTE *entry;
					INT case_number;

					case_number = S->get_INT(j + 1, read_cases_column_of_case);
					entry = S->get_string(j + 1, read_cases_column_of_fname);
					fp << "\t\t" << case_number << " " << entry << " \\" << endl;
					}
				fp << "\t\t" << -1 << " &" << endl;
				}
			}
		else {
			fp << command << " \\" << endl;
			for (j = 0; j < nb_cases; j++) {
				if ((j % N) != i) {
					continue;
					}
				BYTE *entry;
				INT case_number;

				case_number = S->get_INT(j + 1, read_cases_column_of_case);
				entry = S->get_string(j + 1, read_cases_column_of_fname);
				fp << "\t\t" << case_number << " " << entry << " \\" << endl;
				}
			fp << "\t\t" << -1 << " &" << endl;
			}
		for (j = 0; j < nb_final_lines; j++) {
			sprintf(str, final_lines[j], i, i, i, i, i, i, i, i);
			fp << str << endl;
			}

		}
		cout << "Written file " << fname << " of size " << file_size(fname) << endl;
	
		}

	}
	cout << "Written file " << makefile_fname << " of size " << file_size(makefile_fname) << endl;


	if (f_v) {
		cout << "create_files_list_of_cases done" << endl;
		}
}


