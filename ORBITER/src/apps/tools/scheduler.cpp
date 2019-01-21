// scheduler.C
// 
// Anton Betten
// January 16, 2016
//
// 
//

#include "orbiter.h"


using namespace orbiter;


#define MY_BUFSIZE ONE_MILLION

typedef class job_table job_table;

//! job table for the scheduler app for parallel processing


class job_table {

public:

	job_table();
	~job_table();

	int f_task_assigned;
	int task;
	int the_case;
	int job;
	char target_fname[1000];
	char command[1000];
	const char *target_file_mask;
	const char *command_mask;
	char batch_fname[1000];
	char batch_file[10000];
};


// global data:

int t0; // the system time when the program started

void do_collate(int N,
		const char *collate_output_file_mask,
		const char *collated_fname,
		int verbose_level);
void do_scheduling(int N, int *list_of_cases, 
	int J, 
	int f_input_file_mask, const char *input_file_mask, 
	const char *target_file_mask, 
	int f_command_mask, const char *command_mask, 
	int *excluded_cases, int nb_excluded_cases,
	int f_reload, 
	int f_batch, const char *job_fname, const char *batch_template, int template_nb_times, 
	int f_randomized, const char *randomized_fname, 
	int f_log_prefix, const char *log_prefix, 
	int verbose_level);
int find_free_job(job_table *JT, int J);
void assign_task(job_table *JT, int t, int j, 
	int *list_of_cases, const char *log_prefix, 
	int f_batch, const char *job_fname, const char *batch_template, int template_nb_times, 
	int verbose_level);


int main(int argc, char **argv)
{
	int i;
	t0 = os_ticks();
	int verbose_level = 0;
	int f_input_file_mask = FALSE;	
	const char *input_file_mask = NULL;
	int f_target_file_mask = FALSE;	
	const char *target_file_mask = NULL;
	int f_command_mask = FALSE;	
	const char *command_mask = NULL;
	int f_N = FALSE;
	int N = 0;
	int f_list_of_cases = FALSE;
	const char *fname_list_of_cases = NULL;
	int f_J = FALSE;
	int J = 1;
	int nb_excluded_cases = 0;
	int excluded_cases[1000];
	int f_reload = FALSE;
	int f_batch = FALSE;
	const char *job_fname = NULL;
	const char *batch_template = NULL;
	int template_nb_times = 0;
	int f_randomized = FALSE;
	const char *randomized_fname = NULL;
	int f_log_prefix = FALSE;
	const char *log_prefix = "";
	int f_collate = FALSE;
	const char *collate_output_file_mask = NULL;
	const char *collated_fname = NULL;

	int nb_symbols = 0;
	const char *symbol_key[1000];
	const char *symbol_value[1000];


	cout << "scheduler.out" << endl;
	
	// first pass: get definitions:
	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		if (strcmp(argv[i], "-define") == 0) {
			symbol_key[nb_symbols] = argv[++i];
			symbol_value[nb_symbols] = argv[++i];
			cout << "-symbol " << symbol_key[nb_symbols] << " " << symbol_value[nb_symbols] << endl;
			nb_symbols++;
			}
		}

	cout << "we found the following symbol definitions:" << endl;
	for (i = 0; i < nb_symbols; i++) {
		cout << i << " : " << symbol_key[i] << " : " << symbol_value[i] << endl;
		}

	char str[10000];
	char key[10000];
	char *p; 
	int h, j, l, k;
	char **Argv;


	cout << "Arguments before sustitutions:" << endl;
	for (i = 1; i < argc; i++) {
		cout << i << " : " << argv[i] << endl;
		}


	cout << "performing substitutions:" << endl;
	Argv = NEW_pchar(argc + 1);
	for (i = 1; i < argc; i++) {
		str[0] = 0;
		p = argv[i];
		k = 0;
		l = strlen(argv[i]);
		//cout << "parsing " << argv[i] << endl;
		for (j = 0; j < l; j++) {
			//cout << "looking at " << p + j << endl;
			if (strncmp(p + j, "XXX", 3) == 0) {
				for (h = 0; ; h++) {
					if (strncmp(p + j + 3 + h, "XXX", 3) == 0) {
						key[h] = 0;
						h++;
						break;
						}
					if (p[j + 3 + h] == 0) {
						cout << "please use symbols of the form XXX...XXX" << endl;
						exit(1);
						}
					key[h] = p[j + 3 + h];
					}
				//cout << "found symbol " << key << endl;
				for (h = 0; h < nb_symbols; h++) {
					if (strcmp(key, symbol_key[h]) == 0) {
						cout << "the key matches symbol " << h 
							<< " = " << symbol_key[h] << " and will be replaced by " 
							<< symbol_value[h] << endl;
						strcpy(str + k, symbol_value[h]);
						k += strlen(symbol_value[h]);
						str[k] = 0;
						cout << "str=" << str << endl;
						j += strlen(key) + 5;
						break;
						}
					}
				if (h == nb_symbols) {
					cout << "did not find symbol " << key << endl;
					exit(1);
					}
				}
			else {
				str[k++] = p[j];
				}
			}
		str[k] = 0;
		//cout << "after replacement:" << endl;
		cout << str << endl;
		l = strlen(str);
		Argv[i] = NEW_char(l + 1);
		strcpy(Argv[i], str);
		}

	
	cout << "Arguments after sustitutions:" << endl;
	for (i = 1; i < argc; i++) {
		cout << i << " : " << Argv[i] << endl;
		}



	for (i = 1; i < argc; i++) {
		if (strcmp(Argv[i], "-v") == 0) {
			verbose_level = atoi(Argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(Argv[i], "-input_file_mask") == 0) {
			f_input_file_mask = TRUE;
			input_file_mask = Argv[++i];
			cout << "-input_file_mask " << input_file_mask << endl;
			}
		else if (strcmp(Argv[i], "-target_file_mask") == 0) {
			f_target_file_mask = TRUE;
			target_file_mask = Argv[++i];
			cout << "-target_file_mask " << target_file_mask << endl;
			}
		else if (strcmp(Argv[i], "-command_mask") == 0) {
			f_command_mask = TRUE;
			command_mask = Argv[++i];
			cout << "-command_mask " << command_mask << endl;
			}
		else if (strcmp(Argv[i], "-N") == 0) {
			f_N = TRUE;
			N = atoi(Argv[++i]);
			cout << "-N " << N << endl;
			}
		else if (strcmp(Argv[i], "-list_of_cases") == 0) {
			f_list_of_cases = TRUE;
			fname_list_of_cases = Argv[++i];
			cout << "-list_of_cases " << fname_list_of_cases << endl;
			}
		else if (strcmp(Argv[i], "-J") == 0) {
			f_J = TRUE;
			J = atoi(Argv[++i]);
			cout << "-J " << J << endl;
			}
		else if (strcmp(Argv[i], "-exclude") == 0) {
			excluded_cases[nb_excluded_cases] = atoi(Argv[++i]);
			cout << "-exclude " << excluded_cases[nb_excluded_cases] << endl;
			nb_excluded_cases++;
			}
		else if (strcmp(Argv[i], "-reload") == 0) {
			f_reload = TRUE;
			cout << "-reload " << endl;
			}
		else if (strcmp(Argv[i], "-batch") == 0) {
			f_batch = TRUE;
			job_fname = Argv[++i];
			batch_template = Argv[++i];
			template_nb_times = atoi(Argv[++i]);
			cout << "-batch " << job_fname << " \"" << batch_template << "\" " << template_nb_times << endl;
			}
		else if (strcmp(Argv[i], "-randomized") == 0) {
			f_randomized = TRUE;
			randomized_fname = Argv[++i];
			cout << "-randomized " << randomized_fname << endl;
			}
		else if (strcmp(Argv[i], "-log_prefix") == 0) {
			f_log_prefix = TRUE;
			log_prefix = Argv[++i];
			cout << "-log_prefix " << log_prefix << endl;
			}
		else if (strcmp(Argv[i], "-collate") == 0) {
			f_collate = TRUE;
			collate_output_file_mask = Argv[++i];
			collated_fname = Argv[++i];
			cout << "-collate " << collate_output_file_mask << " " << collated_fname << endl;
			}
		}

	if (!f_N && !f_list_of_cases) {
		cout << "please use either option -N <N> or -list_of_cases <fname>" << endl;
		exit(1);
		}
	if (!f_J) {
		cout << "please use option -J <J>" << endl;
		exit(1);
		}
#if 0
	if (!f_input_file_mask) {
		cout << "please use option -input_file_mask <input_file_mask>" << endl;
		exit(1);
		}
#endif
	if (!f_target_file_mask) {
		cout << "please use option -target_file_mask <target_file_mask>" << endl;
		exit(1);
		}

	int *list_of_cases;

	if (f_list_of_cases) {
		read_set_from_file(fname_list_of_cases, list_of_cases, N, verbose_level);
		cout << "nb_cases=N=" << N << endl;
		cout << "list of cases from file: ";
		int_vec_print(cout, list_of_cases, N);
		cout << endl;
		}
	else {
		list_of_cases = NEW_int(N);
		for (i = 0; i < N; i++) {
			list_of_cases[i] = i;
			}
		}

	int_vec_heapsort(excluded_cases, nb_excluded_cases);
	cout << "There are " << nb_excluded_cases << " excluded cases: ";
	int_vec_print(cout, excluded_cases, nb_excluded_cases);
	cout << endl;
	

	do_scheduling(N, list_of_cases, 
		J, 
		f_input_file_mask, input_file_mask, 
		target_file_mask, 
		f_command_mask, command_mask, 
		excluded_cases, nb_excluded_cases,
		f_reload, 
		f_batch, job_fname, batch_template, template_nb_times, 
		f_randomized, randomized_fname, 
		f_log_prefix, log_prefix, 
		verbose_level);

	if (f_collate) {
		do_collate(N, collate_output_file_mask, collated_fname, verbose_level);
		}

	FREE_int(list_of_cases);

	cout << "scheduler.out is done" << endl;
	the_end(t0);
	//the_end_quietly(t0);

}

void do_collate(int N, const char *collate_output_file_mask, const char *collated_fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char output_fname[1000];
	int i, j, a, len, nb_sol, total_sol;
	int *missing_cases;
	int nb_missing_cases = 0;
	char buf[MY_BUFSIZE];
	char *p_buf;

	if (f_v) {
		cout << "do_collate" << endl;
		}

	missing_cases = NEW_int(N);
	total_sol = 0;
	{
	ofstream fp(collated_fname);
	for (i = 0; i < N; i++) {
		sprintf(output_fname, collate_output_file_mask, i);
		if (file_size(output_fname) <= 0) {
			cout << "output file in case " << i << " / " << N << " is missing" << endl;
			missing_cases[nb_missing_cases++] = i;
			}
		else {
			{
			ifstream f(output_fname);
			
			nb_sol = 0;
			while (TRUE) {
				if (f.eof()) {
					break;
					}
		
				//cout << "count_number_of_orbits_in_file reading line, nb_sol = " << nb_sol << endl;
				f.getline(buf, MY_BUFSIZE, '\n');
				if (strlen(buf) == 0) {
					cout << "count_number_of_orbits_in_file reading an empty line" << endl;
					break;
					}
		
				// check for comment line:
				if (buf[0] == '#')
					continue;
			
				p_buf = buf;
				s_scan_int(&p_buf, &len);
				if (len == -1) {
					if (f_v) {
						cout << "found a complete file with " << nb_sol << " solutions" << endl;
						}
					break;
					}
				fp << len;
				for (j = 0; j < len; j++) {
					s_scan_int(&p_buf, &a);
					fp << " " << a;
					}
				fp << endl;
				nb_sol++;
				}
			
			}
			total_sol += nb_sol;
			}
		}
	fp << -1;
	if (nb_missing_cases == 0) {
		fp << " the file is complete" << endl;
		}
	else {
		fp << " the file is incomplete, " << nb_missing_cases << " cases are missing, they are ";
		int_vec_print(fp, missing_cases, nb_missing_cases);
		fp << endl;
		}
	}
	cout << "written file " << collated_fname << " of size " << file_size(collated_fname) << " with " << total_sol << " solutions. Number of missing cases = " << nb_missing_cases << endl;
	cout << "missing cases: ";
	int_vec_print(cout, missing_cases, nb_missing_cases);
	cout << endl;
	
	if (f_v) {
		cout << "do_collate done" << endl;
		}
}

void do_scheduling(int N, int *list_of_cases, 
	int J, 
	int f_input_file_mask, const char *input_file_mask, 
	const char *target_file_mask, 
	int f_command_mask, const char *command_mask, 
	int *excluded_cases, int nb_excluded_cases,
	int f_reload, 
	int f_batch, const char *job_fname, const char *batch_template, int template_nb_times, 
	int f_randomized, const char *randomized_fname, 
	int f_log_prefix, const char *log_prefix, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, c;

	if (f_v) {
		cout << "do_scheduling" << endl;
		cout << "N = " << N << endl;
		cout << "J = " << J << endl;
		if (f_input_file_mask) {
			cout << "input_file_mask = " << input_file_mask << endl;
			}
		cout << "target_file_mask = " << target_file_mask << endl;
		cout << "f_command_mask = " << f_command_mask << endl;
		if (f_command_mask) {
			cout << "command_mask = " << command_mask << endl;
			}
		cout << "f_reload = " << f_reload << endl;
		cout << "f_batch = " << f_batch << endl;
		cout << "f_randomized = " << f_randomized << endl;
		if (f_randomized) {
			cout << "randomized_fname = " << randomized_fname << endl;
			}
		cout << "f_log_prefix = " << f_log_prefix << endl;
		if (f_log_prefix) {
			cout << "log_prefix = " << log_prefix << endl;
			}
		}

	int *task_completed;
	int nb_tasks_completed = 0;
	char input_fname[1000];
	char target_fname[1000];

	int *random_perm = NULL;

	if (f_randomized) {
		int m, n;
		
		int_matrix_read_csv(randomized_fname, random_perm, m, n, verbose_level);
		if (n != 1) {
			cout << "int_matrix_read_csv n != n" << endl;
			exit(1);
			}
		if (m != N) {
			cout << "int_matrix_read_csv m != N" << endl;
			exit(1);
			}
		cout << "read the random permutation of degree " << m << " from file " << randomized_fname << endl;
		}

	
	task_completed = NEW_int(N);

	for (i = 0; i < N; i++) {
		c = list_of_cases[i];
		if (f_input_file_mask) {
			sprintf(input_fname, input_file_mask, c);
			sprintf(target_fname, target_file_mask, c);
			if (file_size(input_fname) <= 0) {
				cout << "The input file does not exist: please check the file " << input_fname << endl;
				exit(1);
				}
			}
		if (file_size(target_fname) > 0) {
			task_completed[i] = 1;
			nb_tasks_completed++;
			}
		else {
			task_completed[i] = 0;
			}
		}
	
	cout << "number of completed tasks = " << nb_tasks_completed << " / " << N << endl;

	cout << "there are " << N - nb_tasks_completed << " open tasks" << endl;
	for (i = 0; i < N; i++) {
		if (task_completed[i] == 0) {
			cout << i << " = " << list_of_cases[i] << ", ";
			}
		}
	cout << endl;


	int idx;
	
	if (f_command_mask) {

		job_table *JT;

		JT = NEW_OBJECTS(job_table, J);
		for (i = 0; i < J; i++) {
			JT[i].job = i;
			JT[i].f_task_assigned = FALSE;
			JT[i].target_file_mask = target_file_mask;
			JT[i].command_mask = command_mask;
			}

		while (TRUE) {

			int t, tt, j;
	
			for (t = 0; t < N; t++) {
				if (f_randomized) {
					tt = random_perm[t];
					}
				else {
					tt = t;
					}
				if (task_completed[tt] == 0 && 
					!int_vec_search(excluded_cases, nb_excluded_cases, tt, idx)) {
					j = find_free_job(JT, J);
					if (j == -1) {
						break;
						}
					assign_task(JT, tt, j, list_of_cases, log_prefix, 
						f_batch, job_fname, batch_template, template_nb_times, 
						verbose_level);
					task_completed[tt] = 2;
					}
				}

			if (!f_reload) {
				break;
				}

			system("sleep 5");


			for (i = 0; i < J; i++) {

				if (!JT[i].f_task_assigned) {
					continue;
					}
				t = JT[i].task;
				c = JT[i].the_case;
				//sprintf(target_fname, target_file_mask, c);
				if (file_size(JT[i].target_fname) > 0) {
					cout << "task completed: " << t << " = " << c << endl;
					task_completed[t] = 1;
					JT[i].f_task_assigned = FALSE;
					nb_tasks_completed++;
					}
				}
	
			cout << "number of completed tasks = " << nb_tasks_completed << " / " << N << endl;

			cout << "there are " << N - nb_tasks_completed << " open tasks" << endl;

			if (nb_tasks_completed == N) {
				break;
				}

			if (N - nb_tasks_completed < 100) {
				for (i = 0; i < N; i++) {
					if (task_completed[i] == 0) {
						cout << i << " = " << list_of_cases[i] << ", ";
						}
					}
				cout << endl;
				}
			else {
				cout << "too many to print" << endl;
				}




			}
		}
}

int find_free_job(job_table *JT, int J)
{
	int i;

	for (i = 0; i < J; i++) {
		if (!JT[i].f_task_assigned) {
			return i;
			}
		}
	return -1;
}

void assign_task(job_table *JT, int t, int j, 
	int *list_of_cases, const char *log_prefix, 
	int f_batch, const char *job_fname, const char *batch_template, int template_nb_times, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char str[1000];
	int c;

	c = list_of_cases[t];
	if (f_v) {
		cout << "assign_task assigning task " << t << " which is case " << c << " to job " << j << endl;
		}
	sprintf(JT[j].target_fname, JT[j].target_file_mask, c);
	sprintf(JT[j].command, JT[j].command_mask, c, c, c);
	if (f_batch) {
		str[0] = 0;
		}
	else {
		sprintf(str, "  >%slog_%d &", log_prefix, c);
		}
	strcat(JT[j].command, str);
	JT[j].task = t;
	JT[j].the_case = c;
	JT[j].f_task_assigned = TRUE;
	if (f_batch) {
		sprintf(JT[j].batch_fname, job_fname, t);
		if (template_nb_times == 0) {
			strcpy(JT[j].batch_file, batch_template);
			}
		else if (template_nb_times == 1) {
			sprintf(JT[j].batch_file, batch_template, t);
			}
		else if (template_nb_times == 2) {
			sprintf(JT[j].batch_file, batch_template, t, t);
			}
		else if (template_nb_times == 3) {
			sprintf(JT[j].batch_file, batch_template, t, t, t);
			}
		else if (template_nb_times == 4) {
			sprintf(JT[j].batch_file, batch_template, t, t, t, t);
			}
		else {
			cout << "template_nb_times is too large" << endl;
			exit(1);
			}
		{
			ofstream fp(JT[j].batch_fname);
			int i, l, h;

			l = strlen(JT[j].batch_file);
			i = 0;
			while (i < l) {
				if (JT[j].batch_file[i] == '\\' && JT[j].batch_file[i + 1] == 'n') {
					//cout << "detected \\n at position " << i << ":" << JT[j].batch_file << endl;
					JT[j].batch_file[i] = 0;
					fp << JT[j].batch_file << endl;
					for (h = 0; JT[j].batch_file[i + 2 + h] != 0; h++) {
						JT[j].batch_file[h] = JT[j].batch_file[i + 2 + h];
						}
					JT[j].batch_file[h] = 0;
					l = h;
					i = 0;
					}
				else {
					i++;
					}
				}
			//fp << JT[j].batch_file << endl;
		}
		cout << "Written file " << JT[j].batch_fname << " of size " << file_size(JT[j].batch_fname) << endl;
		}
	cout << "target_fname: " << JT[j].target_fname << endl;
	cout << "assign task: " << JT[j].command << endl;
	system(JT[j].command);
}

job_table::job_table()
{
}

job_table::~job_table()
{
}


