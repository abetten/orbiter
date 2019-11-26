// run_blt.cpp
// 
// Anton Betten
// January 23, 2018
//
// 
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;



int main(int argc, char **argv)
{
	int i;
	int verbose_level = 0;
	int f_orbiter_path = FALSE;
	const char *orbiter_path = NULL;
	int f_q = FALSE;
	int q = 0;
	int f_slice = FALSE;
	int slice_r = 0;
	int slice_m = 0;

	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-orbiter_path") == 0) {
			f_orbiter_path = TRUE;
			orbiter_path = argv[++i];
			cout << "-orbiter_path " << orbiter_path << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-slice") == 0) {
			f_slice = TRUE;
			slice_r = atoi(argv[++i]);
			slice_m = atoi(argv[++i]);
			cout << "-slice " << slice_r << " " << slice_m << endl;
			}
		}
	if (!f_orbiter_path) {
		cout << "please use -orbiter_path <orbiter_path>" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use -q <q>" << endl;
		exit(1);
		}


	int verbose_level_run = verbose_level - 2;
	char cmd1[1000];
	char cmd2[1000];
	char cmd3[1000];
	char cmd4[1000];
	char cmd5[1000];
	char cmd6[1000];
	char cmd7[1000];
	char cmd8[1000];
	int t[10];
	int dt[10];
	int Mem[10];


	system("rm -rf G ISO SOLUTIONS STARTER_DIR SYSTEMS");
	system("mkdir G ISO SOLUTIONS STARTER_DIR SYSTEMS");


	sprintf(cmd1, "%s/SRC/APPS/BLT/blt_main.out -v %d -BLT -q %d -starter_size 5 -input_prefix STARTER_DIR/ -output_prefix SYSTEMS/ -solution_prefix SOLUTIONS/ -base_fname BLT_%d  -schreier 5 -starter -W", orbiter_path, verbose_level_run + 3, q, q);

	if (f_slice) {
		sprintf(cmd2, "%s/SRC/APPS/BLT/blt_main.out -v %d -BLT -q %d -starter_size 5 -input_prefix STARTER_DIR/ -output_prefix SYSTEMS/ -solution_prefix SOLUTIONS/ -base_fname BLT_%d -create_graphs %d %d 4 -lex -output_prefix G/", orbiter_path, verbose_level_run, q, q, slice_r, slice_m);
		}
	else {
		sprintf(cmd2, "%s/SRC/APPS/BLT/blt_main.out -v %d -BLT -q %d -starter_size 5 -input_prefix STARTER_DIR/ -output_prefix SYSTEMS/ -solution_prefix SOLUTIONS/ -base_fname BLT_%d -create_graphs 0 1 4 -lex -output_prefix G/", orbiter_path, verbose_level_run, q, q);
		}


	if (f_slice) {
		sprintf(cmd3, "%s/SRC/APPS/TOOLS/all_rainbow_cliques.out -v %d -list_of_cases G/list_of_cases_BLT_%d_5_%d_%d.txt G/graph_BLT_%d_5_%%d.bin -output_file SOLUTIONS/BLT_%d_solutions_5_%d_%d.txt", orbiter_path, verbose_level_run, q, slice_r, slice_m, q, q, slice_r, slice_m);
		}
	else {
		sprintf(cmd3, "%s/SRC/APPS/TOOLS/all_rainbow_cliques.out -v %d -list_of_cases G/list_of_cases_BLT_%d_5_0_1.txt G/graph_BLT_%d_5_%%d.bin -output_file SOLUTIONS/BLT_%d_solutions_5_0_1.txt", orbiter_path, verbose_level_run, q, q, q);
		}


//stage4:

	if (f_slice) {
		sprintf(cmd4, "sleep 1");
		}
	else {
		sprintf(cmd4, "%s/SRC/APPS/BLT/blt_main.out -v %d -BLT -q %d -starter_size 5 -input_prefix STARTER_DIR/ -output_prefix SYSTEMS/ -solution_prefix SOLUTIONS/ -base_fname BLT_%d -build_db", orbiter_path, verbose_level_run, q, q);
		}

	if (f_slice) {
		sprintf(cmd5, "sleep 1");
		}
	else {
		sprintf(cmd5, "%s/SRC/APPS/BLT/blt_main.out -v %d -BLT -q %d -starter_size 5 -input_prefix STARTER_DIR/ -output_prefix SYSTEMS/ -solution_prefix SOLUTIONS/ -base_fname BLT_%d -read_solutions_from_clique_finder G/list_of_cases_BLT_%d_5_0_1.txt", orbiter_path,  verbose_level_run, q, q, q);
		}

	if (f_slice) {
		sprintf(cmd6, "sleep 1");
		}
	else {
		sprintf(cmd6, "%s/SRC/APPS/BLT/blt_main.out -v %d -BLT -q %d -starter_size 5 -input_prefix STARTER_DIR/ -output_prefix SYSTEMS/ -solution_prefix SOLUTIONS/ -base_fname BLT_%d -compute_orbits", orbiter_path, verbose_level_run, q, q);
		}

	if (f_slice) {
		sprintf(cmd7, "sleep 1");
		}
	else {
		sprintf(cmd7, "%s/SRC/APPS/BLT/blt_main.out -v %d -BLT -q %d -starter_size 5 -input_prefix STARTER_DIR/ -output_prefix SYSTEMS/ -solution_prefix SOLUTIONS/ -base_fname BLT_%d -isomorph_testing -print_interval 250", orbiter_path, verbose_level_run, q, q);
		}

	if (f_slice) {
		sprintf(cmd8, "sleep 1");
		}
	else {
		sprintf(cmd8, "%s/SRC/APPS/BLT/blt_main.out -v %d -BLT -q %d -starter_size 5 -input_prefix STARTER_DIR/ -output_prefix SYSTEMS/ -solution_prefix SOLUTIONS/ -base_fname BLT_%d -report", orbiter_path, verbose_level_run, q, q);
		}

	int m, n;
	int *M;
	char fname[1000];

	sprintf(fname, "memory_usage.csv");
	t[0] = os_seconds_past_1970();
	file_io Fio;


	system(cmd1);
	t[1] = os_seconds_past_1970();
	Fio.int_matrix_read_csv(fname, M, m, n, 0 /* verbose_level */);
	Mem[0] = M[0];
	FREE_int(M);

	system(cmd2);
	t[2] = os_seconds_past_1970();
	Fio.int_matrix_read_csv(fname, M, m, n, 0 /* verbose_level */);
	Mem[1] = M[0];
	FREE_int(M);

	system(cmd3);
	t[3] = os_seconds_past_1970();
	Fio.int_matrix_read_csv(fname, M, m, n, 0 /* verbose_level */);
	Mem[2] = M[0];
	FREE_int(M);

	system(cmd4);
	t[4] = os_seconds_past_1970();
	Fio.int_matrix_read_csv(fname, M, m, n, 0 /* verbose_level */);
	Mem[3] = M[0];
	FREE_int(M);

	system(cmd5);
	t[5] = os_seconds_past_1970();
	Fio.int_matrix_read_csv(fname, M, m, n, 0 /* verbose_level */);
	Mem[4] = M[0];
	FREE_int(M);

	system(cmd6);
	t[6] = os_seconds_past_1970();
	Fio.int_matrix_read_csv(fname, M, m, n, 0 /* verbose_level */);
	Mem[5] = M[0];
	FREE_int(M);

	system(cmd7);
	t[7] = os_seconds_past_1970();
	Fio.int_matrix_read_csv(fname, M, m, n, 0 /* verbose_level */);
	Mem[6] = M[0];
	FREE_int(M);

	system(cmd8);
	t[8] = os_seconds_past_1970();
	Fio.int_matrix_read_csv(fname, M, m, n, 0 /* verbose_level */);
	Mem[7] = M[0];
	FREE_int(M);

	for (i = 0; i < 8; i++) {
		dt[i] = t[i + 1] - t[i];
		}
	dt[8] = t[8] - t[0];
		
	char fname2[1000];
	int Stats[9+8];

	int_vec_copy(dt, Stats, 9);
	int_vec_copy(Mem, Stats + 9, 8);

	sprintf(fname2, "BLT_%d_stats.csv", q);
	Fio.int_matrix_write_csv(fname2, Stats, 1, 17);

	cout << "Written file " << fname2 << " of size " << Fio.file_size(fname2) << endl;
}




