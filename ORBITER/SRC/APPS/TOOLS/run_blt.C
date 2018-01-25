// run_blt.C
// 
// Anton Betten
// January 23, 2018
//
// 
//

#include "orbiter.h"
#include "discreta.h"


int main(int argc, char **argv)
{
	INT i;
	INT verbose_level = 0;
	INT f_orbiter_path = FALSE;
	const BYTE *orbiter_path = NULL;
	INT f_q = FALSE;
	INT q = 0;

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
		}
	if (!f_orbiter_path) {
		cout << "please use -orbiter_path <orbiter_path>" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use -q <q>" << endl;
		exit(1);
		}


	INT verbose_level_run = verbose_level - 2;
	BYTE cmd1[1000];
	BYTE cmd2[1000];
	BYTE cmd3[1000];
	BYTE cmd4[1000];
	BYTE cmd5[1000];
	BYTE cmd6[1000];
	BYTE cmd7[1000];
	BYTE cmd8[1000];
	INT t[10];
	INT dt[10];


	system("rm -rf G ISO SOLUTIONS STARTER_DIR SYSTEMS");
	system("mkdir G ISO SOLUTIONS STARTER_DIR SYSTEMS");


	sprintf(cmd1, "%s/SRC/APPS/BLT/blt_main.out -v %ld -BLT -q %ld -starter_size 5 -input_prefix STARTER_DIR/ -output_prefix SYSTEMS/ -solution_prefix SOLUTIONS/ -base_fname BLT_%ld  -schreier 5 -starter -W", orbiter_path, verbose_level_run, q, q);

	sprintf(cmd2, "%s/SRC/APPS/BLT/blt_main.out -v %ld -BLT -q %ld -starter_size 5 -input_prefix STARTER_DIR/ -output_prefix SYSTEMS/ -solution_prefix SOLUTIONS/ -base_fname BLT_%ld -create_graphs 0 1 4 -lex -output_prefix G/", orbiter_path, verbose_level_run, q, q);


	sprintf(cmd3, "%s/SRC/APPS/TOOLS/all_rainbow_cliques.out -v %ld -list_of_cases G/list_of_cases_BLT_%ld_5_0_1.txt G/graph_BLT_%ld_5_%%ld.bin -output_file SOLUTIONS/BLT_%ld_solutions_5_0_1.txt", orbiter_path, verbose_level_run, q, q, q);


	sprintf(cmd4, "%s/SRC/APPS/BLT/blt_main.out -v %ld -BLT -q %ld -starter_size 5 -input_prefix STARTER_DIR/ -output_prefix SYSTEMS/ -solution_prefix SOLUTIONS/ -base_fname BLT_%ld -build_db", orbiter_path, verbose_level_run, q, q);

	sprintf(cmd5, "%s/SRC/APPS/BLT/blt_main.out -v %ld -BLT -q %ld -starter_size 5 -input_prefix STARTER_DIR/ -output_prefix SYSTEMS/ -solution_prefix SOLUTIONS/ -base_fname BLT_%ld -read_solutions_from_clique_finder G/list_of_cases_BLT_%ld_5_0_1.txt", orbiter_path,  verbose_level_run, q, q, q);

	sprintf(cmd6, "%s/SRC/APPS/BLT/blt_main.out -v %ld -BLT -q %ld -starter_size 5 -input_prefix STARTER_DIR/ -output_prefix SYSTEMS/ -solution_prefix SOLUTIONS/ -base_fname BLT_%ld -compute_orbits", orbiter_path, verbose_level_run, q, q);

	sprintf(cmd7, "%s/SRC/APPS/BLT/blt_main.out -v %ld -BLT -q %ld -starter_size 5 -input_prefix STARTER_DIR/ -output_prefix SYSTEMS/ -solution_prefix SOLUTIONS/ -base_fname BLT_%ld -isomorph_testing -print_interval 250", orbiter_path, verbose_level_run, q, q);

	sprintf(cmd8, "%s/SRC/APPS/BLT/blt_main.out -v %ld -BLT -q %ld -starter_size 5 -input_prefix STARTER_DIR/ -output_prefix SYSTEMS/ -solution_prefix SOLUTIONS/ -base_fname BLT_%ld -report", orbiter_path, verbose_level_run, q, q);



	t[0] = os_seconds_past_1970();
	system(cmd1);
	t[1] = os_seconds_past_1970();
	system(cmd2);
	t[2] = os_seconds_past_1970();
	system(cmd3);
	t[3] = os_seconds_past_1970();
	system(cmd4);
	t[4] = os_seconds_past_1970();
	system(cmd5);
	t[5] = os_seconds_past_1970();
	system(cmd6);
	t[6] = os_seconds_past_1970();
	system(cmd7);
	t[7] = os_seconds_past_1970();
	system(cmd8);
	t[8] = os_seconds_past_1970();

	for (i = 0; i < 8; i++) {
		dt[i] = t[i + 1] - t[i];
		}
	dt[8] = t[8] - t[0];
		
	BYTE fname[1000];

	sprintf(fname, "BLT_%ld_stats.csv", q);
	INT_matrix_write_csv(fname, dt, 1, 9);

	cout << "Written file " << fname << " of size " << file_size(fname) << endl;
}




