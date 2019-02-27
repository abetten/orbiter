// missing_files.C
// 
// Anton Betten
// February 23, 2018
//
// 
//

#include "orbiter.h"

using namespace std;



using namespace orbiter;


int t0;

int main(int argc, char **argv)
{
	int i;
	int verbose_level = 0;
	int f_N = FALSE;
	int N = 0;
	int f_N_min = FALSE;
	int N_min = 0;
	int f_N2 = FALSE;
	int N2 = 0;
	int f_mask = FALSE;
	const char *mask = NULL;
	int f_save = FALSE;
	const char *fname_out = NULL;
	int f_split = FALSE;
	int split_r = 0;
	int split_m = 0;
	
	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-N") == 0) {
			f_N = TRUE;
			N = atoi(argv[++i]);
			cout << "-N " << N << endl;
			}
		else if (strcmp(argv[i], "-N_min") == 0) {
			f_N_min = TRUE;
			N_min = atoi(argv[++i]);
			cout << "-N_min " << N_min << endl;
			}
		else if (strcmp(argv[i], "-N2") == 0) {
			f_N2 = TRUE;
			N2 = atoi(argv[++i]);
			cout << "-N2 " << N2 << endl;
			}
		else if (strcmp(argv[i], "-mask") == 0) {
			f_mask = TRUE;
			mask = argv[++i];
			cout << "-mask " << mask << endl;
			}
		else if (strcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			fname_out = argv[++i];
			cout << "-save " << fname_out << endl;
			}
		else if (strcmp(argv[i], "-split") == 0) {
			f_split = TRUE;
			split_r = atoi(argv[++i]);
			split_m = atoi(argv[++i]);
			cout << "-split " << split_r << " " << split_m << endl;
			}
		}
	if (!f_N) {
		cout << "Please use option -N <N>" << endl;
		exit(1);
		}
	if (!f_mask) {
		cout << "Please use option -mask <mask>" << endl;
		exit(1);
		}
	
	int nb_missing = 0;
	int *Missing = NULL;
	char fname[1000];
	int missing_n;


	if (f_N2) {
		Missing = NEW_int(N * N2 * 2);
		int j;

		missing_n = 2;
		for (i = 0; i < N; i++) {
			if (f_split) {
				if ((i % split_m) != split_r) {
					continue;
					}
				}
			if (f_N_min) {
				if (i < N_min) {
					continue;
					}
				}
			for (j = 0; j < N2; j++) {
				sprintf(fname, mask, i, j);
				if (file_size(fname) <= 0) {
					Missing[nb_missing * 2 + 0] = i;
					Missing[nb_missing * 2 + 1] = j;
					nb_missing++;
					}
				}
			}
		}
	else {	
		missing_n = 1;
		Missing = NEW_int(N);

		for (i = 0; i < N; i++) {
			if (f_split) {
				if ((i % split_m) != split_r) {
					continue;
					}
				}
			if (f_N_min) {
				if (i < N_min) {
					continue;
					}
				}
			sprintf(fname, mask, i);
			if (file_size(fname) <= 0) {
				Missing[nb_missing] = i;
				nb_missing++;
				}
			}
		}

	cout << "There are " << nb_missing << " missing files" << endl;

	if (f_save) {
		if (is_csv_file(fname_out)) {
			int_matrix_write_csv(fname_out,
					Missing, nb_missing, missing_n);
			}
		else {
			if (missing_n != 1) {
				cout << "missing_n != 1, "
						"cannot use write_set_to_file" << endl;
				exit(1);
				}
			write_set_to_file(fname_out, Missing, nb_missing,
					verbose_level);
			}
		cout << "Written file " << fname_out
				<< " of size " << file_size(fname_out) << endl;
		}

	the_end(t0);
}



