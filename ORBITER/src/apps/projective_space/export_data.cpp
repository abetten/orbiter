// export_data.C
//
// Anton Betten
//
// started:  January 1, 2018




#include "orbiter.h"


int main(int argc, char **argv)
{
	int t0 = os_ticks();
	int verbose_level = 0;
	int i;
	int f_q = FALSE;
	int q;
	int f_spread = FALSE;
	int f_BLT = FALSE;
	int k;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-spread") == 0) {
			f_spread = TRUE;
			k = atoi(argv[++i]);
			cout << "-spread " << k << endl;
			}
		else if (strcmp(argv[i], "-BLT") == 0) {
			f_BLT = TRUE;
			k = atoi(argv[++i]);
			cout << "-BLT " << k << endl;
			}

		}

	int f_v = (verbose_level >= 1);
	char fname[1000];


	if (f_spread) {
		int nb_reps;
		int *p;
		int *Table;
		int width;
		
		sprintf(fname, "Spread_%d_%d.csv", q, k);
		nb_reps = Spread_nb_reps(q, k);
		width = i_power_j(q, k) + 1;
		Table = NEW_int(nb_reps * width);
		for (i = 0; i < nb_reps; i++) {
			int sz;
			
			p = Spread_representative(q, k, i, sz);
			int_vec_copy(p, Table + i * width, width);
			}
		int_matrix_write_csv(fname, Table, nb_reps, width);
		FREE_int(Table);
		}
	else if (f_BLT) {
		int *BLT;

		BLT = BLT_representative(q, k);
		sprintf(fname, "BLT_%d_%d.txt", q, k);
		write_set_to_file(fname, BLT, q + 1, verbose_level - 1);
		}
	if (f_v) {
		cout << "written file " << fname << " of size "
				<< file_size(fname) << endl;
		}
	the_end(t0);
}

