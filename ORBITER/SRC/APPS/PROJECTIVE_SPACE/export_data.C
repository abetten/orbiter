// export_data.C
//
// Anton Betten
//
// started:  January 1, 2018




#include "orbiter.h"


int main(int argc, char **argv)
{
	INT t0 = os_ticks();
	INT verbose_level = 0;
	INT i;
	INT f_q = FALSE;
	INT q;
	INT f_TP = FALSE;
	INT f_BLT = FALSE;
	INT k;

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
		else if (strcmp(argv[i], "-TP") == 0) {
			f_TP = TRUE;
			k = atoi(argv[++i]);
			cout << "-TP " << k << endl;
			}
		else if (strcmp(argv[i], "-BLT") == 0) {
			f_BLT = TRUE;
			k = atoi(argv[++i]);
			cout << "-BLT " << k << endl;
			}

		}

	INT f_v = (verbose_level >= 1);
	BYTE fname[1000];


	if (f_TP) {
		INT nb_reps;
		INT *p;
		INT *Table;
		INT width;
		
		sprintf(fname, "TP_%ld_%ld.csv", q, k);
		nb_reps = TP_nb_reps(q, k);
		width = i_power_j(q, k) + 1;
		Table = NEW_INT(nb_reps * width);
		for (i = 0; i < nb_reps; i++) {
			p = TP_representative(q, k, i);
			INT_vec_copy(p, Table + i * width, width);
			}
		INT_matrix_write_csv(fname, Table, nb_reps, width);
		FREE_INT(Table);
		}
	else if (f_BLT) {
		INT *BLT;

		BLT = BLT_representative(q, k);
		sprintf(fname, "BLT_%ld_%ld.txt", q, k);
		write_set_to_file(fname, BLT, q + 1, verbose_level - 1);
		}
	if (f_v) {
		cout << "written file " << fname << " of size " << file_size(fname) << endl;
		}
	the_end(t0);
}

