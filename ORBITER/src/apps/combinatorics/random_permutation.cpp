// random_permutation.C
// 
// Anton Betten
// April 22, 2016

#include "orbiter.h"

void choose_random_permutation(INT n, INT f_save, const char *fname, INT verbose_level);

int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;
	INT f_n = FALSE;
	INT n;
	INT f_save = FALSE;
	const char *fname = NULL;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			fname = argv[++i];
			cout << "-save " << fname << endl;
			}
		}

	if (!f_n) {
		cout << "Please use option -n <n>" << endl;
		exit(1);
		}

	choose_random_permutation(n, f_save, fname, verbose_level);
}

void choose_random_permutation(INT n, INT f_save, const char *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "choose_random_permutation" << endl;
		}

	INT *P;

	P = NEW_INT(n);
	random_permutation(P, n);

	if (f_save) {
		INT_vec_write_csv(P, n, fname, "perm");
		}
	if (f_v) {
		cout << "choose_random_permutation done" << endl;
		}
}

