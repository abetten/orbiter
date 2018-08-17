// unrank.C
// 
// Anton Betten
// January 21, 2016

#include "orbiter.h"


int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;
	INT f_k_subset = FALSE;
	INT n, k, r;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-k_subset") == 0) {
			f_k_subset = TRUE;
			n = atoi(argv[++i]);
			k = atoi(argv[++i]);
			r = atoi(argv[++i]);
			cout << "-k_subset " << n << " " << k << " " << r  << endl;
			}
		}

	
	if (f_k_subset) {
		INT *set = NEW_INT(k);
		unrank_k_subset(r, set, n, k);
		cout << "set of rank " << r << " is ";
		INT_vec_print(cout, set, k);
		cout << endl;
		FREE_INT(set);
		}

}

