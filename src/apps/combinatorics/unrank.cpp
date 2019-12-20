// unrank.cpp
// 
// Anton Betten
// January 21, 2016

#include "orbiter.h"

using namespace std;


using namespace orbiter;


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_k_subset = FALSE;
	int n, k, r;
	combinatorics_domain Combi;

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
		int *set = NEW_int(k);
		Combi.unrank_k_subset(r, set, n, k);
		cout << "set of rank " << r << " is ";
		int_vec_print(cout, set, k);
		cout << endl;
		FREE_int(set);
		}

}

