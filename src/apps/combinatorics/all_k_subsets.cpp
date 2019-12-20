// all_k_subsets.cpp
// 
// Anton Betten
// January 28, 2015

#include "orbiter.h"

using namespace std;

using namespace orbiter;


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i, h;
	int f_n = FALSE;
	int n;
	int f_k = FALSE;
	int k;
	int *set;
	int N;
	char fname[1000];

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
		else if (strcmp(argv[i], "-k") == 0) {
			f_k = TRUE;
			k = atoi(argv[++i]);
			cout << "-k " << k << endl;
			}
		}

	if (!f_n) {
		cout << "Please use option -n <n>" << endl;
		exit(1);
		}
	if (!f_k) {
		cout << "Please use option -k <k>" << endl;
		exit(1);
		}
	
	combinatorics_domain Combi;

	sprintf(fname, "all_k_subsets_%d_%d.tree", n, k);
	set = NEW_int(k);
	N = Combi.int_n_choose_k(n, k);

	
	{
	ofstream fp(fname);
	
	for (h = 0; h < N; h++) {
		Combi.unrank_k_subset(h, set, n, k);
		fp << k;
		for (i = 0; i < k; i++) {
			fp << " " << set[i];
			}
		fp << endl;
		}
	fp << "-1" << endl;
	}
	FREE_int(set);
}
