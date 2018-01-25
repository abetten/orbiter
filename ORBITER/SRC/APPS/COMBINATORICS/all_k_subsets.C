// all_k_subsets.C
// 
// Anton Betten
// January 28, 2015

#include "orbiter.h"


int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i, h;
	INT f_n = FALSE;
	INT n;
	INT f_k = FALSE;
	INT k;
	INT *set;
	INT N;
	BYTE fname[1000];

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
	
	sprintf(fname, "all_k_subsets_%ld_%ld.tree", n, k);
	set = NEW_INT(k);
	N = INT_n_choose_k(n, k);

	
	{
	ofstream fp(fname);
	
	for (h = 0; h < N; h++) {
		unrank_k_subset(h, set, n, k);
		fp << k;
		for (i = 0; i < k; i++) {
			fp << " " << set[i];
			}
		fp << endl;
		}
	fp << "-1" << endl;
	}
	FREE_INT(set);
}
