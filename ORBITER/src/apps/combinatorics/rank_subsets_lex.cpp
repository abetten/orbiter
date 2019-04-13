// rank_subsets_lex.C
// 
// Anton Betten
// Jul 15, 2016

#include "orbiter.h"

using namespace std;


using namespace orbiter;


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_n = FALSE;
	int n;
	int *set;
	int N, r, sz;
	number_theory_domain NT;

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
		}

	if (!f_n) {
		cout << "Please use option -n <n>" << endl;
		exit(1);
		}
	set = NEW_int(n);

	N = NT.i_power_j(2, n);
	for (i = 0; i < N; i++) {
		cout << " rank " << i << " : ";
		unrank_subset(set, sz, n, i);
		cout << " : set ";
		int_vec_print(cout, set, sz);
		cout << " : ";
		r = rank_subset(set, sz, n);
		cout << " has rank " << r << endl;
		}

	char fname[1000];
	int h;

	sprintf(fname, "subsets_of_%d.tree", n);
	{
	ofstream fp(fname);
	
	N = NT.i_power_j(2, n);
	for (h = 0; h < N; h++) {
		unrank_subset(set, sz, n, h);
		fp << sz;
		for (i = 0; i < sz; i++) {
			fp << " " << set[i];
			}
		fp << endl;
		}
	fp << "-1" << endl;
	}
#if 0
	set[0] = 1;
	set[1] = 3;
	sz = 2;
	r = rank_subset(set, sz, n);
	cout << "The set ";
	int_vec_print(cout, set, sz);
	cout << " has rank " << r << endl;
#endif
}


