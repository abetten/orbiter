// johnson.C
// 
// Anton Betten
// January 20, 2015

#include "orbiter.h"


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i, j, N, sz;
	int *Adj;
	int *set1;
	int *set2;
	int *set3;
	int f_n = FALSE;
	int n;
	int f_k = FALSE;
	int k;
	int f_s = FALSE;
	int s;

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
		else if (strcmp(argv[i], "-s") == 0) {
			f_s = TRUE;
			s = atoi(argv[++i]);
			cout << "-s " << s << endl;
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
	if (!f_s) {
		cout << "Please use option -s <s>" << endl;
		exit(1);
		}

	N = int_n_choose_k(n, k);
	

	Adj = NEW_int(N * N);
	int_vec_zero(Adj, N * N);

	set1 = NEW_int(k);
	set2 = NEW_int(k);
	set3 = NEW_int(k);
	
	for (i = 0; i < N; i++) {
		unrank_k_subset(i, set1, n, k);
		for (j = i + 1; j < N; j++) {
			unrank_k_subset(j, set2, n, k);

			int_vec_intersect_sorted_vectors(set1, k, set2, k, set3, sz);
			if (sz == s) {
				Adj[i * N + j] = 1;
				Adj[j * N + 1] = 1;
				}
			}
		}

	colored_graph *CG;
	char fname[1000];

	CG = NEW_OBJECT(colored_graph);
	CG->init_adjacency_no_colors(N, Adj, verbose_level);

	sprintf(fname, "Johnson_%d_%d_%d.colored_graph", n, k, s);

	CG->save(fname, verbose_level);

	FREE_OBJECT(CG);
	FREE_int(Adj);
	FREE_int(set1);
	FREE_int(set2);
	FREE_int(set3);
}

