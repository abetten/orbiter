// johnson.C
// 
// Anton Betten
// January 20, 2015

#include "orbiter.h"


int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i, j, N, sz;
	INT *Adj;
	INT *set1;
	INT *set2;
	INT *set3;
	INT f_n = FALSE;
	INT n;
	INT f_k = FALSE;
	INT k;
	INT f_s = FALSE;
	INT s;

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

	N = INT_n_choose_k(n, k);
	

	Adj = NEW_INT(N * N);
	INT_vec_zero(Adj, N * N);

	set1 = NEW_INT(k);
	set2 = NEW_INT(k);
	set3 = NEW_INT(k);
	
	for (i = 0; i < N; i++) {
		unrank_k_subset(i, set1, n, k);
		for (j = i + 1; j < N; j++) {
			unrank_k_subset(j, set2, n, k);

			INT_vec_intersect_sorted_vectors(set1, k, set2, k, set3, sz);
			if (sz == s) {
				Adj[i * N + j] = 1;
				Adj[j * N + 1] = 1;
				}
			}
		}

	colored_graph *CG;
	BYTE fname[1000];

	CG = new colored_graph;
	CG->init_adjacency_no_colors(N, Adj, verbose_level);

	sprintf(fname, "Johnson_%ld_%ld_%ld.colored_graph", n, k, s);

	CG->save(fname, verbose_level);

	delete CG;
	FREE_INT(Adj);
	FREE_INT(set1);
	FREE_INT(set2);
	FREE_INT(set3);
}

