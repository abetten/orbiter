// grassmann_graph.cpp
// 

#include "orbiter.h"

using namespace std;


using namespace orbiter;



int main(int argc, char **argv)
{
	finite_field *F;
	grassmann *Gr;
	int verbose_level = 0;
	int i, j, rr, N;
	int *M1; // [k * n]
	int *M2; // [k * n]
	int *M; // [2 * k * n]
	int *Adj;
	int f_q = FALSE;
	int q;
	int f_k = FALSE;
	int k = 0; // vector space dimension of subspaces
	int f_n = FALSE;
	int n = 0; // vector space dimension of whole space
	int f_r = FALSE;
	int r = 0; // two subspaces are incident if the rank of their span is r
	combinatorics_domain Combi;

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
		else if (strcmp(argv[i], "-r") == 0) {
			f_r = TRUE;
			r = atoi(argv[++i]);
			cout << "-r " << r << endl;
			}
		}

	if (!f_q) {
		cout << "Please use option -q <q>" << endl;
		exit(1);
		}
	if (!f_n) {
		cout << "Please use option -n <n>" << endl;
		exit(1);
		}
	if (!f_k) {
		cout << "Please use option -k <k>" << endl;
		exit(1);
		}
	if (!f_r) {
		cout << "Please use option -r <r>" << endl;
		exit(1);
		}
	

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);


	Gr = NEW_OBJECT(grassmann);
	Gr->init(n, k, F, verbose_level);

	N = Combi.generalized_binomial(n, k, q);

	M1 = NEW_int(k * n);
	M2 = NEW_int(k * n);
	M = NEW_int(2 * k * n);

	Adj = NEW_int(N * N);
	int_vec_zero(Adj, N * N);

	for (i = 0; i < N; i++) {
		
		Gr->unrank_int_here(M1, i, 0 /* verbose_level */);

		for (j = i + 1; j < N; j++) {

			Gr->unrank_int_here(M2, j, 0 /* verbose_level */);
		
			int_vec_copy(M1, M, k * n);
			int_vec_copy(M2, M + k * n, k * n);

			rr = F->rank_of_rectangular_matrix(M, 2 * k, n, 0 /* verbose_level */);
			if (rr == r) {
				Adj[i * N + j] = 1;
				Adj[j * N + i] = 1;
				}
			}
		}


	colored_graph *CG;
	char fname[1000];

	CG = NEW_OBJECT(colored_graph);
	CG->init_adjacency_no_colors(N, Adj, verbose_level);

	sprintf(fname, "grassmann_graph_%d_%d_%d_%d.colored_graph", n, k, q, r);

	CG->save(fname, verbose_level);


	

	FREE_OBJECT(CG);
	FREE_int(M1);
	FREE_int(M2);
	FREE_int(M);
	FREE_OBJECT(Gr);
	FREE_OBJECT(F);
}


