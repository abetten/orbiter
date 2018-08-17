// grassmann.C
// 

#include "orbiter.h"


int main(int argc, char **argv)
{
	finite_field *F;
	grassmann *Gr;
	INT verbose_level = 0;
	INT i, j, rr, N;
	INT *M1; // [k * n]
	INT *M2; // [k * n]
	INT *M; // [2 * k * n]
	INT *Adj;
	INT f_q = FALSE;
	INT q;
	INT f_k = FALSE;
	INT k = 0; // vector space dimension of subspaces
	INT f_n = FALSE;
	INT n = 0; // vector space dimension of whole space
	INT f_r = FALSE;
	INT r = 0; // two subspaces are incident if the rank of their span is r

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
	

	F = new finite_field;
	F->init(q, verbose_level);


	Gr = new grassmann;
	Gr->init(n, k, F, verbose_level);

	N = generalized_binomial(n, k, q);

	M1 = NEW_INT(k * n);
	M2 = NEW_INT(k * n);
	M = NEW_INT(2 * k * n);

	Adj = NEW_INT(N * N);
	INT_vec_zero(Adj, N * N);

	for (i = 0; i < N; i++) {
		
		Gr->unrank_INT_here(M1, i, 0 /* verbose_level */);

		for (j = i + 1; j < N; j++) {

			Gr->unrank_INT_here(M2, j, 0 /* verbose_level */);
		
			INT_vec_copy(M1, M, k * n);
			INT_vec_copy(M2, M + k * n, k * n);

			rr = F->rank_of_rectangular_matrix(M, 2 * k, n, 0 /* verbose_level */);
			if (rr == r) {
				Adj[i * N + j] = 1;
				Adj[j * N + i] = 1;
				}
			}
		}


	colored_graph *CG;
	BYTE fname[1000];

	CG = new colored_graph;
	CG->init_adjacency_no_colors(N, Adj, verbose_level);

	sprintf(fname, "grassmann_graph_%ld_%ld_%ld_%ld.colored_graph", n, k, q, r);

	CG->save(fname, verbose_level);


	

	delete CG;
	FREE_INT(M1);
	FREE_INT(M2);
	FREE_INT(M);
	delete Gr;
	delete F;
}


