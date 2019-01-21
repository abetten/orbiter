// schlaefli.C
// 
// Anton Betten
// January 21, 2015

#include "orbiter.h"


using namespace orbiter;


int evaluate_cubic_form(finite_field *F, int *v);

int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i, j, rr, sz, N;
	finite_field *F;
	grassmann *Gr;
	int *Adj;
	int *M1;
	int *M2;
	int *M;
	int v[2];
	int w[4];
	int *List;
	int f_q = FALSE;
	int q;
	int n = 4;
	int k = 2;

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
		}

	if (!f_q) {
		cout << "Please use option -q <q>" << endl;
		exit(1);
		}

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	Gr = NEW_OBJECT(grassmann);
	Gr->init(n, k, F, verbose_level);

	M1 = NEW_int(k * n);
	M2 = NEW_int(k * n);
	M = NEW_int(2 * k * n);

	N = generalized_binomial(n, k, q);

	List = NEW_int(N);
	sz = 0;

	for (i = 0; i < N; i++) {
		Gr->unrank_int_here(M1, i, 0 /* verbose_level */);
		
		for (j = 0; j < q + 1; j++) {
			F->unrank_point_in_PG(v, 2, j);
			F->mult_vector_from_the_left(v, M1, w, k, n);
			if (evaluate_cubic_form(F, w)) {
				break;
				}
			}
		if (j == q + 1) {
			List[sz++] = i;
			} 
		}
	cout << "We found " << sz << " lines" << endl;
	

	Adj = NEW_int(sz * sz);
	int_vec_zero(Adj, sz * sz);

	for (i = 0; i < sz; i++) {
		Gr->unrank_int_here(M1, List[i], 0 /* verbose_level */);

		for (j = i + 1; j < sz; j++) {
			Gr->unrank_int_here(M2, List[j], 0 /* verbose_level */);

			int_vec_copy(M1, M, k * n);
			int_vec_copy(M2, M + k * n, k * n);

			rr = F->rank_of_rectangular_matrix(M, 2 * k, n, 0 /* verbose_level */);
			if (rr == 2 * k) {
				Adj[i * sz + j] = 1;
				Adj[j * sz + i] = 1;
				}
			}
		}

	colored_graph *CG;
	char fname[1000];

	CG = NEW_OBJECT(colored_graph);
	CG->init_adjacency_no_colors(sz, Adj, verbose_level);

	sprintf(fname, "Schlaefli_%d.colored_graph", q);

	CG->save(fname, verbose_level);

	FREE_OBJECT(CG);
	FREE_int(List);
	FREE_int(Adj);
	FREE_int(M1);
	FREE_int(M2);
	FREE_int(M);
	FREE_OBJECT(Gr);
	FREE_OBJECT(F);
}

int evaluate_cubic_form(finite_field *F, int *v)
{
	int a, i;

	a = 0;
	for (i = 0; i < 4; i++) {
		a = F->add(a, F->power(v[i], 3));
		}
	return a;
}

