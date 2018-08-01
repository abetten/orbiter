// schlaefli.C
// 
// Anton Betten
// January 21, 2015

#include "orbiter.h"


INT evaluate_cubic_form(finite_field *F, INT *v);

int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i, j, rr, sz, N;
	finite_field *F;
	grassmann *Gr;
	INT *Adj;
	INT *M1;
	INT *M2;
	INT *M;
	INT v[2];
	INT w[4];
	INT *List;
	INT f_q = FALSE;
	INT q;
	INT n = 4;
	INT k = 2;

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

	F = new finite_field;
	F->init(q, verbose_level);

	Gr = new grassmann;
	Gr->init(n, k, F, verbose_level);

	M1 = NEW_INT(k * n);
	M2 = NEW_INT(k * n);
	M = NEW_INT(2 * k * n);

	N = generalized_binomial(n, k, q);

	List = NEW_INT(N);
	sz = 0;

	for (i = 0; i < N; i++) {
		Gr->unrank_INT_here(M1, i, 0 /* verbose_level */);
		
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
	

	Adj = NEW_INT(sz * sz);
	INT_vec_zero(Adj, sz * sz);

	for (i = 0; i < sz; i++) {
		Gr->unrank_INT_here(M1, List[i], 0 /* verbose_level */);

		for (j = i + 1; j < sz; j++) {
			Gr->unrank_INT_here(M2, List[j], 0 /* verbose_level */);

			INT_vec_copy(M1, M, k * n);
			INT_vec_copy(M2, M + k * n, k * n);

			rr = F->rank_of_rectangular_matrix(M, 2 * k, n, 0 /* verbose_level */);
			if (rr == 2 * k) {
				Adj[i * sz + j] = 1;
				Adj[j * sz + i] = 1;
				}
			}
		}

	colored_graph *CG;
	BYTE fname[1000];

	CG = new colored_graph;
	CG->init_adjacency_no_colors(sz, Adj, verbose_level);

	sprintf(fname, "Schlaefli_%ld.colored_graph", q);

	CG->save(fname, verbose_level);

	delete CG;
	FREE_INT(List);
	FREE_INT(Adj);
	FREE_INT(M1);
	FREE_INT(M2);
	FREE_INT(M);
	delete Gr;
	delete F;
}

INT evaluate_cubic_form(finite_field *F, INT *v)
{
	INT a, i;

	a = 0;
	for (i = 0; i < 4; i++) {
		a = F->add(a, F->power(v[i], 3));
		}
	return a;
}

