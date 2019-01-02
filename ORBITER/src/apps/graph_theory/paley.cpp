// paley.C
// 
// Anton Betten
// January 19, 2015

#include "orbiter.h"


int main(int argc, char **argv)
{
	int verbose_level = 0;
	finite_field *F;
	int i, j, a;
	int *Adj;
	int f_q = FALSE;
	int q;
	int *f_is_square;

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

	if (EVEN(q)) {
		cout << "q must be odd" << endl;
		exit(1);
		}
	if (!DOUBLYEVEN(q - 1)) {
		cout << "q must be congruent to 1 modulo 4" << endl;
		}
	
	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	f_is_square = NEW_int(q);
	int_vec_zero(f_is_square, q);
	
	for (i = 0; i < q; i++) {
		j = F->mult(i, i);
		f_is_square[j] = TRUE;
		}

	Adj = NEW_int(q * q);
	int_vec_zero(Adj, q * q);
	
	for (i = 0; i < q; i++) {
		for (j = i + 1; j < q; j++) {
			a = F->add(i, F->negate(j));
			if (f_is_square[a]) {
				Adj[i * q + j] = 1;
				Adj[j * q + 1] = 1;
				}
			}
		}


	colored_graph *CG;
	char fname[1000];

	CG = NEW_OBJECT(colored_graph);
	CG->init_adjacency_no_colors(q, Adj, verbose_level);

	sprintf(fname, "Paley_%d.colored_graph", q);

	CG->save(fname, verbose_level);

	FREE_OBJECT(CG);
	FREE_int(Adj);
	FREE_int(f_is_square);
	FREE_OBJECT(F);
}

