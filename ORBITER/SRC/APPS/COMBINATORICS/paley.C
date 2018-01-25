// paley.C
// 
// Anton Betten
// January 19, 2015

#include "orbiter.h"


int main(int argc, char **argv)
{
	INT verbose_level = 0;
	finite_field *F;
	INT i, j, a;
	INT *Adj;
	INT f_q = FALSE;
	INT q;
	INT *f_is_square;

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
	
	F = new finite_field;
	F->init(q, verbose_level);

	f_is_square = NEW_INT(q);
	INT_vec_zero(f_is_square, q);
	
	for (i = 0; i < q; i++) {
		j = F->mult(i, i);
		f_is_square[j] = TRUE;
		}

	Adj = NEW_INT(q * q);
	INT_vec_zero(Adj, q * q);
	
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
	BYTE fname[1000];

	CG = new colored_graph;
	CG->init_adjacency_no_colors(q, Adj, verbose_level);

	sprintf(fname, "Paley_%ld.colored_graph", q);

	CG->save(fname, verbose_level);

	delete CG;
	FREE_INT(Adj);
	FREE_INT(f_is_square);
	delete F;
}

