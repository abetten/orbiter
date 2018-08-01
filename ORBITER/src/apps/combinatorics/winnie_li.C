// winnie_li.C
//
// Anton Betten
// November 11, 2011
//
// creates the graphs described in the paper:
// Wen Ching Winnie Li: Character Sums and Abelian Ramanujan Graphs 1991

#include "orbiter.h"


INT t0;

void do_it(INT q, INT f_index, INT index, INT verbose_level);


int main(int argc, char **argv)
{
	INT i;
	INT verbose_level = 0;
	INT f_q = FALSE;
	INT q = 0;
	INT f_index = FALSE;
	INT index = 0;
	
	t0 = os_ticks();

	
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
		else if (strcmp(argv[i], "-index") == 0) {
			f_index = TRUE;
			index = atoi(argv[++i]);
			cout << "-index " << index << endl;
			}
		}
	


	if (!f_q) {
		cout << "please specify -q <q>" << endl;
		exit(1);
		}

	do_it(q, f_index, index, verbose_level);
	
}

void do_it(INT q, INT f_index, INT index, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	finite_field *F;
	INT i, j, h, u, p, k, co_index, q1, relative_norm;
	INT *N1;
	INT *Adj;
	

	F = new finite_field;
	F->init(q, verbose_level - 1);
	p = F->p;

	if (!f_index) {
		index = F->e;
		}

	co_index = F->e / index;

	if (co_index * index != F->e) {
		cout << "the index has to divide the field degree" << endl;
		exit(1);
		}
	q1 = i_power_j(p, co_index);

	k = (q - 1) / (q1 - 1);

	cout << "q=" << q << endl;
	cout << "index=" << index << endl;
	cout << "co_index=" << co_index << endl;
	cout << "q1=" << q1 << endl;
	cout << "k=" << k << endl;

	relative_norm = 0;
	j = 1;
	for (i = 0; i < index; i++) {
		relative_norm += j;
		j *= q1;
		}
	cout << "relative_norm=" << relative_norm << endl;

	N1 = NEW_INT(k);
	j = 0;
	for (i = 0; i < q; i++) {
		if (F->power(i, relative_norm) == 1) {
			N1[j++] = i;
			}
		}
	if (j != k) {
		cout << "j != k" << endl;
		exit(1);
		}
	cout << "found " << k << " norm-one elements:" << endl;
	INT_vec_print(cout, N1, k);
	cout << endl;

	Adj = NEW_INT(q * q);
	for (i = 0; i < q; i++) {
		for (h = 0; h < k; h++) {
			j = N1[h];
			u = F->add(i, j);
			Adj[i * q + u] = 1;
			Adj[u * q + i] = 1;
			}
		}


	colored_graph *CG;
	BYTE fname[1000];

	CG = new colored_graph;
	CG->init_adjacency_no_colors(q, Adj, verbose_level);

	if (f_index) {
		sprintf(fname, "Winnie_Li_%ld_%ld.colored_graph", q, index);
		}
	else {
		sprintf(fname, "Winnie_Li_%ld.colored_graph", q);
		}

	CG->save(fname, verbose_level);





	delete CG;
	FREE_INT(Adj);
	FREE_INT(N1);
	delete F;


}


