// shrikhande.C
//
// Anton Betten
// February 16, 2015
//

#include "orbiter.h"


INT t0;

void do_it(INT verbose_level);



int main(int argc, char **argv)
{
	INT i;
	INT verbose_level = 0;
	
	t0 = os_ticks();

	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		}
	



	do_it(verbose_level);
	
}

void do_it(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "do_it" << endl;
		}

	action *A;
	longinteger_object go;
	INT goi;
	INT **Elt_S;
	vector_ge *gens_G;
	INT *v;
	INT n = 8;
	INT i, j, h;
	INT nb_G, nb_S;

	A = new action;
	A->init_symmetric_group(n, verbose_level);
	A->group_order(go);

	goi = go.as_INT();
	cout << "Created group Sym(" << n << ") of size " << goi << endl;


	nb_G = 2;
	nb_S = 6;
	gens_G = new vector_ge;
	gens_G->init(A);
	gens_G->allocate(nb_G);


	Elt_S = NEW_PINT(nb_S);
	for (i = 0; i < nb_S; i++) {
		Elt_S[i] = NEW_INT(A->elt_size_in_INT);
		}
	v = NEW_INT(n);


	for (i = 0; i < nb_G; i++) {
		if (i == 0) {
			for (j = 0; j < 4; j++) {
				v[j] = (j + 1) % 4;
				}
			for (j = 0; j < 4; j++) {
				v[4 + j] = 4 + j;
				}
			}
		else {
			for (j = 0; j < 4; j++) {
				v[j] = j;
				}
			for (j = 0; j < 4; j++) {
				v[4 + j] = 4 + ((j + 1) % 4);
				}
			}
		A->make_element(gens_G->ith(i), v, 0 /* verbose_level */);
		}

	cout << "generators for G:" << endl;
	for (i = 0; i < nb_G; i++) {
		cout << "generator " << i << ":" << endl;
		A->element_print(gens_G->ith(i), cout);
		}

	for (i = 0; i < nb_S; i++) {
		if (i == 0) {
			for (j = 0; j < 4; j++) {
				v[j] = (j + 1) % 4;
				}
			for (j = 0; j < 4; j++) {
				v[4 + j] = 4 + j;
				}
			}
		else if (i == 1) {
			for (j = 0; j < 4; j++) {
				v[j] = (4 + j - 1) % 4;
				}
			for (j = 0; j < 4; j++) {
				v[4 + j] = 4 + j;
				}
			}
		else if (i == 2) {
			for (j = 0; j < 4; j++) {
				v[j] = j;
				}
			for (j = 0; j < 4; j++) {
				v[4 + j] = 4 + ((j + 1) % 4);
				}
			}
		else if (i == 3) {
			for (j = 0; j < 4; j++) {
				v[j] = j;
				}
			for (j = 0; j < 4; j++) {
				v[4 + j] = 4 + ((4 + j - 1) % 4);
				}
			}
		else if (i == 4) {
			for (j = 0; j < 4; j++) {
				v[j] = (j + 1) % 4;
				}
			for (j = 0; j < 4; j++) {
				v[4 + j] = 4 + ((j + 1) % 4);
				}
			}
		else if (i == 5) {
			for (j = 0; j < 4; j++) {
				v[j] = (4 + j - 1) % 4;
				}
			for (j = 0; j < 4; j++) {
				v[4 + j] = 4 + ((4 + j - 1) % 4);
				}
			}
		A->make_element(Elt_S[i], v, 0 /* verbose_level */);
		}

	cout << "generators for S:" << endl;
	for (i = 0; i < nb_S; i++) {
		cout << "generator " << i << ":" << endl;
		A->element_print(Elt_S[i], cout);
		}
	
	sims *G;


	G = create_sims_from_generators_with_target_group_order_INT(A, 
		gens_G, 16, verbose_level);

	G->group_order(go);

	goi = go.as_INT();

	cout << "created group of order " << goi << endl;

	if (goi != 16) {
		cout << "group order is wrong" << endl;
		exit(1);
		}

	INT *Adj;
	INT *Elt1, *Elt2;

	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Adj = NEW_INT(goi * goi);

	INT_vec_zero(Adj, goi * goi);

	cout << "Computing the Cayley graph:" << endl;
	for (i = 0; i < goi; i++) {
		G->element_unrank_INT(i, Elt1);
		//cout << "i=" << i << endl;
		for (h = 0; h < nb_S; h++) {
			A->element_mult(Elt1, Elt_S[h], Elt2, 0);
#if 0
			cout << "i=" << i << " h=" << h << endl;
			cout << "Elt1=" << endl;
			A->element_print_quick(Elt1, cout);
			cout << "g_h=" << endl;
			A->element_print_quick(gens->ith(h), cout);
			cout << "Elt2=" << endl;
			A->element_print_quick(Elt2, cout);
#endif
			j = G->element_rank_INT(Elt2);
			Adj[i * goi + j] = Adj[j * goi + i] = 1;
			if (i == 0) {
				cout << "edge " << i << " " << j << endl;
				}
			}
		}

	cout << "The adjacency matrix of a graph with " << goi << " vertices has been computed" << endl;
	//INT_matrix_print(Adj, goi, goi);


	colored_graph *CG;
	BYTE fname[1000];

	CG = new colored_graph;
	CG->init_adjacency_no_colors(goi, Adj, verbose_level);

	sprintf(fname, "Shrikhande.colored_graph");

	CG->save(fname, verbose_level);

	cout << "Written file " << fname << " of size " << file_size(fname) << endl;


	delete CG;
	FREE_INT(Adj);
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(v);
	delete gens_G;
	for (i = 0; i < nb_S; i++) {
		FREE_INT(Elt_S[i]);
		}
	FREE_PINT(Elt_S);
		
}

