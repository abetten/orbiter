// cayley.C
//
// Anton Betten
// February 19, 2015
//

#include "orbiter.h"


INT t0;

void do_D1(INT n, INT d, INT verbose_level);



int main(int argc, char **argv)
{
	INT i;
	INT verbose_level = 0;
	INT f_D1 = FALSE;
	INT n = 0;
	INT d = 0;
	
	t0 = os_ticks();

	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-D1") == 0) {
			f_D1 = TRUE;
			n = atoi(argv[++i]);
			d = atoi(argv[++i]);
			cout << "-special" << endl;
			}
		}
	



	if (f_D1) {
		do_D1(n, d, verbose_level);
		}
	
}

void do_D1(INT n, INT d, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, h, m;
	INT n_over_d;
	INT phi_n, phi_n_over_d;
	INT *Rn, *Rn_over_d;

	if (f_v) {
		cout << "do_D1" << endl;
		}

	if (EVEN(n)) {
		cout << "need n to be odd" << endl;
		exit(1);
		}

	m = (n - 1) >> 1;

	n_over_d = n / d;
	phi_n = euler_function(n);
	phi_n_over_d = euler_function(n_over_d);
	cout << "n=" << n << " m=" << m << " d=" << d << " n_over_d=" << n_over_d << endl;
	cout << "phi_n = " << phi_n << endl;
	cout << "phi_n_over_d = " << phi_n_over_d << endl;

	Rn = NEW_INT(phi_n);
	j = 0;
	for (i = 0; i < n; i++) {
		if (gcd_INT(i, n) == 1) {
			Rn[j++] = i;
			}
		}
	if (j != phi_n) {
		cout << "j != phi_n" << endl;
		exit(1);
		}
	cout << "Rn=";
	INT_vec_print(cout, Rn, phi_n);
	cout << endl;
	
	Rn_over_d = NEW_INT(phi_n_over_d);
	j = 0;
	for (i = 0; i < n_over_d; i++) {
		if (gcd_INT(i, n_over_d) == 1) {
			Rn_over_d[j++] = i;
			}
		}
	if (j != phi_n_over_d) {
		cout << "j != phi_n_over_d" << endl;
		exit(1);
		}
	cout << "Rn_over_d=";
	INT_vec_print(cout, Rn_over_d, phi_n_over_d);
	cout << endl;

	action *A;
	longinteger_object go;
	INT goi;

	A = new action;
	A->init_symmetric_group(n, verbose_level);
	A->group_order(go);

	goi = go.as_INT();
	cout << "Created group Sym(" << n << ") of size " << goi << endl;



	INT nb_G;
	INT *perms;
	vector_ge *gens_G;



	generators_dihedral_group(n, nb_G, perms, verbose_level);


	gens_G = new vector_ge;
	gens_G->init(A);
	gens_G->allocate(nb_G);

	for (i = 0; i < nb_G; i++) {
		A->make_element(gens_G->ith(i), perms + i * n, 0 /* verbose_level */);
		}


	cout << "generators:" << endl;
	for (i = 0; i < nb_G; i++) {
		cout << "generator " << i << ":" << endl;
		A->element_print(gens_G->ith(i), cout);
		}


	sims *G;


	G = create_sims_from_generators_with_target_group_order_INT(A, 
		gens_G, 2 * n, verbose_level);

	G->group_order(go);

	goi = go.as_INT();

	cout << "created group of order " << goi << endl;

	if (goi != 2 * n) {
		cout << "group order is wrong" << endl;
		exit(1);
		}

	INT nb_S = 0;
	vector_ge *gens_S;
	INT *Elt1;
	INT *Elt2;
	INT a;

	nb_S = 2 * (phi_n + phi_n_over_d);


	gens_S = new vector_ge;
	gens_S->init(A);
	gens_S->allocate(nb_S);

	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	j = 0;

	
	for (i = 0; i < phi_n; i++) {
		a = Rn[i];
		A->make_element(Elt1, perms + 0 * n, 0 /* verbose_level */);
		A->element_power_INT_in_place(Elt1, a, 0 /* verbose_level */);
		A->element_move(Elt1, gens_S->ith(j), 0 /* verbose_level */);
		j++;
		}
	for (i = 0; i < phi_n_over_d; i++) {
		a = d * Rn_over_d[i];
		A->make_element(Elt1, perms + 0 * n, 0 /* verbose_level */);
		A->element_power_INT_in_place(Elt1, a, 0 /* verbose_level */);
		A->element_move(Elt1, gens_S->ith(j), 0 /* verbose_level */);
		j++;
		}
	for (i = 0; i < phi_n; i++) {
		a = Rn[i];
		A->make_element(Elt1, perms + 0 * n, 0 /* verbose_level */);
		A->make_element(Elt2, perms + 1 * n, 0 /* verbose_level */);
		A->element_power_INT_in_place(Elt1, a, 0 /* verbose_level */);
		A->element_mult(Elt2, Elt1, gens_S->ith(j), 0 /* verbose_level */);
		j++;
		}
	for (i = 0; i < phi_n_over_d; i++) {
		a = d * Rn_over_d[i];
		A->make_element(Elt1, perms + 0 * n, 0 /* verbose_level */);
		A->make_element(Elt2, perms + 1 * n, 0 /* verbose_level */);
		A->element_power_INT_in_place(Elt1, a, 0 /* verbose_level */);
		A->element_mult(Elt2, Elt1, gens_S->ith(j), 0 /* verbose_level */);
		j++;
		}
	if (j != nb_S) {
		cout << "j != nb_S" << endl;
		exit(1);
		}


	INT *Adj;

	Adj = NEW_INT(goi * goi);

	INT_vec_zero(Adj, goi * goi);

	cout << "Computing the Cayley graph:" << endl;
	for (i = 0; i < goi; i++) {
		G->element_unrank_INT(i, Elt1);
		//cout << "i=" << i << endl;
		for (h = 0; h < nb_S; h++) {
			A->element_mult(Elt1, gens_S->ith(h), Elt2, 0);
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


	{
	colored_graph *CG;
	BYTE fname[1000];

	CG = new colored_graph;
	CG->init_adjacency_no_colors(goi, Adj, verbose_level);

	sprintf(fname, "Cayley_D_%ld_%ld.colored_graph", n, d);

	CG->save(fname, verbose_level);

	cout << "Written file " << fname << " of size " << file_size(fname) << endl;
	delete CG;
	}


	for (i = 0; i < goi; i++) {
		for (j = i + 1; j < goi; j++) {
			if (Adj[i * goi + j]) {
				Adj[i * goi + j] = 0;
				Adj[j * goi + i] = 0;
				}
			else {
				Adj[i * goi + j] = 1;
				Adj[j * goi + i] = 1;
				}
			}
		}
	{
	colored_graph *CG;
	BYTE fname[1000];

	CG = new colored_graph;
	CG->init_adjacency_no_colors(goi, Adj, verbose_level);

	sprintf(fname, "Cayley_D_%ld_%ld_complement.colored_graph", n, d);

	CG->save(fname, verbose_level);

	cout << "Written file " << fname << " of size " << file_size(fname) << endl;
	delete CG;
	}

	FREE_INT(Adj);
	FREE_INT(Elt1);
	FREE_INT(Elt2);

	delete G;
	delete gens_G;
	delete gens_S;
	delete A;
	FREE_INT(perms);
	FREE_INT(Rn);
	FREE_INT(Rn_over_d);
}

