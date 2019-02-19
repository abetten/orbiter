// cayley.C
//
// Anton Betten
// February 19, 2015
//

#include "orbiter.h"


using namespace orbiter;

int t0;

void do_D1(int n, int d, int verbose_level);



int main(int argc, char **argv)
{
	int i;
	int verbose_level = 0;
	int f_D1 = FALSE;
	int n = 0;
	int d = 0;
	
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

void do_D1(int n, int d, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h, m;
	int n_over_d;
	int phi_n, phi_n_over_d;
	int *Rn, *Rn_over_d;

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
	cout << "n=" << n << " m=" << m << " d=" << d
			<< " n_over_d=" << n_over_d << endl;
	cout << "phi_n = " << phi_n << endl;
	cout << "phi_n_over_d = " << phi_n_over_d << endl;

	Rn = NEW_int(phi_n);
	j = 0;
	for (i = 0; i < n; i++) {
		if (gcd_int(i, n) == 1) {
			Rn[j++] = i;
			}
		}
	if (j != phi_n) {
		cout << "j != phi_n" << endl;
		exit(1);
		}
	cout << "Rn=";
	int_vec_print(cout, Rn, phi_n);
	cout << endl;
	
	Rn_over_d = NEW_int(phi_n_over_d);
	j = 0;
	for (i = 0; i < n_over_d; i++) {
		if (gcd_int(i, n_over_d) == 1) {
			Rn_over_d[j++] = i;
			}
		}
	if (j != phi_n_over_d) {
		cout << "j != phi_n_over_d" << endl;
		exit(1);
		}
	cout << "Rn_over_d=";
	int_vec_print(cout, Rn_over_d, phi_n_over_d);
	cout << endl;

	action *A;
	longinteger_object go;
	int goi;

	A = NEW_OBJECT(action);
	A->init_symmetric_group(n, verbose_level);
	A->group_order(go);

	goi = go.as_int();
	cout << "Created group Sym(" << n << ") of size " << goi << endl;



	int nb_G;
	int *perms;
	vector_ge *gens_G;



	generators_dihedral_group(n, nb_G, perms, verbose_level);


	gens_G = NEW_OBJECT(vector_ge);
	gens_G->init(A);
	gens_G->allocate(nb_G);

	for (i = 0; i < nb_G; i++) {
		A->make_element(gens_G->ith(i),
				perms + i * n, 0 /* verbose_level */);
		}


	cout << "generators:" << endl;
	for (i = 0; i < nb_G; i++) {
		cout << "generator " << i << ":" << endl;
		A->element_print(gens_G->ith(i), cout);
		}


	sims *G;


	G = A->create_sims_from_generators_with_target_group_order_int(
		gens_G, 2 * n, verbose_level);

	G->group_order(go);

	goi = go.as_int();

	cout << "created group of order " << goi << endl;

	if (goi != 2 * n) {
		cout << "group order is wrong" << endl;
		exit(1);
		}

	int nb_S = 0;
	vector_ge *gens_S;
	int *Elt1;
	int *Elt2;
	int a;

	nb_S = 2 * (phi_n + phi_n_over_d);


	gens_S = NEW_OBJECT(vector_ge);
	gens_S->init(A);
	gens_S->allocate(nb_S);

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	j = 0;

	
	for (i = 0; i < phi_n; i++) {
		a = Rn[i];
		A->make_element(Elt1, perms + 0 * n, 0 /* verbose_level */);
		A->element_power_int_in_place(Elt1, a, 0 /* verbose_level */);
		A->element_move(Elt1, gens_S->ith(j), 0 /* verbose_level */);
		j++;
		}
	for (i = 0; i < phi_n_over_d; i++) {
		a = d * Rn_over_d[i];
		A->make_element(Elt1, perms + 0 * n, 0 /* verbose_level */);
		A->element_power_int_in_place(Elt1, a, 0 /* verbose_level */);
		A->element_move(Elt1, gens_S->ith(j), 0 /* verbose_level */);
		j++;
		}
	for (i = 0; i < phi_n; i++) {
		a = Rn[i];
		A->make_element(Elt1, perms + 0 * n, 0 /* verbose_level */);
		A->make_element(Elt2, perms + 1 * n, 0 /* verbose_level */);
		A->element_power_int_in_place(Elt1, a, 0 /* verbose_level */);
		A->element_mult(Elt2, Elt1, gens_S->ith(j), 0 /* verbose_level */);
		j++;
		}
	for (i = 0; i < phi_n_over_d; i++) {
		a = d * Rn_over_d[i];
		A->make_element(Elt1, perms + 0 * n, 0 /* verbose_level */);
		A->make_element(Elt2, perms + 1 * n, 0 /* verbose_level */);
		A->element_power_int_in_place(Elt1, a, 0 /* verbose_level */);
		A->element_mult(Elt2, Elt1, gens_S->ith(j), 0 /* verbose_level */);
		j++;
		}
	if (j != nb_S) {
		cout << "j != nb_S" << endl;
		exit(1);
		}


	int *Adj;

	Adj = NEW_int(goi * goi);

	int_vec_zero(Adj, goi * goi);

	cout << "Computing the Cayley graph:" << endl;
	for (i = 0; i < goi; i++) {
		G->element_unrank_int(i, Elt1);
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
			j = G->element_rank_int(Elt2);
			Adj[i * goi + j] = Adj[j * goi + i] = 1;
			if (i == 0) {
				cout << "edge " << i << " " << j << endl;
				}
			}
		}

	cout << "The adjacency matrix of a graph with " << goi
			<< " vertices has been computed" << endl;
	//int_matrix_print(Adj, goi, goi);


	{
	colored_graph *CG;
	char fname[1000];

	CG = NEW_OBJECT(colored_graph);
	CG->init_adjacency_no_colors(goi, Adj, verbose_level);

	sprintf(fname, "Cayley_D_%d_%d.colored_graph", n, d);

	CG->save(fname, verbose_level);

	cout << "Written file " << fname << " of size "
			<< file_size(fname) << endl;
	FREE_OBJECT(CG);
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
	char fname[1000];

	CG = NEW_OBJECT(colored_graph);
	CG->init_adjacency_no_colors(goi, Adj, verbose_level);

	sprintf(fname, "Cayley_D_%d_%d_complement.colored_graph", n, d);

	CG->save(fname, verbose_level);

	cout << "Written file " << fname << " of size "
			<< file_size(fname) << endl;
	FREE_OBJECT(CG);
	}

	FREE_int(Adj);
	FREE_int(Elt1);
	FREE_int(Elt2);

	FREE_OBJECT(G);
	FREE_OBJECT(gens_G);
	FREE_OBJECT(gens_S);
	FREE_OBJECT(A);
	FREE_int(perms);
	FREE_int(Rn);
	FREE_int(Rn_over_d);
}

