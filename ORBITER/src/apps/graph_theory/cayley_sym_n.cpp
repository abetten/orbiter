// cayley_sym_n.cpp
//
// Anton Betten
// February 16, 2015
//

#include "orbiter.h"

using namespace std;



using namespace orbiter;

int t0;

void do_it(int n, int f_special, int f_coxeter,
		int f_pancake, int f_burnt_pancake, int verbose_level);
void get_submatrices(int n, int *Adj,
		int N, int N0, int **&P, int verbose_level);
void mult_matrix(int *P, int *Q, int *R, int N);



int main(int argc, char **argv)
{
	int i;
	int verbose_level = 0;
	int f_n = FALSE;
	int n = 0;
	int f_star = FALSE;
	int f_coxeter = FALSE;
	int f_pancake = FALSE;
	int f_burnt_pancake = FALSE;
	
	t0 = os_ticks();

	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-star") == 0) {
			f_star = TRUE;
			cout << "-star" << endl;
			}
		else if (strcmp(argv[i], "-coxeter") == 0) {
			f_coxeter = TRUE;
			cout << "-coxeter" << endl;
			}
		else if (strcmp(argv[i], "-pancake") == 0) {
			f_pancake = TRUE;
			cout << "-pancake" << endl;
			}
		else if (strcmp(argv[i], "-burnt_pancake") == 0) {
			f_burnt_pancake = TRUE;
			cout << "-burnt_pancake" << endl;
			}
		}
	


	if (!f_n) {
		cout << "please specify -n <n>" << endl;
		exit(1);
		}

	do_it(n, f_star, f_coxeter,
			f_pancake, f_burnt_pancake, verbose_level);
	
}

void do_it(int n, int f_star, int f_coxeter,
		int f_pancake, int f_burnt_pancake, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "do_it" << endl;
		}

	action *A;
	longinteger_object go;
	int goi;
	int *v;
	int i, j;
	int nb_gens = 0;
	int deg = 0;
	vector_ge *gens;
	char fname_base[1000];
	graph_theory_domain Graph;


	if (f_star) {
		sprintf(fname_base, "Cayley_Sym_%d_star", n);
		}
	else if (f_coxeter) {
		sprintf(fname_base, "Cayley_Sym_%d_coxeter", n);
		}
	else if (f_pancake) {
		sprintf(fname_base, "Cayley_pancake_%d", n);
		}
	else if (f_burnt_pancake) {
		sprintf(fname_base, "Cayley_burnt_pancake_%d", n);
		}
	else {
		cout << "please specify the type of Cayley graph" << endl;
		exit(1);
		}

	cout << "fname=" << fname_base << endl;


	if (f_star) {
		nb_gens = n - 1;
		deg = n;
		}
	else if (f_coxeter) {
		nb_gens = n - 1;
		deg = n;
		}
	else if (f_pancake) {
		nb_gens = n - 1;
		deg = n;
		}
	else if (f_burnt_pancake) {
		nb_gens = n - 1;
		deg = 2 * n;
		}


	A = NEW_OBJECT(action);
	A->init_symmetric_group(deg, 0 /*verbose_level*/);
	A->group_order(go);

	goi = go.as_int();
	cout << "Created group Sym(" << deg << ") of size " << goi << endl;

	strong_generators *SG;
	group_generators_domain GG;


	if (f_burnt_pancake) {
		int *factors;
		int nb_factors;
		int deg1;
		int nb_perms;
		int *perms;
		vector_ge *G_gens;
		longinteger_object target_go;
		longinteger_domain D;

		
		G_gens = NEW_OBJECT(vector_ge);
		
		GG.order_Bn_group_factorized(n, factors, nb_factors);

		D.multiply_up(target_go, factors, nb_factors, 0 /* verbose_level */);
		cout << "target group order = " << target_go << endl;

		GG.generators_Bn_group(n, deg1, nb_perms, perms, verbose_level);
		
		G_gens->init(A, verbose_level - 2);
		G_gens->allocate(nb_perms, verbose_level - 2);

		for (i = 0; i < nb_perms; i++) {
			A->make_element(G_gens->ith(i),
					perms + i * deg1, 0 /* verbose_level */);
			}

		A->generators_to_strong_generators(
			TRUE /* f_target_go */, target_go, 
			G_gens, SG, verbose_level - 3);
		
		FREE_OBJECT(G_gens);
		FREE_int(perms);
		FREE_int(factors);
		}
	else {
		SG = A->Strong_gens;
		}

	sims *G;
	longinteger_object G_go;
	combinatorics_domain Combi;

	cout << "creating group G:" << endl;
	G = SG->create_sims(0 /* verbose_level */);
	G->group_order(G_go);
	cout << "created group G of order " << G_go << endl;
	goi = G_go.as_int();
	

	gens = NEW_OBJECT(vector_ge);

	gens->init(A, verbose_level - 2);
	gens->allocate(nb_gens, verbose_level - 2);

	v = NEW_int(deg);


	if (f_star) {
		for (i = 0; i < nb_gens; i++) {
			Combi.perm_identity(v, deg);
			v[0] = i + 1;
			v[i + 1] = 0;
			A->make_element(gens->ith(i), v, 0 /* verbose_level */);
			}
		}
	else if (f_coxeter) {
		for (i = 0; i < nb_gens; i++) {
			Combi.perm_identity(v, deg);
			v[i] = i + 1;
			v[i + 1] = i;
			A->make_element(gens->ith(i), v, 0 /* verbose_level */);
			}
		}
	else if (f_pancake) {
		for (i = 0; i < nb_gens; i++) {
			Combi.perm_identity(v, deg);
			for (j = 0; j <= 1 + i; j++) {
				v[j] = 1 + i - j;
				}
			A->make_element(gens->ith(i), v, 0 /* verbose_level */);
			}
		}
	else if (f_burnt_pancake) {
		for (i = 0; i < nb_gens; i++) {
			Combi.perm_identity(v, deg);
			for (j = 0; j <= 1 + i; j++) {
				v[2 * j + 0] = 2 * (1 + i - j) + 1;
				v[2 * j + 1] = 2 * (1 + i - j) + 0;
				}
			A->make_element(gens->ith(i), v, 0 /* verbose_level */);
			}
		}

	cout << "generators:" << endl;
	for (i = 0; i < nb_gens; i++) {
		cout << "generator " << i << ":" << endl;
		A->element_print(gens->ith(i), cout);
		}

	
#if 0
	sims *Sims;

	Sims = A->Sims;	
#endif

	
	int *Adj;
	int goi1;

	G->create_Cayley_graph(gens, Adj, goi1, verbose_level);



	cout << "The adjacency matrix of a graph with " << goi
			<< " vertices has been computed" << endl;
	//int_matrix_print(Adj, goi, goi);


	Graph.save_as_colored_graph_easy(fname_base, goi,
			Adj, 0 /* verbose_level */);


	return;


	int N0;
	int **P;
	//int N2;
	int h;

	N0 = goi / n;
	get_submatrices(n, Adj, goi, N0, P, verbose_level);

	for (h = 0; h < n - 1; h++) {
		cout << "P_" << h << "=" << endl;
		int_matrix_print(P[h], N0, N0);
		}


	for (h = 0; h < n - 1; h++) {
		FREE_int(P[h]);
		}
	FREE_pint(P);

	//delete CG;
	FREE_int(Adj);
	FREE_OBJECT(gens);
	FREE_int(v);
}


void get_submatrices(int n, int *Adj,
		int N, int N0, int **&P, int verbose_level)
{
	
	//int *Q;
	int h, i, j;

	P = NEW_pint(n - 1);
	for (h = 0; h < n - 1; h++) {
		P[h] = NEW_int(N0 * N0);
		for (i = 0; i < N0; i++) {
			for (j = 0; j < N0; j++) {
				P[h][i * N0 + j] = Adj[(1 + h) * N * N0 + i * N + j];
				}
			}
		}
}

#if 0
	Q = NEW_int(N0 * N0);
	
	N2 = N - N0;
	Adj2 = NEW_int(N2 * N2);
	for (u = 0; u < n - 1; u++) {
		for (v = 0; v < n - 1; v++) {
			mult_matrix(P[u], P[v], Q, N0);
			cout << "P_" << u << " * P_" << v << " = " << endl;
			int_matrix_print(Q, N0, N0);
			for (i = 0; i < N0; i++) {
				for (j = 0; j < N0; j++) {
					Adj2[u * N0 * N2 + i * N2 + v * N0 + j] = Q[i * N0 + j]; 
					}
				}
			}
		}

	cout << "Adj2=" << endl;
	int_matrix_print(Adj2, N2, N2);

	FREE_int(Q);
}
#endif

void mult_matrix(int *P, int *Q, int *R, int N)
{
	int i, j, h, a;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			a = 0;
			for (h = 0; h < N; h++) {
				a += P[i * N + h] * Q[h * N + j];
				}
			R[i * N + j] = a;
			}
		}
}

