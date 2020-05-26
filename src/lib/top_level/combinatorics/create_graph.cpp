/*
 * create_graph.cpp
 *
 *  Created on: Nov 28, 2019
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


static int evaluate_cubic_form(finite_field *F, int *v);

create_graph::create_graph()
{
	description = NULL;

	f_has_CG = FALSE;
	CG = NULL;

	N = 0;
	Adj = NULL;

	//char label[1000];
	//char label_tex[1000];

}

create_graph::~create_graph()
{
	if (f_has_CG) {
		FREE_OBJECT(CG);
	}
}

void create_graph::init(
		create_graph_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::init" << endl;
	}
	create_graph::description = description;

	f_has_CG = FALSE;

	if (description->f_load_from_file) {

		f_has_CG = TRUE;
		CG = NEW_OBJECT(colored_graph);
		CG->load(description->fname, verbose_level);
		sprintf(label, "%s", description->fname);
		sprintf(label_tex, "File\\_%s", description->fname);
	}
	else if (description->f_edge_list) {

		combinatorics_domain Combi;
		int h, i, j, a;

		int *Idx;
		int sz;

		int_vec_scan(description->edge_list_text, Idx, sz);

		N = description->n;


		Adj = NEW_int(N * N);
		int_vec_zero(Adj, N * N);
		for (h = 0; h < sz; h++) {
			a = Idx[h];
			Combi.k2ij(a, i, j, N);
			Adj[i * N + j] = 1;
			Adj[j * N + i] = 1;
		}
		FREE_int(Idx);
		sprintf(label, "graph_v%d_e%d", description->n, sz);
		sprintf(label_tex, "Graph\\_%d\\_%d", description->n, sz);
		}
	else if (description->f_edges_as_pairs) {
		int h, i, j;
		int *Idx;
		int sz, sz2;

		int_vec_scan(description->edges_as_pairs_text, Idx, sz);

		N = description->n;


		Adj = NEW_int(N * N);
		int_vec_zero(Adj, N * N);
		sz2 = sz >> 1;
		for (h = 0; h < sz2; h++) {
			i = Idx[2 * h + 0];
			j = Idx[2 * h + 1];
			Adj[i * N + j] = 1;
			Adj[j * N + i] = 1;
		}
		FREE_int(Idx);
		sprintf(label, "graph_v%d_e%d", description->n, sz);
		sprintf(label_tex, "Graph\\_%d\\_%d", description->n, sz);
		}
	else if (description->f_Johnson) {
		if (f_v) {
			cout << "create_graph::init before create_Johnson" << endl;
		}
		create_Johnson(N, Adj, description->Johnson_n,
				description->Johnson_k, description->Johnson_s,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init after create_Johnson" << endl;
		}
	}
	else if (description->f_Paley) {
		if (f_v) {
			cout << "create_graph::init before create_Paley" << endl;
		}
		create_Paley(N, Adj, description->Paley_q,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init after create_Paley" << endl;
		}
	}
	else if (description->f_Sarnak) {
		if (f_v) {
			cout << "create_graph::init before create_Sarnak" << endl;
		}
		create_Sarnak(N, Adj, description->Sarnak_p, description->Sarnak_q,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init after create_Sarnak" << endl;
		}
	}
	else if (description->f_Schlaefli) {
		if (f_v) {
			cout << "create_graph::init before create_Schlaefli" << endl;
		}
		create_Schlaefli(N, Adj, description->Schlaefli_q,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init after create_Schlaefli" << endl;
		}
	}
	else if (description->f_Shrikhande) {
		if (f_v) {
			cout << "create_graph::init before create_Shrikhande" << endl;
		}
		create_Shrikhande(N, Adj, verbose_level);
		if (f_v) {
			cout << "create_graph::init after create_Shrikhande" << endl;
		}
	}
	else if (description->f_Winnie_Li) {
		if (f_v) {
			cout << "create_graph::init before create_Winnie_Li" << endl;
		}
		create_Winnie_Li(N, Adj, description->Winnie_Li_q, description->Winnie_Li_index, verbose_level);
		if (f_v) {
			cout << "create_graph::init after create_Winnie_Li" << endl;
		}
	}
	else if (description->f_Grassmann) {
		if (f_v) {
			cout << "create_graph::init before create_Grassmann" << endl;
		}
		create_Grassmann(N, Adj, description->Grassmann_n, description->Grassmann_k,
				description->Grassmann_q, description->Grassmann_r, verbose_level);
		if (f_v) {
			cout << "create_graph::init after create_Grassmann" << endl;
		}
	}
	else if (description->f_coll_orthogonal) {
		if (f_v) {
			cout << "create_graph::init before create_coll_orthogonal" << endl;
		}
		create_coll_orthogonal(N, Adj, description->coll_orthogonal_epsilon,
				description->coll_orthogonal_d,
				description->coll_orthogonal_q, verbose_level);
		if (f_v) {
			cout << "create_graph::init after create_coll_orthogonal" << endl;
		}
	}

	if (f_v) {
		cout << "create_graph::init done" << endl;
	}
}


void create_graph::create_Johnson(int &N, int *&Adj,
		int n, int k, int s, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_Johnson" << endl;
	}

	combinatorics_domain Combi;
	sorting Sorting;
	int *set1;
	int *set2;
	int *set3;
	int i, j, sz;

	N = Combi.int_n_choose_k(n, k);


	Adj = NEW_int(N * N);
	int_vec_zero(Adj, N * N);

	set1 = NEW_int(k);
	set2 = NEW_int(k);
	set3 = NEW_int(k);

	for (i = 0; i < N; i++) {
		Combi.unrank_k_subset(i, set1, n, k);
		for (j = i + 1; j < N; j++) {
			Combi.unrank_k_subset(j, set2, n, k);

			Sorting.int_vec_intersect_sorted_vectors(set1, k, set2, k, set3, sz);
			if (sz == s) {
				Adj[i * N + j] = 1;
				Adj[j * N + i] = 1;
				}
			}
		}

	sprintf(label, "Johnson_%d_%d_%d", n, k, s);
	sprintf(label_tex, "Johnson\\_%d\\_%d\\_%d", n, k, s);

	FREE_int(set1);
	FREE_int(set2);
	FREE_int(set3);

	if (f_v) {
		cout << "create_graph::create_Johnson done" << endl;
	}
}

void create_graph::create_Paley(int &N, int *&Adj,
		int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_Paley" << endl;
	}


	if (EVEN(q)) {
		cout << "create_graph::create_Paley q must be odd" << endl;
		exit(1);
	}
	if (!DOUBLYEVEN(q - 1)) {
		cout << "create_graph::create_Paley q must be congruent to 1 modulo 4" << endl;
	}

	finite_field *F;
	int *f_is_square;
	int i, j, a;

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
				Adj[j * q + i] = 1;
			}
		}
	}
	N = q;

	sprintf(label, "Paley_%d", q);
	sprintf(label_tex, "Paley\\_%d", q);

	FREE_OBJECT(F);
	FREE_int(f_is_square);

	if (f_v) {
		cout << "create_graph::create_Paley done" << endl;
	}
}

void create_graph::create_Sarnak(int &N, int *&Adj,
		int p, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_Sarnak" << endl;
	}


	int f_vv = (verbose_level >= 2);
	int i, j, h, l, f_special = FALSE;
	number_theory_domain NT;



	l = NT.Legendre(p, q, 0);
	if (f_v) {
		cout << "Legendre(" << p << ", " << q << ")=" << l << endl;
	}


	finite_field *F;
	action *A;
	int f_semilinear = FALSE;
	int f_basis = TRUE;

	F = NEW_OBJECT(finite_field);
	F->init(q, 0);
	//F->init_override_polynomial(q, override_poly, verbose_level);

	A = NEW_OBJECT(action);

	if (l == 1) {
		f_special = TRUE;

		if (f_v) {
			cout << "Creating projective special linear group:" << endl;
		}
		A->init_projective_special_group(2, F,
			f_semilinear,
			f_basis,
			verbose_level - 2);
		}
	else {
		vector_ge *nice_gens;

		if (f_v) {
			cout << "Creating projective linear group:" << endl;
		}
		A->init_projective_group(2, F,
			f_semilinear,
			f_basis, TRUE /* f_init_sims */,
			nice_gens,
			verbose_level - 2);
		FREE_OBJECT(nice_gens);
		}



	sims *Sims;

	Sims = A->Sims;


	longinteger_object go;
	int goi;
	Sims->group_order(go);

	if (f_v) {
		cout << "found a group of order " << go << endl;
	}
	goi = go.as_int();




	int a0, a1, a2, a3;
	int sqrt_p;

	int *sqrt_mod_q;
	int I;
	int *A4;
	int nb_A4 = 0;

	A4 = NEW_int((p + 1) * 4);
	sqrt_mod_q = NEW_int(q);
	for (i = 0; i < q; i++) {
		sqrt_mod_q[i] = -1;
		}
	for (i = 0; i < q; i++) {
		j = F->mult(i, i);
		sqrt_mod_q[j] = i;
		}
	if (f_v) {
		cout << "sqrt_mod_q:" << endl;
		int_vec_print(cout, sqrt_mod_q, q);
		cout << endl;
	}

	sqrt_p = 0;
	for (i = 1; i < p; i++) {
		if (i * i > p) {
			sqrt_p = i - 1;
			break;
			}
		}
	if (f_v) {
		cout << "p=" << p << endl;
		cout << "sqrt_p = " << sqrt_p << endl;
	}


	for (I = 0; I < q; I++) {
		if (F->add(F->mult(I, I), 1) == 0) {
			break;
			}
		}
	if (I == q) {
		cout << "did not find I" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "I=" << I << endl;
	}

	for (a0 = 1; a0 <= sqrt_p; a0++) {
		if (EVEN(a0)) {
			continue;
			}
		for (a1 = -sqrt_p; a1 <= sqrt_p; a1++) {
			if (ODD(a1)) {
				continue;
				}
			for (a2 = -sqrt_p; a2 <= sqrt_p; a2++) {
				if (ODD(a2)) {
					continue;
					}
				for (a3 = -sqrt_p; a3 <= sqrt_p; a3++) {
					if (ODD(a3)) {
						continue;
						}
					if (a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 == p) {
						if (f_v) {
							cout << "solution " << nb_A4 << " : " << a0
									<< ", " << a1 << ", " << a2 << ", "
									<< a3 << ", " << endl;
						}
						if (nb_A4 == p + 1) {
							cout << "too many solutions" << endl;
							exit(1);
							}
						A4[nb_A4 * 4 + 0] = a0;
						A4[nb_A4 * 4 + 1] = a1;
						A4[nb_A4 * 4 + 2] = a2;
						A4[nb_A4 * 4 + 3] = a3;
						nb_A4++;
						}
					}
				}
			}
		}

	if (f_v) {
		cout << "nb_A4=" << nb_A4 << endl;
	}
	if (nb_A4 != p + 1) {
		cout << "nb_A4 != p + 1" << endl;
		exit(1);
		}

	int_matrix_print(A4, nb_A4, 4);

	vector_ge *gens;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int M4[4];
	int det; //, s, sv;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

	gens = NEW_OBJECT(vector_ge);
	gens->init(A, verbose_level - 2);
	gens->allocate(nb_A4, verbose_level - 2);

	if (f_v) {
		cout << "making connection set:" << endl;
	}
	for (i = 0; i < nb_A4; i++) {

		if (f_vv) {
			cout << "making generator " << i << ":" << endl;
			}
		a0 = A4[i * 4 + 0];
		a1 = A4[i * 4 + 1];
		a2 = A4[i * 4 + 2];
		a3 = A4[i * 4 + 3];
		while (a0 < 0) {
			a0 += q;
			}
		while (a1 < 0) {
			a1 += q;
			}
		while (a2 < 0) {
			a2 += q;
			}
		while (a3 < 0) {
			a3 += q;
			}
		a0 = a0 % q;
		a1 = a1 % q;
		a2 = a2 % q;
		a3 = a3 % q;
		if (f_vv) {
			cout << "making generator " << i << ": a0=" << a0
					<< " a1=" << a1 << " a2=" << a2
					<< " a3=" << a3 << endl;
			}
		M4[0] = F->add(a0, F->mult(I, a1));
		M4[1] = F->add(a2, F->mult(I, a3));
		M4[2] = F->add(F->negate(a2), F->mult(I, a3));
		M4[3] = F->add(a0, F->negate(F->mult(I, a1)));

		if (f_vv) {
			cout << "M4=";
			int_vec_print(cout, M4, 4);
			cout << endl;
			}

		if (f_special) {
			det = F->add(F->mult(M4[0], M4[3]),
					F->negate(F->mult(M4[1], M4[2])));

			if (f_vv) {
				cout << "det=" << det << endl;
				}

#if 0
			s = sqrt_mod_q[det];
			if (s == -1) {
				cout << "determinant is not a square" << endl;
				exit(1);
				}
			sv = F->inverse(s);
			if (f_vv) {
				cout << "det=" << det << " sqrt=" << s
						<< " mutiplying by " << sv << endl;
				}
			for (j = 0; j < 4; j++) {
				M4[j] = F->mult(sv, M4[j]);
				}
			if (f_vv) {
				cout << "M4=";
				int_vec_print(cout, M4, 4);
				cout << endl;
				}
#endif
			}

		A->make_element(Elt1, M4, verbose_level - 1);

		if (f_v) {
			cout << "s_" << i << "=" << endl;
			A->element_print_quick(Elt1, cout);
			}

		A->element_move(Elt1, gens->ith(i), 0);
		}


	Adj = NEW_int(goi * goi);

	int_vec_zero(Adj, goi * goi);

	if (f_v) {
		cout << "Computing the Cayley graph:" << endl;
	}
	for (i = 0; i < goi; i++) {
		Sims->element_unrank_lint(i, Elt1);
		//cout << "i=" << i << endl;
		for (h = 0; h < nb_A4; h++) {
			A->element_mult(Elt1, gens->ith(h), Elt2, 0);
#if 0
			cout << "i=" << i << " h=" << h << endl;
			cout << "Elt1=" << endl;
			A->element_print_quick(Elt1, cout);
			cout << "g_h=" << endl;
			A->element_print_quick(gens->ith(h), cout);
			cout << "Elt2=" << endl;
			A->element_print_quick(Elt2, cout);
#endif
			j = Sims->element_rank_lint(Elt2);
			Adj[i * goi + j] = Adj[j * goi + i] = 1;
			if (i == 0) {
				cout << "edge " << i << " " << j << endl;
				}
			}
		}

	if (f_v) {
		cout << "create_graph::create_Sarnak The adjacency matrix of a graph with " << goi
				<< " vertices has been computed" << endl;
		//int_matrix_print(Adj, goi, goi);
	}

	int k;
	k = 0;
	for (i = 0; i < goi; i++) {
		if (Adj[0 * goi + i]) {
			k++;
			}
		}
	if (f_v) {
		cout << "k=" << k << endl;
	}


	N = goi;


	sprintf(label, "Sarnak_%d_%d", p, q);
	sprintf(label_tex, "Sarnak\\_%d\\_%d", p, q);

	FREE_OBJECT(gens);
	FREE_OBJECT(A);
	FREE_int(A4);
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_OBJECT(F);

	if (f_v) {
		cout << "create_graph::create_Sarnak done" << endl;
	}
}


void create_graph::create_Schlaefli(int &N, int *&Adj,
		int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_Schlaefli" << endl;
	}

	int i, j, rr, sz;
	finite_field *F;
	grassmann *Gr;
	int *M1;
	int *M2;
	int *M;
	int v[2];
	int w[4];
	int *List;
	int n = 4;
	int k = 2;
	combinatorics_domain Combi;



	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	Gr = NEW_OBJECT(grassmann);
	Gr->init(n, k, F, verbose_level);

	M1 = NEW_int(k * n);
	M2 = NEW_int(k * n);
	M = NEW_int(2 * k * n);

	N = Combi.generalized_binomial(n, k, q);

	List = NEW_int(N);
	sz = 0;

	for (i = 0; i < N; i++) {
		Gr->unrank_lint_here(M1, i, 0 /* verbose_level */);

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
	N = sz;

	for (i = 0; i < sz; i++) {
		Gr->unrank_lint_here(M1, List[i], 0 /* verbose_level */);

		for (j = i + 1; j < sz; j++) {
			Gr->unrank_lint_here(M2, List[j], 0 /* verbose_level */);

			int_vec_copy(M1, M, k * n);
			int_vec_copy(M2, M + k * n, k * n);

			rr = F->rank_of_rectangular_matrix(M, 2 * k, n, 0 /* verbose_level */);
			if (rr == 2 * k) {
				Adj[i * sz + j] = 1;
				Adj[j * sz + i] = 1;
				}
			}
		}



	sprintf(label, "Schlaefli_%d", q);
	sprintf(label_tex, "Schlaefli\\_%d", q);

	FREE_int(List);
	FREE_int(M1);
	FREE_int(M2);
	FREE_int(M);
	FREE_OBJECT(Gr);
	FREE_OBJECT(F);

	if (f_v) {
		cout << "create_graph::create_Schlaefli done" << endl;
	}
}

void create_graph::create_Shrikhande(int &N, int *&Adj, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_Shrikhande" << endl;
	}

	action *A;
	longinteger_object go;
	int goi;
	int **Elt_S;
	vector_ge *gens_G;
	int *v;
	int n = 8;
	int i, j, h;
	int nb_G, nb_S;

	A = NEW_OBJECT(action);
	A->init_symmetric_group(n, verbose_level);
	A->group_order(go);

	goi = go.as_int();
	cout << "Created group Sym(" << n << ") of size " << goi << endl;


	nb_G = 2;
	nb_S = 6;
	gens_G = NEW_OBJECT(vector_ge);
	gens_G->init(A, verbose_level - 2);
	gens_G->allocate(nb_G, verbose_level - 2);


	Elt_S = NEW_pint(nb_S);
	for (i = 0; i < nb_S; i++) {
		Elt_S[i] = NEW_int(A->elt_size_in_int);
		}
	v = NEW_int(n);


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


	G = A->create_sims_from_generators_with_target_group_order_int(
		gens_G, 16, verbose_level);

	G->group_order(go);

	goi = go.as_int();

	cout << "created group of order " << goi << endl;

	if (goi != 16) {
		cout << "group order is wrong" << endl;
		exit(1);
		}

	int *Elt1, *Elt2;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Adj = NEW_int(goi * goi);

	int_vec_zero(Adj, goi * goi);

	cout << "Computing the Cayley graph:" << endl;
	for (i = 0; i < goi; i++) {
		G->element_unrank_lint(i, Elt1);
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
			j = G->element_rank_lint(Elt2);
			Adj[i * goi + j] = Adj[j * goi + i] = 1;
			if (i == 0) {
				cout << "edge " << i << " " << j << endl;
				}
			}
		}

	cout << "The adjacency matrix of a graph with " << goi << " vertices has been computed" << endl;
	//int_matrix_print(Adj, goi, goi);

	N = goi;

	sprintf(label, "Shrikhande");
	sprintf(label_tex, "Shrikhande");

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(v);
	FREE_OBJECT(gens_G);
	for (i = 0; i < nb_S; i++) {
		FREE_int(Elt_S[i]);
		}
	FREE_pint(Elt_S);

	if (f_v) {
		cout << "create_graph::create_Shrikhande done" << endl;
	}
}

void create_graph::create_Winnie_Li(int &N, int *&Adj,
		int q, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_Winnie_Li" << endl;
	}

	finite_field *F;
	int i, j, h, u, p, k, co_index, q1, relative_norm;
	int *N1;
	number_theory_domain NT;


	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level - 1);
	p = F->p;

#if 0
	if (!f_index) {
		index = F->e;
		}
#endif

	co_index = F->e / index;

	if (co_index * index != F->e) {
		cout << "the index has to divide the field degree" << endl;
		exit(1);
		}
	q1 = NT.i_power_j(p, co_index);

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

	N1 = NEW_int(k);
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
	int_vec_print(cout, N1, k);
	cout << endl;

	Adj = NEW_int(q * q);
	for (i = 0; i < q; i++) {
		for (h = 0; h < k; h++) {
			j = N1[h];
			u = F->add(i, j);
			Adj[i * q + u] = 1;
			Adj[u * q + i] = 1;
			}
		}

	N = q;



	sprintf(label, "Winnie_Li_%d_%d", q, index);
	sprintf(label_tex, "Winnie_Li\\_%d\\_%d", q, index);

	FREE_int(N1);
	FREE_OBJECT(F);


	if (f_v) {
		cout << "create_graph::create_Winnie_Li done" << endl;
	}
}

void create_graph::create_Grassmann(int &N, int *&Adj,
		int n, int k, int q, int r, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_Grassmann" << endl;
	}


	finite_field *F;
	grassmann *Gr;
	int i, j, rr;
	int *M1; // [k * n]
	int *M2; // [k * n]
	int *M; // [2 * k * n]
	combinatorics_domain Combi;

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);


	Gr = NEW_OBJECT(grassmann);
	Gr->init(n, k, F, verbose_level);

	N = Combi.generalized_binomial(n, k, q);

	M1 = NEW_int(k * n);
	M2 = NEW_int(k * n);
	M = NEW_int(2 * k * n);

	Adj = NEW_int(N * N);
	int_vec_zero(Adj, N * N);

	for (i = 0; i < N; i++) {

		Gr->unrank_lint_here(M1, i, 0 /* verbose_level */);

		for (j = i + 1; j < N; j++) {

			Gr->unrank_lint_here(M2, j, 0 /* verbose_level */);

			int_vec_copy(M1, M, k * n);
			int_vec_copy(M2, M + k * n, k * n);

			rr = F->rank_of_rectangular_matrix(M, 2 * k, n, 0 /* verbose_level */);
			if (rr == r) {
				Adj[i * N + j] = 1;
				Adj[j * N + i] = 1;
				}
			}
		}


	sprintf(label, "Grassmann_%d_%d_%d_%d", n, k, q, r);
	sprintf(label_tex, "Grassmann\\_%d\\_%d\\_%d\\_%d", n, k, q, r);


	FREE_int(M1);
	FREE_int(M2);
	FREE_int(M);
	FREE_OBJECT(Gr);
	FREE_OBJECT(F);

	if (f_v) {
		cout << "create_graph::create_Grassmann done" << endl;
	}
}

void create_graph::create_coll_orthogonal(int &N, int *&Adj,
		int epsilon, int d, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_coll_orthogonal" << endl;
	}

	finite_field *F;
	int i, j;
	int n, a, nb_e, nb_inc;
	int c1 = 0, c2 = 0, c3 = 0;
	int *v, *v2;
	int *Gram; // Gram matrix
	geometry_global Gg;


	n = d - 1; // projective dimension

	v = NEW_int(d);
	v2 = NEW_int(d);
	Gram = NEW_int(d * d);

	cout << "epsilon=" << epsilon << " n=" << n << " q=" << q << endl;

	N = Gg.nb_pts_Qepsilon(epsilon, n, q);

	cout << "number of points = " << N << endl;

	F = NEW_OBJECT(finite_field);

	F->init(q, verbose_level - 1);
	F->print();

	if (epsilon == 0) {
		c1 = 1;
		}
	else if (epsilon == -1) {
		F->choose_anisotropic_form(c1, c2, c3, verbose_level - 2);
		//cout << "incma.cpp: epsilon == -1, need irreducible polynomial" << endl;
		//exit(1);
		}
	F->Gram_matrix(epsilon, n, c1, c2, c3, Gram);
	cout << "Gram matrix" << endl;
	print_integer_matrix_width(cout, Gram, d, d, d, 2);

#if 0
	if (f_list_points) {
		for (i = 0; i < N; i++) {
			F->Q_epsilon_unrank(v, 1, epsilon, n, c1, c2, c3, i, 0 /* verbose_level */);
			cout << i << " : ";
			int_vec_print(cout, v, n + 1);
			j = F->Q_epsilon_rank(v, 1, epsilon, n, c1, c2, c3, 0 /* verbose_level */);
			cout << " : " << j << endl;

			}
		}
#endif


	cout << "allocating adjacency matrix" << endl;
	Adj = NEW_int(N * N);
	cout << "allocating adjacency matrix was successful" << endl;
	nb_e = 0;
	nb_inc = 0;
	for (i = 0; i < N; i++) {
		//cout << i << " : ";
		F->Q_epsilon_unrank(v, 1, epsilon, n, c1, c2, c3, i, 0 /* verbose_level */);
		for (j = i + 1; j < N; j++) {
			F->Q_epsilon_unrank(v2, 1, epsilon, n, c1, c2, c3, j, 0 /* verbose_level */);
			a = F->evaluate_bilinear_form(v, v2, n + 1, Gram);
			if (a == 0) {
				//cout << j << " ";
				//k = ij2k(i, j, N);
				//cout << k << ", ";
				nb_e++;
				//if ((nb_e % 50) == 0)
					//cout << endl;
				Adj[i * N + j] = 1;
				Adj[j * N + i] = 1;
				}
			else {
				Adj[i * N + j] = 0;
				Adj[j * N + i] = 0;
				; //cout << " 0";
				nb_inc++;
				}
			}
		//cout << endl;
		Adj[i * N + i] = 0;
		}
	//cout << endl;
	cout << "The adjacency matrix of the collinearity graph has been computed" << endl;


	sprintf(label, "Coll_orthogonal_%d_%d_%d", epsilon, d, q);
	sprintf(label_tex, "Coll_orthogonal\\_%d\\_%d\\_%d", epsilon, d, q);

	FREE_int(v);
	FREE_int(v2);
	FREE_int(Gram);
	FREE_OBJECT(F);

	if (f_v) {
		cout << "create_graph::create_coll_orthogonal done" << endl;
	}
}

static int evaluate_cubic_form(finite_field *F, int *v)
{
	int a, i;

	a = 0;
	for (i = 0; i < 4; i++) {
		a = F->add(a, F->power(v[i], 3));
		}
	return a;
}





}}
