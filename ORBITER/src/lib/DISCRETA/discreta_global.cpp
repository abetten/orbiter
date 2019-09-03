// discreta_global.cpp
//
// Anton Betten
// Nov 19, 2007

#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace discreta {

void free_global_data()
{
	cout << "discreta_global free_global_data freeing global data" << endl;
	orthogonal_points_free_global_data();
	longinteger_free_global_data();
	longinteger_domain_free_tab_q_binomials();
}

void the_end(int t0)
{
	file_io Fio;

	cout << "***************** The End **********************" << endl;
	cout << "nb_calls_to_finite_field_init="
			<< nb_calls_to_finite_field_init << endl;
	free_global_data();
	if (f_memory_debug) {
		//registry_dump();
		//registry_dump_sorted();
		}
	time_check(cout, t0);
	cout << endl;


	int mem_usage;
	char fname[1000];

	mem_usage = os_memory_usage();
	sprintf(fname, "memory_usage.csv");
	Fio.int_matrix_write_csv(fname, &mem_usage, 1, 1);
}

void the_end_quietly(int t0)
{
	//cout << "discreta_global the_end_quietly: freeing global data" << endl;
	free_global_data();
	time_check(cout, t0);
	cout << endl;
}

void calc_Kramer_Mesner_matrix_neighboring(
	poset_classification *gen,
	int level, matrix &M, int verbose_level)
// we assume that we don't use implicit fusion nodes
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;//(verbose_level >= 2);
	int f1, f2, l1, l2, i, j, k, I, J, len;
	poset_orbit_node *O;
	
	if (f_v) {
		cout << "calc_Kramer_Mesner_matrix_neighboring "
				"level=" << level << endl;
		}

	f1 = gen->first_poset_orbit_node_at_level[level];
	f2 = gen->first_poset_orbit_node_at_level[level + 1];
	l1 = gen->nb_orbits_at_level(level); //f2 - f1;
	l2 = gen->nb_orbits_at_level(level + 1); //f3 - f2;

	M.m_mn_n(l1, l2);

	if (f_v) {
		cout << "calc_Kramer_Mesner_matrix_neighboring "
				"the size of the matrix is " << l1 << " x " << l2 << endl;
		}

	for (i = 0; i < l1; i++) {
		if (f_vv) {
			cout << "calc_Kramer_Mesner_matrix_neighboring "
					"i=" << i << " / " << l1 << endl;
			}
		I = f1 + i;
		O = &gen->root[I];
		for (k = 0; k < O->nb_extensions; k++) {
			if (f_vv) {
				cout << "calc_Kramer_Mesner_matrix_neighboring "
						"i=" << i << " / " << l1 << " extension "
						<< k << " / " << O->nb_extensions << endl;
				}
			if (O->E[k].type == EXTENSION_TYPE_EXTENSION) {
				if (f_vv) {
					cout << "calc_Kramer_Mesner_matrix_neighboring "
							"i=" << i << " / " << l1 << " extension "
							<< k << " / " << O->nb_extensions
							<< " type extension node" << endl;
					}
				len = O->E[k].orbit_len;
				J = O->E[k].data;
				j = J - f2;
				M.s_iji(i, j) += len;
				}
			if (O->E[k].type == EXTENSION_TYPE_FUSION) {
				if (f_vv) {
					cout << "calc_Kramer_Mesner_matrix_neighboring "
							"i=" << i << " / " << l1 << " extension "
							<< k << " / " << O->nb_extensions
							<< " type fusion" << endl;
					}
				// fusion node
				len = O->E[k].orbit_len;

				int I1, ext1;
				poset_orbit_node *O1;
				
				I1 = O->E[k].data1;
				ext1 = O->E[k].data2;
				O1 = &gen->root[I1];
				if (O1->E[ext1].type != EXTENSION_TYPE_EXTENSION) {
					cout << "calc_Kramer_Mesner_matrix_neighboring "
							"O1->E[ext1].type != EXTENSION_TYPE_EXTENSION "
							"something is wrong" << endl;
					exit(1);
					}
				J = O1->E[ext1].data;

#if 0
				O->store_set(gen, level - 1);
					// stores a set of size level to gen->S
				gen->S[level] = O->E[k].pt;

				for (ii = 0; ii < level + 1; ii++) {
					gen->set[level + 1][ii] = gen->S[ii];
					}
				
				gen->A->element_one(gen->transporter->ith(level + 1), 0);
				
				J = O->apply_isomorphism(gen, 
					level, I /* current_node */, 
					//0 /* my_node */, 0 /* my_extension */, 0 /* my_coset */, 
					k /* current_extension */, level + 1, 
					FALSE /* f_tolerant */, 
					0/*verbose_level - 2*/);
				if (FALSE) {
					cout << "after apply_isomorphism J=" << J << endl;
					}
#else
				
#endif




#if 0
				//cout << "fusion node:" << endl;
				//int_vec_print(cout, gen->S, level + 1);
				//cout << endl;
				gen->A->element_retrieve(O->E[k].data, gen->Elt1, 0);
	
				gen->A2->map_a_set(gen->S, gen->S0, level + 1, gen->Elt1, 0);
				//int_vec_print(cout, gen->S0, level + 1);
				//cout << endl;

				int_vec_heapsort(gen->S0, level + 1); //int_vec_sort(level + 1, gen->S0);

				//int_vec_print(cout, gen->S0, level + 1);
				//cout << endl;

				J = gen->find_poset_orbit_node_for_set(level + 1, gen->S0, 0);
#endif
				j = J - f2;
				M.s_iji(i, j) += len;
				}
			}
		}
	if (f_v) {
		cout << "calc_Kramer_Mesner_matrix_neighboring "
				"level=" << level << " done" << endl;
		}
}



void Mtk_from_MM(Vector & MM, matrix & Mtk, int t, int k, 
	int f_subspaces, int q,  int verbose_level)
{
	int f_v = (verbose_level >= 1);
	matrix M, N;
	int i;
	
	if (f_v) {
		cout << "Mtk_from_MM t = " << t << ", k = " << k << endl;
		}
	if (k == t) {
		matrix &M1 = MM[t - 1].as_matrix();
		int n = M1.s_n();
		Mtk.m_mn_n(n, n);
		Mtk.one();
		return;
		}
	M = MM[t].as_matrix();
	for (i = t + 2; i <= k; i++) {
		Mtk_via_Mtr_Mrk(t, i - 1, i, f_subspaces, q,  
			M, MM[i - 1].as_matrix(), N, verbose_level - 1);
		N.swap(M);
		}
	M.swap(Mtk);
	if (f_v) {
		cout << "Mtk_from_MM t = " << t
				<< ", k = " << k << " done" << endl;
		}
}

void Mtk_via_Mtr_Mrk(int t, int r, int k, int f_subspaces, int q,  
	matrix & Mtr, matrix & Mrk, matrix & Mtk, int verbose_level)
// Computes $M_{tk}$ via a recursion formula:
// $M_{tk} = {{k - t} \choose {k - r}} \cdot M_{t,r} \cdot M_{r,k}$.
{
	int f_v = (verbose_level >= 1);
	discreta_base s, h;
	int i, j, m, n;
	
	if (f_v) {
		cout << "Mtk_via_Mtr_Mrk(): t = " << t << ", r = "
				<< r << ", k = " << k << endl;
		}
	Mtk.mult(Mtr, Mrk);
		/* Mtk := (k - t) atop (k - r) * M_t,k */
	if (f_subspaces) {
		longinteger_domain D;
		longinteger_object a;
		
		D.q_binomial(a, k - t, r - t, q, 0/* verbose_level*/);
		s.m_i_i(a.as_int());
		}
	else {
		Binomial(k - t, k - r, s);
		}
	m = Mtk.s_m();
	n = Mtk.s_n();
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			Mtk[i][j].integral_division_exact(s, h);
			Mtk[i][j] = h;
			}
		}
	if (f_v) {
		cout << "Mtk_via_Mtr_Mrk matrix M_{" << t << "," << k 
			<< "} of format " << m << " x " << n << " computed" << endl;
		}
}

void Mtk_sup_to_inf(poset_classification *gen,
	int t, int k, matrix & Mtk_sup, matrix & Mtk_inf, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Nt, Nk, i, j, a;
	int *M_sup;
	int *M_inf;	
	
	if (f_v) {
		cout << "Mtk_sup_to_inf" << endl;
		}
	Nt = Mtk_sup.s_m();
	Nk = Mtk_sup.s_n();
	if (f_v) {
		cout << "M_{" << t << "," << k << "} sup is a matrix of size "
				<< Nt << " x " << Nk << endl;
		cout << Mtk_sup << endl;
		}
	M_sup = NEW_int(Nt * Nk);
	M_inf = NEW_int(Nt * Nk);
	for (i = 0; i < Nt; i++) {
		for (j = 0; j < Nk; j++) {
			M_sup[i * Nk + j] = Mtk_sup.s_iji(i, j);
			}
		}
	if (f_v) {
		cout << "Mtk_sup_to_inf before generator::Asup_to_Ainf" << endl;
		}
	
	gen->Asup_to_Ainf(t, k, M_sup, M_inf, verbose_level);

	if (f_v) {
		cout << "Mtk_sup_to_inf after generator::Asup_to_Ainf" << endl;
		}

	Mtk_inf.m_mn_n(Nt, Nk);
	for (i = 0; i < Nt; i++) {
		for (j = 0; j < Nk; j++) {
			a = M_inf[i * Nk + j];
			Mtk_inf.m_iji(i, j, a);
			}
		}
	if (f_v) {
		cout << "Mtk_sup_to_inf" << endl;
		cout << "M_{" << t << "," << k << "} inf is a matrix of size "
				<< Nt << " x " << Nk << endl;
		cout << Mtk_inf << endl;
		}
	
}

void compute_Kramer_Mesner_matrix(poset_classification *gen,
	int t, int k, matrix &Mtk, int f_subspaces, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	// compute Kramer Mesner matrices
	Vector V;
	int i; //, a;
	

	if (f_v) {
		cout << "compute_Kramer_Mesner_matrix t=" << t << " k=" << k << endl;
		}
	V.m_l(k);
	for (i = 0; i < k; i++) {
		V[i].change_to_matrix();
		calc_Kramer_Mesner_matrix_neighboring(gen, i,
			V[i].as_matrix(), verbose_level - 2);


		cout << "matrix at level " << i << " has been computed" << endl;
		
		if (FALSE && f_v) {
			cout << "matrix level " << i << ":" << endl;
			V[i].as_matrix().print(cout);
			}
		}
	
	
	Mtk_from_MM(V, Mtk, t, k, f_subspaces, q, verbose_level - 2);
	cout << "M_{" << t << "," << k << "} sup has been computed" << endl;
	//Mtk.print(cout);

	if (f_v) {
		int m = Mtk.s_m();
		int n = Mtk.s_n();
		cout << "compute_Kramer_Mesner_matrix t=" << t
				<< " k=" << k << " done" << endl;
		cout << "compute_Kramer_Mesner_matrix the matrix has size "
				<< m << " x " << n << endl;
		}
}

void matrix_to_diophant(matrix& M, diophant *&D, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_to_diophant" << endl;
		}
	int nb_rows;
	int nb_cols;
	int i, j;

	D = NEW_OBJECT(diophant);

	nb_rows = M.s_m();
	nb_cols = M.s_n();
	D->open(nb_rows, nb_cols);
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			D->Aij(i, j) = M.s_iji(i, j);
			}
		}
	for (i = 0; i < nb_rows; i++) {
		D->RHSi(i) = 1;
		}
	
	if (f_v) {
		cout << "matrix_to_diophant done" << endl;
		}
}

}}

