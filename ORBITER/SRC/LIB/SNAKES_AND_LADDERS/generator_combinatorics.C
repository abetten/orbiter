// generator_combinatorics.C
//
// Anton Betten
//
// moved here from generator.C
// July 18, 2014


#include "orbiter.h"

void generator::Plesken_matrix_up(INT depth, INT *&P, INT &N, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Nb;
	INT *Fst;
	INT *Pij;
	INT i, j;
	INT N1, N2;
	INT a, b, cnt;

	if (f_v) {
		cout << "generator::Plesken_matrix_up" << endl;
		}
	N = 0;
	Nb = NEW_INT(depth + 1);
	Fst = NEW_INT(depth + 2);
	Fst[0] = 0;
	for (i = 0; i <= depth; i++) {
		Nb[i] = nb_orbits_at_level(i);
		Fst[i + 1] = Fst[i] + Nb[i];
		N += Nb[i];
		}
	P = NEW_INT(N * N);
	for (i = 0; i <= depth; i++) {
		for (j = 0; j <= depth; j++) {
			Plesken_submatrix_up(i, j, Pij, N1, N2, verbose_level - 1);
			for (a = 0; a < N1; a++) {
				for (b = 0; b < N2; b++) {
					cnt = Pij[a * N2 + b];
					P[(Fst[i] + a) * N + Fst[j] + b] = cnt;
					}
				}
			FREE_INT(Pij);
			}
		}
	if (f_v) {
		cout << "generator::Plesken_matrix_up done" << endl;
		}
}

void generator::Plesken_matrix_down(INT depth, INT *&P, INT &N, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Nb;
	INT *Fst;
	INT *Pij;
	INT i, j;
	INT N1, N2;
	INT a, b, cnt;

	if (f_v) {
		cout << "generator::Plesken_matrix_down" << endl;
		}
	N = 0;
	Nb = NEW_INT(depth + 1);
	Fst = NEW_INT(depth + 2);
	Fst[0] = 0;
	for (i = 0; i <= depth; i++) {
		Nb[i] = nb_orbits_at_level(i);
		Fst[i + 1] = Fst[i] + Nb[i];
		N += Nb[i];
		}
	P = NEW_INT(N * N);
	for (i = 0; i <= depth; i++) {
		for (j = 0; j <= depth; j++) {
			Plesken_submatrix_down(i, j, Pij, N1, N2, verbose_level - 1);
			for (a = 0; a < N1; a++) {
				for (b = 0; b < N2; b++) {
					cnt = Pij[a * N2 + b];
					P[(Fst[i] + a) * N + Fst[j] + b] = cnt;
					}
				}
			FREE_INT(Pij);
			}
		}
	if (f_v) {
		cout << "generator::Plesken_matrix_down done" << endl;
		}
}

void generator::Plesken_submatrix_up(INT i, INT j, INT *&Pij, INT &N1, INT &N2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT a, b;

	if (f_v) {
		cout << "generator::Plesken_submatrix_up i=" << i << " j=" << j << endl;
		}
	N1 = nb_orbits_at_level(i);
	N2 = nb_orbits_at_level(j);
	Pij = NEW_INT(N1 * N2);
	for (a = 0; a < N1; a++) {
		for (b = 0; b < N2; b++) {
			Pij[a * N2 + b] = count_incidences_up(i, a, j, b, verbose_level - 1);
			}
		}
	if (f_v) {
		cout << "generator::Plesken_submatrix_up done" << endl;
		}
}

void generator::Plesken_submatrix_down(INT i, INT j, INT *&Pij, INT &N1, INT &N2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT a, b;

	if (f_v) {
		cout << "generator::Plesken_submatrix_down i=" << i << " j=" << j << endl;
		}
	N1 = nb_orbits_at_level(i);
	N2 = nb_orbits_at_level(j);
	Pij = NEW_INT(N1 * N2);
	for (a = 0; a < N1; a++) {
		for (b = 0; b < N2; b++) {
			Pij[a * N2 + b] = count_incidences_down(i, a, j, b, verbose_level - 1);
			}
		}
	if (f_v) {
		cout << "generator::Plesken_submatrix_down done" << endl;
		}
}

INT generator::count_incidences_up(INT lvl1, INT po1, INT lvl2, INT po2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *set;
	INT *set1;
	INT *set2;
	INT ol, i, cnt = 0;
	INT f_contained;

	if (f_v) {
		cout << "generator::count_incidences_up lvl1=" << lvl1 << " po1=" << po1 << " lvl2=" << lvl2 << " po2=" << po2 << endl;
		}
	if (lvl1 > lvl2) {
		return 0;
		}
	set = NEW_INT(lvl2 + 1);
	set1 = NEW_INT(lvl2 + 1);
	set2 = NEW_INT(lvl2 + 1);

	orbit_element_unrank(lvl1, po1, 0 /*el1 */, set1, 0 /* verbose_level */);

	ol = orbit_length_as_INT(po2, lvl2);

	if (f_vv) {
		cout << "set1=";
		INT_vec_print(cout, set1, lvl1);
		cout << endl;
		}

	for (i = 0; i < ol; i++) {

		INT_vec_copy(set1, set, lvl1);


		orbit_element_unrank(lvl2, po2, i, set2, 0 /* verbose_level */);

		if (f_vv) {
			cout << "set2 " << i << " / " << ol << "=";
			INT_vec_print(cout, set2, lvl2);
			cout << endl;
			}

		f_contained = poset_structure_is_contained(set, lvl1, set2, lvl2, verbose_level - 2);
		//f_contained = INT_vec_sort_and_test_if_contained(set, lvl1, set2, lvl2);
		
		if (f_vv) {
			cout << "f_contained=" << f_contained << endl;
			}
						

		if (f_contained) {
			cnt++;
			}
		}

	
	FREE_INT(set);
	FREE_INT(set1);
	FREE_INT(set2);
	if (f_v) {
		cout << "generator::count_incidences_up lvl1=" << lvl1 << " po1=" << po1 << " lvl2=" << lvl2 << " po2=" << po2 << " cnt=" << cnt << endl;
		}
	return cnt;
}

INT generator::count_incidences_down(INT lvl1, INT po1, INT lvl2, INT po2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *set;
	INT *set1;
	INT *set2;
	INT ol, i, cnt = 0;
	INT f_contained;

	if (f_v) {
		cout << "generator::count_incidences_down lvl1=" << lvl1 << " po1=" << po1 << " lvl2=" << lvl2 << " po2=" << po2 << endl;
		}
	if (lvl1 > lvl2) {
		return 0;
		}
	set = NEW_INT(lvl2 + 1);
	set1 = NEW_INT(lvl2 + 1);
	set2 = NEW_INT(lvl2 + 1);

	orbit_element_unrank(lvl2, po2, 0 /*el1 */, set2, 0 /* verbose_level */);

	ol = orbit_length_as_INT(po1, lvl1);

	if (f_vv) {
		cout << "set2=";
		INT_vec_print(cout, set2, lvl2);
		cout << endl;
		}

	for (i = 0; i < ol; i++) {

		INT_vec_copy(set2, set, lvl2);


		orbit_element_unrank(lvl1, po1, i, set1, 0 /* verbose_level */);

		if (f_vv) {
			cout << "set1 " << i << " / " << ol << "=";
			INT_vec_print(cout, set1, lvl1);
			cout << endl;
			}

		
		f_contained = poset_structure_is_contained(set1, lvl1, set, lvl2, verbose_level - 2);
		//f_contained = INT_vec_sort_and_test_if_contained(set1, lvl1, set, lvl2);
						
		if (f_vv) {
			cout << "f_contained=" << f_contained << endl;
			}

		if (f_contained) {
			cnt++;
			}
		}

	
	FREE_INT(set);
	FREE_INT(set1);
	FREE_INT(set2);
	if (f_v) {
		cout << "generator::count_incidences_down lvl1=" << lvl1 << " po1=" << po1 << " lvl2=" << lvl2 << " po2=" << po2 << " cnt=" << cnt << endl;
		}
	return cnt;
}

void generator::Asup_to_Ainf(INT t, INT k, INT *M_sup, INT *M_inf, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object quo, rem, aa, bb, cc;
	longinteger_object go;
	longinteger_object *go_t;
	longinteger_object *go_k;
	longinteger_object *ol_t;
	longinteger_object *ol_k;
	INT Nt, Nk;
	INT i, j, a, c;
	
	if (f_v) {
		cout << "generator::Asup_to_Ainf" << endl;
		}
	Nt = nb_orbits_at_level(t);
	Nk = nb_orbits_at_level(k);
	get_stabilizer_order(0, 0, go);
	if (f_v) {
		cout << "generator::Asup_to_Ainf go=" << go << endl;
		}
	go_t = new longinteger_object[Nt];
	go_k = new longinteger_object[Nk];
	ol_t = new longinteger_object[Nt];
	ol_k = new longinteger_object[Nk];
	if (f_v) {
		cout << "generator::Asup_to_Ainf computing orbit lengths t-orbits" << endl;
		}
	for (i = 0; i < Nt; i++) {
		get_stabilizer_order(t, i, go_t[i]);
		D.integral_division_exact(go, go_t[i], ol_t[i]);
		}
	if (f_v) {
		cout << "i : go_t[i] : ol_t[i]" << endl;
		for (i = 0; i < Nt; i++) {
			cout << i << " : " << go_t[i] << " : " << ol_t[i] << endl;
			}
		}
	if (f_v) {
		cout << "generator::Asup_to_Ainf computing orbit lengths k-orbits" << endl;
		}
	for (i = 0; i < Nk; i++) {
		get_stabilizer_order(k, i, go_k[i]);
		D.integral_division_exact(go, go_k[i], ol_k[i]);
		}
	if (f_v) {
		cout << "i : go_k[i] : ol_k[i]" << endl;
		for (i = 0; i < Nk; i++) {
			cout << i << " : " << go_k[i] << " : " << ol_k[i] << endl;
			}
		}
	if (f_v) {
		cout << "generator::Asup_to_Ainf computing Ainf" << endl;
		}
	for (i = 0; i < Nt; i++) {
		for (j = 0; j < Nk; j++) {
			a = M_sup[i * Nk + j];
			aa.create(a);
			D.mult(ol_t[i], aa, bb);
			D.integral_division(bb, ol_k[j], cc, rem, 0);
			if (!rem.is_zero()) {
				cout << "generator::Asup_to_Ainf stabilizer order does not divide group order" << endl;
				cout << "i=" << i << " j=" << j << " M_sup[i,j] = " << a << " ol_t[i]=" << ol_t[i] << " ol_k[j]=" << ol_k[j] << endl;
				exit(1);
				}
			c = cc.as_INT();
			M_inf[i * Nk + j] = c;
			}
		}
	if (f_v) {
		cout << "generator::Asup_to_Ainf computing Ainf done" << endl;
		}
	delete [] go_t;
	delete [] go_k;
	delete [] ol_t;
	delete [] ol_k;
	if (f_v) {
		cout << "generator::Asup_to_Ainf done" << endl;
		}
}

void generator::test_for_multi_edge_in_classification_graph(INT depth, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, f, l, j, h1;

	if (f_v) {
		cout << "generator::test_for_multi_edge_in_classification_graph depth=" << depth << endl;
		}
	for (i = 0; i <= depth; i++) {
		f = first_oracle_node_at_level[i];
		l = nb_orbits_at_level(i);
		if (f_v) {
			cout << "generator::test_for_multi_edge_in_classification_graph level=" << i << " with " << l << " nodes" << endl;
			}
		for (j = 0; j < l; j++) {
			oracle *O;

			O = &root[f + j];
			for (h1 = 0; h1 < O->nb_extensions; h1++) {
				extension *E1 = O->E + h1;

				if (E1->type != EXTENSION_TYPE_FUSION) {
					continue;
					}

				//cout << "fusion (" << f + j << "/" << h1 << ") -> (" << E1->data1 << "/" << E1->data2 << ")" << endl;
				if (E1->data1 == f + j) {
					cout << "multiedge detected ! level " << i << " with " << l << " nodes, fusion (" << j << "/" << h1 << ") -> (" << E1->data1 - f << "/" << E1->data2 << ")" << endl;
					}

#if 0
				for (h2 = 0; h2 < O->nb_extensions; h2++) {
					extension *E2 = O->E + h2;

					if (E2->type != EXTENSION_TYPE_FUSION) {
						continue;

					if (E2->data1 == E1->data1 && E2->data2 == E1->data2) {
						cout << "multiedge detected!" << endl;
						cout << "fusion (" << f + j << "/" << h1 << ") -> (" << E1->data1 << "/" << E1->data2 << ")" << endl;
						cout << "fusion (" << f + j << "/" << h2 << ") -> (" << E2->data1 << "/" << E2->data2 << ")" << endl;
						}
					}
#endif

				}
			}
		if (f_v) {
			cout << "generator::test_for_multi_edge_in_classification_graph level=" << i << " with " << l << " nodes done" << endl;
			}
		}
	if (f_v) {
		cout << "generator::test_for_multi_edge_in_classification_graph done" << endl;
		}
}




