// poset_classification_combinatorics.cpp
//
// Anton Betten
//
// moved here from poset_classification.cpp
// July 18, 2014


#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

using namespace std;

namespace orbiter {
namespace classification {

void poset_classification::Plesken_matrix_up(int depth,
		int *&P, int &N, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Nb;
	int *Fst;
	int *Pij;
	int i, j;
	int N1, N2;
	int a, b, cnt;

	if (f_v) {
		cout << "poset_classification::Plesken_matrix_up" << endl;
		}
	N = 0;
	Nb = NEW_int(depth + 1);
	Fst = NEW_int(depth + 2);
	Fst[0] = 0;
	for (i = 0; i <= depth; i++) {
		Nb[i] = nb_orbits_at_level(i);
		Fst[i + 1] = Fst[i] + Nb[i];
		N += Nb[i];
		}
	P = NEW_int(N * N);
	for (i = 0; i <= depth; i++) {
		for (j = 0; j <= depth; j++) {
			Plesken_submatrix_up(i, j, Pij, N1, N2, verbose_level - 1);
			for (a = 0; a < N1; a++) {
				for (b = 0; b < N2; b++) {
					cnt = Pij[a * N2 + b];
					P[(Fst[i] + a) * N + Fst[j] + b] = cnt;
					}
				}
			FREE_int(Pij);
			}
		}
	if (f_v) {
		cout << "poset_classification::Plesken_matrix_up done" << endl;
		}
}

void poset_classification::Plesken_matrix_down(int depth,
		int *&P, int &N, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Nb;
	int *Fst;
	int *Pij;
	int i, j;
	int N1, N2;
	int a, b, cnt;

	if (f_v) {
		cout << "poset_classification::Plesken_matrix_down" << endl;
		}
	N = 0;
	Nb = NEW_int(depth + 1);
	Fst = NEW_int(depth + 2);
	Fst[0] = 0;
	for (i = 0; i <= depth; i++) {
		Nb[i] = nb_orbits_at_level(i);
		Fst[i + 1] = Fst[i] + Nb[i];
		N += Nb[i];
		}
	P = NEW_int(N * N);
	for (i = 0; i <= depth; i++) {
		for (j = 0; j <= depth; j++) {
			Plesken_submatrix_down(i, j,
					Pij, N1, N2, verbose_level - 1);
			for (a = 0; a < N1; a++) {
				for (b = 0; b < N2; b++) {
					cnt = Pij[a * N2 + b];
					P[(Fst[i] + a) * N + Fst[j] + b] = cnt;
					}
				}
			FREE_int(Pij);
			}
		}
	if (f_v) {
		cout << "poset_classification::Plesken_matrix_down done" << endl;
		}
}

void poset_classification::Plesken_submatrix_up(int i, int j,
		int *&Pij, int &N1, int &N2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b;

	if (f_v) {
		cout << "poset_classification::Plesken_submatrix_up "
				"i=" << i << " j=" << j << endl;
		}
	N1 = nb_orbits_at_level(i);
	N2 = nb_orbits_at_level(j);
	Pij = NEW_int(N1 * N2);
	for (a = 0; a < N1; a++) {
		for (b = 0; b < N2; b++) {
			Pij[a * N2 + b] = count_incidences_up(
					i, a, j, b, verbose_level - 1);
			}
		}
	if (f_v) {
		cout << "poset_classification::Plesken_submatrix_up done" << endl;
		}
}

void poset_classification::Plesken_submatrix_down(int i, int j,
		int *&Pij, int &N1, int &N2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b;

	if (f_v) {
		cout << "poset_classification::Plesken_submatrix_down "
				"i=" << i << " j=" << j << endl;
		}
	N1 = nb_orbits_at_level(i);
	N2 = nb_orbits_at_level(j);
	Pij = NEW_int(N1 * N2);
	for (a = 0; a < N1; a++) {
		for (b = 0; b < N2; b++) {
			Pij[a * N2 + b] = count_incidences_down(
					i, a, j, b, verbose_level - 1);
			}
		}
	if (f_v) {
		cout << "poset_classification::Plesken_submatrix_down done" << endl;
		}
}

int poset_classification::count_incidences_up(int lvl1, int po1,
		int lvl2, int po2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *set;
	int *set1;
	int *set2;
	int ol, i, cnt = 0;
	int f_contained;

	if (f_v) {
		cout << "poset_classification::count_incidences_up "
				"lvl1=" << lvl1 << " po1=" << po1
				<< " lvl2=" << lvl2 << " po2=" << po2 << endl;
		}
	if (lvl1 > lvl2) {
		return 0;
		}
	set = NEW_int(lvl2 + 1);
	set1 = NEW_int(lvl2 + 1);
	set2 = NEW_int(lvl2 + 1);

	orbit_element_unrank(lvl1, po1, 0 /*el1 */,
			set1, 0 /* verbose_level */);

	ol = orbit_length_as_int(po2, lvl2);

	if (f_vv) {
		cout << "set1=";
		int_vec_print(cout, set1, lvl1);
		cout << endl;
		}

	for (i = 0; i < ol; i++) {

		int_vec_copy(set1, set, lvl1);


		orbit_element_unrank(lvl2, po2, i, set2, 0 /* verbose_level */);

		if (f_vv) {
			cout << "set2 " << i << " / " << ol << "=";
			int_vec_print(cout, set2, lvl2);
			cout << endl;
			}

		f_contained = poset_structure_is_contained(
				set, lvl1, set2, lvl2, verbose_level - 2);
		//f_contained = int_vec_sort_and_test_if_contained(
		// set, lvl1, set2, lvl2);
		
		if (f_vv) {
			cout << "f_contained=" << f_contained << endl;
			}
						

		if (f_contained) {
			cnt++;
			}
		}

	
	FREE_int(set);
	FREE_int(set1);
	FREE_int(set2);
	if (f_v) {
		cout << "poset_classification::count_incidences_up "
				"lvl1=" << lvl1 << " po1=" << po1
				<< " lvl2=" << lvl2 << " po2=" << po2
				<< " cnt=" << cnt << endl;
		}
	return cnt;
}

int poset_classification::count_incidences_down(
		int lvl1, int po1, int lvl2, int po2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *set;
	int *set1;
	int *set2;
	int ol, i, cnt = 0;
	int f_contained;

	if (f_v) {
		cout << "poset_classification::count_incidences_down "
				"lvl1=" << lvl1 << " po1=" << po1
				<< " lvl2=" << lvl2 << " po2=" << po2 << endl;
		}
	if (lvl1 > lvl2) {
		return 0;
		}
	set = NEW_int(lvl2 + 1);
	set1 = NEW_int(lvl2 + 1);
	set2 = NEW_int(lvl2 + 1);

	orbit_element_unrank(lvl2, po2, 0 /*el1 */, set2, 0 /* verbose_level */);

	ol = orbit_length_as_int(po1, lvl1);

	if (f_vv) {
		cout << "set2=";
		int_vec_print(cout, set2, lvl2);
		cout << endl;
		}

	for (i = 0; i < ol; i++) {

		int_vec_copy(set2, set, lvl2);


		orbit_element_unrank(lvl1, po1, i, set1, 0 /* verbose_level */);

		if (f_vv) {
			cout << "set1 " << i << " / " << ol << "=";
			int_vec_print(cout, set1, lvl1);
			cout << endl;
			}

		
		f_contained = poset_structure_is_contained(
				set1, lvl1, set, lvl2, verbose_level - 2);
		//f_contained = int_vec_sort_and_test_if_contained(
		// set1, lvl1, set, lvl2);
						
		if (f_vv) {
			cout << "f_contained=" << f_contained << endl;
			}

		if (f_contained) {
			cnt++;
			}
		}

	
	FREE_int(set);
	FREE_int(set1);
	FREE_int(set2);
	if (f_v) {
		cout << "poset_classification::count_incidences_down "
				"lvl1=" << lvl1 << " po1=" << po1
				<< " lvl2=" << lvl2 << " po2=" << po2
				<< " cnt=" << cnt << endl;
		}
	return cnt;
}

void poset_classification::Asup_to_Ainf(int t, int k,
		int *M_sup, int *M_inf, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object quo, rem, aa, bb, cc;
	longinteger_object go;
	longinteger_object *go_t;
	longinteger_object *go_k;
	longinteger_object *ol_t;
	longinteger_object *ol_k;
	int Nt, Nk;
	int i, j, a, c;
	
	if (f_v) {
		cout << "poset_classification::Asup_to_Ainf" << endl;
		}
	Nt = nb_orbits_at_level(t);
	Nk = nb_orbits_at_level(k);
	get_stabilizer_order(0, 0, go);
	if (f_v) {
		cout << "poset_classification::Asup_to_Ainf go=" << go << endl;
		}
	go_t = NEW_OBJECTS(longinteger_object, Nt);
	go_k = NEW_OBJECTS(longinteger_object, Nk);
	ol_t = NEW_OBJECTS(longinteger_object, Nt);
	ol_k = NEW_OBJECTS(longinteger_object, Nk);
	if (f_v) {
		cout << "poset_classification::Asup_to_Ainf "
				"computing orbit lengths t-orbits" << endl;
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
		cout << "poset_classification::Asup_to_Ainf "
				"computing orbit lengths k-orbits" << endl;
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
		cout << "poset_classification::Asup_to_Ainf computing Ainf" << endl;
		}
	for (i = 0; i < Nt; i++) {
		for (j = 0; j < Nk; j++) {
			a = M_sup[i * Nk + j];
			aa.create(a);
			D.mult(ol_t[i], aa, bb);
			D.integral_division(bb, ol_k[j], cc, rem, 0);
			if (!rem.is_zero()) {
				cout << "poset_classification::Asup_to_Ainf "
						"stabilizer order does not "
						"divide group order" << endl;
				cout << "i=" << i << " j=" << j
						<< " M_sup[i,j] = " << a
						<< " ol_t[i]=" << ol_t[i]
						<< " ol_k[j]=" << ol_k[j] << endl;
				exit(1);
				}
			c = cc.as_int();
			M_inf[i * Nk + j] = c;
			}
		}
	if (f_v) {
		cout << "poset_classification::Asup_to_Ainf computing Ainf done" << endl;
		}
	FREE_OBJECTS(go_t);
	FREE_OBJECTS(go_k);
	FREE_OBJECTS(ol_t);
	FREE_OBJECTS(ol_k);
	if (f_v) {
		cout << "poset_classification::Asup_to_Ainf done" << endl;
		}
}

void poset_classification::test_for_multi_edge_in_classification_graph(
		int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, f, l, j, h1;

	if (f_v) {
		cout << "poset_classification::test_for_multi_edge_in_classification_graph "
				"depth=" << depth << endl;
		}
	for (i = 0; i <= depth; i++) {
		f = first_poset_orbit_node_at_level[i];
		l = nb_orbits_at_level(i);
		if (f_v) {
			cout << "poset_classification::test_for_multi_edge_in_classification_graph "
					"level=" << i << " with " << l << " nodes" << endl;
			}
		for (j = 0; j < l; j++) {
			poset_orbit_node *O;

			O = &root[f + j];
			for (h1 = 0; h1 < O->nb_extensions; h1++) {
				extension *E1 = O->E + h1;

				if (E1->type != EXTENSION_TYPE_FUSION) {
					continue;
					}

				//cout << "fusion (" << f + j << "/" << h1 << ") ->
				// (" << E1->data1 << "/" << E1->data2 << ")" << endl;
				if (E1->data1 == f + j) {
					cout << "multiedge detected ! level "
							<< i << " with " << l << " nodes, fusion ("
							<< j << "/" << h1 << ") -> ("
							<< E1->data1 - f << "/"
							<< E1->data2 << ")" << endl;
					}

#if 0
				for (h2 = 0; h2 < O->nb_extensions; h2++) {
					extension *E2 = O->E + h2;

					if (E2->type != EXTENSION_TYPE_FUSION) {
						continue;

					if (E2->data1 == E1->data1 && E2->data2 == E1->data2) {
						cout << "multiedge detected!" << endl;
						cout << "fusion (" << f + j << "/" << h1
								<< ") -> (" << E1->data1 << "/"
								<< E1->data2 << ")" << endl;
						cout << "fusion (" << f + j << "/" << h2
								<< ") -> (" << E2->data1 << "/"
								<< E2->data2 << ")" << endl;
						}
					}
#endif

				}
			}
		if (f_v) {
			cout << "poset_classification::test_for_multi_edge_in_classification_graph "
					"level=" << i << " with " << l << " nodes done" << endl;
			}
		}
	if (f_v) {
		cout << "poset_classification::test_for_multi_edge_in_classification_graph "
				"done" << endl;
		}
}


}}



