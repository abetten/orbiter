/*
 * hadamard_classify.cpp
 *
 *  Created on: Oct 28, 2019
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {



static void hadamard_classify_early_test_function(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);



void hadamard_classify::init(
		int n,
		int f_draw,
		int verbose_level,
		int verbose_level_clique)
{
	int f_v = (verbose_level = 1);
	int i, j, k, d, cnt, cnt1;
	geometry::other_geometry::geometry_global Gg;


	if (n > (int)sizeof(int) * 8) {
		cout << "n > sizeof(uint) * 8" << endl;
		exit(1);
	}

	hadamard_classify::n = n;

	v = NEW_int(n);

	N = (1 << n);

	if (f_v) {
		cout << "n =" << n << endl;
		cout << "N =" << N << endl;
	}

	N2 = (N * (N - 1)) >> 1;

	if (f_v) {
		cout << "N2 = (N * (N - 1)) >> 1 =" << N2 << endl;
		cout << "list of points:" << endl;
		for (i = 0; i < N; i++) {
			Gg.AG_element_unrank(2, v, 1, n, i);
			cout << i << " : ";
			for (j = 0; j < n; j++) {
				if (v[j]) {
					cout << "+";
				}
				else {
					cout << "-";
				}
			}
			//int_vec_print(cout, v, n);
			cout << endl;
		}
	}

	Bitvec = NEW_OBJECT(other::data_structures::bitvector);
	Bitvec->allocate(N2);
	//bitvector_length = (N2 + 7) >> 3;

	//bitvector_adjacency = NEW_uchar(bitvector_length);

	if (f_v) {
		cout << "after allocating adjacency bitvector" << endl;
		cout << "computing adjacency matrix:" << endl;
	}
	k = 0;
	cnt = 0;
	for (i = 0; i < N; i++) {
		for (j = i + 1; j < N; j++) {

			d = dot_product(i, j, n);

			if (false) {
				cout << "dotproduct i=" << i << " j=" << j << " is " << d << endl;
			}

			if (d == 0) {
				Bitvec->m_i(k, 1);
				cnt++;
			}
			else {
				Bitvec->m_i(k, 0);
			}
			k++;
			if ((k & ((1 << 13) - 1)) == 0) {
				cout << "i=" << i << " j=" << j << " k=" << k << " / " << N2 << endl;
			}
		}
	}
	cout << "We have " << cnt << " edges in the graph" << endl;


#if 0
	// test the graph:

	k = 0;
	cnt1 = 0;
	for (i = 0; i < N; i++) {
		for (j = i + 1; j < N; j++) {

			d = dot_product(i, j, n);
			if (bitvector_s_i(bitvector_adjacency, k)) {
				cnt1++;
			}
			if (bitvector_s_i(bitvector_adjacency, k) && d) {
				cout << "something is wrong in entry i=" << i << " j=" << j << endl;
				cout << "dotproduct i=" << i << " j=" << j << " is " << d << endl;
				cout << "bitvector_s_i(bitvector_adjacency, k)="
						<< bitvector_s_i(bitvector_adjacency, k) << endl;
				exit(1);
			}
			k++;
		}
	}
	cout << "We found " << cnt1 << " edges in the graph" << endl;

	if (cnt1 != cnt) {
		cout << "cnt1 != cnt, something is wrong" << endl;
		cout << "cnt=" << cnt << endl;
		cout << "cnt1=" << cnt1 << endl;
		exit(1);
	}
#endif


	string label, label_tex;

	label = "Hadamard_graph_" + std::to_string(n);
	label_tex = "Hadamard\\_graph\\_" + std::to_string(n);


	{
		combinatorics::graph_theory::colored_graph *CG;
		string fname;

		CG = NEW_OBJECT(combinatorics::graph_theory::colored_graph);
		int *color;

		color = NEW_int(N);
		Int_vec_zero(color, N);


		CG->init_from_bitvector(
				N, 1, 1, color, Bitvec, false, label, label_tex,
				verbose_level);

		fname = label + ".colored_graph";

		CG->save(fname, verbose_level);


		FREE_int(color);
		FREE_OBJECT(CG);
	}




	CG = NEW_OBJECT(combinatorics::graph_theory::colored_graph);

	if (f_v) {
		cout << "initializing colored graph" << endl;
	}

	int *color;

	color = NEW_int(N);
	Int_vec_zero(color, N);

	CG->init_from_bitvector(
			N, 1, 1, color, Bitvec, false, label, label_tex,
			verbose_level);

	if (f_v) {
		cout << "initializing colored graph done" << endl;
	}

	string fname_graph;

	fname_graph = "Hadamard_graph_" + std::to_string(n) + ".magma";

	CG->export_to_magma(fname_graph, 1);

	{
		int *color_graph;

		color_graph = NEW_int(N * N);
		Int_vec_zero(color_graph, N * N);
		k = 0;
		cnt1 = 0;
		for (i = 0; i < N; i++) {
			for (j = i + 1; j < N; j++) {
				if (Bitvec->s_i(k)) {
					cnt1++;
					color_graph[i * N + j] = 2;
					color_graph[j * N + i] = 2;
				}
				else {
					color_graph[i * N + j] = 1;
					color_graph[j * N + i] = 1;
				}
				k++;
			}
		}
		cout << "We found " << cnt1 << " edges in the graph" << endl;

		if (cnt1 != cnt) {
			cout << "cnt1 != cnt, something is wrong" << endl;
			cout << "cnt=" << cnt << endl;
			cout << "cnt1=" << cnt1 << endl;
			exit(1);
		}

		cout << "color graph:" << endl;
		if (N < 30) {
			Int_matrix_print(color_graph, N, N);
		}
		else {
			cout << "Too big to print" << endl;
		}

	#if 0
		int *Pijk;
		int *colors;
		int nb_colors;

		is_association_scheme(color_graph, N, Pijk,
			colors, nb_colors, verbose_level);

		cout << "number of colors = " << nb_colors << endl;
		cout << "colors: ";
		int_vec_print(cout, colors, nb_colors);
		cout << endl;
		cout << "Pijk:" << endl;
		for (i = 0; i < nb_colors; i++) {
			cout << "i=" << i << ":" << endl;
			int_matrix_print(Pijk + i * nb_colors * nb_colors, nb_colors, nb_colors);
		}
		FREE_int(Pijk);
		FREE_int(colors);
	#endif

		FREE_int(color_graph);
	}

	if (f_draw) {
		if (f_v) {
			cout << "drawing adjacency matrix" << endl;
		}

		string fname_base;

		fname_base = "Hadamard_graph_" + std::to_string(n);

		other::graphics::layered_graph_draw_options *Draw_options;

		Draw_options = NEW_OBJECT(other::graphics::layered_graph_draw_options);

		CG->draw(fname_base,
				Draw_options,
				verbose_level);

		if (f_v) {
			cout << "drawing adjacency matrix done" << endl;
		}
	}


	if (f_v) {
		cout << "computing automorphism group of "
				"uncolored graph:" << endl;
	}

	interfaces::nauty_interface_for_graphs Nauty;


	A = Nauty.create_automorphism_group_of_graph_bitvec(
		CG->nb_points, Bitvec,
		verbose_level);

	algebra::ring_theory::longinteger_object go;
	A->group_order(go);
	if (f_v) {
		cout << "computing automorphism group of "
				"uncolored graph done, group order = " << go << endl;
	}

	string fname_group;


	fname_group = "Hadamard_group_" + std::to_string(n) + ".magma";

	interfaces::magma_interface Magma;

	Magma.export_permutation_group_to_magma(
			fname_group,
			A,
			A->Strong_gens,
			verbose_level);




	string prefix;
	prefix = "./had_" + std::to_string(n);

	if (f_v) {
		cout << "Starting the clique finder, "
				"target_depth = " << n << " prefix=" << prefix << endl;
	}

	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Poset;

	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
	Poset->init_subset_lattice(A, A,
			A->Strong_gens,
			verbose_level);
	Poset->add_testing_without_group(
			hadamard_classify_early_test_function,
			this /* void *data */,
			verbose_level);

	gen = NEW_OBJECT(poset_classification::poset_classification);
	Control = NEW_OBJECT(poset_classification::poset_classification_control);
	Control->f_W = true;
	Control->problem_label = prefix;
	Control->f_problem_label = true;


	gen->compute_orbits_on_subsets(
		n /* target_depth */,
		//prefix,
		//true /* f_W */, false /* f_w */,
		Control,
		Poset,
		verbose_level_clique);

	nb_orbits = gen->nb_orbits_at_level(n);

	int h, a, c;
	long int *set;
	int *H;
	int *Ht;
	int *M;

	set = NEW_lint(n);
	H = NEW_int(n * n);
	Ht = NEW_int(n * n);
	M = NEW_int(n * n);
	for (h = 0; h < nb_orbits; h++) {
		gen->get_set_by_level(n, h, set);
		cout << "Orbit " << h << " is the set ";
		Lint_vec_print(cout, set, n);
		cout << endl;


		if (clique_test(set, n)) {
			cout << "is a clique" << endl;
		}
		else {
			cout << "is not a clique, this should not happen" << endl;
			exit(1);
		}

		for (j = 0; j < n; j++) {
			a = set[j];
			for (i = 0; i < n; i++) {
				if (a % 2) {
					H[i * n + j] = 1;
				}
				else {
					H[i * n + j] = -1;
				}
				a >>= 1;
			}
		}
		cout << "The Hadamard matrix " << h << " is:" << endl;
		Int_matrix_print(H, n, n);
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				a = H[i * n + j];
				Ht[j * n + i] = a;
			}
		}

		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				c = 0;
				for (k = 0; k < n; k++) {
					c += H[i * n + k] * Ht[k * n + j];
				}
				M[i * n + j] = c;
			}
		}
		cout << "The matrix H * H^t is:" << endl;
		Int_matrix_print(M, n, n);
	}
}

int hadamard_classify::clique_test(
		long int *set, int sz)
{
	long int i, j, a, b, idx;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	for (i = 0; i < n; i++) {
		a = set[i];
		for (j = i + 1; j < n; j++) {
			b = set[j];
			idx = Combi.ij2k_lint(a, b, N);
			if (Bitvec->s_i(idx)) {
				//cout << "pair (" << i << "," << j << ") vertices " << a << " and " << b << " are adjacent" << endl;
			}
			else {
				//cout << "pair (" << i << "," << j << ") vertices " << a << " and " << b << " are NOT adjacent" << endl;
				return false;
			}
		}
	}
	return true;
}

void hadamard_classify::early_test_func(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	long int j, a, pt;

	if (f_v) {
		cout << "hadamard_classify::early_test_func checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size " << nb_candidates << ":" << endl;
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
	}
	if (len == 0) {
		nb_good_candidates = nb_candidates;
		Lint_vec_copy(candidates, good_candidates, nb_candidates);
		return;
	}

	pt = S[len - 1];

	nb_good_candidates = 0;
	for (j = 0; j < nb_candidates; j++) {
		a = candidates[j];

		if (CG->is_adjacent(pt, a)) {
			good_candidates[nb_good_candidates++] = a;
		}
	} // next j

}


int hadamard_classify::dot_product(
		int a, int b, int n)
{
	int i, c, aa, bb;

	c = 0;
	for (i = 0; i < n; i++) {
		aa = a % 2;
		bb = b % 2;
		if (aa == bb) {
			c++;
		}
		else {
			c--;
		}
		a >>= 1;
		b >>= 1;
	}
	return c;
}

void hadamard_classify_early_test_function(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	hadamard_classify *H = (hadamard_classify *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hadamard_classify_early_test_function for set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
	}
	H->early_test_func(S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
	verbose_level - 2);
	if (f_v) {
		cout << "hadamard_classify_early_test_function done" << endl;
	}
}



}}}



