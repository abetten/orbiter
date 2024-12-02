/*
 * schlaefli_trihedral_pairs.cpp
 *
 *  Created on: Nov 15, 2023
 *      Author: betten
 */





#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace algebraic_geometry {


schlaefli_trihedral_pairs::schlaefli_trihedral_pairs()
{
	Record_birth();

	Schlaefli = NULL;

	Trihedral_pairs = NULL;
	Trihedral_pair_labels = NULL;
	Trihedral_pairs_row_sets = NULL;
	Trihedral_pairs_col_sets = NULL;
	nb_trihedral_pairs = 0;

	Triads = NULL;
	nb_triads = 0;

	Classify_trihedral_pairs_row_values = NULL;
	Classify_trihedral_pairs_col_values = NULL;

	//nb_trihedral_to_Eckardt = 0;
	Axes = NULL;
	nb_axes = 0;
	Axes_sorted = NULL;
	//Axes_sorted_perm_inv = NULL;

	nb_collinear_Eckardt_triples = 0;
	collinear_Eckardt_triples_rank = NULL;

	Classify_collinear_Eckardt_triples = NULL;



}

schlaefli_trihedral_pairs::~schlaefli_trihedral_pairs()
{
	Record_death();
	if (Trihedral_pairs) {
		FREE_int(Trihedral_pairs);
	}
	if (Trihedral_pair_labels) {
		delete [] Trihedral_pair_labels;
	}
	if (Trihedral_pairs_row_sets) {
		FREE_int(Trihedral_pairs_row_sets);
	}
	if (Trihedral_pairs_col_sets) {
		FREE_int(Trihedral_pairs_col_sets);
	}

	if (Triads) {
		FREE_int(Triads);
	}


	if (Classify_trihedral_pairs_row_values) {
		FREE_OBJECT(Classify_trihedral_pairs_row_values);
	}
	if (Classify_trihedral_pairs_col_values) {
		FREE_OBJECT(Classify_trihedral_pairs_col_values);
	}


	if (Axes) {
		FREE_lint(Axes);
	}
	if (Axes_sorted) {
		FREE_OBJECT(Axes_sorted);
	}

	if (collinear_Eckardt_triples_rank) {
		FREE_int(collinear_Eckardt_triples_rank);
	}
	if (Classify_collinear_Eckardt_triples) {
		FREE_OBJECT(Classify_collinear_Eckardt_triples);
	}

}

void schlaefli_trihedral_pairs::init(
		schlaefli *Schlaefli, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::init" << endl;
	}

	schlaefli_trihedral_pairs::Schlaefli = Schlaefli;


	if (f_v) {
		cout << "schlaefli_trihedral_pairs::init "
				"before make_trihedral_pairs" << endl;
	}
	make_trihedral_pairs(verbose_level);
	if (f_v) {
		cout << "schlaefli_trihedral_pairs::init "
				"after make_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::init "
				"before make_triads" << endl;
	}
	make_triads(verbose_level);
	if (f_v) {
		cout << "schlaefli_trihedral_pairs::init "
				"after make_triads" << endl;
	}

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::init "
				"before process_trihedral_pairs" << endl;
	}
	process_trihedral_pairs(verbose_level);
	if (f_v) {
		cout << "schlaefli_trihedral_pairs::init "
				"after process_trihedral_pairs" << endl;
	}


	if (f_v) {
		cout << "schlaefli_trihedral_pairs::init "
				"before init_axes" << endl;
	}
	init_axes(verbose_level);
	if (f_v) {
		cout << "schlaefli_trihedral_pairs::init "
				"after init_axes" << endl;
	}

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::init "
				"before init_collinear_Eckardt_triples" << endl;
	}
	init_collinear_Eckardt_triples(verbose_level);
	if (f_v) {
		cout << "schlaefli_trihedral_pairs::init "
				"after init_collinear_Eckardt_triples" << endl;
	}

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::init done" << endl;
	}
}


void schlaefli_trihedral_pairs::make_trihedral_pairs(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, s, idx;
	int subset[6];
	int second_subset[6];
	int complement[6];
	int subset_complement[6];
	int size_complement;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	other::l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::make_trihedral_pairs" << endl;
	}
	nb_trihedral_pairs = 120;
	Trihedral_pairs = NEW_int(nb_trihedral_pairs * 9);
	Trihedral_pair_labels = new std::string [nb_trihedral_pairs];

	idx = 0;

	// the first type (20 of them, 6 choose 3):
	for (h = 0; h < 20; h++, idx++) {
		Combi.unrank_k_subset(h, subset, 6, 3);
		Combi.set_complement(subset, 3, complement,
			size_complement, 6);

		make_Tijk(Trihedral_pairs + idx * 9, subset[0], subset[1], subset[2]);
		Trihedral_pair_labels[idx] =
				std::to_string(subset[0] + 1)
				+ std::to_string(subset[1] + 1)
				+ std::to_string(subset[2] + 1)
				+ ";"
				+ std::to_string(complement[0] + 1)
				+ std::to_string(complement[1] + 1)
				+ std::to_string(complement[2] + 1);
	}

	// the second type (90 of them, (6 choose 2) times (4 choose 2)):
	for (h = 0; h < 15; h++) {
		Combi.unrank_k_subset(h, subset, 6, 4);
		Combi.set_complement(subset, 4, subset_complement,
			size_complement, 6);
		for (s = 0; s < 6; s++, idx++) {
			Combi.unrank_k_subset(s, second_subset, 4, 2);
			Combi.set_complement(second_subset, 2, complement,
				size_complement, 4);
			make_Tlmnp(Trihedral_pairs + idx * 9,
				subset[second_subset[0]],
				subset[second_subset[1]],
				subset[complement[0]],
				subset[complement[1]]);
			Trihedral_pair_labels[idx] =
					std::to_string(subset[second_subset[0]] + 1)
					+ std::to_string(subset[second_subset[1]] + 1)
					+ ";" + std::to_string(subset[complement[0]] + 1)
					+ std::to_string(subset[complement[1]] + 1)
					+ ";" + std::to_string(subset_complement[0] + 1)
					+ std::to_string(subset_complement[1] + 1);
		}
	}

	// the third type (10 of them, (6 choose 3) divide by 2):
	for (h = 0; h < 10; h++, idx++) {

		Combi.unrank_k_subset(h, subset + 1, 5, 2);

		subset[0] = 0;
		subset[1]++;
		subset[2]++;

		Combi.set_complement(subset, 3, complement,
			size_complement, 6);

		make_Tdefght(Trihedral_pairs + idx * 9,
			subset[0], subset[1], subset[2],
			complement[0], complement[1], complement[2]);

		Trihedral_pair_labels[idx] =
				std::to_string(subset[0] + 1)
				+ std::to_string(subset[1] + 1)
				+ std::to_string(subset[2] + 1)
				+ ","
				+ std::to_string(complement[0] + 1)
				+ std::to_string(complement[1] + 1)
				+ std::to_string(complement[2] + 1);
	}

	if (idx != 120) {
		cout << "schlaefli_trihedral_pairs::make_trihedral_pairs idx != 120" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "The trihedral pairs are:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
				Trihedral_pairs, 120, 9, false /* f_tex */);
		L.print_integer_matrix_with_standard_labels(cout,
				Trihedral_pairs, 120, 9, true /* f_tex */);
	}

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::make_trihedral_pairs done" << endl;
	}
}

void schlaefli_trihedral_pairs::make_Tijk(
		int *T, int i, int j, int k)
{
	T[0] = Schlaefli->line_cij(j, k);
	T[1] = Schlaefli->line_bi(k);
	T[2] = Schlaefli->line_ai(j);
	T[3] = Schlaefli->line_ai(k);
	T[4] = Schlaefli->line_cij(i, k);
	T[5] = Schlaefli->line_bi(i);
	T[6] = Schlaefli->line_bi(j);
	T[7] = Schlaefli->line_ai(i);
	T[8] = Schlaefli->line_cij(i, j);
}

void schlaefli_trihedral_pairs::make_Tlmnp(
		int *T, int l, int m, int n, int p)
{
	int subset[4];
	int complement[2];
	int size_complement;
	int r, s;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	other::data_structures::sorting Sorting;

	subset[0] = l;
	subset[1] = m;
	subset[2] = n;
	subset[3] = p;
	Sorting.int_vec_heapsort(subset, 4);
	Combi.set_complement(subset, 4, complement, size_complement, 6);
	r = complement[0];
	s = complement[1];

	T[0] = Schlaefli->line_ai(l);
	T[1] = Schlaefli->line_bi(p);
	T[2] = Schlaefli->line_cij(l, p);
	T[3] = Schlaefli->line_bi(n);
	T[4] = Schlaefli->line_ai(m);
	T[5] = Schlaefli->line_cij(m, n);
	T[6] = Schlaefli->line_cij(l, n);
	T[7] = Schlaefli->line_cij(m, p);
	T[8] = Schlaefli->line_cij(r, s);
}

void schlaefli_trihedral_pairs::make_Tdefght(
		int *T,
		int d, int e, int f, int g, int h, int t)
{
	T[0] = Schlaefli->line_cij(d, g);
	T[1] = Schlaefli->line_cij(e, h);
	T[2] = Schlaefli->line_cij(f, t);
	T[3] = Schlaefli->line_cij(e, t);
	T[4] = Schlaefli->line_cij(f, g);
	T[5] = Schlaefli->line_cij(d, h);
	T[6] = Schlaefli->line_cij(f, h);
	T[7] = Schlaefli->line_cij(d, t);
	T[8] = Schlaefli->line_cij(e, g);
}

void schlaefli_trihedral_pairs::make_triads(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::sorting Sorting;
	int *Adj;
	int i, j, h, u;

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::make_triads" << endl;
	}
	make_trihedral_pair_disjointness_graph(Adj, verbose_level);
	Triads = NEW_int(40 * 3);
	h = 0;
	for (i = 0; i < nb_trihedral_pairs; i++) {
		for (j = 0; j < i; j++) {
			if (Adj[j * nb_trihedral_pairs + i]) {
				break;
			}
		}
		if (j < i) {
			continue;
		}
		Triads[h * 3 + 0] = i;
		u = 1;
		for (j = i + 1; j < nb_trihedral_pairs; j++) {
			if (Adj[i * nb_trihedral_pairs + j]) {
				Triads[h * 3 + u] = j;
				u++;
			}
		}
		if (u != 3) {
			cout << "schlaefli_trihedral_pairs::make_triads u != 3" << endl;
			exit(1);
		}
		h++;
	}
	if (h != 40) {
		cout << "schlaefli_trihedral_pairs::make_triads h != 40" << endl;
		exit(1);
	}
	nb_triads = h;
	if (f_v) {
		cout << "schlaefli_trihedral_pairs::make_triads done" << endl;
	}
}

void schlaefli_trihedral_pairs::make_trihedral_pair_disjointness_graph(
		int *&Adj, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::sorting Sorting;
	int *T;
	int i, j;

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::make_trihedral_pair_disjointness_graph" << endl;
		cout << "nb_trihedral_pairs=" << nb_trihedral_pairs << endl;
	}
	Adj = NEW_int(nb_trihedral_pairs * nb_trihedral_pairs);
	T = NEW_int(nb_trihedral_pairs * 9);
	for (i = 0; i < nb_trihedral_pairs; i++) {
		Int_vec_copy(Trihedral_pairs + i * 9, T + i * 9, 9);
		Sorting.int_vec_heapsort(T + i * 9, 9);
	}
	Int_vec_zero(Adj, nb_trihedral_pairs * nb_trihedral_pairs);
	for (i = 0; i < nb_trihedral_pairs; i++) {
		for (j = i + 1; j < nb_trihedral_pairs; j++) {
			if (Sorting.int_vecs_are_disjoint(T + i * 9, 9, T + j * 9, 9)) {
				Adj[i * nb_trihedral_pairs + j] = 1;
				Adj[j * nb_trihedral_pairs + 1] = 1;
			}
			else {
			}
		}
	}
	FREE_int(T);

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::make_trihedral_pair_disjointness_graph done" << endl;
	}
}

void schlaefli_trihedral_pairs::process_trihedral_pairs(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int subset[3];
	int i, j, h, rk, a;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	other::data_structures::sorting Sorting;
	other::l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::process_trihedral_pairs" << endl;
	}
	Trihedral_pairs_row_sets = NEW_int(nb_trihedral_pairs * 3);
	Trihedral_pairs_col_sets = NEW_int(nb_trihedral_pairs * 3);
	for (i = 0; i < nb_trihedral_pairs; i++) {
		for (j = 0; j < 3; j++) {
			for (h = 0; h < 3; h++) {
				a = Trihedral_pairs[i * 9 + j * 3 + h];
				subset[h] = a;
			}
			Sorting.int_vec_heapsort(subset, 3);
			rk = Combi.rank_k_subset(subset, 27, 3);
			//rk = Eckardt_point_from_tritangent_plane(subset);
			Trihedral_pairs_row_sets[i * 3 + j] = rk;
		}
	}
	for (i = 0; i < nb_trihedral_pairs; i++) {
		for (j = 0; j < 3; j++) {
			for (h = 0; h < 3; h++) {
				a = Trihedral_pairs[i * 9 + h * 3 + j];
				subset[h] = a;
			}
			Sorting.int_vec_heapsort(subset, 3);
			rk = Combi.rank_k_subset(subset, 27, 3);
			//rk = Eckardt_point_from_tritangent_plane(subset);
			Trihedral_pairs_col_sets[i * 3 + j] = rk;
		}
	}

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::process_trihedral_pairs "
				"The trihedral pairs row sets:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
			Trihedral_pairs_row_sets, 120, 3,
			false /* f_tex */);

		cout << "The trihedral pairs col sets:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
			Trihedral_pairs_col_sets, 120, 3,
			false /* f_tex */);
	}

	Classify_trihedral_pairs_row_values = NEW_OBJECT(other::data_structures::tally);
	Classify_trihedral_pairs_row_values->init(
		Trihedral_pairs_row_sets, 120 * 3, false, 0);

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::process_trihedral_pairs "
				"sorted row values:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
			Classify_trihedral_pairs_row_values->data_sorted,
			120 * 3 / 10, 10, false /* f_tex */);
	}

	Classify_trihedral_pairs_col_values = NEW_OBJECT(other::data_structures::tally);
	Classify_trihedral_pairs_col_values->init(
		Trihedral_pairs_col_sets,
		120 * 3, false, 0);

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::process_trihedral_pairs "
				"sorted col values:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
			Classify_trihedral_pairs_col_values->data_sorted,
			120 * 3 / 10, 10, false /* f_tex */);
	}
	if (f_v) {
		cout << "schlaefli_trihedral_pairs::process_trihedral_pairs done" << endl;
	}
}

void schlaefli_trihedral_pairs::init_axes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t, i, j, rk;
	int tritangent_plane[3];
	other::l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::init_axes" << endl;
	}
	//nb_trihedral_to_Eckardt = nb_trihedral_pairs * 6;
	nb_axes = nb_trihedral_pairs * 2;
	Axes = NEW_lint(nb_trihedral_pairs * 6);
	for (t = 0; t < nb_trihedral_pairs; t++) {
		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				tritangent_plane[j] = Trihedral_pairs[t * 9 + i * 3 + j];
				}
			rk = Schlaefli->Schlaefli_tritangent_planes->Eckardt_point_from_tritangent_plane(tritangent_plane);
			Axes[t * 6 + i] = rk;
		}
		for (j = 0; j < 3; j++) {
			for (i = 0; i < 3; i++) {
				tritangent_plane[i] = Trihedral_pairs[t * 9 + i * 3 + j];
			}
			rk = Schlaefli->Schlaefli_tritangent_planes->Eckardt_point_from_tritangent_plane(tritangent_plane);
			Axes[t * 6 + 3 + j] = rk;
		}
	}
	if (f_v) {
		cout << "Axes:" << endl;
		L.print_lint_matrix_with_standard_labels(
				cout,
				Axes, nb_trihedral_pairs, 6,
			false /* f_tex */);
	}

	other::data_structures::sorting Sorting;


	Axes_sorted = NEW_OBJECT(other::data_structures::int_matrix);
	Axes_sorted->allocate(nb_axes, 3);

	for (t = 0; t < nb_trihedral_pairs; t++) {
		for (i = 0; i < 2; i++) {
			//Axes_sorted_perm_inv[2 * t + i] = 2 * t + i;
			Lint_vec_copy_to_int(Axes + t * 6 + i * 3, Axes_sorted->M + 2 * t + i, 3);
			Sorting.int_vec_heapsort(Axes_sorted->M + 2 * t + i, 3);
		}
	}

	Axes_sorted->sort_rows(verbose_level);




	if (f_v) {
		cout << "schlaefli_trihedral_pairs::init_axes done" << endl;
	}
}

int schlaefli_trihedral_pairs::identify_axis(
		int *axis_E_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::identify_axis" << endl;
	}
	int v[3];
	other::data_structures::sorting Sorting;

	Int_vec_copy(axis_E_idx, v, 3);

	Sorting.int_vec_heapsort(v, 3);

	int idx;

	if (!Axes_sorted->search(v, idx, 0 /*verbose_level*/)) {
		cout << "schlaefli_trihedral_pairs::identify_axis cannot find the axis. Something is wrong." << endl;
		exit(1);
	}
	idx = Axes_sorted->perm_inv[idx];

	return idx;
}

void schlaefli_trihedral_pairs::init_collinear_Eckardt_triples(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t, i, rk, h;
	int subset[3];
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	other::data_structures::sorting Sorting;
	other::l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::init_collinear_Eckardt_triples" << endl;
	}
	nb_collinear_Eckardt_triples = nb_trihedral_pairs * 2;
	collinear_Eckardt_triples_rank = NEW_int(nb_collinear_Eckardt_triples);
	for (t = 0; t < nb_trihedral_pairs; t++) {
		for (i = 0; i < 2; i++) {
			for (h = 0; h < 3; h++) {
				subset[h] = Axes[6 * t + i * 3 + h];
			}
			Sorting.int_vec_heapsort(subset, 3);
			rk = Combi.rank_k_subset(subset, Schlaefli->Schlaefli_tritangent_planes->nb_Eckardt_points, 3);
			collinear_Eckardt_triples_rank[t * 2 + i] = rk;
		}
	}
	if (f_v) {
		cout << "collinear_Eckardt_triples_rank:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
			collinear_Eckardt_triples_rank, nb_trihedral_pairs, 2,
			false /* f_tex */);
	}

	Classify_collinear_Eckardt_triples = NEW_OBJECT(other::data_structures::tally);
	Classify_collinear_Eckardt_triples->init(
		collinear_Eckardt_triples_rank, nb_collinear_Eckardt_triples,
		false, 0);

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::init_collinear_Eckardt_triples done" << endl;
	}
}

void schlaefli_trihedral_pairs::find_trihedral_pairs_from_collinear_triples_of_Eckardt_points(
	int *E_idx, int nb_E,
	int *&T_idx, int &nb_T, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nCk, h, k, rk, idx, i, t_idx;
	int subset[3];
	int set[3];
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "schlaefli_trihedral_pairs::find_trihedral_pairs_from_collinear_triples_of_Eckardt_points" << endl;
	}
	nCk = Combi.int_n_choose_k(nb_E, 3);
	T_idx = NEW_int(nCk);
	nb_T = 0;
	for (h = 0; h < nCk; h++) {
		//cout << "subset " << h << " / " << nCk << ":";
		Combi.unrank_k_subset(h, subset, nb_E, 3);
		//int_vec_print(cout, subset, 3);
		//cout << " = ";

		for (k = 0; k < 3; k++) {
			set[k] = E_idx[subset[k]];
		}
		//int_vec_print(cout, set, 3);
		//cout << " = ";
		Sorting.int_vec_heapsort(set, 3);

		rk = Combi.rank_k_subset(set, Schlaefli->Schlaefli_tritangent_planes->nb_Eckardt_points, 3);


		//int_vec_print(cout, set, 3);
		//cout << " rk=" << rk << endl;

		if (Sorting.int_vec_search(
			Classify_collinear_Eckardt_triples->data_sorted,
			nb_collinear_Eckardt_triples, rk, idx)) {
			//cout << "idx=" << idx << endl;
			for (i = idx; i >= 0; i--) {
				//cout << "i=" << i << " value="
				// << Classify_collinear_Eckardt_triples->data_sorted[i]
				// << " collinear triple index = "
				// << Classify_collinear_Eckardt_triples->sorting_perm_inv[
				// i] / 3 << endl;
				if (Classify_collinear_Eckardt_triples->data_sorted[i] != rk) {
					break;
				}
				t_idx =
				Classify_collinear_Eckardt_triples->sorting_perm_inv[i] / 2;

#if 0
				int idx2, j;

				if (!int_vec_search(T_idx, nb_T, t_idx, idx2)) {
					for (j = nb_T; j > idx2; j--) {
						T_idx[j] = T_idx[j - 1];
					}
					T_idx[idx2] = t_idx;
					nb_T++;
				}
				else {
					cout << "We already have this trihedral pair" << endl;
				}
#else
				T_idx[nb_T++] = t_idx;
#endif
			}
		}

	}


#if 1
	other::data_structures::tally C;

	C.init(T_idx, nb_T, true, 0);
	cout << "The trihedral pairs come in these multiplicities: ";
	C.print_bare(true);
	cout << endl;

	int t2, f2, l2, sz;
	int t1, f1, /*l1,*/ pt;

	for (t2 = 0; t2 < C.second_nb_types; t2++) {
		f2 = C.second_type_first[t2];
		l2 = C.second_type_len[t2];
		sz = C.second_data_sorted[f2];
		if (sz != 1) {
			continue;
		}
		//cout << "fibers of size "
		// << sz << ":" << endl;
		//*fp << "There are " << l2 << " fibers of size " << sz
		// << ":\\\\" << endl;
		for (i = 0; i < l2; i++) {
			t1 = C.second_sorting_perm_inv[f2 + i];
			f1 = C.type_first[t1];
			//l1 = C.type_len[t1];
			pt = C.data_sorted[f1];
			T_idx[i] = pt;
#if 0
			//*fp << "Arc pt " << pt << ", fiber $\\{"; // << l1
			// << " surface points in the list of Pts (local numbering): ";
			for (j = 0; j < l1; j++) {
				u = C.sorting_perm_inv[f1 + j];

				cout << u << endl;
				//*fp << u;
				//cout << Pts[u];
				if (j < l1 - 1) {
					cout << ", ";
				}
			}
#endif
		}
		nb_T = l2;
	}
#endif



	cout << "Found " << nb_T << " special trihedral pairs:" << endl;
	cout << "T_idx: ";
	Int_vec_print(cout, T_idx, nb_T);
	cout << endl;
	for (i = 0; i < nb_T; i++) {
		cout << i << " / " << nb_T << " T_{"
			<< Trihedral_pair_labels[T_idx[i]] << "}" << endl;
	}
	if (f_v) {
		cout << "schlaefli_trihedral_pairs::find_trihedral_pairs_from_collinear_triples_of_Eckardt_points done" << endl;
	}
}

void schlaefli_trihedral_pairs::latex_abstract_trihedral_pair(
		std::ostream &ost, int t_idx)
{
	latex_trihedral_pair_as_matrix(
			ost,
			Trihedral_pairs + t_idx * 9,
			Axes + t_idx * 6);
}

void schlaefli_trihedral_pairs::latex_trihedral_pair_as_matrix(
		std::ostream &ost, int *T, long int *TE)
{
	int i, j;

	ost << "\\begin{array}{*{" << 3 << "}{c}|c}" << endl;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			Schlaefli->print_line(ost, T[i * 3 + j]);
			ost << " & ";
		}
		ost << "\\pi_{";
		Schlaefli->Schlaefli_tritangent_planes->Eckardt_points[TE[i]].latex_index_only(ost);
		ost << "}\\\\" << endl;
	}
	ost << "\\hline" << endl;
	for (j = 0; j < 3; j++) {
		ost << "\\pi_{";
		Schlaefli->Schlaefli_tritangent_planes->Eckardt_points[TE[3 + j]].latex_index_only(ost);
		ost << "} & ";
	}
	ost << "\\\\" << endl;
	ost << "\\end{array}" << endl;
}

void schlaefli_trihedral_pairs::latex_table_of_trihedral_pairs(
		std::ostream &ost)
{
	int i;

	cout << "schlaefli::latex_table_of_trihedral_pairs" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_trihedral_pairs; i++) {
		ost << "$T_{" << i << "} = T_{"
			<< Trihedral_pair_labels[i] << "} = \\\\" << endl;
		//ost << "\\left[" << endl;
		//ost << "\\begin{array}" << endl;
		latex_trihedral_pair_as_matrix(
				ost, Trihedral_pairs + i * 9,
				Axes + i * 6);
		//ost << "\\end{array}" << endl;
		//ost << "\\right]" << endl;
		ost << "$\\\\" << endl;
#if 0
		ost << "planes: $";
		int_vec_print(ost, Trihedral_to_Eckardt + i * 6, 6);
		ost << "$\\\\" << endl;
#endif
	}
	ost << "\\end{multicols}" << endl;

	print_trihedral_pairs(ost);

	cout << "schlaefli_trihedral_pairs::latex_table_of_trihedral_pairs done" << endl;
}



void schlaefli_trihedral_pairs::latex_triads(
		std::ostream &ost)
{
	other::l1_interfaces::latex_interface L;
	int i, j, a;

	cout << "schlaefli_trihedral_pairs::latex_triads" << endl;

	ost << "\\subsection*{Triads}" << endl;
	ost << "The 40 triads are:\\\\" << endl;

	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(
			ost,
			Triads, 40, 3, 0, 0, true /* f_tex*/);
	ost << "$$";

	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_triads; i++) {
		ost << "\\noindent ${\\cal T}_{" << i << "} = \\{";
		for (j = 0; j < 3; j++) {
			a = Triads[i * 3 + j];
			ost << "T_{" << Trihedral_pair_labels[a] << "}";
			if (j < 3 - 1) {
				ost << ", ";
			}
		}
		ost << "\\}$\\\\" << endl;
	}
	ost << "\\end{multicols}" << endl;

	cout << "schlaefli_trihedral_pairs::latex_triads done" << endl;
}

void schlaefli_trihedral_pairs::print_trihedral_pairs(
		std::ostream &ost)
{
	other::l1_interfaces::latex_interface L;
	int i, j, a;

	//ost << "\\clearpage" << endl;

	ost << "\\bigskip" << endl;


	ost << "\\subsection*{Trihedral pairs}" << endl;
	ost << "The 120 trihedral pairs are:\\\\" << endl;
	ost << "{\\renewcommand{\\arraystretch}{1.3}" << endl;
	ost << "$$" << endl;

	int n = 6;
	int n_offset = 0;
	int m = 40;
	int m_offset = 0;
	long int *p = Axes;

	ost << "\\begin{array}{|r|r|*{" << n << "}r|}" << endl;
	ost << "\\hline" << endl;
	ost << " & ";
	for (j = 0; j < n; j++) {
		ost << " & " << n_offset + j;
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < m; i++) {
		ost << m_offset + i << " & S_{";
		ost << Trihedral_pair_labels[m_offset + i] << "}";
		for (j = 0; j < n; j++) {
			a = p[i * n + j];
			ost << " & \\pi_{" << Schlaefli->Schlaefli_tritangent_planes->Eckard_point_label_tex[a] << "}";
		}
		ost << "\\\\";
		ost << endl;
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;

	//L.print_integer_matrix_with_standard_labels(ost,
	//	Surf->Trihedral_to_Eckardt, 40, 6, true /* f_tex */);
	ost << "$$" << endl;


	ost << "$$" << endl;

	m_offset = 40;
	p = Axes + 40 * 6;

	ost << "\\begin{array}{|r|r|*{" << n << "}r|}" << endl;
	ost << "\\hline" << endl;
	ost << " & ";
	for (j = 0; j < n; j++) {
		ost << " & " << n_offset + j;
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < m; i++) {
		ost << m_offset + i << " & S_{";
		ost << Trihedral_pair_labels[m_offset + i] << "}";
		for (j = 0; j < n; j++) {
			a = p[i * n + j];
			ost << " & \\pi_{" << Schlaefli->Schlaefli_tritangent_planes->Eckard_point_label_tex[a] << "}";
		}
		ost << "\\\\";
		ost << endl;
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;


	//L.print_integer_matrix_with_standard_labels_and_offset(ost,
	//	Surf->Trihedral_to_Eckardt + 40 * 6, 40, 6, 40, 0,
	//	true /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;

	m_offset = 80;
	p = Axes + 80 * 6;

	ost << "\\begin{array}{|r|r|*{" << n << "}r|}" << endl;
	ost << "\\hline" << endl;
	ost << " & ";
	for (j = 0; j < n; j++) {
		ost << " & " << n_offset + j;
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < m; i++) {
		ost << m_offset + i << " & S_{";
		ost << Trihedral_pair_labels[m_offset + i] << "}";
		for (j = 0; j < n; j++) {
			a = p[i * n + j];
			ost << " & \\pi_{" << Schlaefli->Schlaefli_tritangent_planes->Eckard_point_label_tex[a] << "}";
		}
		ost << "\\\\";
		ost << endl;
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;


	//L.print_integer_matrix_with_standard_labels_and_offset(ost,
	//	Surf->Trihedral_to_Eckardt + 80 * 6, 40, 6, 80, 0,
	//	true /* f_tex */);
	ost << "$$}" << endl;
}




}}}}


