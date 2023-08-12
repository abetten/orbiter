/*
 * schlaefli.cpp
 *
 *  Created on: Oct 13, 2020
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {


schlaefli::schlaefli()
{
	Surf = NULL;

	Labels = NULL;

	Trihedral_pairs = NULL;
	Trihedral_pair_labels = NULL;
	nb_trihedral_pairs = 0;

	Triads = NULL;
	nb_triads = 0;

	Trihedral_pairs_row_sets = NULL;
	Trihedral_pairs_col_sets = NULL;
	Classify_trihedral_pairs_row_values = NULL;
	Classify_trihedral_pairs_col_values = NULL;
	nb_Eckardt_points = 0;
	Eckardt_points = NULL;
	Eckard_point_label = NULL;
	Eckard_point_label_tex = NULL;

	nb_trihedral_to_Eckardt = 0;
	Trihedral_to_Eckardt = NULL;

	nb_collinear_Eckardt_triples = 0;
	collinear_Eckardt_triples_rank = NULL;

	Classify_collinear_Eckardt_triples = NULL;

	Double_six = NULL;
	Double_six_label_tex = NULL;

	Half_double_six_characteristic_vector = NULL;

	Double_six_characteristic_vector = NULL;

	Half_double_sixes = NULL;
	Half_double_six_label_tex = NULL;
	Half_double_six_to_double_six = NULL;
	Half_double_six_to_double_six_row = NULL;

	adjacency_matrix_of_lines = NULL;
	incidence_lines_vs_tritangent_planes = NULL;
	Lines_in_tritangent_planes = NULL;

}

schlaefli::~schlaefli()
{
	int f_v = false;

	if (Eckard_point_label) {
		delete [] Eckard_point_label;
	}
	if (Eckard_point_label_tex) {
		delete [] Eckard_point_label_tex;
	}

	if (Labels) {
		FREE_OBJECT(Labels);
	}

	if (Trihedral_to_Eckardt) {
		FREE_lint(Trihedral_to_Eckardt);
	}
	if (collinear_Eckardt_triples_rank) {
		FREE_int(collinear_Eckardt_triples_rank);
	}
	if (Classify_collinear_Eckardt_triples) {
		FREE_OBJECT(Classify_collinear_Eckardt_triples);
	}

	if (f_v) {
		cout << "before FREE_int(Trihedral_pairs);" << endl;
	}
	if (Trihedral_pairs) {
		FREE_int(Trihedral_pairs);
	}
	if (Trihedral_pair_labels) {
		delete [] Trihedral_pair_labels;
	}

	if (Triads) {
		FREE_int(Triads);
	}

	if (Trihedral_pairs_row_sets) {
		FREE_int(Trihedral_pairs_row_sets);
	}
	if (Trihedral_pairs_col_sets) {
		FREE_int(Trihedral_pairs_col_sets);
	}
	if (f_v) {
		cout << "before FREE_OBJECT Classify_trihedral_pairs_"
				"row_values;" << endl;
	}
	if (Classify_trihedral_pairs_row_values) {
		FREE_OBJECT(Classify_trihedral_pairs_row_values);
	}
	if (Classify_trihedral_pairs_col_values) {
		FREE_OBJECT(Classify_trihedral_pairs_col_values);
	}
	if (Eckardt_points) {
		FREE_OBJECTS(Eckardt_points);
	}

	if (Double_six) {
		FREE_lint(Double_six);
	}
	if (Double_six_label_tex) {
		delete [] Double_six_label_tex;
	}

	if (Half_double_six_characteristic_vector) {
		FREE_int(Half_double_six_characteristic_vector);
	}

	if (Double_six_characteristic_vector) {
		FREE_int(Double_six_characteristic_vector);
	}

	if (Half_double_sixes) {
		FREE_lint(Half_double_sixes);
	}

	if (Half_double_six_label_tex) {
		delete [] Half_double_six_label_tex;
	}

	if (Half_double_six_to_double_six) {
		FREE_int(Half_double_six_to_double_six);
	}
	if (Half_double_six_to_double_six_row) {
		FREE_int(Half_double_six_to_double_six_row);
	}

	if (adjacency_matrix_of_lines) {
		FREE_int(adjacency_matrix_of_lines);
	}
	if (incidence_lines_vs_tritangent_planes) {
		FREE_int(incidence_lines_vs_tritangent_planes);
	}
	if (Lines_in_tritangent_planes) {
		FREE_lint(Lines_in_tritangent_planes);
	}

}

void schlaefli::init(
		surface_domain *Surf, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schlaefli::init" << endl;
	}

	schlaefli::Surf = Surf;


	Labels = NEW_OBJECT(schlaefli_labels);
	if (f_v) {
		cout << "schlaefli::init "
				"before Labels->init" << endl;
	}
	Labels->init(verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after Labels->init" << endl;
	}


	if (f_v) {
		cout << "schlaefli::init "
				"before make_trihedral_pairs" << endl;
	}
	make_trihedral_pairs(verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after make_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "schlaefli::init "
				"before make_triads" << endl;
	}
	make_triads(verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after make_triads" << endl;
	}

	if (f_v) {
		cout << "schlaefli::init "
				"before process_trihedral_pairs" << endl;
	}
	process_trihedral_pairs(verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after process_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "schlaefli::init "
				"before make_Eckardt_points" << endl;
	}
	make_Eckardt_points(verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after make_Eckardt_points" << endl;
	}

	if (f_v) {
		cout << "schlaefli::init "
				"before init_Trihedral_to_Eckardt" << endl;
	}
	init_Trihedral_to_Eckardt(verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after init_Trihedral_to_Eckardt" << endl;
	}

	if (f_v) {
		cout << "schlaefli::init "
				"before init_collinear_Eckardt_triples" << endl;
	}
	init_collinear_Eckardt_triples(verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after init_collinear_Eckardt_triples" << endl;
	}

	if (f_v) {
		cout << "schlaefli::init "
				"before init_double_sixes" << endl;
	}
	init_double_sixes(verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after init_double_sixes" << endl;
	}

	if (f_v) {
		cout << "schlaefli::init "
				"before create_half_double_sixes" << endl;
	}
	create_half_double_sixes(verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after create_half_double_sixes" << endl;
	}
	//print_half_double_sixes_in_GAP();

	if (f_v) {
		cout << "schlaefli::init "
				"before init_adjacency_matrix_of_lines" << endl;
	}
	init_adjacency_matrix_of_lines(verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after init_adjacency_matrix_of_lines" << endl;
	}

	if (f_v) {
		cout << "schlaefli::init "
				"before init_incidence_matrix_of_lines_vs_tritangent_planes" << endl;
	}
	init_incidence_matrix_of_lines_vs_tritangent_planes(verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after init_incidence_matrix_of_lines_vs_tritangent_planes" << endl;
	}


	if (f_v) {
		cout << "schlaefli::init done" << endl;
	}
}



void schlaefli::find_tritangent_planes_intersecting_in_a_line(
	int line_idx,
	int &plane1, int &plane2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx;
	int three_lines[3];
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "schlaefli::find_tritangent_planes_intersecting_in_a_line" << endl;
	}
	for (plane1 = 0; plane1 < nb_Eckardt_points; plane1++) {

		Eckardt_points[plane1].three_lines(Surf, three_lines);
		if (Sorting.int_vec_search_linear(three_lines, 3, line_idx, idx)) {
			for (plane2 = plane1 + 1;
					plane2 < nb_Eckardt_points;
					plane2++) {

				Eckardt_points[plane2].three_lines(Surf, three_lines);
				if (Sorting.int_vec_search_linear(three_lines, 3, line_idx, idx)) {
					if (f_v) {
						cout << "schlaefli::find_tritangent_planes_"
								"intersecting_in_a_line done" << endl;
						}
					return;
				}
			}
		}
	}
	cout << "schlaefli::find_tritangent_planes_intersecting_in_a_line could not find "
			"two planes" << endl;
	exit(1);
}

void schlaefli::make_triads(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;
	int *Adj;
	int i, j, h, u;

	if (f_v) {
		cout << "schlaefli::make_triads" << endl;
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
			cout << "schlaefli::make_triads u != 3" << endl;
			exit(1);
		}
		h++;
	}
	if (h != 40) {
		cout << "schlaefli::make_triads h != 40" << endl;
		exit(1);
	}
	nb_triads = h;
	if (f_v) {
		cout << "schlaefli::make_triads done" << endl;
	}
}

void schlaefli::make_trihedral_pair_disjointness_graph(
		int *&Adj, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;
	int *T;
	int i, j;

	if (f_v) {
		cout << "schlaefli::make_trihedral_pair_disjointness_graph" << endl;
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
#if 0
	colored_graph *CG;

	CG = NEW_OBJECT(colored_graph);

	CG->init_adjacency_no_colors(nb_trihedral_pairs, Adj,
			verbose_level);

	FREE_int(Adj);
#endif

	if (f_v) {
		cout << "schlaefli::make_trihedral_pair_disjointness_graph done" << endl;
	}
	//return CG;
}

void schlaefli::make_trihedral_pairs(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, s, idx;
	int subset[6];
	int second_subset[6];
	int complement[6];
	int subset_complement[6];
	int size_complement;
	combinatorics::combinatorics_domain Combi;
	l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "schlaefli::make_trihedral_pairs" << endl;
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
		cout << "schlaefli::make_trihedral_pairs idx != 120" << endl;
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
		cout << "schlaefli::make_trihedral_pairs done" << endl;
	}
}

void schlaefli::process_trihedral_pairs(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int subset[3];
	int i, j, h, rk, a;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;
	l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "schlaefli::process_trihedral_pairs" << endl;
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
		cout << "schlaefli::process_trihedral_pairs "
				"The trihedral pairs row sets:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
			Trihedral_pairs_row_sets, 120, 3,
			false /* f_tex */);

		cout << "The trihedral pairs col sets:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
			Trihedral_pairs_col_sets, 120, 3,
			false /* f_tex */);
	}

	Classify_trihedral_pairs_row_values = NEW_OBJECT(data_structures::tally);
	Classify_trihedral_pairs_row_values->init(
		Trihedral_pairs_row_sets, 120 * 3, false, 0);

	if (f_v) {
		cout << "schlaefli::process_trihedral_pairs "
				"sorted row values:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
			Classify_trihedral_pairs_row_values->data_sorted,
			120 * 3 / 10, 10, false /* f_tex */);
	}

	Classify_trihedral_pairs_col_values = NEW_OBJECT(data_structures::tally);
	Classify_trihedral_pairs_col_values->init(
		Trihedral_pairs_col_sets,
		120 * 3, false, 0);

	if (f_v) {
		cout << "schlaefli::process_trihedral_pairs "
				"sorted col values:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
			Classify_trihedral_pairs_col_values->data_sorted,
			120 * 3 / 10, 10, false /* f_tex */);
	}
	if (f_v) {
		cout << "schlaefli::process_trihedral_pairs done" << endl;
	}
}

int schlaefli::line_ai(int i)
{
	if (i >= 6) {
		cout << "schlaefli::line_ai i >= 6" << endl;
		exit(1);
		}
	return i;
}

int schlaefli::line_bi(int i)
{
	if (i >= 6) {
		cout << "schlaefli::line_bi i >= 6" << endl;
		exit(1);
		}
	return 6 + i;
}

int schlaefli::line_cij(int i, int j)
{
	int a;
	combinatorics::combinatorics_domain Combi;

	if (i > j) {
		return line_cij(j, i);
		}
	if (i == j) {
		cout << "schlaefli::line_cij i==j" << endl;
		exit(1);
		}
	if (i >= 6) {
		cout << "schlaefli::line_cij i >= 6" << endl;
		exit(1);
		}
	if (j >= 6) {
		cout << "schlaefli::line_cij j >= 6" << endl;
		exit(1);
		}
	a = Combi.ij2k(i, j, 6);
	return 12 + a;
}

int schlaefli::type_of_line(int line)
// 0 = a_i, 1 = b_i, 2 = c_ij
{
	if (line < 6) {
		return 0;
		}
	else if (line < 12) {
		return 1;
		}
	else if (line < 27) {
		return 2;
		}
	else {
		cout << "schlaefli::type_of_line error" << endl;
		exit(1);
		}
}

void schlaefli::index_of_line(int line, int &i, int &j)
// returns i for a_i, i for b_i and (i,j) for c_ij
{
	int a;
	combinatorics::combinatorics_domain Combi;

	if (line < 6) { // ai
		i = line;
		}
	else if (line < 12) { // bj
		i = line - 6;
		}
	else if (line < 27) { // c_ij
		a = line - 12;
		Combi.k2ij(a, i, j, 6);
		}
	else {
		cout << "schlaefli::index_of_line error" << endl;
		exit(1);
		}
}

int schlaefli::third_line_in_tritangent_plane(
		int l1, int l2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, i, j, k, l, m, n;

	if (f_v) {
		cout << "schlaefli::third_line_in_tritangent_plane" << endl;
	}
	if (l1 > l2) {
		int t = l1;
		l1 = l2;
		l2 = t;
	}
	// now l1 < l2.
	if (l1 < 6) {
		// l1 = ai line
		i = l1;
		if (l2 < 6) {
			cout << "schlaefli::third_line_in_tritangent_plane impossible (1)" << endl;
			exit(1);
		}
		if (l2 < 12) {
			j = l2 - 6;
			return line_cij(i, j);
		}
		else {
			index_of_line(l2, h, k);
			if (h == i) {
				return line_bi(k);
			}
			else if (k == i) {
				return line_bi(h);
			}
			else {
				cout << "schlaefli::third_line_in_tritangent_plane impossible (2)" << endl;
				exit(1);
			}
		}
	}
	else if (l1 < 12) {
		// l1 = bh line
		h = l1 - 6;
		if (l2 < 12) {
			cout << "schlaefli::third_line_in_tritangent_plane impossible (3)" << endl;
			exit(1);
		}
		index_of_line(l2, i, j);
		if (i == h) {
			return line_ai(j);
		}
		else if (h == j) {
			return line_ai(i);
		}
		else {
			cout << "schlaefli::third_line_in_tritangent_plane impossible (4)" << endl;
			exit(1);
		}
	}
	else {
		// now we must be in a tritangent plane c_{ij,kl,mn}
		index_of_line(l1, i, j);
		index_of_line(l2, k, l);

		ijkl2mn(i, j, k, l, m, n);

		return line_cij(m, n);
	}
}

void schlaefli::make_Tijk(int *T, int i, int j, int k)
{
	T[0] = line_cij(j, k);
	T[1] = line_bi(k);
	T[2] = line_ai(j);
	T[3] = line_ai(k);
	T[4] = line_cij(i, k);
	T[5] = line_bi(i);
	T[6] = line_bi(j);
	T[7] = line_ai(i);
	T[8] = line_cij(i, j);
}

void schlaefli::make_Tlmnp(int *T, int l, int m, int n, int p)
{
	int subset[4];
	int complement[2];
	int size_complement;
	int r, s;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	subset[0] = l;
	subset[1] = m;
	subset[2] = n;
	subset[3] = p;
	Sorting.int_vec_heapsort(subset, 4);
	Combi.set_complement(subset, 4, complement, size_complement, 6);
	r = complement[0];
	s = complement[1];

	T[0] = line_ai(l);
	T[1] = line_bi(p);
	T[2] = line_cij(l, p);
	T[3] = line_bi(n);
	T[4] = line_ai(m);
	T[5] = line_cij(m, n);
	T[6] = line_cij(l, n);
	T[7] = line_cij(m, p);
	T[8] = line_cij(r, s);
}

void schlaefli::make_Tdefght(int *T,
		int d, int e, int f, int g, int h, int t)
{
	T[0] = line_cij(d, g);
	T[1] = line_cij(e, h);
	T[2] = line_cij(f, t);
	T[3] = line_cij(e, t);
	T[4] = line_cij(f, g);
	T[5] = line_cij(d, h);
	T[6] = line_cij(f, h);
	T[7] = line_cij(d, t);
	T[8] = line_cij(e, g);
}

void schlaefli::make_Eckardt_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	string str;

	if (f_v) {
		cout << "schlaefli::make_Eckardt_points" << endl;
	}
	nb_Eckardt_points = 45;
	Eckardt_points = NEW_OBJECTS(eckardt_point, nb_Eckardt_points);
	for (i = 0; i < nb_Eckardt_points; i++) {
		Eckardt_points[i].init_by_rank(i);
	}
	Eckard_point_label = new string [nb_Eckardt_points];
	Eckard_point_label_tex = new string [nb_Eckardt_points];
	for (i = 0; i < nb_Eckardt_points; i++) {
		Eckardt_points[i].latex_to_str_without_E(str);
		Eckard_point_label[i].assign(str);
		Eckard_point_label_tex[i].assign(str);
	}
	if (f_v) {
		cout << "schlaefli::make_Eckardt_points done" << endl;
	}
}


void schlaefli::init_Trihedral_to_Eckardt(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t, i, j, rk;
	int tritangent_plane[3];
	l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "schlaefli::init_Trihedral_to_Eckardt" << endl;
	}
	nb_trihedral_to_Eckardt = nb_trihedral_pairs * 6;
	Trihedral_to_Eckardt = NEW_lint(nb_trihedral_to_Eckardt);
	for (t = 0; t < nb_trihedral_pairs; t++) {
		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				tritangent_plane[j] = Trihedral_pairs[t * 9 + i * 3 + j];
				}
			rk = Eckardt_point_from_tritangent_plane(tritangent_plane);
			Trihedral_to_Eckardt[t * 6 + i] = rk;
		}
		for (j = 0; j < 3; j++) {
			for (i = 0; i < 3; i++) {
				tritangent_plane[i] = Trihedral_pairs[t * 9 + i * 3 + j];
			}
			rk = Eckardt_point_from_tritangent_plane(tritangent_plane);
			Trihedral_to_Eckardt[t * 6 + 3 + j] = rk;
		}
	}
	if (f_v) {
		cout << "Trihedral_to_Eckardt:" << endl;
		L.print_lint_matrix_with_standard_labels(cout,
			Trihedral_to_Eckardt, nb_trihedral_pairs, 6,
			false /* f_tex */);
	}
	if (f_v) {
		cout << "schlaefli::init_Trihedral_to_Eckardt done" << endl;
	}
}


int schlaefli::Eckardt_point_from_tritangent_plane(int *tritangent_plane)
{
	int a, b, c, rk;
	eckardt_point E;
	data_structures::sorting Sorting;

	Sorting.int_vec_heapsort(tritangent_plane, 3);
	a = tritangent_plane[0];
	b = tritangent_plane[1];
	c = tritangent_plane[2];
	if (a < 6) {
		E.init2(a, b - 6);
	}
	else {
		if (a < 12) {
			cout << "schlaefli::Eckardt_point_from_tritangent_plane a < 12" << endl;
			exit(1);
		}
		a -= 12;
		b -= 12;
		c -= 12;
		E.init3(a, b, c);
	}
	rk = E.rank();
	return rk;
}


void schlaefli::init_collinear_Eckardt_triples(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t, i, rk, h;
	int subset[3];
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;
	l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "schlaefli::init_collinear_Eckardt_triples" << endl;
	}
	nb_collinear_Eckardt_triples = nb_trihedral_pairs * 2;
	collinear_Eckardt_triples_rank = NEW_int(nb_collinear_Eckardt_triples);
	for (t = 0; t < nb_trihedral_pairs; t++) {
		for (i = 0; i < 2; i++) {
			for (h = 0; h < 3; h++) {
				subset[h] = Trihedral_to_Eckardt[6 * t + i * 3 + h];
			}
			Sorting.int_vec_heapsort(subset, 3);
			rk = Combi.rank_k_subset(subset, nb_Eckardt_points, 3);
			collinear_Eckardt_triples_rank[t * 2 + i] = rk;
		}
	}
	if (f_v) {
		cout << "collinear_Eckardt_triples_rank:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
			collinear_Eckardt_triples_rank, nb_trihedral_pairs, 2,
			false /* f_tex */);
	}

	Classify_collinear_Eckardt_triples = NEW_OBJECT(data_structures::tally);
	Classify_collinear_Eckardt_triples->init(
		collinear_Eckardt_triples_rank, nb_collinear_Eckardt_triples,
		false, 0);

	if (f_v) {
		cout << "schlaefli::init_collinear_Eckardt_triples done" << endl;
	}
}

void schlaefli::find_trihedral_pairs_from_collinear_triples_of_Eckardt_points(
	int *E_idx, int nb_E,
	int *&T_idx, int &nb_T, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nCk, h, k, rk, idx, i, t_idx;
	int subset[3];
	int set[3];
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "schlaefli::find_trihedral_pairs_from_collinear_"
				"triples_of_Eckardt_points" << endl;
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

		rk = Combi.rank_k_subset(set, nb_Eckardt_points, 3);


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
	data_structures::tally C;

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
		//cout << "clebsch::clebsch_map_print_fibers fibers of size "
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
		cout << "schlaefli::find_trihedral_pairs_from_collinear_"
				"triples_of_Eckardt_points done" << endl;
	}
}


void schlaefli::init_double_sixes(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, ij, u, v, l, m, n, h, a, b, c;
	int set[6];
	int size_complement;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "schlaefli::init_double_sixes" << endl;
	}
	Double_six = NEW_lint(36 * 12);
	h = 0;
	// first type: D : a_1,..., a_6; b_1, ..., b_6
	for (i = 0; i < 12; i++) {
		Double_six[h * 12 + i] = i;
	}
	h++;

	// second type:
	// D_{ij} :
	// a_1, b_1, c_23, c_24, c_25, c_26;
	// a_2, b_2, c_13, c_14, c_15, c_16
	for (ij = 0; ij < 15; ij++, h++) {
		//cout << "second type " << ij << " / " << 15 << endl;
		Combi.k2ij(ij, i, j, 6);
		set[0] = i;
		set[1] = j;
		Combi.set_complement(set, 2 /* subset_size */, set + 2,
			size_complement, 6 /* universal_set_size */);
		//cout << "set : ";
		//int_vec_print(cout, set, 6);
		//cout << endl;
		Double_six[h * 12 + 0] = line_ai(i);
		Double_six[h * 12 + 1] = line_bi(i);
		for (u = 0; u < 4; u++) {
			Double_six[h * 12 + 2 + u] = line_cij(j, set[2 + u]);
		}
		Double_six[h * 12 + 6] = line_ai(j);
		Double_six[h * 12 + 7] = line_bi(j);
		for (u = 0; u < 4; u++) {
			Double_six[h * 12 + 8 + u] = line_cij(i, set[2 + u]);
		}
	}

	// third type: D_{ijk} :
	// a_1, a_2, a_3, c_56, c_46, c_45;
	// c_23, c_13, c_12, b_4, b_5, b_6
	for (v = 0; v < 20; v++, h++) {
		//cout << "third type " << v << " / " << 20 << endl;
		Combi.unrank_k_subset(v, set, 6, 3);
		Combi.set_complement(set, 3 /* subset_size */, set + 3,
			size_complement, 6 /* universal_set_size */);
		i = set[0];
		j = set[1];
		k = set[2];
		l = set[3];
		m = set[4];
		n = set[5];
		Double_six[h * 12 + 0] = line_ai(i);
		Double_six[h * 12 + 1] = line_ai(j);
		Double_six[h * 12 + 2] = line_ai(k);
		Double_six[h * 12 + 3] = line_cij(m, n);
		Double_six[h * 12 + 4] = line_cij(l, n);
		Double_six[h * 12 + 5] = line_cij(l, m);
		Double_six[h * 12 + 6] = line_cij(j, k);
		Double_six[h * 12 + 7] = line_cij(i, k);
		Double_six[h * 12 + 8] = line_cij(i, j);
		Double_six[h * 12 + 9] = line_bi(l);
		Double_six[h * 12 + 10] = line_bi(m);
		Double_six[h * 12 + 11] = line_bi(n);
	}

	if (h != 36) {
		cout << "schlaefli::init_double_sixes h != 36" << endl;
		exit(1);
	}

	Double_six_label_tex = new string [36];

	for (i = 0; i < 36; i++) {
		if (i < 1) {
			Double_six_label_tex[i] = "{\\cal D}";
		}
		else if (i < 1 + 15) {
			ij = i - 1;
			Combi.k2ij(ij, a, b, 6);
			set[0] = a;
			set[1] = b;
			Combi.set_complement(set, 2 /* subset_size */, set + 2,
				size_complement, 6 /* universal_set_size */);
			Double_six_label_tex[i] =
					"{\\cal D}_{"
					+ std::to_string(a + 1)
					+ std::to_string(b + 1) + "}";
		}
		else {
			v = i - 16;
			Combi.unrank_k_subset(v, set, 6, 3);
			Combi.set_complement(set, 3 /* subset_size */, set + 3,
				size_complement, 6 /* universal_set_size */);
			a = set[0];
			b = set[1];
			c = set[2];
			Double_six_label_tex[i] =
					"{\\cal D}_{"
					+ std::to_string(a + 1)
					+ std::to_string(b + 1)
					+ std::to_string(c + 1) + "}";
		}
		if (f_v) {
			cout << "creating label " << Double_six_label_tex[i]
				<< " for Double six " << i << endl;
		}
	}

	if (f_v) {
		cout << "schlaefli::init_double_sixes done" << endl;
	}
}

void schlaefli::create_half_double_sixes(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, ij, v, h;
	int set[6];
	int size_complement;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "schlaefli::create_half_double_sixes" << endl;
	}

	Half_double_six_characteristic_vector = NEW_int(72 * 27);
	Half_double_sixes = NEW_lint(72 * 6);
	Half_double_six_to_double_six = NEW_int(72);
	Half_double_six_to_double_six_row = NEW_int(72);

	Double_six_characteristic_vector = NEW_int(36 * 27);
	Int_vec_zero(Double_six_characteristic_vector, 36 * 27);
	for (i = 0; i < 36; i++) {
		for (j = 0; j < 2; j++) {
			for (h = 0; h < 6; h++) {
				a = Double_six[(2 * i + j) * 6 + h];
				Double_six_characteristic_vector[i * 27 + a] = 1;
			}
		}
	}


	Int_vec_zero(Half_double_six_characteristic_vector, 72 * 27);
	for (i = 0; i < 36; i++) {
		for (j = 0; j < 2; j++) {
			for (h = 0; h < 6; h++) {
				a = Double_six[(2 * i + j) * 6 + h];
				Half_double_six_characteristic_vector[(2 * i + j) * 27 + a] = 1;
			}
		}
	}


	Lint_vec_copy(Double_six, Half_double_sixes, 36 * 12);
	for (i = 0; i < 36; i++) {
		for (j = 0; j < 2; j++) {
			Sorting.lint_vec_heapsort(
				Half_double_sixes + (2 * i + j) * 6, 6);
			Half_double_six_to_double_six[2 * i + j] = i;
			Half_double_six_to_double_six_row[2 * i + j] = j;
		}
	}
	Half_double_six_label_tex = new string [72];

	for (i = 0; i < 36; i++) {
		for (j = 0; j < 2; j++) {
			string str;

			if (i < 1) {
				str = "D";
			}
			else if (i < 1 + 15) {
				ij = i - 1;
				Combi.k2ij(ij, a, b, 6);
				set[0] = a;
				set[1] = b;
				Combi.set_complement(set, 2 /* subset_size */,
					set + 2, size_complement,
					6 /* universal_set_size */);
				str = "D_{" + std::to_string(a + 1) + std::to_string(b + 1) + "}";
			}
			else {
				v = i - 16;
				Combi.unrank_k_subset(v, set, 6, 3);
				Combi.set_complement(set, 3 /* subset_size */,
					set + 3, size_complement,
					6 /* universal_set_size */);
				a = set[0];
				b = set[1];
				c = set[2];
				str = "D_{"
						+ std::to_string(a + 1)
						+ std::to_string(b + 1)
						+ std::to_string(c + 1) + "}";
			}


			if (j == 0) {
				str += "^\\top";
			}
			else {
				str += "^\\bot";
			}
			if (f_v) {
				cout << "creating label " << str
					<< " for half double six "
					<< 2 * i + j << endl;
			}
			Half_double_six_label_tex[2 * i + j] = str;
		}
	}

	if (f_v) {
		cout << "schlaefli::create_half_double_sixes done" << endl;
	}
}

int schlaefli::find_half_double_six(long int *half_double_six)
{
	int i;
	data_structures::sorting Sorting;

	Sorting.lint_vec_heapsort(half_double_six, 6);
	for (i = 0; i < 72; i++) {
		if (Sorting.lint_vec_compare(half_double_six,
			Half_double_sixes + i * 6, 6) == 0) {
			return i;
		}
	}
	cout << "schlaefli::find_half_double_six did not find "
			"half double six" << endl;
	exit(1);
}

void schlaefli::ijklm2n(int i, int j, int k, int l, int m, int &n)
{
	int v[6];
	int size_complement;
	combinatorics::combinatorics_domain Combi;

	v[0] = i;
	v[1] = j;
	v[2] = k;
	v[3] = l;
	v[4] = m;
	Combi.set_complement_safe(v, 5, v + 5, size_complement, 6);
	if (size_complement != 1) {
		cout << "schlaefli::ijklm2n size_complement != 1" << endl;
		exit(1);
	}
	n = v[5];
}

void schlaefli::ijkl2mn(int i, int j, int k, int l, int &m, int &n)
{
	int v[6];
	int size_complement;
	combinatorics::combinatorics_domain Combi;

	v[0] = i;
	v[1] = j;
	v[2] = k;
	v[3] = l;
	Combi.set_complement_safe(v, 4, v + 4, size_complement, 6);
	if (size_complement != 2) {
		cout << "schlaefli::ijkl2mn size_complement != 2" << endl;
		exit(1);
	}
	m = v[4];
	n = v[5];
}

void schlaefli::ijk2lmn(int i, int j, int k, int &l, int &m, int &n)
{
	int v[6];
	int size_complement;
	combinatorics::combinatorics_domain Combi;

	v[0] = i;
	v[1] = j;
	v[2] = k;
	cout << "schlaefli::ijk2lmn v=";
	Int_vec_print(cout, v, 3);
	cout << endl;
	Combi.set_complement_safe(v, 3, v + 3, size_complement, 6);
	if (size_complement != 3) {
		cout << "schlaefli::ijk2lmn size_complement != 3" << endl;
		cout << "size_complement=" << size_complement << endl;
		exit(1);
	}
	l = v[3];
	m = v[4];
	n = v[5];
}

void schlaefli::ij2klmn(int i, int j, int &k, int &l, int &m, int &n)
{
	int v[6];
	int size_complement;
	combinatorics::combinatorics_domain Combi;

	v[0] = i;
	v[1] = j;
	Combi.set_complement_safe(v, 2, v + 2, size_complement, 6);
	if (size_complement != 4) {
		cout << "schlaefli::ij2klmn size_complement != 4" << endl;
		exit(1);
	}
	k = v[2];
	l = v[3];
	m = v[4];
	n = v[5];
}

void schlaefli::get_half_double_six_associated_with_Clebsch_map(
	int line1, int line2, int transversal,
	int hds[6],
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t1, t2, t3;
	int i, j, k, l, m, n;
	int i1, j1;
	int null;

	if (f_v) {
		cout << "schlaefli::get_half_double_six_associated_with_Clebsch_map" << endl;
	}

	if (line1 > line2) {
		cout << "schlaefli::get_half_double_six_associated_"
				"with_Clebsch_map line1 > line2" << endl;
		exit(1);
	}
	t1 = type_of_line(line1);
	t2 = type_of_line(line2);
	t3 = type_of_line(transversal);

	if (f_v) {
		cout << "t1=" << t1 << " t2=" << t2 << " t3=" << t3 << endl;
	}
	if (t1 == 0 && t2 == 0) { // ai and aj:
		index_of_line(line1, i, null);
		index_of_line(line2, j, null);
		if (t3 == 1) { // bk
			index_of_line(transversal, k, null);
			//cout << "i=" << i << " j=" << j << " k=" << k <<< endl;
			ijk2lmn(i, j, k, l, m, n);
			// bl, bm, bn, cij, cik, cjk
			hds[0] = line_bi(l);
			hds[1] = line_bi(m);
			hds[2] = line_bi(n);
			hds[3] = line_cij(i, j);
			hds[4] = line_cij(i, k);
			hds[5] = line_cij(j, k);
		}
		else if (t3 == 2) { // cij
			index_of_line(transversal, i1, j1);
				// test whether {i1,j1} =  {i,j}
			if ((i == i1 && j == j1) || (i == j1 && j == i1)) {
				ij2klmn(i, j, k, l, m, n);
				// bi, bj, bk, bl, bm, bn
				hds[0] = line_bi(i);
				hds[1] = line_bi(j);
				hds[2] = line_bi(k);
				hds[3] = line_bi(l);
				hds[4] = line_bi(m);
				hds[5] = line_bi(n);
			}
			else {
				cout << "schlaefli::get_half_doble_six_associated_"
						"with_Clebsch_map not {i,j} = {i1,j1}" << endl;
				exit(1);
			}
		}
	}
	else if (t1 == 1 && t2 == 1) { // bi and bj:
		index_of_line(line1, i, null);
		index_of_line(line2, j, null);
		if (t3 == 0) { // ak
			index_of_line(transversal, k, null);
			ijk2lmn(i, j, k, l, m, n);
			// al, am, an, cij, cik, cjk
			hds[0] = line_ai(l);
			hds[1] = line_ai(m);
			hds[2] = line_ai(n);
			hds[3] = line_cij(i, j);
			hds[4] = line_cij(i, k);
			hds[5] = line_cij(j, k);
		}
		else if (t3 == 2) { // cij
			index_of_line(transversal, i1, j1);
			if ((i == i1 && j == j1) || (i == j1 && j == i1)) {
				ij2klmn(i, j, k, l, m, n);
				// ai, aj, ak, al, am, an
				hds[0] = line_ai(i);
				hds[1] = line_ai(j);
				hds[2] = line_ai(k);
				hds[3] = line_ai(l);
				hds[4] = line_ai(m);
				hds[5] = line_ai(n);
			}
			else {
				cout << "schlaefli::get_half_doble_six_associated_"
						"with_Clebsch_map not {i,j} = {i1,j1}" << endl;
				exit(1);
			}
		}
	}
	else if (t1 == 0 && t2 == 1) { // ai and bi:
		index_of_line(line1, i, null);
		index_of_line(line2, j, null);
		if (j != i) {
			cout << "schlaefli::get_half_double_six_associated_"
					"with_Clebsch_map j != i" << endl;
			exit(1);
		}
		if (t3 != 2) {
			cout << "schlaefli::get_half_double_six_associated_"
					"with_Clebsch_map t3 != 2" << endl;
			exit(1);
		}
		index_of_line(transversal, i1, j1);
		if (i1 == i) {
			j = j1;
		}
		else {
			j = i1;
		}
		ij2klmn(i, j, k, l, m, n);
		// cik, cil, cim, cin, aj, bj
		hds[0] = line_cij(i, k);
		hds[1] = line_cij(i, l);
		hds[2] = line_cij(i, m);
		hds[3] = line_cij(i, n);
		hds[4] = line_ai(j);
		hds[5] = line_bi(j);
	}
	else if (t1 == 1 && t2 == 2) { // bi and cjk:
		index_of_line(line1, i, null);
		index_of_line(line2, j, k);
		if (t3 == 2) { // cil
			index_of_line(transversal, i1, j1);
			if (i1 == i) {
				l = j1;
			}
			else if (j1 == i) {
				l = i1;
			}
			else {
				cout << "schlaefli::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
			}
			ijkl2mn(i, j, k, l, m, n);
			// cin, cim, aj, ak, al, cnm
			hds[0] = line_cij(i, n);
			hds[1] = line_cij(i, m);
			hds[2] = line_ai(j);
			hds[3] = line_ai(k);
			hds[4] = line_ai(l);
			hds[5] = line_cij(n, m);
		}
		else if (t3 == 0) { // aj
			index_of_line(transversal, j1, null);
			if (j1 == k) {
				// swap k and j
				int tmp = k;
				k = j;
				j = tmp;
			}
			if (j1 != j) {
				cout << "schlaefli::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
			}
			ijk2lmn(i, j, k, l, m, n);
			// ak, cil, cim, cin, bk, cij
			hds[0] = line_ai(k);
			hds[1] = line_cij(i, l);
			hds[2] = line_cij(i, m);
			hds[3] = line_cij(i, n);
			hds[4] = line_bi(k);
			hds[5] = line_cij(i, j);
		}
	}
	else if (t1 == 0 && t2 == 2) { // ai and cjk:
		index_of_line(line1, i, null);
		index_of_line(line2, j, k);
		if (t3 == 2) { // cil
			index_of_line(transversal, i1, j1);
			if (i1 == i) {
				l = j1;
			}
			else if (j1 == i) {
				l = i1;
			}
			else {
				cout << "schlaefli::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
			}
			ijkl2mn(i, j, k, l, m, n);
			// cin, cim, bj, bk, bl, cnm
			hds[0] = line_cij(i, n);
			hds[1] = line_cij(i, m);
			hds[2] = line_bi(j);
			hds[3] = line_bi(k);
			hds[4] = line_bi(l);
			hds[5] = line_cij(n, m);
		}
		else if (t3 == 1) { // bj
			index_of_line(transversal, j1, null);
			if (j1 == k) {
				// swap k and j
				int tmp = k;
				k = j;
				j = tmp;
			}
			if (j1 != j) {
				cout << "schlaefli::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
			}
			ijk2lmn(i, j, k, l, m, n);
			// bk, cil, cim, cin, ak, cij
			hds[0] = line_bi(k);
			hds[1] = line_cij(i, l);
			hds[2] = line_cij(i, m);
			hds[3] = line_cij(i, n);
			hds[4] = line_ai(k);
			hds[5] = line_cij(i, j);
		}
	}
	else if (t1 == 2 && t2 == 2) { // cij and cik:
		index_of_line(line1, i, j);
		index_of_line(line2, i1, j1);
		if (i == i1) {
			k = j1;
		}
		else if (i == j1) {
			k = i1;
		}
		else if (j == i1) {
			j = i;
			i = i1;
			k = j1;
		}
		else if (j == j1) {
			j = i;
			i = j1;
			k = i1;
		}
		else {
			cout << "schlaefli::get_half_double_six_associated_"
					"with_Clebsch_map error" << endl;
			exit(1);
		}
		if (t3 == 0) { // ai
			index_of_line(transversal, i1, null);
			if (i1 != i) {
				cout << "schlaefli::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
			}
			ijk2lmn(i, j, k, l, m, n);
			// bi, clm, cnm, cln, bj, bk
			hds[0] = line_bi(i);
			hds[1] = line_cij(l, m);
			hds[2] = line_cij(n, m);
			hds[3] = line_cij(l, n);
			hds[4] = line_bi(j);
			hds[5] = line_bi(k);
		}
		else if (t3 == 1) { // bi
			index_of_line(transversal, i1, null);
			if (i1 != i) {
				cout << "schlaefli::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
			}
			ijk2lmn(i, j, k, l, m, n);
			// ai, clm, cnm, cln, aj, ak
			hds[0] = line_ai(i);
			hds[1] = line_cij(l, m);
			hds[2] = line_cij(n, m);
			hds[3] = line_cij(l, n);
			hds[4] = line_ai(j);
			hds[5] = line_ai(k);
		}
		else if (t3 == 2) { // clm
			index_of_line(transversal, l, m);
			ijklm2n(i, j, k, l, m, n);
			// ai, bi, cmn, cln, ckn, cjn
			hds[0] = line_ai(i);
			hds[1] = line_bi(i);
			hds[2] = line_cij(m, n);
			hds[3] = line_cij(l, n);
			hds[4] = line_cij(k, n);
			hds[5] = line_cij(j, n);
		}
	}
	else {
		cout << "schlaefli::get_half_double_six_associated_"
				"with_Clebsch_map error" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "schlaefli::get_half_double_six_associated_with_Clebsch_map done" << endl;
	}
}

void schlaefli::prepare_clebsch_map(
		int ds, int ds_row,
	int &line1, int &line2, int &transversal,
	int verbose_level)
{
	int ij, i, j, k, l, m, n, size_complement;
	int set[6];
	combinatorics::combinatorics_domain Combi;

	if (ds == 0) {
		if (ds_row == 0) {
			line1 = line_bi(0);
			line2 = line_bi(1);
			transversal = line_cij(0, 1);
			return;
		}
		else {
			line1 = line_ai(0);
			line2 = line_ai(1);
			transversal = line_cij(0, 1);
			return;
		}
	}
	ds--;
	if (ds < 15) {
		ij = ds;
		Combi.k2ij(ij, i, j, 6);

		if (ds_row == 0) {
			line1 = line_ai(j);
			line2 = line_bi(j);
			transversal = line_cij(i, j);
			return;
		}
		else {
			line1 = line_ai(i);
			line2 = line_bi(i);
			transversal = line_cij(i, j);
			return;
		}
	}
	ds -= 15;
	Combi.unrank_k_subset(ds, set, 6, 3);
	Combi.set_complement(set, 3 /* subset_size */, set + 3,
		size_complement, 6 /* universal_set_size */);
	i = set[0];
	j = set[1];
	k = set[2];
	l = set[3];
	m = set[4];
	n = set[5];
	if (ds_row == 0) {
		line1 = line_bi(l);
		line2 = line_bi(m);
		transversal = line_ai(n);
		return;
	}
	else {
		line1 = line_ai(i);
		line2 = line_ai(j);
		transversal = line_bi(k);
		return;
	}
}

void schlaefli::init_adjacency_matrix_of_lines(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, l;

	if (f_v) {
		cout << "schlaefli::init_adjacency_matrix_of_lines" << endl;
	}

	adjacency_matrix_of_lines = NEW_int(27 * 27);
	Int_vec_zero(adjacency_matrix_of_lines, 27 * 27);

	// the ai lines:
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 6; j++) {
			if (j == i) {
				continue;
			}
			set_adjacency_matrix_of_lines(line_ai(i), line_bi(j));
		}
		for (k = 0; k < 6; k++) {
			if (k == i) {
				continue;
			}
			set_adjacency_matrix_of_lines(line_ai(i), line_cij(i, k));
		}
	}


	// the bi lines:
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 6; j++) {
			if (j == i) {
				continue;
			}
			set_adjacency_matrix_of_lines(line_bi(i), line_ai(j));
		}
		for (k = 0; k < 6; k++) {
			if (k == i) {
				continue;
			}
			set_adjacency_matrix_of_lines(line_bi(i), line_cij(i, k));
		}
	}




	// the cij lines:
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 6; j++) {
			if (j == i) {
				continue;
			}
			for (k = 0; k < 6; k++) {
				if (k == i) {
					continue;
				}
				if (k == j) {
					continue;
				}
				for (l = 0; l < 6; l++) {
					if (l == i) {
						continue;
					}
					if (l == j) {
						continue;
					}
					if (k == l) {
						continue;
					}
					set_adjacency_matrix_of_lines(
							line_cij(i, j), line_cij(k, l));
				} // next l
			} // next k
		} // next j
	} // next i

	int r, c;

	for (i = 0; i < 27; i++) {
		r = 0;
		for (j = 0; j < 27; j++) {
			if (get_adjacency_matrix_of_lines(i, j)) {
				r++;
			}
		}
		if (r != 10) {
			cout << "schlaefli::init_adjacency_matrix_of_lines "
					"row sum r != 10, r = " << r << " in row " << i << endl;
		}
	}

	for (j = 0; j < 27; j++) {
		c = 0;
		for (i = 0; i < 27; i++) {
			if (get_adjacency_matrix_of_lines(i, j)) {
				c++;
			}
		}
		if (c != 10) {
			cout << "schlaefli::init_adjacency_matrix_of_lines "
					"col sum c != 10, c = " << c << " in col " << j << endl;
		}
	}

	if (f_v) {
		cout << "schlaefli::init_adjacency_matrix_of_lines done" << endl;
		}
}

void schlaefli::init_incidence_matrix_of_lines_vs_tritangent_planes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h;
	int three_lines[3];

	if (f_v) {
		cout << "schlaefli::init_incidence_matrix_of_lines_vs_tritangent_planes" << endl;
	}

	incidence_lines_vs_tritangent_planes = NEW_int(27 * 45);
	Int_vec_zero(incidence_lines_vs_tritangent_planes, 27 * 45);


	Lines_in_tritangent_planes = NEW_lint(45 * 3);
	Lint_vec_zero(Lines_in_tritangent_planes, 45 * 3);

	for (j = 0; j < nb_Eckardt_points; j++) {
		eckardt_point *E;

		E = Eckardt_points + j;
		E->three_lines(Surf, three_lines);
		for (h = 0; h < 3; h++) {
			Lines_in_tritangent_planes[j * 3 + h] = three_lines[h];
				// conversion to long int
		}
		for (h = 0; h < 3; h++) {
			i = three_lines[h];
			incidence_lines_vs_tritangent_planes[i * 45 + j] = 1;
		}
	}



	if (f_v) {
		cout << "schlaefli::init_incidence_matrix_of_lines_vs_tritangent_planes done" << endl;
	}
}

void schlaefli::set_adjacency_matrix_of_lines(int i, int j)
{
	adjacency_matrix_of_lines[i * 27 + j] = 1;
	adjacency_matrix_of_lines[j * 27 + i] = 1;
}

int schlaefli::get_adjacency_matrix_of_lines(int i, int j)
{
	return adjacency_matrix_of_lines[i * 27 + j];
}

int schlaefli::choose_tritangent_plane_for_Clebsch_map(
		int line_a, int line_b,
			int transversal_line, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j, nb;
	int planes[45];

	if (f_v) {
		cout << "schlaefli::choose_tritangent_plane_for_Clebsch_map" << endl;
	}

	nb = 0;
	for (j = 0; j < 45; j++) {
		if (incidence_lines_vs_tritangent_planes[line_a * 45 + j] == 0 &&
				incidence_lines_vs_tritangent_planes[line_b * 45 + j] == 0 &&
				incidence_lines_vs_tritangent_planes[transversal_line * 45 + j]) {
			planes[nb++] = j;
		}
	}
	if (nb != 3) {
		cout << "schlaefli::choose_tritangent_plane_for_Clebsch_map nb != 3" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "schlaefli::choose_tritangent_plane_for_Clebsch_map done" << endl;
	}
	return planes[0];
}

void schlaefli::print_Steiner_and_Eckardt(std::ostream &ost)
{
	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Eckardt Points}" << endl;
	latex_table_of_Eckardt_points(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Double Sixes}" << endl;
	latex_table_of_double_sixes(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Half Double Sixes}" << endl;
	latex_table_of_half_double_sixes(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Tritangent Planes}" << endl;
	latex_table_of_tritangent_planes(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Steiner Trihedral Pairs}" << endl;
	latex_table_of_trihedral_pairs(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Triads}" << endl;
	latex_triads(ost);

}

void schlaefli::latex_abstract_trihedral_pair(std::ostream &ost, int t_idx)
{
	latex_trihedral_pair(ost, Trihedral_pairs + t_idx * 9,
		Trihedral_to_Eckardt + t_idx * 6);
}

void schlaefli::latex_table_of_Schlaefli_labeling_of_lines(std::ostream &ost)
{
	int i;

	ost << "\\begin{multicols}{5}" << endl;
	ost << "\\noindent";
	for (i = 0; i < 27; i++) {
		ost << "$" << i << " = ";
		print_line(ost, i);
		ost << "$\\\\" << endl;
	}
	ost << "\\end{multicols}" << endl;
}

void schlaefli::latex_trihedral_pair(
		std::ostream &ost, int *T, long int *TE)
{
	int i, j;

	ost << "\\begin{array}{*{" << 3 << "}{c}|c}" << endl;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			print_line(ost, T[i * 3 + j]);
			ost << " & ";
			}
		ost << "\\pi_{";
		Eckardt_points[TE[i]].latex_index_only(ost);
		ost << "}\\\\" << endl;
		}
	ost << "\\hline" << endl;
	for (j = 0; j < 3; j++) {
		ost << "\\pi_{";
		Eckardt_points[TE[3 + j]].latex_index_only(ost);
		ost << "} & ";
		}
	ost << "\\\\" << endl;
	ost << "\\end{array}" << endl;
}

void schlaefli::latex_table_of_trihedral_pairs(std::ostream &ost)
{
	int i;

	cout << "schlaefli::latex_table_of_trihedral_pairs" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_trihedral_pairs; i++) {
		ost << "$T_{" << i << "} = T_{"
			<< Trihedral_pair_labels[i] << "} = \\\\" << endl;
		//ost << "\\left[" << endl;
		//ost << "\\begin{array}" << endl;
		latex_trihedral_pair(ost, Trihedral_pairs + i * 9,
			Trihedral_to_Eckardt + i * 6);
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

	cout << "schlaefli::latex_table_of_trihedral_pairs done" << endl;
}

void schlaefli::latex_triads(std::ostream &ost)
{
	l1_interfaces::latex_interface L;
	int i, j, a;

	cout << "schlaefli::latex_triads" << endl;

	ost << "\\subsection*{Triads}" << endl;
	ost << "The 40 triads are:\\\\" << endl;

	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
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

	cout << "schlaefli::latex_triads done" << endl;
}

void schlaefli::print_trihedral_pairs(std::ostream &ost)
{
	l1_interfaces::latex_interface L;
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
	long int *p = Trihedral_to_Eckardt;

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
			ost << " & \\pi_{" << Eckard_point_label_tex[a] << "}";
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
	p = Trihedral_to_Eckardt + 40 * 6;

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
			ost << " & \\pi_{" << Eckard_point_label_tex[a] << "}";
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
	p = Trihedral_to_Eckardt + 80 * 6;

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
			ost << " & \\pi_{" << Eckard_point_label_tex[a] << "}";
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

#if 0
void schlaefli::print_trihedral_pairs(std::ostream &ost)
{
	latex_interface L;
	int i, j;

	ost << "List of trihedral pairs:\\\\" << endl;
	for (i = 0; i < nb_trihedral_pairs; i++) {
		ost << i << " / " << nb_trihedral_pairs
			<< ": $T_{" << i << "} =  T_{"
			<< Trihedral_pair_labels[i] << "}=(";
		for (j = 0; j < 6; j++) {
			ost << "\\pi_{" << Trihedral_to_Eckardt[i * 6 + j]
				<< "}";
			if (j == 2) {
				ost << "; ";
				}
			else if (j < 6 - 1) {
				ost << ", ";
				}
			}
		ost << ")$\\\\" << endl;
		}
	ost << "List of trihedral pairs numerically:\\\\" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		Trihedral_to_Eckardt, 40, 6, 0, 0, true /* f_tex*/);
	ost << "\\;";
	//ost << "$$" << endl;
	//ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		Trihedral_to_Eckardt + 40 * 6, 40, 6, 40, 0, true /* f_tex*/);
	ost << "\\;";
	//ost << "$$" << endl;
	//ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		Trihedral_to_Eckardt + 80 * 6, 40, 6, 80, 0, true /* f_tex*/);
	ost << "$$" << endl;
}
#endif

void schlaefli::latex_table_of_double_sixes(std::ostream &ost)
{
	int h;

	//cout << "schlaefli::latex_table_of_double_sixes" << endl;



	//ost << "\\begin{multicols}{2}" << endl;
	for (h = 0; h < 36; h++) {

		ost << "$D_{" << h << "} = " << Double_six_label_tex[h] << endl;

		ost << " = " << endl;


		latex_double_six_symbolic(ost, h);

		ost << " = " << endl;


		latex_double_six_index_set(ost, h);

		ost << "$\\\\" << endl;
		}
	//ost << "\\end{multicols}" << endl;

	//cout << "schlaefli::latex_table_of_double_sixes done" << endl;

}


void schlaefli::latex_double_six_symbolic(std::ostream &ost, int idx)
{
	int i, j;
	long int D[12];

	Lint_vec_copy(Double_six + idx * 12, D, 12);


	ost << "\\left[";
	ost << "\\begin{array}{cccccc}" << endl;
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 6; j++) {
			ost << Labels->Line_label_tex[D[i * 6 + j]];
			if (j < 6 - 1) {
				ost << " & ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
}

void schlaefli::latex_double_six_index_set(std::ostream &ost, int idx)
{
	int i, j;
	long int D[12];

	Lint_vec_copy(Double_six + idx * 12, D, 12);


	ost << "\\left[";
	ost << "\\begin{array}{cccccc}" << endl;
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 6; j++) {
			ost << D[i * 6 + j];
			if (j < 6 - 1) {
				ost << " & ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
}


void schlaefli::latex_table_of_half_double_sixes(std::ostream &ost)
{
	int i;

	//cout << "schlaefli::latex_table_of_half_double_sixes" << endl;



	//ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < 72; i++) {


		ost << "$" << endl;

		latex_half_double_six(ost, i);

		ost << "$\\\\" << endl;

	}
	//ost << "\\end{multicols}" << endl;



	//cout << "schlaefli::latex_table_of_double_sixes done" << endl;
}

void schlaefli::latex_half_double_six(std::ostream &ost, int idx)
{
	int j;
	long int H[6];

	//cout << "schlaefli::latex_table_of_half_double_sixes" << endl;




	Lint_vec_copy(Half_double_sixes + idx * 6, H, 6);

	ost << "H_{" << idx << "} = " << Half_double_six_label_tex[idx] << endl;

	ost << " = \\{";
	for (j = 0; j < 6; j++) {
		ost << Labels->Line_label_tex[H[j]];
		if (j < 6 - 1) {
			ost << ", ";
			}
		}
	ost << "\\}";

	ost << "= \\{";

	for (j = 0; j < 6; j++) {
		ost << H[j];
		if (j < 6 - 1) {
			ost << ", ";
		}
	}
	ost << "\\}";




	//cout << "schlaefli::latex_table_of_double_sixes done" << endl;
}



void schlaefli::latex_table_of_Eckardt_points(std::ostream &ost)
{
	int i, j;
	int three_lines[3];

	//cout << "schlaefli::latex_table_of_Eckardt_points" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_Eckardt_points; i++) {

		Eckardt_points[i].three_lines(Surf, three_lines);

		ost << "$E_{" << i << "} = " << endl;
		Eckardt_points[i].latex(ost);
		ost << " = ";
		for (j = 0; j < 3; j++) {
			ost << Labels->Line_label_tex[three_lines[j]];
			if (j < 3 - 1) {
				ost << " \\cap ";
				}
			}
		ost << "$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;
	//cout << "schlaefli::latex_table_of_Eckardt_points done" << endl;
}

void schlaefli::latex_table_of_tritangent_planes(std::ostream &ost)
{
	int i, j;
	int three_lines[3];

	//cout << "schlaefli::latex_table_of_tritangent_planes" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_Eckardt_points; i++) {

		Eckardt_points[i].three_lines(Surf, three_lines);

		ost << "$\\pi_{" << i << "} = \\pi_{" << endl;
		Eckardt_points[i].latex_index_only(ost);
		ost << "} = ";
		for (j = 0; j < 3; j++) {
			ost << Labels->Line_label_tex[three_lines[j]];
			}
		ost << "$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;
	//cout << "schlaefli::latex_table_of_tritangent_planes done" << endl;
}

void schlaefli::print_line(std::ostream &ost, int rk)
{
	combinatorics::combinatorics_domain Combi;

	if (rk < 6) {
		ost << "a_" << rk + 1 << endl;
		}
	else if (rk < 12) {
		ost << "b_" << rk - 6 + 1 << endl;
		}
	else {
		int i, j;

		rk -= 12;
		Combi.k2ij(rk, i, j, 6);
		ost << "c_{" << i + 1 << j + 1 << "}";
		}
}

void schlaefli::print_Schlaefli_labelling(std::ostream &ost)
{
	int j, h;

	ost << "The Schlaefli labeling of lines:\\\\" << endl;
	ost << "$$" << endl;
	for (j = 0; j < 3; j++) {
		ost << "\\begin{array}{|r|r|}" << endl;
		ost << "\\hline" << endl;
		ost << "h &  \\mbox{line} \\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\hline" << endl;
		for (h = 0; h < 9; h++) {
			ost << j * 9 + h << " & "
				<< Labels->Line_label_tex[j * 9 + h] << "\\\\" << endl;
			}
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		if (j < 3 - 1) {
			ost << "\\qquad" << endl;
			}
		}
	ost << "$$" << endl;
}

void schlaefli::print_set_of_lines_tex(
		std::ostream &ost, long int *v, int len)
{
	int i;

	ost << "\\{";
	for (i = 0; i < len; i++) {
		ost << Labels->Line_label_tex[v[i]];
		if (i < len - 1) {
			ost << ", ";
			}
		}
	ost << "\\}";
}

void schlaefli::latex_table_of_clebsch_maps(std::ostream &ost)
{
	int e, line, j, l1, l2, t1, t2, t3, t4, c1, c2, cnt;
	int three_lines[3];
	int transversal_line;
	//int intersecting_lines[10];

	cnt = 0;
	//cout << "schlaefli::latex_table_of_clebsch_maps" << endl;
	//ost << "\\begin{multicols}{2}" << endl;
	for (e = 0; e < nb_Eckardt_points; e++) {

		Eckardt_points[e].three_lines(Surf, three_lines);

		for (line = 0; line < 3; line++) {

			transversal_line = three_lines[line];
			if (line == 0) {
				c1 = three_lines[1];
				c2 = three_lines[2];
			}
			else if (line == 1) {
				c1 = three_lines[0];
				c2 = three_lines[2];
			}
			else if (line == 2) {
				c1 = three_lines[0];
				c2 = three_lines[1];
			}

			for (l1 = 0; l1 < 27; l1++) {
				if (l1 == c1 || l1 == c2) {
					continue;
				}
				if (get_adjacency_matrix_of_lines(
						transversal_line, l1) == 0) {
					continue;
				}
				for (l2 = l1 + 1; l2 < 27; l2++) {
					if (l2 == c1 || l2 == c2) {
						continue;
					}
					if (get_adjacency_matrix_of_lines(
							transversal_line, l2) == 0) {
						continue;
					}



					cout << "e=" << e << endl;
					cout << "transversal_line=" << transversal_line << endl;
					cout << "c1=" << c1 << endl;
					cout << "c2=" << c2 << endl;
					cout << "l1=" << l1 << endl;
					cout << "l2=" << l2 << endl;

					for (t1 = 0; t1 < 27; t1++) {
						if (t1 == three_lines[0] ||
								t1 == three_lines[1] ||
								t1 == three_lines[2]) {
							continue;
						}
						if (t1 == l1 || t1 == l2) {
							continue;
						}
						if (get_adjacency_matrix_of_lines(l1, t1) == 0 ||
								get_adjacency_matrix_of_lines(l2, t1) == 0) {
							continue;
						}
						cout << "t1=" << t1 << endl;

						for (t2 = t1 + 1; t2 < 27; t2++) {
							if (t2 == three_lines[0] ||
									t2 == three_lines[1] ||
									t2 == three_lines[2]) {
								continue;
							}
							if (t2 == l1 || t2 == l2) {
								continue;
							}
							if (get_adjacency_matrix_of_lines(l1, t2) == 0 ||
									get_adjacency_matrix_of_lines(l2, t2) == 0) {
								continue;
							}
							cout << "t2=" << t2 << endl;

							for (t3 = t2 + 1; t3 < 27; t3++) {
								if (t3 == three_lines[0] ||
										t3 == three_lines[1] ||
										t3 == three_lines[2]) {
									continue;
								}
								if (t3 == l1 || t3 == l2) {
									continue;
								}
								if (get_adjacency_matrix_of_lines(l1, t3) == 0 ||
										get_adjacency_matrix_of_lines(l2, t3) == 0) {
									continue;
								}
								cout << "t3=" << t3 << endl;

								for (t4 = t3 + 1; t4 < 27; t4++) {
									if (t4 == three_lines[0] ||
											t4 == three_lines[1] ||
											t4 == three_lines[2]) {
										continue;
									}
									if (t4 == l1 || t4 == l2) {
										continue;
									}
									if (get_adjacency_matrix_of_lines(l1, t4) == 0 ||
											get_adjacency_matrix_of_lines(l2, t4) == 0) {
										continue;
									}
									cout << "t4=" << t4 << endl;


									int tc1[4], tc2[4];
									int n1 = 0, n2 = 0;

									if (get_adjacency_matrix_of_lines(t1, c1)) {
										tc1[n1++] = t1;
									}
									if (get_adjacency_matrix_of_lines(t1, c2)) {
										tc2[n2++] = t1;
									}
									if (get_adjacency_matrix_of_lines(t2, c1)) {
										tc1[n1++] = t2;
									}
									if (get_adjacency_matrix_of_lines(t2, c2)) {
										tc2[n2++] = t2;
									}
									if (get_adjacency_matrix_of_lines(t3, c1)) {
										tc1[n1++] = t3;
									}
									if (get_adjacency_matrix_of_lines(t3, c2)) {
										tc2[n2++] = t3;
									}
									if (get_adjacency_matrix_of_lines(t4, c1)) {
										tc1[n1++] = t4;
									}
									if (get_adjacency_matrix_of_lines(t4, c2)) {
										tc2[n2++] = t4;
									}
									cout << "n1=" << n1 << endl;
									cout << "n2=" << n2 << endl;

									ost << cnt << " : $\\pi_{" << e << "} = \\pi_{";
									Eckardt_points[e].latex_index_only(ost);
									ost << "}$, $\\;$ ";

#if 0
									ost << " = ";
									for (j = 0; j < 3; j++) {
										ost << Line_label_tex[three_lines[j]];
										}
									ost << "$, $\\;$ " << endl;
#endif

									ost << "$" << Labels->Line_label_tex[transversal_line] << "$, $\\;$ ";
									//ost << "$(" << Line_label_tex[c1] << ", " << Line_label_tex[c2];
									//ost << ")$, $\\;$ ";

									ost << "$(" << Labels->Line_label_tex[l1] << "," << Labels->Line_label_tex[l2] << ")$, $\\;$ ";
#if 0
									ost << "$(" << Line_label_tex[t1]
										<< "," << Line_label_tex[t2]
										<< "," << Line_label_tex[t3]
										<< "," << Line_label_tex[t4]
										<< ")$, $\\;$ ";
#endif
									ost << "$"
											<< Labels->Line_label_tex[c1] << " \\cap \\{";
									for (j = 0; j < n1; j++) {
										ost << Labels->Line_label_tex[tc1[j]];
										if (j < n1 - 1) {
											ost << ", ";
										}
									}
									ost << "\\}$ ";
									ost << "$"
											<< Labels->Line_label_tex[c2] << " \\cap \\{";
									for (j = 0; j < n2; j++) {
										ost << Labels->Line_label_tex[tc2[j]];
										if (j < n2 - 1) {
											ost << ", ";
										}
									}
									ost << "\\}$ ";
									ost << "\\\\" << endl;
									cnt++;

								} // next t4
							} // next t3
						} // next t2
					} // next t1
					//ost << "\\hline" << endl;
				} // next l2
			} // next l1

		} // line
		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;
	} // e
	//ost << "\\end{multicols}" << endl;
	//cout << "schlaefli::latex_table_of_clebsch_maps done" << endl;
}

void schlaefli::print_half_double_sixes_in_GAP()
{
	int i, j;

	cout << "[";
	for (i = 0; i < 72; i++) {
		cout << "[";
		for (j = 0; j < 6; j++) {
			cout << Half_double_sixes[i * 6 + j] + 1;
			if (j < 6 - 1) {
				cout << ", ";
			}
		}
		cout << "]";
		if (i < 72 - 1) {
			cout << "," << endl;
		}
	}
	cout << "];" << endl;
}

int schlaefli::identify_Eckardt_point(
		int line1, int line2, int line3, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int lines[3];
	data_structures::sorting Sorting;
	combinatorics::combinatorics_domain Combi;
	int idx;

	if (f_v) {
		cout << "schlaefli::identify_Eckardt_point" << endl;
	}
	lines[0] = line1;
	lines[1] = line2;
	lines[2] = line3;
	Sorting.int_vec_heapsort(lines, 3);
	line1 = lines[0];
	line2 = lines[1];
	line3 = lines[2];
	if (line1 < 6) {
		if (line2 < 6) {
			cout << "schlaefli::identify_Eckardt_point "
					"line1 < 6 and line2 < 6" << endl;
			exit(1);
		}
		idx = Combi.ordered_pair_rank(line1, line2 - 6, 6);
	}
	else {
		int i, j, k, l, m, n;

		if (line1 < 12) {
			cout << "schlaefli::identify_Eckardt_point "
					"second case, line1 < 12" << endl;
			exit(1);
		}
		if (line2 < 12) {
			cout << "schlaefli::identify_Eckardt_point "
					"second case, line2 < 12" << endl;
			exit(1);
		}
		if (line3 < 12) {
			cout << "schlaefli::identify_Eckardt_point "
					"second case, line3 < 12" << endl;
			exit(1);
		}
		Combi.k2ij(line1 - 12, i, j, 6);
		Combi.k2ij(line2 - 12, k, l, 6);
		Combi.k2ij(line3 - 12, m, n, 6);
		idx = 30 + Combi.unordered_triple_pair_rank(i, j, k, l, m, n);
	}
	if (f_v) {
		cout << "schlaefli::identify_Eckardt_point done" << endl;
	}
	return idx;
}

void schlaefli::write_lines_vs_line(
		std::string &prefix, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schlaefli::write_lines_vs_line" << endl;
	}

	orbiter_kernel_system::file_io Fio;
	string fname;

	fname = prefix + "_lines_vs_lines_incma.csv";

	Fio.Csv_file_support->int_matrix_write_csv(fname, adjacency_matrix_of_lines,
			27, 27);


	if (f_v) {
		cout << "schlaefli::write_lines_vs_line done" << endl;
	}

}
void schlaefli::write_lines_vs_tritangent_planes(
		std::string &prefix, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schlaefli::write_lines_vs_tritangent_planes" << endl;
	}

	orbiter_kernel_system::file_io Fio;
	string fname;

	fname = prefix + "_lines_tritplanes_incma.csv";

	Fio.Csv_file_support->int_matrix_write_csv(fname, incidence_lines_vs_tritangent_planes,
			27, 45);

	fname = prefix + "_lines_tritplanes.csv";

	Fio.Csv_file_support->lint_matrix_write_csv(fname, Lines_in_tritangent_planes,
			45, 3);

	if (f_v) {
		cout << "schlaefli::write_lines_vs_tritangent_planes done" << endl;
	}
}

void schlaefli::write_double_sixes(
		std::string &prefix, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schlaefli::write_double_sixes" << endl;
	}

	orbiter_kernel_system::file_io Fio;
	string fname;

	fname = prefix + "_single_sixes_char_vec.csv";

	Fio.Csv_file_support->int_matrix_write_csv(fname, Half_double_six_characteristic_vector,
			72, 27);

	fname = prefix + "_double_sixes_char_vec.csv";

	Fio.Csv_file_support->int_matrix_write_csv(fname, Double_six_characteristic_vector,
			36, 27);



	if (f_v) {
		cout << "schlaefli::write_double_sixes done" << endl;
	}
}





}}}

