/*
 * forest.cpp
 *
 *  Created on: Mar 24, 2025
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace data_structures {


forest::forest()
{
	Record_birth();

	degree = 0;

	orbit = NULL;
	orbit_inv = NULL;

	prev = NULL;
	label = NULL;

	orbit_first = NULL;
	orbit_len = NULL;

	nb_orbits = 0;
}



forest::~forest()
{
	Record_death();

	if (orbit) {
		FREE_int(orbit);
	}
	if (orbit_inv) {
		FREE_int(orbit_inv);
	}
	if (prev) {
		FREE_int(prev);
	}
	if (label) {
		FREE_int(label);
	}
	if (orbit_first) {
		FREE_int(orbit_first);
	}
	if (orbit_len) {
		FREE_int(orbit_len);
	}
}

void forest::init(
		int degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "forest::init n=" << degree << endl;
	}
	forest::degree = degree;

	allocate_tables(verbose_level - 1);

	initialize_tables(verbose_level - 1);

	if (f_v) {
		cout << "forest::init done" << endl;
	}
}

void forest::allocate_tables(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "forest::allocate_tables" << endl;
	}

	orbit = NEW_int(degree);
	orbit_inv = NEW_int(degree);
	prev = NEW_int(degree);
	label = NEW_int(degree);
	orbit_first = NEW_int(degree + 1);
	orbit_len = NEW_int(degree);

	if (f_v) {
		cout << "forest::allocate_tables done" << endl;
	}

}

void forest::initialize_tables(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "forest::initialize_tables" << endl;
	}

	combinatorics::other_combinatorics::combinatorics_domain Combi;
	long int i;

	nb_orbits = 0;
	Combi.Permutations->perm_identity(orbit, degree);
	Combi.Permutations->perm_identity(orbit_inv, degree);
	orbit_first[0] = 0;

	// initialize prev and label with -1:
	for (i = 0; i < degree; i++) {
		prev[i] = -1;
		label[i] = -1;
	}

	if (f_v) {
		cout << "forest::initialize_tables done" << endl;
	}
}

void forest::swap_points(
		int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int pi, pj;

	if (f_v) {
		cout << "forest::swap_points "
				"i=" << i << " j=" << j << endl;
	}
	pi = orbit[i];
	pj = orbit[j];
	orbit[i] = pj;
	orbit[j] = pi;
	orbit_inv[pi] = j;
	orbit_inv[pj] = i;
	if (f_v) {
		cout << "forest::swap_points done" << endl;
	}
}

void forest::move_point_here(
		int here, int pt)
{
	int a, loc;
	if (orbit[here] == pt) {
		return;
	}
	a = orbit[here];
	loc = orbit_inv[pt];
	orbit[here] = pt;
	orbit[loc] = a;
	orbit_inv[a] = loc;
	orbit_inv[pt] = here;
}

int forest::orbit_representative(
		int pt)
{
	int j;

	while (true) {
		j = orbit_inv[pt];
		if (prev[j] == -1) {
			return pt;
		}
		pt = prev[j];
	}
}

int forest::depth_in_tree(
		int j)
// j is a coset, not a point
{
	if (prev[j] == -1) {
		return 0;
	}
	else {
		return depth_in_tree(orbit_inv[prev[j]]) + 1;
	}
}

int forest::sum_up_orbit_lengths()
{
	int i, l, N;

	N = 0;
	for (i = 0; i < nb_orbits; i++) {
		l = orbit_len[i];
		N += l;
	}
	return N;
}

void forest::get_path_and_labels(
		std::vector<int> &path,
		std::vector<int> &labels,
		int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "forest::get_path_and_labels at " << i << endl;
	}
	int ii = orbit_inv[i];

	if (prev[ii] == -1) {
		//path.push_back(i);
		//labels.push_back(label[i]);
	}
	else {
		get_path_and_labels(
				path, labels, prev[ii], verbose_level);
		path.push_back(i);
		labels.push_back(label[ii]);
	}

	if (f_v) {
		cout << "forest::get_path_and_labels done" << endl;
	}
}

void forest::trace_back(
		int i, int &j)
{
	int ii = orbit_inv[i];

	if (prev[ii] == -1) {

		j = 1;
	}
	else {
		trace_back(prev[ii], j);

		j++;
	}
}

void forest::trace_back_and_record_path(
		int *path, int i, int &j)
{
	int ii = orbit_inv[i];

	if (prev[ii] == -1) {
		j = 1;
	}
	else {
		trace_back_and_record_path(path, prev[ii], j);
		j++;
	}
}



void forest::intersection_vector(
		int *set,
		int len, int *intersection_cnt)
// intersection_cnt[nb_orbits]
{
	int i, pt, o;

	Int_vec_zero(intersection_cnt, nb_orbits);
	for (i = 0; i < len; i++) {
		pt = set[i];
		o = orbit_number(pt);
		intersection_cnt[o]++;
	}
}

void forest::get_orbit_partition_of_points_and_lines(
		other::data_structures::partitionstack &S,
		int verbose_level)
{
	int first_column_element, pos, first_column_orbit, i, j, f, l, a;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "forest::get_orbit_partition_of_points_and_lines" << endl;
	}
	first_column_element = S.startCell[1];
	if (f_v) {
		cout << "first_column_element = "
				<< first_column_element << endl;
	}
	pos = orbit_inv[first_column_element];
	first_column_orbit = orbit_number(first_column_element);

	for (i = first_column_orbit - 1; i > 0; i--) {
		f = orbit_first[i];
		l = orbit_len[i];
		for (j = 0; j < l; j++) {
			pos = f + j;
			a = orbit[pos];
			S.subset[j] = a;
		}
		S.subset_size = l;
		S.split_cell(false);
	}
	for (i = nb_orbits - 1; i > first_column_orbit; i--) {
		f = orbit_first[i];
		l = orbit_len[i];
		for (j = 0; j < l; j++) {
			pos = f + j;
			a = orbit[pos];
			S.subset[j] = a;
		}
		S.subset_size = l;
		S.split_cell(false);
	}
}



void forest::get_orbit_partition(
		other::data_structures::partitionstack &S,
	int verbose_level)
{
	int pos, i, j, f, l, a;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "forest::get_orbit_partition" << endl;
	}
	for (i = nb_orbits - 1; i > 0; i--) {
		f = orbit_first[i];
		l = orbit_len[i];
		for (j = 0; j < l; j++) {
			pos = f + j;
			a = orbit[pos];
			S.subset[j] = a;
		}
		S.subset_size = l;
		S.split_cell(false);
	}
}

void forest::get_orbit_in_order(
		std::vector<int> &Orb,
	int orbit_idx, int verbose_level)
{
	int f, l, j, a, pos;

	f = orbit_first[orbit_idx];
	l = orbit_len[orbit_idx];
	for (j = 0; j < l; j++) {
		pos = f + j;
		a = orbit[pos];
		Orb.push_back(a);
	}
}

void forest::get_orbit(
		int orbit_idx, long int *set, int &len,
	int verbose_level)
{
	int f, i;

	f = orbit_first[orbit_idx];
	len = orbit_len[orbit_idx];
	for (i = 0; i < len; i++) {
		set[i] = orbit[f + i];
	}
}

void forest::compute_orbit_statistic(
		int *set, int set_size,
	int *orbit_count, int verbose_level)
// orbit_count[nb_orbits]
{
	int f_v = (verbose_level >= 1);
	int i, a, o;

	if (f_v) {
		cout << "forest::compute_orbit_statistic" << endl;
	}
	Int_vec_zero(orbit_count, nb_orbits);
	for (i = 0; i < set_size; i++) {
		a = set[i];
		o = orbit_number(a);
		orbit_count[o]++;
	}
	if (f_v) {
		cout << "forest::compute_orbit_statistic done" << endl;
	}
}

void forest::compute_orbit_statistic_lint(
		long int *set, int set_size,
	int *orbit_count, int verbose_level)
// orbit_count[nb_orbits]
{
	int f_v = (verbose_level >= 1);
	int i, a, o;

	if (f_v) {
		cout << "forest::compute_orbit_statistic_lint" << endl;
	}
	Int_vec_zero(orbit_count, nb_orbits);
	for (i = 0; i < set_size; i++) {
		a = set[i];
		o = orbit_number(a);
		orbit_count[o]++;
	}
	if (f_v) {
		cout << "forest::compute_orbit_statistic_lint done" << endl;
	}
}


void forest::orbits_as_set_of_sets(
		other::data_structures::set_of_sets *&S,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Sz;
	int i, j, a, f, l;

	if (f_v) {
		cout << "forest::orbits_as_set_of_sets" << endl;
	}
	S = NEW_OBJECT(other::data_structures::set_of_sets);
	Sz = NEW_int(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		l = orbit_len[i];
		Sz[i] = l;
	}

	S->init_basic_with_Sz_in_int(
			degree /* underlying_set_size */,
			nb_orbits, Sz, 0 /* verbose_level */);
	for (i = 0; i < nb_orbits; i++) {
		f = orbit_first[i];
		l = orbit_len[i];
		for (j = 0; j < l; j++) {
			a = orbit[f + j];
			S->Sets[i][j] = a;
		}
	}
	FREE_int(Sz);
	if (f_v) {
		cout << "forest::orbits_as_set_of_sets done" << endl;
	}
}

void forest::get_orbit_reps(
		int *&Reps,
		int &nb_reps, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, f;

	if (f_v) {
		cout << "forest::get_orbit_reps" << endl;
	}
	nb_reps = nb_orbits;
	Reps = NEW_int(nb_reps);
	for (i = 0; i < nb_reps; i++) {
		f = orbit_first[i];
		a = orbit[f];
		Reps[i] = a;
	}
	if (f_v) {
		cout << "forest::get_orbit_reps done" << endl;
	}
}

int forest::find_shortest_orbit_if_unique(
		int &idx)
{
	int l_min = 0, l, i;
	int idx_min = -1;
	int f_is_unique = true;

	for (i = 0; i < nb_orbits; i++) {
		l = orbit_len[i];
		if (idx_min == -1) {
			l_min = l;
			idx_min = i;
			f_is_unique = true;
		}
		else if (l < l_min) {
			l_min = l;
			idx_min = i;
			f_is_unique = true;
		}
		else if (l_min == l) {
			f_is_unique = false;
		}
	}
	idx = idx_min;
	return f_is_unique;
}

void forest::elements_in_orbit_of(
		int pt,
	int *orb, int &nb, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx, f;

	if (f_v) {
		cout << "forest::elements_in_orbit_of" << endl;
	}
	idx = orbit_number(pt);
	f = orbit_first[idx];
	nb = orbit_len[idx];
	Int_vec_copy(orbit + f, orb, nb);
	if (f_v) {
		cout << "forest::elements_in_orbit_of done" << endl;
	}
}

void forest::get_orbit_length(
		int *&orbit_length, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int I, f, l, h, a;

	if (f_v) {
		cout << "forest::get_orbit_length" << endl;
	}
	orbit_length = NEW_int(degree);
	for (I = 0; I < nb_orbits; I++) {
		f = orbit_first[I];
		l = orbit_len[I];
		for (h = 0; h < l; h++) {
			a = orbit[f + h];
			orbit_length[a] = l;
		}
	}
	if (f_v) {
		cout << "forest::get_orbit_length done" << endl;
	}
}

void forest::get_orbit_lengths_once_each(
	int *&orbit_lengths, int &nb_orbit_lengths)
{
	int *val, *mult, len;

	other::orbiter_kernel_system::Orbiter->Int_vec->distribution(
			orbit_len, nb_orbits, val, mult, len);
	//int_distribution_print(ost, val, mult, len);
	//ost << endl;

	nb_orbit_lengths = len;

	orbit_lengths = NEW_int(nb_orbit_lengths);

	Int_vec_copy(val, orbit_lengths, nb_orbit_lengths);

	FREE_int(val);
	FREE_int(mult);
}


int forest::orbit_number(
		int pt)
{
	int pos;
	int idx;
	other::data_structures::sorting Sorting;

	pos = orbit_inv[pt];
	if (Sorting.int_vec_search(orbit_first, nb_orbits, pos, idx)) {
		;
	}
	else {
		if (idx == 0) {
			cout << "forest::orbit_number idx == 0" << endl;
			exit(1);
		}
		idx--;
	}
	if (orbit_first[idx] <= pos &&
			pos < orbit_first[idx] + orbit_len[idx]) {
		return idx;
	}
	else {
		cout << "forest::orbit_number something is wrong, "
				"perhaps the orbit of the point has not yet "
				"been computed" << endl;
		exit(1);
	}
}

void forest::get_orbit_number_and_position(
		int pt, int &orbit_idx, int &orbit_pos,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int pos;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "forest::get_orbit_number_and_position" << endl;
	}
	pos = orbit_inv[pt];
	if (Sorting.int_vec_search(orbit_first, nb_orbits, pos, orbit_idx)) {
		;
	}
	else {
		if (orbit_idx == 0) {
			cout << "forest::get_orbit_number_and_position "
					"orbit_idx == 0" << endl;
			exit(1);
		}
		orbit_idx--;
	}
	if (orbit_first[orbit_idx] <= pos &&
			pos < orbit_first[orbit_idx] + orbit_len[orbit_idx]) {
		orbit_pos = pos - orbit_first[orbit_idx];
	}
	else {
		cout << "forest::get_orbit_number_and_position something is wrong, "
				"perhaps the orbit of the point has not yet "
				"been computed" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "forest::get_orbit_number_and_position done" << endl;
	}
}


void forest::get_orbit_decomposition_scheme_of_graph(
	int *Adj, int n, int *&Decomp_scheme,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int I, J;
	int f1, l1;
	int f2, l2;
	int i, j, r, r0, a, b;

	if (f_v) {
		cout << "forest::get_orbit_decomposition_scheme_of_graph" << endl;
	}
	Decomp_scheme = NEW_int(nb_orbits * nb_orbits);
	Int_vec_zero(Decomp_scheme, nb_orbits * nb_orbits);
	for (I = 0; I < nb_orbits; I++) {
		f1 = orbit_first[I];
		l1 = orbit_len[I];
		if (false) {
			cout << "I = " << I << " f1 = " << f1
					<< " l1 = " << l1 << endl;
		}
		for (J = 0; J < nb_orbits; J++) {
			r0 = 0;
			f2 = orbit_first[J];
			l2 = orbit_len[J];
			if (false) {
				cout << "J = " << J << " f2 = " << f2
						<< " l2 = " << l2 << endl;
			}
			for (i = 0; i < l1; i++) {
				a = orbit[f1 + i];
				r = 0;
				for (j = 0; j < l2; j++) {
					b = orbit[f2 + j];
					if (Adj[a * n + b]) {
						r++;
					}
				}
				if (i == 0) {
					r0 = r;
				}
				else {
					if (r0 != r) {
						cout << "forest::get_orbit_decomposition_scheme_of_graph "
								"not tactical" << endl;
						cout << "I=" << I << endl;
						cout << "J=" << J << endl;
						cout << "r0=" << r0 << endl;
						cout << "r=" << r << endl;
						exit(1);
					}
				}
			}
			if (false) {
				cout << "I = " << I << " J = " << J << " r = " << r0 << endl;
			}
			Decomp_scheme[I * nb_orbits + J] = r0;
		}
	}
	if (f_v) {
		cout << "Decomp_scheme = " << endl;
		Int_matrix_print(Decomp_scheme, nb_orbits, nb_orbits);
	}
	if (f_v) {
		cout << "forest::get_orbit_decomposition_scheme_of_graph done" << endl;
	}
}

void forest::create_point_list_sorted(
		int *&point_list, int &point_list_length)
{
	int i, j, k, f, l, ff, p;
	other::data_structures::sorting Sorting;

	point_list_length = 0;
	for (k = 0; k < nb_orbits; k++) {
		point_list_length += orbit_len[k];
	}
	point_list = NEW_int(point_list_length);

	ff = 0;
	for (k = 0; k < nb_orbits; k++) {
		f = orbit_first[k];
		l = orbit_len[k];
		for (j = 0; j < l; j++) {
			i = f + j;
			p = orbit[i];
			point_list[ff + j] = p;
		}
		ff += l;
	}
	if (ff != point_list_length) {
		cout << "forest::create_point_list_sorted "
				"ff != point_list_length" << endl;
		exit(1);
	}
	Sorting.int_vec_heapsort(point_list, point_list_length);
}

int forest::get_num_points()
// This function returns the number of points in the schreier forest
{

	int total_points_in_forest = 0;

	for (int orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		total_points_in_forest += this->orbit_len[orbit_idx];
	}

	return total_points_in_forest;
}

double forest::get_average_word_length()
// This function returns the average word length of the forest.
{

	double avgwl = 0.0;
	int total_points_in_forest = get_num_points();

	for (int orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		avgwl += get_average_word_length(orbit_idx) * orbit_len[orbit_idx]
				/ total_points_in_forest;
	}

	return avgwl;
}

double forest::get_average_word_length(
		int orbit_idx)
{
	int fst = orbit_first[orbit_idx];
	//int len = orbit_len[orbit_idx];
	//int root = orbit[fst];
	int l;

	// Average and optimal word lengths of old tree
	int last = orbit_first[orbit_idx + 1];
	int L = 0, N = last - fst;
	for (int j = 0; j < last; j++) {
		trace_back(orbit[j], l);
		L += l;
	}

	return L / double(N);
}




}}}}

