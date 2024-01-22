// schreier_vector.cpp
//
// Anton Betten
// moved here from schreier.cpp: December 20, 2015

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace data_structures_groups {


schreier_vector::schreier_vector()
{
	gen_hdl_first = 0;
	nb_gen = 0;
	number_of_orbits = -1;
	sv = NULL;
	f_has_local_generators = false;
	local_gens = NULL;
}

schreier_vector::~schreier_vector()
{
	if (sv) {
		FREE_int(sv);
	}
	if (f_has_local_generators) {
		if (local_gens) {
			FREE_OBJECT(local_gens);
		}
	}
}

void schreier_vector::init(
		int gen_hdl_first, int nb_gen, int *sv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier_vector::init" << endl;
	}
	schreier_vector::gen_hdl_first = gen_hdl_first;
	schreier_vector::nb_gen = nb_gen;
	schreier_vector::sv = sv;
	if (sv) {
		number_of_orbits = count_number_of_orbits();
	}
	else {
		number_of_orbits = -1;
	}
	f_has_local_generators = false;
	if (f_v) {
		cout << "schreier_vector::init done" << endl;
	}
}

void schreier_vector::init_local_generators(
		vector_ge *gens,
		int verbose_level)
// copies gens to local_gens
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier_vector::init_local_generators" << endl;
	}

	gens->copy(local_gens, verbose_level - 2);
	f_has_local_generators = true;

	if (f_v) {
		cout << "schreier_vector::init_local_generators done" << endl;
	}
}

void schreier_vector::set_sv(
		int *sv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier_vector::set_sv" << endl;
	}
	schreier_vector::sv = sv;
	number_of_orbits = count_number_of_orbits();
}

int *schreier_vector::points()
{
	if (sv == NULL) {
		cout << "schreier_vector::points "
				"sv == NULL" << endl;
		exit(1);
	}
	return sv + 1;
}

int *schreier_vector::prev()
{
	if (sv == NULL) {
		cout << "schreier_vector::prev "
				"sv == NULL" << endl;
		exit(1);
	}
	if (nb_gen == 0) {
		cout << "schreier_vector::prev N/A since nb_gen == 0" << endl;
		exit(1);
	}
	int n = sv[0];
	return sv + 1 + n;
}

int *schreier_vector::label()
{
	if (sv == NULL) {
		cout << "schreier_vector::label "
				"sv == NULL" << endl;
		exit(1);
	}
	if (nb_gen == 0) {
		cout << "schreier_vector::label N/A since nb_gen == 0" << endl;
		exit(1);
	}
	int n = sv[0];
	return sv + 1 + 2 * n;
}

int schreier_vector::get_number_of_points()
{
	if (sv == NULL) {
		cout << "schreier_vector::get_number_of_points "
				"sv == NULL" << endl;
		exit(1);
	}
	return sv[0];
}

int schreier_vector::get_number_of_orbits()
{
	if (number_of_orbits == -1) {
		cout << "schreier_vector::get_number_of_orbits "
			"number_of_orbits == -1" << endl;
		exit(1);
	}
	return number_of_orbits;
}

int schreier_vector::count_number_of_orbits()
{
	int i, n;
	int *pts;

	if (sv == NULL) {
		cout << "schreier_vector::count_number_of_orbits "
				"sv == NULL" << endl;
		exit(1);
	}
	n = sv[0];
	pts = sv + 1;
	if (nb_gen == 0) {
		return n;
	}
	else {
		int *prev;
		//int *label;
		int nb = 0;
		prev = pts + n;
		for (i = 0; i < n; i++) {
			if (prev[i] == -1) {
				nb++;
			}
		}
		return nb;
	}
}

void schreier_vector::count_number_of_orbits_and_get_orbit_reps(
		int *&orbit_reps, int &nb_orbits)
{
	int i, n;
	int *pts;

	if (sv == NULL) {
		cout << "schreier_vector::count_number_of_orbits "
				"sv == NULL" << endl;
		exit(1);
	}
	nb_orbits = 0;
	n = sv[0];
	pts = sv + 1;
	if (nb_gen == 0) {
		nb_orbits = n;
		orbit_reps = NEW_int(nb_orbits);
		for (i = 0; i < n; i++) {
			orbit_reps[i] = pts[i];
		}
	}
	else {
		int nb;
		int *prev;
		//int *label;

		prev = pts + n;
		for (i = 0; i < n; i++) {
			if (prev[i] == -1) {
				nb_orbits++;
			}
		}
		orbit_reps = NEW_int(nb_orbits);
		nb = 0;
		for (i = 0; i < n; i++) {
			if (prev[i] == -1) {
				orbit_reps[nb++] = pts[i];
			}
		}
	}
}

int schreier_vector::determine_depth_recursion(
	int n, int *pts, int *prev,
	int *depth, int *ancestor, int pos)
{
	int pt, pt_loc, d;
	data_structures::sorting Sorting;

	pt = prev[pos];
	if (pt == -1) {
		depth[pos] = 0;
		ancestor[pos] = pts[pos];
		return 0;
	}
	if (!Sorting.int_vec_search(pts, n, pt, pt_loc)) {
		int i;

		cout << "schreier_vector::determine_depth_recursion, "
				"fatal: did not find pt" << endl;
		cout << "pt = " << pt << endl;
		cout << "vector of length " << n << endl;
		Int_vec_print(cout, pts, n);
		cout << endl;
		cout << "i : pts[i] : prev[i] : depth[i] : ancestor[i]" << endl;
		for (i = 0; i < n; i++) {
			cout
				<< setw(5) << i << " : "
				<< setw(5) << pts[i] << " : "
				<< setw(5) << prev[i] << " : "
				//<< setw(5) << label[i] << " : "
				<< setw(5) << depth[i] << " : "
				<< setw(5) << ancestor[i]
				<< endl;
		}
		exit(1);
	}
	d = depth[pt_loc];
	if (d >= 0) {
		d++;
	}
	else {
		d = determine_depth_recursion(n,
				pts, prev, depth, ancestor, pt_loc) + 1;
	}
	depth[pos] = d;
	ancestor[pos] = ancestor[pt_loc];
	return d;
}


void schreier_vector::relabel_points(
		induced_actions::action_on_factor_space *AF,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_trivial_group;
	int n;
	int *pts;
	int i, pt, pre, q, pr, new_pr, pos;
	int *new_sv;
	int *new_pts;
	int *new_pts_sorted;
	int *perm;
	int *new_sv_pts;
	int *new_sv_prev;
	int *new_sv_label;

	int nb_old_orbit_reps = 0, idx, j;
	int *old_orbit_reps = NULL;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "schreier_vector::relabel_points" << endl;
	}
	if (nb_gen == 0) {
		f_trivial_group = true;
	}
	else {
		f_trivial_group = false;
	}
#if 0
	if (!f_compact) {
		cout << "schreier_vector::relabel_points "
				"changing point labels: fatal: !f_compact" << endl;
		exit(1);
	}
#endif
	n = sv[0];
	pts = sv + 1;

	if (f_trivial_group) {
		if (f_v) {
			cout << "schreier_vector::relabel_points "
					"trivial group" << endl;
		}
		new_sv = NEW_int(n + 1);
		new_pts = new_sv + 1;
		new_sv[0] = n;
		for (i = 0; i < n; i++) {
			pt = pts[i];
			pre = AF->preimage(pt, 0 /*verbose_level - 3*/);
			q = AF->project_onto_Gauss_reduced_vector(
					pre, 0 /*verbose_level - 2*/);
			if (false) {
				cout << "i=" << i << " pt=" << pt
						<< " pre=" << pre << " q=" << q << endl;
			}
			new_pts[i] = q;
		}
		Sorting.int_vec_heapsort(new_pts, n);
		for (i = 0; i < n + 1; i++) {
			sv[i] = new_sv[i];
		}
		FREE_int(new_sv);
		return;
	}


	int *prev;
	int *label;
	prev = pts + n;
	label = prev + n;


	new_sv = NEW_int(3 * n + 1);
	new_pts = NEW_int(n);
	new_pts_sorted = NEW_int(n);
	perm = NEW_int(n);
	new_sv_pts = new_sv + 1;
	new_sv_prev = new_sv_pts + n;
	new_sv_label = new_sv_prev + n;
	for (i = 0; i < n; i++) {
		perm[i] = i;
	}
	if (f_v) {
		nb_old_orbit_reps = 0;
		cout << "schreier_vector::relabel_points "
				"old orbit reps:" << endl;
		for (i = 0; i < n; i++) {
			if (prev[i] == -1) {
				cout << "orbit rep " << pts[i] << endl;
				nb_old_orbit_reps++;
			}
		}
		old_orbit_reps = NEW_int(nb_old_orbit_reps);
		j = 0;
		for (i = 0; i < n; i++) {
			if (prev[i] == -1) {
				old_orbit_reps[j++] = pts[i];
			}
		}
		Sorting.int_vec_heapsort(old_orbit_reps, nb_old_orbit_reps);
		Int_vec_print(cout, old_orbit_reps, nb_old_orbit_reps);
		cout << endl;
		cout << "schreier_vector::relabel_points "
				"There are " << nb_old_orbit_reps
				<< " old orbit reps, they are:" << endl;
		for (i = 0; i < nb_old_orbit_reps; i++) {
			cout << i << " / " << nb_old_orbit_reps
					<< " : " << old_orbit_reps[i] << endl;
		}
	}
	if (f_vv) {
		cout << "schreier_vector::relabel_points "
				"before:" << endl;
		for (i = 0; i < n; i++) {
			if (Sorting.int_vec_search(old_orbit_reps,
					nb_old_orbit_reps, pts[i], idx)) {
				cout << setw(5) << i << " : "
						<< setw(5) << pts[i] << endl;
			}
		}
	}
	if (f_vv) {
		cout << "schreier_vector::relabel_points "
				"computing new_pts" << endl;
	}
	for (i = 0; i < n; i++) {
		pt = pts[i];
		if (false) {
			cout << "i=" << i << " pt=" << pt << endl;
		}
		pre = AF->preimage(pt, 0/*verbose_level - 3*/);
		if (false) {
			cout << "pre=" << pre << endl;
		}
		q = AF->project_onto_Gauss_reduced_vector(
				pre, 0 /*verbose_level - 2*/);
		if (false) {
			if (Sorting.int_vec_search(old_orbit_reps,
					nb_old_orbit_reps, pt, idx)) {
				cout << "i=" << i << " pt=" << pt
						<< " pre=" << pre << " q=" << q << endl << endl;
			}
		}
		new_pts[i] = q;
	}
	if (f_vv) {
		//cout << "after:" << endl;
		cout << "i : pts[i] : new_pts[i]" << endl;
		for (i = 0; i < n; i++) {
			if (Sorting.int_vec_search(old_orbit_reps,
					nb_old_orbit_reps, pts[i], idx)) {
				cout << setw(5) << i << " : "
						<< setw(5) << pts[i] << " : "
						<< setw(5) << new_pts[i] << endl;
			}
		}
	}
	if (f_vv) {
		cout << "schreier_vector::relabel_points "
				"sorting:" << endl;
	}
	for (i = 0; i < n; i++) {
		new_pts_sorted[i] = new_pts[i];
	}
	Sorting.int_vec_heapsort_with_log(new_pts_sorted, perm, n);
	if (f_vv) {
		cout << "schreier_vector::relabel_points "
				"after sorting:" << endl;
		cout << "i : pts[i] : new_pts_sorted[i] : perm[i]" << endl;
		for (i = 0; i < n; i++) {
			if (Sorting.int_vec_search(old_orbit_reps,
					nb_old_orbit_reps, pts[i], idx)) {
				cout << setw(5) << i << " : "
					<< setw(5) << pts[i] << " : "
					<< setw(5) << new_pts_sorted[i]
					<< " : " << setw(5) << perm[i] << endl;
			}
		}
	}
	new_sv[0] = n;
	for (i = 0; i < n; i++) {
		new_sv_pts[i] = new_pts_sorted[i];
		pos = perm[i];
		pr = prev[pos];
		if (pr == -1) {
			new_pr = -1;
		}
		else {
			new_pr = new_pts[pr];
		}
		new_sv_prev[i] = new_pr;
		new_sv_label[i] = label[pos];
	}
	if (f_vv) {
		cout << "schreier_vector::relabel_points "
				"old / new schreier vector:" << endl;
		cout << "i : pts[i] : prev[i] : label[i] :: i : "
				"new_sv_pts[i] : new_sv_prev[i] : "
				"new_sv_label[i] " << endl;
		for (i = 0; i < n; i++) {
			cout << setw(5) << i << " : "
				<< setw(5) << pts[i] << " : "
				<< setw(5) << prev[i] << " : "
				<< setw(5) << label[i]
				<< " :: ";

			cout << setw(5) << i << " : "
				<< setw(5) << new_sv_pts[i] << " : "
				<< setw(5) << new_sv_prev[i] << " : "
				<< setw(5) << new_sv_label[i]
				<< endl;
		}
		cout << "i : orbit_rep : lexleast : project : "
				"project : preimage" << endl;
		for (i = 0; i < n; i++) {
			if (new_sv_prev[i] == -1) {
				cout << i << " : ";
				//cout << "new_sv_pts[i]=" << new_sv_pts[i] << endl;
				//cout << "AF->lexleast_element_in_coset(new_sv_pts[i], 0)="
				// << AF->lexleast_element_in_coset(new_sv_pts[i], 0) << endl;
				//cout << "AF->project(new_sv_pts[i], 0)="
				// << AF->project(new_sv_pts[i], 0) << endl;
				//cout << "AF->preimage(AF->project(new_sv_pts[i], 0), 0)="
				// << AF->preimage(AF->project(new_sv_pts[i], 0), 0) << endl;
				cout << setw(6) <<
						new_sv_pts[i] << " : ";
				cout << setw(6) <<
						AF->lexleast_element_in_coset(
								new_sv_pts[i], 0) << " : ";
				cout << setw(6)
						<< AF->project(new_sv_pts[i], 0) << " : ";
				cout << setw(6)
						<< AF->preimage(
								AF->project(new_sv_pts[i], 0), 0)
								<< endl;
			}
		}
		cout << "copying over" << endl;
	}
	for (i = 0; i < 3 * n + 1; i++) {
		sv[i] = new_sv[i];
	}
	FREE_int(new_sv);
	FREE_int(new_pts);
	FREE_int(new_pts_sorted);
	FREE_int(perm);
	if (old_orbit_reps) {
		FREE_int(old_orbit_reps);
	}
	if (f_v) {
		cout << "schreier_vector::relabel_points "
				"new schreier vector created" << endl;
		cout << "schreier_vector::relabel_points done" << endl;
	}
}

void schreier_vector::orbit_stats(
		int &nb_orbits, int *&orbit_reps,
		int *&orbit_length, int *&total_depth,
		int verbose_level)
{
	int i, idx;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "schreier_vector::orbit_stats" << endl;
		}
	int n;
	int *pts;
	int *depth;
	int *ancestor;
	data_structures::sorting Sorting;


	n = sv[0];
	pts = sv + 1;


	if (nb_gen == 0) {
		nb_orbits = n;
		orbit_reps = NEW_int(nb_orbits);
		orbit_length = NEW_int(nb_orbits);
		total_depth = NEW_int(nb_orbits);
		Int_vec_copy(pts, orbit_reps, nb_orbits);
		for (i = 0; i < nb_orbits; i++) {
			orbit_length[i] = 1;
			total_depth[i] = 1;
		}
		return;
	}

	int *prev;
	int *label;
	prev = pts + n;
	label = prev + n;
	if (f_v) {
		cout << "schreier_vector::orbit_stats "
				"schreier vector of length " << n << endl;
		}

	depth = NEW_int(n);
	ancestor = NEW_int(n);

	for (i = 0; i < n; i++) {
		depth[i] = -1;
		ancestor[i] = -1;
		}
	if (f_vv) {
		cout << "schreier_vector::orbit_stats "
				"determining depth using schreier_vector_determine_"
				"depth_recursion" << endl;
		}
	for (i = 0; i < n; i++) {
		Sorting.schreier_vector_determine_depth_recursion(n,
				pts, prev, false, NULL, depth, ancestor, i);
		}
	if (f_vv) {
		cout << "schreier_vector::orbit_stats "
				"determining depth using schreier_vector_"
				"determine_depth_recursion done" << endl;
		}
	if (f_vvv && n < 100) {
		cout << "i : pts[i] : prev[i] : label[i] : "
				"depth[i] : ancestor[i]" << endl;
		for (i = 0; i < n; i++) {
			cout
				<< setw(5) << i << " : "
				<< setw(5) << pts[i] << " : "
				<< setw(5) << prev[i] << " : "
				<< setw(5) << label[i] << " : "
				<< setw(5) << depth[i] << " : "
				<< setw(5) << ancestor[i]
				<< endl;
			}
		}

	nb_orbits = 0;
	for (i = 0; i < n; i++) {
		if (prev[i] == -1) {
			nb_orbits++;
			}
	}
	orbit_reps = NEW_int(nb_orbits);
	orbit_length = NEW_int(nb_orbits);
	total_depth = NEW_int(nb_orbits);

	int nb_orb, a;

	nb_orb = 0;
	for (i = 0; i < n; i++) {
		if (prev[i] == -1) {
			orbit_reps[nb_orb] = pts[i];
			orbit_length[nb_orb] = 1;
			total_depth[nb_orb] = 1;
			nb_orb++;
		}
	}
	for (i = 0; i < n; i++) {
		if (prev[i] > 0) {
			a = ancestor[i];
			if (!Sorting.int_vec_search(orbit_reps, nb_orbits, a, idx)) {
				cout << "schreier_vector::orbit_stats "
						"cannot find ancestor in list of orbit reps" << endl;
				exit(1);
			}
			orbit_length[idx]++;
			total_depth[idx] += depth[i] + 1;
		}
	}
	FREE_int(depth);
	FREE_int(ancestor);
}

void schreier_vector::orbit_of_point(
		int pt, long int *&orbit_elts,
		int &orbit_len, int &idx_of_root_node,
		int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "schreier_vector::orbit_of_point "
				"pt=" << pt << endl;
	}
	int n;
	int *pts;
	int *depth;
	int *ancestor;

	int *orbit_elt_idx;

	n = sv[0];
	pts = sv + 1;


	if (nb_gen == 0) {
		orbit_len = 1;
		orbit_elts = NEW_lint(orbit_len);
		orbit_elts[0] = pt;
		return;
	}

	int *prev;
	int *label;
	prev = pts + n;
	label = prev + n;
	if (f_v) {
		cout << "schreier_vector::orbit_of_point "
				"schreier vector of length " << n << endl;
	}


	if (!Sorting.int_vec_search(pts, n, pt, idx_of_root_node)) {
		cout << "schreier_vector::orbit_of_point "
				"fatal: point " << pt << " not found" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "schreier_vector::orbit_of_point idx_of_root_node = " << idx_of_root_node << endl;
	}

	depth = NEW_int(n);
	ancestor = NEW_int(n);
	orbit_elt_idx = NEW_int(n);

	for (i = 0; i < n; i++) {
		depth[i] = -1;
		ancestor[i] = -1;
	}
	if (f_vv) {
		cout << "schreier_vector::orbit_of_point "
				"determining depth using schreier_vector_determine_depth_recursion" << endl;
	}
	for (i = 0; i < n; i++) {
		Sorting.schreier_vector_determine_depth_recursion(n,
				pts, prev, false, NULL, depth, ancestor, i);
	}
	if (f_vv) {
		cout << "schreier_vector::orbit_of_point "
				"determining depth using schreier_vector_determine_depth_recursion done" << endl;
	}
	if (f_vvv && n < 100) {
		cout << "i : pts[i] : prev[i] : label[i] : "
				"depth[i] : ancestor[i]" << endl;
		for (i = 0; i < n; i++) {
			cout
				<< setw(5) << i << " : "
				<< setw(5) << pts[i] << " : "
				<< setw(5) << prev[i] << " : "
				<< setw(5) << label[i] << " : "
				<< setw(5) << depth[i] << " : "
				<< setw(5) << ancestor[i]
				<< endl;
		}
	}
	orbit_len = 0;
	for (i = 0; i < n; i++) {
		if (ancestor[i] == pt) {
			if (i == idx_of_root_node) {
				idx_of_root_node = orbit_len;
			}
			orbit_elt_idx[orbit_len++] = i;
		}
	}
	if (f_v) {
		cout << "schreier_vector::orbit_of_point "
				"found orbit of length " << orbit_len << endl;
	}
	orbit_elts = NEW_lint(orbit_len);
	for (i = 0; i < orbit_len; i++) {
		orbit_elts[i] = pts[orbit_elt_idx[i]];
	}
	if (f_vv) {
		cout << "schreier_vector::orbit_of_point "
				"the points in the orbit are: ";
		Lint_vec_print(cout, orbit_elts, orbit_len);
		cout << endl;
	}


	if (orbit_elts[idx_of_root_node] != pt) {
		cout << "schreier_vector::orbit_of_point "
				"fatal: orbit_elts[idx_of_root_node] != pt" << endl;
		cout << "pt=" << pt << endl;
		cout << "idx_of_root_node=" << idx_of_root_node << endl;
		cout << "orbit_elts[idx_of_root_node]=" << orbit_elts[idx_of_root_node] << endl;
		exit(1);
	}


	for (i = 1; i < orbit_len; i++) {
		if (orbit_elts[i] < orbit_elts[i - 1]) {
			cout << "schreier_vector::orbit_of_point "
					"fatal: orbit_elts[] not increasing" << endl;
			exit(1);
		}
	}

	FREE_int(depth);
	FREE_int(ancestor);
	FREE_int(orbit_elt_idx);
}

void schreier_vector::init_from_schreier(
		groups::schreier *S,
	int f_trivial_group, int verbose_level)
// allocated and creates array sv[size] using NEW_int
// where size is n + 1 if  f_trivial_group is true
// and size is 3 * n + 1 otherwise
// Here, n is the combined size of all orbits counted by nb_orbits
// sv[0] is equal to n
// sv + 1 is the array point_list of size [n],
// listing the point in increasing order
// Unless f_trivial_group, sv + 1 + n is the array prev[n] and
// sv + 1 + 2 * n is the array label[n]
{
	int f_v = (verbose_level >= 1);
	int i, j, p, pr, la, n = 0;
	int *point_list;
	int *svec;

	if (f_v) {
		cout << "schreier_vector::init_from_schreier" << endl;
	}
	S->create_point_list_sorted(point_list, n);


	if (f_trivial_group) {
		svec = NEW_int(n + 1);
		}
	else {
		svec = NEW_int(3 * n + 1);
		}
	svec[0] = n;
	for (i = 0; i < n; i++) {
		svec[1 + i] = point_list[i];
		}
	if (!f_trivial_group) {
		for (i = 0; i < n; i++) {
			p = point_list[i];
			j = S->orbit_inv[p];
			pr = S->prev[j];
			la = S->label[j];
			svec[1 + n + i] = pr;
			svec[1 + 2 * n + i] = la;
			}
		}
	FREE_int(point_list);

	set_sv(svec, verbose_level - 1);

	if (f_v) {
		cout << "schreier_vector::init_from_schreier done" << endl;
	}
}

void schreier_vector::init_shallow_schreier_forest(
		groups::schreier *S,
	int f_trivial_group, int f_randomized,
	int verbose_level)
// initializes local_gens
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int n;
	int *point_list;
	int *svec;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "schreier_vector::init_shallow_schreier_forest" << endl;
	}
	S->create_point_list_sorted(point_list, n);


	if (f_trivial_group) {
		svec = NEW_int(n + 1);
		svec[0] = n;
		Int_vec_copy(point_list, svec + 1, n);
	}
	else {
		int orbit_idx;
		int *points, *prev, *label;
		int fst_gen_idx, fst, len, nb_gens, i;
		int pt, pr, la;
		int j;

		svec = NEW_int(3 * n + 1);
		svec[0] = n;
		points = svec + 1;
		prev = points + n;
		label = prev + n;
		Int_vec_copy(point_list, svec + 1, n);

		f_has_local_generators = true;
		local_gens = NEW_OBJECT(vector_ge);
		local_gens->init(S->A, verbose_level - 2);
		fst_gen_idx = 0;
		for (orbit_idx = 0; orbit_idx < S->nb_orbits; orbit_idx++) {
			if (f_v) {
				cout << "schreier_vector::init_shallow_schreier_forest "
						"orbit_idx=" << orbit_idx
						<< " / " << S->nb_orbits << endl;
			}
			groups::schreier *Shallow_tree;

			if (f_v) {
				cout << "schreier_vector::init_shallow_schreier_forest "
						"creating shallow tree" << endl;
			}
			S->shallow_tree_generators(orbit_idx,
					f_randomized,
					Shallow_tree,
					verbose_level - 2);
			if (f_v) {
				cout << "schreier_vector::init_shallow_schreier_forest "
						"creating shallow tree done" << endl;
			}
			fst = Shallow_tree->orbit_first[0];
			len = Shallow_tree->orbit_len[0];
			nb_gens = Shallow_tree->gens.len;
			if (f_v) {
				cout << "schreier_vector::init_shallow_schreier_forest "
					"orbit " << orbit_idx << " has length " << len << " and "
					<< nb_gens << " generators" << endl;
			}
			for (i = 0; i < nb_gens; i++) {
				local_gens->append(Shallow_tree->gens.ith(i), verbose_level - 2);
			}
			for (i = 0; i < len; i++) {
				pt = Shallow_tree->orbit[fst + i];
				pr = Shallow_tree->prev[fst + i];
				la = Shallow_tree->label[fst + i];
				if (!Sorting.int_vec_search(points, n, pt, j)) {
					cout << "schreier_vector::init_shallow_schreier_forest "
							"could not find point" << endl;
					exit(1);
				}
				prev[j] = pr;
				if (la >= 0) {
					label[j] = la + fst_gen_idx;
				}
				else {
					label[j] = la;
				}
			}
			fst_gen_idx += nb_gens;
			FREE_OBJECT(Shallow_tree);
		}
		if (fst_gen_idx != local_gens->len) {
			cout << "schreier_vector::init_shallow_schreier_forest "
					"fst_gen_idx != local_gens->len" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "schreier_vector::init_shallow_schreier_forest "
				"we have " << local_gens->len
				<< " local generators" << endl;
		}
		if (f_vv) {
			cout << "schreier_vector::init_shallow_schreier_forest "
					"the local generators are:" << endl;
			for (i = 0; i < local_gens->len; i++) {
				cout << "generator " << i << " / "
						<< local_gens->len << ":" << endl;
				local_gens->A->Group_element->element_print_quick(
						local_gens->ith(i), cout);
			}
		}
	}
	FREE_int(point_list);

	set_sv(svec, verbose_level - 1);

	if (f_v) {
		cout << "schreier_vector::init_shallow_schreier_forest "
				"done" << endl;
	}
}


void schreier_vector::export_tree_as_layered_graph(
		int orbit_no, int orbit_rep,
		std::string &fname_mask,
		int verbose_level)
{
	//verbose_level = 3;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int len;
	int *horizontal_position;
	int n, i, j, l, max_depth;
	long int *orbit_elts;
	int *orbit_prev;
	int *orbit_depth;
	int *points;
	int orbit_len;
	int idx_of_root_node;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "schreier_vector::export_tree_as_layered_graph" << endl;
	}

	orbit_of_point(
			orbit_rep, orbit_elts, orbit_len, idx_of_root_node,
			verbose_level);
	len = orbit_len;
	n = sv[0];
	points = sv + 1;

	orbit_prev = NEW_int(orbit_len);
	orbit_depth = NEW_int(orbit_len);

	if (nb_gen == 0) {
		for (j = 0; j < len; j++) {
			orbit_prev[j] = -1;
		}
		max_depth = 0;
	}
	else {
		int pos;
		int *prev;
		prev = points + n;
		max_depth = 0;
		for (j = 0; j < len; j++) {
			if (!Sorting.int_vec_search(points, n, orbit_elts[j], pos)) {
				cout << "schreier_vector::export_tree_as_layered_graph "
						"could not find point" << endl;
				exit(1);
			}
			orbit_prev[j] = prev[pos];
			trace_back(orbit_elts[j], orbit_depth[j]);
			orbit_depth[j]--;
			max_depth = MAX(max_depth, orbit_depth[j]);
		}

		if (false) {
			cout << "j : orbit_elts[j] : orbit_prev[j] : orbit_depth[j]" << endl;
			for (j = 0; j < len; j++) {
				cout << j << " : " << orbit_elts[j] << " : "
						<< orbit_prev[j] << " : " << orbit_depth[j] << endl;
			}
			cout << "orbit_depth:";
			Int_vec_print(cout, orbit_depth, len);
			cout << endl;
		}
	}


	horizontal_position = NEW_int(len);
	int nb_layers;
	nb_layers = max_depth + 1;
	int *Nb;
	int *Nb1;
	int **Node;


	//classify C;
	//C.init(depth, len, false, 0);
	Nb = NEW_int(nb_layers);
	Nb1 = NEW_int(nb_layers);
	Int_vec_zero(Nb, nb_layers);
	Int_vec_zero(Nb1, nb_layers);
	for (j = 0; j < len; j++) {
		l = orbit_depth[j];
		horizontal_position[j] = Nb[l];
		Nb[l]++;
	}
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph" << endl;
		cout << "number of nodes at depth:" << endl;
		for (i = 0; i <= max_depth; i++) {
			cout << i << " : " << Nb[i] << endl;
		}
	}
	Node = NEW_pint(nb_layers);
	for (i = 0; i <= max_depth; i++) {
		Node[i] = NEW_int(Nb[i]);
	}
	for (j = 0; j < len; j++) {
		l = orbit_depth[j];
		Node[l][Nb1[l]] = j;
		Nb1[l]++;
	}
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph" << endl;
		cout << "number of nodes at depth:" << endl;
		for (i = 0; i <= max_depth; i++) {
			cout << i << " : " << Nb[i] << " : ";
			Int_vec_print(cout, Node[i], Nb[i]);
			cout << endl;
		}
	}

	graph_theory::layered_graph *LG;
	int n1, n2, j2;

	LG = NEW_OBJECT(graph_theory::layered_graph);
	if (f_v) {
		cout << "schreier_vector::export_tree_as_layered_graph "
				"before LG->init" << endl;
		}
	//LG->add_data1(data1, 0/*verbose_level*/);

	string dummy;
	dummy.assign("");

	LG->init(nb_layers, Nb, dummy, verbose_level);
	if (f_v) {
		cout << "schreier_vector::export_tree_as_layered_graph "
				"after LG->init" << endl;
		}
	LG->place(verbose_level);
	if (f_v) {
		cout << "schreier_vector::export_tree_as_layered_graph "
				"after LG->place" << endl;
		}
	for (i = 0; i <= max_depth; i++) {
		if (f_vv) {
			cout << "schreier_vector::export_tree_as_layered_graph "
					"adding edges at depth "
					"i=" << i << " / " << max_depth
					<< " Nb[i]=" << Nb[i] << endl;
		}
		for (j = 0; j < Nb[i]; j++) {
			n1 = Node[i][j];
			if (f_vvv) {
				cout << "schreier_vector::export_tree_as_layered_graph "
						"adding edges "
						"i=" << i << " / " << max_depth
						<< " j=" << j << " n1=" << n1 << endl;
			}
			if (orbit_prev[n1] != -1) {
				int pt;
				pt = orbit_prev[n1];
				if (!Sorting.lint_vec_search(orbit_elts, len, pt, n2, 0)) {
					cout << "schreier_vector::export_tree_as_layered_graph "
							"could not find point" << endl;
					exit(1);
				}
				j2 = horizontal_position[n2];
				if (f_vvv) {
					cout << "schreier_vector::export_tree_as_layered_graph "
							"adding edges "
							"i=" << i << " / " << max_depth
							<< " j=" << j << " n1=" << n1 << " pt=" << pt
							<< " n2=" << n2
							<< " j2=horizontal_position=" << j2 << endl;
				}
				if (f_vvv) {
					cout << "schreier_vector::export_tree_as_layered_graph "
							"adding edges "
							"i=" << i << " / " << max_depth
							<< " j=" << j << " n1=" << n1
							<< " n2=" << n2 << " j2=" << j2 << endl;
				}
				if (f_vvv) {
					cout << "adding edge ("<< i - 1 << "," << j2 << ") "
							"-> (" << i << "," << j << ")" << endl;
				}
				LG->add_edge(i - 1, j2, i, j,
						0 /*verbose_level*/);
			}
		}
	}
	for (j = 0; j < len; j++) {
		l = orbit_depth[j];

		string text2;

		text2 = std::to_string(orbit_elts[j]);

		LG->add_text(l, horizontal_position[j],
				text2, 0/*verbose_level*/);
	}

	data_structures::string_tools ST;


	string fname;

	fname = ST.printf_d(fname_mask, orbit_no);


	LG->write_file(fname, 0 /*verbose_level*/);
	FREE_OBJECT(LG);

	FREE_int(Nb);
	FREE_int(Nb1);
	FREE_int(horizontal_position);

	FREE_lint(orbit_elts);
	FREE_int(orbit_prev);
	FREE_int(orbit_depth);

	if (f_v) {
		cout << "schreier_vector::export_tree_as_layered_graph "
				"done" << endl;
	}
}

void schreier_vector::trace_back(
		int pt, int &depth)
{
	int pr, n, pos;
	int *points;
	int *prev;
	data_structures::sorting Sorting;

	n = sv[0];
	points = sv + 1;
	prev = points + n;
	if (nb_gen == 0) {
		depth = 1;
	}
	depth = 1;
	while (true) {
		if (!Sorting.int_vec_search(points, n, pt, pos)) {
			cout << "schreier_vector::trace_back "
					"could not find point" << endl;
			exit(1);
		}
		pr = prev[pos];
		if (pr == -1) {
			break;
		}
		depth++;
		pt = pr;
	}
}






void schreier_vector::print()
{
	int i, n;
	int *pts;

	cout << "schreier_vector::print:" << endl;
	if (sv == NULL) {
		cout << "schreier_vector::print "
				"sv == NULL" << endl;
		return;
	}
	n = sv[0];
	pts = sv + 1;
	if (nb_gen == 0) {
		cout << "nb_gen == 0" << endl;
	}
	else {
		cout << "nb_gen=" << nb_gen << endl;
		int *prev;
		int *label;

		prev = pts + n;
		label = prev + n;
		cout << "i : pts[i] : prev[i] : label[i]" << endl;
		for (i = 0; i < n; i++) {
			cout
				<< setw(5) << i << " : "
				<< setw(5) << pts[i] << " : "
				<< setw(5) << prev[i] << " : "
				<< setw(5) << label[i] << " : "
				<< endl;
		}
	}
}



}}}

