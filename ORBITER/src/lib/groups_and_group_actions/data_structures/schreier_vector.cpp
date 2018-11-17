// schreier_vector.C
//
// Anton Betten
// moved here from schreier.C: December 20, 2015

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

static int schreier_vector_determine_depth_recursion(
	int n, int *pts, int *prev,
	int *depth, int *ancestor, int pos);


schreier_vector::schreier_vector()
{
	null();
}

schreier_vector::~schreier_vector()
{
	freeself();
}

void schreier_vector::null()
{
	nb_gen = 0;
	sv = NULL;
	number_of_orbits = -1;
}

void schreier_vector::freeself()
{
	if (sv) {
		FREE_int(sv);
	}
	null();
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
	if (f_v) {
		cout << "schreier_vector::init done" << endl;
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
	int i, n, nb = 0;
	int *pts;
	int *prev;
	//int *label;

	if (sv == NULL) {
		cout << "schreier_vector::count_number_of_orbits "
				"sv == NULL" << endl;
		exit(1);
	}
	n = sv[0];
	pts = sv + 1;
	prev = pts + n;
	for (i = 0; i < n; i++) {
		if (prev[i] == -1) {
			nb++;
			}
		}
	return nb;
}

int schreier_vector::determine_depth_recursion(
	int n, int *pts, int *prev,
	int *depth, int *ancestor, int pos)
{
	int pt, pt_loc, d;

	pt = prev[pos];
	if (pt == -1) {
		depth[pos] = 0;
		ancestor[pos] = pts[pos];
		return 0;
		}
	if (!int_vec_search(pts, n, pt, pt_loc)) {
		int i;

		cout << "schreier_vector::determine_depth_recursion, "
				"fatal: did not find pt" << endl;
		cout << "pt = " << pt << endl;
		cout << "vector of length " << n << endl;
		int_vec_print(cout, pts, n);
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
	action_on_factor_space *AF,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_trivial_group;
	int n;
	int *pts;
	int *prev;
	int *label;
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

	if (f_v) {
		cout << "schreier_vector::relabel_points" << endl;
		}
	if (nb_gen == 0) {
		f_trivial_group = TRUE;
	} else {
		f_trivial_group = FALSE;
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
	prev = pts + n;
	label = prev + n;

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
			if (FALSE) {
				cout << "i=" << i << " pt=" << pt
						<< " pre=" << pre << " q=" << q << endl;
				}
			new_pts[i] = q;
			}
		int_vec_heapsort(new_pts, n);
		for (i = 0; i < n + 1; i++) {
			sv[i] = new_sv[i];
			}
		FREE_int(new_sv);
		return;
		}


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
		int_vec_heapsort(old_orbit_reps, nb_old_orbit_reps);
		int_vec_print(cout, old_orbit_reps, nb_old_orbit_reps);
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
			if (int_vec_search(old_orbit_reps,
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
		if (FALSE) {
			cout << "i=" << i << " pt=" << pt << endl;
			}
		pre = AF->preimage(pt, 0/*verbose_level - 3*/);
		if (FALSE) {
			cout << "pre=" << pre << endl;
			}
		q = AF->project_onto_Gauss_reduced_vector(
				pre, 0 /*verbose_level - 2*/);
		if (FALSE) {
			if (int_vec_search(old_orbit_reps,
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
			if (int_vec_search(old_orbit_reps,
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
	int_vec_heapsort_with_log(new_pts_sorted, perm, n);
	if (f_vv) {
		cout << "schreier_vector::relabel_points "
				"after sorting:" << endl;
		cout << "i : pts[i] : new_pts_sorted[i] : perm[i]" << endl;
		for (i = 0; i < n; i++) {
			if (int_vec_search(old_orbit_reps,
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
				"old / n e w schreier vector:" << endl;
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
				"n e w schreier vector created" << endl;
		cout << "schreier_vector::relabel_points done" << endl;
		}
}

void schreier_vector::orbit_of_point(
		int pt, int *&orbit_elts, int &orbit_len,
		int verbose_level)
{
	int i, idx;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "schreier_vector::orbit_of_point "
				"pt=" << pt << endl;
		}
	int n;
	int *pts;
	int *prev;
	int *label;
	int *depth;
	int *ancestor;

	int *orbit_elt_idx;

	n = sv[0];
	pts = sv + 1;
	prev = pts + n;
	label = prev + n;
	if (f_v) {
		cout << "schreier_vector::orbit_of_point "
				"schreier vector of length " << n << endl;
		}

	if (!int_vec_search(pts, n, pt, idx)) {
		cout << "schreier_vector::orbit_of_point "
				"fatal: point " << pt << " not found" << endl;
		exit(1);
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
				"determining depth using schreier_vector_determine_"
				"depth_recursion" << endl;
		}
	for (i = 0; i < n; i++) {
		schreier_vector_determine_depth_recursion(n,
				pts, prev, depth, ancestor, i);
		}
	if (f_vv) {
		cout << "schreier_vector::orbit_of_point "
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
	orbit_len = 0;
	for (i = 0; i < n; i++) {
		if (ancestor[i] == pt) {
			orbit_elt_idx[orbit_len++] = i;
			}
		}
	if (f_v) {
		cout << "schreier_vector::orbit_of_point "
				"found orbit of length " << orbit_len << endl;
		}
	orbit_elts = NEW_int(orbit_len);
	for (i = 0; i < orbit_len; i++) {
		orbit_elts[i] = pts[orbit_elt_idx[i]];
		}
	if (f_vv) {
		cout << "schreier_vector::orbit_of_point "
				"the points in the orbit are: ";
		int_vec_print(cout, orbit_elts, orbit_len);
		cout << endl;
		}
	if (orbit_elts[0] != pt) {
		cout << "schreier_vector::orbit_of_point "
				"fatal: orbit_elts[0] != pt" << endl;
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

void schreier_vector::init_from_schreier(schreier *S,
	int f_trivial_group, int verbose_level)
// allocated and creates array sv[size] using NEW_int
// where size is n + 1 if  f_trivial_group is TRUE
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



// #############################################################################
// global functions:
// #############################################################################


static int schreier_vector_determine_depth_recursion(
	int n, int *pts, int *prev,
	int *depth, int *ancestor, int pos)
{
	int pt, pt_loc, d;

	pt = prev[pos];
	if (pt == -1) {
		depth[pos] = 0;
		ancestor[pos] = pts[pos];
		return 0;
		}
	if (!int_vec_search(pts, n, pt, pt_loc)) {
		int i;

		cout << "schreier_vector_determine_depth_recursion, "
				"fatal: did not find pt" << endl;
		cout << "pt = " << pt << endl;
		cout << "vector of length " << n << endl;
		int_vec_print(cout, pts, n);
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
		d = schreier_vector_determine_depth_recursion(n,
				pts, prev, depth, ancestor, pt_loc) + 1;
		}
	depth[pos] = d;
	ancestor[pos] = ancestor[pt_loc];
	return d;
}



