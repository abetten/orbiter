// schreier_vector.C
//
// Anton Betten
// moved here from schreier.C: December 20, 2015

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"


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
	number_of_orbits = count_number_of_orbits();
	if (f_v) {
		cout << "schreier_vector::init done" << endl;
	}
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



// #############################################################################
// global functions:
// #############################################################################


int schreier_vector_coset_rep_inv_general(
	action *A,
	int *sv, int *hdl_gen, int pt, 
	int &pt0, int *cosetrep,
	int *Elt1, int *Elt2, int *Elt3,
	int f_trivial_group, int f_check_image,
	int f_allow_failure,
	int verbose_level)
// determines pt0 to be the first point of the orbit containing pt.
// cosetrep will be a group element that maps pt to pt0.
{
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "schreier_vector_coset_rep_inv "
				"tracing point pt" << endl;
		}
	A->element_one(cosetrep, 0);
	
	//cout << "schreier_vector_coset_rep_inv f_compact="
	//<< f_compact << endl;
	ret = schreier_vector_coset_rep_inv_compact_general(A,
		sv, hdl_gen, pt, pt0,
		cosetrep, Elt1, Elt2, Elt3, 
		f_trivial_group, f_check_image,
		f_allow_failure, verbose_level - 1);
	if (f_v) {
		if (ret) {
			cout << "schreier_vector_coset_rep_inv_general "
					"done " << pt << "->" << pt0 << endl;
			}
		else {
			cout << "schreier_vector_coset_rep_inv_general "
					"failure to find point" << endl;
			}
		}
	return ret;
}

int schreier_vector_coset_rep_inv_compact_general(
	action *A,
	int *sv, int *hdl_gen, int pt, 
	int &pt0, int *cosetrep, int *Elt1, int *Elt2, int *Elt3, 
	int f_trivial_group, int f_check_image, 
	int f_allow_failure, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int hdl, pt_loc, pr, la, n;

	if (f_v) {
		cout << "schreier_vector_coset_rep_inv_compact_general "
				"tracing point " << pt << endl;
		}
	
	//cout << "schreier_vector_coset_rep_inv_compact_general "
	//"pt = " << pt << endl;
	n = sv[0];
	if (!int_vec_search(sv + 1, sv[0], pt, pt_loc)) {
		if (f_allow_failure) {
			return FALSE;
			}
		else {
			cout << "schreier_vector_coset_rep_inv_compact_general, "
					"did not find pt" << endl;
			cout << "pt = " << pt << endl;
			cout << "vector of length " << n << endl;
			int_vec_print(cout, sv + 1, n);
			cout << endl;
			exit(1);
			}
		}
	if (f_trivial_group) {
		pt0 = pt;
		return TRUE;
		}
	pr = sv[1 + n + pt_loc];
	la = sv[1 + 2 * n + pt_loc];
	if (pr != -1) {
		
		if (f_v) {
			cout << "prev = " << pr << " label = " << la << endl;
			}
		hdl = hdl_gen[la];
		A->element_retrieve(hdl, Elt1, 0);
		//cout << "retrieving generator " << gen_idx << endl;
		//A->element_print_verbose(Elt1, cout);
		A->element_invert(Elt1, Elt2, 0);
		
		if (f_check_image) {
			int prev;
			
			prev = A->element_image_of(pt, Elt2, 0);
		
			//cout << "prev = " << prev << endl;
			if (pr != prev) {
				cout << "schreier_vector_coset_rep_inv_compact_general: "
						"pr != prev" << endl;
				cout << "pr = " << pr << endl;
				cout << "prev = " << prev << endl;
				exit(1);
				}
			}
		
		A->element_mult(cosetrep, Elt2, Elt3, 0);
		A->element_move(Elt3, cosetrep, 0);
		
		if (!schreier_vector_coset_rep_inv_compact_general(
			A, sv, hdl_gen, pr, pt0,
			cosetrep, Elt1, Elt2, Elt3, 
			FALSE /* f_trivial_group */, f_check_image, 
			f_allow_failure, verbose_level)) {
			return FALSE;
			}

		}
	else {
		if (f_v) {
			cout << "prev = -1" << endl;
			}
		pt0 = pt;
		}
	return TRUE;
}



void schreier_vector_coset_rep_inv(
	action *A, int *sv, int *hdl_gen, int pt,
	int &pt0, int *cosetrep, int *Elt1, int *Elt2, int *Elt3, 
	int f_trivial_group, int f_compact,
	int f_check_image, int verbose_level)
// determines pt0 to be the first point of the orbit containing pt.
// cosetrep will be a group element that maps pt to pt0.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier_vector_coset_rep_inv "
				"tracing point pt" << endl;
		}
	A->element_one(cosetrep, 0);
	
	//cout << "schreier_vector_coset_rep_inv "
	//"f_compact=" << f_compact << endl;
	if (f_compact) {
		schreier_vector_coset_rep_inv_compact(
			A, sv, hdl_gen, pt, pt0,
			cosetrep, Elt1, Elt2, Elt3, 
			f_trivial_group, f_check_image, verbose_level - 1);
		}
	else {
		schreier_vector_coset_rep_inv1(A, sv, hdl_gen,
				pt, pt0, cosetrep, Elt1, Elt2, Elt3);
		}
	if (f_v) {
		cout << "schreier_vector_coset_rep_inv "
				"done " << pt << "->" << pt0 << endl;
		}
}

void schreier_vector_coset_rep_inv_compact(
	action *A, int *sv, int *hdl_gen, int pt,
	int &pt0, int *cosetrep, int *Elt1, int *Elt2, int *Elt3, 
	int f_trivial_group, int f_check_image,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int hdl, pt_loc, pr, la, n;

	if (f_v) {
		cout << "schreier_vector_coset_rep_inv_compact "
				"tracing point " << pt << endl;
		}
	
	//cout << "schreier_vector_coset_rep_inv_compact "
	//"pt = " << pt << endl;
	n = sv[0];
	if (!int_vec_search(sv + 1, sv[0], pt, pt_loc)) {
		cout << "schreier_vector_coset_rep_inv_compact, "
				"did not find pt" << endl;
		cout << "pt = " << pt << endl;
		cout << "vector of length " << n << endl;
		int_vec_print(cout, sv + 1, n);
		cout << endl;
		exit(1);
		}
	if (f_trivial_group) {
		pt0 = pt;
		return;
		}
	pr = sv[1 + n + pt_loc];
	la = sv[1 + 2 * n + pt_loc];
	if (pr != -1) {
		
		if (f_v) {
			cout << "prev = " << pr << " label = " << la << endl;
			}
		hdl = hdl_gen[la];
		A->element_retrieve(hdl, Elt1, 0);
		//cout << "retrieving generator " << gen_idx << endl;
		//A->element_print_verbose(Elt1, cout);
		A->element_invert(Elt1, Elt2, 0);
		
		if (f_check_image) {
			int prev;
			
			prev = A->element_image_of(pt, Elt2, 0);
		
			//cout << "prev = " << prev << endl;
			if (pr != prev) {
				cout << "schreier_vector_coset_rep_inv_compact: "
						"pr != prev" << endl;
				cout << "pr = " << pr << endl;
				cout << "prev = " << prev << endl;
				exit(1);
				}
			}
		
		A->element_mult(cosetrep, Elt2, Elt3, 0);
		A->element_move(Elt3, cosetrep, 0);
		
		schreier_vector_coset_rep_inv_compact(
			A, sv, hdl_gen, pr, pt0,
			cosetrep, Elt1, Elt2, Elt3, 
			FALSE /* f_trivial_group */, f_check_image,
			verbose_level);

		}
	else {
		if (f_v) {
			cout << "prev = -1" << endl;
			}
		pt0 = pt;
		}
}

void schreier_vector_coset_rep_inv1(
	action *A, int *sv, int *hdl_gen, int pt,
	int &pt0, int *cosetrep, int *Elt1, int *Elt2, int *Elt3)
{
	int gen_idx, hdl, prev;
	
	//cout << "schreier_vector_coset_rep_inv1 pt = " << pt << endl;
	if (sv[2 * pt + 0] != -1) {
		
		gen_idx = sv[2 * pt + 1];
		hdl = hdl_gen[gen_idx];
		A->element_retrieve(hdl, Elt1, 0);
		//cout << "retrieving generator " << gen_idx << endl;
		//A->element_print_verbose(Elt1, cout);
		A->element_invert(Elt1, Elt2, 0);
		
		prev = A->element_image_of(pt, Elt2, 0);
		
		//cout << "prev = " << prev << endl;
		if (prev != sv[2 * pt + 0]) {
			cout << "prev != sv[2 * pt + 0]" << endl;
			cout << "prev = " << prev << endl;
			cout << "sv[2 * pt + 0] = " << sv[2 * pt + 0] << endl;
			exit(1);
			}
		
		A->element_mult(cosetrep, Elt2, Elt3, 0);
		A->element_move(Elt3, cosetrep, 0);
		
		schreier_vector_coset_rep_inv1(A, sv, hdl_gen,
				prev, pt0, cosetrep, Elt1, Elt2, Elt3);

		}
	else {
		pt0 = pt;
		}
}



void schreier_vector_print(int *sv)
{
	int i, n;
	int *pts;
	int *prev;
	int *label;

	n = sv[0];
	pts = sv + 1;
	prev = pts + n;
	label = prev + n;
	cout << "schreier vector of length " << n << ":" << endl;
	if (n >= 100) {
		cout << "too big to print" << endl;
		return;
		}
	for (i = 0; i < n; i++) {
		cout 
			<< setw(5) << i << " : " 
			<< setw(5) << pts[i] << " : " 
			<< setw(5) << prev[i] << " : " 
			<< setw(5) << label[i] 
			<< endl;
		}
}

void schreier_vector_print_tree(int *sv, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, n;
	int *pts;
	int *prev;
	//int *label;
	int *Depth;
	int d, s;
	int /*pt,*/ pr; //, la;
	//int pt1, pr1;

	if (f_v) {
		cout << "schreier_vector_print_tree" << endl;
		}
	n = sv[0];
	pts = sv + 1;
	prev = pts + n;
	//label = prev + n;
	if (f_v) {
		cout << "schreier vector of length " << n << ":" << endl;
		}

	Depth = NEW_int(n);
	int_vec_zero(Depth, n);
	for (i = 0; i < n; i++) {
		//pt = pts[i];
		pr = prev[i];
		if (Depth[i] > 0) {
			continue;
			}
		if (pr == -1) {
			Depth[i] = 1;
			}
		else {
			d = schreier_vector_compute_depth_recursively(
					n, Depth, pts, prev, pr);
			Depth[i] = d + 1;
			}
		}


	s = 0;
	for (i = 0; i < n; i++) {
		//pt = pts[i];
		pr = prev[i];
		//la = label[i];
		d = Depth[i];

#if 0
		if (pr == -1) {
			continue;
			}
#endif
#if 0
		pt1 = i;
		if (!int_vec_search(pts, n, pr, pr1)) {
			cout << "schreier_vector_print_tree, did not find pr" << endl;
			exit(1);
			}
		cout << pr1 << "," << pt1 << "," << la << endl;
#endif


		//cout << pr << "," << pt << "," << d << endl;

		s += d;
		}

	double avg;

	avg = (double) s / (double) n;
	cout << "total depth is " << s << " for " << n
			<< " nodes, average depth is " << avg << endl;
	

	FREE_int(Depth);
	if (f_v) {
		cout << "schreier_vector_print_tree done" << endl;
		}
}

int schreier_vector_compute_depth_recursively(int n,
		int *Depth, int *pts, int *prev, int pt)
{
	int pos, pr, d;
	
	if (!int_vec_search(pts, n, pt, pos)) {
		cout << "schreier_vector_compute_depth_recursively, "
				"did not find pt" << endl;
		exit(1);
		}
	if (Depth[pos] > 0) {
		//cout << "depth of " << pt << " is " << Depth[pos] << endl;
		return Depth[pos];
		}
	pr = prev[pos];
	if (pr == -1) {
		Depth[pos] = 1;
		//cout << "depth of " << pt << " is " << pt << endl;
		return 1;
		}
	else {
		d = schreier_vector_compute_depth_recursively(n,
				Depth, pts, prev, pr);
		Depth[pos] = d + 1;
		//cout << "depth of " << pt << " is " << d + 1 << endl;
		return d + 1;
		}
}

int sv_number_of_orbits(int *sv)
{
	int i, n, nb = 0;
	int *pts;
	int *prev;
	//int *label;

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

void analyze_schreier_vector(int *sv, int verbose_level)
// we assume that the group is not trivial
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int n, i;
	int *depth;
	int *ancestor;
	int *pts;
	int *prev;
	int *label;
	double avg;

	if (f_v) {
		cout << "analyze_schreier_vector" << endl;
		}
	n = sv[0];
	if (f_v) {
		cout << "n=" << n << endl;
		}
	depth = NEW_int(n);	
	ancestor = NEW_int(n);	
	pts = sv + 1;
	prev = pts + n;
	label = prev + n;
	for (i = 0; i < n; i++) {
		depth[i] = -1;
		ancestor[i] = -1;
		}
	if (f_vv) {
		cout << "determining depth using schreier_vector_"
				"determine_depth_recursion" << endl;
		}
	for (i = 0; i < n; i++) {
		schreier_vector_determine_depth_recursion(n,
				pts, prev, depth, ancestor, i);
		}
	if (f_vv) {
		cout << "determining depth using schreier_vector_"
				"determine_depth_recursion done" << endl;
		}
	if (f_vv) {
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
	classify C;
	int *data1;

	data1 = NEW_int(n);

	C.init(ancestor, n, FALSE, verbose_level - 2);
	cout << "orbits:" << endl;
	C.print(FALSE /*f_backwards*/);
	int t, f, l, j;

	cout << "orbit : length : average depth" << endl;
	for (t = 0; t < C.nb_types; t++) {
		if (f_vv) {
			cout << "type " << t << ":" << endl;
			}
		f = C.type_first[t];
		l = C.type_len[t];
		if (f_vv) {
			for (j = 0; j < l; j++) {
				i = C.sorting_perm_inv[f + j];
				cout << i << " ";
				}
			cout << endl;
			}
		for (j = 0; j < l; j++) {
			i = C.sorting_perm_inv[f + j];
			data1[j] = depth[i];
			}
		if (FALSE) {
			cout << "depth vector for orbit " << t << ":" << endl;
			int_vec_print(cout, data1, l);
			cout << endl;
			}
		classify C2;
		C2.init(data1, l, FALSE, verbose_level - 2);
		if (f_vv) {
			cout << "depth multiplicity for orbit "
					<< t << ":" << endl;
			C2.print(FALSE /*f_backwards*/);
			}
		avg = C2.average();
		if (f_vv) {
			cout << "average depth is " << avg << endl;
			}
		if (f_v) {
			cout << setw(5) << i << " : " << setw(5)
					<< l << " : " << avg << " : ";
			C2.print(FALSE /*f_backwards*/);
			}
		}
	FREE_int(depth);
	FREE_int(ancestor);
	FREE_int(data1);
}

int schreier_vector_determine_depth_recursion(
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


