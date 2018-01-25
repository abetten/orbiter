// schreier_vector.C
//
// Anton Betten
// moved here from schreier.C: December 20, 2015

#include "galois.h"
#include "action.h"

INT schreier_vector_coset_rep_inv_general(action *A, 
	INT *sv, INT *hdl_gen, INT pt, 
	INT &pt0, INT *cosetrep, INT *Elt1, INT *Elt2, INT *Elt3, 
	INT f_trivial_group, INT f_check_image, INT f_allow_failure, INT verbose_level)
// determines pt0 to be the first point of the orbit containing pt.
// cosetrep will be a group element that maps pt to pt0.
{
	INT f_v = (verbose_level >= 1);
	INT ret;

	if (f_v) {
		cout << "schreier_vector_coset_rep_inv tracing point pt" << endl;
		}
	A->element_one(cosetrep, 0);
	
	//cout << "schreier_vector_coset_rep_inv f_compact=" << f_compact << endl;
	ret = schreier_vector_coset_rep_inv_compact_general(A, sv, hdl_gen, pt, pt0, 
		cosetrep, Elt1, Elt2, Elt3, 
		f_trivial_group, f_check_image, f_allow_failure, verbose_level - 1);
	if (f_v) {
		if (ret) {
			cout << "schreier_vector_coset_rep_inv_general done " << pt << "->" << pt0 << endl;
			}
		else {
			cout << "schreier_vector_coset_rep_inv_general failure to find point" << endl;
			}
		}
	return ret;
}

INT schreier_vector_coset_rep_inv_compact_general(action *A, 
	INT *sv, INT *hdl_gen, INT pt, 
	INT &pt0, INT *cosetrep, INT *Elt1, INT *Elt2, INT *Elt3, 
	INT f_trivial_group, INT f_check_image, 
	INT f_allow_failure, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT hdl, pt_loc, pr, la, n;

	if (f_v) {
		cout << "schreier_vector_coset_rep_inv_compact_general tracing point " << pt << endl;
		}
	
	//cout << "schreier_vector_coset_rep_inv_compact_general pt = " << pt << endl;
	n = sv[0];
	if (!INT_vec_search(sv + 1, sv[0], pt, pt_loc)) {
		if (f_allow_failure) {
			return FALSE;
			}
		else {
			cout << "schreier_vector_coset_rep_inv_compact_general, did not find pt" << endl;
			cout << "pt = " << pt << endl;
			cout << "vector of length " << n << endl;
			INT_vec_print(cout, sv + 1, n);
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
			INT prev;
			
			prev = A->element_image_of(pt, Elt2, 0);
		
			//cout << "prev = " << prev << endl;
			if (pr != prev) {
				cout << "schreier_vector_coset_rep_inv_compact_general: pr != prev" << endl;
				cout << "pr = " << pr << endl;
				cout << "prev = " << prev << endl;
				exit(1);
				}
			}
		
		A->element_mult(cosetrep, Elt2, Elt3, 0);
		A->element_move(Elt3, cosetrep, 0);
		
		if (!schreier_vector_coset_rep_inv_compact_general(A, sv, hdl_gen, pr, pt0, 
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



void schreier_vector_coset_rep_inv(action *A, INT *sv, INT *hdl_gen, INT pt, 
	INT &pt0, INT *cosetrep, INT *Elt1, INT *Elt2, INT *Elt3, 
	INT f_trivial_group, INT f_compact, INT f_check_image, INT verbose_level)
// determines pt0 to be the first point of the orbit containing pt.
// cosetrep will be a group element that maps pt to pt0.
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier_vector_coset_rep_inv tracing point pt" << endl;
		}
	A->element_one(cosetrep, 0);
	
	//cout << "schreier_vector_coset_rep_inv f_compact=" << f_compact << endl;
	if (f_compact) {
		schreier_vector_coset_rep_inv_compact(A, sv, hdl_gen, pt, pt0, 
			cosetrep, Elt1, Elt2, Elt3, 
			f_trivial_group, f_check_image, verbose_level - 1);
		}
	else {
		schreier_vector_coset_rep_inv1(A, sv, hdl_gen, pt, pt0, cosetrep, Elt1, Elt2, Elt3);
		}
	if (f_v) {
		cout << "schreier_vector_coset_rep_inv done " << pt << "->" << pt0 << endl;
		}
}

void schreier_vector_coset_rep_inv_compact(action *A, INT *sv, INT *hdl_gen, INT pt, 
	INT &pt0, INT *cosetrep, INT *Elt1, INT *Elt2, INT *Elt3, 
	INT f_trivial_group, INT f_check_image, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT hdl, pt_loc, pr, la, n;

	if (f_v) {
		cout << "schreier_vector_coset_rep_inv_compact tracing point " << pt << endl;
		}
	
	//cout << "schreier_vector_coset_rep_inv_compact pt = " << pt << endl;
	n = sv[0];
	if (!INT_vec_search(sv + 1, sv[0], pt, pt_loc)) {
		cout << "schreier_vector_coset_rep_inv_compact, did not find pt" << endl;
		cout << "pt = " << pt << endl;
		cout << "vector of length " << n << endl;
		INT_vec_print(cout, sv + 1, n);
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
			INT prev;
			
			prev = A->element_image_of(pt, Elt2, 0);
		
			//cout << "prev = " << prev << endl;
			if (pr != prev) {
				cout << "schreier_vector_coset_rep_inv_compact: pr != prev" << endl;
				cout << "pr = " << pr << endl;
				cout << "prev = " << prev << endl;
				exit(1);
				}
			}
		
		A->element_mult(cosetrep, Elt2, Elt3, 0);
		A->element_move(Elt3, cosetrep, 0);
		
		schreier_vector_coset_rep_inv_compact(A, sv, hdl_gen, pr, pt0, 
			cosetrep, Elt1, Elt2, Elt3, 
			FALSE /* f_trivial_group */, f_check_image, verbose_level);

		}
	else {
		if (f_v) {
			cout << "prev = -1" << endl;
			}
		pt0 = pt;
		}
}

void schreier_vector_coset_rep_inv1(action *A, INT *sv, INT *hdl_gen, INT pt, 
	INT &pt0, INT *cosetrep, INT *Elt1, INT *Elt2, INT *Elt3)
{
	INT gen_idx, hdl, prev;
	
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
		
		schreier_vector_coset_rep_inv1(A, sv, hdl_gen, prev, pt0, cosetrep, Elt1, Elt2, Elt3);

		}
	else {
		pt0 = pt;
		}
}



void schreier_vector_print(INT *sv)
{
	INT i, n;
	INT *pts;
	INT *prev;
	INT *label;

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

void schreier_vector_print_tree(INT *sv, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, n;
	INT *pts;
	INT *prev;
	INT *label;
	INT *Depth;
	INT d, s;
	INT pt, pr, la;
	//INT pt1, pr1;

	if (f_v) {
		cout << "schreier_vector_print_tree" << endl;
		}
	n = sv[0];
	pts = sv + 1;
	prev = pts + n;
	label = prev + n;
	if (f_v) {
		cout << "schreier vector of length " << n << ":" << endl;
		}

	Depth = NEW_INT(n);
	INT_vec_zero(Depth, n);
	for (i = 0; i < n; i++) {
		pt = pts[i];
		pr = prev[i];
		if (Depth[i] > 0) {
			continue;
			}
		if (pr == -1) {
			Depth[i] = 1;
			}
		else {
			d = schreier_vector_compute_depth_recursively(n, Depth, pts, prev, pr);
			Depth[i] = d + 1;
			}
		}


	s = 0;
	for (i = 0; i < n; i++) {
		pt = pts[i];
		pr = prev[i];
		la = label[i];
		d = Depth[i];

#if 0
		if (pr == -1) {
			continue;
			}
#endif
#if 0
		pt1 = i;
		if (!INT_vec_search(pts, n, pr, pr1)) {
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
	cout << "total depth is " << s << " for " << n << " nodes, average depth is " << avg << endl;
	

	FREE_INT(Depth);
	if (f_v) {
		cout << "schreier_vector_print_tree done" << endl;
		}
}

INT schreier_vector_compute_depth_recursively(INT n, INT *Depth, INT *pts, INT *prev, INT pt)
{
	INT pos, pr, d;
	
	if (!INT_vec_search(pts, n, pt, pos)) {
		cout << "schreier_vector_compute_depth_recursively, did not find pt" << endl;
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
		d = schreier_vector_compute_depth_recursively(n, Depth, pts, prev, pr);
		Depth[pos] = d + 1;
		//cout << "depth of " << pt << " is " << d + 1 << endl;
		return d + 1;
		}
}

INT sv_number_of_orbits(INT *sv)
{
	INT i, n, nb = 0;
	INT *pts;
	INT *prev;
	//INT *label;

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

void analyze_schreier_vector(INT *sv, INT verbose_level)
// we assume that the group is not trivial
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT n, i;
	INT *depth;
	INT *ancestor;
	INT *pts;
	INT *prev;
	INT *label;
	double avg;

	if (f_v) {
		cout << "analyze_schreier_vector" << endl;
		}
	n = sv[0];
	if (f_v) {
		cout << "n=" << n << endl;
		}
	depth = NEW_INT(n);	
	ancestor = NEW_INT(n);	
	pts = sv + 1;
	prev = pts + n;
	label = prev + n;
	for (i = 0; i < n; i++) {
		depth[i] = -1;
		ancestor[i] = -1;
		}
	if (f_vv) {
		cout << "determining depth using schreier_vector_determine_depth_recursion" << endl;
		}
	for (i = 0; i < n; i++) {
		schreier_vector_determine_depth_recursion(n, pts, prev, depth, ancestor, i);
		}
	if (f_vv) {
		cout << "determining depth using schreier_vector_determine_depth_recursion done" << endl;
		}
	if (f_vv) {
		cout << "i : pts[i] : prev[i] : label[i] : depth[i] : ancestor[i]" << endl;
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
	INT *data1;

	data1 = NEW_INT(n);

	C.init(ancestor, n, FALSE, verbose_level - 2);
	cout << "orbits:" << endl;
	C.print(FALSE /*f_backwards*/);
	INT t, f, l, j;

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
			INT_vec_print(cout, data1, l);
			cout << endl;
			}
		classify C2;
		C2.init(data1, l, FALSE, verbose_level - 2);
		if (f_vv) {
			cout << "depth multiplicity for orbit " << t << ":" << endl;
			C2.print(FALSE /*f_backwards*/);
			}
		avg = C2.average();
		if (f_vv) {
			cout << "average depth is " << avg << endl;
			}
		if (f_v) {
			cout << setw(5) << i << " : " << setw(5) << l << " : " << avg << " : ";
			C2.print(FALSE /*f_backwards*/);
			}
		}
	FREE_INT(depth);
	FREE_INT(ancestor);
	FREE_INT(data1);
}

INT schreier_vector_determine_depth_recursion(INT n, INT *pts, INT *prev, 
	INT *depth, INT *ancestor, INT pos)
{
	INT pt, pt_loc, d;
	
	pt = prev[pos];
	if (pt == -1) {
		depth[pos] = 0;
		ancestor[pos] = pts[pos];
		return 0;
		}
	if (!INT_vec_search(pts, n, pt, pt_loc)) {
		INT i;
		
		cout << "schreier_vector_determine_depth_recursion, fatal: did not find pt" << endl;
		cout << "pt = " << pt << endl;
		cout << "vector of length " << n << endl;
		INT_vec_print(cout, pts, n);
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
		d = schreier_vector_determine_depth_recursion(n, pts, prev, depth, ancestor, pt_loc) + 1;
		}
	depth[pos] = d;
	ancestor[pos] = ancestor[pt_loc];
	return d;
}



