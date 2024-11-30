/*
 * projective_space2.cpp
 *
 *  Created on: Nov 3, 2019
 *      Author: anton
 */


#include "foundations.h"


using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace projective_geometry {







int projective_space::is_contained_in_Baer_subline(
	long int *pts, int nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int *subline;
	int sz;
	int i, idx, a;
	int ret = true;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space::is_contained_in_Baer_subline "
				"pts=" << endl;
		Lint_vec_print(cout, pts, nb_pts);
		cout << endl;
		cout << "computing Baer subline determined by the "
				"first three points:" << endl;
	}
	Baer_subline(pts, subline, sz, verbose_level - 2);
	if (f_vv) {
		cout << "projective_space::is_contained_in_Baer_subline "
				"The Baer subline is:" << endl;
		Lint_vec_print(cout, subline, sz);
		cout << endl;
	}
	Sorting.lint_vec_heapsort(subline, sz);
	for (i = 0; i < nb_pts; i++) {
		a = pts[i];
		if (!Sorting.lint_vec_search(subline, sz, a, idx, 0)) {
			ret = false;
			if (f_vv) {
				cout << "did not find " << i << "-th point " << a
						<< " in the list of points, hence not "
						"contained in Baer subline" << endl;
			}
			goto done;
		}

	}
done:
	FREE_lint(subline);

	return ret;
}


void projective_space::circle_type_of_line_subset(
	int *pts, int nb_pts, int *circle_type,
	int verbose_level)
// circle_type[nb_pts]
// this function is not used
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int *subline;
	long int subset[3];
	int idx_set[3];
	int sz;
	int i, idx, a, b;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space::circle_type_of_line_subset "
				"pts=" << endl;
		Int_vec_print(cout, pts, nb_pts);
		cout << endl;
		//cout << "computing Baer subline determined by
		//the first three points:" << endl;
	}
	for (i = 0; i < nb_pts; i++) {
		circle_type[i] = 0;
	}

	Combi.first_k_subset(idx_set, nb_pts, 3);
	do {
		for (i = 0; i < 3; i++) {
			subset[i] = pts[idx_set[i]];
		}
		Baer_subline(subset, subline, sz, verbose_level - 2);
		b = 0;
		Sorting.lint_vec_heapsort(subline, sz);
		for (i = 0; i < nb_pts; i++) {
			a = pts[i];
			if (Sorting.lint_vec_search(subline, sz, a, idx, 0)) {
				b++;
			}
		}


		if (f_v) {
			cout << "projective_space::circle_type_of_line_subset "
					"The Baer subline determined by " << endl;
			Lint_vec_print(cout, subset, 3);
			cout << " is ";
			Lint_vec_print(cout, subline, sz);
			cout << " which intersects in " << b << " points" << endl;
		}



		FREE_lint(subline);
		circle_type[b]++;
	} while (Combi.next_k_subset(idx_set, nb_pts, 3));

	if (f_vv) {
		cout << "projective_space::circle_type_of_line_subset "
				"circle_type before fixing =" << endl;
		Int_vec_print(cout, circle_type, nb_pts);
		cout << endl;
	}
	for (i = 4; i < nb_pts; i++) {
		a = Combi.int_n_choose_k(i, 3);
		if (circle_type[i] % a) {
			cout << "projective_space::circle_type_of_line_subset "
					"circle_type[i] % a" << endl;
			exit(1);
		}
		circle_type[i] /= a;
	}
	if (f_vv) {
		cout << "projective_space::circle_type_of_line_subset "
				"circle_type after fixing =" << endl;
		Int_vec_print(cout, circle_type, nb_pts);
		cout << endl;
	}
}



void projective_space::intersection_of_subspace_with_point_set(
	grassmann *G, int rk, long int *set, int set_size,
	long int *&intersection_set, int &intersection_set_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::intersection_of_subspace_"
				"with_point_set" << endl;
	}

	int h;
	int d = Subspaces->n + 1;
	int k = G->k;
	int *M;

	intersection_set = NEW_lint(set_size);
	M = NEW_int((k + 1) * d);
	intersection_set_size = 0;

	G->unrank_lint(rk, 0 /* verbose_level */);

	for (h = 0; h < set_size; h++) {
		Int_vec_copy(G->M, M, k * d);
		unrank_point(M + k * d, set[h]);
		if (Subspaces->F->Linear_algebra->rank_of_rectangular_matrix(M,
				k + 1, d, 0 /*verbose_level*/) == k) {
			intersection_set[intersection_set_size++] = set[h];
		}
	} // next h

	FREE_int(M);
	if (f_v) {
		cout << "projective_space::intersection_of_subspace_"
				"with_point_set done" << endl;
	}
}

void projective_space::intersection_of_subspace_with_point_set_rank_is_longinteger(
	grassmann *G, ring_theory::longinteger_object &rk,
	long int *set, int set_size,
	long int *&intersection_set, int &intersection_set_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::intersection_of_subspace_with_"
				"point_set_rank_is_longinteger" << endl;
	}
	int h;
	int d = Subspaces->n + 1;
	int k = G->k;
	int *M;

	intersection_set = NEW_lint(set_size);
	M = NEW_int((k + 1) * d);
	intersection_set_size = 0;

	G->unrank_longinteger(rk, 0 /* verbose_level */);

	for (h = 0; h < set_size; h++) {
		Int_vec_copy(G->M, M, k * d);
		unrank_point(M + k * d, set[h]);
		if (Subspaces->F->Linear_algebra->rank_of_rectangular_matrix(M,
				k + 1, d, 0 /*verbose_level*/) == k) {
			intersection_set[intersection_set_size++] = set[h];
		}
	} // next h

	FREE_int(M);
	if (f_v) {
		cout << "projective_space::intersection_of_subspace_with_"
				"point_set_rank_is_longinteger done" << endl;
	}
}

void projective_space::plane_intersection_invariant(
	grassmann *G,
	long int *set, int set_size,
	int *&intersection_type, int &highest_intersection_number,
	int *&intersection_matrix, int &nb_planes,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_object *R;
	long int **Pts_on_plane;
	int *nb_pts_on_plane;
	int nb_planes_total;
	int i, j, a, u, f, l, ii;

	if (f_v) {
		cout << "projective_space::plane_intersection_invariant" << endl;
	}
	if (f_v) {
		cout << "projective_space::plane_intersection_invariant "
				"before plane_intersection_type_fast" << endl;
	}
	plane_intersection_type_fast(
			G,
		set, set_size,
		R, Pts_on_plane, nb_pts_on_plane, nb_planes_total,
		verbose_level - 1);
	if (f_v) {
		cout << "projective_space::plane_intersection_invariant "
				"after plane_intersection_type_fast" << endl;
	}

	data_structures::tally C;
	int f_second = false;

	C.init(nb_pts_on_plane, nb_planes_total, f_second, 0);
	if (f_v) {
		cout << "projective_space::plane_intersection_invariant "
				"plane-intersection type: ";
		C.print(false /* f_backwards*/);
	}

	if (f_v) {
		cout << "The plane intersection type is (";
		C.print_bare(false /*f_backwards*/);
		cout << ")" << endl << endl;
	}
	f = C.type_first[C.nb_types - 1];
	highest_intersection_number = C.data_sorted[f];
	intersection_type = NEW_int(highest_intersection_number + 1);

	Int_vec_zero(intersection_type, highest_intersection_number + 1);

	for (i = 0; i < C.nb_types; i++) {
		f = C.type_first[i];
		l = C.type_len[i];
		a = C.data_sorted[f];
		intersection_type[a] = l;
	}
	f = C.type_first[C.nb_types - 1];
	nb_planes = C.type_len[C.nb_types - 1];

	int *Incma, *Incma_t, *IIt, *ItI;

	Incma = NEW_int(set_size * nb_planes);
	Incma_t = NEW_int(nb_planes * set_size);
	IIt = NEW_int(set_size * set_size);
	ItI = NEW_int(nb_planes * nb_planes);


	for (i = 0; i < set_size * nb_planes; i++) {
		Incma[i] = 0;
	}
	for (i = 0; i < nb_planes; i++) {
		ii = C.sorting_perm_inv[f + i];
		for (j = 0; j < nb_pts_on_plane[ii]; j++) {
			a = Pts_on_plane[ii][j];
			Incma[a * nb_planes + i] = 1;
		}
	}
	if (f_vv) {
		cout << "Incidence matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				Incma, set_size, nb_planes, nb_planes, 1);
	}
	for (i = 0; i < set_size; i++) {
		for (j = 0; j < set_size; j++) {
			a = 0;
			for (u = 0; u < nb_planes; u++) {
				a += Incma[i * nb_planes + u] *
						Incma_t[u * set_size + j];
			}
			IIt[i * set_size + j] = a;
		}
	}
	if (f_vv) {
		cout << "I * I^\\top = " << endl;
		Int_vec_print_integer_matrix_width(cout,
				IIt, set_size, set_size, set_size, 2);
	}
	for (i = 0; i < nb_planes; i++) {
		for (j = 0; j < nb_planes; j++) {
			a = 0;
			for (u = 0; u < set_size; u++) {
				a += Incma[u * nb_planes + i] *
						Incma[u * nb_planes + j];
			}
			ItI[i * nb_planes + j] = a;
		}
	}
	if (f_v) {
		cout << "I^\\top * I = " << endl;
		Int_vec_print_integer_matrix_width(cout,
				ItI, nb_planes, nb_planes, nb_planes, 3);
	}

	intersection_matrix = NEW_int(nb_planes * nb_planes);
	Int_vec_copy(ItI,
			intersection_matrix, nb_planes * nb_planes);

	FREE_int(Incma);
	FREE_int(Incma_t);
	FREE_int(IIt);
	FREE_int(ItI);


	for (i = 0; i < nb_planes_total; i++) {
		FREE_lint(Pts_on_plane[i]);
	}
	FREE_plint(Pts_on_plane);
	FREE_int(nb_pts_on_plane);
	FREE_OBJECTS(R);
	if (f_v) {
		cout << "projective_space::plane_intersection_invariant done" << endl;
	}
}


void projective_space::line_intersection_type(
	long int *set, int set_size, int threshold,
	other_geometry::intersection_type *&Int_type,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::line_intersection_type "
				"threshold = " << threshold << endl;
	}



	Int_type = NEW_OBJECT(other_geometry::intersection_type);

	if (f_v) {
		cout << "projective_space::line_intersection_type "
				"before Int_type->line_intersection_type_slow" << endl;
	}
	Int_type->line_intersection_type_slow(
		set, set_size, threshold,
		this,
		Subspaces->Grass_lines,
		verbose_level);
	if (f_v) {
		cout << "projective_space::line_intersection_type "
				"after Int_type->line_intersection_type_slow" << endl;
	}


	if (f_v) {
		cout << "projective_space::line_intersection_type "
				"before Int_type->compute_heighest_weight_objects" << endl;
	}
	Int_type->compute_heighest_weight_objects(verbose_level);
	if (f_v) {
		cout << "projective_space::line_intersection_type "
				"after Int_type->compute_heighest_weight_objects" << endl;
	}

	if (f_v) {
		cout << "projective_space::line_intersection_type "
				"done" << endl;
	}

}


void projective_space::plane_intersection_type(
	long int *set, int set_size, int threshold,
	other_geometry::intersection_type *&Int_type,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i, f, l, a;

	if (f_v) {
		cout << "projective_space::plane_intersection_type "
				"threshold = " << threshold << endl;
	}
#if 0
	if (f_v) {
		cout << "projective_space::plane_intersection_type "
				"before plane_intersection_type_fast" << endl;
	}
	plane_intersection_type_fast(G,
		set, set_size,
		R,
		Pts_on_plane, nb_pts_on_plane, nb_planes,
		verbose_level - 1);
	if (f_v) {
		cout << "projective_space::plane_intersection_type "
				"after plane_intersection_type_fast" << endl;
	}
	if (f_v) {
		cout << "projective_space::plane_intersection_type "
				"before plane_intersection_type_slow" << endl;
	}
	plane_intersection_type_slow(
		set, set_size, threshold,
		R,
		Pts_on_plane, nb_pts_on_plane, nb_planes,
		verbose_level);
	if (f_v) {
		cout << "projective_space::plane_intersection_type "
				"after plane_intersection_type_slow" << endl;
	}
#endif




	Int_type = NEW_OBJECT(other_geometry::intersection_type);

	if (f_v) {
		cout << "projective_space::plane_intersection_type "
				"before Int_type->plane_intersection_type_slow" << endl;
	}
	Int_type->plane_intersection_type_slow(
		set, set_size, threshold,
		this,
		Subspaces->Grass_planes,
		verbose_level);
	if (f_v) {
		cout << "projective_space::plane_intersection_type "
				"after Int_type->plane_intersection_type_slow" << endl;
	}


	if (f_v) {
		cout << "projective_space::plane_intersection_type "
				"before Int_type->compute_heighest_weight_objects" << endl;
	}
	Int_type->compute_heighest_weight_objects(verbose_level);
	if (f_v) {
		cout << "projective_space::plane_intersection_type "
				"after Int_type->compute_heighest_weight_objects" << endl;
	}

	if (f_v) {
		cout << "projective_space::plane_intersection_type "
				"done" << endl;
	}

}

int projective_space::plane_intersections(
	grassmann *G,
	long int *set, int set_size,
	ring_theory::longinteger_object *&R,
	data_structures::set_of_sets &SoS,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int **Pts_on_plane;
	int *nb_pts_on_plane;
	int nb_planes;
	int i;
	int ret;

	if (f_v) {
		cout << "projective_space::plane_intersections" << endl;
	}
	if (f_v) {
		cout << "projective_space::plane_intersections "
				"before plane_intersection_type_fast" << endl;
	}
	plane_intersection_type_fast(
			G,
		set, set_size,
		R, Pts_on_plane, nb_pts_on_plane, nb_planes,
		verbose_level - 1);
	if (f_v) {
		cout << "projective_space::plane_intersections "
				"after plane_intersection_type_fast, "
				"nb_planes = " << nb_planes << endl;
	}

	if (nb_planes) {
		if (f_v) {
			cout << "projective_space::plane_intersections "
					"before Sos.init" << endl;
		}
		SoS.init_with_Sz_in_int(
				set_size, nb_planes,
				Pts_on_plane, nb_pts_on_plane,
				verbose_level - 1);
		if (f_v) {
			cout << "projective_space::plane_intersections "
					"after Sos.init" << endl;
		}
		ret = true;
	}
	else {
		ret = false;
	}
	for (i = 0; i < nb_planes; i++) {
		FREE_lint(Pts_on_plane[i]);
	}
	FREE_plint(Pts_on_plane);
	FREE_int(nb_pts_on_plane);
	if (f_v) {
		cout << "projective_space::plane_intersections done" << endl;
	}
	return ret;
}


void projective_space::plane_intersection_type_fast(
	grassmann *G,
	long int *set, int set_size,
	ring_theory::longinteger_object *&R,
	long int **&Pts_on_plane, int *&nb_pts_on_plane, int &len,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = false; //(verbose_level >= 3);
	int idx;
	long int r, rk, rr, h, i, j, a, d, N_planes, N, N2, intersection_sz;

	int *Basis;
	int *Basis_save;
	int *f_subset_done;
	int *rank_idx;
	int *Pt_coords;

	int subset[3];
	int subset2[3];
	int subset3[3];
	ring_theory::longinteger_object plane_rk, aa;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;
	linear_algebra::linear_algebra LA;

	if (f_v) {
		cout << "projective_space::plane_intersection_type_fast" << endl;
	}

	LA.init(Subspaces->F, verbose_level);

	if (f_vv) {
		Reporting->print_set_numerical(cout, set, set_size);
	}

	if (!Sorting.test_if_set_with_return_value_lint(
			set, set_size)) {
		cout << "projective_space::plane_intersection_type_fast "
				"the input set if not a set" << endl;
		exit(1);
	}
	d = Subspaces->n + 1;
	N_planes = Subspaces->nb_rk_k_subspaces_as_lint(3);
	N = Combi.int_n_choose_k(set_size, 3);
		// N is the number of 3-subsets of the given set

	if (f_v) {
		cout << "N_planes=" << N_planes << endl;
		cout << "N=number of 3-subsets of the set=" << N << endl;
	}

	// allocate data that is returned:
	R = NEW_OBJECTS(ring_theory::longinteger_object, N);
	Pts_on_plane = NEW_plint(N);
	nb_pts_on_plane = NEW_int(N);

	// allocate temporary data:
	Basis = NEW_int(4 * d);
	Basis_save = NEW_int(4 * d);
	rank_idx = NEW_int(N);
	f_subset_done = NEW_int(N);
	Pt_coords = NEW_int(set_size * d);

	for (i = 0; i < N; i++) {
		f_subset_done[i] = false;
	}
	for (i = 0; i < set_size; i++) {
		unrank_point(Pt_coords + i * d, set[i]);
	}
	if (f_vv) {
		cout << "projective_space::plane_intersection_type_fast "
				"Pt_coords:" << endl;
		Int_matrix_print(Pt_coords, set_size, d);
	}


	len = 0;
	for (rk = 0; rk < N; rk++) {
		Combi.unrank_k_subset(
				rk, subset, set_size, 3);
		if (f_v3) {
			cout << rk << "-th subset ";
			Int_vec_print(cout, subset, 3);
			cout << endl;
		}
		if (f_subset_done[rk]) {
			if (f_v3) {
				cout << "skipping" << endl;
			}
			continue;
		}
		for (j = 0; j < 3; j++) {
			a = subset[j];
			//a = set[subset[j]];
			Int_vec_copy(Pt_coords + a * d, Basis + j * d, d);
			//unrank_point(Basis + j * d, a);
		}
		if (f_v3) {
			cout << "subset: ";
			Int_vec_print(cout, subset, 3);
			cout << " corresponds to Basis:" << endl;
			Int_matrix_print(Basis, 3, d);
		}
		r = Subspaces->F->Linear_algebra->rank_of_rectangular_matrix(
				Basis, 3, d, 0 /* verbose_level */);
		if (r < 3) {
			if (f_v3) {
				cout << "projective_space::plane_intersection_type_fast "
						"not independent, skip" << endl;
				cout << "subset: ";
				Int_vec_print(cout, subset, 3);
				cout << endl;
			}
			rank_idx[rk] = -1;
			continue;
		}
		G->rank_longinteger_here(
				Basis, plane_rk, 0 /* verbose_level */);
		if (f_v3) {
			cout << rk << "-th subset ";
			Int_vec_print(cout, subset, 3);
			cout << " plane_rk=" << plane_rk << endl;
		}

		if (Sorting.longinteger_vec_search(
				R, len, plane_rk, idx)) {
			//rank_idx[rk] = idx;
			// this case should never happen:
			cout << "projective_space::plane_intersection_type_fast "
					"longinteger_vec_search(R, len, plane_rk, idx) "
					"is true" << endl;
			exit(1);
		}
		else {
			if (f_v3) {
				cout << "plane_rk=" << plane_rk
						<< " was not found" << endl;
			}

			long int *pts_on_plane;

			pts_on_plane = NEW_lint(set_size);
			//if (f_v3) {
				//cout << "after allocating pts_on_plane,
				//plane_rk=" << plane_rk << endl;
				//}

			plane_rk.assign_to(aa);
			G->unrank_longinteger_here(Basis_save, aa,
					0 /* verbose_level */);
			if (f_v3) {
				cout << "after unrank " << plane_rk
						<< ", Basis:" << endl;
				Int_matrix_print(Basis_save, 3, d);
			}



			LA.intersect_with_subspace(
					Pt_coords, set_size /* nb_pts */,
					Basis_save, Basis, 3, d,
					pts_on_plane /* Intersection_idx */,
					intersection_sz,
					verbose_level);
			// Pt_coords[nb_pts * n]
			// Basis_save[m * n]
			// Basis[(m + 1) * n]
			// Intersection_idx[nb_pts]



			if (intersection_sz >= 3) {
				if (f_vv) {
					cout << "We found an " << intersection_sz << "-plane, "
							"its rank is " << plane_rk << endl;
					cout << "The ranks of points on that plane are : ";
					Lint_vec_print(cout, pts_on_plane, intersection_sz);
					cout << endl;
				}

				for (j = len; j > idx; j--) {
					R[j].swap_with(R[j - 1]);
					Pts_on_plane[j] = Pts_on_plane[j - 1];
					nb_pts_on_plane[j] = nb_pts_on_plane[j - 1];
				}
				for (j = 0; j < N; j++) {
					if (f_subset_done[j] && rank_idx[j] >= idx) {
						rank_idx[j]++;
					}
				}
				plane_rk.assign_to(R[idx]);
				if (f_vv) {
					cout << "after assign_to, "
							"plane_rk=" << plane_rk << endl;
				}
				rank_idx[rk] = idx;
				len++;




				N2 = Combi.int_n_choose_k(intersection_sz, 3);
				for (i = 0; i < N2; i++) {
					Combi.unrank_k_subset(i, subset2, intersection_sz, 3);
					for (h = 0; h < 3; h++) {
						subset3[h] = pts_on_plane[subset2[h]];
					}
					rr = Combi.rank_k_subset(subset3, set_size, 3);
					if (f_vv) {
						cout << i << "-th subset3 ";
						Int_vec_print(cout, subset3, 3);
						cout << " rr=" << rr << endl;
					}
					if (!f_subset_done[rr]) {
						f_subset_done[rr] = true;
						rank_idx[rr] = idx;
					}
					else if (rank_idx[rr] == -1) {
						rank_idx[rr] = idx;
					}
					else if (rank_idx[rr] != idx) {
						cout << "projective_space::plane_intersection_type_fast "
								"f_subset_done[rr] && "
								"rank_idx[rr] >= 0 && "
								"rank_idx[rr] != idx" << endl;
						exit(1);
					}
				}
				Pts_on_plane[idx] = pts_on_plane;
				nb_pts_on_plane[idx] = intersection_sz;
			}
			else {
				// now l <= 2, we skip those planes:

				FREE_lint(pts_on_plane);
				f_subset_done[rk] = true;
				rank_idx[rk] = -2;
			}
		} // else
	} // next rk

	FREE_int(Basis);
	FREE_int(Basis_save);
	FREE_int(f_subset_done);
	FREE_int(rank_idx);
	FREE_int(Pt_coords);
	if (f_v) {
		cout << "projective_space::plane_intersection_type_fast done" << endl;
	}
}

void projective_space::find_planes_which_intersect_in_at_least_s_points(
	long int *set, int set_size,
	int s,
	vector<int> &plane_ranks,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_v3 = (verbose_level >= 3);
	long int r, rk, i, u, d, N_planes;

	int *Basis;
	int *Basis_save;
	int *Coords;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space::find_planes_which_intersect_in_at_least_s_points" << endl;
	}
	if (f_vv) {
		Reporting->print_set_numerical(cout, set, set_size);
	}
	if (!Sorting.test_if_set_with_return_value_lint(set, set_size)) {
		cout << "projective_space::find_planes_which_intersect_in_at_least_s_points "
				"the input set if not a set" << endl;
		exit(1);
	}
	d = Subspaces->n + 1;
	N_planes = Subspaces->nb_rk_k_subspaces_as_lint(3);

	if (f_v) {
		cout << "N_planes=" << N_planes << endl;
	}

	// allocate temporary data:
	Basis = NEW_int(4 * d);
	Basis_save = NEW_int(4 * d);
	Coords = NEW_int(set_size * d);

	for (i = 0; i < set_size; i++) {
		unrank_point(Coords + i * d, set[i]);
	}
	if (f_vv) {
		cout << "projective_space::find_planes_which_intersect_in_at_least_s_points "
				"Coords:" << endl;
		Int_matrix_print(Coords, set_size, d);
	}

	int one_percent = 0;

	if (N_planes > 1000000) {
		one_percent = N_planes / 100;
	}
	for (rk = 0; rk < N_planes; rk++) {

		if (one_percent > 0) {
			if ((rk % one_percent) == 0) {
				cout << "projective_space::find_planes_which_intersect_in_at_least_s_points "
						<< rk << " / " << N_planes << " which is "
						<< rk / one_percent << " percent done" << endl;
			}
		}
		Subspaces->Grass_planes->unrank_lint_here(
				Basis_save, rk, 0 /* verbose_level */);
		//int_vec_copy(G->M, Basis_save, 3 * d);

		int nb_pts_on_plane = 0;

		for (u = 0; u < set_size; u++) {
			Int_vec_copy(Basis_save, Basis, 3 * d);
			Int_vec_copy(Coords + u * d, Basis + 3 * d, d);
			r = Subspaces->F->Linear_algebra->rank_of_rectangular_matrix(
					Basis,
					4, d, 0 /* verbose_level */);
			if (r < 4) {
				nb_pts_on_plane++;
			}
		}

		if (nb_pts_on_plane >= s) {
			plane_ranks.push_back(rk);
		}
	} // rk
	if (f_v) {
		cout << "projective_space::find_planes_which_intersect_in_at_least_s_points "
				"we found "
				<< plane_ranks.size() << " planes which intersect "
						"in at least " << s << " points" << endl;
	}

	FREE_int(Basis);
	FREE_int(Basis_save);
	FREE_int(Coords);
	if (f_v) {
		cout << "projective_space::find_planes_which_intersect_in_at_least_s_points "
				"done" << endl;
	}
}


void projective_space::line_plane_incidence_matrix_restricted(
	long int *Lines, int nb_lines, int *&M, int &nb_planes,
	int verbose_level)
// requires n >= 3
{
	int f_v = (verbose_level >= 1);
	int *the_lines;
	int line_sz;
	int *Basis; // 3 * (n + 1)
	int *Work; // 5 * (n + 1)
	int i, j;

	if (f_v) {
		cout << "projective_space::line_plane_incidence_matrix_restricted" << endl;
	}
	if (Subspaces->n <= 2) {
		cout << "projective_space::line_plane_incidence_matrix_restricted n <= 2" << endl;
		exit(1);
	}
	line_sz = 2 * (Subspaces->n + 1);
	nb_planes = Subspaces->Nb_subspaces[2];

	M = NEW_int(nb_lines * nb_planes);
	Basis = NEW_int(3 * (Subspaces->n + 1));
	Work = NEW_int(5 * (Subspaces->n + 1));
	the_lines = NEW_int(nb_lines * line_sz);


	Int_vec_zero(M, nb_lines * nb_planes);
	for (i = 0; i < nb_lines; i++) {
		unrank_line(the_lines + i * line_sz, Lines[i]);
	}
	for (j = 0; j < nb_planes; j++) {
		unrank_plane(Basis, j);
		for (i = 0; i < nb_lines; i++) {
			Int_vec_copy(Basis, Work, 3 * (Subspaces->n + 1));
			Int_vec_copy(the_lines + i * line_sz,
					Work + 3 * (Subspaces->n + 1), line_sz);
			if (Subspaces->F->Linear_algebra->Gauss_easy(
					Work, 5, Subspaces->n + 1) == 3) {
				M[i * nb_planes + j] = 1;
			}
		}
	}
	FREE_int(Work);
	FREE_int(Basis);
	FREE_int(the_lines);
	if (f_v) {
		cout << "projective_space::line_plane_incidence_matrix_restricted done" << endl;
	}
}






#if 0
void projective_space::decomposition_from_set_partition(
	int nb_subsets, int *sz, int **subsets,
	incidence_structure *&Inc,
	data_structures::partitionstack *&Stack,
	int verbose_level)
// this function is not used
// this function used to be called decomposition
{
	int f_v = (verbose_level >= 1);
	int nb_pts, nb_lines;
	int *Mtx;
	int *part;
	int i, j, level;

	if (f_v) {
		cout << "projective_space::decomposition_from_set_partition" << endl;
	}
	nb_pts = N_points;
	nb_lines = N_lines;
	if (f_v) {
		cout << "m = N_points = " << nb_pts << endl;
		cout << "n = N_lines = " << nb_lines << endl;
	}
	part = NEW_int(nb_subsets);
	Mtx = NEW_int(nb_pts * nb_lines);
	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < nb_lines; j++) {
			Mtx[i * nb_lines + j] = is_incident(i, j);
		}
	}
	Inc = NEW_OBJECT(incidence_structure);
	Inc->init_by_matrix(nb_pts, nb_lines, Mtx, verbose_level - 2);




	int /*ht0,*/ c, l;

	Stack = NEW_OBJECT(data_structures::partitionstack);
	Stack->allocate(nb_pts + nb_lines, 0);
	Stack->subset_continguous(Inc->nb_points(), Inc->nb_lines());
	Stack->split_cell(0);



	for (level = 0; level < nb_subsets; level++) {


		//ht0 = Stack->ht;

		if (sz[level]) {
			c = Stack->cellNumber[Stack->invPointList[subsets[level][0]]];
			l = Stack->cellSize[c];
			if (sz[level] < l) {
				Stack->split_cell(subsets[level], sz[level], 0);
				part[level] = Stack->ht - 1;
			}
			else {
				part[level] = c;
			}
		}
		else {
			part[level] = -1;
		}


		if (f_v) {
			cout << "projective_space::decomposition level " << level
					<< " : partition stack after splitting:" << endl;
			Stack->print(cout);
			cout << "i : part[i]" << endl;
			for (i = 0; i < nb_subsets; i++) {
				cout << setw(3) << i << " : " << setw(3) << part[i] << endl;
			}
		}


#if 0
		int hash;
		int TDO_depth;
		int f_labeled = true;
		int f_vv = (verbose_level >= 2);


		TDO_depth = nb_pts + nb_lines;

		if (f_vv) {
			cout << "projective_space::decomposition_from_set_partition "
					"before compute_TDO" << endl;
		}
		hash = Inc->compute_TDO(*Stack, ht0, TDO_depth, verbose_level + 2);
		if (f_vv) {
			cout << "projective_space::decomposition_from_set_partition "
					"after compute_TDO" << endl;
		}

		if (false) {
			Inc->print_partitioned(cout, *Stack, f_labeled);
		}
		if (f_v) {
			Inc->get_and_print_decomposition_schemes(*Stack);
			Stack->print_classes(cout);
		}
#endif

	}


	FREE_int(part);
	FREE_int(Mtx);
	if (f_v) {
		cout << "projective_space::decomposition_from_set_partition done" << endl;
	}
}
#endif






}}}}



