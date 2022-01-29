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






void projective_space::print_set_numerical(std::ostream &ost, long int *set, int set_size)
{
	long int i, a;
	int *v;

	v = NEW_int(n + 1);
	for (i = 0; i < set_size; i++) {
		a = set[i];
		unrank_point(v, a);
		ost << setw(3) << i << " : " << setw(5) << a << " : ";
		Int_vec_print(ost, v, n + 1);
		ost << "=";
		F->PG_element_normalize_from_front(v, 1, n + 1);
		Int_vec_print(ost, v, n + 1);
		ost << "\\\\" << endl;
	}
	FREE_int(v);
}

void projective_space::print_set(long int *set, int set_size)
{
	long int i, a;
	int *v;

	v = NEW_int(n + 1);
	for (i = 0; i < set_size; i++) {
		a = set[i];
		unrank_point(v, a);
		cout << setw(3) << i << " : " << setw(5) << a << " : ";
		F->int_vec_print_field_elements(cout, v, n + 1);
		cout << "=";
		F->PG_element_normalize_from_front(v, 1, n + 1);
		F->int_vec_print_field_elements(cout, v, n + 1);
		cout << endl;
	}
	FREE_int(v);
}

void projective_space::print_line_set_numerical(
		long int *set, int set_size)
{
	long int i, a;
	int *v;

	v = NEW_int(2 * (n + 1));
	for (i = 0; i < set_size; i++) {
		a = set[i];
		unrank_line(v, a);
		cout << setw(3) << i << " : " << setw(5) << a << " : ";
		Int_vec_print(cout, v, 2 * (n + 1));
		cout << endl;
	}
	FREE_int(v);
}


int projective_space::is_contained_in_Baer_subline(
	long int *pts, int nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int *subline;
	int sz;
	int i, idx, a;
	int ret = TRUE;
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
			ret = FALSE;
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

int projective_space::determine_hermitian_form_in_plane(
	int *pts, int nb_pts, int *six_coeffs, int verbose_level)
// there is a memory problem in this function
// detected 7/14/11
// solved June 17, 2012:
// coords and system were not freed
// system was allocated too short
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *coords; //[nb_pts * 3];
	int *system; //[nb_pts * 9];
	int kernel[9 * 9];
	int base_cols[9];
	int i, x, y, z, xq, yq, zq, rk;
	int Q, q, little_e;
	int kernel_m, kernel_n;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "projective_space::determine_hermitian_form_in_plane" << endl;
	}
	coords = NEW_int(nb_pts * 3);
	system = NEW_int(nb_pts * 9);
	Q = F->q;
	if (ODD(F->e)) {
		cout << "projective_space::determine_hermitian_"
				"form_in_plane field degree must be even" << endl;
		exit(1);
	}
	little_e = F->e >> 1;
	q = NT.i_power_j(F->p, little_e);
	if (f_v) {
		cout << "projective_space::determine_hermitian_"
				"form_in_plane Q=" << Q << " q=" << q << endl;
	}
	if (n != 2) {
		cout << "projective_space::determine_hermitian_"
				"form_in_plane n != 2" << endl;
		exit(1);
	}
	for (i = 0; i < nb_pts; i++) {
		unrank_point(coords + i * 3, pts[i]);
	}
	if (f_vv) {
		cout << "projective_space::determine_hermitian_"
				"form_in_plane points:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				coords, nb_pts, 3, 3, F->log10_of_q);
	}
	for (i = 0; i < nb_pts; i++) {
		x = coords[i * 3 + 0];
		y = coords[i * 3 + 1];
		z = coords[i * 3 + 2];
		xq = F->frobenius_power(x, little_e);
		yq = F->frobenius_power(y, little_e);
		zq = F->frobenius_power(z, little_e);
		system[i * 9 + 0] = F->mult(x, xq);
		system[i * 9 + 1] = F->mult(y, yq);
		system[i * 9 + 2] = F->mult(z, zq);
		system[i * 9 + 3] = F->mult(x, yq);
		system[i * 9 + 4] = F->mult(y, xq);
		system[i * 9 + 5] = F->mult(x, zq);
		system[i * 9 + 6] = F->mult(z, xq);
		system[i * 9 + 7] = F->mult(y, zq);
		system[i * 9 + 8] = F->mult(z, yq);
	}
	if (f_v) {
		cout << "projective_space::determine_hermitian_"
				"form_in_plane system:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				system, nb_pts, 9, 9, F->log10_of_q);
	}



	rk = F->Linear_algebra->Gauss_simple(system,
			nb_pts, 9, base_cols, verbose_level - 2);
	if (f_v) {
		cout << "projective_space::determine_hermitian_form_in_plane "
				"rk=" << rk << endl;
		Int_vec_print_integer_matrix_width(cout,
				system, rk, 9, 9, F->log10_of_q);
	}
#if 0
	if (rk != 8) {
		if (f_v) {
			cout << "projective_space::determine_hermitian_form_"
					"in_plane system underdetermined" << endl;
		}
		return FALSE;
	}
#endif
	F->Linear_algebra->matrix_get_kernel(system,
			MINIMUM(nb_pts, 9), 9, base_cols, rk,
		kernel_m, kernel_n, kernel, 0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space::determine_hermitian_form_"
				"in_plane kernel:" << endl;
		Int_vec_print_integer_matrix_width(cout, kernel,
				kernel_m, kernel_n, kernel_n, F->log10_of_q);
	}
	six_coeffs[0] = kernel[0 * kernel_n + 0];
	six_coeffs[1] = kernel[1 * kernel_n + 0];
	six_coeffs[2] = kernel[2 * kernel_n + 0];
	six_coeffs[3] = kernel[3 * kernel_n + 0];
	six_coeffs[4] = kernel[5 * kernel_n + 0];
	six_coeffs[5] = kernel[7 * kernel_n + 0];
	if (f_v) {
		cout << "projective_space::determine_hermitian_form_in_plane "
				"six_coeffs:" << endl;
		Int_vec_print(cout, six_coeffs, 6);
		cout << endl;
	}
	FREE_int(coords);
	FREE_int(system);
	return TRUE;
}

void projective_space::circle_type_of_line_subset(
	int *pts, int nb_pts, int *circle_type,
	int verbose_level)
// circle_type[nb_pts]
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

void projective_space::create_unital_XXq_YZq_ZYq_brute_force(
	long int *U, int &sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int *v;
	long int e, i, a;
	long int X, Y, Z, Xq, Yq, Zq;

	if (f_v) {
		cout << "projective_space::create_unital_XXq_YZq_ZYq" << endl;
	}
	if (n != 2) {
		cout << "projective_space::create_unital_XXq_YZq_ZYq "
				"n != 2" << endl;
		exit(1);
 	}
	if (ODD(F->e)) {
		cout << "projective_space::create_unital_XXq_YZq_ZYq "
				"ODD(F->e)" << endl;
		exit(1);
 	}

	v = NEW_int(3);
	e = F->e >> 1;
	if (f_vv) {
		cout << "e=" << e << endl;
	}
	sz = 0;
	for (i = 0; i < N_points; i++) {
		unrank_point(v, i);
		if (f_vvv) {
			cout << "i=" << i << " : ";
			Int_vec_print(cout, v, 3);
			//cout << endl;
		}
		X = v[0];
		Y = v[1];
		Z = v[2];
		Xq = F->frobenius_power(X, e);
		Yq = F->frobenius_power(Y, e);
		Zq = F->frobenius_power(Z, e);
		a = F->add3(F->mult(X, Xq), F->mult(Y, Zq), F->mult(Z, Yq));
		if (f_vvv) {
			cout << " a=" << a << endl;
		}
		if (a == 0) {
			//cout << "a=0, adding i=" << i << endl;
			U[sz++] = i;
			//int_vec_print(cout, U, sz);
			//cout << endl;
		}
	}
	if (f_vv) {
		cout << "we found " << sz << " points:" << endl;
		Lint_vec_print(cout, U, sz);
		cout << endl;
		print_set(U, sz);
	}
	FREE_int(v);

	if (f_v) {
		cout << "projective_space::create_unital_XXq_YZq_ZYq "
				"done" << endl;
	}
}


void projective_space::intersection_of_subspace_with_point_set(
	grassmann *G, int rk, long int *set, int set_size,
	long int *&intersection_set, int &intersection_set_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::intersection_of_subspace_with_point_set" << endl;
	}

	int h;
	int d = n + 1;
	int k = G->k;
	int *M;

	intersection_set = NEW_lint(set_size);
	M = NEW_int((k + 1) * d);
	intersection_set_size = 0;

	G->unrank_lint(rk, 0 /* verbose_level */);

	for (h = 0; h < set_size; h++) {
		Int_vec_copy(G->M, M, k * d);
		unrank_point(M + k * d, set[h]);
		if (F->Linear_algebra->rank_of_rectangular_matrix(M,
				k + 1, d, 0 /*verbose_level*/) == k) {
			intersection_set[intersection_set_size++] = set[h];
		}
	} // next h

	FREE_int(M);
	if (f_v) {
		cout << "projective_space::intersection_of_subspace_with_point_set done" << endl;
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
		cout << "projective_space::intersection_of_subspace_with_point_set_rank_is_longinteger" << endl;
	}
	int h;
	int d = n + 1;
	int k = G->k;
	int *M;

	intersection_set = NEW_lint(set_size);
	M = NEW_int((k + 1) * d);
	intersection_set_size = 0;

	G->unrank_longinteger(rk, 0 /* verbose_level */);

	for (h = 0; h < set_size; h++) {
		Int_vec_copy(G->M, M, k * d);
		unrank_point(M + k * d, set[h]);
		if (F->Linear_algebra->rank_of_rectangular_matrix(M,
				k + 1, d, 0 /*verbose_level*/) == k) {
			intersection_set[intersection_set_size++] = set[h];
		}
	} // next h

	FREE_int(M);
	if (f_v) {
		cout << "projective_space::intersection_of_subspace_with_point_set_rank_is_longinteger done" << endl;
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
		cout << "projective_space::plane_intersection_invariant before plane_intersection_type_fast" << endl;
	}
	plane_intersection_type_fast(G,
		set, set_size,
		R, Pts_on_plane, nb_pts_on_plane, nb_planes_total,
		verbose_level - 1);
	if (f_v) {
		cout << "projective_space::plane_intersection_invariant after plane_intersection_type_fast" << endl;
	}

	data_structures::tally C;
	int f_second = FALSE;

	C.init(nb_pts_on_plane, nb_planes_total, f_second, 0);
	if (f_v) {
		cout << "projective_space::plane_intersection_invariant "
				"plane-intersection type: ";
		C.print(FALSE /* f_backwards*/);
	}

	if (f_v) {
		cout << "The plane intersection type is (";
		C.print_naked(FALSE /*f_backwards*/);
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

void projective_space::plane_intersection_type(
	grassmann *G,
	long int *set, int set_size,
	int *&intersection_type, int &highest_intersection_number,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object *R;
	long int **Pts_on_plane;
	int *nb_pts_on_plane;
	int nb_planes;
	int i, f, l, a;

	if (f_v) {
		cout << "projective_space::plane_intersection_type" << endl;
	}
	if (f_v) {
		cout << "projective_space::plane_intersection_type before plane_intersection_type_fast" << endl;
	}
	plane_intersection_type_fast(G,
		set, set_size,
		R, Pts_on_plane, nb_pts_on_plane, nb_planes,
		verbose_level - 1);
	if (f_v) {
		cout << "projective_space::plane_intersection_type after plane_intersection_type_fast" << endl;
	}

	data_structures::tally C;
	int f_second = FALSE;

	C.init(nb_pts_on_plane, nb_planes, f_second, 0);
	if (f_v) {
		cout << "projective_space::plane_intersection_type "
				"plane-intersection type: ";
		C.print(FALSE /*f_backwards*/);
	}

	if (f_v) {
		cout << "The plane intersection type is (";
		C.print_naked(FALSE /*f_backwards*/);
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

	for (i = 0; i < nb_planes; i++) {
		FREE_lint(Pts_on_plane[i]);
	}
	FREE_plint(Pts_on_plane);
	FREE_int(nb_pts_on_plane);
	FREE_OBJECTS(R);
	if (f_v) {
		cout << "projective_space::plane_intersection_type done" << endl;
	}

}

void projective_space::plane_intersections(
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

	if (f_v) {
		cout << "projective_space::plane_intersections" << endl;
	}
	if (f_v) {
		cout << "projective_space::plane_intersections before plane_intersection_type_fast" << endl;
	}
	plane_intersection_type_fast(G,
		set, set_size,
		R, Pts_on_plane, nb_pts_on_plane, nb_planes,
		verbose_level - 1);
	if (f_v) {
		cout << "projective_space::plane_intersections after plane_intersection_type_fast" << endl;
	}
	if (f_v) {
		cout << "projective_space::plane_intersections "
				"before Sos.init" << endl;
	}
	SoS.init_with_Sz_in_int(set_size, nb_planes,
			Pts_on_plane, nb_pts_on_plane, verbose_level - 1);
	if (f_v) {
		cout << "projective_space::plane_intersections "
				"after Sos.init" << endl;
	}
	for (i = 0; i < nb_planes; i++) {
		FREE_lint(Pts_on_plane[i]);
	}
	FREE_plint(Pts_on_plane);
	FREE_int(nb_pts_on_plane);
	if (f_v) {
		cout << "projective_space::plane_intersections done" << endl;
	}
}

void projective_space::plane_intersection_type_slow(
	grassmann *G,
	long int *set, int set_size,
	ring_theory::longinteger_object *&R,
	long int **&Pts_on_plane, int *&nb_pts_on_plane, int &len,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_v3 = (verbose_level >= 3);
	long int r, rk, i, u, d, N_planes, l;

	int *Basis;
	int *Basis_save;
	int *Coords;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space::plane_intersection_type_slow" << endl;
	}
	if (f_vv) {
		print_set_numerical(cout, set, set_size);
	}
	if (!Sorting.test_if_set_with_return_value_lint(set, set_size)) {
		cout << "projective_space::plane_intersection_type_slow "
				"the input set if not a set" << endl;
		exit(1);
	}
	d = n + 1;
	N_planes = nb_rk_k_subspaces_as_lint(3);

	if (f_v) {
		cout << "N_planes=" << N_planes << endl;
	}
	// allocate data that is returned:
	R = NEW_OBJECTS(ring_theory::longinteger_object, N_planes);
	Pts_on_plane = NEW_plint(N_planes);
	nb_pts_on_plane = NEW_int(N_planes);

	// allocate temporary data:
	Basis = NEW_int(4 * d);
	Basis_save = NEW_int(4 * d);
	Coords = NEW_int(set_size * d);

	for (i = 0; i < set_size; i++) {
		unrank_point(Coords + i * d, set[i]);
	}
	if (f_vv) {
		cout << "projective_space::plane_intersection_type_slow "
				"Coords:" << endl;
		Int_matrix_print(Coords, set_size, d);
	}

	l = 0;
	for (rk = 0; rk < N_planes; rk++) {

		if (N_planes > 1000000) {
			if ((rk % 250000) == 0) {
				cout << "projective_space::plane_intersection_type_slow "
						<< rk << " / " << N_planes << endl;
			}
		}
		G->unrank_lint_here(Basis_save, rk, 0 /* verbose_level */);
		//int_vec_copy(G->M, Basis_save, 3 * d);
		long int *pts_on_plane;
		int nb = 0;

		pts_on_plane = NEW_lint(set_size);

		for (u = 0; u < set_size; u++) {
			Int_vec_copy(Basis_save, Basis, 3 * d);
			Int_vec_copy(Coords + u * d, Basis + 3 * d, d);
			r = F->Linear_algebra->rank_of_rectangular_matrix(Basis,
					4, d, 0 /* verbose_level */);
			if (r < 4) {
				pts_on_plane[nb++] = u;
			}
		}


		Pts_on_plane[l] = pts_on_plane;
		nb_pts_on_plane[l] = nb;
		R[l].create(rk, __FILE__, __LINE__);
		l++;
	} // rk
	len = l;

	FREE_int(Basis);
	FREE_int(Basis_save);
	FREE_int(Coords);
	if (f_v) {
		cout << "projective_space::plane_intersection_type_slow "
				"done" << endl;
	}
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
	int f_v3 = (verbose_level >= 3);
	int r, rk, rr, h, i, j, a, d, N_planes, N, N2, idx, l;

	int *Basis;
	int *Basis_save;
	int *f_subset_done;
	int *rank_idx;
	int *Coords;

	int subset[3];
	int subset2[3];
	int subset3[3];
	ring_theory::longinteger_object plane_rk, aa;
	long int *pts_on_plane;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space::plane_intersection_type_fast" << endl;
	}
	if (f_vv) {
		print_set_numerical(cout, set, set_size);
	}

	if (!Sorting.test_if_set_with_return_value_lint(set, set_size)) {
		cout << "projective_space::plane_intersection_type_fast "
				"the input set if not a set" << endl;
		exit(1);
	}
	d = n + 1;
	N_planes = nb_rk_k_subspaces_as_lint(3);
	N = Combi.int_n_choose_k(set_size, 3);

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
	Coords = NEW_int(set_size * d);

	for (i = 0; i < N; i++) {
		f_subset_done[i] = FALSE;
	}
	for (i = 0; i < set_size; i++) {
		unrank_point(Coords + i * d, set[i]);
	}
	if (f_vv) {
		cout << "projective_space::plane_intersection_type_fast "
				"Coords:" << endl;
		Int_matrix_print(Coords, set_size, d);
	}


	len = 0;
	for (rk = 0; rk < N; rk++) {
		Combi.unrank_k_subset(rk, subset, set_size, 3);
		if (f_v) {
			cout << rk << "-th subset ";
			Int_vec_print(cout, subset, 3);
			cout << endl;
		}
		if (f_subset_done[rk]) {
			if (f_v) {
				cout << "skipping" << endl;
			}
			continue;
		}
		for (j = 0; j < 3; j++) {
			a = subset[j];
			//a = set[subset[j]];
			Int_vec_copy(Coords + a * d, Basis + j * d, d);
			//unrank_point(Basis + j * d, a);
		}
		if (f_v3) {
			cout << "subset: ";
			Int_vec_print(cout, subset, 3);
			cout << " corresponds to Basis:" << endl;
			Int_matrix_print(Basis, 3, d);
		}
		r = F->Linear_algebra->rank_of_rectangular_matrix(
				Basis, 3, d, 0 /* verbose_level */);
		if (r < 3) {
			if (TRUE || f_v) {
				cout << "projective_space::plane_intersection_type_fast "
						"not independent, skip" << endl;
				cout << "subset: ";
				Int_vec_print(cout, subset, 3);
				cout << endl;
			}
			rank_idx[rk] = -1;
			continue;
		}
		G->rank_longinteger_here(Basis, plane_rk, 0 /* verbose_level */);
		if (f_v) {
			cout << rk << "-th subset ";
			Int_vec_print(cout, subset, 3);
			cout << " plane_rk=" << plane_rk << endl;
		}

		if (Sorting.longinteger_vec_search(R, len, plane_rk, idx)) {
			//rank_idx[rk] = idx;
			// this case should never happen:
			cout << "projective_space::plane_intersection_type_fast "
					"longinteger_vec_search(R, len, plane_rk, idx) "
					"is TRUE" << endl;
			exit(1);
		}
		else {
			if (f_v3) {
				cout << "plane_rk=" << plane_rk
						<< " was not found" << endl;
			}
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

			l = 0;
			for (h = 0; h < set_size; h++) {
				if (FALSE && f_v3) {
					cout << "testing point " << h << ":" << endl;
					cout << "plane_rk=" << plane_rk << endl;
				}
#if 0
				plane_rk.assign_to(aa);
				G->unrank_longinteger(aa, 0 /* verbose_level */);
				if (f_v3) {
					cout << "after unrank " << plane_rk << ":" << endl;
					int_matrix_print(G->M, 3, d);
				}
				for (j = 0; j < 3 * d; j++) {
					Basis[j] = G->M[j];
				}
#endif

				Int_vec_copy(Basis_save, Basis, 3 * d);
				//a = set[h];

				Int_vec_copy(Coords + h * d, Basis + 3 * d, d);

				//unrank_point(Basis + 3 * d, set[h]);
				if (FALSE && f_v3) {
					cout << "Basis and point:" << endl;
					Int_matrix_print(Basis, 4, d);
				}
				r = F->Linear_algebra->rank_of_rectangular_matrix(Basis,
						4, d, 0 /* verbose_level */);
				if (r == 3) {
					pts_on_plane[l++] = h;
					if (f_v3) {
						cout << "point " << h
								<< " is on the plane" << endl;
					}
				}
				else {
					if (FALSE && f_v3) {
						cout << "point " << h
								<< " is not on the plane" << endl;
					}
				}
			}
			if (f_v) {
				cout << "We found an " << l << "-plane, "
						"its rank is " << plane_rk << endl;
				cout << "The ranks of points on that plane are : ";
				Lint_vec_print(cout, pts_on_plane, l);
				cout << endl;
			}


			if (l >= 3) {
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
				if (f_v3) {
					cout << "after assign_to, "
							"plane_rk=" << plane_rk << endl;
				}
				rank_idx[rk] = idx;
				len++;




				N2 = Combi.int_n_choose_k(l, 3);
				for (i = 0; i < N2; i++) {
					Combi.unrank_k_subset(i, subset2, l, 3);
					for (h = 0; h < 3; h++) {
						subset3[h] = pts_on_plane[subset2[h]];
					}
					rr = Combi.rank_k_subset(subset3, set_size, 3);
					if (f_v) {
						cout << i << "-th subset3 ";
						Int_vec_print(cout, subset3, 3);
						cout << " rr=" << rr << endl;
					}
					if (!f_subset_done[rr]) {
						f_subset_done[rr] = TRUE;
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
				nb_pts_on_plane[idx] = l;
			}
			else {
				// now l <= 2, we skip those planes:

				FREE_lint(pts_on_plane);
				f_subset_done[rk] = TRUE;
				rank_idx[rk] = -2;
			}
		} // else
	} // next rk

	FREE_int(Basis);
	FREE_int(Basis_save);
	FREE_int(f_subset_done);
	FREE_int(rank_idx);
	FREE_int(Coords);
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
		print_set_numerical(cout, set, set_size);
	}
	if (!Sorting.test_if_set_with_return_value_lint(set, set_size)) {
		cout << "projective_space::find_planes_which_intersect_in_at_least_s_points "
				"the input set if not a set" << endl;
		exit(1);
	}
	d = n + 1;
	N_planes = nb_rk_k_subspaces_as_lint(3);

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
		Grass_planes->unrank_lint_here(Basis_save, rk, 0 /* verbose_level */);
		//int_vec_copy(G->M, Basis_save, 3 * d);

		int nb_pts_on_plane = 0;

		for (u = 0; u < set_size; u++) {
			Int_vec_copy(Basis_save, Basis, 3 * d);
			Int_vec_copy(Coords + u * d, Basis + 3 * d, d);
			r = F->Linear_algebra->rank_of_rectangular_matrix(Basis,
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
		cout << "projective_space::find_planes_which_intersect_in_at_least_s_points we found "
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

void projective_space::plane_intersection(int plane_rank,
		long int *set, int set_size,
		vector<int> &point_indices,
		vector<int> &point_local_coordinates,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int r, i, u, d;

	int *Basis;
	int *Basis_save;
	int *Coords;
	int base_cols[3];
	int coefficients[3];
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space::plane_intersection" << endl;
	}
	d = n + 1;
	// allocate temporary data:
	Basis = NEW_int(4 * d);
	Basis_save = NEW_int(4 * d);
	Coords = NEW_int(set_size * d);

	for (i = 0; i < set_size; i++) {
		unrank_point(Coords + i * d, set[i]);
	}
	if (f_vv) {
		cout << "projective_space::plane_intersection "
				"Coords:" << endl;
		Int_matrix_print(Coords, set_size, d);
	}

	Grass_planes->unrank_lint_here(Basis_save, plane_rank, 0 /* verbose_level */);

	int nb_pts_on_plane = 0;
	int local_rank;

	for (u = 0; u < set_size; u++) {
		Int_vec_copy(Basis_save, Basis, 3 * d);
		Int_vec_copy(Coords + u * d, Basis + 3 * d, d);
		r = F->Linear_algebra->rank_of_rectangular_matrix(Basis,
				4, d, 0 /* verbose_level */);
		if (r < 4) {
			nb_pts_on_plane++;
			point_indices.push_back(u);

			Int_vec_copy(Basis_save, Basis, 3 * d);
			Int_vec_copy(Coords + u * d, Basis + 3 * d, d);

			F->Linear_algebra->Gauss_simple(Basis, 3, d,
					base_cols, 0 /*verbose_level */);
			F->Linear_algebra->reduce_mod_subspace_and_get_coefficient_vector(
				3, d, Basis, base_cols,
				Basis + 3 * d, coefficients, verbose_level);
			F->PG_element_rank_modified(
					coefficients, 1, 3, local_rank);
			point_local_coordinates.push_back(local_rank);
		}
	}

	FREE_int(Basis);
	FREE_int(Basis_save);
	FREE_int(Coords);
	if (f_v) {
		cout << "projective_space::plane_intersection "
				"done" << endl;
	}
}

void projective_space::line_intersection(int line_rank,
		long int *set, int set_size,
		vector<int> &point_indices,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int r, i, u, d;

	int *Basis;
	int *Basis_save;
	int *Coords;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space::line_intersection" << endl;
	}
	d = n + 1;
	// allocate temporary data:
	Basis = NEW_int(3 * d);
	Basis_save = NEW_int(3 * d);
	Coords = NEW_int(set_size * d);

	for (i = 0; i < set_size; i++) {
		unrank_point(Coords + i * d, set[i]);
	}
	if (f_vv) {
		cout << "projective_space::line_intersection "
				"Coords:" << endl;
		Int_matrix_print(Coords, set_size, d);
	}

	Grass_lines->unrank_lint_here(Basis_save, line_rank, 0 /* verbose_level */);

	for (u = 0; u < set_size; u++) {
		Int_vec_copy(Basis_save, Basis, 2 * d);
		Int_vec_copy(Coords + u * d, Basis + 2 * d, d);
		r = F->Linear_algebra->rank_of_rectangular_matrix(Basis,
				3, d, 0 /* verbose_level */);
		if (r < 3) {
			point_indices.push_back(u);
		}
	}

	FREE_int(Basis);
	FREE_int(Basis_save);
	FREE_int(Coords);
	if (f_v) {
		cout << "projective_space::line_intersection "
				"done" << endl;
	}
}

void projective_space::klein_correspondence(
	projective_space *P5,
	long int *set_in, int set_size, long int *set_out,
	int verbose_level)
// Computes the Pluecker coordinates
// for a line in PG(3,q) in the following order:
// (x_1,x_2,x_3,x_4,x_5,x_6) =
// (Pluecker_12, Pluecker_34, Pluecker_13,
//    Pluecker_42, Pluecker_14, Pluecker_23)
// satisfying the quadratic form x_1x_2 + x_3x_4 + x_5x_6 = 0
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int d = n + 1;
	int h;
	int basis8[8];
	int v6[6];
	int *x4, *y4;
	long int a, b, c;
	int f_elements_exponential = TRUE;
	string symbol_for_print;




	if (f_v) {
		cout << "projective_space::klein_correspondence" << endl;
	}


	symbol_for_print.assign("\\alpha");

	for (h = 0; h < set_size; h++) {
		a = set_in[h];
		Grass_lines->unrank_lint(a, 0 /* verbose_level */);
		if (f_vv) {
			cout << setw(5) << h << " : " << setw(5) << a << " :" << endl;
			F->latex_matrix(cout, f_elements_exponential,
				symbol_for_print, Grass_lines->M, 2, 4);
			cout << endl;
		}
		Int_vec_copy(Grass_lines->M, basis8, 8);
		if (f_vv) {
			Int_matrix_print(basis8, 2, 4);
		}
		x4 = basis8;
		y4 = basis8 + 4;
		v6[0] = F->Linear_algebra->Pluecker_12(x4, y4);
		v6[1] = F->Linear_algebra->Pluecker_34(x4, y4);
		v6[2] = F->Linear_algebra->Pluecker_13(x4, y4);
		v6[3] = F->Linear_algebra->Pluecker_42(x4, y4);
		v6[4] = F->Linear_algebra->Pluecker_14(x4, y4);
		v6[5] = F->Linear_algebra->Pluecker_23(x4, y4);
		if (f_vv) {
			cout << "v6 : ";
			Int_vec_print(cout, v6, 6);
			cout << endl;
		}
		a = F->mult(v6[0], v6[1]);
		b = F->mult(v6[2], v6[3]);
		c = F->mult(v6[4], v6[5]);
		d = F->add3(a, b, c);
		//cout << "a=" << a << " b=" << b << " c=" << c << endl;
		//cout << "d=" << d << endl;
		if (d) {
			cout << "d != 0" << endl;
			exit(1);
		}
		set_out[h] = P5->rank_point(v6);
	}
	if (f_v) {
		cout << "projective_space::klein_correspondence done" << endl;
	}
}

void projective_space::Pluecker_coordinates(
	int line_rk, int *v6, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int basis8[8];
	int *x4, *y4;
	int f_elements_exponential = FALSE;
	string symbol_for_print;

	if (f_v) {
		cout << "projective_space::Pluecker_coordinates" << endl;
	}
	symbol_for_print.assign("\\alpha");
	Grass_lines->unrank_lint(line_rk, 0 /* verbose_level */);
	if (f_vv) {
		cout << setw(5) << line_rk << " :" << endl;
		F->latex_matrix(cout, f_elements_exponential,
			symbol_for_print, Grass_lines->M, 2, 4);
		cout << endl;
	}
	Int_vec_copy(Grass_lines->M, basis8, 8);
	if (f_vv) {
		Int_matrix_print(basis8, 2, 4);
	}
	x4 = basis8;
	y4 = basis8 + 4;
	v6[0] = F->Linear_algebra->Pluecker_12(x4, y4);
	v6[1] = F->Linear_algebra->Pluecker_34(x4, y4);
	v6[2] = F->Linear_algebra->Pluecker_13(x4, y4);
	v6[3] = F->Linear_algebra->Pluecker_42(x4, y4);
	v6[4] = F->Linear_algebra->Pluecker_14(x4, y4);
	v6[5] = F->Linear_algebra->Pluecker_23(x4, y4);
	if (f_vv) {
		cout << "v6 : ";
		Int_vec_print(cout, v6, 6);
		cout << endl;
	}
	if (f_v) {
		cout << "projective_space::Pluecker_coordinates done" << endl;
	}
}

void projective_space::klein_correspondence_special_model(
	projective_space *P5,
	int *table, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int d = n + 1;
	int h;
	int basis8[8];
	int x6[6];
	int y6[6];
	int *x4, *y4;
	int a, b, c;
	int half;
	int f_elements_exponential = TRUE;
	string symbol_for_print;
	//int *table;

	if (f_v) {
		cout << "projective_space::klein_correspondence" << endl;
	}
	symbol_for_print.assign("\\alpha");
	half = F->inverse(F->add(1, 1));
	if (f_v) {
		cout << "half=" << half << endl;
		cout << "N_lines=" << N_lines << endl;
	}
	//table = NEW_int(N_lines);
	for (h = 0; h < N_lines; h++) {
		Grass_lines->unrank_lint(h, 0 /* verbose_level */);
		if (f_vv) {
			cout << setw(5) << h << " :" << endl;
			F->latex_matrix(cout, f_elements_exponential,
				symbol_for_print, Grass_lines->M, 2, 4);
			cout << endl;
		}
		Int_vec_copy(Grass_lines->M, basis8, 8);
		if (f_vv) {
			Int_matrix_print(basis8, 2, 4);
		}
		x4 = basis8;
		y4 = basis8 + 4;
		x6[0] = F->Linear_algebra->Pluecker_12(x4, y4);
		x6[1] = F->Linear_algebra->Pluecker_34(x4, y4);
		x6[2] = F->Linear_algebra->Pluecker_13(x4, y4);
		x6[3] = F->Linear_algebra->Pluecker_42(x4, y4);
		x6[4] = F->Linear_algebra->Pluecker_14(x4, y4);
		x6[5] = F->Linear_algebra->Pluecker_23(x4, y4);
		if (f_vv) {
			cout << "x6 : ";
			Int_vec_print(cout, x6, 6);
			cout << endl;
		}
		a = F->mult(x6[0], x6[1]);
		b = F->mult(x6[2], x6[3]);
		c = F->mult(x6[4], x6[5]);
		d = F->add3(a, b, c);
		//cout << "a=" << a << " b=" << b << " c=" << c << endl;
		//cout << "d=" << d << endl;
		if (d) {
			cout << "d != 0" << endl;
			exit(1);
		}
		y6[0] = F->negate(x6[0]);
		y6[1] = x6[1];
		y6[2] = F->mult(half, F->add(x6[2], x6[3]));
		y6[3] = F->mult(half, F->add(x6[2], F->negate(x6[3])));
		y6[4] = x6[4];
		y6[5] = x6[5];
		if (f_vv) {
			cout << "y6 : ";
			Int_vec_print(cout, y6, 6);
			cout << endl;
		}
		table[h] = P5->rank_point(y6);
	}

	cout << "lines in PG(3,q) to points in PG(5,q) "
			"in special model:" << endl;
	for (h = 0; h < N_lines; h++) {
		cout << setw(4) << h << " : " << setw(5) << table[h] << endl;
	}

	//FREE_int(table);
	if (f_v) {
		cout << "projective_space::klein_correspondence_special_model done" << endl;
	}
}

void projective_space::cheat_sheet_points(
		std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::cheat_sheet_points" << endl;
	}
	int i, d;
	int *v;
	string symbol_for_print;

	d = n + 1;

	symbol_for_print.assign("\\alpha");
	v = NEW_int(d);

	f << "PG$(" << n << ", " << q << ")$ has "
			<< N_points << " points:\\\\" << endl;
	if (F->e == 1) {
		f << "\\begin{multicols}{4}" << endl;
		for (i = 0; i < N_points; i++) {
			F->PG_element_unrank_modified(v, 1, d, i);
			f << "$P_{" << i << "}=\\bP";
			Int_vec_print(f, v, d);
			f << "$\\\\" << endl;
		}
		f << "\\end{multicols}" << endl;
	}
	else {
		f << "\\begin{multicols}{2}" << endl;
		for (i = 0; i < N_points; i++) {
			F->PG_element_unrank_modified(v, 1, d, i);
			f << "$P_{" << i << "}=\\bP";
			Int_vec_print(f, v, d);
			f << "=";
			F->int_vec_print_elements_exponential(f, v, d, symbol_for_print);
			f << "$\\\\" << endl;
		}
		f << "\\end{multicols}" << endl;

		f << "\\begin{multicols}{2}" << endl;
		for (i = 0; i < N_points; i++) {
			F->PG_element_unrank_modified(v, 1, d, i);
			f << "$P_{" << i << "}=\\bP";
			Int_vec_print(f, v, d);
			//f << "=";
			//F->int_vec_print_elements_exponential(f, v, d, symbol_for_print);
			f << "$\\\\" << endl;
		}
		f << "\\end{multicols}" << endl;

	}

	if (F->has_quadratic_subfield()) {
		f << "Baer subgeometry:\\\\" << endl;
		f << "\\begin{multicols}{4}" << endl;
		int j, cnt = 0;
		for (i = 0; i < N_points; i++) {
			F->PG_element_unrank_modified(v, 1, d, i);
			F->PG_element_normalize_from_front(v, 1, d);
			for (j = 0; j < d; j++) {
				if (!F->belongs_to_quadratic_subfield(v[j])) {
					break;
				}
			}
			if (j == d) {
				cnt++;
				f << "$P_{" << i << "}=\\bP";
				Int_vec_print(f, v, d);
				f << "$\\\\" << endl;
			}
		}
		f << "\\end{multicols}" << endl;
		f << "There are " << cnt << " elements in the Baer subgeometry.\\\\" << endl;

	}
	//f << "\\clearpage" << endl << endl;

	f << "Normalized from the left:\\\\" << endl;
	f << "\\begin{multicols}{4}" << endl;
	for (i = 0; i < N_points; i++) {
		F->PG_element_unrank_modified(v, 1, d, i);
		F->PG_element_normalize_from_front(v, 1, d);
		f << "$P_{" << i << "}=\\bP";
		Int_vec_print(f, v, d);
		f << "$\\\\" << endl;
		}
	f << "\\end{multicols}" << endl;
	f << "\\clearpage" << endl << endl;


	cheat_polarity(f, verbose_level);


	FREE_int(v);
	if (f_v) {
		cout << "projective_space::cheat_sheet_points done" << endl;
	}
}

void projective_space::cheat_polarity(std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::cheat_polarity" << endl;
	}

	f << "Standard polarity point $\\leftrightarrow$ hyperplane:\\\\" << endl;

	Standard_polarity->report(f);

	f << "Reversal polarity point $\\leftrightarrow$ hyperplane:\\\\" << endl;

	Reversal_polarity->report(f);

	if (f_v) {
		cout << "projective_space::cheat_polarity done" << endl;
	}
}

void projective_space::cheat_sheet_point_table(
		std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::cheat_sheet_point_table" << endl;
	}
	int I, i, j, a, d, nb_rows, nb_cols = 5;
	int nb_rows_per_page = 40, nb_tables;
	int nb_r;
	int *v;

	d = n + 1;

	v = NEW_int(d);

	f << "PG$(" << n << ", " << q << ")$ has " << N_points
			<< " points:\\\\" << endl;

	nb_rows = (N_points + nb_cols - 1) / nb_cols;
	nb_tables = (nb_rows + nb_rows_per_page - 1) / nb_rows_per_page;

	for (I = 0; I < nb_tables; I++) {
		f << "$$" << endl;
		f << "\\begin{array}{r|*{" << nb_cols << "}{r}}" << endl;
		f << "P_{" << nb_cols << "\\cdot i+j}";
		for (j = 0; j < nb_cols; j++) {
			f << " & " << j;
		}
		f << "\\\\" << endl;
		f << "\\hline" << endl;

		if (I == nb_tables - 1) {
			nb_r = nb_rows - I * nb_rows_per_page;
		}
		else {
			nb_r = nb_rows_per_page;
		}

		for (i = 0; i < nb_r; i++) {
			f << (I * nb_rows_per_page + i) * nb_cols;
			for (j = 0; j < nb_cols; j++) {
				a = (I * nb_rows_per_page + i) * nb_cols + j;
				f << " & ";
				if (a < N_points) {
					F->PG_element_unrank_modified(v, 1, d, a);
					Int_vec_print(f, v, d);
					}
				}
			f << "\\\\" << endl;
			}
		f << "\\end{array}" << endl;
		f << "$$" << endl;
		}

	FREE_int(v);
	if (f_v) {
		cout << "projective_space::cheat_sheet_point_table done" << endl;
	}
}


void projective_space::cheat_sheet_points_on_lines(
	std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::cheat_sheet_points_on_lines" << endl;
	}
	orbiter_kernel_system::latex_interface L;


	f << "PG$(" << n << ", " << q << ")$ has " << N_lines
			<< " lines, each with " << k << " points:\\\\" << endl;
	if (Implementation->Lines == NULL) {
		f << "Don't have Lines table\\\\" << endl;
	}
	else {
		int *row_labels;
		int *col_labels;
		int i, nb;

		row_labels = NEW_int(N_lines);
		col_labels = NEW_int(k);
		for (i = 0; i < N_lines; i++) {
			row_labels[i] = i;
		}
		for (i = 0; i < k; i++) {
			col_labels[i] = i;
		}
		//int_matrix_print_tex(f, Lines, N_lines, k);
		for (i = 0; i < N_lines; i += 40) {
			nb = MINIMUM(N_lines - i, 40);
			//f << "i=" << i << " nb=" << nb << "\\\\" << endl;
			f << "$$" << endl;
			L.print_integer_matrix_with_labels(f,
					Implementation->Lines + i * k, nb, k, row_labels + i,
					col_labels, TRUE /* f_tex */);
			f << "$$" << endl;
		}
		FREE_int(row_labels);
		FREE_int(col_labels);
	}
	if (f_v) {
		cout << "projective_space::cheat_sheet_points_on_lines done" << endl;
	}
}

void projective_space::cheat_sheet_lines_on_points(
	std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::cheat_sheet_lines_on_points" << endl;
	}
	orbiter_kernel_system::latex_interface L;

	f << "PG$(" << n << ", " << q << ")$ has " << N_points
			<< " points, each with " << r << " lines:\\\\" << endl;
	if (Implementation->Lines_on_point == NULL) {
		f << "Don't have Lines\\_on\\_point table\\\\" << endl;
	}
	else {
		int *row_labels;
		int *col_labels;
		int i, nb;

		row_labels = NEW_int(N_points);
		col_labels = NEW_int(r);
		for (i = 0; i < N_points; i++) {
			row_labels[i] = i;
		}
		for (i = 0; i < r; i++) {
			col_labels[i] = i;
		}
		for (i = 0; i < N_points; i += 40) {
			nb = MINIMUM(N_points - i, 40);
			//f << "i=" << i << " nb=" << nb << "\\\\" << endl;
			f << "$$" << endl;
			L.print_integer_matrix_with_labels(f,
					Implementation->Lines_on_point + i * r, nb, r,
				row_labels + i, col_labels, TRUE /* f_tex */);
			f << "$$" << endl;
		}
		FREE_int(row_labels);
		FREE_int(col_labels);

#if 0
		f << "$$" << endl;
		int_matrix_print_tex(f, Lines_on_point, N_points, r);
		f << "$$" << endl;
#endif
	}
	if (f_v) {
		cout << "projective_space::cheat_sheet_lines_on_points done" << endl;
	}
}


void projective_space::cheat_sheet_subspaces(
	std::ostream &f, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	grassmann *Gr;
	int *v;
	int n1, k1;
	int nb_k_subspaces;
	int i, j, u;
	int f_need_comma = FALSE;
	combinatorics::combinatorics_domain Combi;


	if (f_v) {
		cout << "projective_space::cheat_sheet_subspaces "
				"k=" << k << endl;
	}
	n1 = n + 1;
	k1 = k + 1;
	v = NEW_int(n1);

	if (F->q >= 10) {
		f_need_comma = TRUE;
	}

	Gr = NEW_OBJECT(grassmann);
	Gr->init(n1, k1, F, 0 /*verbose_level*/);


	//nb_points = N_points;
	nb_k_subspaces = Combi.generalized_binomial(n1, k1, q);


	f << "PG$(" << n << ", " << q << ")$ has "
			<< nb_k_subspaces << " $" << k
			<< "$-subspaces:\\\\" << endl;

	if (nb_k_subspaces > 10000) {
		f << "Too many to print \\\\" << endl;
	}
	else {
		f << "%\\begin{multicols}{2}" << endl;
		for (u = 0; u < nb_k_subspaces; u++) {
			Gr->unrank_lint(u, 0 /* verbose_level*/);
			f << "$L_{" << u << "}=\\bL";
			f << "\\left[" << endl;
			f << "\\begin{array}{c}" << endl;
			for (i = 0; i < k1; i++) {
				for (j = 0; j < n1; j++) {
					f << Gr->M[i * n1 + j];
					if (f_need_comma && j < n1 - 1) {
						f << ", ";
					}
				}
				f << "\\\\" << endl;
			}
			f << "\\end{array}" << endl;
			f << "\\right]" << endl;

			if (n == 3 && k == 1) {
				int v6[6];

				Pluecker_coordinates(u, v6, 0 /* verbose_level */);
				f << "={\\rm\\bf Pl}(" << v6[0] << "," << v6[1] << ","
						<< v6[2] << "," << v6[3] << "," << v6[4]
						<< "," << v6[5] << " ";
				f << ")" << endl;

			}
			f << "$\\\\" << endl;

			if (((u + 1) % 1000) == 0) {
				f << "\\clearpage" << endl << endl;
			}
		}
		f << "%\\end{multicols}" << endl;


		if (n == 3 && k == 1) {
			do_pluecker_reverse(f, Gr, k, nb_k_subspaces, verbose_level);
		}

	}
	if (n == 3 && k == 1) {
		f << "PG$(" << n << ", " << q << ")$ has "
				<< "the following low weight Pluecker lines:\\\\" << endl;
		for (u = 0; u < nb_k_subspaces; u++) {
			int v6[6];
			int w;

			Gr->unrank_lint(u, 0 /* verbose_level*/);
			Pluecker_coordinates(u, v6, 0 /* verbose_level */);
			w = 0;
			for (j = 0; j < 6; j++) {
				if (v6[j]) {
					w++;
				}
			}
			if (w == 1) {
				f << "$L_{" << u << "}=";
				f << "\\left[" << endl;
				f << "\\begin{array}{c}" << endl;
				for (i = 0; i < k1; i++) {
					for (j = 0; j < n1; j++) {
						f << Gr->M[i * n1 + j];
						if (f_need_comma && j < n1 - 1) {
							f << ", ";
							}
						}
					f << "\\\\" << endl;
					}
				f << "\\end{array}" << endl;
				f << "\\right]" << endl;
				f << "={\\rm\\bf Pl}(" << v6[0] << "," << v6[1] << ","
						<< v6[2] << "," << v6[3] << "," << v6[4]
						<< "," << v6[5] << " ";
				f << ")" << endl;
				f << "$\\\\" << endl;

			}
		}


	}

	f << "\\clearpage" << endl << endl;

	FREE_OBJECT(Gr);
	FREE_int(v);

	if (f_v) {
		cout << "projective_space::cheat_sheet_subspaces "
				"done" << endl;
	}
}

void projective_space::do_pluecker_reverse(ostream &ost,
		grassmann *Gr, int k, int nb_k_subspaces, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int v6[6];
	int *T;
	int *Pos;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space::do_pluecker_reverse" << endl;
	}
	T = NEW_int(nb_k_subspaces);
	Pos = NEW_int(nb_k_subspaces);
	for (i = 0; i < nb_k_subspaces; i++) {
		Gr->unrank_lint(i, 0 /* verbose_level*/);
		Pluecker_coordinates(i, v6, 0 /* verbose_level */);
		F->PG_element_rank_modified(v6, 1, 6, j);
		T[i] = j;
		Pos[i] = i;
	}
	Sorting.int_vec_heapsort_with_log(T, Pos, nb_k_subspaces);
	if (f_v) {
		cout << "projective_space::do_pluecker_reverse after sort:" << endl;
		for (i = 0; i < nb_k_subspaces; i++) {
			cout << i << " : " << T[i] << " : " << Pos[i] << endl;
		}
	}


	int u, u0;
	int n1, k1;
	int f_need_comma = FALSE;

	n1 = n + 1;
	k1 = k + 1;
	v = NEW_int(n1);

	ost << "Lines sorted by Pluecker coordinates\\\\" << endl;
	ost << "%\\begin{multicols}{2}" << endl;
	for (u0 = 0; u0 < nb_k_subspaces; u0++) {
		u = Pos[u0];
		Gr->unrank_lint(u, 0 /* verbose_level*/);

		int v6[6];

		Pluecker_coordinates(u, v6, 0 /* verbose_level */);
		F->PG_element_normalize(v6, 1, 6);
		ost << "$" << u0 << /*"=" << u <<*/
				"={\\rm\\bf Pl}(" << v6[0] << "," << v6[1] << ","
				<< v6[2] << "," << v6[3] << "," << v6[4]
				<< "," << v6[5] << " ";
		ost << ")=" << endl;

		ost << "L_{" << u << "}=";
		ost << "\\left[" << endl;
		ost << "\\begin{array}{c}" << endl;
		for (i = 0; i < k1; i++) {
			for (j = 0; j < n1; j++) {
				ost << Gr->M[i * n1 + j];
				if (f_need_comma && j < n1 - 1) {
					ost << ", ";
				}
			}
			ost << "\\\\" << endl;
		}
		ost << "\\end{array}" << endl;
		ost << "\\right]" << endl;


		ost << "$\\\\" << endl;

		if (((u + 1) % 1000) == 0) {
			ost << "\\clearpage" << endl << endl;
		}
	}
	ost << "%\\end{multicols}" << endl;


	FREE_int(T);
	FREE_int(Pos);
	if (f_v) {
		cout << "projective_space::do_pluecker_reverse done" << endl;
	}
}

void projective_space::cheat_sheet_line_intersection(
		std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::cheat_sheet_line_intersection" << endl;
	}
	int i, j, a;


	f << "intersection of 2 lines:" << endl;
	f << "$$" << endl;
	f << "\\begin{array}{|r|*{" << N_points << "}{r}|}" << endl;
	f << "\\hline" << endl;
	for (j = 0; j < N_points; j++) {
		f << "& " << j << endl;
	}
	f << "\\\\" << endl;
	f << "\\hline" << endl;
	for (i = 0; i < N_points; i++) {
		f << i;
		for (j = 0; j < N_points; j++) {
			a = Implementation->Line_intersection[i * N_lines + j];
			f << " & ";
			if (i != j) {
				f << a;
			}
		}
		f << "\\\\[-3pt]" << endl;
	}
	f << "\\hline" << endl;
	f << "\\end{array}" << endl;
	f << "$$" << endl;
	f << "\\clearpage" << endl;

	if (f_v) {
		cout << "projective_space::cheat_sheet_line_intersection done" << endl;
	}

}

void projective_space::cheat_sheet_line_through_pairs_of_points(
		std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::cheat_sheet_line_through_pairs_of_points" << endl;
	}
	int i, j, a;



	f << "line through 2 points:" << endl;
	f << "$$" << endl;
	f << "\\begin{array}{|r|*{" << N_points << "}{r}|}" << endl;
	f << "\\hline" << endl;
	for (j = 0; j < N_points; j++) {
		f << "& " << j << endl;
	}
	f << "\\\\" << endl;
	f << "\\hline" << endl;
	for (i = 0; i < N_points; i++) {
		f << i;
		for (j = 0; j < N_points; j++) {

			a = Implementation->Line_through_two_points[i * N_points + j];
			f << " & ";
			if (i != j) {
				f << a;
			}
		}
		f << "\\\\[-3pt]" << endl;
	}
	f << "\\hline" << endl;
	f << "\\end{array}" << endl;
	f << "$$" << endl;
	f << "\\clearpage" << endl;

	if (f_v) {
		cout << "projective_space::cheat_sheet_line_through_pairs_of_points done" << endl;
	}

}

void projective_space::conic_type_randomized(int nb_times,
	long int *set, int set_size,
	long int **&Pts_on_conic, int *&nb_pts_on_conic, int &len,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::conic_type_randomized" << endl;
	}
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	int rk, h, i, j, a, /*d,*/ N, l, cnt;

	long int input_pts[5];
	int six_coeffs[6];
	int vec[3];

	int subset[5];
	ring_theory::longinteger_object conic_rk, aa;
	long int *pts_on_conic;
	int allocation_length;
	geometry_global Gg;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;
	orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "projective_space::conic_type_randomized" << endl;
	}
	if (n != 2) {
		cout << "projective_space::conic_type_randomized "
				"n != 2" << endl;
		exit(1);
	}
	if (f_vv) {
		print_set_numerical(cout, set, set_size);
	}

	if (!Sorting.test_if_set_with_return_value_lint(set, set_size)) {
		cout << "projective_space::conic_type_randomized "
				"the input set if not a set" << endl;
		exit(1);
	}
	//d = n + 1;
	N = Combi.int_n_choose_k(set_size, 5);

	if (f_v) {
		cout << "set_size=" << set_size << endl;
		cout << "N=number of 5-subsets of the set=" << N << endl;
	}

	// allocate data that is returned:
	allocation_length = 1024;
	Pts_on_conic = NEW_plint(allocation_length);
	nb_pts_on_conic = NEW_int(allocation_length);


	len = 0;
	for (cnt = 0; cnt < nb_times; cnt++) {

		rk = Os.random_integer(N);
		Combi.unrank_k_subset(rk, subset, set_size, 5);
		if (cnt && ((cnt % 1000) == 0)) {
			cout << cnt << " / " << nb_times << " : ";
			Int_vec_print(cout, subset, 5);
			cout << endl;
		}

		for (i = 0; i < len; i++) {
			if (Sorting.lint_vec_is_subset_of(subset, 5,
					Pts_on_conic[i], nb_pts_on_conic[i], 0 /* verbose_level */)) {

#if 0
				cout << "The set ";
				int_vec_print(cout, subset, 5);
				cout << " is a subset of the " << i << "th conic ";
				int_vec_print(cout, Pts_on_conic[i], nb_pts_on_conic[i]);
				cout << endl;
#endif

				break;
			}
		}
		if (i < len) {
			continue;
		}
		for (j = 0; j < 5; j++) {
			a = subset[j];
			input_pts[j] = set[a];
		}
		if (FALSE /* f_v3 */) {
			cout << "subset: ";
			Int_vec_print(cout, subset, 5);
			cout << "input_pts: ";
			Lint_vec_print(cout, input_pts, 5);
		}

		if (!determine_conic_in_plane(input_pts,
				5, six_coeffs, 0 /* verbose_level */)) {
			continue;
		}


		F->PG_element_normalize(six_coeffs, 1, 6);
		Gg.AG_element_rank_longinteger(F->q, six_coeffs, 1, 6, conic_rk);
		if (FALSE /* f_vv */) {
			cout << rk << "-th subset ";
			Int_vec_print(cout, subset, 5);
			cout << " conic_rk=" << conic_rk << endl;
		}

		if (FALSE /* longinteger_vec_search(R, len, conic_rk, idx) */) {

#if 0
			cout << "projective_space::conic_type_randomized "
					"longinteger_vec_search(R, len, conic_rk, idx) "
					"is TRUE" << endl;
			cout << "The current set is ";
			int_vec_print(cout, subset, 5);
			cout << endl;
			cout << "conic_rk=" << conic_rk << endl;
			cout << "The set where it should be is ";
			int_vec_print(cout, Pts_on_conic[idx], nb_pts_on_conic[idx]);
			cout << endl;
			cout << "R[idx]=" << R[idx] << endl;
			cout << "This is the " << idx << "th conic" << endl;
			exit(1);
#endif

		}
		else {
			if (f_v3) {
				cout << "conic_rk=" << conic_rk << " was not found" << endl;
			}
			pts_on_conic = NEW_lint(set_size);
			l = 0;
			for (h = 0; h < set_size; h++) {
				if (FALSE && f_v3) {
					cout << "testing point " << h << ":" << endl;
					cout << "conic_rk=" << conic_rk << endl;
				}

				unrank_point(vec, set[h]);
				a = F->Linear_algebra->evaluate_conic_form(six_coeffs, vec);


				if (a == 0) {
					pts_on_conic[l++] = h;
					if (f_v3) {
						cout << "point " << h << " is on the conic" << endl;
					}
				}
				else {
					if (FALSE && f_v3) {
						cout << "point " << h
								<< " is not on the conic" << endl;
					}
				}
			}
			if (FALSE /*f_v*/) {
				cout << "We found an " << l
						<< "-conic, its rank is " << conic_rk << endl;


			}


			if (l >= 8) {

				if (f_v) {
					cout << "We found an " << l << "-conic, "
							"its rank is " << conic_rk << endl;
					cout << "The " << l << " points on the "
							<< len << "th conic are: ";
					Lint_vec_print(cout, pts_on_conic, l);
					cout << endl;



				}


#if 0
				for (j = len; j > idx; j--) {
					R[j].swap_with(R[j - 1]);
					Pts_on_conic[j] = Pts_on_conic[j - 1];
					nb_pts_on_conic[j] = nb_pts_on_conic[j - 1];
					}
				conic_rk.assign_to(R[idx]);
				Pts_on_conic[idx] = pts_on_conic;
				nb_pts_on_conic[idx] = l;
#else

				//conic_rk.assign_to(R[len]);
				Pts_on_conic[len] = pts_on_conic;
				nb_pts_on_conic[len] = l;

#endif


				len++;
				if (f_v) {
					cout << "We now have found " << len
							<< " conics" << endl;


					data_structures::tally C;
					int f_second = FALSE;

					C.init(nb_pts_on_conic, len, f_second, 0);

					if (f_v) {
						cout << "The conic intersection type is (";
						C.print_naked(FALSE /*f_backwards*/);
						cout << ")" << endl << endl;
					}



				}

				if (len == allocation_length) {
					int new_allocation_length = allocation_length + 1024;


					long int **Pts_on_conic1;
					int *nb_pts_on_conic1;

					Pts_on_conic1 = NEW_plint(new_allocation_length);
					nb_pts_on_conic1 = NEW_int(new_allocation_length);
					for (i = 0; i < len; i++) {
						//R1[i] = R[i];
						Pts_on_conic1[i] = Pts_on_conic[i];
						nb_pts_on_conic1[i] = nb_pts_on_conic[i];
					}
					FREE_plint(Pts_on_conic);
					FREE_int(nb_pts_on_conic);
					Pts_on_conic = Pts_on_conic1;
					nb_pts_on_conic = nb_pts_on_conic1;
					allocation_length = new_allocation_length;
				}




				}
			else {
				// we skip this conic:

				FREE_lint(pts_on_conic);
			}
		} // else
	} // next rk
	if (f_v) {
		cout << "projective_space::conic_type_randomized done" << endl;
	}
}

void projective_space::conic_intersection_type(
	int f_randomized, int nb_times,
	long int *set, int set_size,
	int threshold,
	int *&intersection_type, int &highest_intersection_number,
	int f_save_largest_sets, data_structures::set_of_sets *&largest_sets,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//longinteger_object *R;
	long int **Pts_on_conic;
	int **Conic_eqn;
	int *nb_pts_on_conic;
	int nb_conics;
	int i, j, idx, f, l, a, t;

	if (f_v) {
		cout << "projective_space::conic_intersection_type threshold = " << threshold << endl;
	}

	if (f_randomized) {
		if (f_v) {
			cout << "projective_space::conic_intersection_type "
					"randomized" << endl;
		}
		conic_type_randomized(nb_times,
			set, set_size,
			Pts_on_conic, nb_pts_on_conic, nb_conics,
			verbose_level - 1);
	}
	else {
		if (f_v) {
			cout << "projective_space::conic_intersection_type "
					"not randomized" << endl;
		}
		conic_type(
			set, set_size, threshold,
			Pts_on_conic, Conic_eqn, nb_pts_on_conic, nb_conics,
			verbose_level - 1);
	}

	data_structures::tally C;
	int f_second = FALSE;

	C.init(nb_pts_on_conic, nb_conics, f_second, 0);
	if (f_v) {
		cout << "projective_space::conic_intersection_type "
				"conic-intersection type: ";
		C.print(FALSE /*f_backwards*/);
	}

	if (f_v) {
		cout << "The conic intersection type is (";
		C.print_naked(FALSE /*f_backwards*/);
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

	if (f_save_largest_sets) {
		largest_sets = NEW_OBJECT(data_structures::set_of_sets);
		t = C.nb_types - 1;
		f = C.type_first[t];
		l = C.type_len[t];
		largest_sets->init_basic_constant_size(set_size, l,
				highest_intersection_number, verbose_level);
		for (j = 0; j < l; j++) {
			idx = C.sorting_perm_inv[f + j];
			Lint_vec_copy(Pts_on_conic[idx],
					largest_sets->Sets[j],
					highest_intersection_number);
		}
	}

	for (i = 0; i < nb_conics; i++) {
		FREE_lint(Pts_on_conic[i]);
		FREE_int(Conic_eqn[i]);
	}
	FREE_plint(Pts_on_conic);
	FREE_pint(Conic_eqn);
	FREE_int(nb_pts_on_conic);
	if (f_v) {
		cout << "projective_space::conic_intersection_type done" << endl;
	}

}

void projective_space::determine_nonconical_six_subsets(
	long int *set, int set_size,
	std::vector<int> &Rk,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk, i;
	int threshold = 6;
	int N;

	long int **Pts_on_conic;
	int **Conic_eqn;
	int *nb_pts_on_conic;
	int len;

	//geometry_global Gg;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space::determine_nonconical_six_subsets" << endl;
	}
	if (n != 2) {
		cout << "projective_space::determine_nonconical_six_subsets n != 2" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "projective_space::determine_nonconical_six_subsets before conic_type" << endl;
	}
	conic_type(
		set, set_size,
		threshold,
		Pts_on_conic, Conic_eqn, nb_pts_on_conic, len,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space::determine_nonconical_six_subsets after conic_type" << endl;
	}
	if (f_v) {
		cout << "There are " << len << " conics. They contain the following points:" << endl;
		for (i = 0; i < len; i++) {
			cout << i << " : " << nb_pts_on_conic[i] << " : ";
			Lint_vec_print(cout, Pts_on_conic[i], nb_pts_on_conic[i]);
			cout << endl;
		}
	}

	int subset[6];

	N = Combi.int_n_choose_k(set_size, 6);

	if (f_v) {
		cout << "set_size=" << set_size << endl;
		cout << "N=number of 6-subsets of the set=" << N << endl;
	}


	for (rk = 0; rk < N; rk++) {

		Combi.unrank_k_subset(rk, subset, set_size, 6);
		if (f_v) {
			cout << "projective_space::conic_type rk=" << rk << " / " << N << " : ";
			Int_vec_print(cout, subset, 6);
			cout << endl;
		}

		for (i = 0; i < len; i++) {
			if (Sorting.lint_vec_is_subset_of(subset, 6,
					Pts_on_conic[i], nb_pts_on_conic[i], 0 /* verbose_level */)) {

#if 1
				if (f_v) {
					cout << "The set ";
					Int_vec_print(cout, subset, 6);
					cout << " is a subset of the " << i << "th conic ";
					Lint_vec_print(cout,
							Pts_on_conic[i], nb_pts_on_conic[i]);
					cout << endl;
				}
#endif

				break;
			}
			else {
				if (FALSE) {
					cout << " not on conic " << i << endl;
				}
			}
		}
		if (i == len) {
			Rk.push_back(rk);
		}
	}

	for (i = 0; i < len; i++) {
		FREE_lint(Pts_on_conic[i]);
		FREE_int(Conic_eqn[i]);
	}
	FREE_plint(Pts_on_conic);
	FREE_pint(Conic_eqn);
	FREE_int(nb_pts_on_conic);

	int nb, j, nb_E;
	int *Nb_E;
	long int Arc6[6];

	nb = Rk.size();
	Nb_E = NEW_int(nb);
	if (f_v) {
		cout << "computing Eckardt point number distribution" << endl;
	}
	for (i = 0; i < nb; i++) {
		if ((i % 500) == 0) {
			cout << i << " / " << nb << endl;
		}
		rk = Rk[i];
		Combi.unrank_k_subset(rk, subset, set_size, 6);
		for (j = 0; j < 6; j++) {
			Arc6[j] = set[subset[j]];
		}
		nb_E = nonconical_six_arc_get_nb_Eckardt_points(
				Arc6, 0 /* verbose_level */);
		Nb_E[i] = nb_E;
	}

	data_structures::tally T;

	T.init(Nb_E, nb, FALSE, 0);
	if (f_v) {
		cout << "Eckardt point number distribution : ";
		T.print_file_tex(cout, TRUE /* f_backwards*/);
		cout << endl;
	}

	if (nb) {
		int m, idx;
		int *Idx;
		int nb_idx;
		int *System;

		m = orbiter_kernel_system::Orbiter->Int_vec->maximum(Nb_E, nb);
		T.get_class_by_value(Idx, nb_idx, m /* value */, verbose_level);
		if (f_v) {
			cout << "The class of " << m << " is ";
			Int_vec_print(cout, Idx, nb_idx);
			cout << endl;
		}

		System = NEW_int(nb_idx * 6);

		for (i = 0; i < nb_idx; i++) {
			idx = Idx[i];

			rk = Rk[idx];
			if (f_v) {
				cout << i << " / " << nb_idx << " idx=" << idx << ", rk=" << rk << " :" << endl;
			}
			Combi.unrank_k_subset(rk, subset, set_size, 6);

			Int_vec_copy(subset, System + i * 6, 6);

			for (j = 0; j < 6; j++) {
				Arc6[j] = set[subset[j]];
			}
			nb_E = nonconical_six_arc_get_nb_Eckardt_points(
					Arc6, 0 /* verbose_level */);
			if (nb_E != m) {
				cout << "nb_E != m" << endl;
				exit(1);
			}
			if (f_v) {
				cout << "The subset is ";
				Int_vec_print(cout, subset, 6);
				cout << " : ";
				cout << " the arc is ";
				Lint_vec_print(cout, Arc6, 6);
				cout << " nb_E = " << nb_E << endl;
			}
		}

		orbiter_kernel_system::file_io Fio;
		std::string fname;

		fname.assign("set_system.csv");
		Fio.int_matrix_write_csv(fname, System, nb_idx, 6);
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


		data_structures::tally T2;

		T2.init(System, nb_idx * 6, FALSE, 0);
		if (f_v) {
			cout << "distribution of points: ";
			T2.print_file_tex(cout, TRUE /* f_backwards*/);
			cout << endl;
		}


	}


	if (f_v) {
		cout << "projective_space::determine_nonconical_six_subsets done" << endl;
	}
}

void projective_space::conic_type(
	long int *set, int set_size,
	int threshold,
	long int **&Pts_on_conic, int **&Conic_eqn, int *&nb_pts_on_conic, int &nb_conics,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	int rk, h, i, j, a, /*d,*/ N, l;

	long int input_pts[5];
	int six_coeffs[6];
	int vec[3];

	int subset[5];
	ring_theory::longinteger_object conic_rk, aa;
	int *coords;
	long int *pts_on_conic;
	int allocation_length;
	geometry_global Gg;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space::conic_type, threshold = " << threshold << endl;
	}
	if (n != 2) {
		cout << "projective_space::conic_type n != 2" << endl;
		exit(1);
	}
	if (f_vv) {
		print_set_numerical(cout, set, set_size);
	}

	if (!Sorting.test_if_set_with_return_value_lint(set, set_size)) {
		cout << "projective_space::conic_type the input "
				"set if not a set" << endl;
		exit(1);
	}
	//d = n + 1;
	N = Combi.int_n_choose_k(set_size, 5);

	if (f_v) {
		cout << "set_size=" << set_size << endl;
		cout << "N=number of 5-subsets of the set=" << N << endl;
	}


	coords = NEW_int(set_size * 3);
	for (i = 0; i < set_size; i++) {
		unrank_point(coords + i * 3, set[i]);
	}
	if (f_v) {
		cout << "projective_space::conic_type coords:" << endl;
		Int_vec_print_integer_matrix(cout, coords, set_size, 3);
	}


	// allocate data that is returned:
	allocation_length = 1024;
	Pts_on_conic = NEW_plint(allocation_length);
	Conic_eqn = NEW_pint(allocation_length);
	nb_pts_on_conic = NEW_int(allocation_length);


	nb_conics = 0;
	for (rk = 0; rk < N; rk++) {

		Combi.unrank_k_subset(rk, subset, set_size, 5);
		if (FALSE) {
			cout << "projective_space::conic_type rk=" << rk << " / " << N << " : ";
			Int_vec_print(cout, subset, 5);
			cout << endl;
		}

		for (i = 0; i < nb_conics; i++) {
			if (Sorting.lint_vec_is_subset_of(subset, 5,
					Pts_on_conic[i], nb_pts_on_conic[i], 0)) {

#if 0
				cout << "The set ";
				int_vec_print(cout, subset, 5);
				cout << " is a subset of the " << i << "th conic ";
				int_vec_print(cout,
						Pts_on_conic[i], nb_pts_on_conic[i]);
				cout << endl;
#endif

				break;
			}
		}
		if (i < nb_conics) {
			continue;
		}
		for (j = 0; j < 5; j++) {
			a = subset[j];
			input_pts[j] = set[a];
		}
		if (FALSE) {
			cout << "subset: ";
			Int_vec_print(cout, subset, 5);
			cout << endl;
			cout << "input_pts: ";
			Lint_vec_print(cout, input_pts, 5);
			cout << endl;
		}

		if (!determine_conic_in_plane(input_pts, 5,
				six_coeffs, verbose_level - 2)) {
			if (FALSE) {
				cout << "determine_conic_in_plane returns FALSE" << endl;
			}
			continue;
		}
		if (f_v) {
			cout << "projective_space::conic_type rk=" << rk << " / " << N << " : ";
			Int_vec_print(cout, subset, 5);
			cout << " has not yet been considered and a conic exists" << endl;
		}
		if (f_v) {
			cout << "determine_conic_in_plane the conic exists" << endl;
			cout << "conic: ";
			Int_vec_print(cout, six_coeffs, 6);
			cout << endl;
		}


		F->PG_element_normalize(six_coeffs, 1, 6);
		Gg.AG_element_rank_longinteger(F->q, six_coeffs, 1, 6, conic_rk);
		if (FALSE /* f_vv */) {
			cout << rk << "-th subset ";
			Int_vec_print(cout, subset, 5);
			cout << " conic_rk=" << conic_rk << endl;
		}

		if (FALSE /* longinteger_vec_search(R, len, conic_rk, idx) */) {

#if 0
			cout << "projective_space::conic_type_randomized "
					"longinteger_vec_search(R, len, conic_rk, idx) "
					"is TRUE" << endl;
			cout << "The current set is ";
			int_vec_print(cout, subset, 5);
			cout << endl;
			cout << "conic_rk=" << conic_rk << endl;
			cout << "The set where it should be is ";
			int_vec_print(cout, Pts_on_conic[idx], nb_pts_on_conic[idx]);
			cout << endl;
			cout << "R[idx]=" << R[idx] << endl;
			cout << "This is the " << idx << "th conic" << endl;
			exit(1);
#endif

		}
		else {
			if (f_v) {
				cout << "considering conic of rank conic_rk=" << conic_rk << ":" << endl;
			}
			pts_on_conic = NEW_lint(set_size);
			l = 0;
			for (h = 0; h < set_size; h++) {

				//unrank_point(vec, set[h]);
				Int_vec_copy(coords + h * 3, vec, 3);
				if (f_v) {
					cout << "testing point " << h << ":" << endl;
					Int_vec_print(cout, vec, 3);
					cout << endl;
				}
				a = F->Linear_algebra->evaluate_conic_form(six_coeffs, vec);


				if (a == 0) {
					pts_on_conic[l++] = h;
					if (FALSE) {
						cout << "point " << h
								<< " is on the conic" << endl;
					}
				}
				else {
					if (FALSE && f_v3) {
						cout << "point " << h
								<< " is not on the conic" << endl;
					}
				}
			}
			if (f_v) {
				cout << "We found an " << l << "-conic, "
						"its rank is " << conic_rk << endl;


			}


			if (l >= threshold) {

				if (f_v) {
					cout << "We found an " << l << "-conic, "
							"its rank is " << conic_rk << endl;
					cout << "The " << l << " points on the "
							<< nb_conics << "th conic are: ";
					Lint_vec_print(cout, pts_on_conic, l);
					cout << endl;
				}


#if 0
				for (j = len; j > idx; j--) {
					R[j].swap_with(R[j - 1]);
					Pts_on_conic[j] = Pts_on_conic[j - 1];
					nb_pts_on_conic[j] = nb_pts_on_conic[j - 1];
				}
				conic_rk.assign_to(R[idx]);
				Pts_on_conic[idx] = pts_on_conic;
				nb_pts_on_conic[idx] = l;
#else

				//conic_rk.assign_to(R[len]);
				Pts_on_conic[nb_conics] = pts_on_conic;
				Conic_eqn[nb_conics] = NEW_int(6);
				Int_vec_copy(six_coeffs, Conic_eqn[nb_conics], 6);
				nb_pts_on_conic[nb_conics] = l;

#endif


				nb_conics++;
				if (f_v) {
					cout << "We now have found " << nb_conics
							<< " conics" << endl;


					data_structures::tally C;
					int f_second = FALSE;

					C.init(nb_pts_on_conic, nb_conics, f_second, 0);

					if (f_v) {
						cout << "The conic intersection type is (";
						C.print_naked(FALSE /*f_backwards*/);
						cout << ")" << endl << endl;
					}



				}

				if (nb_conics == allocation_length) {
					int new_allocation_length = allocation_length + 1024;


					long int **Pts_on_conic1;
					int **Conic_eqn1;
					int *nb_pts_on_conic1;

					Pts_on_conic1 = NEW_plint(new_allocation_length);
					Conic_eqn1 = NEW_pint(new_allocation_length);
					nb_pts_on_conic1 = NEW_int(new_allocation_length);
					for (i = 0; i < nb_conics; i++) {
						//R1[i] = R[i];
						Pts_on_conic1[i] = Pts_on_conic[i];
						Conic_eqn1[i] = Conic_eqn[i];
						nb_pts_on_conic1[i] = nb_pts_on_conic[i];
					}
					FREE_plint(Pts_on_conic);
					FREE_pint(Conic_eqn);
					FREE_int(nb_pts_on_conic);
					Pts_on_conic = Pts_on_conic1;
					Conic_eqn = Conic_eqn1;
					nb_pts_on_conic = nb_pts_on_conic1;
					allocation_length = new_allocation_length;
				}




			}
			else {
				// we skip this conic:
				if (f_v) {
					cout << "projective_space::conic_type we skip this conic" << endl;
				}
				FREE_lint(pts_on_conic);
			}
		} // else
	} // next rk

	FREE_int(coords);

	if (f_v) {
		cout << "projective_space::conic_type we found " << nb_conics
				<< " conics intersecting in at least "
				<< threshold << " many points" << endl;
	}

	if (f_v) {
		cout << "projective_space::conic_type done" << endl;
	}
}

void projective_space::find_nucleus(
	int *set, int set_size, int &nucleus,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, l, sz, idx, t1, t2;
	int *Lines;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space::find_nucleus" << endl;
	}

	if (n != 2) {
		cout << "projective_space::find_nucleus n != 2" << endl;
		exit(1);
	}
	if (set_size != F->q + 1) {
		cout << "projective_space::find_nucleus "
				"set_size != F->q + 1" << endl;
		exit(1);
	}

	if (Implementation->Lines_on_point == NULL) {
		init_incidence_structure(verbose_level);
	}

	Lines = NEW_int(r);
	a = set[0];
	for (i = 0; i < r; i++) {
		Lines[i] = Implementation->Lines_on_point[a * r + i];
	}
	sz = r;
	Sorting.int_vec_heapsort(Lines, r);

	for (i = 0; i < set_size - 1; i++) {
		b = set[1 + i];
		l = line_through_two_points(a, b);
		if (!Sorting.int_vec_search(Lines, sz, l, idx)) {
			cout << "projective_space::find_nucleus "
					"cannot find secant in pencil" << endl;
			exit(1);
		}
		for (j = idx + 1; j < sz; j++) {
			Lines[j - 1] = Lines[j];
		}
		sz--;
	}
	if (sz != 1) {
		cout << "projective_space::find_nucleus sz != 1" << endl;
		exit(1);
	}
	t1 = Lines[0];
	if (f_v) {
		cout << "projective_space::find_nucleus t1 = " << t1 << endl;
	}



	a = set[1];
	for (i = 0; i < r; i++) {
		Lines[i] = Implementation->Lines_on_point[a * r + i];
	}
	sz = r;
	Sorting.int_vec_heapsort(Lines, r);

	for (i = 0; i < set_size - 1; i++) {
		if (i == 0) {
			b = set[0];
		}
		else {
			b = set[1 + i];
		}
		l = line_through_two_points(a, b);
		if (!Sorting.int_vec_search(Lines, sz, l, idx)) {
			cout << "projective_space::find_nucleus "
					"cannot find secant in pencil" << endl;
			exit(1);
		}
		for (j = idx + 1; j < sz; j++) {
			Lines[j - 1] = Lines[j];
		}
		sz--;
	}
	if (sz != 1) {
		cout << "projective_space::find_nucleus sz != 1" << endl;
		exit(1);
	}
	t2 = Lines[0];
	if (f_v) {
		cout << "projective_space::find_nucleus t2 = " << t2 << endl;
	}

	nucleus = intersection_of_two_lines(t1, t2);
	if (f_v) {
		cout << "projective_space::find_nucleus "
				"nucleus = " << nucleus << endl;
		int v[3];
		unrank_point(v, nucleus);
		cout << "nucleus = ";
		Int_vec_print(cout, v, 3);
		cout << endl;
	}



	if (f_v) {
		cout << "projective_space::find_nucleus done" << endl;
	}
}

void projective_space::points_on_projective_triangle(
	long int *&set, int &set_size, long int *three_points,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int three_lines[3];
	long int *Pts;
	int sz, h, i, a;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space::points_on_projective_triangle" << endl;
	}
	set_size = 3 * (q - 1);
	set = NEW_lint(set_size);
	sz = 3 * (q + 1);
	Pts = NEW_lint(sz);
	three_lines[0] = line_through_two_points(three_points[0], three_points[1]);
	three_lines[1] = line_through_two_points(three_points[0], three_points[2]);
	three_lines[2] = line_through_two_points(three_points[1], three_points[2]);

	create_points_on_line(three_lines[0], Pts, 0 /* verbose_level */);
	create_points_on_line(three_lines[1], Pts + (q + 1), 0 /* verbose_level */);
	create_points_on_line(three_lines[2], Pts + 2 * (q + 1), 0 /* verbose_level */);
	h = 0;
	for (i = 0; i < sz; i++) {
		a = Pts[i];
		if (a == three_points[0]) {
			continue;
		}
		if (a == three_points[1]) {
			continue;
		}
		if (a == three_points[2]) {
			continue;
		}
		set[h++] = a;
	}
	if (h != set_size) {
		cout << "projective_space::points_on_projective_triangle "
				"h != set_size" << endl;
		exit(1);
	}
	Sorting.lint_vec_heapsort(set, set_size);

	FREE_lint(Pts);
	if (f_v) {
		cout << "projective_space::points_on_projective_triangle "
				"done" << endl;
	}
}

void projective_space::elliptic_curve_addition_table(
	int *A6, int *Pts, int nb_pts, int *&Table,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k;
	int pi, pj, pk;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space::elliptic_curve_addition_table" << endl;
	}
	Table = NEW_int(nb_pts * nb_pts);
	for (i = 0; i < nb_pts; i++) {
		pi = Pts[i];
		for (j = 0; j < nb_pts; j++) {
			pj = Pts[j];
			pk = elliptic_curve_addition(A6, pi, pj,
					0 /* verbose_level */);
			if (!Sorting.int_vec_search(Pts, nb_pts, pk, k)) {
				cout << "projective_space::elliptic_curve_addition_table cannot find point pk" << endl;
				cout << "i=" << i << " pi=" << pi << " j=" << j
						<< " pj=" << pj << " pk=" << pk << endl;
				cout << "Pts: ";
				Int_vec_print(cout, Pts, nb_pts);
				cout << endl;
				exit(1);
			}
			Table[i * nb_pts + j] = k;
		}
	}
	if (f_v) {
		cout << "projective_space::elliptic_curve_addition_table done" << endl;
	}
}

int projective_space::elliptic_curve_addition(
	int *A6, int p1_rk, int p2_rk,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int p1[3];
	int p2[3];
	int p3[3];
	int x1, y1, z1;
	int x2, y2, z2;
	int x3, y3, z3;
	int a1, a2, a3, a4, a6;
	int p3_rk;

	if (f_v) {
		cout << "projective_space::elliptic_curve_addition" << endl;
	}

	a1 = A6[0];
	a2 = A6[1];
	a3 = A6[2];
	a4 = A6[3];
	a6 = A6[5];

	unrank_point(p1, p1_rk);
	unrank_point(p2, p2_rk);
	F->PG_element_normalize(p1, 1, 3);
	F->PG_element_normalize(p2, 1, 3);

	x1 = p1[0];
	y1 = p1[1];
	z1 = p1[2];
	x2 = p2[0];
	y2 = p2[1];
	z2 = p2[2];
	if (f_vv) {
		cout << "projective_space::elliptic_curve_addition "
				"x1=" << x1 << " y1=" << y1 << " z1=" << z1 << endl;
		cout << "projective_space::elliptic_curve_addition "
				"x2=" << x2 << " y2=" << y2 << " z2=" << z2 << endl;
	}
	if (z1 == 0) {
		if (p1_rk != 1) {
			cout << "projective_space::elliptic_curve_addition "
					"z1 == 0 && p1_rk != 1" << endl;
			exit(1);
		}
		x3 = x2;
		y3 = y2;
		z3 = z2;
#if 0
		if (z2 == 0) {
			if (p2_rk != 1) {
				cout << "projective_space::elliptic_curve_addition "
						"z2 == 0 && p2_rk != 1" << endl;
				exit(1);
			}
			x3 = 0;
			y3 = 1;
			z3 = 0;
		}
		else {
			x3 = x2;
			y3 = F->negate(F->add3(y2, F->mult(a1, x2), a3));
			z3 = 1;
		}
#endif

	}
	else if (z2 == 0) {
		if (p2_rk != 1) {
			cout << "projective_space::elliptic_curve_addition "
					"z2 == 0 && p2_rk != 1" << endl;
			exit(1);
		}
		x3 = x1;
		y3 = y1;
		z3 = z1;

#if 0
		// at this point, we know that z1 is not zero.
		x3 = x1;
		y3 = F->negate(F->add3(y1, F->mult(a1, x1), a3));
		z3 = 1;
#endif

	}
	else {
		// now both points are affine.


		int lambda_top, lambda_bottom, lambda, nu_top, nu_bottom, nu;
		int three, two; //, m_one;
		int c;

		c = F->add4(y1, y2, F->mult(a1, x2), a3);

		if (x1 == x2 && c == 0) {
			x3 = 0;
			y3 = 1;
			z3 = 0;
		}
		else {

			two = F->add(1, 1);
			three = F->add(two, 1);
			//m_one = F->negate(1);



			if (x1 == x2) {

				// point duplication:
				lambda_top = F->add4(F->mult3(three, x1, x1),
						F->mult3(two, a2, x1), a4,
						F->negate(F->mult(a1, y1)));
				lambda_bottom = F->add3(F->mult(two, y1),
						F->mult(a1, x1), a3);

				nu_top = F->add4(F->negate(F->mult3(x1, x1, x1)),
						F->mult(a4, x1), F->mult(two, a6),
						F->negate(F->mult(a3, y1)));
				nu_bottom = F->add3(F->mult(two, y1),
						F->mult(a1, x1), a3);

			}
			else {
				// adding different points:
				lambda_top = F->add(y2, F->negate(y1));
				lambda_bottom = F->add(x2, F->negate(x1));

				nu_top = F->add(F->mult(y1, x2), F->negate(F->mult(y2, x1)));
				nu_bottom = lambda_bottom;
			}


			if (lambda_bottom == 0) {
				cout << "projective_space::elliptic_curve_addition "
						"lambda_bottom == 0" << endl;
				cout << "projective_space::elliptic_curve_addition "
						"x1=" << x1 << " y1=" << y1 << " z1=" << z1 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"x2=" << x2 << " y2=" << y2 << " z2=" << z2 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a1=" << a1 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a2=" << a2 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a3=" << a3 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a4=" << a4 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a6=" << a6 << endl;
				exit(1);
			}
			lambda = F->mult(lambda_top, F->inverse(lambda_bottom));

			if (nu_bottom == 0) {
				cout << "projective_space::elliptic_curve_addition "
						"nu_bottom == 0" << endl;
				exit(1);
			}
			nu = F->mult(nu_top, F->inverse(nu_bottom));

			if (f_vv) {
				cout << "projective_space::elliptic_curve_addition "
						"a1=" << a1 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a2=" << a2 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a3=" << a3 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a4=" << a4 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a6=" << a6 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"three=" << three << endl;
				cout << "projective_space::elliptic_curve_addition "
						"lambda_top=" << lambda_top << endl;
				cout << "projective_space::elliptic_curve_addition "
						"lambda=" << lambda << " nu=" << nu << endl;
			}
			x3 = F->add3(F->mult(lambda, lambda), F->mult(a1, lambda),
					F->negate(F->add3(a2, x1, x2)));
			y3 = F->negate(F->add3(F->mult(F->add(lambda, a1), x3), nu, a3));
			z3 = 1;
		}
	}
	p3[0] = x3;
	p3[1] = y3;
	p3[2] = z3;
	if (f_vv) {
		cout << "projective_space::elliptic_curve_addition "
				"x3=" << x3 << " y3=" << y3 << " z3=" << z3 << endl;
	}
	p3_rk = rank_point(p3);
	if (f_v) {
		cout << "projective_space::elliptic_curve_addition "
				"done" << endl;
	}
	return p3_rk;
}


void projective_space::line_plane_incidence_matrix_restricted(
	long int *Lines, int nb_lines, int *&M, int &nb_planes,
	int verbose_level)
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
	if (n <= 2) {
		cout << "projective_space::line_plane_incidence_matrix_"
				"restricted n <= 2" << endl;
		exit(1);
	}
	line_sz = 2 * (n + 1);
	nb_planes = Nb_subspaces[2];

	M = NEW_int(nb_lines * nb_planes);
	Basis = NEW_int(3 * (n + 1));
	Work = NEW_int(5 * (n + 1));
	the_lines = NEW_int(nb_lines * line_sz);


	Int_vec_zero(M, nb_lines * nb_planes);
	for (i = 0; i < nb_lines; i++) {
		unrank_line(the_lines + i * line_sz, Lines[i]);
	}
	for (j = 0; j < nb_planes; j++) {
		unrank_plane(Basis, j);
		for (i = 0; i < nb_lines; i++) {
			Int_vec_copy(Basis, Work, 3 * (n + 1));
			Int_vec_copy(the_lines + i * line_sz,
					Work + 3 * (n + 1), line_sz);
			if (F->Linear_algebra->Gauss_easy(Work, 5, n + 1) == 3) {
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

int projective_space::test_if_lines_are_skew(
	int line1, int line2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis1[4 * 4];
	int Basis2[4 * 4];
	int rk;
	int M[16];

	if (f_v) {
		cout << "projective_space::test_if_lines_are_skew" << endl;
	}
	if (n != 3) {
		cout << "projective_space::test_if_lines_are_skew "
				"n != 3" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "line1=" << line1 << " line2=" << line2 << endl;
	}
	unrank_line(Basis1, line1);
	if (f_v) {
		cout << "line1:" << endl;
		Int_matrix_print(Basis1, 2, 4);
	}
	unrank_line(Basis2, line2);
	if (f_v) {
		cout << "line2:" << endl;
		Int_matrix_print(Basis2, 2, 4);
	}
	F->Linear_algebra->intersect_subspaces(4, 2, Basis1, 2, Basis2,
		rk, M, 0 /* verbose_level */);

	if (f_v) {
		cout << "projective_space::test_if_lines_are_skew done" << endl;
	}

	if (rk == 0) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

int projective_space::point_of_intersection_of_a_line_and_a_line_in_three_space(
	long int line1, long int line2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis1[4 * 4];
	int Basis2[4 * 4];
	int rk, a;
	int M[16];

	if (f_v) {
		cout << "projective_space::point_of_intersection_of_a_line_and_a_line_in_three_space" << endl;
	}
	if (n != 3) {
		cout << "projective_space::point_of_intersection_of_a_line_and_a_line_in_three_space n != 3" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "line1=" << line1 << " line2=" << line2 << endl;
	}
	unrank_line(Basis1, line1);
	if (f_v) {
		cout << "line1:" << endl;
		Int_matrix_print(Basis1, 2, 4);
	}
	unrank_line(Basis2, line2);
	if (f_v) {
		cout << "line2:" << endl;
		Int_matrix_print(Basis2, 2, 4);
	}
	F->Linear_algebra->intersect_subspaces(4, 2, Basis1, 2, Basis2,
		rk, M, 0 /* verbose_level */);
	if (rk != 1) {
		cout << "projective_space::point_of_intersection_of_a_line_and_a_line_in_three_space intersection "
				"is not a point" << endl;
		cout << "line1:" << endl;
		Int_matrix_print(Basis1, 2, 4);
		cout << "line2:" << endl;
		Int_matrix_print(Basis2, 2, 4);
		cout << "rk = " << rk << endl;
		exit(1);
	}
	if (f_v) {
		cout << "intersection:" << endl;
		Int_matrix_print(M, 1, 4);
	}
	a = rank_point(M);
	if (f_v) {
		cout << "point rank = " << a << endl;
	}
	if (f_v) {
		cout << "projective_space::point_of_intersection_of_a_line_and_a_line_in_three_space done" << endl;
	}
	return a;
}

int projective_space::point_of_intersection_of_a_line_and_a_plane_in_three_space(
	long int line, int plane, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis1[4 * 4];
	int Basis2[4 * 4];
	int rk, a;
	int M[16];

	if (f_v) {
		cout << "projective_space::point_of_intersection_of_a_line_and_a_plane_in_three_space" << endl;
	}
	if (n != 3) {
		cout << "projective_space::point_of_intersection_of_a_line_and_a_plane_in_three_space n != 3" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "line=" << line << " plane=" << plane << endl;
	}
	unrank_line(Basis1, line);
	if (f_v) {
		cout << "line:" << endl;
		Int_matrix_print(Basis1, 2, 4);
	}
	unrank_plane(Basis2, plane);
	if (f_v) {
		cout << "plane:" << endl;
		Int_matrix_print(Basis2, 3, 4);
	}
	F->Linear_algebra->intersect_subspaces(4, 2, Basis1, 3, Basis2,
		rk, M, 0 /* verbose_level */);
	if (rk != 1) {
		cout << "projective_space::point_of_intersection_of_a_line_and_a_plane_in_three_space intersection "
				"is not a point" << endl;
	}
	if (f_v) {
		cout << "intersection:" << endl;
		Int_matrix_print(M, 1, 4);
	}
	a = rank_point(M);
	if (f_v) {
		cout << "point rank = " << a << endl;
	}
	if (f_v) {
		cout << "projective_space::point_of_intersection_of_a_line_and_a_plane_in_three_space done" << endl;
	}
	return a;
}

long int projective_space::line_of_intersection_of_two_planes_in_three_space(
	long int plane1, long int plane2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis1[3 * 4];
	int Basis2[3 * 4];
	int rk, a;
	int M[16];

	if (f_v) {
		cout << "projective_space::line_of_intersection_of_two_planes_in_three_space" << endl;
	}
	if (n != 3) {
		cout << "projective_space::line_of_intersection_of_"
				"two_planes_in_three_space n != 3" << endl;
		exit(1);
	}
	unrank_plane(Basis1, plane1);
	unrank_plane(Basis2, plane2);
	F->Linear_algebra->intersect_subspaces(4, 3, Basis1, 3, Basis2,
		rk, M, 0 /* verbose_level */);
	if (rk != 2) {
		cout << "projective_space::line_of_intersection_of_two_planes_in_three_space intersection is not a line" << endl;
	}
	a = rank_line(M);
	if (f_v) {
		cout << "projective_space::line_of_intersection_of_two_planes_in_three_space done" << endl;
	}
	return a;
}

long int projective_space::line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(
	long int plane1, long int plane2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Plane1[4];
	int Plane2[4];
	int Basis[16];
	long int rk;

	if (f_v) {
		cout << "projective_space::line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates" << endl;
	}
	if (n != 3) {
		cout << "projective_space::line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates "
				"n != 3" << endl;
		exit(1);
	}

	unrank_point(Plane1, plane1);
	unrank_point(Plane2, plane2);

	Int_vec_copy(Plane1, Basis, 4);
	Int_vec_copy(Plane2, Basis + 4, 4);
	F->Linear_algebra->RREF_and_kernel(4, 2, Basis, 0 /* verbose_level */);
	rk = Grass_lines->rank_lint_here(Basis + 8, 0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space::line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates done" << endl;
	}
	return rk;
}

long int projective_space::transversal_to_two_skew_lines_through_a_point(
	long int line1, long int line2, int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis1[4 * 4];
	int Basis2[4 * 4];
	int Basis3[4 * 4];
	long int a;

	if (f_v) {
		cout << "projective_space::transversal_to_two_skew_lines_through_a_point" << endl;
	}
	if (n != 3) {
		cout << "projective_space::transversal_to_two_skew_lines_through_a_point "
				"n != 3" << endl;
		exit(1);
	}
	unrank_line(Basis1, line1);
	unrank_point(Basis1 + 8, pt);
	unrank_line(Basis2, line2);
	unrank_point(Basis2 + 8, pt);
	F->Linear_algebra->RREF_and_kernel(4, 3, Basis1, 0 /* verbose_level */);
	F->Linear_algebra->RREF_and_kernel(4, 3, Basis2, 0 /* verbose_level */);
	Int_vec_copy(Basis1 + 12, Basis3, 4);
	Int_vec_copy(Basis2 + 12, Basis3 + 4, 4);
	F->Linear_algebra->RREF_and_kernel(4, 2, Basis3, 0 /* verbose_level */);
	a = rank_line(Basis3 + 8);
	if (f_v) {
		cout << "projective_space::transversal_to_two_skew_lines_through_a_point "
				"done" << endl;
	}
	return a;
}



void projective_space::plane_intersection_matrix_in_three_space(
	long int *Planes, int nb_planes, int *&Intersection_matrix,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, rk;

	if (f_v) {
		cout << "projective_space::plane_intersection_matrix_in_three_space" << endl;
	}
	Intersection_matrix = NEW_int(nb_planes * nb_planes);
	for (i = 0; i < nb_planes; i++) {
		a = Planes[i];
		for (j = i + 1; j < nb_planes; j++) {
			b = Planes[j];
			Intersection_matrix[i * nb_planes + j] = -1;
			rk = line_of_intersection_of_two_planes_in_three_space(
					a, b, 0 /* verbose_level */);
			Intersection_matrix[i * nb_planes + j] = rk;
			Intersection_matrix[j * nb_planes + i] = rk;
		}
	}
	for (i = 0; i < nb_planes; i++) {
		Intersection_matrix[i * nb_planes + i] = -1;
	}

	if (f_v) {
		cout << "projective_space::plane_intersection_matrix_in_three_space done" << endl;
	}
}

long int projective_space::line_rank_using_dual_coordinates_in_plane(
	int *eqn3, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis[3 * 3];
	int rk;
	long int line_rk;

	if (f_v) {
		cout << "projective_space::line_rank_using_dual_coordinates_in_plane" << endl;
	}
	Int_vec_copy(eqn3, Basis, 3);
	rk = F->Linear_algebra->RREF_and_kernel(3, 1, Basis, 0 /* verbose_level*/);
	if (rk != 1) {
		cout << "projective_space::line_rank_using_dual_coordinates_in_plane rk != 1" << endl;
		exit(1);
	}
	line_rk = rank_line(Basis + 1 * 3);
	if (f_v) {
		cout << "projective_space::line_rank_using_dual_coordinates_in_plane" << endl;
	}
	return line_rk;
}

long int projective_space::dual_rank_of_line_in_plane(
	long int line_rank, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis[3 * 3];
	int rk;
	long int dual_rk;

	if (f_v) {
		cout << "projective_space::dual_rank_of_line_in_plane" << endl;
	}
	unrank_line(Basis, line_rank);
	rk = F->Linear_algebra->RREF_and_kernel(3, 2, Basis, 0 /* verbose_level*/);
	if (rk != 2) {
		cout << "projective_space::dual_rank_of_line_in_plane rk != 2" << endl;
		exit(1);
	}
	dual_rk = rank_point(Basis + 2 * 3);
	if (f_v) {
		cout << "projective_space::dual_rank_of_line_in_plane done" << endl;
	}
	return dual_rk;
}



long int projective_space::plane_rank_using_dual_coordinates_in_three_space(
	int *eqn4, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis[4 * 4];
	int rk;
	long int plane_rk;

	if (f_v) {
		cout << "projective_space::plane_rank_using_dual_coordinates_in_three_space" << endl;
	}
	Int_vec_copy(eqn4, Basis, 4);
	rk = F->Linear_algebra->RREF_and_kernel(4, 1, Basis, 0 /* verbose_level*/);
	if (rk != 1) {
		cout << "projective_space::plane_rank_using_dual_coordinates_in_three_space rk != 1" << endl;
		exit(1);
	}
	plane_rk = rank_plane(Basis + 1 * 4);
	if (f_v) {
		cout << "projective_space::plane_rank_using_dual_coordinates_in_three_space" << endl;
	}
	return plane_rk;
}

long int projective_space::dual_rank_of_plane_in_three_space(
	long int plane_rank, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis[4 * 4];
	int rk;
	long int dual_rk;

	if (f_v) {
		cout << "projective_space::dual_rank_of_plane_in_three_space" << endl;
	}
	unrank_plane(Basis, plane_rank);
	rk = F->Linear_algebra->RREF_and_kernel(4, 3, Basis, 0 /* verbose_level*/);
	if (rk != 3) {
		cout << "projective_space::dual_rank_of_plane_"
				"in_three_space rk != 3" << endl;
		exit(1);
	}
	dual_rk = rank_point(Basis + 3 * 4);
	if (f_v) {
		cout << "projective_space::dual_rank_of_plane_in_three_space done" << endl;
	}
	return dual_rk;
}

void projective_space::plane_equation_from_three_lines_in_three_space(
	long int *three_lines, int *plane_eqn4, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis[6 * 4];
	int rk;

	if (f_v) {
		cout << "projective_space::plane_equation_from_three_lines_in_three_space" << endl;
	}
	unrank_lines(Basis, three_lines, 3);
	rk = F->Linear_algebra->RREF_and_kernel(4, 6, Basis, 0 /* verbose_level*/);
	if (rk != 3) {
		cout << "projective_space::plane_equation_from_three_lines_in_three_space rk != 3" << endl;
		exit(1);
	}
	Int_vec_copy(Basis + 3 * 4, plane_eqn4, 4);

	if (f_v) {
		cout << "projective_space::plane_equation_from_three_lines_in_three_space done" << endl;
	}
}

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
		int f_labeled = TRUE;
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

		if (FALSE) {
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



void projective_space::planes_through_a_line(
	long int line_rk, std::vector<long int> &plane_ranks,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int rk;
	int h, d, j, r;
	int *M1;
	int *M2;
	int *base_cols;
	int *embedding;
	int *w;
	int *v;
	int N;
	geometry_global Gg;

	if (f_v) {
		cout << "projective_space::planes_through_a_line" << endl;
	}
	d = n + 1;
	M1 = NEW_int(3 * d);
	M2 = NEW_int(3 * d);
	base_cols = NEW_int(d);
	embedding = NEW_int(d);
	w = NEW_int(d);
	v = NEW_int(d);
	Grass_lines->unrank_lint_here(M1, line_rk, 0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space::planes_through_a_line M1=" << endl;
		Int_matrix_print(M1, 2, d);
	}

	r = F->Linear_algebra->base_cols_and_embedding(2, d, M1,
			base_cols, embedding, 0 /* verbose_level */);
	if (r != 2) {
		cout << "projective_space::planes_through_a_line r != 2" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "projective_space::planes_through_a_line after RREF, M1=" << endl;
		Int_matrix_print(M1, 2, d);
	}
	N = Gg.nb_PG_elements(n - 2, F->q);

	for (h = 0; h < N; h++) {

		F->PG_element_unrank_modified(w, 1, d - 2, h);
		Int_vec_zero(v, d);
		for (j = 0; j < d - 2; j++) {
			v[embedding[j]] = w[j];
		}
		Int_vec_copy(M1, M2, 2 * d);
		Int_vec_copy(v, M2 + 2 * d, d);
		if (f_v) {
			cout << "projective_space::planes_through_a_line h = " << h << ", M2=" << endl;
			Int_matrix_print(M2, 3, d);
		}
		if (F->Linear_algebra->rank_of_rectangular_matrix(M2, 3, d, 0 /*verbose_level*/) == 3) {

			// here, rank means the rank in the sense of linear algebra

			if (f_v) {
				cout << "projective_space::planes_through_a_line h = " << h << ", M2=" << endl;
				Int_matrix_print(M2, 3, d);
			}
			rk = Grass_planes->rank_lint_here(M2, 0 /* verbose_level */);

			// here rank is in the sense of indexing

			if (f_v) {
				cout << "projective_space::planes_through_a_line h = " << h << " rk=" << rk << endl;
			}
			plane_ranks.push_back(rk);
		}
	} // next h
	FREE_int(M1);
	FREE_int(M2);
	FREE_int(base_cols);
	FREE_int(embedding);
	FREE_int(w);
	FREE_int(v);
	if (f_v) {
		cout << "projective_space::planes_through_a_line done" << endl;
	}
}


void projective_space::do_move_two_lines_in_hyperplane_stabilizer(
		long int line1_from, long int line2_from,
		long int line1_to, long int line2_to, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::do_move_two_lines_in_hyperplane_stabilizer" << endl;
	}

	if (n != 3) {
		cout << "projective_space::do_move_two_lines_in_hyperplane_stabilizer n != 3" << endl;
		exit(1);
	}
	geometry_global Gg;
	int A4[16];


	Gg.hyperplane_lifting_with_two_lines_moved(this,
			line1_from, line1_to,
			line2_from, line2_to,
			A4,
			verbose_level);

	cout << "projective_space::do_move_two_lines_in_hyperplane_stabilizer A4=" << endl;
	Int_matrix_print(A4, 4, 4);

	if (f_v) {
		cout << "projective_space::do_move_two_lines_in_hyperplane_stabilizer done" << endl;
	}
}

void projective_space::do_move_two_lines_in_hyperplane_stabilizer_text(
		std::string line1_from_text, std::string line2_from_text,
		std::string line1_to_text, std::string line2_to_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::do_move_two_lines_in_hyperplane_stabilizer_text" << endl;
	}
	if (n != 3) {
		cout << "projective_space::do_move_two_lines_in_hyperplane_stabilizer n != 3" << endl;
		exit(1);
	}

	geometry_global Gg;
	int A4[16];


	int *line1_from_data;
	int *line2_from_data;
	int *line1_to_data;
	int *line2_to_data;
	int sz;

	Int_vec_scan(line1_from_text.c_str(), line1_from_data, sz);
	if (sz != 8) {
		cout << "line1_from_text must contain exactly 8 integers" << endl;
		exit(1);
	}
	Int_vec_scan(line2_from_text.c_str(), line2_from_data, sz);
	if (sz != 8) {
		cout << "line2_from_text must contain exactly 8 integers" << endl;
		exit(1);
	}
	Int_vec_scan(line1_to_text.c_str(), line1_to_data, sz);
	if (sz != 8) {
		cout << "line1_to_text must contain exactly 8 integers" << endl;
		exit(1);
	}
	Int_vec_scan(line2_to_text.c_str(), line2_to_data, sz);
	if (sz != 8) {
		cout << "line2_to_text must contain exactly 8 integers" << endl;
		exit(1);
	}

	long int line1_from;
	long int line2_from;
	long int line1_to;
	long int line2_to;

	line1_from = rank_line(line1_from_data);
	line2_from = rank_line(line2_from_data);
	line1_to = rank_line(line1_to_data);
	line2_to = rank_line(line2_to_data);


	Gg.hyperplane_lifting_with_two_lines_moved(this,
			line1_from, line1_to,
			line2_from, line2_to,
			A4,
			verbose_level);

	cout << "projective_space::do_move_two_lines_in_hyperplane_stabilizer_text A4=" << endl;
	Int_matrix_print(A4, 4, 4);

	if (f_v) {
		cout << "projective_space::do_move_two_lines_in_hyperplane_stabilizer_text done" << endl;
	}
}



}}}


