/*
 * intersection_type.cpp
 *
 *  Created on: Nov 15, 2022
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace other_geometry {



intersection_type::intersection_type()
{
	set = NULL;
	set_size = 0;
	threshold = 0;

	P = NULL;
	Gr = NULL;

	R = NULL;
	Pts_on_subspace = NULL;
	nb_pts_on_subspace = NULL;
	len = 0;

	the_intersection_type = NULL;
	highest_intersection_number = 0;

	Highest_weight_objects = NULL;
	nb_highest_weight_objects = 0;

	Intersection_sets = NULL;

	M = NULL;

}



intersection_type::~intersection_type()
{
	int i;

	for (i = 0; i < len; i++) {
		FREE_lint(Pts_on_subspace[i]);
	}
	FREE_plint(Pts_on_subspace);
	FREE_int(nb_pts_on_subspace);
	FREE_OBJECTS(R);

	if (M) {
		FREE_OBJECT(M);
	}
}

void intersection_type::line_intersection_type_slow(
	long int *set, int set_size, int threshold,
	projective_geometry::projective_space *P,
	projective_geometry::grassmann *Gr,
	int verbose_level)
// Does a complete sweep over all lines
// in the given projective space.
// This is expensive.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_v3 = (verbose_level >= 3);
	long int r, rk, i, u, d, N_subspaces, l, N_subspaces_100;
	int subspace_k;

	int *Basis;
	int *Basis_save;
	int *Coords;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "intersection_type::line_intersection_type_slow" << endl;
	}

	intersection_type::set = set;
	intersection_type::set_size = set_size;
	intersection_type::threshold = threshold;
	intersection_type::P = P;
	intersection_type::Gr = Gr;


	if (f_vv) {
		P->Reporting->print_set_numerical(cout, set, set_size);
	}
	if (!Sorting.test_if_set_with_return_value_lint(set, set_size)) {
		cout << "intersection_type::line_intersection_type_slow "
				"the input set if not a set" << endl;
		exit(1);
	}
	if (Gr->k != 2) {
		cout << "intersection_type::line_intersection_type_slow "
				"we need a Grassmannian of dimension 2" << endl;
		exit(1);

	}
	subspace_k = Gr->k;
	d = P->Subspaces->n + 1;
	N_subspaces = P->Subspaces->nb_rk_k_subspaces_as_lint(3);

	if (f_v) {
		cout << "intersection_type::line_intersection_type_slow "
				"N_subspaces=" << N_subspaces << endl;
	}



	// allocate data that is returned:
	R = NEW_OBJECTS(ring_theory::longinteger_object, N_subspaces);
	Pts_on_subspace = NEW_plint(N_subspaces);
	nb_pts_on_subspace = NEW_int(N_subspaces);

	// allocate temporary data:
	Basis = NEW_int((subspace_k + 1) * d);
	Basis_save = NEW_int((subspace_k + 1) * d);
	Coords = NEW_int(set_size * d);

	if (f_v) {
		cout << "intersection_type::line_intersection_type_slow "
				"before unrank_point" << endl;
	}
	for (i = 0; i < set_size; i++) {
		if (f_v) {
			cout << "intersection_type::line_intersection_type_slow "
					"i=" << i << " set[i] = " << set[i] << endl;
		}
		P->unrank_point(Coords + i * d, set[i]);
	}
	if (f_vv) {
		cout << "intersection_type::line_intersection_type_slow "
				"Coords:" << endl;
		Int_matrix_print(Coords, set_size, d);
	}

	N_subspaces_100 = N_subspaces / 100;
	l = 0;
	for (rk = 0; rk < N_subspaces; rk++) {

		if (N_subspaces > 1000000) {
			if ((rk % N_subspaces_100) == 0) {
				cout << "intersection_type::line_intersection_type_slow "
						<< rk << " / " << N_subspaces << " = "
						<< rk / N_subspaces_100 << " %" << endl;
			}
		}

		Gr->unrank_lint_here(Basis_save, rk, 0 /* verbose_level */);
		//int_vec_copy(G->M, Basis_save, 3 * d);
		long int *pts_on_subspace;
		int nb = 0;

		pts_on_subspace = NEW_lint(set_size);

		for (u = 0; u < set_size; u++) {
			Int_vec_copy(Basis_save, Basis, subspace_k * d);
			Int_vec_copy(Coords + u * d, Basis + subspace_k * d, d);
			r = P->Subspaces->F->Linear_algebra->rank_of_rectangular_matrix(
					Basis,
					subspace_k + 1, d, 0 /* verbose_level */);
			if (r < subspace_k + 1) {
				pts_on_subspace[nb++] = u;
			}
		}

		if (nb >= threshold) {
			Pts_on_subspace[l] = pts_on_subspace;
			nb_pts_on_subspace[l] = nb;
			R[l].create(rk);
			l++;
		}
		else {
			FREE_lint(pts_on_subspace);
		}

	} // rk
	len = l;

	FREE_int(Basis);
	FREE_int(Basis_save);
	FREE_int(Coords);
	if (f_v) {
		cout << "intersection_type::line_intersection_type_slow "
				"done" << endl;
	}
}



void intersection_type::plane_intersection_type_slow(
	long int *set, int set_size, int threshold,
	projective_geometry::projective_space *P,
	projective_geometry::grassmann *Gr,
	int verbose_level)
// Does a complete sweep over all planes
// in the given projective space.
// This is expensive.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_v3 = (verbose_level >= 3);
	long int r, rk, i, u, d, N_subspaces, l, N_subspaces_100;
	int subspace_k;


	int *Basis;
	int *Basis_save;
	int *Coords;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "intersection_type::plane_intersection_type_slow" << endl;
	}

	intersection_type::set = set;
	intersection_type::set_size = set_size;
	intersection_type::threshold = threshold;
	intersection_type::P = P;
	intersection_type::Gr = Gr;


	if (f_vv) {
		P->Reporting->print_set_numerical(cout, set, set_size);
	}
	if (!Sorting.test_if_set_with_return_value_lint(set, set_size)) {
		cout << "intersection_type::plane_intersection_type_slow "
				"the input set if not a set" << endl;
		exit(1);
	}
	if (Gr->k != 3) {
		cout << "intersection_type::plane_intersection_type_slow "
				"we need a Grassmannian of dimension 3" << endl;
		exit(1);

	}
	subspace_k = 3;
	d = P->Subspaces->n + 1;
	N_subspaces = P->Subspaces->nb_rk_k_subspaces_as_lint(3);

	if (f_v) {
		cout << "intersection_type::plane_intersection_type_slow "
				"N_planes=" << N_subspaces << endl;
	}



	// allocate data that is returned:
	R = NEW_OBJECTS(ring_theory::longinteger_object, N_subspaces);
	Pts_on_subspace = NEW_plint(N_subspaces);
	nb_pts_on_subspace = NEW_int(N_subspaces);

	// allocate temporary data:
	Basis = NEW_int((subspace_k + 1) * d);
	Basis_save = NEW_int((subspace_k + 1) * d);
	Coords = NEW_int(set_size * d);

	if (f_v) {
		cout << "intersection_type::plane_intersection_type_slow "
				"before unrank_point" << endl;
	}
	for (i = 0; i < set_size; i++) {
		if (f_v) {
			cout << "intersection_type::plane_intersection_type_slow "
					"i=" << i << " set[i] = " << set[i] << endl;
		}
		P->unrank_point(Coords + i * d, set[i]);
	}
	if (f_vv) {
		cout << "intersection_type::plane_intersection_type_slow "
				"Coords:" << endl;
		Int_matrix_print(Coords, set_size, d);
	}

	N_subspaces_100 = N_subspaces / 100;
	l = 0;
	for (rk = 0; rk < N_subspaces; rk++) {

		if (N_subspaces > 1000000) {
			if ((rk % N_subspaces_100) == 0) {
				cout << "intersection_type::plane_intersection_type_slow "
						<< rk << " / " << N_subspaces << " = "
						<< rk / N_subspaces_100 << " %" << endl;
			}
		}

		Gr->unrank_lint_here(Basis_save, rk, 0 /* verbose_level */);
		//int_vec_copy(G->M, Basis_save, 3 * d);
		long int *pts_on_subspace;
		int nb = 0;

		pts_on_subspace = NEW_lint(set_size);

		for (u = 0; u < set_size; u++) {
			Int_vec_copy(Basis_save, Basis, subspace_k * d);
			Int_vec_copy(Coords + u * d, Basis + subspace_k * d, d);
			r = P->Subspaces->F->Linear_algebra->rank_of_rectangular_matrix(
					Basis,
					subspace_k + 1, d, 0 /* verbose_level */);
			if (r < subspace_k + 1) {
				pts_on_subspace[nb++] = u;
			}
		}

		if (nb >= threshold) {
			Pts_on_subspace[l] = pts_on_subspace;
			nb_pts_on_subspace[l] = nb;
			R[l].create(rk);
			l++;
		}
		else {
			FREE_lint(pts_on_subspace);
		}

	} // rk
	len = l;

	FREE_int(Basis);
	FREE_int(Basis_save);
	FREE_int(Coords);
	if (f_v) {
		cout << "intersection_type::plane_intersection_type_slow "
				"done" << endl;
	}
}

void intersection_type::compute_heighest_weight_objects(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "intersection_type::compute_heighest_weight_objects" << endl;
	}

	data_structures::tally C;
	int f_second = false;

	C.init(nb_pts_on_subspace, len, f_second, 0);
	if (f_v) {
		cout << "intersection_type::compute_heighest_weight_objects "
				"plane-intersection type: ";
		C.print(false /*f_backwards*/);
	}

	if (f_v) {
		cout << "The plane intersection type is (";
		C.print_bare(false /*f_backwards*/);
		cout << ")" << endl << endl;
	}

	int f, i, l, a;

	f = C.type_first[C.nb_types - 1];
	highest_intersection_number = C.data_sorted[f];

	the_intersection_type = NEW_int(highest_intersection_number + 1);
	Int_vec_zero(
			the_intersection_type,
			highest_intersection_number + 1);

	for (i = 0; i < C.nb_types; i++) {
		f = C.type_first[i];
		l = C.type_len[i];
		a = C.data_sorted[f];
		the_intersection_type[a] = l;
	}

	int *Pts;
	int nb_pts;
	data_structures::sorting Sorting;

	C.get_class_by_value(Pts, nb_pts, highest_intersection_number,
			verbose_level);

	Sorting.int_vec_heapsort(Pts, nb_pts);

	if (f_v) {
		cout << "intersection_type::compute_heighest_weight_objects "
				"class with highest intersection "
				"value has size " << nb_pts << endl;
		Int_vec_print(cout, Pts, nb_pts);
		cout << endl;
	}


	Highest_weight_objects = NEW_lint(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		Highest_weight_objects[i] = R[Pts[i]].as_lint();
	}
	nb_highest_weight_objects = nb_pts;

	if (f_v) {
		cout << "intersection_type::compute_heighest_weight_objects "
				"corresponding to the planes with rank: " << endl;
		Lint_vec_print(cout, Highest_weight_objects, nb_pts);
		cout << endl;
	}

	int j;

	Intersection_sets = NEW_int(
			nb_highest_weight_objects * highest_intersection_number);

	for (i = 0; i < nb_highest_weight_objects; i++) {
		a = Pts[i];
		for (j = 0; j < highest_intersection_number; j++) {
			Intersection_sets[i * highest_intersection_number + j]
							  = Pts_on_subspace[a][j];
		}
	}
	if (f_v) {
		cout << "intersection_type::compute_heighest_weight_objects "
				"Intersection_sets: " << endl;
		Int_matrix_print(
				Intersection_sets,
				nb_highest_weight_objects,
				highest_intersection_number);
	}


	M = NEW_OBJECT(data_structures::int_matrix);

	M->allocate_and_init(
			nb_highest_weight_objects,
			highest_intersection_number,
			Intersection_sets);

	M->sort_rows(verbose_level);

	if (f_v) {
		cout << "intersection_type::compute_heighest_weight_objects "
				"Intersection_sets sorted: " << endl;
		Int_matrix_print(
				M->M,
				nb_highest_weight_objects,
				highest_intersection_number);
	}


	FREE_int(Pts);


	if (f_v) {
		cout << "intersection_type::compute_heighest_weight_objects done" << endl;
	}
}

}}}}



