/*
 * projective_space_of_dimension_three.cpp
 *
 *  Created on: Jan 21, 2023
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {


projective_space_of_dimension_three::projective_space_of_dimension_three()
{
	Projective_space = NULL;

	Three_skew_subspaces = NULL;

}

projective_space_of_dimension_three::~projective_space_of_dimension_three()
{
	if (Three_skew_subspaces) {
		FREE_OBJECT(Three_skew_subspaces);
	}
}

void projective_space_of_dimension_three::init(
		projective_space *Projective_space, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_of_dimension_three::init" << endl;
	}
	if (Projective_space->Subspaces->n != 3) {
		cout << "projective_space_of_dimension_three::init "
				"need dimension three" << endl;
		exit(1);
	}
	projective_space_of_dimension_three::Projective_space = Projective_space;


	Three_skew_subspaces = NEW_OBJECT(three_skew_subspaces);

	if (f_v) {
		cout << "projective_space_of_dimension_three::init "
				"before Three_skew_subspaces->init" << endl;
	}
	Three_skew_subspaces->init(
			Projective_space->Subspaces->Grass_lines,
			Projective_space->Subspaces->F,
			2 /*k*/, 4 /* n */,
			verbose_level - 1);
	if (f_v) {
		cout << "projective_space_of_dimension_three::init "
				"before Three_skew_subspaces->init" << endl;
	}


	if (f_v) {
		cout << "projective_space_of_dimension_three::init done" << endl;
	}
}


void projective_space_of_dimension_three::determine_quadric_in_solid(
	long int *nine_pts_or_more,
	int nb_pts, int *ten_coeffs, int verbose_level)
// requires n = 3
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *coords; // [nb_pts * 4]
	int *system; // [nb_pts * 10]
	int kernel[10 * 10];
	int base_cols[10];
	int i, x, y, z, w, rk;
	int kernel_m, kernel_n;

	if (f_v) {
		cout << "projective_space_of_dimension_three::determine_quadric_in_solid" << endl;
	}
	if (Projective_space->Subspaces->n != 3) {
		cout << "projective_space_of_dimension_three::determine_quadric_in_solid "
				"n != 3" << endl;
		exit(1);
	}
	if (nb_pts < 9) {
		cout << "projective_space_of_dimension_three::determine_quadric_in_solid "
				"you need to give at least 9 points" << endl;
		exit(1);
	}
	coords = NEW_int(nb_pts * 4);
	system = NEW_int(nb_pts * 10);
	for (i = 0; i < nb_pts; i++) {
		Projective_space->unrank_point(coords + i * 4, nine_pts_or_more[i]);
	}
	if (f_vv) {
		cout << "projective_space_of_dimension_three::determine_quadric_in_solid "
				"points:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				coords, nb_pts, 4, 4, Projective_space->Subspaces->F->log10_of_q);
	}

	field_theory::finite_field *F;

	F = Projective_space->Subspaces->F;
	for (i = 0; i < nb_pts; i++) {
		x = coords[i * 4 + 0];
		y = coords[i * 4 + 1];
		z = coords[i * 4 + 2];
		w = coords[i * 4 + 3];
		system[i * 10 + 0] = F->mult(x, x);
		system[i * 10 + 1] = F->mult(y, y);
		system[i * 10 + 2] = F->mult(z, z);
		system[i * 10 + 3] = F->mult(w, w);
		system[i * 10 + 4] = F->mult(x, y);
		system[i * 10 + 5] = F->mult(x, z);
		system[i * 10 + 6] = F->mult(x, w);
		system[i * 10 + 7] = F->mult(y, z);
		system[i * 10 + 8] = F->mult(y, w);
		system[i * 10 + 9] = F->mult(z, w);
	}
	if (f_v) {
		cout << "projective_space_of_dimension_three::determine_quadric_in_solid "
				"system:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				system, nb_pts, 10, 10, Projective_space->Subspaces->F->log10_of_q);
	}



	rk = Projective_space->Subspaces->F->Linear_algebra->Gauss_simple(system,
			nb_pts, 10, base_cols, verbose_level - 2);
	if (rk != 9) {
		cout << "projective_space_of_dimension_three::determine_quadric_in_solid "
				"system underdetermined" << endl;
		cout << "rk=" << rk << endl;
		exit(1);
	}
	Projective_space->Subspaces->F->Linear_algebra->matrix_get_kernel(system, 9, 10, base_cols, rk,
		kernel_m, kernel_n, kernel, 0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space_of_dimension_three::determine_quadric_in_solid "
				"conic:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				kernel, 1, 10, 10, Projective_space->Subspaces->F->log10_of_q);
	}
	for (i = 0; i < 10; i++) {
		ten_coeffs[i] = kernel[i];
	}
	if (f_v) {
		cout << "projective_space_of_dimension_three::determine_quadric_in_solid done" << endl;
	}
}


void projective_space_of_dimension_three::quadric_points_brute_force(
	int *ten_coeffs,
	long int *points, int &nb_points, int verbose_level)
// requires n = 3
// quadric in PG(3,q)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[3];
	long int i, a;

	if (f_v) {
		cout << "projective_space_of_dimension_three::quadric_points_brute_force" << endl;
	}
	nb_points = 0;
	for (i = 0; i < Projective_space->Subspaces->N_points; i++) {
		Projective_space->unrank_point(v, i);
		a = Projective_space->Subspaces->F->Linear_algebra->evaluate_quadric_form_in_PG_three(ten_coeffs, v);
		if (f_vv) {
			cout << "point " << i << " = ";
			Int_vec_print(cout, v, 3);
			cout << " gives a value of " << a << endl;
		}
		if (a == 0) {
			if (f_vv) {
				cout << "point " << i << " = ";
				Int_vec_print(cout, v, 4);
				cout << " lies on the quadric" << endl;
			}
			points[nb_points++] = i;
		}
	}
	if (f_v) {
		cout << "projective_space_of_dimension_three::quadric_points_brute_force done, "
				"we found " << nb_points << " points" << endl;
	}
	if (f_vv) {
		cout << "They are : ";
		Lint_vec_print(cout, points, nb_points);
		cout << endl;
	}
	if (f_v) {
		cout << "projective_space_of_dimension_three::quadric_points_brute_force done" << endl;
	}
}

int projective_space_of_dimension_three::point_of_intersection_of_a_line_and_a_line_in_three_space(
	long int line1, long int line2, int verbose_level)
// requires n = 3
{
	int f_v = (verbose_level >= 1);
	int Basis1[4 * 4];
	int Basis2[4 * 4];
	int rk, a;
	int M[16];

	if (f_v) {
		cout << "projective_space_of_dimension_three::point_of_intersection_of_a_line_and_a_line_in_three_space" << endl;
	}
	if (Projective_space->Subspaces->n != 3) {
		cout << "projective_space_of_dimension_three::point_of_intersection_of_a_line_and_a_line_in_three_space n != 3" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "line1=" << line1 << " line2=" << line2 << endl;
	}
	Projective_space->unrank_line(Basis1, line1);
	if (f_v) {
		cout << "line1:" << endl;
		Int_matrix_print(Basis1, 2, 4);
	}
	Projective_space->unrank_line(Basis2, line2);
	if (f_v) {
		cout << "line2:" << endl;
		Int_matrix_print(Basis2, 2, 4);
	}
	Projective_space->Subspaces->F->Linear_algebra->intersect_subspaces(
			4, 2, Basis1, 2, Basis2,
		rk, M, 0 /* verbose_level */);
	if (rk != 1) {
		cout << "projective_space_of_dimension_three::point_of_intersection_of_a_line_and_a_line_in_three_space intersection "
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
	a = Projective_space->rank_point(M);
	if (f_v) {
		cout << "point rank = " << a << endl;
	}
	if (f_v) {
		cout << "projective_space_of_dimension_three::point_of_intersection_of_a_line_and_a_line_in_three_space done" << endl;
	}
	return a;
}

int projective_space_of_dimension_three::point_of_intersection_of_a_line_and_a_plane_in_three_space(
	long int line, int plane, int verbose_level)
// requires n = 3
{
	int f_v = (verbose_level >= 1);
	int Basis1[4 * 4];
	int Basis2[4 * 4];
	int rk, a;
	int M[16];

	if (f_v) {
		cout << "projective_space_of_dimension_three::point_of_intersection_of_a_line_and_a_plane_in_three_space" << endl;
	}
	if (Projective_space->Subspaces->n != 3) {
		cout << "projective_space_of_dimension_three::point_of_intersection_of_a_line_and_a_plane_in_three_space n != 3" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "line=" << line << " plane=" << plane << endl;
	}
	Projective_space->unrank_line(Basis1, line);
	if (f_v) {
		cout << "line:" << endl;
		Int_matrix_print(Basis1, 2, 4);
	}
	Projective_space->unrank_plane(Basis2, plane);
	if (f_v) {
		cout << "plane:" << endl;
		Int_matrix_print(Basis2, 3, 4);
	}
	Projective_space->Subspaces->F->Linear_algebra->intersect_subspaces(
			4, 2, Basis1, 3, Basis2,
		rk, M, 0 /* verbose_level */);
	if (rk != 1) {
		cout << "projective_space_of_dimension_three::point_of_intersection_of_a_line_and_a_plane_in_three_space intersection "
				"is not a point" << endl;
	}
	if (f_v) {
		cout << "intersection:" << endl;
		Int_matrix_print(M, 1, 4);
	}
	a = Projective_space->rank_point(M);
	if (f_v) {
		cout << "point rank = " << a << endl;
	}
	if (f_v) {
		cout << "projective_space_of_dimension_three::point_of_intersection_of_a_line_and_a_plane_in_three_space done" << endl;
	}
	return a;
}

long int projective_space_of_dimension_three::line_of_intersection_of_two_planes_in_three_space(
	long int plane1, long int plane2, int verbose_level)
// requires n = 3
{
	int f_v = (verbose_level >= 1);
	int Basis1[3 * 4];
	int Basis2[3 * 4];
	int rk, a;
	int M[16];

	if (f_v) {
		cout << "projective_space_of_dimension_three::line_of_intersection_of_two_planes_in_three_space" << endl;
	}
	if (Projective_space->Subspaces->n != 3) {
		cout << "projective_space_of_dimension_three::line_of_intersection_of_"
				"two_planes_in_three_space n != 3" << endl;
		exit(1);
	}
	Projective_space->unrank_plane(Basis1, plane1);
	Projective_space->unrank_plane(Basis2, plane2);
	Projective_space->Subspaces->F->Linear_algebra->intersect_subspaces(
			4, 3, Basis1, 3, Basis2,
		rk, M, 0 /* verbose_level */);
	if (rk != 2) {
		cout << "projective_space_of_dimension_three::line_of_intersection_of_two_planes_in_three_space intersection is not a line" << endl;
	}
	a = Projective_space->rank_line(M);
	if (f_v) {
		cout << "projective_space_of_dimension_three::line_of_intersection_of_two_planes_in_three_space done" << endl;
	}
	return a;
}

long int projective_space_of_dimension_three::line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(
	long int plane1, long int plane2, int verbose_level)
// requires n = 3
{
	int f_v = (verbose_level >= 1);
	int Plane1[4];
	int Plane2[4];
	int Basis[16];
	long int rk;

	if (f_v) {
		cout << "projective_space_of_dimension_three::line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates" << endl;
	}
	if (Projective_space->Subspaces->n != 3) {
		cout << "projective_space_of_dimension_three::line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates "
				"n != 3" << endl;
		exit(1);
	}

	Projective_space->unrank_point(Plane1, plane1);
	Projective_space->unrank_point(Plane2, plane2);

	Int_vec_copy(Plane1, Basis, 4);
	Int_vec_copy(Plane2, Basis + 4, 4);
	Projective_space->Subspaces->F->Linear_algebra->RREF_and_kernel(
			4, 2, Basis, 0 /* verbose_level */);
	rk = Projective_space->Subspaces->Grass_lines->rank_lint_here(
			Basis + 8, 0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space_of_dimension_three::line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates done" << endl;
	}
	return rk;
}

long int projective_space_of_dimension_three::transversal_to_two_skew_lines_through_a_point(
	long int line1, long int line2, int pt, int verbose_level)
// requires n = 3
{
	int f_v = (verbose_level >= 1);
	int Basis1[4 * 4];
	int Basis2[4 * 4];
	int Basis3[4 * 4];
	long int a;

	if (f_v) {
		cout << "projective_space_of_dimension_three::transversal_to_two_skew_lines_through_a_point" << endl;
	}
	if (Projective_space->Subspaces->n != 3) {
		cout << "projective_space_of_dimension_three::transversal_to_two_skew_lines_through_a_point "
				"n != 3" << endl;
		exit(1);
	}
	Projective_space->unrank_line(Basis1, line1);
	Projective_space->unrank_point(Basis1 + 8, pt);
	Projective_space->unrank_line(Basis2, line2);
	Projective_space->unrank_point(Basis2 + 8, pt);
	Projective_space->Subspaces->F->Linear_algebra->RREF_and_kernel(
			4, 3, Basis1, 0 /* verbose_level */);
	Projective_space->Subspaces->F->Linear_algebra->RREF_and_kernel(
			4, 3, Basis2, 0 /* verbose_level */);
	Int_vec_copy(Basis1 + 12, Basis3, 4);
	Int_vec_copy(Basis2 + 12, Basis3 + 4, 4);
	Projective_space->Subspaces->F->Linear_algebra->RREF_and_kernel(
			4, 2, Basis3, 0 /* verbose_level */);
	a = Projective_space->rank_line(Basis3 + 8);
	if (f_v) {
		cout << "projective_space_of_dimension_three::transversal_to_two_skew_lines_through_a_point "
				"done" << endl;
	}
	return a;
}

void projective_space_of_dimension_three::plane_intersection_matrix_in_three_space(
	long int *Planes, int nb_planes, int *&Intersection_matrix,
	int verbose_level)
// requires n = 3
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, rk;

	if (f_v) {
		cout << "projective_space_of_dimension_three::plane_intersection_matrix_in_three_space" << endl;
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
		cout << "projective_space_of_dimension_three::plane_intersection_matrix_in_three_space done" << endl;
	}
}





long int projective_space_of_dimension_three::plane_rank_using_dual_coordinates_in_three_space(
	int *eqn4, int verbose_level)
// requires n = 3
{
	int f_v = (verbose_level >= 1);
	int Basis[4 * 4];
	int rk;
	long int plane_rk;

	if (f_v) {
		cout << "projective_space_of_dimension_three::plane_rank_using_dual_coordinates_in_three_space" << endl;
	}
	Int_vec_copy(eqn4, Basis, 4);
	rk = Projective_space->Subspaces->F->Linear_algebra->RREF_and_kernel(
			4, 1, Basis, 0 /* verbose_level*/);
	if (rk != 1) {
		cout << "projective_space_of_dimension_three::plane_rank_using_dual_coordinates_in_three_space "
				"rk != 1" << endl;
		exit(1);
	}
	plane_rk = Projective_space->rank_plane(Basis + 1 * 4);
	if (f_v) {
		cout << "projective_space_of_dimension_three::plane_rank_using_dual_coordinates_in_three_space" << endl;
	}
	return plane_rk;
}

long int projective_space_of_dimension_three::dual_rank_of_plane_in_three_space(
	long int plane_rank, int verbose_level)
// requires n = 3
{
	int f_v = (verbose_level >= 1);
	int Basis[4 * 4];
	int rk;
	long int dual_rk;

	if (f_v) {
		cout << "projective_space_of_dimension_three::dual_rank_of_plane_in_three_space" << endl;
	}
	Projective_space->unrank_plane(Basis, plane_rank);
	rk = Projective_space->Subspaces->F->Linear_algebra->RREF_and_kernel(
			4, 3, Basis, 0 /* verbose_level*/);
	if (rk != 3) {
		cout << "projective_space_of_dimension_three::dual_rank_of_plane_in_three_space "
				"rk != 3" << endl;
		exit(1);
	}
	dual_rk = Projective_space->rank_point(Basis + 3 * 4);
	if (f_v) {
		cout << "projective_space_of_dimension_three::dual_rank_of_plane_in_three_space done" << endl;
	}
	return dual_rk;
}

void projective_space_of_dimension_three::plane_equation_from_three_lines_in_three_space(
	long int *three_lines, int *plane_eqn4, int verbose_level)
// requires n = 3
{
	int f_v = (verbose_level >= 1);
	int Basis[6 * 4];
	int rk;

	if (f_v) {
		cout << "projective_space_of_dimension_three::plane_equation_from_three_lines_in_three_space" << endl;
	}
	Projective_space->unrank_lines(Basis, three_lines, 3);
	rk = Projective_space->Subspaces->F->Linear_algebra->RREF_and_kernel(4, 6, Basis, 0 /* verbose_level*/);
	if (rk != 3) {
		cout << "projective_space_of_dimension_three::plane_equation_from_three_lines_in_three_space rk != 3" << endl;
		exit(1);
	}
	Int_vec_copy(Basis + 3 * 4, plane_eqn4, 4);

	if (f_v) {
		cout << "projective_space_of_dimension_three::plane_equation_from_three_lines_in_three_space done" << endl;
	}
}

long int projective_space_of_dimension_three::plane_from_three_lines(
		long int *three_lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis[6 * 4];
	long int rk;

	if (f_v) {
		cout << "projective_space_of_dimension_three::plane_from_three_lines" << endl;
	}
	Projective_space->unrank_lines(Basis, three_lines, 3);
	rk = Projective_space->Subspaces->F->Linear_algebra->Gauss_easy(Basis, 6, 4);
	if (rk != 3) {
		cout << "projective_space_of_dimension_three::plane_from_three_lines rk != 3" << endl;
		exit(1);
	}
	rk = Projective_space->rank_plane(Basis);

	if (f_v) {
		cout << "projective_space_of_dimension_three::plane_from_three_lines done" << endl;
	}
	return rk;
}

void projective_space_of_dimension_three::make_element_which_moves_a_line_in_PG3q(
		long int line_rk, int *Mtx16,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_of_dimension_three::make_element_which_moves_a_line_in_PG3q" << endl;
	}

	int M[4 * 4];
	//int N[4 * 4 + 1]; // + 1 if f_semilinear
	int base_cols[4];
	int r, c, i, j;

	//int_vec_zero(M, 16);
	Projective_space->Subspaces->Grass_lines->unrank_lint_here(
			M, line_rk, 0 /*verbose_level*/);
	r = Projective_space->Subspaces->Grass_lines->F->Linear_algebra->Gauss_simple(
			M, 2, 4, base_cols, 0 /* verbose_level */);
	Projective_space->Subspaces->Grass_lines->F->Linear_algebra->kernel_columns(
			4, r, base_cols, base_cols + r);

	for (i = r; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			if (j == base_cols[i]) {
				c = 1;
			}
			else {
				c = 0;
			}
			M[i * 4 + j] = c;
		}
	}
	Projective_space->Subspaces->Grass_lines->F->Linear_algebra->matrix_inverse(
			M, Mtx16, 4, 0 /* verbose_level */);
	//N[4 * 4] = 0;
	//A->Group_element->make_element(Elt, N, 0);

	if (f_v) {
		cout << "projective_space_of_dimension_three::make_element_which_moves_a_line_in_PG3q done" << endl;
	}
}

int projective_space_of_dimension_three::test_if_lines_are_skew(
	int line1, int line2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis1[4 * 4];
	int Basis2[4 * 4];
	int rk;
	int M[16];

	if (f_v) {
		cout << "projective_space_of_dimension_three::test_if_lines_are_skew" << endl;
	}
	if (Projective_space->Subspaces->n != 3) {
		cout << "projective_space_of_dimension_three::test_if_lines_are_skew "
				"n != 3" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "line1=" << line1 << " line2=" << line2 << endl;
	}
	Projective_space->unrank_line(Basis1, line1);
	if (f_v) {
		cout << "line1:" << endl;
		Int_matrix_print(Basis1, 2, 4);
	}
	Projective_space->unrank_line(Basis2, line2);
	if (f_v) {
		cout << "line2:" << endl;
		Int_matrix_print(Basis2, 2, 4);
	}
	Projective_space->Subspaces->F->Linear_algebra->intersect_subspaces(4, 2, Basis1, 2, Basis2,
		rk, M, 0 /* verbose_level */);

	if (f_v) {
		cout << "projective_space_of_dimension_three::test_if_lines_are_skew done" << endl;
	}

	if (rk == 0) {
		return true;
	}
	else {
		return false;
	}
}

int projective_space_of_dimension_three::five_plus_one_to_double_six(
	long int *five_lines, long int transversal_line,
	long int *double_six,
	int verbose_level)
// a similar function exists in class surface_domain
// the arguments are almost the same, except that transversal_line is missing.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_subsets;
	int subset[5];
	long int four_lines[5];
	long int P[5];
	long int rk, i, ai4image, P4, Q, a, b, h, k, line3, line4;
	long int b1, b2, b3, b4, b5;
	int size_complement;
	int Q4[4];
	int L[8];
	int v[2];
	int w[4];
	int d;

	// L0,L1,L2 are the first three lines in the regulus on the
	// hyperbolic quadric x_0x_3-x_1x_2 = 0:
	int L0[] = {0,0,1,0, 0,0,0,1};
	int L1[] = {1,0,0,0, 0,1,0,0};
	int L2[] = {1,0,1,0, 0,1,0,1};
	int ell0;

	int pi1[12];
	int pi2[12];
	int *line1;
	int *line2;
	int M[16];
	long int image[2];
	int pt_coord[4 * 4];
	int nb_pts;
	int transformation[17];
	int transformation_inv[17];
	combinatorics::combinatorics_domain Combi;
	field_theory::finite_field *F;

	if (f_v) {
		cout << "projective_space_of_dimension_three::five_plus_one_to_double_six, "
				"verbose_level = " << verbose_level << endl;
	}

	F = Projective_space->Subspaces->F;

#if 0
	if (Recoordinatize == NULL) {
		cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
				"Recoordinatize == NULL" << endl;
		exit(1);
	}
#endif

	if (f_v) {
		cout << "projective_space_of_dimension_three::five_plus_one_to_double_six" << endl;
		cout << "The five lines are ";
		Lint_vec_print(cout, five_lines, 5);
		cout << endl;
	}

	//ell0 = Projective_space->Subspaces->Grass_lines->rank_lint_here(L0, 0 /* verbose_level */);

	ell0 = Projective_space->rank_line(L0);


	Lint_vec_copy(five_lines, double_six, 5);
		// fill in a_1,\ldots,a_5

	double_six[11] = transversal_line;
		// fill in b_6

	for (i = 0; i < 5; i++) {
		if (f_vv) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
					"intersecting line " << i << " = " << five_lines[i]
				<< " with line " << transversal_line << endl;
		}
		P[i] = point_of_intersection_of_a_line_and_a_line_in_three_space(
				five_lines[i], transversal_line,
				0 /* verbose_level */);
	}
	if (f_v) {
		cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
				"The five intersection points are:";
		Lint_vec_print(cout, P, 5);
		cout << endl;
	}


	// Determine b_1,\ldots,b_5:

	// For every 4-subset \{a_1,\ldots,a_5\} \setminus \{a_i\},
	// let b_i be the unique second transversal:

	nb_subsets = Combi.int_n_choose_k(5, 4);
		// 5 choose 4 is of course 5.

	for (rk = 0; rk < nb_subsets; rk++) {

		if (f_vv) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
					"subset " << rk << " / " << nb_subsets << endl;
		}
		// Determine a subset a_{i1},a_{i2},a_{i3},a_{i4};a_{i5}
		Combi.unrank_k_subset(rk, subset, 5, 4);
		Combi.set_complement(subset, 4, subset + 4, size_complement, 5);
		for (i = 0; i < 5; i++) {
			four_lines[i] = five_lines[subset[i]];
		}

		// P4 is the intersection of a_{i4} with the transversal:
		P4 = P[subset[3]];
		if (f_vv) {
			cout << "subset " << rk << " / " << nb_subsets << " : ";
			Int_vec_print(cout, subset, 4);
			cout << " : ";
			Lint_vec_print(cout, four_lines, 5);
			cout << " P4=" << P4 << endl;
		}

		// We map a_{i1},a_{12},a_{i3} to
		// \ell_0,\ell_1,\ell_2, the first three lines in a regulus:
		// This cannot go wrong because we know
		// that the three lines are pairwise skew,
		// and hence determine a regulus.
		// This is because they are part of a
		// partial ovoid on the Klein quadric.

		if (f_vv) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
					"subset " << rk << " / " << nb_subsets
					<< " before Three_skew_subspaces->do_recoordinatize" << endl;
		}

		Three_skew_subspaces->do_recoordinatize(
				four_lines[0], four_lines[1], four_lines[2],
				transformation,
				0 /*verbose_level - 2*/);

		if (f_vv) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six transformation=" << endl;
			Int_matrix_print(transformation, 4, 4);
		}
		//Recoordinatize->do_recoordinatize(
		//		four_lines[0], four_lines[1], four_lines[2],
		//		verbose_level - 2);
		if (f_vv) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
					"subset " << rk << " / " << nb_subsets
					<< " after Three_skew_subspaces->do_recoordinatize" << endl;
		}

		F->Linear_algebra->invert_matrix(
				transformation, transformation_inv, 4 /*n*/,
				0 /* verbose_level*/);

		if (f_vv) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six transformation_inv=" << endl;
			Int_matrix_print(transformation_inv, 4, 4);
		}
		if (f_vv) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six transformation=" << endl;
			Int_matrix_print(transformation, 4, 4);
		}


		transformation[16] = 0;
		transformation_inv[16] = 0;

		//A->Group_element->element_invert(
		//		Recoordinatize->Elt, Elt1, 0);

		if (f_vv) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six mapping a4=four_lines[3]=" << four_lines[3] << endl;
		}

		ai4image = Projective_space->Subspaces->Grass_lines->map_line_in_PG3q(
				four_lines[3], transformation,
				0 /* verbose_level */);

		if (f_vv) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six ai4image=" << ai4image << endl;
		}

		//ai4image = A2->Group_element->element_image_of(
		//		four_lines[3],
		//		Recoordinatize->Elt,
		//		0 /* verbose_level */);


		Q = map_point(
				P4, transformation,
				0 /*verbose_level - 2*/);

		//Q = A->Group_element->element_image_of(P4,
		//		Recoordinatize->Elt,
		//		0 /* verbose_level */);

		if (f_vv) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six ai4image = " << ai4image << " Q=" << Q << endl;
		}
		//Surf->unrank_point(Q4, Q);
		Projective_space->unrank_point(Q4, Q);

		b = F->Linear_algebra->evaluate_quadratic_form_x0x3mx1x2(Q4);
		if (b) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six error: The point Q does not "
					"lie on the quadric" << endl;
			exit(1);
		}


		//Projective_space->Subspaces->Grass_lines->unrank_lint_here(
		//		L, ai4image, 0 /* verbose_level */);

		Projective_space->unrank_line(L, ai4image);


		if (f_vv) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six before F->adjust_basis" << endl;
			cout << "L=" << endl;
			Int_matrix_print(L, 2, 4);
			cout << "Q4=" << endl;
			Int_matrix_print(Q4, 1, 4);
		}

		// Adjust the basis L of the line ai4image so that Q4 is first:
		F->Linear_algebra->adjust_basis(
				L, Q4, 4, 2, 1, verbose_level - 1);
		if (f_vv) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six after F->adjust_basis" << endl;
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six L=" << endl;
			Int_matrix_print(L, 2, 4);
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six Q4=" << endl;
			Int_matrix_print(Q4, 1, 4);
		}

		// Determine the point w which is the second point where
		// the line which is the image of a_{i4} intersects the hyperboloid:
		// To do so, we loop over all points on the line distinct from Q4:

		if (f_vv) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
					"subset " << rk << " / " << nb_subsets
					<< " before loop" << endl;
		}

		for (a = 0; a < F->q; a++) {
			v[0] = a;
			v[1] = 1;
			F->Linear_algebra->mult_matrix_matrix(
					v, L, w, 1, 2, 4,
					0 /* verbose_level */);
			//rk = Surf->rank_point(w);

			// Evaluate the equation of the hyperboloid
			// which is x_0x_3-x_1x_2 = 0,
			// to see if w lies on it:
			b = F->Linear_algebra->evaluate_quadratic_form_x0x3mx1x2(w);
			if (f_vv) {
				cout << "projective_space_of_dimension_three::five_plus_one_to_double_six a=" << a << " v=";
				Int_vec_print(cout, v, 2);
				cout << " w=";
				Int_vec_print(cout, w, 4);
				cout << " b=" << b << endl;
			}
			if (b == 0) {
				break;
			}
		}

		if (f_vv) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
					"subset " << rk << " / " << nb_subsets
					<< " after loop" << endl;
		}

		if (a == F->q) {
			if (f_v) {
				cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
						"we could not find a second intersection point"
						<< endl;
			}
			return false;
		}



		// test that the line is not a line of the quadric:
		F->Linear_algebra->add_vector(
				L, w, pt_coord, 4);
		b = F->Linear_algebra->evaluate_quadratic_form_x0x3mx1x2(
				pt_coord);
		if (b == 0) {
			if (f_v) {
				cout << "projective_space_of_dimension_three::five_plus_one_to_double_six The line lies on the quadric, "
						"this five plus one is not good." << endl;
			}
			return false;
		}

		// Pick two lines out of the three lines ell_0,ell_1,ell_2
		// which do not contain the point w:

		// test if w lies on ell_0 or ell_1 or ell2:
		if (w[0] == 0 && w[1] == 0) {
			// now w lies on ell_0 so we take ell_1 and ell_2:
			line1 = L1;
			line1 = L2;
		}
		else if (w[2] == 0 && w[3] == 0) {
			// now w lies on ell_1 so we take ell_0 and ell_2:
			line1 = L0;
			line1 = L2;
		}
		else if (w[0] == w[2] && w[1] == w[3]) {
			// now w lies on ell_2 so we take ell_0 and ell_1:
			line1 = L0;
			line2 = L1;
		}
		else {
			// Now, w does not lie on ell_0,ell_1,ell_2:
			line1 = L0;
			line2 = L1;
		}

		// Let pi1 be the plane spanned by line1 and w:
		Int_vec_copy(line1, pi1, 8);
		Int_vec_copy(w, pi1 + 8, 4);

		// Let pi2 be the plane spanned by line2 and w:
		Int_vec_copy(line2, pi2, 8);
		Int_vec_copy(w, pi2 + 8, 4);

		// Let line3 be the intersection of pi1 and pi2:
		if (f_v) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
					"subset " << rk << " / " << nb_subsets
					<< " before intersect_subspaces" << endl;
		}
		F->Linear_algebra->intersect_subspaces(
				4, 3, pi1, 3, pi2,
			d, M, 0 /* verbose_level */);
		if (f_v) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
					"subset " << rk << " / " << nb_subsets
					<< " after intersect_subspaces" << endl;
		}
		if (d != 2) {
			if (f_v) {
				cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
						"intersection is not a line" << endl;
			}
			return false;
		}
		line3 = Projective_space->Subspaces->Grass_lines->rank_lint_here(
				M, 0 /* verbose_level */);


		// Map line3 back to get line4 = b_i:
		//line4 = A2->Group_element->element_image_of(
		//		line3, Elt1, 0 /* verbose_level */);
		line4 = Projective_space->Subspaces->Grass_lines->map_line_in_PG3q(
				line3, transformation_inv,
				0 /* verbose_level */);

		double_six[10 - rk] = line4; // fill in b_i
	} // next rk


	if (f_vv) {
		cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
				"b1,...,b5 have been created" << endl;
	}

	// Now, b_1,\ldots,b_5 have been determined.
	b1 = double_six[6];
	b2 = double_six[7];
	b3 = double_six[8];
	b4 = double_six[9];
	b5 = double_six[10];

	// Next, determine a_6 as the transversal of b_1,\ldots,b_5:

	if (f_vv) {
		cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
				"before do_recoordinatize" << endl;
	}
	//Recoordinatize->do_recoordinatize(
	//		b1, b2, b3, verbose_level - 2);

	Three_skew_subspaces->do_recoordinatize(
			b1, b2, b3,
			transformation,
			verbose_level - 2);

	if (f_vv) {
		cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
				"after do_recoordinatize" << endl;
	}

	F->Linear_algebra->invert_matrix(
			transformation, transformation_inv, 4 /*n*/,
			0 /* verbose_level*/);

	//A->Group_element->element_invert(
	//		Recoordinatize->Elt, Elt1, 0);

	// map b4 and b5:
	image[0] = Projective_space->Subspaces->Grass_lines->map_line_in_PG3q(
			b4, transformation,
			0 /* verbose_level */);
	image[1] = Projective_space->Subspaces->Grass_lines->map_line_in_PG3q(
			b5, transformation,
			0 /* verbose_level */);
#if 0
	image[0] = A2->Group_element->element_image_of(
			b4, Recoordinatize->Elt, 0 /* verbose_level */);
	image[1] = A2->Group_element->element_image_of(
			b5, Recoordinatize->Elt, 0 /* verbose_level */);
#endif

	nb_pts = 0;
	for (h = 0; h < 2; h++) {

		//Surf->Gr->unrank_lint_here(
		//		L, image[h], 0 /* verbose_level */);

		//Projective_space->Subspaces->Grass_lines->unrank_lint_here(
		//						L, image[h], 0 /* verbose_level */);

		Projective_space->unrank_line(L, image[h]);


		for (a = 0; a < F->q + 1; a++) {
			F->Projective_space_basic->PG_element_unrank_modified(
					v, 1, 2, a);
			F->Linear_algebra->mult_matrix_matrix(
					v, L, w, 1, 2, 4,
					0 /* verbose_level */);

			// Evaluate the equation of the hyperboloid
			// which is x_0x_3-x_1x_2 = 0,
			// to see if w lies on it:
			b = F->Linear_algebra->evaluate_quadratic_form_x0x3mx1x2(w);
			if (b == 0) {
				Int_vec_copy(w, pt_coord + nb_pts * 4, 4);
				nb_pts++;
				if (nb_pts == 5) {
					cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
							"nb_pts == 5" << endl;
					exit(1);
				}
			}
		}
		if (nb_pts != (h + 1) * 2) {
			cout << "projective_space_of_dimension_three::five_plus_one_to_double_six nb_pts != "
					"(h + 1) * 2" << endl;
			exit(1);
		}
	} // next h

	if (f_v) {
		cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
				"four points have been computed:" << endl;
		Int_matrix_print(pt_coord, 4, 4);
	}
	line3 = -1;
	for (h = 0; h < 2; h++) {
		for (k = 0; k < 2; k++) {

			F->Linear_algebra->add_vector(
					pt_coord + h * 4, pt_coord + (2 + k) * 4, w, 4);

			b = F->Linear_algebra->evaluate_quadratic_form_x0x3mx1x2(w);
			if (b == 0) {
				if (f_vv) {
					cout << "h=" << h << " k=" << k
							<< " define a singular line" << endl;
				}
				Int_vec_copy(pt_coord + h * 4, L, 4);
				Int_vec_copy(pt_coord + (2 + k) * 4, L + 4, 4);


				//line3 = Surf->rank_line(L);

				line3 = Projective_space->Subspaces->Grass_lines->rank_lint_here(
						L, 0 /* verbose_level */);



				if (!test_if_lines_are_skew(
						ell0,
						line3, 0 /* verbose_level */)) {
					if (f_vv) {
						cout << "The line intersects ell_0, so we are good" << endl;
					}
					break;
				}
				// continue on to find another line
			}
		}
		if (k < 2) {
			break;
		}
	}
	if (h == 2) {
		cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
				"could not determine a_6" << endl;
		exit(1);
	}
	if (line3 == -1) {
		cout << "projective_space_of_dimension_three::five_plus_one_to_double_six "
				"line3 == -1" << endl;
		exit(1);
	}
	// Map line3 back to get line4 = a_6:
	//line4 = A2->Group_element->element_image_of(
	//		line3, Elt1, 0 /* verbose_level */);

	line4 = Projective_space->Subspaces->Grass_lines->map_line_in_PG3q(
			line3, transformation_inv,
			0 /* verbose_level */);

	double_six[5] = line4; // fill in a_6

	if (f_v) {
		cout << "projective_space_of_dimension_three::five_plus_one_to_double_six done" << endl;
	}
	return true;
}

long int projective_space_of_dimension_three::map_point(
		long int point, int *transform16, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_of_dimension_three::map_point" << endl;
	}
	long int b;
	int Basis1[4];
	int Basis2[4];

	if (f_v) {
		cout << "projective_space_of_dimension_three::map_point transform16 = " << endl;
		Int_matrix_print(transform16, 4, 4);
	}

	Int_vec_zero(Basis1, 4);
	Projective_space->unrank_point(
			Basis1, point);

	if (f_v) {
		cout << "projective_space_of_dimension_three::map_point point = " << point << endl;
		cout << "projective_space_of_dimension_three::map_point Basis1 = " << endl;
		Int_matrix_print(Basis1, 2, 4);
	}


	Projective_space->Subspaces->F->Linear_algebra->mult_matrix_matrix(
			Basis1, transform16, Basis2,
			1, 4, 4, 0/*verbose_level - 4*/);

	if (f_v) {
		cout << "projective_space_of_dimension_three::map_point Basis2 = " << endl;
		Int_matrix_print(Basis2, 1, 4);
	}

	b = Projective_space->rank_point(Basis2);

	if (f_v) {
		cout << "projective_space_of_dimension_three::map_point image line = " << b << endl;
	}

	if (f_v) {
		cout << "projective_space_of_dimension_three::map_point done" << endl;
	}
	return b;
}


}}}


