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
		P = NULL;
}

projective_space_of_dimension_three::~projective_space_of_dimension_three()
{
}

void projective_space_of_dimension_three::init(projective_space *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_of_dimension_three::init" << endl;
	}
	if (P->n != 3) {
		cout << "projective_space_of_dimension_three::init "
				"need dimension three" << endl;
		exit(1);
	}
	projective_space_of_dimension_three::P = P;
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
	if (P->n != 3) {
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
		P->unrank_point(coords + i * 4, nine_pts_or_more[i]);
	}
	if (f_vv) {
		cout << "projective_space_of_dimension_three::determine_quadric_in_solid "
				"points:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				coords, nb_pts, 4, 4, P->F->log10_of_q);
	}
	for (i = 0; i < nb_pts; i++) {
		x = coords[i * 4 + 0];
		y = coords[i * 4 + 1];
		z = coords[i * 4 + 2];
		w = coords[i * 4 + 3];
		system[i * 10 + 0] = P->F->mult(x, x);
		system[i * 10 + 1] = P->F->mult(y, y);
		system[i * 10 + 2] = P->F->mult(z, z);
		system[i * 10 + 3] = P->F->mult(w, w);
		system[i * 10 + 4] = P->F->mult(x, y);
		system[i * 10 + 5] = P->F->mult(x, z);
		system[i * 10 + 6] = P->F->mult(x, w);
		system[i * 10 + 7] = P->F->mult(y, z);
		system[i * 10 + 8] = P->F->mult(y, w);
		system[i * 10 + 9] = P->F->mult(z, w);
	}
	if (f_v) {
		cout << "projective_space_of_dimension_three::determine_quadric_in_solid "
				"system:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				system, nb_pts, 10, 10, P->F->log10_of_q);
	}



	rk = P->F->Linear_algebra->Gauss_simple(system,
			nb_pts, 10, base_cols, verbose_level - 2);
	if (rk != 9) {
		cout << "projective_space_of_dimension_three::determine_quadric_in_solid "
				"system underdetermined" << endl;
		cout << "rk=" << rk << endl;
		exit(1);
	}
	P->F->Linear_algebra->matrix_get_kernel(system, 9, 10, base_cols, rk,
		kernel_m, kernel_n, kernel, 0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space_of_dimension_three::determine_quadric_in_solid "
				"conic:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				kernel, 1, 10, 10, P->F->log10_of_q);
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
	for (i = 0; i < P->N_points; i++) {
		P->unrank_point(v, i);
		a = P->F->Linear_algebra->evaluate_quadric_form_in_PG_three(ten_coeffs, v);
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
	if (P->n != 3) {
		cout << "projective_space_of_dimension_three::point_of_intersection_of_a_line_and_a_line_in_three_space n != 3" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "line1=" << line1 << " line2=" << line2 << endl;
	}
	P->unrank_line(Basis1, line1);
	if (f_v) {
		cout << "line1:" << endl;
		Int_matrix_print(Basis1, 2, 4);
	}
	P->unrank_line(Basis2, line2);
	if (f_v) {
		cout << "line2:" << endl;
		Int_matrix_print(Basis2, 2, 4);
	}
	P->F->Linear_algebra->intersect_subspaces(4, 2, Basis1, 2, Basis2,
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
	a = P->rank_point(M);
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
	if (P->n != 3) {
		cout << "projective_space_of_dimension_three::point_of_intersection_of_a_line_and_a_plane_in_three_space n != 3" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "line=" << line << " plane=" << plane << endl;
	}
	P->unrank_line(Basis1, line);
	if (f_v) {
		cout << "line:" << endl;
		Int_matrix_print(Basis1, 2, 4);
	}
	P->unrank_plane(Basis2, plane);
	if (f_v) {
		cout << "plane:" << endl;
		Int_matrix_print(Basis2, 3, 4);
	}
	P->F->Linear_algebra->intersect_subspaces(4, 2, Basis1, 3, Basis2,
		rk, M, 0 /* verbose_level */);
	if (rk != 1) {
		cout << "projective_space_of_dimension_three::point_of_intersection_of_a_line_and_a_plane_in_three_space intersection "
				"is not a point" << endl;
	}
	if (f_v) {
		cout << "intersection:" << endl;
		Int_matrix_print(M, 1, 4);
	}
	a = P->rank_point(M);
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
	if (P->n != 3) {
		cout << "projective_space_of_dimension_three::line_of_intersection_of_"
				"two_planes_in_three_space n != 3" << endl;
		exit(1);
	}
	P->unrank_plane(Basis1, plane1);
	P->unrank_plane(Basis2, plane2);
	P->F->Linear_algebra->intersect_subspaces(4, 3, Basis1, 3, Basis2,
		rk, M, 0 /* verbose_level */);
	if (rk != 2) {
		cout << "projective_space_of_dimension_three::line_of_intersection_of_two_planes_in_three_space intersection is not a line" << endl;
	}
	a = P->rank_line(M);
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
	if (P->n != 3) {
		cout << "projective_space_of_dimension_three::line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates "
				"n != 3" << endl;
		exit(1);
	}

	P->unrank_point(Plane1, plane1);
	P->unrank_point(Plane2, plane2);

	Int_vec_copy(Plane1, Basis, 4);
	Int_vec_copy(Plane2, Basis + 4, 4);
	P->F->Linear_algebra->RREF_and_kernel(4, 2, Basis, 0 /* verbose_level */);
	rk = P->Grass_lines->rank_lint_here(Basis + 8, 0 /* verbose_level */);
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
	if (P->n != 3) {
		cout << "projective_space_of_dimension_three::transversal_to_two_skew_lines_through_a_point "
				"n != 3" << endl;
		exit(1);
	}
	P->unrank_line(Basis1, line1);
	P->unrank_point(Basis1 + 8, pt);
	P->unrank_line(Basis2, line2);
	P->unrank_point(Basis2 + 8, pt);
	P->F->Linear_algebra->RREF_and_kernel(4, 3, Basis1, 0 /* verbose_level */);
	P->F->Linear_algebra->RREF_and_kernel(4, 3, Basis2, 0 /* verbose_level */);
	Int_vec_copy(Basis1 + 12, Basis3, 4);
	Int_vec_copy(Basis2 + 12, Basis3 + 4, 4);
	P->F->Linear_algebra->RREF_and_kernel(4, 2, Basis3, 0 /* verbose_level */);
	a = P->rank_line(Basis3 + 8);
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
	rk = P->F->Linear_algebra->RREF_and_kernel(4, 1, Basis, 0 /* verbose_level*/);
	if (rk != 1) {
		cout << "projective_space_of_dimension_three::plane_rank_using_dual_coordinates_in_three_space "
				"rk != 1" << endl;
		exit(1);
	}
	plane_rk = P->rank_plane(Basis + 1 * 4);
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
	P->unrank_plane(Basis, plane_rank);
	rk = P->F->Linear_algebra->RREF_and_kernel(4, 3, Basis, 0 /* verbose_level*/);
	if (rk != 3) {
		cout << "projective_space_of_dimension_three::dual_rank_of_plane_in_three_space "
				"rk != 3" << endl;
		exit(1);
	}
	dual_rk = P->rank_point(Basis + 3 * 4);
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
	P->unrank_lines(Basis, three_lines, 3);
	rk = P->F->Linear_algebra->RREF_and_kernel(4, 6, Basis, 0 /* verbose_level*/);
	if (rk != 3) {
		cout << "projective_space_of_dimension_three::plane_equation_from_three_lines_in_three_space rk != 3" << endl;
		exit(1);
	}
	Int_vec_copy(Basis + 3 * 4, plane_eqn4, 4);

	if (f_v) {
		cout << "projective_space_of_dimension_three::plane_equation_from_three_lines_in_three_space done" << endl;
	}
}




}}}


