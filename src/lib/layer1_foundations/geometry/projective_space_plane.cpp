/*
 * projective_space_plane.cpp
 *
 *  Created on: Jan 21, 2023
 *      Author: betten
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {


projective_space_plane::projective_space_plane()
{
		P = NULL;
}

projective_space_plane::~projective_space_plane()
{
}

void projective_space_plane::init(projective_space *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_plane::init" << endl;
	}
	if (P->Subspaces->n != 2) {
		cout << "projective_space_plane::init "
				"need dimension two" << endl;
		exit(1);
	}
	projective_space_plane::P = P;
	if (f_v) {
		cout << "projective_space_plane::init done" << endl;
	}
}


int projective_space_plane::determine_line_in_plane(
	long int *two_input_pts,
	int *three_coeffs,
	int verbose_level)
// returns false is the rank of the coefficient matrix is not 2.
// true otherwise.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *coords; // [nb_pts * 3];
	int *system; // [nb_pts * 3];
	int kernel[3 * 3];
	int base_cols[3];
	int i, x, y, z, rk;
	int kernel_m, kernel_n;
	int nb_pts = 2;

	if (f_v) {
		cout << "projective_space_plane::determine_line_in_plane" << endl;
	}
	if (P->Subspaces->n != 2) {
		cout << "projective_space_plane::determine_line_in_plane "
				"n != 2" << endl;
		exit(1);
	}



	coords = NEW_int(nb_pts * 3);
	system = NEW_int(nb_pts * 3);
	for (i = 0; i < nb_pts; i++) {
		P->unrank_point(coords + i * 3, two_input_pts[i]);
	}
	if (f_vv) {
		cout << "projective_space_plane::determine_line_in_plane "
				"points:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				coords, nb_pts, 3, 3, P->Subspaces->F->log10_of_q);
	}
	for (i = 0; i < nb_pts; i++) {
		x = coords[i * 3 + 0];
		y = coords[i * 3 + 1];
		z = coords[i * 3 + 2];
		system[i * 3 + 0] = x;
		system[i * 3 + 1] = y;
		system[i * 3 + 2] = z;
	}
	if (f_v) {
		cout << "projective_space_plane::determine_line_in_plane system:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				system, nb_pts, 3, 3, P->Subspaces->F->log10_of_q);
	}



	rk = P->Subspaces->F->Linear_algebra->Gauss_simple(
			system,
			nb_pts, 3, base_cols, verbose_level - 2);
	if (rk != 2) {
		if (f_v) {
			cout << "projective_space_plane::determine_line_in_plane "
					"system undetermined" << endl;
		}
		return false;
	}
	P->Subspaces->F->Linear_algebra->matrix_get_kernel(
			system, 2, 3, base_cols, rk,
		kernel_m, kernel_n, kernel, 0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space_plane::determine_line_in_plane line:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, kernel, 1, 3, 3, P->Subspaces->F->log10_of_q);
	}
	for (i = 0; i < 3; i++) {
		three_coeffs[i] = kernel[i];
	}
	FREE_int(coords);
	FREE_int(system);
	return true;
}




int projective_space_plane::conic_test(
		long int *S, int len, int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = true;
	int subset[5];
	long int the_set[6];
	int six_coeffs[6];
	int i;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "projective_space_plane::conic_test" << endl;
	}
	if (len < 5) {
		return true;
	}
	Combi.first_k_subset(subset, len, 5);
	while (true) {
		for (i = 0; i < 5; i++) {
			the_set[i] = S[subset[i]];
		}
		the_set[5] = pt;
		if (determine_conic_in_plane(
				the_set, 6, six_coeffs,
				0 /*verbose_level*/)) {
			ret = false;
			break;
		}

		if (!Combi.next_k_subset(subset, len, 5)) {
			ret = true;
			break;
		}
	}
	if (f_v) {
		cout << "projective_space_plane::conic_test done" << endl;
	}
	return ret;
}


int projective_space_plane::test_if_conic_contains_point(
		int *six_coeffs, int pt)
{
	int v[3];
	int c[6];
	int x, y, z, s, i;

	P->unrank_point(v, pt);
	x = v[0];
	y = v[1];
	z = v[2];
	c[0] = P->Subspaces->F->mult(x, x);
	c[1] = P->Subspaces->F->mult(y, y);
	c[2] = P->Subspaces->F->mult(z, z);
	c[3] = P->Subspaces->F->mult(x, y);
	c[4] = P->Subspaces->F->mult(x, z);
	c[5] = P->Subspaces->F->mult(y, z);
	s = 0;
	for (i = 0; i < 6; i++) {
		s = P->Subspaces->F->add(s, P->Subspaces->F->mult(six_coeffs[i], c[i]));
	}
	if (s == 0) {
		return true;
	}
	else {
		return false;
	}
 }

int projective_space_plane::determine_conic_in_plane(
	long int *input_pts, int nb_pts,
	int *six_coeffs,
	int verbose_level)
// returns false if the rank of the coefficient
// matrix is not 5. true otherwise.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *coords; // [nb_pts * 3];
	int *system; // [nb_pts * 6];
	int kernel[6 * 6];
	int base_cols[6];
	int i, x, y, z, rk;
	int kernel_m, kernel_n;

	if (f_v) {
		cout << "projective_space_plane::determine_conic_in_plane" << endl;
	}
	if (P->Subspaces->n != 2) {
		cout << "projective_space_plane::determine_conic_in_plane "
				"n != 2" << endl;
		exit(1);
	}
	if (nb_pts < 5) {
		cout << "projective_space_plane::determine_conic_in_plane "
				"need at least 5 points" << endl;
		exit(1);
	}

	if (P->Arc_in_projective_space == NULL) {
		cout << "projective_space_plane::determine_conic_in_plane "
				"P->Arc_in_projective_space == NULL" << endl;
		exit(1);
	}

	if (!P->Arc_in_projective_space->arc_test(
			input_pts, nb_pts, verbose_level)) {
		if (f_v) {
			cout << "projective_space_plane::determine_conic_in_plane "
					"some 3 of the points are collinear" << endl;
		}
		return false;
	}


	coords = NEW_int(nb_pts * 3);
	system = NEW_int(nb_pts * 6);
	for (i = 0; i < nb_pts; i++) {
		P->unrank_point(coords + i * 3, input_pts[i]);
	}
	if (f_vv) {
		cout << "projective_space_plane::determine_conic_in_plane "
				"points:" << endl;
		Int_vec_print_integer_matrix_width(
				cout,
				coords, nb_pts, 3, 3, P->Subspaces->F->log10_of_q);
	}
	for (i = 0; i < nb_pts; i++) {
		x = coords[i * 3 + 0];
		y = coords[i * 3 + 1];
		z = coords[i * 3 + 2];
		system[i * 6 + 0] = P->Subspaces->F->mult(x, x);
		system[i * 6 + 1] = P->Subspaces->F->mult(y, y);
		system[i * 6 + 2] = P->Subspaces->F->mult(z, z);
		system[i * 6 + 3] = P->Subspaces->F->mult(x, y);
		system[i * 6 + 4] = P->Subspaces->F->mult(x, z);
		system[i * 6 + 5] = P->Subspaces->F->mult(y, z);
	}
	if (f_v) {
		cout << "projective_space_plane::determine_conic_in_plane "
				"system:" << endl;
		Int_vec_print_integer_matrix_width(
				cout,
				system, nb_pts, 6, 6, P->Subspaces->F->log10_of_q);
	}



	rk = P->Subspaces->F->Linear_algebra->Gauss_simple(system, nb_pts,
			6, base_cols, verbose_level - 2);
	if (rk != 5) {
		if (f_v) {
			cout << "projective_space_plane::determine_conic_in_plane "
					"system undetermined" << endl;
		}
		return false;
	}
	P->Subspaces->F->Linear_algebra->matrix_get_kernel(
			system, 5, 6, base_cols, rk,
		kernel_m, kernel_n, kernel, 0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space_plane::determine_conic_in_plane "
				"conic:" << endl;
		Int_vec_print_integer_matrix_width(
				cout,
				kernel, 1, 6, 6, P->Subspaces->F->log10_of_q);
	}
	for (i = 0; i < 6; i++) {
		six_coeffs[i] = kernel[i];
	}
	FREE_int(coords);
	FREE_int(system);
	if (f_v) {
		cout << "projective_space_plane::determine_conic_in_plane done" << endl;
	}
	return true;
}


int projective_space_plane::determine_cubic_in_plane(
		ring_theory::homogeneous_polynomial_domain *Poly_3_3,
		int nb_pts, long int *Pts, int *coeff10,
		int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int i, j, r, d;
	int *Pt_coord;
	int *System;
	int *base_cols;

	if (f_v) {
		cout << "projective_space_plane::determine_cubic_in_plane" << endl;
	}
	d = P->Subspaces->n + 1;
	Pt_coord = NEW_int(nb_pts * d);
	System = NEW_int(nb_pts * Poly_3_3->get_nb_monomials());
	base_cols = NEW_int(Poly_3_3->get_nb_monomials());

	if (f_v) {
		cout << "projective_space_plane::determine_cubic_in_plane list of "
				"points:" << endl;
		Lint_vec_print(cout, Pts, nb_pts);
		cout << endl;
	}
	for (i = 0; i < nb_pts; i++) {
		P->unrank_point(Pt_coord + i * d, Pts[i]);
	}
	if (f_v) {
		cout << "projective_space_plane::determine_cubic_in_plane matrix of "
				"point coordinates:" << endl;
		Int_matrix_print(Pt_coord, nb_pts, d);
	}

	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < Poly_3_3->get_nb_monomials(); j++) {
			System[i * Poly_3_3->get_nb_monomials() + j] =
					Poly_3_3->evaluate_monomial(j, Pt_coord + i * d);
		}
	}
	if (f_v) {
		cout << "projective_space_plane::determine_cubic_in_plane "
				"The system:" << endl;
		Int_matrix_print(System, nb_pts, Poly_3_3->get_nb_monomials());
	}
	r = P->Subspaces->F->Linear_algebra->Gauss_simple(
			System, nb_pts, Poly_3_3->get_nb_monomials(),
		base_cols, 0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space_plane::determine_cubic_in_plane "
				"The system in RREF:" << endl;
		Int_matrix_print(System, r, Poly_3_3->get_nb_monomials());
	}
	if (f_v) {
		cout << "projective_space_plane::determine_cubic_in_plane "
				"The system has rank " << r << endl;
	}

	if (r != 9) {
		cout << "r != 9" << endl;
		exit(1);
	}
	int kernel_m, kernel_n;

	P->Subspaces->F->Linear_algebra->matrix_get_kernel(
			System, r, Poly_3_3->get_nb_monomials(),
		base_cols, r,
		kernel_m, kernel_n, coeff10, 0 /* verbose_level */);


	FREE_int(Pt_coord);
	FREE_int(System);
	FREE_int(base_cols);
	if (f_v) {
		cout << "projective_space_plane::determine_cubic_in_plane done" << endl;
	}
	return r;
}

void projective_space_plane::conic_points_brute_force(
	int *six_coeffs,
	long int *points, int &nb_points, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[3];
	int i, a;

	if (f_v) {
		cout << "projective_space_plane::conic_points_brute_force" << endl;
	}
	nb_points = 0;
	for (i = 0; i < P->Subspaces->N_points; i++) {
		P->unrank_point(v, i);
		a = P->Subspaces->F->Linear_algebra->evaluate_conic_form(
				six_coeffs, v);
		if (f_vv) {
			cout << "point " << i << " = ";
			Int_vec_print(cout, v, 3);
			cout << " gives a value of " << a << endl;
		}
		if (a == 0) {
			if (f_vv) {
				cout << "point " << i << " = ";
				Int_vec_print(cout, v, 3);
				cout << " lies on the conic" << endl;
			}
			points[nb_points++] = i;
		}
	}
	if (f_v) {
		cout << "projective_space_plane::conic_points_brute_force done, "
				"we found " << nb_points << " points" << endl;
	}
	if (f_vv) {
		cout << "They are : ";
		Lint_vec_print(cout, points, nb_points);
		cout << endl;
	}
	if (f_v) {
		cout << "projective_space_plane::conic_points_brute_force done" << endl;
	}
}


void projective_space_plane::conic_points(
	long int *five_pts, int *six_coeffs,
	long int *points, int &nb_points, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int Gram_matrix[9];
	int Basis[9];
	int Basis2[9];
	int v[3], w[3];
	int i, j, l, a = 0, av, ma, b, bv, t;


	if (f_v) {
		cout << "projective_space_plane::conic_points" << endl;
	}
	if (P->Subspaces->n != 2) {
		cout << "projective_space_plane::conic_points P->n != 2" << endl;
		exit(1);
	}
	Gram_matrix[0 * 3 + 0] = P->Subspaces->F->add(six_coeffs[0], six_coeffs[0]);
	Gram_matrix[1 * 3 + 1] = P->Subspaces->F->add(six_coeffs[1], six_coeffs[1]);
	Gram_matrix[2 * 3 + 2] = P->Subspaces->F->add(six_coeffs[2], six_coeffs[2]);
	Gram_matrix[0 * 3 + 1] = Gram_matrix[1 * 3 + 0] = six_coeffs[3];
	Gram_matrix[0 * 3 + 2] = Gram_matrix[2 * 3 + 0] = six_coeffs[4];
	Gram_matrix[1 * 3 + 2] = Gram_matrix[2 * 3 + 1] = six_coeffs[5];
	if (f_vv) {
		cout << "projective_space_plane::conic_points Gram matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				Gram_matrix, 3, 3, 3, P->Subspaces->F->log10_of_q);
	}

	P->unrank_point(Basis, five_pts[0]);
	for (i = 1; i < 5; i++) {
		P->unrank_point(Basis + 3, five_pts[i]);
		a = P->Subspaces->F->Linear_algebra->evaluate_bilinear_form(
				3, Basis, Basis + 3, Gram_matrix);
		if (a) {
			break;
		}
	}
	if (i == 5) {
		cout << "projective_space_plane::conic_points did not "
				"find non-orthogonal vector" << endl;
		exit(1);
	}
	if (a != 1) {
		av = P->Subspaces->F->inverse(a);
		for (i = 0; i < 3; i++) {
			Basis[3 + i] = P->Subspaces->F->mult(av, Basis[3 + i]);
		}
	}
	if (f_v) {
		cout << "projective_space_plane::conic_points "
				"Hyperbolic pair:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				Basis, 2, 3, 3, P->Subspaces->F->log10_of_q);
	}
	P->Subspaces->F->Linear_algebra->perp(
			3, 2, Basis, Gram_matrix,
			0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space_plane::conic_points "
				"perp:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				Basis, 3, 3, 3, P->Subspaces->F->log10_of_q);
	}
	a = P->Subspaces->F->Linear_algebra->evaluate_conic_form(
			six_coeffs, Basis + 6);
	if (f_v) {
		cout << "projective_space_plane::conic_points "
				"form value = " << a << endl;
	}
	if (a == 0) {
		cout << "projective_space_plane::conic_points "
				"the form is degenerate or we are in "
				"characteristic zero" << endl;
		exit(1);
	}
	l = P->Subspaces->F->log_alpha(a);
	if ((l % 2) == 0) {
		j = l / 2;
		b = P->Subspaces->F->alpha_power(j);
		bv = P->Subspaces->F->inverse(b);
		for (i = 0; i < 3; i++) {
			Basis[6 + i] = P->Subspaces->F->mult(bv, Basis[6 + i]);
		}
		a = P->Subspaces->F->Linear_algebra->evaluate_conic_form(
				six_coeffs, Basis + 6);
		if (f_v) {
			cout << "form value = " << a << endl;
		}
	}
	for (i = 0; i < 3; i++) {
		Basis2[3 + i] = Basis[6 + i];
	}
	for (i = 0; i < 3; i++) {
		Basis2[0 + i] = Basis[0 + i];
	}
	for (i = 0; i < 3; i++) {
		Basis2[6 + i] = Basis[3 + i];
	}
	if (f_v) {
		cout << "Basis2:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				Basis2, 3, 3, 3, P->Subspaces->F->log10_of_q);
	}
	// Now the form is a^{-1}y_1^2 = y_0y_2
	// (or, equivalently, a^{-1}y_1^2 - y_0y_2 = 0)
	// and  the quadratic form on (0,1,0) in y-coordinates is a.
	//
	// In the y-coordinates, the points on this conic are
	// (1,0,0) and (t^2,t,-a) for t \in GF(q).
	// In the x-coordinates, the points are
	// (1,0,0) * Basis2 and (t^2,t,-a) * Basis2 for t \in GF(q).

	v[0] = 1;
	v[1] = 0;
	v[2] = 0;

	P->Subspaces->F->Linear_algebra->mult_vector_from_the_left(
			v, Basis2, w, 3, 3);
	if (f_v) {
		cout << "vector corresponding to 100:" << endl;
		Int_vec_print(cout, w, 3);
		cout << endl;
	}
	b = P->rank_point(w);
	points[0] = b;
	nb_points = 1;

	ma = P->Subspaces->F->negate(a);

	for (t = 0; t < P->Subspaces->F->q; t++) {
		v[0] = P->Subspaces->F->mult(t, t);
		v[1] = t;
		v[2] = ma;
		P->Subspaces->F->Linear_algebra->mult_vector_from_the_left(
				v, Basis2, w, 3, 3);
		if (f_v) {
			cout << "vector corresponding to t=" << t << ":" << endl;
			Int_vec_print(cout, w, 3);
			cout << endl;
		}
		b = P->rank_point(w);
		points[nb_points++] = b;
	}
	if (f_vv) {
		cout << "projective_space_plane::conic_points conic points:" << endl;
		Lint_vec_print(cout, points, nb_points);
		cout << endl;
	}
	if (f_v) {
		cout << "projective_space_plane::conic_points done" << endl;
	}
}

void projective_space_plane::find_tangent_lines_to_conic(
	int *six_coeffs,
	long int *points, int nb_points,
	long int *tangents, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int v[3];
	int Basis[9];
	int Gram_matrix[9];
	int i;

	if (f_v) {
		cout << "projective_space_plane::find_tangent_lines_to_conic" << endl;
	}
	if (P->Subspaces->n != 2) {
		cout << "projective_space_plane::find_tangent_lines_to_conic "
				"P->n != 2" << endl;
		exit(1);
	}
	Gram_matrix[0 * 3 + 0] = P->Subspaces->F->add(six_coeffs[0], six_coeffs[0]);
	Gram_matrix[1 * 3 + 1] = P->Subspaces->F->add(six_coeffs[1], six_coeffs[1]);
	Gram_matrix[2 * 3 + 2] = P->Subspaces->F->add(six_coeffs[2], six_coeffs[2]);
	Gram_matrix[0 * 3 + 1] = Gram_matrix[1 * 3 + 0] = six_coeffs[3];
	Gram_matrix[0 * 3 + 2] = Gram_matrix[2 * 3 + 0] = six_coeffs[4];
	Gram_matrix[1 * 3 + 2] = Gram_matrix[2 * 3 + 1] = six_coeffs[5];

	for (i = 0; i < nb_points; i++) {
		P->unrank_point(Basis, points[i]);
		P->Subspaces->F->Linear_algebra->perp(
				3, 1, Basis, Gram_matrix,
				0 /* verbose_level */);
		if (f_vv) {
			cout << "perp:" << endl;
			Int_vec_print_integer_matrix_width(cout,
					Basis, 3, 3, 3, P->Subspaces->F->log10_of_q);
		}
		tangents[i] = P->rank_line(Basis + 3);
		if (f_vv) {
			cout << "tangent at point " << i << " is "
					<< tangents[i] << endl;
		}
	}
	if (f_v) {
		cout << "projective_space_plane::find_tangent_lines_to_conic done" << endl;
	}
}

int projective_space_plane::determine_hermitian_form_in_plane(
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
		cout << "projective_space_plane::determine_hermitian_form_in_plane" << endl;
	}
	coords = NEW_int(nb_pts * 3);
	system = NEW_int(nb_pts * 9);
	Q = P->Subspaces->F->q;
	if (ODD(P->Subspaces->F->e)) {
		cout << "projective_space_plane::determine_hermitian_form_in_plane "
				"field degree must be even" << endl;
		exit(1);
	}
	little_e = P->Subspaces->F->e >> 1;
	q = NT.i_power_j(P->Subspaces->F->p, little_e);
	if (f_v) {
		cout << "projective_space_plane::determine_hermitian_form_in_plane "
				"Q=" << Q << " q=" << q << endl;
	}
	if (P->Subspaces->n != 2) {
		cout << "projective_space_plane::determine_hermitian_form_in_plane "
				"n != 2" << endl;
		exit(1);
	}
	for (i = 0; i < nb_pts; i++) {
		P->unrank_point(coords + i * 3, pts[i]);
	}
	if (f_vv) {
		cout << "projective_space_plane::determine_hermitian_form_in_plane "
				"points:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				coords, nb_pts, 3, 3, P->Subspaces->F->log10_of_q);
	}
	for (i = 0; i < nb_pts; i++) {
		x = coords[i * 3 + 0];
		y = coords[i * 3 + 1];
		z = coords[i * 3 + 2];
		xq = P->Subspaces->F->frobenius_power(x, little_e);
		yq = P->Subspaces->F->frobenius_power(y, little_e);
		zq = P->Subspaces->F->frobenius_power(z, little_e);
		system[i * 9 + 0] = P->Subspaces->F->mult(x, xq);
		system[i * 9 + 1] = P->Subspaces->F->mult(y, yq);
		system[i * 9 + 2] = P->Subspaces->F->mult(z, zq);
		system[i * 9 + 3] = P->Subspaces->F->mult(x, yq);
		system[i * 9 + 4] = P->Subspaces->F->mult(y, xq);
		system[i * 9 + 5] = P->Subspaces->F->mult(x, zq);
		system[i * 9 + 6] = P->Subspaces->F->mult(z, xq);
		system[i * 9 + 7] = P->Subspaces->F->mult(y, zq);
		system[i * 9 + 8] = P->Subspaces->F->mult(z, yq);
	}
	if (f_v) {
		cout << "projective_space_plane::determine_hermitian_form_in_plane "
				"system:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				system, nb_pts, 9, 9, P->Subspaces->F->log10_of_q);
	}



	rk = P->Subspaces->F->Linear_algebra->Gauss_simple(system,
			nb_pts, 9, base_cols, verbose_level - 2);
	if (f_v) {
		cout << "projective_space_plane::determine_hermitian_form_in_plane "
				"rk=" << rk << endl;
		Int_vec_print_integer_matrix_width(cout,
				system, rk, 9, 9, P->Subspaces->F->log10_of_q);
	}
#if 0
	if (rk != 8) {
		if (f_v) {
			cout << "projective_space_plane::determine_hermitian_form_"
					"in_plane system under-determined" << endl;
		}
		return false;
	}
#endif
	P->Subspaces->F->Linear_algebra->matrix_get_kernel(system,
			MINIMUM(nb_pts, 9), 9, base_cols, rk,
		kernel_m, kernel_n, kernel, 0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space_plane::determine_hermitian_form_in_plane "
				"kernel:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, kernel,
				kernel_m, kernel_n, kernel_n, P->Subspaces->F->log10_of_q);
	}
	six_coeffs[0] = kernel[0 * kernel_n + 0];
	six_coeffs[1] = kernel[1 * kernel_n + 0];
	six_coeffs[2] = kernel[2 * kernel_n + 0];
	six_coeffs[3] = kernel[3 * kernel_n + 0];
	six_coeffs[4] = kernel[5 * kernel_n + 0];
	six_coeffs[5] = kernel[7 * kernel_n + 0];
	if (f_v) {
		cout << "projective_space_plane::determine_hermitian_form_in_plane "
				"six_coeffs:" << endl;
		Int_vec_print(cout, six_coeffs, 6);
		cout << endl;
	}
	FREE_int(coords);
	FREE_int(system);
	if (f_v) {
		cout << "projective_space_plane::determine_hermitian_form_in_plane done" << endl;
	}
	return true;
}

void projective_space_plane::conic_type_randomized(
		int nb_times,
	long int *set, int set_size,
	long int **&Pts_on_conic,
	int *&nb_pts_on_conic, int &len,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_plane::conic_type_randomized" << endl;
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
		cout << "projective_space_plane::conic_type_randomized" << endl;
	}
	if (P->Subspaces->n != 2) {
		cout << "projective_space_plane::conic_type_randomized "
				"P->Subspaces->n != 2" << endl;
		exit(1);
	}
	if (f_vv) {
		P->Reporting->print_set_numerical(
				cout, set, set_size);
	}

	if (!Sorting.test_if_set_with_return_value_lint(set, set_size)) {
		cout << "projective_space_plane::conic_type_randomized "
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
			if (Sorting.lint_vec_is_subset_of(
					subset, 5,
					Pts_on_conic[i], nb_pts_on_conic[i],
					0 /* verbose_level */)) {

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
		if (false /* f_v3 */) {
			cout << "subset: ";
			Int_vec_print(cout, subset, 5);
			cout << "input_pts: ";
			Lint_vec_print(cout, input_pts, 5);
		}

		if (!determine_conic_in_plane(input_pts,
				5, six_coeffs, 0 /* verbose_level */)) {
			continue;
		}


		P->Subspaces->F->Projective_space_basic->PG_element_normalize(
				six_coeffs, 1, 6);
		Gg.AG_element_rank_longinteger(P->Subspaces->F->q, six_coeffs, 1, 6, conic_rk);
		if (false /* f_vv */) {
			cout << rk << "-th subset ";
			Int_vec_print(cout, subset, 5);
			cout << " conic_rk=" << conic_rk << endl;
		}

		if (false /* longinteger_vec_search(R, len, conic_rk, idx) */) {

#if 0
			cout << "projective_space_plane::conic_type_randomized "
					"longinteger_vec_search(R, len, conic_rk, idx) "
					"is true" << endl;
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
				if (false && f_v3) {
					cout << "testing point " << h << ":" << endl;
					cout << "conic_rk=" << conic_rk << endl;
				}

				P->unrank_point(vec, set[h]);
				a = P->Subspaces->F->Linear_algebra->evaluate_conic_form(
						six_coeffs, vec);


				if (a == 0) {
					pts_on_conic[l++] = h;
					if (f_v3) {
						cout << "point " << h << " is on the conic" << endl;
					}
				}
				else {
					if (false && f_v3) {
						cout << "point " << h
								<< " is not on the conic" << endl;
					}
				}
			}
			if (false /*f_v*/) {
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
					int f_second = false;

					C.init(nb_pts_on_conic, len, f_second, 0);

					if (f_v) {
						cout << "The conic intersection type is (";
						C.print_bare(false /*f_backwards*/);
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
		cout << "projective_space_plane::conic_type_randomized done" << endl;
	}
}

void projective_space_plane::conic_intersection_type(
	int f_randomized, int nb_times,
	long int *set, int set_size,
	int threshold,
	int *&intersection_type, int &highest_intersection_number,
	int f_save_largest_sets,
	data_structures::set_of_sets *&largest_sets,
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
		cout << "projective_space_plane::conic_intersection_type "
				"threshold = " << threshold << endl;
	}

	if (f_randomized) {
		if (f_v) {
			cout << "projective_space_plane::conic_intersection_type "
					"randomized" << endl;
		}
		conic_type_randomized(nb_times,
			set, set_size,
			Pts_on_conic, nb_pts_on_conic, nb_conics,
			verbose_level - 1);
	}
	else {
		if (f_v) {
			cout << "projective_space_plane::conic_intersection_type "
					"not randomized" << endl;
		}
		conic_type(
			set, set_size, threshold,
			Pts_on_conic, Conic_eqn, nb_pts_on_conic, nb_conics,
			verbose_level - 1);
	}

	data_structures::tally C;
	int f_second = false;

	C.init(nb_pts_on_conic, nb_conics, f_second, 0);
	if (f_v) {
		cout << "projective_space_plane::conic_intersection_type "
				"conic-intersection type: ";
		C.print(false /*f_backwards*/);
	}

	if (f_v) {
		cout << "The conic intersection type is (";
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
		cout << "projective_space_plane::conic_intersection_type done" << endl;
	}

}

void projective_space_plane::determine_nonconical_six_subsets(
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

	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space_plane::determine_nonconical_six_subsets" << endl;
	}
	if (P->Subspaces->n != 2) {
		cout << "projective_space_plane::determine_nonconical_six_subsets "
				"P->Subspaces->n != 2" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "projective_space_plane::determine_nonconical_six_subsets "
				"before conic_type" << endl;
	}
	conic_type(
		set, set_size,
		threshold,
		Pts_on_conic, Conic_eqn, nb_pts_on_conic, len,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_plane::determine_nonconical_six_subsets "
				"after conic_type" << endl;
	}
	if (f_v) {
		cout << "There are " << len << " conics. "
				"They contain the following points:" << endl;
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
			cout << "projective_space_plane::conic_type "
					"rk=" << rk << " / " << N << " : ";
			Int_vec_print(cout, subset, 6);
			cout << endl;
		}

		for (i = 0; i < len; i++) {
			if (Sorting.lint_vec_is_subset_of(
					subset, 6,
					Pts_on_conic[i], nb_pts_on_conic[i],
					0 /* verbose_level */)) {

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
				if (false) {
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
	geometry_global Geo;

	nb = Rk.size();
	Nb_E = NEW_int(nb);
	if (f_v) {
		cout << "Computing Eckardt point number distribution" << endl;
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
		nb_E = Geo.nonconical_six_arc_get_nb_Eckardt_points(P,
				Arc6, 0 /* verbose_level */);
		Nb_E[i] = nb_E;
	}

	data_structures::tally T;

	T.init(Nb_E, nb, false, 0);
	if (f_v) {
		cout << "Eckardt point number distribution : ";
		T.print_file_tex(cout, true /* f_backwards*/);
		cout << endl;
	}


	if (nb) {
		int m, idx;
		int *Idx;
		int nb_idx;
		int *System;

		m = Int_vec_maximum(Nb_E, nb);
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
				cout << i << " / " << nb_idx
						<< " idx=" << idx << ", rk=" << rk << " :" << endl;
			}
			Combi.unrank_k_subset(rk, subset, set_size, 6);

			Int_vec_copy(subset, System + i * 6, 6);

			for (j = 0; j < 6; j++) {
				Arc6[j] = set[subset[j]];
			}
			nb_E = Geo.nonconical_six_arc_get_nb_Eckardt_points(P,
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
		Fio.Csv_file_support->int_matrix_write_csv(
				fname, System, nb_idx, 6);
		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;


		data_structures::tally T2;

		T2.init(System, nb_idx * 6, false, 0);
		if (f_v) {
			cout << "distribution of points: ";
			T2.print_file_tex(cout, true /* f_backwards*/);
			cout << endl;
		}


	}


	if (f_v) {
		cout << "projective_space_plane::determine_nonconical_six_subsets done" << endl;
	}
}

void projective_space_plane::conic_type(
	long int *set, int set_size,
	int threshold,
	long int **&Pts_on_conic,
	int **&Conic_eqn,
	int *&nb_pts_on_conic, int &nb_conics,
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
		cout << "projective_space_plane::conic_type, "
				"threshold = " << threshold << endl;
	}
	if (P->Subspaces->n != 2) {
		cout << "projective_space_plane::conic_type "
				"P->Subspaces->n != 2" << endl;
		exit(1);
	}
	if (f_vv) {
		P->Reporting->print_set_numerical(cout, set, set_size);
	}

	if (!Sorting.test_if_set_with_return_value_lint(set, set_size)) {
		cout << "projective_space_plane::conic_type the input "
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
		P->unrank_point(coords + i * 3, set[i]);
	}
	if (f_v) {
		cout << "projective_space_plane::conic_type coords:" << endl;
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
		if (false) {
			cout << "projective_space_plane::conic_type "
					"rk=" << rk << " / " << N << " : ";
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
		if (false) {
			cout << "subset: ";
			Int_vec_print(cout, subset, 5);
			cout << endl;
			cout << "input_pts: ";
			Lint_vec_print(cout, input_pts, 5);
			cout << endl;
		}

		if (!determine_conic_in_plane(input_pts, 5,
				six_coeffs, verbose_level - 2)) {
			if (false) {
				cout << "determine_conic_in_plane returns false" << endl;
			}
			continue;
		}
		if (f_v) {
			cout << "projective_space_plane::conic_type "
					"rk=" << rk << " / " << N << " : ";
			Int_vec_print(cout, subset, 5);
			cout << " has not yet been considered "
					"and a conic exists" << endl;
		}
		if (f_v) {
			cout << "determine_conic_in_plane the conic exists" << endl;
			cout << "conic: ";
			Int_vec_print(cout, six_coeffs, 6);
			cout << endl;
		}


		P->Subspaces->F->Projective_space_basic->PG_element_normalize(
				six_coeffs, 1, 6);
		Gg.AG_element_rank_longinteger(P->Subspaces->F->q, six_coeffs, 1, 6, conic_rk);
		if (false /* f_vv */) {
			cout << rk << "-th subset ";
			Int_vec_print(cout, subset, 5);
			cout << " conic_rk=" << conic_rk << endl;
		}

		if (false /* longinteger_vec_search(R, len, conic_rk, idx) */) {

#if 0
			cout << "projective_space_plane::conic_type_randomized "
					"longinteger_vec_search(R, len, conic_rk, idx) "
					"is true" << endl;
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
				cout << "considering conic of rank "
						"conic_rk=" << conic_rk << ":" << endl;
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
				a = P->Subspaces->F->Linear_algebra->evaluate_conic_form(
						six_coeffs, vec);


				if (a == 0) {
					pts_on_conic[l++] = h;
					if (false) {
						cout << "point " << h
								<< " is on the conic" << endl;
					}
				}
				else {
					if (false && f_v3) {
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
					int f_second = false;

					C.init(nb_pts_on_conic, nb_conics, f_second, 0);

					if (f_v) {
						cout << "The conic intersection type is (";
						C.print_bare(false /*f_backwards*/);
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
					cout << "projective_space_plane::conic_type "
							"we will skip this conic" << endl;
				}
				FREE_lint(pts_on_conic);
			}
		} // else
	} // next rk

	FREE_int(coords);

	if (f_v) {
		cout << "projective_space_plane::conic_type we found " << nb_conics
				<< " conics intersecting in at least "
				<< threshold << " many points" << endl;
	}

	if (f_v) {
		cout << "projective_space_plane::conic_type done" << endl;
	}
}

void projective_space_plane::find_nucleus(
	int *set, int set_size, int &nucleus,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, l, sz, idx, t1, t2;
	int *Lines;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space_plane::find_nucleus" << endl;
	}

	if (P->Subspaces->n != 2) {
		cout << "projective_space_plane::find_nucleus n != 2" << endl;
		exit(1);
	}
	if (set_size != P->Subspaces->F->q + 1) {
		cout << "projective_space_plane::find_nucleus "
				"set_size != F->q + 1" << endl;
		exit(1);
	}

	if (P->Subspaces->Implementation->Lines_on_point == NULL) {
		if (f_v) {
			cout << "projective_space_plane::find_nucleus "
					"before P->init_incidence_structure" << endl;
		}
		P->Subspaces->init_incidence_structure(verbose_level);
		if (f_v) {
			cout << "projective_space_plane::find_nucleus "
					"after P->init_incidence_structure" << endl;
		}
	}

	Lines = NEW_int(P->Subspaces->r);
	a = set[0];
	for (i = 0; i < P->Subspaces->r; i++) {
		Lines[i] = P->Subspaces->Implementation->Lines_on_point[a * P->Subspaces->r + i];
	}
	sz = P->Subspaces->r;
	Sorting.int_vec_heapsort(Lines, P->Subspaces->r);

	for (i = 0; i < set_size - 1; i++) {
		b = set[1 + i];
		l = P->Subspaces->line_through_two_points(a, b);
		if (!Sorting.int_vec_search(Lines, sz, l, idx)) {
			cout << "projective_space_plane::find_nucleus "
					"cannot find secant in pencil" << endl;
			exit(1);
		}
		for (j = idx + 1; j < sz; j++) {
			Lines[j - 1] = Lines[j];
		}
		sz--;
	}
	if (sz != 1) {
		cout << "projective_space_plane::find_nucleus "
				"sz != 1" << endl;
		exit(1);
	}
	t1 = Lines[0];
	if (f_v) {
		cout << "projective_space_plane::find_nucleus "
				"t1 = " << t1 << endl;
	}



	a = set[1];
	for (i = 0; i < P->Subspaces->r; i++) {
		Lines[i] = P->Subspaces->Implementation->Lines_on_point[a * P->Subspaces->r + i];
	}
	sz = P->Subspaces->r;
	Sorting.int_vec_heapsort(Lines, P->Subspaces->r);

	for (i = 0; i < set_size - 1; i++) {
		if (i == 0) {
			b = set[0];
		}
		else {
			b = set[1 + i];
		}
		l = P->Subspaces->line_through_two_points(a, b);
		if (!Sorting.int_vec_search(Lines, sz, l, idx)) {
			cout << "projective_space_plane::find_nucleus "
					"cannot find secant in pencil" << endl;
			exit(1);
		}
		for (j = idx + 1; j < sz; j++) {
			Lines[j - 1] = Lines[j];
		}
		sz--;
	}
	if (sz != 1) {
		cout << "projective_space_plane::find_nucleus sz != 1" << endl;
		exit(1);
	}
	t2 = Lines[0];
	if (f_v) {
		cout << "projective_space_plane::find_nucleus t2 = " << t2 << endl;
	}

	nucleus = P->Subspaces->intersection_of_two_lines(t1, t2);
	if (f_v) {
		cout << "projective_space_plane::find_nucleus "
				"nucleus = " << nucleus << endl;
		int v[3];
		P->unrank_point(v, nucleus);
		cout << "nucleus = ";
		Int_vec_print(cout, v, 3);
		cout << endl;
	}



	if (f_v) {
		cout << "projective_space_plane::find_nucleus done" << endl;
	}
}

void projective_space_plane::points_on_projective_triangle(
	long int *&set, int &set_size, long int *three_points,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int three_lines[3];
	long int *Pts;
	int sz, h, i, a;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space_plane::points_on_projective_triangle" << endl;
	}
	set_size = 3 * (P->Subspaces->q - 1);
	set = NEW_lint(set_size);
	sz = 3 * (P->Subspaces->q + 1);
	Pts = NEW_lint(sz);
	three_lines[0] = P->Subspaces->line_through_two_points(three_points[0], three_points[1]);
	three_lines[1] = P->Subspaces->line_through_two_points(three_points[0], three_points[2]);
	three_lines[2] = P->Subspaces->line_through_two_points(three_points[1], three_points[2]);

	P->Subspaces->create_points_on_line(
			three_lines[0],
			Pts,
			0 /* verbose_level */);
	P->Subspaces->create_points_on_line(
			three_lines[1],
			Pts + (P->Subspaces->q + 1),
			0 /* verbose_level */);
	P->Subspaces->create_points_on_line(
			three_lines[2],
			Pts + 2 * (P->Subspaces->q + 1),
			0 /* verbose_level */);
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
		cout << "projective_space_plane::points_on_projective_triangle "
				"h != set_size" << endl;
		exit(1);
	}
	Sorting.lint_vec_heapsort(set, set_size);

	FREE_lint(Pts);
	if (f_v) {
		cout << "projective_space_plane::points_on_projective_triangle "
				"done" << endl;
	}
}

long int projective_space_plane::dual_rank_of_line_in_plane(
	long int line_rank, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis[3 * 3];
	int rk;
	long int dual_rk;

	if (f_v) {
		cout << "projective_space_plane::dual_rank_of_line_in_plane" << endl;
	}
	P->unrank_line(Basis, line_rank);
	rk = P->Subspaces->F->Linear_algebra->RREF_and_kernel(3, 2, Basis, 0 /* verbose_level*/);
	if (rk != 2) {
		cout << "projective_space_plane::dual_rank_of_line_in_plane rk != 2" << endl;
		exit(1);
	}
	dual_rk = P->rank_point(Basis + 2 * 3);
	if (f_v) {
		cout << "projective_space_plane::dual_rank_of_line_in_plane done" << endl;
	}
	return dual_rk;
}

long int projective_space_plane::line_rank_using_dual_coordinates_in_plane(
	int *eqn3, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis[3 * 3];
	int rk;
	long int line_rk;

	if (f_v) {
		cout << "projective_space_plane::line_rank_using_dual_coordinates_in_plane" << endl;
	}
	Int_vec_copy(eqn3, Basis, 3);
	rk = P->Subspaces->F->Linear_algebra->RREF_and_kernel(
			3, 1, Basis, 0 /* verbose_level*/);
	if (rk != 1) {
		cout << "projective_space_plane::line_rank_using_dual_coordinates_in_plane rk != 1" << endl;
		exit(1);
	}
	line_rk = P->rank_line(Basis + 1 * 3);
	if (f_v) {
		cout << "projective_space_plane::line_rank_using_dual_coordinates_in_plane" << endl;
	}
	return line_rk;
}


}}}


