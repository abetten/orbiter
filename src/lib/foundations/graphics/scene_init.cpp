/*
 * scene_init.cpp
 *
 *  Created on: Apr 4, 2022
 *      Author: betten
 */






#include "foundations.h"

using namespace std;



#define EPSILON 0.01

namespace orbiter {
namespace layer1_foundations {
namespace graphics {


int scene::line6(double *x6)
{
	Line_coords[nb_lines * 6 + 0] = x6[0];
	Line_coords[nb_lines * 6 + 1] = x6[1];
	Line_coords[nb_lines * 6 + 2] = x6[2];
	Line_coords[nb_lines * 6 + 3] = x6[3];
	Line_coords[nb_lines * 6 + 4] = x6[4];
	Line_coords[nb_lines * 6 + 5] = x6[5];
	nb_lines++;
	if (nb_lines >= SCENE_MAX_LINES) {
		cout << "too many lines" << endl;
		exit(1);
	}
	return nb_lines - 1;
}

int scene::line(double x1, double x2, double x3,
	double y1, double y2, double y3)
{
	double coords[6];

	coords[0]= x1;
	coords[1]= x2;
	coords[2]= x3;
	coords[3]= y1;
	coords[4]= y2;
	coords[5]= y3;
	line6(coords);

#if 0
	Line_coords[nb_lines * 6 + 0] = x1;
	Line_coords[nb_lines * 6 + 1] = x2;
	Line_coords[nb_lines * 6 + 2] = x3;
	Line_coords[nb_lines * 6 + 3] = y1;
	Line_coords[nb_lines * 6 + 4] = y2;
	Line_coords[nb_lines * 6 + 5] = y3;
	nb_lines++;
	if (nb_lines >= SCENE_MAX_LINES) {
		cout << "too many lines" << endl;
		exit(1);
	}
#endif

	return nb_lines - 1;
}


int scene::point(double x1, double x2, double x3)
{
	Point_coords[nb_points * 3 + 0] = x1;
	Point_coords[nb_points * 3 + 1] = x2;
	Point_coords[nb_points * 3 + 2] = x3;
	nb_points++;
	if (nb_points >= SCENE_MAX_POINTS) {
		cout << "too many points" << endl;
		exit(1);
	}
	return nb_points - 1;
}

int scene::edge(int pt1, int pt2)
{
	Edge_points[nb_edges * 2 + 0] = pt1;
	Edge_points[nb_edges * 2 + 1] = pt2;
	nb_edges++;
	if (nb_edges >= SCENE_MAX_EDGES) {
		cout << "too many edges" << endl;
		exit(1);
	}
	return nb_edges - 1;
}

int scene::plane(double x1, double x2, double x3, double a)
// A plane is called a polynomial shape because
// it is defined by a first order polynomial equation.
// Given a plane: plane { <A, B, C>, D }
// it can be represented by the equation
// A*x + B*y + C*z - D*sqrt(A^2 + B^2 + C^2) = 0.
// see http://www.povray.org/documentation/view/3.6.1/297/
// Example:
// plane { <0, 1, 0>, 4 }
// This is a plane where straight up is defined in the positive y-direction.
// The plane is 4 units in that direction away from the origin.
// Because most planes are defined with surface normals in the direction
// of an axis you will often see planes defined using the x, y or z
// built-in vector identifiers. The example above could be specified as:
//  plane { y, 4 }


//intersection {
//  box { <-1.5, -1, -1>, <0.5, 1, 1> }
//  cylinder { <0.5, 0, -1>, <0.5, 0, 1>, 1 }
//  }
{
	Plane_coords[nb_planes * 4 + 0] = x1;
	Plane_coords[nb_planes * 4 + 1] = x2;
	Plane_coords[nb_planes * 4 + 2] = x3;
	Plane_coords[nb_planes * 4 + 3] = a;
	nb_planes++;
	if (nb_planes >= SCENE_MAX_PLANES) {
		cout << "too many planes" << endl;
		exit(1);
	}
	return nb_planes - 1;
}


int scene::quadric(double *coeff10)
// povray (lexicographic) ordering of monomials:
// http://www.povray.org/documentation/view/3.6.1/298/
// 1: x^2
// 2: xy
// 3: xz
// 4: x
// 5: y^2
// 6: yz
// 7: y
// 8: z^2
// 9: z
// 10: 1
{
	int i;

	scene_element_of_type_surface *SE_surf;

	SE_surf = NEW_OBJECT(scene_element_of_type_surface);
	SE_surf->init(2, 10, coeff10);

	if (nb_quadrics >= SCENE_MAX_QUADRICS) {
		cout << "too many quadrics" << endl;
		exit(1);
	}
	for (i = 0; i < 10; i++) {
		Quadric_coords[nb_quadrics * 10 + i] = coeff10[i];
		}
	nb_quadrics++;
	return nb_quadrics - 1;
}


int scene::cubic(double *coeff20)
// povray (lexicographic) ordering of monomials:
// http://www.povray.org/documentation/view/3.6.1/298/
// 0: x^3
// 1: x^2y
// 2: x^2z
// 3: x^2
// 4: xy^2
// 5: xyz
// 6: xy
// 7: xz^2
// 8: xz
// 9: x
// 10: y^3
// 11: y^2z
// 12: y^2
// 13: yz^2
// 14: yz
// 15: y
// 16: z^3
// 17: z^2
// 18: z
// 19: 1
{
	int i;

	scene_element_of_type_surface *SE_surf;

	SE_surf = NEW_OBJECT(scene_element_of_type_surface);
	SE_surf->init(3, 20, coeff20);


	for (i = 0; i < 20; i++) {
		Cubic_coords[nb_cubics * 20 + i] = coeff20[i];
	}
	nb_cubics++;
	if (nb_cubics >= SCENE_MAX_CUBICS) {
		cout << "too many cubics" << endl;
		exit(1);
	}
	return nb_cubics - 1;
}

int scene::quartic(double *coeff35)
// povray (lexicographic) ordering of monomials:
// http://www.povray.org/documentation/view/3.6.1/298/
{
	int i;

	scene_element_of_type_surface *SE_surf;

	SE_surf = NEW_OBJECT(scene_element_of_type_surface);
	SE_surf->init(4, 35, coeff35);

	if (nb_quartics >= SCENE_MAX_QUARTICS) {
		cout << "too many quartics" << endl;
		exit(1);
	}
	for (i = 0; i < 35; i++) {
		Quartic_coords[nb_quartics * 35 + i] = coeff35[i];
		}
	nb_quartics++;
	return nb_quartics - 1;
}


int scene::quintic(double *coeff_56)
// povray (lexicographic) ordering of monomials:
// http://www.povray.org/documentation/view/3.6.1/298/
{
	int i;

	scene_element_of_type_surface *SE_surf;

	SE_surf = NEW_OBJECT(scene_element_of_type_surface);
	SE_surf->init(5, 56, coeff_56);

	if (nb_quintics >= SCENE_MAX_QUINTICS) {
		cout << "scene::quintic too many quintics" << endl;
		exit(1);
	}
	for (i = 0; i < 56; i++) {
		Quintic_coords[nb_quintics * 56 + i] = coeff_56[i];
	}
	nb_quintics++;
	return nb_quintics - 1;

}

int scene::octic(double *coeff_165)
{
	int i;

	scene_element_of_type_surface *SE_surf;

	SE_surf = NEW_OBJECT(scene_element_of_type_surface);
	SE_surf->init(8, 165, coeff_165);

	if (nb_octics >= SCENE_MAX_OCTICS) {
		cout << "too many octics" << endl;
		exit(1);
	}
	for (i = 0; i < 165; i++) {
		Octic_coords[nb_octics * 165 + i] = coeff_165[i];
	}
	nb_octics++;
	return nb_octics - 1;

}


int scene::face(int *pts, int nb_pts)
{
	Face_points[nb_faces] = NEW_int(nb_pts);
	Nb_face_points[nb_faces] = nb_pts;
	Int_vec_copy(pts, Face_points[nb_faces], nb_pts);
	nb_faces++;
	if (nb_faces >= SCENE_MAX_FACES) {
		cout << "too many faces" << endl;
		exit(1);
	}
	return nb_faces - 1;
}

int scene::face3(int pt1, int pt2, int pt3)
{
	int pts[3];

	pts[0] = pt1;
	pts[1] = pt2;
	pts[2] = pt3;
	return face(pts, 3);
}

int scene::face4(int pt1, int pt2, int pt3, int pt4)
{
	int pts[4];

	pts[0] = pt1;
	pts[1] = pt2;
	pts[2] = pt3;
	pts[3] = pt4;
	return face(pts, 4);
}

int scene::face5(int pt1, int pt2, int pt3, int pt4, int pt5)
{
	int pts[5];

	pts[0] = pt1;
	pts[1] = pt2;
	pts[2] = pt3;
	pts[3] = pt4;
	pts[4] = pt5;
	return face(pts, 5);
}




int scene::line_pt_and_dir(double *x6, double rad, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double pt[6];
	double pt2[6];
	numerics N;
	int ret;

	if (f_v) {
		cout << "scene::line_pt_and_dir" << endl;
	}
	pt[0] = x6[0];
	pt[1] = x6[1];
	pt[2] = x6[2];
	pt[3] = x6[0] + x6[3];
	pt[4] = x6[1] + x6[4];
	pt[5] = x6[2] + x6[5];
	if (N.line_centered_tolerant(pt, pt + 3,
		pt2, pt2 + 3,
		rad, verbose_level)) {


		line6(pt2);
#if 0
		Line_coords[nb_lines * 6 + 0] = pt2[0];
		Line_coords[nb_lines * 6 + 1] = pt2[1];
		Line_coords[nb_lines * 6 + 2] = pt2[2];
		Line_coords[nb_lines * 6 + 3] = pt2[3];
		Line_coords[nb_lines * 6 + 4] = pt2[4];
		Line_coords[nb_lines * 6 + 5] = pt2[5];
		nb_lines++;
		if (nb_lines >= SCENE_MAX_LINES) {
			cout << "too many lines" << endl;
			exit(1);
		}
#endif
		ret = TRUE;
	}
	else {
		ret = FALSE;
	}
	if (f_v) {
		cout << "scene::line_pt_and_dir done" << endl;
	}
	return ret;
}

int scene::line_pt_and_dir_and_copy_points(double *x6, double rad, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double pt[6];
	double pt2[6];
	numerics N;
	int ret;

	if (f_v) {
		cout << "scene::line_pt_and_dir_and_copy_points" << endl;
	}
	pt[0] = x6[0];
	pt[1] = x6[1];
	pt[2] = x6[2];
	pt[3] = x6[0] + x6[3];
	pt[4] = x6[1] + x6[4];
	pt[5] = x6[2] + x6[5];
	if (N.line_centered_tolerant(pt, pt + 3,
		pt2, pt2 + 3,
		rad, verbose_level)) {

		line6(pt2);
#if 0
		Line_coords[nb_lines * 6 + 0] = pt2[0];
		Line_coords[nb_lines * 6 + 1] = pt2[1];
		Line_coords[nb_lines * 6 + 2] = pt2[2];
		Line_coords[nb_lines * 6 + 3] = pt2[3];
		Line_coords[nb_lines * 6 + 4] = pt2[4];
		Line_coords[nb_lines * 6 + 5] = pt2[5];
		nb_lines++;
		if (nb_lines >= SCENE_MAX_LINES) {
			cout << "too many lines" << endl;
			exit(1);
		}
#endif
		points(pt2, 2);
		ret = TRUE;
	}
	else {
		cout << "line_pt_and_dir_and_copy_points could not create points" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "scene::line_pt_and_dir_and_copy_points done" << endl;
	}
	return ret;
}


int scene::line_through_two_pts(double *x6, double rad)
{
	double pt[6];
	double pt2[6];
	numerics N;
	int verbose_level = 0;

	pt[0] = x6[0];
	pt[1] = x6[1];
	pt[2] = x6[2];
	pt[3] = x6[3];
	pt[4] = x6[4];
	pt[5] = x6[5];
	if (N.line_centered(pt, pt + 3,
		pt2, pt2 + 3,
		rad, verbose_level)) {

		line6(pt2);

#if 0
		Line_coords[nb_lines * 6 + 0] = pt2[0];
		Line_coords[nb_lines * 6 + 1] = pt2[1];
		Line_coords[nb_lines * 6 + 2] = pt2[2];
		Line_coords[nb_lines * 6 + 3] = pt2[3];
		Line_coords[nb_lines * 6 + 4] = pt2[4];
		Line_coords[nb_lines * 6 + 5] = pt2[5];
		nb_lines++;
		if (nb_lines >= SCENE_MAX_LINES) {
			cout << "too many lines" << endl;
			exit(1);
		}
#endif
		return nb_lines - 1;
	}
	else {
		return -1;
	}
}



int scene::line_after_recentering(double x1, double x2, double x3,
	double y1, double y2, double y3, double rad)
{
	double x[3], y[3];
	double xx[3], yy[3];
	double pt2[6];
	numerics N;
	int verbose_level = 0;

	x[0] = x1;
	x[1] = x2;
	x[2] = x3;
	y[0] = y1;
	y[1] = y2;
	y[2] = y3;

	N.line_centered(x, y, xx, yy, rad, verbose_level);

	pt2[0] = xx[0];
	pt2[1] = xx[1];
	pt2[2] = xx[2];
	pt2[3] = yy[0];
	pt2[4] = yy[1];
	pt2[5] = yy[2];

	line6(pt2);

#if 0
	Line_coords[nb_lines * 6 + 0] = xx[0];
	Line_coords[nb_lines * 6 + 1] = xx[1];
	Line_coords[nb_lines * 6 + 2] = xx[2];
	Line_coords[nb_lines * 6 + 3] = yy[0];
	Line_coords[nb_lines * 6 + 4] = yy[1];
	Line_coords[nb_lines * 6 + 5] = yy[2];
	points(Line_coords + nb_lines * 6, 2 /* nb_points */);
	nb_lines++;
	if (nb_lines >= SCENE_MAX_LINES) {
		cout << "too many lines" << endl;
		exit(1);
	}
#endif
	return nb_lines - 1;
}

int scene::line_through_two_points(int pt1, int pt2, double rad)
{
	double x[3], y[3];
	double xx[3], yy[3];
	double coords[6];
	numerics N;
	int verbose_level = 0;

	N.vec_copy(Point_coords + pt1 * 3, x, 3);
	N.vec_copy(Point_coords + pt2 * 3, y, 3);

	N.line_centered(x, y, xx, yy, 10., verbose_level);

	coords[0] = xx[0];
	coords[1] = xx[1];
	coords[2] = xx[2];
	coords[3] = yy[0];
	coords[4] = yy[1];
	coords[5] = yy[2];

	line6(coords);

#if 0
	Line_coords[nb_lines * 6 + 0] = xx[0];
	Line_coords[nb_lines * 6 + 1] = xx[1];
	Line_coords[nb_lines * 6 + 2] = xx[2];
	Line_coords[nb_lines * 6 + 3] = yy[0];
	Line_coords[nb_lines * 6 + 4] = yy[1];
	Line_coords[nb_lines * 6 + 5] = yy[2];
	nb_lines++;
	if (nb_lines >= SCENE_MAX_LINES) {
		cout << "too many lines" << endl;
		exit(1);
	}
#endif
	return nb_lines - 1;
}




int scene::plane_through_three_points(int pt1, int pt2, int pt3)
{
	double p1[3], p2[3], p3[3], n[3], d;
	numerics N;

	N.vec_copy(Point_coords + pt1 * 3, p1, 3);
	N.vec_copy(Point_coords + pt2 * 3, p2, 3);
	N.vec_copy(Point_coords + pt3 * 3, p3, 3);

#if 0
	cout << "p1=" << endl;
	print_system(p1, 1, 3);
	cout << endl;
	cout << "p2=" << endl;
	print_system(p2, 1, 3);
	cout << endl;
	cout << "p3=" << endl;
	print_system(p3, 1, 3);
	cout << endl;
#endif

	N.plane_through_three_points(p1, p2, p3, n, d);
	return plane(n[0], n[1], n[2], d);
}

int scene::quadric_through_three_lines(
	int line_idx1, int line_idx2, int line_idx3,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double coeff[10];
	double points[9 * 3];
	double System[9 * 10];
	double v[3];
	double x, y, z;
	int idx;
	int i, k;
	numerics N;

	if (f_v) {
		cout << "scene::quadric_through_three_lines" << endl;
	}
	N.vec_copy(Line_coords + line_idx1 * 6, points + 0, 3);
	N.vec_copy(Line_coords + line_idx1 * 6 + 3, points + 3, 3);
	N.vec_subtract(points + 3, points + 0, v, 3);
	N.vec_scalar_multiple(v, 2., 3);
	N.vec_add(points + 0, v, points + 6, 3);

	N.vec_copy(Line_coords + line_idx2 * 6, points + 9, 3);
	N.vec_copy(Line_coords + line_idx2 * 6 + 3, points + 12, 3);
	N.vec_subtract(points + 12, points + 9, v, 3);
	N.vec_scalar_multiple(v, 2., 3);
	N.vec_add(points + 9, v, points + 15, 3);

	N.vec_copy(Line_coords + line_idx3 * 6, points + 18, 3);
	N.vec_copy(Line_coords + line_idx3 * 6 + 3, points + 21, 3);
	N.vec_subtract(points + 21, points + 18, v, 3);
	N.vec_scalar_multiple(v, 2., 3);
	N.vec_add(points + 18, v, points + 24, 3);


	for (i = 0; i < 9; i++) {
		x = points[i * 3 + 0];
		y = points[i * 3 + 1];
		z = points[i * 3 + 2];
		System[i * 10 + 0] = x * x;
		System[i * 10 + 1] = x * y;
		System[i * 10 + 2] = x * z;
		System[i * 10 + 3] = x;
		System[i * 10 + 4] = y * y;
		System[i * 10 + 5] = y * z;
		System[i * 10 + 6] = y;
		System[i * 10 + 7] = z * z;
		System[i * 10 + 8] = z;
		System[i * 10 + 9] = 1;
	}

	k = N.Null_space(System, 9, 10, coeff, 0 /* verbose_level */);
	if (k != 1) {
		cout << "scene::quadric_through_three_lines k != 1" << endl;
		exit(1);
	}


	idx = quadric(coeff);
	if (f_v) {
		cout << "scene::quadric_through_three_lines done" << endl;
	}
	return idx;
}


int scene::cubic_in_orbiter_ordering(double *coeff)
{
	double eqn[20];

	eqn[0] = coeff[0];
	eqn[1] = coeff[4];
	eqn[2] = coeff[5];
	eqn[3] = coeff[6];
	eqn[4] = coeff[7];
	eqn[5] = coeff[16];
	eqn[6] = coeff[17];
	eqn[7] = coeff[10];
	eqn[8] = coeff[18];
	eqn[9] = coeff[13];
	eqn[10] = coeff[1];
	eqn[11] = coeff[8];
	eqn[12] = coeff[9];
	eqn[13] = coeff[11];
	eqn[14] = coeff[19];
	eqn[15] = coeff[14];
	eqn[16] = coeff[2];
	eqn[17] = coeff[12];
	eqn[18] = coeff[15];
	eqn[19] = coeff[3];
	return cubic(eqn);
}

void scene::deformation_of_cubic_lex(int nb_frames,
		double angle_start, double angle_max, double angle_min,
		double *coeff1, double *coeff2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "scene::deformation_of_cubic_lex" << endl;
	}
	double phi_step, phi, s1, s2, c, theta, t, mu, lambda;
	double coeff3[20];
	int h, nb_frames_half, i;

	nb_frames_half = nb_frames >> 1;
	phi_step = 2 * 2 * M_PI / (nb_frames - 1);
	s1 = (angle_max - angle_start) / 2.;
	s2 = (angle_start - angle_min) / 2.;

	for (h = 0; h < nb_frames; h++) {
		phi = h * phi_step;
		c = 1. - cos(phi);
		if (h < nb_frames_half) {
			theta = angle_start + c * s1;
		}
		else {
			theta = angle_start - c * s2;
		}
		t = tan(theta);
		if (isnan(t) || ABS(t) > 1000000.) {
			mu = 0;
			lambda = 1;
		}
		else if (t > 1.) {
			mu = 1. / t;
			lambda = 1;
		}
		else {
			mu = 1;
			lambda = t;
		}
		for (i = 0; i < 20; i++) {
			coeff3[i] = mu * coeff1[i] + lambda * coeff2[i];
		}
		cubic(coeff3);
	}
	if (f_v) {
		cout << "scene::deformation_of_cubic_lex done" << endl;
	}
}

int scene::cubic_Goursat_ABC(double A, double B, double C)
{
	double coeffs[20];
	int i;

	for (i = 0; i < 20; i++) {
		coeffs[i] = 0;
	}
	coeffs[5] = A; // xyz
	coeffs[3] = B; // x^2
	coeffs[12] = B; // y^2
	coeffs[17] = B; // z^2
	coeffs[19] = C; // 1
	return cubic(coeffs);
}


int scene::line_extended(
	double x1, double x2, double x3,
	double y1, double y2, double y3, double r)
{
	//double d1, d2;
	double v[3];
	double a, b, c, av, d, e;
	double lambda1, lambda2;

	//d1 = ::distance_from_origin(x1, x2, x3);
	//d2 = ::distance_from_origin(y1, y2, y3);
	v[0] = y1 - x1;
	v[1] = y2 - x2;
	v[2] = y3 - x3;
	// solve
	// (x1+\lambda*v[0])^2 + (x2+\lambda*v[1])^2 + (x3+\lambda*v[2])^2 = r^2
	// which gives
	// (v[0]^2+v[1]^2+v[2]^2) * \lambda^2 +
	// (2*x1*v[0] + 2*x2*v[1] + 2*x3*v[2]) * \lambda +
	// x1^2 + x2^2 + x3^2 - r^2 = 0
	a = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
	b = 2 * (x1 * v[0] + x2 * v[1] + x3 * v[2]);
	c = x1 * x1 + x2 * x2 + x3 * x3 - r * r;
	av = 1. / a;
	b = b * av;
	c = c * av;
	d = b * b * 0.25 - c;
	if (d < 0) {
		cout << "line_extended d < 0" << endl;
		return line(x1, x2, x3, y1, y2, y3);
	}
	e = sqrt(d);
	lambda1 = -b * 0.5 + e;
	lambda2 = -b * 0.5 - e;

	double coords[6];

	coords[0] = x1 + lambda1 * v[0];
	coords[1] = x2 + lambda1 * v[1];
	coords[2] = x3 + lambda1 * v[2];
	coords[3] = x1 + lambda2 * v[0];
	coords[4] = x2 + lambda2 * v[1];
	coords[5] = x3 + lambda2 * v[2];

	line6(coords);

#if 0
	Line_coords[nb_lines * 6 + 0] = x1 + lambda1 * v[0];
	Line_coords[nb_lines * 6 + 1] = x2 + lambda1 * v[1];
	Line_coords[nb_lines * 6 + 2] = x3 + lambda1 * v[2];
	Line_coords[nb_lines * 6 + 3] = x1 + lambda2 * v[0];
	Line_coords[nb_lines * 6 + 4] = x2 + lambda2 * v[1];
	Line_coords[nb_lines * 6 + 5] = x3 + lambda2 * v[2];
	nb_lines++;
	if (nb_lines >= SCENE_MAX_LINES) {
		cout << "too many lines" << endl;
		exit(1);
	}
#endif

	return nb_lines - 1;
}




}}}


