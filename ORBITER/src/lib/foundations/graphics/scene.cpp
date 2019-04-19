// scene.C
//
// Anton Betten
//
// started:  February 13, 2018




#include "foundations.h"

using namespace std;



#define EPSILON 0.01

namespace orbiter {
namespace foundations {



scene::scene()
{
	null();
}

scene::~scene()
{
	freeself();
}

void scene::null()
{
	line_radius = 0.05;
	nb_lines = 0;
	Line_coords = NULL;
	nb_edges = 0;
	Edge_points = NULL;
	nb_points = 0;
	Point_coords = NULL;
	nb_planes = 0;
	Plane_coords = NULL;
	nb_quadrics = 0;
	Quadric_coords = NULL;
	nb_cubics = 0;
	Cubic_coords = NULL;
	nb_faces = 0;
	Nb_face_points = NULL;
	Face_points = NULL;

	extra_data = NULL;

	f_has_affine_space = FALSE;
	affine_space_q = 0;
	affine_space_starting_point = 0;
}

void scene::freeself()
{
	if (Line_coords) {
		delete [] Line_coords;
		}
	if (Edge_points) {
		FREE_int(Edge_points);
		}
	if (Point_coords) {
		delete [] Point_coords;
		}
	if (Plane_coords) {
		delete [] Plane_coords;
		}
	if (Quadric_coords) {
		delete [] Quadric_coords;
		}
	if (Cubic_coords) {
		delete [] Cubic_coords;
		}
	if (Nb_face_points) {
		FREE_int(Nb_face_points);
		}
	if (Face_points) {
		int i;
		for (i = 0; i < nb_faces; i++) {
			FREE_int(Face_points[i]);
			}
		FREE_pint(Face_points);
		}
	null();
}

void scene::init(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "scene::init" << endl;
		}
	Line_coords = new double [SCENE_MAX_LINES * 6];
	nb_lines = 0;
	Edge_points = NEW_int(SCENE_MAX_EDGES * 2);
	nb_edges = 0;
	Point_coords = new double [SCENE_MAX_POINTS * 3];
	nb_points = 0;
	Plane_coords = new double [SCENE_MAX_PLANES * 4];
	nb_planes = 0;
	Quadric_coords = new double [SCENE_MAX_QUADRICS * 10];
	nb_quadrics = 0;
	Cubic_coords = new double [SCENE_MAX_CUBICS * 20];
	nb_cubics = 0;
	Face_points = NEW_pint(SCENE_MAX_FACES);
	Nb_face_points = NEW_int(SCENE_MAX_FACES);
	nb_faces = 0;
	if (f_v) {
		cout << "scene::done" << endl;
		}
}

scene *scene::transformed_copy(double *A4, double *A4_inv, 
	double rad, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	scene *S;
	
	if (f_v) {
		cout << "scene::transformed_copy" << endl;
		}

	S = NEW_OBJECT(scene);
	S->init(verbose_level);
	
	transform_lines(S, A4, A4_inv, rad, verbose_level);
	copy_edges(S, A4, A4_inv, verbose_level);
	transform_points(S, A4, A4_inv, verbose_level);
	transform_planes(S, A4, A4_inv, verbose_level);
	transform_quadrics(S, A4, A4_inv, verbose_level);
	transform_cubics(S, A4, A4_inv, verbose_level);
	copy_faces(S, A4, A4_inv, verbose_level);

	if (f_v) {
		cout << "scene::transformed_copy done" << endl;
		}
	return S;
}

void scene::print()
{
	int i;
	numerics N;
	
	cout << "scene:" << endl;
	cout << "nb_lines=" << nb_lines << endl;
	for (i = 0; i < nb_lines; i++) {
		cout << i << " / " << nb_lines << " : ";
		N.vec_print(Line_coords + i * 6 + 0, 3);
		cout << ", ";
		N.vec_print(Line_coords + i * 6 + 3, 3);
		cout << endl;
		}
	cout << "nb_planes=" << nb_planes << endl;
	for (i = 0; i < nb_planes; i++) {
		cout << i << " / " << nb_planes << " : ";
		N.vec_print(Plane_coords + i * 4, 4);
		cout << endl;
		}
	cout << "nb_points=" << nb_points << endl;
	for (i = 0; i < nb_points; i++) {
		cout << i << " / " << nb_points << " : ";
		N.vec_print(Point_coords + i * 3, 3);
		cout << endl;
		}
	cout << "nb_cubics=" << nb_cubics << endl;
	for (i = 0; i < nb_cubics; i++) {
		cout << i << " / " << nb_cubics << " : ";
		N.vec_print(Cubic_coords + i * 20, 20);
		cout << endl;
		}

	int k;
	
	cout << "nb_edges=" << nb_edges << endl;
	for (k = 0; k < nb_edges; k++) {
		cout << k << " / " << nb_edges << " : ";
		int_vec_print(cout, Edge_points + k * 2, 2);
		cout << endl;
		}
}

void scene::transform_lines(scene *S, 
	double *A4, double *A4_inv, 
	double rad, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double x[4], y[4];
	double xx[4], yy[4];
	double xxx[4], yyy[4];
	int i;
	numerics N;
	
	
	if (f_v) {
		cout << "scene::transform_lines" << endl;
		}

	for (i = 0; i < nb_lines; i++) {
		N.vec_copy(Line_coords + i * 6 + 0, x, 3);
		x[3] = 1.;
		N.vec_copy(Line_coords + i * 6 + 3, y, 3);
		y[3] = 1.;
		N.mult_matrix_4x4(x, A4, xx);
		N.mult_matrix_4x4(y, A4, yy);
		if (ABS(xx[3]) < 0.0001) {
			cout << "scene::transform_lines warning, "
					"point x is moved to infinity" << endl;
			exit(1);
			}
		if (ABS(yy[3]) < 0.0001) {
			cout << "scene::transform_lines warning, "
					"point y is moved to infinity" << endl;
			exit(1);
			}
		N.vec_normalize_from_back(xx, 4);
		if (ABS(xx[3] - 1.) > 0.0001) {
			cout << "scene::transform_lines warning, "
					"point xx is not an affine point" << endl;
			exit(1);
			}
		N.vec_normalize_from_back(yy, 4);
		if (ABS(yy[3] - 1.) > 0.0001) {
			cout << "scene::transform_lines warning, "
					"point yy is not an affine point" << endl;
			exit(1);
			}
		if (N.line_centered(xx, yy, xxx, yyy, 10.)) {
			S->line(xxx[0], xxx[1], xxx[2], yyy[0], yyy[1], yyy[2]);
			}
		}

	if (f_v) {
		cout << "scene::transform_lines done" << endl;
		}

}

void scene::copy_edges(scene *S, double *A4, double *A4_inv, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b;
	int i;
	
	
	if (f_v) {
		cout << "scene::copy_edges" << endl;
		}

	for (i = 0; i < nb_edges; i++) {
		a = Edge_points[i * 2 + 0];
		b = Edge_points[i * 2 + 1];
		S->edge(a, b);
		}
	if (f_v) {
		cout << "scene::copy_edges done" << endl;
		}
}

void scene::transform_points(scene *S, double *A4, double *A4_inv, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double x[4];
	double xx[4];
	int i;
	numerics N;
	
	
	if (f_v) {
		cout << "scene::transform_points" << endl;
		}

	for (i = 0; i < nb_points; i++) {
		N.vec_copy(Point_coords + i * 3, x, 3);
		x[3] = 1.;
		N.mult_matrix_4x4(x, A4, xx);
		if (ABS(xx[3]) < 0.0001) {
			cout << "scene::transform_lines warning, "
					"point x is moved to infinity" << endl;
			exit(1);
			}
		N.vec_normalize_from_back(xx, 4);
		if (ABS(xx[3] - 1.) > 0.0001) {
			cout << "scene::transform_lines warning, "
					"point xx is not an affine point" << endl;
			exit(1);
			}
		S->point(xx[0], xx[1], xx[2]);
		}

	if (f_v) {
		cout << "scene::transform_points done" << endl;
		}

}

void scene::transform_planes(scene *S, double *A4, double *A4_inv, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double x[4];
	double xx[4];
	double Avt[16];
	int i;
	numerics N;
	
	
	if (f_v) {
		cout << "scene::transform_planes" << endl;
		}

	N.transpose_matrix_4x4(A4_inv, Avt);
	

	for (i = 0; i < nb_planes; i++) {
		N.vec_copy(Plane_coords + i * 4, x, 4);
		x[3] *= -1.;

		N.mult_matrix_4x4(x, Avt, xx);

		xx[3] *= -1.;
		S->plane(xx[0], xx[1], xx[2], xx[3]);
		}

	if (f_v) {
		cout << "scene::transform_planes done" << endl;
		}

}

void scene::transform_quadrics(scene *S, double *A4, double *A4_inv, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double coeff_in[10];
	double coeff_out[10];
	int i;
	numerics N;
	
	
	if (f_v) {
		cout << "scene::transform_quadrics" << endl;
		}
	

	for (i = 0; i < nb_quadrics; i++) {
		N.vec_copy(Quadric_coords + i * 10, coeff_in, 10);

		N.substitute_quadric_linear(coeff_in, coeff_out,
			A4_inv, verbose_level);

		S->quadric(coeff_out);
		}

	if (f_v) {
		cout << "scene::transform_quadrics done" << endl;
		}

}

void scene::transform_cubics(scene *S, double *A4, double *A4_inv, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double coeff_in[20];
	double coeff_out[20];
	int i;
	numerics N;
	
	
	if (f_v) {
		cout << "scene::transform_cubics" << endl;
		}
	

	for (i = 0; i < nb_cubics; i++) {
		N.vec_copy(Cubic_coords + i * 20, coeff_in, 20);

		N.substitute_cubic_linear(coeff_in, coeff_out,
			A4_inv, verbose_level);

		S->cubic(coeff_out);
		}

	if (f_v) {
		cout << "scene::transform_cubics done" << endl;
		}

}

void scene::copy_faces(scene *S, double *A4, double *A4_inv, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_pts, *pts;
	int i;
	
	
	if (f_v) {
		cout << "scene::copy_faces" << endl;
		}

	for (i = 0; i < nb_faces; i++) {
		pts = Face_points[i];
		nb_pts = Nb_face_points[i];
		S->face(pts, nb_pts);
		}
	if (f_v) {
		cout << "scene::copy_faces done" << endl;
		}
}


int scene::line_pt_and_dir(double *x6, double rad)
{
	double pt[6];
	double pt2[6];
	numerics N;

	pt[0] = x6[0];
	pt[1] = x6[1];
	pt[2] = x6[2];
	pt[3] = x6[0] + x6[3];
	pt[4] = x6[1] + x6[4];
	pt[5] = x6[2] + x6[5];
	if (N.line_centered(pt, pt + 3,
		pt2, pt2 + 3, 
		rad)) {
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
		return TRUE;
		}
	else {
		return FALSE;
		}
}

int scene::line_through_two_pts(double *x6, double rad)
{
	double pt[6];
	double pt2[6];
	numerics N;

	pt[0] = x6[0];
	pt[1] = x6[1];
	pt[2] = x6[2];
	pt[3] = x6[3];
	pt[4] = x6[4];
	pt[5] = x6[5];
	if (N.line_centered(pt, pt + 3,
		pt2, pt2 + 3,
		rad)) {
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
		return TRUE;
		}
	else {
		return FALSE;
		}
}

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
	return nb_lines - 1;
}

int scene::line_through_two_points(int pt1, int pt2, double rad)
{
	double x[3], y[3];
	double xx[3], yy[3];
	numerics N;

	N.vec_copy(Point_coords + pt1 * 3, x, 3);
	N.vec_copy(Point_coords + pt2 * 3, y, 3);

	N.line_centered(x, y, xx, yy, 10.);

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
	return nb_lines - 1;
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

void scene::points(double *Coords, int nb_points)
{
	int i;
	
	for (i = 0; i < nb_points; i++) {
		point(Coords[i * 3 + 0], Coords[i * 3 + 1], Coords[i * 3 + 2]);
		}
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

int scene::point_center_of_mass_of_face(int face_idx)
{
	return point_center_of_mass(Face_points[face_idx], Nb_face_points[face_idx]);
}

int scene::point_center_of_mass_of_edge(int edge_idx)
{
	return point_center_of_mass(Edge_points + edge_idx * 2, 2);
}

int scene::point_center_of_mass(int *Pt_idx, int nb_pts)
{
	double x[3];
	numerics N;

	N.center_of_mass(Point_coords, 3 /* len */, Pt_idx, nb_pts, x);
	return point(x[0], x[1], x[2]);
}

int scene::triangle(int line1, int line2, int line3, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int pt[3], idx;

	if (f_v) {
		cout << "scene::triangle" << endl;
		}
	pt[0] = point_as_intersection_of_two_lines(line1, line2);
	pt[1] = point_as_intersection_of_two_lines(line2, line3);
	pt[2] = point_as_intersection_of_two_lines(line3, line1);
	idx = face3(pt[0], pt[1], pt[2]);
	if (f_v) {
		cout << "scene::triangle done" << endl;
		}
	return idx;
}

int scene::point_as_intersection_of_two_lines(int line1, int line2)
{
	double x[3], y[3], z[3], u[3], v[3];
	double System[9];
	int ker, idx, i;
	//int base_cols[3];
	double lambda;
	double Ker[9];
	numerics N;

	N.vec_copy(Line_coords + line1 * 6, x, 3);
	N.vec_copy(Line_coords + line2 * 6, y, 3);
	N.vec_subtract(Line_coords + line1 * 6 + 3, Line_coords + line1 * 6, u, 3);
	N.vec_subtract(Line_coords + line2 * 6 + 3, Line_coords + line2 * 6, v, 3);

	// we need to solve 
	// x + lambda * u = y + mu * v
	// so
	// lambda * u + mu * (- v) = y - x

	for (i = 0; i < 3; i++) {
		System[i * 3 + 0] = u[i];
		System[i * 3 + 1] = -1. * v[i];
		System[i * 3 + 2] = y[i] - x[i];
		}
	// rk = Gauss_elimination(System, 3, 3,
	//base_cols, TRUE /* f_complete */, 0 /* verbose_level */);
	ker = N.Null_space(System, 3, 3, Ker, 0 /* verbose_level */);
	if (ker != 1) {
		cout << "scene::point_as_intersection_of_two_lines ker != 1" << endl;
		exit(1);
		}
	N.vec_normalize_from_back(Ker, 3);
	N.vec_normalize_to_minus_one_from_back(Ker, 3);
	lambda = Ker[0];
	for (i = 0; i < 3; i++) {
		z[i] = x[i] + lambda * u[i];
		}
	idx = point(z[0], z[1], z[2]);
	return idx;
}

int scene::plane_from_dual_coordinates(double *x4)
{
	double y[4];
	double d, dv;
	d = sqrt(x4[0] * x4[0] + x4[1] * x4[1] + x4[2] * x4[2]);
	dv = 1. / d;
	y[0] = x4[0] * dv;
	y[1] = x4[1] * dv;
	y[2] = x4[2] * dv;
	y[3] = - x4[3] * dv;
	return plane(y[0], y[1], y[2], y[3]);
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

int scene::quadric(double *coeff)
// povray ordering of monomials:
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
	
	for (i = 0; i < 10; i++) {
		Quadric_coords[nb_quadrics * 10 + i] = coeff[i];
		}
	nb_quadrics++;
	if (nb_quadrics >= SCENE_MAX_QUADRICS) {
		cout << "too many quadrics" << endl;
		exit(1);
		}
	return nb_quadrics - 1;
}



int scene::cubic(double *coeff)
// povray ordering of monomials:
// http://www.povray.org/documentation/view/3.6.1/298/
// 1: x^3
// 2: x^2y
// 3: x^2z
// 4: x^2
// 5: xy^2
// 6: xyz
// 7: xy
// 8: xz^2
// 9: xz
// 10: x
// 11: y^3
// 12: y^2z
// 13: y^2
// 14: yz^2
// 15: yz
// 16: y
// 17: z^3
// 18: z^2
// 19: z
// 20: 1
{
	int i;
	
	for (i = 0; i < 20; i++) {
		Cubic_coords[nb_cubics * 20 + i] = coeff[i];
		}
	nb_cubics++;
	if (nb_cubics >= SCENE_MAX_CUBICS) {
		cout << "too many cubics" << endl;
		exit(1);
		}
	return nb_cubics - 1;
}

int scene::face(int *pts, int nb_pts)
{
	Face_points[nb_faces] = NEW_int(nb_pts);
	Nb_face_points[nb_faces] = nb_pts;
	int_vec_copy(pts, Face_points[nb_faces], nb_pts);
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



void scene::draw_lines_with_selection(int *selection, int nb_select, 
	const char *options, ostream &ost)
{
	int i, j, h, s;
	numerics N;
		
	ost << endl;
	ost << "	union{ // lines" << endl;
	ost << endl;
	ost << "	        #declare r=" << line_radius << "; " << endl;
	//ost << "                #declare b=4;" << endl;
	ost << endl;
	for (i = 0; i < nb_select; i++) {
		s = selection[i];
		j = s;
		ost << "		cylinder{<";
		for (h = 0; h < 3; h++) {
			N.output_double(Line_coords[j * 6 + h], ost);
			if (h < 2) {
				ost << ", ";
				}
			}
		ost << ">,<";
		for (h = 0; h < 3; h++) {
			N.output_double(Line_coords[j * 6 + 3 + h], ost);
			if (h < 2) {
				ost << ", ";
				}
			}
		ost << ">, r } // line " << j << endl;
		}
	ost << endl;
	ost << "		" << options << "" << endl;
	//ost << "		pigment{" << color << "}" << endl;
	ost << "	}" << endl;
}

void scene::draw_line_with_selection(int line_idx, 
	const char *options, ostream &ost)
{
	int j, h, s;
	numerics N;
		
	ost << endl;
	ost << "	union{ // lines" << endl;
	ost << endl;
	ost << "	        #declare r=" << line_radius << "; " << endl;
	//ost << "                #declare b=4;" << endl;
	ost << endl;
	s = line_idx;
	j = s;
	ost << "		cylinder{<";
	for (h = 0; h < 3; h++) {
		N.output_double(Line_coords[j * 6 + h], ost);
		if (h < 2) {
			ost << ", ";
			}
		}
	ost << ">,<";
	for (h = 0; h < 3; h++) {
		N.output_double(Line_coords[j * 6 + 3 + h], ost);
		if (h < 2) {
			ost << ", ";
			}
		}
	ost << ">, r } // line " << j << endl;
	ost << endl;
	ost << "		" << options << "" << endl;
	//ost << "		pigment{" << color << "}" << endl;
	ost << "	}" << endl;
}

void scene::draw_lines_cij_with_selection(int *selection, int nb_select, 
	ostream &ost)
{
	int i, j, h, s;
	numerics N;
		
	ost << endl;
	ost << "	union{ // cij lines" << endl;
	ost << endl;
	ost << "	        #declare r=0.04 ; " << endl;
	//ost << "                #declare b=4;" << endl;
	ost << endl;
	for (i = 0; i < nb_select; i++) {
		s = selection[i];
		j = 12 + s;
		ost << "		cylinder{<";
		for (h = 0; h < 3; h++) {
			N.output_double(Line_coords[j * 6 + h], ost);
			if (h < 2) {
				ost << ", ";
				}
			}	
		ost << ">,<";
		for (h = 0; h < 3; h++) {
			N.output_double(Line_coords[j * 6 + 3 + h], ost);
			if (h < 2) {
				ost << ", ";
				}
			}	
		ost << ">, r } // line " << j << endl;
		}
	ost << endl;
	ost << "		pigment{Yellow}" << endl;
	ost << "	}" << endl;
}

void scene::draw_lines_cij(ostream &ost)
{
	int selection[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14};
	
	draw_lines_cij_with_selection(selection, 15, ost);
} 

void scene::draw_lines_ai_with_selection(int *selection, int nb_select, 
	ostream &ost)
{
	int s, i, j, h;
	numerics N;
		
	ost << endl;
	ost << "	union{ // ai lines" << endl;
	ost << endl;
	ost << "	        #declare r=0.04 ; " << endl;
	//ost << "                #declare b=4;" << endl;
	ost << endl;
	for (i = 0; i < nb_select; i++) {
		s = selection[i];
		j = 0 + s;
		ost << "		cylinder{<";
		for (h = 0; h < 3; h++) {
			N.output_double(Line_coords[j * 6 + h], ost);
			if (h < 2) {
				ost << ", ";
				}
			}	
		ost << ">,<";
		for (h = 0; h < 3; h++) {
			N.output_double(Line_coords[j * 6 + 3 + h], ost);
			if (h < 2) {
				ost << ", ";
				}
			}	
		ost << ">, r } // line " << j << endl;
		}
	ost << endl;
	ost << "		pigment{Red}" << endl;
	ost << "	}" << endl;
}

void scene::draw_lines_ai(ostream &ost)
{
	int selection[] = {0,1,2,3,4,5};
	
	draw_lines_ai_with_selection(selection, 6, ost);
} 

void scene::draw_lines_bj_with_selection(int *selection, int nb_select, 
	ostream &ost)
{
	int s, i, j, h;
	numerics N;
		
	ost << endl;
	ost << "	union{ // bj lines" << endl;
	ost << endl;
	ost << "	        #declare r=0.04 ; " << endl;
	//ost << "                #declare b=4;" << endl;
	ost << endl;
	for (i = 0; i < nb_select; i++) {
		s = selection[i];
		j = 6 + s;
		ost << "		cylinder{<";
		for (h = 0; h < 3; h++) {
			N.output_double(Line_coords[j * 6 + h], ost);
			if (h < 2) {
				ost << ", ";
				}
			}	
		ost << ">,<";
		for (h = 0; h < 3; h++) {
			N.output_double(Line_coords[j * 6 + 3 + h], ost);
			if (h < 2) {
				ost << ", ";
				}
			}	
		ost << ">, sr } // line " << j << endl;
		}
	ost << endl;
	ost << "		pigment{Blue}" << endl;
	ost << "	}" << endl;
}

void scene::draw_lines_bj(ostream &ost)
{
	int selection[] = {0,1,2,3,4,5};
	
	draw_lines_bj_with_selection(selection, 6, ost);
} 



void scene::draw_edges_with_selection(int *selection, int nb_select, 
	const char *options, ostream &ost)
{
	int s, i, j, h, pt1, pt2;
	numerics N;
		
	ost << endl;
	ost << "	union{ // edges" << endl;
	ost << endl;
	ost << "	        #declare r=" << line_radius << " ; " << endl;
	ost << endl;
	for (i = 0; i < nb_select; i++) {
		s = selection[i];
		j = s;
		pt1 = Edge_points[j * 2 + 0];
		pt2 = Edge_points[j * 2 + 1];
		ost << "		cylinder{<";
		for (h = 0; h < 3; h++) {
			N.output_double(Point_coords[pt1 * 3 + h], ost);
			if (h < 2) {
				ost << ", ";
				}
			}	
		ost << ">,<";
		for (h = 0; h < 3; h++) {
			N.output_double(Point_coords[pt2 * 3 + h], ost);
			if (h < 2) {
				ost << ", ";
				}
			}	
		ost << ">, r } // line " << j << endl;
		}
	ost << endl;
	ost << "		" << options << " " << endl;
	//ost << "		pigment{" << color << "}" << endl;
	ost << "	}" << endl;
}

void scene::draw_faces_with_selection(int *selection, int nb_select, 
	double thickness_half, const char *options, ostream &ost)
{
	int s, i, j;
		
	ost << endl;
	ost << "	union{ // faces" << endl;
	ost << endl;
	//ost << "	        #declare r=" << rad << " ; " << endl;
	//ost << endl;
	for (i = 0; i < nb_select; i++) {
		s = selection[i];
		j = s;
		draw_face(j, thickness_half, options, ost);
		}
	ost << endl;
	//ost << "		pigment{" << color << "}" << endl;
	ost << "	}" << endl;
}


void scene::draw_face(int idx, double thickness_half, const char *options, 
	ostream &ost)
{
	int f_v = FALSE;
	int *pts;
	int nb_pts, i;
	double *Pts_in;
	double *Pts_out;
	double abc3[3];
	double angles3[3];
	double T3[3];
	numerics N;

	nb_pts = Nb_face_points[idx];
	pts = Face_points[idx];
	Pts_in = new double [nb_pts * 3];
	Pts_out = new double [nb_pts * 2];
	if (f_v) {
		cout << "scene::draw_face" << endl;
	}
	for (i = 0; i < nb_pts; i++) {
		N.vec_copy(Point_coords + pts[i] * 3, Pts_in + i * 3, 3);
		if (f_v) {
			cout << "vertex i= " << i << " pts[i] = " << pts[i]
					<< " x=" << Pts_in[i * 3 + 0]
					<< " y=" << Pts_in[i * 3 + 1]
					<< " z=" << Pts_in[i * 3 + 2]
					<< endl;
		}
	}

#if 0
	N.triangular_prism(P1, P2, P3,
		abc3, angles3, T3, 
		0 /*verbose_level*/);
#else
	N.general_prism(Pts_in, nb_pts, Pts_out,
		abc3, angles3, T3, 
		0 /*verbose_level*/);
#endif



	//double thickness_half = 0.001;

	ost << "\t\tprism { ";
	N.output_double(-thickness_half, ost);
	ost << ", ";
	N.output_double(thickness_half, ost);
	ost << ", " << nb_pts + 1 << endl;
	ost << "\t\t\t< 0, 0 >," << endl;
	ost << "\t\t\t< ";
	N.output_double(abc3[0], ost);
	ost << ", " << 0 << " >," << endl;
	ost << "\t\t\t< ";
	N.output_double(abc3[1], ost);
	ost << ", ";
	N.output_double(abc3[2], ost);
	ost << " >," << endl;
	for (i = 3; i < nb_pts; i++) {
		ost << "\t\t\t< ";
		N.output_double(Pts_out[i * 2 + 0], ost);
		ost << ", ";
		N.output_double(Pts_out[i * 2 + 1], ost);
		ost << " >," << endl;
		}
	ost << "\t\t\t< 0, 0 >" << endl;
	ost << "\t\t\t " << options << " " << endl;
	//ost << "\t\t\ttexture{ " << endl;
	//ost << "\t\t\t\tpigment{color " << color << " }" << endl;
	//ost << "\t\t\t\tfinish {ambient 0.4 diffuse 0.5 "
	//"roughness 0.001 reflection 0.1 specular .8}" << endl;
	//ost << "\t\t\t}" << endl;
	//ost << "rotate<" << -90 << ",0,0>" << endl;
	ost << "\t\trotate<";
	N.output_double(N.rad2deg(angles3[0]), ost);
	ost << ",0,0>" << endl;
	ost << "\t\trotate<0, ";
	N.output_double(N.rad2deg(angles3[1]), ost);
	ost << ",0>" << endl;
	ost << "\t\trotate<0,0, ";
	N.output_double(N.rad2deg(angles3[2]), ost);
	ost << ">" << endl;
	ost << "\t\ttranslate<";
	N.output_double(T3[0], ost);
	ost << ", ";
	N.output_double(T3[1], ost);
	ost << ", ";
	N.output_double(T3[2], ost);
	ost << ">" << endl;
	ost << "\t\t}" << endl;

	delete [] Pts_in;
	delete [] Pts_out;
}

void scene::draw_text(const char *text,
		double thickness_half, double extra_spacing,
		double scale, 
		double off_x, double off_y, double off_z, 
		const char *color_options, 
		double x, double y, double z, 
		double up_x, double up_y, double up_z, 
		double view_x, double view_y, double view_z, 
		ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double P1[3];
	double P2[3];
	double P3[3];
	double abc3[3];
	double angles3[3];
	double T3[3];
	double u[3];
	numerics N;

	if (f_v) {
		cout << "scene::draw_text" << endl;
		}
	if (f_v) {
		cout << "x,y,z=" << x << ", " << y << " , " << z << endl;
		}
	if (f_v) {
		cout << "view_x,view_y,view_z=" << view_x << ", "
				<< view_y << " , " << view_z << endl;
		}
	if (f_v) {
		cout << "up_x,up_y,up_z=" << up_x << ", " << up_y
				<< " , " << up_z << endl;
		}
	u[0] = view_y * up_z - view_z * up_y;
	u[1] = -1 *(view_x * up_z - up_x * view_z);
	u[2] = view_x * up_y - up_x * view_y;
	if (f_v) {
		cout << "u=" << u[0] << ", " << u[1] << " , " << u[2] << endl;
		}
	P1[0] = x;
	P1[1] = y;
	P1[2] = z;
	P2[0] = x + u[0];
	P2[1] = y + u[1];
	P2[2] = z + u[2];
	P3[0] = x + up_x;
	P3[1] = y + up_y;
	P3[2] = z + up_z;
	
	N.triangular_prism(P1, P2, P3,
		abc3, angles3, T3, 
		verbose_level);
	double offset[3];
	double up[3];
	double view[3];
	up[0] = up_x;
	up[1] = up_y;
	up[2] = up_z;
	view[0] = view_x;
	view[1] = view_y;
	view[2] = view_z;
	N.make_unit_vector(u, 3);
	N.make_unit_vector(up, 3);
	N.make_unit_vector(view, 3);
	if (f_v) {
		cout << "up normalized: ";
		N.vec_print(up, 3);
		cout << endl;
		cout << "u normalized: ";
		N.vec_print(u, 3);
		cout << endl;
		cout << "view normalized: ";
		N.vec_print(view, 3);
		cout << endl;
		}
	
	offset[0] = off_x * u[0] + off_y * up[0] + off_z * view[0]; 
	offset[1] = off_x * u[1] + off_y * up[1] + off_z * view[1]; 
	offset[2] = off_x * u[2] + off_y * up[2] + off_z * view[2]; 

	if (f_v) {
		cout << "offset: ";
		N.vec_print(offset, 3);
		cout << endl;
		}

	ost << "\t\ttext {" << endl;
		ost << "\t\tttf \"timrom.ttf\" \"" << text << "\" "
				<< thickness_half << ", " << extra_spacing << " " << endl;
		ost << "\t\t" << color_options << endl;
		ost << "\t\tscale " << scale << endl;
		ost << "\t\trotate<0,180,0>" << endl;
		ost << "\t\trotate<90,0,0>" << endl;
		ost << "\t\trotate<";
		N.output_double(N.rad2deg(angles3[0]), ost);
		ost << ",0,0>" << endl;
		ost << "\t\trotate<0, ";
		N.output_double(N.rad2deg(angles3[1]), ost);
		ost << ",0>" << endl;
		ost << "\t\trotate<0,0, ";
		N.output_double(N.rad2deg(angles3[2]), ost);
		ost << ">" << endl;
		ost << "\t\ttranslate<";
		N.output_double(T3[0] + offset[0], ost);
		ost << ", ";
		N.output_double(T3[1] + offset[1], ost);
		ost << ", ";
		N.output_double(T3[2] + offset[2], ost);
		ost << ">" << endl;
	ost << "\t\t}" << endl;
		//pigment { BrightGold }
		//finish { reflection .25 specular 1 }
		//translate <0,0,0>
	if (f_v) {
		cout << "scene::draw_text done" << endl;
		}
}



void scene::draw_planes_with_selection(
	int *selection, int nb_select,
	const char *options, ostream &ost)
// for instance color = "Orange transmit 0.5 "
{
	int i, j, h, s;
	numerics N;
		
	ost << endl;
	ost << "	union{ // planes" << endl;
	ost << endl;
	ost << endl;
	for (i = 0; i < nb_select; i++) {
		s = selection[i];
		j = s;
		ost << "		plane{<";
		for (h = 0; h < 3; h++) {
			N.output_double(Plane_coords[j * 4 + h], ost);
			if (h < 2) {
				ost << ", ";
				}
			}	
		ost << ">, ";
		N.output_double(Plane_coords[j * 4 + 3], ost);
		ost << "}" << endl;
		}
	ost << endl;
	ost << "		 " << options << " " << endl;
	//ost << "		texture{ pigment{ color " << color << "}}" << endl;
	ost << "	}" << endl;
}

void scene::draw_points_with_selection(
	int *selection, int nb_select,
	double rad, const char *options, ostream &ost)
// rad = 0.06 works
{
	int i, j, h, s;
	numerics N;
		
	ost << endl;
	ost << "	union{ // points" << endl;
	ost << endl;
        ost << "		#declare r=" << rad << ";" << endl;
	ost << endl;
	for (i = 0; i < nb_select; i++) {
		s = selection[i];
		j = s;
		ost << "		sphere{<";
		for (h = 0; h < 3; h++) {
			N.output_double(Point_coords[j * 3 + h], ost);
			if (h < 2) {
				ost << ", ";
				}
			}	
		ost << ">, r }" << endl;
		}
	//ost << endl;
	ost << "		" << options << " " << endl;
	//ost << "		pigment{" << color << "}" << endl;
	//ost << "		pigment{Cyan*1.3}" << endl;
	//ost << "		finish {ambient 0.4 diffuse 0.6 roughness 0.001 "
	//"reflection 0 specular .8} " << endl;
	ost << "	}" << endl;
}

void scene::draw_cubic_with_selection(int *selection, int nb_select, 
	const char *options, ostream &ost)
{
	int i, j, h, s;
	numerics N;
		
	ost << endl;
	ost << "	union{ // cubics" << endl;
	ost << endl;
	for (i = 0; i < nb_select; i++) {
		s = selection[i];
		j = s;
		ost << "		poly{3, <";

		for (h = 0; h < 20; h++) {
			N.output_double(Cubic_coords[j * 20 + h], ost);
			if (h < 20 - 1) {
				ost << ", ";
				}
			}	
		ost << ">";
		}
	ost << endl;
	ost << "		" << options << " " << endl;
	//ost << "		pigment{" << color << "}" << endl;
	//ost << "		pigment{Cyan*1.3}" << endl;
	//ost << "		finish {ambient 0.4 diffuse 0.5 roughness 0.001 "
	//"reflection 0.1 specular .8} " << endl;
	ost << "		}" << endl;
	ost << "	}" << endl;
}

void scene::draw_quadric_with_selection(int *selection, int nb_select, 
	const char *options, ostream &ost)
{
	int i, j, h, s;
	numerics N;
		
	ost << endl;
	ost << "	union{ // quadrics" << endl;
	ost << endl;
	for (i = 0; i < nb_select; i++) {
		s = selection[i];
		j = s;
		ost << "		poly{2, <";

		for (h = 0; h < 10; h++) {
			N.output_double(Quadric_coords[j * 10 + h], ost);
			if (h < 10 - 1) {
				ost << ", ";
				}
			}	
		ost << ">}" << endl;
		}
	ost << endl;
	ost << "		" << options << " " << endl;
	//ost << "		texture{ pigment{" << color << "}}" << endl;
	//ost << "		pigment{Cyan*1.3}" << endl;
	//ost << "		finish { phong albedo 0.9 phong_size 60 ambient 0.4 "
	//"diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} " << endl;
	ost << "	}" << endl;
}

void scene::draw_quadric_clipped_by_plane(int quadric_idx, int plane_idx,
	const char *options, ostream &ost)
{
	int h;
	numerics N;

	ost << endl;
	ost << "	object{ // quadric clipped by plane" << endl;
	ost << endl;
	ost << "		poly{2, <";
	for (h = 0; h < 10; h++) {
		N.output_double(Quadric_coords[quadric_idx * 10 + h], ost);
		if (h < 10 - 1) {
			ost << ", ";
			}
		}
	ost << ">}" << endl;
	ost << "		clipped_by{ plane{<";
	N.output_double(Plane_coords[plane_idx * 4 + 0], ost);
	ost << ",";
	N.output_double(Plane_coords[plane_idx * 4 + 1], ost);
	ost << ",";
	N.output_double(Plane_coords[plane_idx * 4 + 2], ost);
	ost << ">,";
	N.output_double(Plane_coords[plane_idx * 4 + 3], ost);
	ost << "}";
	ost << "}" << endl;


	ost << endl;
	ost << "		" << options << " " << endl;
	//ost << "		texture{ pigment{" << color << "}}" << endl;
	//ost << "		pigment{Cyan*1.3}" << endl;
	//ost << "		finish { phong albedo 0.9 phong_size 60 ambient 0.4 "
	//"diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} " << endl;
	ost << "	}" << endl;
}


void scene::draw_line_clipped_by_plane(int line_idx, int plane_idx,
		const char *options, ostream &ost)
{
	int h;
	numerics N;

	ost << endl;
	ost << "	object{ // line with clipping" << endl;
	ost << endl;
	ost << "	        #declare r=" << line_radius << "; " << endl;
	//ost << "                #declare b=4;" << endl;
	ost << endl;
	ost << "		cylinder{<";
	for (h = 0; h < 3; h++) {
		N.output_double(Line_coords[line_idx * 6 + h], ost);
		if (h < 2) {
			ost << ", ";
			}
		}
	ost << ">,<";
	for (h = 0; h < 3; h++) {
		N.output_double(Line_coords[line_idx * 6 + 3 + h], ost);
		if (h < 2) {
			ost << ", ";
			}
		}
	ost << ">, r }" << endl;
	ost << "		clipped_by{ plane{<";
	N.output_double(Plane_coords[plane_idx * 4 + 0], ost);
	ost << ",";
	N.output_double(Plane_coords[plane_idx * 4 + 1], ost);
	ost << ",";
	N.output_double(Plane_coords[plane_idx * 4 + 2], ost);
	ost << ">,";
	N.output_double(Plane_coords[plane_idx * 4 + 3], ost);
	ost << "}";
	ost << "}" << endl;
	ost << endl;
	ost << "		" << options << "" << endl;
	//ost << "		pigment{" << color << "}" << endl;
	ost << "	}" << endl;
}





int scene::intersect_line_and_plane(int line_idx, int plane_idx, 
	int &intersection_point_idx, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; // (verbose_level >= 2);
	double B[9];
	double M[12];
	int i;
	double a, av;
	double v[3];
	numerics N;

	if (f_v) {
		cout << "scene::intersect_line_and_plane" << endl;
		}


	// equation of the form:
	// P_0 + \lambda * v = Q_0 + \mu * u + \nu * w

	// where P_0 is a point on the line, 
	// Q_0 is a point on the plane,
	// v is a direction vector of the line
	// u, w are vectors which span the plane.

	// M is the matrix whose columns are 
	// v, -u, -w, -P_0 + Q_0

	// point on the line, brought over on the other side, hence the minus:
	M[0 * 4 + 3] = -1. * Line_coords[line_idx * 6 + 0];
	M[1 * 4 + 3] = -1. * Line_coords[line_idx * 6 + 1];
	M[2 * 4 + 3] = -1. * Line_coords[line_idx * 6 + 2];

	// direction vector of the line:
	// we will need v[] later, hence we store this vector
	for (i = 0; i < 3; i++) {
		v[i] = Line_coords[line_idx * 6 + 3 + i] -
				Line_coords[line_idx * 6 + i];
		M[i * 4 + 0] = v[i];
		}
	
	// compute the vectors u and w:
	B[0 * 3 + 0] = Plane_coords[plane_idx * 4 + 0];
	B[0 * 3 + 1] = Plane_coords[plane_idx * 4 + 1];
	B[0 * 3 + 2] = Plane_coords[plane_idx * 4 + 2];
	if (ABS(B[0 * 3 + 0]) > EPSILON) {
		a = 1. / B[0 * 3 + 0];
		for (i = 0; i < 3; i++) {
			B[i] *= a;
			}
		B[1 * 3 + 0] = 0.;
		B[1 * 3 + 1] = -1.;
		B[1 * 3 + 2] = 0.;
		B[2 * 3 + 0] = 0.;
		B[2 * 3 + 1] = 0.;
		B[2 * 3 + 2] = -1.;
		}
	else {
		cout << "scene::intersect_line_and_plane "
				"ABS(B[0 * 3 + 0]) is too small" << endl;
		exit(1);
		}
	// copy u:
	M[0 * 4 + 1] = -1. * B[0 * 3 + 1];
	M[1 * 4 + 1] = -1. * B[1 * 3 + 1];
	M[2 * 4 + 1] = -1. * B[2 * 3 + 1];
	// copy w:
	M[0 * 4 + 2] = -1. * B[0 * 3 + 2];
	M[1 * 4 + 2] = -1. * B[1 * 3 + 2];
	M[2 * 4 + 2] = -1. * B[2 * 3 + 2];

	// find Q_0:

	while (TRUE) {
		B[0] = (double) random_integer(5);
		B[1] = (double) random_integer(5);
		B[2] = (double) random_integer(5);
		a = B[0] * Plane_coords[plane_idx * 4 + 0];
		a += B[1] * Plane_coords[plane_idx * 4 + 1];
		a += B[2] * Plane_coords[plane_idx * 4 + 2];
		if (ABS(a) > EPSILON) {
			break;
			}
		}
	av = (1. / a) * Plane_coords[plane_idx * 4 + 3];
	for (i = 0; i < 3; i++) {
		B[i] *= av;
		}
	// add Q_0 onto the RHS:
	M[0 * 4 + 3] += B[0];
	M[1 * 4 + 3] += B[1];
	M[2 * 4 + 3] += B[2];

	// solve M:
	int rk;
	int base_cols[4];
	double lambda;

	if (f_vv) {
		cout << "scene::intersect_line_and_plane "
				"before Gauss elimination:" << endl;
		N.print_system(M, 3, 4);
		}
	
	rk = N.Gauss_elimination(M, 3, 4,
			base_cols, TRUE /* f_complete */,
			0 /* verbose_level */);

	if (f_vv) {
		cout << "scene::intersect_line_and_plane "
				"after Gauss elimination:" << endl;
		N.print_system(M, 3, 4);
		}
	

	if (rk < 3) {
		cout << "scene::intersect_line_and_plane "
				"the matrix M does not have full rank" << endl;
		return FALSE;
		}
	lambda = 1. * M[0 * 4 + 3];
	for (i = 0; i < 3; i++) {
		B[i] = Line_coords[line_idx * 6 + i] + lambda * v[i];
		}

	if (f_vv) {
		cout << "scene::intersect_line_and_plane "
				"The intersection point is "
				<< B[0] << ", " << B[1] << ", " << B[2] << endl;
		}
	point(B[0], B[1], B[2]);

	intersection_point_idx = nb_points - 1;
	
	if (f_v) {
		cout << "scene::intersect_line_and_plane done" << endl;
		}
	return TRUE;
}

int scene::intersect_line_and_line(int line1_idx, int line2_idx, 
	double &lambda, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; // (verbose_level >= 2);
	double B[3];
	double M[9];
	int i;
	double v[3];
	numerics N;

	if (f_v) {
		cout << "scene::intersect_line_and_line" << endl;
		}


	// equation of the form:
	// P_0 + \lambda * v = Q_0 + \mu * u

	// where P_0 is a point on the line, 
	// Q_0 is a point on the plane,
	// v is a direction vector of line 1
	// u is a direction vector of line 2

	// M is the matrix whose columns are 
	// v, -u, -P_0 + Q_0

	// point on line 1, brought over on the other side, hence the minus:
	// -P_0
	M[0 * 3 + 2] = -1. * Line_coords[line1_idx * 6 + 0];
	M[1 * 3 + 2] = -1. * Line_coords[line1_idx * 6 + 1];
	M[2 * 3 + 2] = -1. * Line_coords[line1_idx * 6 + 2];
	// +P_1
	M[0 * 3 + 2] += Line_coords[line2_idx * 6 + 0];
	M[1 * 3 + 2] += Line_coords[line2_idx * 6 + 1];
	M[2 * 3 + 2] += Line_coords[line2_idx * 6 + 2];

	// direction vector of line 1:
	// we will need v[] later, hence we store this vector
	for (i = 0; i < 3; i++) {
		v[i] = Line_coords[line1_idx * 6 + 3 + i] -
				Line_coords[line1_idx * 6 + i];
		M[i * 3 + 0] = v[i];
		}
	
	// negative direction vector of line 2:
	for (i = 0; i < 3; i++) {
		M[i * 3 + 1] = -1. * (Line_coords[line2_idx * 6 + 3 + i] -
				Line_coords[line2_idx * 6 + i]);
		}


	// solve M:
	int rk;
	int base_cols[3];

	if (f_vv) {
		cout << "scene::intersect_line_and_line "
				"before Gauss elimination:" << endl;
		N.print_system(M, 3, 3);
		}
	
	rk = N.Gauss_elimination(M, 3, 3,
			base_cols, TRUE /* f_complete */,
			0 /* verbose_level */);

	if (f_vv) {
		cout << "scene::intersect_line_and_line "
				"after Gauss elimination:" << endl;
		N.print_system(M, 3, 3);
		}
	

	if (rk < 2) {
		cout << "scene::intersect_line_and_line "
				"the matrix M does not have full rank" << endl;
		return FALSE;
		}
	lambda = M[0 * 3 + 2];
	for (i = 0; i < 3; i++) {
		B[i] = Line_coords[line1_idx * 6 + i] + lambda * v[i];
		}

	if (f_vv) {
		cout << "scene::intersect_line_and_line "
				"The intersection point is "
				<< B[0] << ", " << B[1] << ", " << B[2] << endl;
		}
	point(B[0], B[1], B[2]);

	
	if (f_v) {
		cout << "scene::intersect_line_and_line done" << endl;
		}
	return TRUE;
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
	// (x1+\lambda*v[0])^2 + (x2+\lambda*v[1])^2 + (x3+\lambda*v[2])^2 = 10^2
	// which gives
	// (v[0]^2+v[1]^2+v[2]^2) * \lambda^2 +
	// (2*x1*v[0] + 2*x2*v[1] + 2*x3*v[2]) * \lambda +
	// x1^2 + x2^2 + x3^2 - 10^2 = 0
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
	return nb_lines - 1;
}

void scene::map_a_line(int line1, int line2, 
	int plane_idx, int line_idx, double spread, 
	int nb_pts, 
	int *New_line_idx, int &nb_new_lines, 
	int *New_pt_idx, int &nb_new_points, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double line_pt1_in[3], line_pt2_in[3];
	double line_pt1[3], line_pt2[3];
	double direction[3], direction_step[3];
	double line_pt[3];
	int i, h;
	numerics N;

	if (f_v) {
		cout << "map_a_line" << endl;
		}

	nb_new_lines = 0;
	nb_new_points = 0;
	N.vec_copy(Line_coords + line_idx * 6, line_pt1_in, 3);
	N.vec_copy(Line_coords + line_idx * 6 + 3, line_pt2_in, 3);
	N.line_centered(line_pt1_in, line_pt2_in,
			line_pt1, line_pt2, spread /* r */);
	
	for (i = 0; i < 3; i++) {
		direction[i] = line_pt2[i] - line_pt1[i];
		}
	for (i = 0; i < 3; i++) {
		direction_step[i] = direction[i] / (double) (nb_pts - 1);
		}
	for (h = 0; h < nb_pts; h++) {
		if (f_v) {
			cout << "map_a_line point " << h << " / "
					<< nb_pts << ":" << endl;
			}
		for (i = 0; i < 3; i++) {
			line_pt[i] = line_pt1[i] + (double) h * direction_step[i];
			}

		New_pt_idx[nb_new_points] = point(line_pt[0], line_pt[1], line_pt[2]);
		nb_new_points++;
		
		if (map_a_point(line1, line2, plane_idx, line_pt, 
			New_line_idx[nb_new_lines], New_pt_idx[nb_new_points], 
			verbose_level)) {
		
			nb_new_lines++;
			nb_new_points++;
			}
		}
	
	
	if (f_v) {
		cout << "map_a_line done" << endl;
		}
}

int scene::map_a_point(int line1, int line2, 
	int plane_idx, double pt_in[3], 
	int &new_line_idx, int &new_pt_idx, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	double line1_pt1[4], line1_pt2[4];
	double line2_pt1[4], line2_pt2[4];
	double M1[4 * 4];
	double M2[4 * 4];
	double K1[4 * 4];
	double K2[4 * 4];
	double K3[4 * 4];
	double K4[4 * 4];
	int k1, k2, k3, i;
	numerics N;

	if (f_v) {
		cout << "map_a_point" << endl;
		}

	N.vec_copy(Line_coords + line1 * 6, line1_pt1, 3);
	line1_pt1[3] = 1.;
	N.vec_copy(Line_coords + line1 * 6 + 3, line1_pt2, 3);
	line1_pt2[3] = 1.;
	
	N.vec_copy(Line_coords + line2 * 6, line2_pt1, 3);
	line2_pt1[3] = 1.;
	N.vec_copy(Line_coords + line2 * 6 + 3, line2_pt2, 3);
	line2_pt2[3] = 1.;

	N.vec_copy(line1_pt1, M1, 4);
	N.vec_copy(line1_pt2, M1 + 4, 4);
	N.vec_copy(pt_in, M1 + 8, 3);
	M1[8 + 3] = 1.;
	if (f_vv) {
		cout << "M1:" << endl;
		N.print_system(M1, 3, 4);
		}

	N.vec_copy(line2_pt1, M2, 4);
	N.vec_copy(line2_pt2, M2 + 4, 4);
	N.vec_copy(pt_in, M2 + 8, 3);
	M2[8 + 3] = 1.;
	if (f_vv) {
		cout << "M2:" << endl;
		N.print_system(M1, 3, 4);
		}

	k1 = N.Null_space(M1, 3, 4, K1, 0 /* verbose_level */);
	if (k1 != 1) {
		cout << "map_a_point k1 != 1" << endl;
		return FALSE;
		}
	if (f_vv) {
		cout << "M1 after:" << endl;
		N.print_system(M1, 3, 4);
		cout << "K1:" << endl;
		N.print_system(K1, 1, 4);
		}

	k2 = N.Null_space(M2, 3, 4, K2, 0 /* verbose_level */);
	if (k2 != 1) {
		cout << "map_a_point k2 != 1" << endl;
		return FALSE;
		}
	if (f_vv) {
		cout << "K2:" << endl;
		N.print_system(K2, 1, 4);
		}

	N.vec_copy(K1, K3, 4);
	N.vec_copy(K2, K3 + 4, 4);
	k3 = N.Null_space(K3, 2, 4, K4, 0 /* verbose_level */);
	if (k3 != 2) {
		cout << "map_a_point k3 != 2" << endl;
		return FALSE;
		}
	if (f_vv) {
		cout << "K4:" << endl;
		N.print_system(K4, 2, 4);
		}

	N.vec_normalize_from_back(K4, 4);
	N.vec_normalize_from_back(K4 + 4, 4);
	if (ABS(K4[3] - 1.) > 0.01 && ABS(K4[4 + 3] - 1.) > 0.01) {
		cout << "K4 (1) and (2) are not affine points, "
				"this is not good" << endl;
		exit(1);
		}
	if (ABS(K4[3] - 1.) > 0.01) {
		for (i = 0; i < 4; i++) {
			K4[i] = K4[i] + K4[4 + i];
			}
		N.vec_normalize_from_back(K4, 4);
		if (ABS(K4[3] - 1.) > 0.01) {
			cout << "after fixing, K4 (1) is not an affine point, "
					"this is not good" << endl;
			return FALSE;
			}
		}
	else if (ABS(K4[4 + 3] - 1.) > 0.01) {
		for (i = 0; i < 4; i++) {
			K4[4 + i] = K4[i] + K4[4 + i];
			}
		N.vec_normalize_from_back(K4 + 4, 4);
		if (ABS(K4[4 + 3] - 1.) > 0.01) {
			cout << "after fixing, K4 (2) is not an affine point, "
					"this is not good" << endl;
			return FALSE;
			}
		}
	new_line_idx = line_extended(K4[0], K4[1], K4[2],
			K4[4 + 0], K4[4 + 1], K4[4 + 2], 10.);
	
	intersect_line_and_plane(new_line_idx, plane_idx,
			new_pt_idx, 0 /* verbose_level */);

	if (f_v) {
		cout << "map_a_point done" << endl;
		}
	return TRUE;
}

void scene::lines_a()
{
        double t1 = 10;
        double t2 = -10;

	line(
		(-sqrt(5)-3) * t1 + (3*sqrt(5)+7)/2 , t1 , -sqrt(5) * t1 +(sqrt(5)+3)/2 ,
		(-sqrt(5)-3) * t2 + (3*sqrt(5)+7)/2 , t2 , -sqrt(5) * t2 +(sqrt(5)+3)/2);
		// the line \ell_{0,5} = a_1

	line( 
		-sqrt(5) * t1 +(sqrt(5)+3)/2, (-sqrt(5)-3) * t1 + (3*sqrt(5)+7)/2 , t1  ,
		-sqrt(5) * t2 +(sqrt(5)+3)/2, (-sqrt(5)-3) * t2 + (3*sqrt(5)+7)/2 , t2);
		// the line \ell_{1,5} = a_2

	line(
		t1, -sqrt(5) * t1 +(sqrt(5)+3)/2, (-sqrt(5)-3) * t1 + (3*sqrt(5)+7)/2   ,
		t2, -sqrt(5) * t2 +(sqrt(5)+3)/2, (-sqrt(5)-3) * t2 + (3*sqrt(5)+7)/2);
		// the line \ell_{2,5} = a_3

	line(
		-(sqrt(5)+3)/4 * t1 + (-sqrt(5)+3)/4 , t1 , (sqrt(5)+1)/4 + (-3*sqrt(5)-5)/4 * t1 ,
		-(sqrt(5)+3)/4 * t2 + (-sqrt(5)+3)/4 , t2 , (sqrt(5)+1)/4 + (-3*sqrt(5)-5)/4 * t2);
		// the line \ell_{0,7} = a_4

	line(
		(sqrt(5)+1)/4 + (-3*sqrt(5)-5)/4 * t1, -(sqrt(5)+3)/4 * t1 + (-sqrt(5)+3)/4 , t1  ,
		(sqrt(5)+1)/4 + (-3*sqrt(5)-5)/4 * t2, -(sqrt(5)+3)/4 * t2 + (-sqrt(5)+3)/4 , t2);
		// the line a_5

	line(
		t1, (sqrt(5)+1)/4 + (-3*sqrt(5)-5)/4 * t1, -(sqrt(5)+3)/4 * t1 + (-sqrt(5)+3)/4   ,
		t2, (sqrt(5)+1)/4 + (-3*sqrt(5)-5)/4 * t2, -(sqrt(5)+3)/4 * t2 + (-sqrt(5)+3)/4);
		// the line a_6
}


void scene::lines_b()
{
        double t1 = 10;
        double t2 = -10;

	line(
		(sqrt(5)-3) * t1 + (-3*sqrt(5)+7)/2 , t1 , (-sqrt(5)+3)/2 + sqrt(5) * t1 ,
		(sqrt(5)-3) * t2 + (-3*sqrt(5)+7)/2 , t2 , (-sqrt(5)+3)/2 + sqrt(5) * t2);
		// the line \ell_{0,8} = b_1

	line(
		(-sqrt(5)+3)/2 + sqrt(5) * t1, (sqrt(5)-3) * t1 + (-3*sqrt(5)+7)/2 , t1  ,
		(-sqrt(5)+3)/2 + sqrt(5) * t2, (sqrt(5)-3) * t2 + (-3*sqrt(5)+7)/2 , t2);
		// the line \ell_{1,8} = b_2
	
	line(
		t1, (-sqrt(5)+3)/2 + sqrt(5) * t1, (sqrt(5)-3) * t1 + (-3*sqrt(5)+7)/2   ,
		t2, (-sqrt(5)+3)/2 + sqrt(5) * t2, (sqrt(5)-3) * t2 + (-3*sqrt(5)+7)/2);
		// the line \ell_{2,8} = b_2

	line(
		(sqrt(5)-3)/4 * t1 + (sqrt(5)+3)/4 , t1 , (-sqrt(5)+1)/4 + (3*sqrt(5)-5)/4 * t1 ,
		(sqrt(5)-3)/4 * t2 + (sqrt(5)+3)/4 , t2 , (-sqrt(5)+1)/4 + (3*sqrt(5)-5)/4 * t2);
		// the line \ell_{0,6} = b_4

	line(
		(-sqrt(5)+1)/4 + (3*sqrt(5)-5)/4 * t1, (sqrt(5)-3)/4 * t1 + (sqrt(5)+3)/4 , t1  ,
		(-sqrt(5)+1)/4 + (3*sqrt(5)-5)/4 * t2, (sqrt(5)-3)/4 * t2 + (sqrt(5)+3)/4 , t2);
		// the line \ell_{1,6} = b_5

	line(
		t1, (-sqrt(5)+1)/4 + (3*sqrt(5)-5)/4 * t1, (sqrt(5)-3)/4 * t1 + (sqrt(5)+3)/4   ,
		t2, (-sqrt(5)+1)/4 + (3*sqrt(5)-5)/4 * t2, (sqrt(5)-3)/4 * t2 + (sqrt(5)+3)/4);
		// the line \ell_{2,6} = b_6

}

void scene::lines_cij()
{
	double b = 4.;

	// 12 = c_12   3
	line(-1. * b, -3. * b - 1, 0., b, 3. * b - 1, 0.);
	//cylinder{< -1.*b,-3.*b-1,0. >,<b,3.*b-1,0. > ,r }   //c_12
	//c_{12} & (t, 3t-1, 0) 

	// 13 = c_13  4
	line(-3. * b -1, 0, -1. * b, 3. * b - 1, 0, b);
	//cylinder{< -3.*b-1,0,-1.*b >,<3.*b-1,0,b > ,r }   //c_13
	//c_{13} & (3t-1, 0,t)  

	// 14 = c_14  12 bottom
	line(b, -1. * b, -1., -1. * b, b, -1.);
	//cylinder{< b,-1.*b,-1. >,<-1.*b,b,-1. > ,r }   //c_14 bottom plane 
	//c_{14} & (-t,t,-1) 

	//  15 = c_15  7 top
	line(2. + b, 1, -1. * b, 2. -1. * b, 1, b);
	//cylinder{< 2.+b,1,-1.*b >,<2.-1.*b,1,b > ,r }   //c_15top plane 
	//c_{15} & (2-t, 1, t)


	// 16 = c_16 15 middle
	line(0, b + 1, -1. * b , 0, -1.*b + 1, b);
	//cylinder{< 0,b+1,-1.*b >,<0,-1.*b+1,b > ,r } //c_16 middle
	//c_{16} & (0, t,1-t) 

	// 17 = c_23 11
	line(0, -1. * b, -3. * b -1., 0, b, 3. * b - 1.);
	//cylinder{< 0,-1.*b,-3.*b-1. >,<0,b,3.*b-1. > ,r } //c_23
	//c_{23} & (0,t,3t-1)  

	// 18 = c_24  10 middle
	line(b + 1, 0, -1. * b, -1. * b + 1, 0, b);
	//cylinder{< b+1,0,-1.*b >,<-1.*b+1,0,b > ,r }   //c_24 middle
	//c_{24} & (1-t, 0, t)  

	//  19 = c_25 8 bottom
	line(-1, b, -1. * b, -1, -1. * b, b);
	//cylinder{< -1,b,-1.*b >,<-1,-1.*b,b > ,r }   //c_25 bottom plane 
	//c_{25} & (-1,t,-t)  

	// 20 = c_26  6 top
	line(2. + b, -1. * b, 1., 2. -1. * b, b, 1.);
	//cylinder{< 2.+b,-1.*b,1. >,<2.-1.*b,b,1. > ,r }  //c_26 top plane 
	//c_{26} & (t,2-t,1)

	// 21 = c_34   1 top
	line(1, 2. + b, -1. * b, 1, 2. -1. * b, b);
	//cylinder{< 1,2.+b,-1.*b >,<1,2.-1.*b,b > ,r }   //c_34 top plane 
	//c_{34} & (1, t,2-t)  

	//22  = c_35  middle
	line(b + 1, -1. * b, 0., -1. * b + 1, b, 0.);
	//cylinder{< b+1,-1.*b,0. >,<-1.*b+1,b,0. > ,r }   //c_35 middle
	//c_{35} & (t,1-t,0)   

	// 23 = c_36  2 bottom
	line(b, -1, -1. * b, -1. * b, -1, b);
	//cylinder{< b,-1,-1.*b >,<-1.*b,-1,b > ,r } //c_36 bottom plane 
	//c_{36} & (t,-1,-t)  

	// 24 = c_45  9
	line(0, -3. * b - 1, -1. * b, 0, 3. * b - 1, b);
	//cylinder{< 0,-3.*b-1,-1.*b >,<0,3.*b-1,b > ,r }   //c_45 
	//c_{45}  & (0, 3t-1, t)

	// 25 = c_46  13
	line(-3. * b - 1, -1. * b, 0., 3. * b - 1, b, 0.);
	//cylinder{< -3.*b-1,-1.*b,0. >,<3.*b-1,b,0. > ,r }   //c_46
	//c_{46} & (3t-1, t,0)  

	// 26 = c_56  5
	line(-1. * b, 0, -3. * b - 1., b, 0, 3. * b - 1.);
	//cylinder{< -1.*b,0,-3.*b-1. >,<b,0,3.*b-1. > ,r }   //c_56
	//c_{56} & (t,0,3t-1)  


}

void scene::fourD_cube(double rad_desired)
{
	int r, i, j, k, h;
	int v[3];
	double x[3];
	double rad;

	int first_pt_idx;

	first_pt_idx = nb_points;
	for (r = 0; r < 2; r++) {
		if (r == 0) {
			rad = 1;
			}
		else {
			rad = 2;
			}
		for (i = 0; i < 2; i++) {
			v[0] = i;
			for (j = 0; j < 2; j++) {
				v[1] = j;
				for (k = 0; k < 2; k++) {
					v[2] = k;
					for (h = 0; h < 3; h++) {
						if (v[h] == 1) {
							x[h] = -1 * rad;
							}
						else {
							x[h] = rad;
							}
						}
					point(x[0], x[1], x[2]);
					}
				}
			}
		}
	fourD_cube_edges(first_pt_idx);

	rescale(first_pt_idx, rad_desired);

}

void scene::rescale(int first_pt_idx, double rad_desired)
{
	int i;
	double rad = 1., a;
	numerics N;

	for (i = first_pt_idx; i < nb_points; i++) {
		a = N.distance_from_origin(
				Point_coords + i * 3, 3);
		if (i == first_pt_idx) {
			rad = a;
			}
		else {
			if (a > rad) {
				rad = a;
				}
			}
		}
	a = rad_desired / rad;
	N.vec_scalar_multiple(Point_coords + first_pt_idx * 3,
			a, 3 * (nb_points - first_pt_idx));
}

double scene::euclidean_distance(int pt1, int pt2)
{
	double d;
	numerics N;

	d = N.distance_euclidean(Point_coords + pt1 * 3, Point_coords + pt2 * 3, 3);
	return d;
}

double scene::distance_from_origin(int pt)
{
	double d;
	numerics N;

	d = N.distance_from_origin(
			Point_coords[pt * 3 + 0],
			Point_coords[pt * 3 + 1], Point_coords[pt * 3 + 2]);
	return d;
}

void scene::fourD_cube_edges(int first_pt_idx)
{
	int i, j;
	double d;

	for (i = 0; i < 8; i++) {
		edge(first_pt_idx + i, first_pt_idx + 8 + i);
		}
	for (i = 0; i < 8; i++) {
		for (j = i + 1; j < 8; j++) {
			d = distance_between_two_points(first_pt_idx + i, first_pt_idx + j);
			cout << "i=" << i << " j=" << j << " d=" << d << endl;
			if (ABS(d - 2) < 0.2) {
				cout << "EDGE " << i << ", " << j << endl;
				edge(first_pt_idx + i, first_pt_idx + j);
				edge(first_pt_idx + 8 + i, first_pt_idx + 8 + j);
				}
			else {
				//cout << endl;
				}
			}
		}
}

// tetrahedron vertices on the unit cube:
//v1 = ( sqrt(8/9), 0 , -1/3 )
//v2 = ( -sqrt(2/9), sqrt(2/3), -1/3 )
//v3 = ( -sqrt(2/9), -sqrt(2/3), -1/3 )
//v4 = ( 0 , 0 , 1 )


void scene::hypercube(int n, double rad_desired)
{
	int N, i, j, h, k, d;
	int *v;
	int *w;
	double *Basis; // [n * 3];
	double x[3];
	double t, dt;
	numerics Num;
	number_theory_domain NT;
	geometry_global Gg;

	int first_pt_idx;

	first_pt_idx = nb_points;
	N = NT.i_power_j(2, n);

	Basis = new double [n * 3];
	v = NEW_int(n);
	w = NEW_int(n);
	
	if (n == 4) {
		Basis[0 * 3 + 0] = sqrt(8./9.); 
		Basis[0 * 3 + 1] = 0; 
		Basis[0 * 3 + 2] = -1 / 3.; 
		Basis[1 * 3 + 0] = -sqrt(2./9.); 
		Basis[1 * 3 + 1] = sqrt(2/3); 
		Basis[1 * 3 + 2] = -1 / 3.; 
		Basis[2 * 3 + 0] = -sqrt(2./9.); 
		Basis[2 * 3 + 1] = -sqrt(2/3); 
		Basis[2 * 3 + 2] = -1 / 3.; 
		Basis[3 * 3 + 0] = 0; 
		Basis[3 * 3 + 1] = 0; 
		Basis[3 * 3 + 2] = -1 / 3.; 
		}
	else {
		dt = 1. / (n - 1);
		for (i = 0; i < n; i++) {
			t = i * dt;
			Basis[i * 3 + 0] = cos(M_PI * t); 
			Basis[i * 3 + 1] = sin(M_PI * t); 
			Basis[i * 3 + 2] = t; 
			}
		}
	for (i = 0; i < n; i++) {
		Num.make_unit_vector(Basis + i * 3, 3);
		}

	for (h = 0; h < N; h++) {
		Gg.AG_element_unrank(2, v, 1, n, h);
		for (j = 0; j < 3; j++) {
			x[j] = 0.;
			}
		for (i = 0; i < n; i++) {
			if (v[i]) {
				for (j = 0; j < 3; j++) {
					x[j] += Basis[i * 3 + j];
					}
				}
			else {
				for (j = 0; j < 3; j++) {
					x[j] -= Basis[i * 3 + j];
					}
				}
			}
		point(x[0], x[1], x[2]);
		}
	for (h = 0; h < N; h++) {
		Gg.AG_element_unrank(2, v, 1, n, h);
		for (k = h + 1; k < N; k++) {
			Gg.AG_element_unrank(2, w, 1, n, k);
			d = 0;
			for (i = 0; i < n; i++) {
				if (v[i] != w[i]) {
					d++;
					}
				}
			if (d == 1) {
				edge(first_pt_idx + h, first_pt_idx + k);
				}
			}
		}
	

	rescale(first_pt_idx, rad_desired);
	delete [] Basis;
	FREE_int(v);
	FREE_int(w);
}


void scene::Eckardt_points()
{
	point(1,1,1. ); //0
	point(0,-1,0. ); //1 
	point(.5,.5,0. ); //2 
	point(0,.5,.5 ); //3
	point(0,0,-1. ); //4 
	point(.5,0,.5 ); //5 
	point(-1,0,0. ); //6 
}

void scene::Dodecahedron_points()
{
	double zero;
	double one, minus_one;
	double phi, minus_phi;
	double phi_inv, minus_phi_inv;

	zero = 0.;
	one = 1.;
	phi = (1. + sqrt(5)) * 0.5;
	phi_inv = (sqrt(5) - 1.) * 0.5;
	minus_one = -1. * one;
	minus_phi = -1. * phi;
	minus_phi_inv = -1. * phi_inv;


	point(one,one,one); //0
	point(one,one,minus_one); //1
	point(one,minus_one,one); //2
	point(one,minus_one,minus_one); //3
	point(minus_one,one,one); //4
	point(minus_one,one,minus_one); //5
	point(minus_one,minus_one,one); //6
	point(minus_one,minus_one,minus_one); //7

	point(zero,phi,phi_inv); //8
	point(zero,phi,minus_phi_inv); //9
	point(zero,minus_phi,phi_inv); //10
	point(zero,minus_phi,minus_phi_inv); //11

	point(phi_inv,zero,phi); //12
	point(phi_inv,zero,minus_phi); //13
	point(minus_phi_inv,zero,phi); //14
	point(minus_phi_inv,zero,minus_phi); //15
	
	point(phi,phi_inv,zero); //16
	point(phi,minus_phi_inv,zero); //17
	point(minus_phi,phi_inv,zero); //18
	point(minus_phi,minus_phi_inv,zero); //19


}

void scene::Dodecahedron_edges(int first_pt_idx)
{
	int i, j;
	double d;
	double phi;
	double two_over_phi;

	phi = (1. + sqrt(5)) * 0.5;
	two_over_phi = 2. / phi;
	
	for (i = 0; i < 20; i++) {
		for (j = i + 1; j < 20; j++) {
			d = distance_between_two_points(
					first_pt_idx + i, first_pt_idx + j);
			//cout << "i=" << i << " j=" << j << " d="
			//<< d << " two_over_phi=" << two_over_phi;
			if (ABS(d - two_over_phi) < 0.2) {
				cout << "EDGE " << i << ", " << j << endl;
				edge(first_pt_idx + i, first_pt_idx + j);
				}
			else {
				//cout << endl;
				}
			}
		}
}

void scene::Dodecahedron_planes(int first_pt_idx)
{
	int i;
	
#if 0
	int faces[] = {
		8, 9, 11, 10, 
		12, 14, 15, 13, 
		16, 17, 19, 18
		};
	for (i = 0; i < 12; i++) {
		faces[i] += first_pt_idx;
		}
	face(faces + 0, 4);
	face(faces + 4, 4);
	face(faces + 8, 4);
#else
	int faces[] = {
		0,16,1,9,8, 
		0,12,2,17,16, 
		0,8,4,14,12,
		1,13,15,5,9,
		1,16,17,3,13,
		2,10,11,3,17,
		2,12,14,6,10,
		4,8,9,5,18,
		5,15,7,19,18,
		6,14,4,18,19,
		3,11,7,15,13,
		19,7,11,10,6
		};
	for (i = 0; i < 12 * 5; i++) {
		faces[i] += first_pt_idx;
		}
	for (i = 0; i < 12; i++) {
		face(faces + i * 5, 5);
		}
#endif
}

void scene::tritangent_planes()
{
	plane(1/sqrt(3),1/sqrt(3),1/sqrt(3), 3*1/sqrt(3));
	plane(1/sqrt(3),1/sqrt(3),1/sqrt(3), 1*1/sqrt(3));
	plane(1/sqrt(3),1/sqrt(3),1/sqrt(3), -1*1/sqrt(3));
}

void scene::clebsch_cubic()
{
	double coeff[20] = {-3,7,7,1,7,-2,-14,7,-14,3,-3,7,1,7,-14,3,-3,1,3,-1.};

	cubic(coeff);
}


double scene::distance_between_two_points(int pt1, int pt2)
{
	double x1, x2, x3;
	double y1, y2, y3;
	double d1, d2, d3;
	double d;

	x1 = Point_coords[pt1 * 3 + 0];
	x2 = Point_coords[pt1 * 3 + 1];
	x3 = Point_coords[pt1 * 3 + 2];
	y1 = Point_coords[pt2 * 3 + 0];
	y2 = Point_coords[pt2 * 3 + 1];
	y3 = Point_coords[pt2 * 3 + 2];
	d1 = y1 - x1;
	d2 = y2 - x2;
	d3 = y3 - x3;
	d = sqrt(d1 * d1 + d2 * d2 + d3 * d3);
	return d;
}

void scene::create_five_plus_one()
{
	plane(0., 0., 1., 0.); // c23'
	plane(0., 1., 0., 0.); // c56'
	line_extended(0., 0., 0., 0., 1., 0., 5.); // c45'
	line_extended(0., 0., 1., 1., 1., 1., 5.); // c46'
	line_extended(-sqrt(5) - 3., 2*sqrt(5) + 4., 0., 
		0., (sqrt(5) + 1.)/2., (-sqrt(5) + 3.)/2., 
		5.); // a2'
	line_extended(-4.-2.*sqrt(5), sqrt(5)+3., 0.,
		(-sqrt(5)-1.)/2., 1., (sqrt(5)+3.)/2., 
		5. ); // a3'
}


void scene::create_Hilbert_model(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "scene::create_Hilbert_model" << endl;
	}
	numerics N;
	double p = 1.;
	double m = -1.;
	double px = p + 1;
	double mx = m - 1;
	int i;

	point(p,p,p); // 0
	point(p,p,m); // 1
	point(p,m,p); // 2
	point(p,m,m); // 3
	point(m,p,p); // 4
	point(m,p,m); // 5
	point(m,m,p); // 6
	point(m,m,m); // 7

	face4(0, 1, 5, 4); // 0, front right
	face4(0, 2, 3, 1); // 1, front left
	face4(0, 4, 6, 2); // 2, top
	face4(1, 5, 7, 3); // 3, bottom
	face4(4, 6, 7, 5); // 4, back right
	face4(2, 3, 7, 6); // 5, back left


	// top front:
	point(px,p,p); // 8
	point(p,px,p); // 9
	point(p,p,px); // 10

	// top back:
	point(mx,m,p); // 11
	point(m,mx,p); // 12
	point(m,m,px); // 13

	// bottom left:
	point(px,m,m); // 14
	point(p,mx,m); // 15
	point(p,m,mx); // 16

	// bottom right:
	point(mx,p,m); // 17
	point(m,px,m); // 18
	point(m,p,mx); // 19

	// the edges of the cube:
	edge(0, 1); // 0
	edge(0, 2); // 1
	edge(0, 4); // 2
	edge(1, 3); // 3
	edge(1, 5); // 4
	edge(2, 3); // 5
	edge(2, 6); // 6
	edge(3, 7); // 7
	edge(4, 5); // 8
	edge(4, 6); // 9
	edge(5, 7); // 10
	edge(6, 7); // 11
	
	// the double six:
	// there is a symmetry (a1,a2,a3)(a4,a5,a6)(b1,b2,b3)(b4,b5,b6)
	// also, a_i^\perp = b_i
	edge(8, 17); // 12 a1
	edge(9, 12); // 13 a2
	edge(10, 16); // 14 a3
	edge(11, 14); // 15 a4
	edge(15, 18); // 16 a5
	edge(13, 19); // 17 a6

	edge(13, 16); // 18 b1
	edge(14, 17); // 19 b2
	edge(12, 18); // 20 b3
	edge(10, 19); // 21 b4
	edge(8, 11); // 22 b5
	edge(9, 15); // 23 b6


	point(2,1,1); // 20
	point(-2,1,-1); // 21
	point(-6,1,-3); // 22
	point(-10,1,-5); // 23

	// the tetrahedron:
	face3(0, 6, 3 ); // 6, left
	face3(0, 6, 5 ); // 7, right
	face3(0, 3, 5 ); // 8, front
	face3(6, 3, 5 ); // 9, back
	edge(0, 6); // 24
	edge(0, 3); // 25
	edge(0, 5); // 26
	edge(6, 3); // 27
	edge(6, 5); // 28
	edge(3, 5); // 29

	// the extended edges of the cube:
	edge(0, 8); // 30
	edge(0, 9); // 31
	edge(0, 10); // 32
	edge(3, 14); // 33
	edge(3, 15); // 34
	edge(3, 16); // 35
	edge(5, 17); // 36
	edge(5, 18); // 37
	edge(5, 19); // 38
	edge(6, 11); // 39
	edge(6, 12); // 40
	edge(6, 13); // 41
	
	// the base points in the plane \pi_{12,34,56}
	point(4./5., -2./5.,-1.); // 24
	point(4./5., -1.,-8./5.); // 25
	point(1., -2./5.,-4./5.); // 26
	point(8./5., -1.,-4./5.); // 27
	point(-2./5., 4./5., -1.); // 28
	point(-1., 4./5.,-8./5.); // 29


	// the base points in the plane \pi_{16,24,35} (at the bottom)
	point(8./5., 1.,4./5.); // 30
	point(-4./5., -8./5.,1.); // 31
	point(1.,-4./5., -8./5.); // 32
	point(-4./5.,-1., 2./5.); // 33
	point(2./5.,-4./5.,-1.); // 34
	point(1.,2./5.,4./5.); // 35
	point(2./5.,-1.,-4./5.); // 36
	point(1.,-8./5.,-4./5.); // 37
	
	double planes[] = {
		1.,0,0,0, // 0, X=0 
		0,1.,0,0, // 1, Y=0 
		0,0,1.,0, // 2, Z=0 
		-1./2., -1., 1., 1., // 3 a1b2
		-2, 1., -1., 1., // 4 a2b1
		-5./7., -5./7., 5./7., 1., // 5 pi_{12,34,56}
		.5,-1,-1,1, // 6 F1 = \pi_7
		2,-1,-1,1, // 7 F2 = \pi_13
		0,0,0,1, // 8 F3 = \pi_38 = plane at infinity (cannot draw)
		(double)5/(double)4,0,0,1, // 9 G1 = \pi_41
		0,-1,0,1, // 10 G2 = \pi_5
		0,0,-1,1, // 11 G3 = \pi_15

		// tritangent planes in dual coordinates, computed using Maple:

		-1, -2, 2, 2, // 12 = start of tritangent planes
		-2, 1, -1, 1, 
		1, -1, -2, 1, 
		-2, 2, -1, 2, 
		0, -1, 0, 1, 
		0, 1, 0, 1, 
		1, -2, -2, 2, 
		2, 1, 1, 1, 
		-1, -1, 2, 1, 
		2, 2, 1, 2, 
		2, -1, -2, 2, 
		-1, -2, 1, 1, 
		2, -1, -1, 1, 
		1, 2, 2, 2, 
		0, 0, -1, 1, 
		0, 0, 1, 1, 
		-2, 1, -2, 2, 
		1, 2, 1, 1,
		 -2, -2, 1, 2, 
		1, 1, 2, 1, 
		-1, 2, -1, 1, 
		2, 1, 2, 2, 
		-1, 0, 0, 1, 
		1, 0, 0, 1, 
		-1, 2, -2, 2, 
		-2, -1, 1, 1, 
		-1, 1, -2, 1, 
		2, -2, -1, 2, 
		-2, -1, 2, 2, 
		1, -2, -1, 1, 
		-5, -5, 5, 7, 
		-5, 5, -5, 1, 
		-5, 0, 0, 4, 
		5, -5, -5, 1, 
		0, 0, -5, 4, 
		-5, 5, -5, 7,
		0, -5, 0, 4, 
		0, 0, 0, 1, 
		0, 5, 0, 4, 
		5, -5, -5, 7, 
		5, 0, 0, 4, 
		5, 5, 5, 1, 
		-5, -5, 5, 1, 
		5, 5, 5, 7, 
		0, 0, 5, 4
		// end of tritangent planes


		};
	for (i = 0; i < 12 + 45; i++) {
		plane_from_dual_coordinates(planes + i * 4);
		}


	double Lines[] = {
		0,1,0,-2,0,-1, // 0 a1
		0,0,1,-1,-2,0, // 1 a2
		1,0,0,0,-1,-2, // 2 a3
		0,-1,0,2,0,-1, // 3 a4
		0,0,-1,-1,2,0, // 4 a5
		-1,0,0,0,1,-2, // 5 a6
		0,-1,0,1,0,-2, // 6 b1
		0,0,-1,-2,1,0, // 7 b2
		-1,0,0,0,2,-1, // 8 b3
		0,1,0,-1,0,-2, // 9 b4
		0,0,1,-2,-1,0, // 10 b5
		1,0,0,0,-2,-1, // 11 b6
		4./5., 3./5., 0, 0., 1., 1., // 12, 0, c12
		3./5., 0., 4./5., 1, 1., 0., // 13, 1, c13
		0.,0.,1., 1.,0.,0., // 14, 2 c14
		-4./5., 3./5., 0., 0., -1., 1., // 15, 3 c15
		-3./5., 0., -4./5., -1., 1., 0., // 16, 4 c16
		-3./5., 4./5., 0., 1., 0., 1., // 17, 5 c23
		-4./5., -3./5., 0., 0., -1., 1., // 18, 6 c24
		0., 1., 0., 1., 0., 0., // 19, 7 c25
		3./5., -4./5., 0., -1., 0., 1., // 20, 8 c26
		3./5., 0., -4./5., -1., 1., 0., // 21, 9 c34
		-3./5., -4./5., 0., -1., 0., 1., // 22, 10 c35
		0., 0., 1., 0., 1., 0., // 23, 11 c36
		4./5., -3./5., 0., 0., 1., 1., // 24, 12 c45
		-3./5., 0., 4./5., 1., 1., 0., // 25, 13 c46
		3./5., 4./5., 0., 1., 0., 1.,  // 26, 14 c56
		};
	double r = sqrt(5);
	for (i = 0; i < 27; i++) {
		line_pt_and_dir(Lines + i * 6, r);
		}

	{
	double coeff[20] = {0,0,0,-1,0,5./2.,0,0,0,0,0,0,-1,0,0,0,0,-1,0,1};
	cubic(coeff);  // cubic 0
	}
	{ // the quadric determined by b2, b3, b4: 
	double coeff[10] = {-2,-5,5,5,-2,-5,-5,-2,5,7};
	quadric(coeff); // quadric 0
	}
	// Cayley's nodal cubic, 6,7,9,15 (starting from 1)
	{
	//double a = 0.2;
	double coeff_in[20] = {0,0,0,0,0,1,1,0,1,0,0,0,0,0,1,0,0,0,0,0};
	double coeff_out[20];
#if 0
	for (i = 0; i < 20; i++) {
		if (coeff[i]) {
			coeff[i] = a;
			}
		}
#endif
	//cubic(coeff);

	double A4[] = {
		1.,1.,1.,1.,
		-1.,-1.,1,1,
		1.,-1.,-1.,1,
		-1.,1.,-1.,1
		};
	double A4_inv[16];

	N.matrix_double_inverse(A4, A4_inv, 4, 0 /* verbose_level*/);

	N.substitute_cubic_linear(coeff_in, coeff_out,
			A4_inv, 0 /*verbose_level*/);

	if (f_v) {
		cout << "i : coeff_in[i] : coeff_out[i]" << endl;
		for (i = 0; i < 20; i++) {
			cout << i << " : " << coeff_in[i] << " : " << coeff_out[i] << endl;
		}
	}

#if 0
0 : 0 : 0
1 : 0 : 0
2 : 0 : 0
3 : 0 : -0.0625 : x^2
4 : 0 : 0
5 : 1 : 0.125 : xyz
6 : 1 : 0
7 : 0 : 0
8 : 1 : 0
9 : 0 : 0
10 : 0 : 0
11 : 0 : 0
12 : 0 : -0.0625 : y^2
13 : 0 : 0
14 : 1 : 0
15 : 0 : 0
16 : 0 : 0
17 : 0 : -0.0625 : z^2
18 : 0 : 0
19 : 0 : 0.0625 : 1
#endif
	cubic(coeff_out); // cubic 1

	// pts of the tetrahedron: 0,6,3,5
	int pts[4] = {0, 6, 3, 5};
	double rad = 5.;

	// create lines 27-32 on Cayley's nodal cubic  (previously: 15,16,17,18,19,20)
	line_through_two_points(pts[0], pts[1], rad);
	line_through_two_points(pts[0], pts[2], rad);
	line_through_two_points(pts[0], pts[3], rad);
	line_through_two_points(pts[1], pts[2], rad);
	line_through_two_points(pts[1], pts[3], rad);
	line_through_two_points(pts[2], pts[3], rad);
#if 0
	pts[4];
	for (i = 0; i < 4; i++) {
		pts[0] = point(A4[0 * 4 + 0], A4[0 * 4 + 1], A4[0 * 4 + 2]);
		}
#endif

	}
	clebsch_cubic(); // cubic 2
	// lines 33-59   (previously: 21, 21+1, ... 21+26=47)
	lines_a();
	lines_b();
	lines_cij();


	double coeff_fermat[20] = {1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1};
	cubic(coeff_fermat); // cubic 3, Fermat
	{
	double Lines[] = {
		-1.,0,0,0,-1.,1, // 60 (previously 48)
		0.,-1.,0,-1.,0,1, // 61 (previously 49)
		0.,0.,-1.,-1.,1,0 // 62 (previously 50)
		};
	double r = 3.6; //sqrt(5);
	// lines 60-62 (previously 48,49,50)
	for (i = 0; i < 3; i++) {
		line_pt_and_dir(Lines + i * 6, r);
		}
	}
	// Cayleys ruled surface:
	//X0X1X2 − X1^3 − X0^2X3
	// XYZ - Y^3 - X^2 coeff (1 based) 6,11,4
	double coeff_cayley_ruled[20] =
		{0,0,0,-1,0,1,0,0,0,0,-1,0,0,0,0,0,0,0,0,0};
	cubic(coeff_cayley_ruled); // cubic 4, Cayley / Chasles ruled

	// -x + y - z^3
	double coeff_cayley_ruled2[20] =
		{0,0,0,0,0,0,0,0,0,-1,0,0,0,0,1,0,-1,0,0,0};
	cubic(coeff_cayley_ruled2); // cubic 5, Cayley / Chasles ruled
	// xy-x^3-z (-1,1, 1,7, -1,19)
	double coeff_cayley_ruled3[20] =
		{-1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,-1,0};
	cubic(coeff_cayley_ruled3); // cubic 6, Cayley / Chasles ruled


	// create lines on Cayley's ruled surface:
	double Line[6];
	double s0, s1;
	r = 10;
	double phi, delta_phi;
	
	int nb_lines0 = nb_lines;
	int nb_lines_wanted = 18;
	int nb_lines_actual = 0;
	delta_phi = M_PI / nb_lines_wanted;
	for (i = 0; i < nb_lines_wanted; i++) {
		phi = M_PI / 2 + (double) i * delta_phi;
		s0 = cos(phi);
		s1 = sin(phi);
		if (ABS(s0) > 0.01) {
			Line[0] = s1 / s0;
			Line[1] = Line[0] * Line[0];
			Line[2] = 0.;
			Line[3] = 0.;
			Line[4] = s0;
			Line[5] = s1;
			if (f_v) {
				cout << "creating line " << i << " phi=" << phi << " ";
				N.vec_print(Line, 6);
				cout << endl;
			}
			if (line_pt_and_dir(Line, r)) {
				nb_lines_actual++;
				}
			else {
				cout << "line could not be created" << endl;
				}
			}
		}
	if (f_v) {
		cout << "nb_lines0 = " << nb_lines0 << endl;
		cout << "nb_lines_actual = " << nb_lines_actual << endl;
		cout << "nb_lines = " << nb_lines << endl;
	}
	point(0,0,0); // 38

	double osculating_tangent_line[6] = {0,0,0,1,0,0};
	line_pt_and_dir(osculating_tangent_line, r); // 

	{ // the quadric determined by b1, b2, b3: 
	double coeff[10] = {2,5,5,5,2,5,5,2,5,3};
	quadric(coeff); // quadric 1 B123
	}
	{ // the quadric determined by b1, b2, b4: 
	double coeff[10] = {-4,-10,0,0,-4,0,0,1,5,4};
	quadric(coeff); // quadric 2 B124
	}
	{ // the quadric determined by b1, b2, b5: 
	double coeff[10] = {-2,0,-5,0,8,0,10,-2,0,2};
	quadric(coeff); // quadric 3 B125
	}
	{ // the quadric determined by b1, b2, b6: 
	double coeff[10] = {-2,-5,-5,-5,-2,5,5,-2,5,7};
	quadric(coeff); // quadric 4 B126
	}

	{
	double coeff_surf_lifted[20] = {0, 0, 0, -4/(double)25, 
		0, -1, 1, 0, 1, -41/(double)25, 
		0, 0, -1, 0, -2, 4, 0, -1, 4, -91/(double)25};
	//double coeff_surf_lifted[20] = {0, 0, 0, -1, 0, -1, -2, 0, 1, 4, 0, 0, 
	//	-1, 0, 1, 4, 0, (double)-4/(double)25, 
	//	(double)-41/(double)25, (double)-91/(double)25};
	cubic(coeff_surf_lifted); // cubic 7, arc_lifting
	}
	{
	double coeff_surf_lifted[20] = {0, 0, 0, (double)-1/(double)4, 0, 
		(double)-5/(double)4, 0, 0, 0, (double)-6/(double)5, 
		0, 0, -1, 0, -3, 0, 0, -1, 0, (double)-11/(double)25};
	cubic(coeff_surf_lifted); // cubic 8, arc_lifting
	}

	int set[3];
	int nCk = int_n_choose_k(8, 3);
	int rk;
	int first_three_face;

	first_three_face = nb_faces;
	if (f_v) {
		cout << "first_three_face = " << first_three_face << endl;
	}
	for (rk = 0; rk < nCk; rk++) {
		unrank_k_subset(rk, set, 8 /*n*/, 3 /*k*/);
		face3(set[0], set[1], set[2]);
		if (f_v) {
			cout << "rk=" << rk << " set=";
			int_vec_print(cout, set, 3);
			cout << endl;
		}
	}


	// the lines in long:
	cout << "the long lines start at " << nb_lines << endl;
	r = 3.6;
	for (i = 0; i < 27; i++) {
		line_pt_and_dir(Lines + i * 6, r);
		}


	point(1,0,0); // P39
	point(0,1,0); // P40
	point(0,0,1); // P41
	edge(38,39); // E42
	edge(38,40); // E42
	edge(38,41); // E42

	if (f_v) {
		cout << "scene::create_Hilbert_model done" << endl;
	}
}


void scene::create_affine_space(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "scene::create_affine_space" << endl;
	}
	numerics N;
	int x, y, z;
	double half = 3.6 / sqrt(3);
	double dx = 2 * half / (double) q;
	double dx_half = dx * 0.5;
	//double center[3];
	double basis[9];
	double v[3];

	basis[0] = sqrt(6) / 3.;
	basis[1] = -1 * sqrt(6) / 6.;
	basis[2] = -1 * sqrt(6) / 6.;
	basis[3] = 0.;
	basis[4] = sqrt(2) * 0.5;
	basis[5] = -1 * sqrt(2) * 0.5;
	basis[6] = -7./15.;
	basis[7] = -7./15.;
	basis[8] = -7./15.;


	cout << "dx = " << dx << endl;
	cout << "dx_half = " << dx_half << endl;
	//center[0] = -half;
	//center[1] = -half;
	//center[2] = -half;

	f_has_affine_space = TRUE;
	affine_space_q = 0;
	affine_space_starting_point = nb_points;

	cout << "start_point for affine space over F_" << q
			<< "=" << affine_space_starting_point << endl;

	for (x = 0; x < q; x++) {
		for (y = 0; y < q; y++) {
			for (z = 0; z < q; z++) {
				N.vec_linear_combination3(
						-half + x * dx + dx_half, basis + 0,
						-half + y * dx + dx_half, basis + 3,
						-half + z * dx + dx_half, basis + 6,
						v, 3);

				point(v[0], v[1], v[2]);
			}
		}
	}

	if (f_v) {
		cout << "scene::create_affine_space done" << endl;
	}
}

#if 0
void scene::create_surface_13_1(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "scene::create_surface_13_1" << endl;
	}
	int q = 13;
	int Pts[] = {
		0, 0, 0, 1,
		1, 1, 1, 1,
		8, 0, 0, 1,
		0, 1, 0, 1,
		12, 1, 0, 1,
		0, 2, 0, 1,
		3, 2, 0, 1,
		0, 3, 0, 1,
		7, 3, 0, 1,
		0, 4, 0, 1,
		11, 4, 0, 1,
		0, 5, 0, 1,
		2, 5, 0, 1,
		0, 6, 0, 1,
		6, 6, 0, 1,
		0, 7, 0, 1,
		10, 7, 0, 1,
		0, 8, 0, 1,
		1, 8, 0, 1,
		0, 9, 0, 1,
		5, 9, 0, 1,
		0, 10, 0, 1,
		9, 10, 0, 1,
		0, 11, 0, 1,
		0, 12, 0, 1,
		4, 12, 0, 1,
		0, 0, 1, 1,
		5, 0, 1, 1,
		12, 1, 1, 1,
		2, 2, 1, 1,
		6, 2, 1, 1,
		0, 3, 1, 1,
		3, 3, 1, 1,
		4, 4, 1, 1,
		7, 4, 1, 1,
		1, 5, 1, 1,
		5, 5, 1, 1,
		6, 6, 1, 1,
		8, 6, 1, 1,
		2, 7, 1, 1,
		7, 7, 1, 1,
		8, 8, 1, 1,
		9, 8, 1, 1,
		3, 9, 1, 1,
		9, 9, 1, 1,
		10, 10, 1, 1,
		4, 11, 1, 1,
		11, 11, 1, 1,
		11, 12, 1, 1,
		12, 12, 1, 1,
		0, 0, 2, 1,
		2, 0, 2, 1,
		3, 1, 2, 1,
		11, 1, 2, 1,
		6, 2, 2, 1,
		7, 2, 2, 1,
		3, 3, 2, 1,
		9, 3, 2, 1,
		12, 4, 2, 1,
		2, 5, 2, 1,
		8, 5, 2, 1,
		4, 6, 2, 1,
		5, 6, 2, 1,
		0, 7, 2, 1,
		8, 7, 2, 1,
		9, 8, 2, 1,
		11, 8, 2, 1,
		1, 9, 2, 1,
		5, 9, 2, 1,
		1, 10, 2, 1,
		4, 10, 2, 1,
		7, 11, 2, 1,
		10, 11, 2, 1,
		6, 12, 2, 1,
		10, 12, 2, 1,
		0, 0, 3, 1,
		12, 0, 3, 1,
		5, 1, 3, 1,
		10, 1, 3, 1,
		7, 2, 3, 1,
		11, 2, 3, 1,
		4, 3, 3, 1,
		1, 4, 3, 1,
		10, 4, 3, 1,
		3, 5, 3, 1,
		11, 5, 3, 1,
		8, 6, 3, 1,
		9, 6, 3, 1,
		2, 7, 3, 1,
		5, 7, 3, 1,
		2, 8, 3, 1,
		8, 8, 3, 1,
		1, 9, 3, 1,
		12, 9, 3, 1,
		7, 10, 3, 1,
		9, 10, 3, 1,
		0, 11, 3, 1,
		6, 11, 3, 1,
		3, 12, 3, 1,
		6, 12, 3, 1,
		0, 0, 4, 1,
		9, 0, 4, 1,
		7, 1, 4, 1,
		9, 1, 4, 1,
		0, 2, 4, 1,
		10, 2, 4, 1,
		1, 5, 4, 1,
		4, 5, 4, 1,
		5, 6, 4, 1,
		7, 6, 4, 1,
		1, 7, 4, 1,
		5, 7, 4, 1,
		10, 9, 4, 1,
		4, 11, 4, 1,
		0, 0, 5, 1,
		6, 0, 5, 1,
		8, 1, 5, 1,
		9, 1, 5, 1,
		4, 3, 5, 1,
		9, 3, 5, 1,
		4, 5, 5, 1,
		5, 5, 5, 1,
		0, 6, 5, 1,
		7, 6, 5, 1,
		6, 9, 5, 1,
		8, 9, 5, 1,
		5, 10, 5, 1,
		7, 10, 5, 1,
		0, 0, 6, 1,
		3, 0, 6, 1,
		7, 1, 6, 1,
		11, 1, 6, 1,
		6, 5, 6, 1,
		7, 5, 6, 1,
		2, 9, 6, 1,
		6, 9, 6, 1,
		0, 10, 6, 1,
		10, 10, 6, 1,
		2, 11, 6, 1,
		10, 11, 6, 1,
		3, 12, 6, 1,
		11, 12, 6, 1,
		0, 0, 7, 1,
		0, 1, 7, 1,
		6, 1, 7, 1,
		2, 2, 7, 1,
		10, 2, 7, 1,
		7, 5, 7, 1,
		10, 5, 7, 1,
		2, 8, 7, 1,
		7, 8, 7, 1,
		4, 9, 7, 1,
		11, 9, 7, 1,
		4, 10, 7, 1,
		0, 0, 8, 1,
		10, 0, 8, 1,
		2, 1, 8, 1,
		5, 1, 8, 1,
		1, 4, 8, 1,
		10, 4, 8, 1,
		0, 5, 8, 1,
		8, 5, 8, 1,
		7, 7, 8, 1,
		8, 7, 8, 1,
		2, 9, 8, 1,
		7, 9, 8, 1,
		1, 10, 8, 1,
		5, 10, 8, 1,
		0, 0, 9, 1,
		7, 0, 9, 1,
		4, 1, 9, 1,
		12, 4, 9, 1,
		3, 5, 9, 1,
		9, 5, 9, 1,
		4, 6, 9, 1,
		9, 6, 9, 1,
		0, 9, 9, 1,
		3, 9, 9, 1,
		7, 12, 9, 1,
		12, 12, 9, 1,
		0, 0, 10, 1,
		4, 0, 10, 1,
		3, 1, 10, 1,
		6, 1, 10, 1,
		3, 2, 10, 1,
		11, 2, 10, 1,
		7, 3, 10, 1,
		12, 3, 10, 1,
		4, 4, 10, 1,
		7, 4, 10, 1,
		6, 5, 10, 1,
		10, 5, 10, 1,
		11, 9, 10, 1,
		12, 9, 10, 1,
		0, 0, 11, 1,
		1, 0, 11, 1,
		2, 1, 11, 1,
		8, 1, 11, 1,
		0, 4, 11, 1,
		11, 4, 11, 1,
		9, 5, 11, 1,
		11, 5, 11, 1,
		1, 8, 11, 1,
		7, 8, 11, 1,
		8, 9, 11, 1,
		9, 9, 11, 1,
		2, 11, 11, 1,
		7, 11, 11, 1,
		0, 0, 12, 1,
		11, 0, 12, 1,
		1, 1, 12, 1,
		10, 1, 12, 1,
		12, 3, 12, 1,
		12, 5, 12, 1,
		1, 7, 12, 1,
		10, 7, 12, 1,
		0, 8, 12, 1,
		11, 8, 12, 1,
		4, 9, 12, 1,
		7, 9, 12, 1,
		4, 12, 12, 1,
		7, 12, 12, 1,
	};
	int nb_affine_pts = 222;
	int i, x, y, z;
	double half = 3.6 / sqrt(3);
	double dx = 2 * half / (double) q;
	double dx_half = dx * 0.5;
	double center[3];

	cout << "dx = " << dx << endl;
	cout << "dx_half = " << dx_half << endl;
	center[0] = -half;
	center[1] = -half;
	center[2] = -half;

	int start_point = nb_points;

	cout << "start_point=" << start_point << endl;

	for (i = 0; i < nb_affine_pts; i++) {
		x = Pts[i * 4 + 0];
		y = Pts[i * 4 + 1];
		z = Pts[i * 4 + 2];
		point(center[0] + x * dx + dx_half,
				center[1] + y * dx + dx_half,
				center[2] + z * dx + dx_half);
	}

	if (f_v) {
		cout << "scene::create_surface_13_1 done" << endl;
	}
}
#endif

}
}

