// scene.cpp
//
// Anton Betten
//
// started:  February 13, 2018




#include "foundations.h"

using namespace std;



#define EPSILON 0.01

namespace orbiter {
namespace foundations {



static double Hilbert_Cohn_Vossen_Lines[] = {
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
	0.,0.,1., 1.,0.,0., // 14, 2 c14 at infinity
	-4./5., 3./5., 0., 0., -1., 1., // 15, 3 c15
	-3./5., 0., -4./5., -1., 1., 0., // 16, 4 c16
	-3./5., 4./5., 0., 1., 0., 1., // 17, 5 c23
	-4./5., -3./5., 0., 0., -1., 1., // 18, 6 c24
	0., 1., 0., 1., 0., 0., // 19, 7 c25 at infinity
	3./5., -4./5., 0., -1., 0., 1., // 20, 8 c26
	3./5., 0., -4./5., -1., 1., 0., // 21, 9 c34
	-3./5., -4./5., 0., -1., 0., 1., // 22, 10 c35
	0., 0., 1., 0., 1., 0., // 23, 11 c36 at infinity
	4./5., -3./5., 0., 0., 1., 1., // 24, 12 c45
	-3./5., 0., 4./5., 1., 1., 0., // 25, 13 c46
	3./5., 4./5., 0., 1., 0., 1.,  // 26, 14 c56
	};


static double Hilbert_Cohn_Vossen_tritangent_planes[] = {
	// tritangent planes in dual coordinates, computed using Maple:

	-1, -2, 2, 2, // 0 = start of tritangent planes
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

scene::scene()
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
	Quartic_coords = NULL;
	nb_quartics = 0;
	Octic_coords = NULL;
	nb_octics = 0;
	Cubic_coords = NULL;
	nb_faces = 0;
	Nb_face_points = NULL;
	Face_points = NULL;

	extra_data = NULL;

	f_has_affine_space = FALSE;
	affine_space_q = 0;
	affine_space_starting_point = 0;

	nb_groups = 0;
	//null();
}

scene::~scene()
{
	freeself();
}

void scene::null()
{
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
	if (Quartic_coords) {
		delete [] Quartic_coords;
	}
	if (Octic_coords) {
		delete [] Octic_coords;
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

double scene::label(int idx, const char *txt)
{
	if (idx >= nb_points) {
		cout << "scene::label idx >= nb_points, "
				"idx=" << idx << " nb_points=" << nb_points << endl;
		exit(1);
	}

	{
	pair<int, string> P(idx, txt);
	Labels.push_back(P);
	}
	return Labels.size() - 1;
}

double scene::point_coords(int idx, int j)
{
	if (idx >= nb_points) {
		cout << "scene::point_coords idx >= nb_points, "
				"idx=" << idx << " nb_points=" << nb_points << endl;
		exit(1);
	}
	if (j >= 3) {
		cout << "scene::point_coords j >= 3, "
				"j=" << j << endl;
		exit(1);
	}
	return Point_coords[idx * 3 + j];
}

double scene::line_coords(int idx, int j)
{
	if (idx >= nb_lines) {
		cout << "scene::line_coords idx >= nb_lines, "
				"idx=" << idx << " nb_lines=" << nb_lines << endl;
		exit(1);
	}
	if (j >= 6) {
		cout << "scene::line_coords j >= 6, "
				"j=" << j << endl;
		exit(1);
	}
	return Line_coords[idx * 6 + j];
}

double scene::plane_coords(int idx, int j)
{
	if (idx >= nb_planes) {
		cout << "scene::plane_coords idx >= nb_planes, "
				"idx=" << idx << " nb_planes=" << nb_planes << endl;
		exit(1);
	}
	if (j >= 3) {
		cout << "scene::plane_coords j >= 4, "
				"j=" << j << endl;
		exit(1);
	}
	return Plane_coords[idx * 4 + j];
}

double scene::cubic_coords(int idx, int j)
{
	if (idx >= nb_cubics) {
		cout << "scene::cubic_coords idx >= nb_cubics, "
				"idx=" << idx << " nb_cubics=" << nb_cubics << endl;
		exit(1);
	}
	if (j >= 20) {
		cout << "scene::cubic_coords j >= 20, "
				"j=" << j << endl;
		exit(1);
	}
	return Cubic_coords[idx * 20 + j];
}

double scene::quadric_coords(int idx, int j)
{
	if (idx >= nb_quadrics) {
		cout << "scene::quadric_coords idx >= nb_quadrics, "
				"idx=" << idx << " nb_quadrics=" << nb_quadrics << endl;
		exit(1);
	}
	if (j >= 10) {
		cout << "scene::quadric_coords j >= 10, "
				"j=" << j << endl;
		exit(1);
	}
	return Quadric_coords[idx * 10 + j];
}

int scene::edge_points(int idx, int j)
{
	if (idx >= nb_planes) {
		cout << "scene::edge_points idx >= nb_edges, "
				"idx=" << idx << " nb_edges=" << nb_edges << endl;
		exit(1);
	}
	if (j >= 2) {
		cout << "scene::edge_points j >= 2, "
				"j=" << j << endl;
		exit(1);
	}
	return Edge_points[idx * 2 + j];
}

void scene::print_point_coords(int idx)
{
	int j;

	for (j = 0; j < 3; j++) {
		cout << Point_coords[idx * 3 + j] << "\t";
	}
	cout << endl;
}

double scene::point_distance_euclidean(int pt_idx, double *y)
{
	numerics Num;
	double d;

	d = Num.distance_euclidean(y, Point_coords + pt_idx * 3, 3);
	return d;
}

double scene::point_distance_from_origin(int pt_idx)
{
	numerics Num;
	double d;

	d = Num.distance_from_origin(Point_coords + pt_idx * 3, 3);
	return d;
}

double scene::distance_euclidean_point_to_point(int pt1_idx, int pt2_idx)
{
	numerics Num;
	double d;

	d = Num.distance_euclidean(Point_coords + pt1_idx * 3, Point_coords + pt2_idx * 3, 3);
	return d;
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
	Quartic_coords = new double [SCENE_MAX_QUARTICS * 35];
	nb_quartics = 0;
	Octic_coords = new double [SCENE_MAX_OCTICS * 165];
	nb_octics = 0;
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
		if (N.line_centered(xx, yy, xxx, yyy, 10., verbose_level - 1)) {
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

		N.substitute_cubic_linear_using_povray_ordering(coeff_in, coeff_out,
			A4_inv, verbose_level);

		S->cubic(coeff_out);
		}

	if (f_v) {
		cout << "scene::transform_cubics done" << endl;
		}

}

void scene::transform_quartics(scene *S, double *A4, double *A4_inv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "scene::transform_quartics" << endl;
		}

	cout << "scene::transform_quartics not yet implemented" << endl;

	double coeff_in[35];
	double coeff_out[35];
	int i;
	numerics N;


	for (i = 0; i < nb_quartics; i++) {
		N.vec_copy(Quartic_coords + i * 35, coeff_in, 35);

		N.substitute_quartic_linear_using_povray_ordering(coeff_in, coeff_out,
			A4_inv, verbose_level);

		S->quartic(coeff_out);
		}

	if (f_v) {
		cout << "scene::transform_quartics done" << endl;
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
		return nb_lines - 1;
		}
	else {
		return -1;
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

int scene::line_after_recentering(double x1, double x2, double x3,
	double y1, double y2, double y3, double rad)
{
	double x[3], y[3];
	double xx[3], yy[3];
	numerics N;
	int verbose_level = 0;

	x[0] = x1;
	x[1] = x2;
	x[2] = x3;
	y[0] = y1;
	y[1] = y2;
	y[2] = y3;

	N.line_centered(x, y, xx, yy, rad, verbose_level);

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

int scene::line_through_two_points(int pt1, int pt2, double rad)
{
	double x[3], y[3];
	double xx[3], yy[3];
	numerics N;
	int verbose_level = 0;

	N.vec_copy(Point_coords + pt1 * 3, x, 3);
	N.vec_copy(Point_coords + pt2 * 3, y, 3);

	N.line_centered(x, y, xx, yy, 10., verbose_level);

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

int scene::octic(double *coeff_165)
{
	int i;

	for (i = 0; i < 165; i++) {
		Octic_coords[nb_octics * 165 + i] = coeff_165[i];
		}
	nb_octics++;
	if (nb_octics >= SCENE_MAX_OCTICS) {
		cout << "too many octics" << endl;
		exit(1);
		}
	return nb_octics - 1;

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

int scene::cubic(double *coeff)
// povray ordering of monomials:
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

int scene::quartic(double *coeff)
// povray ordering of monomials:
// http://www.povray.org/documentation/view/3.6.1/298/
{
	int i;

	for (i = 0; i < 35; i++) {
		Quartic_coords[nb_quartics * 35 + i] = coeff[i];
		}
	nb_quartics++;
	if (nb_quartics >= SCENE_MAX_QUARTICS) {
		cout << "too many quartics" << endl;
		exit(1);
		}
	return nb_quartics - 1;
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
	ost << "	        #declare r=" << line_radius << "; " << endl;
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

void scene::draw_lines_cij_with_offset(int offset, int number_of_lines, ostream &ost)
{
	int selection[15];
	int i;

	for (i = 0; i < number_of_lines; i++) {
		selection[i] = offset + i;
	}
	draw_lines_cij_with_selection(selection, number_of_lines, ost);
}

void scene::draw_lines_ai_with_selection(int *selection, int nb_select, 
	ostream &ost)
{
	int s, i, j, h;
	numerics N;
		
	ost << endl;
	ost << "	union{ // ai lines" << endl;
	ost << endl;
	ost << "	        #declare r=" << line_radius << "; " << endl;
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

void scene::draw_lines_ai_with_offset(int offset, ostream &ost)
{
	int selection[6];
	int i;

	for (i = 0; i < 6; i++) {
		selection[i] = offset + i;
	}
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
	ost << "	        #declare r=" << line_radius << "; " << endl;
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
		ost << ">, r } // line " << j << endl;
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

void scene::draw_lines_bj_with_offset(int offset, ostream &ost)
{
	int selection[6];
	int i;

	for (i = 0; i < 6; i++) {
		selection[i] = offset + i;
	}
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
		ost << "\t\t// face " << j << ":" << endl;
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

void scene::draw_plane(
	int idx,
	const char *options, ostream &ost)
// for instance color = "Orange transmit 0.5 "
{
	int h;
	numerics N;

	ost << endl;
	ost << endl;
	ost << "	object{" << endl;
	ost << "		plane{<";
	for (h = 0; h < 3; h++) {
		N.output_double(Plane_coords[idx * 4 + h], ost);
		if (h < 2) {
			ost << ", ";
			}
		}
	ost << ">, ";
	N.output_double(Plane_coords[idx * 4 + 3], ost);
	ost << "}" << endl;
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

		cout << "scene::draw_cubic_with_selection j=" << j << ":" << endl;
		for (h = 0; h < 20; h++) {
			cout << h << " : " << Cubic_coords[j * 20 + h] << endl;
		}
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

void scene::draw_quartic_with_selection(int *selection, int nb_select,
	const char *options, ostream &ost)
{
	int i, j, h, s;
	numerics N;

	ost << endl;
	ost << "	union{ // quartics" << endl;
	ost << endl;
	for (i = 0; i < nb_select; i++) {
		s = selection[i];
		j = s;

		cout << "scene::draw_quartic_with_selection j=" << j << ":" << endl;
		for (h = 0; h < 35; h++) {
			cout << h << " : " << Quartic_coords[j * 35 + h] << endl;
		}
		ost << "		poly{4, <";

		for (h = 0; h < 35; h++) {
			N.output_double(Quartic_coords[j * 35 + h], ost);
			if (h < 35 - 1) {
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

void scene::draw_octic_with_selection(int *selection, int nb_select,
	const char *options, ostream &ost)
{
	int i, j, h, s;
	numerics N;

	ost << endl;
	ost << "	union{ // octics" << endl;
	ost << endl;
	for (i = 0; i < nb_select; i++) {
		s = selection[i];
		j = s;

		cout << "scene::draw_quartic_with_selection j=" << j << ":" << endl;
		for (h = 0; h < 165; h++) {
			cout << h << " : " << Quartic_coords[j * 165 + h] << endl;
		}
		ost << "		poly{8, <";

		for (h = 0; h < 165; h++) {
			N.output_double(Octic_coords[j * 165 + h], ost);
			if (h < 165 - 1) {
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
	os_interface Os;

	if (f_v) {
		cout << "scene::intersect_line_and_plane" << endl;
		}
	if (f_v) {
		cout << "scene::intersect_line_and_plane line_idx=" << line_idx << endl;
		cout << "scene::intersect_line_and_plane plane_idx=" << plane_idx << endl;
		print_a_line(line_idx);
		print_a_plane(plane_idx);
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
	// -P_0
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
	
	if (f_v) {
		cout << "scene::intersect_line_and_plane M=" << endl;
		numerics Num;

		Num.print_system(M, 3, 4);
		}

	// compute the vectors u and w:
	B[0 * 3 + 0] = Plane_coords[plane_idx * 4 + 0];
	B[0 * 3 + 1] = Plane_coords[plane_idx * 4 + 1];
	B[0 * 3 + 2] = Plane_coords[plane_idx * 4 + 2];

	int b1, b2;
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
		b1 = 1;
		b2 = 2;
	}
	else {
		if (f_v) {
			cout << "scene::intersect_line_and_plane "
					"ABS(B[0 * 3 + 0]) is too small" << endl;
		}
		if (ABS(B[0 * 3 + 1]) > EPSILON) {
			a = 1. / B[0 * 3 + 1];
			for (i = 0; i < 3; i++) {
				B[3 + i] = B[i] * a;
			}
			B[0 * 3 + 0] = -1.;
			B[0 * 3 + 1] = 0;
			B[0 * 3 + 2] = 0.;
			B[2 * 3 + 0] = 0.;
			B[2 * 3 + 1] = 0.;
			B[2 * 3 + 2] = -1.;
			b1 = 0;
			b2 = 2;
		}
		else {
			if (f_v) {
				cout << "scene::intersect_line_and_plane "
						"ABS(B[0 * 3 + 0]) and ABS(B[0 * 3 + 1]) are too small" << endl;
			}
			if (ABS(B[0 * 3 + 2]) > EPSILON) {
				a = 1. / B[0 * 3 + 2];
				for (i = 0; i < 3; i++) {
					B[6 + i] = B[i] * a;
				}
				B[0 * 3 + 0] = -1.;
				B[0 * 3 + 1] = 0;
				B[0 * 3 + 2] = 0.;
				B[1 * 3 + 0] = 0.;
				B[1 * 3 + 1] = -1.;
				B[1 * 3 + 2] = 0;
				b1 = 0;
				b2 = 1;
			}
			else {
				cout << "scene::intersect_line_and_plane the first three entries of B are zero" << endl;
				exit(1);
			}
		}
	}
	// copy u:
	M[0 * 4 + 1] = -1. * B[0 * 3 + b1];
	M[1 * 4 + 1] = -1. * B[1 * 3 + b1];
	M[2 * 4 + 1] = -1. * B[2 * 3 + b1];
	// copy w:
	M[0 * 4 + 2] = -1. * B[0 * 3 + b2];
	M[1 * 4 + 2] = -1. * B[1 * 3 + b2];
	M[2 * 4 + 2] = -1. * B[2 * 3 + b2];

	// find Q_0:

	while (TRUE) {
		B[0] = (double) Os.random_integer(5);
		B[1] = (double) Os.random_integer(5);
		B[2] = (double) Os.random_integer(5);
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
	//int f_vv = FALSE; // (verbose_level >= 2);
	numerics N;

	if (f_v) {
		cout << "scene::intersect_line_and_line" << endl;
		}

#if 1
	double pt_coords[3];

	N.intersect_line_and_line(
			&Line_coords[line1_idx * 6 + 0], &Line_coords[line1_idx * 6 + 3],
			&Line_coords[line2_idx * 6 + 0], &Line_coords[line2_idx * 6 + 3],
			lambda,
			pt_coords,
			verbose_level - 2);
	point(pt_coords[0], pt_coords[1], pt_coords[2]);
#else
	double B[3];
	double M[9];
	int i;
	double v[3];

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
#endif
	
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
			line_pt1, line_pt2, spread /* r */, verbose_level - 1);
	
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

	// opposite pairs are :
	// 0, 7
	// 1, 6
	// 2, 5
	// 3, 4
	// 8, 11
	// 9, 10
	// 12, 15
	// 13, 14
	// 16, 19
	// 17, 18

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


//##############################################################################
// Clebsch cubic, version 1
//##############################################################################

void scene::clebsch_cubic()
{
	double coeff[20] = {-3,7,7,1,7,-2,-14,7,-14,3,-3,7,1,7,-14,3,-3,1,3,-1.};

	cubic(coeff);
}

void scene::clebsch_cubic_lines_a()
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


void scene::clebsch_cubic_lines_b()
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

void scene::clebsch_cubic_lines_cij()
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

void scene::Clebsch_Eckardt_points()
{
	point(1,1,1. ); //0
	point(0,-1,0. ); //1
	point(.5,.5,0. ); //2
	point(0,.5,.5 ); //3
	point(0,0,-1. ); //4
	point(.5,0,.5 ); //5
	point(-1,0,0. ); //6
}

//##############################################################################
// Clebsch cubic, version 2
//##############################################################################

void scene::clebsch_cubic_version2()
{
	// this is X0^3+X1^3+X2^3+X3^3-(X0+X1+X2+X3)^3
	// = -3*x^2*y - 3*x^2*z - 3*x*y^2 - 6*x*y*z - 3*x*z^2 - 3*y^2*z
	// - 3*y*z^2 - 3*x^2 - 6*x*y - 6*x*z - 3*y^2 - 6*y*z - 3*z^2 - 3*x - 3*y - 3*z


	double Eqn[20] = {
			0, // 1 x^3
			-3, // 2 x^2y
			-3, // 3 x^2z
			-3, // 4 x^2
			-3, // 5 xy^2
			-6, // 6 xyz
			-6, // 7 xy
			-3, // 8 xz^2
			-6, // 9 xz
			-3, // 10 x
			0, // 11 y^3
			-3, // 12 y^2z
			-3, // 13 y^2
			-3, // 14 yz^2
			-6, // 15 yz
			-3, // 16 y
			0, // 17 z^3
			-3, // 18 z^2
			-3, // 19 z
			0, // 20 1
	};

	cubic(Eqn);
}

void scene::clebsch_cubic_version2_Hessian()
{

	double Eqn[20] = {
			0, // 1 x^3
			1, // 2 x^2y
			1, // 3 x^2z
			0, // 4 x^2
			1, // 5 xy^2
			2, // 6 xyz
			1, // 7 xy
			1, // 8 xz^2
			1, // 9 xz
			0, // 10 x
			0, // 11 y^3
			1, // 12 y^2z
			0, // 13 y^2
			1, // 14 yz^2
			1, // 15 yz
			0, // 16 y
			0, // 17 z^3
			0, // 18 z^2
			0, // 19 z
			0, // 20 1
	};

	cubic(Eqn);
}

#define alpha ((1. + sqrt(5)) * .5)
#define beta ((-1. + sqrt(5)) * .5)

void scene::clebsch_cubic_version2_lines_a()
{
	double A[] = {beta, -1, 0, 1, -1, beta, 1, 0,
		beta, -1, 1, -beta, -1, beta, 0, -beta,
		beta, -1, -beta, 0, -1, beta, -beta, 1,
		-1, -alpha, 0, 1, -alpha, -1, 1, 0,
		-1, -alpha, 1, alpha, -alpha, -1, 0, alpha,
		-1, -alpha, alpha, 0, -alpha, -1, alpha, 1};
	int i, j;
	double L[8];
	double a, av;
	numerics N;

	for (i = 0; i < 6; i++) {
		N.vec_copy(A + i * 8, L, 8);
		if (ABS(L[3]) < 0.001 && ABS(L[7]) < 0.001) {
			cout << "scene::clebsch_cubic_version2_lines_a line A" << i << " lies at infinity" << endl;
			exit(1);
		}
		if (ABS(L[7]) > ABS(L[3])) {
			N.vec_swap(L, L + 4, 4);
		}
		a = L[3];
		av = 1. / a;
		N.vec_scalar_multiple(L, av, 4);
		a = -L[7];
		for (j = 0; j < 4; j++) {
			L[4 + j] += L[j] * a;
		}
		line_extended(L[0], L[1], L[2], L[0] + L[4], L[1] + L[5], L[2] + L[6], 10.);
	}
}

void scene::clebsch_cubic_version2_lines_b()
{
	double B[] = {-1, -alpha, 1, 0, -alpha, -1, 0, 1,
			-1, -alpha, 0, alpha, -alpha, -1, 1, alpha,
        	-1, -alpha, alpha, 1, -alpha, -1, alpha, 0,
        	-1, beta, 0, 1, beta, -1, 1, 0,
        	-1, beta, 1, -beta, beta, -1, 0, -beta,
        	-1, beta, -beta, 0, beta, -1, -beta, 1};
	int i, j;
	double L[8];
	double a, av;
	numerics N;

	for (i = 0; i < 6; i++) {
		N.vec_copy(B + i * 8, L, 8);
		if (ABS(L[3]) < 0.001 && ABS(L[7]) < 0.001) {
			cout << "scene::clebsch_cubic_version2_lines_b line B" << i << " lies at infinity" << endl;
			exit(1);
		}
		if (ABS(L[7]) > ABS(L[3])) {
			N.vec_swap(L, L + 4, 4);
		}
		a = L[3];
		av = 1. / a;
		N.vec_scalar_multiple(L, av, 4);
		a = -L[7];
		for (j = 0; j < 4; j++) {
			L[4 + j] += L[j] * a;
		}
		line_extended(L[0], L[1], L[2], L[0] + L[4], L[1] + L[5], L[2] + L[6], 10.);
	}

}

void scene::clebsch_cubic_version2_lines_c()
{
	double C[] = {
		-1,0,0,0,0,-1,0,1,
		-1,0,1,0,0,-1,0,0,
		1,-1,0,0,0,0,1,-1,
		0,0,0,-1,0,1,-1,0,
		0,0,1,0,1,0,0,-1,
		-1,0,0,1,0,-1,1,0,
		0,0,0,-1,1,0,-1,0,
		1,-1,0,0,0,0,1,0,

		0,0,-1,1,0,1,0,0,
		0,0,1,0,0,1,0,-1,
		0,0,-1,1,1,0,0,0,
		1,-1,0,0,0,0,0,1,
		0,-1,0,0,-1,0,0,1,
		0,-1,1,0,-1,0,0,0,
		0,-1,0,1,-1,0,1,0,
		};
	int i, j, h;
	double L[8];
	double a, av;
	numerics N;

	h = 0;
	for (i = 0; i < 15; i++) {
		N.vec_copy(C + i * 8, L, 8);
		if (ABS(L[3]) < 0.001 && ABS(L[7]) < 0.001) {
			cout << "scene::clebsch_cubic_version2_lines_c line C" << i << " lies at infinity" << endl;
			continue;
		}
		if (ABS(L[7]) > ABS(L[3])) {
			N.vec_swap(L, L + 4, 4);
		}
		a = L[3];
		av = 1. / a;
		N.vec_scalar_multiple(L, av, 4);
		a = -L[7];
		for (j = 0; j < 4; j++) {
			L[4 + j] += L[j] * a;
		}
		line_extended(L[0], L[1], L[2], L[0] + L[4], L[1] + L[5], L[2] + L[6], 10.);
		h++;
	}


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

void scene::create_Hilbert_Cohn_Vossen_surface(int verbose_level)
// 1 cubic, 27 lines, 54 points, 45 planes
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "scene::create_Hilbert_Cohn_Vossen_surface" << endl;
	}


	if (f_v) {
		cout << "scene::create_Hilbert_Cohn_Vossen_surface creating lines" << endl;
	}

	int i;
	double r = sqrt(5);
	for (i = 0; i < 27; i++) {

		if (f_v) {
			cout << "scene::create_Hilbert_Cohn_Vossen_surface creating line" << i << endl;
		}
		line_pt_and_dir_and_copy_points(Hilbert_Cohn_Vossen_Lines + i * 6, r, verbose_level - 1);
		}

	{
	double coeff[20] = {0,0,0,-1,0,5./2.,0,0,0,0,0,0,-1,0,0,0,0,-1,0,1};
	cubic(coeff);  // cubic 0
	}


	for (i = 0; i < 45; i++) {
		plane_from_dual_coordinates(Hilbert_Cohn_Vossen_tritangent_planes + i * 4);
		}





	if (f_v) {
		cout << "scene::create_Hilbert_Cohn_Vossen_surface done" << endl;
	}
}


void scene::create_Hilbert_model(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "scene::create_Hilbert_model" << endl;
	}

	verbose_level += 2;

	numerics N;
	int i;

	if (f_v) {
		cout << "scene::create_Hilbert_model before create_Hilbert_cube" << endl;
	}
	create_Hilbert_cube(verbose_level);
	if (f_v) {
		cout << "scene::create_Hilbert_model after create_Hilbert_cube" << endl;
	}

#if 0
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
#endif
	
	if (f_v) {
		cout << "scene::create_Hilbert_model creating edges for double six" << endl;
	}
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

	if (f_v) {
		cout << "scene::create_Hilbert_model creating the terahedron" << endl;
	}
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



	if (f_v) {
		cout << "scene::create_Hilbert_model creating lines" << endl;
	}

	double r = sqrt(5);
	for (i = 0; i < 27; i++) {

		if (f_v) {
			cout << "scene::create_Hilbert_model creating line" << i << endl;
		}
		line_pt_and_dir(Hilbert_Cohn_Vossen_Lines + i * 6, r, verbose_level - 1);
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

	if (f_v) {
		cout << "scene::create_Hilbert_model before substitute_cubic_linear_using_povray_ordering" << endl;
	}
	N.substitute_cubic_linear_using_povray_ordering(
			coeff_in, coeff_out,
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

	if (f_v) {
		cout << "scene::create_Hilbert_model lines on Cayley's nodal" << endl;
	}
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
	clebsch_cubic_lines_a();
	clebsch_cubic_lines_b();
	clebsch_cubic_lines_cij();


	if (f_v) {
		cout << "scene::create_Hilbert_model fermat" << endl;
	}
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
		line_pt_and_dir(Lines + i * 6, r, verbose_level - 1);
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


	if (f_v) {
		cout << "scene::create_Hilbert_model create lines on Cayley's ruled surface" << endl;
	}
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
			if (line_pt_and_dir(Line, r, verbose_level - 1)) {
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

	if (f_v) {
		cout << "scene::create_Hilbert_model create osculating_tangent_line" << endl;
	}
	double osculating_tangent_line[6] = {0,0,0,1,0,0};
	line_pt_and_dir(osculating_tangent_line, r, verbose_level - 1); //

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

	if (f_v) {
		cout << "scene::create_Hilbert_model create coeff_surf_lifted" << endl;
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

	if (f_v) {
		cout << "scene::create_Hilbert_model create long lines" << endl;
	}


	// the lines in long:
	cout << "the long lines start at " << nb_lines << endl;
	r = 3.6;
	for (i = 0; i < 27; i++) {
		line_pt_and_dir(Hilbert_Cohn_Vossen_Lines + i * 6, r, verbose_level - 1);
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

void scene::create_Cayleys_nodal_cubic(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	numerics Num;
	int idx;

	if (f_v) {
		cout << "scene::create_Cayleys_nodal_cubic" << endl;
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

	Num.matrix_double_inverse(A4, A4_inv, 4, 0 /* verbose_level*/);

	Num.substitute_cubic_linear_using_povray_ordering(
			coeff_in, coeff_out,
			A4_inv, 0 /*verbose_level*/);

	if (f_v) {
		int i;

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
	idx = cubic(coeff_out); // cubic 1
	cout << "created cubic " << idx << endl;

	// pts of the tetrahedron: 0,6,3,5
	int pts[4] = {0, 6, 3, 5};
	double rad = 5.;


	// create lines 27-32 on Cayley's nodal cubic  (previously: 15,16,17,18,19,20)
	idx = line_through_two_points(pts[0], pts[1], rad); // line 27
	cout << "created line " << idx << endl;
	idx = line_through_two_points(pts[0], pts[2], rad);
	cout << "created line " << idx << endl;
	idx = line_through_two_points(pts[0], pts[3], rad);
	cout << "created line " << idx << endl;
	idx = line_through_two_points(pts[1], pts[2], rad);
	cout << "created line " << idx << endl;
	idx = line_through_two_points(pts[1], pts[3], rad);
	cout << "created line " << idx << endl;
	idx = line_through_two_points(pts[2], pts[3], rad);
	cout << "created line " << idx << endl;

#if 0
	pts[4];
	for (i = 0; i < 4; i++) {
		pts[0] = point(A4[0 * 4 + 0], A4[0 * 4 + 1], A4[0 * 4 + 2]);
		}
#endif

	}

	if (f_v) {
		cout << "scene::create_Cayleys_nodal_cubic done" << endl;
	}
}

void scene::create_Hilbert_cube(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "scene::create_Hilbert_cube" << endl;
	}
	numerics N;
	double p = 1.;
	double m = -1.;
	double px = p + 1;
	double mx = m - 1;


	if (f_v) {
		cout << "scene::create_Hilbert_cube before create_cube" << endl;
	}
	create_cube(verbose_level);
	if (f_v) {
		cout << "scene::create_Hilbert_cube after create_cube" << endl;
	}

#if 0
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
#endif

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
	if (f_v) {
		cout << "scene::create_Hilbert_cube done" << endl;
	}
}

void scene::create_cube(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "scene::create_cube" << endl;
	}
	numerics N;
	double p = 1.;
	double m = -1.;

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


	if (f_v) {
		cout << "scene::create_cube done" << endl;
	}
}

void scene::create_cube_and_tetrahedra(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "scene::create_cube_and_tetrahedra" << endl;
	}

	create_cube(verbose_level);

	int i, j;

	for (i = 0; i < 8; i++) {
		for (j = i + 1; j < 8; j++) {
			edge(i, j); // 0
		}
	}



	// create faces:

	combinatorics_domain Combi;

	int set[3];
	int nCk = Combi.int_n_choose_k(8, 3);
	int rk;
	int first_three_face;

	first_three_face = nb_faces; // = 6, because create_cube creates six faces for the cube
	if (f_v) {
		cout << "first_three_face = " << first_three_face << endl;
	}
	for (rk = 0; rk < nCk; rk++) {
		Combi.unrank_k_subset(rk, set, 8 /*n*/, 3 /*k*/);
		face3(set[0], set[1], set[2]);
		if (f_v) {
			cout << "rk=" << rk << " set=";
			int_vec_print(cout, set, 3);
			cout << endl;
		}
	}

	if (f_v) {
		cout << "scene::create_cube_and_tetrahedra done" << endl;
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


void scene::create_HCV_surface(int N, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "scene::create_HCV_surface" << endl;
	}

	//double coeff_in[20] = {0,3,3,5,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,1};
	// by accident, this is the wrong ordering of coefficients!
	// it is the ordering of coefficients in orbiter.
	// But it should be the ordering of coefficients required by povray.


	double coeff_orig[20] = {0,0,0,-1,0,2.5,0,0,0,0,0,0,-1,0,0,0,0,-1,0,1};
	// HCV original
	// 4: -1*x^2
	// 6: 2.5*xyz
	// 13: -1*y^2
	// 18: -1*z^2
	// 20: 1*1


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

	double coeff_trans[20] = {1,0,0,0,-1,0,0,-1,0,-1,0,0,0,0,2.5,0,0,0,0,0};
	// HCV original

	double coeff_out[20];



	cubic(coeff_orig); // cubic 0
	cubic(coeff_trans); // cubic 1

	numerics Num;

	double A4[] = {
		1.,1.,1.,1.,
		-1.,-1.,1,1,
		1.,-1.,-1.,1,
		-1.,1.,-1.,1
		};
	double A4_inv[16];

	Num.matrix_double_inverse(A4, A4_inv, 4, 0 /* verbose_level*/);

	Num.substitute_cubic_linear_using_povray_ordering(
			coeff_orig, coeff_out,
			A4_inv, 0 /*verbose_level*/);



	if (f_v) {
		cout << "i : coeff_orig[i] : coeff_out[i]" << endl;
		for (i = 0; i < 20; i++) {
			cout << i << " : " << coeff_orig[i] << " : " << coeff_out[i] << endl;
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
	cubic(coeff_out); // cubic 2

	double sqrt3 = sqrt(3.);
	double one_over_sqrt3 = 1. / sqrt3;

	plane(-1 * one_over_sqrt3, one_over_sqrt3, -1 * one_over_sqrt3, -1 * one_over_sqrt3); // plane 0


#if 0
	double lines[] = {
			1,1,1,0,1,1,
			1,1,1,1,1,0,
			-1,-1,1,-1,0,1,
		};
	int k;

	k = 0;
	for (i = 0; i < 3; i++) {
		k = 6 * i;
		S->line_extended(
				lines[k + 0], lines[k + 1], lines[k + 2],
				lines[k + 0] + lines[k + 3], lines[k + 1] + lines[k + 4], lines[k + 2] + lines[k + 5], 10); // line i
	}
#endif

#if 0
	// the lines on the original HCV surface:
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
		0.,0.,1., 1.,0.,0., // 14, 2 c14 at infinity
		-4./5., 3./5., 0., 0., -1., 1., // 15, 3 c15
		-3./5., 0., -4./5., -1., 1., 0., // 16, 4 c16
		-3./5., 4./5., 0., 1., 0., 1., // 17, 5 c23
		-4./5., -3./5., 0., 0., -1., 1., // 18, 6 c24
		0., 1., 0., 1., 0., 0., // 19, 7 c25 at infinity
		3./5., -4./5., 0., -1., 0., 1., // 20, 8 c26
		3./5., 0., -4./5., -1., 1., 0., // 21, 9 c34
		-3./5., -4./5., 0., -1., 0., 1., // 22, 10 c35
		0., 0., 1., 0., 1., 0., // 23, 11 c36 at infinity
		4./5., -3./5., 0., 0., 1., 1., // 24, 12 c45
		-3./5., 0., 4./5., 1., 1., 0., // 25, 13 c46
		3./5., 4./5., 0., 1., 0., 1.,  // 26, 14 c56
		};

	double *Lines4_in;
	double *Lines4_out;

	double T[] = {
			1,1,1,1,
			-1,-1,1,1,
			1,-1,-1,1,
			-1,1,-1,1
	};


	Lines4_in = new double[27 * 2 * 4];
	Lines4_out = new double[27 * 2 * 4];
	for (i = 0; i < 27; i++) {
		for (j = 0; j < 3; j++) {
			Lines4_in[i * 8 + j] = Lines[6 * i + j];
			Lines4_in[i * 8 + 4 + j] = Lines[6 * i + 3 + j];
		}
		if (i == 14 || i == 19 || i == 23) {
			Lines4_in[i * 8 + 3] = 0;
			Lines4_in[i * 8 + 4 + 3] = 0;
		}
		else {
			Lines4_in[i * 8 + 3] = 1;
			Lines4_in[i * 8 + 4 + 3] = 0;
		}
	}
	Num.mult_matrix_matrix(
				Lines4_in, T, Lines4_out, 2 * 27, 4, 4);
		// A is m x n, B is n x o, C is m x o
#endif

	double Lines[] = {
			// 0-5:
			// ai:
			-1,0,0,6,1,1,
			-1./6.,-1./6.,0,-1./6.,-1./6.,1,
			0,1,0,-1,6,1,
			1,6,0,0,-1,1,
			-6,0,1,-1,1,0,
			6,-1,0,1,0,1,
			// 6-11:
			// bj:
			1,-6,0,0,-1,1,
			6,0,1,-1,1,0,
			-6,-1,0,1,0,1,
			-1,0,0,-6,1,1,
			1./6.,1./6.,0,1./6.,1./6.,1,
			0,1,0,-1,-6,1,
			//12-26
			// cij:
			0,-1,0,-2,9,1,
			1./9.,2./9.,0,-1./9.,-2./9.,1,
			1,0,0,0,1,1,
			-9./2.,-1./2.,0,-1,0,1,
			9,0,2,1,1,0,
			1,0,0,9./2.,1./2.,1,
			9,-2,0,-1,0,1,
			0,0,1,1,1,0,
			1./2.,9./2.,0,0,1,1,
			-9./2.,0,1./2.,1,1,0,
			2.,-9.,0,0,1,1,
			0,-1,0,-1,0,1,
			0,-1,0,-1./2.,-9./2.,1,
			-2./9.,-1./9.,0,2./9.,1./9.,1,
			1,0,0,-9,2,1
	};
	double r = 10;
	for (i = 0; i < 27; i++) {
		line_pt_and_dir(Lines + i * 6, r, verbose_level - 1);
		}

	//S->line6(lines); // line 0
	//S->line6(lines + 6); // line 0
	//S->line6(lines + 12); // line 0

	point(0,-1,0); // 0 = Eckardt point
	point(1,0,0); // 1 = Eckardt point
	point(0,0,1); // 2 = Eckardt point


	if (f_v) {
		cout << "scene::create_HCV_surface done" << endl;
	}
}

void scene::create_E4_surface(int N, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "scene::create_E4_surface" << endl;
	}

	//double coeff_in[20] = {0,3,3,5,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,1};
	// by accident, this is the wrong ordering of coefficients!
	// it is the ordering of coefficients in orbiter.
	// But it should be the ordering of coefficients required by povray.

	//double coeff_in[20] = {5,3,3,0,0,0,0,0,0,0,0,0,0,2,0,0,1,0,0,0};
	// transformed, but still wrong ordering

	double coeff_orig[20] = {0,0,0,2,0,0,0,0,0,0,3,0,0,0,1,0,3,0,0,5};
	// original, correct ordering.

	double coeff_trans[20] = {5,0,0,0,0,1,0,0,0,2,3,0,0,0,0,0,3,0,0,0};
	// transformed, correct ordering.


	cubic(coeff_orig);
	cubic(coeff_trans);

	double x6[6];
	int line_idx;

	x6[0] = 0;
	x6[1] = 4;
	x6[2] = -4;
	x6[3] = 0;
	x6[4] = -4;
	x6[5] = 4;

	line_idx = line6(x6);
	cout << "line_idx=" << line_idx << endl;


	if (f_v) {
		cout << "scene::create_E4_surface done" << endl;
	}
}

void scene::create_twisted_cubic(int N, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	numerics Num;
	int i;
	double t, x, y, z;

	if (TRUE) {
		cout << "scene::create_twisted_cubic" << endl;
	}


	double p = 1.;
	double m = 0.;

	point(p,p,p); // 0
	point(p,p,m); // 1
	point(p,m,p); // 2
	point(p,m,m); // 3
	point(m,p,p); // 4
	point(m,p,m); // 5
	point(m,m,p); // 6
	point(m,m,m); // 7


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



	for (i = 0; i < N; i++) {
		t = i * 1. / (double) (N - 1);
		x = t;
		y = t * t;
		z = y * t;
		point(x, y, z); // P_8+i
		if (i) {
			edge(8 + i, 8 + i - 1); // E_i-1

		}
	}

}

void scene::create_triangulation_of_cube(int N, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	numerics Num;

	if (f_v) {
		cout << "scene::create_triangulation_of_cube" << endl;
	}


	double p = 1.;
	double m = -1.;

	point(p,p,p); // 0
	point(p,p,m); // 1
	point(p,m,p); // 2
	point(p,m,m); // 3
	point(m,p,p); // 4
	point(m,p,m); // 5
	point(m,m,p); // 6
	point(m,m,m); // 7


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


	face4(0, 1, 5, 4); // F0, front right
	face4(0, 2, 3, 1); // F1, front left
	face4(0, 4, 6, 2); // F2, top
	face4(1, 5, 7, 3); // F3, bottom
	face4(4, 6, 7, 5); // F4, back right
	face4(2, 3, 7, 6); // F5, back left


	face3(0, 3, 5); // F6
	face3(0, 3, 6); // F7
	face3(0, 5, 6); // F8
	face3(3, 5, 6); // F9

	edge(0, 3); // E12
	edge(0, 5); // E13
	edge(0, 6); // E14
	edge(3, 5); // E15
	edge(3, 6); // E16
	edge(5, 6); // E17

}

void scene::print_a_line(int line_idx)
{
	numerics Num;

	cout << "Line " << line_idx << " is ";
	Num.vec_print(Line_coords + line_idx * 6 + 0, 3);
	cout << " - ";
	Num.vec_print(Line_coords + line_idx * 6 + 3, 3);
	cout << endl;
}


void scene::print_a_plane(int plane_idx)
{
	numerics Num;

	cout << "Plane " << plane_idx << " : ";
	Num.vec_print(Plane_coords + plane_idx * 4, 4);
	cout << endl;
}

void scene::print_a_face(int face_idx)
{
	cout << "face " << face_idx << " has " << Nb_face_points[face_idx] << " points: ";
	int_vec_print(cout, Face_points[face_idx], Nb_face_points[face_idx]);
	cout << endl;

}

#define MY_OWN_BUFSIZE ONE_MILLION

void scene::read_obj_file(const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	char *p_buf;
	double x, y, z;
	double x0, y0, z0;
	double x1, y1, z1;
	double sx = 0, sy = 0, sz = 0;
	int line_nb = -1;
	char str[1000];
	int a, l, h;
	int *w;
	int nb_points0;
	int nb_pts = 0;
	int nb_faces_read = 0;
	int idx, pt_idx;

	if (f_v) {
		cout << "scene::read_obj_file" << endl;
	}

	buf = NEW_char(MY_OWN_BUFSIZE);

	nb_points0 = nb_points;


	{
		ifstream fp(fname);


		while (TRUE) {
			if (fp.eof()) {
				break;
			}
			line_nb++;
			fp.getline(buf, MY_OWN_BUFSIZE, '\n');
			//cout << "read line " << line << " : '" << buf << "'" << endl;
			if (strlen(buf) == 0) {
				//cout << "scene::read_obj_file empty line, skipping" << endl;
				continue;
			}

			if (strncmp(buf, "#", 1) == 0) {
				continue;
			}
			else if (strncmp(buf, "v ", 2) == 0) {
				p_buf = buf + 2;
				s_scan_double(&p_buf, &x);
				s_scan_double(&p_buf, &y);
				s_scan_double(&p_buf, &z);
				if (nb_pts == 0) {
					x0 = x;
					x1 = x;
					y0 = y;
					y1 = y;
					z0 = z;
					z1 = z;
				}
				else {
					x0 = min(x0, x);
					x1 = max(x1, x);
					y0 = min(y0, y);
					y1 = max(y1, y);
					z0 = min(z0, z);
					z1 = max(z1, z);
				}
				if (ABS(x) > ONE_MILLION) {
					cout << "x coordinate out of range, skipping: " << x << endl;
					cout << "read line " << line_nb << " : '" << buf << "'" << endl;
					continue;
				}
				if (ABS(y) > ONE_MILLION) {
					cout << "y coordinate out of range, skipping: " << y << endl;
					cout << "read line " << line_nb << " : '" << buf << "'" << endl;
					continue;
				}
				if (ABS(z) > ONE_MILLION) {
					cout << "z coordinate out of range, skipping: " << z << endl;
					cout << "read line " << line_nb << " : '" << buf << "'" << endl;
					continue;
				}
				sx += x;
				sy += y;
				sz += z;
				nb_pts++;
				//cout << "point : " << x << "," << y << "," << z << endl;
				point(x, y, z);
			}
			else if (strncmp(buf, "f ", 2) == 0) {

				nb_faces_read++;
				//cout << "reading face: " << buf << endl;
				p_buf = buf + 2;
				vector<int> v;
				while (strlen(p_buf)) {
					s_scan_token_arbitrary(&p_buf, str);
					//cout << "read token: " << str << endl;
					if (strlen(str) == 0) {
						continue;
					}
					l = strlen(str);
					for (h = 0; h < l; h++) {
						if (str[h] == '/') {
							str[h] = 0;
							a = atoi(str);
							break;
						}
					}
					if (h == l) {
						a = atoi(str);
					}
					//cout << "reading it as " << a << endl;
					v.push_back(a);
				}
				l = v.size();
				w = NEW_int(l + 1);
				for (h = 0; h < l; h++) {
					w[h] = v[h] - 1 + nb_points0;
				}
				if (w[l - 1] != w[0]) {
					w[l] = w[0];
					l++;
				}
				//cout << "read face : ";
				//int_vec_print(cout, w, l);
				//cout << endl;
				idx = face(w, l);
				if (FALSE && idx == 2920) {
					cout << "added face " << idx << ": ";
					int_vec_print(cout, w, l);
					cout << endl;
					for (h = 0; h < l; h++) {
						pt_idx = w[h];
						double x, y, z;
						x = point_coords(pt_idx, 0);
						y = point_coords(pt_idx, 0);
						z = point_coords(pt_idx, 0);
						cout << "Point " << pt_idx << " : x=" << x << " y=" << y << " z=" << z << endl;
					}
				}
				FREE_int(w);
			}
		}
		cout << "midpoint: " << sx / nb_pts << ", " << sy / nb_pts << ", " << sz / nb_pts << endl;
		cout << "x-interval: [" << x0 << ", " << x1 << "]" << endl;
		cout << "y-interval: [" << y0 << ", " << y1 << "]" << endl;
		cout << "z-interval: [" << z0 << ", " << z1 << "]" << endl;
		cout << "number points read: " << nb_pts << endl;
		cout << "number faces read: " << nb_faces_read << endl;
	}
	if (f_v) {
		cout << "scene::read_obj_file done" << endl;
	}
}

void scene::add_a_group_of_things(int *Idx, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	vector<int> v;
	int i;

	if (f_v) {
		cout << "scene::add_a_group_of_things" << endl;
	}
	for (i = 0; i < sz; i++) {
		v.push_back(Idx[i]);
	}
	group_of_things.push_back(v);
	if (f_v) {
		cout << "scene::add_a_group_of_things done" << endl;
	}
}

void scene::create_regulus(int idx, int nb_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double coeff[10];
	numerics Num;
	int k, i, j;

	if (f_v) {
		cout << "scene::create_regulus" << endl;
	}


	double axx;
	double ayy;
	double azz;
	double axy;
	double axz;
	double ayz;
	double ax;
	double ay;
	double az;
	double a1;

	double A[9];
	double lambda[3];
	double Basis[9];
	double Basis_t[9];
	double B[9];
	double C[9];
	double D[9];
	double E[9];
	double F[9];
	double R[9];

	double vec_a[3];
	double vec_c[3];
	double vec_d[3];
	double vec_e[1];
	double vec_f[1];
	double c1;

	double x6[6];
	double w6[6];
	double y6[6];
	double z6[6];

	int *line_idx;
	int axis_of_symmetry_idx;

	axx = quadric_coords(idx, 0);
	axy = quadric_coords(idx, 1);
	axz = quadric_coords(idx, 2);
	ax = quadric_coords(idx, 3);
	ayy = quadric_coords(idx, 4);
	ayz = quadric_coords(idx, 5);
	ay = quadric_coords(idx, 6);
	azz = quadric_coords(idx, 7);
	az = quadric_coords(idx, 8);
	a1 = quadric_coords(idx, 9);

	coeff[0] = axx;
	coeff[1] = axy;
	coeff[2] = axz;
	coeff[3] = ax;
	coeff[4] = ayy;
	coeff[5] = ayz;
	coeff[6] = ay;
	coeff[7] = azz;
	coeff[8] = az;
	coeff[9] = a1;

	if (f_v) {
		cout << "scene::create_regulus coeff=" << endl;
		Num.print_system(coeff, 10, 1);
	}


	//quadric1_idx = S->quadric(coeff); // Q(2 * h + 0)

	// A is the 3 x 3 symmetric coefficient matrix
	// of the quadratic terms:
	A[0] = axx;
	A[4] = ayy;
	A[8] = azz;
	A[1] = A[3] = axy * 0.5;
	A[2] = A[6] = axz * 0.5;
	A[5] = A[7] = ayz * 0.5;

	// vec_a is the linear terms:
	vec_a[0] = ax;
	vec_a[1] = ay;
	vec_a[2] = az;
	if (f_v) {
		cout << "scene::create_regulus A=" << endl;
		Num.print_system(A, 3, 3);
		cout << "scene::create_regulus a=" << endl;
		Num.print_system(vec_a, 1, 3);
	}


	if (f_v) {
		cout << "scene::create_regulus" << endl;
	}
	Num.eigenvalues(A, 3, lambda, verbose_level - 2);
	Num.eigenvectors(A, Basis,
			3, lambda, verbose_level - 2);

	if (f_v) {
		cout << "scene::create_regulus Basis=" << endl;
		Num.print_system(Basis, 3, 3);
	}
	Num.transpose_matrix_nxn(Basis, Basis_t, 3);

	Num.mult_matrix_matrix(Basis_t, A, B, 3, 3, 3);
	Num.mult_matrix_matrix(B, Basis, C, 3, 3, 3);
		// C = Basis_t * A * Basis = diagonal matrix

	if (f_v) {
		cout << "scene::create_regulus diagonalized matrix is" << endl;
	}
	Num.print_system(C, 3, 3);

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			D[i * 3 + j] = 0;

			if (i == j) {
				if (ABS(C[i * 3 + i]) > 0.0001) {
					D[i * 3 + i] = 1. / C[i * 3 + i]; // 1 / lambda_i
				}
				else {
					cout << "Warning zero eigenvalue" << endl;
					D[i * 3 + i] = 0;
				}
			}
		}
	}
	if (f_v) {
		cout << "scene::create_regulus D=" << endl;
		Num.print_system(D, 3, 3);
	}

	Num.mult_matrix_matrix(Basis, D, E, 3, 3, 3);
	Num.mult_matrix_matrix(E, Basis_t, F, 3, 3, 3);
		// F = Basis * D * Basis_t
	Num.mult_matrix_matrix(F, vec_a, vec_c, 3, 3, 1);
	for (i = 0; i < 3; i++) {
		vec_c[i] *= 0.5;
	}
	// c = 1/2 * Basis * D * Basis_t * a

	if (f_v) {
		cout << "scene::create_regulus c=" << endl;
		Num.print_system(vec_c, 3, 1);
	}


	Num.mult_matrix_matrix(vec_c, A, vec_d, 1, 3, 3);
	Num.mult_matrix_matrix(vec_d, vec_c, vec_e, 1, 3, 1);
	// e = c^\top * A * c

	Num.mult_matrix_matrix(vec_a, vec_c, vec_f, 1, 3, 1);
	// f = a^\top * c

	c1 = vec_e[0] - vec_f[0] + a1;
	// e - f + a1

	if (f_v) {
		cout << "scene::create_regulus e=" << vec_e[0] << endl;
		cout << "scene::create_regulus f=" << vec_f[0] << endl;
		cout << "scene::create_regulus a1=" << a1 << endl;
		cout << "scene::create_regulus c1=" << c1 << endl;
	}

	coeff[0] = C[0 * 3 + 0]; // x^2
	coeff[1] = 0;
	coeff[2] = 0;
	coeff[3] = 0;
	coeff[4] = C[1 * 3 + 1]; // y^2
	coeff[5] = 0;
	coeff[6] = 0;
	coeff[7] = C[2 * 3 + 2]; // z^2
	coeff[8] = 0;
	coeff[9] = c1;

	if (f_v) {
		cout << "scene::create_regulus coeff=" << endl;
		Num.print_system(coeff, 10, 1);
	}


	//quadric2_idx = S->quadric(coeff); // Q(2 * h + 1)

	// the axis of symmetry:
	x6[0] = 0;
	x6[1] = 0;
	x6[2] = -1;
	x6[3] = 0;
	x6[4] = 0;
	x6[5] = 1;



	// mapping x \mapsto Basis * x - c
	Num.mult_matrix_matrix(Basis, x6, y6, 3, 3, 1);
	Num.mult_matrix_matrix(Basis, x6 + 3, y6 + 3, 3, 3, 1);

	Num.vec_linear_combination(1, y6,
			-1, vec_c, z6, 3);
	Num.vec_linear_combination(1, y6 + 3,
			-1, vec_c, z6 + 3, 3);

	// create the axis of symmetry inside the scene
	axis_of_symmetry_idx = line_through_two_pts(z6, sqrt(3) * 100.); // Line h * (TARGET_NB_LINES + 1) + 0


	// create a line on the cone:
	x6[0] = 0;
	x6[1] = 0;
	x6[2] = 0;
	if (lambda[2] < 0) {
		x6[3] = sqrt(-lambda[2]);
		x6[4] = 0;
		x6[5] = sqrt(lambda[0]);
	}
	else {
		x6[3] = sqrt(lambda[2]);
		x6[4] = 0;
		x6[5] = sqrt(-lambda[0]);
	}
	x6[0] = - x6[3];
	x6[1] = - x6[4];
	x6[2] = - x6[5];

	// mapping x \mapsto Basis * x - c
	Num.mult_matrix_matrix(Basis, x6, y6, 3, 3, 1);
	Num.mult_matrix_matrix(Basis, x6 + 3, y6 + 3, 3, 3, 1);

	Num.vec_linear_combination(1, y6,
			-1, vec_c, z6, 3);
	Num.vec_linear_combination(1, y6 + 3,
			-1, vec_c, z6 + 3, 3);


	line_idx = NEW_int(nb_lines);


	line_idx[0] = line_through_two_pts(z6, sqrt(3) * 100.);
		// Line h * (TARGET_NB_LINES + 1) + 1

	// create the remaining lines on the cone using symmetry:

	double phi;

	phi = 2. * M_PI / (double) nb_lines;
	for (k = 1; k < nb_lines; k++) {
		Num.make_Rz(R, (double) k * phi);
		Num.mult_matrix_matrix(R, x6, w6, 3, 3, 1);
		Num.mult_matrix_matrix(R, x6 + 3, w6 + 3, 3, 3, 1);


		// mapping x \mapsto Basis * x - c
		Num.mult_matrix_matrix(Basis, w6, y6, 3, 3, 1);
		Num.mult_matrix_matrix(Basis, w6 + 3, y6 + 3, 3, 3, 1);

		Num.vec_linear_combination(1, y6,
				-1, vec_c, z6, 3);
		Num.vec_linear_combination(1, y6 + 3,
				-1, vec_c, z6 + 3, 3);

		line_idx[k] = line_through_two_pts(z6, sqrt(3) * 100.);
			// Line h * (TARGET_NB_LINES + 1) + 1 + k

	}

	cout << "adding group for axis of symmetry:" << endl;
	add_a_group_of_things(&axis_of_symmetry_idx, 1, verbose_level);

	cout << "adding group for lines of the regulus:" << endl;
	add_a_group_of_things(line_idx, nb_lines, verbose_level);

	FREE_int(line_idx);


	if (f_v) {
		cout << "scene::create_regulus done" << endl;
	}
}

void scene::clipping_by_cylinder(int line_idx, double r, ostream &ost)
{
	int h;
	numerics N;

	ost << "	clipped_by { 	cylinder{<";
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
	ost << ">, " << r << " } } // line " << line_idx << endl;
	ost << "	bounded_by { clipped_by }" << endl;

}

}}

