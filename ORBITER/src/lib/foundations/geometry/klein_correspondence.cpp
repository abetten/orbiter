// klein_correspondence.cpp
//
// Anton Betten
// 
// January 1, 2016

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


klein_correspondence::klein_correspondence()
{
	null();
}

klein_correspondence::~klein_correspondence()
{
	freeself();
}

void klein_correspondence::null()
{
	P3 = NULL;
	P5 = NULL;
	Gr63 = NULL;
	Gr62 = NULL;
	Form = NULL;
	//Line_to_point_on_quadric = NULL;
	//Point_on_quadric_to_line = NULL;
	//Point_on_quadric_embedded_in_P5 = NULL;
	//coordinates_of_quadric_points = NULL;
	//Pt_rk = NULL;
}

void klein_correspondence::freeself()
{
	if (P3) {
		FREE_OBJECT(P3);
	}
	if (P5) {
		FREE_OBJECT(P5);
	}
	if (Gr63) {
		FREE_OBJECT(Gr63);
	}
	if (Gr62) {
		FREE_OBJECT(Gr62);
	}
	if (Form) {
		FREE_int(Form);
	}
#if 0
	if (Line_to_point_on_quadric) {
		FREE_lint(Line_to_point_on_quadric);
	}
	if (Point_on_quadric_to_line) {
		FREE_lint(Point_on_quadric_to_line);
	}
#endif
#if 0
	if (Point_on_quadric_embedded_in_P5) {
		FREE_lint(Point_on_quadric_embedded_in_P5);
	}
#endif
#if 0
	if (coordinates_of_quadric_points) {
		FREE_int(coordinates_of_quadric_points);
	}
	if (Pt_rk) {
		FREE_int(Pt_rk);
	}
#endif
}

void klein_correspondence::init(finite_field *F,
		orthogonal *O, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int d = 6;
	int h, u, v;
	geometry_global Gg;

	if (f_v) {
		cout << "klein_correspondence::init" << endl;
	}
	
	klein_correspondence::F = F;
	klein_correspondence::O = O;
	q = F->q;


	nb_Pts = O->nb_points;
	
	P3 = NEW_OBJECT(projective_space);
	
	if (f_v) {
		cout << "klein_correspondence::init before P3->init" << endl;
	}
	P3->init(3, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level - 2);

	P5 = NEW_OBJECT(projective_space);
	
	if (f_v) {
		cout << "klein_correspondence::init before P5->init" << endl;
	}
	P5->init(5, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level - 2);
	if (f_v) {
		cout << "klein_correspondence::after before P5->init" << endl;
	}

	
	Gr63 = NEW_OBJECT(grassmann);
	Gr62 = NEW_OBJECT(grassmann);

	Gr63->init(6, 3, F, 0 /* verbose_level */);
	Gr62->init(6, 2, F, 0 /* verbose_level */);


	Form = NEW_int(d * d);
	int_vec_zero(Form, d * d);
	// the matrix with blocks
	// [0 1]
	// [1 0]
	// along the diagonal:
	for (h = 0; h < 3; h++) {
		u = 2 * h + 0;
		v = 2 * h + 1;
		Form[u * d + v] = 1;
		Form[v * d + u] = 1;
	}
	if (f_v) {
		cout << "klein_correspondence::init Form matrix:" << endl;
		int_matrix_print(Form, d, d);
	}
#if 0
	if (f_v) {
		cout << "klein_correspondence::init before allocate "
				"Line_to_point_on_quadric P3->N_lines="
				<< P3->N_lines << endl;
	}
	Line_to_point_on_quadric = NEW_lint(P3->N_lines);
	if (f_v) {
		cout << "klein_correspondence::init before allocate "
				"Point_on_quadric_to_line P3->N_lines="
				<< P3->N_lines << endl;
	}
	Point_on_quadric_to_line = NEW_lint(P3->N_lines);
	if (f_v) {
		cout << "klein_correspondence::init before allocate "
				"Point_on_quadric_embedded_in_P5 P3->N_lines="
				<< P3->N_lines << endl;
	}
	//Point_on_quadric_embedded_in_P5 = NEW_lint(P3->N_lines);
#endif

#if 0
	int basis_line[8]; // [2 * 4]
	int v6[6];
	int *x4, *y4, a, b, c, val;
	long int i, j;
	long int N100;
	long int point_rk, line_rk;

	//basis_line = NEW_int(8);
	for (i = 0; i < P3->N_lines; i++) {
		Point_on_quadric_to_line[i] = -1;
	}
	if (f_v) {
		cout << "klein_correspondence::init computing "
				"Line_to_point_on_quadric[] / Point_on_quadric_to_line[]" << endl;
	}

	N100 = P3->N_lines / 100;
	for (i = 0; i < P3->N_lines; i++) {

		if ((i % N100) == 0) {
			cout << "klein_correspondence::init at " << i << " which is " << (double) i / (double) N100 << "%" << endl;
		}
		P3->unrank_line(basis_line, i);
		x4 = basis_line;
		y4 = basis_line + 4;
		v6[0] = F->Pluecker_12(x4, y4);
		v6[1] = F->Pluecker_34(x4, y4);
		v6[2] = F->Pluecker_13(x4, y4);
		v6[3] = F->Pluecker_42(x4, y4);
		v6[4] = F->Pluecker_14(x4, y4);
		v6[5] = F->Pluecker_23(x4, y4);
		a = F->mult(v6[0], v6[1]);
		b = F->mult(v6[2], v6[3]);
		c = F->mult(v6[4], v6[5]);
		val = F->add3(a, b, c);
		//cout << "a=" << a << " b=" << b << " c=" << c << endl;
		//cout << "val=" << val << endl;
		if (val) {
			cout << "klein_correspondence::init point does "
					"not lie on quadric" << endl;
			exit(1);
		}
		//j = P5->rank_point(v6);
		j = O->rank_point(v6, 1, 0 /* verbose_level */);
		if (FALSE) {
			cout << "klein_correspondence::init i=" << i
					<< " / " << P3->N_lines << " v6 : ";
			int_vec_print(cout, v6, 6);
			cout << " : j=" << j << endl;
		}
		if (Point_on_quadric_to_line[j] != -1) {
			cout << "Something is wrong with "
					"Point_on_quadric_to_line: Point_on_quadric_to_line[j] != -1" << endl;
			exit(1);
		}

		point_rk = line_to_point_on_quadric(i, verbose_level);
		if (point_rk != j) {
			cout << "klein_correspondence::init point_rk != j" << endl;
			exit(1);
		}
		line_rk = point_on_quadric_to_line(point_rk, verbose_level);
		if (line_rk != i) {
			cout << "klein_correspondence::init line_rk != i" << endl;
			exit(1);
		}

		Line_to_point_on_quadric[i] = j;
		Point_on_quadric_to_line[j] = i;
	}
	for (i = 0; i < P3->N_lines; i++) {
		if (Point_on_quadric_to_line[i] == -1) {
			cout << "Something is wrong with "
					"Point_on_quadric_to_line" << endl;
			cout << "Point_on_quadric_to_line[i] == -1" << endl;
			cout << "i=" << i << endl;
			exit(1);
		}
	}
#endif


#if 0
	if (f_v) {
		cout << "klein_correspondence::init computing "
				"Point_on_quadric_embedded_in_P5[]" << endl;
	}
	for (i = 0; i < P3->N_lines; i++) {
		O->unrank_point(v6, 1, i, 0);
		Point_on_quadric_embedded_in_P5[i] = P5->rank_point(v6);
	}
#endif


#if 0
	if (f_v) {
		cout << "klein_correspondence::init before coordinates_"
				"of_quadric_points P3->N_lines * d="
				<< P3->N_lines * d << endl;
	}
	coordinates_of_quadric_points = NEW_int(P3->N_lines * d);


	if (f_v) {
		cout << "klein_correspondence::init before allocate "
				"Pt_rk P3->N_lines=" << P3->N_lines << endl;
	}
	Pt_rk = NEW_int(P3->N_lines);

	if (f_v) {
		cout << "klein_correspondence::init after allocate "
				"Pt_rk P3->N_lines=" << P3->N_lines << endl;
		cout << "klein_correspondence::init computing Pt_rk[]" << endl;
	}
	for (i = 0; i < P3->N_lines; i++) {
		O->unrank_point(
			coordinates_of_quadric_points + i * d, 1, i, 0);
		int_vec_copy(
			coordinates_of_quadric_points + i * d, v6, 6);
		F->PG_element_rank_modified(v6, 1, d, a);
		Pt_rk[i] = a;
	}
	if (f_v) {
		cout << "klein_correspondence::init computing Pt_rk[] done" << endl;
	}

	if (f_vv) {
		cout << "Points on the Klein quadric:" << endl;
		if (nb_Pts < 50) {
			for (i = 0; i < nb_Pts; i++) {
				P3->unrank_line(basis_line, i);

				cout << i << " & " << endl;
				cout << "\\left[" << endl;
				cout << "\\begin{array}{cccc}" << endl;
				for (u = 0; u < 2; u++) {
					for (v = 0; v < 4; v++) {
						cout << basis_line[u * 4 + v] << " ";
						if (v < 4 - 1) {
							cout << "&";
						}
					}
					cout << "\\\\" << endl;
				}
				cout << "\\end{array}" << endl;
				cout << "\\right] & " << endl;
				int_vec_print(cout,
						coordinates_of_quadric_points + i * d, d);
				//cout << " : " << Pt_rk[i] << endl;
				cout << "\\\\" << endl;
			}
		}
		else {
			cout << "too many points to print" << endl;
		}
	}
#endif

	nb_pts_PG = Gg.nb_PG_elements(d - 1, q);
	if (f_v) {
		cout << "klein_correspondence::init "
				"nb_pts_PG = " << nb_pts_PG << endl;
	}

#if 0
	// this array is only used by REGULAR_PACKING
	if (f_v) {
		cout << "klein_correspondence::init before "
				"allocate Pt_idx nb_pts_P=" << nb_pts_PG << endl;
	}
	Pt_idx = NEW_int(nb_pts_PG);
	for (i = 0; i < nb_pts_PG; i++) {
		Pt_idx[i] = -1;
	}
	for (i = 0; i < nb_Pts; i++) {
		a = Pt_rk[i];
		Pt_idx[a] = i;
	}
#endif


	if (f_v) {
		cout << "klein_correspondence::init done" << endl;
	}
}

void klein_correspondence::plane_intersections(
	long int *lines_in_PG3, int nb_lines,
	longinteger_object *&R,
	long int **&Pts_on_plane,
	int *&nb_pts_on_plane, 
	int &nb_planes, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int *pts;
	int i;

	if (f_v) {
		cout << "klein_correspondence::plane_intersections" << endl;
		}
	pts = NEW_lint(nb_lines);
	
	P3->klein_correspondence(P5, 
		lines_in_PG3, nb_lines, pts, 0/*verbose_level*/);

	P5->plane_intersection_type_fast(Gr63, pts, nb_lines, 
		R, Pts_on_plane, nb_pts_on_plane, nb_planes, 
		verbose_level - 3);


	if (f_vv) {
		cout << "klein_correspondence::plane_intersections: "
				"We found " << nb_planes << " planes." << endl;
#if 1
		for (i = 0; i < nb_planes; i++) {
			cout << setw(3) << i << " : " << R[i] 
				<< " : " << setw(5) << nb_pts_on_plane[i] << " : ";
			lint_vec_print(cout, Pts_on_plane[i], nb_pts_on_plane[i]);
			cout << endl; 
			}
#endif
		}
	
	FREE_lint(pts);
	if (f_v) {
		cout << "klein_correspondence::plane_intersections done" << endl;
		}
}

long int klein_correspondence::point_on_quadric_embedded_in_P5(long int pt)
{
	int v6[6];
	long int r;

	O->unrank_point(v6, 1, pt, 0);
	r = P5->rank_point(v6);
	return r;
}

long int klein_correspondence::line_to_point_on_quadric(long int line_rk, int verbose_level)
{
	int basis_line[8]; // [2 * 4]
	int v6[6];
	int *x4, *y4, a, b, c, val;
	long int point_rk;

	P3->unrank_line(basis_line, line_rk);
	x4 = basis_line;
	y4 = basis_line + 4;
	v6[0] = F->Pluecker_12(x4, y4);
	v6[1] = F->Pluecker_34(x4, y4);
	v6[2] = F->Pluecker_13(x4, y4);
	v6[3] = F->Pluecker_42(x4, y4);
	v6[4] = F->Pluecker_14(x4, y4);
	v6[5] = F->Pluecker_23(x4, y4);
	a = F->mult(v6[0], v6[1]);
	b = F->mult(v6[2], v6[3]);
	c = F->mult(v6[4], v6[5]);
	val = F->add3(a, b, c);
	//cout << "a=" << a << " b=" << b << " c=" << c << endl;
	//cout << "val=" << val << endl;
	if (val) {
		cout << "klein_correspondence::line_to_point_on_quadric point does "
				"not lie on quadric" << endl;
		exit(1);
	}
	//j = P5->rank_point(v6);
	point_rk = O->rank_point(v6, 1, 0 /* verbose_level */);
	if (FALSE) {
		cout << "klein_correspondence::line_to_point_on_quadric line_rk=" << line_rk
				<< " / " << P3->N_lines << " v6 : ";
		int_vec_print(cout, v6, 6);
		cout << " : point_rk=" << point_rk << endl;
	}

	return point_rk;
}

long int klein_correspondence::point_on_quadric_to_line(long int point_rk, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int basis_line[8]; // [2 * 4]
	int v6[6];
	//int *X4, *Y4, a, b, c, val;
	long int line_rk = 0;
	int p12, p34, p13, p24, p14, p23;
	int x2, x3, x4;
	int y2, y3, y4;

	if (f_v) {
		cout << "klein_correspondence::point_on_quadric_to_line point_rk=" << point_rk << endl;
	}
	O->unrank_point(v6, 1, point_rk, 0);
	if (f_v) {
		cout << "klein_correspondence::point_on_quadric_to_line v6=";
		int_vec_print(cout, v6, 6);
		cout << endl;
	}

	p12 = v6[0];
	p34 = v6[1];
	p13 = v6[2];
	p24 = F->negate(v6[3]);
	p14 = v6[4];
	p23 = v6[5];

	int_vec_zero(basis_line, 8);

	if (p12 == 0 && p13 == 0 && p14 == 0) {
		// this means that x1 = 0
		if (f_v) {
			cout << "klein_correspondence::point_on_quadric_to_line x1=0" << endl;
		}

		if (p23 == 0 && p24 == 0) {
			basis_line[2] = basis_line[7] = 1;
		}
		else {
			y3 = p23;
			y4 = p24;
			if (y3) {
				x4 = F->negate(F->a_over_b(p34, y3));
				basis_line[1] = 1;
				basis_line[3] = x4;
				basis_line[6] = p23;
				basis_line[7] = p24;
			}
			else {
				x3 = F->a_over_b(p34, y4);
				basis_line[1] = 1;
				basis_line[2] = x3;
				basis_line[7] = 1;
			}
		}
	}
	else {
		// at least one of p12, p13, p14 is nonzero,
		// which means that x1 \neq 0
		if (f_v) {
			cout << "klein_correspondence::point_on_quadric_to_line x1=1" << endl;
		}

		y2 = p12;
		y3 = p13;
		y4 = p14;
		if (y2 == 0 && y3 == 0) {
			basis_line[0] = 1;
			basis_line[1] = p24;
			basis_line[2] = p34;
			basis_line[7] = 1;
		}
		else {
			if (p12) {
				x3 = F->negate(F->a_over_b(p23, p12));
				x4 = F->negate(F->a_over_b(p24, p12));
				basis_line[0] = 1;
				basis_line[2] = x3;
				basis_line[3] = x4;
				basis_line[5] = p12;
				basis_line[6] = p13;
				basis_line[7] = y4;
			}
			else {
				x2 = F->a_over_b(p23, p13);
				x4 = F->negate(F->a_over_b(p34, p13));
				basis_line[0] = 1;
				basis_line[1] = x2;
				basis_line[3] = x4;
				basis_line[6] = p13;
				basis_line[7] = y4;
			}
		}

	}

	line_rk = P3->rank_line(basis_line);
	if (f_v) {
		cout << "klein_correspondence::point_on_quadric_to_line point_rk=" << point_rk << " line_rk=" << line_rk << " done" << endl;
	}

	return line_rk;
}


	//long int *Line_to_point_on_quadric; // [P3->N_lines]
	//long int *Point_on_quadric_to_line; // [P3->N_lines]


}}



