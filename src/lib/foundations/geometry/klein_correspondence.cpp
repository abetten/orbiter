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
	nb_lines_orthogonal = 0;
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

	combinatorics::combinatorics_domain Combi;
	longinteger_object la;

	Combi.q_binomial(la, d, 2, q, 0 /* verbose_level */);

	nb_lines_orthogonal = la.as_lint();


	Form = NEW_int(d * d);
	Orbiter->Int_vec->zero(Form, d * d);
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
		Orbiter->Int_vec->matrix_print(Form, d, d);
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
		cout << "klein_correspondence::init nb_pts_PG = " << nb_pts_PG << endl;
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
			Orbiter->Lint_vec->print(cout, Pts_on_plane[i], nb_pts_on_plane[i]);
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
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "klein_correspondence::line_to_point_on_quadric" << endl;
	}
	int v6[6];
	long int point_rk;

	line_to_Pluecker(line_rk, v6, verbose_level);

	point_rk = O->rank_point(v6, 1, 0 /* verbose_level */);
	if (FALSE) {
		cout << "klein_correspondence::line_to_point_on_quadric line_rk=" << line_rk
				<< " / " << P3->N_lines << " v6 : ";
		Orbiter->Int_vec->print(cout, v6, 6);
		cout << " : point_rk=" << point_rk << endl;
	}

	if (f_v) {
		cout << "klein_correspondence::line_to_point_on_quadric done" << endl;
	}
	return point_rk;
}

void klein_correspondence::line_to_Pluecker(long int line_rk, int *v6, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "klein_correspondence::line_to_Pluecker" << endl;
	}
	int basis_line[8]; // [2 * 4]
	int *x4, *y4, a, b, c, val;

	P3->unrank_line(basis_line, line_rk);
	x4 = basis_line;
	y4 = basis_line + 4;
	v6[0] = F->Linear_algebra->Pluecker_12(x4, y4);
	v6[1] = F->Linear_algebra->Pluecker_34(x4, y4);
	v6[2] = F->Linear_algebra->Pluecker_13(x4, y4);
	v6[3] = F->Linear_algebra->Pluecker_42(x4, y4);
	v6[4] = F->Linear_algebra->Pluecker_14(x4, y4);
	v6[5] = F->Linear_algebra->Pluecker_23(x4, y4);
	a = F->mult(v6[0], v6[1]);
	b = F->mult(v6[2], v6[3]);
	c = F->mult(v6[4], v6[5]);
	val = F->add3(a, b, c);
	//cout << "a=" << a << " b=" << b << " c=" << c << endl;
	//cout << "val=" << val << endl;
	if (val) {
		cout << "klein_correspondence::line_to_Pluecker point does "
				"not lie on quadric" << endl;
		exit(1);
	}
	//j = P5->rank_point(v6);
	if (f_v) {
		cout << "klein_correspondence::line_to_Pluecker done" << endl;
	}
}

long int klein_correspondence::point_on_quadric_to_line(long int point_rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "klein_correspondence::point_on_quadric_to_line" << endl;
	}
	int v6[6];
	int basis_line[8]; // [2 * 4]
	long int line_rk = 0;

	O->unrank_point(v6, 1, point_rk, 0);
	if (f_v) {
		cout << "klein_correspondence::point_on_quadric_to_line v6=";
		Orbiter->Int_vec->print(cout, v6, 6);
		cout << endl;
	}

	Pluecker_to_line(v6, basis_line, verbose_level);

	line_rk = P3->rank_line(basis_line);
	if (f_v) {
		cout << "klein_correspondence::point_on_quadric_to_line "
				"point_rk=" << point_rk << " line_rk=" << line_rk << " done" << endl;
	}
	if (line_rk >= P3->N_lines) {
		cout << "klein_correspondence::point_on_quadric_to_line "
				"line_rk >= P3->N_lines" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "klein_correspondence::point_on_quadric_to_line done" << endl;
	}
	return line_rk;
}

void klein_correspondence::Pluecker_to_line(int *v6, int *basis_line, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int p12, p34, p13, p24, p14, p23;
	int v[6];

	if (f_v) {
		cout << "klein_correspondence::Pluecker_to_line" << endl;
	}

	p12 = v6[0];
	p34 = v6[1];
	p13 = v6[2];
	p24 = F->negate(v6[3]);
	p14 = v6[4];
	p23 = v6[5];

	v[0] = p12;
	v[1] = p13;
	v[2] = p14;
	v[3] = p23;
	v[4] = p24;
	v[5] = p34;

	exterior_square_to_line(
			v,
			basis_line, verbose_level);


	if (f_v) {
		cout << "klein_correspondence::Pluecker_to_line done" << endl;
	}
}

long int klein_correspondence::Pluecker_to_line_rk(int *v6, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int basis_line[8];
	long int line_rk;

	if (f_v) {
		cout << "klein_correspondence::Pluecker_to_line_rk" << endl;
	}

	Pluecker_to_line(v6, basis_line, 0 /*verbose_level*/);

	line_rk = P3->rank_line(basis_line);


	if (f_v) {
		cout << "klein_correspondence::Pluecker_to_line_rk done" << endl;
	}

	return line_rk;
}


void klein_correspondence::exterior_square_to_line(int *v, int *basis_line, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int p12, p13, p14, p23, p24, p34;
	int x2, x3, x4;
	int y2, y3, y4;

	if (f_v) {
		cout << "klein_correspondence::exterior_square_to_line" << endl;
	}

	p12 = v[0];
	p13 = v[1];
	p14 = v[2];
	p23 = v[3];
	p24 = v[4];
	p34 = v[5];


	Orbiter->Int_vec->zero(basis_line, 8);

	if (p12 == 0 && p13 == 0 && p14 == 0) {
		// this means that x1 = 0
		if (f_v) {
			cout << "klein_correspondence::exterior_square_to_line x1=0" << endl;
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
			cout << "klein_correspondence::exterior_square_to_line x1=1" << endl;
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
	if (f_v) {
		cout << "klein_correspondence::exterior_square_to_line done" << endl;
	}

}



void klein_correspondence::compute_external_lines(
		std::vector<long int> &External_lines, int verbose_level)
{

	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, j;
	int d = 6;
	int *LineMtx;
	long int *Line;
	int nb_points_covered;

	if (f_v) {
		cout << "klein_correspondence::compute_external_lines" << endl;
	}


	LineMtx = NEW_int(2 * d);

	if (f_v) {
		cout << "klein_correspondence::compute_external_lines "
				"nb_lines_orthogonal=" << nb_lines_orthogonal << endl;
	}

	nb_points_covered = Gr62->nb_points_covered(0 /* verbose_level */);

	if (f_v) {
		cout << "klein_correspondence::compute_external_lines "
				"nb_points_covered=" << nb_points_covered << endl;
	}

	Line = NEW_lint(nb_points_covered);

	if (f_v) {
		cout << "klein_correspondence::compute_external_lines "
				"computing external lines:" << endl;
	}
	int pt, a, b, c, val;
	int v6[6];

	// make a list of all external lines to the Klein quadric:
	for (i = 0; i < nb_lines_orthogonal; i++) {
		Gr62->unrank_lint_here(LineMtx, i, 0 /*verbose_level*/);
		Gr62->points_covered(Line, 0 /* verbose_level */);
		for (j = 0; j < nb_points_covered; j++) {
			pt = Line[j];
			F->PG_element_unrank_modified(v6, 1, 6, pt);
			//K->O->unrank_point(v6, 1, pt, 0);
			a = F->mult(v6[0], v6[1]);
			b = F->mult(v6[2], v6[3]);
			c = F->mult(v6[4], v6[5]);
			val = F->add3(a, b, c);

			if (val == 0) {
				break; // we found a point on the quadric, break off
			}
		}
		if (j == nb_points_covered) {
			External_lines.push_back(i);
		}
	}
	if (f_v) {
		cout << "klein_correspondence::compute_external_lines "
				"We found " << External_lines.size()
				<< " external lines" << endl;
	}

	FREE_int(LineMtx);
	FREE_lint(Line);

	if (f_v) {
		cout << "klein_correspondence::compute_external_lines done" << endl;
	}

}

void klein_correspondence::identify_external_lines_and_spreads(
		spread_tables *T,
		std::vector<long int> &External_lines,
		long int *&spread_to_external_line_idx,
		long int *&external_line_to_spread,
		int verbose_level)
// spread_to_external_line_idx[i] is index into External_lines
// corresponding to regular spread i
// external_line_to_spread[i] is the index of the
// regular spread of PG(3,q) in table T associated with
// External_lines[i]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "klein_correspondence::identify_external_lines_and_spreads" << endl;
	}
	int i, j, rk, idx;
	long int a, b;
	int N100;
	int d = 6;
	int *basis_elliptic_quadric;
	int *basis;
	int *basis_external_line;
	data_structures::sorting Sorting;

	spread_to_external_line_idx = NEW_lint(T->nb_spreads);
	external_line_to_spread = NEW_lint(External_lines.size());

	basis_elliptic_quadric = NEW_int(T->spread_size * d);
	basis = NEW_int(d * d);

	basis_external_line = NEW_int(2 * d);

	for (i = 0; i < External_lines.size(); i++) {
		external_line_to_spread[i] = -1;
	}

	N100 = (T->nb_spreads / 100) + 1;
	for (i = 0; i < T->nb_spreads; i++) {

		if ((i % N100) == 0) {
			cout << "klein_correspondence::identify_external_lines_and_spreads "
					"progress " << ((double)i / N100) << " %" << endl;
		}
		for (j = 0; j < T->spread_size; j++) {
			a = T->spread_table[i * T->spread_size + j];
			b = line_to_point_on_quadric(a, 0 /* verbose_level */);

			O->unrank_point(basis_elliptic_quadric + j * d, 1, b, 0);
		}
		if (FALSE) {
			cout << "klein_correspondence::identify_external_lines_and_spreads"
					"spread " << i
					<< " the elliptic quadric space" << endl;
			Orbiter->Int_vec->matrix_print(basis_elliptic_quadric,
					T->spread_size, d);
		}
		rk = F->Linear_algebra->Gauss_easy(basis_elliptic_quadric, T->spread_size, d);
		if (rk != 4) {
			cout << "klein_correspondence::identify_external_lines_and_spreads "
					"spread " << i << " the elliptic quadric space "
					"does not have rank 4" << endl;
			exit(1);
		}
		Orbiter->Int_vec->copy(basis_elliptic_quadric, basis, 4 * d);
		F->Linear_algebra->perp(d, 4, basis, Form, 0 /* verbose_level */);
		Orbiter->Int_vec->copy(
				basis + 4 * d,
				basis_external_line,
				2 * d);
		a = P5->rank_line(basis_external_line);
		if (!Sorting.vector_lint_search(External_lines, a, idx, 0 /*verbose_level*/)) {
			cout << "klein_correspondence::identify_external_lines_and_spreads spread "
					" cannot find the external line i = " << i << endl;
			exit(1);
		}
		spread_to_external_line_idx[i] = idx;
		external_line_to_spread[idx] = i;
	}

#if 0
	for (i = 0; i < External_lines.size(); i++) {
		if (external_line_to_spread[i] == -1) {
			cout << "klein_correspondence::identify_external_lines_and_spreads "
					"something is wrong with the correspondence" << endl;
			exit(1);
		}
	}
#endif

	FREE_int(basis_elliptic_quadric);
	FREE_int(basis);
	FREE_int(basis_external_line);

	if (f_v) {
		cout << "klein_correspondence::identify_external_lines_and_spreads done" << endl;
	}

}


void klein_correspondence::reverse_isomorphism(int *A6, int *A4, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int A6_copy[36];
	int X[16];
	int Xv[16];
	int Y[16];
	int Z[16];
	int Yv[16];
	int Zv[16];
	int XYv[16];
	int XZv[16];
	int D[16];
	//int i, u1, u2;

	if (f_v) {
		cout << "klein_correspondence::reverse_isomorphism" << endl;
	}

	if (f_v) {
		cout << "A6=" << endl;
		Orbiter->Int_vec->matrix_print(A6, 6, 6);
	}


#if 0
	int Basis1[] = {
			1,0,0,0,0,0,
			0,1,0,0,0,0,
			0,0,1,0,0,0,
			0,0,0,1,0,0,
			0,0,0,0,1,0,
			0,0,0,0,0,1,
	};
	int Basis2[36];
	int Image[36];
	int Image2[36];
	int v[6];
	int w[6];
	sorting Sorting;

	for (i = 0; i < 6; i++) {
		F->wedge_to_klein(Basis1 + i * 6, Basis2 + i * 6);
	}

	F->mult_matrix_matrix(Basis2, A6, Image, 6, 6, 6, 0 /* verbose_level*/);
	for (i = 0; i < 6; i++) {
		F->klein_to_wedge(Image + i * 6, Image2 + i * 6);
	}
#endif

#if 0
	int_vec_copy(A6, A6_copy, 36);

#if 0
	for (i = 0; i < 6; i++) {
		u1 = A6_copy[3 * 6 + i];
		u2 = F->negate(u1);
		A6_copy[3 * 6 + i] = u2;
	}

	for (i = 0; i < 6; i++) {
		u1 = A6_copy[i * 6 + 3];
		u2 = F->negate(u1);
		A6_copy[i * 6 + 3] = u2;
	}
#endif
	if (f_v) {
		cout << "A6_copy=" << endl;
		int_matrix_print(A6_copy, 6, 6);
	}
#endif

	// 12,34:
	exterior_square_to_line(A6, X, 0 /* verbose_level*/);
	exterior_square_to_line(A6 + 5 * 6, X + 8, 0 /* verbose_level*/);

	// 13,24
	exterior_square_to_line(A6 + 1 * 6, Y, 0 /* verbose_level*/);
	exterior_square_to_line(A6 + 4 * 6, Y + 8, 0 /* verbose_level*/);

	// 14,23
	exterior_square_to_line(A6 + 2 * 6, Z, 0 /* verbose_level*/);
	exterior_square_to_line(A6 + 3 * 6, Z + 8, 0 /* verbose_level*/);

	if (f_v) {
		cout << "X=" << endl;
		Orbiter->Int_vec->matrix_print(X, 4, 4);
		cout << "Y=" << endl;
		Orbiter->Int_vec->matrix_print(Y, 4, 4);
		cout << "Z=" << endl;
		Orbiter->Int_vec->matrix_print(Z, 4, 4);
	}

	F->Linear_algebra->invert_matrix(X, Xv, 4, 0 /* verbose_level*/);
	F->Linear_algebra->invert_matrix(Y, Yv, 4, 0 /* verbose_level*/);
	F->Linear_algebra->invert_matrix(Z, Zv, 4, 0 /* verbose_level*/);
	//F->invert_matrix(A, Av, 4, 0 /* verbose_level*/);

	if (f_v) {
		cout << "Xv=" << endl;
		Orbiter->Int_vec->matrix_print(Xv, 4, 4);
		cout << "Yv=" << endl;
		Orbiter->Int_vec->matrix_print(Yv, 4, 4);
		cout << "Zv=" << endl;
		Orbiter->Int_vec->matrix_print(Zv, 4, 4);
	}

	F->Linear_algebra->mult_matrix_matrix(X, Yv, XYv, 4, 4, 4, 0 /* verbose_level*/);

	if (f_v) {
		cout << "XYv=" << endl;
		Orbiter->Int_vec->matrix_print(XYv, 4, 4);
	}

	F->Linear_algebra->mult_matrix_matrix(X, Zv, XZv, 4, 4, 4, 0 /* verbose_level*/);

	if (f_v) {
		cout << "XZv=" << endl;
		Orbiter->Int_vec->matrix_print(XZv, 4, 4);
	}

	//int a, b, c, d, e, f, g, h;

	int M[16 * 8];

	Orbiter->Int_vec->zero(M, 16 * 8);

	M[0 * 8 + 0] = XYv[0 * 4 + 2];
	M[0 * 8 + 1] = XYv[1 * 4 + 2];
	M[1 * 8 + 0] = XYv[0 * 4 + 3];
	M[1 * 8 + 1] = XYv[1 * 4 + 3];
	M[2 * 8 + 4] = XYv[2 * 4 + 2];
	M[2 * 8 + 5] = XYv[3 * 4 + 2];
	M[3 * 8 + 4] = XYv[2 * 4 + 3];
	M[3 * 8 + 5] = XYv[3 * 4 + 3];
	M[4 * 8 + 2] = XYv[0 * 4 + 0];
	M[4 * 8 + 3] = XYv[1 * 4 + 0];
	M[5 * 8 + 2] = XYv[0 * 4 + 1];
	M[5 * 8 + 3] = XYv[1 * 4 + 1];
	M[6 * 8 + 6] = XYv[2 * 4 + 0];
	M[6 * 8 + 7] = XYv[3 * 4 + 0];
	M[7 * 8 + 6] = XYv[2 * 4 + 1];
	M[7 * 8 + 7] = XYv[3 * 4 + 1];

	M[8 * 8 + 0] = XZv[0 * 4 + 2];
	M[8 * 8 + 1] = XZv[1 * 4 + 2];
	M[9 * 8 + 0] = XZv[0 * 4 + 3];
	M[9 * 8 + 1] = XZv[1 * 4 + 3];
	M[10 * 8 + 6] = XZv[2 * 4 + 2];
	M[10 * 8 + 7] = XZv[3 * 4 + 2];
	M[11 * 8 + 6] = XZv[2 * 4 + 3];
	M[11 * 8 + 7] = XZv[3 * 4 + 3];
	M[12 * 8 + 2] = XZv[0 * 4 + 0];
	M[12 * 8 + 3] = XZv[1 * 4 + 0];
	M[13 * 8 + 2] = XZv[0 * 4 + 1];
	M[13 * 8 + 3] = XZv[1 * 4 + 1];
	M[14 * 8 + 4] = XZv[2 * 4 + 0];
	M[14 * 8 + 5] = XZv[3 * 4 + 0];
	M[15 * 8 + 4] = XZv[2 * 4 + 1];
	M[15 * 8 + 5] = XZv[3 * 4 + 1];

	Orbiter->Int_vec->zero(A4, 4 * 4);


	if (f_v) {
		cout << "M=" << endl;
		Orbiter->Int_vec->matrix_print(M, 16, 8);
	}

	int rk;
	int base_cols[8];

	rk = F->Linear_algebra->Gauss_simple(M, 16, 8, base_cols, verbose_level);

	//rk = F->RREF_and_kernel(16, 8, M, verbose_level);

	if (f_v) {
		cout << "has rank " << rk << endl;
		Orbiter->Int_vec->matrix_print(M, rk, 8);
	}
	if (f_v) {
		cout << "base columns: " << endl;
		Orbiter->Int_vec->print(cout, base_cols, rk);
		cout << endl;
	}

	int kernel_m, kernel_n;
	int K[8 * 8];
	int i, j;

	F->Linear_algebra->matrix_get_kernel(M, 16, 8, base_cols, rk,
		kernel_m, kernel_n, K, 0 /* verbose_level */);


	if (f_v) {
		cout << "kernel: " << endl;
		Orbiter->Int_vec->matrix_print(K, 8, kernel_n);
	}


	int abcdefgh[8];
	int a, b, c, d, e, f, g, h;

	for (i = 0; i < 8; i++) {
		abcdefgh[i] = 0;
	}


	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < 8; i++) {
			if (K[i * kernel_n + j]) {
				abcdefgh[i] = K[i * kernel_n + j];
			}
		}
	}

	a = abcdefgh[0];
	b = abcdefgh[1];
	c = abcdefgh[2];
	d = abcdefgh[3];
	e = abcdefgh[4];
	f = abcdefgh[5];
	g = abcdefgh[6];
	h = abcdefgh[7];


	Orbiter->Int_vec->zero(D, 16);
	D[0 * 4 + 0] = a;
	D[0 * 4 + 1] = b;
	D[1 * 4 + 0] = c;
	D[1 * 4 + 1] = d;
	D[2 * 4 + 2] = e;
	D[2 * 4 + 3] = f;
	D[3 * 4 + 2] = g;
	D[3 * 4 + 3] = h;

	if (f_v) {
		cout << "D=" << endl;
		Orbiter->Int_vec->matrix_print(D, 4, 4);
	}

	F->Linear_algebra->mult_matrix_matrix(D, X, A4, 4, 4, 4, 0 /* verbose_level*/);

	if (f_v) {
		cout << "A4=" << endl;
		Orbiter->Int_vec->matrix_print(A4, 4, 4);
	}


	int A6b[36];

	F->Linear_algebra->exterior_square(A4, A6b, 4, 0 /* verbose_level*/);
	//F->lift_to_Klein_quadric(A4, A6b, 0 /* verbose_level*/);

	if (f_v) {
		cout << "A6b=" << endl;
		Orbiter->Int_vec->matrix_print(A6b, 6, 6);
	}


	if (!F->test_if_vectors_are_projectively_equal(A6, A6b, 36)) {
		cout << "matrices are not projectively equal" << endl;
		exit(1);
	}
	else {
		cout << "matrices are projectively the same, success" << endl;
	}

	if (f_v) {
		cout << "klein_correspondence::reverse_isomorphism done" << endl;
	}
}


}}



