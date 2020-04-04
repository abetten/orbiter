// arc_lifting_with_two_lines.cpp
//
// Anton Betten, Fatma Karaoglu
//
// December 27, 2018
//
//
//
//

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


arc_lifting_with_two_lines::arc_lifting_with_two_lines()
{
	null();
}

arc_lifting_with_two_lines::~arc_lifting_with_two_lines()
{
	freeself();
}

void arc_lifting_with_two_lines::null()
{
	q = 0;
	F = NULL;
	Surf = NULL;
	Arc6 = NULL;
	Arc_coords = NULL;

}

void arc_lifting_with_two_lines::freeself()
{
	if (Arc_coords) {
		FREE_int(Arc_coords);
	}
	null();
}

void arc_lifting_with_two_lines::create_surface(
		surface_domain *Surf,
	long int *Arc6, long int line1, long int line2,
	int verbose_level)
// line1 = b1
// line2 = b2
// and Arc6 is the six-arc arising as image of
// the half double-six a1, a2, a3, a4, a5, a6
// The arc must be given as points in PG(3,q), not in PG(2,q).
// The partition is missing.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int q;
	int base_cols[4];
	int Basis[16];
	int Transversals[4 * 8];
	int rk;

	if (f_v) {
		cout << "arc_lifting_with_two_lines::create_surface" << endl;
		cout << "line1=" << line1 << " line2=" << line2 << endl;
		cout << "Arc6: ";
		lint_vec_print(cout, Arc6, 6);
		cout << endl;
	}

	arc_size = 6;
	arc_lifting_with_two_lines::Arc6 = Arc6;
	arc_lifting_with_two_lines::Surf = Surf;
	arc_lifting_with_two_lines::line1 = line1;
	arc_lifting_with_two_lines::line2 = line2;

	F = Surf->F;
	q = F->q;


	Arc_coords = NEW_int(6 * 4);
	Surf->P->unrank_points(Arc_coords, Arc6, 6);

	rk = F->Gauss_simple(
			Arc_coords, 6, 4 /*dimension*/,
			base_cols, 0 /* verbose_level */);
	if (rk != 3) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"the arc does not lie in a plane" << endl;
		cout << "rk=" << rk << endl;
		exit(1);
	}
	Surf->P->unrank_points(Arc_coords, Arc6, 6);

	plane_rk = Surf->rank_plane(Arc_coords);
	if (f_vv) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"plane_rk=" << plane_rk << endl;
		}
	Surf->unrank_line(Basis, line1);
	Surf->unrank_line(Basis + 8, line2);
	if (f_vv) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"Basis of line1 and line2:" << endl;
		cout << "line1=" << line1 << " line2=" << line2 << endl;
		int_matrix_print(Basis, 4, 4);
	}
	rk = F->Gauss_simple(
			Basis, 4, 4 /*dimension*/,
			base_cols,
			0 /* verbose_level */);
	if (rk != 4) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"the two lines are not skew" << endl;
		exit(1);
	}

	P[0] = Surf->P->point_of_intersection_of_a_line_and_a_plane_in_three_space(
			line1,
			plane_rk,
			0 /* verbose_level */);

	P[1] = Surf->P->point_of_intersection_of_a_line_and_a_plane_in_three_space(
			line2,
			plane_rk,
			0 /* verbose_level */);

	if (f_vv) {
		cout << "P[0]=" << P[0] << endl;
		cout << "P[1]=" << P[1] << endl;
	}

	// P[0] and P[1] are points on the arc.
	// Find them and remove them
	// so we can find the remaining four point of the arc:
	int i, a, h;

	h = 2;
	for (i = 0; i < 6; i++) {
		a = Arc6[i];
		if (a == P[0] || a == P[1]) {
			continue;
		}
		P[h++] = a;
	}
	if (h != 6) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"h != 6" << endl;
		exit(1);
	}
	// now P[2], P[3], P[4], P[5] are the remaining four points
	// of the arc.

	if (f_vv) {
		cout << "P[]: ";
		lint_vec_print(cout, P, 6);
		cout << endl;
	}

	transversal_01 = Surf->P->line_through_two_points(P[0], P[1]);
	// the line c_12

	if (f_vv) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"transversal_01=" << transversal_01 << endl;
	}

	transversal_23 = Surf->P->line_through_two_points(P[2], P[3]);
	// the line c_23

	if (f_vv) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"transversal_23=" << transversal_23 << endl;
	}
	transversal_45 = Surf->P->line_through_two_points(P[4], P[5]);
	// the line c_45

	if (f_vv) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"transversal_23=" << transversal_45 << endl;
	}


	Surf->P->unrank_points(Arc_coords, P, 6);

	if (f_vv) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"rearranged arc:" << endl;
		lint_vec_print(cout, P, 6);
		cout << endl;
		int_matrix_print(Arc_coords, 6, 4);
	}

	for (i = 0; i < 4; i++) {
		transversal[i] =
				Surf->P->transversal_to_two_skew_lines_through_a_point(
						line1, line2, P[2 + i],
						0 /* verbose_level */);
		Surf->unrank_line(Transversals + i * 8, transversal[i]);
	}
	if (f_vv) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"transversal:" << endl;
		lint_vec_print(cout, transversal, 4);
		cout << endl;
		cout << "Transversals:" << endl;
		int_matrix_print(Transversals, 8, 4);
	}

	input_Lines[0] = line1;
	input_Lines[1] = line2;
	input_Lines[2] = transversal_01;
	input_Lines[3] = transversal_23;
	input_Lines[4] = transversal_45;
	input_Lines[5] = transversal[0];
	input_Lines[6] = transversal[1];
	input_Lines[7] = transversal[2];
	input_Lines[8] = transversal[3];


	Surf->build_cubic_surface_from_lines(
		9, input_Lines,
		coeff, 0/* verbose_level*/);

	if (f_v) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"coeff:" << endl;
		int_vec_print(cout, coeff, 20);
		cout << endl;

		Surf->Poly3_4->print_equation(cout, coeff);
		cout << endl;
	}

	// now we have a 4 + 2:
	// b1 b2  .  .  .  .
	//  .  . a3 a4 a5 a6
	//
	// where b1 = line1, b2 = line2
	// and a3 = transversal[0]
	// and a4 = transversal[1]
	// and a5 = transversal[2]
	// and a6 = transversal[3]
	//
	// it remains to fill in the missing lines
	// to create a double six.

	// The points of a2 can be found in the following way:
	// Consider any line L through P2 and a point on b1 (but not on c12).
	// Each such line intersects a2
	// in the tritangent plane through c12, b1, a2.
	// The point on a2 can be found as the third point of intersection of
	// F (F is the surface equation)
	// with the line L (besides the point on b1 and the point P2).

	// By doing this for two different lines through P2 and b1,
	// two distinct points on a1 are found.
	// This determines a1.

	int B1[8];
	int P2[4];
	int pt1 = 0, pt2 = 0;
	int i1, i2, j1, j2;

	Surf->unrank_line(B1, line1);
	Surf->P->unrank_point(P2, P[1]);

	for (i1 = 0; i1 <= q; i1++) {
		pt1 = Surf->P->create_point_on_line(
				line1, i1 /*pt_rk*/, 0 /* verbose_level*/);
		if (!Surf->P->is_incident(pt1, transversal_01)) {
			if (f_vv) {
				cout << "found point pt1 which is not on "
						"transversal_01, i1=" << i1 << endl;
			}
			break;
		}
	}
	if (i1 == q + 1) {
		cout << "arc_lifting_with_two_lines::create_surface could "
				"not find first point on line b1" << endl;
		exit(1);
	}
	for (i2 = i1 + 1; i2 <= q; i2++) {
		pt2 = Surf->P->create_point_on_line(
				line1, i2 /*pt_rk*/, 0 /* verbose_level*/);
		if (!Surf->P->is_incident(pt2, transversal_01)) {
			break;
		}
	}
	if (i2 == q + 1) {
		cout << "arc_lifting_with_two_lines::create_surface could "
				"not find second point on line b1" << endl;
		exit(1);
	}

	if (f_vv) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"pt1=" << pt1 << " pt2=" << pt2 << endl;
	}

	int line3, line4;
	int pt3 = 0, pt4 = 0;
	int val;

	line3 = Surf->P->line_through_two_points(P[1], pt1);
	line4 = Surf->P->line_through_two_points(P[1], pt2);

	for (j1 = 0; j1 <= q; j1++) {
		pt3 = Surf->P->create_point_on_line(
				line3, j1 /*pt_rk*/, 0 /* verbose_level */);
		val = Surf->Poly3_4->evaluate_at_a_point_by_rank(coeff, pt3);
		//cout << "j1=" << j1 << " pt3=" << pt3 << " val=" << val << endl;
		if (pt3 == P[1]) {
			if (f_vv) {
				cout << "pt3 == P[1], skipping" << endl;
			}
			continue;
		}
#if 1
		if (pt3 == pt1) {
			if (f_vv) {
				cout << "pt3 == pt1, skipping" << endl;
			}
			continue;
		}
#endif
		if (val == 0) {
			break;
		}
	}
	if (j1 == q + 1) {
		if (f_vv) {
			cout << "arc_lifting_with_two_lines::create_surface "
					"j1 == q + 1, picking pt1" << endl;
		}
		pt3 = pt1;
	}

	for (j2 = 0; j2 <= q; j2++) {
		pt4 = Surf->P->create_point_on_line(
				line4, j2 /*pt_rk*/, 0 /* verbose_level*/);
		if (pt4 == P[1]) {
			continue;
		}
#if 1
		if (pt4 == pt2) {
			continue;
		}
#endif
		val = Surf->Poly3_4->evaluate_at_a_point_by_rank(coeff, pt4);
		//cout << "j2=" << j2 << " pt4=" << pt4 << " val=" << val << endl;
		if (val == 0) {
			break;
		}
	}
	if (j2 == q + 1) {
		if (f_vv) {
			cout << "arc_lifting_with_two_lines::create_surface "
					"j2 == q + 1, picking pt2" << endl;
		}
		pt4 = pt2;
	}

	if (f_vv) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"pt3=" << pt3 << " pt4=" << pt4 << endl;
	}

	int a2;
	int basis_a2[8];

	a2 = Surf->P->line_through_two_points(pt3, pt4);
	Surf->unrank_line(basis_a2, a2);


	if (f_v) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"a2=" << a2 << endl;
		cout << "basis_a2:" << endl;
		int_matrix_print(basis_a2, 2, 4);
	}


	// now we have:
	// b1 b2  .  .  .  .
	//  . a2 a3 a4 a5 a6
	//
	// where b1 = line1, b2 = line2
	// and a3 = transversal[0]
	// and a4 = transversal[1]
	// and a5 = transversal[2]
	// and a6 = transversal[3]
	//

	// We consider the 5 + 1 formed by b1 and a2,a3,a4,a5,a6:
	//
	// We pick the five point in order a3,a4,a5,a6,a2,
	// so the double six will be
	// a3 a4 a5 a6 a2 a1
	// b3 b4 b5 b6 b2 b1

	long int five_pts[5];
	long int double_six_special[12];
	long int double_six[12];

	five_pts[0] = transversal[0];
	five_pts[1] = transversal[1];
	five_pts[2] = transversal[2];
	five_pts[3] = transversal[3];
	five_pts[4] = a2;

	if (f_vv) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"five_pts=";
		lint_vec_print(cout, five_pts, 5);
		cout << endl;
	}
	if (f_vv) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"before Surf->create_double_six_from_five_"
				"lines_with_a_common_transversal" << endl;
	}
	Surf->create_double_six_from_five_lines_with_a_common_transversal(
			five_pts, double_six_special,
			verbose_level - 10);
	if (f_vv) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"after Surf->create_double_six_from_five_"
				"lines_with_a_common_transversal" << endl;
	}

	// the double six is
	// a3 a4 a5 a6 a2 a1
	// b3 b4 b5 b6 b2 b1


	if (f_vv) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"double_six_special=";
		lint_vec_print(cout, double_six_special, 12);
		cout << endl;
	}
	double_six[0] = double_six_special[5];
	double_six[1] = double_six_special[4];
	double_six[2] = double_six_special[0];
	double_six[3] = double_six_special[1];
	double_six[4] = double_six_special[2];
	double_six[5] = double_six_special[3];
	double_six[6 + 0] = double_six_special[6 + 5];
	double_six[6 + 1] = double_six_special[6 + 4];
	double_six[6 + 2] = double_six_special[6 + 0];
	double_six[6 + 3] = double_six_special[6 + 1];
	double_six[6 + 4] = double_six_special[6 + 2];
	double_six[6 + 5] = double_six_special[6 + 3];

	if (f_v) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"double_six=";
		lint_vec_print(cout, double_six, 12);
		cout << endl;
	}
	lint_vec_copy(double_six, lines27, 12);
	Surf->create_the_fifteen_other_lines(double_six,
			lines27 + 12, verbose_level - 10);

	if (f_v) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"done" << endl;
	}
}

}}

