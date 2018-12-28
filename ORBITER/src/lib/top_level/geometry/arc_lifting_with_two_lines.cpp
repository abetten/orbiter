// arc_lifting_with_two_lines.cpp
//
// Anton Betten, Fatma Karaoglu
//
// December 27, 2018
//
//
//
//

#include "orbiter.h"

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
	Surf_A = NULL;
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
	surface_with_action *Surf_A,
	int *Arc6, int line1, int line2,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q;
	surface *Surf;
	int base_cols[4];
	int Basis[16];
	int Transversals[4 * 8];
	int rk;

	if (f_v) {
		cout << "arc_lifting_with_two_lines::create_surface" << endl;
		}

	arc_size = 6;
	arc_lifting_with_two_lines::Arc6 = Arc6;
	arc_lifting_with_two_lines::Surf_A = Surf_A;
	arc_lifting_with_two_lines::line1 = line1;
	arc_lifting_with_two_lines::line2 = line2;

	Surf = Surf_A->Surf;
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
	if (f_v) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"plane_rk=" << plane_rk << endl;
		}
	Surf->unrank_line(Basis, line1);
	Surf->unrank_line(Basis + 8, line2);
	if (f_v) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"Basis of line1 and line2:" << endl;
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

	transversal_01 = Surf->P->line_through_two_points(P[0], P[1]);
	if (f_v) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"transversal_01=" << transversal_01 << endl;
		}
	transversal_23 = Surf->P->line_through_two_points(P[2], P[3]);
	if (f_v) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"transversal_23=" << transversal_23 << endl;
		}
	transversal_45 = Surf->P->line_through_two_points(P[4], P[5]);
	if (f_v) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"transversal_23=" << transversal_45 << endl;
		}


	Surf->P->unrank_points(Arc_coords, P, 6);

	if (f_v) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"rearranged arc:" << endl;
		int_vec_print(cout, P, 6);
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
	if (f_v) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"transversal:" << endl;
		int_vec_print(cout, transversal, 4);
		cout << endl;
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
		}


	if (f_v) {
		cout << "arc_lifting_with_two_lines::create_surface "
				"done" << endl;
		}
}

