/*
 * surface_domain2.cpp
 *
 *  Created on: Nov 3, 2019
 *      Author: anton
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {



void surface_domain::create_equations_for_pencil_of_surfaces_from_trihedral_pair(
	int *The_six_plane_equations, int *The_surface_equations,
	int verbose_level)
// The_six_plane_equations[24]
// The_surface_equations[(q + 1) * 20]
{
	int f_v = (verbose_level >= 1);
	int v[2];
	int l;
	int eqn_F[20];
	int eqn_G[20];
	int eqn_F2[20];
	int eqn_G2[20];

	if (f_v) {
		cout << "surface_domain::create_equations_for_pencil_of_surfaces_from_trihedral_pair" << endl;
	}


	PolynomialDomains->multiply_linear_times_linear_times_linear_in_space(
		The_six_plane_equations + 0 * 4,
		The_six_plane_equations + 1 * 4,
		The_six_plane_equations + 2 * 4,
		eqn_F, FALSE /* verbose_level */);
	PolynomialDomains->multiply_linear_times_linear_times_linear_in_space(
		The_six_plane_equations + 3 * 4,
		The_six_plane_equations + 4 * 4,
		The_six_plane_equations + 5 * 4,
		eqn_G, FALSE /* verbose_level */);


	for (l = 0; l < q + 1; l++) {
		F->PG_element_unrank_modified(v, 1, 2, l);

		Int_vec_copy(eqn_F, eqn_F2, 20);
		F->Linear_algebra->scalar_multiply_vector_in_place(v[0], eqn_F2, 20);
		Int_vec_copy(eqn_G, eqn_G2, 20);
		F->Linear_algebra->scalar_multiply_vector_in_place(v[1], eqn_G2, 20);
		F->Linear_algebra->add_vector(eqn_F2, eqn_G2, The_surface_equations + l * 20, 20);
		F->PG_element_normalize(The_surface_equations + l * 20, 1, 20);
	}

	if (f_v) {
		cout << "surface_domain::create_equations_for_pencil_of_surfaces_from_trihedral_pair done" << endl;
	}
}




long int surface_domain::plane_from_three_lines(long int *three_lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis[6 * 4];
	long int rk;

	if (f_v) {
		cout << "surface_domain::plane_from_three_lines" << endl;
	}
	unrank_lines(Basis, three_lines, 3);
	rk = F->Linear_algebra->Gauss_easy(Basis, 6, 4);
	if (rk != 3) {
		cout << "surface_domain::plane_from_three_lines rk != 3" << endl;
		exit(1);
	}
	rk = rank_plane(Basis);

	if (f_v) {
		cout << "surface_domain::plane_from_three_lines done" << endl;
	}
	return rk;
}

void surface_domain::Trihedral_pairs_to_planes(long int *Lines, long int *Planes_by_rank,
	int verbose_level)
// Planes_by_rank[nb_trihedral_pairs * 6]
{
	int f_v = (verbose_level >= 1);
	int t, i, j;
	long int rk;
	long int lines_in_tritangent_plane[3];
	long int three_lines[3];
	orbiter_kernel_system::latex_interface L;

	if (f_v) {
		cout << "surface_domain::Trihedral_pairs_to_planes" << endl;
	}
	for (t = 0; t < Schlaefli->nb_trihedral_pairs; t++) {
		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				lines_in_tritangent_plane[j] = Schlaefli->Trihedral_pairs[t * 9 + i * 3 + j];
				three_lines[j] = Lines[lines_in_tritangent_plane[j]];
			}
			rk = plane_from_three_lines(three_lines, 0 /* verbose_level */);
			Planes_by_rank[t * 6 + i] = rk;
		}
		for (j = 0; j < 3; j++) {
			for (i = 0; i < 3; i++) {
				lines_in_tritangent_plane[i] = Schlaefli->Trihedral_pairs[t * 9 + i * 3 + j];
				three_lines[i] = Lines[lines_in_tritangent_plane[i]];
			}
			rk = plane_from_three_lines(three_lines, 0 /* verbose_level */);
			Planes_by_rank[t * 6 + 3 + j] = rk;
		}
	}
	if (f_v) {
		cout << "Trihedral_pairs_to_planes:" << endl;
		L.print_lint_matrix_with_standard_labels(cout,
				Planes_by_rank, Schlaefli->nb_trihedral_pairs, 6, FALSE /* f_tex */);
	}
	if (f_v) {
		cout << "surface_domain::Trihedral_pairs_to_planes done" << endl;
	}
}


#if 0
void surface_domain::compute_tritangent_planes_slow(long int *Lines,
	long int *&Tritangent_planes, int &nb_tritangent_planes,
	long int *&Unitangent_planes, int &nb_unitangent_planes,
	long int *&Lines_in_tritangent_plane,
	long int *&Line_in_unitangent_plane,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Inc_lines_planes;
	int *The_plane_type;
	int nb_planes;
	int i, j, h, c;

	if (f_v) {
		cout << "surface_domain::compute_tritangent_planes_slow" << endl;
	}
	if (f_v) {
		cout << "Lines=" << endl;
		lint_vec_print(cout, Lines, 27);
		cout << endl;
	}
	P->line_plane_incidence_matrix_restricted(Lines, 27,
		Inc_lines_planes, nb_planes, 0 /* verbose_level */);

	The_plane_type = NEW_int(nb_planes);
	int_vec_zero(The_plane_type, nb_planes);

	for (j = 0; j < nb_planes; j++) {
		for (i = 0; i < 27; i++) {
			if (Inc_lines_planes[i * nb_planes + j]) {
				The_plane_type[j]++;
			}
		}
	}
	tally Plane_type;

	Plane_type.init(The_plane_type, nb_planes, FALSE, 0);
	if (f_v) {
		cout << "surface_domain::compute_tritangent_planes_slow The plane type is: ";
		Plane_type.print_naked(TRUE);
		cout << endl;
	}


	Plane_type.get_class_by_value_lint(Tritangent_planes,
		nb_tritangent_planes, 3 /* value */, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::compute_tritangent_planes_slow "
				"The tritangent planes are: ";
		lint_vec_print(cout, Tritangent_planes, nb_tritangent_planes);
		cout << endl;
	}
	Lines_in_tritangent_plane = NEW_lint(nb_tritangent_planes * 3);
	for (h = 0; h < nb_tritangent_planes; h++) {
		j = Tritangent_planes[h];
		c = 0;
		for (i = 0; i < 27; i++) {
			if (Inc_lines_planes[i * nb_planes + j]) {
				Lines_in_tritangent_plane[h * 3 + c++] = i;
			}
		}
		if (c != 3) {
			cout << "surface_domain::compute_tritangent_planes_slow c != 3" << endl;
			exit(1);
		}
	}


	Plane_type.get_class_by_value_lint(Unitangent_planes,
		nb_unitangent_planes, 1 /* value */, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::compute_tritangent_planes_slow "
				"The unitangent planes are: ";
		lint_vec_print(cout, Unitangent_planes, nb_unitangent_planes);
		cout << endl;
	}
	Line_in_unitangent_plane = NEW_lint(nb_unitangent_planes);
	for (h = 0; h < nb_unitangent_planes; h++) {
		j = Unitangent_planes[h];
		c = 0;
		for (i = 0; i < 27; i++) {
			if (Inc_lines_planes[i * nb_planes + j]) {
				Line_in_unitangent_plane[h * 1 + c++] = i;
			}
		}
		if (c != 1) {
			cout << "surface_domain::compute_tritangent_planes_slow c != 1" << endl;
			exit(1);
		}
	}

	FREE_int(Inc_lines_planes);
	FREE_int(The_plane_type);

	if (f_v) {
		cout << "surface_domain::compute_tritangent_planes_slow done" << endl;
	}
}
#endif






void surface_domain::prepare_system_from_FG(int *F_planes, int *G_planes,
	int lambda, int *&system, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;


	if (f_v) {
		cout << "surface_domain::prepare_system_from_FG" << endl;
	}
	system = NEW_int(3 * 4 * 3);
	Int_vec_zero(system, 3 * 4 * 3);
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			int *p = system + (i * 4 + j) * 3;
			if (i == 0) {
				p[0] = 0;
				p[1] = F->mult(lambda, G_planes[0 * 4 + j]);
				p[2] = F_planes[2 * 4 + j];
			}
			else if (i == 1) {
				p[0] = F_planes[0 * 4 + j];
				p[1] = 0;
				p[2] = G_planes[1 * 4 + j];
			}
			else if (i == 2) {
				p[0] = G_planes[2 * 4 + j];
				p[1] = F_planes[1 * 4 + j];
				p[2] = 0;
			}
		}
	}
	if (f_v) {
		cout << "surface_domain::prepare_system_from_FG done" << endl;
	}
}


void surface_domain::compute_nine_lines(int *F_planes, int *G_planes,
	long int *nine_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int Basis[16];

	if (f_v) {
		cout << "surface_domain::compute_nine_lines" << endl;
	}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			Int_vec_copy(F_planes + i * 4, Basis, 4);
			Int_vec_copy(G_planes + j * 4, Basis + 4, 4);
			F->Linear_algebra->RREF_and_kernel(4, 2, Basis, 0 /* verbose_level */);
			nine_lines[i * 3 + j] = Gr->rank_lint_here(
				Basis + 8, 0 /* verbose_level */);
		}
	}
	if (f_v) {
		cout << "The nine lines are: ";
		Lint_vec_print(cout, nine_lines, 9);
		cout << endl;
	}
	if (f_v) {
		cout << "surface_domain::compute_nine_lines done" << endl;
	}
}

void surface_domain::compute_nine_lines_by_dual_point_ranks(
	long int *F_planes_rank,
	long int *G_planes_rank, long int *nine_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int F_planes[12];
	int G_planes[12];
	int Basis[16];

	if (f_v) {
		cout << "surface_domain::compute_nine_lines_by_dual_point_ranks" << endl;
	}
	for (i = 0; i < 3; i++) {
		P->unrank_point(F_planes + i * 4, F_planes_rank[i]);
	}
	for (i = 0; i < 3; i++) {
		P->unrank_point(G_planes + i * 4, G_planes_rank[i]);
	}

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			Int_vec_copy(F_planes + i * 4, Basis, 4);
			Int_vec_copy(G_planes + j * 4, Basis + 4, 4);
			F->Linear_algebra->RREF_and_kernel(4, 2, Basis, 0 /* verbose_level */);
			nine_lines[i * 3 + j] = Gr->rank_lint_here(
				Basis + 8, 0 /* verbose_level */);
		}
	}
	if (f_v) {
		cout << "The nine lines are: ";
		Lint_vec_print(cout, nine_lines, 9);
		cout << endl;
	}
	if (f_v) {
		cout << "surface_domain::compute_nine_lines_by_dual_point_ranks done" << endl;
	}
}


void surface_domain::tritangent_plane_to_trihedral_pair_and_position(
	int tritangent_plane_idx,
	int &trihedral_pair_idx, int &position, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	static int Table[] = {
		0, 2, // 0
		0, 5, // 1
		22, 0, // 2
		0, 1, // 3
		20, 0, // 4
		1, 1, // 5
		26, 0, //6
		5, 1, // 7
		32, 0, //8
		6, 1, //9
		0, 0, //10
		25, 0, // 11
		1, 0, // 12
		43, 0, //13
		2, 0, //14
		55, 0, // 15
		3, 0, // 16
		3, 3, //17
		4, 0, //18
		67, 0, // 19
		5, 0, // 20
		73, 0, // 21
		6, 0, // 22
		6, 3, // 23
		7, 0, // 24
		79, 0, // 25
		8, 0, // 26
		8, 3, // 27
		9, 0, // 28
		9, 3, // 29
		115, 0, // 30
		114, 0, // 31
		34, 2, // 32
		113, 0, // 33
		111, 0, // 34
		34, 5, // 35
		74, 2, // 36
		110, 0, // 37
		49, 2, // 38
		26, 5, // 39
		38, 5, // 40
		53, 5, // 41
		36, 5, // 42
		45, 5, // 43
		51, 5, // 44
		};

	if (f_v) {
		cout << "surface_domain::tritangent_plane_to_trihedral_pair_and_position" << endl;
	}
	trihedral_pair_idx = Table[2 * tritangent_plane_idx + 0];
	position = Table[2 * tritangent_plane_idx + 1];
	if (f_v) {
		cout << "surface_domain::tritangent_plane_to_trihedral_pair_and_position done" << endl;
	}
}

void surface_domain::do_arc_lifting_with_two_lines(
	long int *Arc6, int p1_idx, int p2_idx, int partition_rk,
	long int line1, long int line2,
	int *coeff20, long int *lines27,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int arc[6];
	long int P1, P2;
	geometry::geometry_global Gg;


	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines" << endl;
		cout << "Arc6: ";
		Lint_vec_print(cout, Arc6, 6);
		cout << endl;
		cout << "p1_idx=" << p1_idx << " p2_idx=" << p2_idx
				<< " partition_rk=" << partition_rk
				<< " line1=" << line1 << " line2=" << line2 << endl;
	}

	P1 = Arc6[p1_idx];
	P2 = Arc6[p2_idx];

	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines before "
				"Gg.rearrange_arc_for_lifting" << endl;
	}
	Gg.rearrange_arc_for_lifting(Arc6,
				P1, P2, partition_rk, arc,
				verbose_level);

	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines after "
				"Gg.rearrange_arc_for_lifting" << endl;
		cout << "arc: ";
		Lint_vec_print(cout, arc, 6);
		cout << endl;
	}

	arc_lifting_with_two_lines *AL;

	AL = NEW_OBJECT(arc_lifting_with_two_lines);


	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines before "
				"AL->create_surface" << endl;
	}
	AL->create_surface(this, arc, line1, line2, verbose_level);
	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines after "
				"AL->create_surface" << endl;
		cout << "equation: ";
		Int_vec_print(cout, AL->coeff, 20);
		cout << endl;
		cout << "lines: ";
		Lint_vec_print(cout, AL->lines27, 27);
		cout << endl;
	}

	Int_vec_copy(AL->coeff, coeff20, 20);
	Lint_vec_copy(AL->lines27, lines27, 27);


	FREE_OBJECT(AL);


	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines done" << endl;
	}
}

void surface_domain::compute_local_coordinates_of_arc(
		long int *P6, long int *P6_local, int verbose_level)
// assuming we are in the hyperplane W=0
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::compute_local_coordinates_of_arc" << endl;
	}

	int i;
	int v[4];
	int base_cols[3] = {0, 1, 2};
	int coefficients[3];
	int Basis_of_hyperplane[12] = { 1,0,0,0, 0,1,0,0, 0,0,1,0 };

	for (i = 0; i < 6; i++) {
		if (f_v) {
			cout << "surface_domain::compute_local_coordinates_of_arc "
					"i=" << i << endl;
		}
		P->unrank_point(v, P6[i]);
		if (f_v) {
			cout << "surface_domain::compute_local_coordinates_of_arc "
					"which is ";
			Int_vec_print(cout, v, 4);
			cout << endl;
		}
		F->Linear_algebra->reduce_mod_subspace_and_get_coefficient_vector(
			3, 4, Basis_of_hyperplane, base_cols,
			v, coefficients,
			0 /* verbose_level */);
		if (f_v) {
			cout << "surface_domain::compute_local_coordinates_of_arc "
					"local coefficients ";
			Int_vec_print(cout, coefficients, 3);
			cout << endl;
		}
		F->PG_element_rank_modified_lint(coefficients, 1, 3, P6_local[i]);
	}
	if (f_v) {
		cout << "surface_domain::compute_local_coordinates_of_arc" << endl;
		cout << "P6_local=" << endl;
		Lint_vec_print(cout, P6_local, 6);
		cout << endl;
	}

	if (f_v) {
		cout << "surface_domain::compute_local_coordinates_of_arc done" << endl;
	}
}



}}}

