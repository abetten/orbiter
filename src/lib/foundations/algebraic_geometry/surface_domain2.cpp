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


void surface_domain::multiply_conic_times_linear(int *six_coeff,
	int *three_coeff, int *ten_coeff,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx, u;
	int M[3];

	if (f_v) {
		cout << "surface_domain::multiply_conic_times_linear" << endl;
	}


	Orbiter->Int_vec->zero(ten_coeff, 10);
	for (i = 0; i < 6; i++) {
		a = six_coeff[i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < 3; j++) {
			b = three_coeff[j];
			if (b == 0) {
				continue;
			}
			c = F->mult(a, b);

			for (u = 0; u < 3; u++) {
				M[u] = Poly2->get_monomial(i, u) + Poly1->get_monomial(j, u);
			}
			idx = Poly3->index_of_monomial(M);
			if (idx >= 10) {
				cout << "surface_domain::multiply_conic_times_linear "
						"idx >= 10" << endl;
				exit(1);
				}
			ten_coeff[idx] = F->add(ten_coeff[idx], c);
		}
	}


	if (f_v) {
		cout << "surface_domain::multiply_conic_times_linear done" << endl;
	}
}

void surface_domain::multiply_linear_times_linear_times_linear(
	int *three_coeff1, int *three_coeff2, int *three_coeff3,
	int *ten_coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, a, b, c, d, idx, u;
	int M[3];

	if (f_v) {
		cout << "surface_domain::multiply_linear_times_linear_times_linear" << endl;
	}

	Orbiter->Int_vec->zero(ten_coeff, 10);
	for (i = 0; i < 3; i++) {
		a = three_coeff1[i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < 3; j++) {
			b = three_coeff2[j];
			if (b == 0) {
				continue;
			}
			for (k = 0; k < 3; k++) {
				c = three_coeff3[k];
				if (c == 0) {
					continue;
				}
				d = F->mult3(a, b, c);
				for (u = 0; u < 3; u++) {
					M[u] = Poly1->get_monomial(i, u)
							+ Poly1->get_monomial(j, u)
							+ Poly1->get_monomial(k, u);
				}
				idx = Poly3->index_of_monomial(M);
				if (idx >= 10) {
					cout << "surface::multiply_linear_times_"
							"linear_times_linear idx >= 10" << endl;
					exit(1);
					}
				ten_coeff[idx] = F->add(ten_coeff[idx], d);
			}
		}
	}


	if (f_v) {
		cout << "surface_domain::multiply_linear_times_linear_times_linear done" << endl;
	}
}

void surface_domain::multiply_linear_times_linear_times_linear_in_space(
	int *four_coeff1, int *four_coeff2, int *four_coeff3,
	int *twenty_coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, a, b, c, d, idx, u;
	int M[4];

	if (f_v) {
		cout << "surface_domain::multiply_linear_times_linear_times_linear_in_space" << endl;
	}

	Orbiter->Int_vec->zero(twenty_coeff, 20);
	for (i = 0; i < 4; i++) {
		a = four_coeff1[i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < 4; j++) {
			b = four_coeff2[j];
			if (b == 0) {
				continue;
			}
			for (k = 0; k < 4; k++) {
				c = four_coeff3[k];
				if (c == 0) {
					continue;
				}
				d = F->mult3(a, b, c);
				for (u = 0; u < 4; u++) {
					M[u] = Poly1_4->get_monomial(i, u)
							+ Poly1_4->get_monomial(j, u)
							+ Poly1_4->get_monomial(k, u);
				}
				idx = index_of_monomial(M);
				if (idx >= 20) {
					cout << "surface_domain::multiply_linear_times_linear_"
							"times_linear_in_space idx >= 20" << endl;
					exit(1);
					}
				twenty_coeff[idx] = F->add(twenty_coeff[idx], d);
			}
		}
	}


	if (f_v) {
		cout << "surface_domain::multiply_linear_times_linear_times_linear_in_space done" << endl;
	}
}

void surface_domain::multiply_Poly2_3_times_Poly2_3(
	int *input1, int *input2,
	int *result, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx, u;
	int M[3];

	if (f_v) {
		cout << "surface_domain::multiply_Poly2_3_times_Poly2_3" << endl;
	}

	Orbiter->Int_vec->zero(result, Poly4_x123->get_nb_monomials());
	for (i = 0; i < Poly2->get_nb_monomials(); i++) {
		a = input1[i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < Poly2->get_nb_monomials(); j++) {
			b = input2[j];
			if (b == 0) {
				continue;
			}
			c = F->mult(a, b);

			for (u = 0; u < 3; u++) {
				M[u] = Poly2->get_monomial(i, u) + Poly2->get_monomial(j, u);
			}
			idx = Poly4_x123->index_of_monomial(M);
			result[idx] = F->add(result[idx], c);
		}
	}


	if (f_v) {
		cout << "surface_domain::multiply_Poly2_3_times_Poly2_3 done" << endl;
	}
}

void surface_domain::multiply_Poly1_3_times_Poly3_3(int *input1, int *input2,
	int *result, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx, u;
	int M[3];

	if (f_v) {
		cout << "surface_domain::multiply_Poly1_3_times_Poly3_3" << endl;
	}

	Orbiter->Int_vec->zero(result, Poly4_x123->get_nb_monomials());
	for (i = 0; i < Poly1->get_nb_monomials(); i++) {
		a = input1[i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < Poly3->get_nb_monomials(); j++) {
			b = input2[j];
			if (b == 0) {
				continue;
			}
			c = F->mult(a, b);
			for (u = 0; u < 3; u++) {
				M[u] = Poly1->get_monomial(i, u) + Poly3->get_monomial(j, u);
			}
			idx = Poly4_x123->index_of_monomial(M);
			result[idx] = F->add(result[idx], c);
		}
	}

	if (f_v) {
		cout << "surface_domain::multiply_Poly1_3_times_Poly3_3 done" << endl;
	}
}

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


	multiply_linear_times_linear_times_linear_in_space(
		The_six_plane_equations + 0 * 4,
		The_six_plane_equations + 1 * 4,
		The_six_plane_equations + 2 * 4,
		eqn_F, FALSE /* verbose_level */);
	multiply_linear_times_linear_times_linear_in_space(
		The_six_plane_equations + 3 * 4,
		The_six_plane_equations + 4 * 4,
		The_six_plane_equations + 5 * 4,
		eqn_G, FALSE /* verbose_level */);


	for (l = 0; l < q + 1; l++) {
		F->PG_element_unrank_modified(v, 1, 2, l);

		Orbiter->Int_vec->copy(eqn_F, eqn_F2, 20);
		F->Linear_algebra->scalar_multiply_vector_in_place(v[0], eqn_F2, 20);
		Orbiter->Int_vec->copy(eqn_G, eqn_G2, 20);
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
	latex_interface L;

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





void surface_domain::clebsch_cubics(int verbose_level)
{
	int f_v = (verbose_level >= 1);



	if (f_v) {
		cout << "surface_domain::clebsch_cubics" << endl;
	}

	if (!f_has_large_polynomial_domains) {
		cout << "surface::clebsch_cubics f_has_large_"
				"polynomial_domains is FALSE" << endl;
		exit(1);
	}
	int Monomial[27];

	int i, j, idx;

	Clebsch_Pij = NEW_int(3 * 4 * nb_monomials2);
	Clebsch_P = NEW_pint(3 * 4);
	Clebsch_P3 = NEW_pint(3 * 3);

	Orbiter->Int_vec->zero(Clebsch_Pij, 3 * 4 * nb_monomials2);


	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			Clebsch_P[i * 4 + j] =
				Clebsch_Pij + (i * 4 + j) * nb_monomials2;
		}
	}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			Clebsch_P3[i * 3 + j] =
				Clebsch_Pij + (i * 4 + j) * nb_monomials2;
		}
	}
	int coeffs[] = {
		1, 15, 2, 11,
		1, 16, 2, 12,
		1, 17, 2, 13,
		1, 18, 2, 14,
		0, 3, 2, 19,
		0, 4, 2, 20,
		0, 5, 2, 21,
		0, 6, 2, 22,
		0, 23, 1, 7,
		0, 24, 1, 8,
		0, 25, 1, 9,
		0, 26, 1, 10
	};
	int c0, c1;

	if (f_v) {
		cout << "surface_domain::clebsch_cubics "
				"Setting up the matrix P:" << endl;
	}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			cout << "i=" << i << " j=" << j << endl;
			Orbiter->Int_vec->zero(Monomial, 27);
			c0 = coeffs[(i * 4 + j) * 4 + 0];
			c1 = coeffs[(i * 4 + j) * 4 + 1];
			Orbiter->Int_vec->zero(Monomial, 27);
			Monomial[c0] = 1;
			Monomial[c1] = 1;
			idx = Poly2_27->index_of_monomial(Monomial);
			Clebsch_P[i * 4 + j][idx] = 1;
			c0 = coeffs[(i * 4 + j) * 4 + 2];
			c1 = coeffs[(i * 4 + j) * 4 + 3];
			Orbiter->Int_vec->zero(Monomial, 27);
			Monomial[c0] = 1;
			Monomial[c1] = 1;
			idx = Poly2_27->index_of_monomial(Monomial);
			Clebsch_P[i * 4 + j][idx] = 1;
		}
	}


	if (f_v) {
		cout << "surface_domain::clebsch_cubics the matrix "
				"Clebsch_P is:" << endl;
	}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			cout << "Clebsch_P_" << i << "," << j << ":";
			Poly2_27->print_equation(cout, Clebsch_P[i * 4 + j]);
			cout << endl;
		}
	}

	int *Cubics;
	int *Adjugate;
	int *Ad[3 * 3];
	int *C[4];
	int m1;


	if (f_v) {
		cout << "surface_domain::clebsch_cubics allocating cubics" << endl;
	}

	Cubics = NEW_int(4 * nb_monomials6);
	Orbiter->Int_vec->zero(Cubics, 4 * nb_monomials6);

	Adjugate = NEW_int(3 * 3 * nb_monomials4);
	Orbiter->Int_vec->zero(Adjugate, 3 * 3 * nb_monomials4);

	for (i = 0; i < 4; i++) {
		C[i] = Cubics + i * nb_monomials6;
	}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			Ad[i * 3 + j] = Adjugate + (i * 3 + j) * nb_monomials4;
		}
	}

	if (f_v) {
		cout << "surface_domain::clebsch_cubics computing "
				"C[3] = the determinant" << endl;
	}
	// compute C[3] as the negative of the determinant
	// of the matrix of the first three columns:
	//int_vec_zero(C[3], nb_monomials6);
	m1 = F->negate(1);
	multiply_222_27_and_add(Clebsch_P[0 * 4 + 0],
			Clebsch_P[1 * 4 + 1],
			Clebsch_P[2 * 4 + 2], m1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[0 * 4 + 1],
			Clebsch_P[1 * 4 + 2],
			Clebsch_P[2 * 4 + 0], m1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[0 * 4 + 2],
			Clebsch_P[1 * 4 + 0],
			Clebsch_P[2 * 4 + 1], m1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[2 * 4 + 0],
			Clebsch_P[1 * 4 + 1],
			Clebsch_P[0 * 4 + 2], 1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[2 * 4 + 1],
			Clebsch_P[1 * 4 + 2],
			Clebsch_P[0 * 4 + 0], 1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[2 * 4 + 2],
			Clebsch_P[1 * 4 + 0],
			Clebsch_P[0 * 4 + 1], 1, C[3],
			0 /* verbose_level*/);

	int I[3];
	int J[3];
	int size_complement, scalar;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "surface_domain::clebsch_cubics computing adjugate" << endl;
	}
	// compute adjugate:
	for (i = 0; i < 3; i++) {
		I[0] = i;
		Combi.set_complement(I, 1, I + 1, size_complement, 3);
		for (j = 0; j < 3; j++) {
			J[0] = j;
			Combi.set_complement(J, 1, J + 1, size_complement, 3);

			if ((i + j) % 2) {
				scalar = m1;
			}
			else {
				scalar = 1;
			}
			minor22(Clebsch_P3, I[1], I[2], J[1], J[2], scalar,
					Ad[j * 3 + i], 0 /* verbose_level */);
		}
	}

	// multiply adjugate * last column:
	if (f_v) {
		cout << "surface_domain::clebsch_cubics multiply adjugate "
				"times last column" << endl;
	}

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			multiply42_and_add(Ad[i * 3 + j],
					Clebsch_P[j * 4 + 3], C[i], 0 /* verbose_level */);
		}
	}

	if (f_v) {
		cout << "surface_domain::clebsch_cubics We have "
				"computed the Clebsch cubics" << endl;
	}


	int Y[3];
	int M24[24];
	int h;
	data_structures::sorting Sorting;

	Clebsch_coeffs = NEW_int(4 * Poly3->get_nb_monomials() * nb_monomials3);
	Orbiter->Int_vec->zero(Clebsch_coeffs,
			4 * Poly3->get_nb_monomials() * nb_monomials3);
	CC = NEW_pint(4 * Poly3->get_nb_monomials());
	for (i = 0; i < 4; i++) {
		for (j = 0; j < Poly3->get_nb_monomials(); j++) {
			CC[i * Poly3->get_nb_monomials() + j] =
				Clebsch_coeffs + (i * Poly3->get_nb_monomials() + j) * nb_monomials3;
		}
	}
	for (i = 0; i < Poly3->get_nb_monomials(); i++) {
		Orbiter->Int_vec->copy(Poly3->get_monomial_pointer(i), Y, 3);
		for (j = 0; j < nb_monomials6; j++) {
			if (Sorting.int_vec_compare(Y, Poly6_27->get_monomial_pointer(j), 3) == 0) {
				Orbiter->Int_vec->copy(Poly6_27->get_monomial_pointer(j) + 3, M24, 24);
				idx = Poly3_24->index_of_monomial(M24);
				for (h = 0; h < 4; h++) {
					CC[h * Poly3->get_nb_monomials() + i][idx] =
						F->add(CC[h * Poly3->get_nb_monomials() + i][idx], C[h][j]);
				}
			}
		}
	}

	if (f_v) {
		print_clebsch_cubics(cout);
	}

	FREE_int(Cubics);
	FREE_int(Adjugate);

	if (f_v) {
		cout << "surface_domain::clebsch_cubics done" << endl;
	}
}

void surface_domain::multiply_222_27_and_add(int *M1, int *M2, int *M3,
	int scalar, int *MM, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, a, b, c, d, idx;
	int M[27];

	if (f_v) {
		cout << "surface_domain::multiply_222_27_and_add" << endl;
	}

	if (!f_has_large_polynomial_domains) {
		cout << "surface_domain::multiply_222_27_and_add "
				"f_has_large_polynomial_domains is FALSE" << endl;
		exit(1);
	}
	//int_vec_zero(MM, nb_monomials6);
	for (i = 0; i < nb_monomials2; i++) {
		a = M1[i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < nb_monomials2; j++) {
			b = M2[j];
			if (b == 0) {
				continue;
			}
			for (k = 0; k < nb_monomials2; k++) {
				c = M3[k];
				if (c == 0) {
					continue;
				}
				d = F->mult3(a, b, c);
				Orbiter->Int_vec->add3(Poly2_27->get_monomial_pointer(i),
					Poly2_27->get_monomial_pointer(j),
					Poly2_27->get_monomial_pointer(k),
					M, 27);
				idx = Poly6_27->index_of_monomial(M);
				if (idx >= nb_monomials6) {
					cout << "surface_domain::multiply_222_27_and_add "
							"idx >= nb_monomials6" << endl;
					exit(1);
					}
				d = F->mult(scalar, d);
				MM[idx] = F->add(MM[idx], d);
			}
		}
	}


	if (f_v) {
		cout << "surface_domain::multiply_222_27_and_add done" << endl;
	}
}

void surface_domain::minor22(int **P3, int i1, int i2, int j1, int j2,
	int scalar, int *Ad, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, d, idx;
	int M[27];

	if (f_v) {
		cout << "surface_domain::minor22" << endl;
	}

	if (!f_has_large_polynomial_domains) {
		cout << "surface_domain::minor22 "
				"f_has_large_polynomial_domains is FALSE" << endl;
		exit(1);
	}
	Orbiter->Int_vec->zero(Ad, nb_monomials4);
	for (i = 0; i < nb_monomials2; i++) {
		a = P3[i1 * 3 + j1][i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < nb_monomials2; j++) {
			b = P3[i2 * 3 + j2][j];
			if (b == 0) {
				continue;
			}
			d = F->mult(a, b);
			Orbiter->Int_vec->add(Poly2_27->get_monomial_pointer(i),
				Poly2_27->get_monomial_pointer(j),
				M, 27);
			idx = Poly4_27->index_of_monomial(M);
			if (idx >= nb_monomials4) {
				cout << "surface_domain::minor22 "
						"idx >= nb_monomials4" << endl;
				exit(1);
			}
			d = F->mult(scalar, d);
			Ad[idx] = F->add(Ad[idx], d);
		}
	}
	for (i = 0; i < nb_monomials2; i++) {
		a = P3[i2 * 3 + j1][i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < nb_monomials2; j++) {
			b = P3[i1 * 3 + j2][j];
			if (b == 0) {
				continue;
			}
			d = F->mult(a, b);
			Orbiter->Int_vec->add(Poly2_27->get_monomial_pointer(i),
				Poly2_27->get_monomial_pointer(j),
				M, 27);
			idx = Poly4_27->index_of_monomial(M);
			if (idx >= nb_monomials4) {
				cout << "surface_domain::minor22 "
						"idx >= nb_monomials4" << endl;
				exit(1);
			}
			d = F->mult(scalar, d);
			d = F->negate(d);
			Ad[idx] = F->add(Ad[idx], d);
		}
	}


	if (f_v) {
		cout << "surface_domain::minor22 done" << endl;
	}
}

void surface_domain::multiply42_and_add(int *M1, int *M2,
		int *MM, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, d, idx;
	int M[27];

	if (f_v) {
		cout << "surface_domain::multiply42_and_add" << endl;
	}

	if (!f_has_large_polynomial_domains) {
		cout << "surface_domain::multiply42_and_add "
				"f_has_large_polynomial_domains is FALSE" << endl;
		exit(1);
	}
	for (i = 0; i < nb_monomials4; i++) {
		a = M1[i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < nb_monomials2; j++) {
			b = M2[j];
			if (b == 0) {
				continue;
			}
			d = F->mult(a, b);
			Orbiter->Int_vec->add(Poly4_27->get_monomial_pointer(i),
				Poly2_27->get_monomial_pointer(j),
				M, 27);
			idx = Poly6_27->index_of_monomial(M);
			if (idx >= nb_monomials6) {
				cout << "surface_domain::multiply42_and_add "
						"idx >= nb_monomials6" << endl;
				exit(1);
			}
			MM[idx] = F->add(MM[idx], d);
		}
	}

	if (f_v) {
		cout << "surface_domain::multiply42_and_add done" << endl;
	}
}

void surface_domain::prepare_system_from_FG(int *F_planes, int *G_planes,
	int lambda, int *&system, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;


	if (f_v) {
		cout << "surface_domain::prepare_system_from_FG" << endl;
	}
	system = NEW_int(3 * 4 * 3);
	Orbiter->Int_vec->zero(system, 3 * 4 * 3);
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
			Orbiter->Int_vec->copy(F_planes + i * 4, Basis, 4);
			Orbiter->Int_vec->copy(G_planes + j * 4, Basis + 4, 4);
			F->Linear_algebra->RREF_and_kernel(4, 2, Basis, 0 /* verbose_level */);
			nine_lines[i * 3 + j] = Gr->rank_lint_here(
				Basis + 8, 0 /* verbose_level */);
		}
	}
	if (f_v) {
		cout << "The nine lines are: ";
		Orbiter->Lint_vec->print(cout, nine_lines, 9);
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
			Orbiter->Int_vec->copy(F_planes + i * 4, Basis, 4);
			Orbiter->Int_vec->copy(G_planes + j * 4, Basis + 4, 4);
			F->Linear_algebra->RREF_and_kernel(4, 2, Basis, 0 /* verbose_level */);
			nine_lines[i * 3 + j] = Gr->rank_lint_here(
				Basis + 8, 0 /* verbose_level */);
		}
	}
	if (f_v) {
		cout << "The nine lines are: ";
		Orbiter->Lint_vec->print(cout, nine_lines, 9);
		cout << endl;
	}
	if (f_v) {
		cout << "surface_domain::compute_nine_lines_by_dual_point_ranks done" << endl;
	}
}

void surface_domain::split_nice_equation(int *nice_equation,
	int *&f1, int *&f2, int *&f3, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::split_nice_equation" << endl;
	}
	int M[4];
	int i, a, idx;

	f1 = NEW_int(Poly1->get_nb_monomials());
	f2 = NEW_int(Poly2->get_nb_monomials());
	f3 = NEW_int(Poly3->get_nb_monomials());
	Orbiter->Int_vec->zero(f1, Poly1->get_nb_monomials());
	Orbiter->Int_vec->zero(f2, Poly2->get_nb_monomials());
	Orbiter->Int_vec->zero(f3, Poly3->get_nb_monomials());

	for (i = 0; i < 20; i++) {
		a = nice_equation[i];
		if (a == 0) {
			continue;
		}
		Orbiter->Int_vec->copy(Poly3_4->get_monomial_pointer(i), M, 4);
		if (M[0] == 3) {
			cout << "surface_domain::split_nice_equation the x_0^3 "
				"term is supposed to be zero" << endl;
			exit(1);
		}
		else if (M[0] == 2) {
			idx = Poly1->index_of_monomial(M + 1);
			f1[idx] = a;
		}
		else if (M[0] == 1) {
			idx = Poly2->index_of_monomial(M + 1);
			f2[idx] = a;
		}
		else if (M[0] == 0) {
			idx = Poly3->index_of_monomial(M + 1);
			f3[idx] = a;
		}
	}
	if (f_v) {
		cout << "surface_domain::split_nice_equation done" << endl;
	}
}

void surface_domain::assemble_tangent_quadric(
	int *f1, int *f2, int *f3,
	int *&tangent_quadric, int verbose_level)
// 2*x_0*f_1 + f_2
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::assemble_tangent_quadric" << endl;
	}
	int M[4];
	int i, a, idx, two;


	two = F->add(1, 1);
	tangent_quadric = NEW_int(Poly2_4->get_nb_monomials());
	Orbiter->Int_vec->zero(tangent_quadric, Poly2_4->get_nb_monomials());

	for (i = 0; i < Poly1->get_nb_monomials(); i++) {
		a = f1[i];
		if (a == 0) {
			continue;
		}
		Orbiter->Int_vec->copy(Poly1->get_monomial_pointer(i), M + 1, 3);
		M[0] = 1;
		idx = Poly2_4->index_of_monomial(M);
		tangent_quadric[idx] = F->mult(two, a);
	}

	for (i = 0; i < Poly2->get_nb_monomials(); i++) {
		a = f2[i];
		if (a == 0) {
			continue;
		}
		Orbiter->Int_vec->copy(Poly2->get_monomial_pointer(i), M + 1, 3);
		M[0] = 0;
		idx = Poly2_4->index_of_monomial(M);
		tangent_quadric[idx] = a;
	}

	if (f_v) {
		cout << "surface_domain::assemble_tangent_quadric done" << endl;
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


	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines" << endl;
		cout << "Arc6: ";
		Orbiter->Lint_vec->print(cout, Arc6, 6);
		cout << endl;
		cout << "p1_idx=" << p1_idx << " p2_idx=" << p2_idx
				<< " partition_rk=" << partition_rk
				<< " line1=" << line1 << " line2=" << line2 << endl;
	}

	P1 = Arc6[p1_idx];
	P2 = Arc6[p2_idx];

	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines before "
				"P->rearrange_arc_for_lifting" << endl;
	}
	P->rearrange_arc_for_lifting(Arc6,
				P1, P2, partition_rk, arc,
				verbose_level);

	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines after "
				"P->rearrange_arc_for_lifting" << endl;
		cout << "arc: ";
		Orbiter->Lint_vec->print(cout, arc, 6);
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
		Orbiter->Int_vec->print(cout, AL->coeff, 20);
		cout << endl;
		cout << "lines: ";
		Orbiter->Lint_vec->print(cout, AL->lines27, 27);
		cout << endl;
	}

	Orbiter->Int_vec->copy(AL->coeff, coeff20, 20);
	Orbiter->Lint_vec->copy(AL->lines27, lines27, 27);


	FREE_OBJECT(AL);


	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines done" << endl;
	}
}

void surface_domain::compute_local_coordinates_of_arc(
		long int *P6, long int *P6_local, int verbose_level)
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
			Orbiter->Int_vec->print(cout, v, 4);
			cout << endl;
		}
		F->Linear_algebra->reduce_mod_subspace_and_get_coefficient_vector(
			3, 4, Basis_of_hyperplane, base_cols,
			v, coefficients,
			0 /* verbose_level */);
		if (f_v) {
			cout << "surface_domain::compute_local_coordinates_of_arc "
					"local coefficients ";
			Orbiter->Int_vec->print(cout, coefficients, 3);
			cout << endl;
		}
		F->PG_element_rank_modified_lint(coefficients, 1, 3, P6_local[i]);
	}
	if (f_v) {
		cout << "surface_domain::compute_local_coordinates_of_arc" << endl;
		cout << "P6_local=" << endl;
		Orbiter->Lint_vec->print(cout, P6_local, 6);
		cout << endl;
	}

	if (f_v) {
		cout << "surface_domain::compute_local_coordinates_of_arc done" << endl;
	}
}

void surface_domain::compute_gradient(int *equation20, int *&gradient, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "surface_domain::compute_gradient" << endl;
	}


	if (f_v) {
		cout << "surface_domain::compute_gradient Poly2_4->get_nb_monomials() = " << Poly2_4->get_nb_monomials() << endl;
	}

	gradient = NEW_int(4 * Poly2_4->get_nb_monomials());

	for (i = 0; i < 4; i++) {
		if (f_v) {
			cout << "surface_domain::compute_gradient i=" << i << endl;
		}
		if (f_v) {
			cout << "surface_domain::compute_gradient eqn_in=";
			Orbiter->Int_vec->print(cout, equation20, 20);
			cout << " = " << endl;
			Poly3_4->print_equation(cout, equation20);
			cout << endl;
		}
		Partials[i].apply(equation20,
				gradient + i * Poly2_4->get_nb_monomials(),
				verbose_level - 2);
		if (f_v) {
			cout << "surface_domain::compute_gradient "
					"partial=";
			Orbiter->Int_vec->print(cout, gradient + i * Poly2_4->get_nb_monomials(),
					Poly2_4->get_nb_monomials());
			cout << " = ";
			Poly2_4->print_equation(cout,
					gradient + i * Poly2_4->get_nb_monomials());
			cout << endl;
		}
	}


	if (f_v) {
		cout << "surface_domain::compute_gradient done" << endl;
	}
}

long int surface_domain::compute_tangent_plane(int *pt_coords, int *equation20, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_eqns = 4;
	int i;
	int w[4];
	int *gradient;

	if (f_v) {
		cout << "surface_domain::compute_tangent_plane" << endl;
	}
	if (f_v) {
		cout << "surface_domain::compute_tangent_plane before compute_gradient" << endl;
	}
	compute_gradient(equation20, gradient, verbose_level - 2);
	if (f_v) {
		cout << "surface_domain::compute_tangent_plane after compute_gradient" << endl;
	}

	for (i = 0; i < nb_eqns; i++) {
		if (f_vv) {
			cout << "surface_domain::compute_tangent_plane "
					"gradient i=" << i << " / " << nb_eqns << endl;
		}
		if (FALSE) {
			cout << "surface_domain::compute_tangent_plane "
					"gradient " << i << " = ";
			Orbiter->Int_vec->print(cout,
					gradient + i * Poly2_4->get_nb_monomials(),
					Poly2_4->get_nb_monomials());
			cout << endl;
		}
		w[i] = Poly2_4->evaluate_at_a_point(
				gradient + i * Poly2_4->get_nb_monomials(), pt_coords);
		if (f_vv) {
			cout << "surface_domain::compute_tangent_plane "
					"value = " << w[i] << endl;
		}
	}
	for (i = 0; i < nb_eqns; i++) {
		if (w[i]) {
			break;
		}
	}

	if (i == nb_eqns) {
		cout << "surface_domain::compute_tangent_plane the point is singular" << endl;
		exit(1);
	}
	long int plane_rk;

	plane_rk = P->plane_rank_using_dual_coordinates_in_three_space(
			w /* eqn4 */,
			0 /* verbose_level*/);

	FREE_int(gradient);

	return plane_rk;
}



}}}

