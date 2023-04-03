/*
 * quartic_curve_domain.cpp
 *
 *  Created on: May 21, 2021
 *      Author: betten
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {


quartic_curve_domain::quartic_curve_domain()
{
	F = NULL;
	P = NULL;
	Poly1_3 = NULL;
	Poly2_3 = NULL;
	Poly3_3 = NULL;
	Poly4_3 = NULL;
	Poly3_4 = NULL;
	Partials = NULL;
	Schlaefli = NULL;
}


quartic_curve_domain::~quartic_curve_domain()
{
	if (Schlaefli) {
		FREE_OBJECT(Schlaefli);
	}
}


void quartic_curve_domain::init(
		field_theory::finite_field *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_domain::init" << endl;
	}

	quartic_curve_domain::F = F;

	P = NEW_OBJECT(geometry::projective_space);
	if (f_v) {
		cout << "quartic_curve_domain::init "
				"before P->projective_space_init" << endl;
	}
	P->projective_space_init(2, F,
		true /*f_init_incidence_structure */,
		verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_domain::init "
				"after P->projective_space_init" << endl;
	}



	if (f_v) {
		cout << "quartic_curve_domain::init "
				"before init_polynomial_domains" << endl;
	}
	init_polynomial_domains(verbose_level);
	if (f_v) {
		cout << "quartic_curve_domain::init "
				"after init_polynomial_domains" << endl;
	}


	Schlaefli = NEW_OBJECT(algebraic_geometry::schlaefli_labels);
	Schlaefli->init(verbose_level);

	if (f_v) {
		cout << "quartic_curve_domain::init done" << endl;
	}
}

void quartic_curve_domain::init_polynomial_domains(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_domain::init_polynomial_domains" << endl;
	}


	Poly1_3 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"before Poly1_3->init" << endl;
	}
	Poly1_3->init(F,
			3 /* nb_vars */, 1 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"after Poly1_3->init" << endl;
	}



	Poly2_3 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"before Poly2_3->init" << endl;
	}
	Poly2_3->init(F,
			3 /* nb_vars */, 2 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"after Poly2_3->init" << endl;
	}



	Poly3_3 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"before Poly3_3->init" << endl;
	}
	Poly3_3->init(F,
			3 /* nb_vars */, 3 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"after Poly3_3->init" << endl;
	}


	Poly4_3 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"before Poly4_3->init" << endl;
	}
	Poly4_3->init(F,
			3 /* nb_vars */, 4 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"after Poly4_3->init" << endl;
	}

	Poly3_4 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"before Poly3_4->init" << endl;
	}
	Poly3_4->init(F,
			4 /* nb_vars */, 3 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"after Poly3_4->init" << endl;
	}

	Partials = NEW_OBJECTS(ring_theory::partial_derivative, 3);

	int i;

	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"initializing partials" << endl;
	}
	for (i = 0; i < 3; i++) {
		Partials[i].init(Poly4_3, Poly3_3, i, verbose_level);
	}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"initializing partials done" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_domain::init_polynomial_domains done" << endl;
	}

}

void quartic_curve_domain::print_equation_maple(
		std::stringstream &ost, int *coeffs)
{
	Poly4_3->print_equation_str(ost, coeffs);
}

void quartic_curve_domain::print_equation_with_line_breaks_tex(
		std::ostream &ost, int *coeffs)
{
	Poly4_3->print_equation_with_line_breaks_tex(
			ost, coeffs, 8 /* nb_terms_per_line*/,
			"\\\\\n" /* const char *new_line_text*/);
}

void quartic_curve_domain::print_gradient_with_line_breaks_tex(
		std::ostream &ost, int *coeffs)
{
	Poly3_3->print_equation_with_line_breaks_tex(
			ost, coeffs, 8 /* nb_terms_per_line*/,
			"\\\\\n" /* const char *new_line_text*/);
}

void quartic_curve_domain::unrank_point(int *v, long int rk)
{
	P->unrank_point(v, rk);
}

long int quartic_curve_domain::rank_point(int *v)
{
	long int rk;

	rk = P->rank_point(v);
	return rk;
}


void quartic_curve_domain::unrank_line_in_dual_coordinates(int *v, long int rk)
{
	int basis[9];
	//int r;

	P->unrank_line(basis, rk);
	F->Linear_algebra->RREF_and_kernel(3, 2, basis,
			0 /* verbose_level */);
	Int_vec_copy(basis + 6, v, 3);
}

void quartic_curve_domain::print_lines_tex(
		std::ostream &ost, long int *Lines, int nb_lines)
{
	int i;
	l1_interfaces::latex_interface L;

	ost << "The lines are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;

	for (i = 0; i < nb_lines; i++) {
		//fp << "Line " << i << " is " << v[i] << ":\\\\" << endl;
		P->Subspaces->Grass_lines->unrank_lint(Lines[i], 0 /*verbose_level*/);
		ost << "$$" << endl;

		ost << Schlaefli->Line_label_tex[i];
		//ost << "\\ell_{" << i << "}";

#if 0
		if (nb_lines == 27) {
			ost << " = " << Schlaefli->Line_label_tex[i];
		}
#endif
		ost << " = " << endl;
		//print_integer_matrix_width(cout,
		// P->Grass_lines->M, k, n, n, F->log10_of_q + 1);
		P->Subspaces->Grass_lines->latex_matrix(ost, P->Subspaces->Grass_lines->M);
		//print_integer_matrix_tex(ost, P->Grass_lines->M, 2, 4);
		//ost << "\\right]_{" << Lines[i] << "}" << endl;
		ost << "_{" << Lines[i] << "}" << endl;

		if (F->e > 1) {
			ost << "=" << endl;
			ost << "\\left[" << endl;
			L.print_integer_matrix_tex(ost, P->Subspaces->Grass_lines->M, 2, 3);
			ost << "\\right]_{" << Lines[i] << "}" << endl;
		}

		ost << "$$" << endl;
	}
	ost << "\\end{multicols}" << endl;
	ost << "Rank of lines: ";
	Lint_vec_print(ost, Lines, nb_lines);
	ost << "\\\\" << endl;

}



void quartic_curve_domain::compute_points_on_lines(
		long int *Pts, int nb_points,
		long int *Lines, int nb_lines,
		data_structures::set_of_sets *&pts_on_lines,
		int *&f_is_on_line,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, l, r;
	int *pt_coords;
	int Basis[6];
	int Mtx[9];

	if (f_v) {
		cout << "quartic_curve_domain::compute_points_on_lines" << endl;
	}
	f_is_on_line = NEW_int(nb_points);
	Int_vec_zero(f_is_on_line, nb_points);

	pts_on_lines = NEW_OBJECT(data_structures::set_of_sets);
	pts_on_lines->init_basic_constant_size(nb_points,
		nb_lines, F->q + 1, 0 /* verbose_level */);
	pt_coords = NEW_int(nb_points * 3);
	for (i = 0; i < nb_points; i++) {
		P->unrank_point(pt_coords + i * 3, Pts[i]);
	}

	orbiter_kernel_system::Orbiter->Lint_vec->zero(pts_on_lines->Set_size, nb_lines);
	for (i = 0; i < nb_lines; i++) {
		l = Lines[i];
		P->unrank_line(Basis, l);
		for (j = 0; j < nb_points; j++) {
			Int_vec_copy(Basis, Mtx, 6);
			Int_vec_copy(pt_coords + j * 3, Mtx + 6, 3);
			r = F->Linear_algebra->Gauss_easy(Mtx, 3, 3);
			if (r == 2) {
				pts_on_lines->add_element(i, j);
				f_is_on_line[j] = true;
			}
		}
	}

	FREE_int(pt_coords);

	if (f_v) {
		cout << "quartic_curve_domain::compute_points_on_lines done" << endl;
	}
}



void quartic_curve_domain::multiply_conic_times_conic(int *six_coeff_a,
	int *six_coeff_b, int *fifteen_coeff,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx, u;
	int M[3];

	if (f_v) {
		cout << "quartic_curve_domain::multiply_conic_times_conic" << endl;
	}


	Int_vec_zero(fifteen_coeff, 15);
	for (i = 0; i < 6; i++) {
		a = six_coeff_a[i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < 6; j++) {
			b = six_coeff_b[j];
			if (b == 0) {
				continue;
			}
			c = F->mult(a, b);

			for (u = 0; u < 3; u++) {
				M[u] = Poly2_3->get_monomial(i, u) + Poly2_3->get_monomial(j, u);
			}
			idx = Poly4_3->index_of_monomial(M);
			if (idx >= 15) {
				cout << "quartic_curve_domain::multiply_conic_times_conic "
						"idx >= 15" << endl;
				exit(1);
			}
			fifteen_coeff[idx] = F->add(fifteen_coeff[idx], c);
		}
	}


	if (f_v) {
		cout << "quartic_curve_domain::multiply_conic_times_conic done" << endl;
	}
}

void quartic_curve_domain::multiply_conic_times_line(int *six_coeff,
	int *three_coeff, int *ten_coeff,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx, u;
	int M[3];

	if (f_v) {
		cout << "quartic_curve_domain::multiply_conic_times_line" << endl;
	}


	Int_vec_zero(ten_coeff, 10);
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
				M[u] = Poly2_3->get_monomial(i, u) + Poly1_3->get_monomial(j, u);
			}
			idx = Poly3_3->index_of_monomial(M);
			if (idx >= 10) {
				cout << "quartic_curve_domain::multiply_conic_times_line "
						"idx >= 10" << endl;
				exit(1);
			}
			ten_coeff[idx] = F->add(ten_coeff[idx], c);
		}
	}


	if (f_v) {
		cout << "quartic_curve_domain::multiply_conic_times_line done" << endl;
	}
}

void quartic_curve_domain::multiply_line_times_line(int *line1,
	int *line2, int *six_coeff,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx, u;
	int M[3];

	if (f_v) {
		cout << "quartic_curve_domain::multiply_line_times_line" << endl;
	}


	Int_vec_zero(six_coeff, 6);
	for (i = 0; i < 3; i++) {
		a = line1[i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < 3; j++) {
			b = line2[j];
			if (b == 0) {
				continue;
			}
			c = F->mult(a, b);

			for (u = 0; u < 3; u++) {
				M[u] = Poly1_3->get_monomial(i, u) + Poly1_3->get_monomial(j, u);
			}
			idx = Poly2_3->index_of_monomial(M);
			if (idx >= 15) {
				cout << "quartic_curve_domain::multiply_line_times_line "
						"idx >= 6" << endl;
				exit(1);
			}
			six_coeff[idx] = F->add(six_coeff[idx], c);
		}
	}


	if (f_v) {
		cout << "quartic_curve_domain::multiply_line_times_line done" << endl;
	}
}

void quartic_curve_domain::multiply_three_lines(int *line1, int *line2, int *line3,
	int *ten_coeff,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int six[6];

	if (f_v) {
		cout << "quartic_curve_domain::multiply_three_lines" << endl;
	}


	multiply_line_times_line(line1, line2, six, verbose_level);
	multiply_conic_times_line(six, line3, ten_coeff, verbose_level);


	if (f_v) {
		cout << "quartic_curve_domain::multiply_three_lines done" << endl;
	}
}


void quartic_curve_domain::multiply_four_lines(int *line1, int *line2, int *line3, int *line4,
	int *fifteen_coeff,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int six1[6];
	int six2[6];

	if (f_v) {
		cout << "quartic_curve_domain::multiply_four_lines" << endl;
	}


	multiply_line_times_line(line1, line2, six1, verbose_level);
	multiply_line_times_line(line3, line4, six2, verbose_level);
	multiply_conic_times_conic(six1, six2, fifteen_coeff, verbose_level);


	if (f_v) {
		cout << "quartic_curve_domain::multiply_four_lines done" << endl;
	}
}



void quartic_curve_domain::assemble_cubic_surface(int *f1, int *f2, int *f3, int *eqn20,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "quartic_curve_domain::assemble_cubic_surface" << endl;
	}

	Int_vec_zero(eqn20, 20);

	int i, a, idx;
	int mon[4];


	for (i = 0; i < Poly1_3->get_nb_monomials(); i++) {
		a = f1[i];
		if (a == 0) {
			continue;
		}
		if (f_v) {
			cout << "f1[" << i << "] = " << a << endl;
		}
		mon[0] = 2;
		Int_vec_copy(Poly1_3->get_monomial_pointer(i), mon + 1, 3);

		idx = Poly3_4->index_of_monomial(mon);
		if (idx >= 20) {
			cout << "quartic_curve_domain::assemble_cubic_surface "
					"idx >= 20" << endl;
			exit(1);
		}
		eqn20[idx] = F->add(eqn20[idx], a);
	}

	for (i = 0; i < Poly2_3->get_nb_monomials(); i++) {
		a = f2[i];
		if (a == 0) {
			continue;
		}
		if (f_v) {
			cout << "f2[" << i << "] = " << a << endl;
		}
		mon[0] = 1;
		Int_vec_copy(Poly2_3->get_monomial_pointer(i), mon + 1, 3);

		idx = Poly3_4->index_of_monomial(mon);
		if (idx >= 20) {
			cout << "quartic_curve_domain::assemble_cubic_surface "
					"idx >= 20" << endl;
			exit(1);
		}
		eqn20[idx] = F->add(eqn20[idx], a);
	}

	for (i = 0; i < Poly3_3->get_nb_monomials(); i++) {
		a = f3[i];
		if (a == 0) {
			continue;
		}
		if (f_v) {
			cout << "f3[" << i << "] = " << a << endl;
		}
		mon[0] = 0;
		Int_vec_copy(Poly3_3->get_monomial_pointer(i), mon + 1, 3);

		idx = Poly3_4->index_of_monomial(mon);
		if (idx >= 20) {
			cout << "quartic_curve_domain::assemble_cubic_surface "
					"idx >= 20" << endl;
			exit(1);
		}
		eqn20[idx] = F->add(eqn20[idx], a);
	}




	if (f_v) {
		cout << "quartic_curve_domain::assemble_cubic_surface done" << endl;
	}
}

void quartic_curve_domain::create_surface(quartic_curve_object *Q,
		int *eqn20, int verbose_level)
// Given a quartic Q in X1,X2,X3, compute an associated cubic surface
// whose projection from (1,0,0,0) gives back the quartic Q.
// Pick 4 bitangents L0,L1,L2,L3 so that the 8 points of tangency lie on a conic C.
// Then, create the cubic surface with equation
// (- lambda * mu) / 4 * X0^2 * L0 (the equation of the first of the four bitangents)
// + X0 * lambda * C (the conic equation)
// + L1 * L2 * L3 (the product of the equations of the last three bitangents)
// Here 1, lambda, mu are the coefficients of a linear dependency between
// Q (the quartic), C^2, L0*L1*L2*L3, so
// Q + lambda * C^2 + mu * L0*L1*L2*L3 = 0.
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "quartic_curve_domain::create_surface" << endl;
	}

	if (Q->QP == NULL) {
		cout << "quartic_curve_domain::create_surface, QP == NULL" << endl;
		exit(1);
	}

	int *Bitangents;
	int nb_bitangents;
	int set[4];
	int Idx[4];
	int pt_idx[8];
	long int Points[8];
	long int Bitangents4[4];
	int Bitangents_coeffs[16];
	int six_coeffs_conic[6];
	int i, r;
	long int nCk, h;
	combinatorics::combinatorics_domain Combi;
	int conic_squared_15[15];
	int four_lines_15[15];
	int M1[3 * 15];
	int M2[15 * 3];

	Q->QP->Bitangent_line_type->get_class_by_value(Bitangents, nb_bitangents, 2 /*value */,
			verbose_level);

	if (nb_bitangents < 4) {
		cout << "quartic_curve_domain::create_surface, nb_bitangents < 4" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "quartic_curve_domain::create_surface "
				"we found " << nb_bitangents << " bitangents" << endl;
		Int_vec_print(cout, Bitangents, nb_bitangents);
		cout << endl;
	}

	nCk = Combi.binomial_lint(nb_bitangents, 4);
	for (h = 0; h < nCk; h++) {
		Combi.unrank_k_subset(h, set, nb_bitangents, 4);
		if (f_v) {
			cout << "quartic_curve_domain::create_surface "
					"trying subset " << h << " / " << nCk << " which is ";
			Int_vec_print(cout, set, 4);
			cout << endl;
		}
		for (i = 0; i < 4; i++) {
			Idx[i] = Bitangents[set[i]];
			Bitangents4[i] = Q->bitangents28[Idx[i]];
		}

		for (i = 0; i < 4; i++) {

			if (Q->QP->pts_on_lines->Set_size[Idx[i]] != 2) {
				cout << "quartic_curve_domain::create_surface QP->pts_on_lines->Set_size[Idx[i]] != 2" << endl;
				exit(1);
			}
			pt_idx[i * 2 + 0] = Q->QP->pts_on_lines->Sets[Idx[i]][0];
			pt_idx[i * 2 + 1] = Q->QP->pts_on_lines->Sets[Idx[i]][1];

		}
		for (i = 0; i < 8; i++) {
			Points[i] = Q->Pts[pt_idx[i]];
		}
		if (f_v) {
			cout << "quartic_curve_domain::create_surface "
					"trying subset " << h << " / " << nCk << " Points = ";
			Lint_vec_print(cout, Points, 8);
			cout << endl;
		}

		if (P->Plane->determine_conic_in_plane(
				Points, 8,
				six_coeffs_conic,
				verbose_level)) {
			cout << "quartic_curve_domain::create_surface "
					"The four bitangents are syzygetic" << endl;
			break;
		}
	}
	if (h == nCk) {
		cout << "quartic_curve_domain::create_surface, "
				"could not find a syzygetic set of bitangents" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "quartic_curve_domain::create_surface "
				"trying subset " << h << " / " << nCk << " Bitangents4 = ";
		Lint_vec_print(cout, Bitangents4, 4);
		cout << endl;
	}

	multiply_conic_times_conic(six_coeffs_conic,
			six_coeffs_conic, conic_squared_15,
			verbose_level);

	if (f_v) {
		cout << "quartic_curve_domain::create_surface "
				"conic squared = ";
		Int_vec_print(cout, conic_squared_15, 15);
		cout << endl;
	}

	for (i = 0; i < 4; i++) {
		unrank_line_in_dual_coordinates(Bitangents_coeffs + i * 3, Bitangents4[i]);
	}

	if (f_v) {
		cout << "quartic_curve_domain::create_surface "
				"chosen bitangents in dual coordinates = ";
		Int_matrix_print(Bitangents_coeffs, 4, 3);
	}


	multiply_four_lines(Bitangents_coeffs + 0 * 3,
			Bitangents_coeffs + 1 * 3,
			Bitangents_coeffs + 2 * 3,
			Bitangents_coeffs + 3 * 3,
			four_lines_15,
			verbose_level);

	if (f_v) {
		cout << "quartic_curve_domain::create_surface "
				"product of 4 bitangents = ";
		Int_vec_print(cout, four_lines_15, 15);
		cout << endl;
	}

	Int_vec_copy(Q->eqn15, M1, 15);
	Int_vec_copy(conic_squared_15, M1 + 15, 15);
	Int_vec_copy(four_lines_15, M1 + 30, 15);

	orbiter_kernel_system::Orbiter->Int_vec->transpose(M1, 3, 15, M2);

	r = F->Linear_algebra->RREF_and_kernel(3, 15, M2, 0 /* verbose_level*/);

	if (r != 2) {
		cout << "quartic_curve_domain::create_surface r != 2" << endl;
		exit(1);
	}

	F->Projective_space_basic->PG_element_normalize_from_front(
			M2 + 6, 1, 3);
	if (f_v) {
		cout << "quartic_curve_domain::create_surface kernel = ";
		Int_vec_print(cout, M2 + 6, 3);
		cout << endl;
	}
	int lambda, mu;

	lambda = M2[7];
	mu = M2[8];
	if (f_v) {
		cout << "quartic_curve_domain::create_surface "
				"lambda = " << lambda << " mu = " << mu << endl;
	}

	int f1_three_coeff[3]; // - lambda * mu / 4 * the equation of the first of the four bitangents
	int f2_six_coeff[6]; // lambda * conic equation
	int f3_ten_coeff[10]; // the product of the last three bitangents

	multiply_three_lines(
			Bitangents_coeffs + 1 * 3,
			Bitangents_coeffs + 2 * 3,
			Bitangents_coeffs + 3 * 3,
			f3_ten_coeff,
			verbose_level);

#if 0
	int sqrt_lambda;

	if (f_v) {
		cout << "quartic_curve_domain::create_surface computing square root of lambda" << endl;
	}
	F->square_root(lambda, sqrt_lambda);
	if (f_v) {
		cout << "quartic_curve_domain::create_surface sqrt_lambda = " << sqrt_lambda << endl;
	}
#endif

	Poly2_3->multiply_by_scalar(
			six_coeffs_conic, lambda, f2_six_coeff,
			verbose_level);

	int half, fourth, a;

	half = F->inverse(2);
	fourth = F->mult(half, half);
	a = F->mult(F->negate(F->mult(lambda, mu)), fourth);

	Poly1_3->multiply_by_scalar(
			Bitangents_coeffs + 0 * 3, a, f1_three_coeff,
			verbose_level);


	// and now, create the cubic with equation
	// (- lambda * mu) / 4 * X0^2 * L0 (the equation of the first of the four bitangents)
	// + X0 * lambda * conic equation
	// + L1 * L2 * L3 (the product of the equations of the last three bitangents)

	assemble_cubic_surface(
			f1_three_coeff,
			f2_six_coeff,
			f3_ten_coeff,
			eqn20,
		verbose_level);

	if (f_v) {
		cout << "quartic_curve_domain::create_surface "
				"eqn20 = ";
		Int_vec_print(cout, eqn20, 20);
		cout << endl;
	}

	FREE_int(Bitangents);

	if (f_v) {
		cout << "quartic_curve_domain::create_surface done" << endl;
	}
}

void quartic_curve_domain::compute_gradient(int *equation15, int *&gradient, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "quartic_curve_domain::compute_gradient" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_domain::compute_gradient "
				"Poly3_3->get_nb_monomials() = " << Poly3_3->get_nb_monomials() << endl;
	}

	gradient = NEW_int(3 * Poly3_3->get_nb_monomials());

	for (i = 0; i < 3; i++) {
		if (f_v) {
			cout << "quartic_curve_domain::compute_gradient i=" << i << endl;
		}
		if (f_v) {
			cout << "quartic_curve_domain::compute_gradient eqn_in=";
			Int_vec_print(cout, equation15, 15);
			cout << " = " << endl;
			Poly4_3->print_equation(cout, equation15);
			cout << endl;
		}
		Partials[i].apply(equation15,
				gradient + i * Poly3_3->get_nb_monomials(),
				verbose_level - 2);
		if (f_v) {
			cout << "quartic_curve_domain::compute_gradient "
					"partial=";
			Int_vec_print(cout, gradient + i * Poly3_3->get_nb_monomials(),
					Poly3_3->get_nb_monomials());
			cout << " = ";
			Poly3_3->print_equation(cout,
					gradient + i * Poly3_3->get_nb_monomials());
			cout << endl;
		}
	}


	if (f_v) {
		cout << "quartic_curve_domain::compute_gradient done" << endl;
	}
}



}}}


