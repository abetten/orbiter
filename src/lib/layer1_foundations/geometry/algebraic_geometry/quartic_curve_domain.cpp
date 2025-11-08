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
namespace geometry {
namespace algebraic_geometry {


quartic_curve_domain::quartic_curve_domain()
{
	Record_birth();
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
	Record_death();
	if (Schlaefli) {
		FREE_OBJECT(Schlaefli);
	}
}


void quartic_curve_domain::init(
		algebra::field_theory::finite_field *F,
		int verbose_level)
// creates a projective_space object
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_domain::init" << endl;
	}

	quartic_curve_domain::F = F;

	P = NEW_OBJECT(geometry::projective_geometry::projective_space);
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
	init_polynomial_domains(verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_domain::init "
				"after init_polynomial_domains" << endl;
	}


	Schlaefli = NEW_OBJECT(algebraic_geometry::schlaefli_labels);
	if (f_v) {
		cout << "quartic_curve_domain::init "
				"before Schlaefli->init" << endl;
	}
	Schlaefli->init(0 /*verbose_level*/);
	if (f_v) {
		cout << "quartic_curve_domain::init "
				"after Schlaefli->init" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_domain::init done" << endl;
	}
}

void quartic_curve_domain::init_polynomial_domains(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_domain::init_polynomial_domains" << endl;
	}


	Poly1_3 = NEW_OBJECT(algebra::ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"before Poly1_3->init" << endl;
	}
	Poly1_3->init(F,
			3 /* nb_vars */, 1 /* degree */,
			t_PART,
			verbose_level - 2);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"after Poly1_3->init" << endl;
	}



	Poly2_3 = NEW_OBJECT(algebra::ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"before Poly2_3->init" << endl;
	}
	Poly2_3->init(F,
			3 /* nb_vars */, 2 /* degree */,
			t_PART,
			verbose_level - 2);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"after Poly2_3->init" << endl;
	}



	Poly3_3 = NEW_OBJECT(algebra::ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"before Poly3_3->init" << endl;
	}
	Poly3_3->init(F,
			3 /* nb_vars */, 3 /* degree */,
			t_PART,
			verbose_level - 2);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"after Poly3_3->init" << endl;
	}


	Poly4_3 = NEW_OBJECT(algebra::ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"before Poly4_3->init" << endl;
	}
	Poly4_3->init(F,
			3 /* nb_vars */, 4 /* degree */,
			t_PART,
			verbose_level - 2);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"after Poly4_3->init" << endl;
	}

	Poly3_4 = NEW_OBJECT(algebra::ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"before Poly3_4->init" << endl;
	}
	Poly3_4->init(F,
			4 /* nb_vars */, 3 /* degree */,
			t_PART,
			verbose_level - 2);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"after Poly3_4->init" << endl;
	}

	Partials = NEW_OBJECTS(algebra::ring_theory::partial_derivative, 3);

	int i;

	if (f_v) {
		cout << "surface_domain::init_polynomial_domains "
				"initializing partials" << endl;
	}
	for (i = 0; i < 3; i++) {
		Partials[i].init(Poly4_3, Poly3_3, i, verbose_level - 2);
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

std::string quartic_curve_domain::stringify_equation_maple(
		int *eqn15)
{
	stringstream sstr;
	string str;
	print_equation_maple(sstr, eqn15);
	str = sstr.str();
	return str;
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

void quartic_curve_domain::unrank_point(
		int *v, long int rk)
{
	P->unrank_point(v, rk);
}

long int quartic_curve_domain::rank_point(
		int *v)
{
	long int rk;

	rk = P->rank_point(v);
	return rk;
}


void quartic_curve_domain::unrank_line_in_dual_coordinates(
		int *v, long int rk)
{
	int basis[9];
	//int r;

	P->unrank_line(basis, rk);
	F->Linear_algebra->RREF_and_kernel(
			3, 2, basis,
			0 /* verbose_level */);
	Int_vec_copy(basis + 6, v, 3);
}

void quartic_curve_domain::print_lines_tex(
		std::ostream &ost, long int *Lines, int nb_lines)
{
	int i;
	other::l1_interfaces::latex_interface L;

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





void quartic_curve_domain::multiply_conic_times_conic(
		int *six_coeff_a,
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

void quartic_curve_domain::multiply_conic_times_line(
		int *six_coeff,
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

void quartic_curve_domain::multiply_line_times_line(
		int *line1,
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

void quartic_curve_domain::multiply_three_lines(
		int *line1, int *line2, int *line3,
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


void quartic_curve_domain::multiply_four_lines(
		int *line1, int *line2, int *line3, int *line4,
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



void quartic_curve_domain::assemble_cubic_surface(
		int *f1, int *f2, int *f3, int *eqn20,
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

void quartic_curve_domain::create_surface(
		quartic_curve_object *Q,
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
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	int conic_squared_15[15];
	int four_lines_15[15];
	int M1[3 * 15];
	int M2[15 * 3];

	Q->QP->Kovalevski->Bitangent_line_type->get_class_by_value(
			Bitangents, nb_bitangents, 2 /*value */,
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
			Bitangents4[i] = Q->get_line(Idx[i]);
		}

		for (i = 0; i < 4; i++) {

			if (Q->QP->Kovalevski->pts_on_lines->Set_size[Idx[i]] != 2) {
				cout << "quartic_curve_domain::create_surface "
						"QP->pts_on_lines->Set_size[Idx[i]] != 2" << endl;
				exit(1);
			}
			pt_idx[i * 2 + 0] = Q->QP->Kovalevski->pts_on_lines->Sets[Idx[i]][0];
			pt_idx[i * 2 + 1] = Q->QP->Kovalevski->pts_on_lines->Sets[Idx[i]][1];

		}
		for (i = 0; i < 8; i++) {
			Points[i] = Q->get_point(pt_idx[i]);
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
		unrank_line_in_dual_coordinates(
				Bitangents_coeffs + i * 3, Bitangents4[i]);
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

	Int_vec_copy(Q->Variety_object->eqn, M1, 15);
	Int_vec_copy(conic_squared_15, M1 + 15, 15);
	Int_vec_copy(four_lines_15, M1 + 30, 15);

	other::orbiter_kernel_system::Orbiter->Int_vec->transpose(M1, 3, 15, M2);

	r = F->Linear_algebra->RREF_and_kernel(
			3, 15, M2, 0 /* verbose_level*/);

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

void quartic_curve_domain::compute_gradient(
		int *equation15, int *&gradient, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "quartic_curve_domain::compute_gradient" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_domain::compute_gradient "
				"Poly3_3->get_nb_monomials() = "
				<< Poly3_3->get_nb_monomials() << endl;
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

int quartic_curve_domain::create_quartic_curve_by_symbolic_object(
		algebra::ring_theory::homogeneous_polynomial_domain *Poly,
		std::string &name_of_formula,
		quartic_curve_object *&QO,
		int verbose_level)
// returns false if the equation is zero
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_domain::create_quartic_curve_by_symbolic_object" << endl;
		cout << "quartic_curve_domain::create_quartic_curve_by_symbolic_object "
				"name_of_formula=" << name_of_formula << endl;
	}



	algebra::expression_parser::symbolic_object_builder *Symbol;

	Symbol = Get_symbol(name_of_formula);

	// assemble the equation as a vector of coefficients
	// in the ordering of the polynomial ring:

	int *coeffs;
	int nb_coeffs;

	if (f_v) {
		cout << "quartic_curve_domain::create_quartic_curve_by_symbolic_object "
				"before Symbol->Formula_vector->V[0].collect_coefficients_of_equation" << endl;
	}
	Symbol->Formula_vector->V[0].collect_coefficients_of_equation(
			Poly,
			coeffs, nb_coeffs,
			verbose_level);
	if (f_v) {
		cout << "quartic_curve_domain::create_quartic_curve_by_symbolic_object "
				"after Symbol->Formula_vector->V[0].collect_coefficients_of_equation" << endl;
	}

	if (nb_coeffs != 15) {
		cout << "quartic_curve_domain::create_quartic_curve_by_symbolic_object nb_coeffs != 15" << endl;
		exit(1);
	}
	// build a surface_object and compute properties of the surface:


	if (Int_vec_is_zero(coeffs, nb_coeffs)) {
		return false;
	}



	QO = NEW_OBJECT(algebraic_geometry::quartic_curve_object);


	if (f_v) {
		cout << "quartic_curve_domain::create_quartic_curve_by_symbolic_object "
				"before create_surface_by_coefficient_vector" << endl;
	}

	create_quartic_curve_by_coefficient_vector(
			coeffs,
			name_of_formula,
			name_of_formula,
			QO,
			verbose_level);


	if (f_v) {
		cout << "quartic_curve_domain::create_quartic_curve_by_symbolic_object "
				"after create_surface_by_coefficient_vector" << endl;
	}

	FREE_int(coeffs);

	if (f_v) {
		cout << "quartic_curve_domain::create_quartic_curve_by_symbolic_object done" << endl;
	}
	return true;
}


void quartic_curve_domain::create_quartic_curve_by_coefficient_vector(
		int *coeffs15,
		std::string &label_txt,
		std::string &label_tex,
		quartic_curve_object *&QO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_domain::create_quartic_curve_by_coefficient_vector" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_domain::create_quartic_curve_by_coefficient_vector "
				"surface is given by the coefficients" << endl;
	}



	QO = NEW_OBJECT(algebraic_geometry::quartic_curve_object);

	if (f_v) {
		cout << "quartic_curve_domain::create_quartic_curve_by_coefficient_vector "
				"before SO->init_equation" << endl;
	}
	QO->init_equation_but_no_bitangents(
			this,
			coeffs15,
			verbose_level);
	if (f_v) {
		cout << "quartic_curve_domain::create_quartic_curve_by_coefficient_vector "
				"after SO->init_equation" << endl;
	}




	if (f_v) {
		cout << "quartic_curve_domain::create_quartic_curve_by_coefficient_vector "
				"before compute_properties" << endl;
	}
	QO->compute_properties(verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_domain::create_quartic_curve_by_coefficient_vector "
				"after compute_properties" << endl;
	}



	if (f_v) {
		cout << "quartic_curve_domain::create_quartic_curve_by_coefficient_vector done" << endl;
	}

}

void quartic_curve_domain::create_quartic_curve_by_normal_form(
	int *abcdef,
	int *coeffs15,
	long int *Lines_rk,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_domain::create_quartic_curve_by_normal_form" << endl;
	}

	int i;
	int a, b, c, d, e, f;
	int ma, mb, mc, md, me, mf;
	int m1, two, four, mtwo, mfour;
	int s1, s2, s3, s4, s5, s6, s7, s8;
	int delta, epsilon, gamma, lambda, mu, nu, eta, zeta, xi, theta, psi, omega;

	a = abcdef[0];
	b = abcdef[1];
	c = abcdef[2];
	d = abcdef[3];
	e = abcdef[4];
	f = abcdef[5];
	ma = F->negate(a);
	mb = F->negate(b);
	mc = F->negate(c);
	md = F->negate(d);
	me = F->negate(e);
	mf = F->negate(f);

	m1 = F->negate(1);
	two = F->add(1, 1);
	four = F->add(two, two);
	mtwo = F->negate(two);
	mfour = F->negate(four);
	s1 = F->add(a, m1);
	s2 = F->add(b, m1);
	s3 = F->add(c, m1);
	s4 = F->add(d, m1);
	s5 = F->add(a, mb);
	s6 = F->add(a, mc);
	s7 = F->add(b, md);
	s8 = F->add(c, md);

	delta = F->add(F->mult(a, d), F->negate(F->mult(b, c)));
	epsilon = F->add5(
			F->mult3(a, b, c),
			F->mult3(ma, b, d),
			F->mult3(ma, c, d),
			F->mult3(b, c, d),
			delta);
	gamma = F->add5(delta, ma, b, c, md);
	lambda = F->add3(F->mult3(b, b, s8), F->mult3(md, d, s5), delta);
	mu = F->add3(F->mult3(ma, b, d), F->mult3(b, c, d), delta);
	nu = F->add(F->mult3(a, c, s7), F->mult3(mb, d, s6));
	eta = F->add6(
			F->mult4(ma, a, c, d),
			F->mult4(a, b, c, c),
			F->mult3(a, a, d),
			F->mult3(ma, b, d),
			F->mult3(mb, c, c),
			F->mult3(b, c, d)
			);
	zeta = F->mult3(s1, s3, s7);
	xi = F->add5(
			F->mult3(a, a, c),
			F->mult3(ma, a, d),
			F->mult3(ma, c, c),
			F->mult3(b, c, c),
			delta
			);
	theta = F->add6(
			F->mult3(a, b, c),
			F->mult3(ma, c, d),
			F->mult(ma, b),
			F->mult(c, d),
			a,
			mc
			);
	int psi17[17];
	psi17[0] = F->mult6(two, a, a, b, c, d);
	psi17[1] = F->mult5(ma, a, b, d, d);
	psi17[2] = F->mult6(mtwo, a, a, c, d, d);
	psi17[3] = F->mult6(mtwo, a, b, b, c, c);
	psi17[4] = F->mult5(a, b, b, c, d);
	psi17[5] = F->mult6(two, a, b, c, c, d);
	psi17[6] = F->mult5(a, b, c, d, d);
	psi17[7] = F->mult5(mb, b, c, c, d);
	psi17[8] = F->mult4(ma, a, b, c);
	psi17[9] = F->mult4(a, a, c, d);
	psi17[10] = F->mult4(a, a, d, d);
	psi17[11] = F->mult4(a, b, b, c);
	psi17[12] = F->mult4(a, b, c, c);
	psi17[13] = F->mult5(mfour, a, b, c, d);
	psi17[14] = F->mult4(ma, c, c, d);
	psi17[15] = F->mult4(a, c, d, d);
	psi17[16] = F->mult4(b, b, c, c);

	psi = 0;
	for (i = 0; i < 17; i++) {
		psi = F->add(psi, psi17[i]);
	}
	omega = F->add4(a, b, mc, md);

	int y0, y1, y2;
	int y0012;
	int y0022;
	int y0112;
	int y0122;
	int y0222;
	int y1122;
	int y1222;

	y0 = e;
	y1 = f;
	y2 = 1;
	y0012 = F->mult4(y0, y0, y1, y2);
	y0022 = F->mult4(y0, y0, y2, y2);
	y0112 = F->mult4(y0, y1, y1, y2);
	y0122 = F->mult4(y0, y1, y2, y2);
	y0222 = F->mult4(y0, y2, y2, y2);
	y1122 = F->mult4(y1, y1, y2, y2);
	y1222 = F->mult4(y1, y2, y2, y2);

	int P0[4], p0;

	P0[0] = F->mult5(b, d, s6, gamma, y0022);
	P0[1] = F->mult4(m1, s6,
			F->add6(F->mult3(a, d, d),
					F->mult3(mb, b, c),
					F->mult3(b, b, d),
					F->mult3(mb, d, d),
					F->mult(ma, d),
					F->mult(b, c)),
			y0012);
	P0[2] = F->mult4(m1, s6, epsilon, y0112);
	P0[3] = F->mult3(s6,
			F->add6(
					F->mult4(ma, b, d, d),
					F->mult4(b, b, c, d),
					F->mult3(a, b, c),
					F->mult3(ma, c, d),
					F->mult3(a, d, d),
					F->mult3(mb, b, c)
					),
			y0122);
	p0 = 0;
	for (i = 0; i < 4; i++) {
		p0 = F->add(p0, P0[i]);
	}

	int P1[4], p1;

	P1[0] = F->mult5(ma, c, s7, gamma, y1122);
	P1[1] = F->mult3(s7, epsilon, y0012);
	P1[2] = F->mult4(m1, s7, xi, y0112);
	P1[3] = F->mult4(m1, s7, eta, y0122);
	p1 = 0;
	for (i = 0; i < 4; i++) {
		p1 = F->add(p1, P1[i]);
	}

	int P2[4], p2;

	P2[0] = F->mult4(m1, s7, eta, y0222);
	P2[1] = F->mult6(m1, a, c, s7, gamma, y1222);
	P2[2] = F->mult3(s7, epsilon, y0022);
	P2[3] = F->mult4(m1, s7, xi, y0122);
	p2 = 0;
	for (i = 0; i < 4; i++) {
		p2 = F->add(p2, P2[i]);
	}

	int P3[5], p3;

	P3[0] = F->mult4(m1, epsilon, delta, y0222);
	P3[1] = F->mult3(epsilon, delta, y1222);
	P3[2] = F->mult4(m1, s7, epsilon, y0022);
	P3[3] = F->mult4(m1, s6, epsilon, y1122);
	P3[4] = F->mult3(epsilon, omega, y0122);
	p3 = 0;
	for (i = 0; i < 5; i++) {
		p3 = F->add(p3, P3[i]);
	}

	if (f_v) {
		int P[4];

		P[0] = p0;
		P[1] = p1;
		P[2] = p2;
		P[3] = p3;
		cout << "quartic_curve_domain::create_quartic_curve_by_normal_form "
				"p0,p1,p2,p3 = ";
		Int_vec_print(cout, P, 4);
		cout << endl;
	}

	int p00, p01, p02, p03;
	int p11, p12, p13;
	int p22, p23;
	int p33;

	p00 = F->mult(p0, p0);
	p01 = F->mult(p0, p1);
	p02 = F->mult(p0, p2);
	p03 = F->mult(p0, p3);

	p11 = F->mult(p1, p1);
	p12 = F->mult(p1, p2);
	p13 = F->mult(p1, p3);

	p22 = F->mult(p2, p2);
	p23 = F->mult(p2, p3);

	p33 = F->mult(p3, p3);

	int Lambda1[7], lambda1;

	Lambda1[0] = F->mult3(epsilon, omega, p02);
	Lambda1[1] = F->mult3(xi, s7, p03);
	Lambda1[2] = F->mult4(mtwo, s6, epsilon, p12);
	Lambda1[3] = F->mult4(mtwo, s6, epsilon, p13);
	Lambda1[4] = F->mult3(delta, epsilon, p22);
	Lambda1[5] = F->mult(psi, p23);
	Lambda1[6] = F->mult5(c, a, gamma, s7, p33);
	lambda1 = 0;
	for (i = 0; i < 7; i++) {
		lambda1 = F->add(lambda1, Lambda1[i]);
	}


	int Lambda2[7], lambda2;

	Lambda2[0] = F->mult4(m1, epsilon, s7, p00);
	Lambda2[1] = F->mult3(epsilon, omega, p01);
	Lambda2[2] = F->mult4(mtwo, delta, epsilon, p02);
	Lambda2[3] = F->mult3(eta, s7, p03);
	Lambda2[4] = F->mult4(m1, s6, epsilon, p11);
	Lambda2[5] = F->mult4(two, delta, epsilon, p12);
	Lambda2[6] = F->mult(psi, p13);
	lambda2 = 0;
	for (i = 0; i < 7; i++) {
		lambda2 = F->add(lambda2, Lambda2[i]);
	}


	int Lambda3[5], lambda3;

	Lambda3[0] = F->mult3(xi, s7, p01);
	Lambda3[1] = F->mult3(eta, s7, p02);
	Lambda3[2] = F->mult4(m1, s6, epsilon, p11);
	Lambda3[3] = F->mult(psi, p12);
	Lambda3[4] = F->mult6(two, c, a, gamma, s7, p13);
	lambda3 = 0;
	for (i = 0; i < 5; i++) {
		lambda3 = F->add(lambda3, Lambda3[i]);
	}

	int mu112, mu113, mu122, mu123, mu133;

	mu112 = F->mult3(m1, s6, epsilon);
	mu113 = F->mult3(m1, s6, epsilon);
	mu122 = F->mult(delta, epsilon);
	mu123 = psi;
	mu133 = F->mult4(c, a, gamma, s7);

	int nu11, nu22, nu33, nu12, nu13, nu23;

	nu11 = F->mult4(m1, s6, epsilon, F->add(p2, p3));
	nu22 = F->mult3(delta, epsilon, F->add(p1, F->negate(p0)));
	nu33 = F->mult5(c, a, gamma, s7, p1);
	nu12 = F->add4(
			F->mult3(epsilon, omega, p0),
			F->mult4(mtwo, s6, epsilon, p1),
			F->mult4(two, delta, epsilon, p2),
			F->mult(psi, p3)
			);
	nu13 = F->add4(
			F->mult3(xi, s7, p0),
			F->mult4(mtwo, s6, epsilon, p1),
			F->mult(psi, p2),
			F->mult6(two, c, a, gamma, s7, p3)
			);
	nu23 = F->add(
			F->mult3(eta, s7, p0),
			F->mult(psi, p1)
			);

	int c1111, c1112, c1113, c1122, c1123, c1133, c1222, c1223, c1233, c1333, c2222, c2223, c2233, c2333, c3333;

	c1111 = c1112 = c1113 = c1122 = c1123 = c1133 = c1222 = c1223 = c1233 = c1333 = c2222 = c2223 = c2233 = c2333 = c3333 = 0;

	c1112 = F->add(c1112, F->mult3(mfour, lambda1, mu112));

	c1113 = F->add(c1113, F->mult3(mfour, lambda1, mu113));

	c1122 = F->add(c1122, F->mult3(mfour, lambda2, mu112));
	c1122 = F->add(c1122, F->mult3(mfour, lambda1, mu122));

	c1123 = F->add(c1123, F->mult3(mfour, lambda3, mu112));
	c1123 = F->add(c1123, F->mult3(mfour, lambda2, mu113));
	c1123 = F->add(c1123, F->mult3(mfour, lambda1, mu123));

	c1133 = F->add(c1133, F->mult3(mfour, lambda3, mu113));
	c1133 = F->add(c1133, F->mult3(mfour, lambda1, mu133));

	c1222 = F->add(c1222, F->mult3(mfour, lambda2, mu122));

	c1223 = F->add(c1223, F->mult3(mfour, lambda3, mu122));
	c1223 = F->add(c1223, F->mult3(mfour, lambda2, mu123));

	c1233 = F->add(c1233, F->mult3(mfour, lambda3, mu123));
	c1233 = F->add(c1233, F->mult3(mfour, lambda2, mu133));

	c1333 = F->add(c1333, F->mult3(mfour, lambda3, mu133));


	c1111 = F->add(c1111, F->mult(nu11, nu11));

	c1112 = F->add(c1112, F->mult3(two, nu11, nu12));

	c1113 = F->add(c1113, F->mult3(two, nu11, nu13));

	c1122 = F->add(c1122, F->mult3(two, nu11, nu22));
	c1122 = F->add(c1122, F->mult(nu12, nu12));

	c1123 = F->add(c1123, F->mult3(two, nu11, nu23));
	c1123 = F->add(c1123, F->mult3(two, nu12, nu13));

	c1133 = F->add(c1133, F->mult3(two, nu11, nu33));
	c1133 = F->add(c1133, F->mult(nu13, nu13));

	c1222 = F->add(c1222, F->mult3(two, nu12, nu22));

	c1223 = F->add(c1223, F->mult3(two, nu12, nu23));
	c1223 = F->add(c1223, F->mult3(two, nu13, nu22));

	c1233 = F->add(c1233, F->mult3(two, nu12, nu33));
	c1233 = F->add(c1233, F->mult3(two, nu13, nu23));

	c1333 = F->add(c1333, F->mult3(two, nu13, nu33));

	c2222 = F->add(c2222, F->mult(nu22, nu22));

	c2223 = F->add(c2223, F->mult3(two, nu22, nu23));

	c2233 = F->add(c2233, F->mult3(two, nu22, nu33));
	c2233 = F->add(c2233, F->mult(nu23, nu23));

	c2333 = F->add(c2333, F->mult3(two, nu23, nu33));

	c3333 = F->add(c3333, F->mult(nu33, nu33));


	//int coeffs15[15];

	coeffs15[0] = c1111;
	coeffs15[3] = c1112;
	coeffs15[4] = c1113;
	coeffs15[9] = c1122;
	coeffs15[12] = c1123;
	coeffs15[10] = c1133;
	coeffs15[5] = c1222;
	coeffs15[13] = c1223;
	coeffs15[14] = c1233;
	coeffs15[7] = c1333;
	coeffs15[1] = c2222;
	coeffs15[6] = c2223;
	coeffs15[11] = c2233;
	coeffs15[8] = c2333;
	coeffs15[2] = c3333;

	if (f_v) {
		cout << "quartic_curve_domain::create_quartic_curve_by_normal_form "
				"abcdef = ";
		Int_vec_print(cout, abcdef, 6);
		cout << endl;
	}


	if (f_v) {
		int P[4];

		P[0] = p0;
		P[1] = p1;
		P[2] = p2;
		P[3] = p3;
		cout << "quartic_curve_domain::create_quartic_curve_by_normal_form "
				"p0,p1,p2,p3 = ";
		Int_vec_print(cout, P, 4);
		cout << endl;
	}


	if (f_v) {
		cout << "quartic_curve_domain::create_quartic_curve_by_normal_form "
				"coeffs15 = ";
		Int_vec_print(cout, coeffs15, 15);
		cout << endl;
	}


	int a1[6];
	int a2[6];
	int a3[6];
	int a4[6];
	int a5[6];
	int a6[6];

	int b1[6];
	int b2[6];
	int b3[6];
	int b4[6];
	int b5[6];
	int b6[6];

	int c12[6];
	int c13[6];
	int c14[6];
	int c15[6];
	int c16[6];
	int c23[6];
	int c24[6];
	int c25[6];
	int c26[6];
	int c34[6];
	int c35[6];
	int c36[6];
	int c45[6];
	int c46[6];
	int c56[6];

	int t[6];

	int *Lines[28];
	//long int Lines_rk[28];

	Lines[0] = a1;
	Lines[1] = a2;
	Lines[2] = a3;
	Lines[3] = a4;
	Lines[4] = a5;
	Lines[5] = a6;
	Lines[6] = b1;
	Lines[7] = b2;
	Lines[8] = b3;
	Lines[9] = b4;
	Lines[10] = b5;
	Lines[11] = b6;
	Lines[12] = c12;
	Lines[13] = c13;
	Lines[14] = c14;
	Lines[15] = c15;
	Lines[16] = c16;
	Lines[17] = c23;
	Lines[18] = c24;
	Lines[19] = c25;
	Lines[20] = c26;
	Lines[21] = c34;
	Lines[22] = c35;
	Lines[23] = c36;
	Lines[24] = c45;
	Lines[25] = c46;
	Lines[26] = c56;
	Lines[27] = t;

	a1[0] = F->add(F->mult3(s7, epsilon, p0), F->negate(F->mult3(s6, lambda, p1)));
	a1[1] = F->negate(F->mult3(s6, lambda, p2));
	a1[2] = F->negate(F->mult3(s6, lambda, p3));
	a1[3] = F->mult5(s6, b, d, gamma, p1);
	a1[4] = F->add(F->mult5(s6, b, d, gamma, p2), F->negate(F->mult3(s7, epsilon, p0)));
	a1[5] = F->add(F->mult5(s6, b, d, gamma, p3), F->mult3(s7, epsilon, p0));

	if (xi == 0) {
		a2[0] = p1;
		a2[1] = p2;
		a2[2] = p3;
		a2[3] = F->mult5(s7, a, c, gamma, p0);
		a2[4] = 0;
		a2[5] = F->mult3(s6, epsilon, p0);
	}
	else {
		a2[0] = F->add(F->mult3(s7, xi, p0), F->negate(F->mult3(s6, epsilon, p1)));
		a2[1] = F->negate(F->mult3(s6, epsilon, p2));
		a2[2] = F->negate(F->mult3(s6, epsilon, p3));
		a2[3] = F->mult4(a, c, gamma, p1);
		a2[4] = F->mult4(a, c, gamma, p2);
		a2[5] = F->add(F->mult4(a, c, gamma, p3), F->mult(xi, p0));
	}

	a3[0] = 0;
	a3[1] = p0;
	a3[2] = 0;
	a3[3] = 0;
	a3[4] = 0;
	a3[5] = p0;

	a4[0] = F->add(p0, F->negate(p1));
	a4[1] = F->add(p0, F->negate(p2));
	a4[2] = F->negate(p3);
	a4[3] = F->negate(p1);
	a4[4] = F->negate(p2);
	a4[5] = F->add(p0, F->negate(p3));

	a5[0] = F->add(F->mult(b, p0), F->negate(F->mult(a, p1)));
	a5[1] = F->add(F->mult(1, p0), F->negate(F->mult(a, p2)));
	a5[2] = F->negate(F->mult(a, p3));
	a5[3] = F->negate(F->mult(a, p1));
	a5[4] = F->negate(F->mult(a, p2));
	a5[5] = F->add(F->mult(1, p0), F->negate(F->mult(a, p3)));

	a6[0] = F->add(F->mult(d, p0), F->negate(F->mult(c, p1)));
	a6[1] = F->add(F->mult(1, p0), F->negate(F->mult(c, p2)));
	a6[2] = F->negate(F->mult(c, p3));
	a6[3] = F->negate(F->mult(c, p1));
	a6[4] = F->negate(F->mult(c, p2));
	a6[5] = F->add(F->mult(1, p0), F->negate(F->mult(c, p3)));

	b1[0] = p1;
	b1[1] = p2;
	b1[2] = p3;
	b1[3] = 0;
	b1[4] = 0;
	b1[5] = p0;

	b2[0] = p0;
	b2[1] = 0;
	b2[2] = 0;
	b2[3] = 0;
	b2[4] = F->negate(p0);
	b2[5] = p0;

	b3[0] = F->add(F->mult(theta, p1), F->negate(F->mult(zeta, p0)));
	b3[1] = F->mult(theta, p2);
	b3[2] = F->add(F->mult(theta, p3), F->mult(gamma, p0));
	b3[3] = F->add(F->mult(epsilon, p1), F->negate(F->mult(epsilon, p0)));
	b3[4] = F->add(F->mult(epsilon, p2), F->mult(gamma, p0));
	b3[5] = F->mult(epsilon, p3);

	b4[0] = F->add(F->mult(nu, p1), F->negate(F->mult(nu, p0)));
	b4[1] = F->add(F->mult(nu, p2), F->mult(delta, p0));
	b4[2] = F->mult(nu, p3);
	b4[3] = F->mult4(a, c, s7, F->add(p1, F->negate(p0)));
	b4[4] = F->mult4(a, c, s7, p2);
	b4[5] = F->add(F->mult4(a, c, s7, p3), F->mult(delta, p0));

	b5[0] = F->add(F->mult4(s3, s7, delta, p0), F->negate(F->mult4(s4, s6, delta, p1)));
	b5[1] = F->add(F->mult4(s6, s7, s8, p0), F->negate(F->mult4(s4, s6, delta, p2)));
	b5[2] = F->negate(F->mult4(s4, s6, delta, p3));
	b5[3] = F->add(F->mult4(c, s4, s6, p1), F->negate(F->mult4(c, s3, s7, p0)));
	b5[4] = F->mult4(c, s4, s6, p2);
	b5[5] = F->add(F->mult4(c, s4, s6, p3), F->mult3(s6, s8, p0));

	b6[0] = F->add(F->mult4(s1, s7, delta, p0), F->negate(F->mult4(s2, s6, delta, p1)));
	b6[1] = F->add(F->mult4(s5, s6, s7, p0), F->negate(F->mult4(s2, s6, delta, p2)));
	b6[2] = F->negate(F->mult4(s2, s6, delta, p3));
	b6[3] = F->add(F->mult4(a, s2, s6, p1), F->negate(F->mult4(a, s1, s7, p0)));
	b6[4] = F->mult4(a, s2, s6, p2);
	b6[5] = F->add(F->mult4(a, s2, s6, p3), F->mult3(s5, s6, p0));


	c12[0] = p1;
	c12[1] = p2;
	c12[2] = p3;
	c12[3] = p0;
	c12[4] = 0;
	c12[5] = 0;

	c13[0] = F->negate(F->mult(eta, p1));
	c13[1] = F->negate(F->mult(eta, p2));
	c13[2] = F->add(F->mult(epsilon, p0), F->negate(F->mult(eta, p3)));
	c13[3] = F->mult(delta, p1);
	c13[4] = F->add(F->mult(delta, p2), F->mult(s7, p0));
	c13[5] = F->mult(delta, p3);

	c14[0] = F->mult4(a, c, gamma, p1);
	c14[1] = F->mult4(a, c, gamma, p2);
	c14[2] = F->add(F->mult4(a, c, gamma, p3), F->mult(epsilon, p0));
	c14[3] = F->add(F->mult(F->add(delta, F->negate(s6)), p1), F->mult(s7, p0));
	c14[4] = F->add(F->mult(F->add(delta, F->negate(s6)), p2), F->mult(s7, p0));
	c14[5] = F->mult(F->add(delta, F->negate(s6)), p3);

	c15[0] = F->mult4(c, b, gamma, p1);
	c15[1] = F->mult4(c, b, gamma, p2);
	c15[2] = F->add(F->mult4(c, b, gamma, p3), F->mult(epsilon, p0));
	c15[3] = F->mult(b, F->add(p0, F->negate(p1)));
	c15[4] = F->add(p0, F->negate(F->mult(b, p2))); // found an error 10/31/2025, the last mult was add
	c15[5] = F->mult3(m1, b, p3);

	c16[0] = F->mult4(a, d, gamma, p1);
	c16[1] = F->mult4(a, d, gamma, p2);
	c16[2] = F->add(F->mult4(a, d, gamma, p3), F->mult(epsilon, p0));
	c16[3] = F->mult(d, F->add(p0, F->negate(p1)));
	c16[4] = F->add(p0, F->negate(F->mult(d, p2))); // found an error 10/31/2025, the last mult was add
	c16[5] = F->mult3(m1, d, p3);

	c23[0] = F->mult5(s7, a, c, gamma, p0);
	c23[1] = 0;
	c23[2] = F->mult3(epsilon, s6, p0);
	c23[3] = F->mult(delta, p0);
	c23[4] = F->mult(s6, p0);
	c23[5] = 0;

	c24[0] = F->add(F->mult3(delta, zeta, p0), F->mult4(m1, epsilon, s6, p1));
	c24[1] = F->mult4(m1, epsilon, s6, p2);
	c24[2] = F->add(F->mult3(epsilon, s6, p0), F->mult4(m1, epsilon, s6, p3));
	c24[3] = F->add(F->mult(F->add(delta, s7), p0), F->mult3(m1, s6, p1));
	c24[4] = F->mult(s6, F->add(p0, F->negate(p2)));
	c24[5] = F->mult3(m1, s6, p3);

	c25[0] = F->mult(a, F->add(p0, F->negate(p1)));
	c25[1] = F->add(p0, F->mult3(m1, a, p2));
	c25[2] = F->mult3(m1, a, p3);
	c25[3] = F->add(F->mult5(s1, s7, s8, a, p0), F->mult4(m1, a, epsilon, p1));
	c25[4] = F->mult4(m1, a, epsilon, p2);
	c25[5] = F->add(F->mult(epsilon, p0), F->mult4(m1, a, epsilon, p3));

	c26[0] = F->mult(c, F->add(p0, F->negate(p1)));
	c26[1] = F->add(p0, F->mult3(m1, c, p2));
	c26[2] = F->mult3(m1, c, p3);
	c26[3] = F->add(F->mult5(s3, s5, s7, c, p0), F->mult4(m1, c, epsilon, p1));
	c26[4] = F->mult4(m1, c, epsilon, p2);
	c26[5] = F->add(F->mult(epsilon, p0), F->mult4(m1, c, epsilon, p3));

	c34[0] = 0;
	c34[1] = p0;
	c34[2] = 0;
	c34[3] = F->add(p0, F->negate(p1)); // found mistake
	c34[4] = F->add(p0, F->negate(p2));
	c34[5] = F->negate(p3);

	c35[0] = 0;
	c35[1] = F->mult5(m1, s3, s7, a, p0);
	c35[2] = F->mult(epsilon, p0);
	c35[3] = F->add(F->mult3(s3, s7, p0), F->mult4(m1, s4, s6, p1)); // here was a mistake
	c35[4] = F->mult4(m1, s4, s6, p2);
	c35[5] = F->mult4(m1, s4, s6, p3);

	c36[0] = 0;
	c36[1] = F->mult5(m1, s1, s7, c, p0);
	c36[2] = F->mult(epsilon, p0);
	c36[3] = F->add(F->mult3(s1, s7, p0), F->mult4(m1, s2, s6, p1));
	c36[4] = F->mult4(m1, s2, s6, p2);
	c36[5] = F->mult4(m1, s2, s6, p3);

	c45[0] = F->mult5(m1, s5, s6, d, p1);
	c45[1] = F->add(F->mult5(m1, s1, s7, c, p0), F->mult5(m1, s5, s6, d, p2));
	c45[2] = F->add(F->mult(epsilon, p0), F->mult5(m1, s5, s6, d, p3));
	c45[3] = F->add(F->mult3(s7, c, p0), F->mult4(m1, s6, d, p1));
	c45[4] = F->mult4(m1, s6, d, p2);
	c45[5] = F->mult4(m1, s6, d, p3);

	c46[0] = F->mult5(m1, s6, s8, b, p1);
	c46[1] = F->add(F->mult5(m1, s3, s7, a, p0), F->mult5(m1, s6, s8, b, p2));
	c46[2] = F->add(F->mult(epsilon, p0), F->mult5(m1, s6, s8, b, p3));
	c46[3] = F->add(F->mult3(s7, a, p0), F->mult4(m1, s6, b, p1));
	c46[4] = F->mult4(m1, s6, b, p2);
	c46[5] = F->mult4(m1, s6, b, p3);

	c56[0] = F->add(F->mult(b, p0), F->mult3(m1, a, p1));
	c56[1] = F->add(p0, F->mult3(m1, a, p2));
	c56[2] = F->mult3(m1, a, p3);
	c56[3] = F->add(F->mult(d, p0), F->mult3(m1, c, p1));
	c56[4] = F->add(p0, F->mult3(m1, c, p2));
	c56[5] = F->mult3(m1, c, p3);

	if (lambda1) {
		t[0] = lambda2;
		t[1] = F->negate(lambda1);
		t[2] = 0;
		t[3] = lambda3;
		t[4] = 0;
		t[5] = F->negate(lambda1);
	}
	else if (lambda2) {
		t[0] = m1;
		t[1] = 0;
		t[2] = 0;
		t[3] = 0;
		t[4] = lambda3;
		t[5] = F->negate(lambda2);

	}
	else {
		t[0] = 1;
		t[1] = 0;
		t[2] = 0;
		t[3] = 0;
		t[4] = 1;
		t[5] = 0;

	}

	for (i = 0; i < 28; i++) {
		Lines_rk[i] = P->Subspaces->Grass_lines->rank_lint_here(
				Lines[i], 0 /*verbose_level*/);
	}


	//int delta, epsilon, gamma, lambda, mu, nu, eta, zeta, xi, theta, psi, omega;


	int nb_cols = 51;
	int nb_rows = 1;

	string fname_base;
	string *Table;

	fname_base = "qc_q" + std::to_string(F->q) + "_abcd_"
			+ std::to_string(a) + "_"
			+ std::to_string(b) + "_"
			+ std::to_string(c) + "_"
			+ std::to_string(d);

	Table = new string [nb_rows * nb_cols];
	Table[0] = fname_base;
	Table[1] = std::to_string(a);
	Table[2] = std::to_string(b);
	Table[3] = std::to_string(c);
	Table[4] = std::to_string(d);
	Table[5] = std::to_string(delta);
	Table[6] = std::to_string(epsilon);
	Table[7] = std::to_string(gamma);
	Table[8] = std::to_string(lambda);
	Table[9] = std::to_string(mu);
	Table[10] = std::to_string(nu);
	Table[11] = std::to_string(eta);
	Table[12] = std::to_string(zeta);
	Table[13] = std::to_string(xi);
	Table[14] = std::to_string(theta);
	Table[15] = std::to_string(psi);
	Table[16] = std::to_string(omega);
	Table[17] = std::to_string(p0);
	Table[18] = std::to_string(p1);
	Table[19] = std::to_string(p2);
	Table[20] = std::to_string(p3);
	Table[21] = std::to_string(lambda1);
	Table[22] = std::to_string(lambda2);
	Table[23] = std::to_string(lambda3);
	Table[24] = "\"" + Int_vec_stringify(a1, 6) + "\"";
	Table[25] = "\"" + Int_vec_stringify(a2, 6) + "\"";
	Table[26] = "\"" + Int_vec_stringify(a3, 6) + "\"";
	Table[27] = "\"" + Int_vec_stringify(a4, 6) + "\"";
	Table[28] = "\"" + Int_vec_stringify(a5, 6) + "\"";
	Table[29] = "\"" + Int_vec_stringify(a6, 6) + "\"";
	Table[30] = "\"" + Int_vec_stringify(b1, 6) + "\"";
	Table[31] = "\"" + Int_vec_stringify(b2, 6) + "\"";
	Table[32] = "\"" + Int_vec_stringify(b3, 6) + "\"";
	Table[33] = "\"" + Int_vec_stringify(b4, 6) + "\"";
	Table[34] = "\"" + Int_vec_stringify(b5, 6) + "\"";
	Table[35] = "\"" + Int_vec_stringify(b6, 6) + "\"";
	Table[36] = "\"" + Int_vec_stringify(c12, 6) + "\"";
	Table[37] = "\"" + Int_vec_stringify(c13, 6) + "\"";
	Table[38] = "\"" + Int_vec_stringify(c14, 6) + "\"";
	Table[39] = "\"" + Int_vec_stringify(c15, 6) + "\"";
	Table[40] = "\"" + Int_vec_stringify(c16, 6) + "\"";
	Table[41] = "\"" + Int_vec_stringify(c23, 6) + "\"";
	Table[42] = "\"" + Int_vec_stringify(c24, 6) + "\"";
	Table[43] = "\"" + Int_vec_stringify(c25, 6) + "\"";
	Table[44] = "\"" + Int_vec_stringify(c26, 6) + "\"";
	Table[45] = "\"" + Int_vec_stringify(c34, 6) + "\"";
	Table[46] = "\"" + Int_vec_stringify(c35, 6) + "\"";
	Table[47] = "\"" + Int_vec_stringify(c36, 6) + "\"";
	Table[48] = "\"" + Int_vec_stringify(c45, 6) + "\"";
	Table[49] = "\"" + Int_vec_stringify(c46, 6) + "\"";
	Table[50] = "\"" + Int_vec_stringify(c56, 6) + "\"";


	other::orbiter_kernel_system::file_io Fio;


	std::string fname_params;

	fname_params = fname_base + "_params.csv";

	std::string *Col_headings;

	Col_headings = new string [nb_cols];

	Col_headings[0] = "label";
	Col_headings[1] = "a";
	Col_headings[2] = "b";
	Col_headings[3] = "c";
	Col_headings[4] = "d";
	Col_headings[5] = "delta";
	Col_headings[6] = "epsilon";
	Col_headings[7] = "gamma";
	Col_headings[8] = "lambda";
	Col_headings[9] = "mu";
	Col_headings[10] = "nu";
	Col_headings[11] = "eta";
	Col_headings[12] = "zeta";
	Col_headings[13] = "xi";
	Col_headings[14] = "theta";
	Col_headings[15] = "psi";
	Col_headings[16] = "omega";
	Col_headings[17] = "p0";
	Col_headings[18] = "p1";
	Col_headings[19] = "p2";
	Col_headings[20] = "p3";
	Col_headings[21] = "lambda1";
	Col_headings[22] = "lambda2";
	Col_headings[23] = "lambda3";
	Col_headings[24] = "a1";
	Col_headings[25] = "a2";
	Col_headings[26] = "a3";
	Col_headings[27] = "a4";
	Col_headings[28] = "a5";
	Col_headings[29] = "a6";
	Col_headings[30] = "b1";
	Col_headings[31] = "b2";
	Col_headings[32] = "b3";
	Col_headings[33] = "b4";
	Col_headings[34] = "b5";
	Col_headings[35] = "b6";
	Col_headings[36] = "c12";
	Col_headings[37] = "c13";
	Col_headings[38] = "c14";
	Col_headings[39] = "c15";
	Col_headings[40] = "c16";
	Col_headings[41] = "c23";
	Col_headings[42] = "c24";
	Col_headings[43] = "c25";
	Col_headings[44] = "c26";
	Col_headings[45] = "c34";
	Col_headings[46] = "c35";
	Col_headings[47] = "c36";
	Col_headings[48] = "c45";
	Col_headings[49] = "c46";
	Col_headings[50] = "c56";


	if (f_v) {
		cout << "algebra_global_with_action::split_by_classes "
				"writing file " << fname_params << endl;
	}

	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname_params,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::split_by_classes "
				"written file " << fname_params << " of size "
				<< Fio.file_size(fname_params) << endl;
	}



#if 0
	QO = NEW_OBJECT(algebraic_geometry::quartic_curve_object);

	if (f_v) {
		cout << "surface_domain::create_quartic_curve_by_normal_form "
				"before SO->init_equation" << endl;
	}
	QO->init_equation_but_no_bitangents(
			this,
			coeffs15,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::create_quartic_curve_by_normal_form "
				"after SO->init_equation" << endl;
	}




	if (f_v) {
		cout << "surface_domain::create_quartic_curve_by_normal_form "
				"before compute_properties" << endl;
	}
	QO->compute_properties(verbose_level - 2);
	if (f_v) {
		cout << "surface_domain::create_quartic_curve_by_normal_form "
				"after compute_properties" << endl;
	}

#endif

	if (f_v) {
		cout << "surface_domain::create_quartic_curve_by_normal_form done" << endl;
	}
}

}}}}


