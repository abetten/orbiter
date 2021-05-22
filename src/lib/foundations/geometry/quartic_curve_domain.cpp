/*
 * quartic_curve_domain.cpp
 *
 *  Created on: May 21, 2021
 *      Author: betten
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


quartic_curve_domain::quartic_curve_domain()
{
	F = NULL;
	P = NULL;
	Poly3_3 = NULL;
	Poly4_3 = NULL;
	Partials = NULL;
}


quartic_curve_domain::~quartic_curve_domain()
{
}


void quartic_curve_domain::init(finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_domain::init" << endl;
	}

	quartic_curve_domain::F = F;

	P = NEW_OBJECT(projective_space);
	if (f_v) {
		cout << "quartic_curve_domain::init before P->init" << endl;
	}
	P->init(2, F,
		TRUE /*f_init_incidence_structure */,
		verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_domain::init after P->init" << endl;
	}



	if (f_v) {
		cout << "quartic_curve_domain::init before init_polynomial_domains" << endl;
	}
	init_polynomial_domains(verbose_level);
	if (f_v) {
		cout << "quartic_curve_domain::init after init_polynomial_domains" << endl;
	}


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


	Poly3_3 = NEW_OBJECT(homogeneous_polynomial_domain);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly3_3->init" << endl;
	}
	Poly3_3->init(F,
			3 /* nb_vars */, 3 /* degree */,
			FALSE /* f_init_incidence_structure */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly3_3->init" << endl;
	}


	Poly4_3 = NEW_OBJECT(homogeneous_polynomial_domain);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly4_3->init" << endl;
	}
	Poly4_3->init(F,
			3 /* nb_vars */, 4 /* degree */,
			FALSE /* f_init_incidence_structure */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly4_3->init" << endl;
	}

	Partials = NEW_OBJECTS(partial_derivative, 3);

	int i;

	if (f_v) {
		cout << "surface_domain::init_polynomial_domains initializing partials" << endl;
	}
	for (i = 0; i < 3; i++) {
		Partials[i].init(Poly4_3, Poly3_3, i, verbose_level);
	}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains initializing partials done" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_domain::init_polynomial_domains done" << endl;
	}

}

void quartic_curve_domain::print_equation_with_line_breaks_tex(std::ostream &ost, int *coeffs)
{
	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{c}" << endl;
	Poly4_3->print_equation_with_line_breaks_tex(
			ost, coeffs, 8 /* nb_terms_per_line*/,
			"\\\\\n" /* const char *new_line_text*/);
	ost << "=0" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
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

void quartic_curve_domain::print_lines_tex(std::ostream &ost, long int *Lines, int nb_lines)
{
	int i;
	latex_interface L;

	ost << "The lines are:\\\\" << endl;

	for (i = 0; i < nb_lines; i++) {
		//fp << "Line " << i << " is " << v[i] << ":\\\\" << endl;
		P->Grass_lines->unrank_lint(Lines[i], 0 /*verbose_level*/);
		ost << "$$" << endl;
		ost << "\\ell_{" << i << "}";

#if 0
		if (nb_lines == 27) {
			ost << " = " << Schlaefli->Line_label_tex[i];
		}
#endif
		ost << " = " << endl;
		//print_integer_matrix_width(cout,
		// P->Grass_lines->M, k, n, n, F->log10_of_q + 1);
		P->Grass_lines->latex_matrix(ost, P->Grass_lines->M);
		//print_integer_matrix_tex(ost, P->Grass_lines->M, 2, 4);
		//ost << "\\right]_{" << Lines[i] << "}" << endl;
		ost << "_{" << Lines[i] << "}" << endl;
		ost << "=" << endl;
		ost << "\\left[" << endl;
		L.print_integer_matrix_tex(ost, P->Grass_lines->M, 2, 3);
		ost << "\\right]_{" << Lines[i] << "}" << endl;

		ost << "$$" << endl;
	}
	ost << "Rank of lines: ";
	Orbiter->Lint_vec.print(ost, Lines, nb_lines);
	ost << "\\\\" << endl;

}


void quartic_curve_domain::compute_points_on_lines(
		long int *Pts, int nb_points,
		long int *Lines, int nb_lines,
		set_of_sets *&pts_on_lines,
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
	Orbiter->Int_vec.zero(f_is_on_line, nb_points);

	pts_on_lines = NEW_OBJECT(set_of_sets);
	pts_on_lines->init_basic_constant_size(nb_points,
		nb_lines, F->q + 1, 0 /* verbose_level */);
	pt_coords = NEW_int(nb_points * 3);
	for (i = 0; i < nb_points; i++) {
		P->unrank_point(pt_coords + i * 3, Pts[i]);
	}

	Orbiter->Lint_vec.zero(pts_on_lines->Set_size, nb_lines);
	for (i = 0; i < nb_lines; i++) {
		l = Lines[i];
		P->unrank_line(Basis, l);
		for (j = 0; j < nb_points; j++) {
			Orbiter->Int_vec.copy(Basis, Mtx, 6);
			Orbiter->Int_vec.copy(pt_coords + j * 3, Mtx + 6, 3);
			r = F->Gauss_easy(Mtx, 3, 3);
			if (r == 2) {
				pts_on_lines->add_element(i, j);
				f_is_on_line[j] = TRUE;
			}
		}
	}

	FREE_int(pt_coords);

	if (f_v) {
		cout << "quartic_curve_domain::compute_points_on_lines done" << endl;
	}
}


}}

