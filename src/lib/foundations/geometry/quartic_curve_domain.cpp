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



}}

