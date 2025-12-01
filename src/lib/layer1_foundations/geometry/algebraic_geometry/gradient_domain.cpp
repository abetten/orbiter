/*
 * gradient_domain.cpp
 *
 *  Created on: Nov 11, 2025
 *      Author: betten
 */





#include "foundations.h"


using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace algebraic_geometry {



gradient_domain::gradient_domain()
{
	Record_birth();

	Poly = NULL;
	Poly_small = NULL;

	nb_variables = 0;

	Partials = NULL;

}

gradient_domain::~gradient_domain()
{
	Record_death();

	if (Partials) {
		FREE_OBJECTS(Partials);
	}
}

void gradient_domain::init(
		algebra::ring_theory::homogeneous_polynomial_domain *Poly,
		algebra::ring_theory::homogeneous_polynomial_domain *Poly_small,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gradient_domain::init" << endl;
	}
	gradient_domain::Poly = Poly;
	gradient_domain::Poly_small = Poly_small;

	nb_variables = Poly->nb_variables;

	if (Poly_small->nb_variables != nb_variables) {
		cout << "gradient_domain::init the number of variables must be the same" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "gradient_domain::init nb_variables = " << nb_variables << endl;
	}

	Partials = NEW_OBJECTS(algebra::ring_theory::partial_derivative, nb_variables);

	int i;

	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"initializing partials" << endl;
	}
	for (i = 0; i < nb_variables; i++) {
		Partials[i].init(Poly, Poly_small, i, verbose_level);
	}
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"initializing partials done" << endl;
	}

	if (f_v) {
		cout << "gradient_domain::init done" << endl;
	}
}


void gradient_domain::compute_gradient(
		int *equation_in, int *&gradient, int verbose_level)
// input: equation_in
// output: gradient[nb_variables * Poly->get_nb_monomials()]
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "gradient_domain::compute_gradient" << endl;
	}


	if (f_v) {
		cout << "gradient_domain::compute_gradient "
				"Poly2_4->get_nb_monomials() = " << Poly->get_nb_monomials() << endl;
	}

	gradient = NEW_int(nb_variables * Poly->get_nb_monomials());

	for (i = 0; i < 4; i++) {
		if (f_v) {
			cout << "gradient_domain::compute_gradient i=" << i << endl;
		}
		if (f_v) {
			cout << "gradient_domain::compute_gradient "
					"eqn_in=";
			Int_vec_print(cout, equation_in, Poly->get_nb_monomials());
			cout << " = " << endl;
			Poly->print_equation(cout, equation_in);
			cout << endl;
		}
		Partials[i].apply(equation_in,
				gradient + i * Poly->get_nb_monomials(),
				verbose_level - 2);
		if (f_v) {
			cout << "gradient_domain::compute_gradient "
					"partial=";
			Int_vec_print(cout, gradient + i * Poly->get_nb_monomials(),
					Poly_small->get_nb_monomials());
			cout << " = ";
			Poly_small->print_equation(cout,
					gradient + i * Poly_small->get_nb_monomials());
			cout << endl;
		}
	}


	if (f_v) {
		cout << "gradient_domain::compute_gradient done" << endl;
	}
}




}}}}


