/*
 * polynomial_ring_activity.cpp
 *
 *  Created on: Feb 26, 2022
 *      Author: betten
 */



#include "foundations.h"


using namespace std;




namespace orbiter {
namespace layer1_foundations {
namespace ring_theory {



polynomial_ring_activity::polynomial_ring_activity()
{
	Descr = NULL;
	HPD = NULL;

}


polynomial_ring_activity::~polynomial_ring_activity()
{

}

void polynomial_ring_activity::init(polynomial_ring_activity_description *Descr,
		homogeneous_polynomial_domain *HPD,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "polynomial_ring_activity::init" << endl;
	}


	polynomial_ring_activity::Descr = Descr;
	polynomial_ring_activity::HPD = HPD;

	if (f_v) {
		cout << "polynomial_ring_activity::init done" << endl;
	}
}

void polynomial_ring_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "polynomial_ring_activity::perform_activity" << endl;
	}


	if (Descr->f_cheat_sheet) {

		//algebra::algebra_global Algebra;

		//Algebra.do_cheat_sheet_GF(F, verbose_level);
		//HPD->print_monomial_ordering(cout);
	}
	else if (Descr->f_ideal) {



		ring_theory::homogeneous_polynomial_domain *HPD;


		HPD = orbiter_kernel_system::Orbiter->get_object_of_type_polynomial_ring(Descr->ideal_ring_label);

		int dim_kernel;
		int nb_monomials;
		int *Kernel;

		HPD->create_ideal(
				Descr->ideal_label_txt,
				Descr->ideal_label_tex,
				Descr->ideal_point_set_label,
				dim_kernel, nb_monomials, Kernel,
				verbose_level);

		cout << "The ideal has dimension " << dim_kernel << endl;
		cout << "generators for the ideal:" << endl;
		Int_matrix_print(Kernel, dim_kernel, nb_monomials);

		int i;

		for (i = 0; i < dim_kernel; i++) {
			HPD->print_equation_relaxed(cout, Kernel + i * nb_monomials);
			cout << endl;
		}

	}

	if (f_v) {
		cout << "polynomial_ring_activity::perform_activity done" << endl;
	}

}



}}}


