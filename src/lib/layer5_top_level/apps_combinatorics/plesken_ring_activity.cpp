/*
 * plesken_ring_activity.cpp
 *
 *  Created on: Mar 3, 2026
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {



plesken_ring_activity::plesken_ring_activity()
{
	Record_birth();

	Descr = NULL;
	Plesken_ring = NULL;

}

plesken_ring_activity::~plesken_ring_activity()
{
	Record_death();
}


void plesken_ring_activity::init(
		plesken_ring_activity_description *Descr,
		apps_combinatorics::plesken_ring *Plesken_ring,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "plesken_ring_activity::init" << endl;
	}

	plesken_ring_activity::Descr = Descr;
	plesken_ring_activity::Plesken_ring = Plesken_ring;


	if (f_v) {
		cout << "plesken_ring_activity::init done" << endl;
	}
}



void plesken_ring_activity::perform_activity(
		int &nb_output,
		other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "plesken_ring_activity::perform_activity" << endl;
		cout << "plesken_ring_activity::perform_activity verbose_level = " << verbose_level << endl;
	}

	if (Descr->f_report) {
		if (f_v) {
			cout << "plesken_ring_activity::perform_activity f_report" << endl;
		}


		other::graphics::draw_options *Draw_options;

		Draw_options = Get_draw_options(Descr->report_draw_options_label);

		if (f_v) {
			cout << "plesken_ring_activity::perform_activity "
					"before SC->do_report" << endl;
		}
		Plesken_ring->do_report(Draw_options, verbose_level);
		if (f_v) {
			cout << "plesken_ring_activity::perform_activity "
					"after SC->do_report" << endl;
		}

	}
	if (Descr->f_evaluate_join) {
		if (f_v) {
			cout << "plesken_ring_activity::perform_activity f_evaluate_join" << endl;
		}


		algebra::ring_theory::homogeneous_polynomial_domain *Ring;

		Ring = Get_ring(Descr->evaluate_join_ring_label);


		algebra::expression_parser::symbolic_object_builder *Object;

		Object = Get_symbol(Descr->evaluate_join_formula_label);


		if (Ring->get_nb_variables() != Plesken_ring->N) {
			cout << "plesken_ring_activity::perform_activity "
					"number of variables is wrong, is " << Ring->get_nb_variables()
					<< " should be " << Plesken_ring->N << endl;
			exit(1);
		}


		if (f_v) {
			cout << "plesken_ring_activity::perform_activity "
					"before Plesken_ring->evaluate_expression" << endl;
		}
		Plesken_ring->evaluate_expression(true /* f_sup */, Ring, Object, verbose_level);
		if (f_v) {
			cout << "plesken_ring_activity::perform_activity "
					"after Plesken_ring->evaluate_expression" << endl;
		}

	}
	if (Descr->f_evaluate_meet) {
		if (f_v) {
			cout << "plesken_ring_activity::perform_activity f_evaluate_meet" << endl;
		}


		algebra::ring_theory::homogeneous_polynomial_domain *Ring;

		Ring = Get_ring(Descr->evaluate_meet_ring_label);


		algebra::expression_parser::symbolic_object_builder *Object;

		Object = Get_symbol(Descr->evaluate_meet_formula_label);


		if (Ring->get_nb_variables() != Plesken_ring->N) {
			cout << "plesken_ring_activity::perform_activity "
					"number of variables is wrong, is " << Ring->get_nb_variables()
					<< " should be " << Plesken_ring->N << endl;
			exit(1);
		}


		if (f_v) {
			cout << "plesken_ring_activity::perform_activity "
					"before Plesken_ring->evaluate_expression" << endl;
		}
		Plesken_ring->evaluate_expression(false /* f_sup */, Ring, Object, verbose_level);
		if (f_v) {
			cout << "plesken_ring_activity::perform_activity "
					"after Plesken_ring->evaluate_expression" << endl;
		}

	}


	if (f_v) {
		cout << "plesken_ring_activity::perform_activity done" << endl;
	}
}







}}}





