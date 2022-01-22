/*
 * quartic_curve_domain_with_action.cpp
 *
 *  Created on: May 21, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace quartic_curves {



quartic_curve_domain_with_action::quartic_curve_domain_with_action()
{
		PA = NULL;
		f_semilinear = FALSE;
		Dom = NULL;
		A = NULL;
		A_on_lines = NULL;
		Elt1 = NULL;
		AonHPD_4_3 = NULL;
}


quartic_curve_domain_with_action::~quartic_curve_domain_with_action()
{
}

void quartic_curve_domain_with_action::init(algebraic_geometry::quartic_curve_domain *Dom,
		projective_geometry::projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_domain_with_action::init" << endl;
	}
	quartic_curve_domain_with_action::Dom = Dom;
	quartic_curve_domain_with_action::PA = PA;



	A = PA->A;

	if (f_v) {
		cout << "quartic_curve_domain_with_action::init action A:" << endl;
		A->print_info();
	}



	A_on_lines = PA->A_on_lines;
	if (f_v) {
		cout << "quartic_curve_domain_with_action::init action A_on_lines:" << endl;
		A_on_lines->print_info();
	}
	f_semilinear = A->is_semilinear_matrix_group();
	if (f_v) {
		cout << "quartic_curve_domain_with_action::init f_semilinear=" << f_semilinear << endl;
	}


	Elt1 = NEW_int(A->elt_size_in_int);

	AonHPD_4_3 = NEW_OBJECT(induced_actions::action_on_homogeneous_polynomials);
	if (f_v) {
		cout << "quartic_curve_domain_with_action::init "
				"before AonHPD_4_3->init" << endl;
	}
	AonHPD_4_3->init(A, Dom->Poly4_3, verbose_level);

	if (f_v) {
		cout << "quartic_curve_domain_with_action::init done" << endl;
	}
}


}}}}

