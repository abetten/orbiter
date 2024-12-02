/*
 * blt_set_with_action.cpp
 *
 *  Created on: Apr 7, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {


blt_set_with_action::blt_set_with_action()
{
	Record_birth();
	A = NULL;
	Blt_set_domain_with_action = NULL;

	set = NULL;

	//std::string label_txt;
	//std::string label_tex;

	Aut_gens = NULL;
	Inv = NULL;

	T = NULL;
	Pi_ij = NULL;

	Blt_set_group_properties = NULL;
}

blt_set_with_action::~blt_set_with_action()
{
	Record_death();
	if (Inv) {
		FREE_OBJECT(Inv);
	}
	if (T) {
		FREE_lint(T);
	}
	if (Pi_ij) {
		FREE_lint(Pi_ij);
	}
	if (Blt_set_group_properties) {
		FREE_OBJECT(Blt_set_group_properties);
	}
}

void blt_set_with_action::init_set(
		actions::action *A,
		orthogonal_geometry_applications::blt_set_domain_with_action *Blt_set_domain_with_action,
		long int *set,
		std::string &label_txt,
		std::string &label_tex,
		groups::strong_generators *Aut_gens,
		int f_invariants,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_with_action::init_set" << endl;
	}
	blt_set_with_action::A = A;
	blt_set_with_action::Blt_set_domain_with_action = Blt_set_domain_with_action;
	blt_set_with_action::set = set;
	blt_set_with_action::Aut_gens = Aut_gens;
	blt_set_with_action::label_txt.assign(label_txt);
	blt_set_with_action::label_tex.assign(label_tex);

	Inv = NEW_OBJECT(geometry::orthogonal_geometry::blt_set_invariants);
	if (f_v) {
		cout << "blt_set_with_action::init_set "
				"before Inv->init" << endl;
	}
	Inv->init(Blt_set_domain_with_action->Blt_set_domain, set, verbose_level - 1);
	if (f_v) {
		cout << "blt_set_with_action::init_set "
				"after Inv->init" << endl;
	}

	if (f_invariants) {
		if (f_v) {
			cout << "blt_set_with_action::init_set "
					"before Inv->compute" << endl;
		}
		Inv->compute(verbose_level - 1);
		if (f_v) {
			cout << "blt_set_with_action::init_set "
					"after Inv->compute" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "blt_set_with_action::init_set "
					"We don't compute invariants." << endl;
		}

	}


	if (f_v) {
		cout << "blt_set_with_action::init_set "
				"before compute_T" << endl;
	}
	compute_T(verbose_level - 1);
	if (f_v) {
		cout << "blt_set_with_action::init_set "
				"after compute_T" << endl;
	}

	if (f_v) {
		cout << "blt_set_with_action::init_set "
				"before compute_Pi_ij" << endl;
	}
	compute_Pi_ij(verbose_level - 1);
	if (f_v) {
		cout << "blt_set_with_action::init_set "
				"after compute_Pi_ij" << endl;
	}



	if (Aut_gens) {

		Blt_set_group_properties = NEW_OBJECT(blt_set_group_properties);


		if (f_v) {
			cout << "blt_set_with_action::init_set "
					"before Blt_set_group_properties->init_blt_set_group_properties" << endl;
		}
		Blt_set_group_properties->init_blt_set_group_properties(
				this, verbose_level);
		if (f_v) {
			cout << "blt_set_with_action::init_set "
					"after Blt_set_group_properties->init_blt_set_group_properties" << endl;
		}

	}


	if (f_v) {
		cout << "blt_set_with_action::init_set done" << endl;
	}
}


void blt_set_with_action::compute_T(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_with_action::compute_T" << endl;
	}
	int i;
	//int i, j;
	//long int plane_rk1, plane_rk2, a;

	T = NEW_lint(Blt_set_domain_with_action->Blt_set_domain->target_size);
	for (i = 0; i < Blt_set_domain_with_action->Blt_set_domain->target_size; i++) {
		T[i] = Blt_set_domain_with_action->Blt_set_domain->compute_tangent_hyperplane(
				set[i],
				0 /* verbose_level */);
	}

	if (f_v) {
		cout << "blt_set_with_action::compute_T done" << endl;
	}
}


void blt_set_with_action::compute_Pi_ij(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_with_action::compute_Pi_ij" << endl;
	}
	int i, j;
	long int plane_rk1, plane_rk2, a;

	Pi_ij = NEW_lint(Blt_set_domain_with_action->Blt_set_domain->target_size * Blt_set_domain_with_action->Blt_set_domain->target_size);

	Lint_vec_zero(Pi_ij, Blt_set_domain_with_action->Blt_set_domain->target_size * Blt_set_domain_with_action->Blt_set_domain->target_size);

	for (i = 0; i < Blt_set_domain_with_action->Blt_set_domain->target_size; i++) {
		plane_rk1 = T[i];
		for (j = 0; j < Blt_set_domain_with_action->Blt_set_domain->target_size; j++) {
			if (i == j) {
				continue;
			}
			plane_rk2 = T[j];
			a = Blt_set_domain_with_action->Blt_set_domain->intersection_of_hyperplanes(
					plane_rk1, plane_rk2,
					0 /* verbose_level */);
			Pi_ij[i * Blt_set_domain_with_action->Blt_set_domain->target_size + j] = a;
		}
	}

	if (f_v) {
		cout << "blt_set_with_action::compute_Pi_ij done" << endl;
	}
}

void blt_set_with_action::report(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_with_action::report" << endl;
	}

	if (Aut_gens) {


		Blt_set_group_properties->report(ost, verbose_level);

	}



	if (f_v) {
		cout << "blt_set_group_properties::report done" << endl;
	}
}


void blt_set_with_action::report_basics(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_with_action::report_basics" << endl;
	}

	ost << "\\bigskip" << endl;
	ost << "BLT-set and tangent hyperplanes $T$:\\\\" << endl;
	ost << "\\bigskip" << endl;


	Blt_set_domain_with_action->Blt_set_domain->report_given_point_set(ost,
			set, Blt_set_domain_with_action->Blt_set_domain->q + 1, verbose_level);

	ost << "\\bigskip" << endl;


	ost << "Tangent hyperplanes:\\\\" << endl;

	Blt_set_domain_with_action->Blt_set_domain->G54->print_set_tex(
			ost, T, Blt_set_domain_with_action->Blt_set_domain->target_size, verbose_level);

	ost << "\\bigskip" << endl;


	other::l1_interfaces::latex_interface Li;

	ost << "$\\Pi_{ij}$ matrix:" << endl;
	ost << "$$" << endl;
	ost << "\\left[" << endl;

	Li.print_lint_matrix_tex(ost,
			Pi_ij,
			Blt_set_domain_with_action->Blt_set_domain->target_size,
			Blt_set_domain_with_action->Blt_set_domain->target_size);

	ost << "\\right]" << endl;
	ost << "$$" << endl;

	ost << "\\bigskip" << endl;

	ost << "First row of $\\Pi_{ij}$ (with the diagonal entry removed):\\\\" << endl;

	Blt_set_domain_with_action->Blt_set_domain->G53->print_set_tex(ost,
			Pi_ij + 1,
			Blt_set_domain_with_action->Blt_set_domain->q,
			verbose_level);

	ost << "\\bigskip" << endl;


	if (Inv) {
		if (f_v) {
			cout << "blt_set_group_properties::report "
					"before Inv->latex" << endl;
		}
		Inv->latex(ost, verbose_level);
		if (f_v) {
			cout << "blt_set_group_properties::report "
					"after Inv->latex" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "blt_set_group_properties::report "
					"Inv not available" << endl;
		}

	}


	ost << "\\bigskip" << endl;
	ost << "Orthogonal space:\\\\" << endl;
	ost << "\\bigskip" << endl;

	Blt_set_domain_with_action->Blt_set_domain->O->report(ost, verbose_level);

	if (f_v) {
		cout << "blt_set_with_action::report_basics done" << endl;
	}
}


}}}

