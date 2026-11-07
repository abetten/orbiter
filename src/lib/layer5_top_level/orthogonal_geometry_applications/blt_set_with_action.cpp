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

	Blt_set = NULL;

	Aut_gens = NULL;
	//Inv = NULL;


	Blt_set_group_properties = NULL;
}

blt_set_with_action::~blt_set_with_action()
{
	Record_death();

#if 0
	if (Inv) {
		FREE_OBJECT(Inv);
	}
#endif

	if (Blt_set_group_properties) {
		FREE_OBJECT(Blt_set_group_properties);
	}
}

void blt_set_with_action::init_set(
		actions::action *A,
		orthogonal_geometry_applications::blt_set_domain_with_action *Blt_set_domain_with_action,
		geometry::orthogonal_geometry::blt_set *Blt_set, //long int *set,
		groups::strong_generators *Aut_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_with_action::init_set" << endl;
	}
	blt_set_with_action::A = A;
	blt_set_with_action::Blt_set_domain_with_action = Blt_set_domain_with_action;
	blt_set_with_action::Blt_set = Blt_set;
	blt_set_with_action::Aut_gens = Aut_gens;





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



	if (f_v) {
		cout << "blt_set_with_action::report_basics "
				"before Blt_set->report" << endl;
	}
	Blt_set->report(
			ost, verbose_level);
	if (f_v) {
		cout << "blt_set_with_action::report_basics "
				"after Blt_set->report" << endl;
	}



	ost << "\\bigskip" << endl;
	ost << "Orthogonal space:\\\\" << endl;
	ost << "\\bigskip" << endl;

	if (f_v) {
		cout << "blt_set_with_action::report_basics "
				"before Blt_set_domain_with_action->Blt_set_domain->O->report" << endl;
	}
	Blt_set_domain_with_action->Blt_set_domain->O->report(
			ost, verbose_level);
	if (f_v) {
		cout << "blt_set_with_action::report_basics "
				"after Blt_set_domain_with_action->Blt_set_domain->O->report" << endl;
	}

	if (f_v) {
		cout << "blt_set_with_action::report_basics done" << endl;
	}
}


}}}

