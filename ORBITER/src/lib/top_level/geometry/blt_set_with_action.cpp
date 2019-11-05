/*
 * blt_set_with_action.cpp
 *
 *  Created on: Apr 7, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


blt_set_with_action::blt_set_with_action()
{
	Blt_set = NULL;
	Blt_set_domain = NULL;
	Aut_gens = NULL;
	Inv = NULL;
	A_on_points = NULL;
	Orbits_on_points = NULL;
	null();
}

blt_set_with_action::~blt_set_with_action()
{
	freeself();
}

void blt_set_with_action::null()
{
}

void blt_set_with_action::freeself()
{
	if (Inv) {
		FREE_OBJECT(Inv);
	}
	if (A_on_points) {
		FREE_OBJECT(A_on_points);
	}
	if (Orbits_on_points) {
		FREE_OBJECT(Orbits_on_points);
	}
	null();
}

void blt_set_with_action::init_set(
		blt_set_classify *Blt_set, long int *set,
		strong_generators *Aut_gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_with_action::init_set" << endl;
		}
	blt_set_with_action::Blt_set = Blt_set;
	Blt_set_domain = Blt_set->Blt_set_domain;
	blt_set_with_action::Aut_gens = Aut_gens;
	Inv = NEW_OBJECT(blt_set_invariants);
	Inv->init(Blt_set_domain, set, verbose_level);

	init_orbits_on_points(verbose_level);

	if (f_v) {
		cout << "blt_set_with_action::init_set done" << endl;
		}
}


void blt_set_with_action::init_orbits_on_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_with_action::init_orbits_"
				"on_points" << endl;
		}

	if (f_v) {
		cout << "blt_set_with_action action "
				"on points:" << endl;
		}
	A_on_points = Blt_set->A->restricted_action(
			Inv->the_set_in_orthogonal,
			Blt_set_domain->target_size,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_object_with_action action "
				"on points done" << endl;
		}


	if (f_v) {
		cout << "computing orbits on points:" << endl;
		}
	Orbits_on_points = Aut_gens->orbits_on_points_schreier(
			A_on_points, 0 /*verbose_level*/);
	if (f_v) {
		cout << "We found " << Orbits_on_points->nb_orbits
				<< " orbits on points" << endl;
		}

	if (f_v) {
		cout << "blt_set_with_action::init_orbits_"
				"on_points done" << endl;
		}
}

void blt_set_with_action::print_automorphism_group(
	ostream &ost)
{
	longinteger_object go;

	Aut_gens->group_order(go);

	ost << "The automorphism group has order " << go << ".\\\\" << endl;
	ost << "\\bigskip" << endl;
	ost << "Orbits of the automorphism group on points "
			"of the BLT-set:\\\\" << endl;
	Orbits_on_points->print_and_list_orbits_sorted_by_length_tex(ost);
}




}}

