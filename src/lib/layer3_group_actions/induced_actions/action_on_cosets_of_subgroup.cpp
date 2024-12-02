/*
 * action_on_cosets_of_subgroup.cpp
 *
 *  Created on: Oct 27, 2024
 *      Author: betten
 */



#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_on_cosets_of_subgroup::action_on_cosets_of_subgroup()
{
	Record_birth();
	A = NULL;
	Subgroup_gens_H = NULL;
	Subgroup_gens_G = NULL;
	Sims_H = NULL;
	degree = 0;
	coset_reps = NULL;
	coset_reps_inverse = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	Elt4 = NULL;
}

action_on_cosets_of_subgroup::~action_on_cosets_of_subgroup()
{
	Record_death();
	if (Sims_H) {
		FREE_OBJECT(Sims_H);
	}
	if (coset_reps) {
		FREE_OBJECT(coset_reps);
	}
	if (coset_reps_inverse) {
		FREE_OBJECT(coset_reps_inverse);
	}
	if (Elt1) {
		FREE_int(Elt1);
	}
	if (Elt2) {
		FREE_int(Elt2);
	}
	if (Elt3) {
		FREE_int(Elt3);
	}
	if (Elt4) {
		FREE_int(Elt4);
	}
}


void action_on_cosets_of_subgroup::init(
		actions::action *A,
		groups::strong_generators *Subgroup_gens_H,
		groups::strong_generators *Subgroup_gens_G,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "action_on_cosets_of_subgroup::init" << endl;
	}
	action_on_cosets_of_subgroup::A = A;
	action_on_cosets_of_subgroup::Subgroup_gens_H = Subgroup_gens_H;
	action_on_cosets_of_subgroup::Subgroup_gens_G = Subgroup_gens_G;


	if (f_v) {
		cout << "any_group::set_of_coset_representatives "
				"before Subgroup_gens_H->create_sims" << endl;
	}
	Sims_H = Subgroup_gens_H->create_sims(verbose_level);
	if (f_v) {
		cout << "any_group::set_of_coset_representatives "
				"after Subgroup_gens_H->create_sims" << endl;
	}

	groups::group_theory_global Group_theory_global;

	if (f_v) {
		cout << "any_group::set_of_coset_representatives "
				"before Group_theory_global.set_of_coset_representatives" << endl;
	}
	Group_theory_global.set_of_coset_representatives(
			Subgroup_gens_H,
			Subgroup_gens_G,
			coset_reps,
			verbose_level);
	if (f_v) {
		cout << "any_group::set_of_coset_representatives "
				"after Group_theory_global.set_of_coset_representatives" << endl;
	}

	degree = coset_reps->len;
	if (f_v) {
		cout << "any_group::set_of_coset_representatives "
				"degree = " << degree << endl;
	}


	if (f_v) {
		cout << "any_group::set_of_coset_representatives "
				"before coset_reps->make_inverses" << endl;
	}
	coset_reps_inverse = coset_reps->make_inverses(
			verbose_level - 2);
	if (f_v) {
		cout << "any_group::set_of_coset_representatives "
				"after coset_reps->make_inverses" << endl;
	}

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);

	if (f_v) {
		cout << "action_on_cosets_of_subgroup::init done" << endl;
	}
}

long int action_on_cosets_of_subgroup::compute_image(
		int *Elt, long int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "action_on_cosets_of_subgroup::compute_image "
				"i = " << i << endl;
	}
	long int j;
	if (i < 0 || i >= degree) {
		cout << "action_on_cosets_of_subgroup::compute_image "
				"i = " << i << " out of range" << endl;
		exit(1);
	}

	A->Group_element->element_mult(coset_reps->ith(i), Elt, Elt1, false);
	for (j = 0; j < degree; j++) {
		A->Group_element->element_mult(Elt1, coset_reps_inverse->ith(j), Elt2, false);
		A->Group_element->element_move(Elt2, Elt3, false);

		int drop_out_level;
		int image;

		if (Sims_H->strip(
				Elt3, Elt4,
				drop_out_level,
				image, 0 /*verbose_level */)) {
			// returns true if the element sifts through)
			break;
		}
	}
	if (j == degree) {
		cout << "action_on_cosets_of_subgroup::compute_image "
				"could not find the coset containing the image" << endl;
		exit(1);

	}

	if (j < 0 || j >= degree) {
		cout << "action_on_cosets_of_subgroup::compute_image "
				"j = " << j << " out of range" << endl;
		exit(1);
	}
	return j;
}


}}}

