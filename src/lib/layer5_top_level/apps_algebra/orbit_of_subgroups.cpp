/*
 * orbit_of_subgroups.cpp
 *
 *  Created on: Feb 23, 2025
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


orbit_of_subgroups::orbit_of_subgroups()
{
	Record_birth();

	Class = NULL;

	idx = 0;

	go_P = 0;

	Sims_P = NULL;
	Elements_P = NULL;
	Orbits_P = NULL;
}


orbit_of_subgroups::~orbit_of_subgroups()
{
	Record_death();

	if (Sims_P) {
		FREE_OBJECT(Sims_P);
	}
	if (Elements_P) {
		FREE_OBJECT(Elements_P);
	}
	if (Orbits_P) {
		FREE_OBJECT(Orbits_P);
	}
}

void orbit_of_subgroups::init(
		groups::any_group *Any_group,
		groups::sims *Sims_G,
		actions::action *A_conj,
		interfaces::conjugacy_classes_of_subgroups *Classes,
		int idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_subgroups::init" << endl;
	}

	orbit_of_subgroups::idx = idx;


	other::data_structures::sorting Sorting;

	int *Elt;

	Elt = NEW_int(Classes->A->elt_size_in_int);

	long int rk;

	Sims_P = Classes->Conjugacy_class[idx]->gens->create_sims(verbose_level);
	go_P = Sims_P->group_order_lint();

	Elements_P = NEW_lint(go_P);

	int i;
	for (i = 0; i < go_P; i++) {
		Sims_P->element_unrank_lint(i, Elt);
		rk = Sims_G->element_rank_lint(Elt);
		Elements_P[i] = rk;
	}
	Sorting.lint_vec_heapsort(Elements_P, go_P);

	if (f_v) {
		cout << "orbit_of_subgroups::init "
				"before Any_group->A->create_induced_action_by_conjugation" << endl;
	}



	Orbits_P = NEW_OBJECT(orbits_schreier::orbit_of_sets);


	if (f_v) {
		cout << "orbit_of_subgroups::init "
				"before Orbits_P->init" << endl;
	}

	Orbits_P->init(
			Any_group->A,
			A_conj,
			Elements_P, go_P,
			Any_group->Subgroup_gens->gens,
			verbose_level);

	if (f_v) {
		cout << "orbit_of_subgroups::init "
				"after Orbits_P->init" << endl;
	}

	FREE_int(Elt);

	if (f_v) {
		cout << "orbit_of_subgroups::init done" << endl;
	}



}


}}}

