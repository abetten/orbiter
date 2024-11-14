/*
 * orbits_on_subspaces.cpp
 *
 *  Created on: Oct 20, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orbits {


static long int orbits_on_subspaces_rank_point_func(
		int *v, void *data);
static void orbits_on_subspaces_unrank_point_func(
		int *v, long int rk, void *data);
static void orbits_on_subspaces_early_test_func(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);



orbits_on_subspaces::orbits_on_subspaces()
{
	//GTA = NULL;
	Group = NULL;

	orbits_on_subspaces_Poset = NULL;
	orbits_on_subspaces_PC = NULL;
	orbits_on_subspaces_VS = NULL;
	orbits_on_subspaces_M = NULL;
	orbits_on_subspaces_base_cols = NULL;

}



orbits_on_subspaces::~orbits_on_subspaces()
{
}

void orbits_on_subspaces::init(
		groups::any_group *Group,
		poset_classification::poset_classification_control *Control,
		int depth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_subspaces::init" << endl;
	}
	//orbits_on_subspaces::GTA = GTA;
	orbits_on_subspaces::Group = Group;

	if (!Group->A->f_is_linear) {
		cout << "orbits_on_subspaces::init group must be linear" << endl;
		exit(1);
	}
	int n, q;

	n = Group->A->matrix_group_dimension();
	//n = Group->LG->n;
	if (f_v) {
		cout << "orbits_on_subspaces::init n = " << n << endl;
	}
	q = Group->A->matrix_group_finite_field()->q;
	if (f_v) {
		cout << "orbits_on_subspaces::init q = " << q << endl;
	}

	Control->f_depth = true;
	Control->depth = depth;
	if (f_v) {
		cout << "orbits_on_subspaces::init "
				"Control->max_depth=" << Control->depth << endl;
	}


	orbits_on_subspaces_PC = NEW_OBJECT(poset_classification::poset_classification);
	orbits_on_subspaces_Poset = NEW_OBJECT(poset_classification::poset_with_group_action);



	orbits_on_subspaces_M = NEW_int(n * n);
	orbits_on_subspaces_base_cols = NEW_int(n);

	orbits_on_subspaces_VS = NEW_OBJECT(linear_algebra::vector_space);
	orbits_on_subspaces_VS->init(
			Group->A->matrix_group_finite_field(),
			n /* dimension */,
			verbose_level - 1);
	orbits_on_subspaces_VS->init_rank_functions(
			orbits_on_subspaces_rank_point_func,
			orbits_on_subspaces_unrank_point_func,
			this,
			verbose_level - 1);


#if 0
	if (Descr->f_print_generators) {
		int f_print_as_permutation = false;
		int f_offset = true;
		int offset = 1;
		int f_do_it_anyway_even_for_big_degree = true;
		int f_print_cycles_of_length_one = true;

		cout << "orbits_on_subspaces::init "
				"printing generators "
				"for the group:" << endl;
		LG->Strong_gens->gens->print(cout,
			f_print_as_permutation,
			f_offset, offset,
			f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one);
	}
#endif

	orbits_on_subspaces_Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
	orbits_on_subspaces_Poset->init_subspace_lattice(
			Group->A_base /*LG->A_linear*/,
			Group->A /* LG->A2 */,
			Group->Subgroup_gens /* ->LG->Strong_gens */,
			orbits_on_subspaces_VS,
			verbose_level);
	orbits_on_subspaces_Poset->add_testing_without_group(
				orbits_on_subspaces_early_test_func,
				this /* void *data */,
				verbose_level);



	if (f_v) {
		cout << "orbits_on_subspaces::init "
				"Group->label=" << Group->label << endl;
	}

	if (!Control->f_problem_label) {
		Control->problem_label.assign(Group->label);
		Control->f_problem_label = true;
	}

	orbits_on_subspaces_PC->initialize_and_allocate_root_node(
			Control, orbits_on_subspaces_Poset,
			Control->depth, verbose_level);



	int schreier_depth = Control->depth;
	int f_use_invariant_subset_if_available = false;
	int f_debug = false;
	int nb_orbits;

	orbiter_kernel_system::os_interface Os;
	int t0 = Os.os_ticks();

	if (f_v) {
		cout << "orbits_on_subspaces::init "
				"calling generator_main" << endl;
		cout << "A=";
		orbits_on_subspaces_PC->get_A()->print_info();
		cout << "A2=";
		orbits_on_subspaces_PC->get_A2()->print_info();
	}
	orbits_on_subspaces_PC->main(t0,
		schreier_depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level - 1);


	if (f_v) {
		cout << "orbits_on_subspaces::init "
				"done with generator_main" << endl;
	}
	nb_orbits = orbits_on_subspaces_PC->nb_orbits_at_level(Control->depth);
	if (f_v) {
		cout << "orbits_on_subspaces::init we found "
				<< nb_orbits << " orbits at depth "
				<< Control->depth << endl;
	}

	orbits::orbits_global Orbits;

	if (f_v) {
		cout << "orbits_on_subspaces::init "
				"before Orbits.orbits_on_poset_post_processing" << endl;
	}
	Orbits.orbits_on_poset_post_processing(
			Group,
			orbits_on_subspaces_PC, Control->depth,
			verbose_level);
	if (f_v) {
		cout << "orbits_on_subspaces::init "
				"after Orbits.orbits_on_poset_post_processing" << endl;
	}




	if (f_v) {
		cout << "orbits_on_subspaces::init done" << endl;
	}
}



// #############################################################################
// global functions:
// #############################################################################


static long int orbits_on_subspaces_rank_point_func(
		int *v, void *data)
{
	orbits_on_subspaces *OoS;
	//group_theoretic_activity *G;
	poset_classification::poset_classification *gen;
	long int rk;

	//cout << "orbits_on_subspaces_rank_point_func temporarily disabled" << endl;
	//exit(1);


	OoS = (orbits_on_subspaces *) data;
	//G = OoS->GTA;
	gen = OoS->orbits_on_subspaces_PC;
	gen->get_VS()->F->Projective_space_basic->PG_element_rank_modified_lint(
			v, 1,
			gen->get_VS()->dimension, rk);
	return rk;
}

static void orbits_on_subspaces_unrank_point_func(
		int *v, long int rk, void *data)
{
	orbits_on_subspaces *OoS;
	//group_theoretic_activity *G;
	poset_classification::poset_classification *gen;

	//cout << "orbits_on_subspaces_unrank_point_func temporarily disabled" << endl;
	//exit(1);

	OoS = (orbits_on_subspaces *) data;
	//G = OoS->GTA;
	gen = OoS->orbits_on_subspaces_PC;
	gen->get_VS()->F->Projective_space_basic->PG_element_unrank_modified(
			v, 1,
			gen->get_VS()->dimension, rk);
}

static void orbits_on_subspaces_early_test_func(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	//verbose_level = 1;

	orbits_on_subspaces *OoS;
	//group_theoretic_activity *G;
	//poset_classification *gen;
	int f_v = (verbose_level >= 1);
	int i;

	OoS = (orbits_on_subspaces *) data;
	//G = OoS->GTA;

	//gen = G->orbits_on_subspaces_PC;

	if (f_v) {
		cout << "gorbits_on_subspaces_early_test_func" << endl;
		cout << "testing " << nb_candidates << " candidates" << endl;
	}
	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		S[len] = candidates[i];
		if (OoS->Group->Any_group_linear->subspace_orbits_test_set(len + 1, S, verbose_level - 1)) {
			good_candidates[nb_good_candidates++] = candidates[i];
		}
	}
	if (f_v) {
		cout << "orbits_on_subspaces_early_test_func" << endl;
		cout << "Out of " << nb_candidates << " candidates, "
				<< nb_good_candidates << " survive" << endl;
	}
}




}}}


