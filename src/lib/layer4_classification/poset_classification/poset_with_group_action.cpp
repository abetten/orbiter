// poset_with_group_action.cpp
//
// Anton Betten
// November 19, 2018

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


static int callback_test_independence_condition(
		orbit_based_testing *Obt,
		long int *S, int len, void *data, int verbose_level);


poset_with_group_action::poset_with_group_action()
{
	description = NULL;
	f_subset_lattice = false;
	n = 0;
	f_subspace_lattice = false;
	VS = NULL;

	A = NULL;
	A2 = NULL;
	Strong_gens = NULL;

	f_has_orbit_based_testing = false;
	Orbit_based_testing = NULL;

	f_print_function = false;;
	print_function = NULL;
	print_function_data = NULL;

	//null();
}

poset_with_group_action::~poset_with_group_action()
{
	if (f_has_orbit_based_testing) {
		if (Orbit_based_testing) {
			FREE_OBJECT(Orbit_based_testing);
		}
	}
}

void poset_with_group_action::init_subset_lattice(
		actions::action *A, actions::action *A2,
		groups::strong_generators *Strong_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_with_group_action::init_subset_lattice" << endl;
	}
	f_subset_lattice = true;
	n = A2->degree;
	if (f_v) {
		cout << "poset_with_group_action::init_subset_lattice "
				"degree of action = " << n << endl;
	}
	f_subspace_lattice = false;
	poset_with_group_action::A = A;
	poset_with_group_action::A2 = A2;
	poset_with_group_action::Strong_gens = Strong_gens;
	Strong_gens->group_order(go);
	f_has_orbit_based_testing = false;
	if (f_v) {
		cout << "poset_with_group_action::init_subset_lattice done" << endl;
	}
}

void poset_with_group_action::init_subspace_lattice(
		actions::action *A, actions::action *A2,
		groups::strong_generators *Strong_gens,
		linear_algebra::vector_space *VS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_with_group_action::init_subspace_lattice" << endl;
	}
	f_subset_lattice = false;
	n = A2->degree;
	f_subspace_lattice = true;
	poset_with_group_action::VS = VS;
	poset_with_group_action::A = A;
	poset_with_group_action::A2 = A2;
	poset_with_group_action::Strong_gens = Strong_gens;
	Strong_gens->group_order(go);
	f_has_orbit_based_testing = false;
	if (f_v) {
		cout << "poset_with_group_action::init_subspace_lattice done" << endl;
	}
}

#if 0
void poset_with_group_action::init(
		poset_description *description,
		actions::action *A, // the action in which the group is given
		actions::action *A2, // the action in which we do the search
		groups::strong_generators *Strong_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_with_group_action::init" << endl;
	}
	poset_with_group_action::description = description;
	poset_with_group_action::A = A;
	poset_with_group_action::A2 = A2;
	poset_with_group_action::Strong_gens = Strong_gens;


	f_subset_lattice = description->f_subset_lattice;
	n = A2->degree;
	f_subspace_lattice = description->f_subspace_lattice;
	VS = NEW_OBJECT(linear_algebra::vector_space);
	algebra::matrix_group *mtx;
	field_theory::finite_field *F;
	mtx = A->get_matrix_group();
	F = mtx->GFq;
	if (mtx->n != description->dimension) {
		cout << "poset_with_group_action::init mtx->n != description->dimension" << endl;
		exit(1);
	}
	VS->init(F, description->dimension, verbose_level);
	Strong_gens->group_order(go);
	f_has_orbit_based_testing = false;

	if (description->f_independence_condition) {
		add_independence_condition(
				description->independence_condition_value,
				verbose_level);
	}
	if (f_v) {
		cout << "poset_with_group_action::init action A:" << endl;
		A->print_info();
	}
	if (f_v) {
		cout << "poset_with_group_action::init action A2:" << endl;
		A2->print_info();
	}
	if (f_v) {
		cout << "poset_with_group_action::init generators for a group of order " << go
				<< " and degree " << A2->degree << endl;
	}
	if (f_v) {
		cout << "poset_with_group_action::init done" << endl;
	}
}
#endif

void poset_with_group_action::add_independence_condition(
		int independence_value,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_with_group_action::add_independence_condition" << endl;
	}
	if (f_v) {
		cout << "poset_with_group_action::init independence_condition value = "
				<< independence_value << endl;
	}

	algebra::rank_checker *rc;
	algebra::matrix_group *mtx;

	mtx = A->get_matrix_group();

	rc = NEW_OBJECT(algebra::rank_checker);
	if (f_v) {
		cout << "poset_with_group_action::add_independence_condition before "
				"rc->init" << endl;
	}
	rc->init(mtx->GFq,
			mtx->n,
			n,
			independence_value + 1);

	if (Orbit_based_testing == NULL) {
		f_has_orbit_based_testing = true;
		Orbit_based_testing = NEW_OBJECT(orbit_based_testing);
		if (f_v) {
			cout << "poset_with_group_action::add_independence_condition before "
					"Orbit_based_testing->init" << endl;
		}
		Orbit_based_testing->init(
			NULL /* poset_classification *PC */,
			n,
			verbose_level);
	}
	if (f_v) {
		cout << "poset_with_group_action::add_independence_condition "
				"adding callback for testing the "
				"independence condition" << endl;
	}
	Orbit_based_testing->add_callback(
			callback_test_independence_condition,
			(void *) rc,
			verbose_level);
	if (f_v) {
		cout << "poset_with_group_action::add_independence_condition done" << endl;
	}
}


void poset_with_group_action::add_testing(
		int (*func)(orbit_based_testing *Obt,
				long int *S, int len,
				void *data, int verbose_level),
		void *data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_with_group_action::add_testing" << endl;
	}

	if (Orbit_based_testing == NULL) {
		f_has_orbit_based_testing = true;
		Orbit_based_testing = NEW_OBJECT(orbit_based_testing);
		if (f_v) {
			cout << "poset_with_group_action::add_testing before "
					"Orbit_based_testing->init" << endl;
		}
		Orbit_based_testing->init(
			NULL /* poset_classification *PC */,
			n,
			verbose_level);
	}
	if (f_v) {
		cout << "poset_with_group_action::add_testing "
				"adding callback for testing the "
				"independence condition" << endl;
	}
	Orbit_based_testing->add_callback(
			func,
			data,
			verbose_level);

	if (f_v) {
		cout << "poset_with_group_action::add_testing done" << endl;
	}
}

void poset_with_group_action::add_testing_without_group(
		void (*func)(long int *S, int len,
				long int *candidates, int nb_candidates,
				long int *good_candidates, int &nb_good_candidates,
				void *data, int verbose_level),
		void *data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_with_group_action::add_testing_without_group" << endl;
	}

	if (Orbit_based_testing == NULL) {
		f_has_orbit_based_testing = true;
		Orbit_based_testing = NEW_OBJECT(orbit_based_testing);
		if (f_v) {
			cout << "poset_with_group_action::add_testing_without_group before "
					"Orbit_based_testing->init" << endl;
		}
		Orbit_based_testing->init(
			NULL /* poset_classification *PC */,
			n,
			verbose_level);
	}
	if (f_v) {
		cout << "poset_with_group_action::add_testing_without_group "
				"adding callback" << endl;
	}
	Orbit_based_testing->add_callback_no_group(
			func,
			data,
			verbose_level);

	if (f_v) {
		cout << "poset_with_group_action::add_testing_without_group done" << endl;
	}
}


void poset_with_group_action::print()
{
	if (f_subset_lattice) {
		cout << "poset of subsets of an " << n << "-element set" << endl;
	}
	if (f_subspace_lattice) {
		cout << "poset of subspaces of F_{" << VS->F->q << "}^{"
				<< VS->dimension << "}" << endl;
	}
	cout << "group action A:" << endl;
	A->print_info();
	cout << "group action A2:" << endl;
	A2->print_info();
	cout << "group order " << go << endl;
}

void poset_with_group_action::early_test_func(
	long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_with_group_action::early_test_func" << endl;
	}
	if (f_has_orbit_based_testing) {
		if (f_v) {
			cout << "poset_with_group_action::early_test_func "
					"before Orbit_based_testing->early_test_func_by_using_group" << endl;
		}
		Orbit_based_testing->early_test_func(
				S, len,
				candidates, nb_candidates,
				good_candidates, nb_good_candidates,
				verbose_level - 1);
		if (f_v) {
			cout << "poset_with_group_action::early_test_func "
					"after Orbit_based_testing->early_test_func_by_using_group" << endl;
		}
	}
	else {
		Lint_vec_copy(candidates, good_candidates, nb_candidates);
		nb_good_candidates = nb_candidates;
	}
	if (f_v) {
		cout << "poset_with_group_action::early_test_func done" << endl;
	}
}

void poset_with_group_action::unrank_point(int *v, long int rk)
{
	if (!f_subspace_lattice) {
		cout << "poset_with_group_action::unrank_point !f_subspace_lattice" << endl;
		exit(1);
	}
	if (VS == NULL) {
		cout << "poset_with_group_action::unrank_point VS == NULL" << endl;
		exit(1);
	}
	VS->unrank_point(v, rk);
}

long int poset_with_group_action::rank_point(int *v)
{
	long int rk;

	if (!f_subspace_lattice) {
		cout << "poset_with_group_action::rank_point !f_subspace_lattice" << endl;
		exit(1);
	}
	if (VS == NULL) {
		cout << "poset_with_group_action::rank_point VS == NULL" << endl;
		exit(1);
	}
	rk = VS->rank_point(v);
	return rk;
}

void poset_with_group_action::orbits_on_k_sets(
		poset_classification_control *Control,
		int k, long int *&orbit_reps,
		int &nb_orbits, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	poset_classification *Gen;

	if (f_v) {
		cout << "poset_with_group_action::orbits_on_k_sets" << endl;
	}

	Gen = orbits_on_k_sets_compute(Control,
		k, verbose_level);
	if (f_v) {
		cout << "poset_with_group_action::orbits_on_k_sets "
				"done with orbits_on_k_sets_compute" << endl;
	}

	Gen->get_orbit_representatives(k, nb_orbits,
			orbit_reps, verbose_level);


	if (f_v) {
		cout << "poset_with_group_action::orbits_on_k_sets "
				"we found "
				<< nb_orbits << " orbits on " << k << "-sets" << endl;
	}

	FREE_OBJECT(Gen);
	if (f_v) {
		cout << "poset_with_group_action::orbits_on_k_sets done" << endl;
	}
}

poset_classification *poset_with_group_action::orbits_on_k_sets_compute(
		poset_classification_control *Control,
		int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	poset_classification *Gen;


	if (f_v) {
		cout << "poset_with_group_action::orbits_on_k_sets_compute" << endl;
	}
	Gen = NEW_OBJECT(poset_classification);


	if (f_v) {
		cout << "poset_with_group_action::orbits_on_k_sets_compute calling Gen->init" << endl;
	}
	Gen->initialize_and_allocate_root_node(
			Control,
			this,
			k /* sz */,
			verbose_level - 1);

	orbiter_kernel_system::os_interface Os;
	int schreier_depth = k;
	int f_use_invariant_subset_if_available = true;
	int f_debug = false;
	int t0 = Os.os_ticks();

	if (f_v) {
		cout << "poset_with_group_action::orbits_on_k_sets_compute "
				"calling generator_main" << endl;
		}
	Gen->main(t0,
		schreier_depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level - 1);


	if (f_v) {
		cout << "poset_with_group_action::orbits_on_k_sets_compute done" << endl;
	}
	return Gen;
}

void poset_with_group_action::invoke_print_function(
		std::ostream &ost, int sz, long int *set)
{
	(*print_function)(ost, sz, set, print_function_data);
}

//##############################################################################
// global functions:
//##############################################################################


static int callback_test_independence_condition(
		orbit_based_testing *Obt,
		long int *S, int len, void *data, int verbose_level)
{
	int f_v = (verbose_level >= 0);

	if (f_v) {
		cout << "callback_test_independence_condition" << endl;
	}
	algebra::rank_checker *rc;

	rc = (algebra::rank_checker *) data;
	if (rc->check_rank_last_two_are_fixed(len,
		S, verbose_level - 1)) {
		return true;
	}
	else {
		return false;
	}

}

}}}


