// poset.C
//
// Anton Betten
// November 19, 2018

#include "foundations/foundations.h"
#include "groups_and_group_actions/groups_and_group_actions.h"
#include "poset_classification/poset_classification.h"



poset::poset()
{
	null();
}

poset::~poset()
{
	freeself();
}

void poset::null()
{


	description = NULL;
	f_subset_lattice = FALSE;
	n = 0;
	f_subspace_lattice = FALSE;
	VS = NULL;

	A = NULL;
	A2 = NULL;
	Strong_gens = NULL;

	f_has_orbit_based_testing = FALSE;
	Orbit_based_testing = NULL;

}

void poset::freeself()
{
	if (f_has_orbit_based_testing) {
		if (Orbit_based_testing) {
			FREE_OBJECT(Orbit_based_testing);
		}
	}
	null();
}

void poset::init_subset_lattice(action *A, action *A2,
		strong_generators *Strong_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset::init_subset_lattice" << endl;
		}
	f_subset_lattice = TRUE;
	n = A2->degree;
	f_subspace_lattice = FALSE;
	poset::A = A;
	poset::A2 = A2;
	poset::Strong_gens = Strong_gens;
	Strong_gens->group_order(go);
	f_has_orbit_based_testing = FALSE;
	if (f_v) {
		cout << "poset::init_subset_lattice done" << endl;
		}
}

void poset::init_subspace_lattice(action *A, action *A2,
		strong_generators *Strong_gens,
		vector_space *VS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset::init_subspace_lattice" << endl;
		}
	f_subset_lattice = FALSE;
	n = A2->degree;
	f_subspace_lattice = TRUE;
	poset::VS = VS;
	poset::A = A;
	poset::A2 = A2;
	poset::Strong_gens = Strong_gens;
	Strong_gens->group_order(go);
	f_has_orbit_based_testing = FALSE;
	if (f_v) {
		cout << "poset::init_subspace_lattice done" << endl;
		}
}

void poset::init(
		poset_description *description,
		action *A, // the action in which the group is given
		action *A2, // the action in which we do the search
		strong_generators *Strong_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset::init" << endl;
		}
	poset::description = description;
	poset::A = A;
	poset::A2 = A2;
	poset::Strong_gens = Strong_gens;


	f_subset_lattice = description->f_subset_lattice;
	n = A2->degree;
	f_subspace_lattice = description->f_subspace_lattice;
	VS = NEW_OBJECT(vector_space);
	matrix_group *mtx;
	finite_field *F;
	mtx = A->get_matrix_group();
	F = mtx->GFq;
	if (mtx->n != description->dimension) {
		cout << "poset::init mtx->n != description->dimension" << endl;
		exit(1);
	}
	VS->init(F, description->dimension, verbose_level);
	Strong_gens->group_order(go);
	f_has_orbit_based_testing = FALSE;

	if (description->f_independence_condition) {
		add_independence_condition(
				description->independence_condition_value,
				verbose_level);
	}
	if (f_v) {
		cout << "poset::init action A:" << endl;
		A->print_info();
		}
	if (f_v) {
		cout << "poset::init action A2:" << endl;
		A2->print_info();
		}
	if (f_v) {
		cout << "poset::init generators for a group of order " << go
				<< " and degree " << A2->degree << endl;
		}
	if (f_v) {
		cout << "poset::init done" << endl;
		}
}

void poset::add_independence_condition(
		int independence_value,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset::add_independence_condition" << endl;
		}
	if (f_v) {
		cout << "poset::init independence_condition value = "
				<< independence_value << endl;
		}

	rank_checker *rc;
	matrix_group *mtx;

	mtx = A->get_matrix_group();

	rc = NEW_OBJECT(rank_checker);
	if (f_v) {
		cout << "poset::add_independence_condition before "
				"rc->init" << endl;
	}
	rc->init(mtx->GFq,
			mtx->n,
			n,
			independence_value + 1);

	if (Orbit_based_testing == NULL) {
		f_has_orbit_based_testing = TRUE;
		Orbit_based_testing = NEW_OBJECT(orbit_based_testing);
		if (f_v) {
			cout << "poset::add_independence_condition before "
					"Orbit_based_testing->init" << endl;
		}
		Orbit_based_testing->init(
			NULL /* poset_classification *PC */,
			n,
			verbose_level);
	}
	if (f_v) {
		cout << "poset::add_independence_condition "
				"adding callback for testing the "
				"independence condition" << endl;
	}
	Orbit_based_testing->add_callback(
			callback_test_independence_condition,
			(void *) rc,
			verbose_level);
	if (f_v) {
		cout << "poset::add_independence_condition done" << endl;
		}
}


void poset::add_testing_without_group(
		void (*func)(int *S, int len,
				int *candidates, int nb_candidates,
				int *good_candidates, int &nb_good_candidates,
				void *data, int verbose_level),
		void *data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset::add_testing_without_group" << endl;
		}

	if (Orbit_based_testing == NULL) {
		f_has_orbit_based_testing = TRUE;
		Orbit_based_testing = NEW_OBJECT(orbit_based_testing);
		if (f_v) {
			cout << "poset::add_testing_without_group before "
					"Orbit_based_testing->init" << endl;
		}
		Orbit_based_testing->init(
			NULL /* poset_classification *PC */,
			n,
			verbose_level);
	}
	if (f_v) {
		cout << "poset::add_testing_without_group "
				"adding callback for testing the "
				"independence condition" << endl;
	}
	Orbit_based_testing->add_callback_no_group(
			func,
			data,
			verbose_level);

	if (f_v) {
		cout << "poset::add_testing_without_group done" << endl;
		}
}


void poset::print()
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

void poset::early_test_func(
	int *S, int len,
	int *candidates, int nb_candidates,
	int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset::early_test_func" << endl;
	}
	if (f_has_orbit_based_testing) {
		if (f_v) {
			cout << "poset::early_test_func "
					"before Orbit_based_testing->early_test_"
					"func_by_using_group" << endl;
		}
		Orbit_based_testing->early_test_func(
				S, len,
				candidates, nb_candidates,
				good_candidates, nb_good_candidates,
				verbose_level - 1);
		if (f_v) {
			cout << "poset::early_test_func "
					"after Orbit_based_testing->early_test_"
					"func_by_using_group" << endl;
		}
	}
	else {
		int_vec_copy(candidates, good_candidates, nb_candidates);
		nb_good_candidates = nb_candidates;
	}
	if (f_v) {
		cout << "poset::early_test_func done" << endl;
	}
}

void poset::unrank_point(int *v, int rk)
{
	if (!f_subspace_lattice) {
		cout << "poset::unrank_point !f_subspace_lattice" << endl;
		exit(1);
	}
	if (VS == NULL) {
		cout << "poset::unrank_point VS == NULL" << endl;
		exit(1);
	}
	VS->unrank_point(v, rk);
}

int poset::rank_point(int *v)
{
	int rk;

	if (!f_subspace_lattice) {
		cout << "poset::rank_point !f_subspace_lattice" << endl;
		exit(1);
	}
	if (VS == NULL) {
		cout << "poset::rank_point VS == NULL" << endl;
		exit(1);
	}
	rk = VS->rank_point(v);
	return rk;
}



int callback_test_independence_condition(orbit_based_testing *Obt,
		int *S, int len, void *data, int verbose_level)
{
	int f_v = (verbose_level >= 0);

	if (f_v) {
		cout << "callback_test_independence_condition" << endl;
	}
	rank_checker *rc;

	rc = (rank_checker *) data;
	if (rc->check_rank_last_two_are_fixed(len,
		S, verbose_level - 1)) {
		return TRUE;
		}
	else {
		return FALSE;
		}

}
