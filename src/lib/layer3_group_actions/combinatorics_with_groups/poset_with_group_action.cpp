// poset_with_group_action.cpp
//
// Anton Betten
// November 19, 2018

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace combinatorics_with_groups {


static int callback_test_independence_condition(
		orbit_based_testing *Obt,
		long int *S, int len, void *data, int verbose_level);


poset_with_group_action::poset_with_group_action()
{
	Record_birth();
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
	Record_death();
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
		cout << "poset_with_group_action::init_subset_lattice "
				"A  = ";
		A->print_info();
		cout << "poset_with_group_action::init_subset_lattice "
				"A2 = ";
		A2->print_info();
		cout << "poset_with_group_action::init_subset_lattice generators = ";
		Strong_gens->print_generators_tex(cout);
		Strong_gens->print_generators(cout, 0 /* verbose_level */);
	}




	if (f_v) {
		cout << "poset_with_group_action::init_subset_lattice done" << endl;
	}
}

void poset_with_group_action::init_subspace_lattice(
		actions::action *A, actions::action *A2,
		groups::strong_generators *Strong_gens,
		algebra::linear_algebra::vector_space *VS,
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

	algebra::basic_algebra::rank_checker *rc;
	algebra::basic_algebra::matrix_group *mtx;

	mtx = A->get_matrix_group();

	rc = NEW_OBJECT(algebra::basic_algebra::rank_checker);
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
			this,
			//NULL /* poset_classification *PC */,
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
				this,
			//NULL /* poset_classification *PC */,
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
			this,
			//NULL /* poset_classification *PC */,
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

void poset_with_group_action::unrank_point(
		int *v, long int rk)
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

long int poset_with_group_action::rank_point(
		int *v)
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

void poset_with_group_action::unrank_basis(
		int *Basis, long int *S, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		unrank_point(Basis + i * VS->dimension, S[i]);
	}
}

void poset_with_group_action::rank_basis(
		int *Basis, long int *S, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		S[i] = rank_point(Basis + i * VS->dimension);
	}
}



void poset_with_group_action::invoke_print_function(
		std::ostream &ost, int sz, long int *set)
{
	(*print_function)(ost, sz, set, print_function_data);
}


int poset_with_group_action::is_contained(
		long int *set1, int sz1, long int *set2, int sz2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_contained;
	int i, rk1, rk2;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "poset_with_group_action::is_contained" << endl;
	}
	if (f_vv) {
		cout << "set1: ";
		Lint_vec_print(cout, set1, sz1);
		cout << " ; ";
		cout << "set2: ";
		Lint_vec_print(cout, set2, sz2);
		cout << endl;
	}
	if (sz1 > sz2) {
		f_contained = false;
	}
	else {
		if (f_subspace_lattice) {
			int *B1, *B2;
			int dim = VS->dimension;

			B1 = NEW_int(sz1 * dim);
			B2 = NEW_int((sz1 + sz2) * dim);

			for (i = 0; i < sz1; i++) {
				unrank_point(B1 + i * dim, set1[i]);
			}
			for (i = 0; i < sz2; i++) {
				unrank_point(B2 + i * dim, set2[i]);
			}

			rk1 = VS->F->Linear_algebra->Gauss_easy(B1, sz1, dim);
			if (rk1 != sz1) {
				cout << "poset_with_group_action::is_contained "
						"rk1 != sz1" << endl;
				exit(1);
			}

			rk2 = VS->F->Linear_algebra->Gauss_easy(B2, sz2, dim);
			if (rk2 != sz2) {
				cout << "poset_with_group_action::is_contained "
						"rk2 != sz2" << endl;
				exit(1);
			}
			Int_vec_copy(B1,
					B2 + sz2 * dim,
					sz1 * dim);
			rk2 = VS->F->Linear_algebra->Gauss_easy(B2, sz1 + sz2, dim);
			if (rk2 > sz2) {
				f_contained = false;
			}
			else {
				f_contained = true;
			}

			FREE_int(B1);
			FREE_int(B2);
		}
		else {
			f_contained = Sorting.lint_vec_sort_and_test_if_contained(
					set1, sz1, set2, sz2);
		}
	}
	return f_contained;
}


void poset_with_group_action::invoke_early_test_func(
		long int *the_set, int lvl,
		long int *candidates,
		int nb_candidates,
		long int *good_candidates,
		int &nb_good_candidates,
		int verbose_level)
{
	early_test_func(
			the_set, lvl,
			candidates,
			nb_candidates,
			good_candidates,
			nb_good_candidates,
			verbose_level - 2);

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
	algebra::basic_algebra::rank_checker *rc;

	rc = (algebra::basic_algebra::rank_checker *) data;
	if (rc->check_rank_last_two_are_fixed(len,
		S, verbose_level - 1)) {
		return true;
	}
	else {
		return false;
	}

}

}}}


