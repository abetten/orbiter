// orbit_based_testing.cpp
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


orbit_based_testing::orbit_based_testing()
{
	Record_birth();
	int i;

	PC = NULL;
	max_depth = 0;
	local_S = NULL;
	nb_callback = 0;
	for (i = 0; i < MAX_CALLBACK; i++) {
		callback_testing[i] = NULL;
		callback_data[i] = NULL;
	}
	nb_callback_no_group = 0;
	for (i = 0; i < MAX_CALLBACK; i++) {
		callback_testing_no_group[i] = NULL;
		callback_data_no_group[i] = NULL;
	}
}

orbit_based_testing::~orbit_based_testing()
{
	Record_death();
	if (local_S) {
		FREE_lint(local_S);
	}
}

void orbit_based_testing::init(
		poset_classification *PC,
		int max_depth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_based_testing::init" << endl;
	}
	orbit_based_testing::PC = PC;
	orbit_based_testing::max_depth = max_depth;
	local_S = NEW_lint(max_depth);
	if (f_v) {
		cout << "orbit_based_testing::init done" << endl;
	}
}

void orbit_based_testing::add_callback(
		int (*func)(orbit_based_testing *Obt,
				long int *S, int len,
				void *data, int verbose_level),
		void *data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_based_testing::add_callback" << endl;
	}
	callback_testing[nb_callback] = func;
	callback_data[nb_callback] = data;
	nb_callback++;

	if (f_v) {
		cout << "orbit_based_testing::add_callback done" << endl;
	}
}

void orbit_based_testing::add_callback_no_group(
		void (*func)(long int *S, int len,
				long int *candidates, int nb_candidates,
				long int *good_candidates,
				int &nb_good_candidates,
				void *data, int verbose_level),
		void *data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_based_testing::add_callback_no_group" << endl;
	}
	callback_testing_no_group[nb_callback_no_group] = func;
	callback_data_no_group[nb_callback_no_group] = data;
	nb_callback_no_group++;

	if (f_v) {
		cout << "orbit_based_testing::add_callback_no_group done" << endl;
	}
}

void orbit_based_testing::early_test_func(
	long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "orbit_based_testing::early_test_func" << endl;
	}
	if (nb_callback) {
		if (f_v) {
			cout << "orbit_based_testing::early_test_func before early_test_func_by_using_group" << endl;
		}

		early_test_func_by_using_group(
			S, len,
			candidates, nb_candidates,
			good_candidates, nb_good_candidates,
			verbose_level);

		if (f_v) {
			cout << "orbit_based_testing::early_test_func after early_test_func_by_using_group" << endl;
		}
	}
	else if (nb_callback_no_group) {
		if (nb_callback_no_group > 1) {
			cout << "orbit_based_testing::early_test_func "
					"nb_callback_no_group > 1" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "orbit_based_testing::early_test_func nb_callback_no_group = " << nb_callback_no_group << endl;
		}
		if (f_v) {
			cout << "orbit_based_testing::early_test_func before (*callback_testing_no_group[0])" << endl;
		}
		(*callback_testing_no_group[0])(
					S, len,
					candidates, nb_candidates,
					good_candidates, nb_good_candidates,
					callback_data_no_group[0], verbose_level);
		if (f_v) {
			cout << "orbit_based_testing::early_test_func after (*callback_testing_no_group[0])" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "orbit_based_testing::early_test_func no test function" << endl;
		}
	}
	if (f_v) {
		cout << "orbit_based_testing::early_test_func done" << endl;
	}
}

void orbit_based_testing::early_test_func_by_using_group(
	long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "orbit_based_testing::early_test_func_by_using_group" << endl;
	}

	if (f_vv) {
		cout << "S=";
		Lint_vec_print(cout, S, len);
		cout << " testing " << nb_candidates << " candidates" << endl;
		//int_vec_print(cout, candidates, nb_candidates);
		//cout << endl;
	}

	if (len >= max_depth) {
		cout << "orbit_based_testing::early_test_func_by_using_group "
				"len >= max_depth" << endl;
		exit(1);
	}
	Lint_vec_copy(S, local_S, len);


	int i, j, node, f, l, nb_good_orbits;
	long int pt;
	poset_orbit_node *O;
	int f_orbit_is_good;
	//int s, a;

	node = PC->find_poset_orbit_node_for_set(len, local_S,
		false /* f_tolerant */, 0);
	O = PC->get_node(node);

	if (f_v) {
		cout << "orbit_based_testing::early_test_func_by_using_group for ";
		O->print_set(PC);
		cout << endl;
	}

	groups::schreier Schreier;

	Schreier.init(PC->get_A2(), verbose_level - 2);


#if 0
	Schreier.init_generators_by_hdl(
		O->nb_strong_generators, O->hdl_strong_generators, 0);
#else
	{
		vector<int> gen_hdl;

		O->get_strong_generators_handle(gen_hdl, verbose_level);
	}
#endif

	if (f_v) {
		cout << "orbit_based_testing::early_test_func_by_using_group "
				"before Schreier.compute_all_orbits_on_invariant_subset" << endl;
	}
	Schreier.orbits_on_invariant_subset_fast_lint(
		nb_candidates, candidates,
		0/*verbose_level*/);

	if (f_v) {
		cout << "orbit_based_testing::early_test_func_by_using_group "
				"after Schreier.compute_all_orbits_on_invariant_subset, we found "
		<< Schreier.Forest->nb_orbits << " orbits" << endl;
	}
	nb_good_candidates = 0;
	nb_good_orbits = 0;
	for (i = 0; i < Schreier.Forest->nb_orbits; i++) {
		f = Schreier.Forest->orbit_first[i];
		l = Schreier.Forest->orbit_len[i];
		pt = Schreier.Forest->orbit[f];
		local_S[len] = pt;
		f_orbit_is_good = true;
		if (f_v) {
			cout << "orbit_based_testing::early_test_func_by_using_group "
					"testing orbit " << i << " / " << Schreier.Forest->nb_orbits << " of length " << l << endl;
		}
		for (j = 0; j < nb_callback; j++) {
			if (f_v) {
				cout << "orbit_based_testing::early_test_func_by_using_group "
						"testing orbit " << i << " / " << Schreier.Forest->nb_orbits << " of length " << l
						<< " calling test function " << j << " / " << nb_callback << endl;
			}
			if (!(*callback_testing[j])(this,
					local_S, len + 1, callback_data[j], verbose_level - 1)) {
				f_orbit_is_good = false;
				break;
			}
		}
#if 0
		if (f_linear) {
			if (rc.check_rank_last_two_are_fixed(len + 1,
				local_S, verbose_level - 1)) {
				f_orbit_is_good = true;
			}
			else {
				f_orbit_is_good = false;
			}
		}
		else {
			f_orbit_is_good = true;
			for (s = 0; s < len; s++) {
				a = local_S[s];
				if (Hamming_distance(a, pt) < d) {
					f_orbit_is_good = false;
					break;
				}
			}
		}
#endif

		if (f_orbit_is_good) {
			for (j = 0; j < l; j++) {
				pt = Schreier.Forest->orbit[f + j];
				good_candidates[nb_good_candidates++] = pt;
			}
			nb_good_orbits++;
		}
	}

	Sorting.lint_vec_heapsort(good_candidates, nb_good_candidates);
	if (f_v) {
		cout << "orbit_based_testing::early_test_func_by_using_group "
			"after Schreier.compute_all_orbits_on_invariant_subset, "
			"we found "
			<< nb_good_candidates << " good candidates in "
			<< nb_good_orbits << " good orbits" << endl;
	}
}

}}}






