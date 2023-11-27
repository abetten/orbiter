/*
 * hall_system_classify.cpp
 *
 *  Created on: Nov 6, 2019
 *      Author: anton
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {


static void hall_system_print_set(
		std::ostream &ost, int len, long int *S, void *data);
static void hall_system_early_test_function(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);


hall_system_classify::hall_system_classify()
{
	//e = 0;
	n = 0;
	nm1 = 0;
	nb_pairs = 0;
	nb_pairs2 = 0;
	nb_blocks_overall = 0;
	nb_blocks_needed = 0;
	nb_orbits_needed = 0;
	depth = 0;
	N = 0;
	N0 = 0;
	row_sum = NULL;
	pair_covering = NULL;
	triples = NULL;
	A = NULL;
	A_on_triples = NULL;
	Strong_gens_Hall_reflection = NULL;
	Strong_gens_normalizer = NULL;
	S = NULL;

	//std::string prefix;
	//std::string fname_orbits_on_triples;
	Orbits_on_triples = NULL;
	A_on_orbits = NULL;
	f_play_it_safe = false;

	Control = NULL;
	Poset = NULL;
	PC = NULL;
}


hall_system_classify::~hall_system_classify()
{
	int verbose_level = 1;
	int f_v = (verbose_level >= 1);

	if (triples) {
		FREE_lint(triples);
	}
	if (row_sum) {
		FREE_int(row_sum);
	}
	if (pair_covering) {
		FREE_int(pair_covering);
	}
	if (A) {
		FREE_OBJECT(A);
	}
	if (A_on_triples) {
		FREE_OBJECT(A_on_triples);
	}
	if (Strong_gens_Hall_reflection) {
		FREE_OBJECT(Strong_gens_Hall_reflection);
	}
	if (Strong_gens_normalizer) {
		FREE_OBJECT(Strong_gens_normalizer);
	}
	if (Orbits_on_triples) {
		FREE_OBJECT(Orbits_on_triples);
	}
	if (A_on_orbits) {
		FREE_OBJECT(A_on_orbits);
	}
	if (S) {
		FREE_OBJECT(S);
	}
	if (Control) {
		FREE_OBJECT(Control);
	}
	if (Poset) {
		FREE_OBJECT(Poset);
	}
	if (PC) {
		FREE_OBJECT(PC);
	}
	if (f_v) {
		cout << "hall_system_classify::~hall_system_classify done" << endl;
	}
}


void hall_system_classify::init(
	int argc, const char **argv,
	int n, int depth,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "hall_system_classify::init" << endl;
		}
	hall_system_classify::n = n;
	hall_system_classify::depth = depth;
	//n = i_power_j(3, e);
	nm1 = n - 1;
	nb_pairs = nm1 / 2;
	nb_pairs2 = (nm1 * (nm1 - 1)) / 2;
	nb_blocks_overall = (n * (n - 1)) >> 1;
	nb_blocks_needed = nb_blocks_overall - (n - 1) / 2;
	nb_orbits_needed = nb_blocks_needed / 2;
	N0 = (nb_pairs * (nb_pairs - 1) * (nb_pairs - 2)) / 6 * 4;
	N = N0 * 2;

	prefix = "hall_" + std::to_string(n);

	if (f_v) {
		cout << "hall_system_classify::init n=" << n << endl;
		cout << "hall_system_classify::init nb_pairs=" << nb_pairs << endl;
		cout << "hall_system_classify::init nb_pairs2=" << nb_pairs2 << endl;
		cout << "hall_system_classify::init N0=" << N0 << endl;
		cout << "hall_system_classify::init N=" << N << endl;
		cout << "hall_system_classify::init nb_blocks_overall=" << nb_blocks_overall << endl;
		cout << "hall_system_classify::init nb_blocks_needed=" << nb_blocks_needed << endl;
		cout << "hall_system_classify::init nb_orbits_needed=" << nb_orbits_needed << endl;
	}

	triples = NEW_lint(N * 3);
	row_sum = NEW_int(nm1);
	pair_covering = NEW_int(nb_pairs2);

	if (f_v) {
		cout << "hall_system_classify::init "
				"before computing triples" << endl;
	}
	for (i = 0; i < N; i++) {
		if (f_v) {
			cout << "triple " << i << " / " << N << ":" << endl;
		}
		unrank_triple(triples + i * 3, i);
		if (f_v) {
			Lint_vec_print(cout, triples + i * 3, 3);
			cout << endl;
		}
		Sorting.lint_vec_heapsort(triples + i * 3, 3);
	}
	if (f_v) {
		cout << "sorted:" << endl;
		for (i = 0; i < N; i++) {
			Lint_vec_print(cout, triples + i * 3, 3);
			cout << endl;
		}
	}


	if (f_v) {
		cout << "hall_system_classify::init "
				"before A->init_permutation_group" << endl;
		}
	A = NEW_OBJECT(actions::action);
	int f_no_base = false;

	A->Known_groups->init_symmetric_group(
			nm1 /* degree */, f_no_base, verbose_level - 1);

	//A->init_permutation_group(nm1 /* degree */, verbose_level - 1);
	if (f_v) {
		cout << "hall_system_classify::init "
				"after A->init_permutation_group" << endl;
	}

	if (f_v) {
		cout << "hall_system_classify::init "
				"creating Strong_gens_Hall_reflection" << endl;
	}

	int degree; // nb_pairs * 2

	Strong_gens_Hall_reflection = NEW_OBJECT(groups::strong_generators);
	Strong_gens_Hall_reflection->init(A);
	Strong_gens_Hall_reflection->Hall_reflection(
		nb_pairs, degree, verbose_level);


	if (f_v) {
		cout << "hall_system_classify::init "
				"creating Strong_gens_normalizer" << endl;
	}

	Strong_gens_normalizer = NEW_OBJECT(groups::strong_generators);
	Strong_gens_normalizer->init(A);
	Strong_gens_normalizer->normalizer_of_a_Hall_reflection(
		nb_pairs, degree, verbose_level);

	if (f_v) {
		cout << "hall_system_classify::init "
				"before Strong_gens->create_sims" << endl;
	}
	S = Strong_gens_normalizer->create_sims(verbose_level - 1);
	if (f_v) {
		cout << "hall_system_classify::init "
				"after Strong_gens->create_sims" << endl;
	}

	//A_on_triples = NEW_OBJECT(actions::action);
	if (f_v) {
		cout << "hall_system_classify::init "
				"before A->Induced_action->induced_action_on_sets" << endl;
	}
	A_on_triples = A->Induced_action->induced_action_on_sets(
			S /*sims *old_G*/,
			N /* nb_sets*/, 3 /* set_size */, triples,
			false /* f_induce_action*/, verbose_level - 1);
	if (f_v) {
		cout << "hall_system_classify::init "
				"after A->Induced_action->induced_action_on_sets" << endl;
	}

	if (f_v) {
		cout << "hall_system_classify::init "
				"before orbits_on_triples" << endl;
	}
	orbits_on_triples(verbose_level);
	if (f_v) {
		cout << "hall_system_classify::init "
				"after orbits_on_triples" << endl;
	}


	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
	if (f_v) {
		cout << "hall_system_classify::init "
				"before Poset->init_subset_lattice" << endl;
	}
	Poset->init_subset_lattice(
			A, A_on_orbits,
			Strong_gens_normalizer,
			verbose_level);
	if (f_v) {
		cout << "hall_system_classify::init "
				"after Poset->init_subset_lattice" << endl;
	}
	Poset->add_testing_without_group(
			hall_system_early_test_function,
				this /* void *data */,
				verbose_level);

	Poset->f_print_function = true;
	Poset->print_function = hall_system_print_set;
	Poset->print_function_data = (void *) this;


	Control = NEW_OBJECT(poset_classification::poset_classification_control);
	PC = NEW_OBJECT(poset_classification::poset_classification);
	//PC->read_arguments(argc, argv, 0);
	if (f_v) {
		cout << "hall_system_classify::init "
				"before PC->initialize_and_allocate_root_node" << endl;
	}
	PC->initialize_and_allocate_root_node(
			Control, Poset, depth, verbose_level - 3);
	if (f_v) {
		cout << "hall_system_classify::init "
				"after PC->initialize_and_allocate_root_node" << endl;
	}


	int depth_completed;
	int f_use_invariant_subset_if_available = true;
	int f_debug = false;
	int schreier_depth = INT_MAX;
	int t0;
	orbiter_kernel_system::os_interface Os;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "hall_system_classify::init_generator "
				"before PC->main" << endl;
	}
	depth_completed = PC->main(t0, schreier_depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level);
	if (f_v) {
		cout << "hall_system_classify::init_generator "
				"after PC->main" << endl;
	}
	cout << "hall_system_classify returns "
			"depth_completed=" << depth_completed << endl;

	if (f_v) {
		cout << "hall_system_classify::init done" << endl;
	}
}

void hall_system_classify::orbits_on_triples(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "hall_system_classify::orbits_on_triples" << endl;
	}

	fname_orbits_on_triples = prefix + "_orbits_on_triples.bin";

	if (Fio.file_size(fname_orbits_on_triples) > 0) {


		if (f_v) {
			cout << "hall_system_classify::orbits_on_triples "
					"reading orbits from file "
					<< fname_orbits_on_triples << endl;
		}

		Orbits_on_triples = NEW_OBJECT(groups::schreier);

		Orbits_on_triples->init(A_on_triples, verbose_level - 2);
		Orbits_on_triples->initialize_tables();
		Orbits_on_triples->init_generators(
				*Strong_gens_Hall_reflection->gens, verbose_level - 2);
		{
			ifstream fp(fname_orbits_on_triples);
			Orbits_on_triples->read_from_file_binary(fp, verbose_level);
		}
		if (f_v) {
			cout << "hall_system_classify::orbits_on_triples "
					"read orbits from file "
					<< fname_orbits_on_triples << endl;
		}
	}
	else {

		if (f_v) {
			cout << "hall_system_classify::orbits_on_triples "
					"computing orbits of the selected group" << endl;
		}

		Orbits_on_triples =
				Strong_gens_Hall_reflection->compute_all_point_orbits_schreier(
				A_on_triples, 0 /*verbose_level*/);

		if (f_v) {
			cout << "hall_system_classify::orbits_on_triples "
					"computing orbits done" << endl;
			cout << "We found " << Orbits_on_triples->nb_orbits
					<< " orbits of the selected group on lines" << endl;
		}


		{
			ofstream fp(fname_orbits_on_triples);
			Orbits_on_triples->write_to_file_binary(fp, verbose_level);
		}
		cout << "Written file " << fname_orbits_on_triples << " of size "
				<< Fio.file_size(fname_orbits_on_triples) << endl;
	}

	if (f_v) {
		cout << "Orbits_on_triples:" << endl;
		Orbits_on_triples->print(cout);
	}
	//A_on_orbits = NEW_OBJECT(actions::action);
	A_on_orbits = A_on_triples->Induced_action->induced_action_on_orbits(
			Orbits_on_triples, f_play_it_safe, verbose_level);

	if (f_v) {
		cout << "hall_system_classify::orbits_on_triples "
				"created action on orbits of degree "
				<< A_on_orbits->degree << endl;
	}



	if (f_v) {
		cout << "hall_system_classify::orbits_on_triples done" << endl;
	}
}



void hall_system_classify::print(
		std::ostream &ost, long int *S, int len)
{
	int i;
	int orb, f, l, j, t, a;
	long int T[3];

	for (i = 0; i < len; i++) {
		ost << S[i] << " ";
	}
	ost << endl;
	for (i = 0; i < len; i++) {

		orb = S[i];

		f = Orbits_on_triples->orbit_first[orb];
		l = Orbits_on_triples->orbit_len[orb];
		for (j = 0; j < l; j++ ) {
			t = Orbits_on_triples->orbit[f + j];
			unrank_triple(T, t);
			for (a = 0; a < 3; a++) {
				cout << T[a];
				if (a < 3 - 1) {
					cout << ", ";
				}
			} // next a
			if (j < l) {
				cout << "; ";
			}
		} // next j
		cout << endl;
	} // next i
}

void hall_system_classify::unrank_triple(
		long int *T, int rk)
{
	int a, b, i;
	int set[3];
	int binary[3];
	combinatorics::combinatorics_domain Combi;

	b = rk % 8;
	a = rk / 8;
	Combi.unrank_k_subset(a, set, nb_pairs, 3);
	binary[0] = b % 2;
	b >>= 1;
	binary[1] = b % 2;
	b >>= 1;
	binary[2] = b % 2;
	for (i = 0; i < 3; i++) {
		T[i] = 2 * set[i] + binary[i];
	}
}

void hall_system_classify::unrank_triple_pair(
		long int *T1, long int *T2, int rk)
{
	int a, b, i;
	int set[3];
	int binary[3];
	combinatorics::combinatorics_domain Combi;

	b = rk % 8;
	a = rk / 8;
	Combi.unrank_k_subset(a, set, nb_pairs, 3);
	binary[0] = b % 2;
	b >>= 1;
	binary[1] = b % 2;
	b >>= 1;
	binary[2] = b % 2;
	for (i = 0; i < 3; i++) {
		T1[i] = 2 * set[i] + binary[i];
		T2[i] = 2 * set[i] + 1 - binary[i];
	}
}

void hall_system_classify::early_test_func(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	//verbose_level = 10;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, a, b, p;
	int orb, f, l, t, h;
	long int T[3];
	int f_OK;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "hall_system_classify::early_test_func checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size " << nb_candidates << ":" << endl;
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
	}


	Int_vec_zero(row_sum, nm1);
	Int_vec_zero(pair_covering, nb_pairs2);


	for (i = 0; i < len; i++) {

		orb = S[i];

		f = Orbits_on_triples->orbit_first[orb];
		l = Orbits_on_triples->orbit_len[orb];
		for (j = 0; j < l; j++ ) {
			t = Orbits_on_triples->orbit[f + j];
			unrank_triple(T, t);
			for (a = 0; a < 3; a++) {
				row_sum[T[a]]++;
				for (b = a + 1; b < 3; b++) {
					p = Combi.ij2k(T[a], T[b], nm1);
					pair_covering[p] = true;
				} // next b
			} // next a
		} // next j
	} // next i
	if (f_vv) {
		cout << "hall_system::early_test_func "
				"pair_covering before testing:" << endl;
		cout << "row_sum: " << endl;
		Int_vec_print(cout, row_sum, nm1);
		cout << endl;
		cout << "pair_covering: " << endl;
		Int_vec_print(cout, pair_covering, nb_pairs2);
		cout << endl;
	}


	nb_good_candidates = 0;

	for (j = 0; j < nb_candidates; j++) {


		orb = candidates[j];

		f = Orbits_on_triples->orbit_first[orb];
		l = Orbits_on_triples->orbit_len[orb];

		if (f_vv) {
			cout << "Testing candidate " << j << " = "
					<< candidates[j] << " which is ";
			for (h = 0; h < l; h++ ) {
				t = Orbits_on_triples->orbit[f + h];
				unrank_triple(T, t);
				Lint_vec_print(cout, T, 3);
				cout << ", ";
			}
			cout << endl;

		}

		f_OK = true;

		for (h = 0; h < l; h++ ) {
			t = Orbits_on_triples->orbit[f + h];
			unrank_triple(T, t);
			for (a = 0; a < 3; a++) {
				if (row_sum[T[a]] == nb_pairs - 1) {
					if (f_v) {
						cout << "bad because of row sum "
								"in row " << T[a] << endl;
					}
					f_OK = false;
					break;
				} // if
			} // next a
			if (!f_OK) {
				break;
			}
		} // next h

		if (f_OK) {
			for (h = 0; h < l; h++ ) {
				t = Orbits_on_triples->orbit[f + h];
				unrank_triple(T, t);
				for (a = 0; a < 3; a++) {
					for (b = a + 1; b < 3; b++) {
						p = Combi.ij2k(T[a], T[b], nm1);
						if (pair_covering[p]) {
							if (f_v) {
								cout << "bad because of pair covering in pair "
									<< T[a] << "," << T[b] << "=" << p << endl;
							}
							f_OK = false;
							break;
							}
						}
					if (!f_OK) {
						break;
					}
				} // next a
				if (!f_OK) {
					break;
				}
			} // next h
		}


		if (f_OK) {
			if (f_vv) {
				cout << "Testing candidate " << j << " = "
						<< candidates[j] << " is good" << endl;
				}
			good_candidates[nb_good_candidates++] = candidates[j];
			}
		}
}


// #############################################################################
// global functions:
// #############################################################################




static void hall_system_print_set(
		std::ostream &ost, int len, long int *S, void *data)
{
	hall_system_classify *H = (hall_system_classify *) data;

	//print_vector(ost, S, len);
	H->print(ost, S, len);
}


static void hall_system_early_test_function(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	hall_system_classify *H = (hall_system_classify *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hall_system_early_test_function for set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
	}
	H->early_test_func(S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
		verbose_level - 2);
	if (f_v) {
		cout << "hall_system_early_test_function done" << endl;
	}
}



}}}

