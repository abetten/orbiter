/*
 * compute_stabilizer.cpp
 *
 *  Created on: Apr 24, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



compute_stabilizer::compute_stabilizer()
{
	A = NULL;
	A2 = NULL;
	Stab = NULL;
	A_on_the_set = NULL;


	null1();

	Strong_gens_G = NULL;

	Stab_orbits = NULL;
	orbit_count1 = NULL;
	orbit_count2 = NULL;


	interesting_orbits = NULL;
	interesting_points = NULL;
	interesting_orbit_first = NULL;
	interesting_orbit_len = NULL;


	A_induced = NULL;
	Aut = NULL;
	Aut_original = NULL;

	transporter_witness = NULL;
	transporter1 = NULL;
	T1 = NULL;
	T1v = NULL;
	transporter2 = NULL;
	T2 = NULL;

	U = NULL;

}

compute_stabilizer::~compute_stabilizer()
{
	free1();

	if (Stab) {
		FREE_OBJECT(Stab);
		}
	if (A_on_the_set) {
		FREE_OBJECT(A_on_the_set);
		}
	if (Strong_gens_G) {
		FREE_OBJECT(Strong_gens_G);
		}
	if (Stab_orbits) {
		FREE_OBJECT(Stab_orbits);
		}

	if (orbit_count1) {
		FREE_int(orbit_count1);
		FREE_int(orbit_count2);
		}

	if (interesting_points) {
		FREE_lint(interesting_points);
		FREE_int(interesting_orbits);
		FREE_int(interesting_orbit_first);
		FREE_int(interesting_orbit_len);
		}

	if (A_induced) {
		FREE_OBJECT(A_induced);
		}
	if (Aut) {
		FREE_OBJECT(Aut);
		}
	if (Aut_original) {
		FREE_OBJECT(Aut_original);
		}

	if (transporter_witness) {
		FREE_int(transporter_witness);
		FREE_int(transporter1);
		FREE_int(transporter2);
		FREE_int(T1);
		FREE_int(T1v);
		FREE_int(T2);
		}

	if (U) {
		FREE_OBJECT(U);
		}

}


void compute_stabilizer::init(long int *the_set, int set_size,
		poset_classification *gen, action *A, action *A2,
		int level, int interesting_orbit,
		int nb_interesting_subsets, long int *interesting_subsets,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_stabilizer::init" << endl;
		cout << "interesting_orbit = " << interesting_orbit << endl;
		cout << "nb_interesting_subsets = " << nb_interesting_subsets << endl;
		}
	compute_stabilizer::the_set = the_set;
	compute_stabilizer::set_size = set_size;
	compute_stabilizer::gen = gen;
	compute_stabilizer::A = A;
	compute_stabilizer::A2 = A2;
	compute_stabilizer::level = level;
	compute_stabilizer::interesting_orbit = interesting_orbit;
	compute_stabilizer::nb_interesting_subsets = nb_interesting_subsets;
	compute_stabilizer::interesting_subsets = interesting_subsets;

	Stab = NEW_OBJECT(sims);
	Stab->init(A, 0 /* verbose_level */);
	Stab->init_trivial_group(verbose_level - 1);

	init_U(verbose_level);


	A_on_the_set = A2->restricted_action(the_set, set_size, 0/*verbose_level*/);


	reduced_set_size = set_size - level;
	first_at_level = gen->first_node_at_level(level);

	allocate1();

	gen->get_stabilizer_generators(
			Strong_gens_G,
			level, interesting_orbit, 0 /* verbose_level*/);

	if (f_v) {
		cout << "compute_stabilizer::init the group has order " << go_G << endl;
		}


	if (f_v) {
		cout << "compute_stabilizer::init before compute_orbits" << endl;
		}
	compute_orbits(verbose_level);
	if (f_v) {
		cout << "compute_stabilizer::init after compute_orbits" << endl;
		}

	if (f_v) {
		cout << "compute_stabilizer::init before restricted_action" << endl;
		}
	restricted_action(verbose_level);
	if (f_v) {
		cout << "compute_stabilizer::init after restricted_action" << endl;
		}


	if (f_v) {
		cout << "compute_stabilizer::init before main_loop" << endl;
		}
	main_loop(verbose_level);
	if (f_v) {
		cout << "compute_stabilizer::init after main_loop" << endl;
		cout << "compute_stabilizer::init backtrack_nodes_first_time = " << backtrack_nodes_first_time << endl;
		cout << "compute_stabilizer::init backtrack_nodes_total_in_loop = " << backtrack_nodes_total_in_loop << endl;
		}

	if (f_v) {
		cout << "compute_stabilizer::init done" << endl;
		}
}


void compute_stabilizer::init_U(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_stabilizer::init_U before U->init" << endl;
		}
	U = NEW_OBJECT(union_find_on_k_subsets);

	U->init(A2, Stab,
		the_set, set_size, level,
		interesting_subsets, nb_interesting_subsets,
		verbose_level);


	if (f_v) {
		cout << "compute_stabilizer::init_U after U->init" << endl;
		}
}

void compute_stabilizer::compute_orbits(int verbose_level)
// uses Strong_gens_G to compute orbits on points in action A2
{
	int f_v = (verbose_level >= 1);
	int i, j, k, f, l, ii, jj, a;

	if (f_v) {
		cout << "compute_stabilizer::compute_orbits computing orbits on points" << endl;
		}
	Stab_orbits = Strong_gens_G->orbits_on_points_schreier(A2, 0 /*verbose_level*/);
	if (f_v) {
		cout << "compute_stabilizer::compute_orbits computing orbits on points done, we found " << Stab_orbits->nb_orbits << " orbits" << endl;
		}

	nb_orbits = Stab_orbits->nb_orbits;
	orbit_count1 = NEW_int(nb_orbits);
	orbit_count2 = NEW_int(nb_orbits);
	Orbiter->Int_vec.zero(orbit_count1, nb_orbits);


	if (f_v) {
		cout << "compute_stabilizer::compute_orbits mapping the first subset to its canonical form" << endl;
		}

	map_the_first_set(0 /* cnt */, verbose_level);
		// reduced_set1 has size set_size - level (=reduced_set_size)
		// compute orbit_count1[] for reduced_set1[].
		// orbit_count1[i] is the number of points from reduced_set1[] contained in orbit i



	// An orbit is interesting if it contains points from reduced_set1[],
	// i.e., orbit i is interesting if orbit_count1[i] is not equal to zero
	// Let interesting_orbits[nb_interesting_orbits] be the list of interesting orbits

	nb_interesting_orbits = 0;
	for (i = 0; i < nb_orbits; i++) {
		if (orbit_count1[i] == 0 /*|| orbit_count1[i] == Stab_orbits->orbit_len[i]*/) {
			continue;
			}
		nb_interesting_orbits++;
		}
	if (FALSE) {
		for (i = 0; i < nb_orbits; i++) {
			if (orbit_count1[i] == 0 /*|| orbit_count1[i] == Stab_orbits->orbit_len[i]*/) {
				continue;
				}
			cout << i << "^" << orbit_count1[i] << " ";
			}
		cout << endl;
		cout << "compute_stabilizer::compute_orbits nb_interesting_points = " << nb_interesting_points << endl;
		}

	interesting_orbits = NEW_int(nb_interesting_orbits);
	j = 0;
	for (i = 0; i < nb_orbits; i++) {
		if (orbit_count1[i] == 0 /*|| orbit_count1[i] == Stab_orbits->orbit_len[i]*/) {
			continue;
			}
		interesting_orbits[j++] = i;
		}




	// compute the set interesting_points[]
	// These are the points that lie in orbits
	// that intersect the set of interesting_points:

	nb_interesting_points = 0;
	for (k = 0; k < nb_interesting_orbits; k++) {
		i = interesting_orbits[k];
		nb_interesting_points += Stab_orbits->orbit_len[i];
		}

	if (f_v) {
		cout << "compute_stabilizer::compute_orbits reduced_set_size = " << reduced_set_size << endl;
		cout << "compute_stabilizer::compute_orbits nb_interesting_orbits = " << nb_interesting_orbits << endl;
		cout << "compute_stabilizer::compute_orbits nb_interesting_points = " << nb_interesting_points << endl;
		}


	interesting_points = NEW_lint(nb_interesting_points);

	interesting_orbit_first = NEW_int(nb_interesting_orbits);
	interesting_orbit_len = NEW_int(nb_interesting_orbits);


	j = 0;
	for (k = 0; k < nb_interesting_orbits; k++) {
		i = interesting_orbits[k];
		f = Stab_orbits->orbit_first[i];
		l = Stab_orbits->orbit_len[i];
		interesting_orbit_first[k] = j;
		interesting_orbit_len[k] = l;
		for (ii = 0; ii < l; ii++) {
			jj = f + ii;
			interesting_points[j++] = Stab_orbits->orbit[jj];
			}
		}

	if (f_v) {
		cout << "compute_stabilizer::compute_orbits interesting orbits: " << endl;
		if (nb_interesting_orbits < 10) {
			for (k = 0; k < nb_interesting_orbits; k++) {
				i = interesting_orbits[k];
				f = Stab_orbits->orbit_first[i];
				l = Stab_orbits->orbit_len[i];
				a = orbit_count1[i];
				cout << i << " : " << a << " / " << l << endl;
				}
			}
		else {
			cout << "too many orbits to print" << endl;
			}
		}
	if (FALSE) {
		cout << "compute_stabilizer::compute_orbits interesting_points:" << endl;
		Orbiter->Lint_vec.print(cout, interesting_points, nb_interesting_points);
		cout << endl;
		cout << "by interesting orbits:" << endl;
		cout << "k : interesting_orbits[k] : interesting_orbit_first[k] : interesting_orbit_len[k]" << endl;
		for (k = 0; k < nb_interesting_orbits; k++) {
			cout << setw(3) << k << " : "
				<< setw(4) << interesting_orbits[k] << " : "
				<< setw(4) << interesting_orbit_first[k] << " : "
				<< setw(4) << interesting_orbit_len[k] << endl;
			}
		}

	// Let reduced_set1_new_labels[] be the set reduced_set1[] in the restricted action,
	// and let the set be ordered increasingly:

	for (i = 0; i < reduced_set_size; i++) {
		a = reduced_set1[i];
		for (j = 0; j < nb_interesting_points; j++) {
			if (interesting_points[j] == a) {
				reduced_set1_new_labels[i] = j;
				break;
				}
			}
		if (j == nb_interesting_points) {
			cout << "did not find point " << a << endl;
			exit(1);
			}
		}
	if (FALSE) {
		cout << "compute_stabilizer::compute_orbits reduced_set1_new_labels:" << endl;
		Orbiter->Lint_vec.print(cout, reduced_set1_new_labels, reduced_set_size);
		cout << endl;
		}


	sorting Sorting;

	Sorting.lint_vec_heapsort(reduced_set1_new_labels, reduced_set_size);
	if (FALSE) {
		cout << "compute_stabilizer::compute_orbits compute_stabilizer sorted:" << endl;
		Orbiter->Lint_vec.print(cout, reduced_set1_new_labels, reduced_set_size);
		cout << endl;
		}
	if (f_v) {
		cout << "compute_stabilizer::compute_orbits computing orbits on points done" << endl;
		}
}


void compute_stabilizer::restricted_action(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nodes;
	longinteger_domain D;

	if (f_v) {
		cout << "compute_stabilizer::restricted_action" << endl;
		}
	// Compute the restricted action on the set of
	// interesting points and call it A_induced :
	// Determine the kernel
	if (f_v) {
		cout << "compute_stabilizer::restricted_action computing induced action by restriction" << endl;
		}
#if 0
	A_induced = NEW_OBJECT(action);
	A_induced->induced_action_by_restriction(*A2,
		TRUE, G->S,
		nb_interesting_points, interesting_points,
		0 /*verbose_level - 3*/);
#endif

	A_induced = A2->restricted_action(interesting_points, nb_interesting_points, 0/*verbose_level*/);
	A_induced->group_order(induced_go);
	A_induced->Kernel->group_order(K_go);
	if (f_v) {
		cout << "compute_stabilizer::restricted_action induced action by restriction: group order = " << induced_go << endl;
		cout << "kernel group order = " << K_go << endl;
		//cout << "strong generators for induced action:" << endl;
		//A_induced.strong_generators->print_as_permutation(cout);
		//cout << "strong generators for kernel:" << endl;
		//A_induced.Kernel->gens.print_as_permutation(cout);
		}

	transporter_witness = NEW_int(A_induced->elt_size_in_int);
	transporter1 = NEW_int(A_induced->elt_size_in_int);
	transporter2 = NEW_int(A_induced->elt_size_in_int);
	T1 = NEW_int(A->elt_size_in_int);
	T1v = NEW_int(A->elt_size_in_int);
	T2 = NEW_int(A->elt_size_in_int);

	K = NEW_OBJECT(sims);
	Kernel_original = NEW_OBJECT(sims);

	K->init(A, 0 /* verbose_level */);
	K->init_trivial_group(verbose_level - 1);

	A_induced->Kernel->group_order(K_go);
	if (f_v) {
		cout << "compute_stabilizer::restricted_action kernel has order " << K_go << endl;
		}

	// conjugate the Kernel so that it is a subgroup of
	// the stabilizer of the set the_set[] that we wanted to stabilizer originally:
	// remember that elt1 is the transporter that was computed in map_it() above

	Kernel_original->conjugate(A_induced->Kernel->A, A_induced->Kernel, elt1,
		FALSE, 0 /*verbose_level - 3*/);
	Kernel_original->group_order(K_go);
	if (f_v) {
		cout << "compute_stabilizer::restricted_action after conjugation, kernel has order " << K_go << endl;
		}

	if (f_v) {
		cout << "compute_stabilizer::restricted_action adding kernel of order " << K_go
			<< " to the stabilizer (in action " << Stab->A->label << ")" << endl;
		}
	Stab->build_up_group_random_process(K, Kernel_original,
		K_go, FALSE, NULL, 0 /*verbose_level - 3*/);
	Stab->group_order(stab_order);

	if (f_v) {
		cout << "compute_stabilizer::restricted_action kernel of action on the set has been added to stabilizer" << endl;
		cout << "compute_stabilizer::restricted_action current stabilizer order " << stab_order << endl;
		}
#if 0
	if (!Stab->test_if_in_set_stabilizer(A, the_set, set_size, verbose_level)) {
		cout << "set stabilizer does not stabilize" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "set stabilizer of order " << stab_order << " is OK" << endl;
		}
#endif
	// here we need the stabilizer of the set the_set[]
	// and the kernel of the action has to go into Stab first.



	Aut = NEW_OBJECT(sims);
	Aut_original = NEW_OBJECT(sims);


	// computes the stabilizer of reduced_set[] in the stabilizer
	// of the k-subset and in the induced action:
	if (f_v) {
		cout << "compute_stabilizer::restricted_action before A_induced.make_canonical" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	A_induced->make_canonical(
		reduced_set_size, reduced_set1_new_labels,
		canonical_set1, transporter1, nodes,
		TRUE, Aut,
		verbose_level /*- 3*/);

	if (f_v) {
		cout << "compute_stabilizer::restricted_action after A_induced.make_canonical" << endl;
		}

	// Now, Aut is the stabilizer of canonical_set1 in the induced action (A_induced)

	backtrack_nodes_first_time = nodes;


	Aut->group_order(ago);
	if (f_v) {
		cout << "compute_stabilizer::restricted_action backtrack_nodes=" << nodes << endl;
#if 0
		cout << "canonical set1: ";
		INT_vec_print(cout, canonical_set1, reduced_set_size);
		cout << endl;
#endif
		cout << "compute_stabilizer::restricted_action automorphism group order " << ago << endl;
		}
	if (FALSE) {
		cout << "transporter1:" << endl;
		A_induced->element_print(transporter1, cout);
		cout << endl;
		A_induced->element_print_as_permutation(transporter1, cout);
		cout << endl;
		}

#if 0
	if (f_v) {
		cout << "testing stabilizer of canonical set" << endl;
		if (!Aut.test_if_in_set_stabilizer(&A_induced, canonical_set1, reduced_set_size, verbose_level)) {
			cout << "set stabilizer does not stabilize" << endl;
			exit(1);
			}
		}
#endif

	A->mult(elt1, transporter1, T1);

	if (FALSE) {
		cout << "T1:" << endl;
		A->element_print(T1, cout);
		cout << endl;
		}
	A->element_invert(T1, T1v, FALSE);

	// T1 := elt1 * transporter1
	// moves the_set to the canonical set.

	Aut->group_order(ago);
	Aut_original->conjugate(A /*Aut.A */, Aut, T1,
		TRUE /* f_overshooting_OK */, 0 /*verbose_level - 3*/);
	Aut_original->group_order(ago1);
	if (f_v) {
		cout << "compute_stabilizer::restricted_action after conjugation, group in action " << Aut_original->A->label << endl;
		cout << "compute_stabilizer::restricted_action automorphism group order before = " << ago << endl;
		cout << "compute_stabilizer::restricted_action automorphism group order after = " << ago1 << endl;
		}

	D.mult(K_go, ago, target_go);
	if (f_v) {
		cout << "compute_stabilizer::restricted_action target_go=" << target_go << endl;
		cout << "compute_stabilizer::restricted_action adding automorphisms to set-stabilizer" << endl;
		}
	Stab->build_up_group_random_process(K, Aut_original,
		target_go, FALSE, NULL, 0 /*verbose_level - 3*/);
	Stab->group_order(stab_order);
	if (f_v) {
		cout << "compute_stabilizer::restricted_action set stabilizer is added to stabilizer" << endl;
		cout << "compute_stabilizer::restricted_action current stabilizer order " << stab_order << endl;
		}
#if 0
	if (!Stab->test_if_in_set_stabilizer(A, the_set, set_size, verbose_level)) {
		cout << "set stabilizer does not stabilize" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "set stabilizer of order " << stab_order << " is OK" << endl;
		}
#endif

	//A->element_mult(elt1, transporter1, Elt1, FALSE);
	//A->element_invert(Elt1, Elt1_inv, FALSE);
	if (f_v) {
		cout << "compute_stabilizer::restricted_action done" << endl;
		}
}


void compute_stabilizer::main_loop(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_v4 = (verbose_level >= 4);
	int cnt;

	if (f_v) {
		cout << "compute_stabilizer::main_loop" << endl;
		}
	// In the second step, we consider all possible images for the chosen k-subset
	// and try to pick up coset representatives fro the subgroup in the
	// full set stabilizer:


	nb_times_orbit_count_does_not_match_up = 0;
	backtrack_nodes_total_in_loop = 0;


	for (cnt = 1; cnt < nb_interesting_subsets; cnt++) {

		if (U->is_minimal(cnt, 0 /* verbose_level */)) {
			main_loop_handle_case(cnt, verbose_level);
			}

		}

	if (f_v) {
		cout << "compute_stabilizer::main_loop nb_interesting_subsets = " << nb_interesting_subsets;
		cout << " nb_times_orbit_count_does_not_match_up = " << nb_times_orbit_count_does_not_match_up << endl;
		}
	if (f_v) {
		cout << "compute_stabilizer::main_loop done" << endl;
		}
}

void compute_stabilizer::main_loop_handle_case(int cnt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v4 = (verbose_level >= 4);
	int cmp;

	if (f_v || ((cnt % 10) == 0)) {
		cout << "compute_stabilizer::main_loop_handle_case STABILIZER loop " << cnt << " / " << nb_interesting_subsets << " stab_order=" << stab_order << endl;
		}

	map_the_second_set(cnt, verbose_level);

	if (f_v4) {
		print_orbit_count(TRUE /* f_both */);
		}


	if (!check_orbit_count()) {
		nb_times_orbit_count_does_not_match_up++;
		if (f_vv) {
			cout << "compute_stabilizer::main_loop_handle_case STABILIZER loop " << cnt << " / " << nb_interesting_subsets << " orbit count does not match up, we skip, nb_times_orbit_count_does_not_match_up=" << nb_times_orbit_count_does_not_match_up << endl;
			}
		return;
		}


	if (!compute_second_reduced_set()) {
		if (f_vv) {
			cout << "compute_stabilizer::main_loop_handle_case did not find point, we skip" << endl;
			}
		return;
		}




	if (f_vv) {
		cout << "compute_stabilizer::main_loop_handle_case before make_canonical_second_set" << endl;
		}

	make_canonical_second_set(verbose_level);


	cmp = lint_vec_compare(canonical_set1, canonical_set2, reduced_set_size);
	if (FALSE) {
		//INT_vec_print(cout, canonical_set2, reduced_set_size);
		cout << "comparing the two canonical vectors cmp=" << cmp << endl;
		}

	if (cmp != 0) {
		if (f_vv) {
			cout << "compute_stabilizer::main_loop_handle_case the two canonical sets are not the same, so we skip" << endl;
			}
		return;
		}
	else {
		if (f_v) {
			cout << "compute_stabilizer::main_loop_handle_case the two canonical sets are the same, so we are looking for an automorphism" << endl;
			}
		}


	retrieve_automorphism(verbose_level);


		// new automorphism is now in new_automorphism


	add_automorphism(verbose_level);


	update_stabilizer(verbose_level);



	if (f_v || ((cnt % 10) == 0)) {
		cout << "compute_stabilizer::main_loop_handle_case STABILIZER loop " << cnt << " / " << nb_interesting_subsets << " stab_order=" << stab_order << " done" << endl;
		}
}

void compute_stabilizer::map_the_first_set(int cnt, int verbose_level)
{
	sorting Sorting;

	gen->map_to_canonical_k_subset(the_set, set_size,
			level /* subset_size */, interesting_subsets[cnt],
			reduced_set1, elt1 /*transporter */, local_idx1, verbose_level - 4);
		// reduced_set1 has size set_size - level (=reduced_set_size)


	Sorting.lint_vec_heapsort(reduced_set1, reduced_set_size);
	if (FALSE) {
		cout << setw(4) << 0 << " : " << setw(4) << interesting_subsets[0] << " : ";
		Orbiter->Lint_vec.print(cout, reduced_set1, reduced_set_size);
		cout << endl;
		}
	if (FALSE) {
		cout << "elt1:" << endl;
		A->element_print(elt1, cout);
		cout << endl;
		}


	// compute orbit_count1[] for reduced_set1[].
	// orbit_count1[i] is the number of points from reduced_set1[] contained in orbit i

	Stab_orbits->compute_orbit_statistic_lint(reduced_set1, reduced_set_size, orbit_count1, verbose_level - 1);
}


void compute_stabilizer::map_the_second_set(int cnt, int verbose_level)
{
	sorting Sorting;

	gen->map_to_canonical_k_subset(the_set, set_size,
			level /* subset_size */, interesting_subsets[cnt],
			reduced_set2, elt2 /*transporter */, local_idx2, verbose_level - 4);
		// reduced_set2 has size set_size - level (=reduced_set_size)


	Sorting.lint_vec_heapsort(reduced_set2, reduced_set_size);
	if (FALSE) {
		cout << "compute_stabilizer::map_the_second_set STABILIZER "
				<< setw(4) << cnt << " : " << setw(4) << interesting_subsets[cnt] << " : ";
		Orbiter->Lint_vec.print(cout, reduced_set2, reduced_set_size);
		cout << endl;
		}

	Stab_orbits->compute_orbit_statistic_lint(reduced_set2, reduced_set_size, orbit_count2, verbose_level - 1);
}

void compute_stabilizer::update_stabilizer(int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	longinteger_domain D;
	int cmp;

	Stab->group_order(new_stab_order);
	cmp = D.compare_unsigned(new_stab_order, stab_order);
	if (cmp) {
		cout << "compute_stabilizer::update_stabilizer new stabilizer order is " << new_stab_order << endl;
		new_stab_order.assign_to(stab_order);


		strong_generators *Strong_gens;

		Strong_gens = NEW_OBJECT(strong_generators);
		Strong_gens->init_from_sims(Stab, 0);
		Strong_gens->print_generators(cout);
		FREE_OBJECT(Strong_gens);

#if 0
		char fname[1000];
		sprintf(fname, "stab_order_%ld.txt", new_stab_order.as_INT());
		Stab->write_sgs(fname, 0 /*verbose_level */);
		cout << "Written file " << fname << " of size " << file_size(fname) << endl;
#endif

		FREE_OBJECT(U);

		init_U(verbose_level);
		}

#if 0
	cout << "stabilizer transversal length:" << endl;
	Stab->print_transversal_lengths();
	cout << endl;
	if (!Stab->test_if_in_set_stabilizer(A, the_set, set_size, verbose_level)) {
		cout << "set stabilizer does not stabilize" << endl;
		exit(1);
		}
#endif
}


void compute_stabilizer::add_automorphism(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_v4 = (verbose_level >= 4);
	int drop_out_level, image, f_added;

	//cout << "stabilizer transversal length:" << endl;
	//Stab->print_transversal_lengths();
	//cout << endl;

	if (Stab->strip(new_automorphism, Elt4, drop_out_level, image, 0/*verbose_level - 3*/)) {
		if (f_v4) {
			cout << "compute_stabilizer::main_loop_handle_case element strips through" << endl;
			if (FALSE) {
				cout << "compute_stabilizer residue:" << endl;
				A->element_print(Elt4, cout);
				cout << endl;
				}
			}
		f_added = FALSE;
		//Stab->closure_group(2000, verbose_level - 1);
		}
	else {
		f_added = TRUE;
		if (f_v4) {
			cout << "compute_stabilizer::main_loop_handle_case element needs to be inserted at level = "
				<< drop_out_level << " with image " << image << endl;
			if (FALSE) {
				A->element_print(Elt4, cout);
				cout  << endl;
				}
			}
		if (!A2->check_if_in_set_stabilizer(Elt4, set_size, the_set, 0/*verbose_level*/)) {
			cout << "compute_stabilizer::main_loop_handle_case residue does not stabilize original set" << endl;
			exit(1);
			}
		Stab->add_generator_at_level(Elt4, drop_out_level, 0/*verbose_level - 3*/);
		if (f_v) {
			cout << "compute_stabilizer::main_loop_handle_case calling closure_group" << endl;
			}
		Stab->closure_group(2000, 0 /*verbose_level - 1*/);
		}
}

void compute_stabilizer::retrieve_automorphism(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	A->element_mult(T2, T1v, new_automorphism, FALSE);
	if (f_v) {
		cout << "compute_stabilizer::retrieve_automorphism found automorphism" << endl;
		}
	if (FALSE) {
		cout << "compute_stabilizer::retrieve_automorphism automorphism:" << endl;
		A->element_print(new_automorphism, cout);
		cout << endl;
		}
	if (!A2->check_if_in_set_stabilizer(new_automorphism, set_size, the_set, verbose_level)) {
		cout << "compute_stabilizer::retrieve_automorphism does not stabilize original set" << endl;
		exit(1);
		}
	if (FALSE) {
		cout << "compute_stabilizer::retrieve_automorphism is in the set stabilizer" << endl;
		}

	cout << "the automorphism is: " << endl;
	A_on_the_set->element_print(new_automorphism, cout);
	cout << "the automorphism acts on the set as: " << endl;
	A_on_the_set->element_print_as_permutation(new_automorphism, cout);

}

void compute_stabilizer::make_canonical_second_set(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_v4 = (verbose_level >= 4);
	int nodes;

	A_induced->make_canonical(
		reduced_set_size, reduced_set2_new_labels,
		canonical_set2, transporter2, nodes,
		TRUE, Aut,
		0 /*verbose_level - 1*/);

	backtrack_nodes_total_in_loop += nodes;
	if (f_v) {
		cout << "compute_stabilizer::make_canonical_second_set nodes=" << nodes << endl;
		}
	if (f_v4) {
		cout << "compute_stabilizer::make_canonical_second_set canonical set2: ";
		Orbiter->Lint_vec.print(cout, canonical_set2, reduced_set_size);
		cout << endl;
		}
	if (FALSE) {
		cout << "compute_stabilizer::make_canonical_second_set transporter2:" << endl;
		A_induced->element_print(transporter2, cout);
		cout << endl;
		A_induced->element_print_as_permutation(transporter2, cout);
		cout << endl;
		}
	A->mult(elt2, transporter2, T2);
	if (FALSE) {
		cout << "compute_stabilizer::make_canonical_second_set T2:" << endl;
		A->element_print(T2, cout);
		cout << endl;
		//A_induced.element_print_as_permutation(transporter2, cout);
		//cout << endl;
		}
}


int compute_stabilizer::compute_second_reduced_set()
{
	int i, j, a;
	sorting Sorting;

	for (i = 0; i < reduced_set_size; i++) {
		a = reduced_set2[i];
		for (j = 0; j < nb_interesting_points; j++) {
			if (interesting_points[j] == a) {
				reduced_set2_new_labels[i] = j;
				break;
				}
			}
		if (j == nb_interesting_points) {
			break;
			}
		}
	if (i < reduced_set_size) {
		return FALSE;
		}

	Sorting.lint_vec_heapsort(reduced_set2_new_labels, reduced_set_size);
#if 0
	if (f_vv) {
		cout << "reduced_set2_new_labels:" << endl;
		INT_vec_print(cout, reduced_set2_new_labels, reduced_set_size);
		cout << endl;
		}
#endif
#if 0
	if (f_vv) {
		cout << "sorted: ";
		INT_vec_print(cout, reduced_set2_new_labels, reduced_set_size);
		cout << endl;
		cout << "orbit invariant: ";
		for (i = 0; i < nb_orbits; i++) {
			if (orbit_count2[i] == 0)
				continue;
			cout << i << "^" << orbit_count2[i] << " ";
			}
		cout << endl;
		}
#endif

	return TRUE;
}

int compute_stabilizer::check_orbit_count()
{
	int i, j;

	for (i = 0; i < nb_interesting_orbits; i++) {
		j = interesting_orbits[i];
		if (orbit_count2[j] != orbit_count1[j]) {
			break;
			}
		}
	if (i < nb_interesting_orbits) {
		return FALSE;
		}
	else {
		return TRUE;
		}
}

void compute_stabilizer::print_orbit_count(int f_both)
{
	int i, j;

	cout << "orbit count:" << endl;
	for (i = 0; i < nb_interesting_orbits; i++) {
		j = interesting_orbits[i];
		cout << j << " : " << orbit_count1[j];
		if (f_both) {
			cout << " - " << orbit_count2[j];
			}
		cout << endl;
		}
}


void compute_stabilizer::null1()
{
	reduced_set1 = NULL;
	reduced_set2 = NULL;
	reduced_set1_new_labels = NULL;
	reduced_set2_new_labels = NULL;
	canonical_set1 = NULL;
	canonical_set2 = NULL;
	elt1 = NULL;
	Elt1 = NULL;
	Elt1_inv = NULL;
	new_automorphism = NULL;
	Elt4 = NULL;
	elt2 = NULL;
	Elt2 = NULL;
}

void compute_stabilizer::allocate1()
{

	reduced_set1 = NEW_lint(set_size);
	reduced_set2 = NEW_lint(set_size);
	reduced_set1_new_labels = NEW_lint(set_size);
	reduced_set2_new_labels = NEW_lint(set_size);
	canonical_set1 = NEW_lint(set_size);
	canonical_set2 = NEW_lint(set_size);
	elt1 = NEW_int(A->elt_size_in_int);
	elt2 = NEW_int(A->elt_size_in_int);
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt1_inv = NEW_int(A->elt_size_in_int);
	new_automorphism = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);
}

void compute_stabilizer::free1()
{
	if (reduced_set1) {
		FREE_lint(reduced_set1);
		FREE_lint(reduced_set2);
		FREE_lint(reduced_set1_new_labels);
		FREE_lint(reduced_set2_new_labels);
		FREE_lint(canonical_set1);
		FREE_lint(canonical_set2);
		FREE_int(elt1);
		FREE_int(elt2);
		FREE_int(Elt1);
		FREE_int(Elt2);
		FREE_int(Elt1_inv);
		FREE_int(new_automorphism);
		FREE_int(Elt4);
		}
}





}}
