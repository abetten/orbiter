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

	set_size = 0;
	the_set = NULL;

	A = NULL;
	A2 = NULL;
	PC = NULL;

	Elt1 = NULL;

	A_on_the_set = NULL;
		// only used to print the induced action on the set
		// of the set stabilizer

	Stab = NULL;
	//longinteger_object stab_order, new_stab_order;
	nb_times_orbit_count_does_not_match_up = 0;
	backtrack_nodes_first_time = 0;
	backtrack_nodes_total_in_loop = 0;

	level = 0;
	interesting_orbit = 0; // previously orb_idx

	interesting_subsets = NULL; // [nb_interesting_subsets]
	nb_interesting_subsets = 0;

	selected_set_stab_gens = NULL;
	selected_set_stab = NULL;


	reduced_set_size = 0; // = set_size - level




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
	transporter0 = NULL;


	//longinteger_object go_G;

	Stab_orbits = NULL;
	nb_orbits = 0;
	orbit_count1 = NULL; // [nb_orbits]
	orbit_count2 = NULL; // [nb_orbits]


	nb_interesting_subsets_reduced = 0;
	interesting_subsets_reduced = NULL;

	Orbit_patterns = NULL; // [nb_interesting_subsets * nb_orbits]



	orbit_to_interesting_orbit = NULL; // [nb_orbits]
	nb_interesting_orbits = 0;
	interesting_orbits = NULL;
	nb_interesting_points = 0;
	interesting_points = NULL;
	interesting_orbit_first = NULL;
	interesting_orbit_len = NULL;
	local_idx1 = local_idx2 = 0;





	A_induced = NULL;
	//longinteger_object induced_go, K_go;

	transporter_witness = NULL;
	transporter1 = NULL;
	transporter2 = NULL;
	T1 = NULL;
	T1v = NULL;
	T2 = NULL;

	Kernel_original = NULL;
	K = NULL; // kernel for building up Stab



	Aut = NULL;
	Aut_original = NULL;
	//longinteger_object ago;
	//longinteger_object ago1;
	//longinteger_object target_go;


	U = NULL;

	Canonical_forms = NULL;
	nb_interesting_subsets_rr = 0;
	interesting_subsets_rr = NULL;
}

compute_stabilizer::~compute_stabilizer()
{
	//free1();

	if (Stab) {
		FREE_OBJECT(Stab);
	}
	if (A_on_the_set) {
		FREE_OBJECT(A_on_the_set);
	}
	if (Elt1) {
		FREE_int(Elt1);
	}

	if (selected_set_stab_gens) {
		FREE_OBJECT(selected_set_stab_gens);
	}
	if (selected_set_stab) {
		FREE_OBJECT(selected_set_stab);
	}

	if (interesting_subsets_reduced) {
		FREE_lint(interesting_subsets_reduced);
	}
	if (Orbit_patterns) {
		FREE_int(Orbit_patterns);
	}

	if (Stab_orbits) {
		FREE_OBJECT(Stab_orbits);
	}

	if (orbit_count1) {
		FREE_int(orbit_count1);
		FREE_int(orbit_count2);
	}

	if (orbit_to_interesting_orbit) {
		FREE_int(orbit_to_interesting_orbit);
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

	if (Canonical_forms) {
		FREE_lint(Canonical_forms);
	}

}


void compute_stabilizer::init(long int *the_set, int set_size,
		long int *canonical_pts,
		poset_classification *PC, action *A, action *A2,
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
	compute_stabilizer::PC = PC;
	compute_stabilizer::A = A;
	compute_stabilizer::A2 = A2;
	compute_stabilizer::level = level;
	compute_stabilizer::interesting_orbit = interesting_orbit;
	compute_stabilizer::nb_interesting_subsets = nb_interesting_subsets;
	compute_stabilizer::interesting_subsets = interesting_subsets;

	if (f_v) {
		cout << "compute_stabilizer::init A=" << endl;
		A->print_info();
	}

	if (f_v) {
		cout << "compute_stabilizer::init A2=" << endl;
		A2->print_info();
	}


	int size;

	PC->get_set(level, interesting_orbit, canonical_pts, size);
	if (f_v) {
		cout << "compute_stabilizer::init canonical substructure: ";
		Orbiter->Lint_vec.print(cout, canonical_pts, size);
		cout << endl;
	}


	if (f_v) {
		cout << "compute_stabilizer::init before PC->get_stabilizer_generators" << endl;
	}
	PC->get_stabilizer_generators(
			selected_set_stab_gens,
			level, interesting_orbit, 0 /*verbose_level*/);
	if (f_v) {
		cout << "compute_stabilizer::init after PC->get_stabilizer_generators" << endl;
		selected_set_stab_gens->print_generators_tex();
	}
	selected_set_stab_gens->group_order(go_G);
	if (f_v) {
		cout << "compute_stabilizer::init go_G=" << go_G << endl;
	}

	if (f_v) {
		cout << "compute_stabilizer::init before selected_set_stab_gens->create_sims" << endl;
	}
	selected_set_stab = selected_set_stab_gens->create_sims(verbose_level);
	if (f_v) {
		cout << "compute_stabilizer::init after selected_set_stab_gens->create_sims" << endl;
	}



	if (f_v) {
		cout << "compute_stabilizer::init before creating Stab" << endl;
	}


	Stab = NEW_OBJECT(sims);
	Stab->init(A, 0 /* verbose_level */);
	Stab->init_trivial_group(0 /* verbose_level - 1*/);



	if (f_v) {
		cout << "compute_stabilizer::init before A2->restricted_action" << endl;
	}
	A_on_the_set = A2->restricted_action(the_set, set_size, 0/*verbose_level*/);
	if (f_v) {
		cout << "compute_stabilizer::init after A2->restricted_action" << endl;
	}


	reduced_set_size = set_size - level;

	allocate1();


	if (f_v) {
		cout << "compute_stabilizer::init before compute_orbits_and_find_minimal_pattern" << endl;
	}
	compute_orbits_and_find_minimal_pattern(0 /*verbose_level*/);
	if (f_v) {
		cout << "compute_stabilizer::init after compute_orbits_and_find_minimal_pattern" << endl;
		cout << "compute_stabilizer::init nb_interesting_subsets_reduced = " << nb_interesting_subsets_reduced << endl;
	}

	if (f_v) {
		cout << "compute_stabilizer::init before find_interesting_orbits" << endl;
	}
	find_interesting_orbits(verbose_level);
	if (f_v) {
		cout << "compute_stabilizer::init after find_interesting_orbits" << endl;
	}




	if (f_v) {
		cout << "compute_stabilizer::init before restricted_action_on_interesting_points" << endl;
	}
	restricted_action_on_interesting_points(verbose_level);
	if (f_v) {
		cout << "compute_stabilizer::init after restricted_action_on_interesting_points" << endl;
	}



	if (f_v) {
		cout << "compute_stabilizer::init before compute_canonical_form" << endl;
	}
	compute_canonical_form(verbose_level);
	if (f_v) {
		cout << "compute_stabilizer::init after compute_canonical_form" << endl;
		cout << "compute_stabilizer::init backtrack_nodes_first_time = " << backtrack_nodes_first_time << endl;
		cout << "compute_stabilizer::init backtrack_nodes_total_in_loop = " << backtrack_nodes_total_in_loop << endl;
	}

	if (f_v) {
		cout << "compute_stabilizer::init before init_U" << endl;
	}

	init_U(verbose_level);
	if (f_v) {
		cout << "compute_stabilizer::init after init_U" << endl;
	}


	if (f_v) {
		cout << "compute_stabilizer::init before compute_automorphism_group" << endl;
	}
	compute_automorphism_group(verbose_level);
	if (f_v) {
		cout << "compute_stabilizer::init after compute_automorphism_group" << endl;
		cout << "compute_stabilizer::init backtrack_nodes_first_time = " << backtrack_nodes_first_time << endl;
		cout << "compute_stabilizer::init backtrack_nodes_total_in_loop = " << backtrack_nodes_total_in_loop << endl;
	}







	int i;

	for (i = 0; i < reduced_set_size; i++) {
		canonical_pts[level + i] = interesting_points[canonical_set1[i]];
	}

	if (f_v) {
		cout << "compute_stabilizer::init input    : ";
		Orbiter->Lint_vec.print(cout, the_set, set_size);
		cout << endl;
		cout << "compute_stabilizer::init canonical: ";
		Orbiter->Lint_vec.print(cout, canonical_pts, set_size);
		cout << endl;
	}

	if (f_v) {
		cout << "compute_stabilizer::init done" << endl;
	}
}


void compute_stabilizer::compute_automorphism_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int cnt2;
	sorting Sorting;

	if (f_v) {
		cout << "compute_stabilizer::compute_automorphism_group" << endl;
	}

	backtrack_nodes_total_in_loop = 0;



	for (cnt2 = 0; cnt2 < nb_interesting_subsets_rr; cnt2++) {


		//if (U->is_minimal(cnt, 0 /* verbose_level */)) {
			compute_automorphism_group_handle_case(cnt2, verbose_level);
		//}

	}

	if (f_v) {
		cout << "compute_stabilizer::compute_automorphism_group nb_interesting_subsets_reduced = " << nb_interesting_subsets_reduced << endl;
		cout << "compute_stabilizer::compute_automorphism_group nb_times_orbit_count_does_not_match_up = " << nb_times_orbit_count_does_not_match_up << endl;
	}



	if (f_v) {
		cout << "compute_stabilizer::compute_automorphism_group done" << endl;
	}
}


void compute_stabilizer::compute_automorphism_group_handle_case(int cnt2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int cnt;

	if (f_v) {
		cout << "compute_stabilizer::compute_automorphism_group_handle_case STABILIZER loop "
				<< cnt2 << " / " << nb_interesting_subsets_rr << " stab_order=" << stab_order << endl;
	}

	cnt = interesting_subsets_rr[cnt2];


	if (f_vv) {
		cout << "compute_stabilizer::compute_automorphism_group_handle_case before map_reduced_set_and_do_orbit_counting" << endl;
	}
	map_reduced_set_and_do_orbit_counting(cnt, interesting_subsets_reduced[cnt], elt1, verbose_level);
	if (f_vv) {
		cout << "compute_stabilizer::compute_automorphism_group_handle_case after map_reduced_set_and_do_orbit_counting" << endl;
	}

	if (f_v) {
		Orbiter->Lint_vec.print(cout, reduced_set1, reduced_set_size);
		cout << endl;
	}


	if (!check_orbit_count()) {
		cout << "compute_automorphism_group_handle_case !check_orbit_count()" << endl;
		exit(1);
	}


	if (f_vv) {
		cout << "compute_stabilizer::compute_automorphism_group_handle_case before compute_local_labels" << endl;
	}
	compute_local_labels(reduced_set1, reduced_set1_new_labels, reduced_set_size, verbose_level);
	if (f_vv) {
		cout << "compute_stabilizer::compute_automorphism_group_handle_case after compute_local_labels" << endl;
	}
	if (f_v) {
		cout << "local labels:" << endl;
		Orbiter->Lint_vec.print(cout, reduced_set1_new_labels, reduced_set_size);
		cout << endl;
	}


	sims *Stab0 = NULL;

	if (cnt2 == 0) {


		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case before compute_canonical_set_and_group" << endl;
		}
		compute_canonical_set_and_group(reduced_set1_new_labels, canonical_set1, reduced_set_size,
				elt2 /*int *transporter*/, Stab0,
				verbose_level);
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case after compute_canonical_set_and_group" << endl;
		}
		A->element_mult(elt1, elt2, transporter0, FALSE);

	}
	else {
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case before compute_canonical_set" << endl;
		}
		compute_canonical_set(reduced_set1_new_labels, canonical_set1, reduced_set_size,
				elt2 /*int *transporter*/, verbose_level);
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case after compute_canonical_set" << endl;
		}
		A->element_mult(elt1, elt2, T2, FALSE);
	}
	if (f_v) {
		cout << "canonical form:" << endl;
		Orbiter->Lint_vec.print(cout, canonical_set1, reduced_set_size);
		cout << endl;
	}

	int cmp;

	cmp = lint_vec_compare(canonical_set1, canonical_set2, reduced_set_size);

	if (cmp != 0) {
		cout << "compute_stabilizer::compute_automorphism_group_handle_case "
				"the two canonical sets are not the same, error" << endl;
		exit(1);
	}

	if (cnt2 == 0) {
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case before setup_stabilizer" << endl;
		}
		setup_stabilizer(Stab0, verbose_level);
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case after setup_stabilizer" << endl;
		}
	}
	else {
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case before retrieve_automorphism" << endl;
		}
		retrieve_automorphism(verbose_level);
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case after retrieve_automorphism" << endl;
		}


			// new automorphism is now in new_automorphism


		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case before add_automorphism" << endl;
		}
		add_automorphism(verbose_level);
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case after add_automorphism" << endl;
		}


		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case before update_stabilizer" << endl;
		}
		update_stabilizer(verbose_level);
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case after update_stabilizer" << endl;
		}


	}

	if (f_v) {
		cout << "compute_stabilizer::compute_automorphism_group_handle_case done" << endl;
	}
}

void compute_stabilizer::setup_stabilizer(sims *Stab0, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer" << endl;
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
	K->init_trivial_group(0 /*verbose_level - 1*/);

	A_induced->Kernel->group_order(K_go);
	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer kernel has order " << K_go << endl;
	}

	// conjugate the Kernel so that it is a subgroup of
	// the stabilizer of the set the_set[] that we wanted to stabilizer originally:
	// remember that elt1 is the transporter that was computed in map_it() above

	Kernel_original->conjugate(A_induced->Kernel->A,
			A_induced->Kernel, transporter0,
			FALSE, 0 /*verbose_level - 3*/);
	Kernel_original->group_order(K_go);
	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer after conjugation, kernel has order " << K_go << endl;
	}

	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer adding kernel of order " << K_go
			<< " to the stabilizer (in action " << Stab->A->label << ")" << endl;
	}
	Stab->build_up_group_random_process(K, Kernel_original,
		K_go, FALSE, NULL, 0 /*verbose_level - 3*/);
	Stab->group_order(stab_order);

	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer kernel of action on the set has been added to stabilizer" << endl;
		cout << "compute_stabilizer::setup_stabilizer current stabilizer order " << stab_order << endl;
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
		cout << "compute_stabilizer::setup_stabilizer before A_induced.make_canonical" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "before make_canonical: ";
		Orbiter->Lint_vec.print(cout, reduced_set1_new_labels, reduced_set_size);
		cout << endl;
	}


	// Stab0 is the stabilizer of canonical_set1 in the induced action (A_induced)

	Aut = Stab0;

	Aut->group_order(ago);
	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer current stabilizer order " << ago << endl;
	}
	if (FALSE) {
		cout << "transporter1:" << endl;
		A_induced->element_print(transporter0, cout);
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

	A->element_move(transporter0, T1, 0);

	if (FALSE) {
		cout << "T1:" << endl;
		A->element_print(T1, cout);
		cout << endl;
	}
	A->element_invert(T1, T1v, FALSE);

	// T1 := elt1 * elt2
	// moves the_set to the canonical set.

	Aut->group_order(ago);
	Aut_original->conjugate(A /*Aut.A */, Aut, T1,
		TRUE /* f_overshooting_OK */, 0 /*verbose_level - 3*/);
	Aut_original->group_order(ago1);
	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer after conjugation, group in action " << Aut_original->A->label << endl;
		cout << "compute_stabilizer::setup_stabilizer automorphism group order before = " << ago << endl;
		cout << "compute_stabilizer::setup_stabilizer automorphism group order after = " << ago1 << endl;
	}

	longinteger_domain D;

	D.mult(K_go, ago, target_go);
	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer target_go=" << target_go << endl;
		cout << "compute_stabilizer::setup_stabilizer adding automorphisms to set-stabilizer" << endl;
	}
	Stab->build_up_group_random_process(K, Aut_original,
		target_go, FALSE, NULL, 0 /*verbose_level - 3*/);
	Stab->group_order(stab_order);
	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer set stabilizer is added to stabilizer" << endl;
		cout << "compute_stabilizer::setup_stabilizer current stabilizer order " << stab_order << endl;
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
		cout << "compute_stabilizer::setup_stabilizer done" << endl;
	}

}

void compute_stabilizer::compute_canonical_form(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int cnt;
	sorting Sorting;

	if (f_v) {
		cout << "compute_stabilizer::compute_canonical_form" << endl;
	}
	// In the second step, we consider all possible images for the chosen k-subset
	// and try to pick up coset representatives for the subgroup in the
	// full set stabilizer:


	nb_times_orbit_count_does_not_match_up = 0;
	backtrack_nodes_total_in_loop = 0;


	Canonical_forms = NEW_lint(nb_interesting_subsets_reduced * reduced_set_size);

	for (cnt = 0; cnt < nb_interesting_subsets_reduced; cnt++) {

		//if (U->is_minimal(cnt, 0 /* verbose_level */)) {
			compute_canonical_form_handle_case(cnt, verbose_level);
		//}

	}

	if (f_v) {
		cout << "compute_stabilizer::compute_canonical_form nb_interesting_subsets_reduced = " << nb_interesting_subsets_reduced << endl;
		cout << "compute_stabilizer::compute_canonical_form nb_times_orbit_count_does_not_match_up = " << nb_times_orbit_count_does_not_match_up << endl;
	}


	if (f_v) {
		cout << "Canonical forms:" << endl;
		Orbiter->Lint_vec.matrix_print(Canonical_forms, nb_interesting_subsets_reduced, reduced_set_size, 2);
		cout << endl;
	}

	interesting_subsets_rr = NEW_lint(nb_interesting_subsets_reduced);

	for (cnt = 0; cnt < nb_interesting_subsets_reduced; cnt++) {
		if (cnt == 0) {
			Orbiter->Lint_vec.copy(Canonical_forms + cnt * reduced_set_size, canonical_set2, reduced_set_size);
			nb_interesting_subsets_rr = 0;
			interesting_subsets_rr[nb_interesting_subsets_rr++] = cnt;
		}
		else {
			int cmp;

			cmp = Sorting.lint_vec_compare(Canonical_forms + cnt * reduced_set_size, canonical_set2, reduced_set_size);

			if (cmp > 0) {
				Orbiter->Lint_vec.copy(Canonical_forms + cnt * reduced_set_size, canonical_set2, reduced_set_size);
				nb_interesting_subsets_rr = 0;
				interesting_subsets_rr[nb_interesting_subsets_rr++] = cnt;
			}
			else if (cmp == 0) {
				interesting_subsets_rr[nb_interesting_subsets_rr++] = cnt;
			}
		}

	}

#if 1
	if (f_v) {
		cout << "canonical form : " << endl;
		Orbiter->Lint_vec.matrix_print(canonical_set2, 1, reduced_set_size, 2);
		cout << "nb_interesting_subsets_rr = " << nb_interesting_subsets_rr << endl;
		cout << "interesting_subsets_rr:" << endl;
		Orbiter->Lint_vec.print(cout, interesting_subsets_rr, nb_interesting_subsets_rr);
		cout << endl;
	}
#endif

	if (f_v) {
		cout << "compute_stabilizer::compute_canonical_form done" << endl;
	}
}

void compute_stabilizer::compute_canonical_form_handle_case(int cnt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "compute_stabilizer::compute_canonical_form_handle_case STABILIZER loop " << cnt << " / " << nb_interesting_subsets_reduced << " subset_idx=" << interesting_subsets_reduced[cnt] << endl;
	}

	if (f_vv) {
		cout << "compute_stabilizer::compute_canonical_form_handle_case before map_reduced_set_and_do_orbit_counting" << endl;
	}
	map_reduced_set_and_do_orbit_counting(cnt, interesting_subsets_reduced[cnt], elt1, verbose_level);
	if (f_vv) {
		cout << "compute_stabilizer::compute_canonical_form_handle_case after map_reduced_set_and_do_orbit_counting" << endl;
	}

	if (f_v) {
		Orbiter->Lint_vec.print(cout, reduced_set1, reduced_set_size);
		cout << endl;
	}


	if (!check_orbit_count()) {
		cout << "compute_canonical_form_handle_case !check_orbit_count()" << endl;
		cout << "reduced_set1: ";
		Orbiter->Lint_vec.print(cout, reduced_set1, reduced_set_size);
		cout << endl;
		cout << "orbit_count1: ";
		Orbiter->Int_vec.print(cout, orbit_count1, nb_orbits);
		cout << endl;
		cout << "orbit_count2: ";
		Orbiter->Int_vec.print(cout, orbit_count2, nb_orbits);
		cout << endl;
		exit(1);
	}


	if (f_vv) {
		cout << "compute_stabilizer::compute_canonical_form_handle_case before compute_local_labels" << endl;
	}
	compute_local_labels(reduced_set1, reduced_set1_new_labels, reduced_set_size, verbose_level);
	if (f_vv) {
		cout << "compute_stabilizer::compute_canonical_form_handle_case after compute_local_labels" << endl;
	}
	if (f_v) {
		cout << "local labels:" << endl;
		Orbiter->Lint_vec.print(cout, reduced_set1_new_labels, reduced_set_size);
		cout << endl;
	}


	sims *stab;

	if (f_vv) {
		cout << "compute_stabilizer::compute_canonical_form_handle_case before compute_canonical_set" << endl;
	}
	compute_canonical_set_and_group(reduced_set1_new_labels, canonical_set1, reduced_set_size,
			elt2 /*int *transporter*/, stab, verbose_level);
	if (f_vv) {
		cout << "compute_stabilizer::compute_canonical_form_handle_case after compute_canonical_set" << endl;
	}
	if (FALSE) {
		cout << "canonical form:" << endl;
		Orbiter->Lint_vec.print(cout, canonical_set1, reduced_set_size);
		cout << endl;
	}

	Orbiter->Lint_vec.copy(canonical_set1, Canonical_forms + cnt * reduced_set_size, reduced_set_size);


}

void compute_stabilizer::compute_canonical_set(long int *set_in, long int *set_out, int sz,
		int *transporter, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_v4 = (verbose_level >= 4);
	int nb_nodes;

	if (f_v) {
		cout << "compute_stabilizer::compute_canonical_set" << endl;
	}
	if (f_v) {
		cout << "compute_stabilizer::compute_canonical_set before A_induced->make_canonical" << endl;
	}

	sims *my_Aut;

	my_Aut = NEW_OBJECT(sims);

	A_induced->make_canonical(
		sz, set_in,
		set_out, transporter, nb_nodes,
		TRUE, my_Aut,
		0 /*verbose_level - 1*/);

	FREE_OBJECT(my_Aut);

	if (f_v) {
		cout << "compute_stabilizer::compute_canonical_set after A_induced->make_canonical" << endl;
	}
	if (f_v) {
		cout << "compute_stabilizer::compute_canonical_set done" << endl;
	}
}

void compute_stabilizer::compute_canonical_set_and_group(
		long int *set_in, long int *set_out, int sz,
		int *transporter, sims *&stab, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_nodes;

	if (f_v) {
		cout << "compute_stabilizer::compute_canonical_set_and_group" << endl;
	}
	if (f_v) {
		cout << "compute_stabilizer::compute_canonical_set_and_group before A_induced->make_canonical" << endl;
	}


	stab = NEW_OBJECT(sims);

	A_induced->make_canonical(
		sz, set_in,
		set_out, transporter, nb_nodes,
		TRUE, stab,
		0 /*verbose_level - 1*/);

	if (f_v) {
		cout << "compute_stabilizer::compute_canonical_set_and_group after A_induced->make_canonical" << endl;
	}
	if (f_v) {
		cout << "compute_stabilizer::compute_canonical_set_and_group done" << endl;
	}
}


void compute_stabilizer::compute_local_labels(long int *set_in, long int *set_out, int sz, int verbose_level)
{
	// Let reduced_set1_new_labels[] be the set reduced_set1[] in the restricted action,
	// and let the set be ordered increasingly:

	int i, idx, idx1, f, l, pos_local;
	long int a;
	sorting Sorting;

	for (i = 0; i < sz; i++) {
		a = set_in[i];
		idx = Stab_orbits->orbit_number(a);
		idx1 = orbit_to_interesting_orbit[idx];
		f = interesting_orbit_first[idx1];
		l = interesting_orbit_len[idx1];
		if (!Sorting.lint_vec_search(interesting_points + f, l, a, pos_local, 0 /* verbose_level */)) {
			cout << "compute_stabilizer::compute_local_labels did not find point " << a << endl;
			exit(1);
		}
		set_out[i] = f + pos_local;
	}

	Sorting.lint_vec_heapsort(set_out, sz);

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
		interesting_subsets_reduced, nb_interesting_subsets_reduced,
		verbose_level);


	if (f_v) {
		cout << "compute_stabilizer::init_U after U->init" << endl;
	}
}

void compute_stabilizer::compute_orbits_and_find_minimal_pattern(int verbose_level)
// uses selected_set_stab_gens to compute orbits on points in action A2
{
	int f_v = (verbose_level >= 1);
	sorting Sorting;

	if (f_v) {
		cout << "compute_stabilizer::compute_orbits_and_find_minimal_pattern" << endl;
	}
	if (f_v) {
		cout << "compute_stabilizer::compute_orbits_and_find_minimal_pattern computing orbits on points" << endl;
	}
	Stab_orbits = selected_set_stab_gens->orbits_on_points_schreier(A2, 0 /*verbose_level*/);
	if (f_v) {
		cout << "compute_stabilizer::compute_orbits_and_find_minimal_pattern computing orbits on points done, we found " << Stab_orbits->nb_orbits << " orbits" << endl;
	}

	nb_orbits = Stab_orbits->nb_orbits;
	orbit_count1 = NEW_int(nb_orbits);
	orbit_count2 = NEW_int(nb_orbits);
	Orbiter->Int_vec.zero(orbit_count1, nb_orbits);

	int cnt;

	interesting_subsets_reduced = NEW_lint(nb_interesting_subsets);
	Orbit_patterns = NEW_int(nb_interesting_subsets * nb_orbits);

	if (f_v) {
		cout << "compute_stabilizer::compute_orbits_and_find_minimal_pattern computing Orbit_patterns" << endl;
	}

	for (cnt = 0; cnt < nb_interesting_subsets; cnt++) {

		if (f_v) {
			cout << "compute_stabilizer::compute_orbits_and_find_minimal_pattern computing Orbit_patterns cnt = " << cnt << endl;
		}
		find_orbit_pattern(cnt, elt1 /* transp */, verbose_level - 4);


		Orbiter->Int_vec.copy(orbit_count1, Orbit_patterns + cnt * nb_orbits, nb_orbits);

	}

	if (f_v) {
		cout << "compute_stabilizer::compute_orbits_and_find_minimal_pattern computing Orbit_patterns done" << endl;
	}


	if (f_v) {
		cout << "orbit patterns (top row is orbit length): " << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				Stab_orbits->orbit_len, 1, nb_orbits, nb_orbits, 2);
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				Orbit_patterns, nb_interesting_subsets, nb_orbits, nb_orbits, 2);
		//Orbiter->Int_vec.print(cout, Orbit_patterns, nb_orbits);
		//cout << endl;
	}


	for (cnt = 0; cnt < nb_interesting_subsets; cnt++) {
		if (cnt == 0) {
			Orbiter->Int_vec.copy(Orbit_patterns + cnt * nb_orbits, orbit_count2, nb_orbits);
			nb_interesting_subsets_reduced = 0;
			interesting_subsets_reduced[nb_interesting_subsets_reduced++] = interesting_subsets[cnt];
		}
		else {
			int cmp;

			cmp = Sorting.integer_vec_compare(Orbit_patterns + cnt * nb_orbits, orbit_count2, nb_orbits);

			if (cmp > 0) {
				Orbiter->Int_vec.copy(Orbit_patterns + cnt * nb_orbits, orbit_count2, nb_orbits);
				nb_interesting_subsets_reduced = 0;
				interesting_subsets_reduced[nb_interesting_subsets_reduced++] = interesting_subsets[cnt];
			}
			else if (cmp == 0) {
				interesting_subsets_reduced[nb_interesting_subsets_reduced++] = interesting_subsets[cnt];
			}
		}

	}

#if 1
	if (f_v) {
		cout << "minimal orbit pattern : " << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				Stab_orbits->orbit_len, 1, nb_orbits, nb_orbits, 2);
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				orbit_count2, 1, nb_orbits, nb_orbits, 2);
		cout << "nb_interesting_subsets_reduced = " << nb_interesting_subsets_reduced << endl;
		cout << "interesting_subsets_reduced:" << endl;
		Orbiter->Lint_vec.print(cout, interesting_subsets_reduced, nb_interesting_subsets_reduced);
		cout << endl;
	}
#endif

	if (f_v) {
		cout << "compute_stabilizer::compute_orbits_and_find_minimal_pattern done" << endl;
	}
}

void compute_stabilizer::find_interesting_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_stabilizer::find_interesting_orbits" << endl;
	}
	// An orbit is interesting if it contains points from reduced_set1[],
	// i.e., orbit i is interesting if orbit_count1[i] is not equal to zero
	// Let interesting_orbits[nb_interesting_orbits] be the list of interesting orbits

	int i;

	nb_interesting_orbits = 0;
	nb_interesting_points = 0;
	orbit_to_interesting_orbit = NEW_int(nb_orbits);
	interesting_orbits = NEW_int(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		if (orbit_count2[i]) {
			orbit_to_interesting_orbit[i] = nb_interesting_orbits;
			interesting_orbits[nb_interesting_orbits++] = i;
			nb_interesting_points += Stab_orbits->orbit_len[i];
		}
		else {
			orbit_to_interesting_orbit[i] = -1;
		}
	}
	if (f_v) {
		cout << "nb_interesting_orbits = " << nb_interesting_orbits << endl;
		cout << "nb_interesting_points = " << nb_interesting_points << endl;
		cout << "interesting_orbits:" << endl;
		Orbiter->Int_vec.print(cout, interesting_orbits, nb_interesting_orbits);
		cout << endl;
		cout << "orbit_to_interesting_orbit:" << endl;
		Orbiter->Int_vec.print(cout, orbit_to_interesting_orbit, nb_orbits);
		cout << endl;
	}

	interesting_points = NEW_lint(nb_interesting_points);

	interesting_orbit_first = NEW_int(nb_interesting_orbits);
	interesting_orbit_len = NEW_int(nb_interesting_orbits);

	int idx, j, f, l, k, ii;
	sorting Sorting;

	j = 0;
	for (k = 0; k < nb_interesting_orbits; k++) {
		idx = interesting_orbits[k];
		f = Stab_orbits->orbit_first[idx];
		l = Stab_orbits->orbit_len[idx];
		interesting_orbit_first[k] = j;
		interesting_orbit_len[k] = l;
		for (ii = 0; ii < l; ii++) {
			interesting_points[j++] = Stab_orbits->orbit[f + ii];
		}
		Sorting.lint_vec_heapsort(interesting_points + interesting_orbit_first[k], l);
	}

	if (f_v) {
		cout << "interesting_points:" << endl;
		for (k = 0; k < nb_interesting_orbits; k++) {
			f = interesting_orbit_first[k];
			l = interesting_orbit_len[k];
			Orbiter->Lint_vec.print(cout, interesting_points + f, l);
			if (k < nb_interesting_orbits - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}


}

void compute_stabilizer::find_orbit_pattern(int cnt, int *transp, int verbose_level)
// computes transporter to transp
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_stabilizer::find_orbit_pattern cnt=" << cnt
				<< " interesting_subsets[cnt]=" << interesting_subsets[cnt] << endl;
	}
	sorting Sorting;

	if (f_v) {
		cout << "compute_stabilizer::find_orbit_pattern before PC->map_to_canonical_k_subset" << endl;
	}
	PC->map_to_canonical_k_subset(the_set, set_size,
			level /* subset_size */, interesting_subsets[cnt],
			reduced_set1, transp /*transporter */, local_idx1, verbose_level - 4);
		// reduced_set1 has size set_size - level (=reduced_set_size)
	if (f_v) {
		cout << "compute_stabilizer::find_orbit_pattern after PC->map_to_canonical_k_subset" << endl;
	}


	Sorting.lint_vec_heapsort(reduced_set1, reduced_set_size);
	if (FALSE) {
		cout << "compute_stabilizer::find_orbit_pattern STABILIZER "
				<< setw(4) << cnt << " : " << setw(4) << interesting_subsets[cnt] << " : ";
		Orbiter->Lint_vec.print(cout, reduced_set1, reduced_set_size);
		cout << endl;
	}

	Stab_orbits->compute_orbit_statistic_lint(reduced_set1, reduced_set_size,
			orbit_count1, verbose_level - 1);

	if (f_v) {
		cout << "compute_stabilizer::find_orbit_pattern" << endl;
	}
}

void compute_stabilizer::compute_orbits(int verbose_level)
// uses selected_set_stab_gens to compute orbits on points in action A2
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_stabilizer::compute_orbits computing orbits on points" << endl;
	}
	Stab_orbits = selected_set_stab_gens->orbits_on_points_schreier(A2, 0 /*verbose_level*/);
	if (f_v) {
		cout << "compute_stabilizer::compute_orbits computing orbits on points done, we found " << Stab_orbits->nb_orbits << " orbits" << endl;
	}

	nb_orbits = Stab_orbits->nb_orbits;
	orbit_count1 = NEW_int(nb_orbits);
	orbit_count2 = NEW_int(nb_orbits);
	Orbiter->Int_vec.zero(orbit_count1, nb_orbits);


	if (f_v) {
		cout << "compute_stabilizer::compute_orbits done" << endl;
	}
}


void compute_stabilizer::restricted_action_on_interesting_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points" << endl;
	}
	// Compute the restricted action on the set of
	// interesting points and call it A_induced :
	// Determine the kernel


	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points interesting_points=" << endl;
		Orbiter->Lint_vec.print(cout, interesting_points, nb_interesting_points);
		cout << endl;
	}

	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points computing induced action by restriction" << endl;
	}

	A_induced = A2->restricted_action(interesting_points, nb_interesting_points, 0 /*verbose_level*/);
	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points after A2->restricted_action" << endl;
	}
	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points before A_induced->group_order" << endl;
	}
	A_induced->group_order(induced_go);
	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points induced_go = " << induced_go << endl;
		cout << "compute_stabilizer::restricted_action_on_interesting_points A_induced:" << endl;
		A_induced->print_info();
	}

	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points before A_induced->induced_action_override_sims" << endl;
	}
	A_induced->induced_action_override_sims(
		*A, selected_set_stab,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points after A_induced->induced_action_override_sims" << endl;
	}

	if (!A_induced->f_has_kernel) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points A_induced does not have kernel" << endl;
		exit(1);
	}
	A_induced->Kernel->group_order(K_go);
	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points induced action by restriction: group order = " << induced_go << endl;
		cout << "kernel group order = " << K_go << endl;
		//cout << "strong generators for induced action:" << endl;
		//A_induced.strong_generators->print_as_permutation(cout);
		//cout << "strong generators for kernel:" << endl;
		//A_induced.Kernel->gens.print_as_permutation(cout);
	}



	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points done" << endl;
	}
}



void compute_stabilizer::map_the_first_set_and_do_orbit_counting(int cnt, int verbose_level)
{
	sorting Sorting;

	PC->map_to_canonical_k_subset(the_set, set_size,
			level /* subset_size */, interesting_subsets[cnt],
			reduced_set1, elt1 /*transporter */, local_idx1, verbose_level - 4);

		// map the chosen subset interesting_subsets[cnt]
		// to the canonical orbit rep and move it to the beginning.
		// The remaining points are mapped as well and are arranged after the canonical subset.
		// the remaining points are stored in reduced_set1.
		// local_idx1 is the (local) orbit index of the chosen set in the orbits at level
		// reduced_set1 has size set_size - level (=reduced_set_size)


	Sorting.lint_vec_heapsort(reduced_set1, reduced_set_size);
	if (FALSE) {
		cout << setw(4) << cnt << " : " << setw(4) << interesting_subsets[cnt] << " : ";
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

	Stab_orbits->compute_orbit_statistic_lint(reduced_set1, reduced_set_size,
			orbit_count1, verbose_level - 1);
}


void compute_stabilizer::map_reduced_set_and_do_orbit_counting(int cnt,
		long int subset_idx, int *transporter, int verbose_level)
{
	sorting Sorting;

	PC->map_to_canonical_k_subset(the_set, set_size,
			level /* subset_size */, subset_idx,
			reduced_set1, transporter, local_idx1, verbose_level - 4);
		// reduced_set2 has size set_size - level (=reduced_set_size)


	Sorting.lint_vec_heapsort(reduced_set1, reduced_set_size);
	if (FALSE) {
		cout << "compute_stabilizer::map_the_second_set STABILIZER "
				<< setw(4) << cnt << " : " << setw(4) << interesting_subsets_reduced[cnt] << " : ";
		Orbiter->Lint_vec.print(cout, reduced_set1, reduced_set_size);
		cout << endl;
	}

	Stab_orbits->compute_orbit_statistic_lint(reduced_set1, reduced_set_size, orbit_count1, verbose_level - 1);
}

void compute_stabilizer::update_stabilizer(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	int cmp;

	Stab->group_order(new_stab_order);
	cmp = D.compare_unsigned(new_stab_order, stab_order);
	if (cmp) {

		if (f_v) {
			cout << "compute_stabilizer::update_stabilizer new stabilizer order is " << new_stab_order << endl;
		}
		new_stab_order.assign_to(stab_order);


		strong_generators *Strong_gens;

		Strong_gens = NEW_OBJECT(strong_generators);
		Strong_gens->init_from_sims(Stab, 0);

		if (f_v) {
			Strong_gens->print_generators(cout);
		}
		FREE_OBJECT(Strong_gens);

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

	if (f_v) {
		cout << "the automorphism is: " << endl;
		A_on_the_set->element_print(new_automorphism, cout);
		cout << endl;
		cout << "the automorphism acts on the set as: " << endl;
		A_on_the_set->element_print_as_permutation(new_automorphism, cout);
		cout << endl;
	}
}

void compute_stabilizer::make_canonical_second_set(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_v4 = (verbose_level >= 4);
	int nb_nodes;

	if (f_v) {
		cout << "compute_stabilizer::make_canonical_second_set" << endl;
	}
	if (f_v) {
		cout << "compute_stabilizer::make_canonical_second_set before A_induced->make_canonical" << endl;
	}
	A_induced->make_canonical(
		reduced_set_size, reduced_set2_new_labels,
		canonical_set2, transporter2, nb_nodes,
		TRUE, Aut,
		0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "compute_stabilizer::make_canonical_second_set after A_induced->make_canonical" << endl;
	}

	backtrack_nodes_total_in_loop += nb_nodes;
	if (f_v) {
		cout << "compute_stabilizer::make_canonical_second_set nb_nodes=" << nb_nodes << endl;
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
	if (f_v) {
		cout << "compute_stabilizer::make_canonical_second_set done" << endl;
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
	int i;

	for (i = 0; i < nb_orbits; i++) {
		if (orbit_count2[i] != orbit_count1[i]) {
			break;
		}
	}
	if (i < nb_orbits) {
		return FALSE;
	}
	else {
		return TRUE;
	}
}

void compute_stabilizer::print_orbit_count(int f_both)
{
	int i;

	cout << "orbit count:" << endl;
	for (i = 0; i < nb_orbits; i++) {
		cout << i << " : " << orbit_count1[i];
		if (f_both) {
			cout << " - " << orbit_count2[i];
		}
		cout << endl;
	}
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
	transporter0 = NEW_int(A->elt_size_in_int);
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
		FREE_int(transporter0);
	}
}

void compute_stabilizer::report(std::ostream &ost)
{
	ost << "Input set of size " << set_size << " : ";
	ost << "$";
	Orbiter->Lint_vec.print_fully(ost, the_set, set_size);
	ost << "$";
	ost << "\\\\" << endl;

	ost << "Level = " << level << "\\\\" << endl;
	ost << "Interesting_orbit = " << interesting_orbit << "\\\\" << endl;
	ost << "Nb_interesting_subsets = " << nb_interesting_subsets << "\\\\" << endl;

	ost << "interesting_subsets: ";
	//ost << "$";
	Orbiter->Lint_vec.print_fully(ost, interesting_subsets, nb_interesting_subsets);
	//ost << "$";
	ost << "\\\\" << endl;

	ost << "Classification of small sets: ";
	PC->report(ost, 0 /* verbose_level */);

}




}}
