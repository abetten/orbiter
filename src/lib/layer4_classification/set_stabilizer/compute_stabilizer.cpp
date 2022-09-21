/*
 * compute_stabilizer.cpp
 *
 *  Created on: Apr 24, 2021
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {



compute_stabilizer::compute_stabilizer()
{

	SubSt = NULL;


	A_on_the_set = NULL;
		// only used to print the induced action on the set
		// of the set stabilizer

	Stab = NULL;
	//longinteger_object stab_order, new_stab_order;
	nb_times_orbit_count_does_not_match_up = 0;
	backtrack_nodes_first_time = 0;
	backtrack_nodes_total_in_loop = 0;


	Stab_orbits = NULL;






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


	//U = NULL;

	Canonical_form_input = NULL;
	Canonical_forms = NULL;
	Canonical_form_transporter = NULL;

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

#if 0
	if (U) {
		FREE_OBJECT(U);
	}
#endif

	if (Canonical_form_input) {
		FREE_lint(Canonical_form_input);
	}
	if (Canonical_forms) {
		FREE_lint(Canonical_forms);
	}
	if (Canonical_form_transporter) {
		FREE_int(Canonical_form_transporter);
	}

}


void compute_stabilizer::init(
		substructure_stats_and_selection *SubSt,
		long int *canonical_pts,
		int verbose_level)
// computes canonical_pts[] (lower part), Stab_orbits, Stab, A_on_the_set
// and then calls restricted_action_on_interesting_points,
// compute_canonical_form, compute_automorphism_group,
// and then computes canonical_pts[] (upper part).
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_stabilizer::init" << endl;
		cout << "fname_out = " << SubSt->fname_case_out << endl;
		cout << "SubSt->selected_orbit = " << SubSt->selected_orbit << endl;
		cout << "nb_interesting_subsets = " << SubSt->nb_interesting_subsets << endl;
	}
	compute_stabilizer::SubSt = SubSt;

	if (f_v) {
		cout << "compute_stabilizer::init A=" << endl;
		SubSt->SubC->A->print_info();
	}

	if (f_v) {
		cout << "compute_stabilizer::init A2=" << endl;
		SubSt->SubC->A2->print_info();
	}


	int size;

	SubSt->SubC->PC->get_set(SubSt->SubC->substructure_size, SubSt->selected_orbit, canonical_pts, size);
	if (f_v) {
		cout << "compute_stabilizer::init canonical substructure: ";
		Lint_vec_print(cout, canonical_pts, size);
		cout << endl;
	}


	///

	Stab_orbits = NEW_OBJECT(stabilizer_orbits_and_types);


	if (f_v) {
		cout << "compute_stabilizer::init before Stab_orbits->init" << endl;
	}

	Stab_orbits->init(this, verbose_level);

	if (f_v) {
		cout << "compute_stabilizer::init after Stab_orbits->init" << endl;
	}


	if (f_v) {
		cout << "compute_stabilizer::init before creating Stab" << endl;
	}


	Stab = NEW_OBJECT(groups::sims);
	Stab->init(SubSt->SubC->A, 0 /* verbose_level */);
	Stab->init_trivial_group(0 /* verbose_level - 1*/);



	if (f_v) {
		cout << "compute_stabilizer::init before A2->restricted_action" << endl;
	}
	A_on_the_set = SubSt->SubC->A2->restricted_action(SubSt->Pts, SubSt->nb_pts, 0/*verbose_level*/);
	if (f_v) {
		cout << "compute_stabilizer::init after A2->restricted_action" << endl;
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

#if 0
	if (f_v) {
		cout << "compute_stabilizer::init before init_U" << endl;
	}

	init_U(verbose_level);
	if (f_v) {
		cout << "compute_stabilizer::init after init_U" << endl;
	}
#endif


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

	for (i = 0; i < Stab_orbits->reduced_set_size; i++) {
		canonical_pts[SubSt->SubC->substructure_size + i] =
				Stab_orbits->interesting_points[Stab_orbits->canonical_set1[i]];
	}

	if (f_v) {
		cout << "compute_stabilizer::init input    : ";
		Lint_vec_print(cout, SubSt->Pts, SubSt->nb_pts);
		cout << endl;
		cout << "compute_stabilizer::init canonical: ";
		Lint_vec_print(cout, canonical_pts, SubSt->nb_pts);
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
	data_structures::sorting Sorting;

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
		cout << "compute_stabilizer::compute_automorphism_group "
				"nb_interesting_subsets_reduced = " << Stab_orbits->nb_interesting_subsets_reduced << endl;
		cout << "compute_stabilizer::compute_automorphism_group "
				"nb_times_orbit_count_does_not_match_up = " << nb_times_orbit_count_does_not_match_up << endl;
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
	data_structures::sorting Sorting;


	if (f_v) {
		cout << "compute_stabilizer::compute_automorphism_group_handle_case STABILIZER loop "
				<< cnt2 << " / " << nb_interesting_subsets_rr << " stab_order=" << stab_order << endl;
	}

	cnt = interesting_subsets_rr[cnt2];


	if (f_vv) {
		cout << "compute_stabilizer::compute_automorphism_group_handle_case "
				"before Stab_orbits->map_reduced_set_and_do_orbit_counting" << endl;
	}
	Stab_orbits->map_reduced_set_and_do_orbit_counting(cnt,
			Stab_orbits->interesting_subsets_reduced[cnt],
			Stab_orbits->elt1,
			verbose_level);
	if (f_vv) {
		cout << "compute_stabilizer::compute_automorphism_group_handle_case "
				"after Stab_orbits->map_reduced_set_and_do_orbit_counting" << endl;
	}

	if (f_v) {
		Lint_vec_print(cout, Stab_orbits->reduced_set1, Stab_orbits->reduced_set_size);
		cout << endl;
	}


	if (!Stab_orbits->check_orbit_count()) {
		cout << "compute_automorphism_group_handle_case !Stab_orbits->check_orbit_count()" << endl;
		exit(1);
	}


	if (f_vv) {
		cout << "compute_stabilizer::compute_automorphism_group_handle_case "
				"before Stab_orbits->compute_local_labels" << endl;
	}
	Stab_orbits->compute_local_labels(Stab_orbits->reduced_set1,
			Stab_orbits->reduced_set1_new_labels,
			Stab_orbits->reduced_set_size,
			verbose_level);
	if (f_vv) {
		cout << "compute_stabilizer::compute_automorphism_group_handle_case "
				"after Stab_orbits->compute_local_labels" << endl;
	}
	if (f_v) {
		cout << "local labels:" << endl;
		Lint_vec_print(cout, Stab_orbits->reduced_set1_new_labels, Stab_orbits->reduced_set_size);
		cout << endl;
	}


	groups::sims *Stab0 = NULL;

	if (cnt2 == 0) {


		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case "
					"before compute_canonical_set_and_group" << endl;
		}
		compute_canonical_set_and_group(Stab_orbits->reduced_set1_new_labels,
				Stab_orbits->canonical_set1,
				Stab_orbits->reduced_set_size,
				Stab_orbits->elt2 /*int *transporter*/, Stab0,
				verbose_level);
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case "
					"after compute_canonical_set_and_group" << endl;
		}
		SubSt->SubC->A->element_mult(
				Stab_orbits->elt1,
				Stab_orbits->elt2,
				Stab_orbits->transporter0,
				FALSE);

	}
	else {
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case "
					"before compute_canonical_set" << endl;
		}
		compute_canonical_set(
				Stab_orbits->reduced_set1_new_labels,
				Stab_orbits->canonical_set1,
				Stab_orbits->reduced_set_size,
				Stab_orbits->elt2 /*int *transporter*/,
				verbose_level);
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case "
					"after compute_canonical_set" << endl;
		}
		SubSt->SubC->A->element_mult(Stab_orbits->elt1, Stab_orbits->elt2, T2, FALSE);
	}
	if (f_v) {
		cout << "canonical form:" << endl;
		Lint_vec_print(cout, Stab_orbits->canonical_set1, Stab_orbits->reduced_set_size);
		cout << endl;
	}

	int cmp;

	cmp = Sorting.lint_vec_compare(Stab_orbits->canonical_set1, Stab_orbits->canonical_set2, Stab_orbits->reduced_set_size);

	if (cmp != 0) {
		cout << "compute_stabilizer::compute_automorphism_group_handle_case "
				"the two canonical sets are not the same, error" << endl;
		exit(1);
	}

	if (cnt2 == 0) {
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case "
					"before setup_stabilizer" << endl;
		}
		setup_stabilizer(Stab0, verbose_level);
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case "
					"after setup_stabilizer" << endl;
		}
	}
	else {
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case "
					"before retrieve_automorphism" << endl;
		}
		retrieve_automorphism(verbose_level);
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case "
					"after retrieve_automorphism" << endl;
		}


			// new automorphism is now in new_automorphism


		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case "
					"before add_automorphism" << endl;
		}
		add_automorphism(verbose_level);
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case "
					"after add_automorphism" << endl;
		}


		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case "
					"before update_stabilizer" << endl;
		}
		update_stabilizer(verbose_level);
		if (f_vv) {
			cout << "compute_stabilizer::compute_automorphism_group_handle_case "
					"after update_stabilizer" << endl;
		}


	}

	if (f_v) {
		cout << "compute_stabilizer::compute_automorphism_group_handle_case done" << endl;
	}
}

void compute_stabilizer::setup_stabilizer(groups::sims *Stab0, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer" << endl;
	}

	transporter_witness = NEW_int(A_induced->elt_size_in_int);
	transporter1 = NEW_int(A_induced->elt_size_in_int);
	transporter2 = NEW_int(A_induced->elt_size_in_int);
	T1 = NEW_int(SubSt->SubC->A->elt_size_in_int);
	T1v = NEW_int(SubSt->SubC->A->elt_size_in_int);
	T2 = NEW_int(SubSt->SubC->A->elt_size_in_int);

	K = NEW_OBJECT(groups::sims);
	Kernel_original = NEW_OBJECT(groups::sims);

	K->init(SubSt->SubC->A, 0 /* verbose_level */);
	K->init_trivial_group(0 /*verbose_level - 1*/);

	A_induced->Kernel->group_order(K_go);
	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer kernel has order " << K_go << endl;
	}

	// conjugate the Kernel so that it is a subgroup of
	// the stabilizer of the set Pts[] that we wanted to stabilize originally:
	// remember that elt1 is the transporter that was computed in map_it() above

	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer before Kernel_original->conjugate" << endl;
	}
	Kernel_original->conjugate(A_induced->Kernel->A,
			A_induced->Kernel, Stab_orbits->transporter0,
			FALSE, 0 /*verbose_level - 3*/);
	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer after Kernel_original->conjugate" << endl;
	}
	Kernel_original->group_order(K_go);
	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer after conjugation, kernel has order " << K_go << endl;
	}

	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer adding kernel of order " << K_go
			<< " to the stabilizer (in action " << Stab->A->label << ")" << endl;
	}
	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer before Stab->build_up_group_random_process" << endl;
	}
	Stab->build_up_group_random_process(K, Kernel_original,
		K_go, FALSE, NULL, 0 /*verbose_level - 3*/);
	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer after Stab->build_up_group_random_process" << endl;
	}
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


	Aut = NEW_OBJECT(groups::sims);
	Aut_original = NEW_OBJECT(groups::sims);


	// computes the stabilizer of reduced_set[] in the stabilizer
	// of the k-subset and in the induced action:
	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer before A_induced.make_canonical" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "before make_canonical: ";
		Lint_vec_print(cout, Stab_orbits->reduced_set1_new_labels, Stab_orbits->reduced_set_size);
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
		A_induced->element_print(Stab_orbits->transporter0, cout);
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

	SubSt->SubC->A->element_move(Stab_orbits->transporter0, T1, 0);

	if (FALSE) {
		cout << "T1:" << endl;
		SubSt->SubC->A->element_print(T1, cout);
		cout << endl;
	}
	SubSt->SubC->A->element_invert(T1, T1v, FALSE);

	// T1 := elt1 * elt2
	// moves the_set to the canonical set.

	Aut->group_order(ago);
	Aut_original->conjugate(SubSt->SubC->A /*Aut.A */, Aut, T1,
		TRUE /* f_overshooting_OK */, 0 /*verbose_level - 3*/);
	Aut_original->group_order(ago1);
	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer after conjugation, group in action " << Aut_original->A->label << endl;
		cout << "compute_stabilizer::setup_stabilizer automorphism group order before = " << ago << endl;
		cout << "compute_stabilizer::setup_stabilizer automorphism group order after = " << ago1 << endl;
	}

	ring_theory::longinteger_domain D;

	D.mult(K_go, ago, target_go);
	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer K_go=" << K_go << endl;
		cout << "compute_stabilizer::setup_stabilizer ago=" << ago << endl;
		cout << "compute_stabilizer::setup_stabilizer target_go=" << target_go << endl;
		cout << "compute_stabilizer::setup_stabilizer adding automorphisms to set-stabilizer" << endl;
	}
	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer before Stab->build_up_group_random_process in action ";
		Stab->A->print_info();
		cout << endl;
	}
	Stab->build_up_group_random_process(K, Aut_original,
		target_go, FALSE, NULL, 0 /*verbose_level - 3*/);
	if (f_v) {
		cout << "compute_stabilizer::setup_stabilizer after Stab->build_up_group_random_process" << endl;
	}
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

void compute_stabilizer::restricted_action_on_interesting_points(int verbose_level)
// computes A_induced, induced_go, K_go
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points" << endl;
	}
	// Compute the restricted action on the set of
	// interesting points and call it A_induced :
	// Determine the kernel


	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points "
				"interesting_points=" << endl;
		Lint_vec_print(cout,
				Stab_orbits->interesting_points,
				Stab_orbits->nb_interesting_points);
		cout << endl;
	}

	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points "
				"computing induced action by restriction" << endl;
	}

	A_induced = SubSt->SubC->A2->restricted_action(
			Stab_orbits->interesting_points,
			Stab_orbits->nb_interesting_points,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points "
				"after A2->restricted_action" << endl;
	}
	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points "
				"before A_induced->group_order" << endl;
	}
	A_induced->group_order(induced_go);
	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points "
				"induced_go = " << induced_go << endl;
		cout << "compute_stabilizer::restricted_action_on_interesting_points "
				"A_induced:" << endl;
		A_induced->print_info();
	}

	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points "
				"before A_induced->induced_action_override_sims" << endl;
	}
	A_induced->induced_action_override_sims(
		*SubSt->SubC->A, Stab_orbits->selected_set_stab,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points "
				"after A_induced->induced_action_override_sims" << endl;
	}

	if (!A_induced->f_has_kernel) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points "
				"A_induced does not have kernel" << endl;
		exit(1);
	}
	A_induced->Kernel->group_order(K_go);
	if (f_v) {
		cout << "compute_stabilizer::restricted_action_on_interesting_points "
				"induced action by restriction: group order = " << induced_go << endl;
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



void compute_stabilizer::compute_canonical_form(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int cnt;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "compute_stabilizer::compute_canonical_form" << endl;
	}
	// In the second step, we consider all possible images for the chosen k-subset
	// and try to pick up coset representatives for the subgroup in the
	// full set stabilizer:


	nb_times_orbit_count_does_not_match_up = 0;
	backtrack_nodes_total_in_loop = 0;


	Canonical_form_input = NEW_lint(Stab_orbits->nb_interesting_subsets_reduced * Stab_orbits->reduced_set_size);
	Canonical_forms = NEW_lint(Stab_orbits->nb_interesting_subsets_reduced * Stab_orbits->reduced_set_size);
	Canonical_form_transporter = NEW_int(Stab_orbits->nb_interesting_subsets_reduced * A_induced->elt_size_in_int);


	for (cnt = 0; cnt < Stab_orbits->nb_interesting_subsets_reduced; cnt++) {

		//if (U->is_minimal(cnt, 0 /* verbose_level */)) {
		if (f_v) {
			cout << "compute_stabilizer::compute_canonical_form case " << cnt << " / " << Stab_orbits->nb_interesting_subsets_reduced << endl;
		}
		compute_canonical_form_handle_case(cnt, verbose_level);
		//}

	}

	if (f_v) {
		cout << "compute_stabilizer::compute_canonical_form nb_interesting_subsets_reduced = " << Stab_orbits->nb_interesting_subsets_reduced << endl;
		cout << "compute_stabilizer::compute_canonical_form nb_times_orbit_count_does_not_match_up = " << nb_times_orbit_count_does_not_match_up << endl;
	}


	if (f_v) {
		cout << "Canonical forms:" << endl;
		orbiter_kernel_system::Orbiter->Lint_vec->matrix_print(Canonical_forms,
				Stab_orbits->nb_interesting_subsets_reduced,
				Stab_orbits->reduced_set_size, 2);
		cout << endl;
	}

	interesting_subsets_rr = NEW_lint(Stab_orbits->nb_interesting_subsets_reduced);

	for (cnt = 0; cnt < Stab_orbits->nb_interesting_subsets_reduced; cnt++) {
		if (cnt == 0) {
			Lint_vec_copy(Canonical_forms + cnt * Stab_orbits->reduced_set_size,
					Stab_orbits->canonical_set2, Stab_orbits->reduced_set_size);
			nb_interesting_subsets_rr = 0;
			interesting_subsets_rr[nb_interesting_subsets_rr++] = cnt;
		}
		else {
			int cmp;

			cmp = Sorting.lint_vec_compare(Canonical_forms + cnt * Stab_orbits->reduced_set_size,
					Stab_orbits->canonical_set2, Stab_orbits->reduced_set_size);

			if (cmp > 0) {
				Lint_vec_copy(Canonical_forms + cnt * Stab_orbits->reduced_set_size,
						Stab_orbits->canonical_set2, Stab_orbits->reduced_set_size);
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
		orbiter_kernel_system::Orbiter->Lint_vec->matrix_print(Stab_orbits->canonical_set2, 1, Stab_orbits->reduced_set_size, 2);
		cout << "nb_interesting_subsets_rr = " << nb_interesting_subsets_rr << endl;
		cout << "interesting_subsets_rr:" << endl;
		Lint_vec_print(cout, interesting_subsets_rr, nb_interesting_subsets_rr);
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
		cout << "compute_stabilizer::compute_canonical_form_handle_case "
				"STABILIZER loop " << cnt << " / " << Stab_orbits->nb_interesting_subsets_reduced
				<< " subset_idx=" << Stab_orbits->interesting_subsets_reduced[cnt] << endl;
	}

	if (f_vv) {
		cout << "compute_stabilizer::compute_canonical_form_handle_case "
				"before Stab_orbits->map_subset_and_compute_local_labels" << endl;
	}
	Stab_orbits->map_subset_and_compute_local_labels(cnt, verbose_level - 2);
	if (f_vv) {
		cout << "compute_stabilizer::compute_canonical_form_handle_case "
				"after Stab_orbits->map_subset_and_compute_local_labels" << endl;
	}

	groups::sims *stab;

	if (f_vv) {
		cout << "compute_stabilizer::compute_canonical_form_handle_case before compute_canonical_set" << endl;
	}
	compute_canonical_set_and_group(
			Stab_orbits->reduced_set1_new_labels /* input set */,
			Stab_orbits->canonical_set1 /* output set */,
			Stab_orbits->reduced_set_size,
			Stab_orbits->elt2 /* transporter */,
			stab /* set stabilizer in the induced action */,
			verbose_level - 2);
	if (f_vv) {
		cout << "compute_stabilizer::compute_canonical_form_handle_case after compute_canonical_set" << endl;
	}
	if (FALSE) {
		cout << "canonical form:" << endl;
		Lint_vec_print(cout, Stab_orbits->canonical_set1, Stab_orbits->reduced_set_size);
		cout << endl;
	}

	Lint_vec_copy(Stab_orbits->reduced_set1_new_labels,
			Canonical_form_input + cnt * Stab_orbits->reduced_set_size,
			Stab_orbits->reduced_set_size);

	Lint_vec_copy(Stab_orbits->canonical_set1,
			Canonical_forms + cnt * Stab_orbits->reduced_set_size,
			Stab_orbits->reduced_set_size);

	Int_vec_copy(Stab_orbits->elt2,
			Canonical_form_transporter + cnt * A_induced->elt_size_in_int,
			A_induced->elt_size_in_int);



	if (f_v) {
		cout << "compute_stabilizer::compute_canonical_form_handle_case "
				"STABILIZER loop " << cnt << " / " << Stab_orbits->nb_interesting_subsets_reduced
				<< " subset_idx=" << Stab_orbits->interesting_subsets_reduced[cnt] << " done" << endl;
	}

}

void compute_stabilizer::compute_canonical_set(long int *set_in, long int *set_out, int sz,
		int *transporter, int verbose_level)
// calls A_induced->make_canonical and computes a transporter.
// does not compute the set stabilizer
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

	groups::sims *my_Aut;

	my_Aut = NEW_OBJECT(groups::sims);

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
		int *transporter, groups::sims *&stab, int verbose_level)
// calls A_induced->make_canonical and computes a transporter and the set stabilizer
{
	int f_v = (verbose_level >= 1);
	int nb_nodes;

	if (f_v) {
		cout << "compute_stabilizer::compute_canonical_set_and_group" << endl;
	}
	if (f_v) {
		cout << "compute_stabilizer::compute_canonical_set_and_group before A_induced->make_canonical" << endl;
	}


	stab = NEW_OBJECT(groups::sims);

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

#if 0
void compute_stabilizer::init_U(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_stabilizer::init_U before U->init" << endl;
	}
	U = NEW_OBJECT(union_find_on_k_subsets);

	U->init(SubSt->SubC->A2, Stab,
			SubSt->Pts, SubSt->nb_pts, SubSt->SubC->substructure_size,
			Stab_orbits->interesting_subsets_reduced, Stab_orbits->nb_interesting_subsets_reduced,
		verbose_level);


	if (f_v) {
		cout << "compute_stabilizer::init_U after U->init" << endl;
	}
}
#endif




void compute_stabilizer::update_stabilizer(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_domain D;
	int cmp;

	Stab->group_order(new_stab_order);
	cmp = D.compare_unsigned(new_stab_order, stab_order);
	if (cmp) {

		if (f_v) {
			cout << "compute_stabilizer::update_stabilizer new stabilizer order is " << new_stab_order << endl;
		}
		new_stab_order.assign_to(stab_order);


		groups::strong_generators *Strong_gens;

		Strong_gens = NEW_OBJECT(groups::strong_generators);
		Strong_gens->init_from_sims(Stab, 0);

		if (f_v) {
			Strong_gens->print_generators(cout);
		}
		FREE_OBJECT(Strong_gens);

		//FREE_OBJECT(U);

		//init_U(verbose_level);
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
	int drop_out_level, image; //, f_added;

	//cout << "stabilizer transversal length:" << endl;
	//Stab->print_transversal_lengths();
	//cout << endl;

	if (Stab->strip(Stab_orbits->new_automorphism, Stab_orbits->Elt4, drop_out_level, image, 0/*verbose_level - 3*/)) {
		if (f_v4) {
			cout << "compute_stabilizer::main_loop_handle_case element strips through" << endl;
			if (FALSE) {
				cout << "compute_stabilizer residue:" << endl;
				SubSt->SubC->A->element_print(Stab_orbits->Elt4, cout);
				cout << endl;
			}
		}
		//f_added = FALSE;
		//Stab->closure_group(2000, verbose_level - 1);
	}
	else {
		//f_added = TRUE;
		if (f_v4) {
			cout << "compute_stabilizer::main_loop_handle_case element needs to be inserted at level = "
				<< drop_out_level << " with image " << image << endl;
			if (FALSE) {
				SubSt->SubC->A->element_print(Stab_orbits->Elt4, cout);
				cout  << endl;
			}
		}
		if (!SubSt->SubC->A2->check_if_in_set_stabilizer(Stab_orbits->Elt4, SubSt->nb_pts, SubSt->Pts, 0/*verbose_level*/)) {
			cout << "compute_stabilizer::main_loop_handle_case residue does not stabilize original set" << endl;
			exit(1);
		}
		Stab->add_generator_at_level(Stab_orbits->Elt4, drop_out_level, 0/*verbose_level - 3*/);
		if (f_v) {
			cout << "compute_stabilizer::main_loop_handle_case calling closure_group" << endl;
		}
		Stab->closure_group(2000, 0 /*verbose_level - 1*/);
	}
}

void compute_stabilizer::retrieve_automorphism(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	SubSt->SubC->A->element_mult(T2, T1v, Stab_orbits->new_automorphism, FALSE);
	if (f_v) {
		cout << "compute_stabilizer::retrieve_automorphism found automorphism" << endl;
	}
	if (FALSE) {
		cout << "compute_stabilizer::retrieve_automorphism automorphism:" << endl;
		SubSt->SubC->A->element_print(Stab_orbits->new_automorphism, cout);
		cout << endl;
	}
	if (!SubSt->SubC->A2->check_if_in_set_stabilizer(Stab_orbits->new_automorphism,
			SubSt->nb_pts, SubSt->Pts,
			verbose_level)) {
		cout << "compute_stabilizer::retrieve_automorphism does not stabilize original set" << endl;
		exit(1);
	}
	if (FALSE) {
		cout << "compute_stabilizer::retrieve_automorphism is in the set stabilizer" << endl;
	}

	if (f_v) {
		cout << "the automorphism is: " << endl;
		A_on_the_set->element_print(Stab_orbits->new_automorphism, cout);
		cout << endl;
		cout << "the automorphism acts on the set as: " << endl;
		A_on_the_set->element_print_as_permutation(Stab_orbits->new_automorphism, cout);
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
			Stab_orbits->reduced_set_size, Stab_orbits->reduced_set2_new_labels,
			Stab_orbits->canonical_set2, transporter2, nb_nodes,
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
		Lint_vec_print(cout, Stab_orbits->canonical_set2, Stab_orbits->reduced_set_size);
		cout << endl;
	}
	if (FALSE) {
		cout << "compute_stabilizer::make_canonical_second_set transporter2:" << endl;
		A_induced->element_print(transporter2, cout);
		cout << endl;
		A_induced->element_print_as_permutation(transporter2, cout);
		cout << endl;
	}
	SubSt->SubC->A->mult(Stab_orbits->elt2, transporter2, T2);
	if (FALSE) {
		cout << "compute_stabilizer::make_canonical_second_set T2:" << endl;
		SubSt->SubC->A->element_print(T2, cout);
		cout << endl;
		//A_induced.element_print_as_permutation(transporter2, cout);
		//cout << endl;
	}
	if (f_v) {
		cout << "compute_stabilizer::make_canonical_second_set done" << endl;
	}
}





void compute_stabilizer::report(std::ostream &ost)
{
	ost << "Input set of size " << SubSt->nb_pts << " : ";
	ost << "$";
	Lint_vec_print_fully(ost, SubSt->Pts, SubSt->nb_pts);
	ost << "$";
	ost << "\\\\" << endl;

	ost << "Level = " << SubSt->SubC->substructure_size << "\\\\" << endl;
	ost << "SubSt->selected_orbit = " << SubSt->selected_orbit << "\\\\" << endl;
	ost << "Nb_interesting_subsets = " << SubSt->nb_interesting_subsets << "\\\\" << endl;

	ost << "interesting_subsets: ";
	//ost << "$";
	Lint_vec_print_fully(ost, SubSt->interesting_subsets, SubSt->nb_interesting_subsets);
	//ost << "$";
	ost << "\\\\" << endl;

	ost << "Classification of small sets: ";

	poset_classification::poset_classification_report_options Opt;

	SubSt->SubC->PC->report(ost, &Opt, 0 /* verbose_level */);

}

void compute_stabilizer::print_canonical_sets()
{
	int i;

	for (i = 0; i < Stab_orbits->nb_interesting_subsets_reduced; i++) {
		cout << "STABILIZER loop " << i << " / " << Stab_orbits->nb_interesting_subsets_reduced
				<< " subset_idx=" << Stab_orbits->interesting_subsets_reduced[i] << endl;
		cout << "input set: ";
		Lint_vec_print(cout,
					Canonical_form_input + i * Stab_orbits->reduced_set_size,
					Stab_orbits->reduced_set_size);
		cout << endl;
		cout << "output set: ";
		Lint_vec_print(cout,
				Canonical_forms + i * Stab_orbits->reduced_set_size,
					Stab_orbits->reduced_set_size);
		cout << endl;
		cout << "transporter: ";
		Int_vec_print(cout,
				Canonical_form_transporter + i * A_induced->elt_size_in_int,
				A_induced->elt_size_in_int);
		cout << endl;

	}
}





}}
