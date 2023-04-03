// poset_orbit_node_downstep.cpp
//
// Anton Betten
// July 23, 2007
//
// this is the downstep for action on subsets only

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {

void poset_orbit_node::compute_flag_orbits(
	poset_classification *gen,
	int lvl, 
	int f_create_schreier_vector,
	int f_use_invariant_subset_if_available, 
	int f_implicit_fusion, 
	int verbose_level)
// Called from poset_classification::compute_flag_orbits if we are acting on sets
// (i.e., not on subspaces).
// Calls downstep_orbits, 
// downstep_orbit_test_and_schreier_vector and 
// downstep_implicit_fusion
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int nb_orbits;
	int good_orbits1, nb_points1;
	int f_using_invariant_subset = false;
	groups::schreier Schreier;
	actions::action *AR;

	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits" << endl;
		store_set(gen, lvl - 1); // stores a set of size lvl
		gen->print_level_info(lvl, node);
		cout << " : Downstep for ";
		print_set(gen);
		cout << " verbose_level=" << verbose_level << endl;
		if (f_vvv) {
			print_set_verbose(gen);
		}
	}

	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits "
				"before schreier_forest" << endl;
	}
	schreier_forest(gen, Schreier, AR,
		lvl, 
		f_use_invariant_subset_if_available, 
		f_using_invariant_subset, 
			verbose_level - 1);
	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits "
				"after schreier_forest" << endl;
	}

#if 0
	if (node == 50) {
		cout << "poset_orbit_node::compute_flag_orbits "
				"after downstep_orbits" << endl;
		gen->root[49].print_extensions(cout);
	}
#endif
	nb_orbits = Schreier.nb_orbits;


#if 0
	// ToDo:
	save_schreier_forest(
		gen,
		&Schreier,
		verbose_level);

	draw_schreier_forest(
			gen,
			&Schreier,
			f_using_invariant_subset, AR,
			verbose_level);
#endif

	
	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits "
				"before downstep_orbit_test_and_schreier_vector" << endl;
	}
	downstep_orbit_test_and_schreier_vector(
		gen, &Schreier, AR,
		lvl, 
		f_use_invariant_subset_if_available, 
		f_using_invariant_subset,
		f_create_schreier_vector,
		good_orbits1, nb_points1, 
		verbose_level - 1);


	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits "
				"after downstep_orbit_test_and_schreier_vector" << endl;
	}

#if 0
	if (node == 50) {
		cout << "poset_orbit_node::compute_flag_orbits "
				"after downstep_orbit_test_and_schreier_vector" << endl;
		gen->root[49].print_extensions(cout);
	}
#endif

	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits "
				"before downstep_implicit_fusion" << endl;
	}
	downstep_implicit_fusion(
		gen, Schreier, AR, f_using_invariant_subset,
		lvl, 
		f_implicit_fusion, 
		good_orbits1, nb_points1, 
		verbose_level - 1);
	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits "
				"after downstep_implicit_fusion" << endl;
	}

#if 0
	if (node == 50) {
		cout << "poset_orbit_node::compute_flag_orbits "
				"after downstep_implicit_fusion" << endl;
		gen->root[49].print_extensions(cout);
	}
#endif

	// ToDo:
	//save_shallow_schreier_forest(gen, verbose_level);

	
	if (f_vvv) {
		gen->print_level_info(lvl, node);
		cout << " : calling find_extensions" << endl;
	}
	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits "
				"before find_extensions" << endl;
	}
	find_extensions(
		gen, Schreier, AR, f_using_invariant_subset,
		lvl, 
		verbose_level - 2);
	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits "
				"after find_extensions" << endl;
	}
	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << " : after test_orbits and find_extensions, "
				"we have " << nb_extensions << " extensions" << endl;
	}

	if (false) {
		print_extensions(gen);
	}
	
	
	
	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << " : found " << nb_extensions << " extensions (out of "
				<< nb_orbits << " orbits) with "
				<< nb_extension_points() << " points " << endl;
	}

	FREE_OBJECT(AR);

	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits done" << endl;
	}
}


void poset_orbit_node::compute_schreier_vector(
	poset_classification *gen,
	int lvl, int verbose_level)
// called from generator::recreate_schreier_vectors_at_level
// and from generator::count_live_points
// calls downstep_apply_early_test
// and check_orbits
// and Schreier.get_schreier_vector
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	groups::schreier *Schreier;
	//int f_trivial_group;
	//int f_using_invariant_subset = false;
	//int f_use_incremental_test_func_if_available = true;
	long int *candidates = NULL;
	int nb_candidates;
	long int *live_points = NULL;
	int nb_live_points;
	actions::action *AR = NULL;

	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector" << endl;
	}
	

	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector: "
				"before get_candidates" << endl;
	}
	get_candidates(
			gen,
			lvl,
			candidates, nb_candidates,
			verbose_level - 1);
	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector: "
				"after get_candidates, nb_candidates=" << nb_candidates << endl;
	}

	live_points = NEW_lint(nb_candidates);

	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector: "
				"before downstep_apply_early_test" << endl;
	}
	downstep_apply_early_test(gen, lvl,
			nb_candidates, candidates,
			live_points, nb_live_points,
			verbose_level);
	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector: "
				"after downstep_apply_early_test, nb_live_points=" << nb_live_points << endl;
	}


	std::string label_of_set;

	label_of_set.assign("live_points");

	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector: "
				"before gen->Poset->A2->create_induced_action_by_restriction" << endl;
	}
	AR = gen->get_A2()->Induced_action->create_induced_action_by_restriction(
		NULL /*sims *old_G*/,
		nb_live_points, live_points, label_of_set,
		false /*f_induce_action*/,
		verbose_level - 2);
	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector: "
				"after gen->Poset->A2->create_induced_action_by_restriction" << endl;
	}

	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector: "
				"before gen->Poset->A2->create_induced_action_by_restriction" << endl;
	}
	Schreier = NEW_OBJECT(groups::schreier);

	Schreier->init(AR, verbose_level - 2);



#if 0
	if (lvl &&
		gen->root[prev].Schreier_vector) {

		int n = gen->root[prev].get_nb_of_live_points();
		int *subset = gen->root[prev].live_points();


		f_using_invariant_subset = true;

		candidates = NEW_lint(n);
		
		downstep_apply_early_test(gen, lvl, 
			n, subset, 
			candidates, nb_candidates, 
			verbose_level);


		//action *create_induced_action_by_restriction(
		//		sims *S, int size, int *set, int f_induce,
		//		int verbose_level);

		AR = gen->Poset->A2->create_induced_action_by_restriction(
			NULL /*sims *old_G*/,
			nb_candidates, candidates,
			false /*f_induce_action*/,
			verbose_level - 2);

		//if (f_vv) {
		//	cout << "calling orbits_on_invariant_subset_fast" << endl;
		//	}
		//Schreier.orbits_on_invariant_subset_fast(
		// n, subset, verbose_level);
		Schreier.init(AR, verbose_level - 2);
#if 0
		if (f_vv) {
			cout << "poset_orbit_node::compute_schreier_vector "
					"the stabilizer has " << Schreier.nb_orbits
					<< " orbits on the live point" << endl;
			}
#endif
		}
	else if (lvl == 0) {
		long int *subset;
		int i;
		int n = gen->Poset->A2->degree;
		
		subset = NEW_lint(n);
		for (i = 0; i < n; i++) {
			subset[i] = i;
			}
		
		f_using_invariant_subset = true;

		candidates = NEW_lint(n);
		
		downstep_apply_early_test(gen, lvl, 
			n, subset, 
			candidates, nb_candidates, 
			verbose_level);
		AR = gen->Poset->A2->create_induced_action_by_restriction(
			NULL /*sims *old_G*/,
			nb_candidates, candidates,
			false /*f_induce_action*/,
			verbose_level - 2);
		//if (f_vv) {
		//	cout << "calling orbits_on_invariant_subset_fast" << endl;
		//	}
		//Schreier.orbits_on_invariant_subset_fast(n, subset, verbose_level);
		Schreier.init(AR, verbose_level - 2);
		FREE_lint(subset);
#if 0
		if (f_vv) {
			cout << "poset_orbit_node::compute_schreier_vector "
					"the stabilizer has " << Schreier.nb_orbits
					<< " orbits on the live point" << endl;
			}
#endif
		}
	else {
		f_using_invariant_subset = false;
		Schreier.init(gen->Poset->A2, verbose_level - 2);
		}
#endif

	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector "
				"before Schreier->init_generators_by_handle" << endl;
	}

	std::vector<int> gen_handle;

	get_strong_generators_handle(gen_handle, verbose_level - 2);


	Schreier->init_generators_by_handle(
			gen_handle,
			verbose_level - 1);
	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector "
				"before Schreier->init_generators_by_handle" << endl;
	}

	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector "
				"calling compute_all_point_orbits" << endl;
	}
	if (lvl == 0) {
		Schreier->compute_all_point_orbits(0 /*verbose_level - 1 */);
	}
	else {
		Schreier->compute_all_point_orbits(0 /* verbose_level */);
	}
	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector "
				"the stabilizer has " << Schreier->nb_orbits
				<< " orbits overall" << endl;
		}
	
	check_orbits(gen, Schreier, AR,
		lvl,
		verbose_level - 2);
	
	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector "
				"the stabilizer has " << Schreier->nb_orbits
			<< " good orbits with "
			<< Schreier->sum_up_orbit_lengths() << " points" << endl;
		}

	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector "
				"before create_schreier_vector_wrapper" << endl;
	}

		// ToDo: schreier vector strategy
		// if lvl < 5, do the ai method
		// otherwise do the default method

	create_schreier_vector_wrapper(
			gen,
			true /* f_create_schreier_vector */,
			Schreier, verbose_level - 1);

	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector "
				"after create_schreier_vector_wrapper" << endl;
	}

	//Schreier.get_schreier_vector(sv,
	//		f_trivial_group, f_compact);
	//Schreier.test_sv(gen->A, hdl_strong_generators,
	// sv, f_compact, verbose_level);

	if (true /* f_using_invariant_subset */) {
		if (f_v) {
			cout << "poset_orbit_node::compute_schreier_vector "
					"before relabel_schreier_vector" << endl;
		}
		relabel_schreier_vector(AR, verbose_level - 1);
		if (f_v) {
			cout << "poset_orbit_node::compute_schreier_vector: "
					"after relabeling: Schreier vector is" << endl;
			//Schreier_vector->print();
			}
	}

	FREE_OBJECT(AR);
	FREE_OBJECT(Schreier);

	if (candidates) {
		FREE_lint(candidates);
		}
	if (live_points) {
		FREE_lint(live_points);
		}
	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector: "
				"Schreier vector has been computed" << endl;
		}	
}


void poset_orbit_node::get_candidates(
	poset_classification *gen,
	int lvl,
	long int *&candidates, int &nb_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "poset_orbit_node::get_candidates" << endl;
	}

	if (lvl &&
		gen->node_has_schreier_vector(prev)) {

		nb_candidates = gen->get_node(prev)->get_nb_of_live_points();
		int *subset = gen->get_node(prev)->live_points();

		candidates = NEW_lint(nb_candidates);
		for (i = 0; i < nb_candidates; i++) {
			candidates[i] = subset[i];
		}
	}
	else {
		nb_candidates = gen->get_A2()->degree;

		candidates = NEW_lint(nb_candidates);
		for (i = 0; i < nb_candidates; i++) {
			candidates[i] = i;
		}
	}
	if (f_v) {
		cout << "poset_orbit_node::get_candidates done" << endl;
	}
}


// #############################################################################
// first level under downstep:
// #############################################################################



void poset_orbit_node::schreier_forest(
	poset_classification *gen, groups::schreier &Schreier,
	actions::action *&AR,
	int lvl, 
	int f_use_invariant_subset_if_available, 
	int &f_using_invariant_subset, 
	int verbose_level)
// calls downstep_get_invariant_subset, downstep_apply_early_test, 
// and AR.induced_action_by_restriction
// if f_use_invariant_subset_if_available and f_using_invariant_subset
//
// Sets up the schreier data structure Schreier 
// If f_using_invariant_subset, we will use the 
// restricted action AR, otherwise the action gen->A2
// In this action, the orbits are computed using 
// Schreier.compute_all_point_orbits
// and possibly printed using downstep_orbits_print
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v4 = (verbose_level >= 4);
	int n = 0;
	long int *subset = NULL;
	long int *candidates = NULL;
	int nb_candidates = 0;

	f_using_invariant_subset = false;

	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << "poset_orbit_node::schreier_forest" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	
	

	if (f_use_invariant_subset_if_available) {
		if (lvl == 0) {
			if (f_v) {
				cout << "poset_orbit_node::schreier_forest we are trying "
						"to find an invariant subset" << endl;
				}
			}
		f_using_invariant_subset = downstep_get_invariant_subset(
			gen, 
			lvl, 
			n, subset,
			verbose_level - 2);




		if (lvl == 0 && !f_using_invariant_subset) {
			cout << "poset_orbit_node::schreier_forest we are trying "
						"to find an invariant subset. We did not find an invariant subset" << endl;
			}
		}
	else {
		if (lvl == 0) {
			cout << "poset_orbit_node::schreier_forest we are NOT using "
					"an invariant subset" << endl;
			}
		}
	
	if (f_using_invariant_subset) {

		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : poset_orbit_node::schreier_forest we are using an invariant subset : "
					"live points at the predecessor node: number=" << n;
			if (f_v4) {
				cout << " : ";
				if (n < 100) {
					Lint_vec_print(cout, subset, n);
				}
				else {
					cout << "too large to print";
				}
				cout << endl; 
				}
			else {
				cout << endl; 
				}
			}
		candidates = NEW_lint(n);
		
		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : poset_orbit_node::schreier_forest before downstep_apply_early_test" << endl;
		}
		downstep_apply_early_test(gen, lvl, 
			n, subset, 
			candidates, nb_candidates, 
			verbose_level - 2);
		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : poset_orbit_node::schreier_forest after downstep_apply_early_test" << endl;
		}

		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : poset_orbit_node::schreier_forest live points after downstep_apply_early_test: "
					"number=" << nb_candidates;
			Lint_vec_print(cout, candidates, nb_candidates);
			cout << " reduced from a set of size " << nb_candidates << endl;
#if 0
			if (f_v4) {
				cout << " : ";
				if (nb_candidates < 100) {
					int_vec_print(cout, candidates, nb_candidates);
				}
				else {
					cout << "too large to print";
				}
				cout << endl; 
			}
			else {
				cout << endl;
			}
#endif
		}

		std::string label_of_set;

		label_of_set.assign("candidates");


		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : poset_orbit_node::schreier_forest before create_induced_action_by_restriction" << endl;
		}
		AR = gen->get_A2()->Induced_action->create_induced_action_by_restriction(
			NULL /*sims *old_G*/,
			nb_candidates, candidates, label_of_set,
			false /*f_induce_action*/,
			verbose_level - 2);
		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : poset_orbit_node::schreier_forest after create_induced_action_by_restriction" << endl;
			gen->print_level_info(lvl, node);
			cout << " : poset_orbit_node::schreier_forest created action " << AR->label << endl;
		}
		
		if (f_vv) {
			cout << "poset_orbit_node::schreier_forest created restricted action ";
			AR->print_info();
			}
		Schreier.init(AR /*gen->A2*/, verbose_level - 2);
		}
	else {
		gen->print_level_info(lvl, node);
		cout << " : poset_orbit_node::schreier_forest we are NOT using an invariant subset" << endl;
		Schreier.init(gen->get_A2(), verbose_level - 2);
		}


	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << " : poset_orbit_node::schreier_forest initializing generators. There are "
				<< nb_strong_generators  << " strong generators" << endl;
		//cout << "hdl_strong_generators=";
		//int_vec_print(cout, hdl_strong_generators, nb_strong_generators);
		//cout << endl;
		}


	std::vector<int> gen_handle;

	get_strong_generators_handle(gen_handle, verbose_level - 2);

	Schreier.init_generators_by_handle(
			gen_handle,
			verbose_level - 1);

	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << " : poset_orbit_node::schreier_forest calling Schreier.compute_all_point_orbits "
				"for a set of size " << Schreier.A->degree << endl;
		}


	if (false) {
		gen->print_level_info(lvl, node);
		cout << " : generators:" << endl;
		Schreier.print_generators();
		}


	//Schreier.compute_all_point_orbits_with_preferred_labels(
	// n, subset, verbose_level - 4);


	if (gen->get_control()->f_preferred_choice) {

		for (int i = 0; i < gen->get_control()->preferred_choice.size(); i++) {
			if (gen->get_control()->preferred_choice[i][0] == node) {
				Schreier.init_preferred_choice_function(
						poset_classification_control_preferred_choice_function,
						gen, node,
						verbose_level);
			}
		}
	}


#if 0
	if (lvl == 0) {
		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : poset_orbit_node::schreier_forest before Schreier.compute_all_point_orbits" << endl;
		}
		Schreier.compute_all_point_orbits( 0 /*verbose_level - 1 */);
		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : poset_orbit_node::schreier_forest after Schreier.compute_all_point_orbits" << endl;
		}
	}
	else {
		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : poset_orbit_node::schreier_forest before Schreier.compute_all_point_orbits" << endl;
		}
		Schreier.compute_all_point_orbits(0 /*verbose_level - 4*/);
		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : poset_orbit_node::schreier_forest after Schreier.compute_all_point_orbits" << endl;
		}
	}
#else
	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << " : poset_orbit_node::schreier_forest before Schreier.compute_all_point_orbits" << endl;
	}
	Schreier.compute_all_point_orbits( 0 /*verbose_level - 1 */);
	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << " : poset_orbit_node::schreier_forest after Schreier.compute_all_point_orbits" << endl;
	}
#endif

	if (false) {
		int f_print_orbits = false;
		if (f_vv) {
			f_print_orbits = true;
			}
		//int max_orbits = 50;
		//int max_points_per_orbit = 25;
		if (f_using_invariant_subset) {
			downstep_orbits_print(gen, 
				&Schreier, AR, lvl,
				f_print_orbits, 
				gen->max_number_of_orbits_to_print(),
				gen->max_number_of_points_to_print_in_orbit());
			}
		}
	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << "poset_orbit_node::schreier_forest: we found "
				<< Schreier.nb_orbits << " orbits" << endl;
	}
	if (subset) {
		FREE_lint(subset);
	}
	if (candidates) {
		FREE_lint(candidates);
	}
}

void poset_orbit_node::downstep_orbit_test_and_schreier_vector(
	poset_classification *gen, groups::schreier *Schreier,
	actions::action *AR,
	int lvl, 
	int f_use_invariant_subset_if_available, 
	int f_using_invariant_subset,
	int f_create_schreier_vector,
	int &nb_good_orbits, int &nb_points, 
	int verbose_level)
// called from downstep once downstep_orbits is completed
// Calls check_orbits_wrapper and create_schreier_vector_wrapper
// The order in which these two functions are called matters.
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_print_orbits = false;
	if (f_vvv) {
		f_print_orbits = true;
		}
	int max_orbits = 50;
	int max_points_per_orbit = 25;

	if (verbose_level > 4) {
		max_points_per_orbit = INT_MAX;
		}

	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << "poset_orbit_node::downstep_orbit_"
				"test_and_schreier_vector" << endl;
		}
	if (f_use_invariant_subset_if_available) {
		check_orbits_wrapper(gen, Schreier,
			AR,
			lvl, nb_good_orbits, nb_points, 
			verbose_level - 1);

		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : after check_orbits_wrapper:" << endl;
			cout << "nb_good_orbits=" << nb_good_orbits << endl;
			cout << "nb_points=" << nb_points << endl;
			}
		if (false) {
			downstep_orbits_print(gen, 
				Schreier, AR, lvl, 
				f_print_orbits, 
				max_orbits, max_points_per_orbit);
			}

		create_schreier_vector_wrapper(
			gen,
			f_create_schreier_vector, 
			Schreier, 
			verbose_level - 1);
		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : after creating Schreier vector." << endl;
			}

		if (f_using_invariant_subset && f_create_schreier_vector) {
			relabel_schreier_vector(AR, verbose_level - 1);
			if (f_v) {
				gen->print_level_info(lvl, node);
				cout << " : after relabeling Schreier vector." << endl;
				//Schreier_vector->print();
				}
			}
		}
	else {
		// in this case, we need all orbits in the schreier vector.
		// that's why we do the orbit checking afterwards
		create_schreier_vector_wrapper(
			gen,
			f_create_schreier_vector, 
			Schreier, 
			verbose_level - 1);
		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : after creating Schreier vector." << endl;
			}

		check_orbits_wrapper(gen,
			Schreier, AR,
			lvl, nb_good_orbits, nb_points, 
			verbose_level - 1);



		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : after check_orbits_wrapper:" << endl;
			cout << "nb_good_orbits=" << nb_good_orbits << endl;
			cout << "nb_points=" << nb_points << endl;
		}
		if (false) {
			downstep_orbits_print(gen, 
				Schreier, AR, lvl, 
				f_print_orbits, 
				max_orbits, max_points_per_orbit);
		}
	}
}

void poset_orbit_node::downstep_implicit_fusion(
	poset_classification *gen, groups::schreier &Schreier,
	actions::action *AR,
	int f_using_invariant_subset,
	int lvl, 
	int f_implicit_fusion, 
	int good_orbits1, int nb_points1, 
	int verbose_level)
// called from downstep, 
// once downstep_orbit_test_and_schreier_vector is done
// calls test_orbits_for_implicit_fusion
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << "poset_orbit_node::downstep_implicit_fusion" << endl;
		}
	if (f_implicit_fusion) {
		int good_orbits2, nb_points2;
		
		if (f_vv) {
			gen->print_level_info(lvl, node);
			cout << " : calling test_orbits_for_implicit_fusion" << endl;
			}

		test_orbits_for_implicit_fusion(gen, 
			Schreier, AR,
			f_using_invariant_subset, lvl,
			verbose_level - 3);

		good_orbits2 = Schreier.nb_orbits;
		nb_points2 = Schreier.sum_up_orbit_lengths();

		if (f_vv) {
			gen->print_level_info(lvl, node);
			cout << " : after eliminating implicit fusion nodes: "
				"the stabilizer has " << good_orbits2
				<< " good orbits with "
				<< nb_points2 << " points" << endl;
			cout << "we have eliminated " << good_orbits1 - good_orbits2 
				<< " implicit fusion orbits with "
				<< nb_points1 - nb_points2 << " points" << endl;
			}
		}
	else {
		if (f_vv) {
			gen->print_level_info(lvl, node);
			cout << " : no implicit fusion" << endl;
			}
		}
}


void poset_orbit_node::find_extensions(
		poset_classification *gen,
		groups::schreier &O,
		actions::action *AR, int f_using_invariant_subset,
	int lvl, 
	int verbose_level)
// called by downstep
// prepares all extension nodes and marks them as unprocessed.
// we are at depth lvl, i.e., currently, we have a set of size lvl.
// removes implicit fusion orbits
// removes orbits that are contained in the set
{
	//verbose_level = 2;
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);
	int f_vvv = false; //(verbose_level >= 3);
	int h, k, fst, /*len,*/ rep;
	induced_actions::action_by_restriction *ABR = NULL;

	if (f_using_invariant_subset) {
		ABR = AR->G.ABR;
		}
	
	if (f_v) {
		cout << "poset_orbit_node::find_extensions computing all possible "
				"extensions (out of " << O.nb_orbits << " orbits)" << endl;
		}
	if (f_vv) {
		cout << "the stabilizer orbits are:" << endl;
		cout << "i : orbit length : representative" << endl;
		for (k = 0; k < O.nb_orbits; k++) {
			fst = O.orbit_first[k];
			rep = O.orbit[fst];
			if (f_using_invariant_subset) {
				rep = ABR->points[rep];
				}
			cout << k << " : " << O.orbit_len[k] << " : " << rep << endl;
			}
		}
	E = NEW_OBJECTS(extension, O.nb_orbits);

	store_set(gen, lvl - 1);

	nb_extensions = 0;
	for (k = 0; k < O.nb_orbits; k++) {
		fst = O.orbit_first[k];
		//len = O.orbit_len[k];
		rep = O.orbit[fst];
		if (f_using_invariant_subset) {
			rep = ABR->points[rep];
			}

#if 0
		if (f_implicit_fusion) {
			// use implicit fusion nodes
			if (lvl) {
				if (rep <= pt) {
					if (f_vv) {
						cout << "orbit " << k << " is not accepted because "
							<< "we use implicit fusion nodes and " 
							<< rep << " is less than " 
							<< pt << endl;
						}
					continue;
					}
				if (f_vv) {
					cout << "orbit " << k << " is accepted" << endl;
					}
				}
			}
		else {
#endif


#if 1

			// we need to check whether the point is already in the set:
			int ii;
			
			for (ii = 0; ii < lvl; ii++) {
				if (gen->get_S()[ii] == rep)
					break;
				}
			if (ii < lvl) {
				if (f_vv) {
					cout << "orbit " << k << " is in the set "
							"so we skip" << endl;
					}
				continue;
				}
			if (f_vv) {
				cout << "orbit " << k << " is accepted" << endl;
				}
#endif



#if 0
			}
#endif

			

		E[nb_extensions].set_pt(rep);
		E[nb_extensions].set_orbit_len(O.orbit_len[k]);
		//E[nb_extensions].type = EXTENSION_TYPE_UNPROCESSED;
		//E[nb_extensions].data = 0;
		//E[nb_extensions].data1 = 0;
		//E[nb_extensions].data2 = 0;
		nb_extensions++;
		}
	//nb_extensions = O.nb_orbits;
	

	if (f_vv) {
		cout << "found " << nb_extensions << " extensions with "
				<< nb_extension_points() << " points (out of "
				<< O.nb_orbits << " orbits)" << endl;
		}
#if 0
	if (node == 49) {
		cout << "Node 49:" << endl;
		print_extensions(cout);
		}
	else if (node > 49) {
		cout << "Node 49:" << endl;
		gen->root[49].print_extensions(cout);
		}
#endif

	if (f_vvv) {
		cout << "i : orbit_length : representing point" << endl;
		for (h = 0; h < nb_extensions; h++) {
			cout << h << " : " << E[h].get_orbit_len() << " : " << E[h].get_pt() << endl;
			}
		}
}



// #############################################################################
// second level under downstep:
// #############################################################################


int poset_orbit_node::downstep_get_invariant_subset(
	poset_classification *gen,
	int lvl, 
	int &n, long int *&subset,
	int verbose_level)
// called from schreier_forest
// Gets the live points at the present node.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret = false;
	int i;
	data_structures::sorting Sorting;
	

	n = -1;
	subset = NULL;

	if (f_v) {
		cout << "poset_orbit_node::downstep_get_invariant_subset" << endl;
		}

	if (gen->has_base_case() && lvl == gen->get_Base_case()->size) {
		if (f_vv) {
			gen->print_level_info(lvl, node);
			cout << "poset_orbit_node::downstep_get_invariant_subset "
					"Getting live points for the starter" << endl;
			}
		n = gen->get_Base_case()->nb_live_points;
		//subset = gen->starter_live_points;
		subset = NEW_lint(n);
		for (i = 0; i < n; i++) {
			subset[i] = gen->get_Base_case()->live_points[i];
		}
		ret = true;
		goto the_end;
	}


	else if (lvl == 0 && gen->has_invariant_subset_for_root_node()) {
		cout << "poset_orbit_node::downstep_get_invariant_subset "
				"root node has an invariant subset of size " << n << endl;
		n = gen->size_of_invariant_subset_for_root_node();
		subset = NEW_lint(n);
		for (i = 0; i < n; i++) {
			subset[i] = gen->get_invariant_subset_for_root_node()[i];
		}
		ret = true;
		goto the_end;
	}

	else if (lvl == 0) {
		n = gen->get_A2()->degree;
		subset = NEW_lint(n);
		int i;
		for (i = 0; i < n; i++) {
			subset[i] = i;
			}
		ret = true;
		goto the_end;
	}

	else if (lvl && gen->node_has_schreier_vector(prev)) {
		
		if (f_vv) {
			gen->print_level_info(lvl, node);
			cout << "poset_orbit_node::downstep_get_invariant_subset "
					"Getting live points from previous level" << endl;
			}
		n = gen->get_node(prev)->get_nb_of_live_points();
		//subset = gen->root[prev].live_points();
		subset = NEW_lint(n);
		for (i = 0; i < n; i++) {
			subset[i] = gen->get_node(prev)->live_points()[i];
		}
		ret = true;
		goto the_end;
	}

	else if (lvl) {
		if (f_vv) {
			gen->print_level_info(lvl, node);
			cout << "poset_orbit_node::downstep_get_invariant_subset "
					"Getting live points from previous level "
					"using orbit calculations" << endl;
			}
		poset_orbit_node *O = gen->get_node(prev);
		int i, j, l, len, pt, cur_length, a;

		len = 0;
		for (i = 0; i < O->nb_extensions; i++) {
			l = O->E[i].get_orbit_len();
			len += l;
		}
		if (f_v && O->nb_extensions < 500) {
			cout << "len=" << len << "=";
			for (i = 0; i < O->nb_extensions; i++) {
				l = O->E[i].get_orbit_len();
				cout << l;
				if (i < O->nb_extensions - 1)
					cout << "+";
			}
			cout << endl;
		}

		subset = NEW_lint(len);
		if (O->nb_strong_generators) {
			cur_length = 0;
			for (i = 0; i < O->nb_extensions; i++) {
				l = O->E[i].get_orbit_len();
				pt = O->E[i].get_pt();
				groups::schreier S;

				S.init(gen->get_A2(), verbose_level - 2);

				std::vector<int> gen_handle;

				O->get_strong_generators_handle(gen_handle, verbose_level - 2);

				S.init_generators_by_handle(
						gen_handle,
						verbose_level - 1);


				S.compute_point_orbit(pt, 0/*verbose_level*/);
				if (S.orbit_len[0] != l) {
					cout << "poset_orbit_node::downstep_get_invariant_subset "
							"fatal: S.orbit_len[0] != l" << endl;
					exit(1);
				}
				for (j = 0; j < S.orbit_len[0]; j++) {
					a = S.orbit[S.orbit_first[0] + j];
					subset[cur_length++] = a;
				}
			}
			if (cur_length != len) {
				cout << "poset_orbit_node::downstep_get_invariant_subset "
						"fatal: cur_length != len" << endl;
				exit(1);
			}
		}
		else {
			for (i = 0; i < O->nb_extensions; i++) {
				subset[i] = O->E[i].get_pt();
			}
		}
		Sorting.lint_vec_heapsort(subset, len);
		n = len;
		ret = true;
		goto the_end;
	}

the_end:
	if (f_v) {
		cout << "poset_orbit_node::downstep_get_invariant_subset "
				"subset has size " << n << endl;
		cout << "poset_orbit_node::downstep_get_invariant_subset done" << endl;
	}
	return ret;
}

void poset_orbit_node::downstep_apply_early_test(
	poset_classification *gen,
	int lvl, 
	int n, long int *subset,
	long int *candidates, int &nb_candidates,
	int verbose_level)
// called from compute_schreier_vector and from downstep_orbits
// calls the callback early test function if available
// and calls test_point_using_check_functions otherwise
// 
// This function takes the set of live points from the 
// previous level and considers it for the current level. 
// The problem is that the set may not be invariant under 
// the stabilizer of the current orbit representative 
// (because of the possibility 
// that the stabilizer of the current orbit-rep is larger 
// than the stabilizer of the previous orbit-rep).
// So, either there is a function called 'early_test_func' available 
// that does the work, 
// or we call the test_function for each point in the set, and test 
// if that point is a live point for the current orbit-rep. 
// The subset of points that survive this test are stored in 
// candidates[nb_candidates]
// This set is invariant under the stabilizer of the current orbit-rep, 
// and hence will be the set of live points for the current node.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int *the_set;
	int i;
		
	if (f_vv) {
		gen->print_level_info(lvl, node);
		cout << " : downstep_apply_early_test "
				"number of live points = " << n << endl;
		}
	the_set = NEW_lint(lvl + 1);
		// we add one more so that the early test
		// function can use the_set
		// for its own testing purposes
	store_set_to(gen, lvl - 1, the_set);
	if (f_vv) {
		gen->print_level_info(lvl, node);
		cout << " : downstep_apply_early_test "
				"number of live points = " << n << endl;
		}

	if (f_vv) {
		cout << "calling Poset->early_test_func_by_using_group" << endl;
		}

	gen->invoke_early_test_func(
			the_set, lvl,
			subset /* int *candidates*/,
			n /*int nb_candidates*/,
			candidates /*int *good_candidates*/,
			nb_candidates /*int &nb_good_candidates*/,
			verbose_level - 2);

	
	if (f_v) {
		cout << "poset_orbit_node::downstep_apply_early_test "
				"nb_candidates=" << nb_candidates << endl;
		}
	if (false && f_vv) {
		cout << "candidates: ";
		//int_vec_print(cout, candidates, nb_candidates);
		//cout << endl;
		for (i = 0; i < nb_candidates; i++) {
			cout << candidates[i] << " ";
			}
		cout << endl;
		}

	FREE_lint(the_set);
}

void poset_orbit_node::check_orbits_wrapper(
	poset_classification *gen,
	groups::schreier *Schreier, actions::action *AR,
	int lvl, 
	int &nb_good_orbits1, int &nb_points1, 
	int verbose_level)
// called from downstep_orbit_test_and_schreier_vector
// This function and create_schreier_vector_wrapper are used in pairs.
// Except, the order in which the function is used matters.
// Calls check_orbits
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "poset_orbit_node::check_orbits_wrapper" << endl;
		}

	check_orbits(gen,
			Schreier,
			AR,
			lvl,
			verbose_level - 3);


	nb_good_orbits1 = Schreier->nb_orbits;
	nb_points1 = Schreier->sum_up_orbit_lengths();
	
	if (f_v) {
		cout << "poset_orbit_node::check_orbits_wrapper "
				"the stabilizer has " << nb_good_orbits1
			<< " good orbits with "
			<< nb_points1 << " points" << endl;
	}
	if (f_v) {
		cout << "poset_orbit_node::check_orbits_wrapper done" << endl;
	}

}

void poset_orbit_node::test_orbits_for_implicit_fusion(
	poset_classification *gen,
	groups::schreier &Schreier, actions::action *AR, int f_using_invariant_subset,
	int lvl, int verbose_level)
// called from downstep_implicit_fusion
// eliminates implicit fusion orbits
// from the Schreier data structure,
{
	//verbose_level = 8;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int k, u = 0, L;
	int fst, len, rep;
	induced_actions::action_by_restriction *ABR = NULL;

	if (f_using_invariant_subset) {
		ABR = AR->G.ABR;
		}
	
	store_set(gen, lvl - 1);
	
	L = Schreier.nb_orbits;
	if (f_v) {
		cout << "test_orbits_for_implicit_fusion: "
				"testing " << L << " orbits" << endl;
		}
	for (k = 0; k < L; k++) {
		fst = Schreier.orbit_first[k];
		len = Schreier.orbit_len[k];
		rep = Schreier.orbit[fst];
		if (f_using_invariant_subset) {
			rep = ABR->points[rep];
			}


		if (lvl) {
			if (rep <= pt) {
				if (f_vv) {
					cout << "orbit " << k
						<< " is not accepted because "
						<< "we use implicit fusion nodes and " 
						<< rep << " is less than " 
						<< pt << endl;
					}
				continue;
				}
			if (f_vv) {
				cout << "orbit " << k << " is accepted" << endl;
				}
			}

		Schreier.orbit_first[u] = fst;
		Schreier.orbit_len[u] = len;
		u++;
		}
	Schreier.nb_orbits = u;
	if (f_v) {
		cout << "test_orbits_for_implicit_fusion: "
				"orbit testing "
				"finished: " << u << " orbits out of "
				<< L << " accepted" << endl;
		}
	if (f_vvv) {
		cout << "the good orbits are:" << endl;
		cout << "i : representative : orbit length" << endl;
		for (k = 0; k < Schreier.nb_orbits; k++) {
			fst = Schreier.orbit_first[k];
			len = Schreier.orbit_len[k];
			rep = Schreier.orbit[fst];
			if (f_using_invariant_subset) {
				rep = ABR->points[rep];
				}
			cout << setw(5) << k << " : " << setw(5)
					<< rep << " : " << setw(5) << len << endl;
			}
		}
	
}

void poset_orbit_node::check_orbits(
	poset_classification *gen,
	groups::schreier *Schreier, actions::action *AR,
	int lvl, 
	int verbose_level)
// called from compute_schreier_vector 
// and check_orbits_wrapper (which is called
// from downstep_orbit_test_and_schreier_vector)
// calls test_point_using_check_functions
// eliminates bad orbits from the Schreier data structure, 
// does not eliminate implicit fusion orbits
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int k, j, u = 0, L;
	int fst, len, rep, f_accept;
	induced_actions::action_by_restriction *ABR = NULL;

	ABR = AR->G.ABR;
	
	if (f_v) {
		cout << "poset_orbit_node::check_orbits" << endl;
		}
	store_set(gen, lvl - 1);
	
	L = Schreier->nb_orbits;
	if (f_v) {
		cout << "check_orbits: testing " << L << " orbits" << endl;
		}
	if (L > 100) {
		f_vv = false;
		//f_vvv = false;
	}
	for (k = 0; k < L; k++) {
		fst = Schreier->orbit_first[k];
		len = Schreier->orbit_len[k];
		rep = Schreier->orbit[fst];
		//if (f_using_invariant_subset) {
			rep = ABR->points[rep];
		//	}


		// check if the point is already in the set:
		// if it is, j will be less than lvl
		for (j = 0; j < lvl; j++) {
			if (gen->get_S()[j] == rep) {
				// we will temporarily accept the orbit anyway.
				// but we will not call the test function on this orbit
				break;
				}
			}

		f_accept = true;
		if (j == lvl) {
			if (f_vv) {
				cout << "poset_orbit_node::check_orbits "
						"calling test_point_using_check_functions"
						<< endl;
				}
			f_accept = true;

#if 0
				test_point_using_check_functions(gen,
				lvl, rep, gen->S, 
				verbose_level - 10);
#endif
			}
		if (f_accept) {
			if (f_vv) {
				cout << "orbit " << k << " of point " << rep 
					<< " of length " << len
					<< " is accepted as orbit " << u << endl;
				}
			}
		else {
			if (f_vv) {
				cout << "orbit " << k << " of point " << rep 
					<< " of length " << len
					<< " is not accepted" << endl;
				}
			continue;
			}

		Schreier->orbit_first[u] = fst;
		Schreier->orbit_len[u] = len;
		u++;
		}
	Schreier->nb_orbits = u;
	if (f_v) {
		cout << "check_orbits: orbit testing finished: " << u
				<< " orbits out of " << L << " accepted" << endl;
		}
	if (false) {
		cout << "the good orbits are:" << endl;
		cout << "i : representative : orbit length" << endl;
		for (k = 0; k < Schreier->nb_orbits; k++) {
			fst = Schreier->orbit_first[k];
			len = Schreier->orbit_len[k];
			rep = Schreier->orbit[fst];
			//if (f_using_invariant_subset) {
				rep = ABR->points[rep];
			//	}
			cout << setw(5) << k << " : " << setw(5) << rep << " : " 
				<< setw(5) << len << endl;
			}
		}
	
}


#if 0
int poset_orbit_node::test_point_using_check_functions(
	poset_classification *gen,
	int lvl, int rep, int *the_set, 
	int verbose_level)
// called by check_orbits and downstep_apply_early_test 
// Calls gen->check_the_set_incrementally
// (if gen->f_candidate_incremental_check_func).
// Otherwise, calls gen->check_the_set
// (if gen->f_candidate_check_func).
// Otherwise accepts any point.
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_accept = true;
	
	if (f_v) {
		cout << "poset_orbit_node::test_point_using_check_functions" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
#if 0
	if (gen->f_candidate_incremental_check_func) {
		if (f_vv) {
			cout << "checking point " << rep
					<< " incrementally" << endl;
			}
		the_set[lvl] = rep;
		if (gen->check_the_set_incrementally(lvl + 1,
				the_set, verbose_level - 2)) {
			}
		else {
			f_accept = false;
			}
		}
	else if (gen->f_candidate_check_func) {
		if (f_vv) {
			cout << "checking point " << rep << endl;
			}
		the_set[lvl] = rep;
		if (f_vv) {
			cout << "calling gen->check_the_set" << endl;
			}
		if (gen->check_the_set(lvl + 1, the_set, verbose_level - 2)) {
			}
		else {
			f_accept = false;
			}
		}
	else {
		//cout << "neither incremental nor ordinary check function" << endl;
		}
#endif
	return f_accept;
}
#endif

void poset_orbit_node::relabel_schreier_vector(
		actions::action *AR, int verbose_level)
// called from compute_schreier_vector,
// downstep_orbit_test_and_schreier_vector
// Replaces the points in the arrays pts[]
// and prev[] by the corresponding
// point in ABR.points[]. Does not sort. 
{
	int f_v = (verbose_level >= 1);
	int f_v5 = (verbose_level >= 5);
	induced_actions::action_by_restriction *ABR;
	int n, i;
	int *pts;
	int *prev;
	//int *label;

	if (f_v) {
		cout << "poset_orbit_node::relabel_schreier_vector" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	ABR = AR->G.ABR;
	n = get_nb_of_live_points();
	pts = live_points();
	//n = sv[0];
	//pts = sv + 1;
	if (f_v5) {
		cout << "poset_orbit_node::relabel_schreier_vector "
				"sv before:" << endl;
		//schreier_vector_print(sv);
		}
	for (i = 0; i < n; i++) {
		pts[i] = ABR->points[pts[i]];
		}
	if (nb_strong_generators) {
		prev = pts + n;
		//label = prev + n;
		for (i = 0; i < n; i++) {
			if (prev[i] >= 0) {
				prev[i] = ABR->points[prev[i]];
				}
			}
		}
	if (f_v5) {
		cout << "poset_orbit_node::relabel_schreier_vector "
				"sv after:" << endl;
		//schreier_vector_print(sv);
		}
	if (f_v) {
		cout << "poset_orbit_node::relabel_schreier_vector "
				"done" << endl;
		}
}


void poset_orbit_node::downstep_orbits_print(
	poset_classification *gen,
	groups::schreier *Schreier, actions::action *AR,
	int lvl, 
	int f_print_orbits,
	int max_orbits, int max_points_per_orbit)
{
	gen->print_level_info(lvl, node);
	cout << "The " << Schreier->nb_orbits << " orbits are:" << endl;
	int h, rep;
	induced_actions::action_by_restriction *ABR = NULL;
	
	cout << "h : orbit_len[h] : points[rep[h]] : "
			"orbit (if size is less than "
			<< max_points_per_orbit << ")" << endl;
	if (Schreier->nb_orbits <= max_orbits) {
		//if (f_using_invariant_subset) {
			ABR = AR->G.ABR;
		//	}
			
		for (h = 0; h < Schreier->nb_orbits; h++) {
			rep = Schreier->orbit[Schreier->orbit_first[h]];
			//if (f_using_invariant_subset) {
				rep = ABR->points[rep];
			//	}
			cout << setw(4) << h << " : " 
				<< setw(5) << Schreier->orbit_len[h] <<  " : "
				<< setw(5) << rep;
			if (f_print_orbits) {
				if (Schreier->orbit_len[h] <= max_points_per_orbit) {
					cout << " : ";
					//if (f_using_invariant_subset) {
						Schreier->print_orbit_through_labels(
								cout, h, ABR->points);
					//	}
					//else {
					//	Schreier.print_orbit(h);
					//	}
					}
				else {
					cout << " : too long to print";
					}
				}
			cout << endl;
			}
		if (false) {
			Schreier->print(cout);
			Schreier->print_generators();
			if (gen->get_A()->degree < 1000 && false) {
				Schreier->print_tree(0);
				Schreier->print_tables(cout, false /* f_with_cosetrep */);
				}
			}
		}
	else {
		cout << "Too many orbits to print: we have "
				<< Schreier->nb_orbits << " orbits" << endl;
		}
}



}}}


