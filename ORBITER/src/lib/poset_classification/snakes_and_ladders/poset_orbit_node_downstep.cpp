// poset_orbit_node_downstep.C
//
// Anton Betten
// July 23, 2007
//
// this is the downstep for action on subsets only

#include "foundations/foundations.h"
#include "groups_and_group_actions/groups_and_group_actions.h"
#include "poset_classification/poset_classification.h"

void poset_orbit_node::downstep(poset_classification *gen,
	int lvl, 
	int f_create_schreier_vector,
	int f_use_invariant_subset_if_available, 
	int f_implicit_fusion, 
	int verbose_level)
// Called from generator::downstep if we are acting on sets 
// (i.e., not on subspaces).
// Calls downstep_orbits, 
// downstep_orbit_test_and_schreier_vector and 
// downstep_implicit_fusion
{
#if 0
	if (node == 50) {
		//verbose_level += 10;
		}
#endif
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int nb_orbits;
	int good_orbits1, nb_points1;
	int f_using_invariant_subset = FALSE;
	schreier Schreier;
	action AR;
	int f_node_is_dead_because_of_clique_testing = FALSE;

	if (f_v) {
		cout << "poset_orbit_node::downstep" << endl;
		store_set(gen, lvl - 1); // stores a set of size lvl
		gen->print_level_info(lvl, node);
		cout << " : Downstep for ";
		print_set(gen);
		cout << " verbose_level=" << verbose_level << endl;
		if (f_vvv) {
			print_set_verbose(gen);
			}

#if 0
		if (prev >= 0 && gen->root[prev].sv) {
			//cout << "computing live points, prev=" << prev << endl;
			int nb = gen->root[prev].sv[0];
			cout << " with " << nb << " live points" << endl;
			}
		cout << endl;
#endif
		}

	//cout << "calling downstep_orbits" << endl;
	if (f_v) {
		cout << "poset_orbit_node::downstep before downstep_orbits" << endl;
		}
	downstep_orbits(gen, Schreier, AR, 
		lvl, 
		f_use_invariant_subset_if_available, 
		f_using_invariant_subset, 
		f_node_is_dead_because_of_clique_testing, 
		verbose_level - 1);
	if (f_v) {
		cout << "poset_orbit_node::downstep after downstep_orbits" << endl;
		}

#if 0
	if (node == 50) {
		cout << "poset_orbit_node::downstep after downstep_orbits" << endl;
		gen->root[49].print_extensions(cout);
		}
#endif
	nb_orbits = Schreier.nb_orbits;

#if 1
	if (gen->f_export_schreier_trees) {
		int orbit_no;

		for (orbit_no = 0; orbit_no < nb_orbits; orbit_no++) {
			char fname_mask_base[1000];
			char fname_mask[1000];

			gen->create_schreier_tree_fname_mask_base(fname_mask_base, node);

			sprintf(fname_mask, "%s.layered_graph", fname_mask_base);

			Schreier.export_tree_as_layered_graph(orbit_no,
					fname_mask,
					verbose_level);
		}
	}
	if (gen->f_draw_schreier_trees) {
		int i;
	
		for (i = 0; i < nb_orbits; i++) {
			char label[1000];
			int xmax = gen->schreier_tree_xmax;
			int ymax =  gen->schreier_tree_ymax;
			int f_circletext = gen->schreier_tree_f_circletext;
			int rad = gen->schreier_tree_rad;
			int f_embedded = gen->schreier_tree_f_embedded;
			int f_sideways = gen->schreier_tree_f_sideways;
			double scale = gen->schreier_tree_scale;
			double line_width = gen->schreier_tree_line_width;
			int f_has_point_labels = FALSE;
			int *point_labels = NULL;
			
			sprintf(label, "%sschreier_tree_node_%d_%d",
					gen->schreier_tree_prefix, node, i);

			if (f_using_invariant_subset) {
				f_has_point_labels = TRUE;
				point_labels = AR.G.ABR->points;
				}

			cout << "Node " << node << " " << i << " drawing schreier tree" << endl;
			Schreier.draw_tree(label, i, xmax, ymax, 
				f_circletext, rad, 
				f_embedded, f_sideways, 
				scale, line_width, 
				f_has_point_labels, point_labels, 
				verbose_level + 3);
			}
		
		char label_data[1000];
		sprintf(label_data, "%sschreier_data_node_%d.tex",
				gen->schreier_tree_prefix, node);
		Schreier.latex(label_data);
		}
#endif

	
	if (f_v) {
		cout << "poset_orbit_node::downstep before downstep_orbit_test_and_"
				"schreier_vector" << endl;
		}
	downstep_orbit_test_and_schreier_vector(
		gen, Schreier, AR, 
		lvl, 
		f_use_invariant_subset_if_available, 
		f_using_invariant_subset,
		f_create_schreier_vector,
		good_orbits1, nb_points1, 
		verbose_level - 1);
	if (f_v) {
		cout << "poset_orbit_node::downstep after downstep_orbit_test_and_"
				"schreier_vector" << endl;
		}

#if 0
	if (node == 50) {
		cout << "poset_orbit_node::downstep after downstep_orbit_test_and_"
				"schreier_vector" << endl;
		gen->root[49].print_extensions(cout);
		}
#endif

	if (f_v) {
		cout << "poset_orbit_node::downstep before downstep_implicit_fusion" << endl;
		}
	downstep_implicit_fusion(
		gen, Schreier, AR, f_using_invariant_subset,
		lvl, 
		f_implicit_fusion, 
		good_orbits1, nb_points1, 
		verbose_level - 1);
	if (f_v) {
		cout << "poset_orbit_node::downstep after downstep_implicit_fusion" << endl;
		}

#if 0
	if (node == 50) {
		cout << "poset_orbit_node::downstep after downstep_implicit_fusion" << endl;
		gen->root[49].print_extensions(cout);
		}
#endif


	
	if (f_vvv) {
		gen->print_level_info(lvl, node);
		cout << " : calling find_extensions" << endl;
		}
	if (f_v) {
		cout << "poset_orbit_node::downstep before find_extensions" << endl;
		}
	find_extensions(
		gen, Schreier, AR, f_using_invariant_subset,
		lvl, 
		verbose_level - 2);
	if (f_v) {
		cout << "poset_orbit_node::downstep after find_extensions" << endl;
		}
	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << " : after test_orbits and find_extensions, we have "
				<< nb_extensions << " extensions" << endl;
		}

	if (f_vvv) {
		print_extensions(gen);
		}
	
	
	
	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << " : found " << nb_extensions << " extensions (out of "
				<< nb_orbits << " orbits) with "
				<< nb_extension_points() << " points " << endl;
		}
	if (f_v) {
		cout << "poset_orbit_node::downstep done" << endl;
		}

}


void poset_orbit_node::compute_schreier_vector(
	poset_classification *gen,
	int lvl, int f_compact, int verbose_level)
// called from generator::recreate_schreier_vectors_at_level
// and from generator::count_live_points
// calls downstep_apply_early_test
// and check_orbits
// and Schreier.get_schreier_vector
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	schreier Schreier;
	int f_trivial_group;
	int f_using_invariant_subset = FALSE;
	int f_use_incremental_test_func_if_available = TRUE;
	int *candidates = NULL;
	int nb_candidates;
	action AR;

	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector: "
				"computing Schreier vector" << endl;
		}	
	
	if (nb_strong_generators == 0) {
		f_trivial_group = TRUE;
		}
	else {
		f_trivial_group = FALSE;
		}
	if (FALSE) {
		cout << "generators:" << endl;
		Schreier.print_generators();
		}
	
	if (lvl &&
		gen->root[prev].Schreier_vector) {

#if 0
		int *osv = gen->root[prev].sv;
		int n = osv[0];
		int *subset = osv + 1;
#else
		int n = gen->root[prev].get_nb_of_live_points();
		int *subset = gen->root[prev].live_points();
#endif
		f_using_invariant_subset = TRUE;

		candidates = NEW_int(n);
		
		downstep_apply_early_test(gen, lvl, 
			n, subset, 
			candidates, nb_candidates, 
			verbose_level);

		AR.induced_action_by_restriction(*gen->A2, 
			FALSE /*f_induce_action*/, NULL /*sims *old_G*/, 
			nb_candidates, candidates,
			verbose_level - 2);

		//if (f_vv) {
		//	cout << "calling orbits_on_invariant_subset_fast" << endl;
		//	}
		//Schreier.orbits_on_invariant_subset_fast(
		// n, subset, verbose_level);
		Schreier.init(&AR);
#if 0
		if (f_vv) {
			cout << "poset_orbit_node::compute_schreier_vector "
					"the stabilizer has " << Schreier.nb_orbits
					<< " orbits on the live point" << endl;
			}
#endif
		}
	else if (lvl == 0) {
		int *subset;
		int i;
		int n = gen->A2->degree;
		
		subset = NEW_int(n);
		for (i = 0; i < n; i++) {
			subset[i] = i;
			}
		
		f_using_invariant_subset = TRUE;

		candidates = NEW_int(n);
		
		downstep_apply_early_test(gen, lvl, 
			n, subset, 
			candidates, nb_candidates, 
			verbose_level);
		AR.induced_action_by_restriction(*gen->A2, 
			FALSE /*f_induce_action*/, NULL /*sims *old_G*/, 
			nb_candidates, candidates,
			verbose_level - 2);
		//if (f_vv) {
		//	cout << "calling orbits_on_invariant_subset_fast" << endl;
		//	}
		//Schreier.orbits_on_invariant_subset_fast(n, subset, verbose_level);
		Schreier.init(&AR);
		FREE_int(subset);
#if 0
		if (f_vv) {
			cout << "poset_orbit_node::compute_schreier_vector "
					"the stabilizer has " << Schreier.nb_orbits
					<< " orbits on the live point" << endl;
			}
#endif
		}
	else {
		f_using_invariant_subset = FALSE;
		Schreier.init(gen->A2);
			// here was a mistake, it was gen->A
			// A. Betten, Dec 17, 2011 !!!
		}
	Schreier.init_generators_by_hdl(nb_strong_generators,
			hdl_strong_generators, verbose_level - 1);
	if (f_vv) {
		cout << "poset_orbit_node::compute_schreier_vector "
				"calling compute_all_point_orbits" << endl;
		}
	if (lvl == 0) {
		Schreier.compute_all_point_orbits(verbose_level - 1 /*FALSE*/);
		}
	else {
		Schreier.compute_all_point_orbits(FALSE);
		}
	if (f_vv) {
		cout << "poset_orbit_node::compute_schreier_vector "
				"the stabilizer has " << Schreier.nb_orbits
				<< " orbits overall" << endl;
		}
	
	check_orbits(gen, Schreier, AR, f_using_invariant_subset, 
		lvl, f_use_incremental_test_func_if_available,
		verbose_level - 2);
		// here was a mistake, 
		// f_use_incremental_test_func_if_available
		// was f_using_invariant_subset
		// A. Betten, Dec 17, 2011 !!!
	
	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector "
				"the stabilizer has " << Schreier.nb_orbits
			<< " good orbits with "
			<< Schreier.sum_up_orbit_lengths() << " points" << endl;
		}

	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector "
				"calling get_schreier_vector" << endl;
		}	

	create_schreier_vector_wrapper(
			TRUE /* f_create_schreier_vector */,
			Schreier, verbose_level - 1);

	//Schreier.get_schreier_vector(sv,
	//		f_trivial_group, f_compact);
	//Schreier.test_sv(gen->A, hdl_strong_generators,
	// sv, f_compact, verbose_level);

	if (f_using_invariant_subset) {
		relabel_schreier_vector(AR, verbose_level - 1);
		}

	if (candidates) {
		FREE_int(candidates);
		}
	if (f_v) {
		cout << "poset_orbit_node::compute_schreier_vector: "
				"Schreier vector has been computed" << endl;
		}	
}





// #############################################################################
// first level under downstep:
// #############################################################################



void poset_orbit_node::downstep_orbits(
	poset_classification *gen, schreier &Schreier, action &AR,
	int lvl, 
	int f_use_invariant_subset_if_available, 
	int &f_using_invariant_subset, 
	int &f_node_is_dead_because_of_clique_testing, 
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
	int *subset = NULL;
	int *candidates = NULL;
	int nb_candidates = 0;
	int f_subset_is_allocated = FALSE;

	f_node_is_dead_because_of_clique_testing = FALSE;
	f_using_invariant_subset = FALSE;

	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << "poset_orbit_node::downstep_orbits" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	
	

	if (f_use_invariant_subset_if_available) {
		if (lvl == 0) {
			if (f_v) {
				cout << "poset_orbit_node::downstep_orbits we are trying "
						"to find an invariant subset" << endl;
				}
			}
		f_using_invariant_subset = downstep_get_invariant_subset(
			gen, 
			lvl, 
			n, subset, f_subset_is_allocated, 
			verbose_level /*- 2 */);

		if (lvl == 0 && !f_using_invariant_subset) {
			cout << "We did not find an invariant subset" << endl;
			}
		}
	else {
		if (lvl == 0) {
			cout << "poset_orbit_node::downstep_orbits we are NOT using "
					"an invariant subset" << endl;
			}
		}
	
	if (f_using_invariant_subset) {

		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : we are using an invariant subset : "
					"live points at the predecessor node: number=" << n;
			if (f_v4) {
				cout << " : ";
				int_vec_print(cout, subset, n);
				cout << endl; 
				}
			else {
				cout << endl; 
				}
			}
		candidates = NEW_int(n);
		
		downstep_apply_early_test(gen, lvl, 
			n, subset, 
			candidates, nb_candidates, 
			verbose_level - 2);

		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : live points after downstep_apply_early_test: "
					"number=" << nb_candidates;
			if (f_v4) {
				cout << " : ";
				int_vec_print(cout, candidates, nb_candidates);
				cout << endl; 
				}
			else {
				cout << endl;
				}	
			}



		AR.induced_action_by_restriction(*gen->A2, 
			FALSE /*f_induce_action*/, NULL /*sims *old_G*/, 
			nb_candidates, candidates, verbose_level - 2);
		
		if (f_vv) {
			cout << "created restricted action ";
			AR.print_info();
			}
		Schreier.init(&AR /*gen->A2*/);
		}
	else {
		gen->print_level_info(lvl, node);
		cout << " : we are NOT using an invariant subset" << endl;
		Schreier.init(gen->A2);
		}


	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << " : initializing generators. There are "
				<< nb_strong_generators  << " strong generators" << endl;
		}


	Schreier.init_generators_by_hdl(nb_strong_generators,
			hdl_strong_generators, verbose_level - 1);
	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << " : calling Schreier.compute_all_point_orbits "
				"for a set of size " << Schreier.A->degree << endl;
		}


	if (FALSE /*f_v4*/) {
		gen->print_level_info(lvl, node);
		cout << " : generators:" << endl;
		Schreier.print_generators();
		}


	//Schreier.compute_all_point_orbits_with_preferred_labels(
	// n, subset, verbose_level - 4);
	if (lvl == 0) {
		Schreier.compute_all_point_orbits(verbose_level);
		}
	else {
		Schreier.compute_all_point_orbits(verbose_level - 4);
		}

	if (f_v) {
		int f_print_orbits = FALSE;
		if (f_vv) {
			f_print_orbits = TRUE;
			}
		//int max_orbits = 50;
		//int max_points_per_orbit = 25;
		if (f_using_invariant_subset) {
			downstep_orbits_print(gen, 
				Schreier, AR, lvl, 
				f_using_invariant_subset, 
				f_print_orbits, 
				gen->downstep_orbits_print_max_orbits,
				gen->downstep_orbits_print_max_points_per_orbit);
			}
		}
	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << "poset_orbit_node::downstep_orbits: we found "
				<< Schreier.nb_orbits << " orbits" << endl;
		}
	if (f_using_invariant_subset && f_subset_is_allocated) {
		FREE_int(subset);
		}
	if (candidates) {
		FREE_int(candidates);
		}
}

void poset_orbit_node::downstep_orbit_test_and_schreier_vector(
	poset_classification *gen, schreier &Schreier, action &AR,
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
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_print_orbits = FALSE;
	if (f_vvv) {
		f_print_orbits = TRUE;
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
			AR, f_using_invariant_subset,
			lvl, nb_good_orbits, nb_points, 
			f_using_invariant_subset
			/*f_use_incremental_test_func_if_available*/,
			verbose_level - 1);

		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : after check_orbits_wrapper:" << endl;
			cout << "nb_good_orbits=" << nb_good_orbits << endl;
			cout << "nb_points=" << nb_points << endl;
			}
		if (f_vv) {
			downstep_orbits_print(gen, 
				Schreier, AR, lvl, 
				f_using_invariant_subset, 
				f_print_orbits, 
				max_orbits, max_points_per_orbit);
			}

		create_schreier_vector_wrapper(
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
				}
			}
		}
	else {
		// in this case, we need all orbits in the schreier vector.
		// that's why we do the orbit checking afterwards
		create_schreier_vector_wrapper(
			f_create_schreier_vector, 
			Schreier, 
			verbose_level - 1);
		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : after creating Schreier vector." << endl;
			}

		check_orbits_wrapper(gen,
			Schreier, AR, f_using_invariant_subset,
			lvl, nb_good_orbits, nb_points, 
			FALSE /*f_use_incremental_test_func_if_available*/, 
			verbose_level - 1);

		if (f_v) {
			gen->print_level_info(lvl, node);
			cout << " : after check_orbits_wrapper:" << endl;
			cout << "nb_good_orbits=" << nb_good_orbits << endl;
			cout << "nb_points=" << nb_points << endl;
			}
		if (f_vv) {
			downstep_orbits_print(gen, 
				Schreier, AR, lvl, 
				f_using_invariant_subset, 
				f_print_orbits, 
				max_orbits, max_points_per_orbit);
			}
		}
}

void poset_orbit_node::downstep_implicit_fusion(
	poset_classification *gen, schreier &Schreier, action &AR,
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


void poset_orbit_node::find_extensions(poset_classification *gen,
	schreier &O, action &AR, int f_using_invariant_subset, 
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
	int f_vv = FALSE; //(verbose_level >= 2);
	int f_vvv = FALSE; //(verbose_level >= 3);
	int h, k, fst, /*len,*/ rep;
	action_by_restriction *ABR = NULL;

	if (f_using_invariant_subset) {
		ABR = AR.G.ABR;
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
				if (gen->S[ii] == rep)
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

			

		E[nb_extensions].pt = rep;
		E[nb_extensions].orbit_len = O.orbit_len[k];
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
			cout << h << " : " << E[h].orbit_len << " : " << E[h].pt << endl;
			}
		}
}



// #############################################################################
// second level under downstep:
// #############################################################################


int poset_orbit_node::downstep_get_invariant_subset(
		poset_classification *gen,
	int lvl, 
	int &n, int *&subset, int &f_subset_is_allocated, 
	int verbose_level)
// called from downstep_orbits
// Gets the live points at the present node.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret = FALSE;
	

	n = -1;
	subset = NULL;

	if (f_v) {
		cout << "poset_orbit_node::downstep_get_invariant_subset" << endl;
		}
	if (gen->f_starter && lvl == gen->starter_size) {
		if (f_vv) {
			gen->print_level_info(lvl, node);
			cout << "poset_orbit_node::downstep_get_invariant_subset "
					"Getting live points for the starter" << endl;
			}
		n = gen->starter_nb_live_points;
		subset = gen->starter_live_points;
		f_subset_is_allocated = FALSE;
		ret = TRUE;
		goto the_end;
		}
	else if (lvl == 0 && gen->f_has_invariant_subset_for_root_node) {
		cout << "poset_orbit_node::downstep_get_invariant_subset "
				"root node has an invariant subset of size " << n << endl;
		subset = gen->invariant_subset_for_root_node;
		n = gen->invariant_subset_for_root_node_size;
		f_subset_is_allocated = FALSE;
		ret = TRUE;
		goto the_end;
		}
	else if (lvl == 0) {
		n = gen->A2->degree;
		subset = NEW_int(n);
		int i;
		for (i = 0; i < n; i++) {
			subset[i] = i;
			}
		f_subset_is_allocated = TRUE;
		ret = TRUE;
		goto the_end;
		}
	else if (lvl && gen->root[prev].Schreier_vector) {
		
		if (f_vv) {
			gen->print_level_info(lvl, node);
			cout << "poset_orbit_node::downstep_get_invariant_subset "
					"Getting live points from previous level" << endl;
			}
		n = gen->root[prev].get_nb_of_live_points();
		subset = gen->root[prev].live_points();
#if 0
		int *osv = gen->root[prev].sv;
		n = osv[0];
		subset = osv + 1;
#endif
		f_subset_is_allocated = FALSE;
		ret = TRUE;
		goto the_end;
		}
	else if (lvl) {
		if (f_vv) {
			gen->print_level_info(lvl, node);
			cout << "poset_orbit_node::downstep_get_invariant_subset "
					"Getting live points from previous level "
					"using orbit calculations" << endl;
			}
		poset_orbit_node *O = &gen->root[prev];
		int i, j, l, len, pt, cur_length, a;

		len = 0;
		for (i = 0; i < O->nb_extensions; i++) {
			l = O->E[i].orbit_len;
			len += l;
			}
		if (f_v && O->nb_extensions < 500) {
			cout << "len=" << len << "=";
			for (i = 0; i < O->nb_extensions; i++) {
				l = O->E[i].orbit_len;
				cout << l;
				if (i < O->nb_extensions - 1)
					cout << "+";
				}
			cout << endl;
			}
		subset = NEW_int(len);
		if (O->nb_strong_generators) {
			cur_length = 0;
			for (i = 0; i < O->nb_extensions; i++) {
				l = O->E[i].orbit_len;
				pt = O->E[i].pt;
				schreier S;

				S.init(gen->A2);
				S.init_generators_by_hdl(O->nb_strong_generators, 
					O->hdl_strong_generators, verbose_level - 1);
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
				subset[i] = O->E[i].pt;
				}
			}
		int_vec_heapsort(subset, len);
		n = len;
		f_subset_is_allocated = TRUE;
		ret = TRUE;
		goto the_end;
		}
the_end:
	if (f_v) {
		cout << "poset_orbit_node::downstep_get_invariant_subset done" << endl;
		}
	return ret;
}

void poset_orbit_node::downstep_apply_early_test(
	poset_classification *gen,
	int lvl, 
	int n, int *subset, 
	int *candidates, int &nb_candidates, 
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
	int *the_set;
	int i;
		
	if (f_vv) {
		gen->print_level_info(lvl, node);
		cout << " : downstep_apply_early_test "
				"number of live points = " << n << endl;
		}
	the_set = NEW_int(lvl + 1);
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
		cout << "calling early_test_func" << endl;
		}
	if (!gen->f_early_test_func) {
		if (gen->f_its_OK_to_not_have_an_early_test_func) {
			
			// simply copy the set over, no testing

			for (i = 0; i < n; i++) {
				candidates[i] = subset[i];
				}
			nb_candidates = n;
			}
		else {
			//cout << "poset_orbit_node::downstep_apply_early_test
			// not gen->f_early_test_func" << endl;
			//exit(1);
			int rep; 

			if (f_vv) {
				cout << "poset_orbit_node::downstep_apply_early_test "
						"not gen->f_early_test_func, using the check "
						"functions instead" << endl;
				}
			nb_candidates = 0;
			for (i = 0; i < n; i++) {
				rep = subset[i];
				if (test_point_using_check_functions(gen, 
					lvl, rep, the_set, 
					verbose_level - 1)) {
					candidates[nb_candidates++] = rep;
					}
				}

			}
		}
	else {
		(*gen->early_test_func)(the_set, lvl, subset, n, 
			candidates, nb_candidates,
			gen->early_test_func_data, verbose_level - 1);
		}
	
	if (f_v) {
		cout << "poset_orbit_node::downstep_apply_early_test "
				"nb_candidates=" << nb_candidates << endl;
		}
	if (FALSE && f_vv) {
		cout << "candidates: ";
		//int_vec_print(cout, candidates, nb_candidates);
		//cout << endl;
		for (i = 0; i < nb_candidates; i++) {
			cout << candidates[i] << " ";
			}
		cout << endl;
		}

	FREE_int(the_set);
}

void poset_orbit_node::check_orbits_wrapper(
	poset_classification *gen,
	schreier &Schreier, action &AR, int f_using_invariant_subset, 
	int lvl, 
	int &nb_good_orbits1, int &nb_points1, 
	int f_use_incremental_test_func_if_available, 
	int verbose_level)
// called from downstep_orbit_test_and_schreier_vector
// This function and create_schreier_vector_wrapper are used in pairs.
// Except, the order in which the function is used matters.
// Calls check_orbits
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "poset_orbit_node::check_orbits_wrapper "
				"calling check_orbits f_use_incremental_test_func_"
				"if_available="
				<< f_use_incremental_test_func_if_available << endl;
		}

	check_orbits(gen,
			Schreier,
			AR,
			f_using_invariant_subset,
			lvl,
			f_use_incremental_test_func_if_available,
			verbose_level - 3);


	nb_good_orbits1 = Schreier.nb_orbits;
	nb_points1 = Schreier.sum_up_orbit_lengths();
	
	if (f_v) {
		cout << "the stabilizer has " << nb_good_orbits1
			<< " good orbits with "
			<< nb_points1 << " points" << endl;
		}

}

void poset_orbit_node::test_orbits_for_implicit_fusion(
	poset_classification *gen,
	schreier &Schreier, action &AR, int f_using_invariant_subset, 
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
	action_by_restriction *ABR = NULL;

	if (f_using_invariant_subset) {
		ABR = AR.G.ABR;
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
				"orbit testing '"
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

int poset_orbit_node::nb_extension_points()
// sums up the lengths of orbits in all extensions
{
	int i, n;
	
	n = 0;
	for (i = 0; i < nb_extensions; i++) {
		n += E[i].orbit_len;
		}
	return n;
	
}

void poset_orbit_node::check_orbits(
	poset_classification *gen,
	schreier &Schreier, action &AR,
	int f_using_invariant_subset,
	int lvl, 
	int f_use_incremental_test_func_if_available, 
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
	int f_vvv = (verbose_level >= 3);
	int k, j, u = 0, L;
	int fst, len, rep, f_accept;
	action_by_restriction *ABR = NULL;

	if (f_using_invariant_subset) {
		ABR = AR.G.ABR;
		}
	
	if (f_v) {
		cout << "poset_orbit_node::check_orbits" << endl;
		cout << "f_use_incremental_test_func_if_available="
				<< f_use_incremental_test_func_if_available << endl;
		cout << "f_using_invariant_subset="
				<< f_using_invariant_subset << endl;
		}
	store_set(gen, lvl - 1);
	
	L = Schreier.nb_orbits;
	if (f_v) {
		cout << "check_orbits: testing " << L << " orbits" << endl;
		}
	for (k = 0; k < L; k++) {
		fst = Schreier.orbit_first[k];
		len = Schreier.orbit_len[k];
		rep = Schreier.orbit[fst];
		if (f_using_invariant_subset) {
			rep = ABR->points[rep];
			}


		// check if the point is already in the set:
		// if it is, j will be less than lvl
		for (j = 0; j < lvl; j++) {
			if (gen->S[j] == rep) {
				// we will temporarily accept the orbit anyway.
				// but we will not call the test function on this orbit
				break;
				}
			}

		f_accept = TRUE;
		if (j == lvl) {
			if (f_vv) {
				cout << "poset_orbit_node::check_orbits "
						"calling test_point_using_check_functions"
						<< endl;
				}
			f_accept = test_point_using_check_functions(gen, 
				lvl, rep, gen->S, 
				verbose_level - 4);
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

		Schreier.orbit_first[u] = fst;
		Schreier.orbit_len[u] = len;
		u++;
		}
	Schreier.nb_orbits = u;
	if (f_v) {
		cout << "check_orbits: orbit testing finished: " << u
				<< " orbits out of " << L << " accepted" << endl;
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
			cout << setw(5) << k << " : " << setw(5) << rep << " : " 
				<< setw(5) << len << endl;
			}
		}
	
}


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
	int f_vv = (verbose_level >= 2);
	int f_accept = TRUE;
	
	if (f_v) {
		cout << "poset_orbit_node::test_point_using_check_functions" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
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
			f_accept = FALSE;
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
			f_accept = FALSE;
			}
		}
	else {
		//cout << "neither incremental nor ordinary check function" << endl;
		}
	return f_accept;
}

void poset_orbit_node::relabel_schreier_vector(
	action &AR, int verbose_level)
// called from compute_schreier_vector,
// downstep_orbit_test_and_schreier_vector
// Replaces the points in the arrays pts[]
// and prev[] by the corresponding
// point in ABR.points[]. Does not sort. 
{
	int f_v = (verbose_level >= 1);
	int f_v5 = (verbose_level >= 5);
	action_by_restriction *ABR;
	int n, i;
	int *pts;
	int *prev;
	//int *label;

	if (f_v) {
		cout << "poset_orbit_node::relabel_schreier_vector" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	ABR = AR.G.ABR;
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
	schreier &Schreier, action &AR, 
	int lvl, 
	int f_using_invariant_subset, int f_print_orbits, 
	int max_orbits, int max_points_per_orbit)
{
	gen->print_level_info(lvl, node);
	cout << "The " << Schreier.nb_orbits << " orbits are:" << endl;
	int h, rep;
	action_by_restriction *ABR = NULL;
	
	cout << "h : orbit_len[h] : points[rep[h]] : "
			"orbit (if size is less than "
			<< max_points_per_orbit << ")" << endl;
	if (Schreier.nb_orbits <= max_orbits) {
		if (f_using_invariant_subset) {
			ABR = AR.G.ABR;
			}
			
		for (h = 0; h < Schreier.nb_orbits; h++) {
			rep = Schreier.orbit[Schreier.orbit_first[h]];
			if (f_using_invariant_subset) {
				rep = ABR->points[rep];
				}
			cout << setw(4) << h << " : " 
				<< setw(5) << Schreier.orbit_len[h] <<  " : " 
				<< setw(5) << rep;
			if (f_print_orbits) {
				if (Schreier.orbit_len[h] <= max_points_per_orbit) {
					cout << " : ";
					if (f_using_invariant_subset) {
						Schreier.print_orbit_through_labels(
								cout, h, ABR->points);
						}
					else {
						Schreier.print_orbit(h);
						}
					}
				else {
					cout << " : too long to print";
					}
				}
			cout << endl;
			}
		if (FALSE) {
			Schreier.print(cout);
			Schreier.print_generators();
			if (gen->A->degree < 1000 && FALSE) {
				Schreier.print_tree(0);
				Schreier.print_tables(cout, FALSE /* f_with_cosetrep */);
				}
			}
		}
	else {
		cout << "Too many orbits to print: we have "
				<< Schreier.nb_orbits << " orbits" << endl;
		}
}




