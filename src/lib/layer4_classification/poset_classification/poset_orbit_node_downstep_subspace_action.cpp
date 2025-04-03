// poset_orbit_node_downstep_subspace_action.cpp
//
// Anton Betten
// Jan 21, 2010

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


void poset_orbit_node::compute_flag_orbits_subspace_action(
	poset_classification *gen,
	int lvl, 
	int f_create_schreier_vector,
	int f_use_invariant_subset_if_available, 
	int f_implicit_fusion, 
	int verbose_level)
// called from poset_classification::compute_flag_orbits
// creates action *A_factor_space
// and action_on_factor_space *AF
// and disposes them at the end.
{
	//if (node == 0) {verbose_level += 20;
	// cout << "poset_orbit_node::compute_flag_orbits_subspace_action "
	//"node 0 reached" << endl;}
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_v4 = (verbose_level >= 4);
	int nb_orbits;
	//int good_orbits1, nb_points1;
	int f_using_invariant_subset = false;
	groups::schreier *Schreier;


	induced_actions::action_on_factor_space *AF;
	actions::action *A_factor_space;


	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits_subspace_action" << endl;
	}
	store_set(gen, lvl - 1); // stores a set of size lvl to gen->S
	

	Schreier = NEW_OBJECT(groups::schreier);
	AF = NEW_OBJECT(induced_actions::action_on_factor_space);
	//A_factor_space = NEW_OBJECT(actions::action);
	
	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << " : Downstep for ";
		print_set(gen);
		cout <<  " in subspace_action";
		cout << " verbose_level=" << verbose_level << endl;
		//cout << "gen->f_early_test_func="
		//		<< gen->f_early_test_func << endl;
		if (f_vvv) {
			print_set_verbose(gen);
		}
		if (prev >= 0 && gen->node_has_schreier_vector(prev)) {
			int nb = gen->get_node(prev)->get_nb_of_live_points();
			cout << " with " << nb << " live points";
			if (f_vvv) {
				cout << " : ";
				Int_vec_print(cout, gen->get_node(prev)->live_points(),
						gen->get_node(prev)->get_nb_of_live_points());
				cout << endl;
			}
			else {
				cout << endl;
			}
		}
		cout << endl;
	}

	groups::strong_generators *Strong_gens;
	algebra::ring_theory::longinteger_object go;

	get_stabilizer_generators(gen, Strong_gens, verbose_level);
	Strong_gens->group_order(go);
	if (f_v) {
		cout << "The stabilizer is a group of order " << go << endl;
		if (f_vv) {
			cout << "With the following generators:" << endl;
			Strong_gens->print_generators(cout, verbose_level - 1);
		}
	}
	

	//if (true /*gen->f_early_test_func*/) {

		if (f_v) {
			cout << "poset_orbit_node::compute_flag_orbits_subspace_action "
					"before setup_factor_space_action_with_early_test"
					<< endl;
		}
		//cout << "setup_factor_space_action_with_early_test unavailable" << endl;
		//exit(1);
#if 1
		setup_factor_space_action_with_early_test(gen, 
			AF, A_factor_space,
			lvl, verbose_level - 2);
#endif

		if (f_v) {
			cout << "poset_orbit_node::compute_flag_orbits_subspace_action "
					"after setup_factor_space_action_with_early_test"
					<< endl;
		}

		//}
#if 0
	else {
		if (f_v) {
			cout << "poset_orbit_node::compute_flag_orbits_subspace_action "
					"before setup_factor_space_action" << endl;
		}
		setup_factor_space_action(gen,
				*AF, *A_factor_space, lvl,
				true /*f_compute_tables*/,
				verbose_level - 7);
		if (f_v) {
			cout << "poset_orbit_node::compute_flag_orbits_subspace_action "
					"after setup_factor_space_action" << endl;
		}
	}
#endif
	
	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits_subspace_action "
				"before Schreier.init" << endl;
	}



#if 1
	if (A_on_upset) {
		cout << "poset_orbit_node::compute_flag_orbits_subspace_action "
				"A_on_upset is already allocated" << endl;
		FREE_OBJECT(A_on_upset);
	}
	A_on_upset = A_factor_space;
#endif

	Schreier->init(A_factor_space, verbose_level - 2);




	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits_subspace_action "
				"before Schreier.init_generators_by_handle" << endl;
	}

	std::vector<int> gen_handle;

	get_strong_generators_handle(gen_handle, verbose_level - 2);

	Schreier->Generators_and_images->init_generators_by_handle(
			gen_handle,
			verbose_level - 1);

	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits_subspace_action "
				"before downstep_orbits_subspace_action" << endl;
	}
	downstep_orbits_subspace_action(
		gen,
		*Schreier,
		lvl, 
		f_use_invariant_subset_if_available, 
		f_using_invariant_subset, 
		verbose_level - 2);
	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits_subspace_action "
				"after downstep_orbits_subspace_action" << endl;
	}



	nb_orbits = Schreier->Forest->nb_orbits;
	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits_subspace_action "
				"after downstep_orbits_subspace_action "
				"nb_orbits=" << nb_orbits << endl;
	}
	
	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits_subspace_action "
				"before create_schreier_vector_wrapper_subspace_action "
				<< endl;
	}
	create_schreier_vector_wrapper_subspace_action(
		gen,
		f_create_schreier_vector, 
		*Schreier, 
		A_factor_space, AF, 
		verbose_level - 2);
	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits_subspace_action "
				"after create_schreier_vector_wrapper_subspace_action "
				<< endl;
	}

	if (f_v4) {
		gen->print_level_info(lvl, node);
		cout << " : calling find_extensions_subspace_action" << endl;
	}
	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits_subspace_action "
				"before find_extensions_subspace_action" << endl;
	}
	find_extensions_subspace_action(
		gen, *Schreier, 
		A_factor_space, AF, 
		lvl, f_implicit_fusion,
		verbose_level - 1);
	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits_subspace_action "
				"after find_extensions_subspace_action" << endl;
	}
	if (f_v4) {
		gen->print_level_info(lvl, node);
		cout << " : after test_orbits and find_extensions, "
				"we have " << nb_extensions << " extensions" << endl;
	}
	
	
	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << " : found " << nb_extensions << " extensions "
				"(out of " << nb_orbits << " orbits) with "
				<< nb_extension_points() << " points " << endl;
	}
	if (f_vv) {
		downstep_subspace_action_print_orbits(
			gen, *Schreier, 
			lvl, 
			f_vvv /* f_print_orbits */, 
			verbose_level);
	}
	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits_subspace_action "
				"before deleting things" << endl;
	}
	FREE_OBJECT(Strong_gens);
	FREE_OBJECT(Schreier);


	//FREE_OBJECT(A_factor_space);


	//FREE_OBJECT(AF);
	if (f_v) {
		cout << "poset_orbit_node::compute_flag_orbits_subspace_action "
				"done" << endl;
	}

}

void poset_orbit_node::setup_factor_space_action_light(
	poset_classification *gen,
	induced_actions::action_on_factor_space &AF,
	int lvl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *the_set;

	if (f_v) {
		cout << "poset_orbit_node::setup_factor_space_action_light "
				"lvl=" << lvl << endl;
		cout << "poset_orbit_node::setup_factor_space_action_light "
				"node=" << node << " prev=" << prev
				<< " pt=" << pt << endl;
	}
	the_set = NEW_lint(lvl);
	store_set_to(gen, lvl - 1, the_set);

	AF.init_light(
		gen->get_VS(),
		*gen->get_A(),
		*gen->get_A2(),
		the_set, lvl,
		verbose_level - 1);
	FREE_lint(the_set);
}

void poset_orbit_node::setup_factor_space_action_with_early_test(
	poset_classification *gen,
	induced_actions::action_on_factor_space *AF,
	actions::action *&A_factor_space,
	int lvl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int *subset;
	long int *the_set;
	int n, i;

	if (f_v) {
		cout << "poset_orbit_node::setup_factor_space_action_with_early_test "
				"lvl=" << lvl << endl;
		cout << "poset_orbit_node::setup_factor_space_action_with_early_test "
				"node=" << node << " prev=" << prev
				<< " pt=" << pt << endl;
		cout << "poset_orbit_node::setup_factor_space_action_with_early_test "
				"A2->degree=" << gen->get_A2()->degree << endl;
	}
	the_set = NEW_lint(lvl + 1);
		// the +1 is for safety !!!
		// Namely, so that the test function has one more entry
		// to store the candidate point in.
		// A Betten Nov 21, 2011


	store_set_to(gen, lvl - 1, the_set);

	if (lvl) {
		if (f_vv) {
			cout << "poset_orbit_node::setup_factor_space_action_with_early_test "
					"retrieving candidate set from Schreier vector "
					"of node " << prev << endl;
		}
		int *old_subset;
		n = gen->get_node(prev)->get_nb_of_live_points();
		old_subset = gen->get_node(prev)->live_points();
		subset = NEW_lint(n);
		for (i = 0; i < n; i++) {
			subset[i] = old_subset[i];
		}
		//int *osv = gen->root[prev].sv;
		//n = osv[0];
		//subset = osv + 1;
	}
	else {
		n = gen->get_A2()->degree;
		subset = NEW_lint(n);
		for (i = 0; i < n; i++) {
			subset[i] = i;
		}
	}
	long int *candidates;
	int nb_candidates;

	if (f_vv) {
		gen->print_level_info(lvl, node);
		cout << " : number of live points = " << n << endl;
	}
	candidates = NEW_lint(gen->get_A2()->degree);

	if (f_vv) {
		cout << "poset_orbit_node::setup_factor_space_action_with_early_test "
				"calling early_test_func, degree = "
				<< gen->get_A2()->degree
				<< " n = " << n << endl;
	}

	gen->invoke_early_test_func(
			the_set, lvl,
			subset /* candidates */,
			n /* nb_candidates */,
			candidates  /* good_candidates */,
			nb_candidates /* nb_good_candidates */,
			verbose_level - 2);

	if (f_vv) {
		cout << "poset_orbit_node::setup_factor_space_action_with_early_test "
				"after early_test_func, "
				"degree = " << gen->get_A2()->degree
				<< " n = " << n
				<< " nb_candidates=" << nb_candidates << endl;
		//cout << "candidates: ";
		//int_vec_print(cout, candidates, nb_candidates);
		//cout << endl;
	}

	if (f_vv) {
		cout << "poset_orbit_node::setup_factor_space_action_with_early_test "
				"before AF->init_by_rank_table_mode" << endl;
	}
	AF->init_by_rank_table_mode(
		gen->get_VS(),
		*gen->get_A(),
		*gen->get_A2(),
		the_set, lvl,
		candidates, nb_candidates,
		verbose_level - 3);
	if (f_vv) {
		cout << "poset_orbit_node::setup_factor_space_action_with_early_test "
				"after AF->init_by_rank_table_mode" << endl;
	}

	FREE_lint(candidates);
	FREE_lint(the_set);
	if (true /* lvl == 0*/) {
		FREE_lint(subset);
	}

	if (f_vv) {
		cout << "poset_orbit_node::setup_factor_space_action_with_early_test "
				"before gen->get_A2()->Induced_action->induced_action_on_factor_space"
				<< endl;
	}

	A_factor_space = gen->get_A2()->Induced_action->induced_action_on_factor_space(
		AF,
		false /*f_induce_action*/,
		NULL /* sims */,
		0/*verbose_level - 3*/);

	if (f_vv) {
		cout << "poset_orbit_node::setup_factor_space_action_with_early_test "
				"after gen->get_A2()->Induced_action->induced_action_on_factor_space"
				<< endl;
	}

	if (f_v) {
		cout << "poset_orbit_node::setup_factor_space_action_with_early_test "
				"lvl=" << lvl << endl;
		cout << "poset_orbit_node::setup_factor_space_action_with_early_test "
				"node=" << node << " prev=" << prev
				<< " pt=" << pt << " done" << endl;
	}
}

void poset_orbit_node::setup_factor_space_action(
	poset_classification *gen,
	induced_actions::action_on_factor_space *AF,
	actions::action *&A_factor_space,
	int lvl, int f_compute_tables, int verbose_level)
// called from poset_orbit_node::init_extension_node,
// poset_orbit_node::orbit_representative_and_coset_rep_inv_subspace_action
// (in poset_orbit_node_upstep_subspace_action)
// poset_orbit_node::downstep_subspace_action
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v20 = (verbose_level >= 20);
	long int *the_set;
	int *coordinates;
	//int i;

	if (f_v) {
		cout << "poset_orbit_node::setup_factor_space_action "
				"lvl=" << lvl << endl;
		cout << "poset_orbit_node::setup_factor_space_action "
				"node=" << node << " prev=" << prev
				<< " pt=" << pt << endl;
		cout << "f_compute_tables=" << f_compute_tables << endl;
	}
	the_set = NEW_lint(lvl);
	coordinates = NEW_int(lvl * gen->get_VS()->dimension);
	store_set_to(gen, lvl - 1, the_set);

	if (f_v) {
		cout << "the set: ";
		Lint_vec_print(cout, the_set, lvl);
		cout << endl;
	}
	gen->unrank_basis(coordinates, the_set, lvl);


	if (f_v) {
		cout << "poset_orbit_node::setup_factor_space_action "
				"initializing action_on_factor_space "
				"dimension=" << gen->get_VS()->dimension
				<< " before AF->init_from_coordinate_vectors" << endl;
	}
	AF->init_from_coordinate_vectors(
		gen->get_VS(),
		*gen->get_A(), *gen->get_A2(),
		coordinates, lvl, f_compute_tables,
		verbose_level);
	if (f_v) {
		cout << "poset_orbit_node::setup_factor_space_action "
				"initializing action_on_factor_space "
				"dimension=" << gen->get_VS()->dimension
				<< " after AF->init_from_coordinate_vectors" << endl;
	}

	if (f_v20) {
		AF->list_all_elements();
	}
	if (f_vv) {
		cout << "poset_orbit_node::setup_factor_space_action "
				"before gen->get_A2()->Induced_action->induced_action_on_factor_space"
				<< endl;
	}
	A_factor_space = gen->get_A2()->Induced_action->induced_action_on_factor_space(
		AF,
		false /*f_induce_action*/,
		NULL /* sims */,
		0/*verbose_level - 3*/);
	if (f_vv) {
		cout << "poset_orbit_node::setup_factor_space_action "
				"after gen->get_A2()->Induced_action->induced_action_on_factor_space"
				<< endl;
	}

	FREE_lint(the_set);
	FREE_int(coordinates);
	if (f_v) {
		cout << "poset_orbit_node::setup_factor_space_action "
				"lvl=" << lvl << endl;
		cout << "poset_orbit_node::setup_factor_space_action "
				"node=" << node << " prev=" << prev
				<< " pt=" << pt << " done" << endl;
	}
}

void poset_orbit_node::downstep_subspace_action_print_orbits(
	poset_classification *gen,
	groups::schreier &Schreier,
	int lvl, 
	int f_print_orbits, 
	int verbose_level)
{
	int h, first, len, rep;
	induced_actions::action_on_factor_space *AF;
	
	cout << "poset_orbit_node::downstep_subspace_action_"
			"print_orbits" << endl;
	gen->print_level_info(lvl, node);
	cout << "poset_orbit_node::downstep_subspace_action_"
			"print_orbits: "
			"The " << Schreier.Forest->nb_orbits
			<< " orbits are:" << endl;
	
	AF = Schreier.Generators_and_images->A->G.AF;
	if (f_print_orbits) {
		cout << "i : orbit rep : orbit length : orbit" << endl;
	}
	else {
		cout << "i : orbit rep : orbit length" << endl;
	}

	for (h = 0; h < Schreier.Forest->nb_orbits; h++) {
		first = Schreier.Forest->orbit_first[h];
		len = Schreier.Forest->orbit_len[h];
		rep = AF->preimage_table[Schreier.Forest->orbit[first + 0]];
		cout << setw(4) << h << " : " 
			<< setw(5) << rep;

		gen->unrank_point(gen->get_VS()->v1, rep);

		cout << " = ";
		Int_vec_print(cout,
				gen->get_VS()->v1,
				gen->get_VS()->dimension);
		cout << " : " << setw(5) << len;
		if (f_print_orbits) {
			if (len < 25) {
				cout << " : ";
				Schreier.Forest->print_orbit_through_labels(
						cout, h, AF->preimage_table);
			}
			else {
				cout << " : too long to print";
			}
		}
		cout << endl;
	}
	cout << "poset_orbit_node::downstep_subspace_action_"
			"print_orbits done" << endl;
}

void poset_orbit_node::downstep_orbits_subspace_action(
	poset_classification *gen, groups::schreier &Schreier,
	int lvl, 
	int f_use_invariant_subset_if_available, 
	int &f_using_invariant_subset, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	//int f_v10 = (verbose_level >= 10);
	induced_actions::action_on_factor_space *AF;

	if (f_v) {
		cout << "poset_orbit_node::downstep_orbits_subspace_action" << endl;
		gen->print_level_info(lvl, node);
		cout << "poset_orbit_node::downstep_orbits_subspace_action "
				"verbose_level = " << verbose_level << endl;
	}
		
	if (false) {
		gen->print_level_info(lvl, node);
		cout << " : generators:" << endl;
		Schreier.Generators_and_images->print_generators();
	}

	if (f_vv) {
		gen->print_level_info(lvl, node);
		cout << " : calling Schreier.compute_all_point_orbits_with_"
				"preferred_labels" << endl;
	}
	//Schreier.compute_all_point_orbits(verbose_level - 4);
	AF = Schreier.Generators_and_images->A->G.AF;

	Schreier.compute_all_point_orbits_with_preferred_labels(
			AF->preimage_table, verbose_level);

	if (f_vv) {
		gen->print_level_info(lvl, node);
		cout << "poset_orbit_node::downstep_orbits_subspace_action: "
				"The " << Schreier.Forest->nb_orbits
				<< " orbits are:" << endl;
		int h;
		cout << "h : orbit_len[h] : orbit : "
				"orbit (via AF->preimage_table)" << endl;
		for (h = 0; h < Schreier.Forest->nb_orbits; h++) {
			cout << setw(4) << h << " / " << Schreier.Forest->nb_orbits << " : ";

			int first, len, rep;


			first = Schreier.Forest->orbit_first[h];
			len = Schreier.Forest->orbit_len[h];
			rep = AF->preimage_table[Schreier.Forest->orbit[first + 0]];

			gen->unrank_point(gen->get_VS()->v1, rep);

			cout << rep << " = ";
			Int_vec_print(cout,
					gen->get_VS()->v1, gen->get_VS()->dimension);

			cout << " : " << setw(5) << Schreier.Forest->orbit_len[h];
			//if (f_vvv) {
			cout << " : ";

			if (len < 100) {
				Schreier.Forest->print_orbit(cout, h);
				cout << " : ";
				Schreier.Forest->print_orbit_through_labels(cout,
						h, AF->preimage_table);
			}
			else {
				cout << "too large to print";
			}
			cout << endl;
		}
		cout << "the orbit elements are:" << endl;
		for (h = 0; h < Schreier.Forest->nb_orbits; h++) {
			cout << "orbit " << setw(4) << h << " / "
				<< Schreier.Forest->nb_orbits << " : " << endl;

			int first, len, rep, j;

			first = Schreier.Forest->orbit_first[h];
			len = Schreier.Forest->orbit_len[h];

			if (len < 100) {
				for (j = 0; j < len; j++) {
					rep = AF->preimage_table[Schreier.Forest->orbit[first + j]];

					gen->unrank_point(gen->get_VS()->v1, rep);

					cout << setw(3) << j << " / " << setw(3) << len
							<< " : " << rep << " = ";
					Int_vec_print(cout,
							gen->get_VS()->v1, gen->get_VS()->dimension);
					cout << " : ";
					cout << Schreier.Forest->prev[first + j];
					cout << " : ";
					cout << Schreier.Forest->label[first + j];
					cout << endl;
				}
			}
			else {
				cout << "too large to print" << endl;
			}
		}
	}

#if 0
	if (false) {
		cout << "Schreier structure generators:" << endl;
		Schreier.list_elements_as_permutations_vertically(cout);


		cout << "Schreier structure table:" << endl;
		Schreier.print_tables(cout, false/* f_with_cosetrep*/);

		int h;
		string fname;
		int xmax = 2000000;
		int ymax = 1000000;
		int f_circletext = true;
		int rad = 20000;
		int f_embedded = true;
		int f_sideways = true;
		double scale = 0.35;
		double line_width = 0.6;
		int f_has_point_labels = false;
		long int *point_labels = NULL;

		for (h = 0; h < Schreier.nb_orbits; h++) {
			fname = "node_" + std::to_string(node) + "_tree_" + std::to_string(h);
			cout << "before Schreier.draw_tree fname = " << fname << endl;
			Schreier.draw_tree(fname, h /* orbit_no */,
				xmax, ymax, f_circletext, rad,
				f_embedded, f_sideways,
				scale, line_width,
				f_has_point_labels, point_labels,
				0 /*verbose_level */);
		}
	}
#endif


#if 0
	if (f_v10) {
		Schreier.print(cout);
		Schreier.print_generators();
		Schreier.print_tables(cout, false /* f_with_cosetrep */);
		if (gen->A->degree < 1000 && false) {
			//Schreier.print_tree(0);
			Schreier.print_tables(cout, false /* f_with_cosetrep */);
			}
		}
#endif
	if (f_v) {
		gen->print_level_info(lvl, node);
		cout << "poset_orbit_node::downstep_orbits: "
				"we found " << Schreier.Forest->nb_orbits << " orbits" << endl;
	}
	if (f_v) {
		cout << "poset_orbit_node::downstep_orbits_subspace_action done" << endl;
	}
}

void poset_orbit_node::find_extensions_subspace_action(
	poset_classification *gen, groups::schreier &O,
	actions::action *A_factor_space,
	induced_actions::action_on_factor_space *AF,
	int lvl, int f_implicit_fusion, int verbose_level)
// prepares all extension nodes and marks them as unprocessed.
// we are at depth lvl, i.e., currently, we have a set of size lvl.
// removes implicit fusion orbits
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int h, k, fst, /*len,*/ pt, pt1;
	
	if (f_v) {
		cout << "poset_orbit_node::find_extensions_subspace_action" << endl;
		gen->print_level_info(lvl, node);
		cout << "poset_orbit_node::find_extensions_subspace_action "
				"computing all possible extensions (out of "
				<< O.Forest->nb_orbits << " orbits)" << endl;
	}
	if (f_vv) {
		cout << "the stabilizer orbits are:" << endl;
		cout << "i : representative : orbit length" << endl;
		for (k = 0; k < O.Forest->nb_orbits; k++) {
			fst = O.Forest->orbit_first[k];
			pt = O.Forest->orbit[fst];
			cout << k << " : " << pt << " : " << O.Forest->orbit_len[k] << endl;
		}
	}
	nb_extensions = 0;
	E = NEW_OBJECTS(extension, O.Forest->nb_orbits);

	store_set(gen, lvl - 1);

	if (f_vv) {
		cout << "k : orbit_rep[k] : preimage[k]" << endl;
	}
	for (k = 0; k < O.Forest->nb_orbits; k++) {
		fst = O.Forest->orbit_first[k];
		//len = O.Forest->orbit_len[k];
		pt1 = O.Forest->orbit[fst];
		pt = AF->preimage(pt1, verbose_level - 2);
		if (f_vv) {
			cout << setw(5) << k << " : " << setw(7)
					<< pt1 << " : " << setw(7) << pt << endl;
		}


		if (f_implicit_fusion) {
			// use implicit fusion nodes
			if (lvl) {
				if (pt <= poset_orbit_node::pt) {
					if (f_vv) {
						cout << "poset_orbit_node::find_extensions_subspace_action "
							"orbit " << k << " is not accepted because "
							<< "we use implicit fusion nodes and " 
							<< pt << " is less than " 
							<< poset_orbit_node::pt << endl;
					}
					continue;
				}
				if (f_vv) {
					cout << "poset_orbit_node::find_extensions_subspace_action "
							"orbit " << k << " is accepted" << endl;
				}
			}
		}
		else {
			// we need to check whether the point is already in the set:
			int ii;
			
			for (ii = 0; ii < lvl; ii++) {
				if (gen->get_S()[ii] == pt) {
					break;
				}
			}
			if (ii < lvl) {
				if (f_vv) {
					cout << "poset_orbit_node::find_extensions_subspace_action "
						"orbit " << k << " is in the set so we skip" << endl;
				}
				continue;
			}
			if (f_vv) {
				cout << "poset_orbit_node::find_extensions_subspace_action "
					"orbit " << k << " is accepted" << endl;
			}
		}

			

		E[nb_extensions].set_pt(pt);
		E[nb_extensions].set_orbit_len(O.Forest->orbit_len[k]);
		E[nb_extensions].set_type(EXTENSION_TYPE_UNPROCESSED);
		nb_extensions++;
	}
	
#if 1
	// reallocate:
	extension *E2 = E;
	int nb_extension_points = 0;
	
	E = NEW_OBJECTS(extension, nb_extensions);
	for (k = 0; k < nb_extensions; k++) {
		E[k] = E2[k]; 
		nb_extension_points += E[k].get_orbit_len();
	}
	FREE_OBJECTS(E2);
#endif

	if (f_v) {
		cout << "poset_orbit_node::find_extensions_subspace_action "
				"found " << nb_extensions << " extensions with "
				<< nb_extension_points << " points (out of "
				<< O.Forest->nb_orbits << " orbits)" << endl;
	}
	if (f_vv) {
		cout << "i : representing point : orbit_length" << endl;
		for (h = 0; h < nb_extensions; h++) {
			cout << h << " : " << E[h].get_pt() << " : "
				<< E[h].get_orbit_len() << endl;
		}
	}
}


}}}


