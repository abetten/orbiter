// oracle_downstep_subspace_action.C
//
// Anton Betten
// Jan 21, 2010

#include "orbiter.h"

void oracle::setup_factor_space_action_light(generator *gen, 
	action_on_factor_space &AF, 
	INT lvl, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *the_set;
		
	if (f_v) {
		cout << "oracle::setup_factor_space_action_light lvl=" << lvl << endl;
		cout << "oracle::setup_factor_space_action_light node=" << node << " prev=" << prev << " pt=" << pt << endl;
		}
	the_set = NEW_INT(lvl);
	store_set_to(gen, lvl - 1, the_set);
	
	AF.init_light(*gen->A, *gen->A2, gen->vector_space_dimension, gen->F, 
		the_set, lvl, 
		gen->rank_point_func, 
		gen->unrank_point_func, 
		gen->rank_point_data, 
		verbose_level - 1);
	FREE_INT(the_set);
}

void oracle::setup_factor_space_action_with_early_test(generator *gen, 
	action_on_factor_space &AF, action &A_factor_space, 
	INT lvl, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *subset;
	INT *the_set;
	INT n, i;
		
	if (f_v) {
		cout << "oracle::setup_factor_space_action_with_early_test lvl=" << lvl << endl;
		cout << "oracle::setup_factor_space_action_with_early_test node=" << node << " prev=" << prev << " pt=" << pt << endl;
		cout << "oracle::setup_factor_space_action_with_early_test A2->degree=" << gen->A2->degree << endl;
		}
	the_set = NEW_INT(lvl + 1);
		// the +1 is for safety !!! 
		// Namely, so that the test function has one more entry 
		// to store the candidate point in.
		// A Betten Nov 21, 2011


	store_set_to(gen, lvl - 1, the_set);
	
	if (lvl) {
		if (f_vv) {
			cout << "oracle::setup_factor_space_action_with_early_test retrieving candidate set from Schreier vector of node " << prev << endl;
			}
		INT *osv = gen->root[prev].sv;
		n = osv[0];
		subset = osv + 1;
		}
	else {
		n = gen->A2->degree;
		subset = NEW_INT(n);
		for (i = 0; i < n; i++) {
			subset[i] = i;
			}
		}
	INT *candidates;
	INT nb_candidates;
		
	if (f_vv) {
		gen->print_level_info(lvl + 1, node);
		cout << " : number of live points = " << n << endl;
		}
	candidates = NEW_INT(gen->A2->degree);

	if (f_vv) {
		cout << "oracle::setup_factor_space_action_with_early_test calling early_test_func, degree = " << gen->A2->degree << " n = " << n << endl;
		}
	(*gen->early_test_func)(the_set, lvl, subset, n, 
		candidates, nb_candidates, gen->early_test_func_data, verbose_level - 2);
	if (f_vv) {
		cout << "oracle::setup_factor_space_action_with_early_test after early_test_func, degree = " << gen->A2->degree << " n = " << n << " nb_candidates=" << nb_candidates << endl;
		//cout << "candidates: ";
		//INT_vec_print(cout, candidates, nb_candidates);
		//cout << endl;
		}

	if (f_vv) {
		cout << "oracle::setup_factor_space_action_with_early_test before AF.init_by_rank_table_mode" << endl;
		}
	AF.init_by_rank_table_mode(*gen->A, *gen->A2, gen->vector_space_dimension, gen->F, 
		the_set, lvl, 
		candidates, nb_candidates, 
		gen->rank_point_func, 
		gen->unrank_point_func, 
		gen->rank_point_data, 
		verbose_level - 3);
	if (f_vv) {
		cout << "oracle::setup_factor_space_action_with_early_test after AF.init_by_rank_table_mode" << endl;
		}
	
	FREE_INT(candidates);
	FREE_INT(the_set);
	if (lvl == 0) {
		FREE_INT(subset);
		}
		
	if (f_vv) {
		cout << "oracle::setup_factor_space_action_with_early_test before A_factor_space.induced_action_on_factor_space" << endl;
		}
	A_factor_space.induced_action_on_factor_space(gen->A2, &AF, 
		FALSE /*f_induce_action*/, NULL /* sims */, 0/*verbose_level - 3*/);
	if (f_vv) {
		cout << "oracle::setup_factor_space_action_with_early_test after A_factor_space.induced_action_on_factor_space" << endl;
		}

	if (f_v) {
		cout << "oracle::setup_factor_space_action_with_early_test lvl=" << lvl << endl;
		cout << "oracle::setup_factor_space_action_with_early_test node=" << node << " prev=" << prev << " pt=" << pt << " done" << endl;
		}
}

void oracle::setup_factor_space_action(generator *gen, 
	action_on_factor_space &AF, action &A_factor_space, 
	INT lvl, INT f_compute_tables, INT verbose_level)
// called from oracle::init_extension_node, 
// oracle::orbit_representative_and_coset_rep_inv_subspace_action (in oracle_upstep_subspace_action)
// oracle::downstep_subspace_action
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_v20 = (verbose_level >= 20);
	INT *the_set;
	INT *coordinates;
	INT i;

	if (f_v) {
		cout << "oracle::setup_factor_space_action lvl=" << lvl << endl;
		cout << "oracle::setup_factor_space_action node=" << node << " prev=" << prev << " pt=" << pt << endl;
		cout << "f_compute_tables=" << f_compute_tables << endl;
		}
	the_set = NEW_INT(lvl);
	coordinates = NEW_INT(lvl * gen->vector_space_dimension);
	store_set_to(gen, lvl - 1, the_set);

	if (f_v) {
		cout << "the set: ";
		INT_vec_print(cout, the_set, lvl);
		cout << endl;
		cout << "oracle::setup_factor_space_action initializing action_on_factor_space dimension=" << gen->vector_space_dimension << endl;
		}
	for (i = 0; i < lvl; i++) {
		(*gen->unrank_point_func)(coordinates + i * gen->vector_space_dimension, the_set[i], 
			gen->rank_point_data);
		}
	//AF.init_by_rank(*gen->A2, gen->vector_space_dimension, gen->F, the_set, lvl, verbose_level);
	AF.init_from_coordinate_vectors(*gen->A, *gen->A2, gen->vector_space_dimension, gen->F, 
		coordinates, lvl, f_compute_tables, verbose_level);	
	if (f_v20) {
		AF.list_all_elements();
		}
	if (f_vv) {
		cout << "oracle::setup_factor_space_action before A_factor_space->induced_action_on_factor_space" << endl;
		}
	A_factor_space.induced_action_on_factor_space(gen->A2, &AF, 
		FALSE /*f_induce_action*/, NULL /* sims */, 0/*verbose_level - 3*/);
	if (f_vv) {
		cout << "oracle::setup_factor_space_action after A_factor_space->induced_action_on_factor_space" << endl;
		}
	
	FREE_INT(the_set);
	FREE_INT(coordinates);
}

void oracle::downstep_subspace_action(generator *gen, 
	INT lvl, 
	INT f_create_schreier_vector, INT f_compact, 
	INT f_use_invariant_subset_if_available, 
	INT f_implicit_fusion, 
	INT verbose_level)
{
	//if (node == 0) {verbose_level += 20; cout << "oracle::downstep_subspace_action node 0 reached" << endl;}
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT f_v4 = (verbose_level >= 4);
	INT nb_orbits;
	//INT good_orbits1, nb_points1;
	INT f_using_invariant_subset = FALSE;
	schreier *Schreier;
	action_on_factor_space *AF;
	action *A_factor_space;


	if (f_v) {
		cout << "oracle::downstep_subspace_action" << endl;
		}
	store_set(gen, lvl - 1); // stores a set of size lvl to gen->S
	

	Schreier = new schreier;
	AF = new action_on_factor_space;
	A_factor_space = new action;
	
	if (f_v) {
		gen->print_level_info(lvl + 1, node);
		cout << " : Downstep for ";
		print_set(gen);
		cout <<  " in subspace_action";
		cout << " verbose_level=" << verbose_level << endl;
		cout << "gen->f_early_test_func=" << gen->f_early_test_func << endl;
		if (f_vvv) {
			print_set_verbose(gen);
			}
		if (prev >= 0 && gen->root[prev].sv) {
			INT nb = gen->root[prev].sv[0];
			cout << " with " << nb << " live points";
			if (f_vvv) {
				cout << " : ";
				INT_vec_print(cout, gen->root[prev].sv + 1, nb);
				cout << endl;
				}
			else {
				cout << endl;
				}
			}
		cout << endl;
		}

	strong_generators *Strong_gens;
	longinteger_object go;

	get_stabilizer_generators(gen, Strong_gens, verbose_level);
	Strong_gens->group_order(go);
	if (f_v) {
		cout << "The stabilizer is a group of order " << go << endl;
		if (f_vv) {
			cout << "With the following generators:" << endl;
			Strong_gens->print_generators_ost(cout);
			}
		}
	

	if (gen->f_early_test_func) {

		if (f_v) {
			cout << "oracle::downstep_subspace_action before setup_factor_space_action_with_early_test" << endl;
			}
		setup_factor_space_action_with_early_test(gen, 
			*AF, *A_factor_space, 
			lvl, verbose_level - 2);

		if (f_v) {
			cout << "oracle::downstep_subspace_action after setup_factor_space_action_with_early_test" << endl;
			}

		}
	else {
		if (f_v) {
			cout << "oracle::downstep_subspace_action before setup_factor_space_action" << endl;
			}
		setup_factor_space_action(gen, *AF, *A_factor_space, lvl, TRUE /*f_compute_tables*/, verbose_level - 7);
		if (f_v) {
			cout << "oracle::downstep_subspace_action after setup_factor_space_action" << endl;
			}
		}
	
	
	if (f_v) {
		cout << "oracle::downstep_subspace_action before Schreier.init" << endl;
		}


	Schreier->init(A_factor_space);




	if (f_v) {
		cout << "oracle::downstep_subspace_action before Schreier.init_generators_by_hdl" << endl;
		}
	Schreier->init_generators_by_hdl(nb_strong_generators, hdl_strong_generators, verbose_level - 1);

	if (f_v) {
		cout << "oracle::downstep_subspace_action before downstep_orbits_subspace_action" << endl;
		}
	downstep_orbits_subspace_action(gen, *Schreier, 
		lvl, 
		f_use_invariant_subset_if_available, 
		f_using_invariant_subset, 
		verbose_level - 2);



	nb_orbits = Schreier->nb_orbits;
	if (f_v) {
		cout << "oracle::downstep_subspace_action after downstep_orbits_subspace_action nb_orbits=" << nb_orbits << endl;
		}
	
	if (f_v) {
		cout << "oracle::downstep_subspace_action before create_schreier_vector_wrapper_subspace_action " << endl;
		}
	create_schreier_vector_wrapper_subspace_action(
		f_create_schreier_vector, 
		f_compact, 
		*Schreier, 
		A_factor_space, AF, 
		verbose_level - 2);
	if (f_v) {
		cout << "oracle::downstep_subspace_action after create_schreier_vector_wrapper_subspace_action " << endl;
		}

#if 0
	downstep_orbit_test_and_schreier_vector(
		gen, Schreier, 
		lvl, 
		f_use_invariant_subset_if_available, 
		f_using_invariant_subset,
		f_create_schreier_vector,
		f_compact, 
		good_orbits1, nb_points1, 
		verbose_level - 1);


	downstep_implicit_fusion(
		gen, Schreier, 
		lvl, 
		f_implicit_fusion, 
		good_orbits1, nb_points1, 
		verbose_level - 1);

#endif
	if (f_v4) {
		gen->print_level_info(lvl + 1, node);
		cout << " : calling find_extensions_subspace_action" << endl;
		}
	if (f_v) {
		cout << "oracle::downstep_subspace_action before find_extensions_subspace_action" << endl;
		}
	find_extensions_subspace_action(
		gen, *Schreier, 
		A_factor_space, AF, 
		lvl, f_implicit_fusion, verbose_level - 1);
	if (f_v) {
		cout << "oracle::downstep_subspace_action after find_extensions_subspace_action" << endl;
		}
	if (f_v4) {
		gen->print_level_info(lvl + 1, node);
		cout << " : after test_orbits and find_extensions, we have " << nb_extensions << " extensions" << endl;
		}
	
	
	if (f_v) {
		gen->print_level_info(lvl + 1, node);
		cout << " : found " << nb_extensions << " extensions (out of " << nb_orbits << " orbits) with " << nb_extension_points() << " points " << endl;
		}
	if (f_vv) {
		downstep_subspace_action_print_orbits(
			gen, *Schreier, 
			lvl, 
			f_vvv /* f_print_orbits */, 
			verbose_level);
		}
	if (f_v) {
		cout << "oracle::downstep_subspace_action before deleting things" << endl;
		}
	delete Strong_gens;
	delete Schreier;
	delete A_factor_space;
	delete AF;
	if (f_v) {
		cout << "oracle::downstep_subspace_action done" << endl;
		}

}

void oracle::downstep_subspace_action_print_orbits(
	generator *gen, schreier &Schreier, 
	INT lvl, 
	INT f_print_orbits, 
	INT verbose_level)
{
	INT h, first, len, rep;
	action_on_factor_space *AF;
	
	cout << "oracle::downstep_subspace_action_print_orbits" << endl;
	gen->print_level_info(lvl + 1, node);
	cout << "oracle::downstep_subspace_action_print_orbits: The " << Schreier.nb_orbits << " orbits are:" << endl;
	
	AF = Schreier.A->G.AF;
	if (f_print_orbits) {
		cout << "i : orbit rep : orbit length : orbit" << endl;
		}
	else {
		cout << "i : orbit rep : orbit length" << endl;
		}

	for (h = 0; h < Schreier.nb_orbits; h++) {
		first = Schreier.orbit_first[h];
		len = Schreier.orbit_len[h];
		rep = AF->preimage_table[Schreier.orbit[first + 0]];
		cout << setw(4) << h << " : " 
			<< setw(5) << rep << " : "
			<< setw(5) << len;
		if (f_print_orbits) {
			if (len < 25) {
				cout << " : ";
				Schreier.print_orbit_through_labels(cout, h, AF->preimage_table);
				}
			else {
				cout << " : too long to print";
				}
			}
		cout << endl;
		}
	cout << "oracle::downstep_subspace_action_print_orbits done" << endl;
}

void oracle::downstep_orbits_subspace_action(
	generator *gen, schreier &Schreier, 
	INT lvl, 
	INT f_use_invariant_subset_if_available, 
	INT &f_using_invariant_subset, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	//INT f_v10 = (verbose_level >= 10);
	action_on_factor_space *AF;

	if (f_v) {
		cout << "oracle::downstep_orbits_subspace_action" << endl;
		gen->print_level_info(lvl + 1, node);
		cout << "oracle::downstep_orbits_subspace_action" << endl;
		}
		
	if (FALSE) {
		gen->print_level_info(lvl + 1, node);
		cout << " : generators:" << endl;
		Schreier.print_generators();
		}

	if (f_vv) {
		gen->print_level_info(lvl + 1, node);
		cout << " : calling Schreier.compute_all_point_orbits_with_preferred_labels" << endl;
		}
	//Schreier.compute_all_point_orbits(verbose_level - 4);
	AF = Schreier.A->G.AF;
	Schreier.compute_all_point_orbits_with_preferred_labels(AF->preimage_table, verbose_level);

	if (f_vv) {
		gen->print_level_info(lvl + 1, node);
		cout << "oracle::downstep_orbits_subspace_action: The " << Schreier.nb_orbits << " orbits are:" << endl;
		INT h;
		cout << "h : orbit_len[h] : orbit : orbit (via AF->preimage_table)" << endl;
		for (h = 0; h < Schreier.nb_orbits; h++) {
			cout << setw(4) << h << " / " << Schreier.nb_orbits << " : " << setw(5) << Schreier.orbit_len[h];
			//if (f_vvv) {
			cout << " : ";
			Schreier.print_orbit(cout, h);
			cout << " : ";
			Schreier.print_orbit_through_labels(cout, h, AF->preimage_table);
			//	}
			cout << endl;
			}
		}
#if 0
	if (f_v10) {
		Schreier.print(cout);
		Schreier.print_generators();
		Schreier.print_tables(cout, FALSE /* f_with_cosetrep */);
		if (gen->A->degree < 1000 && FALSE) {
			//Schreier.print_tree(0);
			Schreier.print_tables(cout, FALSE /* f_with_cosetrep */);
			}
		}
#endif
	if (f_v) {
		gen->print_level_info(lvl + 1, node);
		cout << "oracle::downstep_orbits: we found " << Schreier.nb_orbits << " orbits" << endl;
		}
	if (f_v) {
		cout << "oracle::downstep_orbits_subspace_action done" << endl;
		}
}

void oracle::find_extensions_subspace_action(generator *gen, schreier &O, 
	action *A_factor_space, action_on_factor_space *AF, 
	INT lvl, INT f_implicit_fusion, INT verbose_level)
// prepares all extension nodes and marks them as unprocessed.
// we are at depth lvl, i.e., currently, we have a set of size lvl.
// removes implicit fusion orbits
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT h, k, fst, len, pt, pt1;
	
	if (f_v) {
		cout << "oracle::find_extensions_subspace_action" << endl;
		gen->print_level_info(lvl + 1, node);
		cout << "oracle::find_extensions_subspace_action computing all possible extensions (out of " << O.nb_orbits << " orbits)" << endl;
		}
	if (f_vv) {
		cout << "the stabilizer orbits are:" << endl;
		cout << "i : representative : orbit length" << endl;
		for (k = 0; k < O.nb_orbits; k++) {
			fst = O.orbit_first[k];
			pt = O.orbit[fst];
			cout << k << " : " << pt << " : " << O.orbit_len[k] << endl;
			}
		}
	nb_extensions = 0;
	E = new extension[O.nb_orbits];

	store_set(gen, lvl - 1);

	if (f_vv) {
		cout << "k : orbit_rep[k] : preimage[k]" << endl;
		}
	for (k = 0; k < O.nb_orbits; k++) {
		fst = O.orbit_first[k];
		len = O.orbit_len[k];
		pt1 = O.orbit[fst];
		pt = AF->preimage(pt1, verbose_level - 2);
		if (f_vv) {
			cout << setw(5) << k << " : " << setw(7) << pt1 << " : " << setw(7) << pt << endl;
			}


		if (f_implicit_fusion) {
			// use implicit fusion nodes
			if (lvl) {
				if (pt <= oracle::pt) {
					if (f_vv) {
						cout << "oracle::find_extensions_subspace_action orbit " << k << " is not accepted because "
							<< "we use implicit fusion nodes and " 
							<< pt << " is less than " 
							<< oracle::pt << endl;
						}
					continue;
					}
				if (f_vv) {
					cout << "oracle::find_extensions_subspace_action orbit " << k << " is accepted" << endl;
					}
				}
			}
		else {
			// we need to check whether the point is already in the set:
			INT ii;
			
			for (ii = 0; ii < lvl; ii++) {
				if (gen->S[ii] == pt)
					break;
				}
			if (ii < lvl) {
				if (f_vv) {
					cout << "oracle::find_extensions_subspace_action orbit " << k << " is in the set so we skip" << endl;
					}
				continue;
				}
			if (f_vv) {
				cout << "oracle::find_extensions_subspace_action orbit " << k << " is accepted" << endl;
				}
			}

			

		E[nb_extensions].pt = pt;
		E[nb_extensions].orbit_len = O.orbit_len[k];
		E[nb_extensions].type = EXTENSION_TYPE_UNPROCESSED;
		nb_extensions++;
		}
	
#if 1
	// reallocate:
	extension *E2 = E;
	INT nb_extension_points = 0;
	
	E = new extension[nb_extensions];
	for (k = 0; k < nb_extensions; k++) {
		E[k] = E2[k]; 
		nb_extension_points += E[k].orbit_len;
		}
	delete [] E2;
#endif

	if (f_v) {
		cout << "oracle::find_extensions_subspace_action found " << nb_extensions << " extensions with " << nb_extension_points << " points (out of " << O.nb_orbits << " orbits)" << endl;
		}
	if (f_vv) {
		cout << "i : representing point : orbit_length" << endl;
		for (h = 0; h < nb_extensions; h++) {
			cout << h << " : " << E[h].pt << " : " << E[h].orbit_len << endl;
			}
		}
}

void oracle::create_schreier_vector_wrapper_subspace_action(
	INT f_create_schreier_vector, INT f_compact, 
	schreier &Schreier, 
	action *A_factor_space, action_on_factor_space *AF, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 10);
	
	if (f_v) {
		cout << "oracle::create_schreier_vector_wrapper_subspace_action" << endl;
		}
	if (/*nb_strong_generators &&*/ f_create_schreier_vector) {
		INT f_trivial_group;
		
		if (f_vv) {
			cout << "calling get_schreier_vector" << endl;
			}
		if (nb_strong_generators == 0) {
			f_trivial_group = TRUE;
			}
		else {
			f_trivial_group = FALSE;
			}
		Schreier.get_schreier_vector(sv, f_trivial_group, f_compact);
		//Schreier.test_sv(gen->A, hdl_strong_generators, sv, f_trivial_group, f_compact, verbose_level);

		if (f_vv) {
			cout << "schreier vector before relabeling :" << endl;
			INT_vec_print(cout, sv + 1, sv[0]);
			cout << endl;
			}
		if (f_v) {
			cout << "oracle::create_schreier_vector_wrapper_subspace_action changing point labels:" << endl;
			}
		schreier_vector_relabel_points(sv, AF, f_compact, f_trivial_group, verbose_level - 4);
		if (f_v) {
			cout << "oracle::create_schreier_vector_wrapper_subspace_action changing point labels done" << endl;
			}
		if (f_vv) {
			cout << "schreier vector after relabeling :" << endl;
			INT_vec_print(cout, sv + 1, sv[0]);
			cout << endl;
			}
		}
	else {
		sv = NULL;
		}
	if (f_v) {
		cout << "oracle::create_schreier_vector_wrapper_subspace_action done" << endl;
		}
}

void schreier_vector_relabel_points(INT *sv, action_on_factor_space *AF, 
	INT f_compact, INT f_trivial_group, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT n;
	INT *pts;
	INT *prev;
	INT *label;
	INT i, pt, pre, q, pr, new_pr, pos;
	INT *new_sv;
	INT *new_pts;
	INT *new_pts_sorted;
	INT *perm;
	INT *new_sv_pts;
	INT *new_sv_prev;
	INT *new_sv_label;

	INT nb_old_orbit_reps = 0, idx, j;
	INT *old_orbit_reps = NULL;

	if (f_v) {
		cout << "schreier_vector_relabel_points" << endl;
		}
	if (!f_compact) {
		cout << "schreier_vector_relabel_points changing point labels: fatal: !f_compact" << endl;
		exit(1);
		}
	n = sv[0];
	pts = sv + 1;
	prev = pts + n;
	label = prev + n;

	if (f_trivial_group) {
		if (f_v) {
			cout << "trivial group" << endl;
			}
		new_sv = NEW_INT(n + 1);
		new_pts = new_sv + 1;
		new_sv[0] = n;
		for (i = 0; i < n; i++) {
			pt = pts[i];
			pre = AF->preimage(pt, 0 /*verbose_level - 3*/);
			q = AF->project_onto_Gauss_reduced_vector(pre, 0 /*verbose_level - 2*/);
			if (FALSE) {
				cout << "i=" << i << " pt=" << pt << " pre=" << pre << " q=" << q << endl;
				}
			new_pts[i] = q;
			}
		INT_vec_heapsort(new_pts, n);
		for (i = 0; i < n + 1; i++) {
			sv[i] = new_sv[i];
			}
		FREE_INT(new_sv);
		return;
		}


	new_sv = NEW_INT(3 * n + 1);
	new_pts = NEW_INT(n);
	new_pts_sorted = NEW_INT(n);
	perm = NEW_INT(n);
	new_sv_pts = new_sv + 1;
	new_sv_prev = new_sv_pts + n;
	new_sv_label = new_sv_prev + n;
	for (i = 0; i < n; i++) {
		perm[i] = i;
		}
	if (f_v) {
		nb_old_orbit_reps = 0;
		cout << "old orbit reps:" << endl;
		for (i = 0; i < n; i++) {
			if (prev[i] == -1) {
				cout << "orbit rep " << pts[i] << endl;
				nb_old_orbit_reps++;
				}
			}
		old_orbit_reps = NEW_INT(nb_old_orbit_reps);
		j = 0;
		for (i = 0; i < n; i++) {
			if (prev[i] == -1) {
				old_orbit_reps[j++] = pts[i];
				}
			}
		INT_vec_heapsort(old_orbit_reps, nb_old_orbit_reps);
		INT_vec_print(cout, old_orbit_reps, nb_old_orbit_reps);
		cout << endl;
		cout << "There are " << nb_old_orbit_reps << " old orbit reps, they are:" << endl;
		for (i = 0; i < nb_old_orbit_reps; i++) {
			cout << i << " / " << nb_old_orbit_reps << " : " << old_orbit_reps[i] << endl;
			}
		}
	if (f_vv) {
		cout << "before:" << endl;
		for (i = 0; i < n; i++) {
			if (INT_vec_search(old_orbit_reps, nb_old_orbit_reps, pts[i], idx)) {
				cout << setw(5) << i << " : " << setw(5) << pts[i] << endl;
				}
			}
		}
	if (f_vv) {
		cout << "computing new_pts" << endl;
		}
	for (i = 0; i < n; i++) {
		pt = pts[i];
		if (FALSE) {
			cout << "i=" << i << " pt=" << pt << endl;
			}
		pre = AF->preimage(pt, 0/*verbose_level - 3*/);
		if (FALSE) {
			cout << "pre=" << pre << endl;
			}
		q = AF->project_onto_Gauss_reduced_vector(pre, 0 /*verbose_level - 2*/);
		if (FALSE) {
			if (INT_vec_search(old_orbit_reps, nb_old_orbit_reps, pt, idx)) {
				cout << "i=" << i << " pt=" << pt << " pre=" << pre << " q=" << q << endl << endl;
				}
			}
		new_pts[i] = q;
		}
	if (f_vv) {
		//cout << "after:" << endl;
		cout << "i : pts[i] : new_pts[i]" << endl;
		for (i = 0; i < n; i++) {
			if (INT_vec_search(old_orbit_reps, nb_old_orbit_reps, pts[i], idx)) {
				cout << setw(5) << i << " : " << setw(5) << pts[i] << " : " << setw(5) << new_pts[i] << endl;
				}
			}
		}
	if (f_vv) {
		cout << "sorting:" << endl;
		}
	for (i = 0; i < n; i++) {
		new_pts_sorted[i] = new_pts[i];
		}
	INT_vec_heapsort_with_log(new_pts_sorted, perm, n);
	if (f_vv) {
		cout << "after sorting:" << endl;
		cout << "i : pts[i] : new_pts_sorted[i] : perm[i]" << endl;
		for (i = 0; i < n; i++) {
			if (INT_vec_search(old_orbit_reps, nb_old_orbit_reps, pts[i], idx)) {
				cout << setw(5) << i << " : " 
					<< setw(5) << pts[i] << " : " 
					<< setw(5) << new_pts_sorted[i] << " : " << setw(5) << perm[i] << endl;
				}
			}
		}
	new_sv[0] = n;
	for (i = 0; i < n; i++) {
		new_sv_pts[i] = new_pts_sorted[i];
		pos = perm[i];
		pr = prev[pos];
		if (pr == -1) {
			new_pr = -1;
			}
		else {
			new_pr = new_pts[pr];
			}
		new_sv_prev[i] = new_pr;
		new_sv_label[i] = label[pos];
		}
	if (f_vv) {
		cout << "old / new schreier vector:" << endl;
		cout << "i : pts[i] : prev[i] : label[i] :: i : new_sv_pts[i] : new_sv_prev[i] : new_sv_label[i] " << endl;
		for (i = 0; i < n; i++) {
			cout << setw(5) << i << " : " 
				<< setw(5) << pts[i] << " : " 
				<< setw(5) << prev[i] << " : " 
				<< setw(5) << label[i] 
				<< " :: ";

			cout << setw(5) << i << " : " 
				<< setw(5) << new_sv_pts[i] << " : " 
				<< setw(5) << new_sv_prev[i] << " : " 
				<< setw(5) << new_sv_label[i] 
				<< endl;
			}
		cout << "i : orbit_rep : lexleast : project : project : preimage" << endl;
		for (i = 0; i < n; i++) {
			if (new_sv_prev[i] == -1) {
				cout << i << " : ";
				//cout << "new_sv_pts[i]=" << new_sv_pts[i] << endl;
				//cout << "AF->lexleast_element_in_coset(new_sv_pts[i], 0)=" << AF->lexleast_element_in_coset(new_sv_pts[i], 0) << endl;
				//cout << "AF->project(new_sv_pts[i], 0)=" << AF->project(new_sv_pts[i], 0) << endl;
				//cout << "AF->preimage(AF->project(new_sv_pts[i], 0), 0)=" << AF->preimage(AF->project(new_sv_pts[i], 0), 0) << endl;
				cout << setw(6) << new_sv_pts[i] << " : ";
				cout << setw(6) << AF->lexleast_element_in_coset(new_sv_pts[i], 0) << " : ";
				cout << setw(6) << AF->project(new_sv_pts[i], 0) << " : ";
				cout << setw(6) << AF->preimage(AF->project(new_sv_pts[i], 0), 0) << endl;
				}
			}
		cout << "copying over" << endl;
		}
	for (i = 0; i < 3 * n + 1; i++) {
		sv[i] = new_sv[i];
		}
	FREE_INT(new_sv);
	FREE_INT(new_pts);
	FREE_INT(new_pts_sorted);
	FREE_INT(perm);
	if (old_orbit_reps) {
		FREE_INT(old_orbit_reps);
		}
	if (f_v) {
		cout << "new schreier vector created" << endl;
		cout << "schreier_vector_relabel_points done" << endl;
		}
}

