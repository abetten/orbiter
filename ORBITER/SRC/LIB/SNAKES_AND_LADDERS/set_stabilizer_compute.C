// set_stabilizer_compute.C
// 
// Anton Betten
// moved here from STABILIZER/stabilizer.C
// Dec 28, 2008
//
//

#include "orbiter.h"


set_stabilizer_compute::set_stabilizer_compute()
{
	A = NULL;
	the_set = NULL;
	the_set_sorted = NULL;
	the_set_sorting_perm = NULL;
	the_set_sorting_perm_inv = NULL;
	gen = NULL;
}

set_stabilizer_compute::~set_stabilizer_compute()
{
	INT verbose_level = 0;
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_stabilizer_compute::~set_stabilizer_compute" << endl;
		}
	if (the_set) {
		FREE_INT(the_set);
		the_set = NULL;
		}
	if (the_set_sorted) {
		FREE_INT(the_set_sorted);
		the_set_sorted = NULL;
		}
	if (the_set_sorting_perm) {
		FREE_INT(the_set_sorting_perm);
		the_set_sorting_perm = NULL;
		}
	if (the_set_sorting_perm_inv) {
		FREE_INT(the_set_sorting_perm_inv);
		the_set_sorting_perm_inv = NULL;
		}
	if (gen) {
		if (f_v) {
			cout << "set_stabilizer_compute::~set_stabilizer_compute before freeing gen" << endl;
			}
		delete gen;
		if (f_v) {
			cout << "set_stabilizer_compute::~set_stabilizer_compute after freeing gen" << endl;
			}
		gen = NULL;
		}
	if (f_v) {
		cout << "set_stabilizer_compute::~set_stabilizer_compute done" << endl;
		}
}

void set_stabilizer_compute::init(action *A, INT *set, INT size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	longinteger_object go;

	if (f_v) {
		A->group_order(go);
		cout << "set_stabilizer_compute::init group of degree " << A->degree 
			<< " and order " << go << ", set of size " << size << endl;
		}
	
	if (!test_if_set_with_return_value(set, size)) {
		cout << "set_stabilizer_compute::init the set is not a set" << endl;
		exit(1);
		}
	if (!A->f_has_strong_generators) {
		cout << "set_stabilizer_compute::init action has no strong generators" << endl;
		cout << "the action is " << endl;
		A->print_info();
		exit(1);
		}
	init_with_strong_generators(A, A, A->Strong_gens, set, size, verbose_level);
}

void set_stabilizer_compute::init_with_strong_generators(action *A, action *A0, 
	strong_generators *Strong_gens, 
	INT *set, INT size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT t;
	longinteger_object go;
	
	set_stabilizer_compute::A = A0;
	set_stabilizer_compute::A2 = A;
	gen = new generator;
	
	set_stabilizer_compute::set_size = size;
	
	if (f_v) {
		Strong_gens->group_order(go);
		cout << "set_stabilizer_compute::init_with_strong_generators group of degree " << A->degree 
			<< " and order " << go << ", set of size " << size << endl;
		}
	
	the_set = NEW_INT(set_size);
	the_set_sorted = NEW_INT(set_size);
	the_set_sorting_perm = NEW_INT(set_size);
	the_set_sorting_perm_inv = NEW_INT(set_size);
	
	INT_vec_copy(set, the_set, set_size);
	INT_vec_copy(set, the_set_sorted, set_size);

	INT_vec_sorting_permutation(the_set_sorted, set_size, 
		the_set_sorting_perm, 
		the_set_sorting_perm_inv, TRUE /* f_increasingly */);
	for (t = 0; t < set_size; t++) {
		the_set_sorted[the_set_sorting_perm[t]] = the_set[t];
		}
	
	if (f_v) {
		cout << "Stabilizing the set ";
		INT_vec_print(cout, the_set_sorted, set_size);
		cout << endl;
		}
	if (FALSE /*f_vv*/) {
		cout << "set_stabilizer_compute::init_with_strong_generators  the_set:" << endl;
		cout << "t : the_set : the_set_sorted : the_set_sorting_perm : the_set_sorting_perm_inv" << endl;
		for (t = 0; t < set_size; t++) {
			cout << setw(4) << t << " : " << setw(4) << the_set[t] << " : " << setw(4) << the_set_sorted[t] << " : " << setw(4) << the_set_sorting_perm[t] << " : " << setw(4) << the_set_sorting_perm_inv[t] << endl;
			}
		}

	gen->depth = set_size;
	
	if (f_vv) {
		cout << "set_stabilizer_compute::init_with_strong_generators calling gen->init:" << endl;
		}
	gen->init(A0, A2, Strong_gens, gen->depth /* sz */, verbose_level - 2);

	//gen->init_check_func(set_stabilizer_compute_callback_check_conditions, 
	//	this /* check_conditions_data */);

	if (f_vv) {
		cout << "set_stabilizer_compute::init_with_strong_generators gen->init finished" << endl;
		}

#if 0
	gen->f_print_function = TRUE;
	gen->print_function = print_code;
	gen->print_function_data = this;
#endif
	gen->f_its_OK_to_not_have_an_early_test_func = TRUE;

	INT nb_oracle_nodes = ONE_MILLION;
	
	gen->init_oracle(nb_oracle_nodes, verbose_level - 1);

	if (f_vv) {
		cout << "set_stabilizer_compute::init_with_strong_generators calling init_root_node:" << endl;
		}

	gen->init_root_node(verbose_level - 1);

	if (f_vv) {
		cout << "set_stabilizer_compute::init_with_strong_generators init_root_node finished" << endl;
		}
}

void set_stabilizer_compute::compute_set_stabilizer(INT t0, INT &nb_backtrack_nodes, strong_generators *&Aut_gens, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, lvl, depth_completed = 0;
	INT f_create_schreier_vector = TRUE;
	INT f_compact = TRUE;
	INT f_use_invariant_subset_if_available = TRUE;
	INT f_write_candidate_file = FALSE;
// f_use_invariant_subset_if_available is an option that affects the downstep.
// if FALSE, the orbits of the stabilizer on all points are computed. 
// if TRUE, the orbits of the stabilizer on the set of points that were 
// possible in the previous level are computed only 
// (using Schreier.orbits_on_invariant_subset_fast).
// The set of possible points is stored 
// inside the schreier vector data structure (sv).
	INT vl;
	INT f_debug = FALSE;
	
	if (f_v) {
		cout << "set_stabilizer_compute::compute_set_stabilizer depth = " << gen->depth << endl;
		}

	INT *frequency = NULL; // [nb_orbits]
	INT nb_orbits = 0;
	INT *orbit_idx_of_subset; // [n_choose_k]
	INT n_choose_k;
	INT counter = 0;

	nb_backtrack_nodes = 0;
	for (lvl = depth_completed; lvl <= gen->depth; lvl++) {

		if (f_v) {
			cout << "set_stabilizer_compute::compute_set_stabilizer: computing frequencies of " << lvl << "-set orbits in the set" << endl;
			}
		
		compute_frequencies(lvl, frequency, nb_orbits, orbit_idx_of_subset, n_choose_k, 0 /*verbose_level - 1*/);
		if (f_v) {
			cout << "set_stabilizer_compute::compute_set_stabilizer: computing frequencies of " << lvl << "-set orbits finished" << endl;
			}
		if (f_vv) {
			print_frequencies(lvl, frequency, nb_orbits);
			}
			
		if (handle_frequencies(lvl, 
			frequency, nb_orbits, orbit_idx_of_subset, 
			counter, n_choose_k, Aut_gens, verbose_level)) {
			nb_backtrack_nodes += overall_backtrack_nodes;
			cout << "set_stabilizer_compute::compute_set_stabilizer stabilizer has been computed" << endl;
			return;
			}
		
		if (lvl == gen->depth) {
			break;
			}
			
		if (f_v) {
			cout << "set_stabilizer_compute::compute_set_stabilizer: calling extend_level " << lvl << endl;
			}

		gen->extend_level(lvl, /* lvl,*/ 
			f_create_schreier_vector, f_compact, 
			f_use_invariant_subset_if_available, 
			f_debug, 
			f_write_candidate_file, 
			MINIMUM(1, verbose_level - 1));

		if (f_v) {
			cout << "orbits at level " << lvl + 1 << ":" << endl;
			gen->print_representatives_at_level(lvl + 1);
			//gen->list_all_orbits_at_level(lvl + 1);
			}

		
		if (lvl + 1 == gen->sz) {
			vl = verbose_level;
			}
		else {
			vl = verbose_level - 1;
			}
		gen->housekeeping_no_data_file(lvl + 1, t0, vl);
	
		FREE_INT(frequency);
		FREE_INT(orbit_idx_of_subset);
		
		} // next lvl
	if (f_v) {
		cout << "set_stabilizer_compute::compute_set_stabilizer level " << lvl << " reached" << endl;
		}
	INT f, idx = -1, nd;
	
	for (i = 0; i < nb_orbits; i++) {
		if (frequency[i]) {
			if (idx == -1) {
				idx = i;
				}
			else {
				cout << "set_stabilizer_compute::compute_set_stabilizer error, two entries in frequency are non zero at level " << lvl << endl;
				exit(1);
				}
			}
		}
	f = gen->first_oracle_node_at_level[lvl];
	nd = f + idx;
	if (f_v) {
		cout << "the set is isomorphic to orbit " << idx << " i.e. node = " << nd << endl;
		}

	group G;
	oracle *O;
	longinteger_object go, go1;
	INT *Elt1, *Elt2, *Elt3;
	
	O = &gen->root[nd];
	O->store_set_to(gen, lvl - 1, gen->set0);
	
	if (f_v) {
		cout << "the original set is ";
		INT_vec_print(cout, the_set, lvl);
		cout << endl;
		}
	
	if (f_v) {
		cout << "the canonical set is ";
		INT_vec_print(cout, gen->set0, lvl);
		cout << endl;
		}
	
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Elt3 = NEW_INT(A->elt_size_in_INT);
	A->element_move(gen->transporter->ith(lvl), Elt1, FALSE);
	A->element_invert(Elt1, Elt2, FALSE);
	
	if (f_v) {
		cout << "the transporter is" << endl;
		A->element_print(Elt1, cout);
		cout << endl;
		}
	
	
	G.init(gen->A);
	G.init_strong_generators_by_hdl(O->nb_strong_generators, O->hdl_strong_generators, O->tl, FALSE);
	G.schreier_sims(0);
	G.group_order(go);

	if (f_v) {
		cout << "stabilizer has order " << go << endl;
		}
	
#if 0
	Stab->init(gen->A);
	Stab->init_trivial_group(verbose_level - 1);
	//Stab->group_order(stab_order);
#endif


#if 0
	if (!go.is_one()) {

		Stab->conjugate(gen->A, G.S, Elt1, FALSE, verbose_level - 1);
		Stab->group_order(go1);

#if 0
		G.require_strong_generators();
		cout << "tl: ";
		for (i = 0; i < G.A->base_len; i++)
			cout << G.tl[i] << " ";
		cout << endl;
		cout << G.SG->len << " strong generators by rank: " << endl;
		for (i = 0; i < G.SG->len; i++) {
			cout << i << " : " << endl;
			
			G.A->element_print(G.SG->ith(i), cout);
			cout << endl;

			G.A->element_rank(rk, G.SG->ith(i), 0);
			cout << "\"" << rk << "\", ";
			cout << endl;
			}
		//for (i = 0; i < G.SG->len; i++) {
			//}
		aut_gens.init(A);
		aut_gens.allocate(G.SG->len);
		
		for (i = 0; i < G.SG->len; i++) {
			A->element_mult(elt1, G.SG->ith(i), elt3, FALSE);
			A->element_mult(elt3, elt2, aut_gens.ith(i), FALSE);
			}
		cout << "set stabilizer is generated by" << endl;
		for (i = 0; i < G.SG->len; i++) {
			cout << i << " : " << endl;
			
			G.A->element_print(aut_gens.ith(i), cout);
			cout << endl;

			G.A->element_rank(rk, aut_gens.ith(i), 0);
			cout << "\"" << rk << "\", ";
			cout << endl;
			}
#endif
		}
#endif

	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(Elt3);
}

void set_stabilizer_compute::print_frequencies(INT lvl, INT *frequency, INT nb_orbits)
{
	INT i, f;

	cout << "orbit i : frequency[i] : stab order" << endl;

	f = gen->first_oracle_node_at_level[lvl];

	if (nb_orbits < 50) {
		for (i = 0; i < nb_orbits; i++) {
			oracle *O;
			group G;
			longinteger_object go;

			O = &gen->root[f + i];

			G.init(gen->A);
			G.init_strong_generators_by_hdl(O->nb_strong_generators, O->hdl_strong_generators, O->tl, FALSE);
			G.schreier_sims(0);
			G.group_order(go);
			cout << setw(4) << i << " : " << setw(10) << frequency[i] << setw(20) << go << endl;
			}
		}
	else {
		cout << "Too many to print" << endl;
		}
#if 0
	cout << "i : orbit_idx_of_subset[i]" << endl;
	for (i = 0; i < n_choose_k; i++) {
		cout << setw(4) << i << " : " << setw(10) << orbit_idx_of_subset[i] << endl;
		}
#endif
}

INT set_stabilizer_compute::handle_frequencies(INT lvl, 
	INT *frequency, INT nb_orbits, INT *orbit_idx_of_subset, 
	INT &counter, INT n_choose_k, strong_generators *&Aut_gens, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	classify C;
	INT nb_types;
	INT i, j, selected_type, selected_frequency, first, len, orb_idx;
	INT nb_interesting_orbits;	
		
	if (f_v) {
		cout << "set_stabilizer_compute::handle_frequencies" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		cout << "lvl = " << lvl << endl;
		cout << "nb_orbits = " << nb_orbits << endl;
		}
	C.init(frequency, nb_orbits, FALSE, verbose_level);
	if (f_v) {
		cout << "orbit type: ";
		C.print(FALSE /*f_backwards*/);
		//cout << endl;
		}

	nb_types = C.nb_types;
	if (f_v) {
		cout << "nb_types = " << nb_types << endl;
		}

	nb_interesting_orbits = 0;
	for (i = 0; i < nb_types; i++) {
		if (C.data_sorted[C.type_first[i]]) {
			nb_interesting_orbits += C.type_len[i];
			}
		}

	if (f_v) {
		cout << "nb_interesting_orbits = " << nb_interesting_orbits << endl;
		}

	// Search for the isomorphism type of lvl-sets which is represented 
	// the fewest number of times amongst the lvl-subsets of the given set (but not zero times).
	// If there is more than one isomorphism type with the same frequency, one of them is picked.
	// It does not matter which one is picked.
	for (selected_type = 0; selected_type < nb_types; selected_type++) {
		if (C.data_sorted[C.type_first[selected_type]]) {
			break;
			}
		}
	if (selected_type == nb_types) {
		cout << "selected_type == nb_types, error" << endl;
		exit(1);
		}

	// selected_frequency is how many lvl-subsets are of this isomorphism type:
	selected_frequency = C.data_sorted[C.type_first[selected_type]];
	if (f_v) {
		cout << "selected_frequency = " << selected_frequency << endl;
		}


	if (nb_interesting_orbits > 1 /*lvl >= 1 && counter >= 1*/ /* lvl == 4 */) {


		// interesting_subsets are the lvl-subsets of the given set
		// which are of the chosen type.
		// There is nb_interesting_subsets of them.
		INT *interesting_subsets;
		INT nb_interesting_subsets;

		
		cout << "set_stabilizer_compute::handle_frequencies we decide to go for subsets of size " << lvl << ", selected_frequency = " << selected_frequency << endl;
		first = C.type_first[selected_type];
		len = C.type_len[selected_type];
		orb_idx = C.sorting_perm_inv[first];
		nb_interesting_subsets = C.data_sorted[first];

		if (nb_interesting_subsets != selected_frequency) {
			cout << "nb_interesting_subsets != selected_frequency" << endl;
			exit(1);
			}
			
		cout << "orbit " << orb_idx << " has frequency " << nb_interesting_subsets << " first=" << first << " len=" << len << endl;
		cout << "n_choose_k=" << n_choose_k << endl;
		j = 0;
		interesting_subsets = NEW_INT(nb_interesting_subsets);
		for (i = 0; i < n_choose_k; i++) {
			if (orbit_idx_of_subset[i] == orb_idx) {
				interesting_subsets[j++] = i;
				//cout << "subset of rank " << i << " is isomorphic to orbit " << orb_idx << " j=" << j << endl;
				}
			}
		if (j != nb_interesting_subsets) {
			cout << "j != nb_interesting_subsets" << endl;
			exit(1);
			}
		if (f_vv) {
			print_interesting_subsets(lvl, nb_interesting_subsets, interesting_subsets);
			}
		
		
		overall_backtrack_nodes = 0;
		if (f_v) {
			cout << "set_stabilizer_compute::handle_frequencies calling compute_stabilizer_function" << endl;
			}
		//INT nodes;


		//compute_stabilizer_function(the_set, set_size, A, A, gen, lvl, orb_idx, orb_mult, interesting_subsets, Stab, nodes, verbose_level);

		compute_stabilizer *CS;

		CS = new compute_stabilizer;

		CS->init(the_set, set_size, gen, A, A2, lvl, orb_idx, nb_interesting_subsets, interesting_subsets, verbose_level);

		
		Aut_gens = new strong_generators;

		Aut_gens->init_from_sims(CS->Stab, verbose_level);

		if (f_v) {
			cout << "set_stabilizer_compute::handle_frequencies done with compute_stabilizer" << endl;
			cout << "compute_stabilizer::init backtrack_nodes_first_time = " << CS->backtrack_nodes_first_time << endl;
			cout << "compute_stabilizer::init backtrack_nodes_total_in_loop = " << CS->backtrack_nodes_total_in_loop << endl;
			}


		delete CS;

		//overall_backtrack_nodes += CS->nodes;

		FREE_INT(interesting_subsets);

		cout << "set_stabilizer_compute::handle_frequencies: Stabilizer computation finished.";
		//the_end_quietly(t0);
		//exit(0);
		return TRUE;
		
		}
	return FALSE;
}

void set_stabilizer_compute::print_interesting_subsets(INT lvl, INT nb_interesting_subsets, INT *interesting_subsets)
{
	INT i, j;
	
	cout << "the ranks of the corresponding subsets are:" << endl;
	INT_vec_print(cout, interesting_subsets, nb_interesting_subsets);
	cout << endl;
	INT set[1000];
	cout << "the subsets are:" << endl;

	if (nb_interesting_subsets < 50) {
		for (i = 0; i < nb_interesting_subsets; i++) {
			INT ii;
			j = interesting_subsets[i];
			unrank_k_subset(j, set, set_size, lvl);
			cout << setw(3) << i << " : " << setw(6) << j << " : (";
			for (ii = 0; ii < lvl; ii++) {
				cout << setw(3) << set[ii];
				if (ii < lvl - 1)
					cout << ", ";
				}
			//INT_vec_print(cout, set, lvl);
			cout << ")" << endl;
#if 0
			cout << " : (";
			for (ii = 0; ii < lvl; ii++) {
				cout << setw(6) << the_set[set[ii]];
				if (ii < lvl - 1)
					cout << ", ";
				}
			cout << ") : (";
			for (ii = 0; ii < lvl; ii++) {
				A->print_point(the_set[set[ii]], cout);
				if (ii < lvl - 1)
					cout << ", ";
				}
			cout << ")" << endl;
#endif
			}
		}
	else {
		cout << "Too many to print" << endl;
		}
}

void set_stabilizer_compute::compute_frequencies(INT level, 
	INT *&frequency, INT &nb_orbits, 
	INT *&isomorphism_type_of_subset, INT &n_choose_k, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	//INT f_vvvv = (verbose_level >= 4);
	INT set[1000];
	INT subset[1000];
	INT canonical_subset[1000];
	INT i;
	INT idx, f, local_idx, subset_rk, print_mod;
	INT *Elt1;
	//INT f_implicit_fusion = TRUE;
	

	Elt1 = NEW_INT(A->elt_size_in_INT);

	f = gen->first_oracle_node_at_level[level];
	nb_orbits = gen->nb_orbits_at_level(level);
	n_choose_k = INT_n_choose_k(set_size, level);
	
	if (f_v) {
		cout << "set_stabilizer_compute::compute_frequencies level=" << level << " nb_orbits=" << nb_orbits << " n_choose_k=" << n_choose_k << endl;
		}	
	
	frequency = NEW_INT(nb_orbits);
	isomorphism_type_of_subset = NEW_INT(n_choose_k);
	
	INT_vec_zero(frequency, nb_orbits);

	first_k_subset(set, set_size, level);
	subset_rk = 0;
	if (n_choose_k > 100000) {
		print_mod = 10000;
		}
	else {
		print_mod = 1000;
		}
	while (TRUE) {
		INT f_v2 = FALSE;

		if ((subset_rk % print_mod) == 0) {
			f_v2 = TRUE;
			}
		if (f_v2) {
			cout << "set_stabilizer_compute::compute_frequencies level=" << level 
				<< " testing set " << subset_rk << " / " << n_choose_k 
				<< " = " << 100. * (double) subset_rk / (double) n_choose_k << " % : ";
			INT_vec_print(cout, set, level);
			cout << endl;
			}
		for (i = 0; i < level; i++) {
			subset[i] = the_set[set[i]];
			gen->set[0][i] = subset[i];
			}
		if (FALSE /*f_v2*/) {
			cout << "corresponding to set ";
			INT_vec_print(cout, subset, level);
			cout << endl;
			}
		gen->A->element_one(gen->transporter->ith(0), 0);
		
		if (level == 0) {
			frequency[0] = 1;
			isomorphism_type_of_subset[0] = 0;
			}
		else {

			if (FALSE) {
				cout << "set_stabilizer_compute::compute_frequencies before trace_set" << endl;
				}
			local_idx = gen->trace_set(subset, level, level, 
				canonical_subset, Elt1, 
				//f_implicit_fusion, 
				verbose_level - 3);
			if (FALSE) {
				cout << "set_stabilizer_compute::compute_frequencies after trace_set, local_idx = " << local_idx << endl;
				}
			idx = local_idx + f;
			
			if (FALSE /*f_vvv*/) {
				cout << "local_idx=" << local_idx << endl;
				}
			frequency[local_idx]++;
			isomorphism_type_of_subset[subset_rk] = local_idx;
			if (FALSE) {
				cout << "the transporter is" << endl;
				gen->A->element_print(Elt1, cout);
				cout << endl;
				}

			}
		subset_rk++;
		if (!next_k_subset(set, set_size, level)) {
			break;
			}
		}
	if (subset_rk != n_choose_k) {
		cout << "subset_rk != n_choose_k" << endl;
		exit(1);
		}
	if (f_v) {
		//cout << "i : frequency[i]" << endl;
		cout << "frequency vector of orbits at level " << level << " : ";
		for (i = 0; i < nb_orbits; i++) {
			if (frequency[i] == 0)
				continue;
			cout << i << "^" << frequency[i] << " ";
			}
		cout << endl;
		cout << "n_choose_k = " << n_choose_k << endl;
		}
	if (f_vv) {
		cout << "isomorphism_type_of_subset:";
		INT_vec_print(cout, isomorphism_type_of_subset, MINIMUM(100, n_choose_k));
		cout << endl;
		}
	FREE_INT(Elt1);
}




