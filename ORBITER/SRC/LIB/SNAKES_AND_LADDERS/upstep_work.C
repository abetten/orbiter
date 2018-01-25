// upstep_work.C
//
// Anton Betten
// March 10, 2010

#include "orbiter.h"

upstep_work::upstep_work()
{
	gen = NULL;
	nb_fusion_nodes = 0;
	nb_fuse_cur = 0;
	nb_ext_cur = 0;
	mod_for_printing = 1;
	G = NULL;
	H = NULL;
	coset_table = NULL;
	path = NULL;

}

upstep_work::~upstep_work()
{
	INT verbose_level = 0;
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "upstep_work::~upstep_work" << endl;
		}
	if (G) {
		if (f_v) {
			cout << "upstep_work::~upstep_work before delete G" << endl;
			}
		delete G;
		G = NULL;
		}
	if (H) {
		if (f_v) {
			cout << "upstep_work::~upstep_work before delete H" << endl;
			}
		delete H;
		H = NULL;
		}
	if (coset_table) {
		if (f_v) {
			cout << "upstep_work::~upstep_work before delete coset_table" << endl;
			}
		delete [] coset_table;
		coset_table = NULL;
		}
	if (path) {
		if (f_v) {
			cout << "upstep_work::~upstep_work before FREE_INT(path)" << endl;
			}
		FREE_INT(path);
		path = NULL;
		}
	if (f_v) {
		cout << "upstep_work::~upstep_work done" << endl;
		}
}

void upstep_work::init(generator *gen, 
	INT size,
	INT prev,
	INT prev_ex,
	INT cur,
	INT f_debug,
	INT f_implicit_fusion,
	INT f_indicate_not_canonicals, 
	FILE *fp, 
	INT verbose_level)
// called from generator::extend_node
{
	//verbose_level = 1;
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "upstep_work::init size=" << size << " prev=" << prev << " prev_ex=" << prev_ex << " cur=" << cur << endl;
		}
	upstep_work::gen = gen;
	upstep_work::size = size;
	upstep_work::prev = prev;
	upstep_work::prev_ex = prev_ex;
	upstep_work::cur = cur;
	upstep_work::f_debug = f_debug;
	upstep_work::f_implicit_fusion = f_implicit_fusion;
	upstep_work::f_indicate_not_canonicals = f_indicate_not_canonicals;
	upstep_work::f = fp;
	if (gen->root[prev].nb_extensions > 25) {
		mod_for_printing = 25;
		}
	if (gen->root[prev].nb_extensions > 50) {
		mod_for_printing = 50;
		}
	if (gen->root[prev].nb_extensions > 100) {
		mod_for_printing = 100;
		}
	if (gen->root[prev].nb_extensions > 500) {
		mod_for_printing = 500;
		}
	O_prev = &gen->root[prev];
	path = NEW_INT(size + 1);
	path[size] = prev;
	for (i = size - 1; i >= 0; i--) {
		path[i] = gen->root[path[i + 1]].prev;
		}
	if (f_v) {
		cout << "upstep_work::init path: ";
		INT_vec_print(cout, path, size + 1);
		cout << endl;
		}
	if (f_v) {
		cout << "upstep_work::init done" << endl;
		}
}

void upstep_work::handle_extension(INT &nb_fuse_cur, INT &nb_ext_cur, INT verbose_level)
// called from generator::extend_node
// Calls handle_extension_fusion_type 
// or handle_extension_unprocessed_type
//
// Handles the extension 'cur_ex' in node 'prev'.
// We are extending a set of size 'size' to a set of size 'size' + 1. 
// Calls oracle::init_extension_node for the new node that is (possibly) created
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT type;

	if (f_v) {
		cout << "upstep_work::handle_extension verbose_level = " << verbose_level << endl;
		cout << "prev=" << prev << " prev_ex=" << prev_ex << endl;
		}
	pt = O_prev->E[prev_ex].pt;
	type = O_prev->E[prev_ex].type;

	if (f_v) {
		gen->print_level_extension_info(size + 1, prev, prev_ex);
		cout << "type ";
		print_extension_type(cout, type);
		cout << endl;
		}

	if (type == EXTENSION_TYPE_FUSION) {
		if (f_v) {
			cout << "upstep_work::handle_extension fusion type" << endl;
			}
		handle_extension_fusion_type(verbose_level - 2);	
		nb_fuse_cur++;
		}
	else if (type == EXTENSION_TYPE_UNPROCESSED) {
		if (f_v) {
			cout << "upstep_work::handle_extension unprocessed type" << endl;
			}
		handle_extension_unprocessed_type(verbose_level - 2);
		nb_ext_cur++;
		}
	else {
		gen->print_level_extension_info(size + 1, prev, prev_ex);
		cout << endl;
		cout << "upstep_work::handle_extension extension not of unprocessed type, error" << endl;
		cout << "type is ";
		print_extension_type(cout, type);
		cout << endl;
		exit(1);
		}
	if (f_v) {
		cout << "upstep_work::handle_extension prev=" << prev << " prev_ex=" << prev_ex << " done" << endl;
		}
}

void upstep_work::handle_extension_fusion_type(INT verbose_level)
// called from upstep_work::handle_extension
// Handles the extension 'cur_ex' in node 'prev'.
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	
	// fusion node, nothing to do
	
	if (f_v) {
		cout << "upstep_work::handle_extension_fusion_type" << endl;
		}		
	if (f_vv) {
		INT *set;
		set = NEW_INT(size + 1);
		O_prev->store_set_to(gen, size - 1, set /*gen->S1*/);
			// store_set_to(k) stores a set of size k+1
			// so here we store the first size points of the set
			// namely the current set.
								
			// next we store the size+1 th point:
		set[size] = pt; //gen->S1[size] = pt;
			// so, we really have a set of size size + 1

		gen->print_level_extension_info(size + 1, prev, prev_ex);
		cout << " point " << pt << " ";
		INT_set_print(cout, set /*gen->S1*/, size + 1);
		cout << " is a fusion node, skipping" << endl;
		FREE_INT(set);
#if 0
		if (f_vvv) {
			if (gen->f_print_function) {
				(*gen->print_function)(cout, size + 1, gen->S1, gen->print_function_data);
				}
			gen->generator_apply_fusion_element_no_transporter(
				size, size + 1, prev, prev_ex, 
				gen->S1, gen->S2, 
				verbose_level - 3);
			cout << "fusion elt: " << endl;
			//A->element_print_quick(Elt1, cout);
			cout << "maps it to: ";
			INT_set_print(cout, gen->S2, size + 1);
			cout << endl;
			if (gen->f_print_function) {
				(*gen->print_function)(cout, size + 1, gen->S2, gen->print_function_data);
				}
			}
#endif
		}
}

void upstep_work::handle_extension_unprocessed_type(INT verbose_level)
// called from upstep_work::handle_extension
// calls init_extension_node
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);

	INT ret, type;
	
	if (f_v) {
		cout << "upstep_work::handle_extension_unprocessed_type" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	type = O_prev->E[prev_ex].type;
		
	if (f_vv) {
		gen->print_level_extension_info(size + 1, prev, prev_ex);
		cout << "with point " << pt << " : " << endl;
		}
	if (type != EXTENSION_TYPE_UNPROCESSED) {
		cout << "extension not of unprocessed type, error" << endl;
		cout << "type is ";
		print_extension_type(cout, type);
		cout << endl;
		exit(1);
		}
				
	// process the node and create a new set orbit at level size + 1:
				
	pt_orbit_len = O_prev->E[prev_ex].orbit_len;

	size++;
		
	if (f_vv) {
		gen->print_level_extension_info(size, prev, prev_ex);
		cout << "with point " << pt << " : before init_extension_node" << endl;
		}
	
	ret = init_extension_node(verbose_level - 3);

	if (f_vv) {
		gen->print_level_extension_info(size, prev, prev_ex);
		cout << "with point " << pt << " : done " << endl;
		cout << "nb_cosets_processed=" << nb_cosets_processed << endl;
		}
	if (f_vvv) {
		cout << "upstep_work::handle_extension_unprocessed_type coset_table:" << endl;
		print_coset_table(coset_table, nb_cosets_processed);
		}
	if (ret) {
		if (f_vv) {
			cout << "init_extension_node returns TRUE" << endl;
			}
		}
	else {
		if (f_vv) {
			cout << "init_extension_node returns FALSE, the set is not canonical" << endl;
		//cout << "u=" << gen->split_case << " @(" << prev << "," << prev_ex << ") not canonical" << endl;
			}
		if (f_vv) {
			cout << "the set is not canonical, we skip it" << endl;
			}
		cout << "setting type of extension to EXTENSION_TYPE_NOT_CANONICAL" << endl;
		O_prev->E[prev_ex].type = EXTENSION_TYPE_NOT_CANONICAL;
		cur--;
		cout << "reducing cur to " << cur << endl;
		}


	cur++;
	size--;
	if (f_vvv) {
		cout << "cur=" << cur << endl;
		}

	if (f_v) {
		gen->print_level_extension_info(size + 1, prev, prev_ex);
		cout << "with point " << pt << " done" << endl;
		cout << "upstep_work::handle_extension_unprocessed_type done" << endl;
		}
}

INT upstep_work::init_extension_node(INT verbose_level)
// Called from upstep_work::handle_extension_unprocessed_type
// Calls upstep_subspace_action or upstep_for_sets, 
// depending on the type of action
// then changes the type of the extension to EXTENSION_TYPE_EXTENSION
//
// Establishes a new node at depth 'size' (i.e., a set of size 'size') as an extension 
// of a previous node (prev) at depth size - 1 
// with respect to a given point (pt).
// This function is to be called for the next free oracle node which will 
// become the descendant of the previous node (prev).
// the extension node corresponds to the point pt. 
// returns FALSE if the set is not canonical (provided f_indicate_not_canonicals is TRUE)
{
	//if (prev == 1) {verbose_level += 40; cout << "node with prev == 1 reached" << endl;}

#if 0
	if (prev == 12 && prev_ex == 3) {
		cout << "upstep_work::init_extension_node we are at node (12,3)" << endl;
		verbose_level += 10;
		}
#endif

#if 0
	if (cur == 26) {
		cout << "upstep_work::init_extension_node Node=26" << endl;
		}
#endif

	//longinteger_domain D;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	


	if (f_v) {
		gen->print_level_extension_info(size, prev, prev_ex);
		cout << "upstep_work::init_extension_node cur=" << cur 
			<< " verbose_level=" << verbose_level << endl;
		}

	if (cur == -1) {
		gen->print_level_extension_info(size, prev, prev_ex);
		cout << "upstep_work::init_extension_node cur=" << cur << endl;
		exit(1);
		}

	O_cur = &gen->root[cur];
		

	O_cur->freeself();
	O_cur->node = cur;
	O_cur->prev = prev;
	O_cur->pt = pt;

	//if (f_v) {cout << "after freeself" << endl;}
	O_cur->store_set(gen, size - 1); // stores a set of size 'size' to gen->S
	
	if (f_v) {
		gen->print_level_extension_info(size, prev, prev_ex);
		cout << "upstep_work::init_extension_node initializing Node " << cur << " ";
		INT_vec_print(cout, gen->S, size);
		cout << " f_indicate_not_canonicals=" << f_indicate_not_canonicals;
		cout << " verbose_level=" << verbose_level;
		cout << endl;
		}

	if (f_vv) {
		cout << "point " << pt << " lies in an orbit of length " << pt_orbit_len << " verbose_level = " << verbose_level << endl;
		}


	if (G) {
		cout << "upstep_work::init_extension_node G is already allocated" << endl;
		exit(1);
		}
	if (H) {
		cout << "upstep_work::init_extension_node H is already allocated" << endl;
		exit(1);
		}
	G = new group;
	H = new group;
	

	if (f_v) {
		gen->print_level_extension_info(size, prev, prev_ex);
		INT_set_print(cout, gen->S, size);
		cout << "upstep_work::init_extension_node before O_cur->init_extension_node_prepare_G" << endl;
		}
	O_cur->init_extension_node_prepare_G(gen, 
		prev, prev_ex, size, *G, go_G, 
		verbose_level - 4);
		// oracle.C
	if (f_v) {
		gen->print_level_extension_info(size, prev, prev_ex);
		INT_set_print(cout, gen->S, size);
		cout << "upstep_work::init_extension_node after O_cur->init_extension_node_prepare_G" << endl;
		}

	

	if (f_vv) {
		gen->print_level_extension_info(size, prev, prev_ex);
		INT_set_print(cout, gen->S, size);
		cout << endl;
		}
	if (f_vvv) {
		if (gen->f_print_function) {
			(*gen->print_function)(size, gen->S, gen->print_function_data);
			}
		}
	if (f_vv) {
		cout << "(orbit length = " << pt_orbit_len << ")" << endl;
		}
	
	O_prev->E[prev_ex].type = EXTENSION_TYPE_PROCESSING; // currently processing
	O_prev->E[prev_ex].data = cur;
	

	//group H;


	if (f_v) {
		gen->print_level_extension_info(size, prev, prev_ex);
		INT_set_print(cout, gen->S, size);
		cout << "upstep_work::init_extension_node before O_cur->init_extension_node_prepare_H" << endl;
		}
	
	O_cur->init_extension_node_prepare_H(gen, 
		prev, prev_ex, size, 
		*G, go_G,
		*H, go_H, 
		pt, pt_orbit_len, 
		verbose_level - 4);
		// in oracle.C


#if 0
	if (cur == 26) {
		cout << "upstep_work::init_extension_node Node=26" << endl;
		cout << "go_G=" << go_G << endl;
		cout << "go_H=" << go_H << endl;
		}
#endif
	
	if (f_v) {
		gen->print_level_extension_info(size, prev, prev_ex);
		INT_set_print(cout, gen->S, size);
		cout << "upstep_work::init_extension_node after O_cur->init_extension_node_prepare_H" << endl;
		}

	
	if (f_v) {
		gen->print_level_extension_info(size, prev, prev_ex);
		INT_vec_print(cout, gen->S, size);
		cout << "upstep_work::init_extension_node calling upstep" << endl;
		}

	if (gen->f_on_subspaces) {
		if (f_v) {
			gen->print_level_extension_info(size, prev, prev_ex);
			INT_vec_print(cout, gen->S, size);
			cout << "upstep_work::init_extension_node calling upstep_subspace_action" << endl;
			}
		if (!upstep_subspace_action(verbose_level - 2)) {

			if (f_indicate_not_canonicals) {
				if (f_vv) {
					cout << "the set is not canonical" << endl;
					}
				return FALSE;
				}
			cout << "upstep_subspace_action returns FALSE, the set is not canonical, this should not happen" << endl;
			exit(1);
			}
		if (f_v) {
			gen->print_level_extension_info(size, prev, prev_ex);
			INT_vec_print(cout, gen->S, size);
			cout << "upstep_work::init_extension_node after upstep_subspace_action" << endl;
			}
		}
	else {
		if (f_v) {
			gen->print_level_extension_info(size, prev, prev_ex);
			INT_vec_print(cout, gen->S, size);
			cout << "upstep_work::init_extension_node calling upstep_for_sets, verbose_level = " << verbose_level - 2 << endl;
			}
		if (!upstep_for_sets(verbose_level - 2)) {
			if (f_indicate_not_canonicals) {
				if (f_vv) {
					cout << "the set is not canonical" << endl;
					}
				return FALSE;
				}
			cout << "upstep returns FALSE, the set is not canonical, this should not happen" << endl;
			exit(1);
			}
		if (f_v) {
			gen->print_level_extension_info(size, prev, prev_ex);
			INT_vec_print(cout, gen->S, size);
			cout << "upstep_work::init_extension_node after upstep_for_sets" << endl;
			}
		}
	if (f_vv) {
		gen->print_level_extension_info(size, prev, prev_ex);
		INT_vec_print(cout, gen->S, size);
		cout << "extension with point " << pt << " : " << endl;
		cout << "after upstep_for_sets/upstep_subspace_action" << endl;
		//print_coset_table(coset_table, nb_cosets_processed);
		}

	gen->change_extension_type(size - 1, prev, prev_ex, EXTENSION_TYPE_EXTENSION, 0/* verbose_level*/);


	if (f_vv) {
		gen->print_level_extension_info(size, prev, prev_ex);
		INT_vec_print(cout, gen->S, size);
		cout << "_{";
		H->print_group_order(cout);
		cout << "}" << endl;
		}

	strong_generators *Strong_gens;

	Strong_gens = new strong_generators;
	Strong_gens->init_from_sims(H->S, 0);

#if 0
	if (cur == 26) {
		longinteger_object go;
		cout << "upstep_work::init_extension_node Node=26 before upstep_subspace_action group order=";
		Strong_gens->group_order(go);
		cout << go;
		cout << endl;
		cout << "generators are:" << endl;
		Strong_gens->print_generators();
		cout << "tl:";
		INT_vec_print(cout, Strong_gens->tl, gen->A->base_len);
		cout << endl;
		}
#endif

	O_cur->store_strong_generators(gen, Strong_gens);
	delete Strong_gens;
	

	if (f_v) {
		longinteger_object go;
		
		gen->stabilizer_order(cur, go);
		gen->print_level_extension_info(size, prev, prev_ex);
		INT_vec_print(cout, gen->S, size);
		cout << "_{";
		cout << go;
		cout << "} (double check)" << endl;
		cout << "upstep_work::init_extension_node done" << endl;
		}
	return TRUE;
}

INT upstep_work::upstep_for_sets(INT verbose_level)
// This routine is called from upstep_work::init_extension_node
// It is testing a set of size 'size'. The newly added point is in gen->S[size - 1]
// returns FALSE if the set is not canonical (provided f_indicate_not_canonicals is TRUE)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT f_v4 = (verbose_level >= 4);
	INT f_v5 = (verbose_level >= 5);
	schreier up_orbit;
	INT h, possible_image;
	INT *aut, idx;
	trace_result r;
	action A_by_restriction;
	INT final_node, final_ex;
	union_find UF;
	
	O_cur->store_set(gen, size - 1); // stores a set of size 'size'
	if (f_v) {
		gen->print_level_extension_info(size, prev, prev_ex);
		cout << "upstep for set ";
		INT_set_print(cout, gen->S, size);
		cout << " verbose_level=" << verbose_level;
		cout << " f_indicate_not_canonicals=" << f_indicate_not_canonicals << endl;
		//cout << endl;
		}
	A_by_restriction.induced_action_by_restriction(*gen->A2, 
		FALSE /* f_induce_action */, NULL /*sims *old_G */, 
		size, gen->S, 0 /*verbose_level - 2*/);
	
	// the newly added point:
	if (gen->S[size - 1] != pt) {
		cout << "upstep_work::upstep_for_sets fatal: gen->S[size - 1] != pt" << endl;
		exit(1);
		}

	if (f_v) {
		print_level_extension_info();
		cout << "initializing up_orbit with restricted action ";
		A_by_restriction.print_info();
		}	
	up_orbit.init(&A_by_restriction);
	//up_orbit.init(gen->A2);
	if (f_v) {
		print_level_extension_info();
		cout << "initializing up_orbit with generators" << endl;
		}	
	up_orbit.init_generators(*H->SG);



	if (f_v) {
		print_level_extension_info();
		cout << "computing orbit of point " << pt << endl;
		}	
	up_orbit.compute_point_orbit(size - 1 /*pt*/, 0);
		// the orbits of the group H
		// up_orbit will be extended as soon 
		// as new automorphisms are found



	if (f_vv) {
		cout << "upstep_work::upstep_for_sets initializing union_find:" << endl;		
		}
	UF.init(&A_by_restriction, 0 /*verbose_level - 8*/);
	if (f_vv) {
		cout << "upstep_work::upstep_for_sets adding generators to union_find:" << endl;		
		}
	UF.add_generators(H->SG, 0 /*verbose_level - 8*/);
	if (f_vv) {
		cout << "upstep_work::upstep_for_sets initializing union_find done" << endl;	
		}
	if (f_vvv) {
		UF.print();
		}


	if (coset_table) {
		cout << "upstep_work::upstep_for_sets coset_table is allocated" << endl;
		exit(1);
		}
	nb_cosets = size;
	nb_cosets_processed = 0;
	coset_table = new coset_table_entry[nb_cosets];
	
	for (coset = 0; coset < size - 1; coset++) { 
		if (f_v) {
			cout << "upstep_work::upstep_for_sets coset=" << coset << " / " << nb_cosets << endl;
			}
		// for all the previous (=old) points
		possible_image = gen->S[coset];
		if (f_vv) {
			print_level_extension_coset_info();
			cout << " we are trying to map " << possible_image << " to " << pt << endl;
			}	


		idx = UF.ancestor(coset);
		if (idx < coset) {
			gen->nb_times_trace_was_saved++;
			if (f_vv) {
				cout << "coset " << coset << " / " << nb_cosets << " is at " << idx << " which has already been done, so we save one trace" << endl;
				}
			continue;
			}




		if (f_v4) {
			print_level_extension_coset_info();
			cout << " orbit length upstep so far: " << up_orbit.orbit_len[0] 
				<< " checking possible image " << possible_image << endl;
			}


		// initialize set[0] and transporter[0] for the tracing
		for (h = 0; h < size; h++) {
			gen->set[0][h] = gen->S[h];
			}
		gen->set[0][coset] = pt;
		gen->set[0][size - 1] = possible_image;
		gen->A->element_one(gen->transporter->ith(0), 0);


		if (f_v4) {
			print_level_extension_coset_info();
			cout << "exchanged set: ";
			INT_set_print(cout, gen->set[0], size);
			cout << endl;
			cout << "calling find_automorphism()" << endl;
			}
		
		INT nb_times_image_of_called0 = gen->A->nb_times_image_of_called;
		INT nb_times_mult_called0 = gen->A->nb_times_mult_called;
		INT nb_times_invert_called0 = gen->A->nb_times_invert_called;
		INT nb_times_retrieve_called0 = gen->A->nb_times_retrieve_called;
		
		r = find_automorphism_by_tracing(final_node, 
				final_ex, 
				TRUE /*f_tolerant*/, 
				verbose_level - 4);
				// in upstep_work_trace.C

		if (f_v) {
			cout << "upstep_work::upstep_for_sets find_automorphism_by_tracing returns " << trace_result_as_text(r) << endl;
			}
		


		coset_table[nb_cosets_processed].coset = coset;
		coset_table[nb_cosets_processed].type = r;
		coset_table[nb_cosets_processed].node = final_node;
		coset_table[nb_cosets_processed].ex = final_ex;
		coset_table[nb_cosets_processed].nb_times_image_of_called = 
			gen->A->nb_times_image_of_called - nb_times_image_of_called0;
		coset_table[nb_cosets_processed].nb_times_mult_called = 
			gen->A->nb_times_mult_called - nb_times_mult_called0;
		coset_table[nb_cosets_processed].nb_times_invert_called = 
			gen->A->nb_times_invert_called - nb_times_invert_called0;
		coset_table[nb_cosets_processed].nb_times_retrieve_called = 
			gen->A->nb_times_retrieve_called - nb_times_retrieve_called0;
		nb_cosets_processed++;

		if (f_vvv) {
			print_level_extension_coset_info();
			cout << "upstep_work::upstep_for_sets calling find_automorphism returns " << trace_result_as_text(r) << endl;
			}
		
		
		if (r == found_automorphism) {
			aut = gen->transporter->ith(size);
			if (f_vvv) {
				print_level_extension_coset_info();
				cout << "upstep_work::upstep_for_sets found automorphism mapping " << possible_image << " to " << pt << endl;
				//gen->A->element_print_as_permutation(aut, cout);
				if (gen->f_allowed_to_show_group_elements && f_v5) {
					gen->A->element_print_quick(aut, cout);
					cout << endl;
					}
				}
			if (gen->A2->element_image_of(possible_image, aut, 0) != pt) {
				cout << "upstep_work::upstep_for_sets image of possible_image is not pt" << endl;
				exit(1);
				}
			UF.add_generator(aut, 0 /*verbose_level - 5*/);
			up_orbit.extend_orbit(aut, verbose_level - 5);
			if (f_vvv) {
				cout << "upstep_work::upstep_for_sets new orbit length upstep = " << up_orbit.orbit_len[0] << endl;
				}
			}
		else if (r == not_canonical) {
			if (f_indicate_not_canonicals) {
				if (f_vvv) {
					cout << "upstep_work::upstep_for_sets not canonical" << endl;
					}
				return FALSE;
				}
#if 0
			print_level_extension_coset_info();
			cout << "upstep_work::upstep_for_sets fatal: find_automorphism_by_tracing returns not_canonical, this should not happen" << endl;
			exit(1);
#endif
			}
		else if (r == no_result_extension_not_found) {
			if (f_vvv) {
				cout << "upstep_work::upstep_for_sets no_result_extension_not_found" << endl;
				}
			}
		else if (r == no_result_fusion_node_installed) {
			if (f_vvv) {
				cout << "upstep_work::upstep_for_sets no_result_fusion_node_installed" << endl;
				}
			}
		else if (r == no_result_fusion_node_already_installed) {
			if (f_vvv) {
				cout << "upstep_work::upstep_for_sets no_result_fusion_node_already_installed" << endl;
				}
			}
		} // next j
	if (f_v) {
		print_level_extension_info();
		cout << "upstep_work::upstep_for_sets upstep orbit length for set ";
		INT_set_print(cout, gen->S, size);
		cout << " is " << up_orbit.orbit_len[0] << endl;
		}
	vector_ge SG_extension;
	INT *tl_extension = NEW_INT(gen->A->base_len);
	INT f_tolerant = TRUE;
	
	H->S->transitive_extension_tolerant(up_orbit, SG_extension, tl_extension, f_tolerant, verbose_level - 3);
	H->delete_strong_generators();
	H->init_strong_generators(SG_extension, tl_extension);
	
	FREE_INT(tl_extension);
	
	if (f_v) {
		print_level_extension_info();
		cout << "upstep_work::upstep_for_sets done nb_cosets_processed = " << nb_cosets_processed << endl;
		}
	return TRUE;
}




void upstep_work::print_level_extension_info()
{
	gen->print_level_extension_info(size, prev, prev_ex);
}

void upstep_work::print_level_extension_coset_info()
{
	gen->print_level_extension_coset_info(size, prev, prev_ex, coset, nb_cosets);
}

void print_coset_table(coset_table_entry *coset_table, INT len)
{
	INT i;
	
	cout << "coset table" << endl;
	cout << "i : coset : node : ex : nb_times_mult : nb_times_invert : nb_times_retrieve : trace_result" << endl;
	for (i = 0; i < len; i++) {
		cout << setw(3) << i << " : " 
			<< setw(5) << coset_table[i].coset << " : " 
			<< setw(5) << coset_table[i].node << " : " 
			<< setw(5) << coset_table[i].ex << " : (" 
			<< coset_table[i].nb_times_mult_called << "/" 
			<< coset_table[i].nb_times_invert_called << "/" 
			<< coset_table[i].nb_times_retrieve_called << ") : " 
			<< setw(5) << trace_result_as_text((trace_result) coset_table[i].type) << endl; 
		}
}



