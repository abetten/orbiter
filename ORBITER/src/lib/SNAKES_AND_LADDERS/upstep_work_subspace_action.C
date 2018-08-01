// upstep_work_subspace_action.C
//
// Anton Betten
// March 10, 2010
// moved here: June 28, 2014


#include "orbiter.h"

INT upstep_work::upstep_subspace_action(INT verbose_level)
// This routine is called from upstep_work::init_extension_node
// It computes coset_table.
// It is testing a set of size 'size'. 
// The newly added point is in gen->S[size - 1]
// The extension is initiated from node 'prev' and from extension 'prev_ex' 
// The node 'prev' is at depth 'size' - 1 
// returns FALSE if the set is not canonical (provided f_indicate_not_canonicals is TRUE)
{
	//if (prev == 1)  verbose_level += 20;
	
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 5);
	INT f_vvv = (verbose_level >= 6);
	//INT f_v4 = (verbose_level >= 7);
	INT f_v5 = (verbose_level >= 8);
	schreier up_orbit;
	union_find UF;
	INT *aut;
	trace_result r;
	INT final_node, final_ex;
	
	matrix_group *M;
	finite_field *F;
	{
	grassmann G;
	action_on_grassmannian *AG;
	{
	action A_on_hyperplanes;
	INT big_n, n, k, rk, degree, i, h, idx;
	INT *ambient_space;
	INT *base_change_matrix;
	INT *base_cols;
	INT *embedding;
	INT *changed_space;
	
	AG = new action_on_grassmannian;
	
	O_cur->store_set(gen, size - 1); // stores a set of size 'size' to gen->S
	if (f_v) {
		print_level_extension_info();
		cout << "upstep_work::upstep_subspace_action upstep in subspace action for set ";
		INT_set_print(cout, gen->S, size);
		cout << " verbose_level=" << verbose_level;
		cout << " f_indicate_not_canonicals=" << f_indicate_not_canonicals << endl;
		//cout << endl;
		}

	if (gen->A2->type_G != matrix_group_t && 
		gen->A2->type_G != action_on_wedge_product_t && 
		gen->A2->type_G != action_by_subfield_structure_t && 
		gen->A2->type_G != action_by_representation_t && 
		gen->A2->type_G != action_on_spread_set_t) {
		cout << "upstep_work::upstep_subspace_action action is not linear" << endl;
		exit(1);
		}
	if (gen->A2->type_G == matrix_group_t) {
		M = gen->A2->G.matrix_grp;
		}
	else {
		action *sub = gen->A2->subaction;
		M = sub->G.matrix_grp;
		}
	F = M->GFq;
#if 1
	if (gen->A2->type_G == action_by_subfield_structure_t) {
		F = gen->A2->G.SubfieldStructure->Fq;
			// we need the small field because we work 
			// in the large vector space over the small field.
		}
#endif
	big_n = gen->vector_space_dimension;
	n = size;
	k = size - 1;
	if (f_vv) {
		cout << "big_n=" << big_n << endl;
		cout << "n=" << n << endl;
		cout << "k=" << k << endl;
		}
	ambient_space = NEW_INT(n * big_n);
	base_change_matrix = NEW_INT(n * n);
	base_cols = NEW_INT(n);
	embedding = NEW_INT(n);
	changed_space = NEW_INT(n * big_n);
	
	for (i = 0; i < n; i++) {
		gen->unrank_point(ambient_space + i * big_n, gen->S[i]);
		}
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action ambient space:" << endl;
		print_integer_matrix_width(cout, ambient_space, n, big_n, big_n, F->log10_of_q);
		cout << "setting up grassmannian n=" << n << " k=" << k << " q=" << F->q << endl;
		}
	G.init(n, k, F, 0/*verbose_level - 10*/);
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action grassmann initialized" << endl;
		}
	AG->init(*gen->A2, &G, 0/*verbose_level - 2*/);
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action after AG.init" << endl;
		}
	AG->init_embedding(big_n, ambient_space, verbose_level - 8);
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action after AG.init_embedding, big_n=" << big_n << endl;
		}
		
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action AG->GE->degree = " << AG->GE->degree << endl;
		cout << "upstep_work::upstep_subspace_action before induced_action_on_grassmannian" << endl;
		}
	

	A_on_hyperplanes.induced_action_on_grassmannian(gen->A2, 
		AG, 
		FALSE /* f_induce_action*/, NULL /*sims *old_G*/, 
		0/*verbose_level - 3*/);
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action after A_on_hyperplanes->induced_action_on_grassmannian" << endl;
		}
	degree = A_on_hyperplanes.degree;
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action The action on hyperplanes has degree = " << degree << endl;
		}
	if (degree != AG->GE->degree) {
		cout << "upstep_work::upstep_subspace_action degree != AG->GE->degree" << endl;
		exit(1);
		}

	up_orbit.init(&A_on_hyperplanes);
	up_orbit.init_generators(*H->SG);
	if (f_vvv) {
		cout << "generators:" << endl;
		H->print_strong_generators(cout, TRUE);

#if 1
		cout << "generators in the action on hyperplanes:" << endl;
		H->print_strong_generators_with_different_action_verbose(cout, 
			&A_on_hyperplanes, verbose_level - 2);
#endif

		}
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action computing initial orbits of hyperplane action:" << endl;
		}
	up_orbit.compute_point_orbit(0 /* the initial hyperplane */, 0/*verbose_level*/);
		// computes the orbits of the group H
		// up_orbit will be extended as soon 
		// as new automorphisms are found
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action computing initial orbits of hyperplane action done" << endl;
		}
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action the initial orbits on hyperplanes are:" << endl;		
		up_orbit.print_and_list_orbits(cout);
		}

	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action initializing union_find:" << endl;		
		}
	UF.init(&A_on_hyperplanes, verbose_level - 8);
	UF.add_generators(H->SG, 0 /*verbose_level - 8 */);
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action initializing union_find done" << endl;	
		}
	if (f_vvv) {
		UF.print();
		}

	if (f_vv) {
		cout << "we will now loop over the " << degree << " cosets of the hyperplane stabilizer:" << endl;		
		}

	coset_table = new coset_table_entry[degree];
	nb_cosets = degree;
	nb_cosets_processed = 0;

	for (coset = 0; coset < degree; coset++) { 

		idx = UF.ancestor(coset);
		if (idx < coset) {
			gen->nb_times_trace_was_saved++;
			if (f_vv) {
				cout << "coset " << coset << " is at " << idx << " which has already been done, so we save one trace" << endl;
				}
			continue;
			}
#if 0
		idx = up_orbit.orbit_inv[coset];
		if (idx < up_orbit.orbit_len[0]) {
			if (f_v) {
				cout << "coset " << coset << " is at " << idx << " which is part of the current orbit, so we save one trace" << endl;
				}
			continue;
			}
#endif

		// for all the previous (=old) points
		if (f_vv) {
			print_level_extension_coset_info();
			cout << endl;
			}
		if (f_vvv) {
			cout << "unranking " << coset << ":" << endl;
			}
		G.unrank_INT(coset, verbose_level - 5);
		for (h = 0; h < k * n; h++) {
			base_change_matrix[h] = G.M[h];
			}
		if (f_vvv) {
			cout << "upstep_work::upstep_subspace_action base_change_matrix (hyperplane part) for coset " << coset << ":" << endl;
			print_integer_matrix_width(cout, base_change_matrix, k, n, n, F->log10_of_q);
			}
		rk = F->base_cols_and_embedding(k, n, base_change_matrix, base_cols, embedding, 0/*verbose_level*/);
		if (rk != k) {
			cout << "rk != k" << endl;
			exit(1);
			}
		if (f_v5) {
			cout << "upstep_work::upstep_subspace_action base_cols:";
			INT_vec_print(cout, base_cols, rk);
			cout << " embedding:";
			INT_vec_print(cout, embedding, n - rk);
			cout << endl;
			}
		for (h = 0; h < n; h++) {
			base_change_matrix[(n - 1) * n + h] = 0;
			}
		base_change_matrix[(n - 1) * n + embedding[0]] = 1;
		if (f_v5) {
			cout << "upstep_work::upstep_subspace_action extended base_change_matrix (hyperplane part) for coset " << coset << ":" << endl;
			print_integer_matrix_width(cout, base_change_matrix, n, n, n, F->log10_of_q);
			}
		if (f_v5) {
			cout << "upstep_work::upstep_subspace_action AG->GE->M:" << endl;
			print_integer_matrix_width(cout, AG->GE->M, n, big_n, big_n, F->log10_of_q);
			}


		// now base_change_matrix is invertible
		rk = F->base_cols_and_embedding(n, n, base_change_matrix, base_cols, embedding, 0/*verbose_level*/);
		if (rk != n) {
			cout << "upstep_work::upstep_subspace_action rk != n" << endl;
			exit(1);
			}
		F->mult_matrix_matrix(base_change_matrix, AG->GE->M, changed_space, n, n, big_n);
		if (f_v5) {
			cout << "upstep_work::upstep_subspace_action changed_space for coset " << coset << ":" << endl;
			print_integer_matrix_width(cout, changed_space, n, big_n, big_n, F->log10_of_q);
			}

		// initialize set[0] for the tracing (keep gen->S as it is):
		for (h = 0; h < n; h++) {
			gen->set[0][h] = gen->rank_point(changed_space + h * big_n);
#if 0
			gen->set[0][h] = (*gen->rank_point_func)(changed_space + h * big_n, 
				gen->rank_point_data);
#endif			
			}
		if (f_vvv) {
			cout << "upstep_work::upstep_subspace_action changed_space for coset " << coset << " as rank vector: ";
			INT_vec_print(cout, gen->set[0], n);
			cout << endl; 
			}
		
		
		// initialize transporter[0] for the tracing
		gen->A->element_one(gen->transporter->ith(0), 0);


		if (f_vv) {
			print_level_extension_coset_info();
			cout << "upstep_work::upstep_subspace_action exchanged set: ";
			INT_set_print(cout, gen->set[0], size);
			cout << endl;
			cout << "calling find_automorphism_by_tracing()" << endl;
			}
		
#if 0		
		if (prev == 1 && prev_ex == 1) {
			verbose_level += 20;
			}
#endif
		
		INT nb_times_image_of_called0 = gen->A->nb_times_image_of_called;
		INT nb_times_mult_called0 = gen->A->nb_times_mult_called;
		INT nb_times_invert_called0 = gen->A->nb_times_invert_called;
		INT nb_times_retrieve_called0 = gen->A->nb_times_retrieve_called;
		

		r = find_automorphism_by_tracing(final_node, final_ex, 
				TRUE /* f_tolerant */, verbose_level - 1);
			// upstep_work_trace.C
		
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

		if (f_v) {
			print_level_extension_coset_info();
			cout << "upstep_work::upstep_subspace_action calling find_automorphism returns " << trace_result_as_text(r) << endl;
			}
		
		
		if (r == found_automorphism) {
			aut = gen->transporter->ith(size);
			if (f_v) {
				print_level_extension_coset_info();
				cout << "upstep_work::upstep_subspace_action found automorphism in coset " << coset << endl;
				if (coset > 0 && TRUE /*gen->f_allowed_to_show_group_elements*/ && f_v) {
					gen->A->element_print_quick(aut, cout);
					cout << endl;
#if 0
					cout << "in the action " << gen->A2->label << ":" << endl;
					gen->A2->element_print_as_permutation(aut, cout);
#endif
					cout << "in the action " << A_on_hyperplanes.label << " on the hyperplanes:" << endl;
					A_on_hyperplanes.element_print_as_permutation_verbose(aut, 
						cout, 0/*verbose_level - 5*/);
					}
				}
#if 0
			if (A_on_hyperplanes.element_image_of(coset, aut, FALSE) != 0) {
				cout << "oracle::upstep_subspace_action fatal: automorphism does not map " << coset << " to 0 as it should" << endl;
				exit(1);
				}
#endif

			UF.add_generator(aut, 0 /*verbose_level - 5*/);
			up_orbit.extend_orbit(aut, verbose_level - 8);
			if (f_vv) {
				cout << "upstep_work::upstep_subspace_action new orbit length upstep = " << up_orbit.orbit_len[0] << endl;
				}
			}
		else if (r == not_canonical) {
			if (f_indicate_not_canonicals) {
				if (f_vvv) {
					cout << "not canonical" << endl;
					}
				return FALSE;
				}
			cout << "upstep_work::upstep_subspace_action: find_automorphism_by_tracing returns not_canonical, this should not happen" << endl;
			exit(1);
			}
		else if (r == no_result_extension_not_found) {
			if (f_vvv) {
				cout << "no_result_extension_not_found" << endl;
				}
			cout << "upstep_work::upstep_subspace_action fatal: no_result_extension_not_found" << endl;
			exit(1);
			}
		else if (r == no_result_fusion_node_installed) {
			if (f_vvv) {
				cout << "no_result_fusion_node_installed" << endl;
				}
			}
		else if (r == no_result_fusion_node_already_installed) {
			if (f_vvv) {
				cout << "no_result_fusion_node_already_installed" << endl;
				}
			}
		} // next coset

	
	if (f_v) {
		print_level_extension_info();
		cout << "upstep orbit length for set ";
		INT_set_print(cout, gen->S, size);
		cout << " is " << up_orbit.orbit_len[0] << endl;
		}

	if (f_vv) {
		cout << "the final orbits on hyperplanes are:" << endl;		
		up_orbit.print_and_list_orbits(cout);
		}



	if (gen->f_do_group_extension_in_upstep) {
		vector_ge SG_extension;
		INT *tl_extension = NEW_INT(gen->A->base_len);
		INT f_OK;
		INT f_tolerant = FALSE;
	
#if 0
		if (cur == 26) {
			cout << "upstep_work::upstep_subspace_action node " << cur << ":" << endl;
			}
#endif
		f_OK = H->S->transitive_extension_tolerant(up_orbit, SG_extension, tl_extension, 
			f_tolerant, verbose_level - 8);
		if (!f_OK) {
			cout << "upstep_work::upstep_subspace_action overshooting the group order" << endl;
			}
		H->delete_strong_generators();
		H->init_strong_generators(SG_extension, tl_extension);


		FREE_INT(tl_extension);
		}
	
	FREE_INT(ambient_space);
	FREE_INT(base_change_matrix);
	FREE_INT(base_cols);
	FREE_INT(embedding);
	FREE_INT(changed_space);
	
	if (f_vv) {
		cout << "before freeing A_on_hyperplanes" << endl;
		}
	}

	if (f_vv) {
		cout << "before freeing the rest" << endl;
		}
	}
	return TRUE;
}


