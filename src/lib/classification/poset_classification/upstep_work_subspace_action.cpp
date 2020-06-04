// upstep_work_subspace_action.cpp
//
// Anton Betten
// March 10, 2010
// moved here: June 28, 2014


#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

using namespace std;

namespace orbiter {
namespace classification {

int upstep_work::upstep_subspace_action(int verbose_level)
// This routine is called from upstep_work::init_extension_node
// It computes coset_table.
// It is testing a set of size 'size'. 
// The newly added point is in gen->S[size - 1]
// The extension is initiated from node 'prev'
// and from extension 'prev_ex'
// The node 'prev' is at depth 'size' - 1 
// returns FALSE if the set is not canonical
// (provided f_indicate_not_canonicals is TRUE)
{
	//if (prev == 1)  verbose_level += 20;
	
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);
	int f_vvv = (verbose_level >= 6);
	//int f_v4 = (verbose_level >= 7);
	int f_v5 = (verbose_level >= 8);
	schreier up_orbit;
	union_find UF;
	int *aut;
	trace_result r;
	int final_node, final_ex;
	
	//wreath_product *W;
	matrix_group *M;
	finite_field *F;
	{
	grassmann G;
	action_on_grassmannian *AG;
	{
	action A_on_hyperplanes;
	int big_n, n, k, rk, degree, idx;
	int *ambient_space; // [n * big_n]
	int *base_change_matrix; // [n * n]
	int *base_cols; // [n]
	int *embedding; // [n]
	int *changed_space; // [n * big_n]
	
	AG = NEW_OBJECT(action_on_grassmannian);
	

	O_cur->store_set(gen, size - 1);
		// stores a set of size 'size' to gen->S
	if (f_v) {
		print_level_extension_info();
		cout << "upstep_work::upstep_subspace_action "
				"upstep in subspace action for set ";
		lint_set_print(cout, gen->get_S(), size);
		cout << " verbose_level=" << verbose_level;
		cout << " f_indicate_not_canonicals="
				<< f_indicate_not_canonicals << endl;
		//cout << endl;
	}

	if (!gen->get_A2()->f_is_linear) {
		cout << "upstep_work::upstep_subspace_action "
				"action is not linear" << endl;
		exit(1);
	}
#if 0
	if (gen->get_A2()->type_G == matrix_group_t) {
		M = gen->get_A2()->G.matrix_grp;
		F = M->GFq;
	}
	else {
		action *sub = gen->get_A2()->subaction;
		if (sub->type_G == wreath_product_t) {
			W = sub->G.wreath_product_group;
			F = W->F;
		}
		else {
			M = sub->G.matrix_grp;
			F = M->GFq;
		}
	}
#else
	M = gen->get_A2()->get_matrix_group();
	F = M->GFq;
#endif



#if 1
	if (gen->get_A2()->type_G == action_by_subfield_structure_t) {
		F = gen->get_A2()->G.SubfieldStructure->Fq;
			// we need the small field because we work 
			// in the large vector space over the small field.
	}
#endif
	big_n = gen->get_VS()->dimension;
	n = size;
	k = size - 1;
	if (f_vv) {
		cout << "big_n=" << big_n << endl;
		cout << "n=" << n << endl;
		cout << "k=" << k << endl;
	}
	ambient_space = NEW_int(n * big_n);
	base_change_matrix = NEW_int(n * n);
	base_cols = NEW_int(n);
	embedding = NEW_int(n);
	changed_space = NEW_int(n * big_n);
	
	gen->unrank_basis(ambient_space, gen->get_S(), n);

	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action "
				"ambient space:" << endl;

		print_integer_matrix_width(cout, ambient_space,
				n, big_n, big_n, F->log10_of_q);

		cout << "setting up grassmannian n=" << n
				<< " k=" << k << " q=" << F->q << endl;
	}
	G.init(n, k, F, 0 /*verbose_level - 1*/);
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action "
				"grassmann initialized" << endl;
		cout << "upstep_work::upstep_subspace_action "
				"setting up action_on_grassmannian:" << endl;
	}
	AG->init(*gen->get_A2(), &G, verbose_level - 2);
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action "
				"after AG.init" << endl;
	}
	AG->init_embedding(big_n, ambient_space, verbose_level - 8);
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action "
				"after AG.init_embedding, big_n=" << big_n << endl;
	}
		
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action "
				"AG->GE->degree = " << AG->GE->degree << endl;
		cout << "upstep_work::upstep_subspace_action "
				"before induced_action_on_grassmannian" << endl;
	}
	

	A_on_hyperplanes.induced_action_on_grassmannian(
		gen->get_A2(),
		AG, 
		FALSE /* f_induce_action*/,
		NULL /*sims *old_G*/,
		verbose_level - 3);
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action "
				"after A_on_hyperplanes->induced_action_on_grassmannian"
				<< endl;
	}
	degree = A_on_hyperplanes.degree;
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action "
				"The action on hyperplanes has degree = "
				<< degree << endl;
	}
	if (degree != AG->GE->degree) {
		cout << "upstep_work::upstep_subspace_action "
				"degree != AG->GE->degree" << endl;
		exit(1);
	}

	up_orbit.init(&A_on_hyperplanes, verbose_level - 2);
	up_orbit.init_generators(*H->SG, verbose_level - 2);
	if (f_vvv) {
		cout << "upstep_work::upstep_subspace_action "
				"generators for H:" << endl;
		H->print_strong_generators(cout, TRUE);

#if 1
		cout << "generators in the action on hyperplanes:" << endl;
		H->print_strong_generators_with_different_action_verbose(
			cout,
			&A_on_hyperplanes,
			verbose_level - 2);
#endif

	}
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action "
				"computing initial orbits of hyperplane action:"
				<< endl;
		}
	up_orbit.compute_point_orbit(
			0 /* the initial hyperplane */,
			verbose_level);
		// computes the orbits of the group H
		// up_orbit will be extended as soon 
		// as n e w automorphisms are found
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action "
				"computing initial orbits of hyperplane action done"
				<< endl;
	}
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action "
				"the initial orbits on hyperplanes are:" << endl;
		up_orbit.print_and_list_orbits(cout);
	}

	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action "
				"initializing union_find:" << endl;
		}
	UF.init(&A_on_hyperplanes, verbose_level - 8);
	UF.add_generators(H->SG, 0 /*verbose_level - 8 */);
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action "
				"initializing union_find done" << endl;
	}
	if (f_vvv) {
		UF.print();
	}

	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action "
				"we will now loop over the " << degree
				<< " cosets of the hyperplane stabilizer:" << endl;
	}

	coset_table = NEW_OBJECTS(coset_table_entry, degree);
	nb_cosets = degree;
	nb_cosets_processed = 0;

	for (coset = 0; coset < degree; coset++) { 

		if (f_vv) {
			print_level_extension_info();
			cout << " upstep_work::upstep_subspace_action coset "
					<< coset << " / " << degree << endl;
		}
		idx = UF.ancestor(coset);
		if (idx < coset) {
			//gen->nb_times_trace_was_saved++;
			if (f_vv) {
				cout << "upstep_work::upstep_subspace_action coset "
						<< coset << " / " << degree << " is at " << idx
						<< " which has already been done, "
						"so we save one trace" << endl;
			}
			continue;
		}
#if 0
		idx = up_orbit.orbit_inv[coset];
		if (idx < up_orbit.orbit_len[0]) {
			if (f_v) {
				cout << "upstep_work::upstep_subspace_action "
						"coset " << coset << " is at " << idx
						<< " which is part of the current orbit, "
						"so we save one trace" << endl;
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
			cout << "upstep_work::upstep_subspace_action "
					"unranking " << coset << ":" << endl;
		}
		G.unrank_lint(coset, 0 /*verbose_level - 5*/);
		int_vec_copy(G.M, base_change_matrix, k * n);

		if (f_vvv) {
			cout << "upstep_work::upstep_subspace_action "
					"base_change_matrix (hyperplane part) for coset "
					<< coset << ":" << endl;
			print_integer_matrix_width(cout,
					base_change_matrix,
					k, n, n, F->log10_of_q);
		}
		rk = F->base_cols_and_embedding(
				k, n,
				base_change_matrix,
				base_cols,
				embedding,
				0/*verbose_level*/);
		if (rk != k) {
			cout << "rk != k" << endl;
			exit(1);
		}
		if (f_v5) {
			cout << "upstep_work::upstep_subspace_action base_cols:";
			int_vec_print(cout, base_cols, rk);
			cout << " embedding:";
			int_vec_print(cout, embedding, n - rk);
			cout << endl;
		}

		// fill the matrix up and make it invertible:
		int_vec_zero(base_change_matrix + (n - 1) * n, n);
		base_change_matrix[(n - 1) * n + embedding[0]] = 1;

		if (f_v5) {
			cout << "upstep_work::upstep_subspace_action "
					"extended base_change_matrix (hyperplane part) "
					"for coset " << coset << ":" << endl;
			print_integer_matrix_width(cout,
					base_change_matrix,
					n, n, n, F->log10_of_q);
		}
		if (f_v5) {
			cout << "upstep_work::upstep_subspace_action "
					"AG->GE->M:" << endl;
			print_integer_matrix_width(cout,
					AG->GE->M, n, big_n, big_n, F->log10_of_q);
		}


		// now base_change_matrix is invertible
		rk = F->base_cols_and_embedding(n, n,
				base_change_matrix, base_cols, embedding,
				0/*verbose_level*/);
		if (rk != n) {
			cout << "upstep_work::upstep_subspace_action "
					"rk != n" << endl;
			exit(1);
		}
		F->mult_matrix_matrix(
				base_change_matrix,
				AG->GE->M,
				changed_space,
				n, n, big_n,
				0 /* verbose_level */);
		if (f_v5) {
			cout << "upstep_work::upstep_subspace_action "
					"changed_space for coset " << coset << ":" << endl;
			print_integer_matrix_width(cout,
					changed_space,
					n, big_n, big_n, F->log10_of_q);
		}

		// initialize set[0] for the tracing
		// (keep gen->S as it is):
		gen->rank_basis(changed_space, gen->get_set_i(0), n);

		if (f_vvv) {
			cout << "upstep_work::upstep_subspace_action "
					"changed_space for coset " << coset
					<< " as rank vector: ";
			lint_vec_print(cout, gen->get_set_i(0), n);
			cout << endl; 
		}
		
		
		// initialize transporter[0] for the tracing
		gen->get_A()->element_one(gen->get_transporter()->ith(0), 0);


		if (f_vv) {
			print_level_extension_coset_info();
			cout << "upstep_work::upstep_subspace_action exchanged set: ";
			lint_set_print(cout, gen->get_set_i(0), size);
			cout << endl;
			cout << "upstep_work::upstep_subspace_action "
					"calling recognize" << endl;
		}
		
#if 0		
		if (prev == 1 && prev_ex == 1) {
			verbose_level += 20;
			}
#endif
		
		int nb_times_image_of_called0 = gen->get_A()->ptr->nb_times_image_of_called;
		int nb_times_mult_called0 = gen->get_A()->ptr->nb_times_mult_called;
		int nb_times_invert_called0 = gen->get_A()->ptr->nb_times_invert_called;
		int nb_times_retrieve_called0 = gen->get_A()->ptr->nb_times_retrieve_called;

		
		if (f_vv) {
			print_level_extension_info();
			cout << " upstep_work::upstep_subspace_action coset "
					<< coset << " / " << degree
					<< " before recognize " << endl;
		}

		r = recognize(
				final_node, final_ex,
				TRUE /* f_tolerant */,
				verbose_level - 1);
			// upstep_work_trace.cpp
			// gen->set[0] is the set we want to trace
			// gen->transporter->ith(0) is the identity


		if (f_vv) {
			print_level_extension_info();
			cout << " upstep_work::upstep_subspace_action coset "
					<< coset << " / " << degree
					<< " after recognize " << endl;
		}

		
		coset_table[nb_cosets_processed].coset = coset;
		coset_table[nb_cosets_processed].type = r;
		coset_table[nb_cosets_processed].node = final_node;
		coset_table[nb_cosets_processed].ex = final_ex;
		coset_table[nb_cosets_processed].nb_times_image_of_called = 
			gen->get_A()->ptr->nb_times_image_of_called - nb_times_image_of_called0;
		coset_table[nb_cosets_processed].nb_times_mult_called = 
			gen->get_A()->ptr->nb_times_mult_called - nb_times_mult_called0;
		coset_table[nb_cosets_processed].nb_times_invert_called = 
			gen->get_A()->ptr->nb_times_invert_called - nb_times_invert_called0;
		coset_table[nb_cosets_processed].nb_times_retrieve_called = 
			gen->get_A()->ptr->nb_times_retrieve_called - nb_times_retrieve_called0;
		nb_cosets_processed++;

		if (f_v) {
			print_level_extension_coset_info();
			cout << "upstep_work::upstep_subspace_action calling "
					"find_automorphism returns "
					<< trace_result_as_text(r) << endl;
		}
		
		
		if (r == found_automorphism) {
			aut = gen->get_transporter()->ith(size);
			if (f_v) {
				print_level_extension_coset_info();
				cout << "upstep_work::upstep_subspace_action "
						"found automorphism in coset " << coset << endl;
				if (coset > 0 &&
						TRUE /*gen->f_allowed_to_show_group_elements*/
						&& f_v) {
					gen->get_A()->element_print_quick(aut, cout);
					cout << endl;
#if 0
					cout << "in the action " << gen->A2->label
							<< ":" << endl;
					gen->A2->element_print_as_permutation(aut, cout);
#endif
					cout << "in the action "
							<< A_on_hyperplanes.label
							<< " on the hyperplanes:" << endl;
					A_on_hyperplanes.element_print_as_permutation_verbose(
						aut,
						cout, 0/*verbose_level - 5*/);
				}
			}
#if 0
			if (A_on_hyperplanes.element_image_of(coset,
					aut, FALSE) != 0) {
				cout << "upstep_work::upstep_subspace_action fatal: "
						"automorphism does not map " << coset
						<< " to 0 as it should" << endl;
				exit(1);
			}
#endif

			UF.add_generator(aut, 0 /*verbose_level - 5*/);
			up_orbit.extend_orbit(aut, verbose_level - 8);
			if (f_vv) {
				cout << "upstep_work::upstep_subspace_action "
						"n e w orbit length upstep = "
						<< up_orbit.orbit_len[0] << endl;
			}
		}
		else if (r == not_canonical) {
			if (f_indicate_not_canonicals) {
				if (f_vvv) {
					cout << "upstep_work::upstep_subspace_action "
							"not canonical" << endl;
				}
				return FALSE;
			}
			cout << "upstep_work::upstep_subspace_action: "
					"recognize returns not_canonical, "
					"this should not happen" << endl;
			exit(1);
		}
		else if (r == no_result_extension_not_found) {
			if (f_vvv) {
				cout << "upstep_work::upstep_subspace_action "
						"no_result_extension_not_found" << endl;
			}
			cout << "upstep_work::upstep_subspace_action "
					"fatal: no_result_extension_not_found" << endl;
			exit(1);
		}
		else if (r == no_result_fusion_node_installed) {
			if (f_vvv) {
				cout << "upstep_work::upstep_subspace_action "
						"no_result_fusion_node_installed" << endl;
			}
		}
		else if (r == no_result_fusion_node_already_installed) {
			if (f_vvv) {
				cout << "upstep_work::upstep_subspace_action "
						"no_result_fusion_node_already_installed" << endl;
			}
		}
	} // next coset

	
	if (f_v) {
		print_level_extension_info();
		cout << "upstep_work::upstep_subspace_action "
				"upstep orbit length for set ";
		lint_set_print(cout, gen->get_S(), size);
		cout << " is " << up_orbit.orbit_len[0] << endl;
	}

	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action "
				"the final orbits on hyperplanes are:" << endl;
		up_orbit.print_and_list_orbits(cout);
	}



	if (gen->do_group_extension_in_upstep()) {
		vector_ge SG_extension;
		int *tl_extension = NEW_int(gen->get_A()->base_len());
		int f_OK;
		int f_tolerant = FALSE;
	
#if 0
		if (cur == 26) {
			cout << "upstep_work::upstep_subspace_action "
					"node " << cur << ":" << endl;
		}
#endif
		if (f_vv) {
			cout << "upstep_work::upstep_subspace_action "
					"before H->S->transitive_extension_tolerant"
					<< endl;
		}
		f_OK = H->S->transitive_extension_tolerant(
				up_orbit, SG_extension,
				tl_extension,
				f_tolerant,
				0 /*verbose_level - 8*/);
		if (f_vv) {
			cout << "upstep_work::upstep_subspace_action "
					"after H->S->transitive_extension_tolerant"
					<< endl;
		}
		if (!f_OK) {
			cout << "upstep_work::upstep_subspace_action "
					"overshooting the group order" << endl;
		}
		H->delete_strong_generators();
		H->init_strong_generators(SG_extension, tl_extension, verbose_level - 2);


		FREE_int(tl_extension);
	}
	
	FREE_int(ambient_space);
	FREE_int(base_change_matrix);
	FREE_int(base_cols);
	FREE_int(embedding);
	FREE_int(changed_space);
	
	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action "
				"before freeing A_on_hyperplanes" << endl;
	}
	} // end A_on_hyperplanes

	FREE_OBJECT(AG);

	if (f_vv) {
		cout << "upstep_work::upstep_subspace_action "
				"before freeing the rest" << endl;
		}
	}
	if (f_v) {
		cout << "upstep_work::upstep_subspace_action done" << endl;
	}
	return TRUE;
}

}}


