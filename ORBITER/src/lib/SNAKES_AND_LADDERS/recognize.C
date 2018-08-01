// recognize.C
//
// Anton Betten
//
// started July 19, 2014

#include "orbiter.h"

void recognize_start_over(
	generator *gen, 
	INT size, INT f_implicit_fusion, 
	INT lvl, INT current_node, 
	INT &final_node, INT verbose_level)
// Called from oracle::find_automorphism_by_tracing_recursion
// when trace_next_point returns FALSE
// This can happen only if f_implicit_fusion is TRUE
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "recognize_start_over" << endl;
		}
	// this is needed if implicit fusion nodes are used:
	if (lvl == size - 1) {
		if (f_v) {
			cout << "recognize_start_over lvl == size - 1" << endl;
			}
		final_node = current_node;
		exit(1);
		}


	INT_vec_heapsort(gen->set[lvl + 1], size /* - 1 */);
		// we don't keep the last point (i.e., the (len + 1)-th) extra
	INT_vec_copy(gen->set[lvl + 1], gen->set[0], size);
	//INT_vec_copy(size, gen->set[lvl + 1], gen->set[0]);
	if (f_vv) {
		INT_set_print(cout, gen->set[0], size);
		cout << endl;
		}
	gen->A->element_move(gen->transporter->ith(lvl + 1), 
		gen->transporter->ith(0), FALSE);
	if (f_v) {
		cout << "recognize_start_over before recognize_recursion" << endl;
		}
	recognize_recursion(gen, size, f_implicit_fusion, 
		0, 0, final_node, verbose_level);
	if (f_v) {
		cout << "recognize_start_over after recognize_recursion" << endl;
		}
	if (f_v) {
		cout << "recognize_start_over done" << endl;
		}
}

void recognize_recursion(
	generator *gen, 
	INT size, INT f_implicit_fusion, 
	INT lvl, INT current_node, INT &final_node, 
	INT verbose_level)
// this routine is called by upstep_work::find_automorphism_by_tracing
// we are dealing with a set of size size.
// the tracing starts at lvl = 0 with current_node = 0
{
	//if (my_node == 9 && my_extension == 4) {verbose_level += 10;}
	INT pt0, current_extension;
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//INT f_v10 = (verbose_level >= 10);
	INT f_vvv = (verbose_level >= 3);
	INT f_v4 = (verbose_level >= 3);
	INT f_v5 = (verbose_level >= 3);
	INT f_failure_to_find_point;
	INT node;


	node = current_node - gen->first_oracle_node_at_level[lvl];
	

	if (f_v) {
		cout << "recognize_recursion at ";
		cout << "(" << lvl << "/" << node  << ")" << endl;
		}


	if (lvl == size) {
		if (f_v) {
			cout << "recognize_recursion at ";
			cout << "(" << lvl << "/" << node  << ") lvl == size, terminating" << endl;
			}
		final_node = current_node;
		return;
#if 0
		return handle_last_level(
			lvl, current_node, current_extension, pt0, 
			final_node, 
			verbose_level);
#endif
		}

	oracle *O;

	O = &gen->root[current_node];
	if (f_vvv) {
		cout << "recognize_recursion"
			<< " lvl = " << lvl 
			<< " current_node = " << current_node 
			<< " verbose_level = " << verbose_level 
			<< endl;
		cout << "node=" << O->node << " prev=" << O->prev << " pt=" << O->pt << endl;
		INT_set_print(cout, gen->set[lvl], size);
		cout << endl;
		}
#if 0
	if (current_node < path[lvl]) {
		cout << "upstep_work::find_automorphism_by_tracing_recursion: not canonical" << endl;
		cout << "current_node=" << current_node << endl;
		cout << "path[lvl]=" << path[lvl] << endl;
		return not_canonical;
		}
#endif
	if (f_v4) {
		if (gen->f_print_function) {
			(*gen->print_function)(size, gen->set[lvl], gen->print_function_data);
			}
		}
	
#if 0
	if (f_debug) {
		if (!O->check_node_and_set_consistency(gen, lvl - 1, gen->set[lvl])) {
			print_level_extension_coset_info();
			cout << "upstep_work::find_automorphism_by_tracing_recursion: node and set inconsistent, the node corresponds to" << endl;
			O->store_set_to(gen, lvl - 1, gen->set3);
			INT_set_print(cout, gen->set3, lvl);
			cout << endl;
			exit(1);
			}
		}
#endif

	if (lvl == 0 && gen->f_starter) {
		INT *cur_set = gen->set[0];
		INT *next_set = gen->set[0 + gen->starter_size];
		INT *cur_transporter = gen->transporter->ith(0);
		INT *next_transporter = gen->transporter->ith(0 + gen->starter_size);
		
		O->trace_starter(gen, size, 
			cur_set, next_set,
			cur_transporter, next_transporter, 
			0 /*verbose_level */);
		if (f_v) {
			cout << "recognize_recursion at ";
			cout << "(" << lvl << "/" << node << ") after trace_starter, calling recognize_recursion" << endl;
			}
		recognize_recursion(
			gen, size, f_implicit_fusion, 
			gen->starter_size, gen->starter_size, final_node, 
			verbose_level);

		if (f_v) {
			cout << "recognize_recursion at ";
			cout << "(" << lvl << "/" << node << ") after recognize_recursion" << endl;
			}

		return;
		}
	
	if (f_v) {
		cout << "recognize_recursion at ";
		cout << "(" << lvl << "/" << node << ") before O->trace_next_point_wrapper" << endl;
		}
	if (!O->trace_next_point_wrapper(gen, 
		lvl, current_node, size - 1 /*len*/, 
		f_implicit_fusion, f_failure_to_find_point, 0 /*verbose_level - 5*/)) {

		// FALSE in trace_next_point_wrapper can only happen if f_implicit_fusion is true.
		
		
		if (f_v) {
			cout << "recognize_recursion at ";
			cout << "(" << lvl << "/" << node << ") O->trace_next_point_wrapper returns FALSE, starting over" << endl;
			}

		
		recognize_start_over(
			gen, size, f_implicit_fusion, 
			lvl, current_node, final_node, 
			verbose_level);
		if (f_v) {
			cout << "recognize_recursion at ";
			cout << "(" << lvl << "/" << node << ") O->trace_next_point_wrapper returnsed FALSE, after over" << endl;
			}
		}

	if (f_v) {
		cout << "recognize_recursion at ";
		cout << "(" << lvl << "/" << node << ") after O->trace_next_point_wrapper" << endl;
		}
	
	if (f_failure_to_find_point) {
		cout << "recognize_recursion failure to find point" << endl;
		exit(1);
		}

	pt0 = gen->set[lvl + 1][lvl];

	if (f_v) {
		cout << "recognize_recursion at ";
		cout << "(" << lvl << "/" << node << ") trying to find extension for point pt0=" << pt0 << endl;
		}




	current_extension = O->find_extension_from_point(gen, pt0, FALSE);
	
	if (f_v) {
		cout << "recognize_recursion at ";
		cout << "(" << lvl << "/" << node << "/" << current_extension << ") current_extension=" << current_extension << endl;
		}
	if (current_extension == -1) {

		cout << "recognize_recursion failure in find_extension_from_point" << endl;
		
		cout << "the original set is" << endl;
		INT_set_print(cout, gen->set[0], size);
		cout << endl;
		//if (gen->f_print_function) {
			//(*gen->print_function)(cout, size, gen->set[0], gen->print_function_data);
			//}
		cout << "the current set is" << endl;
		INT_set_print(cout, gen->set[lvl + 1], size);
		cout << endl;
		//if (gen->f_print_function) {
			//(*gen->print_function)(cout, size, gen->set[lvl + 1], gen->print_function_data);
			//}
		cout << "the node corresponds to" << endl;
		O->store_set_to(gen, lvl - 1, gen->set3);
		INT_set_print(cout, gen->set3, lvl);
		cout << endl;

		cout << "lvl = " << lvl << endl;
		cout << "current_node = " << current_node << endl;

		exit(1);

		}



	if (f_v5) {
		cout << "recognize_recursion point " << pt0 << " is extension no " << current_extension << endl;
		}
	if (gen->f_allowed_to_show_group_elements && f_v4) {
		INT *transporter = gen->transporter->ith(lvl + 1);
		cout << "recognize_recursion transporter element:" << endl;
		gen->A2->element_print_quick(transporter, cout);
		//gen->A2->element_print_as_permutation(transporter, cout);
		cout << endl;
		}
	

	
	// now lvl < size - 1
	
	if (O->E[current_extension].type == EXTENSION_TYPE_FUSION) {
		INT next_node;
		
		if (f_v4) {
			cout << "recognize_recursion at ";
			cout << "(" << lvl << "/" << node << "/" << current_extension << ")";
			cout << " fusion node " << O->node << endl;
			}
		next_node = O->apply_fusion_element(gen, 
			lvl, current_node, 
			current_extension, size - 1 /* len */, FALSE /* f_tolerant */, verbose_level - 6);
		
		if (f_v) {
			cout << "recognize_recursion lvl " << lvl << " at ";
			cout << "(" << lvl << "/" << node << "/" << current_extension << ")";
			cout << " fusion from " << O->node << " to " << next_node << endl;
			}
		if (next_node == -1) {
			cout << "next_node == -1" << endl;
			exit(1);
			}
		if (f_v5) {
			cout << "recognize_recursion at ";
			cout << "(" << lvl << "/" << node << "/" << current_extension << ")";
			cout << " after apply_fusion_element, next_node=" << next_node << endl;
			}
#if 0
		if (next_node < path[lvl + 1]) {
			if (f_v) {
				cout << "recognize_recursion lvl " << lvl << " not canonical" << endl;
				cout << "next_node=" << next_node << endl;
				//cout << "path[lvl + 1]=" << path[lvl + 1] << endl;
				}
			return not_canonical;
			}
#endif
		


		recognize_recursion(
			gen, size, f_implicit_fusion, 
			lvl + 1, next_node, final_node, verbose_level);

		return;

		}
	else if (O->E[current_extension].type == EXTENSION_TYPE_EXTENSION) {
		INT next_node;
		
		if (f_v4) {
			cout << "recognize_recursion extension node" << endl;
			}
		next_node = O->E[current_extension].data;
		if (f_v) {
			cout << "recognize_recursion at ";
			cout << "(" << lvl << "/" << node << "/" << current_extension << ")";
			cout << " extension from " << O->node << " to " << next_node << endl;
			}

		recognize_recursion(
			gen, size, f_implicit_fusion, 
			lvl + 1, next_node, final_node, verbose_level);
		
		return;
		}
	else if (O->E[current_extension].type == EXTENSION_TYPE_UNPROCESSED) {
		cout << "recognize_recursion unprocessed node, this should not happen" << endl;
		exit(1);
		}
	else if (O->E[current_extension].type == EXTENSION_TYPE_PROCESSING) {
		cout << "recognize_recursion processing node, this should not happen" << endl;
		exit(1);
		}
	cout << "recognize_recursion unknown type of extension" << endl;
	exit(1);
}

void recognize(
	generator *gen, 
	INT *the_set, INT size, INT *transporter, INT f_implicit_fusion, 
	INT &final_node, INT verbose_level)
// This routine is called from upstep (upstep_work::upstep_subspace_action).
// It in turn calls oracle::find_automorphism_by_tracing_recursion
// It tries to compute an isomorphism of the set in set[0][0,...,len] 
// (i.e. of size len+1) to the 
// set in S[0,...,len] which sends set[0][len] to S[len].
// Since set[0][0,...,len] is a permutation of S[0,...,len], this isomorphism is 
// in fact an automorphism which maps S[len] to one of the points in S[0,...,len - 1].
// If this is done for all possible points in S[0,...,len - 1], 
// a transversal for H in the stabilizer of S[0,...,len] results, 
// where H is the point stabilizer of S[len] in the set-stabilizer of S[0,...,len-1], 
// (which is a subgroup of S[0,...,len]).  
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	
	if (f_vvv) {
		cout << "recognize" << endl;
		}
	// put in default values just in case we are doing a 
	// tolerant search and do not have a final result
	final_node = -1;
	
	INT_vec_copy(the_set, gen->set[0], size);

	gen->A->element_one(gen->transporter->ith(0), 0);

	if (f_vv) {
		INT_vec_print(cout, gen->set[0], size);
		cout << endl;
		if (gen->f_print_function) {
			(*gen->print_function)(size, gen->set[0], gen->print_function_data);
			}
		}
	if (size > gen->sz) {
		cout << "recognize size > sz" << endl;
		cout << "size=" << size << endl;
		cout << "gen->sz=" << gen->sz << endl;
		exit(1);
		}

	gen->nb_times_trace++;

#if 0
	if ((gen->nb_times_trace & ((1 << 17) - 1)) == 0) {
		cout << "upstep_work::find_automorphism_by_tracing() " << endl;
		cout << "nb_times_trace=" << gen->nb_times_trace << endl;
		cout << "nb_times_trace_was_saved=" << gen->nb_times_trace_was_saved << endl;
		}
#endif
	
	INT_vec_copy(gen->set[0], gen->set0, size);
#if 0
	INT_vec_heapsort(gen->set0, size - 1); // INT_vec_sort(len, set0);  
		// important: we keep the last point separate
#endif

	recognize_recursion(
		gen, 
		size, f_implicit_fusion, 
		0, 0,  // start from the very first node
		final_node, 
		verbose_level);

	if (f_v) {
		cout << "recognize after recognize_recursion, copying transporter" << endl;
		}


	gen->A->element_move(gen->transporter->ith(size), transporter, 0);


	if (f_v) {
		cout << "recognize done" << endl;
		}
}



