// generator_recognize.C
//
// Anton Betten
//
// started July 19, 2014

#include "GALOIS/galois.h"
#include "ACTION/action.h"
#include "SNAKES_AND_LADDERS/snakesandladders.h"

void generator::recognize_start_over(
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
		cout << "generator::recognize_start_over" << endl;
		}
	// this is needed if implicit fusion nodes are used:
	if (lvl == size - 1) {
		if (f_v) {
			cout << "generator::recognize_start_over "
					"lvl == size - 1" << endl;
			}
		final_node = current_node;
		exit(1);
		}


	INT_vec_heapsort(set[lvl + 1], size /* - 1 */);
		// we don't keep the last point (i.e., the (len + 1)-th) extra
	INT_vec_copy(set[lvl + 1], set[0], size);
	//INT_vec_copy(size, set[lvl + 1], gen->set[0]);
	if (f_vv) {
		INT_set_print(cout, set[0], size);
		cout << endl;
		}
	A->element_move(
		transporter->ith(lvl + 1),
		transporter->ith(0), FALSE);
	if (f_v) {
		cout << "generator::recognize_start_over "
				"before recognize_recursion" << endl;
		}
	recognize_recursion(
		size, f_implicit_fusion,
		0, 0, final_node,
		verbose_level);
	if (f_v) {
		cout << "generator::recognize_start_over "
				"after recognize_recursion" << endl;
		}
	if (f_v) {
		cout << "generator::recognize_start_over done" << endl;
		}
}

void generator::recognize_recursion(
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


	node = current_node - first_oracle_node_at_level[lvl];
	

	if (f_v) {
		cout << "generator::recognize_recursion at ";
		cout << "(" << lvl << "/" << node  << ")" << endl;
		}


	if (lvl == size) {
		if (f_v) {
			cout << "generator::recognize_recursion at ";
			cout << "(" << lvl << "/" << node  << ") "
					"lvl == size, terminating" << endl;
			}
		final_node = current_node;
		return;
		}

	oracle *O;

	O = &root[current_node];
	if (f_vvv) {
		cout << "generator::recognize_recursion"
			<< " lvl = " << lvl 
			<< " current_node = " << current_node 
			<< " verbose_level = " << verbose_level 
			<< endl;
		cout << "node=" << O->node << " prev=" << O->prev
				<< " pt=" << O->pt << endl;
		INT_set_print(cout, set[lvl], size);
		cout << endl;
		}
	if (f_v4) {
		if (f_print_function) {
			(*print_function)(size, set[lvl],
					print_function_data);
			}
		}
	
#if 0
	if (f_debug) {
		if (!O->check_node_and_set_consistency(this, lvl - 1,
				gen->set[lvl])) {
			print_level_extension_coset_info();
			cout << "upstep_work::find_automorphism_by_tracing_recursion: "
					"node and set inconsistent, the node "
					"corresponds to" << endl;
			O->store_set_to(this, lvl - 1, set3);
			INT_set_print(cout, set3, lvl);
			cout << endl;
			exit(1);
			}
		}
#endif

	if (lvl == 0 && f_starter) {
		INT *cur_set = set[0];
		INT *next_set = set[0 + starter_size];
		INT *cur_transporter = transporter->ith(0);
		INT *next_transporter = transporter->ith(
				0 + starter_size);
		
		O->trace_starter(this, size,
			cur_set, next_set,
			cur_transporter, next_transporter, 
			0 /*verbose_level */);
		if (f_v) {
			cout << "generator::recognize_recursion at ";
			cout << "(" << lvl << "/" << node << ") after trace_starter, "
					"calling recognize_recursion" << endl;
			}
		recognize_recursion(
			size, f_implicit_fusion,
			starter_size, starter_size, final_node,
			verbose_level);

		if (f_v) {
			cout << "generator::recognize_recursion at ";
			cout << "(" << lvl << "/" << node << ") "
					"after recognize_recursion" << endl;
			}

		return;
		}
	
	if (f_v) {
		cout << "generator::recognize_recursion at ";
		cout << "(" << lvl << "/" << node << ") "
				"before O->trace_next_point_wrapper" << endl;
		}
	if (!O->trace_next_point_wrapper(this,
		lvl, current_node, size - 1 /*len*/, 
		f_implicit_fusion, f_failure_to_find_point,
		0 /*verbose_level - 5*/)) {

		// FALSE in trace_next_point_wrapper
		// can only happen if f_implicit_fusion is true.
		
		
		if (f_v) {
			cout << "generator::recognize_recursion at ";
			cout << "(" << lvl << "/" << node
					<< ") O->trace_next_point_wrapper "
							"returns FALSE, starting over" << endl;
			}

		
		recognize_start_over(
			size, f_implicit_fusion,
			lvl, current_node, final_node, 
			verbose_level);
		if (f_v) {
			cout << "generator::recognize_recursion at ";
			cout << "(" << lvl << "/" << node << ") "
					"O->trace_next_point_wrapper returns FALSE, after over" << endl;
			}
		}

	if (f_v) {
		cout << "generator::recognize_recursion at ";
		cout << "(" << lvl << "/" << node << ") after "
				"O->trace_next_point_wrapper" << endl;
		}
	
	if (f_failure_to_find_point) {
		cout << "generator::recognize_recursion failure to find point" << endl;
		exit(1);
		}

	pt0 = set[lvl + 1][lvl];

	if (f_v) {
		cout << "generator::recognize_recursion at ";
		cout << "(" << lvl << "/" << node << ") trying to find "
				"extension for point pt0=" << pt0 << endl;
		}




	current_extension = O->find_extension_from_point(this, pt0, FALSE);
	
	if (f_v) {
		cout << "generator::recognize_recursion at ";
		cout << "(" << lvl << "/" << node << "/" << current_extension
				<< ") current_extension=" << current_extension << endl;
		}
	if (current_extension == -1) {

		cout << "generator::recognize_recursion failure in "
				"find_extension_from_point" << endl;
		
		cout << "generator::recognize_recursion "
				"the original set is" << endl;
		INT_set_print(cout, set[0], size);
		cout << endl;
		//if (gen->f_print_function) {
			//(*gen->print_function)(cout, size, gen->set[0],
			//gen->print_function_data);
			//}
		cout << "generator::recognize_recursion "
				"the current set is" << endl;
		INT_set_print(cout, set[lvl + 1], size);
		cout << endl;
		//if (f_print_function) {
			//(*print_function)(cout, size, set[lvl + 1],
			//print_function_data);
			//}
		cout << "generator::recognize_recursion "
				"the node corresponds to" << endl;
		O->store_set_to(this, lvl - 1, set3);
		INT_set_print(cout, set3, lvl);
		cout << endl;

		cout << "generator::recognize_recursion "
				"lvl = " << lvl << endl;
		cout << "generator::recognize_recursion "
				"current_node = " << current_node << endl;

		exit(1);

		}



	if (f_v5) {
		cout << "generator::recognize_recursion point " << pt0
				<< " is extension no " << current_extension << endl;
		}
	if (f_allowed_to_show_group_elements && f_v4) {
		INT *transporter1 = transporter->ith(lvl + 1);
		cout << "recognize_recursion transporter element:" << endl;
		A2->element_print_quick(transporter1, cout);
		//A2->element_print_as_permutation(transporter1, cout);
		cout << endl;
		}
	

	
	// now lvl < size - 1
	
	if (O->E[current_extension].type == EXTENSION_TYPE_FUSION) {
		INT next_node;
		
		if (f_v4) {
			cout << "generator::recognize_recursion at ";
			cout << "(" << lvl << "/" << node << "/"
					<< current_extension << ")";
			cout << " fusion node " << O->node << endl;
			}
		next_node = O->apply_fusion_element(this,
			lvl, current_node, 
			current_extension, size - 1 /* len */,
			FALSE /* f_tolerant */, verbose_level - 6);
		
		if (f_v) {
			cout << "generator::recognize_recursion lvl " << lvl << " at ";
			cout << "(" << lvl << "/" << node << "/"
					<< current_extension << ")";
			cout << " fusion from " << O->node << " to "
					<< next_node << endl;
			}
		if (next_node == -1) {
			cout << "generator::recognize_recursion next_node == -1" << endl;
			exit(1);
			}
		if (f_v5) {
			cout << "generator::recognize_recursion at ";
			cout << "(" << lvl << "/" << node << "/"
					<< current_extension << ")";
			cout << " after apply_fusion_element, "
					"next_node=" << next_node << endl;
			}
#if 0
		if (next_node < path[lvl + 1]) {
			if (f_v) {
				cout << "generator::recognize_recursion lvl "
						<< lvl << " not canonical" << endl;
				cout << "next_node=" << next_node << endl;
				//cout << "path[lvl + 1]=" << path[lvl + 1] << endl;
				}
			return not_canonical;
			}
#endif
		


		recognize_recursion(
			size, f_implicit_fusion,
			lvl + 1, next_node, final_node,
			verbose_level);

		return;

		}
	else if (O->E[current_extension].type == EXTENSION_TYPE_EXTENSION) {
		INT next_node;
		
		if (f_v4) {
			cout << "generator::recognize_recursion "
					"extension node" << endl;
			}
		next_node = O->E[current_extension].data;
		if (f_v) {
			cout << "generator::recognize_recursion at ";
			cout << "(" << lvl << "/" << node << "/"
					<< current_extension << ")";
			cout << " extension from " << O->node << " to "
					<< next_node << endl;
			}

		recognize_recursion(
			size, f_implicit_fusion,
			lvl + 1, next_node, final_node,
			verbose_level);
		
		return;
		}
	else if (O->E[current_extension].type == EXTENSION_TYPE_UNPROCESSED) {
		cout << "generator::recognize_recursion unprocessed node, "
				"this should not happen" << endl;
		exit(1);
		}
	else if (O->E[current_extension].type == EXTENSION_TYPE_PROCESSING) {
		cout << "generator::recognize_recursion processing node, "
				"this should not happen" << endl;
		exit(1);
		}
	cout << "generator::recognize_recursion unknown type of extension" << endl;
	exit(1);
}

void generator::recognize(
	INT *the_set, INT size, INT *transporter,
	INT f_implicit_fusion,
	INT &final_node, INT verbose_level)
// This routine is called from upstep
// (upstep_work::upstep_subspace_action).
// It in turn calls oracle::find_automorphism_by_tracing_recursion
// It tries to compute an isomorphism
// of the set in set[0][0,...,len]
// (i.e. of size len+1) to the 
// set in S[0,...,len] which sends set[0][len] to S[len].
// Since set[0][0,...,len] is a permutation
// of S[0,...,len], this isomorphism is
// in fact an automorphism which maps S[len]
// to one of the points in S[0,...,len - 1].
// If this is done for all possible points
// in S[0,...,len - 1],
// a transversal for H in the stabilizer
// of S[0,...,len] results,
// where H is the point stabilizer of S[len]
// in the set-stabilizer of S[0,...,len-1],
// (which is a subgroup of S[0,...,len]).  
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	
	if (f_vvv) {
		cout << "generator::recognize" << endl;
		}
	// put in default values just in case we are doing a 
	// tolerant search and do not have a final result
	final_node = -1;
	
	INT_vec_copy(the_set, set[0], size);

	A->element_one(generator::transporter->ith(0), 0);

	if (f_vv) {
		INT_vec_print(cout, set[0], size);
		cout << endl;
		if (f_print_function) {
			(*print_function)(size, set[0],
					print_function_data);
			}
		}
	if (size > sz) {
		cout << "generator::recognize size > sz" << endl;
		cout << "size=" << size << endl;
		cout << "gen->sz=" << sz << endl;
		exit(1);
		}

	nb_times_trace++;

	
	INT_vec_copy(set[0], set0, size);

	recognize_recursion(
		size, f_implicit_fusion, 
		0, 0,  // start from the very first node
		final_node, 
		verbose_level);

	if (f_v) {
		cout << "generator::recognize after recognize_recursion, "
				"copying transporter" << endl;
		}


	A->element_move(generator::transporter->ith(size), transporter, 0);


	if (f_v) {
		cout << "generator::recognize done" << endl;
		}
}



