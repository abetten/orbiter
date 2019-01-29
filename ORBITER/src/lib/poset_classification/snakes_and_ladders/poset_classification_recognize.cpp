// poset_classification_recognize.C
//
// Anton Betten
//
// started July 19, 2014

#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "poset_classification/poset_classification.h"

namespace orbiter {
namespace classification {

void poset_classification::recognize_start_over(
	int size, int f_implicit_fusion, 
	int lvl, int current_node, 
	int &final_node, int verbose_level)
// Called from poset_orbit_node::recognize_recursion
// when trace_next_point returns FALSE
// This can happen only if f_implicit_fusion is TRUE
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "poset_classification::recognize_start_over" << endl;
		}
	// this is needed if implicit fusion nodes are used:
	if (lvl == size - 1) {
		if (f_v) {
			cout << "poset_classification::recognize_start_over "
					"lvl == size - 1" << endl;
			}
		final_node = current_node;
		exit(1);
		}


	int_vec_heapsort(set[lvl + 1], size /* - 1 */);
		// we don't keep the last point (i.e., the (len + 1)-th) extra
	int_vec_copy(set[lvl + 1], set[0], size);
	//int_vec_copy(size, set[lvl + 1], gen->set[0]);
	if (f_vv) {
		int_set_print(cout, set[0], size);
		cout << endl;
		}
	Poset->A->element_move(
		transporter->ith(lvl + 1),
		transporter->ith(0), FALSE);
	if (f_v) {
		cout << "poset_classification::recognize_start_over "
				"before recognize_recursion" << endl;
		}
	recognize_recursion(
		size, f_implicit_fusion,
		0, 0, final_node,
		verbose_level);
	if (f_v) {
		cout << "poset_classification::recognize_start_over "
				"after recognize_recursion" << endl;
		}
	if (f_v) {
		cout << "poset_classification::recognize_start_over done" << endl;
		}
}

void poset_classification::recognize_recursion(
	int size, int f_implicit_fusion, 
	int lvl, int current_node, int &final_node, 
	int verbose_level)
// this routine is called by upstep_work::recognize
// we are dealing with a set of size size.
// the tracing starts at lvl = 0 with current_node = 0
{
	//if (my_node == 9 && my_extension == 4) {verbose_level += 10;}
	int pt0, current_extension;
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_v10 = (verbose_level >= 10);
	int f_vvv = (verbose_level >= 3);
	int f_v4 = (verbose_level >= 3);
	int f_v5 = (verbose_level >= 3);
	int f_failure_to_find_point;
	int node;


	node = current_node - first_poset_orbit_node_at_level[lvl];
	

	if (f_v) {
		cout << "poset_classification::recognize_recursion at ";
		cout << "(" << lvl << "/" << node  << ")" << endl;
		}


	if (lvl == size) {
		if (f_v) {
			cout << "poset_classification::recognize_recursion at ";
			cout << "(" << lvl << "/" << node  << ") "
					"lvl == size, terminating" << endl;
			}
		final_node = current_node;
		return;
		}

	poset_orbit_node *O;

	O = &root[current_node];
	if (f_vvv) {
		cout << "poset_classification::recognize_recursion"
			<< " lvl = " << lvl 
			<< " current_node = " << current_node 
			<< " verbose_level = " << verbose_level 
			<< endl;
		cout << "node=" << O->node << " prev=" << O->prev
				<< " pt=" << O->pt << endl;
		int_set_print(cout, set[lvl], size);
		cout << endl;
		}
	if (f_v4) {
		if (f_print_function) {
			(*print_function)(cout, size, set[lvl],
					print_function_data);
			}
		}
	
#if 0
	if (f_debug) {
		if (!O->check_node_and_set_consistency(this, lvl - 1,
				gen->set[lvl])) {
			print_level_extension_coset_info();
			cout << "upstep_work::recognize_recursion: "
					"node and set inconsistent, the node "
					"corresponds to" << endl;
			O->store_set_to(this, lvl - 1, set3);
			int_set_print(cout, set3, lvl);
			cout << endl;
			exit(1);
			}
		}
#endif

	if (lvl == 0 && f_starter) {
		int *cur_set = set[0];
		int *next_set = set[0 + starter_size];
		int *cur_transporter = transporter->ith(0);
		int *next_transporter = transporter->ith(
				0 + starter_size);
		
		O->trace_starter(this, size,
			cur_set, next_set,
			cur_transporter, next_transporter, 
			0 /*verbose_level */);
		if (f_v) {
			cout << "poset_classification::recognize_recursion at ";
			cout << "(" << lvl << "/" << node << ") "
					"after trace_starter, "
					"calling recognize_recursion" << endl;
			}
		recognize_recursion(
			size, f_implicit_fusion,
			starter_size, starter_size, final_node,
			verbose_level);

		if (f_v) {
			cout << "poset_classification::recognize_recursion at ";
			cout << "(" << lvl << "/" << node << ") "
					"after recognize_recursion" << endl;
			}

		return;
		}
	
	if (f_v) {
		cout << "poset_classification::recognize_recursion at ";
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
			cout << "poset_classification::recognize_recursion at ";
			cout << "(" << lvl << "/" << node
					<< ") O->trace_next_point_wrapper "
							"returns FALSE, starting over" << endl;
			}

		
		recognize_start_over(
			size, f_implicit_fusion,
			lvl, current_node, final_node, 
			verbose_level);
		if (f_v) {
			cout << "poset_classification::recognize_recursion at ";
			cout << "(" << lvl << "/" << node << ") "
					"O->trace_next_point_wrapper "
					"returns FALSE, after over" << endl;
			}
		}

	if (f_v) {
		cout << "poset_classification::recognize_recursion at ";
		cout << "(" << lvl << "/" << node << ") after "
				"O->trace_next_point_wrapper" << endl;
		}
	
	if (f_failure_to_find_point) {
		cout << "poset_classification::recognize_recursion "
				"failure to find point" << endl;
		exit(1);
		}

	pt0 = set[lvl + 1][lvl];

	if (f_v) {
		cout << "poset_classification::recognize_recursion at ";
		cout << "(" << lvl << "/" << node << ") trying to find "
				"extension for point pt0=" << pt0 << endl;
		}




	current_extension = O->find_extension_from_point(this, pt0, FALSE);
	
	if (f_v) {
		cout << "poset_classification::recognize_recursion at ";
		cout << "(" << lvl << "/" << node << "/" << current_extension
				<< ") current_extension=" << current_extension << endl;
		}
	if (current_extension == -1) {

		cout << "poset_classification::recognize_recursion failure in "
				"find_extension_from_point" << endl;
		
		cout << "poset_classification::recognize_recursion "
				"the original set is" << endl;
		int_set_print(cout, set[0], size);
		cout << endl;
		//if (gen->f_print_function) {
			//(*gen->print_function)(cout, size, gen->set[0],
			//gen->print_function_data);
			//}
		cout << "poset_classification::recognize_recursion "
				"the current set is" << endl;
		int_set_print(cout, set[lvl + 1], size);
		cout << endl;
		//if (f_print_function) {
			//(*print_function)(cout, size, set[lvl + 1],
			//print_function_data);
			//}
		cout << "poset_classification::recognize_recursion "
				"the node corresponds to" << endl;
		O->store_set_to(this, lvl - 1, set3);
		int_set_print(cout, set3, lvl);
		cout << endl;

		cout << "poset_classification::recognize_recursion "
				"lvl = " << lvl << endl;
		cout << "poset_classification::recognize_recursion "
				"current_node = " << current_node << endl;

		exit(1);

		}



	if (f_v5) {
		cout << "poset_classification::recognize_recursion point " << pt0
				<< " is extension no " << current_extension << endl;
		}
	if (f_allowed_to_show_group_elements && f_v4) {
		int *transporter1 = transporter->ith(lvl + 1);
		cout << "recognize_recursion transporter element:" << endl;
		Poset->A2->element_print_quick(transporter1, cout);
		//A2->element_print_as_permutation(transporter1, cout);
		cout << endl;
		}
	

	
	// now lvl < size - 1
	
	if (O->E[current_extension].type == EXTENSION_TYPE_FUSION) {
		int next_node;
		
		if (f_v4) {
			cout << "poset_classification::recognize_recursion at ";
			cout << "(" << lvl << "/" << node << "/"
					<< current_extension << ")";
			cout << " fusion node " << O->node << endl;
			}
		next_node = O->apply_isomorphism(this,
			lvl, current_node, 
			current_extension, size - 1 /* len */,
			FALSE /* f_tolerant */, verbose_level - 6);
		
		if (f_v) {
			cout << "poset_classification::recognize_recursion "
					"lvl " << lvl << " at ";
			cout << "(" << lvl << "/" << node << "/"
					<< current_extension << ")";
			cout << " fusion from " << O->node << " to "
					<< next_node << endl;
			}
		if (next_node == -1) {
			cout << "poset_classification::recognize_recursion "
					"next_node == -1" << endl;
			exit(1);
			}
		if (f_v5) {
			cout << "poset_classification::recognize_recursion at ";
			cout << "(" << lvl << "/" << node << "/"
					<< current_extension << ")";
			cout << " after apply_isomorphism, "
					"next_node=" << next_node << endl;
			}
#if 0
		if (next_node < path[lvl + 1]) {
			if (f_v) {
				cout << "poset_classification::recognize_recursion "
						"lvl " << lvl << " not canonical" << endl;
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
		int next_node;
		
		if (f_v4) {
			cout << "poset_classification::recognize_recursion "
					"extension node" << endl;
			}
		next_node = O->E[current_extension].data;
		if (f_v) {
			cout << "poset_classification::recognize_recursion at ";
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
		cout << "poset_classification::recognize_recursion "
				"unprocessed node, "
				"this should not happen" << endl;
		exit(1);
		}
	else if (O->E[current_extension].type == EXTENSION_TYPE_PROCESSING) {
		cout << "poset_classification::recognize_recursion "
				"processing node, "
				"this should not happen" << endl;
		exit(1);
		}
	cout << "poset_classification::recognize_recursion "
			"unknown type of extension" << endl;
	exit(1);
}

void poset_classification::recognize(
	int *the_set, int size, int *transporter,
	int f_implicit_fusion,
	int &final_node, int verbose_level)
// This routine is called from upstep
// (upstep_work::upstep_subspace_action).
// It in turn calls poset_orbit_node::recognize_recursion
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
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	
	if (f_vvv) {
		cout << "poset_classification::recognize" << endl;
		}
	// put in default values just in case we are doing a 
	// tolerant search and do not have a final result
	final_node = -1;
	
	int_vec_copy(the_set, set[0], size);

	Poset->A->element_one(poset_classification::transporter->ith(0), 0);

	if (f_vv) {
		int_vec_print(cout, set[0], size);
		cout << endl;
		if (f_print_function) {
			(*print_function)(cout, size, set[0],
					print_function_data);
			}
		}
	if (size > sz) {
		cout << "poset_classification::recognize size > sz" << endl;
		cout << "size=" << size << endl;
		cout << "gen->sz=" << sz << endl;
		exit(1);
		}

	nb_times_trace++;

	
	int_vec_copy(set[0], set0, size);

	recognize_recursion(
		size, f_implicit_fusion, 
		0, 0,  // start from the very first node
		final_node, 
		verbose_level);

	if (f_v) {
		cout << "poset_classification::recognize "
				"after recognize_recursion, "
				"copying transporter" << endl;
		}


	Poset->A->element_move(
			poset_classification::transporter->ith(size),
			transporter, 0);


	if (f_v) {
		cout << "poset_classification::recognize done" << endl;
		}
}


}}

