/*
 * orbit_tracer.cpp
 *
 *  Created on: Apr 22, 2022
 *      Author: betten
 */


#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


orbit_tracer::orbit_tracer()
{
	PC = NULL;

	Transporter = NULL;

	Set = NULL;

	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;

}

orbit_tracer::~orbit_tracer()
{
	int i;

	if (Transporter) {
		FREE_OBJECT(Transporter);
	}
	if (Set) {
		for (i = 0; i <= PC->get_sz(); i++) {
			FREE_lint(Set[i]);
		}
		FREE_plint(Set);
	}
	if (Elt1) {
		FREE_int(Elt1);
	}
	if (Elt2) {
		FREE_int(Elt2);
	}
	if (Elt3) {
		FREE_int(Elt3);
	}
}

void orbit_tracer::init(poset_classification *PC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_tracer::init" << endl;
	}

	int i;

	orbit_tracer::PC = PC;

	Transporter = NEW_OBJECT(data_structures_groups::vector_ge);
	Transporter->init(PC->get_poset()->A, verbose_level - 2);
	Transporter->allocate(PC->get_sz() + 1, verbose_level - 2);
	PC->get_poset()->A->Group_element->element_one(Transporter->ith(0), FALSE);

	Set = NEW_plint(PC->get_sz() + 1);
	for (i = 0; i <= PC->get_sz(); i++) {
		Set[i] = NEW_lint(PC->get_max_set_size());
	}

	Elt1 = NEW_int(PC->get_poset()->A->elt_size_in_int);
	Elt2 = NEW_int(PC->get_poset()->A->elt_size_in_int);
	Elt3 = NEW_int(PC->get_poset()->A->elt_size_in_int);

	if (f_v) {
		cout << "orbit_tracer::init done" << endl;
	}
}

data_structures_groups::vector_ge *orbit_tracer::get_transporter()
{
	return Transporter;
}

long int *orbit_tracer::get_set_i(int i)
{
	return Set[i];
}


void orbit_tracer::recognize_start_over(
	int size,
	int lvl, int current_node,
	int &final_node, int verbose_level)
// Called from poset_orbit_node::recognize_recursion
// when trace_next_point returns FALSE
// This can happen only if f_implicit_fusion is TRUE
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "orbit_tracer::recognize_start_over" << endl;
	}
	// this is needed if implicit fusion nodes are used:
	if (lvl == size - 1) {
		if (f_v) {
			cout << "orbit_tracer::recognize_start_over "
					"lvl == size - 1" << endl;
		}
		final_node = current_node;
		exit(1);
	}


	Sorting.lint_vec_heapsort(Set[lvl + 1], size /* - 1 */);
		// we don't keep the last point (i.e., the (len + 1)-th) extra
	Lint_vec_copy(Set[lvl + 1], Set[0], size);

	if (f_vv) {
		orbiter_kernel_system::Orbiter->Lint_vec->set_print(cout, Set[0], size);
		cout << endl;
	}
	PC->get_poset()->A->Group_element->element_move(
		Transporter->ith(lvl + 1),
		Transporter->ith(0), FALSE);
	if (f_v) {
		cout << "orbit_tracer::recognize_start_over "
				"before recognize_recursion" << endl;
	}
	recognize_recursion(
		size,
		0, 0, final_node,
		verbose_level);
	if (f_v) {
		cout << "orbit_tracer::recognize_start_over "
				"after recognize_recursion" << endl;
	}
	if (f_v) {
		cout << "orbit_tracer::recognize_start_over done" << endl;
	}
}

void orbit_tracer::recognize_recursion(
	int size,
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


	node = current_node - PC->get_Poo()->first_node_at_level(lvl);


	if (f_v) {
		cout << "orbit_tracer::recognize_recursion at ";
		cout << "(" << lvl << "/" << node  << ")" << endl;
	}


	if (lvl == size) {
		if (f_v) {
			cout << "orbit_tracer::recognize_recursion at ";
			cout << "(" << lvl << "/" << node  << ") "
					"lvl == size, terminating" << endl;
		}
		final_node = current_node;
		return;
	}

	poset_orbit_node *O;

	//O = &root[current_node];
	O = PC->get_Poo()->get_node(current_node);
	if (f_vvv) {
		cout << "orbit_tracer::recognize_recursion"
			<< " lvl = " << lvl
			<< " current_node = " << current_node
			<< " verbose_level = " << verbose_level
			<< endl;
		cout << "node=" << O->get_node() << " prev=" << O->get_prev()
				<< " pt=" << O->get_pt() << endl;
		orbiter_kernel_system::Orbiter->Lint_vec->set_print(cout, Set[lvl], size);
		cout << endl;
	}
	if (f_v4) {
		if (PC->get_poset()->f_print_function) {
			(*PC->get_poset()->print_function)(cout, size, Set[lvl],
					PC->get_poset()->print_function_data);
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

	if (lvl == 0 && PC->has_base_case()) {
		long int *cur_set = Set[0];
		long int *next_set = Set[0 + PC->get_Base_case()->size];
		int *cur_transporter = Transporter->ith(0);
		int *next_transporter = Transporter->ith(0 + PC->get_Base_case()->size);

		O->trace_starter(PC, size,
			cur_set, next_set,
			cur_transporter, next_transporter,
			0 /*verbose_level */);
		if (f_v) {
			cout << "orbit_tracer::recognize_recursion at ";
			cout << "(" << lvl << "/" << node << ") "
					"after trace_starter, "
					"calling recognize_recursion" << endl;
		}
		recognize_recursion(
			size,
			PC->get_Base_case()->size, PC->get_Base_case()->size, final_node,
			verbose_level);

		if (f_v) {
			cout << "orbit_tracer::recognize_recursion at ";
			cout << "(" << lvl << "/" << node << ") "
					"after recognize_recursion" << endl;
		}

		return;
	}

	if (f_v) {
		cout << "orbit_tracer::recognize_recursion at ";
		cout << "(" << lvl << "/" << node << ") "
				"before O->trace_next_point_wrapper" << endl;
	}
	if (!O->trace_next_point_wrapper(PC,
		lvl, current_node, size - 1 /*len*/,
		PC->get_control()->f_lex /*f_implicit_fusion*/,
		Elt3 /* cosetrep */,
		f_failure_to_find_point,
		verbose_level)) {

		// FALSE in trace_next_point_wrapper
		// can only happen if f_implicit_fusion is true.


		if (f_v) {
			cout << "orbit_tracer::recognize_recursion at ";
			cout << "(" << lvl << "/" << node
					<< ") O->trace_next_point_wrapper "
							"returns FALSE, starting over" << endl;
		}


		recognize_start_over(
			size,
			lvl, current_node, final_node,
			verbose_level);
		if (f_v) {
			cout << "orbit_tracer::recognize_recursion at ";
			cout << "(" << lvl << "/" << node << ") "
					"O->trace_next_point_wrapper "
					"returns FALSE, after over" << endl;
		}
	}

	if (f_v) {
		cout << "orbit_tracer::recognize_recursion at ";
		cout << "(" << lvl << "/" << node << ") after "
				"O->trace_next_point_wrapper" << endl;
	}

	if (f_failure_to_find_point) {
		cout << "orbit_tracer::recognize_recursion "
				"failure to find point" << endl;
		exit(1);
	}

	pt0 = Set[lvl + 1][lvl];

	if (f_v) {
		cout << "orbit_tracer::recognize_recursion at ";
		cout << "(" << lvl << "/" << node << ") trying to find "
				"extension for point pt0=" << pt0 << endl;
	}




	current_extension = O->find_extension_from_point(PC, pt0, FALSE);

	if (f_v) {
		cout << "orbit_tracer::recognize_recursion at ";
		cout << "(" << lvl << "/" << node << "/" << current_extension
				<< ") current_extension=" << current_extension << endl;
	}
	if (current_extension == -1) {

		cout << "orbit_tracer::recognize_recursion failure in "
				"find_extension_from_point" << endl;

		cout << "orbit_tracer::recognize_recursion "
				"the original set is" << endl;
		orbiter_kernel_system::Orbiter->Lint_vec->set_print(cout, Set[0], size);
		cout << endl;
		//if (gen->f_print_function) {
			//(*gen->print_function)(cout, size, gen->set[0],
			//gen->print_function_data);
			//}
		cout << "orbit_tracer::recognize_recursion "
				"the current set is" << endl;
		orbiter_kernel_system::Orbiter->Lint_vec->set_print(cout, Set[lvl + 1], size);
		cout << endl;
		//if (f_print_function) {
			//(*print_function)(cout, size, set[lvl + 1],
			//print_function_data);
			//}
		cout << "orbit_tracer::recognize_recursion "
				"the node corresponds to" << endl;
		O->store_set_to(PC, lvl - 1, PC->get_Poo()->set3);
		orbiter_kernel_system::Orbiter->Lint_vec->set_print(cout, PC->get_Poo()->set3, lvl);
		cout << endl;

		cout << "orbit_tracer::recognize_recursion "
				"lvl = " << lvl << endl;
		cout << "orbit_tracer::recognize_recursion "
				"current_node = " << current_node << endl;

		exit(1);

	}



	if (f_v5) {
		cout << "orbit_tracer::recognize_recursion point " << pt0
				<< " is extension no " << current_extension << endl;
	}
	if (PC->allowed_to_show_group_elements() && f_v4) {
		int *transporter1 = Transporter->ith(lvl + 1);
		cout << "recognize_recursion transporter element:" << endl;
		PC->get_poset()->A2->Group_element->element_print_quick(transporter1, cout);
		//A2->element_print_as_permutation(transporter1, cout);
		cout << endl;
	}



	// now lvl < size - 1

	if (O->get_E(current_extension)->get_type() == EXTENSION_TYPE_FUSION) {
		int next_node;

		if (f_v4) {
			cout << "orbit_tracer::recognize_recursion at ";
			cout << "(" << lvl << "/" << node << "/"
					<< current_extension << ")";
			cout << " fusion node " << O->get_node() << endl;
		}
		next_node = O->apply_isomorphism(PC,
			lvl, current_node,
			current_extension, size - 1 /* len */,
			FALSE /* f_tolerant */,
			Elt1, Elt2,
			verbose_level - 6);

		if (f_v) {
			cout << "orbit_tracer::recognize_recursion "
					"lvl " << lvl << " at ";
			cout << "(" << lvl << "/" << node << "/"
					<< current_extension << ")";
			cout << " fusion from " << O->get_node() << " to "
					<< next_node << endl;
		}
		if (next_node == -1) {
			cout << "orbit_tracer::recognize_recursion "
					"next_node == -1" << endl;
			exit(1);
		}
		if (f_v5) {
			cout << "orbit_tracer::recognize_recursion at ";
			cout << "(" << lvl << "/" << node << "/"
					<< current_extension << ")";
			cout << " after apply_isomorphism, "
					"next_node=" << next_node << endl;
		}
#if 0
		if (next_node < path[lvl + 1]) {
			if (f_v) {
				cout << "orbit_tracer::recognize_recursion "
						"lvl " << lvl << " not canonical" << endl;
				cout << "next_node=" << next_node << endl;
				//cout << "path[lvl + 1]=" << path[lvl + 1] << endl;
			}
			return not_canonical;
		}
#endif



		recognize_recursion(
			size, //f_implicit_fusion,
			lvl + 1, next_node, final_node,
			verbose_level);

		return;

	}
	else if (O->get_E(current_extension)->get_type() == EXTENSION_TYPE_EXTENSION) {
		int next_node;

		if (f_v4) {
			cout << "orbit_tracer::recognize_recursion "
					"extension node" << endl;
		}
		next_node = O->get_E(current_extension)->get_data();
		if (f_v) {
			cout << "orbit_tracer::recognize_recursion at ";
			cout << "(" << lvl << "/" << node << "/"
					<< current_extension << ")";
			cout << " extension from " << O->get_node() << " to "
					<< next_node << endl;
		}

		recognize_recursion(
			size, //f_implicit_fusion,
			lvl + 1, next_node, final_node,
			verbose_level);

		return;
	}
	else if (O->get_E(current_extension)->get_type() == EXTENSION_TYPE_UNPROCESSED) {
		cout << "orbit_tracer::recognize_recursion "
				"unprocessed node, "
				"this should not happen" << endl;
		exit(1);
	}
	else if (O->get_E(current_extension)->get_type() == EXTENSION_TYPE_PROCESSING) {
		cout << "orbit_tracer::recognize_recursion "
				"processing node, "
				"this should not happen" << endl;
		exit(1);
	}
	cout << "orbit_tracer::recognize_recursion "
			"unknown type of extension" << endl;
	exit(1);
}

void orbit_tracer::recognize(
	long int *the_set, int size, int *transporter,
	//int f_implicit_fusion,
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
// The input set the_set[] is not modified.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_vvv) {
		cout << "orbit_tracer::recognize" << endl;
	}
	// put in default values just in case we are doing a
	// tolerant search and do not have a final result
	final_node = -1;

	Lint_vec_copy(the_set, Set[0], size);

	PC->get_poset()->A->Group_element->element_one(Transporter->ith(0), 0);

	if (f_vv) {
		Lint_vec_print(cout, Set[0], size);
		cout << endl;
		if (PC->get_poset()->f_print_function) {
			(*PC->get_poset()->print_function)(cout, size, Set[0],
					PC->get_poset()->print_function_data);
		}
	}
	if (size > PC->get_sz()) {
		cout << "orbit_tracer::recognize size > sz" << endl;
		cout << "size=" << size << endl;
		cout << "PC->get_sz()=" << PC->get_sz() << endl;
		exit(1);
	}

	//nb_times_trace++;


	Lint_vec_copy(Set[0], PC->get_Poo()->set0, size);

	recognize_recursion(
		size, //f_implicit_fusion,
		0, 0,  // start from the very first node
		final_node,
		verbose_level);

	if (f_v) {
		cout << "orbit_tracer::recognize "
				"after recognize_recursion, "
				"copying transporter" << endl;
	}


	PC->get_poset()->A->Group_element->element_move(
			Transporter->ith(size),
			transporter, 0);


	if (f_v) {
		cout << "orbit_tracer::recognize done" << endl;
	}
}


void orbit_tracer::identify(long int *data, int sz,
		int *transporter, int &orbit_at_level,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_implicit_fusion = FALSE;
	int final_node;

	if (f_v) {
		cout << "orbit_tracer::identify" << endl;
	}
	if (f_v) {
		cout << "orbit_tracer::identify identifying the set ";
		Lint_vec_print(cout, data, sz);
		cout << endl;
	}

	if (f_v) {
		cout << "orbit_tracer::identify before recognize" << endl;
	}

	recognize(data, sz,
		transporter, //f_implicit_fusion,
		final_node,
		verbose_level - 2);

	if (f_v) {
		cout << "orbit_tracer::identify after recognize" << endl;
	}

	ring_theory::longinteger_object go;

	orbit_at_level = final_node - PC->get_Poo()->first_node_at_level(sz);
	PC->get_stabilizer_order(sz, orbit_at_level, go);

	if (f_v) {
		cout << "orbit_tracer::identify trace returns "
				"final_node = " << final_node << " which is "
						"isomorphism type " << orbit_at_level
						<< " with ago=" << go << endl;
	}
	if (f_v) {
		cout << "orbit_tracer::identify transporter:" << endl;
		PC->get_poset()->A->Group_element->element_print_quick(transporter, cout);
	}

	if (f_v) {
		cout << "orbit_tracer::identify done" << endl;
	}

}



}}}


