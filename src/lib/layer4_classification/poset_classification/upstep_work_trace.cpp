// upstep_work_trace.cpp
//
// Anton Betten
//
// moved out of upstep_work.cpp: Dec 20, 2011

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


trace_result upstep_work::recognize(
	int &final_node, int &final_ex, int f_tolerant,
	int verbose_level)
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
// a transversal for H in the stabilizer of
// S[0,...,len] results,
// where H is the point stabilizer of S[len]
// in the set-stabilizer of S[0,...,len-1],
// (which is a subgroup of S[0,...,len]).  
{
	trace_result r;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int len = size - 1;
	data_structures::sorting Sorting;
	
	if (f_v) {
		print_level_extension_coset_info();
		cout << "upstep_work::recognize "
				"verbose_level = " << verbose_level << endl;
		}
	// put in default values just in case we are doing a 
	// tolerant search and do not have a final result
	final_node = -1;
	final_ex = -1;
	
	if (f_vv) {
		print_level_extension_coset_info();
		Lint_vec_print(cout, gen->get_set_i(0), len + 1);
		cout << endl;
		if (gen->get_poset()->f_print_function) {
			gen->get_poset()->invoke_print_function(cout, size, gen->get_set_i(0));
		}
	}
#if 0
	if (len >= gen->sz) {
		cout << "upstep_work::recognize "
				"len >= sz, "
				"gen->nb_times_trace="
				<< gen->nb_times_trace << endl;
		cout << "len=" << len << endl;
		cout << "gen->sz=" << gen->sz << endl;
		exit(1);
	}
#endif
	//gen->nb_times_trace++;

#if 0
	if ((gen->nb_times_trace & ((1 << 17) - 1)) == 0) {
		cout << "upstep_work::recognize() " << endl;
		cout << "nb_times_trace=" << gen->nb_times_trace << endl;
		cout << "nb_times_trace_was_saved="
				<< gen->nb_times_trace_was_saved << endl;
	}
#endif
	
	Lint_vec_copy(gen->get_set_i(0), gen->get_set0(), size);
	Sorting.lint_vec_heapsort(gen->get_set0(), size - 1);
		// important: we keep the last point separate
	
	if (f_v) {
		print_level_extension_coset_info();
		cout << "upstep_work::recognize before recognize_recursion" << endl;
	}
	r = recognize_recursion(
		0, 0,  // start from the very first node
		final_node, final_ex, 
		f_tolerant, 
		verbose_level);
	if (f_v) {
		print_level_extension_coset_info();
		cout << "upstep_work::recognize after recognize_recursion" << endl;
	}
	if (f_v) {
		cout << "upstep_work::recognize after recognize_recursion" << endl;
		print_level_extension_coset_info();
		cout << " returning " << trace_result_as_text(r) << endl;
	}
	return r;
}

trace_result upstep_work::recognize_recursion(
	int lvl, int current_node,
	int &final_node, int &final_ex,
	int f_tolerant,
	int verbose_level)
// this routine is called from
// upstep_work::recognize
// we are dealing with a set of size len + 1.
// but we can only trace the first len points.
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
	int len = size - 1;
	int f_failure_to_find_point;
	
	poset_orbit_node *O;

	O = gen->get_node(current_node);
	if (f_vvv) {
		print_level_extension_coset_info();
		cout << "upstep_work::recognize_recursion"
			<< " lvl = " << lvl 
			<< " current_node = " << current_node 
			<< " verbose_level = " << verbose_level 
			<< endl;
		cout << "node=" << O->get_node() << " prev="
				<< O->get_prev() << " pt=" << O->get_pt() << endl;
		Lint_vec_set_print(cout, gen->get_set_i(lvl), size);
		cout << endl;
	}
	if (current_node < path[lvl]) {
		print_level_extension_coset_info();
		cout << "upstep_work::recognize_recursion: "
				"not canonical" << endl;
		cout << "current_node=" << current_node << endl;
		cout << "path[lvl]=" << path[lvl] << endl;
		return not_canonical;
	}
	if (f_v4) {
		if (gen->get_poset()->f_print_function) {
			gen->get_poset()->invoke_print_function(cout, size, gen->get_set_i(lvl));
		}
	}
	
	if (f_debug) {
		if (!O->check_node_and_set_consistency(gen,
				lvl - 1, gen->get_set_i(lvl))) {
			print_level_extension_coset_info();
			cout << "upstep_work::recognize_recursion: "
					"node and set inconsistent, "
					"the node corresponds to" << endl;
			O->store_set_to(gen, lvl - 1, gen->get_set3());
			Lint_vec_set_print(cout, gen->get_set3(), lvl);
			cout << endl;
			exit(1);
		}
	}
	
	if (lvl == 0 && gen->has_base_case()) {
		if (f_v4) {
			print_level_extension_coset_info();
			cout << "upstep_work::recognize_recursion "
					"special handling because of starter" << endl;
			}
		long int *cur_set = gen->get_set_i(0);
		long int *next_set =
				gen->get_set_i(0 + gen->get_Base_case()->size);
		int *cur_transporter =
				gen->get_transporter()->ith(0);
		int *next_transporter =
				gen->get_transporter()->ith(0 + gen->get_Base_case()->size);
		
		O->trace_starter(gen,
			size,
			cur_set,
			next_set,
			cur_transporter,
			next_transporter,
			0 /*verbose_level */);

		if (f_v) {
			print_level_extension_coset_info();
			cout << "upstep_work::recognize_recursion after trace_starter, "
					"calling find_automorphism_by_tracing_recursion for node "
					<< gen->get_Base_case()->size << endl;
		}
		trace_result r;

		r = recognize_recursion(
			gen->get_Base_case()->size,
			gen->get_Base_case()->size,
			final_node,
			final_ex,
			f_tolerant,
			verbose_level);
		if (f_v) {
			print_level_extension_coset_info();
			cout << "upstep_work::recognize_recursion "
					"after trace_starter, "
					"after recognize_recursion for node "
					<< gen->get_Base_case()->size << endl;
		}
		return r;
	}
	
	if (f_v4) {
		print_level_extension_coset_info();
		cout << "upstep_work::recognize_recursion lvl=" << lvl
				<< " current_node=" << current_node
				<< " calling O->trace_next_point_wrapper" << endl;
	}
	if (!O->trace_next_point_wrapper(
		gen,
		lvl,
		current_node,
		len,
		f_implicit_fusion,
		Elt3,
		f_failure_to_find_point,
		verbose_level - 5)) {

		// false in trace_next_point_wrapper can
		// only happen if f_implicit_fusion is true.
		
		
		if (f_v) {
			print_level_extension_coset_info();
			cout << "upstep_work::recognize_"
					"recursion trace_next_point_wrapper returns false, "
					"starting over" << endl;
		}
		trace_result r;

		r = start_over(
			lvl,
			current_node,
			final_node,
			final_ex,
			f_tolerant,
			verbose_level);
		if (f_v) {
			print_level_extension_coset_info();
			cout << "upstep_work::recognize_after start_over" << endl;
		}
		return r;
	}

	if (f_v4) {
		cout << "upstep_work::recognize_recursion "
				"after trace_next_point_wrapper" << endl;
	}
	
	if (f_failure_to_find_point) {
		if (f_v) {
			cout << "upstep_work::recognize_recursion "
					"failure to find point" << endl;
			}
		return no_result_extension_not_found;
	}

	pt0 = gen->get_set_i(lvl + 1)[lvl];
	current_extension = O->find_extension_from_point(gen, pt0, false);
	
	if (current_extension == -1) {

		cout << "upstep_work::recognize_recursion "
				"failure in find_extension_from_point" << endl;
		cout << "the original set is" << endl;
		Lint_vec_set_print(cout, gen->get_set_i(0), len + 1);
		cout << endl;
		//if (gen->f_print_function) {
			//(*gen->print_function)(cout, len + 1, gen->set[0],
			// gen->print_function_data);
			//}
		cout << "the current set is" << endl;
		Lint_vec_set_print(cout, gen->get_set_i(lvl + 1), len + 1);
		cout << endl;
		//if (gen->f_print_function) {
			//(*gen->print_function)(cout, len + 1, gen->set[lvl + 1],
			// gen->print_function_data);
			//}
		cout << "the node corresponds to" << endl;
		O->store_set_to(gen, lvl - 1, gen->get_set3());
		Lint_vec_set_print(cout, gen->get_set3(), lvl);
		cout << endl;

		cout << "lvl = " << lvl << endl;
		cout << "current_node = " << current_node << endl;

		cout << "pt0=" << pt0 << endl;
		cout << "gen->set[0][lvl]=" << gen->get_set_i(0)[lvl] << endl;

		if (current_node == path[lvl]
				&& pt0 < gen->get_set_i(0)[lvl]) {
			cout << "since pt0=" << pt0 << " is less than '"
					"gen->set[0][lvl]=" << gen->get_set_i(0)[lvl]
					<< ", we return not_canonical" << endl;
			return not_canonical;
		}
#if 0
		if (gen->f_print_function) {
			(*gen->print_function)(cout, lvl,
					gen->set3, gen->print_function_data);
		}
		gen->print_tree();
		exit(1);
#else
		// it is possible that we trace a set that is not admissible.
		// This can happen in the set stabilizer routine.
		// The set stabilizer routine does not know which sets are admissible.
		// It reduces set-orbits to those that appear in the given set 
		// to be stabilized. 
		// So, when computing the next level, it considers all extensions 
		// and hence may come across a set that is not admissible.
		// Returning no_result suffices, since we are not really interested 
		// in the full automorphism group of the set in this case.
		return no_result_extension_not_found;
#endif
	}



	if (f_v5) {
		cout << "upstep_work::recognize_recursion "
				"point " << pt0 << " is extension no "
				<< current_extension << endl;
	}
	if (gen->allowed_to_show_group_elements() && f_v4) {
		cout << "upstep_work::recognize_recursion "
				"transporter element:" << endl;
		int *transporter = gen->get_transporter()->ith(lvl + 1);
		gen->get_A2()->Group_element->element_print_quick(transporter, cout);
		//gen->A2->element_print_as_permutation(transporter, cout);
		cout << endl;
	}
	
	
	if (lvl == len) {
		if (f_v) {
			print_level_extension_coset_info();
			cout << "upstep_work::recognize_recursion "
					"calling handle_last_level" << endl;
			cout << "The element \\alpha is equal to " << endl;
			gen->get_A2()->Group_element->element_print_quick(
					gen->get_transporter()->ith(lvl + 1), cout);
		}
		trace_result r;
		if (f_v) {
			print_level_extension_coset_info();
			cout << "upstep_work::recognize_recursion "
					"before handle_last_level" << endl;
		}
		r = handle_last_level(
			lvl,
			current_node,
			current_extension,
			pt0,
			final_node,
			final_ex,
			verbose_level);
		if (f_v) {
			print_level_extension_coset_info();
			cout << "upstep_work::recognize_recursion "
					"after handle_last_level" << endl;
		}
		return r;
	}
	
	// now lvl < len
	
	if (O->get_E(current_extension)->get_type() == EXTENSION_TYPE_FUSION) {
		int next_node;
		
		if (f_v4) {
			print_level_extension_coset_info();
			cout << "upstep_work::find_automorphism_by_"
					"tracing_recursion at ";
			cout << "(" << current_node
					<< "/" << current_extension << ")";
			cout << " fusion node " << O->get_node() << endl;
		}
		next_node = O->apply_isomorphism(
			gen,
			lvl,
			current_node,
			current_extension,
			len,
			f_tolerant,
			Elt1, Elt2,
			verbose_level - 6);
		
		if (f_v) {
			print_level_extension_coset_info();
			cout << "upstep_work::find_automorphism_by_"
					"tracing_recursion lvl "
					<< lvl << " at ";
			cout << "(" << current_node
					<< "/" << current_extension << ")";
			cout << " fusion from " << O->get_node()
					<< " to " << next_node << endl;
		}
		if (next_node == -1) {
			cout << "We return no_result_extension_not_found" << endl;
			return no_result_extension_not_found;
		}
		if (f_v5) {
			print_level_extension_coset_info();
			cout << "upstep_work::find_automorphism_by_"
					"tracing_recursion at ";
			cout << "(" << current_node << "/"
					<< current_extension << ")";
			cout << " after apply_isomorphism, "
					"next_node=" << next_node << endl;
		}
		if (next_node < path[lvl + 1]) {
			if (f_v) {
				print_level_extension_coset_info();
				cout << "upstep_work::find_automorphism_by_"
						"tracing_recursion lvl " << lvl
						<< " not canonical" << endl;
				cout << "next_node=" << next_node << endl;
				cout << "path[lvl + 1]=" << path[lvl + 1] << endl;
			}
			return not_canonical;
		}
		
		if (f_v) {
			print_level_extension_coset_info();
			cout << "upstep_work::recognize_recursion "
					"calling recognize_recursion" << endl;
		}
		trace_result r;

		r = recognize_recursion(
			lvl + 1,
			next_node,
			final_node,
			final_ex,
			f_tolerant,
			verbose_level);

		if (f_v) {
			print_level_extension_coset_info();
			cout << "upstep_work::recognize_recursion "
					"after recognize_recursion" << endl;
		}
		return r;

	}
	else if (O->get_E(current_extension)->get_type() == EXTENSION_TYPE_EXTENSION) {
		int next_node;
		
		if (f_v4) {
			cout << "extension node" << endl;
		}
		next_node = O->get_E(current_extension)->get_data();
		if (f_v) {
			print_level_extension_coset_info();
			cout << "upstep_work::find_automorphism_by_"
					"tracing_recursion at ";
			cout << "(" << current_node << "/" << current_extension << ")";
			cout << " extension from " << O->get_node() << " to "
					<< next_node << endl;
		}
		if (f_v) {
			cout << "upstep_work::recognize_recursion "
					"calling recognize_recursion" << endl;
		}
		trace_result r;
		r = recognize_recursion(
			lvl + 1,
			next_node,
			final_node,
			final_ex,
			f_tolerant,
			verbose_level);
		if (f_v) {
			print_level_extension_coset_info();
			cout << "upstep_work::recognize_recursion "
					"after recognize_recursion" << endl;
		}
		return r;
	}
	else if (O->get_E(current_extension)->get_type() == EXTENSION_TYPE_UNPROCESSED) {
		cout << "unprocessed node at level len, "
				"this should not happen" << endl;
		exit(1);
	}
	else if (O->get_E(current_extension)->get_type() == EXTENSION_TYPE_PROCESSING) {
		cout << "processing node at level len, "
				"this should not happen" << endl;
		exit(1);
	}
	cout << "unknown type of extension" << endl;
	exit(1);
}

trace_result upstep_work::handle_last_level(
	int lvl, int current_node,
	int current_extension, int pt0,
	int &final_node, int &final_ex,  
	int verbose_level)
// called from poset_orbit_node::recognize_recursion
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int next_node;
	int my_current_node;

	poset_orbit_node *O = gen->get_node(current_node);

	if (f_v) {
		print_level_extension_coset_info();
		cout << "upstep_work::handle_last_level lvl=" << lvl 
			<< " node=" << O->get_node()
			<< " current_node=" << current_node 
			<< " current_extension=" << current_extension 
			<< " pt0=" << pt0 
			<< endl;
	}
	if (current_node < path[size - 1]) {
		if (f_v) {
			print_level_extension_coset_info();
			cout << "upstep_work::handle_last_level "
					"current_node=" << current_node
				<< " < " << path[size - 1]
				<< ", i.e. not canonical" << endl;
		}
		return not_canonical;
	}
	//my_current_node = gen->root[my_node].E[my_extension].data;
	my_current_node = cur;
	if (f_v) {
		print_level_extension_coset_info();
		cout << "upstep_work::handle_last_level "
				"my_current_node=" << my_current_node << endl;
	}
	
	if (O->get_E(current_extension)->get_type() == EXTENSION_TYPE_UNPROCESSED) {
		if (f_vv) {
			print_level_extension_coset_info();
			cout << "upstep_work::handle_last_level "
					"calling install_fusion_node at ";
			cout << "(" << current_node << "/"
					<< current_extension << ")" << endl;
		}

		O->install_fusion_node(
			gen,
			lvl,
			current_node,
			prev,
			prev_ex,
			coset,
			pt0,
			current_extension,
			f_debug,
			f_implicit_fusion,
			Elt1,
			verbose_level - 2);

		if (f_vv) {
			print_level_extension_coset_info();
			cout << "upstep_work::handle_last_level "
					"after install_fusion_node at ";
			cout << "(" << current_node << "/"
					<< current_extension << ")" << endl;
		}

		if (f_v) {
			print_level_extension_coset_info();
			cout << "install fusion node (" << current_node
					<< "/" << current_extension << ") -> ("
					<< prev << "/" << prev_ex << ")" << endl;
		}
		if (f_vv) {
			print_level_extension_coset_info();
			cout << "upstep_work::handle_last_level "
					"install_fusion_node at ";
			cout << "(" << current_node << "/"
					<< current_extension << ")";
			cout << " done, returning no_result_fusion_"
					"node_installed" << endl;
		}
		final_node = current_node;
		final_ex = current_extension;

		return no_result_fusion_node_installed;
	}
	else if (O->get_E(current_extension)->get_type() == EXTENSION_TYPE_FUSION) {
		if (f_vv) {
			print_level_extension_coset_info();
			cout << "upstep_work::handle_last_level at ";
			cout << "(" << current_node << "/" << current_extension
					<< ") fusion node already installed, "
					"returning no_result" << endl;
		}
		final_node = current_node;
		final_ex = current_extension;
		// fusion node is already installed
		return no_result_fusion_node_already_installed;

	}
	else if (O->get_E(current_extension)->get_type() == EXTENSION_TYPE_PROCESSING) {
		if (f_v) {
			print_level_extension_coset_info();
			cout << "upstep_work::handle_last_level: at ";
			cout << "(" << current_node << "/"
					<< current_extension << ")";
			cout << " extension node is current node, "
					"i.e. found an automorphism" << endl;
			if (gen->allowed_to_show_group_elements() && f_vv) {
				gen->get_A()->Group_element->element_print_quick(
						gen->get_transporter()->ith(lvl + 1), cout);
				cout << endl;
			}
		}
		final_node = current_node;
		final_ex = current_extension;
		
		return found_automorphism;
	}
	else if (O->get_E(current_extension)->get_type() == EXTENSION_TYPE_EXTENSION) {
#if 1
		print_level_extension_coset_info();
		cout << "upstep_work::handle_last_level: at";
		cout << "(" << current_node << "/"
				<< current_extension << ")";
		cout << " extension node at level len, "
				"this should not happen" << endl;
		cout << "the original set is" << endl;
		Lint_vec_set_print(cout, gen->get_set_i(0), lvl + 1);
		cout << endl;
		cout << "the current set is" << endl;
		Lint_vec_set_print(cout, gen->get_set_i(lvl + 1), lvl + 1);
		cout << endl;
		cout << "the node corresponds to" << endl;
		O->store_set_to(gen, lvl - 1, gen->get_set3());
		Lint_vec_set_print(cout, gen->get_set3(), lvl);
		cout << endl;
		exit(1);
#else
		return no_result_extension_not_found; // A Betten Dec 17, 2011 !!!
#endif
	}
	else if (O->get_E(current_extension)->get_type() == EXTENSION_TYPE_NOT_CANONICAL) {
		print_level_extension_coset_info();
		cout << "upstep_work::handle_last_level "
				"reached EXTENSION_TYPE_NOT_CANONICAL, "
				"returning not_canonical" << endl;
		return not_canonical;
	}
	cout << "upstep_work::handle_last_level: "
			"fatal: unknown type of extension" << endl;
	exit(1);
}

trace_result upstep_work::start_over(
	int lvl, int current_node, 
	int &final_node, int &final_ex,
	int f_tolerant, int verbose_level)
// Called from poset_orbit_node::recognize_recursion
// when trace_next_point returns false
// This can happen only if f_implicit_fusion is true
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	data_structures::sorting Sorting;

	if (f_v) {
		print_level_extension_coset_info();
		cout << "upstep_work::start_over" << endl;
	}
	// this is needed if implicit fusion nodes are used:
	if (lvl == size - 1) {
		if (f_v) {
			print_level_extension_coset_info();
			cout << "upstep_work::start_over lvl == size - 1, "
					"so we return not_canonical" << endl;
		}
		final_node = current_node;
		final_ex = -1;
		return not_canonical;
	}


	Sorting.lint_vec_heapsort(gen->get_set_i(lvl + 1), size - 1);
		// we keep the last point (i.e., the (len + 1)-th) extra
	Lint_vec_copy(gen->get_set_i(lvl + 1), gen->get_set_i(0), size);

	if (f_vv) {
		Lint_vec_set_print(cout, gen->get_set_i(0), size);
		cout << endl;
	}
	gen->get_A()->Group_element->element_move(
		gen->get_transporter()->ith(lvl + 1),
		gen->get_transporter()->ith(0),
		false);

	trace_result r;
	if (f_v) {
		print_level_extension_coset_info();
		cout << "upstep_work::start_over "
				"before recognize_recursion" << endl;
	}
	r = recognize_recursion(
		0,
		0,
		final_node,
		final_ex,
		f_tolerant,
		verbose_level);
	if (f_v) {
		print_level_extension_coset_info();
		cout << "upstep_work::start_over "
				"after recognize_recursion" << endl;
	}
	if (f_v) {
		print_level_extension_coset_info();
		cout << "upstep_work::start_over done" << endl;
	}
	return r;
}

}}}


