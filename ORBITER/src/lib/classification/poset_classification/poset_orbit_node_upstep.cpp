// poset_orbit_node_upstep.C
//
// Anton Betten
// December 27, 2004
// July 23, 2007

#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

namespace orbiter {
namespace classification {

int poset_orbit_node::apply_isomorphism(poset_classification *gen,
	int lvl, int current_node, 
	int current_extension, int len, int f_tolerant,
	int verbose_level)
// returns next_node
{
	int f_v = (verbose_level >= 1);
	int next_node;
	int *set;

	if (f_v) {
		cout << "poset_orbit_node::apply_isomorphism" << endl;
		}

	//set = NEW_int(len + 1); // this call should be eliminated
	set = gen->tmp_set_apply_fusion;

	gen->Poset->A->element_retrieve(
			E[current_extension].data,
			gen->Elt1, 0);
		// A Betten March 18 2012, this was gen->A2 previously
	
	if (f_v) {
		cout << "poset_orbit_node::apply_isomorphism "
				"applying fusion element" << endl;
		if (gen->f_allowed_to_show_group_elements) {
			gen->Poset->A2->element_print_quick(gen->Elt1, cout);
			}
		cout << "in action " << gen->Poset->A2->label << ":" << endl;
		if (gen->f_allowed_to_show_group_elements) {
			gen->Poset->A2->element_print_as_permutation(gen->Elt1, cout);
			}
		cout << "to the set ";
		int_vec_print(cout, gen->set[lvl + 1], len + 1);
		cout << endl;
		}
	gen->Poset->A2->map_a_set(
			gen->set[lvl + 1],
			set,
			len + 1,
			gen->Elt1, 0);
	if (f_v) {
		cout << "poset_orbit_node::apply_isomorphism the set becomes: ";
		int_vec_print(cout, set, len + 1);
		cout << endl;
		}

	gen->Poset->A2->element_mult(
			gen->transporter->ith(lvl + 1),
			gen->Elt1, gen->Elt2, 0);
	if (f_v) {
		int_vec_print(cout,
				gen->set[lvl + 1], len + 1);
		cout << endl;
		}
	gen->Poset->A2->move(gen->Elt2,
			gen->transporter->ith(lvl + 1));

	if (gen->Poset->f_subspace_lattice) {
		next_node = gen->find_node_for_subspace_by_rank(
				set,
				lvl + 1,
				verbose_level - 1);

		int_vec_copy(set, gen->set[lvl + 1], len + 1);
		}
	else {
		int_vec_heapsort(set, lvl + 1);
		int_vec_copy(set, gen->set[lvl + 1], len + 1);
		if (f_v) {
			cout << "poset_orbit_node::apply_isomorphism after sorting: ";
			}
		if (f_v) {
			cout << "poset_orbit_node::apply_isomorphism "
					"calling find_poset_orbit_node_for_set: ";
			int_vec_print(cout, gen->set[lvl + 1], lvl + 1);
			cout << endl;
			}
		next_node = gen->find_poset_orbit_node_for_set(
				lvl + 1,
				gen->set[lvl + 1],
				f_tolerant, 0);
		}

	//FREE_int(set);
	if (f_v) {
		cout << "poset_orbit_node::apply_isomorphism the set ";
		int_vec_print(cout, gen->set[lvl + 1], lvl + 1);
		cout << " is node " << next_node << endl;
		}
	return next_node;
}

void poset_orbit_node::install_fusion_node(
	poset_classification *gen,
	int lvl, int current_node, 
	int my_node, int my_extension, int my_coset, 
	int pt0, int current_extension, 
	int f_debug, int f_implicit_fusion, 
	int verbose_level)
// Called from poset_orbit_node::handle_last_level
// current_node is the same as oracle::node !!!
// pt0 is the same as E[current_extension].pt !!!
{
	int f_v = (verbose_level >= 1);
	//int f_v10 = (verbose_level >= 10);
	int hdl, cmp;	
	
	if (f_v) {
		cout << "poset_orbit_node::install_fusion_node "
				" verbose_level = " << verbose_level <<
				"lvl=" << lvl
			<< " node=" << node 
			<< " current_node=" << current_node 
			<< " my_node=" << my_node 
			<< " current_extension=" << current_extension 
			<< " pt0=" << pt0 
			<< " E[current_extension].pt=" << E[current_extension].pt 
			<< endl;
		}
		
	if (E[current_extension].pt != pt0) {
		cout << "poset_orbit_node::install_fusion_node "
				"E[current_extension].pt != pt0" << endl;
		exit(1);
		}
	if (current_node != node) {
		cout << "poset_orbit_node::install_fusion_node "
				"current_node != node" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "poset_orbit_node::install_fusion_node: "
				"unprocessed extension, ";
		cout << "we will now install a fusion node at node " << node 
			<< " , extension " << current_extension << endl;
		}
	if (f_v) {
		cout << "poset_orbit_node::install_fusion_node" << endl;
		cout << "transporter[lvl + 1]=" << endl;
		gen->Poset->A->element_print_quick(
				gen->transporter->ith(lvl + 1), cout);
		gen->Poset->A2->element_print_as_permutation_verbose(
				gen->transporter->ith(lvl + 1), cout, 0);
	}
	gen->Poset->A->element_invert(
			gen->transporter->ith(lvl + 1),
			gen->Elt1, FALSE);
	if (f_v) {
		cout << "poset_orbit_node::install_fusion_node" << endl;
		cout << "transporter[lvl + 1]^-1=Elt1=" << endl;
		gen->Poset->A->element_print_quick(gen->Elt1, cout);
		gen->Poset->A2->element_print_as_permutation_verbose(
				gen->Elt1, cout, 0);
	}
	if (f_v) {
		cout << "poset_orbit_node::install_fusion_node: "
				"fusion element:" << endl;
		if (gen->f_allowed_to_show_group_elements) {
			gen->Poset->A->element_print_quick(gen->Elt1, cout);
			gen->Poset->A2->element_print_as_permutation(gen->Elt1, cout);
			cout << endl;
			}
		}
	hdl = gen->Poset->A->element_store(gen->Elt1, FALSE);
	//E[current_extension].type = EXTENSION_TYPE_FUSION;
	gen->change_extension_type(lvl,
			current_node, current_extension,
			EXTENSION_TYPE_FUSION,
			0/* verbose_level*/);
	E[current_extension].data = hdl;
	E[current_extension].data1 = my_node;
	E[current_extension].data2 = my_extension;
	if (f_v) {
		cout << "FUSION NODE at lvl " << lvl
				<< " node/extension=" << current_node
				<< "/" << current_extension << " pt=" << pt0
				<< " hdl=" << hdl << " to node/extension="
				<< E[current_extension].data1 /*my_node*/
				<< "/" << E[current_extension].data2 /*my_extension*/
				<< " : ";
		int_vec_print(cout, gen->set0, lvl + 1);
		cout << endl;
#if 0
		if (current_node == 9 && pt0 == 39371) {
			gen->A->element_print_quick(gen->Elt1, cout);
			gen->A2->element_print_as_permutation(gen->Elt1, cout);
			cout << endl;
			}
#endif
		//cout << "FUSION from=" << current_node << " / "
		//<< current_extension << " hdl=" << hdl << " to="
		// << my_node << "/" << my_extension << endl;
		}

	
	// we check it:
	store_set_to(gen, lvl - 1, gen->set1);
	gen->set1[lvl] = pt0;
			
#if 0
	if (node == my_node || f_v) {
		cout << "poset_orbit_node::install_fusion_node "
				"fusion element stored in Node " << node
				<< ", extension " << current_extension
				<< " my_node = " << my_node << endl;
		gen->A->element_print_verbose(gen->Elt1, cout);
		cout << endl;
		cout << "Node " << node << " fusion from ";
		int_set_print(cout, gen->set1, lvl + 1);
		cout << " to ";
		int_set_print(cout, gen->set0, lvl + 1);
		cout << endl;
		if (node == my_node) {
			exit(1);
			}
		}
#endif

	if (f_v) {
		cout << "poset_orbit_node::install_fusion_node set1=";
		int_vec_print(cout, gen->set1, lvl + 1);
		cout << endl;
		cout << "Elt1=" << endl;
		gen->Poset->A->element_print_quick(gen->Elt1, cout);
		cout << "before map_a_set" << endl;
	}
	gen->Poset->A2->map_a_set(
			gen->set1,
			gen->set3,
			lvl + 1,
			gen->Elt1, 0);
	if (f_v) {
		cout << "poset_orbit_node::install_fusion_node "
				"after map_a_set set3=";
		int_vec_print(cout, gen->set3, lvl + 1);
		cout << endl;
	}

	if (gen->Poset->f_subspace_lattice) {
		cmp = gen->Poset->VS->compare_subspaces_ranked(
				gen->set3, gen->set0, lvl + 1, verbose_level);
#if 0
		cmp = gen->F->compare_subspaces_ranked_with_unrank_function(
			gen->set3, gen->set0, lvl + 1, 
			gen->vector_space_dimension, 
			gen->unrank_point_func,
			gen->rank_point_data, 
			verbose_level);
#endif
		}
	else {
		int_vec_heapsort(gen->set3, lvl);
		cmp = int_vec_compare(gen->set3, gen->set0, lvl + 1);
		}


	if (cmp != 0) {
		cout << "poset_orbit_node::install_fusion_node "
				"something is wrong" << endl;
		cout << "comparing ";
		int_set_print(cout, gen->set3, lvl + 1);
		cout << " with ";
		int_set_print(cout, gen->set0, lvl + 1);
		cout << endl;
		if (gen->Poset->f_subspace_lattice) {
			int *v;
			int i;

			v = NEW_int(gen->Poset->VS->dimension);
			int_set_print(cout, gen->set3, lvl + 1);
			cout << " is " << endl;
			for (i = 0; i < lvl + 1; i++) {
				gen->unrank_point(v, gen->set3[i]);
				int_vec_print(cout, v, gen->Poset->VS->dimension);
				cout << endl;
				}
			int_set_print(cout, gen->set0, lvl + 1);
			cout << " is " << endl;
			for (i = 0; i < lvl + 1; i++) {
				gen->unrank_point(v, gen->set0[i]);
				int_vec_print(cout, v, gen->Poset->VS->dimension);
				cout << endl;
				}

			FREE_int(v);			
			}
		exit(1);
		}
	if (f_v) {
		cout << "poset_orbit_node::install_fusion_node done" << endl;
	}
}

int poset_orbit_node::trace_next_point_wrapper(
	poset_classification *gen,
	int lvl, int current_node,
	int len, int f_implicit_fusion,
	int &f_failure_to_find_point,
	int verbose_level)
// Called from upstep_work::recognize_recursion
// applies the permutation which maps the point with index lvl 
// (i.e. the lvl+1-st point) to its orbit representative.
// also maps all the other points under that permutation.
// we are dealing with a set of size len + 1
// returns FALSE if we are using implicit fusion
// nodes and the set becomes lexicographically
// less than before, in which case trace has to be restarted.
{
	int f_v = (verbose_level >= 1);
	int ret;
	
	if (f_v) {
		cout << "poset_orbit_node::trace_next_point_wrapper" << endl;
		}
	if (f_v) {
		cout << "poset_orbit_node::trace_next_point_wrapper "
				"before trace_next_point" << endl;
		}
	ret = trace_next_point(gen,
		lvl,
		current_node,
		len + 1,
		gen->set[lvl],
		gen->set[lvl + 1],
		gen->transporter->ith(lvl),
		gen->transporter->ith(lvl + 1),
		f_implicit_fusion,
		f_failure_to_find_point,
		verbose_level);
	if (f_v) {
		cout << "poset_orbit_node::trace_next_point_wrapper done" << endl;
		}
	return ret;
}

int poset_orbit_node::trace_next_point_in_place(
	poset_classification *gen,
	int lvl,
	int current_node,
	int size,
	int *cur_set,
	int *tmp_set,
	int *cur_transporter,
	int *tmp_transporter,
	int f_implicit_fusion,
	int &f_failure_to_find_point,
	int verbose_level)
// called by poset_classification::trace_set_recursion
{
	int ret;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "poset_orbit_node::trace_next_point_in_place" << endl;
		cout << "poset_orbit_node::trace_next_point_in_place "
				"verbose_level = " << verbose_level << endl;
		}
	ret = trace_next_point(gen,
		lvl,
		current_node,
		size,
		cur_set,
		tmp_set,
		cur_transporter,
		tmp_transporter,
		f_implicit_fusion,
		f_failure_to_find_point,
		verbose_level);
	if (f_v) {
		cout << "poset_orbit_node::trace_next_point_in_place, "
				"after trace_next_point" << endl;
		}
	int_vec_copy(tmp_set, cur_set, size);
	gen->Poset->A->element_move(tmp_transporter,
			cur_transporter, 0);
	if (f_v) {
		cout << "poset_orbit_node::trace_next_point_in_place done" << endl;
		}
	return ret;
}

void poset_orbit_node::trace_starter(
	poset_classification *gen, int size,
	int *cur_set, int *next_set,
	int *cur_transporter, int *next_transporter, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Elt;
	int i;

	if (f_v) {
		cout << "poset_orbit_node::trace_starter" << endl;
		cout << "set:" << endl;
		int_vec_print(cout, cur_set, size);
		cout << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	Elt = gen->starter_canonize_Elt;
	
	(*gen->starter_canonize)(cur_set, size, Elt, 
		gen->starter_canonize_data,
		verbose_level - 1);

	if (f_vv) {
		cout << "applying:" << endl;
		if (gen->f_allowed_to_show_group_elements) {
			gen->Poset->A2->element_print(Elt, cout);
			cout << endl;
			}
		}
		
	for (i = 0; i < size; i++) {
		next_set[i] = gen->Poset->A2->element_image_of(
				cur_set[i], Elt, FALSE);
		}

	gen->Poset->A->element_mult(cur_transporter,
			Elt,
			next_transporter,
			FALSE);

	if (f_v) {
		cout << "after canonize:" << endl;
		int_vec_print(cout, next_set, size);
		cout << endl;
		}
	if (f_v) {
		cout << "poset_orbit_node::trace_starter done" << endl;
		}
}


int poset_orbit_node::trace_next_point(
	poset_classification *gen,
	int lvl, int current_node, int size, 
	int *cur_set, int *next_set,
	int *cur_transporter, int *next_transporter, 
	int f_implicit_fusion, int &f_failure_to_find_point,
	int verbose_level)
// Called by poset_orbit_node::trace_next_point_wrapper
// and by poset_orbit_node::trace_next_point_in_place
// returns FALSE only if f_implicit_fusion is TRUE and
// the set becomes lexcographically less 
{
	int the_point, pt0, i;
	int *cosetrep;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	//int f_v10 = (verbose_level >= 10);
	int ret;
	
	f_failure_to_find_point = FALSE;
	the_point = cur_set[lvl];

	if (f_v) {
		cout << "poset_orbit_node::trace_next_point lvl = " << lvl
				<< " the_point=" << the_point << endl;
		cout << "poset_orbit_node::trace_next_point node=" << node
				<< " prev=" << prev << " pt=" << pt << endl;
		cout << "poset_orbit_node::trace_next_point verbose_level = "
				<< verbose_level << endl;
		}
	
	if (gen->Poset->f_subspace_lattice) {
		if (f_v) {
			cout << "poset_orbit_node::trace_next_point before "
					"orbit_representative_and_coset_rep_inv_"
					"subspace_action" << endl;
		}
		orbit_representative_and_coset_rep_inv_subspace_action(
			gen, lvl,
			the_point, pt0, cosetrep,
			verbose_level - 1);

			// poset_orbit_node_upstep_subspace_action.C
		if (f_v) {
			cout << "poset_orbit_node::trace_next_point lvl = " << lvl
					<< " the_point=" << the_point
					<< " is traced to " << pt0 << endl;
		}
	}
	else {
		if (f_v) {
			cout << "poset_orbit_node::trace_next_point "
					"before orbit_representative_and_coset_rep_inv" << endl;
		}
		if (!orbit_representative_and_coset_rep_inv(
			gen, lvl,
			the_point, pt0, cosetrep,
			verbose_level - 1)) {
			if (f_v) {
				cout << "poset_orbit_node::trace_next_point "
						"orbit_representative_"
						"and_coset_rep_inv returns FALSE, "
						"f_failure_to_find_point = TRUE" << endl;
				}
			f_failure_to_find_point = TRUE;
			return TRUE;
			}
		if (f_v) {
			cout << "poset_orbit_node::trace_next_point "
					"after orbit_representative_and_coset_rep_inv" << endl;
		}
		}
	if (f_v) {
		cout << "poset_orbit_node::trace_next_point lvl = " << lvl
				<< " mapping "
				<< the_point << "->" << pt0
				<< " under the element " << endl;
		gen->Poset->A2->element_print_quick(cosetrep, cout);
		cout << "in action " << gen->Poset->A2->label << endl;
		if (gen->f_allowed_to_show_group_elements) {
			gen->Poset->A2->element_print_as_permutation_verbose(
					cosetrep, cout, 0);
			cout << endl;
		}
	}
	if (pt0 == the_point) {
		if (f_v) {
			cout << "Since the image point is equal "
					"to the original point, "
					"we apply no element and copy the set "
					"and the transporter over:" << endl;
			}
		int_vec_copy(cur_set, next_set, size);
		gen->Poset->A2->element_move(cur_transporter, next_transporter, FALSE);
		}
	else {
		if (f_v) {
			cout << "poset_orbit_node::trace_next_point lvl = " << lvl
					<< " applying:" << endl;
			gen->Poset->A2->element_print_quick(cosetrep, cout);
			cout << "in action " << gen->Poset->A2->label << endl;
			if (gen->f_allowed_to_show_group_elements) {
				gen->Poset->A2->element_print_as_permutation_verbose(
						cosetrep, cout, 0);
				cout << endl;
				}
			cout << "cur_set: ";
			int_vec_print(cout, cur_set, size);
			cout << endl;
			}
		
		int_vec_copy(cur_set, next_set, lvl);
		next_set[lvl] = pt0;
		for (i = lvl + 1; i < size; i++) {
			if (f_v) {
				cout << "poset_orbit_node::trace_next_point lvl = " << lvl
						<< " mapping point " << i << " / " << size
						<< "cur_set[i]=" << cur_set[i] << endl;
			}
			next_set[i] = gen->Poset->A2->element_image_of(
					cur_set[i], cosetrep, 0 /*verbose_level*/);
			if (f_v) {
				cout << "poset_orbit_node::trace_next_point lvl = " << lvl
						<< " mapping point " << i << " / " << size
						<< "next_set[i]=" << next_set[i] << endl;
			}
			if (f_v) {
				cout << "poset_orbit_node::trace_next_point "
						"lvl = " << lvl << ": ";
				cout << "mapping " << i << "-th point: "
						<< cur_set[i] << "->" << next_set[i] << endl;
				}
			}
		if (f_v) {
			cout << "poset_orbit_node::trace_next_point next_set: ";
			int_vec_print(cout, next_set, size);
			cout << endl;
			}

		//gen->A->map_a_set(gen->set[lvl],
		// gen->set[lvl + 1], len + 1, cosetrep);

		//int_vec_sort(len, gen->set[lvl + 1]);
		// we keep the last point extra

#if 0
		if (f_v) {
			cout << "poset_orbit_node::trace_next_point "
					"before element_mult" << endl;
			cout << "cur_transporter:" << endl;
			gen->A->element_print_quick(cur_transporter, cout);
			cout << "cosetrep:" << endl;
			gen->A->element_print_quick(cosetrep, cout);
			}
#endif
		gen->Poset->A->element_mult(cur_transporter,
				cosetrep, next_transporter, 0);
#if 0
		if (f_v) {
			cout << "poset_orbit_node::trace_next_point "
					"after element_mult" << endl;
			}
#endif
		}
	
	if (f_v) {
		cout << "poset_orbit_node::trace_next_point lvl = " << lvl
			<< " mapping " << the_point << "->" << pt0
			<< " done, the set becomes ";
		int_set_print(cout, next_set, size);
		cout << endl;

		if (gen->f_print_function && f_vv) {
			(*gen->print_function)(cout, size,
					next_set, gen->print_function_data);
			}
		if (gen->f_allowed_to_show_group_elements && f_vv) {
			cout << "poset_orbit_node::trace_next_point the n e w "
					"transporter is" << endl;
			gen->Poset->A2->element_print_quick(next_transporter, cout);
			gen->Poset->A2->element_print_as_permutation(
					next_transporter, cout);
			cout << endl;
			}
		
		}
	
	if (f_implicit_fusion) {
		// this is needed if implicit fusion nodes are used
	
		if (lvl > 0 &&
				next_set[lvl] < next_set[lvl - 1]) {
			if (f_v) {
				cout << "poset_orbit_node::trace_next_point the set becomes "
						"lexicographically less" << endl;
				}
			ret = FALSE;
			}
		else {
			ret = TRUE;
			}
		}
	else {
		ret = TRUE;
		}
	if (f_v) {
		cout << "poset_orbit_node::trace_next_point "
				"lvl = " << lvl
				<< " done, ret=" << ret << endl;
		}
	return ret;
}

int poset_orbit_node::orbit_representative_and_coset_rep_inv(
	poset_classification *gen,
	int lvl, int pt_to_trace,
	int &pt0, int *&cosetrep,
	int verbose_level)
// called by poset_orbit_node::trace_next_point
{
	int f_v = (verbose_level >= 1);
	//int f_check_image = FALSE;
	//int f_allow_failure = TRUE;

	if (f_v) {
		cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv "
				"lvl=" << lvl
				<< " pt_to_trace=" << pt_to_trace << endl;
		}
	if (nb_strong_generators == 0) {
		cosetrep = gen->Elt1;
		gen->Poset->A->element_one(gen->Elt1, FALSE);
		pt0 = pt_to_trace;
		return TRUE;
		}
	if (Schreier_vector) {

#if 0
		int f_trivial_group;
		
		if (nb_strong_generators) 
			f_trivial_group = FALSE;
		else 
			f_trivial_group = TRUE;
#endif

		//cout << "Node " << node
		//<< " poset_orbit_node::orbit_representative_and_"
		//"coset_rep_inv calling schreier_vector_coset_rep_inv" << endl;


		// in ACTION/schreier.C:
		if (f_v) {
			cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv "
					"before gen->Schreier_vector_handler->coset_rep_inv "
					"verbose_level=" << verbose_level
					<< endl;
		}
		if (!gen->Schreier_vector_handler->coset_rep_inv(
				Schreier_vector,
				pt_to_trace,
				pt0,
				verbose_level - 1)) {

#if 0
			gen->A2,
			sv, 
			hdl_strong_generators, 
			pt_to_trace, 
			pt0, 
			gen->Elt1, gen->Elt2, gen->Elt3, gen->Elt4, 
			f_trivial_group, 
			f_check_image, 
			f_allow_failure, 
			verbose_level - 1))
#endif
			if (f_v) {
				cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv "
						"schreier_vector_coset_rep_inv_general returns FALSE, "
						"point not found" << endl;
				}
			return FALSE;
			}
		cosetrep = gen->Schreier_vector_handler->cosetrep;
		if (f_v) {
			cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv "
					"after gen->Schreier_vector_handler->coset_rep_inv"
					<< endl;
		}

		// gen->Elt1 contains the element that maps pt_to_trace to pt0
		//cout << "Node " << node << " poset_orbit_node::orbit_representative_and_"
		//"coset_rep_inv schreier_vector_coset_rep_inv done" << endl;
		if (f_v) {
			cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv "
					"schreier vector: pt_to_trace=" << pt_to_trace
					<< " pt0=" << pt0 << endl;
			cout << "cosetrep:" << endl;
			gen->Poset->A2->element_print_quick(cosetrep, cout);
			}
		return TRUE;
		}
	else {
		if (f_v) {
			cout << "Node " << node
					<< " poset_orbit_node::orbit_representative_and_"
					"coset_rep_inv Schreier_vector not available" << endl;
			}
		}
	cosetrep = gen->Elt1;
	pt0 = gen->Poset->A2->least_image_of_point_generators_by_handle(
		nb_strong_generators, 
		hdl_strong_generators, 
		pt_to_trace, 
		gen->Elt1, 
		verbose_level - 1);
	if (f_v) {
		cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv "
				"pt_to_trace=" << pt_to_trace
				<< " pt0=" << pt0 <<  " done" << endl;
		}
	return TRUE;
}

}}


