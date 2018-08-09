// oracle_upstep.C
//
// Anton Betten
// December 27, 2004
// July 23, 2007

#include "GALOIS/galois.h"
#include "ACTION/action.h"
#include "SNAKES_AND_LADDERS/snakesandladders.h"

INT oracle::apply_fusion_element(generator *gen, 
	INT lvl, INT current_node, 
	INT current_extension, INT len, INT f_tolerant, INT verbose_level)
// returns next_node
{
	INT f_v = (verbose_level >= 1);
	INT next_node;
	INT *set;

	if (f_v) {
		cout << "oracle::apply_fusion_element" << endl;
		}

	set = NEW_INT(len + 1);
	//set = gen->tmp_set_apply_fusion;

	gen->A->element_retrieve(E[current_extension].data, gen->Elt1, 0);
		// A Betten March 18 2012, this was gen->A2 previously
	
	if (f_v) {
		cout << "oracle::apply_fusion_element "
				"applying fusion element" << endl;
		if (gen->f_allowed_to_show_group_elements) {
			gen->A2->element_print_quick(gen->Elt1, cout);
			}
		cout << "in action " << gen->A2->label << ":" << endl;
		if (gen->f_allowed_to_show_group_elements) {
			gen->A2->element_print_as_permutation(gen->Elt1, cout);
			}
		cout << "to the set ";
		INT_vec_print(cout, gen->set[lvl + 1], len + 1);
		cout << endl;
		}
	gen->A2->map_a_set(
			gen->set[lvl + 1],
			set /* gen->S0 */, len + 1,
			gen->Elt1, 0);
	if (f_v) {
		cout << "oracle::apply_fusion_element the set becomes: ";
		INT_vec_print(cout, set /* gen->S0 */, len + 1);
		cout << endl;
		}

	gen->A2->element_mult(
			gen->transporter->ith(lvl + 1),
			gen->Elt1, gen->Elt2, 0);
	if (f_v) {
		INT_vec_print(cout, gen->set[lvl + 1], len + 1);
		cout << endl;
		}
	gen->A2->move(gen->Elt2, gen->transporter->ith(lvl + 1));

	if (gen->f_on_subspaces) {
#if 0
		//rk = gen->F->Gauss_canonical_form_ranked(gen->S0,
		// gen->set[lvl + 1], lvl + 1,
		//	gen->vector_space_dimension, verbose_level);
		rk = gen->F->lexleast_canonical_form_ranked(gen->S0,
			gen->set[lvl + 1], lvl + 1,
			gen->vector_space_dimension, verbose_level - 2);
		if (f_v) {
			cout << "after lexleast_canonical_form_ranked" << endl;
			}
		for (ii = lvl + 1; ii < len + 1; ii++) {
			gen->set[lvl + 1][ii] = gen->S0[ii];
			}
		//INT_vec_heapsort(gen->set[lvl + 1], lvl + 1);
		// INT_vec_sort(lvl + 1, gen->S0);
		if (f_v) {
			cout << "oracle::apply_fusion_element "
					"after lexleast_canonical_form_ranked: ";
			}
#endif
		next_node = gen->find_node_for_subspace_by_rank(
				set /* gen->S0 */, lvl + 1, verbose_level - 1);

		INT_vec_copy(set, gen->set[lvl + 1], len + 1);
		}
	else {
		INT_vec_heapsort(set /* gen->S0 */, lvl + 1);
		INT_vec_copy(set, gen->set[lvl + 1], len + 1);
		if (f_v) {
			cout << "oracle::apply_fusion_element after sorting: ";
			}
		if (f_v) {
			cout << "oracle::apply_fusion_element "
					"calling find_oracle_node_for_set: ";
			INT_vec_print(cout, gen->set[lvl + 1], lvl + 1);
			cout << endl;
			}
		next_node = gen->find_oracle_node_for_set(
				lvl + 1, gen->set[lvl + 1], f_tolerant, 0);
		}

	FREE_INT(set);
	if (f_v) {
		cout << "oracle::apply_fusion_element the set ";
		INT_vec_print(cout, gen->set[lvl + 1], lvl + 1);
		cout << " is node " << next_node << endl;
		}
	return next_node;
}

void oracle::install_fusion_node(generator *gen, 
	INT lvl, INT current_node, 
	INT my_node, INT my_extension, INT my_coset, 
	INT pt0, INT current_extension, 
	INT f_debug, INT f_implicit_fusion, 
	INT verbose_level)
// Called from oracle::handle_last_level
// current_node is the same as oracle::node !!!
// pt0 is the same as E[current_extension].pt !!!
{
	INT f_v = (verbose_level >= 1);
	//INT f_v10 = (verbose_level >= 10);
	INT hdl, cmp;	
	
	if (f_v) {
		cout << "oracle::install_fusion_node "
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
		cout << "oracle::install_fusion_node E[current_extension].pt "
				"!= pt0" << endl;
		exit(1);
		}
	if (current_node != node) {
		cout << "oracle::install_fusion_node current_node != node" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "oracle::install_fusion_node: unprocessed extension, ";
		cout << "we will now install a fusion node at node " << node 
			<< " , extension " << current_extension << endl;
		}
	gen->A->element_invert(
			gen->transporter->ith(lvl + 1),
			gen->Elt1, FALSE);
	if (f_v) {
		cout << "oracle::install_fusion_node: fusion element:" << endl;
		if (gen->f_allowed_to_show_group_elements) {
			gen->A->element_print_quick(gen->Elt1, cout);
			gen->A2->element_print_as_permutation(gen->Elt1, cout);
			cout << endl;
			}
		}
	hdl = gen->A->element_store(gen->Elt1, FALSE);
	//E[current_extension].type = EXTENSION_TYPE_FUSION;
	gen->change_extension_type(lvl, current_node, current_extension,
			EXTENSION_TYPE_FUSION, 0/* verbose_level*/);
	E[current_extension].data = hdl;
	E[current_extension].data1 = my_node;
	E[current_extension].data2 = my_extension;
	if (f_v) {
		cout << "FUSION NODE at lvl " << lvl << " node/extension=" << current_node
				<< "/" << current_extension << " pt=" << pt0
				<< " hdl=" << hdl << " to node/extension="
				<< E[current_extension].data1 /*my_node*/
				<< "/" << E[current_extension].data2 /*my_extension*/
				<< " : ";
		INT_vec_print(cout, gen->set0, lvl + 1);
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
		cout << "fusion element stored in Node " << node
				<< ", extension " << current_extension
				<< " my_node = " << my_node << endl;
		gen->A->element_print_verbose(gen->Elt1, cout);
		cout << endl;
		cout << "Node " << node << " fusion from ";
		INT_set_print(cout, gen->set1, lvl + 1);
		cout << " to ";
		INT_set_print(cout, gen->set0, lvl + 1);
		cout << endl;
		if (node == my_node) {
			exit(1);
			}
		}
#endif

	gen->A2->map_a_set(gen->set1, gen->set3, lvl + 1, gen->Elt1, 0);

	if (gen->f_on_subspaces) {
		cmp = gen->F->compare_subspaces_ranked_with_unrank_function(
			gen->set3, gen->set0, lvl + 1, 
			gen->vector_space_dimension, 
			gen->unrank_point_func,
			gen->rank_point_data, 
			verbose_level);
		}
	else {
		INT_vec_heapsort(gen->set3, lvl);
		cmp = INT_vec_compare(gen->set3, gen->set0, lvl + 1);
		}


	if (cmp != 0) {
		cout << "oracle::install_fusion_node something is wrong" << endl;
		cout << "comparing ";
		INT_set_print(cout, gen->set3, lvl + 1);
		cout << " with ";
		INT_set_print(cout, gen->set0, lvl + 1);
		cout << endl;
		if (gen->f_on_subspaces) {
			INT *v;
			INT i;

			v = NEW_INT(gen->vector_space_dimension);
			INT_set_print(cout, gen->set3, lvl + 1);
			cout << " is " << endl;
			for (i = 0; i < lvl + 1; i++) {
				gen->unrank_point(v, gen->set3[i]);
				//(*gen->unrank_point_func)(v,
				// gen->set3[i], gen->rank_point_data);
				INT_vec_print(cout, v, gen->vector_space_dimension);
				cout << endl;
				}
			INT_set_print(cout, gen->set0, lvl + 1);
			cout << " is " << endl;
			for (i = 0; i < lvl + 1; i++) {
				gen->unrank_point(v, gen->set0[i]);
				//(*gen->unrank_point_func)(v, gen->set0[i],
				// gen->rank_point_data);
				INT_vec_print(cout, v, gen->vector_space_dimension);
				cout << endl;
				}

			FREE_INT(v);			
			}
		exit(1);
		}
}

INT oracle::trace_next_point_wrapper(generator *gen,
	INT lvl, INT current_node,
	INT len, INT f_implicit_fusion,
	INT &f_failure_to_find_point,
	INT verbose_level)
// Called from upstep_work::find_automorphism_by_tracing_recursion
// applies the permutation which maps the point with index lvl 
// (i.e. the lvl+1-st point) to its orbit representative.
// also maps all the other points under that permutation.
// we are dealing with a set of size len + 1
// returns FALSE if we are using implicit fusion
// nodes and the set becomes lexicographically
// less than before, in which case trace has to be restarted.
{
	INT f_v = (verbose_level >= 1);
	INT ret;
	
	if (f_v) {
		cout << "oracle::trace_next_point_wrapper" << endl;
		}
	ret = trace_next_point(gen, lvl, current_node, len + 1, 
		gen->set[lvl], gen->set[lvl + 1], 
		gen->transporter->ith(lvl), gen->transporter->ith(lvl + 1), 
		f_implicit_fusion, f_failure_to_find_point, verbose_level);
	if (f_v) {
		cout << "oracle::trace_next_point_wrapper done" << endl;
		}
	return ret;
}

INT oracle::trace_next_point_in_place(generator *gen, 
	INT lvl, INT current_node, INT size, 
	INT *cur_set, INT *tmp_set,
	INT *cur_transporter, INT *tmp_transporter, 
	INT f_implicit_fusion, INT &f_failure_to_find_point,
	INT verbose_level)
// called by generator::trace_set_recursion
{
	INT ret, i;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "oracle::trace_next_point_in_place" << endl;
		cout << "oracle::trace_next_point_in_place "
				"verbose_level = " << verbose_level << endl;
		}
	ret = trace_next_point(gen, lvl,
			current_node, size, cur_set, tmp_set,
		cur_transporter, tmp_transporter, f_implicit_fusion, 
		f_failure_to_find_point, verbose_level);
	if (f_v) {
		cout << "oracle::trace_next_point_in_place, "
				"after trace_next_point" << endl;
		}
	for (i = 0; i < size; i++) {
		cur_set[i] = tmp_set[i];
		}
	gen->A->element_move(tmp_transporter, cur_transporter, 0);
	if (f_v) {
		cout << "oracle::trace_next_point_in_place done" << endl;
		}
	return ret;
}

void oracle::trace_starter(generator *gen, INT size, 
	INT *cur_set, INT *next_set,
	INT *cur_transporter, INT *next_transporter, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *Elt;
	INT i;

	if (f_v) {
		cout << "oracle::trace_starter" << endl;
		cout << "set:" << endl;
		INT_vec_print(cout, cur_set, size);
		cout << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	Elt = gen->starter_canonize_Elt;
	
	(*gen->starter_canonize)(cur_set, size, Elt, 
		gen->starter_canonize_data, verbose_level - 1);

	if (f_vv) {
		cout << "applying:" << endl;
		if (gen->f_allowed_to_show_group_elements) {
			gen->A2->element_print(Elt, cout);
			cout << endl;
			}
		}
		
	for (i = 0; i < size; i++) {
		next_set[i] = gen->A2->element_image_of(cur_set[i], Elt, FALSE);
		}

	gen->A->element_mult(cur_transporter, Elt, next_transporter, FALSE);

	if (f_v) {
		cout << "after canonize:" << endl;
		INT_vec_print(cout, next_set, size);
		cout << endl;
		}
	if (f_v) {
		cout << "oracle::trace_starter done" << endl;
		}
}


INT oracle::trace_next_point(generator *gen, 
	INT lvl, INT current_node, INT size, 
	INT *cur_set, INT *next_set,
	INT *cur_transporter, INT *next_transporter, 
	INT f_implicit_fusion, INT &f_failure_to_find_point,
	INT verbose_level)
// Called by oracle::trace_next_point_wrapper 
// and by oracle::trace_next_point_in_place
// returns FALSE only if f_implicit_fusion is TRUE and
// the set becomes lexcographically less 
{
	INT the_point, pt0, i;
	INT *cosetrep;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT f_v10 = (verbose_level >= 10);
	INT ret;
	
	f_failure_to_find_point = FALSE;
	the_point = cur_set[lvl];

	if (f_v) {
		cout << "oracle::trace_next_point lvl = " << lvl
				<< " the_point=" << the_point << endl;
		cout << "oracle::trace_next_point node=" << node
				<< " prev=" << prev << " pt=" << pt << endl;
		cout << "oracle::trace_next_point verbose_level = "
				<< verbose_level << endl;
		}
	
	if (gen->f_on_subspaces) {
		orbit_representative_and_coset_rep_inv_subspace_action(gen, lvl, 
			the_point, pt0, cosetrep, verbose_level - 1);

			// oracle_upstep_subspace_action.C

		}
	else {
		if (!orbit_representative_and_coset_rep_inv(gen, lvl, 
			the_point, pt0, cosetrep, 0 /*verbose_level - 1*/)) {
			if (f_v) {
				cout << "oracle::trace_next_point orbit_representative_"
						"and_coset_rep_inv returns FALSE, "
						"f_failure_to_find_point = TRUE" << endl;
				}
			f_failure_to_find_point = TRUE;
			return TRUE;
			}
		}
	if (f_vv) {
		cout << "oracle::trace_next_point lvl = " << lvl << " mapping "
				<< the_point << "->" << pt0 << " under the element " << endl;
		if (gen->f_allowed_to_show_group_elements) {
			gen->A2->element_print_quick(cosetrep, cout);
			}
		cout << "in action " << gen->A2->label << endl;
		if (gen->f_allowed_to_show_group_elements)
		gen->A2->element_print_as_permutation_verbose(cosetrep, cout, 0);
		cout << endl;
		}
	if (pt0 == the_point) {
		if (f_vv) {
			cout << "Since the image point is equal to the original point, "
					"we apply no element and copy the set over:" << endl;
			}
		for (i = 0; i < size; i++) {
			next_set[i] = cur_set[i];
			}
		gen->A2->element_move(cur_transporter, next_transporter, FALSE);
		}
	else {
		if (f_vv) {
			cout << "oracle::trace_next_point lvl = " << lvl
					<< " applying:" << endl;
			if (gen->f_allowed_to_show_group_elements) {
				gen->A2->element_print_quick(cosetrep, cout);
				}
			cout << "in action " << gen->A2->label << endl;
			if (gen->f_allowed_to_show_group_elements) {
				gen->A2->element_print_as_permutation_verbose(
						cosetrep, cout, 0);
				cout << endl;
				}
			cout << "cur_set: ";
			INT_vec_print(cout, cur_set, size);
			cout << endl;
			}
		
		for (i = 0; i < lvl; i++) {
			next_set[i] = cur_set[i];
			}
		next_set[lvl] = pt0;
		for (i = lvl + 1; i < size; i++) {
			next_set[i] = gen->A2->element_image_of(
					cur_set[i], cosetrep, 0);
			if (f_vvv) {
				cout << "oracle::trace_next_point lvl = " << lvl << ": ";
				cout << "mapping " << i << "-th point: "
						<< cur_set[i] << "->" << next_set[i] << endl;
				}
			}
		if (f_v) {
			cout << "oracle::trace_next_point next_set: ";
			INT_vec_print(cout, next_set, size);
			cout << endl;
			}

		//gen->A->map_a_set(gen->set[lvl],
		// gen->set[lvl + 1], len + 1, cosetrep);

		//INT_vec_sort(len, gen->set[lvl + 1]);
		// we keep the last point extra

#if 0
		if (f_v) {
			cout << "oracle::trace_next_point "
					"before element_mult" << endl;
			cout << "cur_transporter:" << endl;
			gen->A->element_print_quick(cur_transporter, cout);
			cout << "cosetrep:" << endl;
			gen->A->element_print_quick(cosetrep, cout);
			}
#endif
		gen->A->element_mult(cur_transporter,
				cosetrep, next_transporter, 0);
#if 0
		if (f_v) {
			cout << "oracle::trace_next_point "
					"after element_mult" << endl;
			}
#endif
		}
	
	if (f_vv) {
		cout << "oracle::trace_next_point lvl = " << lvl 
			<< " mapping " << the_point << "->" << pt0
			<< " done, the set becomes ";
		INT_set_print(cout, next_set, size);
		cout << endl;
		if (gen->f_print_function && f_vvv) {
			(*gen->print_function)(size,
					next_set, gen->print_function_data);
			}
		if (gen->f_allowed_to_show_group_elements && f_v10) {
			cout << "oracle::trace_next_point the new "
					"transporter is" << endl;
			gen->A2->element_print_quick(next_transporter, cout);
			gen->A2->element_print_as_permutation(
					next_transporter, cout);
			cout << endl;
			}
		
		}
	
	if (f_implicit_fusion) {
		// this is needed if implicit fusion nodes are used
	
		if (lvl > 0 && next_set[lvl] < next_set[lvl - 1]) {
			if (f_v) {
				cout << "oracle::trace_next_point the set becomes "
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
		cout << "oracle::trace_next_point lvl = " << lvl
				<< " done, ret=" << ret << endl;
		}
	return ret;
}

INT oracle::orbit_representative_and_coset_rep_inv(
	generator *gen,
	INT lvl, INT pt_to_trace, INT &pt0, INT *&cosetrep,
	INT verbose_level)
// called by oracle::oracle::trace_next_point
{
	INT f_v = (verbose_level >= 1);
	INT f_check_image = FALSE;
	INT f_allow_failure = TRUE;

	if (f_v) {
		cout << "oracle::orbit_representative_and_coset_rep_inv "
				"lvl=" << lvl << " pt_to_trace=" << pt_to_trace << endl;
		}
	cosetrep = gen->Elt1;
	if (nb_strong_generators == 0) {
		gen->A->element_one(gen->Elt1, FALSE);
		pt0 = pt_to_trace;
		return TRUE;
		}
	if (sv) {
		INT f_trivial_group;
		
		if (nb_strong_generators) 
			f_trivial_group = FALSE;
		else 
			f_trivial_group = TRUE;
		//cout << "Node " << node << " oracle::orbit_representative_and_"
		//"coset_rep_inv calling schreier_vector_coset_rep_inv" << endl;


		// in ACTION/schreier.C:
		if (!schreier_vector_coset_rep_inv_general(gen->A2, 
			sv, 
			hdl_strong_generators, 
			pt_to_trace, 
			pt0, 
			gen->Elt1, gen->Elt2, gen->Elt3, gen->Elt4, 
			f_trivial_group, 
			//TRUE /* f_compact */, 
			f_check_image, 
			f_allow_failure, 
			verbose_level - 1)) {
			if (f_v) {
				cout << "oracle::orbit_representative_and_coset_rep_inv "
						"schreier_vector_coset_rep_inv_general returns FALSE, "
						"point not found" << endl;
				}
			return FALSE;
			}

		// gen->Elt1 contains the element that maps pt_to_trace to pt0
		//cout << "Node " << node << " oracle::orbit_representative_and_"
		//"coset_rep_inv schreier_vector_coset_rep_inv done" << endl;
		if (f_v) {
			cout << "oracle::orbit_representative_and_coset_rep_inv "
					"schreier vector: pt_to_trace=" << pt_to_trace
					<< " pt0=" << pt0 << endl;
			}
		return TRUE;
		}
	else {
		if (f_v) {
			cout << "Node " << node << " oracle::orbit_representative_and_"
					"coset_rep_inv sv not available" << endl;
			}
		}
	pt0 = gen->A2->least_image_of_point_generators_by_handle(
		nb_strong_generators, 
		hdl_strong_generators, 
		pt_to_trace, 
		gen->Elt1, 
		verbose_level - 1);
	if (f_v) {
		cout << "oracle::orbit_representative_and_coset_rep_inv "
				"pt_to_trace=" << pt_to_trace
				<< " pt0=" << pt0 <<  " done" << endl;
		}
	return TRUE;
}


