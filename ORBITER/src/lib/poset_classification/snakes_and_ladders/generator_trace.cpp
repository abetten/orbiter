// generator_trace.C
//
// Anton Betten
//
// moved out of generator.C: Jan 21, 2010

#include "foundations/foundations.h"
#include "groups_and_group_actions/groups_and_group_actions.h"
#include "poset_classification/poset_classification.h"

INT generator::find_isomorphism(
		INT *set1, INT *set2, INT sz,
		INT *transporter, INT &orbit_idx,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *set1_canonical;
	INT *set2_canonical;
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;
	INT orb1;
	INT orb2;
	INT ret;

	if (f_v) {
		cout << "generator::find_isomorphism" << endl;
		}
	
	set1_canonical = NEW_INT(sz);
	set2_canonical = NEW_INT(sz);
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Elt3 = NEW_INT(A->elt_size_in_INT);

	orb1 = trace_set(set1, sz, sz, 
		set1_canonical, Elt1, 
		0 /* verbose_level */);

	orb2 = trace_set(set2, sz, sz, 
		set2_canonical, Elt2, 
		0 /* verbose_level */);

	if (orb1 == orb2) {
		ret = TRUE;
		A->element_invert(Elt2, Elt3, 0);
		A->element_mult(Elt1, Elt3, transporter, 0);
		orbit_idx = orb1;
		}
	else {
		orbit_idx = -1;
		ret = FALSE;
		}

	FREE_INT(set1_canonical);
	FREE_INT(set2_canonical);
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(Elt3);

	if (f_v) {
		cout << "generator::find_isomorphism done" << endl;
		}
	return ret;
}

set_and_stabilizer *generator::identify_and_get_stabilizer(
		INT *set, INT sz, INT *transporter,
		INT &orbit_at_level,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//set_and_stabilizer *SaS0;
	set_and_stabilizer *SaS;
	INT *Elt;

	if (f_v) {
		cout << "generator::identify_and_get_stabilizer" << endl;
		}
	if (f_v) {
		cout << "generator::identify_and_get_stabilizer "
				"identifying the set ";
		INT_vec_print(cout, set, sz);
		cout << endl;
		}
	Elt = NEW_INT(A->elt_size_in_INT);
	identify(set, sz, transporter,
			orbit_at_level, verbose_level - 2);

	SaS = get_set_and_stabilizer(sz,
			orbit_at_level, 0 /* verbose_level */);
	A->element_invert(transporter, Elt, 0);
	SaS->apply_to_self(Elt, 0 /* verbose_level */);

	if (f_v) {
		cout << "generator::identify_and_get_stabilizer "
				"input set=";
		INT_vec_print(cout, set, sz);
		cout << endl;
		cout << "generator::identify_and_get_stabilizer "
				"SaS->set=";
		INT_vec_print(cout, SaS->data, SaS->sz);
		cout << endl;
		}
	if (compare_sets(set, SaS->data, sz, SaS->sz)) {
		cout << "generator::identify_and_get_stabilizer "
				"the sets do not agree" << endl;
		exit(1);
		}
	
	FREE_INT(Elt);
	if (f_v) {
		cout << "generator::identify_and_get_stabilizer "
				"done" << endl;
		}
	return SaS;
}

void generator::identify(INT *data, INT sz,
		INT *transporter, INT &orbit_at_level,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT f_implicit_fusion = FALSE;
	INT final_node;

	if (f_v) {
		cout << "generator::identify" << endl;
		}
	if (f_v) {
		cout << "generator::identify identifying the set ";
		INT_vec_print(cout, data, sz);
		cout << endl;
		}

	if (f_v) {
		cout << "generator::identify before recognize" << endl;
		}

	recognize(data, sz,
		transporter, f_implicit_fusion,
		final_node,
		verbose_level);

	if (f_v) {
		cout << "generator::identify after recognize" << endl;
		}

	longinteger_object go;

	orbit_at_level = final_node - first_oracle_node_at_level[sz];
	get_stabilizer_order(sz, orbit_at_level, go);

	if (f_v) {
		cout << "generator::identify trace returns "
				"final_node = " << final_node << " which is "
						"isomorphism type " << orbit_at_level
						<< " with ago=" << go << endl;
		}
	if (f_v) {
		cout << "generator::identify transporter:" << endl;
		A->element_print_quick(transporter, cout);
		}

	if (f_v) {
		cout << "generator::identify done" << endl;
		}

}

void generator::test_identify(INT level, INT nb_times,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT *transporter;
	INT f_implicit_fusion = FALSE;
	INT final_node;
	INT *Elt;
	INT nb_orbits, cnt, r, r2;
	INT *set1;
	INT *set2;
	sims *S;
	longinteger_object go;

	if (f_v) {
		cout << "generator::test_identify, "
				"level = " << level
				<< " nb_times = " << nb_times << endl;
		}

	Elt = NEW_INT(A->elt_size_in_INT);
	transporter = NEW_INT(A->elt_size_in_INT);
	nb_orbits = nb_orbits_at_level(level);
	set1 = NEW_INT(level);
	set2 = NEW_INT(level);

	S = Strong_gens->create_sims(0 /*verbose_level*/);

	S->group_order(go);
	cout << "Group of order " << go << " created" << endl;




	for (cnt = 0; cnt < nb_times; cnt++) {
		r = random_integer(nb_orbits);
		if (f_v) {
			cout << "random orbit " << r << " / " << nb_orbits << endl;
			}
		get_set_by_level(level, r, set1);
		if (f_v) {
			cout << "random orbit " << r << " / "
					<< nb_orbits << " is represented by ";
			INT_vec_print(cout, set1, level);
			cout << endl;
			}
		A->random_element(S, Elt, 0 /* verbose_level */);
		A2->map_a_set_and_reorder(set1, set2, level, Elt,
				0 /* verbose_level */);
		cout << "mapped set is ";
		INT_vec_print(cout, set2, level);
		cout << endl;

		recognize(set2, level, transporter, f_implicit_fusion,
			final_node, verbose_level);
		
		r2 = final_node - first_oracle_node_at_level[level];
		if (r2 != r) {
			cout << "recognition fails" << endl;
			exit(1);
			}
		else {
			cout << "recognition is successful" << endl;
			}
		}

	delete S;
	FREE_INT(Elt);
	FREE_INT(transporter);
	FREE_INT(set1);
	FREE_INT(set2);
	if (f_v) {
		cout << "generator::test_identify done" << endl;
		}
}



#if 1
void generator::generator_apply_fusion_element_no_transporter(
	INT cur_level, INT size, INT cur_node, INT cur_ex, 
	INT *set_in, INT *set_out, 
	INT verbose_level)
// Called by upstep_work::handle_extension_fusion_type
{
	INT *Elt1;
	INT *Elt2;
	INT *set_tmp;
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "generator::generator_apply_fusion_element_"
				"no_transporter" << endl;
		}

	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	set_tmp = NEW_INT(size);
	A->element_one(Elt1, 0);

	generator_apply_fusion_element(cur_level, size, cur_node, cur_ex, 
		set_in, set_out, set_tmp, 
		Elt1, Elt2, 
		TRUE /* f_tolerant */, 
		0 /*verbose_level*/);

	#if 0
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "generator::generator_apply_fusion_element_"
				"no_transporter" << endl;
		}
	A->element_retrieve(
			root[current_node].E[current_extension].data, Elt1, 0);
	
	A2->map_a_set(S1, S2, len, Elt1, 0);

	INT_vec_heapsort(S2, len);

	//A->element_mult(gen->transporter->ith(lvl + 1),
	//Elt1, gen->transporter->ith(0), 0);
#endif
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(set_tmp);

	if (f_v) {
		cout << "generator::generator_apply_fusion_element_"
				"no_transporter done" << endl;
		}
}
#endif

#if 0
void generator::generator_apply_fusion_element(
		INT cur_level, INT cur_node, INT size, INT level,
	INT current_extension, 
	INT *canonical_set, INT *tmp_set, 
	INT *Elt_transporter, INT *tmp_Elt, 
	INT verbose_level)
{
	oracle *O = &root[cur_node];
	INT i;

	A->element_retrieve(O->E[current_extension].data, Elt1, 0);
	
	A2->map_a_set(canonical_set, tmp_set, size, Elt1, 0);

	
	INT_vec_heapsort(tmp_set, level); //INT_vec_sort(level, tmp_set);

	A->element_mult(Elt_transporter, Elt1, tmp_Elt, 0);

	for (i = 0; i < size; i++) {
		canonical_set[i] = tmp_set[i];
		}
	A->element_move(tmp_Elt, Elt_transporter, 0);

}
#endif

INT generator::generator_apply_fusion_element(INT level, INT size, 
	INT current_node, INT current_extension, 
	INT *set_in, INT *set_out, INT *set_tmp, 
	INT *transporter_in, INT *transporter_out, 
	INT f_tolerant, 
	INT verbose_level)
// returns next_node
{
	INT f_v = (verbose_level >= 1);
	INT next_node;
	oracle *O;

	O = &root[current_node];

	if (f_v) {
		cout << "generator::generator_apply_fusion_element "
				"current_node=" << current_node
				<< " current_extension=" << current_extension << endl;
		cout << "level=" << level << endl;		
		cout << "applying fusion element to the set ";
		INT_set_print(cout, set_in, size);
		cout << endl;
		}

	A2->element_retrieve(O->E[current_extension].data, Elt1, 0);
	
	if (f_v) {
		cout << "generator::generator_apply_fusion_element "
				"applying fusion element" << endl;
		A2->element_print_quick(Elt1, cout);
		cout << "in action " << A2->label << ":" << endl;
		A2->element_print_as_permutation(Elt1, cout);
		cout << "to the set ";
		INT_vec_print(cout, set_in, size);
		cout << endl;
		}
	A2->map_a_set(set_in, set_tmp, size, Elt1, 0);
	if (f_v) {
		cout << "generator::generator_apply_fusion_element "
				"the set becomes: ";
		INT_vec_print(cout, set_tmp, size);
		cout << endl;
		}

	A2->element_mult(transporter_in, Elt1, Elt2, 0);
	if (f_v) {
		INT_vec_print(cout, set_in, size);
		cout << endl;
		}
	A2->move(Elt2, transporter_out);

	if (f_on_subspaces) {
		next_node = find_node_for_subspace_by_rank(set_tmp,
				level + 1, verbose_level - 1);
		INT_vec_copy(set_tmp, set_out, size);
		}
	else {
		INT_vec_heapsort(set_tmp, level + 1);
		INT_vec_copy(set_tmp, set_out, size);
		if (f_v) {
			cout << "generator::generator_apply_fusion_element "
					"after sorting: ";
			}
		if (f_v) {
			cout << "generator::generator_apply_fusion_element "
					"calling find_oracle_node_for_set: ";
			INT_vec_print(cout, set_out, size);
			cout << endl;
			}

		next_node = find_oracle_node_for_set(level + 1 /*size*/,
				set_out, f_tolerant, 0);
		// changed A Betten 2/19/2011

		}
	if (f_v) {
		cout << "generator::generator_apply_fusion_element from ";
		INT_vec_print(cout, set_in, size);
		cout << " to ";
		INT_vec_print(cout, set_out, size);
		cout << ", which is node " << next_node << endl;
		cout << "we are done" << endl;
		}
	return next_node;
}


INT generator::trace_set_recursion(
	INT cur_level, INT cur_node,
	INT size, INT level,
	INT *canonical_set, INT *tmp_set1, INT *tmp_set2, 
	INT *Elt_transporter, INT *tmp_Elt1, 
	INT f_tolerant, 
	INT verbose_level)
// called by generator::trace_set
// returns the node in the generator
// that corresponds to the canonical_set
// or -1 if f_tolerant and the node could not be found
{
	INT f_v = (verbose_level >= 1);
	INT pt, pt0, current_extension, i, t, next_node;
	INT f_failure_to_find_point;
	oracle *O = &root[cur_node];
	
	if (f_v) {
		cout << "generator::trace_set_recursion "
				"cur_level = " << cur_level
				<< " cur_node = " << cur_node << " : ";
		INT_vec_print(cout, canonical_set, size);
		cout << endl;
		}
	pt = canonical_set[cur_level];
	if (f_v) {
		cout << "tracing point " << pt << endl;
		}
	if (!O->trace_next_point_in_place(this, 
		cur_level, cur_node, size, 
		canonical_set, tmp_set1,
		Elt_transporter, tmp_Elt1, 
		f_lex, 
		f_failure_to_find_point, 
		verbose_level - 1)) {
		
		if (f_v) {
			cout << "generator::trace_set_recursion "
					"cur_level = " << cur_level
					<< " cur_node = " << cur_node << " : ";
			cout << "O->trace_next_point_in_place returns FALSE, "
					"sorting and restarting" << endl;
			}
		// this can only happen if f_lex is TRUE
		// we need to sort and restart the trace:

		INT_vec_heapsort(canonical_set, cur_level + 1);
		
		
		return trace_set_recursion(0, 0, 
			size, level, 
			canonical_set, tmp_set1, tmp_set2, 
			Elt_transporter, tmp_Elt1, 
			f_tolerant, 
			verbose_level);
		}

	if (f_failure_to_find_point) {
		cout << "generator::trace_set_recursion: "
				"f_failure_to_find_point" << endl;
		exit(1);
		}
	pt0 = canonical_set[cur_level];
	if (f_v) {
		cout << "generator::trace_set_recursion "
				"cur_level = " << cur_level
				<< " cur_node = " << cur_node << " : ";
		INT_vec_print(cout, canonical_set, size);
		cout << " point " << pt
				<< " has been mapped to " << pt0 << endl;
		}
	current_extension = O->find_extension_from_point(
			this, pt0, FALSE);

	if (current_extension < 0) {
		cout << "generator::trace_set_recursion: "
				"did not find point" << endl;
		exit(1);
		}
	t = O->E[current_extension].type;
	if (t == EXTENSION_TYPE_EXTENSION) {
		// extension node
		next_node = O->E[current_extension].data;
		if (f_v) {
			cout << "generator::trace_set_recursion "
					"cur_level = " << cur_level
					<< " cur_node = " << cur_node << " : ";
			INT_vec_print(cout, canonical_set, size);
			cout << " point " << pt << " has been mapped to "
					<< pt0 << " next node is node " << next_node << endl;
			}
		if (cur_level + 1 == level) {
			return next_node;
			}
		else {
			return trace_set_recursion(cur_level + 1, next_node, 
				size, level, canonical_set, tmp_set1, tmp_set2,  
				Elt_transporter, tmp_Elt1, 
				f_tolerant, 
				verbose_level);
			}
		}
	else if (t == EXTENSION_TYPE_FUSION) {
		// fusion node

		if (f_v) {
			cout << "generator::trace_set_recursion "
					"cur_level = " << cur_level
					<< " cur_node = " << cur_node << " : ";
			cout << "before generator_apply_fusion_element" << endl;
			}
		next_node = generator_apply_fusion_element(cur_level, size, 
			cur_node, current_extension, 
			canonical_set, tmp_set1, tmp_set2, 
			Elt_transporter, tmp_Elt1, 
			f_tolerant, 
			verbose_level);
		if (f_v) {
			cout << "generator::trace_set_recursion "
					"cur_level = " << cur_level
					<< " cur_node = " << cur_node << " : ";
			cout << "after generator_apply_fusion_element" << endl;
			}
		if (f_v) {
			cout << "generator::trace_set_recursion "
					"cur_level = " << cur_level
				<< " cur_node = " << cur_node << " : " 
				<< " current_extension = " << current_extension 
				<< " : fusion from ";
			INT_vec_print(cout, canonical_set, size);
			cout << " to ";
			INT_vec_print(cout, tmp_set1, size);
			cout << " : we continue with node " << next_node << endl; 
			cout << endl;
			}

		if (next_node == -1) { // can only happen if f_tolerant is TRUE
			if (f_v) {
				cout << "generator::trace_set_recursion "
						"cur_level = " << cur_level
					<< " cur_node = " << cur_node << " : " 
					<< " current_extension = " << current_extension 
					<< " : fusion from ";
				INT_vec_print(cout, canonical_set, size);
				cout << " to ";
				INT_vec_print(cout, tmp_set1, size);
				cout << "we stop tracing" << endl;
				}
			return -1;
			}
		A->element_move(tmp_Elt1, Elt_transporter, 0);
		for (i = 0; i < size; i++) {
			canonical_set[i] = tmp_set1[i];
			}

		if (cur_level + 1 == level) {
			return next_node;
			}
		else {
			return trace_set_recursion(cur_level + 1, next_node, 
				size, level, canonical_set, tmp_set1, tmp_set2,  
				Elt_transporter, tmp_Elt1, 
				f_tolerant, 
				verbose_level);
			}
#if 0
		// we need to restart the trace:
		return trace_set_recursion(0, 0, 
			size, level, 
			canonical_set, tmp_set1, tmp_set2, 
			Elt_transporter, tmp_Elt1, 
			f_implicit_fusion, verbose_level);
#endif
		}
	cout << "generator::trace_set_recursion "
			"unknown type " << t << endl;
	exit(1);
}

INT generator::trace_set(INT *set, INT size, INT level, 
	INT *canonical_set, INT *Elt_transporter, 
	INT verbose_level)
// called by map_set_to_set_BLT in orbits.C
// returns the case number of the canonical set
{
	INT i, n, case_nb;
	INT f_v = (verbose_level >= 1);
	INT *tmp_set1, *tmp_set2;
	INT *tmp_Elt;

	tmp_set1 = NEW_INT(size);
	tmp_set2 = NEW_INT(size);
	tmp_Elt = NEW_INT(A->elt_size_in_INT);

	if (f_v) {
		cout << "generator::trace_set" << endl;
		cout << "tracing set ";
		INT_vec_print(cout, set, size);	
		cout << endl;
		cout << "level=" << level << endl;
		cout << "f_lex=" << f_lex << endl;
		}
	
	for (i = 0; i < size; i++) {
		canonical_set[i] = set[i];
		}
	A->element_one(Elt_transporter, 0);

	n = trace_set_recursion(
		0 /* cur_level */,
		0 /* cur_node */,  size, level,
		canonical_set, tmp_set1, tmp_set2, 
		Elt_transporter, tmp_Elt, 
		FALSE /*f_tolerant*/, 
		verbose_level);

	case_nb = n - first_oracle_node_at_level[level];

	if (case_nb < 0) {
		cout << "generator::trace_set, "
				"case_nb < 0, case_nb = " << case_nb << endl;
		exit(1);
		}
	FREE_INT(tmp_set1);
	FREE_INT(tmp_set2);
	FREE_INT(tmp_Elt);
	return case_nb;
}

INT generator::find_node_for_subspace_by_rank(
		INT *set, INT len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *v;
	INT *basis;
	INT *base_cols;
	INT rk, node, i, j, pt;

	if (f_v) {
		cout << "generator::find_node_for_subspace_by_rank for set ";
		INT_vec_print(cout, set, len);
		cout << endl;
		}
	v = tmp_find_node_for_subspace_by_rank1;
	basis = tmp_find_node_for_subspace_by_rank2;
	base_cols = tmp_find_node_for_subspace_by_rank3;
	//v = NEW_INT(vector_space_dimension);
	//basis = NEW_INT(len * vector_space_dimension);
	//base_cols = NEW_INT(vector_space_dimension);
	for (i = 0; i < len; i++) {
		unrank_point(basis + i * vector_space_dimension, set[i]);
		//(*unrank_point_func)(basis + i * vector_space_dimension,
		//set[i], rank_point_data);
		}
	rk = F->Gauss_simple(
			basis, len, vector_space_dimension,
			base_cols, 0 /* verbose_level */);
	if (rk != len) {
		cout << "generator::find_node_for_subspace_by_rank "
				"rk != len" << endl;
		exit(1);
		}
	node = 0;
	for (i = 0; i < len; i++) {
		oracle *O;

		O = &root[node];
		for (j = 0; j < O->nb_extensions; j++) {
			if (O->E[j].type != EXTENSION_TYPE_EXTENSION) {
				continue;
				}
			pt = O->E[j].pt;
			unrank_point(v, pt);
			if (!F->is_contained_in_subspace(len,
					vector_space_dimension, basis, base_cols,
				v, verbose_level)) {
				continue;
				}
			if (f_vv) {
				cout << "generator::find_node_for_subspace_by_rank "
						"at node " << node << " extension " << j
						<< " with point " << pt << " to node "
						<< O->E[j].data << endl;
				}
			node = O->E[j].data;
			set[i] = pt;
			break;
			}
		if (j == O->nb_extensions) {
			cout << "generator::find_node_for_subspace_by_rank "
					"at node " << node << " fatal, "
							"could not find extension" << endl;
			exit(1);
			}
		}
	if (f_v) {
		cout << "generator::find_node_for_subspace_by_rank "
				"the canonical set is ";
		INT_vec_print(cout, set, len);
		cout << " at node " << node << endl;
		}
	
	//FREE_INT(v);
	//FREE_INT(basis);
	//FREE_INT(base_cols);
	return node;
}

