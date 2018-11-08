// poset_classification_trace.C
//
// Anton Betten
//
// moved out of poset_classification.C: Jan 21, 2010

#include "foundations/foundations.h"
#include "groups_and_group_actions/groups_and_group_actions.h"
#include "poset_classification/poset_classification.h"

int poset_classification::find_isomorphism(
		int *set1, int *set2, int sz,
		int *transporter, int &orbit_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *set1_canonical;
	int *set2_canonical;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int orb1;
	int orb2;
	int ret;

	if (f_v) {
		cout << "poset_classification::find_isomorphism" << endl;
		}
	
	set1_canonical = NEW_int(sz);
	set2_canonical = NEW_int(sz);
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

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

	FREE_int(set1_canonical);
	FREE_int(set2_canonical);
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

	if (f_v) {
		cout << "poset_classification::find_isomorphism done" << endl;
		}
	return ret;
}

set_and_stabilizer *poset_classification::identify_and_get_stabilizer(
		int *set, int sz, int *transporter,
		int &orbit_at_level,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//set_and_stabilizer *SaS0;
	set_and_stabilizer *SaS;
	int *Elt;

	if (f_v) {
		cout << "poset_classification::identify_and_get_stabilizer" << endl;
		}
	if (f_v) {
		cout << "poset_classification::identify_and_get_stabilizer "
				"identifying the set ";
		int_vec_print(cout, set, sz);
		cout << endl;
		}
	Elt = NEW_int(A->elt_size_in_int);
	identify(set, sz, transporter,
			orbit_at_level, verbose_level - 2);

	SaS = get_set_and_stabilizer(sz,
			orbit_at_level, 0 /* verbose_level */);
	A->element_invert(transporter, Elt, 0);
	SaS->apply_to_self(Elt, 0 /* verbose_level */);

	if (f_v) {
		cout << "poset_classification::identify_and_get_stabilizer "
				"input set=";
		int_vec_print(cout, set, sz);
		cout << endl;
		cout << "poset_classification::identify_and_get_stabilizer "
				"SaS->set=";
		int_vec_print(cout, SaS->data, SaS->sz);
		cout << endl;
		}
	if (compare_sets(set, SaS->data, sz, SaS->sz)) {
		cout << "poset_classification::identify_and_get_stabilizer "
				"the sets do not agree" << endl;
		exit(1);
		}
	
	FREE_int(Elt);
	if (f_v) {
		cout << "poset_classification::identify_and_get_stabilizer "
				"done" << endl;
		}
	return SaS;
}

void poset_classification::identify(int *data, int sz,
		int *transporter, int &orbit_at_level,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_implicit_fusion = FALSE;
	int final_node;

	if (f_v) {
		cout << "poset_classification::identify" << endl;
		}
	if (f_v) {
		cout << "poset_classification::identify identifying the set ";
		int_vec_print(cout, data, sz);
		cout << endl;
		}

	if (f_v) {
		cout << "poset_classification::identify before recognize" << endl;
		}

	recognize(data, sz,
		transporter, f_implicit_fusion,
		final_node,
		verbose_level);

	if (f_v) {
		cout << "poset_classification::identify after recognize" << endl;
		}

	longinteger_object go;

	orbit_at_level = final_node - first_poset_orbit_node_at_level[sz];
	get_stabilizer_order(sz, orbit_at_level, go);

	if (f_v) {
		cout << "poset_classification::identify trace returns "
				"final_node = " << final_node << " which is "
						"isomorphism type " << orbit_at_level
						<< " with ago=" << go << endl;
		}
	if (f_v) {
		cout << "poset_classification::identify transporter:" << endl;
		A->element_print_quick(transporter, cout);
		}

	if (f_v) {
		cout << "poset_classification::identify done" << endl;
		}

}

void poset_classification::test_identify(int level, int nb_times,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *transporter;
	int f_implicit_fusion = FALSE;
	int final_node;
	int *Elt;
	int nb_orbits, cnt, r, r2;
	int *set1;
	int *set2;
	sims *S;
	longinteger_object go;

	if (f_v) {
		cout << "poset_classification::test_identify, "
				"level = " << level
				<< " nb_times = " << nb_times << endl;
		}

	Elt = NEW_int(A->elt_size_in_int);
	transporter = NEW_int(A->elt_size_in_int);
	nb_orbits = nb_orbits_at_level(level);
	set1 = NEW_int(level);
	set2 = NEW_int(level);

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
			int_vec_print(cout, set1, level);
			cout << endl;
			}
		A->random_element(S, Elt, 0 /* verbose_level */);
		A2->map_a_set_and_reorder(set1, set2, level, Elt,
				0 /* verbose_level */);
		cout << "mapped set is ";
		int_vec_print(cout, set2, level);
		cout << endl;

		recognize(set2, level, transporter, f_implicit_fusion,
			final_node, verbose_level);
		
		r2 = final_node - first_poset_orbit_node_at_level[level];
		if (r2 != r) {
			cout << "recognition fails" << endl;
			exit(1);
			}
		else {
			cout << "recognition is successful" << endl;
			}
		}

	delete S;
	FREE_int(Elt);
	FREE_int(transporter);
	FREE_int(set1);
	FREE_int(set2);
	if (f_v) {
		cout << "poset_classification::test_identify done" << endl;
		}
}



#if 1
void poset_classification::poset_classification_apply_isomorphism_no_transporter(
	int cur_level, int size, int cur_node, int cur_ex, 
	int *set_in, int *set_out, 
	int verbose_level)
// Called by upstep_work::handle_extension_fusion_type
{
	int *Elt1;
	int *Elt2;
	int *set_tmp;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::poset_classification_apply_isomorphism_"
				"no_transporter" << endl;
		}

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	set_tmp = NEW_int(size);
	A->element_one(Elt1, 0);

	poset_classification_apply_isomorphism(cur_level, size, cur_node, cur_ex, 
		set_in, set_out, set_tmp, 
		Elt1, Elt2, 
		TRUE /* f_tolerant */, 
		0 /*verbose_level*/);

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(set_tmp);

	if (f_v) {
		cout << "poset_classification::poset_classification_apply_isomorphism_"
				"no_transporter done" << endl;
		}
}
#endif


int poset_classification::poset_classification_apply_isomorphism(
	int level, int size,
	int current_node, int current_extension, 
	int *set_in, int *set_out, int *set_tmp, 
	int *transporter_in, int *transporter_out, 
	int f_tolerant, 
	int verbose_level)
// returns next_node
{
	int f_v = (verbose_level >= 1);
	int next_node;
	poset_orbit_node *O;

	O = &root[current_node];

	if (f_v) {
		cout << "poset_classification::poset_"
				"classification_apply_isomorphism "
				"current_node=" << current_node
				<< " current_extension=" << current_extension << endl;
		cout << "level=" << level << endl;		
		cout << "applying fusion element to the set ";
		int_set_print(cout, set_in, size);
		cout << endl;
		}

	A2->element_retrieve(O->E[current_extension].data, Elt1, 0);
	
	if (f_v) {
		cout << "poset_classification::poset_"
				"classification_apply_isomorphism "
				"applying fusion element" << endl;
		A2->element_print_quick(Elt1, cout);
		cout << "in action " << A2->label << ":" << endl;
		A2->element_print_as_permutation(Elt1, cout);
		cout << "to the set ";
		int_vec_print(cout, set_in, size);
		cout << endl;
		}
	A2->map_a_set(set_in, set_tmp, size, Elt1, 0);
	if (f_v) {
		cout << "poset_classification::poset_"
				"classification_apply_isomorphism "
				"the set becomes: ";
		int_vec_print(cout, set_tmp, size);
		cout << endl;
		}

	A2->element_mult(transporter_in, Elt1, Elt2, 0);
	if (f_v) {
		int_vec_print(cout, set_in, size);
		cout << endl;
		}
	A2->move(Elt2, transporter_out);

	if (f_on_subspaces) {
		next_node = find_node_for_subspace_by_rank(set_tmp,
				level + 1, verbose_level - 1);
		int_vec_copy(set_tmp, set_out, size);
		}
	else {
		int_vec_heapsort(set_tmp, level + 1);
		int_vec_copy(set_tmp, set_out, size);
		if (f_v) {
			cout << "poset_classification::poset_"
					"classification_apply_isomorphism "
					"after sorting: ";
			}
		if (f_v) {
			cout << "poset_classification::poset_"
					"classification_apply_isomorphism "
					"calling find_poset_orbit_node_for_set: ";
			int_vec_print(cout, set_out, size);
			cout << endl;
			}

		next_node = find_poset_orbit_node_for_set(level + 1 /*size*/,
				set_out, f_tolerant, 0);
		// changed A Betten 2/19/2011

		}
	if (f_v) {
		cout << "poset_classification::poset_"
				"classification_apply_isomorphism from ";
		int_vec_print(cout, set_in, size);
		cout << " to ";
		int_vec_print(cout, set_out, size);
		cout << ", which is node " << next_node << endl;
		cout << "we are done" << endl;
		}
	return next_node;
}


int poset_classification::trace_set_recursion(
	int cur_level, int cur_node,
	int size, int level,
	int *canonical_set, int *tmp_set1, int *tmp_set2, 
	int *Elt_transporter, int *tmp_Elt1, 
	int f_tolerant, 
	int verbose_level)
// called by poset_classification::trace_set
// returns the node in the poset_classification
// that corresponds to the canonical_set
// or -1 if f_tolerant and the node could not be found
{
	int f_v = (verbose_level >= 1);
	int pt, pt0, current_extension, i, t, next_node;
	int f_failure_to_find_point;
	poset_orbit_node *O = &root[cur_node];
	
	if (f_v) {
		cout << "poset_classification::trace_set_recursion "
				"cur_level = " << cur_level
				<< " cur_node = " << cur_node << " : ";
		int_vec_print(cout, canonical_set, size);
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
			cout << "poset_classification::trace_set_recursion "
					"cur_level = " << cur_level
					<< " cur_node = " << cur_node << " : ";
			cout << "O->trace_next_point_in_place returns FALSE, "
					"sorting and restarting" << endl;
			}
		// this can only happen if f_lex is TRUE
		// we need to sort and restart the trace:

		int_vec_heapsort(canonical_set, cur_level + 1);
		
		
		return trace_set_recursion(0, 0, 
			size, level, 
			canonical_set, tmp_set1, tmp_set2, 
			Elt_transporter, tmp_Elt1, 
			f_tolerant, 
			verbose_level);
		}

	if (f_failure_to_find_point) {
		cout << "poset_classification::trace_set_recursion: "
				"f_failure_to_find_point" << endl;
		exit(1);
		}
	pt0 = canonical_set[cur_level];
	if (f_v) {
		cout << "poset_classification::trace_set_recursion "
				"cur_level = " << cur_level
				<< " cur_node = " << cur_node << " : ";
		int_vec_print(cout, canonical_set, size);
		cout << " point " << pt
				<< " has been mapped to " << pt0 << endl;
		}
	current_extension = O->find_extension_from_point(
			this, pt0, FALSE);

	if (current_extension < 0) {
		cout << "poset_classification::trace_set_recursion: "
				"did not find point" << endl;
		exit(1);
		}
	t = O->E[current_extension].type;
	if (t == EXTENSION_TYPE_EXTENSION) {
		// extension node
		next_node = O->E[current_extension].data;
		if (f_v) {
			cout << "poset_classification::trace_set_recursion "
					"cur_level = " << cur_level
					<< " cur_node = " << cur_node << " : ";
			int_vec_print(cout, canonical_set, size);
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
			cout << "poset_classification::trace_set_recursion "
					"cur_level = " << cur_level
					<< " cur_node = " << cur_node << " : ";
			cout << "before poset_classification_apply_isomorphism" << endl;
			}
		next_node = poset_classification_apply_isomorphism(cur_level, size, 
			cur_node, current_extension, 
			canonical_set, tmp_set1, tmp_set2, 
			Elt_transporter, tmp_Elt1, 
			f_tolerant, 
			verbose_level);
		if (f_v) {
			cout << "poset_classification::trace_set_recursion "
					"cur_level = " << cur_level
					<< " cur_node = " << cur_node << " : ";
			cout << "after poset_classification_apply_isomorphism" << endl;
			}
		if (f_v) {
			cout << "poset_classification::trace_set_recursion "
					"cur_level = " << cur_level
				<< " cur_node = " << cur_node << " : " 
				<< " current_extension = " << current_extension 
				<< " : fusion from ";
			int_vec_print(cout, canonical_set, size);
			cout << " to ";
			int_vec_print(cout, tmp_set1, size);
			cout << " : we continue with node " << next_node << endl; 
			cout << endl;
			}

		if (next_node == -1) { // can only happen if f_tolerant is TRUE
			if (f_v) {
				cout << "poset_classification::trace_set_recursion "
						"cur_level = " << cur_level
					<< " cur_node = " << cur_node << " : " 
					<< " current_extension = " << current_extension 
					<< " : fusion from ";
				int_vec_print(cout, canonical_set, size);
				cout << " to ";
				int_vec_print(cout, tmp_set1, size);
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
	cout << "poset_classification::trace_set_recursion "
			"unknown type " << t << endl;
	exit(1);
}

int poset_classification::trace_set(int *set, int size, int level, 
	int *canonical_set, int *Elt_transporter, 
	int verbose_level)
// called by map_set_to_set_BLT in orbits.C
// returns the case number of the canonical set
{
	int i, n, case_nb;
	int f_v = (verbose_level >= 1);
	int *tmp_set1, *tmp_set2;
	int *tmp_Elt;

	tmp_set1 = NEW_int(size);
	tmp_set2 = NEW_int(size);
	tmp_Elt = NEW_int(A->elt_size_in_int);

	if (f_v) {
		cout << "poset_classification::trace_set" << endl;
		cout << "tracing set ";
		int_vec_print(cout, set, size);	
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

	case_nb = n - first_poset_orbit_node_at_level[level];

	if (case_nb < 0) {
		cout << "poset_classification::trace_set, "
				"case_nb < 0, case_nb = " << case_nb << endl;
		exit(1);
		}
	FREE_int(tmp_set1);
	FREE_int(tmp_set2);
	FREE_int(tmp_Elt);
	return case_nb;
}

int poset_classification::find_node_for_subspace_by_rank(
		int *set, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *v;
	int *basis;
	int *base_cols;
	int rk, node, i, j, pt;

	if (f_v) {
		cout << "poset_classification::find_node_for_subspace_by_rank for set ";
		int_vec_print(cout, set, len);
		cout << endl;
		}
	v = tmp_find_node_for_subspace_by_rank1;
	basis = tmp_find_node_for_subspace_by_rank2;
	base_cols = tmp_find_node_for_subspace_by_rank3;
	//v = NEW_int(vector_space_dimension);
	//basis = NEW_int(len * vector_space_dimension);
	//base_cols = NEW_int(vector_space_dimension);
	for (i = 0; i < len; i++) {
		unrank_point(basis + i * vector_space_dimension, set[i]);
		//(*unrank_point_func)(basis + i * vector_space_dimension,
		//set[i], rank_point_data);
		}
	rk = F->Gauss_simple(
			basis, len, vector_space_dimension,
			base_cols, 0 /* verbose_level */);
	if (rk != len) {
		cout << "poset_classification::find_node_for_subspace_by_rank "
				"rk != len" << endl;
		exit(1);
		}
	node = 0;
	for (i = 0; i < len; i++) {
		poset_orbit_node *O;

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
				cout << "poset_classification::find_node_for_subspace_by_rank "
						"at node " << node << " extension " << j
						<< " with point " << pt << " to node "
						<< O->E[j].data << endl;
				}
			node = O->E[j].data;
			set[i] = pt;
			break;
			}
		if (j == O->nb_extensions) {
			cout << "poset_classification::find_node_for_subspace_by_rank "
					"at node " << node << " fatal, "
							"could not find extension" << endl;
			exit(1);
			}
		}
	if (f_v) {
		cout << "poset_classification::find_node_for_subspace_by_rank "
				"the canonical set is ";
		int_vec_print(cout, set, len);
		cout << " at node " << node << endl;
		}
	
	//FREE_int(v);
	//FREE_int(basis);
	//FREE_int(base_cols);
	return node;
}

