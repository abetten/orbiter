// poset_classification_trace.cpp
//
// Anton Betten
//
// moved out of poset_classification.cpp: Jan 21, 2010

#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

using namespace std;

namespace orbiter {
namespace classification {

int poset_classification::find_isomorphism(
		long int *set1, long int *set2, int sz,
		int *transporter, int &orbit_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *set1_canonical;
	long int *set2_canonical;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int orb1;
	int orb2;
	int ret;

	if (f_v) {
		cout << "poset_classification::find_isomorphism" << endl;
	}
	
	set1_canonical = NEW_lint(sz);
	set2_canonical = NEW_lint(sz);
	Elt1 = NEW_int(Poset->A->elt_size_in_int);
	Elt2 = NEW_int(Poset->A->elt_size_in_int);
	Elt3 = NEW_int(Poset->A->elt_size_in_int);

	orb1 = trace_set(set1, sz, sz, 
		set1_canonical, Elt1, 
		0 /* verbose_level */);

	orb2 = trace_set(set2, sz, sz, 
		set2_canonical, Elt2, 
		0 /* verbose_level */);

	if (orb1 == orb2) {
		ret = TRUE;
		Poset->A->element_invert(Elt2, Elt3, 0);
		Poset->A->element_mult(Elt1, Elt3, transporter, 0);
		orbit_idx = orb1;
	}
	else {
		orbit_idx = -1;
		ret = FALSE;
	}

	FREE_lint(set1_canonical);
	FREE_lint(set2_canonical);
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

	if (f_v) {
		cout << "poset_classification::find_isomorphism done" << endl;
	}
	return ret;
}

set_and_stabilizer *poset_classification::identify_and_get_stabilizer(
		long int *set, int sz, int *transporter,
		int &orbit_at_level,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//set_and_stabilizer *SaS0;
	set_and_stabilizer *SaS;
	int *Elt;
	sorting Sorting;

	if (f_v) {
		cout << "poset_classification::identify_and_get_stabilizer" << endl;
	}
	if (f_v) {
		cout << "poset_classification::identify_and_get_stabilizer "
				"identifying the set ";
		Orbiter->Lint_vec.print(cout, set, sz);
		cout << endl;
	}
	Elt = NEW_int(Poset->A->elt_size_in_int);
	identify(set, sz, transporter,
			orbit_at_level, verbose_level - 2);

	SaS = get_set_and_stabilizer(sz,
			orbit_at_level, 0 /* verbose_level */);

	Poset->A->element_invert(transporter, Elt, 0);
	SaS->apply_to_self(Elt, 0 /* verbose_level */);

	if (f_v) {
		cout << "poset_classification::identify_and_get_stabilizer "
				"input set=";
		Orbiter->Lint_vec.print(cout, set, sz);
		cout << endl;
		cout << "poset_classification::identify_and_get_stabilizer "
				"SaS->set=";
		Orbiter->Lint_vec.print(cout, SaS->data, SaS->sz);
		cout << endl;
	}
	if (Sorting.compare_sets_lint(set, SaS->data, sz, SaS->sz)) {
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

void poset_classification::identify(long int *data, int sz,
		int *transporter, int &orbit_at_level,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_implicit_fusion = FALSE;
	int final_node;

	if (f_v) {
		cout << "poset_classification::identify" << endl;
	}
	if (f_v) {
		cout << "poset_classification::identify identifying the set ";
		Orbiter->Lint_vec.print(cout, data, sz);
		cout << endl;
	}

	if (f_v) {
		cout << "poset_classification::identify before recognize" << endl;
	}

	recognize(data, sz,
		transporter, f_implicit_fusion,
		final_node,
		verbose_level - 2);

	if (f_v) {
		cout << "poset_classification::identify after recognize" << endl;
	}

	longinteger_object go;

	orbit_at_level = final_node - Poo->first_node_at_level(sz);
	get_stabilizer_order(sz, orbit_at_level, go);

	if (f_v) {
		cout << "poset_classification::identify trace returns "
				"final_node = " << final_node << " which is "
						"isomorphism type " << orbit_at_level
						<< " with ago=" << go << endl;
	}
	if (f_v) {
		cout << "poset_classification::identify transporter:" << endl;
		Poset->A->element_print_quick(transporter, cout);
	}

	if (f_v) {
		cout << "poset_classification::identify done" << endl;
	}

}

void poset_classification::test_identify(int level, int nb_times,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *transporter;
	int f_implicit_fusion = FALSE;
	int final_node;
	int *Elt;
	int nb_orbits, cnt, r, r2;
	long int *set1;
	long int *set2;
	sims *S;
	longinteger_object go;
	os_interface Os;

	if (f_v) {
		cout << "poset_classification::test_identify, "
				"level = " << level
				<< " nb_times = " << nb_times << endl;
	}

	Elt = NEW_int(Poset->A->elt_size_in_int);
	transporter = NEW_int(Poset->A->elt_size_in_int);
	nb_orbits = nb_orbits_at_level(level);
	set1 = NEW_lint(level);
	set2 = NEW_lint(level);

	S = Poset->Strong_gens->create_sims(0 /*verbose_level*/);

	S->group_order(go);

	if (f_v) {
		cout << "poset_classification::test_identify "
				"Group of order " << go << " has been created" << endl;
	}



	for (cnt = 0; cnt < nb_times; cnt++) {
		r = Os.random_integer(nb_orbits);
		if (f_v) {
			cout << "random orbit " << r << " / " << nb_orbits << endl;
		}
		get_set_by_level(level, r, set1);
		if (f_v) {
			cout << "random orbit " << r << " / "
					<< nb_orbits << " is represented by ";
			Orbiter->Lint_vec.print(cout, set1, level);
			cout << endl;
		}
		Poset->A->random_element(S, Elt, 0 /* verbose_level */);
		Poset->A2->map_a_set_and_reorder(set1, set2, level, Elt,
				0 /* verbose_level */);
		if (f_v) {
			cout << "mapped set is ";
			Orbiter->Lint_vec.print(cout, set2, level);
			cout << endl;
		}

		recognize(set2, level, transporter, f_implicit_fusion,
			final_node, verbose_level);
		
		r2 = final_node - Poo->first_node_at_level(level);
		if (r2 != r) {
			cout << "recognition fails" << endl;
			exit(1);
		}
		else {
			if (f_v) {
				cout << "recognition is successful" << endl;
			}
		}
	}

	FREE_OBJECT(S);
	FREE_int(Elt);
	FREE_int(transporter);
	FREE_lint(set1);
	FREE_lint(set2);
	if (f_v) {
		cout << "poset_classification::test_identify done" << endl;
	}
}



#if 1
void poset_classification::poset_classification_apply_isomorphism_no_transporter(
	int cur_level, int size, int cur_node, int cur_ex, 
	long int *set_in, long int *set_out,
	int verbose_level)
// Called by upstep_work::handle_extension_fusion_type
{
	int *Elt1;
	int *Elt2;
	long int *set_tmp;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::poset_classification_apply_isomorphism_"
				"no_transporter" << endl;
	}

	Elt1 = NEW_int(Poset->A->elt_size_in_int);
	Elt2 = NEW_int(Poset->A->elt_size_in_int);
	set_tmp = NEW_lint(size);
	Poset->A->element_one(Elt1, 0);

	poset_classification_apply_isomorphism(cur_level, size, cur_node, cur_ex, 
		set_in, set_out, set_tmp, 
		Elt1, Elt2, 
		TRUE /* f_tolerant */, 
		0 /*verbose_level*/);

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_lint(set_tmp);

	if (f_v) {
		cout << "poset_classification::poset_classification_apply_isomorphism_"
				"no_transporter done" << endl;
	}
}
#endif


int poset_classification::poset_classification_apply_isomorphism(
	int level, int size,
	int current_node, int current_extension, 
	long int *set_in, long int *set_out, long int *set_tmp,
	int *transporter_in, int *transporter_out, 
	int f_tolerant, 
	int verbose_level)
// returns next_node
{
	int f_v = (verbose_level >= 1);
	int next_node;
	poset_orbit_node *O;
	sorting Sorting;

	O = get_node(current_node);
	//O = &root[current_node];

	if (f_v) {
		cout << "poset_classification::poset_"
				"classification_apply_isomorphism "
				"current_node=" << current_node
				<< " current_extension=" << current_extension << endl;
		cout << "level=" << level << endl;		
		cout << "applying fusion element to the set ";
		Orbiter->Lint_vec.set_print(cout, set_in, size);
		cout << endl;
	}

	Poset->A2->element_retrieve(O->get_E(current_extension)->get_data(), Elt1, 0);
	
	if (f_v) {
		cout << "poset_classification::poset_"
				"classification_apply_isomorphism "
				"applying fusion element" << endl;
		Poset->A2->element_print_quick(Elt1, cout);
		cout << "in action " << Poset->A2->label << ":" << endl;
		Poset->A2->element_print_as_permutation(Elt1, cout);
		cout << "to the set ";
		Orbiter->Lint_vec.print(cout, set_in, size);
		cout << endl;
	}
	Poset->A2->map_a_set(set_in, set_tmp, size, Elt1, 0);
	if (f_v) {
		cout << "poset_classification::poset_"
				"classification_apply_isomorphism "
				"the set becomes: ";
		Orbiter->Lint_vec.print(cout, set_tmp, size);
		cout << endl;
	}

	Poset->A2->element_mult(transporter_in, Elt1, Elt2, 0);
	if (f_v) {
		Orbiter->Lint_vec.print(cout, set_in, size);
		cout << endl;
	}
	Poset->A2->move(Elt2, transporter_out);

	if (Poset->f_subspace_lattice) {
		next_node = find_node_for_subspace_by_rank(set_tmp,
				level + 1, verbose_level - 1);
		Orbiter->Lint_vec.copy(set_tmp, set_out, size);
	}
	else {
		Sorting.lint_vec_heapsort(set_tmp, level + 1);
		Orbiter->Lint_vec.copy(set_tmp, set_out, size);
		if (f_v) {
			cout << "poset_classification::poset_"
					"classification_apply_isomorphism "
					"after sorting: ";
		}
		if (f_v) {
			cout << "poset_classification::poset_"
					"classification_apply_isomorphism "
					"calling find_poset_orbit_node_for_set: ";
			Orbiter->Lint_vec.print(cout, set_out, size);
			cout << endl;
		}

		next_node = find_poset_orbit_node_for_set(level + 1 /*size*/,
				set_out, f_tolerant, 0);
		// changed A Betten 2/19/2011

	}
	if (f_v) {
		cout << "poset_classification::poset_"
				"classification_apply_isomorphism from ";
		Orbiter->Lint_vec.print(cout, set_in, size);
		cout << " to ";
		Orbiter->Lint_vec.print(cout, set_out, size);
		cout << ", which is node " << next_node << endl;
		cout << "we are done" << endl;
	}
	return next_node;
}


int poset_classification::trace_set_recursion(
	int cur_level, int cur_node,
	int size, int level,
	long int *canonical_set, long int *tmp_set1, long int *tmp_set2,
	int *Elt_transporter, int *tmp_Elt1, 
	int f_tolerant, 
	int verbose_level)
// called by poset_classification::trace_set
// returns the node in the poset_classification
// that corresponds to the canonical_set
// or -1 if f_tolerant and the node could not be found
{
	int f_v = (verbose_level >= 1);
	long int pt, pt0;
	int current_extension, i, t, next_node;
	int f_failure_to_find_point;
	poset_orbit_node *O = get_node(cur_node); //&root[cur_node];
	sorting Sorting;
	
	if (f_v) {
		cout << "poset_classification::trace_set_recursion "
				"cur_level = " << cur_level
				<< " cur_node = " << cur_node << " : ";
		Orbiter->Lint_vec.print(cout, canonical_set, size);
		cout << endl;
	}
	pt = canonical_set[cur_level];
	if (f_v) {
		cout << "tracing point " << pt << endl;
	}
	if (f_v) {
		cout << "poset_classification::trace_set_recursion "
				"before O->trace_next_point_in_place" << endl;
	}
	if (!O->trace_next_point_in_place(this, 
		cur_level, cur_node, size, 
		canonical_set, tmp_set1,
		Elt_transporter, tmp_Elt1, 
		Control->f_lex,
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

		Sorting.lint_vec_heapsort(canonical_set, cur_level + 1);
		
		
		if (f_v) {
			cout << "poset_classification::trace_set_recursion "
					"before trace_set_recursion" << endl;
		}
		int r;

		r = trace_set_recursion(0, 0,
			size, level, 
			canonical_set, tmp_set1, tmp_set2, 
			Elt_transporter, tmp_Elt1, 
			f_tolerant, 
			verbose_level);
		if (f_v) {
			cout << "poset_classification::trace_set_recursion "
					"after trace_set_recursion, r = " << r << endl;
		}
		return r;
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
		Orbiter->Lint_vec.print(cout, canonical_set, size);
		cout << " point " << pt
				<< " has been mapped to " << pt0 << endl;
	}
	if (f_v) {
		cout << "poset_classification::trace_set_recursion "
				"before O->find_extension_from_point" << endl;
	}
	current_extension = O->find_extension_from_point(
			this, pt0, 0 /* verbose_level */);
	if (f_v) {
		cout << "poset_classification::trace_set_recursion "
				"after O->find_extension_from_point "
				"current_extension=" << current_extension<< endl;
	}

	if (current_extension < 0) {
		cout << "poset_classification::trace_set_recursion: "
				"did not find point" << endl;
		exit(1);
	}
	t = O->get_E(current_extension)->get_type();
	if (t == EXTENSION_TYPE_EXTENSION) {
		// extension node
		if (f_v) {
			cout << "poset_classification::trace_set_recursion "
					"EXTENSION_TYPE_EXTENSION" << endl;
		}
		next_node = O->get_E(current_extension)->get_data();
		if (f_v) {
			cout << "poset_classification::trace_set_recursion "
					"cur_level = " << cur_level
					<< " cur_node = " << cur_node << " : ";
			Orbiter->Lint_vec.print(cout, canonical_set, size);
			cout << " point " << pt << " has been mapped to "
					<< pt0 << " next node is node " << next_node << endl;
		}
		if (cur_level + 1 == level) {
			return next_node;
		}
		else {
			int r;

			if (f_v) {
				cout << "poset_classification::trace_set_recursion "
						"before trace_set_recursion" << endl;
			}
			r = trace_set_recursion(cur_level + 1, next_node,
				size, level, canonical_set, tmp_set1, tmp_set2,  
				Elt_transporter, tmp_Elt1, 
				f_tolerant, 
				verbose_level);
			if (f_v) {
				cout << "poset_classification::trace_set_recursion "
						"after trace_set_recursion" << endl;
			}
			return r;
		}
	}
	else if (t == EXTENSION_TYPE_FUSION) {
		// fusion node
		if (f_v) {
			cout << "poset_classification::trace_set_recursion "
					"EXTENSION_TYPE_FUSION" << endl;
		}

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
			Orbiter->Lint_vec.print(cout, canonical_set, size);
			cout << " to ";
			Orbiter->Lint_vec.print(cout, tmp_set1, size);
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
				Orbiter->Lint_vec.print(cout, canonical_set, size);
				cout << " to ";
				Orbiter->Lint_vec.print(cout, tmp_set1, size);
				cout << "we stop tracing" << endl;
			}
			return -1;
		}
		Poset->A->element_move(tmp_Elt1, Elt_transporter, 0);

		for (i = 0; i < size; i++) {
			canonical_set[i] = tmp_set1[i];
		}

		if (cur_level + 1 == level) {
			return next_node;
		}
		else {
			int r;

			if (f_v) {
				cout << "poset_classification::trace_set_recursion "
						"before trace_set_recursion" << endl;
			}
			r = trace_set_recursion(cur_level + 1, next_node,
				size, level, canonical_set, tmp_set1, tmp_set2,  
				Elt_transporter, tmp_Elt1, 
				f_tolerant, 
				verbose_level);

			if (f_v) {
				cout << "poset_classification::trace_set_recursion "
						"after trace_set_recursion" << endl;
			}
			return r;
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

int poset_classification::trace_set(
		long int *set,
		int size, int level,
		long int *canonical_set, int *Elt_transporter,
		int verbose_level)
// called by map_set_to_set_BLT in orbits.cpp
// returns the case number of the canonical set
{
	int n, case_nb;
	int f_v = (verbose_level >= 1);
	long int *tmp_set1, *tmp_set2;
	int *tmp_Elt;

	tmp_set1 = NEW_lint(size);
	tmp_set2 = NEW_lint(size);
	tmp_Elt = NEW_int(Poset->A->elt_size_in_int);

	if (f_v) {
		cout << "poset_classification::trace_set" << endl;
		cout << "tracing set ";
		Orbiter->Lint_vec.print(cout, set, size);
		cout << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "level=" << level << endl;
		cout << "f_lex=" << Control->f_lex << endl;
	}
	
	Orbiter->Lint_vec.copy(set, canonical_set, size);

	Poset->A->element_one(Elt_transporter, 0);

	if (f_v) {
		cout << "poset_classification::trace_set "
				"before trace_set_recursion" << endl;
	}
	n = trace_set_recursion(
		0 /* cur_level */,
		0 /* cur_node */,  size, level,
		canonical_set, tmp_set1, tmp_set2, 
		Elt_transporter, tmp_Elt, 
		FALSE /*f_tolerant*/, 
		verbose_level - 1);
	if (f_v) {
		cout << "poset_classification::trace_set "
				"after trace_set_recursion" << endl;
		cout << "n = " << n << endl;
	}

	case_nb = n - Poo->first_node_at_level(level);

	if (case_nb < 0) {
		cout << "poset_classification::trace_set, "
				"case_nb < 0, case_nb = " << case_nb << endl;
		cout << "poset_classification::trace_set, "
				"level = " << level << endl;
		cout << "poset_classification::trace_set, "
				"first_poset_orbit_node_at_level[level] = " << Poo->first_node_at_level(level) << endl;
		exit(1);
	}
	FREE_lint(tmp_set1);
	FREE_lint(tmp_set2);
	FREE_int(tmp_Elt);
	return case_nb;
}

long int poset_classification::find_node_for_subspace_by_rank(
		long int *set, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *v;
	int *basis;
	long int rk, node, i, j, pt;

	if (f_v) {
		cout << "poset_classification::find_node_for_subspace_by_rank for set ";
		Orbiter->Lint_vec.print(cout, set, len);
		cout << endl;
	}
	v = tmp_find_node_for_subspace_by_rank1;
	basis = tmp_find_node_for_subspace_by_rank2;

	unrank_basis(basis, set, len);

	rk = Poset->VS->RREF_and_rank(basis, len);

	if (rk != len) {
		cout << "poset_classification::find_node_for_subspace_by_rank "
				"rk != len" << endl;
		exit(1);
	}
	node = 0;
	for (i = 0; i < len; i++) {
		poset_orbit_node *O;

		//O = &root[node];
		O = get_node(node);
		for (j = 0; j < O->get_nb_of_extensions(); j++) {
			if (O->get_E(j)->get_type() != EXTENSION_TYPE_EXTENSION) {
				continue;
			}
			pt = O->get_E(j)->get_pt();
			unrank_point(v, pt);
			if (!Poset->VS->is_contained_in_subspace(v, basis, len)) {
				continue;
			}
			if (f_vv) {
				cout << "poset_classification::find_node_for_subspace_by_rank "
						"at node " << node << " extension " << j
						<< " with point " << pt << " to node "
						<< O->get_E(j)->get_data() << endl;
			}
			node = O->get_E(j)->get_data();
			set[i] = pt;
			break;
		}
		if (j == O->get_nb_of_extensions()) {
			cout << "poset_classification::find_node_for_subspace_by_rank "
					"at node " << node << " fatal, "
							"could not find extension" << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "poset_classification::find_node_for_subspace_by_rank "
				"the canonical set is ";
		Orbiter->Lint_vec.print(cout, set, len);
		cout << " at node " << node << endl;
	}
	
	return node;
}

// #############################################################################
// global:
// #############################################################################



const char *trace_result_as_text(trace_result r)
{
	if (r == found_automorphism) {
		return "found_automorphism";
	}
	else if (r == not_canonical) {
		return "not_canonical";
	}
	else if (r == no_result_extension_not_found) {
		return "no_result_extension_not_found";
	}
	else if (r == no_result_fusion_node_installed) {
		return "no_result_fusion_node_installed";
	}
	else if (r == no_result_fusion_node_already_installed) {
		return "no_result_fusion_node_already_installed";
	}
	else {
		return "unkown trace result";
	}
}

int trace_result_is_no_result(trace_result r)
{
	if (r == no_result_extension_not_found ||
		r == no_result_fusion_node_installed ||
		r == no_result_fusion_node_already_installed) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}






}}

