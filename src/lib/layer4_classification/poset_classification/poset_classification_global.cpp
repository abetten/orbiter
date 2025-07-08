/*
 * poset_classification_global.cpp
 *
 *  Created on: Jul 4, 2025
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


poset_classification_global::poset_classification_global()
{
	Record_birth();

	PC = NULL;
}

poset_classification_global::~poset_classification_global()
{
	Record_death();

}

void poset_classification_global::init(
		poset_classification *PC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification_global::init" << endl;
	}

	poset_classification_global::PC = PC;

	if (f_v) {
		cout << "poset_classification_global::init done" << endl;
	}
}

void poset_classification_global::count_automorphism_group_orders(
	int lvl, int &nb_agos,
	algebra::ring_theory::longinteger_object *&agos,
	int *&multiplicities,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification_global::count_automorphism_group_orders" << endl;
	}

	int i, l, j, c, h, f_added;
	algebra::ring_theory::longinteger_object ago;
	algebra::ring_theory::longinteger_object *tmp_agos;
	int *tmp_multiplicities;
	algebra::ring_theory::longinteger_domain D;

	l = PC->nb_orbits_at_level(lvl);
	if (f_v) {
		cout << "collecting the automorphism group orders of "
				<< l << " orbits" << endl;
	}
	nb_agos = 0;
	agos = NULL;
	multiplicities = NULL;
	for (i = 0; i < l; i++) {
		PC->get_stabilizer_order(lvl, i, ago);
		f_added = false;
		for (j = 0; j < nb_agos; j++) {
			c = D.compare_unsigned(ago, agos[j]);
			//cout << "comparing " << ago << " with " << agos[j]
			// << " yields " << c << endl;
			if (c >= 0) {
				if (c == 0) {
					multiplicities[j]++;
				}
				else {
					tmp_agos = agos;
					tmp_multiplicities = multiplicities;
					agos = NEW_OBJECTS(algebra::ring_theory::longinteger_object, nb_agos + 1);
					multiplicities = NEW_int(nb_agos + 1);
					for (h = 0; h < j; h++) {
						tmp_agos[h].swap_with(agos[h]);
						multiplicities[h] = tmp_multiplicities[h];
					}
					ago.swap_with(agos[j]);
					multiplicities[j] = 1;
					for (h = j; h < nb_agos; h++) {
						tmp_agos[h].swap_with(agos[h + 1]);
						multiplicities[h + 1] = tmp_multiplicities[h];
					}
					nb_agos++;
					if (tmp_agos) {
						FREE_OBJECTS(tmp_agos);
						FREE_int(tmp_multiplicities);
					}
				}
				f_added = true;
				break;
			}
		}
		if (!f_added) {
			// add at the end (including the case that the list is empty)
			tmp_agos = agos;
			tmp_multiplicities = multiplicities;
			agos = NEW_OBJECTS(algebra::ring_theory::longinteger_object, nb_agos + 1);
			multiplicities = NEW_int(nb_agos + 1);
			for (h = 0; h < nb_agos; h++) {
				tmp_agos[h].swap_with(agos[h]);
				multiplicities[h] = tmp_multiplicities[h];
			}
			ago.swap_with(agos[nb_agos]);
			multiplicities[nb_agos] = 1;
			nb_agos++;
			if (tmp_agos) {
				FREE_OBJECTS(tmp_agos);
				FREE_int(tmp_multiplicities);
			}
		}
	}
	if (f_v) {
		cout << "poset_classification_global::count_automorphism_group_orders done" << endl;
	}
}

std::string poset_classification_global::compute_and_stringify_automorphism_group_orders(
		int lvl,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification_global::compute_and_print_automorphism_group_orders" << endl;
	}

	int j, nb_agos;
	algebra::ring_theory::longinteger_object *agos;
	int *multiplicities;
	int N, r, h;
	algebra::ring_theory::longinteger_object S, S1, Q;
	algebra::ring_theory::longinteger_domain D;

	count_automorphism_group_orders(
			lvl, nb_agos, agos,
			multiplicities, verbose_level);

	S.create(0);
	N = 0;
	for (j = 0; j < nb_agos; j++) {
		N += multiplicities[j];
		for (h = 0; h < multiplicities[j]; h++) {
			D.add(S, agos[j], S1);
			S1.assign_to(S);
		}
	}
	D.integral_division_by_int(S, N, Q, r);

	std::string s;

	s = "(";
	for (j = 0; j < nb_agos; j++) {
		s += agos[j].stringify();
		if (multiplicities[j] == 1) {
		}
		else if (multiplicities[j] >= 10) {
			s += "^{" + std::to_string(multiplicities[j]) + "}";
		}
		else  {
			s += "^" + std::to_string(multiplicities[j]);
		}
		if (j < nb_agos - 1) {
			s += ", ";
		}
	}
	s += ") average is " + Q.stringify() + " + " + std::to_string(r) + " / " + std::to_string(N);
	if (nb_agos) {
		FREE_OBJECTS(agos);
		FREE_int(multiplicities);
	}
	if (f_v) {
		cout << "poset_classification_global::compute_and_print_automorphism_group_orders done" << endl;
	}
	return s;

}

void poset_classification_global::find_interesting_k_subsets(
	long int *the_set, int n, int k,
	int *&interesting_sets, int &nb_interesting_sets,
	int &orbit_idx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::tally *C;
	int j, t, f, l, l_min, t_min = 0;

	if (f_v) {
		cout << "poset_classification_global::find_interesting_k_subsets "
				"n = " << n << " k = " << k << endl;
	}


	classify_k_subsets(the_set, n, k, C, verbose_level);


	if (f_v) {
		C->print_bare(false);
		cout << endl;
	}

	l_min = INT_MAX;
	f = 0;
	for (t = 0; t < C->nb_types; t++) {
		f = C->type_first[t];
		l = C->type_len[t];
		if (l < l_min) {
			l_min = l;
			t_min = t;
		}
	}
	interesting_sets = NEW_int(l_min);
	nb_interesting_sets = l_min;
	for (j = 0; j < l_min; j++) {
		interesting_sets[j] = C->sorting_perm_inv[f + j];
	}
	orbit_idx = C->data_sorted[f];
	if (f_v) {
		cout << "poset_classification_global::find_interesting_k_subsets "
				"l_min = " << l_min << " t_min = " << t_min
				<< " orbit_idx = " << orbit_idx << endl;
	}
	if (f_v) {
		cout << "poset_classification_global::find_interesting_k_subsets "
				"interesting set of size "
				<< nb_interesting_sets << " : ";
		Int_vec_print(cout, interesting_sets, nb_interesting_sets);
		cout << endl;
	}

	FREE_OBJECT(C);

	if (f_v) {
		cout << "poset_classification_global::find_interesting_k_subsets "
				"n = " << n << " k = " << k << " done" << endl;
	}
}

void poset_classification_global::classify_k_subsets(
		long int *the_set, int n, int k,
		other::data_structures::tally *&C, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int nCk;
	int *isotype;

	if (f_v) {
		cout << "poset_classification_global::classify_k_subsets "
				"n = " << n << " k = " << k << endl;
	}

	trace_all_k_subsets(the_set, n, k, nCk, isotype, verbose_level);

	C = NEW_OBJECT(other::data_structures::tally);

	C->init(isotype, nCk, false, 0);

	if (f_v) {
		cout << "poset_classification_global::classify_k_subsets "
				"n = " << n << " k = " << k << " done" << endl;
	}
}

void poset_classification_global::trace_all_k_subsets_and_compute_frequencies(
		long int *the_set,
		int n, int k, int &nCk,
		int *&isotype, int *&orbit_frequencies, int &nb_orbits,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a;


	if (f_v) {
		cout << "poset_classification_global::trace_all_k_subsets_and_compute_frequencies "
				"n = " << n << " k = " << k << " nCk=" << nCk << endl;
	}

	trace_all_k_subsets(
			the_set,
			n, k, nCk, isotype,
			verbose_level);

	nb_orbits = PC->nb_orbits_at_level(k);
	orbit_frequencies = NEW_int(nb_orbits);
	Int_vec_zero(orbit_frequencies, nb_orbits);

	for (i = 0; i < nCk; i++) {
		a = isotype[i];
		orbit_frequencies[a]++;
	}

	if (f_v) {
		cout << "poset_classification_global::trace_all_k_subsets_and_compute_frequencies done" << endl;
	}
}

void poset_classification_global::trace_all_k_subsets(
		long int *the_set,
		int n, int k, int &nCk, int *&isotype,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = false; //(verbose_level >= 2);
	int *index_set;
	long int *subset;
	long int *canonical_subset;
	int *Elt;
	long int subset_rk, local_idx, i;
	//int f_implicit_fusion = true;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	nCk = Combi.int_n_choose_k(n, k);
	if (f_v) {
		cout << "poset_classification_global::trace_all_k_subsets "
				"n = " << n << " k = " << k
				<< " nCk = " << nCk << endl;
	}

	Elt = NEW_int(PC->get_poset()->A->elt_size_in_int);

	index_set = NEW_int(k);
	subset = NEW_lint(k);
	canonical_subset = NEW_lint(k);
	isotype = NEW_int(nCk);

	Int_vec_zero(isotype, nCk);

	Combi.first_k_subset(index_set, n, k);
	subset_rk = 0;

	while (true) {
		if (true && ((subset_rk % 10000) == 0)) {
			cout << "poset_classification_global::trace_all_k_subsets "
					"k=" << k
				<< " testing set " << subset_rk << " / " << nCk
				<< " = " << 100. * (double) subset_rk /
				(double) nCk << " % : ";
			Int_vec_print(cout, index_set, k);
			cout << endl;
		}
		for (i = 0; i < k; i++) {
			subset[i] = the_set[index_set[i]];
		}
		//Lint_vec_copy(subset, set[0], k);

		if (false /*f_v2*/) {
			cout << "poset_classification_global::trace_all_k_subsets "
					"corresponding to set ";
			Lint_vec_print(cout, subset, k);
			cout << endl;
		}
		//Poset->A->element_one(transporter->ith(0), 0);

		if (k == 0) {
			isotype[0] = 0;
		}
		else {

			if (false) {
				cout << "poset_classification_global::trace_all_k_subsets "
						"before trace_set" << endl;
			}
			local_idx = PC->trace_set(
					subset, k, k,
				canonical_subset, Elt,
				0 /*verbose_level - 3*/);
			if (false) {
				cout << "poset_classification_global::trace_all_k_subsets "
						"after trace_set, local_idx = "
						<< local_idx << endl;
			}

			if (false /*f_vvv*/) {
				cout << "poset_classification_global::trace_all_k_subsets "
						"local_idx=" << local_idx << endl;
			}
			isotype[subset_rk] = local_idx;
			if (false) {
				cout << "poset_classification_global::trace_all_k_subsets "
						"the transporter is" << endl;
				PC->get_poset()->A->Group_element->element_print(Elt, cout);
				cout << endl;
			}

		}
		subset_rk++;
		if (!Combi.next_k_subset(index_set, n, k)) {
			break;
		}
	}
	if (subset_rk != nCk) {
		cout << "poset_classification_global::trace_all_k_subsets "
				"subset_rk != nCk" << endl;
		exit(1);
	}


	FREE_int(index_set);
	FREE_lint(subset);
	FREE_lint(canonical_subset);
	FREE_int(Elt);
	if (f_v) {
		cout << "poset_classification_global::trace_all_k_subsets done" << endl;
	}
}


void poset_classification_global::list_all_orbits_at_level(
	int depth,
	int f_has_print_function,
	void (*print_function)(
			std::ostream &ost,
			int len, long int *S, void *data),
	void *print_function_data,
	int f_show_orbit_decomposition, int f_show_stab,
	int f_save_stab, int f_show_whole_orbit)
{
	int l, i;

	l = PC->nb_orbits_at_level(depth);

	cout << "poset_classification_global::list_all_orbits_at_level "
			"listing all orbits "
			"at depth " << depth << ":" << endl;
	for (i = 0; i < l; i++) {
		cout << "poset_classification_global::list_all_orbits_at_level "
			"listing orbit "
			<< i << " / " << l << endl;
		list_whole_orbit(depth, i,
			f_has_print_function, print_function, print_function_data,
			f_show_orbit_decomposition, f_show_stab,
			f_save_stab, f_show_whole_orbit);
	}
}

void poset_classification_global::compute_integer_property_of_selected_list_of_orbits(
	int depth,
	int nb_orbits, int *Orbit_idx,
	int (*compute_function)(
			int len, long int *S, void *data),
	void *compute_function_data,
	int *&Data)
{
	int l, i, j, d;
	long int *set;

	set = NEW_lint(depth);
	l = PC->nb_orbits_at_level(depth);

	Data = NEW_int(nb_orbits);

	cout << "computing integer property for a set of "
			<< nb_orbits << " orbits at "
			"depth " << depth << ":" << endl;
	for (j = 0; j < nb_orbits; j++) {
		i = Orbit_idx[j];
		if (i >= l) {
			cout << "orbit idx is out of range" << endl;
			exit(1);
		}
		cout << "Orbit " << j << " / " << nb_orbits
				<< " which is no " << i << ":" << endl;

		PC->get_set_by_level(depth, i, set);

		d = (*compute_function)(depth, set, compute_function_data);
		Data[j] = d;
	}

	FREE_lint(set);
}

void poset_classification_global::list_selected_set_of_orbits_at_level(
	int depth,
	int nb_orbits, int *Orbit_idx,
	int f_has_print_function,
	void (*print_function)(std::ostream &ost,
			int len, long int *S, void *data),
	void *print_function_data,
	int f_show_orbit_decomposition, int f_show_stab,
	int f_save_stab, int f_show_whole_orbit)
{
	int l, i, j;

	l = PC->nb_orbits_at_level(depth);

	cout << "listing a set of " << nb_orbits
			<< " orbits at depth " << depth << ":" << endl;
	for (j = 0; j < nb_orbits; j++) {
		i = Orbit_idx[j];
		if (i >= l) {
			cout << "orbit idx is out of range" << endl;
			exit(1);
		}
		cout << "Orbit " << j << " / " << nb_orbits
				<< " which is no " << i << ":" << endl;
		list_whole_orbit(depth, i,
			f_has_print_function, print_function, print_function_data,
			f_show_orbit_decomposition, f_show_stab,
			f_save_stab, f_show_whole_orbit);
	}
}

void poset_classification_global::test_property(
		int depth,
	int (*test_property_function)(
			int len, long int *S, void *data),
	void *test_property_data,
	int &nb, int *&Orbit_idx)
{
	int N, i;
	long int *set;

	set = NEW_lint(depth);
	N = PC->nb_orbits_at_level(depth);
	Orbit_idx = NEW_int(N);
	nb = 0;
	for (i = 0; i < N; i++) {
		PC->get_set_by_level(depth, i, set);
		if ((*test_property_function)(depth, set, test_property_data)) {
			Orbit_idx[nb++] = i;
		}
	}
	FREE_lint(set);
}

#if 0
void poset_classification::print_schreier_vectors_at_depth(
		int depth, int verbose_level)
{
	int i, l;

	l = nb_orbits_at_level(depth);
	for (i = 0; i < l; i++) {
		print_schreier_vector(depth, i, verbose_level);
		}
}

void poset_classification::print_schreier_vector(
		int depth,
		int orbit_idx, int verbose_level)
{
	int *set;
	int len;
	//strong_generators *Strong_gens;
	longinteger_object Len, L, go;
	//longinteger_domain D;

	set = NEW_int(depth);

	orbit_length(orbit_idx, depth, Len);
	len = orbit_length_as_int(orbit_idx, depth);
	L.create(len);

	get_stabilizer_order(depth, orbit_idx, go);


	cout << "orbit " << orbit_idx << " / " << nb_orbits_at_level(depth)
			<< " (=node " << first_poset_orbit_node_at_level[depth] + orbit_idx
			<< ") at depth " << depth << " has length " << Len << " : ";

	get_set_by_level(depth, orbit_idx, set);
	int_set_print(cout, set, depth);
	cout << "_" << go << endl;

	cout << "schreier tree:" << endl;

	int *sv;


	sv = root[first_poset_orbit_node_at_level[depth] + orbit_idx].sv;

	if (sv == NULL) {
		cout << "No schreier vector available" << endl;
		}

	schreier_vector_print_tree(sv, 0 /*verbose_level */);
}
#endif

void poset_classification_global::list_whole_orbit(
	int depth, int orbit_idx,
	int f_has_print_function,
	void (*print_function)(std::ostream &ost,
			int len, long int *S, void *data),
	void *print_function_data,
	int f_show_orbit_decomposition, int f_show_stab,
	int f_save_stab, int f_show_whole_orbit)
{
	long int *set;
	int rank, len;
	groups::strong_generators *Strong_gens;
	algebra::ring_theory::longinteger_object Len, L, go;
	algebra::ring_theory::longinteger_domain D;

	set = NEW_lint(depth);

	PC->orbit_length(orbit_idx, depth, Len);
	len = PC->orbit_length_as_int(orbit_idx, depth);
	L.create(len);

	PC->get_stabilizer_order(depth, orbit_idx, go);


	cout << "poset_classification_global::list_whole_orbit "
			"depth " << depth
			<< " orbit " << orbit_idx
			<< " / " << PC->nb_orbits_at_level(depth)
			<< " (=node " << PC->get_Poo()->first_node_at_level(depth) + orbit_idx
			<< ") at depth " << depth << " has length " << Len << " : ";

	PC->get_set_by_level(depth, orbit_idx, set);
	Lint_vec_print(cout, set, depth);
	cout << "_" << go << " ";

	//print_lex_rank(set, depth);
	cout << endl;

	if (f_has_print_function) {
		(*print_function)(cout, depth, set, print_function_data);
	}

	PC->get_stabilizer_generators(
			Strong_gens, depth, orbit_idx,
			0 /* verbose_level*/);


	if (f_show_orbit_decomposition) {
		if (PC->get_poset()->f_subset_lattice) {
			cout << "poset_classification_global::list_whole_orbit "
					"orbits on the set:" << endl;

			// ToDo:
			//Strong_gens->compute_and_print_orbits_on_a_given_set(
			//		Poset->A2, set, depth, 0 /* verbose_level*/);
		}
		else {
			cout << "subspace_lattice not yet implemented" << endl;
		}

		cout << "poset_classification_global::list_whole_orbit "
				"orbits in the original "
				"action on the whole space:" << endl;
		Strong_gens->compute_and_print_orbits(
				PC->get_poset()->A,
				0 /* verbose_level*/);
	}

	if (f_show_stab) {
		cout << "The stabilizer is generated by:" << endl;
		Strong_gens->print_generators(cout, 0 /* verbose_level*/);
	}

	if (f_save_stab) {
		string fname;

		fname = PC->get_problem_label_with_path()
				+ "_stab_"
				+ std::to_string(depth)
				+ "_" + std::to_string(orbit_idx)
				+ ".bin";

		cout << "saving stabilizer poset_classifications "
				"to file " << fname << endl;
		Strong_gens->write_file(fname, PC->get_control()->verbose_level);
	}


	if (f_show_whole_orbit) {
		int max_len;
		if (len > 1000) {
			max_len = 10;
		}
		else {
			max_len = len;
		}

		if (D.compare(L, Len) != 0) {
			cout << "orbit is too long to show" << endl;
		}
		else {
			for (rank = 0; rank < max_len; rank++) {
				PC->orbit_element_unrank(
						depth, orbit_idx,
						rank, set,
						0 /* verbose_level */);
				cout << setw(5) << rank << " : ";
				Lint_vec_set_print(
						cout, set, depth);
				cout << endl;
			}
			if (max_len < len) {
				cout << "output truncated" << endl;
			}
		}
	}

	FREE_lint(set);
	FREE_OBJECT(Strong_gens);
	cout << "poset_classification_global::list_whole_orbit done" << endl;
}

void poset_classification_global::get_whole_orbit(
	int depth, int orbit_idx,
	long int *&Orbit, int &orbit_length,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int rank;
	algebra::ring_theory::longinteger_object Len, L, go;
	algebra::ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "poset_classification_global::get_whole_orbit" << endl;
	}
	PC->orbit_length(orbit_idx, depth, Len);
	orbit_length = PC->orbit_length_as_int(orbit_idx, depth);
	L.create(orbit_length);

	if (f_v) {
		cout << "poset_classification_global::get_whole_orbit "
				"orbit_length=" << orbit_length << endl;
	}
	if (D.compare(L, Len) != 0) {
		cout << "poset_classification_global::get_whole_orbit "
				"orbit is too long" << endl;
		exit(1);
	}

	Orbit = NEW_lint(orbit_length * depth);
	for (rank = 0; rank < orbit_length; rank++) {
		if (f_v) {
			cout << "poset_classification_global::get_whole_orbit "
					"element " << rank << " / " << orbit_length << endl;
		}
		PC->orbit_element_unrank(
				depth, orbit_idx,
				rank,
				Orbit + rank * depth,
				0 /* verbose_level */);
	}
	if (f_v) {
		cout << "poset_classification_global::get_whole_orbit done" << endl;
	}
}

void poset_classification_global::compute_Kramer_Mesner_matrix(
		int t, int k,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "poset_classification_global::compute_Kramer_Mesner_matrix" << endl;
		cout << "poset_classification_global::compute_Kramer_Mesner_matrix verbose_level = " << verbose_level << endl;
	}


	// compute Stack of neighboring Kramer Mesner matrices
	long int **pM;
	int *Nb_rows, *Nb_cols;
	int h;

	pM = NEW_plint(k);
	Nb_rows = NEW_int(k);
	Nb_cols = NEW_int(k);
	for (h = 0; h < k; h++) {

		if (f_v) {
			cout << "poset_classification_global::compute_Kramer_Mesner_matrix "
					"level " << h << " / " << k
					<< " before Kramer_Mesner_matrix_neighboring" << endl;
		}
		Kramer_Mesner_matrix_neighboring(
				h, pM[h], Nb_rows[h], Nb_cols[h],
				verbose_level - 2);

		if (f_v) {
			cout << "poset_classification_global::compute_Kramer_Mesner_matrix "
					"matrix level " << h << " computed" << endl;
	#if 0
			int j;
			for (i = 0; i < Nb_rows[h]; i++) {
				for (j = 0; j < Nb_cols[h]; j++) {
					cout << pM[h][i * Nb_cols[h] + j];
					if (j < Nb_cols[h]) {
						cout << ",";
					}
				}
				cout << endl;
			}
	#endif
		}
	}

	long int *Mtk;
	int nb_r, nb_c;


	if (f_v) {
		cout << "poset_classification_global::compute_Kramer_Mesner_matrix "
				"before Mtk_from_MM" << endl;
	}
	Mtk_from_MM(pM, Nb_rows, Nb_cols,
			t, k,
			Mtk,
			nb_r, nb_c,
			verbose_level);
	if (f_v) {
		cout << "poset_classification_global::compute_Kramer_Mesner_matrix "
				"after Mtk_from_MM" << endl;
		cout << "poset_classification_global::compute_Kramer_Mesner_matrix "
				"M_{" << t << "," << k << "} "
				"has size " << nb_r << " x " << nb_c << "." << endl;

#if 0
		int j;
		for (i = 0; i < nb_r; i++) {
			for (j = 0; j < nb_c; j++) {
				cout << Mtk[i * nb_c + j];
				if (j < nb_c - 1) {
					cout << ",";
				}
			}
			cout << endl;
		}
#endif

	}

	long int *Mtk_inf;


	if (f_v) {
		cout << "poset_classification_global::compute_Kramer_Mesner_matrix "
				"before Asup_to_Ainf" << endl;
	}
	Asup_to_Ainf(
			t, k,
			Mtk, Mtk_inf, verbose_level);
	if (f_v) {
		cout << "poset_classification_global::compute_Kramer_Mesner_matrix "
				"after Asup_to_Ainf" << endl;
	}


	other::orbiter_kernel_system::file_io Fio;
	int i;

	string fname;

	fname = PC->get_problem_label() + "_KM_" + std::to_string(t) + "_" + std::to_string(k) + ".csv";

	Fio.Csv_file_support->lint_matrix_write_csv(
			fname, Mtk, nb_r, nb_c);

	//Mtk.print(cout);
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


	fname = PC->get_problem_label() + "_KM_inf_" + std::to_string(t) + "_" + std::to_string(k) + ".csv";

	Fio.Csv_file_support->lint_matrix_write_csv(
			fname, Mtk_inf, nb_r, nb_c);

	//Mtk.print(cout);
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;



	if (f_v) {
		cout << "poset_classification_global::compute_Kramer_Mesner_matrix "
				"computing Kramer Mesner matrices done" << endl;
	}
	for (i = 0; i < k; i++) {
		FREE_lint(pM[i]);
	}
	FREE_plint(pM);
	FREE_lint(Mtk);
	FREE_lint(Mtk_inf);

	if (f_v) {
		cout << "poset_classification_global::compute_Kramer_Mesner_matrix done" << endl;
	}
}

void poset_classification_global::Plesken_matrix_up(
		int depth,
		int *&P, int &N, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Nb;
	int *Fst;
	int *Pij;
	int i, j;
	int N1, N2;
	int a, b, cnt;

	if (f_v) {
		cout << "poset_classification_global::Plesken_matrix_up" << endl;
	}
	N = 0;
	Nb = NEW_int(depth + 1);
	Fst = NEW_int(depth + 2);
	Fst[0] = 0;
	for (i = 0; i <= depth; i++) {
		Nb[i] = PC->nb_orbits_at_level(i);
		Fst[i + 1] = Fst[i] + Nb[i];
		N += Nb[i];
	}
	P = NEW_int(N * N);
	for (i = 0; i <= depth; i++) {
		for (j = 0; j <= depth; j++) {

			Plesken_submatrix_up(
					i, j, Pij, N1, N2, verbose_level - 1);

			for (a = 0; a < N1; a++) {
				for (b = 0; b < N2; b++) {
					cnt = Pij[a * N2 + b];
					P[(Fst[i] + a) * N + Fst[j] + b] = cnt;
				}
			}
			FREE_int(Pij);
		}
	}
	if (f_v) {
		cout << "poset_classification_global::Plesken_matrix_up done" << endl;
	}
}

void poset_classification_global::Plesken_matrix_down(
		int depth,
		int *&P, int &N, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Nb;
	int *Fst;
	int *Pij;
	int i, j;
	int N1, N2;
	int a, b, cnt;

	if (f_v) {
		cout << "poset_classification_global::Plesken_matrix_down" << endl;
	}
	N = 0;
	Nb = NEW_int(depth + 1);
	Fst = NEW_int(depth + 2);
	Fst[0] = 0;
	for (i = 0; i <= depth; i++) {
		Nb[i] = PC->nb_orbits_at_level(i);
		Fst[i + 1] = Fst[i] + Nb[i];
		N += Nb[i];
	}
	P = NEW_int(N * N);
	for (i = 0; i <= depth; i++) {
		for (j = 0; j <= depth; j++) {

			Plesken_submatrix_down(i, j,
					Pij, N1, N2, verbose_level - 1);

			for (a = 0; a < N1; a++) {
				for (b = 0; b < N2; b++) {
					cnt = Pij[a * N2 + b];
					P[(Fst[i] + a) * N + Fst[j] + b] = cnt;
				}
			}
			FREE_int(Pij);
		}
	}
	if (f_v) {
		cout << "poset_classification_global::Plesken_matrix_down done" << endl;
	}
}

void poset_classification_global::Plesken_submatrix_up(
		int i, int j,
		int *&Pij, int &N1, int &N2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b;

	if (f_v) {
		cout << "poset_classification_global::Plesken_submatrix_up "
				"i=" << i << " j=" << j << endl;
	}
	N1 = PC->nb_orbits_at_level(i);
	N2 = PC->nb_orbits_at_level(j);
	Pij = NEW_int(N1 * N2);
	for (a = 0; a < N1; a++) {
		for (b = 0; b < N2; b++) {
			Pij[a * N2 + b] = count_incidences_up(
					i, a, j, b, verbose_level - 1);
		}
	}
	if (f_v) {
		cout << "poset_classification_global::Plesken_submatrix_up done" << endl;
	}
}

void poset_classification_global::Plesken_submatrix_down(
		int i, int j,
		int *&Pij, int &N1, int &N2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b;

	if (f_v) {
		cout << "poset_classification_global::Plesken_submatrix_down "
				"i=" << i << " j=" << j << endl;
	}
	N1 = PC->nb_orbits_at_level(i);
	N2 = PC->nb_orbits_at_level(j);
	Pij = NEW_int(N1 * N2);
	for (a = 0; a < N1; a++) {
		for (b = 0; b < N2; b++) {
			Pij[a * N2 + b] = count_incidences_down(
					i, a, j, b, verbose_level - 1);
		}
	}
	if (f_v) {
		cout << "poset_classification_global::Plesken_submatrix_down done" << endl;
	}
}

int poset_classification_global::count_incidences_up(
		int lvl1, int po1,
		int lvl2, int po2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int *set;
	long int *set1;
	long int *set2;
	int ol, i, cnt = 0;
	int f_contained;

	if (f_v) {
		cout << "poset_classification_global::count_incidences_up "
				"lvl1=" << lvl1 << " po1=" << po1
				<< " lvl2=" << lvl2 << " po2=" << po2 << endl;
	}
	if (lvl1 > lvl2) {
		return 0;
	}
	set = NEW_lint(lvl2 + 1);
	set1 = NEW_lint(lvl2 + 1);
	set2 = NEW_lint(lvl2 + 1);

	PC->orbit_element_unrank(
			lvl1, po1, 0 /*el1 */,
			set1, 0 /* verbose_level */);

	ol = PC->orbit_length_as_int(po2, lvl2);

	if (f_vv) {
		cout << "set1=";
		Lint_vec_print(cout, set1, lvl1);
		cout << endl;
	}

	for (i = 0; i < ol; i++) {

		Lint_vec_copy(set1, set, lvl1);


		PC->orbit_element_unrank(
				lvl2, po2, i, set2, 0 /* verbose_level */);

		if (f_vv) {
			cout << "set2 " << i << " / " << ol << "=";
			Lint_vec_print(cout, set2, lvl2);
			cout << endl;
		}

		f_contained = PC->poset_structure_is_contained(
				set, lvl1, set2, lvl2, verbose_level - 2);

		if (f_vv) {
			cout << "f_contained=" << f_contained << endl;
		}


		if (f_contained) {
			cnt++;
		}
	}


	FREE_lint(set);
	FREE_lint(set1);
	FREE_lint(set2);
	if (f_v) {
		cout << "poset_classification_global::count_incidences_up "
				"lvl1=" << lvl1 << " po1=" << po1
				<< " lvl2=" << lvl2 << " po2=" << po2
				<< " cnt=" << cnt << endl;
	}
	return cnt;
}

int poset_classification_global::count_incidences_down(
		int lvl1, int po1, int lvl2, int po2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int *set;
	long int *set1;
	long int *set2;
	int ol, i, cnt = 0;
	int f_contained;

	if (f_v) {
		cout << "poset_classification_global::count_incidences_down "
				"lvl1=" << lvl1 << " po1=" << po1
				<< " lvl2=" << lvl2 << " po2=" << po2 << endl;
	}
	if (lvl1 > lvl2) {
		return 0;
	}
	set = NEW_lint(lvl2 + 1);
	set1 = NEW_lint(lvl2 + 1);
	set2 = NEW_lint(lvl2 + 1);

	PC->orbit_element_unrank(
			lvl2, po2, 0 /*el1 */, set2,
			0 /* verbose_level */);

	ol = PC->orbit_length_as_int(po1, lvl1);

	if (f_vv) {
		cout << "set2=";
		Lint_vec_print(cout, set2, lvl2);
		cout << endl;
	}

	for (i = 0; i < ol; i++) {

		Lint_vec_copy(set2, set, lvl2);


		PC->orbit_element_unrank(
				lvl1, po1, i, set1,
				0 /* verbose_level */);

		if (f_vv) {
			cout << "set1 " << i << " / " << ol << "=";
			Lint_vec_print(cout, set1, lvl1);
			cout << endl;
		}


		f_contained = PC->poset_structure_is_contained(
				set1, lvl1, set, lvl2, verbose_level - 2);

		if (f_vv) {
			cout << "f_contained=" << f_contained << endl;
		}

		if (f_contained) {
			cnt++;
		}
	}


	FREE_lint(set);
	FREE_lint(set1);
	FREE_lint(set2);
	if (f_v) {
		cout << "poset_classification_global::count_incidences_down "
				"lvl1=" << lvl1 << " po1=" << po1
				<< " lvl2=" << lvl2 << " po2=" << po2
				<< " cnt=" << cnt << endl;
	}
	return cnt;
}

void poset_classification_global::Asup_to_Ainf(
		int t, int k,
		long int *M_sup, long int *&M_inf, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object quo, rem, aa, bb, cc;
	algebra::ring_theory::longinteger_object go;
	algebra::ring_theory::longinteger_object *go_t;
	algebra::ring_theory::longinteger_object *go_k;
	algebra::ring_theory::longinteger_object *ol_t;
	algebra::ring_theory::longinteger_object *ol_k;
	int Nt, Nk;
	int i, j;
	long int a, c;

	if (f_v) {
		cout << "poset_classification_global::Asup_to_Ainf" << endl;
	}
	Nt = PC->nb_orbits_at_level(t);
	Nk = PC->nb_orbits_at_level(k);
	PC->get_stabilizer_order(0, 0, go);

	M_inf = NEW_lint(Nt * Nk);

	if (f_v) {
		cout << "poset_classification_global::Asup_to_Ainf go=" << go << endl;
	}
	go_t = NEW_OBJECTS(algebra::ring_theory::longinteger_object, Nt);
	go_k = NEW_OBJECTS(algebra::ring_theory::longinteger_object, Nk);
	ol_t = NEW_OBJECTS(algebra::ring_theory::longinteger_object, Nt);
	ol_k = NEW_OBJECTS(algebra::ring_theory::longinteger_object, Nk);
	if (f_v) {
		cout << "poset_classification_global::Asup_to_Ainf "
				"computing orbit lengths t-orbits" << endl;
	}
	for (i = 0; i < Nt; i++) {
		PC->get_stabilizer_order(t, i, go_t[i]);
		D.integral_division_exact(go, go_t[i], ol_t[i]);
	}
	if (f_v) {
		cout << "i : go_t[i] : ol_t[i]" << endl;
		for (i = 0; i < Nt; i++) {
			cout << i << " : " << go_t[i] << " : " << ol_t[i] << endl;
		}
	}
	if (f_v) {
		cout << "poset_classification_global::Asup_to_Ainf "
				"computing orbit lengths k-orbits" << endl;
	}
	for (i = 0; i < Nk; i++) {
		PC->get_stabilizer_order(k, i, go_k[i]);
		D.integral_division_exact(go, go_k[i], ol_k[i]);
	}
	if (f_v) {
		cout << "i : go_k[i] : ol_k[i]" << endl;
		for (i = 0; i < Nk; i++) {
			cout << i << " : " << go_k[i] << " : " << ol_k[i] << endl;
		}
	}
	if (f_v) {
		cout << "poset_classification_global::Asup_to_Ainf "
				"computing Ainf" << endl;
	}
	for (i = 0; i < Nt; i++) {
		for (j = 0; j < Nk; j++) {
			a = M_sup[i * Nk + j];
			aa.create(a);
			D.mult(ol_t[i], aa, bb);
			D.integral_division(bb, ol_k[j], cc, rem, 0);
			if (!rem.is_zero()) {
				cout << "poset_classification_global::Asup_to_Ainf "
						"stabilizer order does not "
						"divide group order" << endl;
				cout << "i=" << i << " j=" << j
						<< " M_sup[i,j] = " << a
						<< " ol_t[i]=" << ol_t[i]
						<< " ol_k[j]=" << ol_k[j] << endl;
				exit(1);
			}
			c = cc.as_lint();
			M_inf[i * Nk + j] = c;
		}
	}
	if (f_v) {
		cout << "poset_classification_global::Asup_to_Ainf "
				"computing Ainf done" << endl;
	}
	FREE_OBJECTS(go_t);
	FREE_OBJECTS(go_k);
	FREE_OBJECTS(ol_t);
	FREE_OBJECTS(ol_k);
	if (f_v) {
		cout << "poset_classification_global::Asup_to_Ainf done" << endl;
	}
}

void poset_classification_global::test_for_multi_edge_in_classification_graph(
		int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, f, l, j, h1;

	if (f_v) {
		cout << "poset_classification_global::test_for_multi_edge_in_classification_graph "
				"depth=" << depth << endl;
	}
	for (i = 0; i <= depth; i++) {
		f = PC->get_Poo()->first_node_at_level(i);
		l = PC->nb_orbits_at_level(i);
		if (f_v) {
			cout << "poset_classification_global::test_for_multi_edge_in_classification_graph "
					"level=" << i << " with " << l << " nodes" << endl;
		}
		for (j = 0; j < l; j++) {
			poset_orbit_node *O;

			//O = &root[f + j];
			O = PC->get_node_ij(i, j);
			for (h1 = 0; h1 < O->get_nb_of_extensions(); h1++) {
				extension *E1 = O->get_E(h1); // O->E + h1;

				if (E1->get_type() != EXTENSION_TYPE_FUSION) {
					continue;
				}

				//cout << "fusion (" << f + j << "/" << h1 << ") ->
				// (" << E1->data1 << "/" << E1->data2 << ")" << endl;
				if (E1->get_data1() == f + j) {
					cout << "multi_edge detected ! level "
							<< i << " with " << l << " nodes, "
							"fusion (" << j << "/" << h1 << ") -> "
							"(" << E1->get_data1() - f << "/"
							<< E1->get_data2() << ")" << endl;
				}

#if 0
				for (h2 = 0; h2 < O->get_nb_of_extensions(); h2++) {
					extension *E2 = O->E + h2;

					if (E2->get_type() != EXTENSION_TYPE_FUSION) {
						continue;

					if (E2->data1 == E1->data1 && E2->data2 == E1->data2) {
						cout << "multiedge detected!" << endl;
						cout << "fusion (" << f + j << "/" << h1
								<< ") -> (" << E1->get_data1() << "/"
								<< E1->get_data2() << ")" << endl;
						cout << "fusion (" << f + j << "/" << h2
								<< ") -> (" << E2->get_data1() << "/"
								<< E2->get_data2() << ")" << endl;
					}
				}
#endif

			}
		}
		if (f_v) {
			cout << "poset_classification_global::test_for_multi_edge_in_classification_graph "
					"level=" << i << " with " << l << " nodes done" << endl;
		}
	}
	if (f_v) {
		cout << "poset_classification_global::test_for_multi_edge_in_classification_graph "
				"done" << endl;
	}
}

void poset_classification_global::Kramer_Mesner_matrix_neighboring(
		int level, long int *&M,
		int &nb_rows, int &nb_cols, int verbose_level)
// we assume that we don't use implicit fusion nodes
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f1, f2, i, j, k, I, J, len;
	poset_orbit_node *O;

	if (f_v) {
		cout << "poset_classification_global::Kramer_Mesner_matrix_neighboring "
				"level=" << level << endl;
	}

	f1 = PC->first_node_at_level(level);
	f2 = PC->first_node_at_level(level + 1);
	if (f_v) {
		cout << "poset_classification_global::Kramer_Mesner_matrix_neighboring "
				"f1=" << f1 << endl;
		cout << "poset_classification_global::Kramer_Mesner_matrix_neighboring "
				"f2=" << f2 << endl;
	}
	nb_rows = PC->nb_orbits_at_level(level);
	if (f_v) {
		cout << "poset_classification_global::Kramer_Mesner_matrix_neighboring "
				"nb_rows=" << nb_rows << endl;
	}
	if (nb_rows < 0) {
		cout << "poset_classification_global::Kramer_Mesner_matrix_neighboring nb_rows < 0" << endl;
		exit(1);
	}
	nb_cols = PC->nb_orbits_at_level(level + 1);
	if (f_v) {
		cout << "poset_classification_global::Kramer_Mesner_matrix_neighboring "
				"nb_cols=" << nb_cols << endl;
	}
	if (nb_cols < 0) {
		cout << "poset_classification_global::Kramer_Mesner_matrix_neighboring nb_cols < 0" << endl;
		exit(1);
	}

	M = NEW_lint(nb_rows * nb_cols);

	for (i = 0; i < nb_rows * nb_cols; i++) {
		M[i] = 0;
	}


	if (f_v) {
		cout << "poset_classification_global::Kramer_Mesner_matrix_neighboring "
				"the size of the matrix is " << nb_rows << " x " << nb_cols << endl;
	}

	for (i = 0; i < nb_rows; i++) {
		if (f_vv) {
			cout << "poset_classification_global::Kramer_Mesner_matrix_neighboring "
					"row i=" << i << " / " << nb_rows << endl;
		}
		I = f1 + i;
		O = PC->get_node(I);
		for (k = 0; k < O->get_nb_of_extensions(); k++) {
			if (f_vv) {
				cout << "poset_classification_global::Kramer_Mesner_matrix_neighboring "
						"row i=" << i << " / " << nb_rows << " extension "
						<< k << " / " << O->get_nb_of_extensions() << endl;
			}
			if (O->get_E(k)->get_type() == EXTENSION_TYPE_EXTENSION) {
				if (f_vv) {
					cout << "poset_classification_global::Kramer_Mesner_matrix_neighboring "
							"row i=" << i << " / " << nb_rows << " extension "
							<< k << " / " << O->get_nb_of_extensions()
							<< " type extension node" << endl;
				}
				len = O->get_E(k)->get_orbit_len();
				J = O->get_E(k)->get_data();
				j = J - f2;
				M[i * nb_cols + j] += len;
			}
			if (O->get_E(k)->get_type() == EXTENSION_TYPE_FUSION) {
				if (f_vv) {
					cout << "poset_classification_global::Kramer_Mesner_matrix_neighboring "
							"row i=" << i << " / " << nb_rows << " extension "
							<< k << " / " << O->get_nb_of_extensions()
							<< " type fusion" << endl;
				}
				// fusion node
				len = O->get_E(k)->get_orbit_len();

				int I1, ext1;
				poset_orbit_node *O1;

				I1 = O->get_E(k)->get_data1();
				ext1 = O->get_E(k)->get_data2();
				O1 = PC->get_node(I1);
				if (O1->get_E(ext1)->get_type() != EXTENSION_TYPE_EXTENSION) {
					cout << "poset_classification_global::Kramer_Mesner_matrix_neighboring "
							"O1->get_E(ext1)->type != EXTENSION_TYPE_EXTENSION "
							"something is wrong" << endl;
					exit(1);
				}
				J = O1->get_E(ext1)->get_data();

#if 0
				O->store_set(gen, level - 1);
					// stores a set of size level to gen->S
				gen->S[level] = O->E[k].pt;

				for (ii = 0; ii < level + 1; ii++) {
					gen->set[level + 1][ii] = gen->S[ii];
				}

				gen->A->element_one(gen->transporter->ith(level + 1), 0);

				J = O->apply_isomorphism(gen,
					level, I /* current_node */,
					//0 /* my_node */, 0 /* my_extension */, 0 /* my_coset */,
					k /* current_extension */, level + 1,
					false /* f_tolerant */,
					0/*verbose_level - 2*/);
				if (false) {
					cout << "after apply_isomorphism J=" << J << endl;
				}
#else

#endif




#if 0
				//cout << "fusion node:" << endl;
				//int_vec_print(cout, gen->S, level + 1);
				//cout << endl;
				gen->A->element_retrieve(O->E[k].data, gen->Elt1, 0);

				gen->A2->map_a_set(gen->S, gen->S0, level + 1, gen->Elt1, 0);
				//int_vec_print(cout, gen->S0, level + 1);
				//cout << endl;

				int_vec_heapsort(gen->S0, level + 1); //int_vec_sort(level + 1, gen->S0);

				//int_vec_print(cout, gen->S0, level + 1);
				//cout << endl;

				J = gen->find_poset_orbit_node_for_set(level + 1, gen->S0, 0);
#endif
				j = J - f2;
				M[i * nb_cols + j] += len;
			}
		}
	}
	if (f_v) {
		cout << "poset_classification_global::Kramer_Mesner_matrix_neighboring "
				"level=" << level << " done" << endl;
	}
}

void poset_classification_global::Mtk_via_Mtr_Mrk(
		int t, int r, int k,
		long int *Mtr, long int *Mrk, long int *&Mtk,
		int nb_r1, int nb_c1, int nb_r2, int nb_c2,
		int &nb_r3, int &nb_c3,
		int verbose_level)
// Computes $M_{tk}$ via a recursion formula:
// $M_{tk} = {{k - t} \choose {k - r}} \cdot M_{t,r} \cdot M_{r,k}$.
{
	int f_v = (verbose_level >= 1);
	int i, j, h, a, b, c, s = 0;
	combinatorics::other_combinatorics::combinatorics_domain C;

	if (f_v) {
		cout << "poset_classification_global::Mtk_via_Mtr_Mrk "
				"t = " << t << ", r = "
				<< r << ", k = " << k << endl;
	}
	if (nb_c1 != nb_r2) {
		cout << "poset_classification_global::Mtk_via_Mtr_Mrk "
				"nb_c1 != nb_r2" << endl;
		exit(1);
	}

	nb_r3 = nb_r1;
	nb_c3 = nb_c2;
	Mtk = NEW_lint(nb_r3 * nb_c3);
	for (i = 0; i < nb_r3; i++) {
		for (j = 0; j < nb_c3; j++) {
			c = 0;
			for (h = 0; h < nb_c1; h++) {
				a = Mtr[i * nb_c1 + h];
				b = Mrk[h * nb_c2 + j];
				c += a * b;
			}
			Mtk[i * nb_c3 + j] = c;
		}
	}


	//Mtk.mult(Mtr, Mrk);

	// Mtk := {(k - t) \atop (k - r)} * M_t,k


	algebra::ring_theory::longinteger_object S;

	if (PC->get_poset()->f_subset_lattice) {
		C.binomial(
				S, k - t, k - r,
				0/* verbose_level*/);
		s = S.as_lint();
	}
	else if (PC->get_poset()->f_subspace_lattice) {
		C.q_binomial(
				S, k - t, r - t, PC->get_poset()->VS->F->q,
				0/* verbose_level*/);
		s = S.as_lint();
	}
	else {
		cout << "poset_classification_global::Mtk_via_Mtr_Mrk "
				"unknown type of lattice" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "poset_classification_global::Mtk_via_Mtr_Mrk "
				"dividing by " << s << endl;
	}


	for (i = 0; i < nb_r3; i++) {
		for (j = 0; j < nb_c3; j++) {
			Mtk[i * nb_c3 + j] /= s;
		}
	}
	if (f_v) {
		cout << "poset_classification_global::Mtk_via_Mtr_Mrk matrix "
				"M_{" << t << "," << k << "} "
						"of format " << nb_r3 << " x " << nb_c3
						<< " has been computed" << endl;
		}
}

void poset_classification_global::Mtk_from_MM(
		long int **pM,
	int *Nb_rows, int *Nb_cols,
	int t, int k,
	long int *&Mtk, int &nb_r, int &nb_c,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "poset_classification_global::Mtk_from_MM "
				"t = " << t << ", k = " << k << endl;
	}
	if (k == t) {
		cout << "poset_classification_global::Mtk_from_MM "
				"k == t" << endl;
		exit(1);
	}

	long int *T;
	long int *T2;
	int Tr, Tc;
	int T2r, T2c;

	Tr = Nb_rows[t];
	Tc = Nb_cols[t];

	T = NEW_lint(Tr * Tc);
	for (i = 0; i < Tr; i++) {
		for (j = 0; j < Tc; j++) {
			T[i * Tc + j] = pM[t][i * Tc + j];
		}
	}
	if (f_v) {
		cout << "poset_classification_global::Mtk_from_MM "
				"Tr=" << Tr << " Tc=" << Tc << endl;
	}

	if (t + 1 < k) {
		for (i = t + 2; i <= k; i++) {

			if (f_v) {
				cout << "poset_classification_global::Mtk_from_MM "
						"i = " << i << " calling Mtk_via_Mtr_Mrk" << endl;
			}

			Mtk_via_Mtr_Mrk(
					t, i - 1, i,
				T, pM[i - 1], T2,
				Tr, Tc, Nb_rows[i - 1], Nb_cols[i - 1], T2r, T2c,
				verbose_level - 1);

			FREE_lint(T);
			T = T2;
			Tr = T2r;
			Tc = T2c;
			T2 = NULL;
		}
		Mtk = T;
		nb_r = Tr;
		nb_c = Tc;
	}
	else {
		Mtk = T;
		nb_r = Tr;
		nb_c = Tc;
	}


	if (f_v) {
		cout << "poset_classification_global::Mtk_from_MM "
				"nb_r=" << nb_r << " nb_c=" << nb_c << endl;
	}

	if (f_v) {
		cout << "poset_classification_global::Mtk_from_MM "
				"t = " << t << ", k = " << k << " done" << endl;
	}
}





}}}


