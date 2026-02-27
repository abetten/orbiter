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

	l = PC->get_Poo()->nb_orbits_at_level(lvl);
	if (f_v) {
		cout << "collecting the automorphism group orders of "
				<< l << " orbits" << endl;
	}
	nb_agos = 0;
	agos = NULL;
	multiplicities = NULL;
	for (i = 0; i < l; i++) {
		PC->get_Poo()->get_stabilizer_order(lvl, i, ago);
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

	nb_orbits = PC->get_Poo()->nb_orbits_at_level(k);
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

	l = PC->get_Poo()->nb_orbits_at_level(depth);

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
	l = PC->get_Poo()->nb_orbits_at_level(depth);

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

		PC->get_Poo()->get_set_by_level(depth, i, set);

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

	l = PC->get_Poo()->nb_orbits_at_level(depth);

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
	N = PC->get_Poo()->nb_orbits_at_level(depth);
	Orbit_idx = NEW_int(N);
	nb = 0;
	for (i = 0; i < N; i++) {
		PC->get_Poo()->get_set_by_level(depth, i, set);
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

	PC->get_Poo()->orbit_length(orbit_idx, depth, Len);
	len = PC->get_Poo()->orbit_length_as_int(orbit_idx, depth);
	L.create(len);

	PC->get_Poo()->get_stabilizer_order(depth, orbit_idx, go);


	cout << "poset_classification_global::list_whole_orbit "
			"depth " << depth
			<< " orbit " << orbit_idx
			<< " / " << PC->get_Poo()->nb_orbits_at_level(depth)
			<< " (=node " << PC->get_Poo()->first_node_at_level(depth) + orbit_idx
			<< ") at depth " << depth << " has length " << Len << " : ";

	PC->get_Poo()->get_set_by_level(depth, orbit_idx, set);
	Lint_vec_print(cout, set, depth);
	cout << "_" << go << " ";

	//print_lex_rank(set, depth);
	cout << endl;

	if (f_has_print_function) {
		(*print_function)(cout, depth, set, print_function_data);
	}

	PC->get_Poo()->get_stabilizer_generators(
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
				PC->get_Poo()->orbit_element_unrank(
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
	PC->get_Poo()->orbit_length(orbit_idx, depth, Len);
	orbit_length = PC->get_Poo()->orbit_length_as_int(orbit_idx, depth);
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
		PC->get_Poo()->orbit_element_unrank(
				depth, orbit_idx,
				rank,
				Orbit + rank * depth,
				0 /* verbose_level */);
	}
	if (f_v) {
		cout << "poset_classification_global::get_whole_orbit done" << endl;
	}
}


void poset_classification_global::recognize(
		std::string &set_to_recognize,
		int h, int nb_to_recognize,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification_global::recognize" << endl;
	}
	long int *recognize_set;
	int recognize_set_sz;
	int orb;
	long int *canonical_set;
	int *Elt_transporter;
	int *Elt_transporter_inv;

	cout << "poset_classification_global::recognize "
			"recognize " << h << " / " << nb_to_recognize << endl;
	Lint_vec_scan(set_to_recognize, recognize_set, recognize_set_sz);
	cout << "poset_classification_global::recognize "
			"input set = " << h << " / " << nb_to_recognize << " : ";
	Lint_vec_print(cout, recognize_set, recognize_set_sz);
	cout << endl;

	canonical_set = NEW_lint(recognize_set_sz);
	Elt_transporter = NEW_int(PC->get_A()->elt_size_in_int);
	Elt_transporter_inv = NEW_int(PC->get_A()->elt_size_in_int);


	data_structures_groups::set_and_stabilizer *SaS_original;
	data_structures_groups::set_and_stabilizer *SaS_canonical;
	int orbit_at_level;


	PC->identify_and_get_stabilizer(
			recognize_set, recognize_set_sz, Elt_transporter,
			orbit_at_level,
			SaS_original,
			SaS_canonical,
			verbose_level);


	orb = PC->trace_set(
			recognize_set,
		recognize_set_sz, recognize_set_sz /* level */,
		canonical_set, Elt_transporter,
		0 /*verbose_level */);

	cout << "poset_classification_global::recognize "
			"recognize " << h << " / " << nb_to_recognize << endl;
	cout << "poset_classification_global::recognize "
			"canonical set = ";
	Lint_vec_print(cout, canonical_set, recognize_set_sz);
	cout << endl;
	cout << "poset_classification_global::recognize "
			"is orbit " << orb << endl;
	cout << "poset_classification_global::recognize "
			"recognize " << h << " / " << nb_to_recognize << endl;
	cout << "poset_classification_global::recognize "
			"transporter:" << endl;
	PC->get_A()->Group_element->element_print_quick(Elt_transporter, cout);

	PC->get_A()->Group_element->element_invert(Elt_transporter, Elt_transporter_inv, 0);
	cout << "poset_classification_global::recognize "
			"recognize " << h << " / " << nb_to_recognize << endl;
	cout << "poset_classification_global::recognize "
			"transporter inverse:" << endl;
	PC->get_A()->Group_element->element_print_quick(Elt_transporter_inv, cout);

	cout << "poset_classification_global::recognize "
			"Stabilizer of the given set:" << endl;
	SaS_original->print_generators_tex(cout);

	cout << "poset_classification_global::recognize "
			"Stabilizer of the canonical set:" << endl;
	SaS_canonical->print_generators_tex(cout);

	FREE_lint(canonical_set);
	FREE_int(Elt_transporter);
	FREE_int(Elt_transporter_inv);
	FREE_lint(recognize_set);

	if (f_v) {
		cout << "poset_classification_global::recognize before FREE_OBJECT" << endl;
	}
	FREE_OBJECT(SaS_original);
	FREE_OBJECT(SaS_canonical);

	if (f_v) {
		cout << "poset_classification_global::recognize done" << endl;
	}
}



void poset_classification_global::orbits_on_k_sets(
		layer3_group_actions::combinatorics_with_groups::poset_with_group_action *Poset,
		poset_classification_control *Control,
		int k, long int *&orbit_reps,
		int &nb_orbits, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	poset_classification *Gen;

	if (f_v) {
		cout << "poset_classification_global::orbits_on_k_sets" << endl;
	}

	Gen = orbits_on_k_sets_compute(
			Poset,
			Control,
		k, verbose_level);
	if (f_v) {
		cout << "poset_classification_global::orbits_on_k_sets "
				"done with orbits_on_k_sets_compute" << endl;
	}

	Gen->get_Poo()->get_orbit_representatives(k, nb_orbits,
			orbit_reps, verbose_level);


	if (f_v) {
		cout << "poset_classification_global::orbits_on_k_sets "
				"we found "
				<< nb_orbits << " orbits on " << k << "-sets" << endl;
	}

	FREE_OBJECT(Gen);
	if (f_v) {
		cout << "poset_classification_global::orbits_on_k_sets done" << endl;
	}
}

poset_classification *poset_classification_global::orbits_on_k_sets_compute(
		layer3_group_actions::combinatorics_with_groups::poset_with_group_action *Poset,
		poset_classification_control *Control,
		int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	poset_classification *Gen;


	if (f_v) {
		cout << "poset_classification_global::orbits_on_k_sets_compute" << endl;
	}
	Gen = NEW_OBJECT(poset_classification);


	if (f_v) {
		cout << "poset_classification_global::orbits_on_k_sets_compute "
				"before Gen->initialize_and_allocate_root_node" << endl;
	}
	Gen->initialize_and_allocate_root_node(
			Control,
			Poset,
			k /* sz */,
			verbose_level - 1);
	if (f_v) {
		cout << "poset_classification_global::orbits_on_k_sets_compute "
				"after Gen->initialize_and_allocate_root_node" << endl;
	}

	if (f_v) {
		cout << "poset_classification_global::orbits_on_k_sets_compute generators = ";
		Poset->Strong_gens->print_generators_tex(cout);
		Poset->Strong_gens->print_generators(cout, 0 /* verbose_level */);
	}


	other::orbiter_kernel_system::os_interface Os;
	int schreier_depth = k;
	int f_use_invariant_subset_if_available = true;
	int f_debug = false;
	int t0 = Os.os_ticks();

	if (f_v) {
		cout << "poset_classification_global::orbits_on_k_sets_compute "
				"before Gen->poset_classification_main" << endl;
	}
	Gen->poset_classification_main(t0,
		schreier_depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level - 1);
	if (f_v) {
		cout << "poset_classification_global::orbits_on_k_sets_compute "
				"after Gen->poset_classification_main" << endl;
	}


	if (f_v) {
		cout << "poset_classification_global::orbits_on_k_sets_compute done" << endl;
	}
	return Gen;
}





}}}


