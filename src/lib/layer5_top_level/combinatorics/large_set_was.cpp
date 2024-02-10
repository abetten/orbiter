/*
 * large_set_was.cpp
 *
 *  Created on: May 26, 2021
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {




large_set_was::large_set_was()
{
	Descr = NULL;

	LS = NULL;

	H_gens = NULL;

	H_orbits = NULL;

	N_gens = NULL;

	N_orbits = NULL;

	orbit_length = 0;
	nb_of_orbits_to_choose = 0;
	type_idx = 0;
	Orbit1 = NULL;
	Orbit2 = NULL;

	A_on_orbits = NULL;
	A_on_orbits_restricted = NULL;


	// used in do_normalizer_on_orbits_of_a_given_length_multiple_orbits::
	PC = NULL;
	Control = NULL;
	Poset = NULL;

	orbit_length2 = 0;
	type_idx2 = 0;

	selected_type_idx = 0;
}





large_set_was::~large_set_was()
{
}

void large_set_was::init(
		large_set_was_description *Descr,
		large_set_classify *LS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_was::init" << endl;
	}


	large_set_was::Descr = Descr;
	large_set_was::LS = LS;



	H_gens = NEW_OBJECT(groups::strong_generators);

	if (f_v) {
		cout << "design_activity::do_load_table before H_gens->init_from_data_with_go" << endl;
	}
	H_gens->init_from_data_with_go(
			LS->DC->A, Descr->H_generators_text,
			Descr->H_go,
			verbose_level);
	if (f_v) {
		cout << "design_activity::do_load_table after H_gens->init_from_data_with_go" << endl;
	}


	if (f_v) {
		cout << "large_set_was::init "
				"computing orbits on reduced set of designs:" << endl;
	}

	if (!Descr->f_prefix) {
		cout << "please use -prefix" << endl;
		exit(1);
	}
	if (!Descr->f_selected_orbit_length) {
		cout << "please use -selected_orbit_length" << endl;
		exit(1);
	}



	H_orbits = NEW_OBJECT(groups::orbits_on_something);

	H_orbits->init(LS->A_on_designs,
			H_gens,
			false /* f_load_save */,
			Descr->prefix,
			verbose_level);

	// computes all orbits and classifies the orbits by their length



	if (f_v) {
		cout << "large_set_was::init "
				"orbits of H on the set of designs are:" << endl;
		H_orbits->report_classified_orbit_lengths(cout);
	}

	//int prev_nb;

	if (f_v) {
		cout << "large_set_was::init after OoS->test_orbits_of_a_certain_length "
				"the number of orbits before filtering is " << endl;
		int type_idx;

		for (type_idx = 0; type_idx < H_orbits->Classify_orbits_by_length->nb_types; type_idx++) {
			cout << type_idx << " : " << H_orbits->Classify_orbits_by_length->Set_partition->Set_size[type_idx] << endl;
		}
	}

	if (f_v) {
		cout << "large_set_was::init before OoS->test_all_orbits_by_length" << endl;
	}
	H_orbits->test_all_orbits_by_length(
			large_set_was_design_test_orbit,
			this /* *test_function_data*/,
			verbose_level);
	if (f_v) {
		cout << "large_set_was::init after OoS->test_all_orbits_by_length" << endl;
	}


	if (f_v) {
		cout << "large_set_was::init after OoS->test_orbits_of_a_certain_length "
				"the number of orbits after filtering is " << endl;
		int type_idx;

		for (type_idx = 0; type_idx < H_orbits->Classify_orbits_by_length->nb_types; type_idx++) {
			cout << type_idx << " : " << H_orbits->Classify_orbits_by_length->Set_partition->Set_size[type_idx] << endl;
		}

		//int type_idx;

		for (type_idx = 0; type_idx < H_orbits->Classify_orbits_by_length->nb_types; type_idx++) {
			H_orbits->print_orbits_of_a_certain_length(H_orbits->Classify_orbits_by_length->data_values[type_idx]);
		}

	}


	if (f_v) {
		cout << "large_set_was::init done" << endl;
	}
}




void large_set_was::do_normalizer_on_orbits_of_a_given_length(
		int orbit_length,
		int nb_of_orbits_to_choose,
		poset_classification::poset_classification_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length, "
				"orbit_length=" << orbit_length << " nb_of_orbits_to_choose=" << nb_of_orbits_to_choose << endl;
	}
	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length control=" << endl;
		Control->print();
	}

	large_set_was::orbit_length = orbit_length;
	large_set_was::nb_of_orbits_to_choose = nb_of_orbits_to_choose;
	type_idx = H_orbits->get_orbit_type_index(orbit_length);

	Orbit1 = NEW_lint(orbit_length);
	Orbit2 = NEW_lint(orbit_length);

	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length computing orbits "
				"of normalizer on orbits of length " << orbit_length
				<< ", type_idx=" << type_idx
				<< ", number of orbits = " << H_orbits->Classify_orbits_by_length->Set_partition->Set_size[type_idx] << endl;
	}


	N_gens = NEW_OBJECT(groups::strong_generators);

	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length before H_gens->init_from_data_with_go" << endl;
	}
	N_gens->init_from_data_with_go(
			LS->DC->A, Descr->N_generators_text,
			Descr->N_go,
			verbose_level);
	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length after H_gens->init_from_data_with_go" << endl;
	}


	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length normalizer has order ";
		N_gens->print_group_order(cout);
		cout << endl;
	}




	//action *A_on_orbits;
	//action *A_on_orbits_restricted;

	//A_on_orbits = NEW_OBJECT(actions::action);
	A_on_orbits = LS->A_on_designs->Induced_action->induced_action_on_orbits(
			H_orbits->Sch /* H_orbits_on_spreads*/,
			true /*f_play_it_safe*/,
			verbose_level - 1);

	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_on_orbits");
	label_of_set_tex.assign("\\_on\\_orbits");


	A_on_orbits_restricted = A_on_orbits->Induced_action->restricted_action(
			H_orbits->Classify_orbits_by_length->Set_partition->Sets[type_idx],
			H_orbits->Classify_orbits_by_length->Set_partition->Set_size[type_idx],
			label_of_set, label_of_set_tex,
			verbose_level);



	if (nb_of_orbits_to_choose == 1) {

		do_normalizer_on_orbits_of_a_given_length_single_orbit(
				orbit_length,
				verbose_level);

	}
	else {

		do_normalizer_on_orbits_of_a_given_length_multiple_orbits(
				orbit_length,
				nb_of_orbits_to_choose,
				Control,
				verbose_level);
	}

#if 0
	FREE_OBJECT(A_on_orbits_restricted);
	A_on_orbits_restricted = NULL;

	FREE_OBJECT(A_on_orbits);
	A_on_orbits = NULL;

	FREE_lint(Orbit1);
	Orbit1 = NULL;

	FREE_lint(Orbit2);
	Orbit2 = NULL;
#endif

	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length "
				"computing orbits of normalizer done" << endl;
	}


	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length done" << endl;
	}

}

void large_set_was::do_normalizer_on_orbits_of_a_given_length_single_orbit(
		int orbit_length,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length_single_orbit, "
				"orbit_length=" << orbit_length << endl;
	}

	actions::action_global AG;
	groups::schreier *Sch;

	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length before "
				"compute_orbits_on_points for the restricted action "
				"on the good orbits" << endl;
	}

	AG.compute_orbits_on_points(
			A_on_orbits_restricted,
			Sch, N_gens->gens,
			verbose_level - 1);

	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length "
				"the number of orbits of the normalizer on the "
				"good orbits is " << Sch->nb_orbits << endl;
		Sch->print_and_list_orbits_tex(cout);

#if 0
		cout << "printing orbits through Design_table_reduced_idx:" << endl;
		Sch->print_and_list_orbits_using_labels(
				cout, Design_table_reduced_idx);
#endif
	}

	{
		long int *Orbits_under_N;
		orbiter_kernel_system::file_io Fio;
		string fname_out;
		int i, a, l;


		Orbits_under_N = NEW_lint(Sch->nb_orbits * 2);

		fname_out = Descr->prefix + "_N_orbit_reps.csv";

		for (i = 0; i < Sch->nb_orbits; i++) {
			l = Sch->orbit_len[i];
			a = Sch->orbit[Sch->orbit_first[i]];
			Orbits_under_N[2 * i + 0] = a;
			Orbits_under_N[2 * i + 1] = l;
		}
		Fio.Csv_file_support->lint_matrix_write_csv(
				fname_out, Orbits_under_N, Sch->nb_orbits, 2);

		FREE_lint(Orbits_under_N);
	}

	FREE_OBJECT(Sch);

	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length_single_orbit done" << endl;
	}
}


void large_set_was::do_normalizer_on_orbits_of_a_given_length_multiple_orbits(
		int orbit_length,
		int nb_of_orbits_to_choose,
		poset_classification::poset_classification_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length_multiple_orbits, "
				"orbit_length=" << orbit_length << " nb_of_orbits_to_choose=" << nb_of_orbits_to_choose << endl;
	}


	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length_multiple_orbits before "
				"compute_orbits_on_points for the restricted action "
				"on the good orbits" << endl;
	}


	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);

#if 0
	Control = NEW_OBJECT(poset_classification_control);
	Control->f_depth = true;
	Control->depth = nb_of_orbits_to_choose;
#endif

#if 0
	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		cout << "please use option -poset_classification_control" << endl;
		exit(1);
	}
#endif
	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length_multiple_orbits control=" << endl;
		Control->print();
	}


	Poset->init_subset_lattice(
			LS->DC->A,
			A_on_orbits_restricted,
			N_gens,
			verbose_level);

	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length_multiple_orbits before "
				"Poset->add_testing_without_group" << endl;
		}
	Poset->add_testing_without_group(
			large_set_was_normalizer_orbits_early_test_func_callback,
			this /* void *data */,
			verbose_level);



	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length_multiple_orbits "
				"before Poset->orbits_on_k_sets_compute, nb_of_orbits_to_choose=" << nb_of_orbits_to_choose << endl;
	}
	PC = Poset->orbits_on_k_sets_compute(
			Control,
			nb_of_orbits_to_choose,
			verbose_level);
	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length_multiple_orbits "
				"after Poset->orbits_on_k_sets_compute" << endl;
	}

	//FREE_OBJECT(Control);

	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length_multiple_orbits done" << endl;
	}


	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length_multiple_orbits done" << endl;
	}
}




void large_set_was::create_graph_on_orbits_of_length(
		std::string &fname, int orbit_length,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_was::create_graph_on_orbits_of_length" << endl;
	}

	graph_theory::colored_graph *CG;

	H_orbits->create_graph_on_orbits_of_a_certain_length(
		CG,
		fname,
		orbit_length,
		selected_type_idx,
		false /*f_has_user_data*/, NULL /* int *user_data */, 0 /* user_data_size */,
		true /* f_has_colors */, LS->nb_colors, LS->design_color_table,
		large_set_was_classify_test_pair_of_orbits,
		this /* *test_function_data */,
		verbose_level);

	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"after OoS->create_graph_on_orbits_of_a_certain_length" << endl;
	}
	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"before CG->save" << endl;
	}

	CG->save(fname, verbose_level);

	FREE_OBJECT(CG);

	if (f_v) {
		cout << "large_set_was::create_graph_on_orbits_of_length done" << endl;
	}
}

void large_set_was::create_graph_on_orbits_of_length_based_on_N_orbits(
		std::string &fname_mask,
		int orbit_length2, int nb_N_orbits_preselected,
		int orbit_r, int orbit_m,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_was::create_graph_on_orbits_of_length_based_on_N_orbits, "
				"orbit_length2=" << orbit_length2
				<< " nb_of_orbits_to_choose=" << nb_of_orbits_to_choose
				<< " nb_N_orbits_known=" << nb_N_orbits_preselected
				<< " orbit_r=" << orbit_r
				<< " orbit_m=" << orbit_m
				<< endl;
	}

	large_set_was::orbit_length2 = orbit_length2;
	type_idx2 = H_orbits->get_orbit_type_index(orbit_length2);

	int nb_N_orbits;
	int idx_N;

	long int *Orbit1_idx;
	long int *extracted_set;
	int extracted_set_size;

	Orbit1_idx = NEW_lint(nb_of_orbits_to_choose);

	extracted_set_size = nb_N_orbits_preselected * orbit_length;
	extracted_set = NEW_lint(extracted_set_size);

	nb_N_orbits = PC->nb_orbits_at_level(nb_N_orbits_preselected);

	if (f_v) {
		cout << "large_set_was::create_graph_on_orbits_of_length_based_on_N_orbits, "
				"nb_N_orbits = " << nb_N_orbits << endl;
	}

	for (idx_N = 0; idx_N < nb_N_orbits; idx_N++) {

#if 0
		if (idx_N != 3239) {
			continue;
		}
#endif

		if ((idx_N % orbit_m) != orbit_r) {
			continue;
		}

		if (f_v) {
			cout << "large_set_was::create_graph_on_orbits_of_length_based_on_N_orbits, "
					"idx_N = " << idx_N << " / " << nb_N_orbits << endl;
		}

		PC->get_set_by_level(nb_N_orbits_preselected, idx_N, Orbit1_idx);


		H_orbits->extract_orbits_using_classification(
			orbit_length,
			nb_N_orbits_preselected,
			Orbit1_idx,
			extracted_set,
			verbose_level);


		data_structures::string_tools ST;


		std::string fname;

		fname = ST.printf_d(fname_mask, idx_N);

		if (f_v) {
			cout << "large_set_was::create_graph_on_orbits_of_length_based_on_N_orbits, "
					"fname = " << fname << endl;
			cout << "large_set_was::create_graph_on_orbits_of_length_based_on_N_orbits, "
					"extracted set = ";
			Lint_vec_print(cout, extracted_set, extracted_set_size);
			cout << endl;
		}

		graph_theory::colored_graph *CG;


		H_orbits->create_graph_on_orbits_of_a_certain_length_after_filtering(
				CG,
				fname,
				extracted_set /*filter_by_set*/,
				extracted_set_size /*filter_by_set_size*/,
				orbit_length2,
				type_idx2,
				true /*f_has_user_data*/,
					extracted_set /* long int *user_data */,
					extracted_set_size /* user_data_size */,
				true /* f_has_colors */,
					LS->nb_colors,
					LS->design_color_table,
				large_set_was_classify_test_pair_of_orbits,
				this /* *test_function_data */,
				verbose_level);


		if (f_v) {
			cout << "large_set_classify::create_graph_on_orbits_of_length_based_on_N_orbits "
					"after OoS->create_graph_on_orbits_of_a_certain_length" << endl;
		}
		if (f_v) {
			cout << "large_set_classify::create_graph_on_orbits_of_length_based_on_N_orbits "
					"before CG->save" << endl;
		}

		CG->save(fname, verbose_level);

		FREE_OBJECT(CG);
	}

	if (f_v) {
		cout << "large_set_was::create_graph_on_orbits_of_length_based_on_N_orbits done" << endl;
	}
}

void large_set_was::read_solution_file(
		std::string &solution_file_name,
		long int *starter_set,
		int starter_set_sz,
		int orbit_length,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_was::read_solution_file" << endl;
	}

	long int *Large_sets;
	//int nb_large_sets;
	long int *Packings_explicit;
	int Sz;


	if (f_v) {
		cout << "large_set_was::read_solution_file "
				"trying to read solution file " << solution_file_name << endl;
	}
	int i, j, a, b, l, h;

	orbiter_kernel_system::file_io Fio;
	int nb_solutions;
	long int *Solutions;
	int solution_size;

	Fio.read_solutions_from_file_and_get_solution_size(solution_file_name,
			nb_solutions, Solutions, solution_size,
			verbose_level);
	cout << "Read the following solutions from file:" << endl;
	if (nb_solutions < 100) {
		Lint_matrix_print(Solutions, nb_solutions, solution_size);
	}
	else {
		cout << "too large to print" << endl;
	}
	cout << "Number of solutions = " << nb_solutions << endl;
	cout << "solution_size = " << solution_size << endl;

	int sz = starter_set_sz + solution_size * orbit_length;

	if (sz != LS->size_of_large_set) {
		cout << "large_set_was::read_solution_file sz != LS->size_of_large_set" << endl;
		exit(1);
	}



	//nb_large_sets = nb_solutions;
	Large_sets = NEW_lint(nb_solutions * sz);
	for (i = 0; i < nb_solutions; i++) {
		Lint_vec_copy(starter_set, Large_sets + i * sz, starter_set_sz);
		for (j = 0; j < solution_size; j++) {
#if 0
			a = Solutions[i * solution_size + j];
			b = H_orbits->Orbits_classified->Sets[selected_type_idx][a];
#else
			b = Solutions[i * solution_size + j];
				// the labels in the graph are set according to
				// H_orbits->Orbits_classified->Sets[selected_type_idx][]
			//b = H_orbits->Orbits_classified->Sets[selected_type_idx][a];
#endif
			H_orbits->Sch->get_orbit(b,
					Large_sets + i * sz + starter_set_sz + j * orbit_length,
					l, 0 /* verbose_level*/);
			if (l != orbit_length) {
				cout << "large_set_was::read_solution_file l != orbit_length" << endl;
				exit(1);
			}
		}
		for (j = 0; j < solution_size * orbit_length; j++) {
			a = Large_sets[i * sz + starter_set_sz + j];
			//b = Design_table_reduced_idx[a];
			b = a;
			Large_sets[i * sz + starter_set_sz + j] = b;
		}
	}
	{
		orbiter_kernel_system::file_io Fio;
		string fname_out;
		data_structures::string_tools ST;

		fname_out.assign(solution_file_name);
		ST.replace_extension_with(fname_out, "_packings_design_indices.csv");


		Fio.Csv_file_support->lint_matrix_write_csv(
				fname_out, Large_sets, nb_solutions, sz);
	}
	Sz = sz * LS->design_size;

	combinatorics::combinatorics_domain Combi;

	Packings_explicit = NEW_lint(nb_solutions * Sz);
	for (i = 0; i < nb_solutions; i++) {
		for (j = 0; j < sz; j++) {
			a = Large_sets[i * sz + j];
			for (h = 0; h < LS->design_size; h++) {
				b = LS->Design_table->the_table[a * LS->design_size + h];
				Packings_explicit[i * Sz + j * LS->design_size + h] = b;
			}
		}
		if (!Combi.is_permutation_lint(Packings_explicit + i * Sz, Sz)) {
			cout << "error: the packing does not pass the permutation test" << endl;
			exit(1);
		}
	}

	if (f_v) {
		cout << "all packings pass the permutation test" << endl;
	}

	{
		orbiter_kernel_system::file_io Fio;
		string fname_out;
		data_structures::string_tools ST;

		fname_out.assign(solution_file_name);
		ST.replace_extension_with(fname_out, "_packings_explicit.csv");

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname_out, Packings_explicit, nb_solutions, Sz);
	}
	FREE_lint(Large_sets);
	FREE_lint(Packings_explicit);


	if (f_v) {
		cout << "large_set_was::read_solution_file done" << endl;
	}

}

void large_set_was::normalizer_orbits_early_test_func(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int j;
	int f_OK;

	if (f_v) {
		cout << "large_set_was::normalizer_orbits_early_test_func checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
	}


	if (len == 0) {
		Lint_vec_copy(candidates, good_candidates, nb_candidates);
		nb_good_candidates = nb_candidates;
	}
	else {
		nb_good_candidates = 0;

		if (f_vv) {
			cout << "large_set_was::normalizer_orbits_early_test_func before testing" << endl;
		}
		for (j = 0; j < nb_candidates; j++) {

			S[len] = candidates[j];

			f_OK = normalizer_orbits_check_conditions(S, len + 1, verbose_level);
			if (f_vv) {
				cout << "large_set_was::normalizer_orbits_early_test_func "
						"testing " << j << " / "
						<< nb_candidates << endl;
			}

			if (f_OK) {
				good_candidates[nb_good_candidates++] = candidates[j];
			}
		} // next j
	} // else
}

int large_set_was::normalizer_orbits_check_conditions(
		long int *S, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx, i;
	long int a, b;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "large_set_was::normalizer_orbits_check_conditions "
				"checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		//cout << "offset=" << offset << endl;
	}

	b = S[len - 1];
	if (Sorting.lint_vec_search_linear(S, len - 1, b, idx)) {
		if (f_v) {
			cout << "large_set_was::normalizer_orbits_check_conditions "
					"not OK, "
					"repeat entry" << endl;
		}
		return false;
	}

	for (i = 0; i < len - 1; i++) {
		a = S[i];

		if (!H_orbits->test_pair_of_orbits_of_a_equal_length(
				orbit_length,
				type_idx,
				a, b,
				Orbit1,
				Orbit2,
				large_set_was_classify_test_pair_of_orbits,
				this /*  test_function_data */,
				verbose_level)) {
			return false;
		}
	}
	return true;
}



// #############################################################################
// global functions:
// #############################################################################


void large_set_was_normalizer_orbits_early_test_func_callback(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	large_set_was *LSW = (large_set_was *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_was_normalizer_orbits_early_test_func_callback for set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
	}
	LSW->normalizer_orbits_early_test_func(S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
		verbose_level - 2);
	if (f_v) {
		cout << "large_set_was_normalizer_orbits_early_test_func_callback done" << endl;
	}
}





// globals:

int large_set_was_design_test_orbit(
		long int *orbit, int orbit_length,
		void *extra_data)
{
	large_set_was *LSW = (large_set_was *) extra_data;
	int ret = false;

	ret = LSW->LS->Design_table->test_set_within_itself(orbit, orbit_length);

	return ret;
}

int large_set_was_classify_test_pair_of_orbits(
		long int *orbit1, int orbit_length1,
		long int *orbit2, int orbit_length2,
		void *extra_data)
{
	large_set_was *LSW = (large_set_was *) extra_data;
	int ret = false;

	ret = LSW->LS->Design_table->test_between_two_sets(orbit1, orbit_length1,
			orbit2, orbit_length2);

	return ret;
}







}}}
