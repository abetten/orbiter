/*
 * large_set_was_activity.cpp
 *
 *  Created on: May 27, 2021
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



large_set_was_activity::large_set_was_activity()
{
	Descr = NULL;
	LSW = NULL;
}


large_set_was_activity::~large_set_was_activity()
{
}


void large_set_was_activity::perform_activity(large_set_was_activity_description *Descr,
		large_set_was *LSW, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_was_activity::perform_activity" << endl;
	}

	large_set_was_activity::Descr = Descr;
	large_set_was_activity::LSW = LSW;


	if (Descr->f_normalizer_on_orbits_of_a_given_length) {
		LSW->do_normalizer_on_orbits_of_a_given_length(
				Descr->normalizer_on_orbits_of_a_given_length_length,
				verbose_level);
	}

	if (Descr->f_create_graph_on_orbits_of_length) {
		LSW->create_graph_on_orbits_of_length(
				Descr->create_graph_on_orbits_of_length_fname,
				Descr->create_graph_on_orbits_of_length_length,
				verbose_level);
	}

	if (Descr->f_read_solution_file) {

		long int *starter_set = NULL;
		int starter_set_sz = 0;

		LSW->read_solution_file(
				Descr->read_solution_file_name,
				starter_set,
				starter_set_sz,
				Descr->read_solution_file_orbit_length,
				verbose_level);
	}


	if (f_v) {
		cout << "large_set_was_activity::perform_activity done" << endl;
	}

}


#if 0
if (f_compute_normalizer_orbits) {
	if (f_v) {
		cout << "large_set_classify::process_starter_case computing orbits "
				"of normalizer on orbits of index " << selected_type_idx << endl;
	}

	action *A_on_orbits;
	action *A_on_orbits_restricted;
	schreier *Sch;

	A_on_orbits = NEW_OBJECT(action);
	A_on_orbits->induced_action_on_orbits(A_reduced,
			OoS->Sch /* H_orbits_on_spreads*/,
			TRUE /*f_play_it_safe*/, verbose_level - 1);

	A_on_orbits_restricted = A_on_orbits->restricted_action(
			OoS->Orbits_classified->Sets[selected_type_idx],
			OoS->Orbits_classified->Set_size[selected_type_idx],
			verbose_level);

	if (f_v) {
		cout << "large_set_classify::process_starter_case before "
				"compute_orbits_on_points for the restricted action "
				"on the good orbits" << endl;
	}
	A_on_orbits_restricted->compute_orbits_on_points(
			Sch, N_gens->gens, verbose_level - 1);

	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"the number of orbits of the normalizer on the "
				"good orbits is " << Sch->nb_orbits << endl;
		Sch->print_and_list_orbits_tex(cout);
		cout << "printing orbits through Design_table_reduced_idx:" << endl;
		Sch->print_and_list_orbits_using_labels(
				cout, Design_table_reduced_idx);
	}

	{
		long int *Orbits_under_N;
		file_io Fio;
		string fname_out;
		int i, a, l;


		Orbits_under_N = NEW_lint(Sch->nb_orbits * 2);

		fname_out.assign(prefix);
		fname_out.append("_graph_");
		fname_out.append(group_label);
		fname_out.append("_N_orbit_reps.csv");

		for (i = 0; i < Sch->nb_orbits; i++) {
			l = Sch->orbit_len[i];
			a = Sch->orbit[Sch->orbit_first[i]];
			Orbits_under_N[2 * i + 0] = a;
			Orbits_under_N[2 * i + 1] = l;
		}
		Fio.lint_matrix_write_csv(fname_out, Orbits_under_N, Sch->nb_orbits, 2);

		FREE_lint(Orbits_under_N);
	}

	FREE_OBJECT(Sch);
	FREE_OBJECT(A_on_orbits_restricted);
	FREE_OBJECT(A_on_orbits);
	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"computing orbits of normalizer done" << endl;
	}
}


if (f_read_solution_file) {
	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"trying to read solution file " << solution_file_name << endl;
	}
	int i, j, a, b, l, h;

	file_io Fio;
	int nb_solutions;
	int *Solutions;
	int solution_size;

	Fio.read_solutions_from_file_and_get_solution_size(solution_file_name,
			nb_solutions, Solutions, solution_size,
			verbose_level);
	cout << "Read the following solutions from file:" << endl;
	Orbiter->Int_vec.matrix_print(Solutions, nb_solutions, solution_size);
	cout << "Number of solutions = " << nb_solutions << endl;
	cout << "solution_size = " << solution_size << endl;

	int sz = starter_set_sz + solution_size * orbit_length;

	if (sz != size_of_large_set) {
		cout << "large_set_classify::process_starter_case sz != size_of_large_set" << endl;
		exit(1);
	}


	nb_large_sets = nb_solutions;
	Large_sets = NEW_lint(nb_solutions * sz);
	for (i = 0; i < nb_solutions; i++) {
		Orbiter->Lint_vec.copy(starter_set, Large_sets + i * sz, starter_set_sz);
		for (j = 0; j < solution_size; j++) {
#if 0
			a = Solutions[i * solution_size + j];
			b = OoS->Orbits_classified->Sets[selected_type_idx][a];
#else
			b = Solutions[i * solution_size + j];
				// the labels in the graph are set according to
				// OoS->Orbits_classified->Sets[selected_type_idx][]
			//b = OoS->Orbits_classified->Sets[selected_type_idx][a];
#endif
			OoS->Sch->get_orbit(b,
					Large_sets + i * sz + starter_set_sz + j * orbit_length,
					l, 0 /* verbose_level*/);
			if (l != orbit_length) {
				cout << "large_set_classify::process_starter_case l != orbit_length" << endl;
				exit(1);
			}
		}
		for (j = 0; j < solution_size * orbit_length; j++) {
			a = Large_sets[i * sz + starter_set_sz + j];
			b = Design_table_reduced_idx[a];
			Large_sets[i * sz + starter_set_sz + j] = b;
		}
	}
	{
		file_io Fio;
		string fname_out;
		string_tools ST;

		fname_out.assign(solution_file_name);
		ST.replace_extension_with(fname_out, "_packings.csv");

		ST.replace_extension_with(fname_out, "_packings.csv");

		Fio.lint_matrix_write_csv(fname_out, Large_sets, nb_solutions, sz);
	}
	long int *Packings_explicit;
	int Sz = sz * design_size;

	Packings_explicit = NEW_lint(nb_solutions * Sz);
	for (i = 0; i < nb_solutions; i++) {
		for (j = 0; j < sz; j++) {
			a = Large_sets[i * sz + j];
			for (h = 0; h < design_size; h++) {
				b = Design_table[a * design_size + h];
				Packings_explicit[i * Sz + j * design_size + h] = b;
			}
		}
	}
	{
		file_io Fio;
		string fname_out;
		string_tools ST;

		fname_out.assign(solution_file_name);
		ST.replace_extension_with(fname_out, "_packings_explicit.csv");

		Fio.lint_matrix_write_csv(fname_out, Packings_explicit, nb_solutions, Sz);
	}
	FREE_lint(Large_sets);
	FREE_lint(Packings_explicit);

}
else {
	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"before OoS->create_graph_on_orbits_of_a_certain_length" << endl;
	}


	OoS->create_graph_on_orbits_of_a_certain_length(
		CG,
		fname,
		orbit_length,
		selected_type_idx,
		f_has_user_data, NULL /* int *user_data */, 0 /* user_data_size */,
		TRUE /* f_has_colors */, nb_remaining_colors, reduced_design_color_table,
		large_set_design_test_pair_of_orbits,
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
}
#endif
}}


