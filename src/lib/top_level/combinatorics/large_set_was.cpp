/*
 * large_set_was.cpp
 *
 *  Created on: May 26, 2021
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {




large_set_was::large_set_was()
{
	Descr = NULL;

	LS = NULL;

	H_gens = NULL;

	H_orbits = NULL;

	N_gens = NULL;

	N_orbits = NULL;

#if 0
	Design_table_reduced = NULL;

	Design_table_reduced_idx = NULL;

	nb_remaining_colors = 0;

	reduced_design_color_table = NULL;

	A_reduced = NULL;

	Orbits_on_reduced = NULL;

	color_of_reduced_orbits = NULL;
#endif

	selected_type_idx = 0;
}





large_set_was::~large_set_was()
{
}

void large_set_was::init(large_set_was_description *Descr,
		large_set_classify *LS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_was::init" << endl;
	}


	large_set_was::Descr = Descr;
	large_set_was::LS = LS;



	H_gens = NEW_OBJECT(strong_generators);

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



	H_orbits = NEW_OBJECT(orbits_on_something);

	H_orbits->init(LS->A_on_designs,
			H_gens,
				FALSE /* f_load_save */,
				Descr->prefix,
				verbose_level);

	// computes all orbits and classifies the orbits by their length



	if (f_v) {
		cout << "large_set_was::init "
				"orbits of H on the set of designs are:" << endl;
		H_orbits->report_classified_orbit_lengths(cout);
	}

	int prev_nb;


	if (f_v) {
		cout << "large_set_was::init before OoS->test_orbits_of_a_certain_length" << endl;
	}
	H_orbits->test_orbits_of_a_certain_length(
			Descr->selected_orbit_length,
			selected_type_idx,
			prev_nb,
			large_set_was_design_test_orbit,
			this /* *test_function_data*/,
			verbose_level);
	if (f_v) {
		cout << "large_set_was::init after OoS->test_orbits_of_a_certain_length" << endl;
	}


	if (f_v) {
		cout << "large_set_was::init after OoS->test_orbits_of_a_certain_length "
				"the number of filtered orbits is " << H_orbits->Orbits_classified->Set_size[selected_type_idx] << endl;

		Orbiter->Lint_vec.print(cout,
				H_orbits->Orbits_classified->Sets[selected_type_idx],
				H_orbits->Orbits_classified->Set_size[selected_type_idx]);
		cout << endl;

	}






	if (f_v) {
		cout << "large_set_was::init done" << endl;
	}
}




void large_set_was::do_normalizer_on_orbits_of_a_given_length(
		int select_orbits_of_length_length,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length" << endl;
	}

	int type_idx;

	type_idx = H_orbits->get_orbit_type_index(select_orbits_of_length_length);

	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length computing orbits "
				"of normalizer on orbits of index " << type_idx << endl;
	}


	N_gens = NEW_OBJECT(strong_generators);

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





	action *A_on_orbits;
	action *A_on_orbits_restricted;
	schreier *Sch;

	A_on_orbits = NEW_OBJECT(action);
	A_on_orbits->induced_action_on_orbits(LS->A_on_designs,
			H_orbits->Sch /* H_orbits_on_spreads*/,
			TRUE /*f_play_it_safe*/,
			verbose_level - 1);

	A_on_orbits_restricted = A_on_orbits->restricted_action(
			H_orbits->Orbits_classified->Sets[type_idx],
			H_orbits->Orbits_classified->Set_size[type_idx],
			verbose_level);

	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length before "
				"compute_orbits_on_points for the restricted action "
				"on the good orbits" << endl;
	}
	A_on_orbits_restricted->compute_orbits_on_points(
			Sch, N_gens->gens, verbose_level - 1);

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
		file_io Fio;
		string fname_out;
		int i, a, l;


		Orbits_under_N = NEW_lint(Sch->nb_orbits * 2);

		fname_out.assign(Descr->prefix);
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
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length "
				"computing orbits of normalizer done" << endl;
	}


	if (f_v) {
		cout << "large_set_was::do_normalizer_on_orbits_of_a_given_length done" << endl;
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
	int nb_large_sets;
	long int *Packings_explicit;
	int Sz;


	if (f_v) {
		cout << "large_set_was::read_solution_file "
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

	if (sz != LS->size_of_large_set) {
		cout << "large_set_was::read_solution_file sz != LS->size_of_large_set" << endl;
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
		file_io Fio;
		string fname_out;
		string_tools ST;

		fname_out.assign(solution_file_name);
		ST.replace_extension_with(fname_out, "_packings.csv");

		ST.replace_extension_with(fname_out, "_packings.csv");

		Fio.lint_matrix_write_csv(fname_out, Large_sets, nb_solutions, sz);
	}
	Sz = sz * LS->design_size;

	Packings_explicit = NEW_lint(nb_solutions * Sz);
	for (i = 0; i < nb_solutions; i++) {
		for (j = 0; j < sz; j++) {
			a = Large_sets[i * sz + j];
			for (h = 0; h < LS->design_size; h++) {
				b = LS->Design_table->the_table[a * LS->design_size + h];
				Packings_explicit[i * Sz + j * LS->design_size + h] = b;
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


	if (f_v) {
		cout << "large_set_was::read_solution_file done" << endl;
	}

}



// globals:

int large_set_was_design_test_orbit(long int *orbit, int orbit_length,
		void *extra_data)
{
	large_set_was *LSW = (large_set_was *) extra_data;
	int ret = FALSE;

	ret = LSW->LS->Design_table->test_set_within_itself(orbit, orbit_length);


	return ret;
}






}}
