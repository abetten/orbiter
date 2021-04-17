/*
 * large_set_classify.cpp
 *
 *  Created on: Sep 19, 2019
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {




large_set_classify::large_set_classify()
{
	DC = NULL;
	design_size = 0;
	nb_points = 0;
	nb_lines = 0;
	search_depth = 0;

	//char starter_directory_name[1000];
	//char prefix[1000];
	//char path[1000];
	//char prefix_with_directory[1000];


	f_lexorder_test = FALSE;
	size_of_large_set = 0;


	Design_table = NULL;
	//design_table_prefix = NULL;
	nb_designs = 0;
	nb_colors = 0;
	design_color_table = NULL;

	A_on_designs = NULL;


	Bitvec = NULL;
	//bitvector_adjacency = NULL;
	//bitvector_length = 0;
	degree = 0;

	Control = NULL;
	Poset = NULL;
	gen = NULL;

	nb_needed = 0;

	Design_table_reduced = NULL;
	Design_table_reduced_idx = NULL;
	nb_reduced = 0;
	nb_remaining_colors = 0;
	reduced_design_color_table = NULL;

	A_reduced = NULL;
	Orbits_on_reduced = NULL;
	color_of_reduced_orbits = NULL;

	OoS = NULL;
	selected_type_idx = 0;


	//null();
}

large_set_classify::~large_set_classify()
{
	freeself();
}

void large_set_classify::null()
{
}

void large_set_classify::freeself()
{
	if (Design_table) {
		FREE_lint(Design_table);
	}
	if (Bitvec) {
		FREE_OBJECT(Bitvec);
		}
	if (design_color_table) {
		FREE_int(design_color_table);
	}
	if (Design_table_reduced) {
		FREE_lint(Design_table_reduced);
	}
	if (Design_table_reduced_idx) {
		FREE_lint(Design_table_reduced_idx);
	}
	if (OoS) {
		FREE_OBJECT(OoS);
	}
	null();
}

void large_set_classify::init(design_create *DC,
		std::string &input_prefix, std::string &base_fname,
		int search_depth,
		int f_lexorder_test,
		std::string &design_table_prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_classify::init" << endl;
		}

	large_set_classify::DC = DC;
	large_set_classify::f_lexorder_test = f_lexorder_test;
	design_size = DC->sz;
	nb_points = DC->A->degree;
	nb_lines = DC->A2->degree;
	size_of_large_set = nb_lines / design_size;

	large_set_classify::search_depth = search_depth;

	if (f_v) {
		cout << "large_set_classify::init nb_points=" << nb_points << endl;
		cout << "large_set_classify::init nb_lines=" << nb_lines << endl;
		cout << "large_set_classify::init design_size=" << design_size << endl;
		cout << "large_set_classify::init input_prefix=" << input_prefix << endl;
		cout << "large_set_classify::init base_fname=" << base_fname << endl;
		cout << "large_set_classify::init size_of_large_set=" << size_of_large_set << endl;
		cout << "large_set_classify::init search_depth=" << search_depth << endl;
		}


	starter_directory_name.assign(input_prefix);
	prefix.assign(base_fname);
	path.assign(starter_directory_name);
	prefix_with_directory.assign(starter_directory_name);
	prefix_with_directory.append(base_fname);

	large_set_classify::design_table_prefix.assign(design_table_prefix);


	if (f_v) {
		cout << "large_set_classify::init done" << endl;
		}
}

void large_set_classify::init_designs(orbit_of_sets *SetOrb,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_classify::init_designs" << endl;
	}

	long int **Sets;
	int i;
	sorting Sorting;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "large_set_classify::init_designs" << endl;
	}

	nb_designs = SetOrb->used_length;
	Sets = NEW_plint(nb_designs);
	for (i = 0; i < nb_designs; i++) {

		Sets[i] = NEW_lint(design_size);
		Orbiter->Lint_vec.copy(SetOrb->Sets[i], Sets[i], design_size);
	}

	if (f_v) {
		cout << "large_set_classify::init_designs before "
				"sorting design table of size " << nb_designs << endl;
	}

	Sorting.Heapsort_general(Sets, nb_designs,
			large_set_design_compare_func,
			large_set_swap_func,
			this);
	if (f_v) {
		cout << "large_set_classify::init_designs after "
				"sorting design table of size " << nb_designs << endl;
	}

	Design_table = NEW_lint(nb_designs * design_size);
	for (i = 0; i < nb_designs; i++) {
		Orbiter->Lint_vec.copy(Sets[i], Design_table + i * design_size, design_size);
	}

	cout << "Designs:" << endl;
	if (nb_designs < 100) {
		for (i = 0; i < nb_designs; i++) {
			cout << i << " : ";
			Orbiter->Lint_vec.print(cout, Design_table + i * design_size, design_size);
			cout << endl;
		}
	}
	else {
		cout << "too many to print" << endl;
	}

	for (i = 0; i < nb_designs; i++) {
		FREE_lint(Sets[i]);
	}
	FREE_plint(Sets);

	if (f_v) {
		cout << "large_set_classify::init_designs before compute_colors" << endl;
	}
	compute_colors(Design_table, nb_designs, design_color_table,
				verbose_level);
	if (f_v) {
		cout << "large_set_classify::init_designs after compute_colors" << endl;
	}

	if (f_v) {
		cout << "large_set_classify::init_designs "
				"creating graph" << endl;
	}


#if 0
	char prefix_for_graph[1000];

	sprintf(prefix_for_graph, "large_sets_PG_2_%d", DC->q);
	Combi.compute_adjacency_matrix(
			Design_table, nb_designs, design_size,
			prefix_for_graph,
				bitvector_adjacency,
				bitvector_length,
				verbose_level);
	if (f_v) {
		cout << "large_set_classify::init_designs "
				"creating graph done" << endl;
	}
#endif

	if (f_v) {
		cout << "large_set_classify::init_designs "
				"creating action A_on_designs" << endl;
	}
	A_on_designs = DC->A2->create_induced_action_on_sets(
			nb_designs, design_size,
			Design_table,
			//f_induce,
			0 /* verbose_level */);

	if (f_v) {
		cout << "large_set_classify::init_designs "
				"A_on_designs->degree=" << A_on_designs->degree << endl;
	}

	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(DC->A, A_on_designs,
			DC->A->Strong_gens,
			verbose_level);

	if (f_v) {
		cout << "large_set_classify::init_designs before "
				"Poset->add_testing_without_group" << endl;
	}
	Poset->add_testing_without_group(
			large_set_early_test_function,
				this /* void *data */,
				verbose_level);


	Control = NEW_OBJECT(poset_classification_control);
	gen = NEW_OBJECT(poset_classification);

	Control->f_T = TRUE;
	Control->f_W = TRUE;
	Control->problem_label = prefix;
	Control->f_problem_label = TRUE;
	Control->path = path;
	Control->f_path = TRUE;
	Control->f_depth = TRUE;
	Control->depth = search_depth;

#if 0
	Control->f_print_function = TRUE;
	Control->print_function = print_set;
	Control->print_function_data = this;
#endif
	if (f_v) {
		cout << "large_set_classify::init_designs "
				"calling gen->initialize" << endl;
	}

	gen->initialize_and_allocate_root_node(Control, Poset,
		search_depth,
		verbose_level - 1);






	if (f_v) {
		cout << "large_set_classify::init_designs done" << endl;
	}
}

void large_set_classify::compute(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int schreier_depth = search_depth;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	int t0;
	os_interface Os;

	t0 = Os.os_ticks();

	gen->main(t0,
		schreier_depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level - 1);

	int length;

	if (f_v) {
		cout << "large_set_classify::compute done with generator_main" << endl;
	}
	length = gen->nb_orbits_at_level(search_depth);
	if (f_v) {
		cout << "large_set_classify::compute We found "
			<< length << " orbits on "
			<< search_depth << "-sets" << endl;
	}
}


void large_set_classify::read_classification(orbit_transversal *&T,
		int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname_classification_at_level;

	if (f_v) {
		cout << "large_set_classify::read_classification" << endl;
	}

	gen->make_fname_lvl_file(fname_classification_at_level,
			gen->get_problem_label_with_path(), level);

	if (f_v) {
		cout << "reading all orbit representatives from "
				"file " << fname_classification_at_level << endl;
		}

	T = NEW_OBJECT(orbit_transversal);

	T->read_from_file(gen->get_A(), gen->get_A2(),
			fname_classification_at_level, verbose_level - 1);

	if (f_v) {
		cout << "large_set_classify::read_classification "
				"We read all orbit representatives. "
				"There are " << T->nb_orbits << " orbits" << endl;
		}
}

void large_set_classify::read_classification_single_case(set_and_stabilizer *&Rep,
		int level, int case_nr, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname_classification_at_level;

	if (f_v) {
		cout << "large_set_classify::read_classification_single_case" << endl;
	}

	gen->make_fname_lvl_file(fname_classification_at_level,
			gen->get_problem_label_with_path(), level);

	if (f_v) {
		cout << "reading all orbit representatives from "
				"file " << fname_classification_at_level << endl;
		}

	Rep = NEW_OBJECT(set_and_stabilizer);

	orbit_transversal *T;
	T = NEW_OBJECT(orbit_transversal);

	T->read_from_file_one_case_only(gen->get_A(), gen->get_A2(),
			fname_classification_at_level, case_nr, verbose_level - 1);

	if (f_v) {
		cout << "large_set_classify::read_classification_single_case before copy" << endl;
	}
	*Rep = T->Reps[case_nr];
	if (f_v) {
		cout << "large_set_classify::read_classification_single_case before null" << endl;
	}
	T->Reps[case_nr].null();

	if (f_v) {
		cout << "large_set_classify::read_classification_single_case before FREE_OBJECT(T)" << endl;
	}
	FREE_OBJECT(T);
	if (f_v) {
		cout << "large_set_classify::read_classification_single_case after FREE_OBJECT(T)" << endl;
	}

	if (f_v) {
		cout << "large_set_classify::read_classification_single_case done" << endl;
		}
}

void large_set_classify::make_reduced_design_table(
		long int *set, int set_sz,
		long int *&Design_table_out, long int *&Design_table_out_idx, int &nb_out,
		int verbose_level)
// Design_table_out[nb_designs * design_size]
{
	int f_v = (verbose_level >= 1);
	long int i, j, a;

	if (f_v) {
		cout << "large_set_classify::make_reduced_design_table" << endl;
	}
	Design_table_out = NEW_lint(nb_designs * design_size);
	Design_table_out_idx = NEW_lint(nb_designs);
	nb_out = 0;
	for (i = 0; i < nb_designs; i++) {
		for (j = 0; j < set_sz; j++) {
			a = set[j];
			if (!designs_are_disjoint(i, a)) {
				break;
			}
		}
		if (j == set_sz) {
			Orbiter->Lint_vec.copy(Design_table + i * design_size,
					Design_table_out + nb_out * design_size, design_size);
			Design_table_out_idx[nb_out] = i;
			nb_out++;
		}
	}
	if (f_v) {
		cout << "large_set_classify::make_reduced_design_table done" << endl;
	}
}


void large_set_classify::compute_colors(
		long int *Design_table, int nb_designs, int *&design_color_table,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "large_set_classify::compute_colors" << endl;
	}
	nb_colors = DC->get_nb_colors_as_two_design(0 /* verbose_level */);
	design_color_table = NEW_int(nb_designs);
	for (i = 0; i < nb_designs; i++) {
		design_color_table[i] =
				DC->get_color_as_two_design_assume_sorted(
						Design_table + i * design_size,
						0 /* verbose_level */);
	}

	if (f_v) {
		cout << "large_set_classify::compute_colors done" << endl;
	}
}

void large_set_classify::compute_reduced_colors(
		long int *chosen_set, int chosen_set_sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, idx, c, s;
	int *chosen_set_color;

	if (f_v) {
		cout << "large_set_classify::compute_reduced_colors" << endl;
	}
	chosen_set_color = NEW_int(chosen_set_sz);
	for (i = 0; i < chosen_set_sz; i++) {
		chosen_set_color[i] = design_color_table[chosen_set[i]];
	}

	if (DC->k != 4) {
		cout << "large_set_classify::compute_reduced_colors DC->k != 4" << endl;
		exit(1);
	}
	nb_remaining_colors = nb_colors - chosen_set_sz; // we assume that k = 4
	if (f_v) {
		cout << "large_set_classify::compute_reduced_colors "
				"nb_remaining_colors=" << nb_remaining_colors << endl;
	}
	reduced_design_color_table = NEW_int(nb_reduced);
	for (i = 0; i < nb_reduced; i++) {
		idx = Design_table_reduced_idx[i];
		c = design_color_table[idx];
		s = 0;
		for (j = 0; j < chosen_set_sz; j++) {
			if (c > chosen_set_color[j]) {
				s++;
			}
		}
		reduced_design_color_table[i] = c - s;
	}
	FREE_int(chosen_set_color);
	if (f_v) {
		cout << "large_set_classify::compute_reduced_colors done" << endl;
	}
}


int large_set_classify::designs_are_disjoint(int i, int j)
{
	long int *p1, *p2;

	p1 = Design_table + i * design_size;
	p2 = Design_table + j * design_size;
	if (test_if_sets_are_disjoint_assuming_sorted_lint(
			p1, p2, design_size, design_size)) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}



void large_set_classify::process_starter_case(
		long int *starter_set, int starter_set_sz,
		strong_generators *SG, std::string &prefix,
		const char *group_label, int orbit_length,
		int f_read_solution_file, std::string &solution_file_name,
		long int *&Large_sets, int &nb_large_sets,
		int f_compute_normalizer_orbits, strong_generators *N_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_classify::process_starter_case" << endl;
	}
	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"before make_reduced_design_table" << endl;
	}
	make_reduced_design_table(
			starter_set, starter_set_sz,
			Design_table_reduced, Design_table_reduced_idx, nb_reduced,
			verbose_level);
	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"after make_reduced_design_table" << endl;
	}
	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"The reduced design table has length " << nb_reduced << endl;
	}

	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"before compute_reduced_colors" << endl;
	}
	compute_reduced_colors(starter_set, starter_set_sz, verbose_level);
	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"after compute_reduced_colors" << endl;
	}


	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"creating A_reduced:" << endl;
	}
	A_reduced = A_on_designs->restricted_action(
			Design_table_reduced_idx, nb_reduced,
			verbose_level);


	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"computing orbits on reduced set of designs:" << endl;
	}


	OoS = NEW_OBJECT(orbits_on_something);

	OoS->init(A_reduced,
				SG,
				FALSE /* f_load_save */,
				prefix,
				verbose_level);

	// computes all orbits and classifies the orbits by their length



	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"orbits on the reduced set of designs are:" << endl;
		OoS->report_classified_orbit_lengths(cout);
	}



	colored_graph *CG;
	std::string fname;
	int f_has_user_data = FALSE;

	fname.assign(prefix);
	fname.append(group_label);

	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"before OoS->test_orbits_of_a_certain_length" << endl;
	}
	int prev_nb;

	OoS->test_orbits_of_a_certain_length(
			orbit_length,
			selected_type_idx,
			prev_nb,
			large_set_design_test_orbit,
			this /* *test_function_data*/,
			verbose_level);

	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"after OoS->test_orbits_of_a_certain_length "
				"prev_nb=" << prev_nb << " cur_nb="
				<< OoS->Orbits_classified->Set_size[selected_type_idx] << endl;
	}


	//Orbits_classified->Set_size[type_idx] = j;

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

	//A_reduced->compute_orbits_on_points(Orbits_on_reduced,
	//		SG->gens, 0 /*verbose_level*/);

#if 0
	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"The orbits on the reduced set of designs are:" << endl;
		//Orbits_on_reduced->print_and_list_orbits_sorted_by_length(
		//	cout, TRUE /* f_tex */);
	}


	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"Distribution of orbit lengths:" << endl;
		Orbits_on_reduced->print_orbit_length_distribution(cout);
	}
#endif

	int i;
	int *reduced_design_color;

	reduced_design_color = NEW_int(nb_reduced);
	for (i = 0; i < nb_reduced; i++) {
		reduced_design_color[i] = DC->get_color_as_two_design_assume_sorted(
			Design_table_reduced + i * design_size,
			0 /* verbose_level */);
	}
	tally C;

	C.init(reduced_design_color, nb_reduced, FALSE, 0);
	cout << "color distribution of reduced designs:" << endl;
	C.print_naked_tex(cout, FALSE /* f_backwards */);
	cout << endl;

	FREE_int(reduced_design_color);




	if (f_v) {
		cout << "large_set_classify::process_starter_case done" << endl;
	}
}

int large_set_classify::test_orbit(long int *orbit, int orbit_length)
{
	int i, j, a, b;
	long int *p1;
	long int *p2;

	for (i = 0; i < orbit_length; i++) {
		a = orbit[i];
		p1 = Design_table_reduced + a * design_size;
		for (j = i + 1; j < orbit_length; j++) {
			b = orbit[j];
			p2 = Design_table_reduced + b * design_size;
			if (!test_if_sets_are_disjoint_assuming_sorted_lint(
					p1, p2, design_size, design_size)) {
				return FALSE;
			}
		}
	}
	return TRUE;
}

int large_set_classify::test_pair_of_orbits(
		long int *orbit1, int orbit_length1,
		long int *orbit2, int orbit_length2)
{
	int i, j, a, b;
	long int *p1;
	long int *p2;

	for (i = 0; i < orbit_length1; i++) {
		a = orbit1[i];
		p1 = Design_table_reduced + a * design_size;
		for (j = 0; j < orbit_length2; j++) {
			b = orbit2[j];
			p2 = Design_table_reduced + b * design_size;
			if (!test_if_sets_are_disjoint_assuming_sorted_lint(
					p1, p2, design_size, design_size)) {
				return FALSE;
			}
		}
	}
	return TRUE;
}

int large_set_design_test_orbit(long int *orbit, int orbit_length,
		void *extra_data)
{
	large_set_classify *LS = (large_set_classify *) extra_data;
	int ret = FALSE;

	ret = LS->test_orbit(orbit, orbit_length);

	return ret;
}

int large_set_design_test_pair_of_orbits(long int *orbit1, int orbit_length1,
		long int *orbit2, int orbit_length2, void *extra_data)
{
	large_set_classify *LS = (large_set_classify *) extra_data;
	int ret = FALSE;

	ret = LS->test_pair_of_orbits(orbit1, orbit_length1,
			orbit2, orbit_length2);

	return ret;
}

int large_set_design_compare_func_for_invariants(void *data, int i, int j, void *extra_data)
{
	//large_set_classify *LS = (large_set_classify *) extra_data;
	int **Invariant = (int **) data;
	int ret;

	ret = int_vec_compare(Invariant[i], Invariant[j], 3);
	return ret;
}

void large_set_swap_func_for_invariants(void *data, int i, int j, void *extra_data)
{
	//large_set_classify *LS = (large_set_classify *) extra_data;
	int **Invariant = (int **) data;
	int *p;

	p = Invariant[i];
	Invariant[i] = Invariant[j];
	Invariant[j] = p;
}




int large_set_design_compare_func(void *data, int i, int j, void *extra_data)
{
	large_set_classify *LS = (large_set_classify *) extra_data;
	int **Sets = (int **) data;
	int ret;

	ret = int_vec_compare(Sets[i], Sets[j], LS->design_size);
	return ret;
}

void large_set_swap_func(void *data, int i, int j, void *extra_data)
{
	//large_set_classify *LS = (large_set_classify *) extra_data;
	int **Sets = (int **) data;
	int *p;

	p = Sets[i];
	Sets[i] = Sets[j];
	Sets[j] = p;
}

void large_set_early_test_function(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	large_set_classify *LS = (large_set_classify *) data;
	int f_v = (verbose_level >= 1);
	int i, k, a, b;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "large_set_early_test_function for set ";
		print_set(cout, len, S);
		cout << endl;
	}
	if (len == 0) {
		Orbiter->Lint_vec.copy(candidates, good_candidates, nb_candidates);
		nb_good_candidates = nb_candidates;
	}
	else {
		a = S[len - 1];
		nb_good_candidates = 0;
		for (i = 0; i < nb_candidates; i++) {
			b = candidates[i];

			if (b == a) {
				continue;
			}
			if (LS->Bitvec) {
				k = Combi.ij2k(a, b, LS->nb_designs);
				if (LS->Bitvec->s_i(k)) {
					good_candidates[nb_good_candidates++] = b;
				}
			}
			else {
				//cout << "large_set_early_test_function bitvector_adjacency has not been computed" << endl;
				//exit(1);
				if (LS->designs_are_disjoint(a, b)) {
					good_candidates[nb_good_candidates++] = b;
				}
			}
		}
	}
	if (f_v) {
		cout << "large_set_early_test_function done" << endl;
		}
}

int large_set_compute_color_of_reduced_orbits_callback(schreier *Sch,
		int orbit_idx, void *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	large_set_classify *LS = (large_set_classify *) data;

	int a, c;

	if (f_v) {
		cout << "large_set_compute_color_of_reduced_orbits_callback" << endl;
	}
	a = Sch->orbit[Sch->orbit_first[orbit_idx]];
	c = LS->DC->get_color_as_two_design_assume_sorted(
			LS->Design_table_reduced + a * LS->design_size, 0 /* verbose_level */);
	if (f_v) {
		cout << "large_set_compute_color_of_reduced_orbits_callback done" << endl;
	}
	return c;
}






}}

