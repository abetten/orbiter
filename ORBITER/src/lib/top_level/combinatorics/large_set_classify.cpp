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
	design_table_prefix = NULL;
	nb_designs = 0;
	nb_colors = 0;
	design_color_table = NULL;

	A_on_designs = NULL;


	bitvector_adjacency = NULL;
	bitvector_length = 0;
	degree = 0;

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
	if (bitvector_adjacency) {
		FREE_uchar(bitvector_adjacency);
		}
	if (design_color_table) {
		FREE_int(design_color_table);
	}
	if (Design_table_reduced) {
		FREE_int(Design_table_reduced);
	}
	if (Design_table_reduced_idx) {
		FREE_int(Design_table_reduced_idx);
	}
	if (OoS) {
		FREE_OBJECT(OoS);
	}
	null();
}

void large_set_classify::init(design_create *DC,
		const char *input_prefix, const char *base_fname,
		int search_depth,
		int f_lexorder_test,
		const char *design_table_prefix,
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


	strcpy(starter_directory_name, input_prefix);
	strcpy(prefix, base_fname);
	sprintf(path, "%s",
			starter_directory_name);
	sprintf(prefix_with_directory, "%s%s",
			starter_directory_name, base_fname);

	large_set_classify::design_table_prefix = design_table_prefix;


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

	int **Sets;
	int i;
	sorting Sorting;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "large_set_classify::init_designs" << endl;
	}

	nb_designs = SetOrb->used_length;
	Sets = NEW_pint(nb_designs);
	for (i = 0; i < nb_designs; i++) {

		Sets[i] = NEW_int(design_size);
		int_vec_copy(SetOrb->Sets[i], Sets[i], design_size);
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

	Design_table = NEW_int(nb_designs * design_size);
	for (i = 0; i < nb_designs; i++) {
		int_vec_copy(Sets[i], Design_table + i * design_size, design_size);
	}

	cout << "Designs:" << endl;
	if (nb_designs < 100) {
		for (i = 0; i < nb_designs; i++) {
			cout << i << " : ";
			int_vec_print(cout, Design_table + i * design_size, design_size);
			cout << endl;
		}
	}
	else {
		cout << "too many to print" << endl;
	}

	for (i = 0; i < nb_designs; i++) {
		FREE_int(Sets[i]);
	}
	FREE_pint(Sets);

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


	gen = NEW_OBJECT(poset_classification);

	gen->f_T = TRUE;
	gen->f_W = TRUE;

	if (f_v) {
		cout << "large_set_classify::init_designs "
				"calling gen->initialize" << endl;
	}

	gen->initialize(Poset,
		search_depth,
		path, prefix,
		verbose_level - 1);



#if 0
	gen->f_print_function = TRUE;
	gen->print_function = print_set;
	gen->print_function_data = this;
#endif



	if (f_v) {
		cout << "large_set_classify::init_designs done" << endl;
	}
}

void large_set_classify::compute(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int schreier_depth = gen->depth;
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
	length = gen->nb_orbits_at_level(gen->depth);
	if (f_v) {
		cout << "large_set_classify::compute We found "
			<< length << " orbits on "
			<< gen->depth << "-sets" << endl;
	}
}


void large_set_classify::read_classification(orbit_transversal *&T,
		int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname_classification_at_level[1000];

	if (f_v) {
		cout << "large_set_classify::read_classification" << endl;
	}

	gen->make_fname_lvl_file(fname_classification_at_level,
			gen->fname_base, level);

	if (f_v) {
		cout << "reading all orbit representatives from "
				"file " << fname_classification_at_level << endl;
		}

	T = NEW_OBJECT(orbit_transversal);

	T->read_from_file(gen->Poset->A, gen->Poset->A2,
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
	char fname_classification_at_level[1000];

	if (f_v) {
		cout << "large_set_classify::read_classification_single_case" << endl;
	}

	gen->make_fname_lvl_file(fname_classification_at_level,
			gen->fname_base, level);

	if (f_v) {
		cout << "reading all orbit representatives from "
				"file " << fname_classification_at_level << endl;
		}

	Rep = NEW_OBJECT(set_and_stabilizer);

	orbit_transversal *T;
	T = NEW_OBJECT(orbit_transversal);

	T->read_from_file_one_case_only(gen->Poset->A, gen->Poset->A2,
			fname_classification_at_level, case_nr, verbose_level - 1);

	if (f_v) {
		cout << "large_set_classify::read_classification_single_case before copy" << endl;
	}
	*Rep = T->Reps[case_nr];
	if (f_v) {
		cout << "large_set_classify::read_classification_single_case before null()" << endl;
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
		int *set, int set_sz,
		int *&Design_table_out, int *&Design_table_out_idx, int &nb_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a;

	if (f_v) {
		cout << "large_set_classify::make_reduced_design_table" << endl;
	}
	Design_table_out = NEW_int(nb_designs * design_size);
	Design_table_out_idx = NEW_int(nb_designs);
	nb_out = 0;
	for (i = 0; i < nb_designs; i++) {
		for (j = 0; j < set_sz; j++) {
			a = set[j];
			if (!designs_are_disjoint(i, a)) {
				break;
			}
		}
		if (j == set_sz) {
			int_vec_copy(Design_table + i * design_size, Design_table_out + nb_out * design_size, design_size);
			Design_table_out_idx[nb_out] = i;
			nb_out++;
		}
	}
	if (f_v) {
		cout << "large_set_classify::make_reduced_design_table done" << endl;
	}
}


void large_set_classify::compute_colors(
		int *Design_table, int nb_designs, int *&design_color_table,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a;

	if (f_v) {
		cout << "large_set_classify::compute_colors" << endl;
	}
	nb_colors = DC->get_nb_colors_as_two_design(0 /* verbose_level */);
	design_color_table = NEW_int(nb_designs);
	for (i = 0; i < nb_designs; i++) {
		design_color_table[i] =
				DC->get_color_as_two_design_assume_sorted(
						Design_table + i * design_size, 0 /* verbose_level */);
	}

	if (f_v) {
		cout << "large_set_classify::compute_colors done" << endl;
	}
}

void large_set_classify::compute_reduced_colors(
		int *set, int set_sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, idx, c;
	int *set_color;

	if (f_v) {
		cout << "large_set_classify::compute_reduced_colors" << endl;
	}
	set_color = NEW_int(set_sz);
	for (i = 0; i < set_sz; i++) {
		set_color[i] = design_color_table[set[i]];
	}

	if (DC->k != 4) {
		cout << "large_set_classify::compute_reduced_colors DC->k != 4" << endl;
		exit(1);
	}
	nb_remaining_colors = nb_colors - set_sz; // we assume that k = 4
	reduced_design_color_table = NEW_int(nb_reduced);
	for (i = 0; i < nb_reduced; i++) {
		idx = Design_table_reduced_idx[i];
		c = design_color_table[idx];
		for (j = 0; j < set_sz; j++) {
			if (c > set_color[j]) {
				c--;
			}
		}
		reduced_design_color_table[i] = c;
	}
	FREE_int(set_color);
	if (f_v) {
		cout << "large_set_classify::compute_reduced_colors done" << endl;
	}
}


int large_set_classify::designs_are_disjoint(int i, int j)
{
	int *p1, *p2;

	p1 = Design_table + i * design_size;
	p2 = Design_table + j * design_size;
	if (test_if_sets_are_disjoint_assuming_sorted(
			p1, p2, design_size, design_size)) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}



void large_set_classify::process_starter_case(set_and_stabilizer *Rep,
		strong_generators *SG, const char *prefix,
		char *group_label, int orbit_length,
		int f_read_solution_file, const char *solution_file_name,
		int *&Large_sets, int &nb_large_sets,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_classify::process_starter_case" << endl;
	}
	if (f_v) {
		cout << "large_set_classify::process_starter_case before make_reduced_design_table" << endl;
	}
	make_reduced_design_table(
			Rep->data, Rep->sz,
			Design_table_reduced, Design_table_reduced_idx, nb_reduced,
			verbose_level);
	if (f_v) {
		cout << "large_set_classify::process_starter_case after make_reduced_design_table" << endl;
	}
	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"The reduced design table has length " << nb_reduced << endl;
	}

	if (f_v) {
		cout << "large_set_classify::process_starter_case before compute_reduced_colors" << endl;
	}
	compute_reduced_colors(
			Rep->data, Rep->sz,
			verbose_level);
	if (f_v) {
		cout << "large_set_classify::process_starter_case after compute_reduced_colors" << endl;
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

	colored_graph *CG;
	char fname[1000];
	int f_has_user_data = FALSE;

	sprintf(fname, "%s_graph_%s.bin", prefix, group_label);

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


	if (f_read_solution_file) {
		if (f_v) {
			cout << "large_set_classify::process_starter_case "
					"trying to read solution file " << solution_file_name << endl;
		}

		file_io Fio;
		int nb_solutions;
		int *Solutions;
		int solution_size;

		Fio.read_solutions_from_file_and_get_solution_size(solution_file_name,
				nb_solutions, Solutions, solution_size,
				verbose_level);
		cout << "Read the following solutions from file:" << endl;
		int_matrix_print(Solutions, nb_solutions, solution_size);
		cout << "Number of solutions = " << nb_solutions << endl;
		cout << "solution_size = " << solution_size << endl;

		int sz = Rep->sz + solution_size * orbit_length;

		if (sz != size_of_large_set) {
			cout << "large_set_classify::process_starter_case sz != size_of_large_set" << endl;
			exit(1);
		}
		nb_large_sets = nb_solutions;
		Large_sets = NEW_int(nb_solutions * sz);
		int i, j, a, b, l;
		for (i = 0; i < nb_solutions; i++) {
			int_vec_copy(Rep->data, Large_sets + i * sz, Rep->sz);
			for (j = 0; j < solution_size; j++) {
				a = Solutions[i * solution_size + j];
				b = OoS->Orbits_classified->Sets[selected_type_idx][a];
				OoS->Sch->get_orbit(b, Large_sets + i * sz + Rep->sz + j * orbit_length, l, 0 /* verbose_level*/);
				if (l != orbit_length) {
					cout << "large_set_classify::process_starter_case l != orbit_length" << endl;
					exit(1);
				}
			}
			for (j = 0; j < solution_size * orbit_length; j++) {
				a = Large_sets[i * sz + Rep->sz + j];
				b = Design_table_reduced_idx[a];
				Large_sets[i * sz + Rep->sz + j] = b;
			}
		}

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
			Design_table_reduced + i * design_size, 0 /* verbose_level */);
	}
	classify C;

	C.init(reduced_design_color, nb_reduced, FALSE, 0);
	cout << "color distribution of reduced designs:" << endl;
	C.print_naked_tex(cout, FALSE /* f_backwards */);
	cout << endl;

	FREE_int(reduced_design_color);


#if 0
	if (f_v) {
		cout << "large_set_classify::process_starter_case "
				"computing coloring of reduced orbits:" << endl;
	}
	Orbits_on_reduced->compute_orbit_invariant(color_of_reduced_orbits,
				large_set_compute_color_of_reduced_orbits_callback,
				this /* compute_orbit_invariant_data */,
				verbose_level);


	int **Invariant;
	int i;
	sorting Sorting;

	Invariant = NEW_pint(Orbits_on_reduced->nb_orbits);
	for (i = 0; i < Orbits_on_reduced->nb_orbits; i++) {
		Invariant[i] = NEW_int(3);
		Invariant[i][0] = color_of_reduced_orbits[i];
		Invariant[i][1] = Orbits_on_reduced->orbit_len[i];
		Invariant[i][2] = i;
	}
	Sorting.Heapsort_general(Invariant, Orbits_on_reduced->nb_orbits,
			large_set_design_compare_func_for_invariants,
			large_set_swap_func_for_invariants,
			Invariant /* extra_data */);

	int f;
	int f_continue;

	f = 0;
	for (i = 0; i < Orbits_on_reduced->nb_orbits; i++) {
		f_continue = TRUE;
		if (i < Orbits_on_reduced->nb_orbits - 1) {
			if (int_vec_compare(Invariant[i], Invariant[i + 1], 2)) {
				f_continue = FALSE;
			}
		}
		if (f_continue) {
			continue;
		}
		cout << "block of " << i + 1 - f << " orbits of color "
				<< Invariant[i][0] << " and of length " << Invariant[i][1] << endl;
		f = i + 1;
	}
#endif



	if (f_v) {
		cout << "large_set_classify::process_starter_case done" << endl;
	}
}

int large_set_classify::test_orbit(int *orbit, int orbit_length)
{
	int i, j, a, b;
	int *p1;
	int *p2;

	for (i = 0; i < orbit_length; i++) {
		a = orbit[i];
		p1 = Design_table_reduced + a * design_size;
		for (j = i + 1; j < orbit_length; j++) {
			b = orbit[j];
			p2 = Design_table_reduced + b * design_size;
			if (!test_if_sets_are_disjoint_assuming_sorted(
					p1, p2, design_size, design_size)) {
				return FALSE;
			}
		}
	}
	return TRUE;
}

int large_set_classify::test_pair_of_orbits(
		int *orbit1, int orbit_length1,
		int *orbit2, int orbit_length2)
{
	int i, j, a, b;
	int *p1;
	int *p2;

	for (i = 0; i < orbit_length1; i++) {
		a = orbit1[i];
		p1 = Design_table_reduced + a * design_size;
		for (j = 0; j < orbit_length2; j++) {
			b = orbit2[j];
			p2 = Design_table_reduced + b * design_size;
			if (!test_if_sets_are_disjoint_assuming_sorted(
					p1, p2, design_size, design_size)) {
				return FALSE;
			}
		}
	}
	return TRUE;
}

int large_set_design_test_orbit(int *orbit, int orbit_length,
		void *extra_data)
{
	large_set_classify *LS = (large_set_classify *) extra_data;
	int ret = FALSE;

	ret = LS->test_orbit(orbit, orbit_length);

	return ret;
}

int large_set_design_test_pair_of_orbits(int *orbit1, int orbit_length1,
		int *orbit2, int orbit_length2, void *extra_data)
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

void large_set_early_test_function(int *S, int len,
	int *candidates, int nb_candidates,
	int *good_candidates, int &nb_good_candidates,
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
		int_vec_copy(candidates, good_candidates, nb_candidates);
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
			if (LS->bitvector_adjacency) {
				k = Combi.ij2k(a, b, LS->nb_designs);
				if (bitvector_s_i(LS->bitvector_adjacency, k)) {
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

