// packing_classify.cpp
// 
// Anton Betten
// Feb 6, 2013
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {




packing_classify::packing_classify()
{
	T = NULL;
	F = NULL;
	spread_size = 0;
	nb_lines = 0;
	//search_depth = 0;

	//starter_directory_name[0] = 0;
	//prefix[0] = 0;
	//prefix_with_directory[0] = 0;


	f_lexorder_test = TRUE;
	q = 0;
	size_of_packing = 0;
		// the number of spreads in a packing,
		// which is q^2 + q + 1

	P3 = NULL;
	P5 = NULL;
	the_packing = NULL;
	spread_iso_type = NULL;
	dual_packing = NULL;
	list_of_lines = NULL;
	list_of_lines_klein_image = NULL;
	Gr = NULL;


	spread_tables_prefix = NULL;

	spread_reps = NULL;
	spread_reps_idx = NULL;
	spread_orbit_length = NULL;
	nb_spread_reps = 0;
	total_nb_of_spreads = 0;
	nb_iso_types_of_spreads = 0;
	// the number of spreads
	// from the classification


	Spread_tables = NULL;
	tmp_isomorphism_type_of_spread = NULL;

	A_on_spreads = NULL;


	bitvector_adjacency = NULL;
	bitvector_length = 0;
	degree = NULL;

	Control = NULL;
	Poset = NULL;
	gen = NULL;

	nb_needed = 0;

	//null();
}

packing_classify::~packing_classify()
{
	freeself();
}

void packing_classify::null()
{
}

void packing_classify::freeself()
{
	if (bitvector_adjacency) {
		FREE_uchar(bitvector_adjacency);
	}
	if (spread_reps) {
		FREE_lint(spread_reps);
	}
	if (spread_reps_idx) {
		FREE_int(spread_reps_idx);
	}
	if (spread_orbit_length) {
		FREE_lint(spread_orbit_length);
	}
	if (Spread_tables) {
		FREE_OBJECT(Spread_tables);
	}
	if (P3) {
		FREE_OBJECT(P3);
	}
	if (P5) {
		FREE_OBJECT(P5);
	}
	if (the_packing) {
		FREE_lint(the_packing);
	}
	if (spread_iso_type) {
		FREE_lint(spread_iso_type);
	}
	if (dual_packing) {
		FREE_lint(dual_packing);
	}
	if (list_of_lines) {
		FREE_lint(list_of_lines);
	}
	if (list_of_lines_klein_image) {
		FREE_lint(list_of_lines_klein_image);
	}
	if (Gr) {
		FREE_OBJECT(Gr);
	}
	null();
}

void packing_classify::init(spread_classify *T,
	int f_select_spread,
	const char *select_spread_text,
	//const char *input_prefix, const char *base_fname,
	int f_lexorder_test,
	const char *path_to_spread_tables,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_classify::init" << endl;
	}

	int *select_spread = NULL;
	int select_spread_nb;

	if (f_select_spread) {
		int_vec_scan(select_spread_text, select_spread, select_spread_nb);
		if (f_v) {
			cout << "packing_select_spread = ";
			int_vec_print(cout, select_spread, select_spread_nb);
			cout << endl;
		}
	}
	else {
		select_spread_nb = 0;
	}


	packing_classify::T = T;
	F = T->Mtx->GFq;
	packing_classify::f_lexorder_test = f_lexorder_test;
	q = T->q;
	spread_size = T->spread_size;
	size_of_packing = q * q + q + 1;
	nb_lines = T->A2->degree;

	
	if (f_v) {
		cout << "packing_classify::init q=" << q << endl;
		cout << "packing_classify::init nb_lines=" << nb_lines << endl;
		cout << "packing_classify::init spread_size=" << spread_size << endl;
		cout << "packing_classify::init size_of_packing=" << size_of_packing << endl;
		//cout << "packing_classify::init input_prefix=" << input_prefix << endl;
		//cout << "packing_classify::init base_fname=" << base_fname << endl;
	}

	init_P3_and_P5(verbose_level - 1);

#if 0
	strcpy(starter_directory_name, input_prefix);
	strcpy(prefix, base_fname);
	sprintf(prefix_with_directory, "%s%s",
			starter_directory_name, base_fname);
#endif


	if (f_select_spread) {
		cout << "packing_classify::init selected spreads are "
				"from the following orbits: ";
		int_vec_print(cout,
				select_spread,
				select_spread_nb);
		cout << endl;
	}
	

	Spread_tables = NEW_OBJECT(spread_tables);


	algebra_global_with_action Algebra;

	if (f_v) {
		cout << "packing_classify::init before Algebra.predict_spread_table_length" << endl;
	}
	Algebra.predict_spread_table_length(
		q, T->k /* dimension_of_spread_elements */, T->spread_size,
		T->A, T->LG->Strong_gens,
		f_select_spread,
		select_spread, select_spread_nb,
		spread_reps, spread_reps_idx, spread_orbit_length,
		nb_spread_reps,
		total_nb_of_spreads,
		nb_iso_types_of_spreads,
		verbose_level - 1);
	if (f_v) {
		cout << "packing_classify::init before Algebra.predict_spread_table_length" << endl;
		cout << "packing_classify::init before Algebra.predict_spread_table_length "
				"total_nb_of_spreads = " << total_nb_of_spreads << endl;
	}

	Spread_tables->nb_spreads = total_nb_of_spreads;

	if (f_v) {
		cout << "packing_classify::init before Spread_tables->init" << endl;
	}

	Spread_tables->init(F,
				FALSE /* f_load */,
				nb_iso_types_of_spreads,
				path_to_spread_tables,
				verbose_level);

	if (f_v) {
		cout << "packing_classify::init after Spread_tables->init" << endl;
	}

	if (f_v) {
		cout << "We will use " << nb_spread_reps << " isomorphism types of spreads, "
				"this will give a total number of " << Spread_tables->nb_spreads
				<< " labeled spreads" << endl;
	}

	FREE_int(select_spread);

	if (f_v) {
		cout << "packing_classify::init done" << endl;
	}
}

void packing_classify::init2(poset_classification_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_classify::init2" << endl;
	}

	if (f_v) {
		cout << "packing_classify::init2 "
				"before create_action_on_spreads" << endl;
	}
	create_action_on_spreads(verbose_level - 1);
	if (f_v) {
		cout << "packing_classify::init2 "
				"after create_action_on_spreads" << endl;
	}


	
	if (f_v) {
		cout << "packing_classify::init "
				"before prepare_generator" << endl;
	}
	prepare_generator(Control, verbose_level - 1);
	if (f_v) {
		cout << "packing_classify::init "
				"after prepare_generator" << endl;
	}

	if (f_v) {
		cout << "packing_classify::init done" << endl;
	}
}

void packing_classify::compute_spread_table(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_classify::compute_spread_table" << endl;
	}




	if (Spread_tables->files_exist(verbose_level)) {
		if (f_v) {
			cout << "packing_classify::compute_spread_table files exist, "
					"reading" << endl;
		}

		Spread_tables->load(verbose_level);

		if (f_v) {
			cout << "packing_classify::compute_spread_table "
					"after Spread_tables->load" << endl;
		}
	}
	else {

		if (f_v) {
			cout << "packing_classify::compute_spread_table "
					"files do not exist, computing the spread table" << endl;
		}

		if (f_v) {
			cout << "packing_classify::compute_spread_table "
					"before compute_spread_table_from_scratch" << endl;
		}
		compute_spread_table_from_scratch(verbose_level - 1);
		if (f_v) {
			cout << "packing_classify::compute_spread_table "
					"after compute_spread_table_from_scratch" << endl;
		}
	}



	if (f_v) {
		cout << "packing_classify::compute_spread_table done" << endl;
	}
}

void packing_classify::compute_spread_table_from_scratch(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_classify::compute_spread_table_from_scratch" << endl;
	}

	int i;
	long int **Sets;
	int nb_spreads;
	int *isomorphism_type_of_spread;
	long int *Spread_table;
	sorting Sorting;


	nb_spreads = Spread_tables->nb_spreads;

	if (f_v) {
		cout << "packing_classify::compute_spread_table_from_scratch "
				"before Algebra.make_spread_table" << endl;
	}

	algebra_global_with_action Algebra;

	Algebra.make_spread_table(
			T->A, T->A2, T->LG->Strong_gens,
			T->spread_size,
			spread_reps, spread_reps_idx, spread_orbit_length,
			nb_spread_reps,
			total_nb_of_spreads,
			Sets, isomorphism_type_of_spread,
			verbose_level);

	// does not sort the spread table

	if (f_v) {
		cout << "packing_classify::compute_spread_table_from_scratch "
				"after Algebra.make_spread_table" << endl;
	}


	if (f_v) {
		cout << "packing_classify::compute_spread_table_from_scratch before "
				"sorting spread table of size " << total_nb_of_spreads << endl;
	}
	tmp_isomorphism_type_of_spread = isomorphism_type_of_spread;
		// for packing_swap_func
	Sorting.Heapsort_general(Sets, total_nb_of_spreads,
			packing_spread_compare_func,
			packing_swap_func,
			this);
	if (f_v) {
		cout << "packing_classify::compute_spread_table_from_scratch after "
				"sorting spread table of size " << total_nb_of_spreads << endl;
	}

	Spread_table = NEW_lint(nb_spreads * spread_size);
	for (i = 0; i < nb_spreads; i++) {
		lint_vec_copy(Sets[i], Spread_table + i * spread_size, spread_size);
	}


	Spread_tables->init(F, FALSE, nb_iso_types_of_spreads,
			spread_tables_prefix,
			verbose_level);


	Spread_tables->init_spread_table(nb_spreads,
			Spread_table, isomorphism_type_of_spread,
			verbose_level);
	long int *Dual_spread_idx;
	long int *self_dual_spread_idx;
	int nb_self_dual_spreads;

	Spread_tables->compute_dual_spreads(Sets,
				Dual_spread_idx,
				self_dual_spread_idx,
				nb_self_dual_spreads,
				verbose_level);



	Spread_tables->init_tables(nb_spreads,
			Spread_table, isomorphism_type_of_spread,
			Dual_spread_idx,
			self_dual_spread_idx, nb_self_dual_spreads,
			verbose_level);

	Spread_tables->save(verbose_level);


	if (nb_spreads < 10000) {
		cout << "packing_classify::compute_spread_table_from_scratch "
				"We are computing the adjacency matrix" << endl;
		compute_adjacency_matrix(verbose_level - 1);
		cout << "packing_classify::compute_spread_table_from_scratch "
				"The adjacency matrix has been computed" << endl;
	}
	else {
		cout << "packing_classify::compute_spread_table_from_scratch "
				"We are NOT computing the adjacency matrix" << endl;
	}


	for (i = 0; i < nb_spreads; i++) {
		FREE_lint(Sets[i]);
	}
	FREE_plint(Sets);

	if (f_v) {
		cout << "packing_classify::compute_spread_table_from_scratch done" << endl;
	}
}

void packing_classify::init_P3_and_P5(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_classify::init_P3_and_P5" << endl;
	}
	P3 = NEW_OBJECT(projective_space);
	
	P3->init(3, T->Mtx->GFq,
		TRUE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "packing_classify::init_P3_and_P5 P3->N_points=" << P3->N_points << endl;
		cout << "packing_classify::init_P3_and_P5 P3->N_lines=" << P3->N_lines << endl;
	}

	P5 = NEW_OBJECT(projective_space);

	P5->init(5, T->Mtx->GFq,
		TRUE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "packing_classify::init_P3_and_P5 P5->N_points=" << P5->N_points << endl;
		cout << "packing_classify::init_P3_and_P5 P5->N_lines=" << P5->N_lines << endl;
	}

	the_packing = NEW_lint(size_of_packing);
	spread_iso_type = NEW_lint(size_of_packing);
	dual_packing = NEW_lint(size_of_packing);
	list_of_lines = NEW_lint(size_of_packing * spread_size);
	list_of_lines_klein_image = NEW_lint(size_of_packing * spread_size);

	Gr = NEW_OBJECT(grassmann);

	Gr->init(6, 3, F, 0 /* verbose_level */);

	if (f_v) {
		cout << "packing_classify::init_P3_and_P5 done" << endl;
	}
}






int packing_classify::test_if_packing_is_self_dual(int *packing, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = FALSE;
	int *sorted_packing;
	int *dual_packing;
	int i, a, b;
	sorting Sorting;

	if (f_v) {
		cout << "packing_classify::test_if_packing_is_self_dual" << endl;
	}
	sorted_packing = NEW_int(size_of_packing);
	dual_packing = NEW_int(size_of_packing);
	for (i = 0; i < size_of_packing; i++) {
		a = packing[i];
		sorted_packing[i] = a;
	}
	Sorting.int_vec_heapsort(sorted_packing, size_of_packing);

	for (i = 0; i < size_of_packing; i++) {
		a = packing[i];
		b = Spread_tables->dual_spread_idx[a];
		dual_packing[i] = b;
	}
	Sorting.int_vec_heapsort(dual_packing, size_of_packing);
	if (int_vec_compare(sorted_packing, dual_packing, size_of_packing) == 0) {
		ret = TRUE;
	}

	if (f_v) {
		cout << "packing_classify::test_if_packing_is_self_dual done" << endl;
	}
	return ret;
}


void packing_classify::compute_adjacency_matrix(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_classify::compute_adjacency_matrix" << endl;
	}

	Spread_tables->compute_adjacency_matrix(
			bitvector_adjacency,
			bitvector_length,
			verbose_level);

	
	if (f_v) {
		cout << "packing_classify::compute_adjacency_matrix done" << endl;
	}
}



void packing_classify::prepare_generator(
		poset_classification_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "packing_classify::prepare_generator" << endl;
	}
	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(T->A, A_on_spreads,
			T->A->Strong_gens,
			verbose_level);

	if (f_v) {
		cout << "packing_classify::prepare_generator before "
				"Poset->add_testing_without_group" << endl;
	}
	Poset->add_testing_without_group(
			packing_early_test_function,
				this /* void *data */,
				verbose_level);




#if 0
	Control->f_print_function = TRUE;
	Control->print_function = print_set;
	Control->print_function_data = this;
#endif


	if (f_v) {
		cout << "packing_classify::prepare_generator "
				"calling gen->initialize" << endl;
	}

	gen = NEW_OBJECT(poset_classification);

	gen->initialize_and_allocate_root_node(Control, Poset,
			size_of_packing,
			verbose_level - 1);

	if (f_v) {
		cout << "packing_classify::prepare_generator done" << endl;
	}
}


void packing_classify::compute(int search_depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int schreier_depth = search_depth;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	int t0;
	os_interface Os;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "packing_classify::compute" << endl;
	}

	gen->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		verbose_level - 1);
	
	int length;
	
	if (f_v) {
		cout << "packing_classify::compute done with generator_main" << endl;
	}
	length = gen->nb_orbits_at_level(search_depth);
	if (f_v) {
		cout << "packing_classify::compute We found "
			<< length << " orbits on "
			<< search_depth << "-sets" << endl;
	}
	if (f_v) {
		cout << "packing_classify::compute done" << endl;
	}

}

void packing_classify::lifting_prepare_function_new(
	exact_cover *E, int starter_case,
	long int *candidates, int nb_candidates,
	strong_generators *Strong_gens,
	diophant *&Dio, long int *&col_labels,
	int &f_ruled_out, 
	int verbose_level)
{
	verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_v3 = (verbose_level >= 3);
	long int *points_covered_by_starter;
	int nb_points_covered_by_starter;
	long int *free_points2;
	int nb_free_points2;
	long int *free_point_idx;
	long int *live_blocks2;
	int nb_live_blocks2;
	int nb_needed, /*nb_rows,*/ nb_cols;


	if (f_v) {
		cout << "packing_classify::lifting_prepare_function "
				"nb_candidates=" << nb_candidates << endl;
	}

	nb_needed = size_of_packing - E->starter_size;


	if (f_v) {
		cout << "packing_classify::lifting_prepare_function "
				"before compute_covered_points" << endl;
	}

	compute_covered_points(points_covered_by_starter, 
		nb_points_covered_by_starter, 
		E->starter, E->starter_size, 
		verbose_level - 1);


	if (f_v) {
		cout << "packing_classify::lifting_prepare_function "
				"before compute_free_points2" << endl;
	}

	compute_free_points2(
		free_points2, nb_free_points2, free_point_idx,
		points_covered_by_starter, nb_points_covered_by_starter, 
		E->starter, E->starter_size, 
		verbose_level - 1);

	if (f_v) {
		cout << "packing_classify::lifting_prepare_function "
				"before compute_live_blocks2" << endl;
	}

	compute_live_blocks2(
		E, starter_case, live_blocks2, nb_live_blocks2,
		points_covered_by_starter, nb_points_covered_by_starter, 
		E->starter, E->starter_size, 
		verbose_level - 1);


	if (f_v) {
		cout << "packing_classify::lifting_prepare_function "
				"after compute_live_blocks2" << endl;
	}

	//nb_rows = nb_free_points2;
	nb_cols = nb_live_blocks2;
	col_labels = NEW_lint(nb_cols);


	lint_vec_copy(live_blocks2, col_labels, nb_cols);


	if (f_vv) {
		cout << "packing_classify::lifting_prepare_function_new candidates: ";
		lint_vec_print(cout, col_labels, nb_cols);
		cout << " (nb_candidates=" << nb_cols << ")" << endl;
	}



	if (E->f_lex) {
		int nb_cols_before;

		nb_cols_before = nb_cols;
		E->lexorder_test(col_labels, nb_cols, Strong_gens->gens, 
			verbose_level - 2);
		if (f_v) {
			cout << "packing_classify::lifting_prepare_function_new after "
					"lexorder test nb_candidates before: " << nb_cols_before
					<< " reduced to  " << nb_cols << " (deleted "
					<< nb_cols_before - nb_cols << ")" << endl;
		}
	}

	if (f_vv) {
		cout << "packing_classify::lifting_prepare_function_new "
				"after lexorder test" << endl;
		cout << "packing::lifting_prepare_function_new "
				"nb_cols=" << nb_cols << endl;
	}

	Spread_tables->make_exact_cover_problem(Dio,
			free_point_idx, nb_free_points2,
			live_blocks2, nb_live_blocks2,
			nb_needed,
			verbose_level);

	FREE_lint(points_covered_by_starter);
	FREE_lint(free_points2);
	FREE_lint(free_point_idx);
	FREE_lint(live_blocks2);
	if (f_v) {
		cout << "packing_classify::lifting_prepare_function done" << endl;
	}
}


void packing_classify::compute_covered_points(
	long int *&points_covered_by_starter,
	int &nb_points_covered_by_starter,
	long int *starter, int starter_size,
	int verbose_level)
// points_covered_by_starter are the lines that
// are contained in the spreads chosen for the starter
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int a, s;
	
	if (f_v) {
		cout << "packing_classify::compute_covered_points" << endl;
	}
	points_covered_by_starter = NEW_lint(starter_size * spread_size);
	for (i = 0; i < starter_size; i++) {
		s = starter[i];
		for (j = 0; j < spread_size; j++) {
			a = Spread_tables->spread_table[s * spread_size + j];
			points_covered_by_starter[i * spread_size + j] = a;
		}
	}
#if 0
	cout << "covered lines:" << endl;
	int_vec_print(cout, covered_lines, starter_size * spread_size);
	cout << endl;
#endif
	if (f_v) {
		cout << "packing_classify::compute_covered_points done" << endl;
	}
}

void packing_classify::compute_free_points2(
	long int *&free_points2, int &nb_free_points2, long int *&free_point_idx,
	long int *points_covered_by_starter,
	int nb_points_covered_by_starter,
	long int *starter, int starter_size,
	int verbose_level)
// free_points2 are actually the free lines,
// i.e., the lines that are not
// yet part of the partial packing
{
	int f_v = (verbose_level >= 1);
	int i, a;
	
	if (f_v) {
		cout << "packing_classify::compute_free_points2" << endl;
	}
	free_point_idx = NEW_lint(nb_lines);
	free_points2 = NEW_lint(nb_lines);
	for (i = 0; i < nb_lines; i++) {
		free_point_idx[i] = 0;
	}
	for (i = 0; i < starter_size * spread_size; i++) {
		a = points_covered_by_starter[i];
		free_point_idx[a] = -1;
	}
	nb_free_points2 = 0;
	for (i = 0; i < nb_lines; i++) {
		if (free_point_idx[i] == 0) {
			free_points2[nb_free_points2] = i;
			free_point_idx[i] = nb_free_points2;
			nb_free_points2++;
		}
	}
#if 0
	cout << "free points2:" << endl;
	int_vec_print(cout, free_points2, nb_free_points2);
	cout << endl;
#endif
	if (f_v) {
		cout << "packing_classify::compute_free_points2 done" << endl;
	}
}

void packing_classify::compute_live_blocks2(
	exact_cover *EC, int starter_case,
	long int *&live_blocks2, int &nb_live_blocks2,
	long int *points_covered_by_starter, int nb_points_covered_by_starter,
	long int *starter, int starter_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	
	if (f_v) {
		cout << "packing_classify::compute_live_blocks2" << endl;
	}
	live_blocks2 = NEW_lint(Spread_tables->nb_spreads);
	nb_live_blocks2 = 0;
	for (i = 0; i < Spread_tables->nb_spreads; i++) {
		for (j = 0; j < starter_size; j++) {
			if (!is_adjacent(starter[j], i)) {
				break;
			}
		}
		if (j == starter_size) {
			live_blocks2[nb_live_blocks2++] = i;
		}
	}
	if (f_v) {
		cout << "packing_classify::compute_live_blocks2 done" << endl;
	}

	if (f_v) {
		cout << "packing_classify::compute_live_blocks2 STARTER_CASE "
			<< starter_case << " / " << EC->starter_nb_cases
			<< " : Found " << nb_live_blocks2 << " live spreads" << endl;
	}
}

int packing_classify::is_adjacent(int i, int j)
{
	int k;
	combinatorics_domain Combi;
	
	if (i == j) {
		return FALSE;
	}
	if (bitvector_adjacency) {
		k = Combi.ij2k(i, j, Spread_tables->nb_spreads);
		if (bitvector_s_i(bitvector_adjacency, k)) {
			return TRUE;
		}
		else {
			return FALSE;
		}
	}
	else {
		if (Spread_tables->test_if_spreads_are_disjoint(i, j)) {
			return TRUE;
		}
		else {
			return FALSE;
		}
	}
}

void packing_classify::read_spread_table(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "packing_classify::read_spread_table" << endl;
	}

	Spread_tables = NEW_OBJECT(spread_tables);

	if (f_v) {
		cout << "packing_classify::read_spread_table "
				"before Spread_tables->init" << endl;
	}

	Spread_tables->init(F,
			TRUE /* f_load */, nb_iso_types_of_spreads,
			spread_tables_prefix,
			verbose_level);

	{
		int *type;
		set_of_sets *SoS;
		int a, b;

		Spread_tables->classify_self_dual_spreads(type,
				SoS,
				verbose_level);
		cout << "the self-dual spreads belong to the "
				"following isomorphism types:" << endl;
		for (i = 0; i < nb_iso_types_of_spreads; i++) {
			cout << i << " : " << type[i] << endl;
		}
		SoS->print();
		for (a = 0; a < SoS->nb_sets; a++) {
			if (SoS->Set_size[a] < 10) {
				cout << "iso type " << a << endl;
				lint_vec_print(cout, SoS->Sets[a], SoS->Set_size[a]);
				cout << endl;
				for (i = 0; i < SoS->Set_size[a]; i++) {
					b = SoS->Sets[a][i];
					cout << i << " : " << b << " : ";
					lint_vec_print(cout, Spread_tables->spread_table +
							b * spread_size, spread_size);
					cout << endl;
				}
			}
		}
		FREE_int(type);
	}

	if (f_v) {
		cout << "packing_classify::read_spread_table "
				"after Spread_tables->init" << endl;
	}



	if (f_v) {
		cout << "packing_classify::read_spread_table done" << endl;
	}
}

void packing_classify::create_action_on_spreads(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_classify::create_action_on_spreads" << endl;
	}

	if (f_v) {
		cout << "packing_classify::create_action_on_spreads "
				"creating action A_on_spreads" << endl;
	}
	A_on_spreads = T->A2->create_induced_action_on_sets(
			Spread_tables->nb_spreads, spread_size,
			Spread_tables->spread_table,
			0 /* verbose_level */);

	cout << "created action on spreads" << endl;

	if (f_v) {
		cout << "packing_classify::create_action_on_spreads "
				"creating action A_on_spreads done" << endl;
	}
}



void packing_classify::report_fixed_objects(int *Elt,
		char *fname_latex, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i, j, cnt;
	//int v[4];
	//file_io Fio;

	if (f_v) {
		cout << "packing_classify::report_fixed_objects" << endl;
	}


	{
		ofstream fp(fname_latex);
		char title[1000];
		latex_interface L;

		sprintf(title, "Fixed Objects");

		L.head(fp,
			FALSE /* f_book */, TRUE /* f_title */,
			title, "" /* const char *author */,
			FALSE /* f_toc */, FALSE /* f_landscape */, TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */, TRUE /* f_pagenumbers */,
			NULL /* extra_praeamble */);
		//latex_head_easy(fp);

	
		T->A->report_fixed_objects_in_P3(fp,
				P3,
				Elt,
				verbose_level);
	
#if 0
		fp << "\\section{Fixed Objects}" << endl;



		fp << "The element" << endl;
		fp << "$$" << endl;
		T->A->element_print_latex(Elt, fp);
		fp << "$$" << endl;
		fp << "has the following fixed objects:" << endl;


		fp << "\\subsection{Fixed Points}" << endl;

		cnt = 0;
		for (i = 0; i < P3->N_points; i++) {
			j = T->A->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				cnt++;
				}
			}

		fp << "There are " << cnt << " fixed points, they are: \\\\" << endl;
		for (i = 0; i < P3->N_points; i++) {
			j = T->A->element_image_of(i, Elt, 0 /* verbose_level */);
			F->PG_element_unrank_modified(v, 1, 4, i);
			if (j == i) {
				fp << i << " : ";
				int_vec_print(fp, v, 4);
				fp << "\\\\" << endl;
				cnt++;
				}
			}
	
		fp << "\\subsection{Fixed Lines}" << endl;

		{
		action *A2;

		A2 = T->A->induced_action_on_grassmannian(2, 0 /* verbose_level*/);

		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				cnt++;
				}
			}

		fp << "There are " << cnt << " fixed lines, they are: \\\\" << endl;
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				fp << i << " : $\\left[";
				A2->G.AG->G->print_single_generator_matrix_tex(fp, i);
				fp << "\\right]$\\\\" << endl;
				cnt++;
				}
			}

		FREE_OBJECT(A2);
		}
	
		fp << "\\subsection{Fixed Planes}" << endl;

		{
		action *A2;

		A2 = T->A->induced_action_on_grassmannian(3, 0 /* verbose_level*/);

		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				cnt++;
				}
			}

		fp << "There are " << cnt << " fixed planes, they are: \\\\" << endl;
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				fp << i << " : $\\left[";
				A2->G.AG->G->print_single_generator_matrix_tex(fp, i);
				fp << "\\right]$\\\\" << endl;
				cnt++;
				}
			}

		FREE_OBJECT(A2);
		}
#endif


		L.foot(fp);
	}
	file_io Fio;

	cout << "Written file " << fname_latex << " of size "
			<< Fio.file_size(fname_latex) << endl;

	
	if (f_v) {
		cout << "packing::report_fixed_objects done" << endl;
	}
}


int packing_classify::test_if_orbit_is_partial_packing(
	schreier *Orbits, int orbit_idx,
	long int *orbit1, int verbose_level)
{
	int f_v = FALSE; // (verbose_level >= 1);
	int len;

	if (f_v) {
		cout << "packing_classify::test_if_orbit_is_partial_packing "
				"orbit_idx = " << orbit_idx << endl;
	}
	Orbits->get_orbit(orbit_idx, orbit1, len, 0 /* verbose_level*/);
	return Spread_tables->test_if_set_of_spreads_is_line_disjoint(orbit1, len);
}

int packing_classify::test_if_pair_of_orbits_are_adjacent(
	schreier *Orbits, int a, int b,
	long int *orbit1, long int *orbit2,
	int verbose_level)
// tests if every spread from orbit a
// is line-disjoint from every spread from orbit b
{
	int f_v = FALSE; // (verbose_level >= 1);
	int len1, len2;

	if (f_v) {
		cout << "packing_classify::test_if_pair_of_orbits_are_adjacent "
				"a=" << a << " b=" << b << endl;
	}
	if (a == b) {
		return FALSE;
	}
	Orbits->get_orbit(a, orbit1, len1, 0 /* verbose_level*/);
	Orbits->get_orbit(b, orbit2, len2, 0 /* verbose_level*/);

	return Spread_tables->test_if_pair_of_sets_are_adjacent(
			orbit1, len1,
			orbit2, len2,
			verbose_level);
}



// #############################################################################
// global functions:
// #############################################################################

void callback_packing_compute_klein_invariants(
		isomorph *Iso, void *data, int verbose_level)
{
	packing_classify *P = (packing_classify *) data;
	int f_split = FALSE;
	int split_r = 0;
	int split_m = 1;
	
	P->compute_klein_invariants(Iso, f_split, split_r, split_m,
			verbose_level);
}


void callback_packing_report(isomorph *Iso,
		void *data, int verbose_level)
{
	packing_classify *P = (packing_classify *) data;
	
	P->report(Iso, verbose_level);
}


void packing_lifting_prepare_function_new(
	exact_cover *EC, int starter_case,
	long int *candidates, int nb_candidates,
	strong_generators *Strong_gens,
	diophant *&Dio, long int *&col_labels,
	int &f_ruled_out, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	packing_classify *P = (packing_classify *) EC->user_data;

	if (f_v) {
		cout << "packing_lifting_prepare_function_new "
				"nb_candidates=" << nb_candidates << endl;
	}

	P->lifting_prepare_function_new(
		EC, starter_case,
		candidates, nb_candidates, Strong_gens, 
		Dio, col_labels, f_ruled_out, 
		verbose_level);


	if (f_v) {
		cout << "packing_lifting_prepare_function_new "
				"after lifting_prepare_function_new" << endl;
	}

	if (f_v) {
		cout << "packing_lifting_prepare_function_new "
				"nb_rows=" << Dio->m
				<< " nb_cols=" << Dio->n << endl;
	}

	if (f_v) {
		cout << "packing_lifting_prepare_function_new done" << endl;
	}
}



void packing_early_test_function(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	packing_classify *P = (packing_classify *) data;
	int f_v = (verbose_level >= 1);
	long int i, k, a, b;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "packing_early_test_function for set ";
		print_set(cout, len, S);
		cout << endl;
	}
	a = S[len - 1];
	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		b = candidates[i];

		if (b == a) {
			continue;
		}
		if (P->bitvector_adjacency) {
			k = Combi.ij2k_lint(a, b, P->Spread_tables->nb_spreads);
			if (bitvector_s_i(P->bitvector_adjacency, k)) {
				good_candidates[nb_good_candidates++] = b;
			}
		}
		else {
			if (P->Spread_tables->test_if_spreads_are_disjoint(a, b)) {
				good_candidates[nb_good_candidates++] = b;
			}
		}
	}
	if (f_v) {
		cout << "packing_early_test_function done" << endl;
	}
}




int count(int *Inc, int n, int m, int *set, int t)
{
	int i, j;
	int nb, h;
	
	nb = 0;
	for (j = 0; j < m; j++) {
		for (h = 0; h < t; h++) {
			i = set[h];
			if (Inc[i * m + j] == 0) {
				break;
			}
		}
		if (h == t) {
			nb++;
		}
	}
	return nb;
}

int count_and_record(int *Inc,
		int n, int m, int *set, int t, int *occurances)
{
	int i, j;
	int nb, h;
	
	nb = 0;
	for (j = 0; j < m; j++) {
		for (h = 0; h < t; h++) {
			i = set[h];
			if (Inc[i * m + j] == 0) {
				break;
			}
		}
		if (h == t) {
			occurances[nb++] = j;
		}
	}
	return nb;
}

int packing_spread_compare_func(void *data, int i, int j, void *extra_data)
{
	packing_classify *P = (packing_classify *) extra_data;
	long int **Sets = (long int **) data;
	int ret;

	ret = lint_vec_compare(Sets[i], Sets[j], P->spread_size);
	return ret;
}

void packing_swap_func(void *data, int i, int j, void *extra_data)
{
	packing_classify *P = (packing_classify *) extra_data;
	int *d = P->tmp_isomorphism_type_of_spread;
	long int **Sets = (long int **) data;
	long int *p;
	int a;

	p = Sets[i];
	Sets[i] = Sets[j];
	Sets[j] = p;

	a = d[i];
	d[i] = d[j];
	d[j] = a;
}


}}

