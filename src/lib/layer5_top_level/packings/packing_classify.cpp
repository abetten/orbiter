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
namespace layer5_applications {
namespace packings {


#if 0
static void callback_packing_compute_klein_invariants(
		isomorph *Iso, void *data, int verbose_level);
static void callback_packing_report(isomorph *Iso,
		void *data, int verbose_level);
static void packing_lifting_prepare_function_new(
	exact_cover *EC, int starter_case,
	long int *candidates, int nb_candidates,
	groups::strong_generators *Strong_gens,
	solvers::diophant *&Dio, long int *&col_labels,
	int &f_ruled_out,
	int verbose_level);
#endif
static void packing_early_test_function(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
#if 0
static int count(int *Inc, int n, int m, int *set, int t);
static int count_and_record(int *Inc, int n, int m,
		int *set, int t, int *occurances);
#endif

packing_classify::packing_classify()
{
	Record_birth();
	PA = NULL;
	T = NULL;
	F = NULL;
	spread_size = 0;
	nb_lines = 0;


	f_lexorder_test = true;
	q = 0;
	size_of_packing = 0;
		// the number of spreads in a packing,
		// which is q^2 + q + 1

	Spread_table_with_selection = NULL;

	P3 = NULL;
	P5 = NULL;
	the_packing = NULL;
	spread_iso_type = NULL;
	dual_packing = NULL;
	list_of_lines = NULL;
	list_of_lines_klein_image = NULL;
	Gr = NULL;


	degree = NULL;

	Control = NULL;
	Poset = NULL;
	gen = NULL;

	nb_needed = 0;

}

packing_classify::~packing_classify()
{
	Record_death();
	if (Spread_table_with_selection) {
		FREE_OBJECT(Spread_table_with_selection);
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
}


void packing_classify::init(
		projective_geometry::projective_space_with_action *PA3,
		projective_geometry::projective_space_with_action *PA5,
		spreads::spread_table_with_selection
			*Spread_table_with_selection,
		int f_lexorder_test,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_classify::init" << endl;
	}


	packing_classify::PA = PA3;
	packing_classify::Spread_table_with_selection = Spread_table_with_selection;


	packing_classify::T = Spread_table_with_selection->T;
	F = Spread_table_with_selection->T->SD->F;
	q = Spread_table_with_selection->T->SD->q;
	spread_size = Spread_table_with_selection->T->SD->spread_size;
	size_of_packing = q * q + q + 1;
	nb_lines = Spread_table_with_selection->T->A2->degree;

	int nb_points;


	nb_points = Spread_table_with_selection->T->A->degree;

	packing_classify::f_lexorder_test = f_lexorder_test;

	
	if (f_v) {
		cout << "packing_classify::init q=" << q << endl;
		cout << "packing_classify::init nb_points=" << nb_points << endl;
		cout << "packing_classify::init nb_lines=" << nb_lines << endl;
		cout << "packing_classify::init spread_size=" << spread_size << endl;
		cout << "packing_classify::init size_of_packing=" << size_of_packing << endl;
		//cout << "packing_classify::init input_prefix=" << input_prefix << endl;
		//cout << "packing_classify::init base_fname=" << base_fname << endl;
	}

	if (f_v) {
		cout << "packing_classify::init before init_P3_and_P5_and_Gr" << endl;
	}
	init_P3_and_P5_and_Gr(PA3->P, PA5->P, verbose_level - 1);
	if (f_v) {
		cout << "packing_classify::init after init_P3_and_P5_and_Gr" << endl;
	}


	if (f_v) {
		cout << "packing_classify::init done" << endl;
	}
}

void packing_classify::init2(
		poset_classification::poset_classification_control *Control,
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
	Spread_table_with_selection->create_action_on_spreads(verbose_level - 1);
	if (f_v) {
		cout << "packing_classify::init2 "
				"after create_action_on_spreads" << endl;
	}

	packing_classify::Control = Control;

	
	if (f_v) {
		cout << "packing_classify::init "
				"before prepare_generator" << endl;
	}
	prepare_generator(verbose_level - 1);
	if (f_v) {
		cout << "packing_classify::init "
				"after prepare_generator" << endl;
	}

	if (f_v) {
		cout << "packing_classify::init done" << endl;
	}
}



void packing_classify::init_P3_and_P5_and_Gr(
		geometry::projective_geometry::projective_space *P3,
		geometry::projective_geometry::projective_space *P5,
		int verbose_level)
// creates a Grassmann 6,3
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_classify::init_P3_and_P5_and_Gr" << endl;
	}

	packing_classify::P3 = P3;
	packing_classify::P5 = P5;

#if 0
	P3 = NEW_OBJECT(geometry::projective_space);
	
	P3->projective_space_init(3, F,
		true /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);


	P5 = NEW_OBJECT(geometry::projective_space);

	P5->projective_space_init(5, F,
		true /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);

#endif
	if (f_v) {
		cout << "packing_classify::init_P3_and_P5_and_Gr P3->N_points=" << P3->Subspaces->N_points << endl;
		cout << "packing_classify::init_P3_and_P5_and_Gr P3->N_lines=" << P3->Subspaces->N_lines << endl;
		cout << "packing_classify::init_P3_and_P5_and_Gr P5->N_points=" << P5->Subspaces->N_points << endl;
		cout << "packing_classify::init_P3_and_P5_and_Gr P5->N_lines=" << P5->Subspaces->N_lines << endl;
	}

	the_packing = NEW_lint(size_of_packing);
	spread_iso_type = NEW_lint(size_of_packing);
	dual_packing = NEW_lint(size_of_packing);
	list_of_lines = NEW_lint(size_of_packing * spread_size);
	list_of_lines_klein_image = NEW_lint(size_of_packing * spread_size);

	Gr = NEW_OBJECT(geometry::projective_geometry::grassmann);

	Gr->init(6, 3, F, 0 /* verbose_level */);

	if (f_v) {
		cout << "packing_classify::init_P3_and_P5_and_Gr done" << endl;
	}
}









void packing_classify::prepare_generator(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "packing_classify::prepare_generator" << endl;
	}
	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
	Poset->init_subset_lattice(
			T->A,
			Spread_table_with_selection->A_on_spreads,
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
	Control->f_print_function = true;
	Control->print_function = print_set;
	Control->print_function_data = this;
#endif


	if (f_v) {
		cout << "packing_classify::prepare_generator "
				"calling gen->initialize" << endl;
	}

	gen = NEW_OBJECT(poset_classification::poset_classification);

	gen->initialize_and_allocate_root_node(
			Control, Poset,
			size_of_packing,
			verbose_level - 1);

	if (f_v) {
		cout << "packing_classify::prepare_generator done" << endl;
	}
}


void packing_classify::compute(
		int search_depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int schreier_depth = search_depth;
	int f_use_invariant_subset_if_available = true;
	int f_debug = false;
	int t0;
	other::orbiter_kernel_system::os_interface Os;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "packing_classify::compute" << endl;
	}

	if (f_v) {
		cout << "packing_classify::compute before poset_classification_main" << endl;
	}
	gen->poset_classification_main(
			t0,
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		verbose_level - 1);
	
	int length;
	
	if (f_v) {
		cout << "packing_classify::compute after poset_classification_main" << endl;
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
		solvers_package::exact_cover *E,
		int starter_case,
	long int *candidates, int nb_candidates,
	groups::strong_generators *Strong_gens,
	combinatorics::solvers::diophant *&Dio, long int *&col_labels,
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

	Spread_table_with_selection->compute_covered_points(
			points_covered_by_starter,
		nb_points_covered_by_starter, 
		E->starter, E->starter_size, 
		verbose_level - 1);


	if (f_v) {
		cout << "packing_classify::lifting_prepare_function "
				"before compute_free_points2" << endl;
	}

	Spread_table_with_selection->compute_free_points2(
		free_points2, nb_free_points2, free_point_idx,
		points_covered_by_starter, nb_points_covered_by_starter, 
		E->starter, E->starter_size, 
		verbose_level - 1);

	if (f_v) {
		cout << "packing_classify::lifting_prepare_function "
				"before compute_live_blocks2" << endl;
	}

	Spread_table_with_selection->compute_live_blocks2(
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


	Lint_vec_copy(live_blocks2, col_labels, nb_cols);


	if (f_vv) {
		cout << "packing_classify::lifting_prepare_function_new candidates: ";
		Lint_vec_print(cout, col_labels, nb_cols);
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

	Spread_table_with_selection->Spread_tables->make_exact_cover_problem(
			Dio,
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






int packing_classify::test_if_orbit_is_partial_packing(
		groups::schreier *Orbits, int orbit_idx,
	long int *orbit1, int verbose_level)
{
	int f_v = false; // (verbose_level >= 1);
	int len;

	if (f_v) {
		cout << "packing_classify::test_if_orbit_is_partial_packing "
				"orbit_idx = " << orbit_idx << endl;
	}
	Orbits->Forest->get_orbit(orbit_idx, orbit1, len, 0 /* verbose_level*/);
	return Spread_table_with_selection->Spread_tables->test_if_set_of_spreads_is_line_disjoint(orbit1, len);
}

int packing_classify::test_if_pair_of_orbits_are_adjacent(
		groups::schreier *Orbits, int a, int b,
	long int *orbit1, long int *orbit2,
	int verbose_level)
// tests if every spread from orbit a
// is line-disjoint from every spread from orbit b
{
	int f_v = false; // (verbose_level >= 1);
	int len1, len2;

	if (f_v) {
		cout << "packing_classify::test_if_pair_of_orbits_are_adjacent "
				"a=" << a << " b=" << b << endl;
	}
	if (a == b) {
		return false;
	}
	Orbits->Forest->get_orbit(a, orbit1, len1, 0 /* verbose_level*/);
	Orbits->Forest->get_orbit(b, orbit2, len2, 0 /* verbose_level*/);

	return Spread_table_with_selection->Spread_tables->test_if_pair_of_sets_are_adjacent(
			orbit1, len1,
			orbit2, len2,
			verbose_level);
}


int packing_classify::find_spread(
		long int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx;

	if (f_v) {
		cout << "packing_classify::find_spread" << endl;
	}
	idx = Spread_table_with_selection->find_spread(set, verbose_level);
	return idx;
}



// #############################################################################
// global functions:
// #############################################################################

#if 0
static void callback_packing_compute_klein_invariants(
		isomorph *Iso, void *data, int verbose_level)
{
	packing_classify *P = (packing_classify *) data;
	int f_split = false;
	int split_r = 0;
	int split_m = 1;
	
	P->compute_klein_invariants(Iso, f_split, split_r, split_m,
			verbose_level);
}

static void callback_packing_report(isomorph *Iso,
		void *data, int verbose_level)
{
	packing_classify *P = (packing_classify *) data;
	
	P->report(Iso, verbose_level);
}


static void packing_lifting_prepare_function_new(
	exact_cover *EC, int starter_case,
	long int *candidates, int nb_candidates,
	groups::strong_generators *Strong_gens,
	solvers::diophant *&Dio, long int *&col_labels,
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
#endif



static void packing_early_test_function(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	packing_classify *P = (packing_classify *) data;
	int f_v = (verbose_level >= 1);
	long int i, a, b;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "packing_early_test_function for set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
	}
	if (len == 0) {
		nb_good_candidates = nb_candidates;
		Lint_vec_copy(candidates, good_candidates, nb_candidates);
	}
	else {
		a = S[len - 1];
		if (f_v) {
			cout << "packing_early_test_function a = " << a << endl;
		}
		nb_good_candidates = 0;
		for (i = 0; i < nb_candidates; i++) {
			b = candidates[i];

			if (b == a) {
				continue;
			}
#if 0
			if (P->bitvector_adjacency) {
				k = Combi.ij2k_lint(a, b, P->Spread_table_with_selection->Spread_tables->nb_spreads);
				if (bitvector_s_i(P->bitvector_adjacency, k)) {
					good_candidates[nb_good_candidates++] = b;
				}
			}
			else {
				if (P->Spread_table_with_selection->Spread_tables->test_if_spreads_are_disjoint(a, b)) {
					good_candidates[nb_good_candidates++] = b;
				}
			}
#else
			if (P->Spread_table_with_selection->is_adjacent(a, b)) {
				good_candidates[nb_good_candidates++] = b;
			}
#endif
		}
	}
	if (f_v) {
		cout << "packing_early_test_function "
				"nb_candidates: " << nb_candidates << " -> " <<  nb_good_candidates << endl;
	}
	if (f_v) {
		cout << "packing_early_test_function done" << endl;
	}
}




#if 0
static int count(int *Inc, int n, int m, int *set, int t)
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

static int count_and_record(int *Inc,
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
#endif


}}}

