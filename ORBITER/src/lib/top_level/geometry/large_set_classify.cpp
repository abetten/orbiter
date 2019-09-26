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

	A_on_designs = NULL;


	bitvector_adjacency = NULL;
	bitvector_length = 0;
	degree = 0;

	Poset = NULL;
	gen = NULL;

	nb_needed = 0;
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
	for (i = 0; i < nb_designs; i++) {
		cout << i << " : ";
		int_vec_print(cout, Design_table + i * design_size, design_size);
		cout << endl;
	}

	for (i = 0; i < nb_designs; i++) {
		FREE_int(Sets[i]);
	}
	FREE_pint(Sets);


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

	t0 = os_ticks();

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

void large_set_classify::read_classification_single_case(orbit_rep *&Rep,
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

	Rep = NEW_OBJECT(orbit_rep);

	orbit_transversal *T;
	T = NEW_OBJECT(orbit_transversal);

	T->read_from_file_one_case_only(gen->Poset->A, gen->Poset->A2,
			fname_classification_at_level, case_nr, verbose_level - 1);

	//cout << "large_set_classify::read_classification_single_case before memcpy" << endl;
	memcpy(Rep, &T->Reps[case_nr], sizeof(orbit_rep));
	//cout << "large_set_classify::read_classification_single_case before null()" << endl;
	T->Reps[case_nr].null();

	//cout << "large_set_classify::read_classification_single_case before FREE_OBJECT(T)" << endl;
	FREE_OBJECT(T);
	//cout << "large_set_classify::read_classification_single_case after FREE_OBJECT(T)" << endl;

	if (f_v) {
		cout << "large_set_classify::read_classification_single_case done" << endl;
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



}}

