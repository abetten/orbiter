// regular_ls_classify.cpp
// 
// Anton Betten
// Jan 1, 2013

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


regular_ls_classify::regular_ls_classify()
{
	Descr = NULL;

	m2 = 0;
	v1 = NULL; // [k]

	Poset = NULL;
	gen = NULL;
	A = NULL;
	A2 = NULL;
	Aonk = NULL; // only a pointer, do not free

	row_sum = NULL;
	pairs = NULL;
	open_rows = NULL;
	open_row_idx = NULL;
	open_pairs = NULL;
	open_pair_idx = NULL;
	//null();
}

regular_ls_classify::~regular_ls_classify()
{
	freeself();
}

void regular_ls_classify::null()
{
}


void regular_ls_classify::freeself()
{
	if (row_sum) {
		FREE_int(row_sum);
	}
	if (pairs) {
		FREE_int(pairs);
	}
	if (open_rows) {
		FREE_int(open_rows);
	}
	if (open_row_idx) {
		FREE_int(open_row_idx);
	}
	if (open_pairs) {
		FREE_int(open_pairs);
	}
	if (open_pair_idx) {
		FREE_int(open_pair_idx);
	}
	null();
}



void regular_ls_classify::init_and_run(
		regular_linear_space_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "regular_ls_classify::init_and_run" << endl;
		cout << "regular_ls_classify::init_and_run "
				"m=" << Descr->m << " n=" << Descr->n << " k=" << Descr->k << " r=" << Descr->r << endl;
		cout << "regular_ls_classify::init_basic starter_size=" << Descr->starter_size << endl;
	}

	regular_ls_classify::Descr = Descr;
	if (!Descr->f_has_control) {
		cout << "regular_ls_classify::init_and_run please use option -control" << endl;
		exit(1);
	}

	m2 = (Descr->m * (Descr->m - 1)) >> 1;
	v1 = NEW_int(Descr->m);


	row_sum = NEW_int(Descr->m);
	pairs = NEW_int(m2);
	open_rows = NEW_int(Descr->m);
	open_row_idx = NEW_int(Descr->m);
	open_pairs = NEW_int(m2);
	open_pair_idx = NEW_int(m2);

	if (f_v) {
		cout << "regular_ls_classify::init_and_run before init_group" << endl;
	}
	init_group(verbose_level);
	if (f_v) {
		cout << "regular_ls_classify::init_and_run after init_group" << endl;
	}

	if (f_v) {
		cout << "regular_ls_classify::init_and_run before init_action_on_k_subsets" << endl;
	}
	init_action_on_k_subsets(Descr->k, verbose_level);
	if (f_v) {
		cout << "regular_ls_classify::init_and_run after init_action_on_k_subsets" << endl;
	}

	if (f_v) {
		cout << "regular_ls_classify::init_and_run before init_generator" << endl;
	}
	init_generator(
			Descr->Control,
			A->Strong_gens,
			verbose_level);
	if (f_v) {
		cout << "regular_ls_classify::init_and_run after init_generator" << endl;
	}


	os_interface Os;
	int schreier_depth = Descr->target_size;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	int t0 = Os.os_ticks();

	if (f_v) {
		cout << "regular_ls_classify::init_and_run "
				"calling gen->main" << endl;
	}
	gen->main(t0,
		schreier_depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level - 1);
	if (f_v) {
		cout << "regular_ls_classify::init_and_run "
				"after gen->main" << endl;
	}


	if (f_v) {
		cout << "regular_ls_classify::init_and_run done" << endl;
	}
}

void regular_ls_classify::init_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "regular_ls_classify::init_group" << endl;
	}

	if (f_v) {
		cout << "regular_ls_classify::init_group "
				"creating symmetric group of degree " << Descr->m << endl;
	}
	A = NEW_OBJECT(action);
	int f_no_base = FALSE;

	A->init_symmetric_group(Descr->m /* degree */, f_no_base, 0 /* verbose_level - 2*/);
	

	if (f_v) {
		cout << "regular_ls_generator::init_group done" << endl;
	}
}

void regular_ls_classify::init_action_on_k_subsets(
		int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "regular_ls_classify::init_action_on_k_subsets" << endl;
	}

	//regular_ls_generator::onk = onk;

	if (f_v) {
		cout << "regular_ls_classify::init_action_on_k_subsets "
				"creating action on k-subsets for k=" << k << endl;
	}
	A2 = NEW_OBJECT(action);
	A2->induced_action_on_k_subsets(*A, k, verbose_level - 2);

	Aonk = A2->G.on_k_subsets;
	
	if (f_v) {
		cout << "regular_ls_classify::init_action_on_k_subsets "
				"before A2->induced_action_override_sims" << endl;
	}

	if (f_v) {
		cout << "regular_ls_classify::init_action_on_k_subsets "
				"done" << endl;
	}
}

void regular_ls_classify::init_generator(
		poset_classification_control *Control,
		strong_generators *Strong_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i;

	if (f_v) {
		cout << "regular_ls_classify::init_generator" << endl;
	}
	



	Poset = NEW_OBJECT(poset_with_group_action);
	Poset->init_subset_lattice(A, A2,
			Strong_gens,
			verbose_level);
	Poset->add_testing_without_group(
			regular_ls_classify_early_test_function,
				this /* void *data */,
				verbose_level);

	Poset->f_print_function = FALSE;
	Poset->print_function = regular_ls_classify_print_set;
	Poset->print_function_data = (void *) this;

	
	gen = NEW_OBJECT(poset_classification);

	gen->initialize_and_allocate_root_node(
			Control, Poset,
			Descr->target_size /* gen->depth ToDo */,
			0/*verbose_level - 3*/);
	
	

	if (f_v) {
		cout << "regular_ls_classify::init_generator done" << endl;
	}
}


void regular_ls_classify::early_test_func(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	//verbose_level = 10;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, a, b, p;
	int f_OK;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "regular_ls_classify::early_test_func checking set ";
		Orbiter->Lint_vec.print(cout, S, len);
		cout << endl;
		cout << "candidate set of size " << nb_candidates << ":" << endl;
		Orbiter->Lint_vec.print(cout, candidates, nb_candidates);
		cout << endl;
	}
	Orbiter->Int_vec.zero(pairs, m2);
	Orbiter->Int_vec.zero(row_sum, Descr->m);
	//int_vec_copy(initial_pair_covering, pairs, m2);

#if 0
	if (f_vv) {
		cout << "pairs initially:" << endl;
		int_vec_print(cout, pairs, m2);
		cout << endl;
	}
#endif

	for (i = 0; i < len; i++) {

		Combi.unrank_k_subset(S[i], v1, Descr->m, Descr->k);
		for (a = 0; a < Descr->k; a++) {
			row_sum[v1[a]]++;
			for (b = a + 1; b < Descr->k; b++) {
				p = Combi.ij2k(v1[a], v1[b], Descr->m);
				pairs[p] = TRUE;
			}
		}
		
	}
	if (f_vv) {
		cout << "pairs after adding in the chosen sets, pairs=" << endl;
		Orbiter->Int_vec.print(cout, pairs, m2);
		cout << endl;
	}
	

	nb_good_candidates = 0;
	
	for (j = 0; j < nb_candidates; j++) {
		f_OK = TRUE;

		if (f_vv) {
			cout << "Testing candidate " << j << " = "
					<< candidates[j] << endl;
		}

		// do candidate testing:

		Combi.unrank_k_subset(candidates[j], v1, Descr->m, Descr->k);
		if (f_vv) {
			cout << "Testing candidate " << j << " = "
					<< candidates[j] << " = ";
			Orbiter->Int_vec.print(cout, v1, Descr->k);
			cout << endl;
		}
		for (a = 0; a < Descr->k; a++) {
			if (row_sum[v1[a]] == Descr->r) {
				f_OK = FALSE;
				break;
			}
			for (b = a + 1; b < Descr->k; b++) {
				p = Combi.ij2k(v1[a], v1[b], Descr->m);
				if (pairs[p]) {
					f_OK = FALSE;
					break;
				}
			}
			if (!f_OK) {
				break;
			}
		}


		if (f_OK) {
			if (f_vv) {
				cout << "Testing candidate " << j << " = "
						<< candidates[j] << " is good" << endl;
			}
			good_candidates[nb_good_candidates++] = candidates[j];
		}
	}
}

void regular_ls_classify::print(ostream &ost, long int *S, int len)
{
	int i;
	
	for (i = 0; i < len; i++) {
		ost << S[i] << " ";
	}
	ost << endl;
}

void regular_ls_classify::lifting_prepare_function_new(
	exact_cover *E, int starter_case,
	long int *candidates, int nb_candidates, strong_generators *Strong_gens,
	diophant *&Dio, long int *&col_labels,
	int &f_ruled_out, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, a, h1, h2, p, idx;
	int nb_needed;
	int nb_open_rows, nb_open_pairs;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "regular_ls_classify::lifting_prepare_function_new "
				"nb_candidates=" << nb_candidates << endl;
	}

	nb_needed = Descr->target_size - E->starter_size;
	f_ruled_out = FALSE;


	//int_vec_copy(initial_pair_covering, pairs, m2);
	Orbiter->Int_vec.zero(pairs, m2);
	Orbiter->Int_vec.zero(row_sum, Descr->m);

#if 0
	if (f_vv) {
		cout << "pairs initially:" << endl;
		int_vec_print(cout, pairs, m2);
		cout << endl;
	}
#endif

	for (i = 0; i < E->starter_size; i++) {

		Combi.unrank_k_subset(E->starter[i], v1, Descr->m, Descr->k);
		for (h1 = 0; h1 < Descr->k; h1++) {
			row_sum[v1[h1]]++;
			for (h2 = h1 + 1; h2 < Descr->k; h2++) {
				p = Combi.ij2k(v1[h1], v1[h2], Descr->m);
				pairs[p] = TRUE;
			}
		}
	}

	nb_open_rows = 0;
	Orbiter->Int_vec.mone(open_row_idx, Descr->m);
	for (i = 0; i < Descr->m; i++) {
		if (row_sum[i] < Descr->r) {
			open_rows[nb_open_rows] = i;
			open_row_idx[i] = nb_open_rows;
			nb_open_rows++;
		}
	}

	nb_open_pairs = 0;
	Orbiter->Int_vec.mone(open_pair_idx, m2);

	for (i = 0; i < m2; i++) {
		if (pairs[i] == FALSE) {
			open_pairs[nb_open_pairs] = i;
			open_pair_idx[i] = nb_open_pairs;
			nb_open_pairs++;
		}
	}

	
	col_labels = NEW_lint(nb_candidates);


	Orbiter->Lint_vec.copy(candidates, col_labels, nb_candidates);

	if (E->f_lex) {
		E->lexorder_test(col_labels, nb_candidates, Strong_gens->gens, 
			verbose_level - 2);
	}

	if (f_vv) {
		cout << "regular_ls_classify::lifting_prepare_function_new "
				"after lexorder test" << endl;
		cout << "regular_ls_classify::lifting_prepare_function_new "
				"nb_candidates=" << nb_candidates << endl;
	}

	// compute the incidence matrix between
	// open rows and open pairs versus candidate blocks:


	int nb_rows;
	int nb_cols;

	nb_rows = nb_open_rows + nb_open_pairs;
	nb_cols = nb_candidates;

	Dio = NEW_OBJECT(diophant);
	Dio->open(nb_rows, nb_cols);
	Dio->f_has_sum = TRUE;
	Dio->sum = nb_needed;

	for (i = 0; i < nb_open_rows; i++) {
		Dio->type[i] = t_EQ;
		Dio->RHS[i] = Descr->r - row_sum[open_rows[i]];
	}

	for (i = 0; i < nb_open_pairs; i++) {
		Dio->type[nb_open_rows + i] = t_LE;
		Dio->RHS[nb_open_rows + i] = 1;
	}

	Dio->fill_coefficient_matrix_with(0);
	Dio->set_x_min_constant(0);
	Dio->set_x_max_constant(1);


	for (i = 0; i < nb_candidates; i++) {
		a = col_labels[i];


		Combi.unrank_k_subset(a, v1, Descr->m, Descr->k);

		for (h1 = 0; h1 < Descr->k; h1++) {

			if (row_sum[v1[h1]] == Descr->r) {
				cout << "regular_ls_classify::lifting_prepare_function_new "
						"row_sum[v1[h1]] == Descr->r" << endl;
				exit(1);
			}
			idx = open_row_idx[v1[h1]];
			Dio->Aij(idx, i) = 1;
			
			for (h2 = h1 + 1; h2 < Descr->k; h2++) {
				p = Combi.ij2k(v1[h1], v1[h2], Descr->m);
				if (pairs[p]) {
					cout << "regular_ls_classify::lifting_prepare_function_new "
							"pairs[p]" << endl;
					exit(1);
				}
				idx = open_pair_idx[p];
				Dio->Aij(nb_open_rows + idx, i) = 1;
			}
		}
	}


	
	if (f_v) {
		cout << "regular_ls_classify::lifting_prepare_function_new "
				"done" << endl;
	}
}





// #############################################################################
// global functions:
// #############################################################################



void regular_ls_classify_print_set(ostream &ost, int len, long int *S, void *data)
{
	regular_ls_classify *Gen = (regular_ls_classify *) data;
	
	//print_vector(ost, S, len);
	Gen->print(ost, S, len);
}

void regular_ls_classify_early_test_function(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	regular_ls_classify *Gen = (regular_ls_classify *) data;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "regular_ls_classify_early_test_function for set ";
		Orbiter->Lint_vec.print(cout, S, len);
		cout << endl;
	}
	Gen->early_test_func(S, len, 
		candidates, nb_candidates, 
		good_candidates, nb_good_candidates, 
		verbose_level - 2);
	if (f_v) {
		cout << "regular_ls_classify_early_test_function done" << endl;
	}
}

void regular_ls_classify_lifting_prepare_function_new(
	exact_cover *EC, int starter_case,
	long int *candidates, int nb_candidates, strong_generators *Strong_gens,
	diophant *&Dio, long int *&col_labels,
	int &f_ruled_out, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	regular_ls_classify *Gen = (regular_ls_classify *) EC->user_data;

	if (f_v) {
		cout << "regular_ls_classify_lifting_prepare_function_new "
				"nb_candidates=" << nb_candidates << endl;
	}

	Gen->lifting_prepare_function_new(EC, starter_case, 
		candidates, nb_candidates, Strong_gens, 
		Dio, col_labels, f_ruled_out, 
		verbose_level - 1);


	if (f_v) {
		cout << "regular_ls_classify_lifting_prepare_function_new "
				"nb_rows=" << Dio->m << " nb_cols=" << Dio->n << endl;
	}

	if (f_v) {
		cout << "regular_ls_classify_lifting_prepare_function_new "
				"done" << endl;
	}
}

}}


