// regular_ls_generator.C
// 
// Anton Betten
// Jan 1, 2013

#include "orbiter.h"
#include "regular_ls.h"

void regular_ls_generator::init_basic(int argc, const char **argv, 
	const char *input_prefix, const char *base_fname, 
	INT starter_size, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "regular_ls_generator::init_basic" << endl;
		}

	regular_ls_generator::starter_size = starter_size;

	gen = NEW_OBJECT(poset_classification);
	
	if (f_vv) {
		cout << "regular_ls_generator::init_basic before read_arguments" << endl;
		}

	read_arguments(argc, argv);

	strcpy(starter_directory_name, input_prefix);
	strcpy(prefix, base_fname);
	sprintf(prefix_with_directory, "%s%s", starter_directory_name, base_fname);

	target_size = n;

	m2 = (m * (m - 1)) >> 1;
	v1 = NEW_INT(m);

	row_sum = NEW_INT(m);
	pairs = NEW_INT(m2);
	open_rows = NEW_INT(m);
	open_row_idx = NEW_INT(m);
	open_pairs = NEW_INT(m2);
	open_pair_idx = NEW_INT(m2);

}

void regular_ls_generator::read_arguments(int argc, const char **argv)
{
	INT i;
	INT f_m = FALSE;
	INT f_n = FALSE;
	INT f_k = FALSE;
	INT f_r = FALSE;
	
	gen->read_arguments(argc, argv, 0);
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-m") == 0) {
			f_m = TRUE;
			m = atoi(argv[++i]);
			cout << "-m " << m << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-k") == 0) {
			f_k = TRUE;
			k = atoi(argv[++i]);
			cout << "-k " << k << endl;
			}
		else if (strcmp(argv[i], "-r") == 0) {
			f_r = TRUE;
			r = atoi(argv[++i]);
			cout << "-r " << r << endl;
			}
		}
	if (!f_m) {
		cout << "regular_ls_generator::read_arguments Please use option -m <m>" << endl;
		exit(1);
		}
	if (!f_n) {
		cout << "regular_ls_generator::read_arguments Please use option -n <n>" << endl;
		exit(1);
		}
	if (!f_k) {
		cout << "regular_ls_generator::read_arguments Please use option -k <k>" << endl;
		exit(1);
		}
	if (!f_r) {
		cout << "regular_ls_generator::read_arguments Please use option -r <r>" << endl;
		exit(1);
		}
}

regular_ls_generator::regular_ls_generator()
{
	null();
}

regular_ls_generator::~regular_ls_generator()
{
	freeself();
}

void regular_ls_generator::null()
{
	gen = NULL;
	A = NULL;
	A2 = NULL;
	initial_pair_covering = NULL;
	row_sum = NULL;
	pairs = NULL;
	open_rows = NULL;
	open_row_idx = NULL;
	open_pairs = NULL;
	open_pair_idx = NULL;
}


void regular_ls_generator::freeself()
{
	if (initial_pair_covering) {
		FREE_INT(initial_pair_covering);
		}
	if (row_sum) {
		FREE_INT(row_sum);
		}
	if (pairs) {
		FREE_INT(pairs);
		}
	if (open_rows) {
		FREE_INT(open_rows);
		}
	if (open_row_idx) {
		FREE_INT(open_row_idx);
		}
	if (open_pairs) {
		FREE_INT(open_pairs);
		}
	if (open_pair_idx) {
		FREE_INT(open_pair_idx);
		}
	null();
}

void regular_ls_generator::init_group(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "regular_ls_generator::init_group" << endl;
		}

	if (f_v) {
		cout << "regular_ls_generator::init_group creating symmetric group of degree " << m << endl;
		}
	A = NEW_OBJECT(action);
	A->init_symmetric_group(m /* degree */, 0 /* verbose_level - 2*/);
	

	if (f_v) {
		cout << "regular_ls_generator::init_group done" << endl;
		}
}

void regular_ls_generator::init_action_on_k_subsets(INT k, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "regular_ls_generator::init_action_on_k_subsets" << endl;
		}

	//regular_ls_generator::onk = onk;

	if (f_v) {
		cout << "regular_ls_generator::init_action_on_k_subsets creating action on k-subsets for k=" << k << endl;
		}
	A2 = NEW_OBJECT(action);
	A2->induced_action_on_k_subsets(*A, k, verbose_level - 2);

	Aonk = A2->G.on_k_subsets;
	
	if (f_v) {
		cout << "regular_ls_generator::init_action_on_k_subsets before A2->induced_action_override_sims" << endl;
		}

	if (f_v) {
		cout << "regular_ls_generator::init_action_on_k_subsets done" << endl;
		}
}

void regular_ls_generator::init_generator(
	INT f_has_initial_pair_covering, INT *initial_pair_covering,
	strong_generators *Strong_gens, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT i;

	if (f_v) {
		cout << "regular_ls_generator::init_generator" << endl;
		}
	if (regular_ls_generator::initial_pair_covering) {
		FREE_INT(regular_ls_generator::initial_pair_covering);
		}
	if (gen->f_max_depth) {
		gen->depth = gen->max_depth;
		}
	else {
		gen->depth = target_size;
		}
	regular_ls_generator::initial_pair_covering = NEW_INT(m2);
	if (f_has_initial_pair_covering) {
		for (i = 0; i < m2; i++) {
			regular_ls_generator::initial_pair_covering[i] = initial_pair_covering[i];
			}
		}
	else {
		for (i = 0; i < m2; i++) {
			regular_ls_generator::initial_pair_covering[i] = FALSE;
			}
		}
	
	if (f_v) {
		cout << "regular_ls_generator::init_generator depth = " << gen->depth << endl;
		}


	strcpy(gen->fname_base, prefix_with_directory);
	
	gen->init(A, A2, Strong_gens, gen->depth, 0/*verbose_level - 3*/);
	
#if 0
	// not needed since we have an early_test_func:
	gen->init_check_func(::check_conditions, 
		(void *)this /* candidate_check_data */);
#endif

	// we have an early test function:

	gen->init_early_test_func(
		rls_generator_early_test_function, 
		this,  
		verbose_level);

	// We also have an incremental check function. 
	// This is only used by the clique finder:
	gen->init_incremental_check_func(
		check_function_incremental_callback, 
		this /* candidate_check_data */);


	gen->f_print_function = TRUE;
	gen->print_function = print_set;
	gen->print_function_data = (void *) this;
	
	
	INT nb_nodes = ONE_MILLION;
	
	if (f_vv) {
		cout << "regular_ls_generator::init_generator calling init_poset_orbit_node with " << nb_nodes << " nodes" << endl;
		}
	
	gen->init_poset_orbit_node(nb_nodes, verbose_level - 1);

	if (f_vv) {
		cout << "regular_ls_generator::init_generator after init_root_node" << endl;
		}
	
	//cout << "verbose_level = " << verbose_level << endl;
	//cout << "verbose_level_group_theory = " << verbose_level_group_theory << endl;
	
	gen->root[0].init_root_node(gen, 0/*verbose_level - 2*/);
	if (f_v) {
		cout << "regular_ls_generator::init_generator done" << endl;
		}
}

void regular_ls_generator::compute_starter(
	//INT f_lex, 
	INT f_write_candidate_file, 
	INT f_draw_poset, INT f_embedded, INT f_sideways, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//char cmd[1000];

	if (f_v) {
		cout << "regular_ls_generator::compute_starter" << endl;
		}

	
	
	gen->f_W = TRUE;
	gen->compute_orbits(0 /* from_level */, starter_size /* to_level */, 
		//f_lex, 
		f_write_candidate_file, 
		verbose_level);


	if (f_draw_poset) {
		if (f_v) {
			cout << "regular_ls_generator::compute_starter before gen->draw_poset" << endl;
			}

		gen->draw_poset(prefix_with_directory, starter_size, 0 /* data1 */, f_embedded, f_sideways, 0 /* gen->verbose_level */);
		
		}
	if (f_v) {
		cout << "regular_ls_generator::compute_starter done" << endl;
		}

}

void regular_ls_generator::early_test_func(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	INT verbose_level)
{
	//verbose_level = 10;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, a, b, p;
	INT f_OK;

	if (f_v) {
		cout << "regular_ls_generator::early_test_func checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size " << nb_candidates << ":" << endl;
		INT_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		}
	INT_vec_zero(row_sum, m);
	INT_vec_copy(initial_pair_covering, pairs, m2);

	if (f_vv) {
		cout << "pairs initially:" << endl;
		INT_vec_print(cout, pairs, m2);
		cout << endl;
		}
	for (i = 0; i < len; i++) {

		unrank_k_subset(S[i], v1, m, k);
		for (a = 0; a < k; a++) {
			row_sum[v1[a]]++;
			for (b = a + 1; b < k; b++) {
				p = ij2k(v1[a], v1[b], m);
				pairs[p] = TRUE;
				}
			}
		
		}
	if (f_vv) {
		cout << "pairs after adding in the chosen sets:" << endl;
		INT_vec_print(cout, pairs, m2);
		cout << endl;
		}
	

	nb_good_candidates = 0;
	
	for (j = 0; j < nb_candidates; j++) {
		f_OK = TRUE;

		if (f_vv) {
			cout << "Testing candidate " << j << " = " << candidates[j] << endl;
			}

		// do some testing
		unrank_k_subset(candidates[j], v1, m, k);
		if (f_vv) {
			cout << "Testing candidate " << j << " = " << candidates[j] << " = ";
			INT_vec_print(cout, v1, k);
			cout << endl;
			}
		for (a = 0; a < k; a++) {
			if (row_sum[v1[a]] == r) {
				f_OK = FALSE;
				break;
				}
			for (b = a + 1; b < k; b++) {
				p = ij2k(v1[a], v1[b], m);
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
				cout << "Testing candidate " << j << " = " << candidates[j] << " is good" << endl;
				}
			good_candidates[nb_good_candidates++] = candidates[j];
			}
		}
}

INT regular_ls_generator::check_function_incremental(INT len, INT *S, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, a, b, p;
	INT f_OK;
		
	if (f_v) {
		cout << "regular_ls_generator::check_function_incremental checking set ";
		print_set(cout, len, S);
		cout << endl;
		}

	INT_vec_zero(row_sum, m);
	INT_vec_copy(initial_pair_covering, pairs, m2);

	for (i = 0; i < len - 1; i++) {

		unrank_k_subset(S[i], v1, m, k);
		for (a = 0; a < k; a++) {
			row_sum[v1[a]]++;
			for (b = a + 1; b < k; b++) {
				p = ij2k(v1[a], v1[b], m);
				pairs[p] = TRUE;
				}
			}
		
		}

	f_OK = TRUE;
	unrank_k_subset(S[len - 1], v1, m, k);
	for (a = 0; a < k; a++) {
		if (row_sum[v1[a]] == r) {
			f_OK = FALSE;
			break;
			}
		for (b = a + 1; b < k; b++) {
			p = ij2k(v1[a], v1[b], m);
			if (pairs[p]) {
				f_OK = FALSE;
				break;
				}
			}
		if (!f_OK) {
			break;
			}
		}

	return f_OK;
}

void regular_ls_generator::print(INT *S, INT len)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		cout << S[i] << " ";
		}
	cout << endl;
}

void regular_ls_generator::lifting_prepare_function_new(exact_cover *E, INT starter_case, 
	INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, INT *&col_labels, 
	INT &f_ruled_out, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, a, h1, h2, p, idx;
	INT nb_needed;
	INT nb_open_rows, nb_open_pairs;

	if (f_v) {
		cout << "regular_ls_generator::lifting_prepare_function_new nb_candidates=" << nb_candidates << endl;
		}

	nb_needed = target_size - E->starter_size;
	f_ruled_out = FALSE;


	INT_vec_zero(row_sum, m);
	INT_vec_copy(initial_pair_covering, pairs, m2);

	if (f_vv) {
		cout << "pairs initially:" << endl;
		INT_vec_print(cout, pairs, m2);
		cout << endl;
		}
	for (i = 0; i < E->starter_size; i++) {

		unrank_k_subset(E->starter[i], v1, m, k);
		for (h1 = 0; h1 < k; h1++) {
			row_sum[v1[h1]]++;
			for (h2 = h1 + 1; h2 < k; h2++) {
				p = ij2k(v1[h1], v1[h2], m);
				pairs[p] = TRUE;
				}
			}
		
		}

	nb_open_rows = 0;
	INT_vec_mone(open_row_idx, m);
	for (i = 0; i < m; i++) {
		if (row_sum[i] < r) {
			open_rows[nb_open_rows] = i;
			open_row_idx[i] = nb_open_rows;
			nb_open_rows++;
			}
		}

	nb_open_pairs = 0;
	INT_vec_mone(open_pair_idx, m2);

	for (i = 0; i < m2; i++) {
		if (pairs[i] == FALSE) {
			open_pairs[nb_open_pairs] = i;
			open_pair_idx[i] = nb_open_pairs;
			nb_open_pairs++;
			}
		}

	
	col_labels = NEW_INT(nb_candidates);


	INT_vec_copy(candidates, col_labels, nb_candidates);

	if (E->f_lex) {
		E->lexorder_test(col_labels, nb_candidates, Strong_gens->gens, 
			verbose_level - 2);
		}

	if (f_vv) {
		cout << "regular_ls_generator::lifting_prepare_function_new "
				"after lexorder test" << endl;
		cout << "regular_ls_generator::lifting_prepare_function_new "
				"nb_candidates=" << nb_candidates << endl;
		}

	// compute the incidence matrix between
	// open rows and open pairs versus candidate blocks:


	INT nb_rows;
	INT nb_cols;

	nb_rows = nb_open_rows + nb_open_pairs;
	nb_cols = nb_candidates;

	Dio = NEW_OBJECT(diophant);
	Dio->open(nb_rows, nb_cols);
	Dio->sum = nb_needed;

	for (i = 0; i < nb_open_rows; i++) {
		Dio->type[i] = t_EQ;
		Dio->RHS[i] = r - row_sum[open_rows[i]];
		}

	for (i = 0; i < nb_open_pairs; i++) {
		Dio->type[nb_open_rows + i] = t_LE;
		Dio->RHS[nb_open_rows + i] = 1;
		}

	Dio->fill_coefficient_matrix_with(0);


	for (i = 0; i < nb_candidates; i++) {
		a = col_labels[i];


		unrank_k_subset(a, v1, m, k);

		for (h1 = 0; h1 < k; h1++) {

			if (row_sum[v1[h1]] == r) {
				cout << "regular_ls_generator::lifting_prepare_"
						"function_new row_sum[v1[h1]] == onr" << endl;
				exit(1);
				}
			idx = open_row_idx[v1[h1]];
			Dio->Aij(idx, i) = 1;
			
			for (h2 = h1 + 1; h2 < k; h2++) {
				p = ij2k(v1[h1], v1[h2], m);
				if (pairs[p]) {
					cout << "regular_ls_generator::lifting_prepare_"
							"function_new pairs[p]" << endl;
					exit(1);
					}
				idx = open_pair_idx[p];
				Dio->Aij(nb_open_rows + idx, i) = 1;
				}
			}

		}


	
	if (f_v) {
		cout << "regular_ls_generator::lifting_prepare_function_new "
				"done" << endl;
		}
}





// #############################################################################
// global functions:
// #############################################################################



void print_set(INT len, INT *S, void *data)
{
	regular_ls_generator *Gen = (regular_ls_generator *) data;
	
	//print_vector(ost, S, len);
	Gen->print(S, len);
}

void rls_generator_early_test_function(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level)
{
	regular_ls_generator *Gen = (regular_ls_generator *) data;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "rls_generator_early_test_function for set ";
		print_set(cout, len, S);
		cout << endl;
		}
	Gen->early_test_func(S, len, 
		candidates, nb_candidates, 
		good_candidates, nb_good_candidates, 
		verbose_level - 2);
	if (f_v) {
		cout << "rls_generator_early_test_function done" << endl;
		}
}

INT check_function_incremental_callback(INT len, INT *S, void *data, INT verbose_level)
{
	regular_ls_generator *Gen = (regular_ls_generator *) data;
	INT f_OK;
	
	f_OK = Gen->check_function_incremental(len, S, verbose_level);
	return f_OK; 
}


void rls_generator_lifting_prepare_function_new(exact_cover *EC, INT starter_case, 
	INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, INT *&col_labels, 
	INT &f_ruled_out, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	regular_ls_generator *Gen = (regular_ls_generator *) EC->user_data;

	if (f_v) {
		cout << "rls_generator_lifting_prepare_function_new nb_candidates=" << nb_candidates << endl;
		}

	Gen->lifting_prepare_function_new(EC, starter_case, 
		candidates, nb_candidates, Strong_gens, 
		Dio, col_labels, f_ruled_out, 
		verbose_level - 1);


	if (f_v) {
		cout << "rls_generator_lifting_prepare_function_new nb_rows=" << Dio->m << " nb_cols=" << Dio->n << endl;
		}

	if (f_v) {
		cout << "rls_generator_lifting_prepare_function_new done" << endl;
		}
}



