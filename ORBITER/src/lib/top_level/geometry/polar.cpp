// polar.C
// 
// Anton Betten
// started: Feb 8, 2010
// moved to DISCRETA: June 1, 2010
//
// 
//
//

#include "orbiter.h"

polar::polar()
{
	A = NULL;
	//AG = NULL;
	Mtx = NULL;
	tmp_M = NULL;
	base_cols = NULL;
	Gen = NULL;
	f_has_strong_generators = FALSE;
	f_has_strong_generators_allocated = FALSE;
	Strong_gens = NULL;
	//SG = NULL;
	//tl = NULL;
	f_print_generators = FALSE;
	first_node = 0;
	nb_orbits = 0;
	nb_elements = 0;
}

polar::~polar()
{
	if (tmp_M) {
		FREE_INT(tmp_M);
		}
	if (base_cols) {
		FREE_INT(base_cols);
		}
	if (Gen) {
		FREE_OBJECT(Gen);
		}
	if (f_has_strong_generators
		&& f_has_strong_generators_allocated) {
		if (Strong_gens) {
			FREE_OBJECT(Strong_gens);
			}
		}
}

void polar::init_group_by_base_images(
	INT *group_generator_data, INT group_generator_size, 
	INT f_group_order_target, const char *group_order_target, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	vector_ge gens;

	if (f_v) {
		cout << "polar::init_group, calling "
				"A->init_group_from_generators_by_base_images" << endl;
		}
	A->init_group_from_generators_by_base_images(
		group_generator_data, group_generator_size, 
		f_group_order_target, group_order_target, 
		&gens, Strong_gens, 
		verbose_level);
	f_has_strong_generators = TRUE;
	f_has_strong_generators_allocated = TRUE;
}

void polar::init_group(
	INT *group_generator_data, INT group_generator_size, 
	INT f_group_order_target, const char *group_order_target, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	vector_ge gens;

	if (f_v) {
		cout << "polar::init_group, calling "
				"A->init_group_from_generators" << endl;
		}
	A->init_group_from_generators(
		group_generator_data, group_generator_size, 
		f_group_order_target, group_order_target, 
		&gens, Strong_gens, 
		verbose_level);
	f_has_strong_generators = TRUE;
	f_has_strong_generators_allocated = TRUE;
}

void polar::init(int argc, const char **argv,
	action *A, orthogonal *O,
	INT epsilon, INT n, INT k, finite_field *F, 
	INT depth, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	polar::epsilon = epsilon;
	polar::n = n;
	polar::k = k;
	polar::F = F;
	polar::q = F->q;
	polar::depth = depth;
	polar::A = A;
	polar::O = O;
	
	if (f_v) {
		cout << "polar::init n=" << n << " k=" << k << " q=" << q << endl;
		}
	
	//matrix_group *M;
	//M = A->subaction->G.matrix_grp;
	//O = M->O;

	
	
	
	tmp_M = NEW_INT(n * n);
	base_cols = NEW_INT(n);
	Gen = NEW_OBJECT(poset_classification);

	Gen->read_arguments(argc, argv, 0);

	//Gen->prefix[0] = 0;
	sprintf(Gen->fname_base, "polar_%ld_%ld_%ld_%ld", epsilon, n, k, q);
	
	
	Gen->depth = depth;
	Gen->verbose_level = verbose_level - 2;
	
}

void polar::init2(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//vector_ge *gens;
	//INT *transversal_lengths;
	strong_generators *gens;
	
	if (f_v) {
		cout << "polar::init2" << endl;
		}
	if (f_has_strong_generators) {
		if (f_v) {
			cout << "initializing with strong generators" << endl;
			}
		gens = Strong_gens;
		//gens = SG;
		//transversal_lengths = tl;
		}
	else {
		if (f_v) {
			cout << "initializing full group" << endl;
			}
		gens = A->Strong_gens;
		//gens = A->strong_generators;
		//transversal_lengths = A->tl;
		}

	if (f_print_generators) {
		INT f_print_as_permutation = TRUE;
		INT f_offset = TRUE;
		INT offset = 1;
		INT f_do_it_anyway_even_for_big_degree = TRUE;
		INT f_print_cycles_of_length_one = FALSE;
		
		cout << "printing generators for the group:" << endl;
		gens->gens->print(cout, f_print_as_permutation, 
			f_offset, offset, 
			f_do_it_anyway_even_for_big_degree, 
			f_print_cycles_of_length_one);
		}
	Gen->init(A, A, gens,
			Gen->depth /* sz */, verbose_level);

	Gen->init_check_func(
		polar_callback_test_func, 
		this /* candidate_check_data */);

	//Gen->init_incremental_check_func(
		//check_mindist_incremental, 
		//this /* candidate_check_data */);

	Gen->init_vector_space_action(n /* vector_space_dimension */, 
		F, 
		polar_callback_rank_point_func, 
		polar_callback_unrank_point_func, 
		this, 
		verbose_level);

	Gen->init_early_test_func(
		polar_callback_early_test_func, 
		this,  
		verbose_level);
#if 0
	Gen->f_print_function = TRUE;
	Gen->print_function = print_set;
	Gen->print_function_data = this;
#endif	

	INT nb_poset_orbit_nodes = 1000;
	
	if (f_v) {
		cout << "Gen->init_poset_orbit_node" << endl;
		}
	Gen->init_poset_orbit_node(nb_poset_orbit_nodes, verbose_level - 1);
	if (f_v) {
		cout << "calling Gen->init_root_node" << endl;
		}
	Gen->root[0].init_root_node(Gen, verbose_level);
	
	schreier_depth = Gen->depth;
	f_use_invariant_subset_if_available = FALSE;
	//f_implicit_fusion = FALSE;
	f_debug = FALSE;
}

void polar::compute_orbits(INT t0, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "polar::compute_orbits calling generator_main" << endl;
		cout << "A=";
		Gen->A->print_info();
		cout << "A2=";
		Gen->A2->print_info();
		}
	Gen->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		//f_implicit_fusion, 
		f_debug, 
		verbose_level - 1);
		
	if (f_v) {
		cout << "done with generator_main" << endl;
		}
	first_node = Gen->first_poset_orbit_node_at_level[depth];
	nb_orbits = Gen->first_poset_orbit_node_at_level[depth + 1] - first_node;

	INT i;
	nb_elements = 0;
	for (i = 0; i < nb_orbits; i++) {
		nb_elements += Gen->orbit_length_as_INT(i, depth);
		}
	if (f_v) {
		cout << "we found " << nb_orbits << " orbits containing "
				<< nb_elements << " elements at depth " << depth << endl;
		}
}

void polar::compute_cosets(INT depth, INT orbit_idx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT i, c, cc, node2, index_INT;
	INT *the_set1;
	INT *the_set2;
	INT *M1;
	INT *M2;
	INT *Elt1, *Elt2;
	longinteger_domain D;
	longinteger_object go1, go2, index, rem, Rank;
	poset_orbit_node *O2;

	if (f_v) {
		cout << "polar::compute_cosets" << endl;
		}
	Elt1 = NEW_INT(Gen->A->elt_size_in_INT);
	Elt2 = NEW_INT(Gen->A->elt_size_in_INT);
	the_set1 = NEW_INT(depth);
	the_set2 = NEW_INT(depth);
	M1 = NEW_INT(k * n);
	M2 = NEW_INT(k * n);
	
	node2 = Gen->first_poset_orbit_node_at_level[depth] + orbit_idx;
	O2 = &Gen->root[node2];

	Gen->stabilizer_order(0, go1);
	Gen->stabilizer_order(node2, go2);
	D.integral_division(go1, go2, index, rem, 0);

	index_INT = index.as_INT();
	if (f_v) {
		cout << "polar::compute_cosets index=" << index_INT << endl;
		}

	O2->store_set_to(Gen, depth - 1, the_set1);
	
	if (f_v) {
		cout << "the set representing orbit " << orbit_idx 
			<< " at level " << depth << " is ";
		INT_vec_print(cout, the_set1, depth);
		cout << endl;
		}
	for (i = 0; i < k; i++) {
		unrank_point(M1 + i * n, the_set1[i]);
		//polar_callback_unrank_point_func(M1 + i * n, the_set1[i], this);
		}
	if (f_vv) {
		cout << "corresponding to the subspace with basis:" << endl;
		print_integer_matrix_width(cout, M1, k, n, n, F->log10_of_q);
		}

	grassmann Grass;
	
	Grass.init(n, k, F, 0 /*verbose_level*/);
	for (i = 0; i < k * n; i++) {
		Grass.M[i] = M1[i];
		}
	Grass.rank_longinteger(Rank, 0/*verbose_level - 3*/);
	cout << "Rank=" << Rank << endl;
	

	for (c = 0; c < index_INT; c++) {

		//if (!(c == 2 || c == 4)) {continue;}
		if (f_v) {
			cout << "Coset " << c << endl;
			}
		Gen->coset_unrank(depth, orbit_idx, c, Elt1, 0/*verbose_level*/);

		if (f_vvv) {
			cout << "Left coset " << c << " is represented by" << endl;
			Gen->A->element_print_quick(Elt1, cout);
			cout << endl;
			}

		Gen->A->element_invert(Elt1, Elt2, 0);


		if (f_vvv) {
			cout << "Right coset " << c << " is represented by" << endl;
			Gen->A->element_print_quick(Elt2, cout);
			cout << endl;
			}

		for (i = 0; i < k; i++) {
			A->element_image_of_low_level(M1 + i * n, M2 + i * n,
					Elt2, 0/* verbose_level*/);
			}
		if (f_vv) {
			cout << "basis of subspace that is the image "
					"under this element:" << endl;
			print_integer_matrix_width(cout, M2, k, n, n, F->log10_of_q);
			}
		for (i = 0; i < k * n; i++) {
			Grass.M[i] = M2[i];
			}
		Grass.rank_longinteger(Rank, 0/*verbose_level - 3*/);
		if (f_vv) {
			cout << "Coset " << c << " Rank=" << Rank << endl;
			}
		
		cc = Gen->coset_rank(depth, orbit_idx, Elt1, 0/*verbose_level*/);
		if (cc != c) {
			cout << "error in polar::compute_cosets" << endl;
			cout << "cc != c" << endl;
			cout << "c=" << c << endl;
			cout << "cc=" << cc << endl;
			//cc = Gen->coset_rank(depth, orbit_idx, Elt1, verbose_level);
			exit(1);
			}
		}
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(the_set1);
	FREE_INT(the_set2);
	FREE_INT(M1);
	FREE_INT(M2);
}

void polar::dual_polar_graph(INT depth, INT orbit_idx, 
	longinteger_object *&Rank_table, INT &nb_maximals,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, c, node2, index_INT;
	INT *the_set1;
	INT *the_set2;
	INT *M1;
	INT *M2;
	INT *Elt1, *Elt2;
	longinteger_domain D;
	longinteger_object go1, go2, index, rem, Rank;
	poset_orbit_node *O2;
	INT *Adj;
	INT **M;
	INT witt;

	if (f_v) {
		cout << "polar::dual_polar_graph" << endl;
		}
	Elt1 = NEW_INT(Gen->A->elt_size_in_INT);
	Elt2 = NEW_INT(Gen->A->elt_size_in_INT);
	the_set1 = NEW_INT(depth);
	the_set2 = NEW_INT(depth);
	M1 = NEW_INT(k * n);
	M2 = NEW_INT(k * n);

	witt = Witt_index(epsilon, n - 1);
		
	node2 = Gen->first_poset_orbit_node_at_level[depth] + orbit_idx;
	O2 = &Gen->root[node2];

	Gen->stabilizer_order(0, go1);
	Gen->stabilizer_order(node2, go2);
	D.integral_division(go1, go2, index, rem, 0);

	index_INT = index.as_INT();
	if (f_v) {
		cout << "polar::dual_polar_graph index=" << index_INT << endl;
		cout << "polar::dual_polar_graph witt=" << witt << endl;
		}

	nb_maximals = index_INT;
	Rank_table = NEW_OBJECTS(longinteger_object, index_INT);
	Adj = NEW_INT(index_INT * index_INT);
	M = NEW_PINT(index_INT);

	for (i = 0; i < index_INT; i++) {
		M[i] = NEW_INT(k * n);
		}
	for (i = 0; i < index_INT * index_INT; i++) {
		Adj[i] = 0;
		}
	
	O2->store_set_to(Gen, depth - 1, the_set1);
	
	if (f_v) {
		cout << "the set representing orbit " << orbit_idx 
			<< " at level " << depth << " is ";
		INT_vec_print(cout, the_set1, depth);
		cout << endl;
		}
	for (i = 0; i < k; i++) {
		unrank_point(M1 + i * n, the_set1[i]);
		//polar_callback_unrank_point_func(M1 + i * n, the_set1[i], this);
		}

	if (f_v) {
		cout << "corresponding to the subspace with basis:" << endl;
		print_integer_matrix_width(cout, M1, k, n, n, F->log10_of_q);
		}

	grassmann Grass;
	
	Grass.init(n, k, F, verbose_level - 2);
	for (i = 0; i < k * n; i++) {
		Grass.M[i] = M1[i];
		}
	Grass.rank_longinteger(Rank, 0/*verbose_level - 3*/);
	cout << "Rank=" << Rank << endl;
	

	for (c = 0; c < index_INT; c++) {

		//if (!(c == 2 || c == 4)) {continue;}

		if (FALSE) {
			cout << "Coset " << c << endl;
			}
		Gen->coset_unrank(depth, orbit_idx, c, Elt1, 0/*verbose_level*/);

		if (FALSE) {
			cout << "Left coset " << c << " is represented by" << endl;
			Gen->A->element_print_quick(Elt1, cout);
			cout << endl;
			}

		Gen->A->element_invert(Elt1, Elt2, 0);


		if (FALSE) {
			cout << "Right coset " << c
					<< " is represented by" << endl;
			Gen->A->element_print_quick(Elt2, cout);
			cout << endl;
			}

		for (i = 0; i < k; i++) {
			A->element_image_of_low_level(M1 + i * n,
					M2 + i * n, Elt2, 0/* verbose_level*/);
			}

		F->Gauss_easy(M2, k, n);
		
		if (f_vv) {
			cout << "subspace " << c << ":" << endl;
			print_integer_matrix_width(cout, M2, k, n, n, F->log10_of_q);
			}

		

		
		for (i = 0; i < k * n; i++) {
			Grass.M[i] = M2[i];
			}
		Grass.rank_longinteger(Rank, 0/*verbose_level - 3*/);
		if (f_vv) {
			cout << "Coset " << c << " Rank=" << Rank << endl;
			}
		
		Rank.assign_to(Rank_table[c]);
		
		for (i = 0; i < k * n; i++) {
			M[c][i] = M2[i];
			}
		
		}
	
	INT c1, c2, rk, nb_e, e;
	INT *MM;
	INT *Inc;
	
	MM = NEW_INT(2 * k * n);
	
	nb_e = 0;
	for (c1 = 0; c1 < index_INT; c1++) {
		for (c2 = 0; c2 < index_INT; c2++) {
			for (i = 0; i < k * n; i++) {
				MM[i] = M[c1][i];
				}
			for (i = 0; i < k * n; i++) {
				MM[k * n + i] = M[c2][i];
				}
			rk = F->rank_of_rectangular_matrix(MM,
					2 * k, n, 0 /* verbose_level*/);
			//rk1 = rk - k;
			//Adj[c1 * index_INT + c2] = rk1;
			if (rk == k + 1) {
				Adj[c1 * index_INT + c2] = 1;
				}
			}
		}

	if (f_vv) {
		cout << "adjacency matrix:" << endl;
		print_integer_matrix_width(cout, Adj,
				index_INT, index_INT, index_INT, 1);
		}

	if (f_vv) {
		cout << "neighborhood lists:" << endl;
		for (c1 = 0; c1 < index_INT; c1++) {
			cout << "N(" << c1 << ")={";
			for (c2 = 0; c2 < index_INT; c2++) {
				if (Adj[c1 * index_INT + c2]) {
					cout << c2 << " ";
					}
				}
			cout << "}" << endl;
			}
		}

	nb_e = 0;
	for (c1 = 0; c1 < index_INT; c1++) {
		for (c2 = c1 + 1; c2 < index_INT; c2++) {
			if (Adj[c1 * index_INT + c2]) {
				nb_e++;
				}
			}
		}
	if (f_vv) {
		cout << "with " << nb_e << " edges" << endl;
		}


	Inc = NEW_INT(index_INT * nb_e);
	for (i = 0; i < index_INT * nb_e; i++) {
		Inc[i] = 0;
		}
	
	e = 0;
	for (c1 = 0; c1 < index_INT; c1++) {
		for (c2 = c1 + 1; c2 < index_INT; c2++) {
			if (Adj[c1 * index_INT + c2]) {
				Inc[c1 * nb_e + e] = 1;
				Inc[c2 * nb_e + e] = 1;
				e++;
				}
			}
		}
	if (f_vv) {
		cout << "Incidence matrix:" << index_INT
				<< " x " << nb_e << endl;
		print_integer_matrix_width(cout, Inc,
				index_INT, nb_e, nb_e, 1);
		}

	{
	char fname[1000];

	sprintf(fname, "dual_polar_graph_O_%ld_%ld_%ld.inc", epsilon, n, q);
	{
	ofstream f(fname);
	f << index_INT << " " << nb_e << " " << 2 * nb_e << endl;
	for (i = 0; i < index_INT * nb_e; i++) {
		if (Inc[i]) {
			f << i << " ";
			}
		}
	f << endl;
	f << -1 << endl;
	}
	if (f_v) {
		cout << "written file " << fname << " of size "
				<< file_size(fname) << endl;
		}
	}

	FREE_INT(Inc);
	FREE_INT(MM);

	
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(the_set1);
	FREE_INT(the_set2);
	FREE_INT(M1);
	FREE_INT(M2);
	FREE_INT(Adj);
	for (i = 0; i < index_INT; i++) {
		FREE_INT(M[i]);
		}
	FREE_PINT(M);
}

void polar::show_stabilizer(INT depth, INT orbit_idx, INT verbose_level)
{
	//INT node2;
	//oracle *O2;
	//vector_ge *gens;
	//INT *tl;
	INT *Elt;
	INT goi, i, order;
	strong_generators *Strong_gens;

	Elt = NEW_INT(A->elt_size_in_INT);	
	//node2 = Gen->first_oracle_node_at_level[depth] + orbit_idx;
	//O2 = &Gen->root[node2];

	Gen->get_stabilizer_generators(Strong_gens,  
		depth, orbit_idx, 0 /* verbose_level*/);
	//Gen->get_stabilizer(gens, tl, depth, orbit_idx, verbose_level);

	sims *S;
	S = create_sims_from_generators_with_target_group_order_factorized(
		A, Strong_gens->gens, Strong_gens->tl, A->base_len,
		verbose_level);
	longinteger_object go;

	S->group_order(go);	
	cout << "polar::show_stabilizer created group of order " << go << endl;
	goi = go.as_INT();
	for (i = 0; i < goi; i++) {
		S->element_unrank_INT(i, Elt);
		order = A->element_order(Elt);
		cout << "element " << i << " of order " << order << ":" << endl;
		A->element_print_quick(Elt, cout);
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
		}
	
	FREE_OBJECT(Strong_gens);
	FREE_OBJECT(S);
	FREE_INT(Elt);
}

#if 0
void polar::get_maximals(INT depth, INT orbit_idx, INT verbose_level)
{
	INT node2;
	oracle *O2;
	vector_ge *gens;
	INT *tl;
	INT *Elt;
	INT goi, i, order;

	gens = NEW_OBJECT(vector_ge);
	tl = NEW_INT(A->base_len);
	Elt = NEW_INT(A->elt_size_in_INT);	
	node2 = Gen->first_oracle_node_at_level[depth] + orbit_idx;
	O2 = &Gen->root[node2];
	Gen->get_stabilizer(gens, tl, depth, orbit_idx, verbose_level);

	sims *S;
	S = create_sims_from_generators_with_target_group_order_factorized(
		A, gens, tl, A->base_len, verbose_level);
	longinteger_object go;

	S->group_order(go);	
	cout << "polar::show_stabilizer created group of order " << go << endl;
	goi = go.as_INT();
	for (i = 0; i < goi; i++) {
		S->element_unrank_INT(i, Elt);
		order = A->element_order(Elt);
		cout << "element " << i << " of order " << order << ":" << endl;
		A->element_print_quick(Elt, cout);
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
		}
	
	FREE_OBJECT(S);
	FREE_OBJECT(gens);
	FREE_INT(tl);
	FREE_INT(Elt);
}
#endif

void polar::compute_Kramer_Mesner_matrix(INT t, INT k, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "compute_Kramer_Mesner_matrix t=" << t
				<< " k=" << k << ":" << endl;
		}

	// compute Kramer Mesner matrices
	Vector V;
	INT i;
	
	V.m_l(k);
	for (i = 0; i < k; i++) {
		V[i].change_to_matrix();
		calc_Kramer_Mesner_matrix_neighboring(Gen, i,
				V[i].as_matrix(), verbose_level - 2);
		if (f_v) {
			cout << "matrix level " << i << ":" << endl;
			V[i].as_matrix().print(cout);
			}
		}
	
	matrix Mtk, Mtk_inf;
	
	Mtk_from_MM(V, Mtk, t, k, TRUE, q, verbose_level - 2);
	cout << "M_{" << t << "," << k << "} sup:" << endl;
	Mtk.print(cout);
	
	
	Mtk_sup_to_inf(Gen, t, k, Mtk, Mtk_inf, verbose_level - 2);	
	cout << "M_{" << t << "," << k << "} inf:" << endl;
	Mtk_inf.print(cout);
	
}

INT polar::test(INT *S, INT len, INT verbose_level)
// test if totally isotropic, i.e. contained in its own perp
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, rk;
	INT f_OK = TRUE;

	if (f_v) {
		cout << "polar::test" << endl;
		}
	for (i = 0; i < len; i++) {
		O->unrank_point(tmp_M + i * n, 1, S[i], 0);
		//PG_element_unrank_modified(*P->F, tmp_M, 1, n, S[i]);
		}
	if (f_v) {
		cout << "coordinate matrix:" << endl;
		print_integer_matrix_width(cout, tmp_M, len, n, n, F->log10_of_q);
		}
	F->perp(n, k, tmp_M, O->Gram_matrix);
	if (f_vv) {
		cout << "after perp:" << endl;
		print_integer_matrix_width(cout, tmp_M, n, n, n, 
			F->log10_of_q + 1);
		}
	rk = F->Gauss_simple(tmp_M, 
		len, n, base_cols, verbose_level - 2);
	if (f_v) {
		cout << "the matrix has rank " << rk << endl;
		}
	if (rk > n - len) {
		f_OK = FALSE;
		}
	if (rk < n - len) {
		cout << "polar::test rk < n - len, fatal. "
				"This should not happen" << endl;
		cout << "rk=" << rk << endl;
		cout << "n=" << n << endl;
		cout << "len=" << len << endl;
		exit(1);
		}
	if (f_v) {
		cout << "polar::test done, f_OK=" << f_OK << endl;
		}
	return f_OK;
}

void polar::test_if_in_perp(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, f, c;
	

	if (f_v) {
		cout << "polar::test_if_in_perp done for ";
		INT_set_print(cout, S, len);
		cout << endl;
		}
	if (len == 0) {
		for (i = 0; i < nb_candidates; i++) {
			good_candidates[i] = candidates[i];
			}
		nb_good_candidates = nb_candidates;
		return;
		}

	nb_good_candidates = 0;

	O->unrank_point(tmp_M + 0 * n, 1, S[len - 1], 0);
	for (i = 0; i < nb_candidates; i++) {
		c = candidates[i];
		O->unrank_point(tmp_M + 1 * n, 1, c, 0);
		if (f_vv) {
			cout << "candidate " << i << " = " << c << ":" << endl;
			print_integer_matrix_width(cout,
					tmp_M, 2, n, n, F->log10_of_q);
			}
		f = O->evaluate_bilinear_form(tmp_M + 0 * n, tmp_M + 1 * n, 1);
		if (f_vv) {
			cout << "form value " << f << endl;
			}
		if (f == 0) {
			good_candidates[nb_good_candidates++] = c;
			}
		}

	
	if (f_v) {
		cout << "polar::test_if_in_perp done for ";
		INT_set_print(cout, S, len);
		cout << "; # of candidates reduced from " << nb_candidates
				<< " to " << nb_good_candidates << endl;
		}
	if (f_vv) {
		cout << "good candidates: ";
		INT_vec_print(cout, good_candidates, nb_good_candidates);
		cout << endl;
		}
}

void polar::test_if_closed_under_cosets(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, h, c, d, y, f_OK, idx, nb, nb0;
	INT *M;
	INT *N0;
	INT *N;
	INT *v;
	INT *w;
	INT *candidates_expanded;
	INT nb_candidates_expanded;
	INT *tmp_candidates;
	INT nb_tmp_candidates;

	if (f_v) {
		cout << "polar::test_if_closed_under_cosets for ";
		INT_set_print(cout, S, len);
		cout << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "candidates: ";
		INT_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		}
	if (len == 0) {
		for (i = 0; i < nb_candidates; i++) {
			good_candidates[i] = candidates[i];
			}
		nb_good_candidates = nb_candidates;
		return;
		}

	nb = nb_PG_elements(len - 1, F->q);
	if (len >= 2) {
		nb0 = nb_PG_elements(len - 2, F->q);
		}
	else {
		nb0 = 1;
		}
	M = NEW_INT(len * n);
	N0 = NEW_INT(nb0 * n);
	N = NEW_INT(nb * n);
	v = NEW_INT(n);
	w = NEW_INT(n);
	candidates_expanded = NEW_INT(2 * nb_candidates * (1 + nb0));
	tmp_candidates = NEW_INT(2 * nb_candidates * (1 + nb0));
	for (i = 0; i < len; i++) {
		O->unrank_point(M + i * n, 1, S[i], 0);
		}
	if (f_v) {
		cout << "the basis is ";
		INT_vec_print(cout, S, len);
		cout << endl;
		cout << "corresponding to the vectors:" << endl;
		print_integer_matrix_width(cout, M, len, n, n, F->log10_of_q);
		cout << "nb=" << nb << endl;
		}
	if (len >= 2) {
		for (i = 0; i < nb0; i++) {
			PG_element_unrank_modified(*F, v, 1, len - 1, i);
			F->mult_vector_from_the_left(v, M, N0 + i * n, len - 1, n);
			}
		if (f_v) {
			cout << "the list of points N0:" << endl;
			print_integer_matrix_width(cout, N0, nb0, n, n, F->log10_of_q);
			}
		}
	for (i = 0; i < nb; i++) {
		PG_element_unrank_modified(*F, v, 1, len, i);
		F->mult_vector_from_the_left(v, M, N + i * n, len, n);
		}
	if (f_v) {
		cout << "the list of points N:" << endl;
		print_integer_matrix_width(cout, N, nb, n, n, F->log10_of_q);
		}
	if (len >= 2) {
		// the expand step:
		if (f_v) {
			cout << "expand:" << endl;
			}
		nb_candidates_expanded = 0;
		for (i = 0; i < nb_candidates; i++) {
			c = candidates[i];
			candidates_expanded[nb_candidates_expanded++] = c;
			if (INT_vec_search(S, len, c, idx)) {
				continue;
				}
			O->unrank_point(v, 1, c, 0);
			if (f_v) {
				cout << "i=" << i;
				INT_vec_print(cout, v, n);
				cout << endl;
				}
			for (j = 0; j < nb0; j++) {
				for (y = 1; y < F->q; y++) {
					for (h = 0; h < n; h++) {
						w[h] = F->add(v[h], F->mult(y, N0[j * n + h]));
						}
					if (f_v) {
						cout << "j=" << j << " y=" << y << " : w=";
						INT_vec_print(cout, w, n);
						cout << endl;
						}
					d = O->rank_point(w, 1, 0);
					if (f_v) {
						cout << "d=" << d << endl;
						}
					candidates_expanded[nb_candidates_expanded++] = d;
					} // next y
				} // next j
			} // next i
		if (f_v) {
			cout << "expanded candidate set:" << endl;
			INT_vec_print(cout,
					candidates_expanded, nb_candidates_expanded);
			cout << endl;
			}
		INT_vec_heapsort(candidates_expanded, nb_candidates_expanded);
		if (f_v) {
			cout << "expanded candidate set after sort:" << endl;
			INT_vec_print(cout, candidates_expanded, nb_candidates_expanded);
			cout << endl;
			}
		}
	else {
		nb_candidates_expanded = 0;
		for (i = 0; i < nb_candidates; i++) {
			c = candidates[i];
			candidates_expanded[nb_candidates_expanded++] = c;
			}
		}

	// now we are doing the test if the full coset is present:
	nb_tmp_candidates = 0;
	for (i = 0; i < nb_candidates_expanded; i++) {
		c = candidates_expanded[i];
		if (INT_vec_search(S, len, c, idx)) {
			tmp_candidates[nb_tmp_candidates++] = c;
			continue;
			}
		O->unrank_point(v, 1, c, 0);
		if (f_v) {
			cout << "i=" << i;
			INT_vec_print(cout, v, n);
			cout << endl;
			}
		f_OK = TRUE;
		for (j = 0; j < nb; j++) {
			for (y = 1; y < F->q; y++) {
				for (h = 0; h < n; h++) {
					w[h] = F->add(v[h], F->mult(y, N[j * n + h]));
					}
				if (f_v) {
					cout << "j=" << j << " y=" << y << " : w=";
					INT_vec_print(cout, w, n);
					cout << endl;
					}
				d = O->rank_point(w, 1, 0);
				if (!INT_vec_search(candidates_expanded,
						nb_candidates_expanded, d, idx)) {
					if (f_vv) {
						cout << "polar::test_if_closed_under_cosets "
								"point " << c << " is ruled out, "
								"coset point " << d << " is not found "
								"j=" << j << " y=" << y << endl;
						}
					f_OK = FALSE;
					break;
					}
				}
			if (!f_OK) {
				break;
				}
			}
		if (f_OK) {
			tmp_candidates[nb_tmp_candidates++] = c;
			}
		}
	if (f_v) {
		cout << "tmp_candidates:" << endl;
		INT_vec_print(cout, tmp_candidates, nb_tmp_candidates);
		cout << endl;
		}

	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		c = candidates[i];
		if (INT_vec_search(tmp_candidates, nb_tmp_candidates, c, idx)) {
			good_candidates[nb_good_candidates++] = c;
			continue;
			}
		}
	
	if (f_v) {
		cout << "polar::test_if_closed_under_cosets for ";
		INT_set_print(cout, S, len);
		cout << "; # of candidates reduced from "
			<< nb_candidates << " to " << nb_good_candidates << endl;
		}
	if (f_vv) {
		cout << "good candidates: ";
		INT_vec_print(cout, good_candidates, nb_good_candidates);
		cout << endl;
		}
	FREE_INT(M);
	FREE_INT(N0);
	FREE_INT(N);
	FREE_INT(v);
	FREE_INT(w);
	FREE_INT(candidates_expanded);
	FREE_INT(tmp_candidates);
}


void polar::get_stabilizer(INT orbit_idx, group &G,
		longinteger_object &go_G)
{
	Gen->root[first_node + orbit_idx].get_stabilizer(Gen,
			G, go_G, 0 /*verbose_level - 2*/);
}

void polar::get_orbit_length(INT orbit_idx,
		longinteger_object &length)
{
	Gen->orbit_length(orbit_idx, depth, length);
}

INT polar::get_orbit_length_as_INT(INT orbit_idx)
{
	return Gen->orbit_length_as_INT(orbit_idx, depth);
}

void polar::orbit_element_unrank(INT orbit_idx,
		INT rank, INT *set, INT verbose_level)
{
	return Gen->orbit_element_unrank(depth,
			orbit_idx, rank, set, verbose_level);
}

void polar::orbit_element_rank(INT &orbit_idx,
		INT &rank, INT *set, INT verbose_level)
{
	return Gen->orbit_element_rank(depth,
			orbit_idx, rank, set, verbose_level);
}

void polar::unrank_point(INT *v, INT rk)
{
	O->unrank_point(v, 1, rk, 0);
}

INT polar::rank_point(INT *v)
{
	return O->rank_point(v, 1, 0);
}

void polar::list_whole_orbit(INT depth,
		INT orbit_idx, INT f_limit, INT limit)
{
	INT *set;
	INT len, j, h, ii, jj;
	group G;
	longinteger_object go_G, Rank;
	INT *M1;
	INT *base_cols;

	set = NEW_INT(depth);
	M1 = NEW_INT(depth * n);
	base_cols = NEW_INT(n);
	get_stabilizer(orbit_idx, G, go_G);
	cout << "the stabilizer of orbit rep " << orbit_idx
			<< " has order " << go_G << endl;

	len = get_orbit_length_as_INT(orbit_idx);
		
	cout << "the orbit length of orbit " << orbit_idx
			<< " is " << len << endl;
	for (j = 0; j < len; j++) {
		//if (j != 2) continue;
		
		if (f_limit && j >= limit) {
			cout << "..." << endl;
			break;
			}
		orbit_element_unrank(orbit_idx, j, set, 0/*verbose_level*/);
		cout << setw(4) << j << " : ";
		INT_vec_print(cout, set, depth);
		cout << endl;
			
		for (h = 0; h < depth; h++) {
			unrank_point(M1 + h * n, set[h]);
			}
		cout << "corresponding to the subspace with basis:" << endl;
		print_integer_matrix_width(cout, M1, k, n, n, F->log10_of_q);
			
		F->Gauss_simple(M1, depth, n, base_cols, 0/* verbose_level*/);

		cout << "basis in echelon form:" << endl;
		print_integer_matrix_width(cout, M1, k, n, n, F->log10_of_q);
		


		grassmann Grass;
	
		Grass.init(n, k, F, 0/*verbose_level*/);
		for (h = 0; h < k * n; h++) {
			Grass.M[h] = M1[h];
			}
		Grass.rank_longinteger(Rank, 0/*verbose_level - 3*/);
		cout << "Rank=" << Rank << endl;
			
		orbit_element_rank(ii, jj, set, 0/*verbose_level*/);
		cout << setw(2) << ii << " : " << setw(4) << jj << endl;
		if (ii != orbit_idx) {
			cout << "polar::list_whole_orbit: "
					"fatal: ii != orbit_idx" << endl;
			exit(1);
			}
		if (jj != j) {
			cout << "polar::list_whole_orbit: "
					"fatal: jj != j" << endl;
			exit(1);
			}
		}
	FREE_INT(set);
	FREE_INT(M1);
	FREE_INT(base_cols);
}


// #############################################################################
// global functions:
// #############################################################################

INT polar_callback_rank_point_func(INT *v, void *data)
{
	polar *P = (polar *) data;
	//generator *gen = P->Gen;
	INT rk;
	
	rk = P->O->rank_point(v, 1, 0);
	//PG_element_rank_modified(*gen->F, v, 1,
	// gen->vector_space_dimension, rk);
	return rk;
}

void polar_callback_unrank_point_func(INT *v, INT rk, void *data)
{
	polar *P = (polar *) data;
	//generator *gen = P->Gen;
	
	P->O->unrank_point(v, 1, rk, 0);
	//PG_element_unrank_modified(*gen->F, v, 1,
	// gen->vector_space_dimension, rk);
}

INT polar_callback_test_func(INT len, INT *S,
		void *data, INT verbose_level)
{
	polar *P = (polar *) data;
	INT f_OK = TRUE;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "checking set ";
		print_set(cout, len, S);
		}
	f_OK = P->test(S, len, verbose_level - 2);
	if (f_OK) {
		if (f_v) {
			cout << "OK" << endl;
			}
		return TRUE;
		}
	else {
		return FALSE;
		}
}

void polar_callback_early_test_func(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level)
{
	polar *P = (polar *) data;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "polar_callback_early_test_func for set ";
		print_set(cout, len, S);
		cout << endl;
		}
	P->test_if_in_perp(S, len, 
		candidates, nb_candidates, 
		good_candidates, nb_good_candidates, 
		verbose_level - 2);
	if (f_v) {
		cout << "polar_callback_early_test_func done" << endl;
		}
}



