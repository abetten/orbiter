// translation_plane.C
// 
// Anton Betten
// November 17, 2009
//
// moved to TOP_LEVEL: November 2, 2013
// 
//
//

#include "orbiter.h"


translation_plane::translation_plane()
{
	null();
}

translation_plane::~translation_plane()
{
	freeself();
}

void translation_plane::null()
{
	f_override_schreier_depth = FALSE;
	f_print_generators = FALSE;

	
	A = NULL;
	A2 = NULL;
	AG = NULL;
	Grass = NULL;
	F = NULL;

	f_recoordinatize = FALSE;
	R = NULL;
	Starter = NULL;
	Starter_Strong_gens = NULL;
	tmp_M1 = NULL;
	tmp_M2 = NULL;
	tmp_M3 = NULL;
	tmp_M4 = NULL;
	gen = NULL;
	Sing = NULL;
	O = NULL;
	Klein = NULL;

	Data1 = NULL;
	Data2 = NULL;
	//Data3 = NULL;
}

void translation_plane::freeself()
{
	if (A) {
		delete A;
		}
	if (A2) {
		delete A2;
		}
#if 0
	if (AG) {
		delete AG;
		}
#endif
	if (Grass) {
		delete Grass;
		}

	if (R) {
		delete R;
		}
	if (Starter) {
		FREE_INT(Starter);
		}
	if (Starter_Strong_gens) {
		delete Starter_Strong_gens;
		}
	if (tmp_M1) {
		FREE_INT(tmp_M1);
		}
	if (tmp_M2) {
		FREE_INT(tmp_M2);
		}
	if (tmp_M3) {
		FREE_INT(tmp_M3);
		}
	if (tmp_M4) {
		FREE_INT(tmp_M4);
		}

	
	if (Sing) {
		delete Sing;
		}
	if (O) {
		delete O;
		}
	if (Klein) {
		delete Klein;
		}
	if (Data1) {
		FREE_INT(Data1);
		}
	if (Data2) {
		FREE_INT(Data2);
		}
#if 0
	if (Data3) {
		FREE_INT(Data3);
		}
#endif
	null();
}

void translation_plane::init(INT order, INT n, INT k, INT max_depth, 
	finite_field *F, INT f_recoordinatize, 
	const BYTE *input_prefix, 
	const BYTE *base_fname,
	INT starter_size,  
	int argc, const char **argv, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	longinteger_object go;
	
	
	if (f_v) {
		cout << "translation_plane::init" << endl;
		cout << "n=" << n << endl;
		cout << "k=" << k << endl;
		cout << "q=" << F->q << endl;
		}
	translation_plane::argc = argc;
	translation_plane::argv = argv;
	
	translation_plane::order = order;
	spread_size = order + 1;
	translation_plane::n = n;
	translation_plane::k = k;
	translation_plane::max_depth = max_depth;
	kn = k * n;
	translation_plane::F = F;
	translation_plane::f_recoordinatize = f_recoordinatize;
	q = F->q;
	
	strcpy(starter_directory_name, input_prefix);
	strcpy(prefix, base_fname);
	//sprintf(prefix_with_directory, "%s%s", starter_directory_name, base_fname);
	translation_plane::starter_size = starter_size;


	tmp_M1 = NEW_INT(n * n);
	tmp_M2 = NEW_INT(n * n);
	tmp_M3 = NEW_INT(n * n);
	tmp_M4 = NEW_INT(n * n);
	
	gen = new generator;
	gen->read_arguments(argc, argv, 1);


	f_projective = TRUE;
	f_semilinear = TRUE;
	f_basis = TRUE;
	f_induce_action = FALSE;

	if (is_prime(q)) {
		if (f_v) {
			cout << "translation_plane::init q=" << q << " is a prime, putting f_semilinear = FALSE" << endl;
			}
		f_semilinear = FALSE;
		}
	else {
		if (f_v) {
			cout << "translation_plane::init q=" << q << " is not a prime" << endl;
			}
		}


	A = new action;
	A2 = new action;
	AG = new action_on_grassmannian;


	if (f_v) {
		cout << "translation_plane::init before init_projective_group" << endl;
		}
	A->init_projective_group(n, F, f_semilinear, f_basis, 0 /*verbose_level*/);
	
	if (f_v) {
		cout << "translation_plane::init after init_projective_group, checking group order" << endl;
		}
	A->Sims->group_order(go);
	if (f_v) {
		cout << "translation_plane::init after init_projective_group group of order " << go << " has been created" <<  endl;
		}


	if (f_vv) {
		cout << "action A created: ";
		A->print_info();
		}



	Grass = new grassmann;
	Grass->init(n, k, F, 0 /*MINIMUM(verbose_level - 1, 1)*/);
	
	nCkq = generalized_binomial(n, k, q);
	block_size = r = generalized_binomial(k, 1, q);
	nb_points_total = nb_pts = generalized_binomial(n, 1, q);
	
	if (f_v) {
		cout << "translation_plane::init nCkq = {n \\choose k}_q = " << nCkq << endl;
		cout << "translation_plane::init r = {k \\choose 1}_q = " << r << endl;
		cout << "translation_plane::init nb_pts = {n \\choose 1}_q = " << nb_pts << endl;
		}



	if (f_v) {
		cout << "translation_plane::init before AG->init" <<  endl;
		}
	
	AG->init(*A, Grass, 0 /*verbose_level - 2*/);
	
	if (f_v) {
		cout << "translation_plane::init after AG->init" <<  endl;
		}

	A2->induced_action_on_grassmannian(A, AG, 
		f_induce_action, NULL /*sims *old_G */, 
		MINIMUM(verbose_level - 2, 2));
	
	if (f_v) {
		cout << "translation_plane::init after A2->induced_action_on_grassmannian" <<  endl;
		}

	if (f_vv) {
		cout << "action A2 created: ";
		A2->print_info();
		}

#if 0
	if (!A->f_has_strong_generators) {
		cout << "action does not have strong generators" << endl;
		exit(1);
		}
#endif

	//INT len;
	


	if (f_print_generators) {
		INT f_print_as_permutation = TRUE;
		INT f_offset = FALSE;
		INT offset = 1;
		INT f_do_it_anyway_even_for_big_degree = TRUE;
		INT f_print_cycles_of_length_one = FALSE;
		
		cout << "printing generators for the group:" << endl;
		A->Strong_gens->gens->print(cout, f_print_as_permutation, 
			f_offset, offset, 
			f_do_it_anyway_even_for_big_degree, 
			f_print_cycles_of_length_one);
		}


#if 0
	len = gens->len;
	for (i = 0; i < len; i++) {
		cout << "generator " << i << ":" << endl;
		A->element_print(gens->ith(i), cout);
		cout << endl;
		if (A2->degree < 150) {
			A2->element_print_as_permutation(gens->ith(i), cout);
			cout << endl;
			}
		}
#endif


	if (nb_pts < 50) {
		print_points();
		}



	if (A2->degree < 150) {
		print_elements();
		print_elements_and_points();
		}


	if (TRUE /*f_v*/) {
		longinteger_object go;
		
		A->Strong_gens->group_order(go);
		cout << "translation_plane::init The order of PGGL(n,q) is " << go << endl;
		}

	
	if (f_recoordinatize) {
		if (f_v) {
			cout << "translation_plane::init before recoordinatize::init" << endl;
			}
		R = new recoordinatize;
		R->init(n, k, F, Grass, A, A2, 
			f_projective, f_semilinear, 
			translation_plane_check_function_incremental, (void *) this, 
			verbose_level);

		if (f_v) {
			cout << "translation_plane::init before recoordinatize::compute_starter" << endl;
			}
		R->compute_starter(Starter, Starter_size, 
			Starter_Strong_gens, MINIMUM(verbose_level - 1, 1));

		longinteger_object go;
		Starter_Strong_gens->group_order(go);
		if (TRUE /*f_v*/) {
			cout << "translation_plane::init The stabilizer of the first three components has order " << go << endl;
			}


		Nb = R->nb_live_points;
		}
	else {
		if (f_v) {
			cout << "translation_plane::init we are not using recoordinatization, please use option -recoordinatize" << endl;
			//exit(1);
			}
		Nb = generalized_binomial(n, k, q); //R->nCkq; // this makes no sense
		}

	if (f_v) {
		cout << "translation_plane::init Nb = " << Nb << endl;
		cout << "translation_plane::init kn = " << kn << endl;
		cout << "translation_plane::init n = " << n << endl;
		cout << "translation_plane::init k = " << k << endl;
		cout << "translation_plane::init allocating Data1 and Data2" << endl;
		}
	
	Data1 = NEW_INT(max_depth * kn);
	Data2 = NEW_INT(n * n);
	//Data3 = NEW_INT(n * n);
	

#if 0
	if (k == 2 && is_prime(q)) {
		Sing = new singer_cycle;
		if (f_v) {
			cout << "translation_plane::init before singer_cycle::init" << endl;
			}
		Sing->init(4, F, A, A2, 0 /*verbose_level*/);
		Sing->init_lines(0 /*verbose_level*/);
		}
#endif

	if (k == 2) {
		
		if (f_v) {
			cout << "translation_plane::init k = 2, initializing klein correspondence" << endl;
			}
		Klein = new klein_correspondence;
		O = new orthogonal;
		
		O->init(1 /* epsilon */, 6, F, 0 /* verbose_level*/);
		Klein->init(F, O, 0 /* verbose_level */);
		}
	else {
		if (f_v) {
			cout << "translation_plane::init we are not initializing klein correspondence" << endl;
			}
		O = NULL;
		Klein = NULL;
		}
	
	if (f_v) {
		cout << "translation_plane::init done" << endl;
		}
}

void translation_plane::print_points()
{
	INT *v;
	INT i;

	cout << "translation_plane::print_points" << endl;
	v = NEW_INT(n);
	for (i = 0; i < nb_pts; i++) {
		PG_element_unrank_modified(*F, v, 1, n, i);
		cout << "point " << i << " : ";
		INT_vec_print(cout, v, n);
		cout << endl;
		}
	FREE_INT(v);
}

void translation_plane::print_elements()
{
	INT i, j;
	
	for (i = 0; i < nCkq; i++) {
		if (FALSE) {
			cout << i << ":" << endl;
			}
		Grass->unrank_INT(i, 0);
		if (FALSE) {
			print_integer_matrix_width(cout, Grass->M, k, n, n, F->log10_of_q + 1);
			}
		j = Grass->rank_INT(0);
		if (j != i) {
			cout << "rank yields " << j << " != " << i << endl;
			exit(1);
			}
		}
}

void translation_plane::print_elements_and_points()
{
	INT i, a, b;
	INT *v, *w;
	INT *Line;

	cout << "translation_plane::print_elements_and_points" << endl;
	v = NEW_INT(k);
	w = NEW_INT(n);
	Line = NEW_INT(r);
	for (i = 0; i < nCkq; i++) {
		if (FALSE) {
			cout << i << ":" << endl;
			}
		Grass->unrank_INT(i, 0);
		for (a = 0; a < r; a++) {
			PG_element_unrank_modified(*F, v, 1, k, a);
			F->mult_matrix(v, Grass->M, w, 1, k, n);
			PG_element_rank_modified(*F, w, 1, n, b);
			Line[a] = b;
			}
		cout << "line " << i << ":" << endl;
		print_integer_matrix_width(cout, Grass->M, k, n, n, F->log10_of_q + 1);
		cout << "points on line " << i << " : ";
		INT_vec_print(cout, Line, r);
		cout << endl;
		}
	FREE_INT(v);
	FREE_INT(w);
	FREE_INT(Line);
}

void translation_plane::read_arguments(int argc, const char **argv)
{
	INT i;
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-schreier") == 0) {
			f_override_schreier_depth = TRUE;
			override_schreier_depth = atoi(argv[++i]);
			cout << "-schreier " << override_schreier_depth << endl;
			}
		else if (strcmp(argv[i], "-print_generators") == 0) {
			f_print_generators = TRUE;
			cout << "-print_generators " << endl;
			}
		}
}

void translation_plane::init2(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//INT depth;
	
	if (f_v) {
		cout << "translation_plane::init2" << endl;
		}
	//depth = order + 1;

	
	
	if (f_recoordinatize) {
		if (f_v) {
			cout << "translation_plane::init2 before gen->initialize_with_starter" << endl;
			}
		gen->initialize_with_starter(A, A2, 
			A->Strong_gens, 
			order + 1, 
			starter_directory_name, 
			prefix, 
			Starter_size, 
			Starter, 
			Starter_Strong_gens, 
			R->live_points, 
			R->nb_live_points, 
			this /*starter_canonize_data*/, 
			starter_canonize_callback, 
			verbose_level - 2);
		if (f_v) {
			cout << "translation_plane::init2 after gen->initialize_with_starter" << endl;
			}
		}
	else {
		if (f_v) {
			cout << "translation_plane::init2 before gen->initialize" << endl;
			}
		gen->initialize(A, A2, 
			A->Strong_gens, 
			order + 1, 
			starter_directory_name, prefix, 
			verbose_level - 2);
		if (f_v) {
			cout << "translation_plane::init2 after gen->initialize" << endl;
			}
		}

	gen->f_allowed_to_show_group_elements = TRUE;

#if 0
	// not needed since we have an early_test_func:

	gen->init_check_func(
		check_function_callback, 
		this /* candidate_check_data */);
#endif


	// we have an early test function:

	gen->init_early_test_func(
		translation_plane_early_test_func_callback, 
		this,  
		verbose_level);

	// We also have an incremental check function. 
	// This is only used by the clique finder:
	gen->init_incremental_check_func(
		translation_plane_check_function_incremental_callback, 
		this /* candidate_check_data */);


#if 0
	gen->f_print_function = TRUE;
	gen->print_function = print_set;
	gen->print_function_data = this;
#endif	


	if (f_v) {
		cout << "translation_plane::init2 done" << endl;
		}
}

void translation_plane::compute(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT schreier_depth = gen->depth;
	INT f_use_invariant_subset_if_available = TRUE;
	INT f_debug = FALSE;
	INT t0;


	if (f_v) {
		cout << "translation_plane::compute" << endl;
		}

	
	if (f_override_schreier_depth) {
		schreier_depth = override_schreier_depth;
		}
	if (f_v) {
		cout << "translation_plane::compute calling generator_main" << endl;
		}

	gen->f_max_depth = TRUE;
	gen->max_depth = starter_size;
	
	t0 = os_ticks();
	gen->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		verbose_level - 1);
	
	INT length;
	
	if (f_v) {
		cout << "translation_plane::compute done with generator_main" << endl;
		}
	length = gen->nb_orbits_at_level(gen->max_depth);
	if (f_v) {
		cout << "translation_plane::compute We found " << length << " orbits on " 
			<< gen->max_depth << "-sets of " << k 
			<< "-subspaces in PG(" << n - 1 << "," << q << ")" 
			<< " satisfying the partial spread condition" << endl;
		}



	if (f_v) {
		cout << "translation_plane::compute done" << endl;
		}
}


void translation_plane::early_test_func(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i0, i, j, rk;
	INT *M;
	INT *MM;
	INT *B, *base_cols;
		
	if (f_v) {
		cout << "translation_plane::early_test_func checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size " << nb_candidates << ":" << endl;
		INT_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		if (f_vv) {
			for (i = 0; i < nb_candidates; i++) {
				Grass->unrank_INT(candidates[i], 0/*verbose_level - 4*/);
				cout << "candidate " << i << "=" << candidates[i] << ":" << endl;
				print_integer_matrix_width(cout, Grass->M, k, n, n, F->log10_of_q + 1);
				}
			}
		}

	if (len + 1 > max_depth) {
		cout << "translation_plane::early_test_func len + 1 > max_depth" << endl;
		exit(1);
		}
	M = Data2;
	MM = Data1;
	//M = NEW_INT(n * n);
	//MM = NEW_INT((len + 1) * k * n);
	B = tmp_M3;
	base_cols = tmp_M4;

	for (i = 0; i < len; i++) {
		Grass->unrank_INT_here(MM + i * k * n, S[i], 0/*verbose_level - 4*/);
		}
	if (f_v) {
		for (i = 0; i < len; i++) {
			cout << "p_" << i << "=" << S[i] << ":" << endl;
			print_integer_matrix_width(cout, MM + i * k * n, k, n, n, F->log10_of_q + 1);
			}
		}
	
	nb_good_candidates = 0;
	for (j = 0; j < nb_candidates; j++) {
		Grass->unrank_INT(candidates[j], 0/*verbose_level - 4*/);
		if (len == 0) {
			i0 = 0;
			}
		else {
			i0 = len - 1;
			}
		for (i = i0; i < len; i++) {
			INT_vec_copy(MM + i * k * n, M, k * n);
			INT_vec_copy(Grass->M, M + k * n, k * n);

			if (f_vv) {
				cout << "testing (p_" << i << ",candidates[" << j << "])=(" << S[i] <<  "," << candidates[j] << ")" << endl;
				print_integer_matrix_width(cout, M, n, n, n, F->log10_of_q + 1);
				}
			rk = F->rank_of_matrix_memory_given(M, n, B, base_cols, 0);
			if (rk < n) {
				if (f_vv) {
					cout << "rank is " << rk << " which is bad" << endl;
					}
				break;
				}
			else {
				if (f_vv) {
					cout << "rank is " << rk << " which is OK" << endl;
					}
				}
			} // next i
		if (i == len) {
			good_candidates[nb_good_candidates++] = candidates[j];
			}
		} // next j
	

	//FREE_INT(M);
	//FREE_INT(MM);

}

INT translation_plane::check_function(INT len, INT *S, INT verbose_level)
// checks all {len \choose 2} pairs. This is very inefficient.
{
	INT f_OK = TRUE;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, rk;
	INT *M, *M1;
	INT *B, *base_cols;
		
	if (f_v) {
		cout << "translation_plane::check_function checking set ";
		print_set(cout, len, S);
		cout << endl;
		}
	M1 = tmp_M1;
	M = tmp_M2;
	B = tmp_M3;
	base_cols = tmp_M4;
	//M1 = NEW_INT(k * n);
	//M = NEW_INT(n * n);
	
	if (f_v) {
		for (i = 0; i < len; i++) {
			cout << "p_" << i << "=" << S[i] << ":" << endl;
			Grass->unrank_INT(S[i], 0/*verbose_level - 4*/);
			print_integer_matrix_width(cout, Grass->M, k, n, n, F->log10_of_q + 1);
			}
		}

	for (i = 0; i < len; i++) {
		Grass->unrank_INT_here(M1, S[i], 0/*verbose_level - 4*/);
		for (j = i + 1; j < len; j++) {
			INT_vec_copy(M1, M, k * n);
			Grass->unrank_INT_here(M + k * n, S[j], 0/*verbose_level - 4*/);

			if (f_vv) {
				cout << "testing (p_" << i << ",p_" << j << ")=(" << S[i] << "," << S[j] << ")" << endl;
				print_integer_matrix_width(cout, M, n, n, n, F->log10_of_q + 1);
				}
			rk = F->rank_of_matrix_memory_given(M, n, B, base_cols, 0);
			if (rk < n) {
				if (f_vv) {
					cout << "rank is " << rk << " which is bad" << endl;
					}
				f_OK = FALSE;
				break;
				}
			else {
				if (f_vv) {
					cout << "rank is " << rk << " which is OK" << endl;
					}
				}
			}
		if (f_OK == FALSE)
			break;
		}

	//FREE_INT(M1);
	//FREE_INT(M);

	if (f_OK) {
		if (f_v) {
			cout << "OK" << endl;
			}
		return TRUE;
		}
	else {
		if (f_v) {
			cout << "not OK" << endl;
			}
		return FALSE;
		}

}

INT translation_plane::check_function_incremental(INT len, INT *S, INT verbose_level)
// checks the pairs (0,len-1),(1,len-1),\ldots,(len-2,len-1) 
{
	INT f_OK = TRUE;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, rk;
	INT *M, *M1;
	INT *B, *base_cols;
		
	if (f_v) {
		cout << "translation_plane::check_function_incremental checking set ";
		print_set(cout, len, S);
		cout << endl;
		}
	if (len <= 1) {
		f_OK = TRUE;
		goto finish;
		}
	M1 = tmp_M1;
	M = tmp_M2;
	B = tmp_M3;
	base_cols = tmp_M4;
	//M = NEW_INT(n * n);
	//M1 = NEW_INT(k * n);
	
	if (f_v) {
		for (i = 0; i < len; i++) {
			cout << "p_" << i << "=" << S[i] << ":" << endl;
			Grass->unrank_INT(S[i], 0/*verbose_level - 4*/);
			print_integer_matrix_width(cout, Grass->M, k, n, n, F->log10_of_q + 1);
			}
		}
	
	j = len - 1;
	
	Grass->unrank_INT_here(M1, S[j], 0/*verbose_level - 4*/);
	for (i = 0; i < len - 1; i++) {
		Grass->unrank_INT_here(M, S[i], 0/*verbose_level - 4*/);
		INT_vec_copy(M1, M + k * n, k * n);
		
		if (f_vv) {
			cout << "testing (p_" << i << ",p_" << j << ")=(" << S[i] <<  "," << S[j] << ")" << endl;
			print_integer_matrix_width(cout, M, n, n, n, F->log10_of_q + 1);
			}
		rk = F->rank_of_matrix_memory_given(M, n, B, base_cols, 0);
		if (rk < n) {
			if (f_vv) {
				cout << "rank is " << rk << " which is bad" << endl;
				}
			f_OK = FALSE;
			break;
			}
		else {
			if (f_vv) {
				cout << "rank is " << rk << " which is OK" << endl;
				}
			}
		}

	//FREE_INT(M);
	//FREE_INT(M1);

finish:
	if (f_OK) {
		if (f_v) {
			cout << "OK" << endl;
			}
		return TRUE;
		}
	else {
		if (f_v) {
			cout << "not OK" << endl;
			}
		return FALSE;
		}

}

INT translation_plane::check_function_pair(INT rk1, INT rk2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT rk;
	INT *M;
	INT *B, *base_cols;
		
	if (f_v) {
		cout << "translation_plane::check_function_pair checking (" << rk1 << "," << rk2 << ")" << endl;
		}
	M = tmp_M1;
	//M = NEW_INT(n * n);
	B = tmp_M3;
	base_cols = tmp_M4;
	
	Grass->unrank_INT_here(M, rk1, 0/*verbose_level - 4*/);
	Grass->unrank_INT_here(M + k * n, rk2, 0/*verbose_level - 4*/);

	if (f_vv) {
		cout << "testing (" << rk1 <<  "," << rk2 << ")" << endl;
		print_integer_matrix_width(cout, M, n, n, n, F->log10_of_q + 1);
		}
	rk = F->rank_of_matrix_memory_given(M, n, B, base_cols, 0);

	//FREE_INT(M);

	if (rk < n) {
		if (f_v) {
			cout << "rank is " << rk << " which is bad" << endl;
			}
		return FALSE;
		}
	else {
		if (f_v) {
			cout << "rank is " << rk << " which is OK" << endl;
			}
		return TRUE;
		}
}

void translation_plane::lifting_prepare_function_new(exact_cover *E, INT starter_case, 
	INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, INT *&col_labels, 
	INT &f_ruled_out, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_v3 = (verbose_level >= 3);
	INT nb_free_points, nb_needed;
	
	if (f_v) {
		cout << "translation_plane::lifting_prepare_function_new nb_candidates=" << nb_candidates << endl;
		}



	INT *points_covered_by_starter;
	INT nb_points_covered_by_starter;

	INT *free_point_list; // [nb_free_points]
	INT *point_idx; // [nb_points_total]
		// point_idx[i] = index of a point in free_point_list 
		// or -1 if the point is in points_covered_by_starter


	INT i, j, h, idx, a, b;
	INT *point_list;
	INT nb_points;

	points_covered_by_starter = NEW_INT(E->starter_size * block_size);
	for (i = 0; i < E->starter_size; i++) {
		a = E->starter[i];
		Grass->unrank_INT(a, 0/*verbose_level - 4*/);
		all_PG_elements_in_subspace(F, Grass->M, k, n, point_list, nb_points, 0 /*verbose_level - 2*/);
		if (nb_points != block_size) {
			cout << "translation_plane::lifting_prepare_function_new nb_points != block_size" << endl;
			exit(1);
			}
		for (j = 0; j < block_size; j++) {
			points_covered_by_starter[i * block_size + j] = point_list[j];
			}

		if (f_v3) {
			cout << "starter element " << i << " / " << E->starter_size << " is " << a << ":" << endl;
			INT_matrix_print(Grass->M, k, n);
			//cout << endl;
			cout << "points_covered_by_starter: " << endl;
			INT_vec_print(cout, points_covered_by_starter + i * block_size, block_size);
			cout << endl;
			}

		FREE_INT(point_list);
		}
	nb_points_covered_by_starter = E->starter_size * block_size;
	INT_vec_heapsort(points_covered_by_starter, nb_points_covered_by_starter);
	if (f_vv) {
		cout << "translation_plane::lifting_prepare_function_new covered points computed:" << endl;
		cout << "translation_plane::lifting_prepare_function_new nb_points_covered_by_starter=" << nb_points_covered_by_starter << endl;
		INT_vec_print(cout, points_covered_by_starter, nb_points_covered_by_starter);
		cout << endl;
		}
	
	nb_free_points = nb_points_total - nb_points_covered_by_starter;
	if (f_vv) {
		cout << "translation_plane::lifting_prepare_function_new nb_free_points=" << nb_free_points << endl;
		}
	free_point_list = NEW_INT(nb_free_points);
	point_idx = NEW_INT(nb_points_total);
	j = 0;
	for (i = 0; i < nb_points_total; i++) {
		if (INT_vec_search(points_covered_by_starter, nb_points_covered_by_starter, i, idx)) {
			point_idx[i] = -1;
			}
		else {
			free_point_list[j] = i;
			point_idx[i] = j;
			j++;
			}
		}
	if (j != nb_free_points) {
		cout << "translation_plane::lifting_prepare_function_new j != nb_free_points" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "translation_plane::lifting_prepare_function_new computed free points" << endl;
		}
	nb_needed = spread_size - E->starter_size;
	if (f_vv) {
		cout << "translation_plane::lifting_prepare_function_new nb_needed=" << nb_needed << endl;
		cout << "translation_plane::lifting_prepare_function_new nb_candidates=" << nb_candidates << endl;
		}



#if 0
	nb_live_blocks2 = nb_candidates;
	live_blocks2 = NEW_INT(nb_live_blocks2);
	for (j = 0; j < nb_candidates; j++) {
		a = candidates[j];
		live_blocks2[j] = a;
		}
#endif

	col_labels = NEW_INT(nb_candidates);


	INT_vec_copy(candidates, col_labels, nb_candidates);


	INT nb_rows = nb_free_points;
	INT nb_cols = nb_candidates;


	if (f_vv) {
		cout << "translation_plane::lifting_prepare_function_new candidates: ";
		INT_vec_print(cout, candidates, nb_candidates);
		cout << " (nb_candidates=" << nb_candidates << ")" << endl;
		}




	if (E->f_lex) {
		INT nb_cols_before;

		nb_cols_before = nb_cols;
		E->lexorder_test(col_labels, nb_cols, Strong_gens->gens, 
			verbose_level - 2);
		if (f_v) {
			cout << "translation_plane::lifting_prepare_function_new after lexorder test nb_candidates before: " << nb_cols_before << " reduced to  " << nb_cols << " (deleted " << nb_cols_before - nb_cols << ")" << endl;
			}
		}

	if (f_vv) {
		cout << "translation_plane::lifting_prepare_function_new after lexorder test" << endl;
		cout << "translation_plane::lifting_prepare_function_new nb_cols=" << nb_cols << endl;
		}




	Dio = new diophant;
	Dio->open(nb_rows, nb_cols);
	Dio->sum = nb_needed;

	for (i = 0; i < nb_rows; i++) {
		Dio->type[i] = t_EQ;
		Dio->RHS[i] = 1;
		}

	Dio->fill_coefficient_matrix_with(0);
	if (f_vv) {
		cout << "translation_plane::lifting_prepare_function_new initializing Inc" << endl;
		cout << "translation_plane::lifting_prepare_function_new nb_rows = " << nb_rows << endl;
		cout << "translation_plane::lifting_prepare_function_new nb_cols = " << nb_cols << endl;
		}
	for (j = 0; j < nb_cols; j++) {
		a = col_labels[j];
		Grass->unrank_INT(a, 0/*verbose_level - 4*/);
		if (f_vv) {
			cout << "candidate " << j << " / " << nb_cols << " is " << a << " is " << endl;
			INT_matrix_print(Grass->M, k, n);
			}
		all_PG_elements_in_subspace(F, Grass->M, k, n, point_list, nb_points, 0 /*verbose_level*/);
		if (nb_points != block_size) {
			cout << "translation_plane::lifting_prepare_function_new nb_points != E->block_size" << endl;
			exit(1);
			}
		if (FALSE /*f_vv*/) {
			cout << "List of points: ";
			INT_vec_print(cout, point_list, nb_points);
			cout << endl;
			}

		if (f_v3) {
			cout << "candidate element " << i << " / " << nb_cols << " is " << a << ":" << endl;
			INT_matrix_print(Grass->M, k, n);
			cout << endl; 
			}



		for (h = 0; h < block_size; h++) {
			b = point_list[h];
			i = point_idx[b];
			if (i == -1) {
				cout << "translation_plane::lifting_prepare_function_new candidate block contains point that is already covered" << endl;
				exit(1);
				}
			if (i < 0 || i >= nb_free_points) {
				cout << "translation_plane::lifting_prepare_function_new i < 0 || i >= nb_free_points" << endl;
				exit(1);
				}
			Dio->Aij(i, j) = 1;
			}
		FREE_INT(point_list);
		}

	if (f_vv) {
		cout << "translation_plane::lifting_prepare_function_new coefficient matrix" << endl;
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				cout << Dio->Aij(i, j);
				}
			cout << endl;
			}
		}

	FREE_INT(points_covered_by_starter);
	FREE_INT(free_point_list);
	FREE_INT(point_idx);
	if (f_v) {
		cout << "translation_plane::lifting_prepare_function_new nb_free_points=" << nb_free_points << " nb_candidates=" << nb_candidates << endl;
		}

	if (f_v) {
		cout << "translation_plane::lifting_prepare_function_new done" << endl;
		}
}

void translation_plane::compute_dual_spread(INT *spread, INT *dual_spread, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 5);
	INT *Basis;
	INT i, a, b;

	if (f_v) {
		cout << "translation_plane::compute_dual_spread" << endl;
		}
	Basis = NEW_INT(n * n);
	if (f_v) {
		cout << "translation_plane::compute_dual_spread The spread is : ";
		INT_vec_print(cout, spread, spread_size);
		cout << endl;
		}
	for (i = 0; i < spread_size; i++) {
		a = spread[i];
		Grass->unrank_INT_here(Basis, a, 0/*verbose_level - 4*/);
		if (f_vv) {
			cout << i << "-th Line has rank " << a << " and is generated by" << endl;
			INT_matrix_print(Basis, k, n);
			}
		F->perp_standard(n, k, Basis, 0 /*verbose_level*/);
		if (f_vv) {
			cout << "after perp:" << endl;
			INT_matrix_print(Basis, n, n);
			}
		b = Grass->rank_INT_here(Basis + k * n, 0/*verbose_level - 4*/);
		if (f_vv) {
			cout << i << "-th Line dual has rank " << b << " and is generated by" << endl;
			INT_matrix_print(Basis + k * n, k, n);
			}
		dual_spread[i] = b;
		}
	if (f_v) {
		cout << "translation_plane::compute_dual_spread The dual spread is : ";
		INT_vec_print(cout, dual_spread, spread_size);
		cout << endl;
		}
	
	FREE_INT(Basis);
	if (f_v) {
		cout << "translation_plane::compute_dual_spread done" << endl;
		}
}

void translation_plane::print(INT len, INT *S)
{
	INT i;
	INT f_elements_exponential = FALSE;
	const BYTE *symbol_for_print = "\\alpha";
	
	if (len == 0) {
		return;
		}
	for (i = 0; i < len; i++) {
		cout << "$S_{" << i + 1 << "}$ has rank " << S[i] << " and is generated by\\\\" << endl;
		Grass->unrank_INT(S[i], 0);
		cout << "$$" << endl;
		cout << "\\left[" << endl;
		F->latex_matrix(cout, f_elements_exponential, symbol_for_print, 
			Grass->M, k, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl << endl;
		}
	
}


// ####################################################################################
// global functions:
// ####################################################################################


void translation_plane_lifting_early_test_function(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level)
{
	translation_plane *T = (translation_plane *) data;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "translation_plane_lifting_early_test_function for set ";
		print_set(cout, len, S);
		cout << endl;
		}
	T->early_test_func(S, len, 
		candidates, nb_candidates, 
		good_candidates, nb_good_candidates, 
		verbose_level - 2);
	if (f_v) {
		cout << "translation_plane_lifting_early_test_function done" << endl;
		}
}

void translation_plane_lifting_prepare_function_new(exact_cover *EC, INT starter_case, 
	INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, INT *&col_labels, 
	INT &f_ruled_out, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	translation_plane *T = (translation_plane *) EC->user_data;

	if (f_v) {
		cout << "translation_plane_lifting_prepare_function_new nb_candidates=" << nb_candidates << endl;
		}

	T->lifting_prepare_function_new(EC, starter_case, 
		candidates, nb_candidates, Strong_gens, 
		Dio, col_labels, f_ruled_out, 
		verbose_level);


	if (f_v) {
		cout << "translation_plane_lifting_prepare_function_new after lifting_prepare_function_new" << endl;
		}

	if (f_v) {
		cout << "translation_plane_lifting_prepare_function_new nb_rows=" << Dio->m << " nb_cols=" << Dio->n << endl;
		}

	if (f_v) {
		cout << "translation_plane_lifting_prepare_function_new done" << endl;
		}
}




INT starter_canonize_callback(INT *Set, INT len, INT *Elt, void *data, INT verbose_level)
{
	translation_plane *T = (translation_plane *) data;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "starter_canonize_callback" << endl;
		}
	T->R->do_recoordinatize(Set[0], Set[1], Set[2], verbose_level - 2);
	T->A->element_move(T->R->Elt, Elt, FALSE);
	if (f_v) {
		cout << "starter_canonize_callback done" << endl;
		}
	if (f_vv) {
		cout << "transporter:" << endl;
		T->A->element_print(Elt, cout);
		}
	return TRUE;
}

INT translation_plane_check_function_incremental(INT len, INT *S, void *data, INT verbose_level)
{
	translation_plane *T = (translation_plane *) data;
	INT ret;

	ret = T->check_function_incremental(len, S, verbose_level);
	return ret;
}



