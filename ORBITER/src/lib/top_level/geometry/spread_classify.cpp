// spread_classify.cpp
// 
// Anton Betten
// November 17, 2009
//
// moved to TOP_LEVEL: November 2, 2013
// renamed to spread.cpp from translation_plane.cpp: March 25, 2018
// renamed spread_classify.cpp from spread.cpp: Aug 4, 2019
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


spread_classify::spread_classify()
{
	null();
}

spread_classify::~spread_classify()
{
	freeself();
}

void spread_classify::null()
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
	Poset = NULL;
	gen = NULL;
	Sing = NULL;
	O = NULL;
	Klein = NULL;

	Data1 = NULL;
	Data2 = NULL;
	//Data3 = NULL;
}

void spread_classify::freeself()
{
	if (A) {
		FREE_OBJECT(A);
		}
	if (A2) {
		FREE_OBJECT(A2);
		}
#if 0
	if (AG) {
		FREE_OBJECT(AG);
		}
#endif
	if (Grass) {
		FREE_OBJECT(Grass);
		}

	if (R) {
		FREE_OBJECT(R);
		}
	if (Starter) {
		FREE_int(Starter);
		}
	if (Starter_Strong_gens) {
		FREE_OBJECT(Starter_Strong_gens);
		}
	if (tmp_M1) {
		FREE_int(tmp_M1);
		}
	if (tmp_M2) {
		FREE_int(tmp_M2);
		}
	if (tmp_M3) {
		FREE_int(tmp_M3);
		}
	if (tmp_M4) {
		FREE_int(tmp_M4);
		}

	
	if (Sing) {
		FREE_OBJECT(Sing);
		}
	if (O) {
		FREE_OBJECT(O);
		}
	if (Klein) {
		FREE_OBJECT(Klein);
		}
	if (Data1) {
		FREE_int(Data1);
		}
	if (Data2) {
		FREE_int(Data2);
		}
#if 0
	if (Data3) {
		FREE_int(Data3);
		}
#endif
	null();
}

void spread_classify::init(int order, int n, int k, int max_depth,
	finite_field *F, int f_recoordinatize, 
	const char *input_prefix, 
	const char *base_fname,
	int starter_size,  
	int argc, const char **argv, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_object go;
	number_theory_domain NT;
	combinatorics_domain Combi;
	
	
	if (f_v) {
		cout << "spread::init" << endl;
		cout << "n=" << n << endl;
		cout << "k=" << k << endl;
		cout << "q=" << F->q << endl;
		}
	spread_classify::argc = argc;
	spread_classify::argv = argv;
	
	spread_classify::order = order;
	spread_size = order + 1;
	spread_classify::n = n;
	spread_classify::k = k;
	spread_classify::max_depth = max_depth;
	kn = k * n;
	spread_classify::F = F;
	spread_classify::f_recoordinatize = f_recoordinatize;
	q = F->q;
	
	strcpy(starter_directory_name, input_prefix);
	strcpy(prefix, base_fname);
	//sprintf(prefix_with_directory, "%s%s",
	//starter_directory_name, base_fname);
	spread_classify::starter_size = starter_size;


	tmp_M1 = NEW_int(n * n);
	tmp_M2 = NEW_int(n * n);
	tmp_M3 = NEW_int(n * n);
	tmp_M4 = NEW_int(n * n);
	
	gen = NEW_OBJECT(poset_classification);
	gen->read_arguments(argc, argv, 1);


	f_projective = TRUE;
	f_semilinear = TRUE;
	f_basis = TRUE;
	f_induce_action = FALSE;

	if (NT.is_prime(q)) {
		if (f_v) {
			cout << "spread::init q=" << q << " is a prime, "
					"putting f_semilinear = FALSE" << endl;
			}
		f_semilinear = FALSE;
		}
	else {
		if (f_v) {
			cout << "spread::init q=" << q
					<< " is not a prime" << endl;
			}
		}


	A = NEW_OBJECT(action);
	A2 = NEW_OBJECT(action);
	AG = NEW_OBJECT(action_on_grassmannian);


	if (f_v) {
		cout << "spread::init "
				"before init_projective_group" << endl;
		}

	vector_ge *nice_gens;

	A->init_projective_group(n, F, f_semilinear,
			f_basis,
			nice_gens,
			0 /*verbose_level*/);
	FREE_OBJECT(nice_gens);
	
	if (f_v) {
		cout << "spread::init "
				"after init_projective_group, "
				"checking group order" << endl;
		}
	A->Sims->group_order(go);
	if (f_v) {
		cout << "spread::init "
				"after init_projective_group "
				"group of order " << go
				<< " has been created" <<  endl;
		}


	if (f_vv) {
		cout << "action A created: ";
		A->print_info();
		}



	Grass = NEW_OBJECT(grassmann);
	Grass->init(n, k, F, 0 /*MINIMUM(verbose_level - 1, 1)*/);
	
	nCkq = Combi.generalized_binomial(n, k, q);
	block_size = r = Combi.generalized_binomial(k, 1, q);
	nb_points_total = nb_pts = Combi.generalized_binomial(n, 1, q);
	
	if (f_v) {
		cout << "spread::init "
				"nCkq = {n \\choose k}_q = " << nCkq << endl;
		cout << "spread::init "
				"r = {k \\choose 1}_q = " << r << endl;
		cout << "spread::init "
				"nb_pts = {n \\choose 1}_q = " << nb_pts << endl;
		}



	if (f_v) {
		cout << "spread::init before AG->init" <<  endl;
		}
	
	AG->init(*A, Grass, 0 /*verbose_level - 2*/);
	
	if (f_v) {
		cout << "spread::init after AG->init" <<  endl;
		}

	A2->induced_action_on_grassmannian(A, AG, 
		f_induce_action, NULL /*sims *old_G */, 
		MINIMUM(verbose_level - 2, 2));
	
	if (f_v) {
		cout << "spread::init after "
				"A2->induced_action_on_grassmannian" <<  endl;
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

	//int len;
	


	if (f_print_generators) {
		int f_print_as_permutation = TRUE;
		int f_offset = FALSE;
		int offset = 1;
		int f_do_it_anyway_even_for_big_degree = TRUE;
		int f_print_cycles_of_length_one = FALSE;
		
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
		cout << "spread::init The order "
				"of PGGL(n,q) is " << go << endl;
		}

	
	if (f_recoordinatize) {
		if (f_v) {
			cout << "spread::init before "
					"recoordinatize::init" << endl;
			}
		R = NEW_OBJECT(recoordinatize);
		R->init(n, k, F, Grass, A, A2, 
			f_projective, f_semilinear, 
			callback_incremental_check_function, (void *) this,
			verbose_level);

		if (f_v) {
			cout << "spread::init before "
					"recoordinatize::compute_starter" << endl;
			}
		R->compute_starter(Starter, Starter_size, 
			Starter_Strong_gens, MINIMUM(verbose_level - 1, 1));

		longinteger_object go;
		Starter_Strong_gens->group_order(go);
		if (TRUE /*f_v*/) {
			cout << "spread::init The stabilizer of the "
					"first three components has order " << go << endl;
			}


		Nb = R->nb_live_points;
		}
	else {
		if (f_v) {
			cout << "spread::init we are not using "
					"recoordinatization, please use option "
					"-recoordinatize" << endl;
			//exit(1);
			}
		Nb = Combi.generalized_binomial(n, k, q); //R->nCkq; // this makes no sense
		}

	if (f_v) {
		cout << "spread::init Nb = " << Nb << endl;
		cout << "spread::init kn = " << kn << endl;
		cout << "spread::init n = " << n << endl;
		cout << "spread::init k = " << k << endl;
		cout << "spread::init allocating Data1 and Data2" << endl;
		}
	
	Data1 = NEW_int(max_depth * kn);
	Data2 = NEW_int(n * n);
	//Data3 = NEW_int(n * n);
	

#if 0
	if (k == 2 && is_prime(q)) {
		Sing = NEW_OBJECT(singer_cycle);
		if (f_v) {
			cout << "spread::init "
					"before singer_cycle::init" << endl;
			}
		Sing->init(4, F, A, A2, 0 /*verbose_level*/);
		Sing->init_lines(0 /*verbose_level*/);
		}
#endif

	if (k == 2) {
		
		if (f_v) {
			cout << "spread::init k = 2, "
					"initializing klein correspondence" << endl;
			}
		Klein = NEW_OBJECT(klein_correspondence);
		O = NEW_OBJECT(orthogonal);
		
		O->init(1 /* epsilon */, 6, F, 0 /* verbose_level*/);
		Klein->init(F, O, 0 /* verbose_level */);
		}
	else {
		if (f_v) {
			cout << "spread::init we are not "
					"initializing klein correspondence" << endl;
			}
		O = NULL;
		Klein = NULL;
		}
	
	if (f_v) {
		cout << "spread::init done" << endl;
		}
}

void spread_classify::unrank_point(int *v, int a)
{
	F->PG_element_unrank_modified(v, 1, n, a);
}

int spread_classify::rank_point(int *v)
{
	int a;
	
	F->PG_element_rank_modified(v, 1, n, a);
	return a;
}

void spread_classify::unrank_subspace(int *M, int a)
{
	Grass->unrank_int_here(M, a, 0/*verbose_level - 4*/);
}

int spread_classify::rank_subspace(int *M)
{
	int a;
	
	a = Grass->rank_int_here(M, 0 /*verbose_level*/);
	return a;
}

void spread_classify::print_points()
{
	int *v;
	int i;

	cout << "spread_classify::print_points" << endl;
	v = NEW_int(n);
	for (i = 0; i < nb_pts; i++) {
		unrank_point(v, i);
		cout << "point " << i << " : ";
		int_vec_print(cout, v, n);
		cout << endl;
		}
	FREE_int(v);
}

void spread_classify::print_points(int *pts, int len)
{
	int *v;
	int h, i;

	cout << "spread_classify::print_points" << endl;
	v = NEW_int(n);
	for (h = 0; h < len; h++) {
		i = pts[h];
		unrank_point(v, i);
		cout << "point " << h << " : " << i << " : ";
		int_vec_print(cout, v, n);
		cout << endl;
		}
	FREE_int(v);
}

void spread_classify::print_elements()
{
	int i, j;
	int *M;
	
	M = NEW_int(kn);
	for (i = 0; i < nCkq; i++) {
		if (FALSE) {
			cout << i << ":" << endl;
			}
		unrank_subspace(M, i);
		if (FALSE) {
			print_integer_matrix_width(cout, M,
					k, n, n, F->log10_of_q + 1);
			}
		j = rank_subspace(M);
		if (j != i) {
			cout << "rank yields " << j << " != " << i << endl;
			exit(1);
			}
		}
	FREE_int(M);
}

void spread_classify::print_elements_and_points()
{
	int i, a, b;
	int *M, *v, *w;
	int *Line;

	cout << "spread_classify::print_elements_and_points" << endl;
	M = NEW_int(kn);
	v = NEW_int(k);
	w = NEW_int(n);
	Line = NEW_int(r);
	for (i = 0; i < nCkq; i++) {
		if (FALSE) {
			cout << i << ":" << endl;
			}
		unrank_subspace(M, i);
		for (a = 0; a < r; a++) {
			F->PG_element_unrank_modified(v, 1, k, a);
			F->mult_matrix_matrix(v, M, w, 1, k, n,
					0 /* verbose_level */);
			b = rank_point(w);
			Line[a] = b;
			}
		cout << "line " << i << ":" << endl;
		print_integer_matrix_width(cout, M,
				k, n, n, F->log10_of_q + 1);
		cout << "points on subspace " << i << " : ";
		int_vec_print(cout, Line, r);
		cout << endl;
		}
	FREE_int(M);
	FREE_int(v);
	FREE_int(w);
	FREE_int(Line);
}

void spread_classify::read_arguments(int argc, const char **argv)
{
	int i;
	
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

void spread_classify::init2(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int depth;
	
	if (f_v) {
		cout << "spread_classify::init2" << endl;
		}
	//depth = order + 1;

	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(A, A2,
			A->Strong_gens,
			verbose_level);
	Poset->add_testing_without_group(
			spread_early_test_func_callback,
				this /* void *data */,
				verbose_level);

	
	if (f_recoordinatize) {
		if (f_v) {
			cout << "spread_classify::init2 "
					"before gen->initialize_with_starter" << endl;
			}
		gen->initialize_with_starter(Poset,
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
			cout << "spread_classify::init2 "
					"after gen->initialize_with_starter" << endl;
			}
		}
	else {
		if (f_v) {
			cout << "spread_classify::init2 "
					"before gen->initialize" << endl;
			}
		gen->initialize(Poset,
			order + 1, 
			starter_directory_name, prefix, 
			verbose_level - 2);
		if (f_v) {
			cout << "spread_classify::init2 "
					"after gen->initialize" << endl;
			}
		}

	gen->f_allowed_to_show_group_elements = TRUE;


#if 0
	gen->f_print_function = TRUE;
	gen->print_function = callback_spread_print;
	gen->print_function_data = this;
#endif	


	if (f_v) {
		cout << "spread_classify::init2 done" << endl;
		}
}

void spread_classify::compute(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int schreier_depth = gen->depth;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	int t0;
	os_interface Os;


	if (f_v) {
		cout << "spread_classify::compute starter_size=" << starter_size << endl;
		}

	
	if (f_override_schreier_depth) {
		schreier_depth = override_schreier_depth;
		}
	if (f_v) {
		cout << "spread_classify::compute calling generator_main" << endl;
		}

	gen->f_max_depth = TRUE;
	gen->max_depth = starter_size;
	
	t0 = Os.os_ticks();
	gen->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		verbose_level - 1);
	
	int length;
	
	if (f_v) {
		cout << "spread_classify::compute done with generator_main" << endl;
		}
	length = gen->nb_orbits_at_level(gen->max_depth);
	if (f_v) {
		cout << "spread_classify::compute We found " << length << " orbits on "
			<< gen->max_depth << "-sets of " << k 
			<< "-subspaces in PG(" << n - 1 << "," << q << ")" 
			<< " satisfying the partial spread condition" << endl;
		}



	if (f_v) {
		cout << "spread_classify::compute done" << endl;
		}
}


void spread_classify::early_test_func(int *S, int len,
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	int verbose_level)
// for poset classification
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i0, i, j, rk;
	int *M;
	int *MM;
	int *B, *base_cols;
		
	if (f_v) {
		cout << "spread_classify::early_test_func checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size " << nb_candidates << ":" << endl;
		int_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		if (f_vv) {
			if (nb_candidates < 100) {
				for (i = 0; i < nb_candidates; i++) {
					Grass->unrank_int(candidates[i], 0/*verbose_level - 4*/);
					cout << "candidate " << i << "="
							<< candidates[i] << ":" << endl;
					print_integer_matrix_width(cout,
							Grass->M, k, n, n, F->log10_of_q + 1);
					}
				}
			else {
				cout << "too many to print" << endl;
				f_vv = FALSE;
			}
			}
		}

	if (len + 1 > max_depth) {
		cout << "spread_classify::early_test_func len + 1 > max_depth" << endl;
		exit(1);
		}
	M = Data2; // [n * n]
	MM = Data1; // [(len + 1) * kn]
	B = tmp_M3;
	base_cols = tmp_M4;

	for (i = 0; i < len; i++) {
		unrank_subspace(MM + i * kn, S[i]);
		}
	if (f_v) {
		for (i = 0; i < len; i++) {
			cout << "p_" << i << "=" << S[i] << ":" << endl;
			print_integer_matrix_width(cout,
					MM + i * k * n, k, n, n, F->log10_of_q + 1);
			}
		}
	
	nb_good_candidates = 0;
	for (j = 0; j < nb_candidates; j++) {
		Grass->unrank_int(candidates[j], 0/*verbose_level - 4*/);
		if (len == 0) {
			i0 = 0;
			}
		else {
			i0 = len - 1;
			}
		for (i = i0; i < len; i++) {
			int_vec_copy(MM + i * kn, M, k * n);
			int_vec_copy(Grass->M, M + kn, k * n);

			if (f_vv) {
				cout << "testing (p_" << i << ",candidates[" << j << "])="
						"(" << S[i] <<  "," << candidates[j] << ")" << endl;
				print_integer_matrix_width(cout, M,
						n, n, n, F->log10_of_q + 1);
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
	
	if (f_v) {
		cout << "spread_classify::early_test_func we found " << nb_good_candidates
				<< " good candidates" << endl;
		}
	if (f_v) {
		cout << "spread_classify::early_test_func done" << endl;
		}
}

int spread_classify::check_function(int len, int *S, int verbose_level)
// checks all {len \choose 2} pairs. This is very inefficient.
// This function should not be used for poset classification!
{
	int f_OK = TRUE;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, rk;
	int *M, *M1;
	int *B, *base_cols;
		
	if (f_v) {
		cout << "spread_classify::check_function checking set ";
		print_set(cout, len, S);
		cout << endl;
		}
	M1 = tmp_M1; // [kn]
	M = tmp_M2; // [n * n]
	B = tmp_M3;
	base_cols = tmp_M4;
	
	if (f_v) {
		for (i = 0; i < len; i++) {
			cout << "p_" << i << "=" << S[i] << ":" << endl;
			Grass->unrank_int(S[i], 0/*verbose_level - 4*/);
			print_integer_matrix_width(cout, Grass->M,
					k, n, n, F->log10_of_q + 1);
			}
		}

	for (i = 0; i < len; i++) {
		unrank_subspace(M1, S[i]);
		for (j = i + 1; j < len; j++) {
			int_vec_copy(M1, M, kn);
			unrank_subspace(M + kn, S[j]);

			if (f_vv) {
				cout << "testing (p_" << i << ",p_" << j << ")"
						"=(" << S[i] << "," << S[j] << ")" << endl;
				print_integer_matrix_width(cout, M,
						n, n, n, F->log10_of_q + 1);
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

int spread_classify::incremental_check_function(int len, int *S, int verbose_level)
// checks the pairs (0,len-1),(1,len-1),\ldots,(len-2,len-1) 
// for recoordinatize
{
	int f_OK = TRUE;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, rk;
	int *M, *M1;
	int *B, *base_cols;
		
	if (f_v) {
		cout << "spread_classify::incremental_check_function checking set ";
		print_set(cout, len, S);
		cout << endl;
		}
	if (len <= 1) {
		f_OK = TRUE;
		goto finish;
		}
	M1 = tmp_M1; // [kn]
	M = tmp_M2; // [n * n]
	B = tmp_M3;
	base_cols = tmp_M4;
	
	if (f_v) {
		for (i = 0; i < len; i++) {
			cout << "p_" << i << "=" << S[i] << ":" << endl;
			Grass->unrank_int(S[i], 0/*verbose_level - 4*/);
			print_integer_matrix_width(cout,
					Grass->M, k, n, n, F->log10_of_q + 1);
			}
		}
	
	j = len - 1;
	
	unrank_subspace(M1, S[j]);
	for (i = 0; i < len - 1; i++) {
		unrank_subspace(M, S[i]);
		int_vec_copy(M1, M + kn, kn);
		
		if (f_vv) {
			cout << "testing (p_" << i << ",p_" << j << ")"
					"=(" << S[i] <<  "," << S[j] << ")" << endl;
			print_integer_matrix_width(cout, M,
					n, n, n, F->log10_of_q + 1);
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

#if 0
int spread_classify::check_function_pair(int rk1, int rk2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int rk;
	int *M;
	int *B, *base_cols;
		
	if (f_v) {
		cout << "spread_classify::check_function_pair "
				"checking (" << rk1 << "," << rk2 << ")" << endl;
		}
	M = tmp_M1; // [n * n]
	B = tmp_M3;
	base_cols = tmp_M4;
	
	unrank_subspace(M, rk1);
	unrank_subspace(M + kn, rk2);

	if (f_vv) {
		cout << "testing (" << rk1 <<  "," << rk2 << ")" << endl;
		print_integer_matrix_width(cout, M,
				n, n, n, F->log10_of_q + 1);
		}
	rk = F->rank_of_matrix_memory_given(M, n, B, base_cols, 0);

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
#endif

void spread_classify::lifting_prepare_function_new(
	exact_cover *E, int starter_case,
	int *candidates, int nb_candidates,
	strong_generators *Strong_gens,
	diophant *&Dio, int *&col_labels, 
	int &f_ruled_out, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_v3 = (verbose_level >= 3);
	
	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"nb_candidates=" << nb_candidates << endl;
		}


	spread_lifting *SL;

	SL = NEW_OBJECT(spread_lifting);

	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"before SL->init" << endl;
		}
	SL->init(this, E, 
		E->starter, E->starter_size, 
		starter_case, E->starter_nb_cases, 
		candidates, nb_candidates, Strong_gens, 
		E->f_lex, 
		verbose_level);
	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"after SL->init" << endl;
		}

	
	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"before SL->create_system" << endl;
		}

	Dio = SL->create_system(verbose_level - 2);
	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"after SL->create_system" << endl;
		}

	int *col_color;
	int nb_colors;

	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"before SL->find_coloring" << endl;
		}
	SL->find_coloring(Dio, 
		col_color, nb_colors, 
		verbose_level - 2);
	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"after SL->find_coloring" << endl;
		}

	if (f_v3) {
		cout << "col_color=";
		int_vec_print(cout, col_color, Dio->n);
		cout << endl;
		}

	uchar *Adj;
	
	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"before Dio->make_clique_graph_adjacency_matrix" << endl;
		}
	Dio->make_clique_graph_adjacency_matrix(Adj, verbose_level - 2);
	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"after Dio->make_clique_graph_adjacency_matrix" << endl;
		}

	colored_graph *CG;

	CG = NEW_OBJECT(colored_graph);

	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"before CG->init_with_point_labels" << endl;
		}
	CG->init_with_point_labels(SL->nb_cols, nb_colors, 1,
		col_color, Adj, TRUE /* f_ownership_of_bitvec */, 
		SL->col_labels /* point_labels */, 
		verbose_level);
	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"after CG->init_with_point_labels" << endl;
		}
	
	char fname_clique_graph[1000];
	file_io Fio;

	sprintf(fname_clique_graph, "%sgraph_%d.bin",
			E->output_prefix, starter_case);
	CG->save(fname_clique_graph, verbose_level - 1);
	if (f_v) {
		cout << "Written file " << fname_clique_graph
				<< " of size " << Fio.file_size(fname_clique_graph) << endl;
		}

	FREE_OBJECT(CG);

	col_labels = SL->col_labels;
	SL->col_labels = NULL;

	FREE_OBJECT(SL);
	//FREE_uchar(Adj);
	FREE_int(col_color);
	
	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"after SL->create_system" << endl;
		}

	if (f_v) {
		cout << "spread_classify::lifting_prepare_function_new "
				"done" << endl;
		}
}



void spread_classify::compute_dual_spread(int *spread,
		int *dual_spread, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_classify::compute_dual_spread" << endl;
		}

	Grass->compute_dual_spread(spread, dual_spread,
			spread_size, verbose_level - 1);

	if (f_v) {
		cout << "spread_classify::compute_dual_spread done" << endl;
		}
}

void spread_classify::print(ostream &ost, int len, int *S)
{
	int i;
	int f_elements_exponential = FALSE;
	const char *symbol_for_print = "\\alpha";
	
	if (len == 0) {
		return;
		}
	for (i = 0; i < len; i++) {
		ost << "$S_{" << i + 1 << "}$ has rank " << S[i]
			<< " and is generated by\\\\" << endl;
		Grass->unrank_int(S[i], 0);
		ost << "$$" << endl;
		ost << "\\left[" << endl;
		F->latex_matrix(ost, f_elements_exponential, symbol_for_print,
			Grass->M, k, n);
		ost << "\\right]" << endl;
		ost << "$$" << endl << endl;
		}
	
}


// #############################################################################
// global functions:
// #############################################################################


void spread_lifting_early_test_function(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level)
{
	spread_classify *Spread = (spread_classify *) data;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "spread_lifting_early_test_function for set ";
		print_set(cout, len, S);
		cout << endl;
		}
	Spread->early_test_func(S, len,
		candidates, nb_candidates, 
		good_candidates, nb_good_candidates, 
		verbose_level - 2);
	if (f_v) {
		cout << "spread_lifting_early_test_function done" << endl;
		}
}

void spread_lifting_prepare_function_new(
	exact_cover *EC, int starter_case,
	int *candidates, int nb_candidates,
	strong_generators *Strong_gens,
	diophant *&Dio, int *&col_labels, 
	int &f_ruled_out, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	spread_classify *Spread = (spread_classify *) EC->user_data;

	if (f_v) {
		cout << "spread_lifting_prepare_function_new "
				"nb_candidates=" << nb_candidates << endl;
		}

	Spread->lifting_prepare_function_new(EC, starter_case,
		candidates, nb_candidates, Strong_gens, 
		Dio, col_labels, f_ruled_out, 
		verbose_level);


	if (f_v) {
		cout << "spread_lifting_prepare_function_new "
				"after lifting_prepare_function_new" << endl;
		}

	if (f_v) {
		cout << "spread_lifting_prepare_function_new "
				"nb_rows=" << Dio->m
				<< " nb_cols=" << Dio->n << endl;
		}

	if (f_v) {
		cout << "spread_lifting_prepare_function_new "
				"done" << endl;
		}
}




int starter_canonize_callback(int *Set, int len,
		int *Elt, void *data, int verbose_level)
// for starter, interface to recoordinatize,
// which uses callback_incremental_check_function
{
	spread_classify *Spread = (spread_classify *) data;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "starter_canonize_callback" << endl;
		}
	Spread->R->do_recoordinatize(Set[0], Set[1], Set[2], verbose_level - 2);
	Spread->A->element_move(Spread->R->Elt, Elt, FALSE);
	if (f_v) {
		cout << "starter_canonize_callback done" << endl;
		}
	if (f_vv) {
		cout << "transporter:" << endl;
		Spread->A->element_print(Elt, cout);
		}
	return TRUE;
}

int callback_incremental_check_function(
		int len, int *S, void *data, int verbose_level)
// for recoordinatize
{
	spread_classify *Spread = (spread_classify *) data;
	int ret;

	ret = Spread->incremental_check_function(len, S, verbose_level);
	return ret;
}

}}



