/*
 * semifield_classify.cpp
 *
 *  Created on: Apr 17, 2019
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



semifield_classify::semifield_classify()
{
	n = 0;
	k = 0;
	k2 = 0;
	F = NULL;
	q = 0;
	order = 0;

	f_level_two_prefix = FALSE;
	level_two_prefix = NULL;

	f_level_three_prefix = FALSE;
	level_three_prefix = NULL;

	T = NULL;

	A = NULL;
	Elt1 = NULL;
	G = NULL;

	A0 = NULL;
	A0_linear = NULL;
	A_on_S = NULL;
	AS = NULL;
	Strong_gens = NULL;
	Poset = NULL;
	Gen = NULL;
	Symmetry_group = NULL;

	vector_space_dimension = 0;
	schreier_depth = 0;

	// for test_partial_semifield:
	test_base_cols = NULL;
	test_v = NULL;
	test_w = NULL;
	test_Basis = NULL;
	Basis1 = NULL;
	Basis2 = NULL;
	desired_pivots = NULL;
	//null();
}

semifield_classify::~semifield_classify()
{
	freeself();
}

void semifield_classify::null()
{


}

void semifield_classify::freeself()
{
	if (A0) {
		FREE_OBJECT(A0);
		}
	if (A0_linear) {
		FREE_OBJECT(A0_linear);
		}
	if (T) {
		FREE_OBJECT(T);
		}
	if (Elt1) {
		FREE_int(Elt1);
		}
	if (Symmetry_group) {
		FREE_OBJECT(Symmetry_group);
		}
	if (Poset) {
		FREE_OBJECT(Poset);
	}
	if (Gen) {
		FREE_OBJECT(Gen);
	}
	if (test_base_cols) {
		FREE_int(test_base_cols);
		}
	if (test_v) {
		FREE_int(test_v);
		}
	if (test_w) {
		FREE_int(test_w);
		}
	if (test_Basis) {
		FREE_int(test_Basis);
		}
	if (Basis1) {
		FREE_int(Basis1);
		}
	if (Basis2) {
		FREE_int(Basis2);
		}
	if (desired_pivots) {
		FREE_int(desired_pivots);
		}
	null();
}

void semifield_classify::init(int argc, const char **argv,
	int order, int n, int k,
	finite_field *F,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object go;
	int i;
	number_theory_domain NT;

	if (f_v) {
		cout << "semifield_classify::init" << endl;
		}

	semifield_classify::n = n;
	semifield_classify::k = k;
	k2 = k * k;
	semifield_classify::F = F;
	semifield_classify::q = F->q;
	semifield_classify::order = order;
	if (order != NT.i_power_j(q, k)) {
		cout << "semifield_classify::init "
				"order != i_power_j(q, k)" << endl;
		exit(1);
		}

	if (sizeof(long int) * 8 - 1 < k2) {
		cout << "sizeof(long int) * 8 - 1 < k2, overflow will happen" << endl;
		cout << "sizeof(long int)=" << sizeof(long int) << endl;
		cout << "k2=" << k2 << endl;
		exit(1);
	}

	if (f_v) {
		cout << "semifield_classify::init q=" << q << endl;
		cout << "semifield_classify::init k=" << k << endl;
		cout << "semifield_classify::init n=" << n << endl;
		cout << "semifield_classify::init order=" << order << endl;
		}

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-level2_prefix") == 0) {
			f_level_two_prefix = TRUE;
			level_two_prefix = argv[++i];
			cout << "-level2_prefix " << level_two_prefix << endl;
			}
		else if (strcmp(argv[i], "-level3_prefix") == 0) {
			f_level_three_prefix = TRUE;
			level_three_prefix = argv[++i];
			cout << "-level3_prefix " << level_three_prefix << endl;
			}
		}

	vector_space_dimension = k2;

	// for test_partial_semifield:
	test_base_cols = NEW_int(n);
	test_v = NEW_int(n);
	test_w = NEW_int(k2);
	test_Basis = NEW_int(k * k2);
	Basis1 = NEW_int(k * k2);
	Basis2 = NEW_int(k * k2);


	T = NEW_OBJECT(spread);

	T->read_arguments(argc, argv);

	if (f_v) {
		cout << "semifield_classify::init "
				"before T->init" << endl;
		}

	int max_depth = k + 1;

	T->init(order, n, k, max_depth,
		F, FALSE /* f_recoordinatize */,
		"TP_STARTER", "TP", order + 1,
		argc, argv,
		verbose_level - 2);

	if (f_v) {
		cout << "semifield_classify::init "
				"after T->init" << endl;
		}

	longinteger_object go1, go2;
	int f_semilinear = TRUE;

	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
		}

	A0 = NEW_OBJECT(action);
	A0_linear = NEW_OBJECT(action);

	if (f_v) {
		cout << "semifield_classify::init "
				"before A0->init_projective_group" << endl;
		}

	vector_ge *nice_gens;

	A0->init_projective_group(
		k, F, f_semilinear,
		TRUE /* f_basis */,
		nice_gens,
		0 /* verbose_level */);
	FREE_OBJECT(nice_gens);

	if (f_v) {
		cout << "semifield_classify::init "
				"after init_projective_group, "
				"checking group order of Sims of A0" << endl;
		}
	A0->Sims->group_order(go);
	if (f_v) {
		cout << "semifield_classify::init "
				"after init_projective_group "
				"group of order " << go << " has been created" <<  endl;
		}



	A0->group_order(go1);
	if (f_v) {
		cout << "semifield_classify::init "
				"target_go=" << go1
			<< " = order of PGGL(" << k << "," << q << ")" << endl;
		cout << "action A0 created: ";
		A0->print_info();
		}

	A0_linear->init_projective_group(k,
		F, FALSE /*f_semilinear */,
		TRUE /* f_basis */,
		nice_gens,
		0 /* verbose_level */);
	FREE_OBJECT(nice_gens);

	if (f_v) {
		cout << "semifield_classify::init "
				"after init_projective_group, "
				"checking group order of Sims of A0_linear" << endl;
		}
	A0_linear->Sims->group_order(go);
	if (f_v) {
		cout << "semifield_classify::init "
				"after init_projective_group "
				"group of order " << go << " has been created" <<  endl;
		}

	A0_linear->group_order(go2);
	if (f_v) {
		cout << "semifield_classify::init order of PGL(" << k << ","
				<< q << ") is " << go2 << endl;
		cout << "action A0_linear created: ";
		A0_linear->print_info();
		}




	A = T->A;

	Elt1 = NEW_int(A->elt_size_in_int);

	G = A0_linear->Sims;




	A_on_S = NEW_OBJECT(action_on_spread_set);

	if (f_v) {
		cout << "semifield_classify::init "
				"before A_on_S->init" << endl;
		}
	A_on_S->init(T->A /* A_PGL_n_q */,
		A0 /* A_PGL_k_q */,
		A0_linear->Sims /* G_PGL_k_q */,
		k, F,
		verbose_level - 2);
	if (f_v) {
		cout << "semifield_classify::init "
				"after A_on_S->init" << endl;
		}




	AS = NEW_OBJECT(action);

	if (f_v) {
		cout << "semifield_classify::init "
				"before induced_action_on_spread_set" << endl;
		}
	AS->induced_action_on_spread_set(T->A,
		A_on_S,
		FALSE /* f_induce_action */,
		NULL /* old_G */,
		verbose_level - 2);
	if (f_v) {
		cout << "semifield_classify::init "
				"after induced_action_on_spread_set "
				"the degree of the induced action "
				"is " << AS->degree << endl;
		}


	if (f_v) {
		cout << "semifield_classify::init "
				"before list_points" << endl;
		}
	list_points();
	if (f_v) {
		cout << "semifield_classify::init "
				"after list_points" << endl;
		}



	if (f_v) {
		cout << "semifield_classify::init "
				"before Strong_gens->generators_for_"
				"the_stabilizer_of_two_components" << endl;
		}
	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->generators_for_the_stabilizer_of_two_components(
		T->A /* A_PGL_n_q */,
		T->A->G.matrix_grp /* Mtx*/,
		verbose_level);

	if (f_v) {
		cout << "semifield_classify::init "
				"after Strong_gens->generators_for_"
				"the_stabilizer_of_two_components" << endl;
		}


	Symmetry_group = Strong_gens->create_sims(verbose_level);




	init_desired_pivots(verbose_level);


	if (f_v) {
		cout << "semifield_classify::init done" << endl;
		}
}


void semifield_classify::init_poset_classification(
		int argc, const char **argv,
		const char *prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object go;

	if (f_v) {
		cout << "semifield_classify::init_poset_classification" << endl;
		}

	Poset = NEW_OBJECT(poset);
	vector_space *VS;
	VS = NEW_OBJECT(vector_space);
	VS->init(F, vector_space_dimension,
			verbose_level);
	VS->init_rank_functions(
			semifield_classify_rank_point_func,
			semifield_classify_unrank_point_func,
			this,
			verbose_level);

#if 0
	Poset->init_subset_lattice(T->A, AS,
			Strong_gens,
			verbose_level);
#endif
	Poset->init_subspace_lattice(T->A, AS,
			Strong_gens,
			VS,
			verbose_level);

	if (f_v) {
		cout << "semifield_classify::init before "
				"Poset->add_testing_without_group" << endl;
		}
	Poset->add_testing_without_group(
			semifield_classify_early_test_func,
				this /* void *data */,
				verbose_level);



	Gen = NEW_OBJECT(poset_classification);

	Gen->read_arguments(argc, argv, 0);

	//Gen->prefix[0] = 0;
	sprintf(Gen->fname_base, "%s", prefix);


	Gen->depth = k;

	if (f_v) {
		cout << "semifield_classify::init before Gen->init" << endl;
		}
	Gen->init(Poset,
			Gen->depth /* sz */,
			verbose_level - 2);
	if (f_v) {
		cout << "semifield_classify::init after Gen->init" << endl;
		}



	int nb_nodes = 1000;

	if (f_v) {
		cout << "semifield_classify::init before "
				"Gen->init_poset_orbit_node" << endl;
		}
	Gen->init_poset_orbit_node(nb_nodes,
			verbose_level - 2);
	if (f_v) {
		cout << "semifield_classify::init calling "
				"Gen->init_root_node" << endl;
		}
	Gen->root[0].init_root_node(Gen, verbose_level - 2);

	schreier_depth = Gen->depth;

	if (f_v) {
		cout << "semifield_classify::init_poset_classification done" << endl;
		}
}


void semifield_classify::compute_orbits(int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t0 = os_ticks();
	file_io Fio;

	if (f_v) {
		cout << "semifield_classify::compute_orbits "
				"calling generator_main" << endl;
		cout << "A=";
		Gen->Poset->A->print_info();
		cout << "A2=";
		Gen->Poset->A2->print_info();
		}
	Gen->depth = depth;
	Gen->main(t0,
		schreier_depth,
		FALSE /*f_use_invariant_subset_if_available*/,
		FALSE /*f_debug*/,
		verbose_level - 1);

	int nb_orbits;

	if (f_v) {
		cout << "semifield_classify::compute_orbits "
				"done with generator_main" << endl;
		}
	nb_orbits = Gen->nb_orbits_at_level(Gen->depth);
	if (f_v) {
		cout << "semifield_classify::compute_orbits "
				"we found " << nb_orbits
				<< " orbits at depth " << Gen->depth << endl;
		}

	char fname[1000];

	sprintf(fname, "semifield_list_order%d.csv", order);
	{
	int *set;
	int *Table;
	int *v;
	int i, j;

	set = NEW_int(k);
	Table = NEW_int(nb_orbits * k);
	v = NEW_int(k2);
	for (i = 0; i < nb_orbits; i++) {
		Gen->get_set_by_level(k, i, set);
		for (j = 0; j < k; j++) {
			unrank_point(v, set[j], 0/* verbose_level*/);
			set[j] = matrix_rank(v);
			}
		int_vec_copy(set, Table + i * k, k);
		}
	Fio.int_matrix_write_csv(fname, Table, nb_orbits, k);

	FREE_int(set);
	FREE_int(Table);
	FREE_int(v);
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
}


void semifield_classify::list_points()
{
	int *v;
	int rk;
	longinteger_object go;
	int goi;

	cout << "semifield_classify::list_points" << endl;
	v = NEW_int(k2);
	G->group_order(go);
	goi = go.as_int();
	cout << "semifield_classify::list_points go=" << goi << endl;
	if (goi < 1000) {
		for (rk = 0; rk < goi; rk++) {
			unrank_point(v, rk, 0 /* verbose_level */);
			cout << rk << " / " << goi << ":" << endl;
			int_matrix_print(v, k, k);
			cout << endl;
			}
		}
	else {
		cout << "too many points to list" << endl;
		}
	FREE_int(v);
}

int semifield_classify::rank_point(int *v, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int /*r,*/ rk;

	if (f_v) {
		cout << "semifield_classify::rank_point" << endl;
		}
	int_vec_copy(v, A_on_S->mtx1, k2);
	G->A->make_element(Elt1, A_on_S->mtx1, 0);
	if (f_vv) {
		cout << "semifield_classify::rank_point "
				"The rank of" << endl;
		int_matrix_print(A_on_S->mtx1, k, k);
		}
	rk = G->element_rank_int(Elt1);
	if (f_vv) {
		cout << "is " << rk << endl;
		}
	if (f_v) {
		cout << "semifield_classify::rank_point done" << endl;
		}
	return rk;
}

void semifield_classify::unrank_point(int *v, int rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "semifield_classify::unrank_point" << endl;
		}
	if (rk >= AS->degree) {
		cout << "semifield_classify::unrank_point "
				"rk >= AS->degree" << endl;
		cout << "rk=" << rk << endl;
		cout << "degree=" << AS->degree << endl;
		exit(1);
		}
	G->element_unrank_int(rk, Elt1);
	int_vec_copy(Elt1, v, k2);
	if (f_vv) {
		cout << "semifield_classify::unrank_point "
				"The element of "
				"rank " << rk << " is " << endl;
		int_matrix_print(v, k, k);
		}
	if (f_v) {
		cout << "semifield_classify::unrank_point done" << endl;
		}
}

void semifield_classify::early_test_func(int *S, int len,
	int *candidates, int nb_candidates,
	int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *M;
	int *v, *w;
	int i, j, N, r;
	number_theory_domain NT;
	geometry_global Gg;

	if (f_v) {
		cout << "semifield_classify::early_test_func" << endl;
		cout << "testing " << nb_candidates << " candidates" << endl;
		}
	if (len == 0) {
		nb_good_candidates = 0;
		for (i = 0; i < nb_candidates; i++) {
			good_candidates[nb_good_candidates++] = candidates[i];
			}
		return;
		}
	//M = NEW_int((len + 1) * k2);
	//v = NEW_int(len + 1);
	//w = NEW_int(k2);
	v = test_v; // [n]
	w = test_w; // [k2]
	M = test_Basis; // [k * k2]

	N = NT.i_power_j(q, len);
	for (i = 0; i < len; i++) {
		unrank_point(M + i * k2, S[i], 0 /*verbose_level - 2*/);
		}
	if (f_vv) {
		cout << "semifield_classify::early_test_func current set:" << endl;
		for (i = 0; i < len; i++) {
			cout << "matrix " << i << " / " << len << ":" << endl;
			int_matrix_print(M + i * k2, k, k);
			}
		}
	if (f_vv) {
		cout << "semifield_classify::early_test_func testing candidates:" << endl;
		}
	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		if ((i % 5000) == 0) {
			cout << i << " / " << nb_candidates
					<< " nb_good_candidates = "
					<< nb_good_candidates << endl;
			}
		unrank_point(M + len * k2, candidates[i], 0 /*verbose_level - 2*/);
		for (j = 0; j < N; j++) {
			if (len) {
				Gg.AG_element_unrank(q, v, 1, len, j);
				}
			v[len] = 1;
			F->mult_matrix_matrix(v, M, w, 1, len + 1, k2,
					0 /* verbose_level */);
			r = A_on_S->F->Gauss_easy(w, k, k);
			if (r != k) {
				break;
				}
			}
		if (j == N) {
			if (FALSE) {
				cout << "The candidate " << i << " / " << nb_candidates
						<< " which is " << candidates[i]
						<< " survives" << endl;
				}
			good_candidates[nb_good_candidates++] = candidates[i];
			}
		else {
			if (FALSE) {
				cout << "The candidate " << i << " / " << nb_candidates
						<< " which is " << candidates[i]
						<< " is eliminated" << endl;
				}
			}
		}
	if (f_vv) {
		cout << "The " << nb_good_candidates
				<< " accepted candidates are:" << endl;
		for (i = 0; i < nb_good_candidates; i++) {
			unrank_point(M, good_candidates[i], 0 /*verbose_level - 2*/);
			cout << i << " / " << nb_good_candidates << " is "
					<< good_candidates[i] << ":" << endl;
			int_matrix_print(M, k, k);
			}
		}
	//FREE_int(M);
	//FREE_int(v);
	//FREE_int(w);
	if (f_v) {
		cout << "semifield_classify::early_test_func done" << endl;
		cout << "Out of " << nb_candidates << " candidates, "
				<< nb_good_candidates << " survive" << endl;
		}
}

int semifield_classify::test_candidate(
		int **Mtx_stack, int stack_size, int *Mtx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = TRUE;
	int *v;
	int *w;
	int *base_cols;
	int N, h, i, j, a, b, c, r;
	number_theory_domain NT;
	geometry_global Gg;

	if (f_v) {
		cout << "semifield_classify::test_candidate" << endl;
		}
	if (stack_size > k) {
		cout << "semifield_classify::test_candidate "
				"stack_size > k" << endl;
		exit(1);
		}
	base_cols = test_base_cols;
	v = test_v;
	w = test_w;
	//base_cols = NEW_int(k);
	//v = NEW_int(stack_size);
	//w = NEW_int(k2);
	N = NT.i_power_j(q, stack_size);
	for (h = 0; h < N; h++) {
		Gg.AG_element_unrank(q, v, 1, stack_size, h);
		for (i = 0; i < k2; i++) {
			c = 0;
			for (j = 0; j < stack_size; j++) {
				a = v[j];
				b = F->mult(a, Mtx_stack[j][i]);
				c = F->add(c, b);
				}
			w[i] = F->add(c, Mtx[i]);
			}
		r = A_on_S->F->Gauss_easy_memory_given(
				w, k, k, base_cols);
		if (r != k) {
			ret = FALSE;
			break;
			}
		}
	//FREE_int(base_cols);
	//FREE_int(v);
	//FREE_int(w);
	if (f_v) {
		cout << "semifield_classify::test_candidate done" << endl;
		}
	return ret;
}

int semifield_classify::test_partial_semifield_numerical_data(
		long int *data, int data_sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Basis;
	int ret, i;

	if (f_v) {
		cout << "semifield_classify::test_partial_"
				"semifield_numerical_data" << endl;
		}
	if (data_sz > k) {
		cout << "semifield_classify::test_partial_"
				"semifield_numerical_data data_sz > k" << endl;
		exit(1);
		}
	Basis = test_Basis;
	//Basis = NEW_int(data_sz * k2);
	for (i = 0; i < data_sz; i++) {
		matrix_unrank(data[i], Basis + i * k2);
		}
	if (f_vv) {
		for (i = 0; i < data_sz; i++) {
			cout << "Basis element " << i << " is "
					<< data[i] << ":" << endl;
			int_matrix_print(Basis + i * k2, k, k);
			cout << endl;
			}
		}

	ret = test_partial_semifield(
			Basis, data_sz, verbose_level - 1);


	//FREE_int(Basis);
	if (f_v) {
		cout << "semifield_classify::test_partial_"
				"semifield_numerical_data done" << endl;
		}
	return ret;
}


int semifield_classify::test_partial_semifield(
		int *Basis, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = TRUE;
	int N, h, i, j, a, b, c, r;
	int *base_cols;
	int *v;
	int *w;
	number_theory_domain NT;
	geometry_global Gg;

	if (f_v) {
		cout << "semifield_classify::test_partial_semifield" << endl;
		}
	base_cols = test_base_cols;
	v = test_v;
	w = test_w;

	//base_cols = NEW_int(n);
	//v = NEW_int(n);
	//w = NEW_int(k2);

	N = NT.i_power_j(q, sz);
	for (h = 1; h < N; h++) {
		Gg.AG_element_unrank(q, v, 1, k, h);
		for (i = 0; i < k2; i++) {
			c = 0;
			for (j = 0; j < sz; j++) {
				a = v[j];
				b = F->mult(a, Basis[j * k2 + i]);
				c = F->add(c, b);
				}
			w[i] = c;
			}
		r = F->Gauss_easy_memory_given(w, k, k, base_cols);
		if (r != k) {
			ret = FALSE;
			if (TRUE) {
				cout << "semifield_classify::test_partial_semifield "
						"fail for vector h=" << h << " / " << N << " : ";
				cout << "r=" << r << endl;
				cout << "v=";
				int_vec_print(cout, v, sz);
				cout << endl;
				basis_print(Basis, sz);
				cout << "linear combination:" << endl;
				for (i = 0; i < k2; i++) {
					c = 0;
					for (j = 0; j < sz; j++) {
						a = v[j];
						b = F->mult(a, Basis[j * k2 + i]);
						c = F->add(c, b);
					}
					w[i] = c;
				}
				int_matrix_print(w, k, k);
			}
			break;
			}
		}
	//FREE_int(base_cols);
	//FREE_int(v);
	//FREE_int(w);
	if (f_v) {
		cout << "semifield_classify::test_partial_semifield done" << endl;
		}
	return ret;
}

void semifield_classify::test_rank_unrank()
{
	int *Mtx;
	int r1, r2;
	number_theory_domain NT;

	Mtx = NEW_int(k2);
	for (r1 = 0; r1 < NT.i_power_j(q, k2); r1++) {
		matrix_unrank(r1, Mtx);
		r2 = matrix_rank(Mtx);
		if (r1 != r2) {
			cout << "semifield_classify::test_rank_unrank "
					"r1 != r2" << endl;
			exit(1);
			}
		}
}

void semifield_classify::matrix_unrank(long int rk, int *Mtx)
{
	int i, j, a;

	for (j = k - 1; j >= 0; j--) {
		for (i = k - 1; i >= 0; i--) {
			a = rk % q;
			if (a) {
				Mtx[i * k + j] = 1;
				}
			else {
				Mtx[i * k + j] = 0;
				}
			rk /= q;
			}
		}
}

long int semifield_classify::matrix_rank(int *Mtx)
{
	int i, j;
	long int rk;

	rk = 0;
	for (j = 0; j < k; j++) {
		for (i = 0; i < k; i++) {
			rk *= q;
			rk += Mtx[i * k + j];
			}
		}
	return rk;
}

long int semifield_classify::matrix_rank_without_first_column(int *Mtx)
{
	int i, j;
	long int rk;

	rk = 0;
	for (j = 1; j < k; j++) {
		for (i = 0; i < k; i++) {
			rk *= q;
			rk += Mtx[i * k + j];
			}
		}
	return rk;
}

void semifield_classify::basis_print(int *Mtx, int sz)
{
	int i;
	long int *A;

	cout << "Basis of size " << sz << ":" << endl;
	A = NEW_lint(sz);
	for (i = 0; i < sz; i++) {
		cout << "Elt " << i << ":" << endl;
		int_matrix_print(Mtx + i * k2, k, k);
		A[i] = matrix_rank(Mtx + i * k2);
		}
	lint_vec_print(cout, A, sz);
	cout << endl;
	FREE_lint(A);
}

void semifield_classify::basis_print_numeric(long int *Rk, int sz)
{
	int i;

	cout << "Basis of size " << sz << ":" << endl;
	for (i = 0; i < sz; i++) {
		cout << "Elt " << i << ":" << endl;
		matrix_print_numeric(Rk[i]);
		}
	lint_vec_print(cout, Rk, sz);
	cout << endl;
}

void semifield_classify::matrix_print(int *Mtx)
{
	int_matrix_print(Mtx, k, k);
}

void semifield_classify::matrix_print_numeric(long int rk)
{
	int *Mtx;

	Mtx = NEW_int(k2);
	matrix_unrank(rk, Mtx);
	int_matrix_print(Mtx, k, k);
	FREE_int(Mtx);
}

void semifield_classify::print_set_of_matrices_numeric(
		long int *Rk, int nb)
{
	int *Mtx;
	int i;

	Mtx = NEW_int(k2);
	for (i = 0; i < nb; i++) {
		cout << "Matrix " << i << " / " << nb << " has rank "
				<< Rk[i] << ":" << endl;
		matrix_unrank(Rk[i], Mtx);
		int_matrix_print(Mtx, k, k);
		}
	FREE_int(Mtx);
}

void semifield_classify::apply_element(int *Elt,
	int *basis_in, int *basis_out,
	int first, int last_plus_one, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "semifield_classify::apply_element" << endl;
		}
	for (i = first; i < last_plus_one; i++) {
		A_on_S->compute_image_low_level(Elt,
				basis_in + i * k2,
				basis_out + i * k2,
				0 /* verbose_level */);
		}
	if (f_v) {
		cout << "semifield_classify::apply_element done" << endl;
		}
}

void semifield_classify::apply_element_and_copy_back(int *Elt,
	int *basis_in, int *basis_out,
	int first, int last_plus_one, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_classify::apply_element_and_copy_back" << endl;
		}
	apply_element(Elt,
		basis_in, basis_out,
		first, last_plus_one, verbose_level);
	int_vec_copy(basis_out + first * k2,
			basis_in + first * k2,
			(last_plus_one - first) * k2);
	if (f_v) {
		cout << "semifield_classify::apply_element_and_copy_back done" << endl;
		}
}


int semifield_classify::test_if_third_basis_vector_is_ok(int *Basis)
{
	int *v = test_v;
	int i;

	for (i = 0; i < k; i++) {
		v[i] = Basis[2 * k2 + i * k + 0];
		}
	if (!F->is_unit_vector(v, k, k - 1)) {
		return FALSE;
		}
	return TRUE;
}

void semifield_classify::candidates_classify_by_first_column(
	long int *Input_set, int input_set_sz,
	int window_bottom, int window_size,
	long int **&Set, int *&Set_sz, int &Nb_sets,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *window;
	int *Mtx;
	int *Tmp_sz;
	int h, u, i, t, w;
	long int a;
	number_theory_domain NT;
	geometry_global Gg;


	if (f_v) {
		cout << "semifield_classify::candidates_classify_"
				"by_first_column input_set_sz = " << input_set_sz << endl;
		}
	Nb_sets = NT.i_power_j(q, window_size);
	window = NEW_int(window_size);
	Mtx = NEW_int(k * k);
	Set_sz = NEW_int(Nb_sets);
	Tmp_sz = NEW_int(Nb_sets);
	int_vec_zero(Set_sz, Nb_sets);
	int_vec_zero(Tmp_sz, Nb_sets);
	for (h = 0; h < input_set_sz; h++) {
		if ((h % (256 * 1024)) == 0) {
			cout << "semifield_classify::candidates_classify_"
					"by_first_column " << h << " / "
					<< input_set_sz << endl;
			}
		a = Input_set[h];
		matrix_unrank(a, Mtx);
		for (u = 0; u < window_size; u++) {
			t = Mtx[(window_bottom - u) * k + 0];
			window[u] = t;
			}
		Gg.AG_element_rank(q, window, 1, window_size, w);
		Set_sz[w]++;
		}
	if (f_vv) {
		cout << "semifield_classify::candidates_classify_"
				"by_first_column" << endl;
		cout << "a : #" << endl;
		for (u = 0; u < Nb_sets; u++) {
			cout << u << " : " << Set_sz[u] << endl;
			}
		}

	if (f_vv) {
		cout << "semifield_classify::candidates_classify_"
				"by_first_column computing efficient "
				"representations input_set_sz = " << input_set_sz << endl;
		}
	Set = NEW_plint(Nb_sets);
	for (u = 0; u < Nb_sets; u++) {
		Set[u] = NEW_lint(Set_sz[u]);
		}
	for (h = 0; h < input_set_sz; h++) {
		if ((h % (256 * 1024)) == 0) {
			cout << "semifield_classify::candidates_classify_"
					"by_first_column " << h << " / " << input_set_sz << endl;
			}
		a = Input_set[h];
		matrix_unrank(a, Mtx);
		for (u = 0; u < window_size; u++) {
			t = Mtx[(window_bottom - u) * k + 0];
			window[u] = t;
			}
		Gg.AG_element_rank(q, window, 1, window_size, w);

		// zero out the first column to make it fit into a machine word:

		for (i = 0; i < k; i++) {
			Mtx[i * k + 0] = 0;
			}
		a = matrix_rank(Mtx);


		Set[w][Tmp_sz[w]++] = a; //Input_set[h];
		}
	for (u = 0; u < Nb_sets; u++) {
		if (Tmp_sz[u] != Set_sz[u]) {
			cout << "semifield_classify::candidates_classify_"
					"by_first_column Tmp_sz[u] != Set_sz[u]" << endl;
			exit(1);
			}
		}


	FREE_int(window);
	FREE_int(Mtx);
	FREE_int(Tmp_sz);
	if (f_v) {
		cout << "semifield_classify::candidates_classify_"
				"by_first_column done" << endl;
		}
}

void semifield_classify::make_fname_candidates_at_level_two_orbit(
	char *fname, int orbit)
{
	if (f_level_two_prefix) {
		sprintf(fname, "%sCandidates_lvl_2_orbit_%d_int8.bin",
				level_two_prefix, orbit);
		}
	else {
		sprintf(fname, "Candidates_lvl_2_orbit_%d_int8.bin", orbit);
		}
}

void semifield_classify::make_fname_candidates_at_level_two_orbit_txt(
	char *fname, int orbit)
{
	if (f_level_two_prefix) {
		sprintf(fname, "%sCandidates_lvl_2_orbit_%d",
				level_two_prefix, orbit);
		}
	else {
		sprintf(fname, "Candidates_lvl_2_orbit_%d", orbit);
		}
}

void semifield_classify::make_fname_candidates_at_level_three_orbit(
	char *fname, int orbit)
{
	if (f_level_three_prefix) {
		sprintf(fname, "%sCandidates_lvl_3_orbit_%d",
				level_three_prefix, orbit);
		}
	else {
		sprintf(fname, "Candidates_lvl_3_orbit_%d", orbit);
		}
}

void semifield_classify::make_fname_candidates_at_level_two_orbit_by_type(
	char *fname, int orbit, int h)
{
	if (f_level_two_prefix) {
		sprintf(fname, "%sC2_orbit%d_type%d_int8.bin",
				level_two_prefix, orbit, h);
		}
	else {
		sprintf(fname, "C2_orbit%d_type%d_int8.bin", orbit, h);
		}
}



void semifield_classify::compute_orbit_of_subspaces(
	long int *input_data,
	strong_generators *stabilizer_gens,
	orbit_of_subspaces *&Orb,
	int verbose_level)
// allocates an orbit_of_subspaces data structure in Orb
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_classify::compute_orbit_"
				"of_subspaces" << endl;
		}

	Orb = NEW_OBJECT(orbit_of_subspaces);


	Orb->init_lint(A, AS, F,
		input_data, k, k2 /* n */,
		TRUE /* f_has_desired_pivots */, desired_pivots,
		TRUE /* f_has_rank_functions */, this /* rank_unrank_data */,
		canonial_form_rank_vector_callback,
		canonial_form_unrank_vector_callback,
		canonial_form_compute_image_of_vector_callback,
		this /* compute_image_of_vector_callback_data */,
		stabilizer_gens->gens,
		verbose_level - 1);


	if (f_v) {
		cout << "semifield_classify::compute_orbit_"
				"of_subspaces done" << endl;
		}
}



void semifield_classify::init_desired_pivots(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "semifield_classify::init_desired_pivots" << endl;
		}
	desired_pivots = NEW_int(k);

	for (i = 0; i < k; i++) {
		if (i < 2) {
			desired_pivots[i] = i * k;
			}
		else {
			desired_pivots[i] = (k - 1 - (i - 2)) * k;
			}
		}
	if (f_vv) {
		cout << "semifield_classify::init_desired_pivots "
				"desired_pivots: ";
		int_vec_print(cout, desired_pivots, k);
		cout << endl;
		}
	if (f_v) {
		cout << "semifield_classify::init_desired_pivots done" << endl;
		}
}

void semifield_classify::knuth_operation(int t,
		long int *data_in, long int *data_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a;
	static int perm[] = {
			0,1,2,
			0,2,1,
			1,0,2,
			1,2,0,
			2,0,1,
			2,1,0
	};
	int I[3], J[3];

	if (f_v) {
		cout << "semifield_classify::knuth_operation" << endl;
	}
	for (i = 0; i < k; i++) {
		matrix_unrank(data_in[i], Basis1 + i * k2);
	}
	for (I[0] = 0; I[0] < k; I[0]++) {
		for (I[1] = 0; I[1] < k; I[1]++) {
			for (I[2] = 0; I[2] < k; I[2]++) {
				J[0] = I[perm[t * 3 + 0]];
				J[1] = I[perm[t * 3 + 1]];
				J[2] = I[perm[t * 3 + 2]];
				a = Basis1[J[0] * k2 + J[1] * k + J[2]];
				Basis2[I[0] * k2 + I[1] * k + I[2]] = a;
				}
			}
		}
	for (i = 0; i < k; i++) {
		data_out[i] = matrix_rank(Basis2 + i * k2);
		}
	if (f_v) {
		cout << "semifield_classify::knuth_operation done" << endl;
	}
}


//##############################################################################
// global function:
//##############################################################################

void semifield_classify_early_test_func(int *S, int len,
	int *candidates, int nb_candidates,
	int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	semifield_classify *Semi;
	int f_v = (verbose_level >= 1);

	Semi = (semifield_classify *) data;

	if (f_v) {
		cout << "semifield_classify_early_test_func" << endl;
		cout << "testing " << nb_candidates << " candidates" << endl;
		}

	Semi->early_test_func(S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
		verbose_level);

	if (f_v) {
		cout << "semifield_classify_early_test_func" << endl;
		cout << "Out of " << nb_candidates << " candidates, "
				<< nb_good_candidates << " survive" << endl;
		}
}



int semifield_classify_rank_point_func(int *v, void *data)
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);
	semifield_classify *Semi;
	int rk;

	if (f_v) {
		cout << "semifield_classify_rank_point_func" << endl;
		}
	Semi = (semifield_classify *) data;
	rk = Semi->rank_point(v, verbose_level - 1);
	if (f_v) {
		cout << "semifield_classify_rank_point_func done" << endl;
		}
	return rk;
}

void semifield_classify_unrank_point_func(int *v, int rk, void *data)
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);
	semifield_classify *Semi;

	if (f_v) {
		cout << "semifield_classify_unrank_point_func" << endl;
		}
	Semi = (semifield_classify *) data;

	Semi->unrank_point(v, rk, verbose_level);

	if (f_v) {
		cout << "semifield_classify_unrank_point_func done" << endl;
		}
}


long int canonial_form_rank_vector_callback(int *v,
		int n, void *data, int verbose_level)
{
	semifield_classify *SC = (semifield_classify *) data;
	long int r;

	r = SC->matrix_rank(v);
	return r;
}

void canonial_form_unrank_vector_callback(long int rk,
		int *v, int n, void *data, int verbose_level)
{
	semifield_classify *SC = (semifield_classify *) data;

	SC->matrix_unrank(rk, v);
}

void canonial_form_compute_image_of_vector_callback(
		int *v, int *w, int *Elt, void *data,
		int verbose_level)
{
	semifield_classify *SC = (semifield_classify *) data;


	SC->A_on_S->compute_image_low_level(Elt, v, w,
			0 /* verbose_level */);
}







}}


