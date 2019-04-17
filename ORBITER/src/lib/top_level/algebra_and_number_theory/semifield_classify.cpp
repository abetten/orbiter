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
	null();
}

semifield_classify::~semifield_classify()
{
	freeself();
}

void semifield_classify::null()
{
	k = 0;
	k2 = 0;
	F = NULL;
	q = 0;
	order = 0;

	f_level_three_prefix = FALSE;
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

	//SFS = NULL;

	vector_space_dimension = 0;
	schreier_depth = 0;
}

void semifield_classify::freeself()
{
	if (A0) {
		delete A0;
		}
	if (A0_linear) {
		delete A0_linear;
		}
	if (T) {
		delete T;
		}
	if (Elt1) {
		FREE_int(Elt1);
		}
#if 0
	if (SFS) {
		delete SFS;
		}
#endif
	if (Symmetry_group) {
		delete Symmetry_group;
		}
	if (Poset) {
		FREE_OBJECT(Poset);
	}
	if (Gen) {
		FREE_OBJECT(Gen);
	}
	null();
}

void semifield_classify::init(int argc, const char **argv,
	int order, int n, int k,
	finite_field *F,
	const char *prefix,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object go;
	int i;
	number_theory_domain NT;

	if (f_v) {
		cout << "semifield_classify::init" << endl;
		}

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

	if (f_v) {
		cout << "semifield_classify::init q=" << q << endl;
		cout << "semifield_classify::init k=" << k << endl;
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


#if 0
	Gen->init_vector_space_action(vector_space_dimension,
		T->F,
		semifield_rank_point_func,
		semifield_unrank_point_func,
		this,
		verbose_level - 2);
#endif

#if 0
	Gen->f_print_function = TRUE;
	Gen->print_function = print_set;
	Gen->print_function_data = this;
#endif

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
		cout << "semifield_classify::init done" << endl;
		}
}

void semifield_classify::compute_orbits(int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t0 = os_ticks();

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
	int_matrix_write_csv(fname, Table, nb_orbits, k);

	FREE_int(set);
	FREE_int(Table);
	FREE_int(v);
	}
	cout << "Written file " << fname << " of size "
			<< file_size(fname) << endl;
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
#if 0
	int_vec_copy(v, A_on_S->mtx2, k2);
	r = A_on_S->F->Gauss_easy(A_on_S->mtx2, k, k);
	if (r != k) {
		cout << "semifield_classify::rank_point "
				"r != k" << endl;
		exit(1);
		}
#endif
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

void semifield_classify::matrix_unrank(int rk, int *Mtx)
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

int semifield_classify::matrix_rank(int *Mtx)
{
	int i, j, rk;

	rk = 0;
	for (j = 0; j < k; j++) {
		for (i = 0; i < k; i++) {
			rk *= q;
			rk += Mtx[i * k + j];
			}
		}
	return rk;
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
	M = NEW_int((len + 1) * k2);
	v = NEW_int(len + 1);
	w = NEW_int(k2);
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
				AG_element_unrank(q, v, 1, len, j);
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
	FREE_int(M);
	FREE_int(v);
	FREE_int(w);
	if (f_v) {
		cout << "semifield_classify::early_test_func done" << endl;
		cout << "Out of " << nb_candidates << " candidates, "
				<< nb_good_candidates << " survive" << endl;
		}
}

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

#if 0
void semifield_classify::init_semifield_starter(
		int f_orbits_light, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	SFS = new semifield_starter;

	if (f_v) {
		cout << "semifield_classify::init_semifield_starter "
				"before semifield_starter->init" << endl;
		}
	SFS->init(this, f_orbits_light, verbose_level);
	if (f_v) {
		cout << "semifield_classify::init_semifield_starter "
				"after semifield_starter->init" << endl;
		}
}
#endif




}}


