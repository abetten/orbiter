/*
 * linear_set_classify.cpp
 *
 *  Created on: Oct 28, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {




linear_set_classify::linear_set_classify()
{
	s = n = m = q = Q = depth = 0;
	f_semilinear = FALSE;
	schreier_depth = 0;
	f_use_invariant_subset_if_available = FALSE;
	f_debug = FALSE;
	f_has_extra_test_func = FALSE;
	extra_test_func = NULL;
	extra_test_func_data = NULL;
	Basis = NULL;
	base_cols = NULL;

	Fq = NULL;
	FQ = NULL;
	SubS = NULL;
	P = NULL;
	Aq = NULL;
	AQ = NULL;
	A_PGLQ = NULL;
	VS = NULL;
	Control1 = NULL;
	Poset1 = NULL;
	Gen = NULL;
	vector_space_dimension = 0;
	Strong_gens = NULL;
	D = NULL;
	n1 = 0;
	m1 = 0;
	D1 = NULL;
	spread_embedding = NULL;

	f_identify = FALSE;
	k = 0;
	order = 0;
	T = NULL;

	secondary_level = 0;
	secondary_orbit_at_level = 0;
	secondary_depth = 0;
	secondary_candidates = NULL;
	secondary_nb_candidates = 0;
	secondary_schreier_depth = 0;


	Control_stab = NULL;
	Poset_stab = NULL;
	Gen_stab = NULL;
	Control2 = NULL;
	Poset2 = NULL;
	Gen2 = NULL;
	is_allowed = NULL;

}

linear_set_classify::~linear_set_classify()
{
	freeself();
}

void linear_set_classify::null()
{
}

void linear_set_classify::freeself()
{
	int f_v = FALSE;

	if (VS) {
		if (f_v) {
			cout << "linear_set_classify::freeself before delete VS" << endl;
		}
		FREE_OBJECT(VS);
	}
	if (Poset1) {
		if (f_v) {
			cout << "linear_set_classify::freeself before delete Poset1" << endl;
		}
		FREE_OBJECT(Poset1);
	}
	if (Gen) {
		if (f_v) {
			cout << "linear_set_classify::freeself before delete Gen" << endl;
		}
		FREE_OBJECT(Gen);
	}
	if (Strong_gens) {
		if (f_v) {
			cout << "linear_set_classify::freeself before delete Strong_gens" << endl;
		}
		FREE_OBJECT(Strong_gens);
	}
	if (D) {
		if (f_v) {
			cout << "linear_set_classify::freeself before delete D" << endl;
		}
		FREE_OBJECT(D);
	}
	if (D1) {
		if (f_v) {
			cout << "linear_set_classify::freeself before delete D1" << endl;
		}
		FREE_OBJECT(D1);
	}
	if (spread_embedding) {
		if (f_v) {
			cout << "linear_set_classify::freeself before delete spread_embedding" << endl;
		}
		FREE_OBJECT(spread_embedding);
	}
	if (P) {
		if (f_v) {
			cout << "linear_set_classify::freeself before delete P" << endl;
		}
		FREE_OBJECT(P);
	}
	if (Aq) {
		if (f_v) {
			cout << "linear_set_classify::freeself before delete Aq" << endl;
		}
		FREE_OBJECT(Aq);
	}
	if (AQ) {
		if (f_v) {
			cout << "linear_set_classify::freeself before delete AQ" << endl;
		}
		FREE_OBJECT(AQ);
	}
	if (A_PGLQ) {
		if (f_v) {
			cout << "linear_set_classify::freeself before delete A_PGLQ" << endl;
		}
		FREE_OBJECT(A_PGLQ);
	}
	if (SubS) {
		if (f_v) {
			cout << "linear_set_classify::freeself before delete SubS" << endl;
		}
		FREE_OBJECT(SubS);
	}
	if (Fq) {
		if (f_v) {
			cout << "linear_set_classify::freeself before delete Fq" << endl;
		}
		FREE_OBJECT(Fq);
	}
	if (FQ) {
		if (f_v) {
			cout << "linear_set_classify::freeself before delete FQ" << endl;
		}
		FREE_OBJECT(FQ);
	}
	if (Basis) {
		FREE_int(Basis);
	}
	if (base_cols) {
		FREE_int(base_cols);
	}
	if (Poset2) {
		FREE_OBJECT(Poset2);
	}
	if (Gen2) {
		FREE_OBJECT(Gen2);
	}
	if (is_allowed) {
		FREE_int(is_allowed);
	}
	if (T) {
		FREE_OBJECT(T);
	}
	null();
}

void linear_set_classify::init(
	int s, int n, int q, const char *poly_q, const char *poly_Q,
	int depth, int f_identify, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	number_theory_domain NT;

	if (f_v) {
		cout << "linear_set_classify::init" << endl;
	}

	linear_set_classify::s = s;
	linear_set_classify::n = n;
	linear_set_classify::q = q;
	linear_set_classify::depth = depth;
	linear_set_classify::f_identify = f_identify;
	if (f_v) {
		cout << "linear_set_classify::init s=" << s << endl;
		cout << "linear_set_classify::init n=" << n << endl;
		cout << "linear_set_classify::init q=" << q << endl;
		cout << "linear_set_classify::init depth=" << depth << endl;
		cout << "linear_set_classify::init f_identify=" << f_identify << endl;
	}


	Q = NT.i_power_j(q, s);
	m = n / s;
	if (m * s != n) {
		cout << "linear_set_classify::init s must divide n" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "linear_set_classify::init m=" << m << endl;
		cout << "linear_set_classify::init Q=" << Q << endl;
	}

	vector_space_dimension = n;
	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}

	Fq = NEW_OBJECT(finite_field);
	if (f_v) {
		cout << "linear_set_classify::init before Fq->init" << endl;
	}
	Fq->init_override_polynomial(q, poly_q, 0);

	FQ = NEW_OBJECT(finite_field);
	if (f_v) {
		cout << "linear_set_classify::init before FQ->init" << endl;
	}
	FQ->init_override_polynomial(Q, poly_Q, 0);

	SubS = NEW_OBJECT(subfield_structure);
	if (f_v) {
		cout << "linear_set_classify::init before SubS->init" << endl;
	}
	SubS->init(FQ, Fq, verbose_level);

	if (f_v) {
		cout << "Field-basis: ";
		int_vec_print(cout, SubS->Basis, s);
		cout << endl;
	}


	P = NEW_OBJECT(projective_space);
	if (f_v) {
		cout << "linear_set_classify::init before P->init" << endl;
	}
	P->init(n - 1, Fq,
		FALSE /* f_init_incidence_structure */,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "linear_set_classify::init after P->init" << endl;
	}



	if (f_v) {
		cout << "linear_set_classify::init before init_general_linear_group "
				"GL(" << n << "," << Fq->q << ")" << endl;
	}

	vector_ge *nice_gens;

	Aq = NEW_OBJECT(action);
	Aq->init_general_linear_group(n, Fq,
		FALSE /* f_semilinear */,
		TRUE /* f_basis */, TRUE /* f_init_sims */,
		nice_gens,
		verbose_level - 2);
	FREE_OBJECT(nice_gens);

	if (f_v) {
		cout << "linear_set_classify::init after init_general_linear_group "
				"GL(" << n << "," << Fq->q << ")" << endl;
	}


	AQ = NEW_OBJECT(action);

	if (f_v) {
		cout << "linear_set_classify::init before init_general_linear_group "
				"GL(" << m << "," << FQ->q << ")" << endl;
	}
	AQ->init_general_linear_group(m, FQ,
		FALSE /* f_semilinear */,
		TRUE /* f_basis */, TRUE /* f_init_sims */,
		nice_gens,
		verbose_level - 2);
	FREE_OBJECT(nice_gens);
	if (f_v) {
		cout << "linear_set_classify::init after init_general_linear_group "
				"GL(" << m << "," << FQ->q << ")" << endl;
	}

	if (f_vv) {
		cout << "Strong generators are:" << endl;
		AQ->Strong_gens->print_generators(cout);
		AQ->Strong_gens->print_generators_tex();
	}


	A_PGLQ = NEW_OBJECT(action);
	if (f_v) {
		cout << "linear_set_classify::init before init_projective_group "
				"PGL(" << m << "," << FQ->q << ")" << endl;
	}
	A_PGLQ->init_projective_group(m, FQ,
		FALSE /* f_semilinear */,
		TRUE /* f_basis */, TRUE /* f_init_sims */,
		nice_gens,
		verbose_level - 2);
	FREE_OBJECT(nice_gens);
	if (f_v) {
		cout << "linear_set_classify::init after init_projective_group "
				"PGL(" << m << "," << FQ->q << ")" << endl;
	}


	if (f_v) {
		cout << "linear_set_classify::init before linear_set_lift_generators_"
				"to_subfield_structure" << endl;
	}

	action_global AG;

	AG.lift_generators_to_subfield_structure(n, s,
		SubS, Aq, AQ, Strong_gens,
		verbose_level);

	if (f_v) {
		cout << "linear_set_classify::init after linear_set_lift_generators_"
				"to_subfield_structure" << endl;
	}

	if (f_v) {
		cout << "After lift, strong generators are:" << endl;
		Strong_gens->print_generators(cout);
		Strong_gens->print_generators_tex();
	}



	Basis = NEW_int(depth * vector_space_dimension);
	base_cols = NEW_int(vector_space_dimension);


	D = NEW_OBJECT(desarguesian_spread);
	if (f_v) {
		cout << "linear_set_classify::init before D->init" << endl;
	}
	D->init(n, m, s,
		SubS,
		verbose_level);
	if (f_v) {
		cout << "linear_set_classify::init after D->init" << endl;
	}

	m1 = m + 1;
	n1 = s * m1; // = n + s
	D1 = NEW_OBJECT(desarguesian_spread);
	if (f_v) {
		cout << "linear_set_classify::init before D1->init" << endl;
	}
	D1->init(n1, m1, s,
		SubS,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "linear_set_classify::init after D1->init" << endl;
	}

	int *vec;
	int i, j;

	if (f_v) {
		cout << "linear_set_classify::init computing spread_embedding" << endl;
	}
	spread_embedding = NEW_int(D->N);
	vec = NEW_int(m1);
	for (i = 0; i < D->N; i++) {
		FQ->PG_element_unrank_modified(vec, 1, m, i);
		vec[m] = 0;
		FQ->PG_element_rank_modified(vec, 1, m1, j);
		spread_embedding[i] = j;
	}

	FREE_int(vec);
	if (f_v) {
		cout << "linear_set_classify::init computing spread_embedding done" << endl;
	}

	VS = NEW_OBJECT(vector_space);
	VS->init(P->F, vector_space_dimension /* dimension */,
			verbose_level - 1);
	VS->init_rank_functions(
			linear_set_classify_rank_point_func,
			linear_set_classify_unrank_point_func,
			this,
			verbose_level - 1);


	Control1 = NEW_OBJECT(poset_classification_control);
	Poset1 = NEW_OBJECT(poset);
	Gen = NEW_OBJECT(poset_classification);



	Control1->f_depth = TRUE;
	Control1->depth = depth;

	Poset1->init_subspace_lattice(Aq, Aq, Strong_gens, VS,
			verbose_level);


	if (f_v) {
		cout << "linear_set_classify::init before Gen->init" << endl;
	}
	Gen->initialize_and_allocate_root_node(Control1, Poset1, depth /* sz */, verbose_level);
	if (f_v) {
		cout << "linear_set_classify::init after Gen->init" << endl;
	}


	schreier_depth = depth;
	f_use_invariant_subset_if_available = TRUE;
	f_debug = FALSE;


	if (f_identify) {
		T = NEW_OBJECT(spread_classify);

		//int f_recoordinatize = TRUE;

		k = n >> 1;
		order = NT.i_power_j(q, k);

		if (f_v) {
			cout << "Classifying spreads of order " << order << endl;
		}

		//int max_depth = order + 1;
		poset_classification_control *Control;
		linear_group *LG;

		Control = NEW_OBJECT(poset_classification_control);
		LG = NEW_OBJECT(linear_group); // hack !!! ToDo

		T->init(LG, k, Control,
			MINIMUM(verbose_level - 1, 2));

#if 0
		T->init(order, n, k, max_depth,
			Fq, f_recoordinatize,
			"SPREADS_STARTER", "Spreads", order + 1,
			argc, argv,
			MINIMUM(verbose_level - 1, 2));
#endif

		//T->read_arguments(argc, argv);

		//T->init2(Control, verbose_level);

		if (f_v) {
			cout << "Classifying spreads planes of order "
					<< order << ":" << endl;
		}
		T->compute(0 /*verbose_level*/);
		if (f_v) {
			cout << "Spreads of order " << order
					<< " have been classified" << endl;
		}
	}

	if (f_v) {
		cout << "linear_set_classify::init done" << endl;
	}
}

void linear_set_classify::do_classify(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_set_classify::do_classify" << endl;
	}

	//int t0 = os_ticks();

	if (f_v) {
		cout << "linear_set_classify::do_classify calling generator_main" << endl;
		cout << "A=";
		Gen->get_A()->print_info();
		cout << "A2=";
		Gen->get_A2()->print_info();
	}

	Gen->compute_orbits(0, depth,
		verbose_level);

#if 0
	Gen->main(t0,
		schreier_depth,
		f_use_invariant_subset_if_available,
		f_lex,
		f_debug,
		verbose_level - 1);
#endif

	int nb_orbits;

	if (f_v) {
		cout << "linear_set_classify::do_classify done with generator_main" << endl;
	}
	nb_orbits = Gen->nb_orbits_at_level(depth);
	if (f_v) {
		cout << "linear_set_classify::do_classify we found " << nb_orbits
				<< " orbits at depth " << depth<< endl;
	}


	if (f_v) {
		cout << "linear_set_classify::do_classify done" << endl;
	}
}

int linear_set_classify::test_set(int len, long int *S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret = TRUE;
	int i, rk;

	if (f_v) {
		cout << "linear_set_classify::test_set" << endl;
		cout << "Testing set ";
		lint_vec_print(cout, S, len);
		cout << endl;
	}
	for (i = 0; i < len; i++) {
		Fq->PG_element_unrank_modified(
			Basis + i * vector_space_dimension, 1,
			vector_space_dimension, S[i]);
	}
	if (f_vv) {
		cout << "coordinate matrix:" << endl;
		print_integer_matrix_width(cout, Basis,
			len, vector_space_dimension, vector_space_dimension,
			Fq->log10_of_q);
	}
	rk = Fq->Gauss_simple(Basis, len, vector_space_dimension,
			base_cols, 0 /*verbose_level - 2*/);
	if (f_v) {
		cout << "the matrix has rank " << rk << endl;
	}
	if (rk < len) {
		ret = FALSE;
	}
	if (ret) {
		if (f_has_extra_test_func) {
			ret = (*extra_test_func)(this, len, S,
				extra_test_func_data, verbose_level);
		}
	}

	if (ret) {
		if (f_v) {
			cout << "OK" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "not OK" << endl;
		}
	}
	return ret;
}

void linear_set_classify::compute_intersection_types_at_level(int level,
	int &nb_nodes, int *&Intersection_dimensions, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int node, i;
	long int *set;

	if (f_v) {
		cout << "linear_set_classify::compute_intersection_types_at_level" << endl;
	}

	set = NEW_lint(level);

	nb_nodes = Gen->nb_orbits_at_level(level);
	Intersection_dimensions = NEW_int(nb_nodes * D->N);
	for (node = 0; node < nb_nodes; node++) {
		Gen->get_set_by_level(level, node, set);
		for (i = 0; i < level; i++) {
			Fq->PG_element_unrank_modified(Basis + i * n, 1, n, set[i]);
		}
		D->compute_intersection_type(level, Basis,
			Intersection_dimensions + node * D->N, 0 /*verbose_level - 1*/);
	}


	FREE_lint(set);

	if (f_v) {
		cout << "linear_set_classify::compute_intersection_types_at_level done" << endl;
	}
}

void linear_set_classify::calculate_intersections(int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_set_classify::calculate_intersections" << endl;
	}

	int level;
	int *Nb_nodes;
	int **Intersection_dimensions;
	long int ***Sets;
	int **Set_sz;
	longinteger_object go;
	int i, h, j;

	Nb_nodes = NEW_int(depth + 1);
	Intersection_dimensions = NEW_pint(depth + 1);
	Sets = NEW_pplint(depth + 1);
	Set_sz = NEW_pint(depth + 1);

	for (level = 0; level <= depth; level++) {
		cout << "Computing intersection types at level " << level
				<< " / " << depth << ":" << endl;
		compute_intersection_types_at_level(level,
			Nb_nodes[level], Intersection_dimensions[level],
			verbose_level - 1);
		cout << "nb_nodes=" << Nb_nodes[level] << endl;
	}
	for (level = 0; level <= depth; level++) {
		cout << "Intersection types at level " << level << " / "
				<< depth << " with " << Nb_nodes[level] << " orbits:" << endl;
		for (i = 0; i < Nb_nodes[level]; i++) {
			cout << setw(3) << i << " : ";
			int_vec_print(cout,
					Intersection_dimensions[level] + i * D->N, D->N);
			cout << " : ";
			{
				tally C;

				C.init(Intersection_dimensions[level] + i * D->N, D->N, FALSE, 0);
				C.print_naked(TRUE);
			}
			cout << " : ";
			Gen->get_stabilizer_order(level, i, go);
			cout << go;
			cout << endl;
		}
		//int_matrix_print(Intersection_dimensions[level],
		// Nb_nodes[level], LS->D->N);
	}
	for (level = 0; level <= depth; level++) {
		cout << "Level " << level << ":" << endl;
		Sets[level] = NEW_plint(Nb_nodes[level]);
		Set_sz[level] = NEW_int(Nb_nodes[level]);
		for (h = 0; h < Nb_nodes[level]; h++) {
			int *I;

			I = Intersection_dimensions[level] + h * D->N;
			Set_sz[level][h] = 0;
			for (i = 0; i < D->N; i++) {
				if (I[i]) {
					Set_sz[level][h]++;
				}
			}
			Sets[level][h] = NEW_lint(Set_sz[level][h]);
			j = 0;
			for (i = 0; i < D->N; i++) {
				if (I[i]) {
					Sets[level][h][j++] = i;
				}
			}
			cout << h << " : ";
			lint_vec_print(cout, Sets[level][h], Set_sz[level][h]);
			cout << endl;
		}
	}


	for (level = 0; level <= depth; level++) {
		for (h = 0; h < Nb_nodes[level]; h++) {
			FREE_lint(Sets[level][h]);
		}
		FREE_plint(Sets[level]);
		FREE_int(Set_sz[level]);
		FREE_int(Intersection_dimensions[level]);
	}
	FREE_pplint(Sets);
	FREE_pint(Set_sz);
	FREE_pint(Intersection_dimensions);
	FREE_int(Nb_nodes);

	if (f_v) {
		cout << "linear_set_classify::calculate_intersections done" << endl;
	}
}

void linear_set_classify::read_data_file(int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int depth_completed;
	string fname;
	char str[1000];

	if (f_v) {
		cout << "linear_set_classify::read_data_file" << endl;
	}
	fname.assign(Gen->get_problem_label_with_path());
	sprintf(str, "_%d.data", depth);
	fname.append(str);

	Gen->read_data_file(depth_completed, fname, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "linear_set_classify::read_data_file after read_data_file" << endl;
	}

	int level;
	string prefix;


	prefix.assign(Gen->get_problem_label_with_path());
	prefix.append("b");
	for (level = 0; level < depth; level++) {
		if (f_v) {
			cout << "linear_set_classify::read_data_file before "
					"read_sv_level_file_binary level=" << level << endl;
		}
		Gen->read_sv_level_file_binary(level, prefix,
			FALSE /* f_split */, 0 /* split_mod */, 0 /*split_case*/,
			FALSE /*f_recreate_extensions*/, FALSE /* f_dont_keep_sv */,
			verbose_level - 2);
	}

#if 0
	cout << "before print_tree" << endl;
	Gen->print_tree();

	cout << "before draw_poset" << endl;
	Gen->draw_poset("test",
			depth, 0 /* data1 */,
			TRUE /* f_embedded */,
			10 /* gen->verbose_level */);
#endif

	if (f_v) {
		cout << "linear_set_classify::read_data_file done" << endl;
	}
}


void linear_set_classify::print_orbits_at_level(int level)
{
	int len, orbit_at_level, i;
	longinteger_object go;
	long int *set;
	int *Basis;

	set = NEW_lint(level);
	Basis = NEW_int(level * n);

	len = Gen->nb_orbits_at_level(level);
	for (orbit_at_level = 0; orbit_at_level < len; orbit_at_level++) {
		Gen->get_set_by_level(level, orbit_at_level, set);
		for (i = 0; i < level; i++) {
			Fq->PG_element_unrank_modified(Basis + i * n, 1, n, set[i]);
		}
		Gen->get_stabilizer_order(level, orbit_at_level, go);
		cout << "orbit " << orbit_at_level << " / " << len
				<< " stabilizer order " << go << ":" << endl;
		cout << "set: ";
		lint_vec_print(cout, set, level);
		cout << endl;
		cout << "Basis:" << endl;
		int_matrix_print(Basis, level, n);
	}

	FREE_lint(set);
	FREE_int(Basis);

}

void linear_set_classify::classify_secondary(int argc, const char **argv,
	int level, int orbit_at_level,
	strong_generators *strong_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, nb_allowed;
	long int *set;

	if (f_v) {
		cout << "linear_set_classify::classify_secondary" << endl;
	}

	secondary_level = level;
	secondary_orbit_at_level = orbit_at_level;

	set = NEW_lint(level);
	is_allowed = NEW_int(Aq->degree);

	Gen->get_set_by_level(level, orbit_at_level, set);
	for (i = 0; i < level; i++) {
		Fq->PG_element_unrank_modified(Basis + i * n, 1, n, set[i]);
	}
	cout << "set: ";
	lint_vec_print(cout, set, level);
	cout << endl;
	cout << "Basis:" << endl;
	int_matrix_print(Basis, level, n);



	D->compute_shadow(Basis, level, is_allowed, verbose_level - 1);
	for (i = 0; i < Aq->degree; i++) {
		is_allowed[i] = !is_allowed[i];
	}


	nb_allowed = 0;
	for (i = 0; i < Aq->degree; i++) {
		if (is_allowed[i]) {
			nb_allowed++;
		}
	}

	cout << "degree=" << Aq->degree << endl;
	cout << "nb_allowed=" << nb_allowed << endl;


#if 0
	int *candidates;
	int nb_candidates;
	int a;
	char fname_candidates[1000];

	Gen->make_fname_candidates_file_default(fname_candidates, level);

	cout << "reading file " << fname_candidates << endl;

	generator_read_candidates_of_orbit(fname_candidates, orbit_at_level,
		candidates, nb_candidates, verbose_level);

	int *good_candidates;
	int nb_good_candidates;

	good_candidates = NEW_int(nb_candidates);
	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		a = candidates[i];
		if (is_allowed[a]) {
			good_candidates[nb_good_candidates++] = a;
			}
		}
	cout << "Out of " << nb_candidates << " candidates, "
			<< nb_good_candidates << " survive" << endl;

	int *good_candidates;
	int nb_good_candidates;

	good_candidates = NEW_int(nb_candidates);
	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		a = candidates[i];
		if (is_allowed[a]) {
			good_candidates[nb_good_candidates++] = a;
			}
		}

#endif

	long int *candidates;
	int nb_candidates;

	candidates = NEW_lint(nb_allowed);
	nb_candidates = 0;
	for (i = 0; i < Aq->degree; i++) {
		if (is_allowed[i]) {
			candidates[nb_candidates++] = i;
		}
	}
	cout << "candidates:" << nb_candidates << endl;
	lint_vec_print(cout, candidates, nb_candidates);
	cout << endl;


#if 0
	strong_generators *Strong_gens_previous;

	Gen->get_stabilizer_generators(Strong_gens_previous,
		level, orbit_at_level, verbose_level);
#endif

	init_secondary(argc, argv,
		candidates, nb_candidates,
		strong_gens /* Strong_gens_previous*/,
		verbose_level);



	FREE_lint(set);
	//FREE_int(is_allowed);
		// don't free is_allowed,
		// it is part of linear_set now.
	FREE_lint(candidates);
}

void linear_set_classify::init_secondary(int argc, const char **argv,
	long int *candidates, int nb_candidates,
	strong_generators *Strong_gens_previous,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_set_classify::init_secondary" << endl;
	}

	secondary_candidates = candidates;
	secondary_nb_candidates = nb_candidates;

	Control2 = NEW_OBJECT(poset_classification_control);
	Poset2 = NEW_OBJECT(poset);
	Gen2 = NEW_OBJECT(poset_classification);

	secondary_depth = n - secondary_level;

	Control2->f_depth = TRUE;
	Control2->depth = secondary_depth;


	char label[1000]; // ToDo

	sprintf(label, "subspaces_%d_%d_%d_secondary_%d_%d", n, q, s,
		secondary_level, secondary_orbit_at_level);


	if (f_v) {
		cout << "linear_set_classify::init_secondary "
				"secondary_level = " << secondary_level << endl;
		cout << "linear_set_classify::init_secondary "
				"secondary_depth = " << secondary_depth << endl;
	}


	if (f_v) {
		cout << "linear_set_classify::init_secondary generators are:" << endl;
		Strong_gens_previous->print_generators(cout);
		//Strong_gens_previous->print_generators_as_permutations();
	}

	cout << "linear_set_classify::init_secondary before Gen2->initialize_and_allocate_root_node" << endl;
	Poset2->init_subspace_lattice(Aq, Aq,
			Strong_gens_previous, VS,
			verbose_level);
	Gen2->initialize_and_allocate_root_node(Control2, Poset2,
			secondary_depth /* sz */,
			verbose_level);
	cout << "linear_set_classify::init_secondary after Gen2->initialize_and_allocate_root_node" << endl;


#if 0
	// ToDo
	Gen2->init_early_test_func(
			linear_set_classify_secondary_early_test_func,
		this /*void *data */,
		verbose_level);
#endif


	secondary_schreier_depth = secondary_depth;
	//f_use_invariant_subset_if_available = TRUE;
	//f_lex = FALSE;
	//f_debug = FALSE;



	// the following works only for actions on subsets:
#if 0
	if (f_v) {
		cout << "linear_set_classify::init_secondary before "
				"Gen2->init_root_node_invariant_subset" << endl;
	}
	Gen2->init_root_node_invariant_subset(
		secondary_candidates, secondary_nb_candidates, verbose_level);
	if (f_v) {
		cout << "linear_set_classify::init_secondary after "
				"Gen2->init_root_node_invariant_subset" << endl;
	}
#endif

	if (f_v) {
		cout << "linear_set_classify::init_secondary before "
				"do_classify_secondary" << endl;
	}
	do_classify_secondary(verbose_level);
	if (f_v) {
		cout << "linear_set_classify::init_secondary after "
				"do_classify_secondary" << endl;
	}
	if (f_v) {
		cout << "linear_set_classify::init_secondary done" << endl;
	}

}

void linear_set_classify::do_classify_secondary(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	os_interface Os;

	if (f_v) {
		cout << "linear_set_classify::do_classify_secondary" << endl;
	}

	int t0 = Os.os_ticks();

	if (f_v) {
		cout << "linear_set_classify::do_classify_secondary "
				"calling generator_main" << endl;
		cout << "A=";
		Gen2->get_A()->print_info();
		cout << "A2=";
		Gen2->get_A2()->print_info();
	}
	Gen2->main(t0,
		secondary_schreier_depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level - 1);

	int nb_orbits;

	if (f_v) {
		cout << "linear_set_classify::do_classify_secondary "
				"done with generator_main" << endl;
	}
	nb_orbits = Gen2->nb_orbits_at_level(secondary_depth);
	if (f_v) {
		cout << "linear_set_classify::do_classify_secondary we found "
				<< nb_orbits << " orbits at depth " << secondary_depth<< endl;
	}

	int h, i;

	long int *set1;
	long int *set2;
	int *Basis1;
	int *Basis2;

	set1 = NEW_lint(secondary_level);
	set2 = NEW_lint(secondary_depth);
	Basis1 = NEW_int(secondary_level * n);
	Basis2 = NEW_int(secondary_depth * n);

	Gen->get_set_by_level(secondary_level, secondary_orbit_at_level, set1);
	for (i = 0; i < secondary_level; i++) {
		Fq->PG_element_unrank_modified(Basis1 + i * n, 1, n, set1[i]);
	}
	cout << "set1: ";
	lint_vec_print(cout, set1, secondary_level);
	cout << endl;
	cout << "Basis1:" << endl;
	int_matrix_print(Basis1, secondary_level, n);


	int *Intersection_dimensions;

	Intersection_dimensions = NEW_int(D->N);

	for (h = 0; h < nb_orbits; h++) {
		cout << "Orbit " << h << " / " << nb_orbits << ":" << endl;
		Gen2->get_set_by_level(secondary_depth, h, set2);
		for (i = 0; i < secondary_depth; i++) {
			Fq->PG_element_unrank_modified(Basis2 + i * n, 1, n, set2[i]);
		}
		cout << "set2: ";
		lint_vec_print(cout, set2, secondary_depth);
		cout << endl;
		cout << "Basis2:" << endl;
		int_matrix_print(Basis2, secondary_depth, n);


		D->compute_intersection_type(secondary_depth, Basis2,
			Intersection_dimensions, 0 /*verbose_level - 1*/);

		cout << "Intersection_dimensions:";
		int_vec_print(cout, Intersection_dimensions, D->N);
		cout << endl;

		strong_generators *Strong_gens2;
		longinteger_object go;

		Gen2->get_stabilizer_generators(Strong_gens2,
			secondary_depth, h, 0 /*verbose_level*/);

		Strong_gens2->group_order(go);

		cout << "The stabilizer has order " << go
				<< " and is generated by:" << endl;
		Strong_gens2->print_generators(cout);

		FREE_OBJECT(Strong_gens2);
	}

	FREE_int(Intersection_dimensions);


	if (f_v) {
		cout << "linear_set_classify::do_classify_secondary done" << endl;
	}
}

int linear_set_classify::test_set_secondary(int len, long int *S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret = TRUE;
	int i, rk;
	int *v;
	int *w;
	int nb;

	if (f_v) {
		cout << "linear_set_classify::test_set_secondary" << endl;
		cout << "Testing set ";
		lint_vec_print(cout, S, len);
		cout << endl;
	}
	for (i = 0; i < len; i++) {
		Fq->PG_element_unrank_modified(
				Basis + i * vector_space_dimension, 1,
				vector_space_dimension, S[i]);
	}

	if (f_vv) {
		cout << "coordinate matrix:" << endl;
		print_integer_matrix_width(cout,
				Basis, len, vector_space_dimension,
				vector_space_dimension, Fq->log10_of_q);
	}

	rk = Fq->Gauss_simple(Basis, len,
			vector_space_dimension, base_cols,
			0 /*verbose_level - 2*/);
	if (f_v) {
		cout << "the matrix has rank " << rk << endl;
	}
	if (rk < len) {
		ret = FALSE;
	}

	if (ret) {
		// need to make sure that the whole space
		// consists of allowable vectors:

		geometry_global Gg;

		v = NEW_int(len);
		w = NEW_int(n);
		nb = Gg.nb_PG_elements(len - 1, q);


		for (i = 0; i < nb; i++) {
			Fq->PG_element_unrank_modified(v, 1, len, i);
			Fq->mult_vector_from_the_left(v, Basis, w, len, n);
			Fq->PG_element_rank_modified(w, 1, n, rk);
			if (is_allowed[rk] == FALSE) {
				ret = FALSE;
				break;
			}
		}

		FREE_int(v);
		FREE_int(w);
	}

	if (ret) {
		if (f_has_extra_test_func) {
			ret = (*extra_test_func)(this, len, S,
					extra_test_func_data, verbose_level);
		}
	}

	if (ret) {
		if (f_v) {
			cout << "OK" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "not OK" << endl;
		}
	}
	return ret;
}

void linear_set_classify::compute_stabilizer_of_linear_set(
	int argc, const char **argv,
	int level, int orbit_at_level,
	strong_generators *&strong_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, nb_allowed;
	long int *set;

	if (f_v) {
		cout << "linear_set_classify::compute_stabilizer_of_linear_set" << endl;
	}

	set = NEW_lint(level);
	is_allowed = NEW_int(Aq->degree);

	Gen->get_set_by_level(level, orbit_at_level, set);
	for (i = 0; i < level; i++) {
		Fq->PG_element_unrank_modified(Basis + i * n, 1, n, set[i]);
	}
	cout << "set: ";
	lint_vec_print(cout, set, level);
	cout << endl;
	cout << "Basis:" << endl;
	int_matrix_print(Basis, level, n);



	D->compute_shadow(Basis, level, is_allowed, verbose_level - 1);

#if 0
	for (i = 0; i < Aq->degree; i++) {
		is_allowed[i] = !is_allowed[i];
	}
#endif

	nb_allowed = 0;
	for (i = 0; i < Aq->degree; i++) {
		if (is_allowed[i]) {
			nb_allowed++;
		}
	}

	cout << "degree=" << Aq->degree << endl;
	cout << "nb_allowed=" << nb_allowed << endl;



	long int *candidates;
	int nb_candidates;

	candidates = NEW_lint(nb_allowed);
	nb_candidates = 0;
	for (i = 0; i < Aq->degree; i++) {
		if (is_allowed[i]) {
			candidates[nb_candidates++] = i;
		}
	}
	cout << "candidates:" << nb_candidates << endl;
	lint_vec_print(cout, candidates, nb_candidates);
	cout << endl;


	strong_generators *Strong_gens_previous;

	Gen->get_stabilizer_generators(Strong_gens_previous,
		level, orbit_at_level, verbose_level);


	init_compute_stabilizer(argc, argv,
		level, orbit_at_level,
		candidates, nb_candidates,
		Strong_gens_previous,
		strong_gens,
		verbose_level);


	FREE_lint(set);
	//FREE_int(is_allowed);
	// don't free is_allowed,
	//it is part of linear_set now.
	FREE_lint(candidates);
}

void linear_set_classify::init_compute_stabilizer(int argc, const char **argv,
	int level, int orbit_at_level,
	long int *candidates, int nb_candidates,
	strong_generators *Strong_gens_previous,
	strong_generators *&strong_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_set_classify::init_compute_stabilizer" << endl;
	}

	Control_stab = NEW_OBJECT(poset_classification_control);
	Poset_stab = NEW_OBJECT(poset);


	Control_stab->f_depth = TRUE;
	Control_stab->depth = level;


	char label[1000];

	sprintf(label,
		"subspaces_%d_%d_%d_stabilizer_%d_%d", n, q, s,
		level, orbit_at_level);


	if (f_v) {
		cout << "linear_set_classify::init_compute_stabilizer "
				"depth = " << level << endl;
	}


	if (f_v) {
		cout << "linear_set_classify::init_compute_stabilizer "
				"generators are:" << endl;
		Strong_gens_previous->print_generators(cout);
		//Strong_gens_previous->print_generators_as_permutations();
	}

	cout << "linear_set_classify::init_compute_stabilizer "
			"before Gen_stab->init" << endl;
	Poset_stab->init_subspace_lattice(Aq, Aq,
			Strong_gens_previous, VS,
			verbose_level);

	Gen_stab = NEW_OBJECT(poset_classification);

	Gen_stab->initialize_and_allocate_root_node(Control_stab,
			Poset_stab,
			level /* sz */,
			verbose_level);
	cout << "linear_set_classify::init_compute_stabilizer "
			"after Gen_stab->init" << endl;


#if 0
	Gen_stab->init_check_func(
		subspace_orbits_test_func,
		this /* candidate_check_data */);
#endif


#if 0
	// ToDo
	Gen_stab->init_early_test_func(
			linear_set_classify_secondary_early_test_func,
		this /*void *data */,
		verbose_level);
#endif

		// we can use the same test function:
		// test if the whole subspace consists of allowed vectors

	//Gen_stab->init_incremental_check_func(
		//check_mindist_incremental,
		//this /* candidate_check_data */);

#if 0
	Gen_stab->init_vector_space_action(vector_space_dimension,
		P->F,
		linear_set_rank_point_func,
		linear_set_unrank_point_func,
		this,
		verbose_level);
#endif
#if 0
	Gen->f_print_function = TRUE;
	Gen->print_function = print_set;
	Gen->print_function_data = this;
#endif



	// the following works only for actions on subsets:
#if 0
	if (f_v) {
		cout << "linear_set_classify::init_secondary before "
				"Gen_stab->init_root_node_invariant_subset" << endl;
	}
	Gen_stab->init_root_node_invariant_subset(
		secondary_candidates, secondary_nb_candidates, verbose_level);
	if (f_v) {
		cout << "linear_set_classify::init_secondary after "
				"Gen_stab->init_root_node_invariant_subset" << endl;
	}
#endif

	if (f_v) {
		cout << "linear_set_classify::init_compute_stabilizer "
				"before do_compute_stabilizer" << endl;
	}
	do_compute_stabilizer(level, orbit_at_level,
		candidates, nb_candidates,
		strong_gens,
		verbose_level);
	if (f_v) {
		cout << "linear_set_classify::init_compute_stabilizer "
				"after do_compute_stabilizer" << endl;
	}
	if (f_v) {
		cout << "linear_set_classify::init_compute_stabilizer done" << endl;
	}

}

void linear_set_classify::do_compute_stabilizer(
	int level, int orbit_at_level,
	long int *candidates, int nb_candidates,
	strong_generators *&strong_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	os_interface Os;

	if (f_v) {
		cout << "linear_set_classify::do_compute_stabilizer" << endl;
	}

	int t0 = Os.os_ticks();

	if (f_v) {
		cout << "linear_set_classify::do_compute_stabilizer "
				"calling generator_main" << endl;
		cout << "A=";
		Gen_stab->get_A()->print_info();
		cout << "A2=";
		Gen_stab->get_A2()->print_info();
	}
	Gen_stab->main(t0,
			level,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level - 1);

	int nb_orbits;

	if (f_v) {
		cout << "linear_set_classify::do_compute_stabilizer "
				"done with generator_main" << endl;
	}
	nb_orbits = Gen_stab->nb_orbits_at_level(level);
	if (f_v) {
		cout << "linear_set_classify::do_compute_stabilizer we found "
				<< nb_orbits << " orbits at depth "
				<< level << endl;
	}

	long int *set1;
	long int *set2;
	long int *set3;
	int *Basis1;
	int *Basis2;
	int i, h, orbit;

	set1 = NEW_lint(level);
	set2 = NEW_lint(level);
	set3 = NEW_lint(level);
	Basis1 = NEW_int(level * n);
	Basis2 = NEW_int(level * n);


	Gen->get_set_by_level(level, orbit_at_level, set1);
	for (i = 0; i < level; i++) {
		Fq->PG_element_unrank_modified(Basis1 + i * n, 1, n, set1[i]);
	}
	cout << "set1: ";
	lint_vec_print(cout, set1, level);
	cout << endl;
	cout << "Basis1:" << endl;
	int_matrix_print(Basis1, level, n);


	long int *linear_set;
	int linear_set_sz;

	D->compute_linear_set(Basis1, level,
		linear_set, linear_set_sz,
		verbose_level);

#if 0
	int *Intersection_dimensions1;
	int *linear_set;
	int linear_set_sz, j;

	Intersection_dimensions1 = NEW_int(D->N);

	D->compute_intersection_type(level, Basis1,
		Intersection_dimensions1, 0 /*verbose_level - 1*/);

	linear_set_sz = 0;
	for (i = 0; i < D->N; i++) {
		if (Intersection_dimensions1[i]) {
			linear_set_sz++;
		}
	}
	linear_set = NEW_int(linear_set_sz);
	j = 0;
	for (i = 0; i < D->N; i++) {
		if (Intersection_dimensions1[i]) {
			linear_set[j++] = i;
		}
	}
	cout << "The linear set is: ";
	int_vec_print(cout, linear_set, linear_set_sz);
	cout << endl;
#endif


	int *Intersection_dimensions;
	int *Elt1;
	vector_ge *aut_gens;
	strong_generators *Strong_gens_previous;
	int group_index, orbit_len, go_int;
	longinteger_object go;

	Gen->get_stabilizer_generators(Strong_gens_previous,
		level, orbit_at_level, verbose_level);

	Strong_gens_previous->group_order(go);
	go_int = go.as_int();


	Elt1 = NEW_int(Aq->elt_size_in_int);
	Intersection_dimensions = NEW_int(D->N);
	aut_gens = NEW_OBJECT(vector_ge);

	aut_gens->init(Aq, verbose_level - 2);
	aut_gens->allocate(Strong_gens_previous->gens->len, verbose_level - 2);
	for (i = 0; i < Strong_gens_previous->gens->len; i++) {
		Aq->element_move(Strong_gens_previous->gens->ith(i),
				aut_gens->ith(i), 0);
	}


	group_index = 0;
	for (h = 0; h < nb_orbits; h++) {
		orbit_len = Gen_stab->orbit_length_as_int(h, level);
		cout << h << " / " << nb_orbits << " orbit if length "
				<< orbit_len << ":" << endl;
		Gen_stab->get_set_by_level(level, h, set2);
		for (i = 0; i < level; i++) {
			Fq->PG_element_unrank_modified(Basis2 + i * n, 1, n, set2[i]);
		}
		cout << "set2: ";
		lint_vec_print(cout, set2, level);
		cout << endl;
		cout << "Basis2:" << endl;
		int_matrix_print(Basis2, level, n);

		D->compute_intersection_type(level, Basis2,
			Intersection_dimensions, 0 /*verbose_level - 1*/);

		cout << "Intersection_dimensions:";
		int_vec_print(cout, Intersection_dimensions, D->N);
		cout << endl;

		//int f_lex = TRUE;

		orbit = Gen->trace_set(set2, level, level,
			set3 /* canonical_set */, Elt1 /* *Elt_transporter */,
			0 /*verbose_level */);

		if (orbit == orbit_at_level) {
			if (f_v) {
				cout << "linear_set_classify::do_compute_stabilizer orbit "
						<< h << " leads to an automorphism" << endl;
				Aq->element_print_quick(Elt1, cout);
				}
			if (!Aq->test_if_set_stabilizes(Elt1,
					nb_candidates, candidates, 0 /* verbose_level */)) {
				cout << "The automorphism does not "
						"stabilize the candidate set" << endl;
				exit(1);
			}
			else {
				if (f_v) {
					cout << "The automorphism is OK" << endl;
				}
			}
			aut_gens->append(Elt1, verbose_level - 2);
			strong_generators *Strong_gens_next;

			Gen_stab->get_stabilizer_generators(Strong_gens_next,
				level, h, verbose_level);

			for (i = 0; i < Strong_gens_next->gens->len; i++) {
				aut_gens->append(Strong_gens_next->gens->ith(i), verbose_level - 2);
			}
			FREE_OBJECT(Strong_gens_next);
			group_index += orbit_len;

		}
	}

	cout << "old stabilizer order = " << go_int << endl;
	cout << "group_index = " << group_index << endl;

	sims *Aut;
	int target_go;

	target_go = go_int * group_index;
	cout << "target_go = " << target_go << endl;
	cout << "creating group of order " << target_go << ":" << endl;
	Aut = Aq->create_sims_from_generators_with_target_group_order_lint(
		aut_gens, target_go, verbose_level);
	cout << "Stabilizer created successfully" << endl;

	//strong_generators *Aut_gens;

	//Aut_gens = NEW_OBJECT(strong_generators);

	strong_gens = NEW_OBJECT(strong_generators);

	strong_gens->init_from_sims(Aut, 0);
	cout << "Generators for the stabilizer of order "
			<< target_go << " are:" << endl;
	strong_gens->print_generators(cout);

	vector_ge *gensQ;


	action_global AG;
	AG.retract_generators(strong_gens->gens, gensQ,
			AQ, SubS, n, verbose_level);

	cout << "Generators over FQ:" << endl;
	gensQ->print_quick(cout);


#if 0
	set_stabilizer_compute *STAB;
	sims *StabQ;
	//int t0;
	int nb_backtrack_nodes;
	longinteger_object goQ;

	t0 = os_ticks();

	StabQ = NEW_OBJECT(sims);
	STAB = NEW_OBJECT(set_stabilizer_compute);
	STAB->init(A_PGLQ, StabQ, linear_set, linear_set_sz, verbose_level);
	STAB->compute_set_stabilizer(t0, nb_backtrack_nodes, verbose_level);
	StabQ->group_order(goQ);
	cout << "order of stabilizer in PGL(m,q)=" << goQ << endl;

	FREE_OBJECT(STAB);
#endif


	FREE_lint(linear_set);
	FREE_OBJECT(gensQ);

	FREE_lint(set1);
	FREE_lint(set2);
	FREE_lint(set3);
	FREE_int(Basis1);
	FREE_int(Basis2);
	FREE_int(Intersection_dimensions);
	FREE_int(Elt1);
	FREE_OBJECT(Strong_gens_previous);

	if (f_v) {
		cout << "linear_set_classify::do_classify_secondary done" << endl;
	}
}

void linear_set_classify::construct_semifield(int orbit_for_W, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	sorting Sorting;

	if (f_v) {
		cout << "linear_set_classify::construct_semifield orbit_for_W=" << orbit_for_W << endl;
	}

	long int *set1;
	long int *set2;
	int dimU, dimW;
	int *Basis1;
	int *Basis2;
	int *BasisU;
	int *BasisW;
	int i;

	set1 = NEW_lint(secondary_level);
	set2 = NEW_lint(secondary_depth);
	dimU = secondary_level + 1;
	dimW = secondary_depth;
	Basis1 = NEW_int(secondary_level * n);
	Basis2 = NEW_int(secondary_depth * n);
	BasisU = NEW_int(dimU * n1);
	BasisW = NEW_int(dimW * n1);

	int_vec_zero(BasisU, dimU * n1);
	int_vec_zero(BasisW, dimW * n1);

	Gen->get_set_by_level(secondary_level, secondary_orbit_at_level, set1);
	for (i = 0; i < secondary_level; i++) {
		Fq->PG_element_unrank_modified(Basis1 + i * n, 1, n, set1[i]);
	}
	for (i = 0; i < secondary_level; i++) {
		Fq->PG_element_unrank_modified(BasisU + i * n1, 1, n, set1[i]);
	}
	BasisU[secondary_level * n1 + n] = 1; // the vector v
	if (f_vv) {
		cout << "set1: ";
		lint_vec_print(cout, set1, secondary_level);
		cout << endl;
		cout << "Basis1:" << endl;
		int_matrix_print(Basis1, secondary_level, n);
		cout << "BasisU:" << endl;
		int_matrix_print(BasisU, dimU, n1);
	}


	Gen2->get_set_by_level(secondary_depth, orbit_for_W, set2);
	for (i = 0; i < secondary_depth; i++) {
		Fq->PG_element_unrank_modified(Basis2 + i * n, 1, n, set2[i]);
	}
	for (i = 0; i < secondary_depth; i++) {
		Fq->PG_element_unrank_modified(BasisW + i * n1, 1, n, set2[i]);
	}

	if (f_vv) {
		cout << "set2: ";
		lint_vec_print(cout, set2, secondary_depth);
		cout << endl;
		cout << "Basis2:" << endl;
		int_matrix_print(Basis2, secondary_depth, n);
		cout << "BasisW:" << endl;
		int_matrix_print(BasisW, dimW, n1);
	}


	long int *large_linear_set;
	int large_linear_set_sz;
	long int *small_linear_set;
	int small_linear_set_sz;
	long int *small_linear_set_W;
	int small_linear_set_W_sz;

	D1->compute_linear_set(BasisU, dimU,
		large_linear_set, large_linear_set_sz,
		0 /*verbose_level*/);

	if (f_vv) {
		cout << "The large linear set of size "
				<< large_linear_set_sz << " is ";
		lint_vec_print(cout, large_linear_set, large_linear_set_sz);
		cout << endl;
		D1->print_linear_set_tex(large_linear_set, large_linear_set_sz);
		cout << endl;
	}

	D->compute_linear_set(Basis1, secondary_level,
		small_linear_set, small_linear_set_sz,
		0 /*verbose_level*/);
	if (f_vv) {
		cout << "The small linear set of size "
				<< small_linear_set_sz << " is ";
		lint_vec_print(cout, small_linear_set, small_linear_set_sz);
		cout << endl;
		D->print_linear_set_tex(small_linear_set, small_linear_set_sz);
		cout << endl;
	}


	D->compute_linear_set(Basis2, secondary_depth,
		small_linear_set_W, small_linear_set_W_sz,
		0 /*verbose_level*/);
	if (f_vv) {
		cout << "The small linear set for W of size "
				<< small_linear_set_W_sz << " is ";
		lint_vec_print(cout, small_linear_set_W, small_linear_set_W_sz);
		cout << endl;
		D->print_linear_set_tex(small_linear_set_W, small_linear_set_W_sz);
		cout << endl;
	}

	int *is_deleted;
	int a, b, idx;

	for (i = 0; i < small_linear_set_sz; i++) {
		a = small_linear_set[i];
		b = spread_embedding[a];
		small_linear_set[i] = b;
	}
	if (f_vv) {
		cout << "After embedding, the small linear set of size "
				<< small_linear_set_sz << " is ";
		lint_vec_print(cout, small_linear_set, small_linear_set_sz);
		cout << endl;
		D1->print_linear_set_tex(small_linear_set, small_linear_set_sz);
		cout << endl;
	}


	is_deleted = NEW_int(large_linear_set_sz);
	for (i = 0; i < large_linear_set_sz; i++) {
		is_deleted[i] = FALSE;
	}

	for (i = 0; i < small_linear_set_sz; i++) {
		a = small_linear_set[i];
		if (!Sorting.lint_vec_search(large_linear_set,
				large_linear_set_sz, a, idx, 0)) {
			cout << "Cannot find embedded spread element "
					"in large linear set, something is wrong" << endl;
			exit(1);
		}
		is_deleted[idx] = TRUE;
	}

	long int *linear_set;
	int linear_set_sz;
	int j;

	linear_set_sz = 0;
	for (i = 0; i < large_linear_set_sz; i++) {
		if (!is_deleted[i]) {
			linear_set_sz++;
		}
	}
	linear_set = NEW_lint(linear_set_sz);
	j = 0;
	for (i = 0; i < large_linear_set_sz; i++) {
		if (!is_deleted[i]) {
			linear_set[j++] = large_linear_set[i];
		}
	}
	if (f_vv) {
		cout << "The linear set of size " << linear_set_sz << " is ";
		lint_vec_print(cout, linear_set, linear_set_sz);
		cout << endl;
		D1->print_linear_set_tex(linear_set, linear_set_sz);
		cout << endl;
	}


	int *base_cols;
	int *kernel_cols;
	int *Spread_element_basis;
	int *Basis_elt;
	int *Basis_infinity;
	int h;
	int *v1, *v2;
	int n2;

	n2 = n1 - dimW;
	Spread_element_basis = NEW_int(D1->spread_element_size);
	Basis_infinity = NEW_int(s * n2);
	Basis_elt = NEW_int(dimW * n2);
	base_cols = NEW_int(n1);
	kernel_cols = NEW_int(n1);
	if (Fq->Gauss_simple(BasisW, dimW, n1, base_cols,
			0/* verbose_level*/) != dimW) {
		cout << "BasisW does not have the correct rank" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "BasisW:" << endl;
		int_matrix_print(BasisW, dimW, n1);
		cout << "base_cols:";
		int_vec_print(cout, base_cols, dimW);
		cout << endl;
	}

	Fq->kernel_columns(n1, dimW, base_cols, kernel_cols);
	if (f_vv) {
		cout << "kernel_cols:";
		int_vec_print(cout, kernel_cols, n2);
		cout << endl;
	}



	int_vec_zero(Basis_infinity, s * n2);
	for (i = 0; i < s; i++) {
		//a = kernel_cols[i] - s;
		Basis_infinity[i * n2 + i] = 1;
	}
	if (f_vv) {
		cout << "Basis element infinity:" << endl;
		int_matrix_print(Basis_infinity, s, n2);
	}


	int nb_components;
	int **Components;
	int *Spread_set;

	nb_components = linear_set_sz + 1;
	Components = NEW_pint(nb_components);
	Spread_set = NEW_int(linear_set_sz * s * s);


	Components[0] = NEW_int(s * n2);
	int_vec_copy(Basis_infinity, Components[0], s * n2);

	for (h = 0; h < linear_set_sz; h++) {
		if (f_v3) {
			cout << "spread element " << h << " / "
					<< linear_set_sz << ":" << endl;
		}
		a = linear_set[h];
		int_vec_copy(
				D1->Spread_elements + a * D1->spread_element_size,
				Spread_element_basis, D1->spread_element_size);
		if (f_v3) {
			cout << "Spread element " << a << " is:" << endl;
			int_matrix_print(Spread_element_basis, s, n1);
		}

		for (i = 0; i < dimW; i++) {
			a = base_cols[i];
			v1 = BasisW + i * n1;
			for (j = 0; j < s; j++) {
				v2 = Spread_element_basis + j * n1;
				if (v2[a]) {
					Fq->Gauss_step(v1, v2, n1, a, 0 /* verbose_level*/);
				}
			}
		}
		if (f_v3) {
			cout << "Basis after reduction mod W:" << endl;
			int_matrix_print(Spread_element_basis, s, n1);
		}

		for (i = 0; i < dimW; i++) {
			for (j = 0; j < n2; j++) {
				a = kernel_cols[j];
				Basis_elt[i * n2 + j] = Spread_element_basis[i * n1 + a];
			}
		}

		if (f_v3) {
			cout << "Basis element:" << endl;
			int_matrix_print(Basis_elt, s, n2);
		}

		Fq->Gauss_easy(Basis_elt, s, n2);

		if (f_v3) {
			cout << "Basis element after RREF:" << endl;
			int_matrix_print(Basis_elt, s, n2);
		}

		for (i = 0; i < s; i++) {
			for (j = 0; j < s; j++) {
				a = Basis_elt[i * n2 + s + j];
				Spread_set[h * s * s + i * s + j] = a;
			}
		}

		Components[h + 1] = NEW_int(s * n2);
		int_vec_copy(Basis_elt, Components[h + 1], s * n2);
	}

	if (f_v3) {
		cout << "The components are:" << endl;
		for (h = 0; h < linear_set_sz + 1; h++) {
			cout << "Component " << h << " / "
					<< linear_set_sz << ":" << endl;
			int_matrix_print(Components[h], s, n2);
		}
	}

	int h2;

	h2 = 0;
	for (h = 0; h < linear_set_sz + 1; h++) {
		if (h == 1) {
			continue;
		}
		for (i = 0; i < s; i++) {
			for (j = 0; j < s; j++) {
				a = Components[h][i * n2 + s + j];
				Spread_set[h2 * s * s + i * s + j] = a;
			}
		}

		h2++;
	}

	int h1, k3;
	int *Intersection;

	Intersection = NEW_int(n2 * n2);
	for (h1 = 0; h1 < nb_components; h1++) {
		for (h2 = h1 + 1; h2 < nb_components; h2++) {
			Fq->intersect_subspaces(n2, s,
				Components[h1], s, Components[h2],
				k3, Intersection, 0 /* verbose_level */);
			if (k3) {
				cout << "Components " << h1 << " and "
						<< h2 << " intersect non-trivially!" << endl;
				cout << "Component " << h1 << " / "
						<< nb_components << ":" << endl;
				int_matrix_print(Components[h1], s, n2);
				cout << "Component " << h2 << " / "
						<< nb_components << ":" << endl;
				int_matrix_print(Components[h2], s, n2);
			}
		}
	}
	if (f_vv) {
		cout << "The components are disjoint!" << endl;
	}


	int rk;

	if (f_v3) {
		cout << "The spread_set is:" << endl;
		int_matrix_print(Spread_set, linear_set_sz, s * s);
	}
	rk = Fq->Gauss_easy(Spread_set, linear_set_sz, s * s);
	if (f_v) {
		cout << "rank = " << rk << endl;
	}
	if (f_v3) {
		cout << "The spread_set basis is:" << endl;
		int_matrix_print(Spread_set, rk, s * s);
		for (h = 0; h < rk; h++) {
			cout << "basis elt " << h << " / " << rk << ":" << endl;
			int_matrix_print(Spread_set + h * s * s, s, s);
		}
	}



	if (f_v3) {
		cout << "opening grassmann:" << endl;
	}
	grassmann *Grass;
	Grass = NEW_OBJECT(grassmann);
	Grass->init(n2, s, Fq, 0 /*verbose_level*/);

	long int *spread_elements_numeric;

	spread_elements_numeric = NEW_lint(nb_components);
	for (h = 0; h < nb_components; h++) {
		spread_elements_numeric[h] =
				Grass->rank_lint_here(Components[h], 0);
	}

	if (f_vv) {
		cout << "spread elements numeric:" << endl;
		for (i = 0; i < nb_components; i++) {
			cout << setw(3) << i << " : "
					<< spread_elements_numeric[i] << endl;
		}
	}

	if (f_identify) {
		cout << "linear_set::construct_semifield "
				"before T->identify" << endl;

		if (nb_components != order + 1) {
			cout << "nb_components != order + 1" << endl;
			exit(1);
		}

		int *transporter;
		int f_implicit_fusion = FALSE;
		int final_node;

		transporter = NEW_int(T->gen->get_A()->elt_size_in_int);

		T->gen->recognize(
			spread_elements_numeric, nb_components,
			transporter, f_implicit_fusion,
			final_node, 0 /*verbose_level*/);
		//T->identify(spread_elements_numeric, nb_components, verbose_level);

		longinteger_object go;
		int lvl;
		int orbit_at_lvl;

		lvl = order + 1;
		orbit_at_lvl = final_node - T->gen->first_node_at_level(lvl);

		T->gen->get_stabilizer_order(lvl, orbit_at_lvl, go);

		cout << "linear_set::construct_semifield after recognize" << endl;
		cout << "final_node=" << final_node
				<< " which is isomorphism type " << orbit_at_lvl
				<< " with stabilizer order " << go << endl;
		cout << "transporter=" << endl;
		T->gen->get_A()->element_print_quick(transporter, cout);

		FREE_int(transporter);
	}

#if 0
	andre_construction *Andre;

	Andre = NEW_OBJECT(andre_construction);

	cout << "Creating the projective plane using "
			"the Andre construction:" << endl;
	Andre->init(Fq, s, spread_elements_numeric, 0 /*verbose_level*/);
	cout << "Done creating the projective plane using "
			"the Andre construction." << endl;
#endif


	FREE_lint(large_linear_set);
	FREE_lint(small_linear_set);
	FREE_lint(linear_set);
	FREE_lint(small_linear_set_W);
	FREE_lint(set1);
	FREE_lint(set2);
	FREE_int(Basis1);
	FREE_int(Basis2);
	FREE_int(BasisU);
	FREE_int(BasisW);
}


// #############################################################################
// global functions:
// #############################################################################


long int linear_set_classify_rank_point_func(int *v, void *data)
{
	linear_set_classify *LS;
	long int rk;
	geometry_global Gg;

	LS = (linear_set_classify *) data;
	rk = Gg.AG_element_rank(LS->Fq->q, v, 1, LS->vector_space_dimension);
	//PG_element_rank_modified(*LS->Fq, v, 1,
	//LS->vector_space_dimension, rk);
	return rk;
}

void linear_set_classify_unrank_point_func(int *v, long int rk, void *data)
{
	linear_set_classify *LS;
	geometry_global Gg;

	LS = (linear_set_classify *) data;
	Gg.AG_element_unrank(LS->Fq->q, v, 1, LS->vector_space_dimension, rk);
	//PG_element_unrank_modified(*LS->Fq, v, 1,
	//LS->vector_space_dimension, rk);
}

void linear_set_classify_early_test_func(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	verbose_level = 2;

	linear_set_classify *LS;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	LS = (linear_set_classify *) data;

	if (f_v) {
		cout << "linear_set_classify_early_test_func" << endl;
		cout << "testing " << nb_candidates << " candidates" << endl;
	}
	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		S[len] = candidates[i];
		if (candidates[i] != 0) {
			// avoid the zero vector for subspace computations
			// recall that the group is not projective
			if (LS->test_set(len + 1, S, verbose_level - 1)) {
				good_candidates[nb_good_candidates++] = candidates[i];
				if (f_vv) {
					cout << "candidate " << i << " / " << nb_candidates
							<< " which is " << candidates[i]
							<< " is accepted" << endl;
				}
			}
			else {
				if (f_vv) {
					cout << "candidate " << i << " / " << nb_candidates
							<< " which is " << candidates[i]
							<< " is rejected" << endl;
				}
			}
		}
	}
	if (f_v) {
		cout << "linear_set_classify_early_test_func" << endl;
		cout << "Out of " << nb_candidates << " candidates, "
				<< nb_good_candidates << " survive" << endl;
	}
}

void linear_set_classify_secondary_early_test_func(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	//verbose_level = 1;

	linear_set_classify *LS;
	int f_v = (verbose_level >= 1);
	int i;

	LS = (linear_set_classify *) data;

	if (f_v) {
		cout << "linear_set_classify_secondary_early_test_func" << endl;
		cout << "testing " << nb_candidates << " candidates" << endl;
	}
	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		S[len] = candidates[i];
		if (LS->is_allowed[candidates[i]]) {
			if (LS->test_set_secondary(len + 1, S, verbose_level - 1)) {
				good_candidates[nb_good_candidates++] = candidates[i];
			}
		}
	}
	if (f_v) {
		cout << "linear_set_classify_secondary_early_test_func" << endl;
		cout << "Out of " << nb_candidates << " candidates, "
				<< nb_good_candidates << " survive" << endl;
	}
}



}}

