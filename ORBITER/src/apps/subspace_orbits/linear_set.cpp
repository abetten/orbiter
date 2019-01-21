// linear_set.C
// 
// Anton Betten
// July 8, 2014
//
//
//

#include <orbiter.h>

using namespace orbiter;

#include "linear_set.h"





linear_set::linear_set()
{
	null();
}

linear_set::~linear_set()
{
	freeself();
}

void linear_set::null()
{
	f_has_extra_test_func = FALSE;
	extra_test_func = NULL;
	extra_test_func_data = NULL;
	Fq = NULL;
	FQ = NULL;
	SubS = NULL;
	P = NULL;
	Aq = NULL;
	AQ = NULL;
	A_PGLQ = NULL;
	Strong_gens = NULL;
	D = NULL;
	D1 = NULL;
	spread_embedding = NULL;
	VS = NULL;
	Poset1 = NULL;
	Gen = NULL;
	Basis = NULL;
	base_cols = NULL;
	Poset_stab = NULL;
	Poset2 = NULL;
	Gen2 = NULL;
	is_allowed = NULL;
	f_identify = FALSE;
	T = NULL;
}

void linear_set::freeself()
{
	int f_v = FALSE;

	if (VS) {
		if (f_v) {
			cout << "linear_set::freeself before delete VS" << endl;
			}
		FREE_OBJECT(VS);
		}
	if (Poset1) {
		if (f_v) {
			cout << "linear_set::freeself before delete Poset1" << endl;
			}
		FREE_OBJECT(Poset1);
		}
	if (Gen) {
		if (f_v) {
			cout << "linear_set::freeself before delete Gen" << endl;
			}
		FREE_OBJECT(Gen);
		}
	if (Strong_gens) {
		if (f_v) {
			cout << "linear_set::freeself before delete Strong_gens" << endl;
			}
		FREE_OBJECT(Strong_gens);
		}
	if (D) {
		if (f_v) {
			cout << "linear_set::freeself before delete D" << endl;
			}
		FREE_OBJECT(D);
		}
	if (D1) {
		if (f_v) {
			cout << "linear_set::freeself before delete D1" << endl;
			}
		FREE_OBJECT(D1);
		}
	if (spread_embedding) {
		if (f_v) {
			cout << "linear_set::freeself before delete spread_embedding" << endl;
			}
		FREE_OBJECT(spread_embedding);
		}
	if (P) {
		if (f_v) {
			cout << "linear_set::freeself before delete P" << endl;
			}
		FREE_OBJECT(P);
		}
	if (Aq) {
		if (f_v) {
			cout << "linear_set::freeself before delete Aq" << endl;
			}
		FREE_OBJECT(Aq);
		}
	if (AQ) {
		if (f_v) {
			cout << "linear_set::freeself before delete AQ" << endl;
			}
		FREE_OBJECT(AQ);
		}
	if (A_PGLQ) {
		if (f_v) {
			cout << "linear_set::freeself before delete A_PGLQ" << endl;
			}
		FREE_OBJECT(A_PGLQ);
		}
	if (SubS) {
		if (f_v) {
			cout << "linear_set::freeself before delete SubS" << endl;
			}
		FREE_OBJECT(SubS);
		}
	if (Fq) {
		if (f_v) {
			cout << "linear_set::freeself before delete Fq" << endl;
			}
		FREE_OBJECT(Fq);
		}
	if (FQ) {
		if (f_v) {
			cout << "linear_set::freeself before delete FQ" << endl;
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

void linear_set::init(int argc, const char **argv, 
	int s, int n, int q, const char *poly_q, const char *poly_Q, 
	int depth, int f_identify, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "linear_set::init" << endl;
		}

	linear_set::s = s;
	linear_set::n = n;
	linear_set::q = q;
	linear_set::depth = depth;
	linear_set::f_identify = f_identify;
	if (f_v) {
		cout << "linear_set::init s=" << s << endl;
		cout << "linear_set::init n=" << n << endl;
		cout << "linear_set::init q=" << q << endl;
		cout << "linear_set::init depth=" << depth << endl;
		cout << "linear_set::init f_identify=" << f_identify << endl;
		}


	Q = i_power_j(q, s);
	m = n / s;
	if (m * s != n) {
		cout << "linear_set::init s must divide n" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "linear_set::init m=" << m << endl;
		cout << "linear_set::init Q=" << Q << endl;
		}

	vector_space_dimension = n;
	if (is_prime(q)) {
		f_semilinear = FALSE;
		}
	else {
		f_semilinear = TRUE;
		}

	Fq = NEW_OBJECT(finite_field);
	if (f_v) {
		cout << "linear_set::init before Fq->init" << endl;
		}
	Fq->init_override_polynomial(q, poly_q, 0);

	FQ = NEW_OBJECT(finite_field);
	if (f_v) {
		cout << "linear_set::init before FQ->init" << endl;
		}
	FQ->init_override_polynomial(Q, poly_Q, 0);

	SubS = NEW_OBJECT(subfield_structure);
	if (f_v) {
		cout << "linear_set::init before SubS->init" << endl;
		}
	SubS->init(FQ, Fq, verbose_level);

	if (f_v) {
		cout << "Field-basis: ";
		int_vec_print(cout, SubS->Basis, s);
		cout << endl;
		}
	

	P = NEW_OBJECT(projective_space);
	if (f_v) {
		cout << "linear_set::init before P->init" << endl;
		}
	P->init(n - 1, Fq, 
		FALSE /* f_init_incidence_structure */, 
		0 /*verbose_level*/);
	if (f_v) {
		cout << "linear_set::init after P->init" << endl;
		}



	if (f_v) {
		cout << "linear_set::init before init_general_linear_group "
				"GL(" << n << "," << Fq->q << ")" << endl;
		}
	Aq = NEW_OBJECT(action);
	Aq->init_general_linear_group(n, Fq, 
		FALSE /* f_semilinear */, 
		TRUE /* f_basis */, 
		verbose_level - 2);
	if (f_v) {
		cout << "linear_set::init after init_general_linear_group "
				"GL(" << n << "," << Fq->q << ")" << endl;
		}


	AQ = NEW_OBJECT(action);
	
	if (f_v) {
		cout << "linear_set::init before init_general_linear_group "
				"GL(" << m << "," << FQ->q << ")" << endl;
		}
	AQ->init_general_linear_group(m, FQ, 
		FALSE /* f_semilinear */, 
		TRUE /* f_basis */, 
		verbose_level - 2);
	if (f_v) {
		cout << "linear_set::init after init_general_linear_group "
				"GL(" << m << "," << FQ->q << ")" << endl;
		}

	if (f_vv) {
		cout << "Strong generators are:" << endl;
		AQ->Strong_gens->print_generators();
		AQ->Strong_gens->print_generators_tex();
		}


	A_PGLQ = NEW_OBJECT(action);
	if (f_v) {
		cout << "linear_set::init before init_projective_group "
				"PGL(" << m << "," << FQ->q << ")" << endl;
		}
	A_PGLQ->init_projective_group(m, FQ, 
		FALSE /* f_semilinear */, 
		TRUE /* f_basis */, 
		verbose_level - 2);
	if (f_v) {
		cout << "linear_set::init after init_projective_group "
				"PGL(" << m << "," << FQ->q << ")" << endl;
		}


	if (f_v) {
		cout << "linear_set::init before linear_set_lift_generators_"
				"to_subfield_structure" << endl;
		}
	lift_generators_to_subfield_structure(n, s, 
		SubS, Aq, AQ, Strong_gens, 
		verbose_level);
		// in ACTION/action_global.C
	if (f_v) {
		cout << "linear_set::init after linear_set_lift_generators_"
				"to_subfield_structure" << endl;
		}

	if (f_v) {
		cout << "After lift, strong generators are:" << endl;
		Strong_gens->print_generators();
		Strong_gens->print_generators_tex();
		}



	Basis = NEW_int(depth * vector_space_dimension);
	base_cols = NEW_int(vector_space_dimension);


	D = NEW_OBJECT(desarguesian_spread);
	if (f_v) {
		cout << "linear_set::init before D->init" << endl;
		}
	D->init(n, m, s, 
		SubS, 
		verbose_level);
	if (f_v) {
		cout << "linear_set::init after D->init" << endl;
		}

	m1 = m + 1;
	n1 = s * m1; // = n + s
	D1 = NEW_OBJECT(desarguesian_spread);
	if (f_v) {
		cout << "linear_set::init before D1->init" << endl;
		}
	D1->init(n1, m1, s, 
		SubS, 
		0 /*verbose_level*/);
	if (f_v) {
		cout << "linear_set::init after D1->init" << endl;
		}

	int *vec;
	int i, j;

	if (f_v) {
		cout << "linear_set::init computing spread_embedding" << endl;
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
		cout << "linear_set::init computing spread_embedding done" << endl;
		}	

	VS = NEW_OBJECT(vector_space);
	VS->init(P->F, vector_space_dimension /* dimension */,
			verbose_level - 1);
	VS->init_rank_functions(
			linear_set_rank_point_func,
			linear_set_unrank_point_func,
			this,
			verbose_level - 1);


	Poset1 = NEW_OBJECT(poset);
	Gen = NEW_OBJECT(poset_classification);

	Gen->read_arguments(argc, argv, 0);

	//Gen->prefix[0] = 0;
	sprintf(Gen->fname_base, "subspaces_%d_%d_%d", n, q, s);
	
	
	Poset1->init_subspace_lattice(Aq, Aq, Strong_gens, VS,
			verbose_level);

	Gen->depth = depth;

	if (f_v) {
		cout << "linear_set::init before Gen->init" << endl;
		}
	Gen->init(Poset1, Gen->depth /* sz */, verbose_level);
	if (f_v) {
		cout << "linear_set::init after Gen->init" << endl;
		}


#if 0
	Gen->init_check_func(
		subspace_orbits_test_func, 
		this /* candidate_check_data */);
#endif

#if 0
	// ToDo
	Gen->init_early_test_func(
		linear_set_early_test_func, 
		this /*void *data */,  
		verbose_level);
#endif


	//Gen->init_incremental_check_func(
		//check_mindist_incremental, 
		//this /* candidate_check_data */);

#if 0
	Gen->init_vector_space_action(vector_space_dimension, 
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

	int nb_nodes = 1000;
	
	if (f_v) {
		cout << "linear_set::init before Gen->init_poset_orbit_node" << endl;
		}
	Gen->init_poset_orbit_node(nb_nodes, verbose_level - 1);
	if (f_v) {
		cout << "linear_set::init calling Gen->init_root_node" << endl;
		}
	Gen->root[0].init_root_node(Gen, verbose_level - 1);
	
	schreier_depth = Gen->depth;
	f_use_invariant_subset_if_available = TRUE;
	//f_lex = FALSE;
	f_debug = FALSE;


	if (f_identify) {
		T = NEW_OBJECT(spread);

		int f_recoordinatize = TRUE;
		
		k = n >> 1;
		order = i_power_j(q, k);

		if (f_v) {
			cout << "Classifying spreads of order " << order << endl;
			}

		int max_depth = order + 1;

		T->init(order, n, k, max_depth, 
			Fq, f_recoordinatize, 
			"SPREADS_STARTER", "Spreads", order + 1, 
			argc, argv, 
			MINIMUM(verbose_level - 1, 2));
	
		T->read_arguments(argc, argv);

		T->init2(verbose_level);

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
		cout << "linear_set::init done" << endl;
		}
}

void linear_set::do_classify(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_set::do_classify" << endl;
		}

	//int t0 = os_ticks();
	
	if (f_v) {
		cout << "linear_set::do_classify calling generator_main" << endl;
		cout << "A=";
		Gen->Poset->A->print_info();
		cout << "A2=";
		Gen->Poset->A2->print_info();
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
		cout << "linear_set::do_classify done with generator_main" << endl;
		}
	nb_orbits = Gen->nb_orbits_at_level(depth);
	if (f_v) {
		cout << "linear_set::do_classify we found " << nb_orbits
				<< " orbits at depth " << depth<< endl;
		}


	if (f_v) {
		cout << "linear_set::do_classify done" << endl;
		}
}

int linear_set::test_set(int len, int *S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret = TRUE;
	int i, rk;
	
	if (f_v) {
		cout << "linear_set::test_set" << endl;
		cout << "Testing set ";
		int_vec_print(cout, S, len);
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

void linear_set::compute_intersection_types_at_level(int level, 
	int &nb_nodes, int *&Intersection_dimensions, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int node, i;
	int *set;
	
	if (f_v) {
		cout << "linear_set::compute_intersection_types_at_level" << endl;
		}
	
	set = NEW_int(level);

	nb_nodes = Gen->nb_orbits_at_level(level);
	Intersection_dimensions = NEW_int(nb_nodes * D->N);
	for (node = 0; node < nb_nodes; node++) {
		Gen->get_set_by_level(level, node, set);
		for (i = 0; i < level; i++) {
			Fq->PG_element_unrank_modified(
				Basis + i * n, 1, n, set[i]);
			}
		D->compute_intersection_type(level, Basis, 
			Intersection_dimensions + node * D->N, 
			0 /*verbose_level - 1*/);
		}


	FREE_int(set);

	if (f_v) {
		cout << "linear_set::compute_intersection_types_at_level done" << endl;
		}
}

void linear_set::calculate_intersections(int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "linear_set::calculate_intersections" << endl;
		}

	int level;
	int *Nb_nodes;
	int **Intersection_dimensions;
	int ***Sets;
	int **Set_sz;
	longinteger_object go;
	int i, h, j;
	
	Nb_nodes = NEW_int(depth + 1);
	Intersection_dimensions = NEW_pint(depth + 1);
	Sets = NEW_ppint(depth + 1);
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
			classify C;

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
		Sets[level] = NEW_pint(Nb_nodes[level]);
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
			Sets[level][h] = NEW_int(Set_sz[level][h]);
			j = 0;
			for (i = 0; i < D->N; i++) {
				if (I[i]) {
					Sets[level][h][j++] = i;
					}
				}
			cout << h << " : ";
			int_vec_print(cout, Sets[level][h], Set_sz[level][h]);
			cout << endl;
			}
		}


	for (level = 0; level <= depth; level++) {
		for (h = 0; h < Nb_nodes[level]; h++) {
			FREE_int(Sets[level][h]);
			}
		FREE_pint(Sets[level]);
		FREE_int(Set_sz[level]);
		FREE_int(Intersection_dimensions[level]);
		}
	FREE_ppint(Sets);
	FREE_pint(Set_sz);
	FREE_pint(Intersection_dimensions);
	FREE_int(Nb_nodes);

	if (f_v) {
		cout << "linear_set::calculate_intersections done" << endl;
		}
}

void linear_set::read_data_file(int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int depth_completed;
	char fname[1000];

	if (f_v) {
		cout << "linear_set::read_data_file" << endl;
		}
	sprintf(fname, "%s_%d.data", Gen->fname_base, depth);
	Gen->read_data_file(depth_completed, fname, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "linear_set::read_data_file after read_data_file" << endl;
		}

	int level;
	char prefix[1000];


	sprintf(prefix, "%sb", Gen->fname_base);
	for (level = 0; level < depth; level++) {
		if (f_v) {
			cout << "linear_set::read_data_file before "
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
		cout << "linear_set::read_data_file done" << endl;
		}
}


void linear_set::print_orbits_at_level(int level)
{
	int len, orbit_at_level, i;
	longinteger_object go;
	int *set;
	int *Basis;

	set = NEW_int(level);
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
		int_vec_print(cout, set, level);
		cout << endl;
		cout << "Basis:" << endl;
		int_matrix_print(Basis, level, n);
		}

	FREE_int(set);
	FREE_int(Basis);

}

void linear_set::classify_secondary(int argc, const char **argv, 
	int level, int orbit_at_level, 
	strong_generators *strong_gens, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, nb_allowed;
	int *set;
	
	if (f_v) {
		cout << "linear_set::classify_secondary" << endl;
		}
	
	secondary_level = level;
	secondary_orbit_at_level = orbit_at_level;

	set = NEW_int(level);
	is_allowed = NEW_int(Aq->degree);

	Gen->get_set_by_level(level, orbit_at_level, set);
	for (i = 0; i < level; i++) {
		Fq->PG_element_unrank_modified(Basis + i * n, 1, n, set[i]);
		}
	cout << "set: ";
	int_vec_print(cout, set, level);
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

	int *candidates;
	int nb_candidates;
	
	candidates = NEW_int(nb_allowed);
	nb_candidates = 0;
	for (i = 0; i < Aq->degree; i++) {
		if (is_allowed[i]) {
			candidates[nb_candidates++] = i;
			}
		}
	cout << "candidates:" << nb_candidates << endl;
	int_vec_print(cout, candidates, nb_candidates);
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


	
	FREE_int(set);
	//FREE_int(is_allowed);
		// don't free is_allowed,
		// it is part of linear_set now.
	FREE_int(candidates);
}

void linear_set::init_secondary(int argc, const char **argv, 
	int *candidates, int nb_candidates, 
	strong_generators *Strong_gens_previous, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_set::init_secondary" << endl;
		}

	secondary_candidates = candidates;
	secondary_nb_candidates = nb_candidates;

	Poset2 = NEW_OBJECT(poset);
	Gen2 = NEW_OBJECT(poset_classification);

	Gen2->read_arguments(argc, argv, 0);

	//Gen2->prefix[0] = 0;
	sprintf(Gen2->fname_base, "subspaces_%d_%d_%d_secondary_%d_%d", n, q, s, 
		secondary_level, secondary_orbit_at_level);
	
	
	secondary_depth = n - secondary_level;
	Gen2->depth = secondary_depth;
	if (f_v) {
		cout << "linear_set::init_secondary "
				"secondary_level = " << secondary_level << endl;
		cout << "linear_set::init_secondary "
				"secondary_depth = " << secondary_depth << endl;
		}

	
	if (f_v) {
		cout << "linear_set::init_secondary generators are:" << endl;
		Strong_gens_previous->print_generators();
		//Strong_gens_previous->print_generators_as_permutations();
		}
	
	cout << "linear_set::init_secondary before Gen2->init" << endl;
	Poset2->init_subspace_lattice(Aq, Aq,
			Strong_gens_previous, VS,
			verbose_level);
	Gen2->init(Poset2,
			Gen2->depth /* sz */,
			verbose_level);
	cout << "linear_set::init_secondary after Gen2->init" << endl;

	Gen2->f_max_depth = FALSE;
		// could have been set to true because of -depth option

#if 0
	Gen2->init_check_func(
		subspace_orbits_test_func, 
		this /* candidate_check_data */);
#endif

#if 0
	// ToDo
	Gen2->init_early_test_func(
		linear_set_secondary_early_test_func, 
		this /*void *data */,  
		verbose_level);
#endif

	//Gen2->init_incremental_check_func(
		//check_mindist_incremental, 
		//this /* candidate_check_data */);

#if 0
	Gen2->init_vector_space_action(vector_space_dimension, 
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

	int nb_nodes = 1000;
	
	if (f_v) {
		cout << "linear_set::init_secondary before "
				"Gen2->init_poset_orbit_node" << endl;
		}
	Gen2->init_poset_orbit_node(nb_nodes, verbose_level - 1);
	if (f_v) {
		cout << "linear_set::init_secondary calling "
				"Gen2->init_root_node" << endl;
		}
	Gen2->root[0].init_root_node(Gen2, verbose_level - 1);
	
	secondary_schreier_depth = Gen2->depth;
	//f_use_invariant_subset_if_available = TRUE;
	//f_lex = FALSE;
	//f_debug = FALSE;



	// the following works only for actions on subsets:
#if 0
	if (f_v) {
		cout << "linear_set::init_secondary before "
				"Gen2->init_root_node_invariant_subset" << endl;
		}
	Gen2->init_root_node_invariant_subset(
		secondary_candidates, secondary_nb_candidates, verbose_level);
	if (f_v) {
		cout << "linear_set::init_secondary after "
				"Gen2->init_root_node_invariant_subset" << endl;
		}
#endif

	if (f_v) {
		cout << "linear_set::init_secondary before "
				"do_classify_secondary" << endl;
		}
	do_classify_secondary(verbose_level);
	if (f_v) {
		cout << "linear_set::init_secondary after "
				"do_classify_secondary" << endl;
		}
	if (f_v) {
		cout << "linear_set::init_secondary done" << endl;
		}

}

void linear_set::do_classify_secondary(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_set::do_classify_secondary" << endl;
		}

	int t0 = os_ticks();
	
	if (f_v) {
		cout << "linear_set::do_classify_secondary "
				"calling generator_main" << endl;
		cout << "A=";
		Gen2->Poset->A->print_info();
		cout << "A2=";
		Gen2->Poset->A2->print_info();
		}
	Gen2->main(t0, 
		secondary_schreier_depth, 
		f_use_invariant_subset_if_available, 
		//f_lex, 
		f_debug, 
		verbose_level - 1);
	
	int nb_orbits;
	
	if (f_v) {
		cout << "linear_set::do_classify_secondary "
				"done with generator_main" << endl;
		}
	nb_orbits = Gen2->nb_orbits_at_level(secondary_depth);
	if (f_v) {
		cout << "linear_set::do_classify_secondary we found "
				<< nb_orbits << " orbits at depth " << secondary_depth<< endl;
		}

	int h, i;
	
	int *set1;
	int *set2;
	int *Basis1;
	int *Basis2;

	set1 = NEW_int(secondary_level);
	set2 = NEW_int(secondary_depth);
	Basis1 = NEW_int(secondary_level * n);
	Basis2 = NEW_int(secondary_depth * n);

	Gen->get_set_by_level(secondary_level, secondary_orbit_at_level, set1);
	for (i = 0; i < secondary_level; i++) {
		Fq->PG_element_unrank_modified(Basis1 + i * n, 1, n, set1[i]);
		}
	cout << "set1: ";
	int_vec_print(cout, set1, secondary_level);
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
		int_vec_print(cout, set2, secondary_depth);
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
		Strong_gens2->print_generators();

		delete Strong_gens2;
		}

	FREE_int(Intersection_dimensions);


	if (f_v) {
		cout << "linear_set::do_classify_secondary done" << endl;
		}
}

int linear_set::test_set_secondary(int len, int *S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret = TRUE;
	int i, rk;
	int *v;
	int *w;
	int nb;
	
	if (f_v) {
		cout << "linear_set::test_set_secondary" << endl;
		cout << "Testing set ";
		int_vec_print(cout, S, len);
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


		v = NEW_int(len);
		w = NEW_int(n);
		nb = nb_PG_elements(len - 1, q);

	
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

void linear_set::compute_stabilizer_of_linear_set(
	int argc, const char **argv,
	int level, int orbit_at_level, 
	strong_generators *&strong_gens, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, nb_allowed;
	int *set;
	
	if (f_v) {
		cout << "linear_set::compute_stabilizer_of_linear_set" << endl;
		}
	
	set = NEW_int(level);
	is_allowed = NEW_int(Aq->degree);

	Gen->get_set_by_level(level, orbit_at_level, set);
	for (i = 0; i < level; i++) {
		Fq->PG_element_unrank_modified(Basis + i * n, 1, n, set[i]);
		}
	cout << "set: ";
	int_vec_print(cout, set, level);
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



	int *candidates;
	int nb_candidates;
	
	candidates = NEW_int(nb_allowed);
	nb_candidates = 0;
	for (i = 0; i < Aq->degree; i++) {
		if (is_allowed[i]) {
			candidates[nb_candidates++] = i;
			}
		}
	cout << "candidates:" << nb_candidates << endl;
	int_vec_print(cout, candidates, nb_candidates);
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

	
	FREE_int(set);
	//FREE_int(is_allowed);
	// don't free is_allowed,
	//it is part of linear_set now.
	FREE_int(candidates);
}

void linear_set::init_compute_stabilizer(int argc, const char **argv, 
	int level, int orbit_at_level,  
	int *candidates, int nb_candidates, 
	strong_generators *Strong_gens_previous, 
	strong_generators *&strong_gens, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_set::init_compute_stabilizer" << endl;
		}

	Poset_stab = NEW_OBJECT(poset);
	Gen_stab = NEW_OBJECT(poset_classification);

	Gen_stab->read_arguments(argc, argv, 0);

	//Gen_stab->prefix[0] = 0;
	sprintf(Gen_stab->fname_base,
		"subspaces_%d_%d_%d_stabilizer_%d_%d", n, q, s,
		level, orbit_at_level);
	
	
	Gen_stab->depth = level;
	if (f_v) {
		cout << "linear_set::init_compute_stabilizer "
				"depth = " << Gen_stab->depth << endl;
		}

	
	if (f_v) {
		cout << "linear_set::init_compute_stabilizer "
				"generators are:" << endl;
		Strong_gens_previous->print_generators();
		//Strong_gens_previous->print_generators_as_permutations();
		}
	
	cout << "linear_set::init_compute_stabilizer "
			"before Gen_stab->init" << endl;
	Poset_stab->init_subspace_lattice(Aq, Aq,
			Strong_gens_previous, VS,
			verbose_level);
	Gen_stab->init(Poset_stab,
			Gen_stab->depth /* sz */,
			verbose_level);
	cout << "linear_set::init_compute_stabilizer "
			"after Gen_stab->init" << endl;

	Gen_stab->f_max_depth = FALSE; // could have been set to true because of -depth option

#if 0
	Gen_stab->init_check_func(
		subspace_orbits_test_func, 
		this /* candidate_check_data */);
#endif


#if 0
	// ToDo
	Gen_stab->init_early_test_func(
		linear_set_secondary_early_test_func, 
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

	int nb_nodes = 1000;
	
	if (f_v) {
		cout << "linear_set::init_compute_stabilizer "
				"before Gen_stab->init_poset_orbit_node" << endl;
		}
	Gen_stab->init_poset_orbit_node(nb_nodes, verbose_level - 1);
	if (f_v) {
		cout << "linear_set::init_compute_stabilizer "
				"calling Gen_stab->init_root_node" << endl;
		}
	Gen_stab->root[0].init_root_node(Gen_stab, verbose_level - 1);
	
	//stabilizer_schreier_depth = Gen_stab->depth;
	//f_use_invariant_subset_if_available = TRUE;
	//f_lex = FALSE;
	//f_debug = FALSE;



	// the following works only for actions on subsets:
#if 0
	if (f_v) {
		cout << "linear_set::init_secondary before "
				"Gen_stab->init_root_node_invariant_subset" << endl;
		}
	Gen_stab->init_root_node_invariant_subset(
		secondary_candidates, secondary_nb_candidates, verbose_level);
	if (f_v) {
		cout << "linear_set::init_secondary after "
				"Gen_stab->init_root_node_invariant_subset" << endl;
		}
#endif

	if (f_v) {
		cout << "linear_set::init_compute_stabilizer "
				"before do_compute_stabilizer" << endl;
		}
	do_compute_stabilizer(level, orbit_at_level, 
		candidates, nb_candidates, 
		strong_gens, 
		verbose_level);
	if (f_v) {
		cout << "linear_set::init_compute_stabilizer "
				"after do_compute_stabilizer" << endl;
		}
	if (f_v) {
		cout << "linear_set::init_compute_stabilizer done" << endl;
		}

}

void linear_set::do_compute_stabilizer(
	int level, int orbit_at_level,
	int *candidates, int nb_candidates, 
	strong_generators *&strong_gens, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_set::do_compute_stabilizer" << endl;
		}

	int t0 = os_ticks();
	
	if (f_v) {
		cout << "linear_set::do_compute_stabilizer "
				"calling generator_main" << endl;
		cout << "A=";
		Gen_stab->Poset->A->print_info();
		cout << "A2=";
		Gen_stab->Poset->A2->print_info();
		}
	Gen_stab->main(t0, 
		Gen_stab->depth, 
		f_use_invariant_subset_if_available, 
		//f_lex, 
		f_debug, 
		verbose_level - 1);
	
	int nb_orbits;
	
	if (f_v) {
		cout << "linear_set::do_compute_stabilizer "
				"done with generator_main" << endl;
		}
	nb_orbits = Gen_stab->nb_orbits_at_level(Gen_stab->depth);
	if (f_v) {
		cout << "linear_set::do_compute_stabilizer we found "
				<< nb_orbits << " orbits at depth "
				<< Gen_stab->depth << endl;
		}

	int *set1;
	int *set2;
	int *set3;
	int *Basis1;
	int *Basis2;
	int i, h, orbit;
	
	set1 = NEW_int(level);
	set2 = NEW_int(level);
	set3 = NEW_int(level);
	Basis1 = NEW_int(level * n);
	Basis2 = NEW_int(level * n);


	Gen->get_set_by_level(level, orbit_at_level, set1);
	for (i = 0; i < level; i++) {
		Fq->PG_element_unrank_modified(Basis1 + i * n, 1, n, set1[i]);
		}
	cout << "set1: ";
	int_vec_print(cout, set1, level);
	cout << endl;
	cout << "Basis1:" << endl;
	int_matrix_print(Basis1, level, n);

	
	int *linear_set;
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

	aut_gens->init(Aq);
	aut_gens->allocate(Strong_gens_previous->gens->len);
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
		int_vec_print(cout, set2, level);
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
			//f_lex, 
			0 /*verbose_level */);

		if (orbit == orbit_at_level) {
			if (f_v) {
				cout << "linear_set::do_compute_stabilizer orbit "
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
			aut_gens->append(Elt1);
			strong_generators *Strong_gens_next;

			Gen_stab->get_stabilizer_generators(Strong_gens_next,  
				level, h, verbose_level);

			for (i = 0; i < Strong_gens_next->gens->len; i++) {
				aut_gens->append(Strong_gens_next->gens->ith(i));
				}
			delete Strong_gens_next;
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
	Aut = create_sims_from_generators_with_target_group_order_int(Aq, 
		aut_gens, target_go, verbose_level);
	cout << "Stabilizer created successfully" << endl;
	
	//strong_generators *Aut_gens;

	//Aut_gens = NEW_OBJECT(strong_generators);

	strong_gens = NEW_OBJECT(strong_generators);

	strong_gens->init_from_sims(Aut, 0);
	cout << "Generators for the stabilizer of order "
			<< target_go << " are:" << endl;
	strong_gens->print_generators();

	vector_ge *gensQ;

	
	retract_generators(strong_gens->gens, gensQ,
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


	FREE_int(linear_set);
	FREE_OBJECT(gensQ);

	FREE_int(set1);
	FREE_int(set2);
	FREE_int(set3);
	FREE_int(Basis1);
	FREE_int(Basis2);
	FREE_int(Intersection_dimensions);
	FREE_int(Elt1);
	FREE_OBJECT(Strong_gens_previous);

	if (f_v) {
		cout << "linear_set::do_classify_secondary done" << endl;
		}
}


// #############################################################################
// global functions:
// #############################################################################


int linear_set_rank_point_func(int *v, void *data)
{
	linear_set *LS;
	int rk;
	
	LS = (linear_set *) data;
	AG_element_rank(LS->Fq->q, v, 1, LS->vector_space_dimension, rk);
	//PG_element_rank_modified(*LS->Fq, v, 1,
	//LS->vector_space_dimension, rk);
	return rk;
}

void linear_set_unrank_point_func(int *v, int rk, void *data)
{
	linear_set *LS;
	
	LS = (linear_set *) data;
	AG_element_unrank(LS->Fq->q, v, 1, LS->vector_space_dimension, rk);
	//PG_element_unrank_modified(*LS->Fq, v, 1,
	//LS->vector_space_dimension, rk);
}

void linear_set_early_test_func(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level)
{
	verbose_level = 2;

	linear_set *LS;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	LS = (linear_set *) data;

	if (f_v) {
		cout << "linear_set_early_test_func" << endl;
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
		cout << "linear_set_early_test_func" << endl;
		cout << "Out of " << nb_candidates << " candidates, "
				<< nb_good_candidates << " survive" << endl;
		}
}

void linear_set_secondary_early_test_func(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level)
{
	//verbose_level = 1;

	linear_set *LS;
	int f_v = (verbose_level >= 1);
	int i;

	LS = (linear_set *) data;

	if (f_v) {
		cout << "linear_set_secondary_early_test_func" << endl;
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
		cout << "linear_set_secondary_early_test_func" << endl;
		cout << "Out of " << nb_candidates << " candidates, "
				<< nb_good_candidates << " survive" << endl;
		}
}




