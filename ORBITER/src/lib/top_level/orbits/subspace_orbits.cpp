// subspace_orbits.C
// 
// Anton Betten
//
// started:    January 25, 2010
// moved here: March 29, 2012
// 
//
//

#include "orbiter.h"


subspace_orbits::subspace_orbits()
{
	LG = NULL;
	tmp_M = NULL;
	tmp_M2 = NULL;
	tmp_M3 = NULL;
	base_cols = NULL;
	v = NULL;
	w = NULL;
	weights = NULL;
	Gen = NULL;
	
	f_print_generators = FALSE;
	f_has_extra_test_func = FALSE;
}

subspace_orbits::~subspace_orbits()
{
	if (tmp_M) {
		FREE_INT(tmp_M);
		}
	if (tmp_M2) {
		FREE_INT(tmp_M2);
		}
	if (tmp_M3) {
		FREE_INT(tmp_M3);
		}
	if (base_cols) {
		FREE_INT(base_cols);
		}
	if (v) {
		FREE_INT(v);
		}
	if (w) {
		FREE_INT(w);
		}
	if (weights) {
		FREE_INT(weights);
		}
	if (Gen) {
		FREE_OBJECT(Gen);
		}
}

void subspace_orbits::init(int argc, const char **argv, 
	linear_group *LG, INT depth, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subspace_orbits::init" << endl;
		}

	subspace_orbits::LG = LG;
	subspace_orbits::depth = depth;
	n = LG->vector_space_dimension;
	F = LG->F;
	q = F->q;

	if (f_v) {
		cout << "subspace_orbits::init "
				"n=" << n << " q=" << q << endl;
		}


	
	tmp_M = NEW_INT(n * n);
	tmp_M2 = NEW_INT(n * n);
	tmp_M3 = NEW_INT(n * n);
	base_cols = NEW_INT(n);
	v = NEW_INT(n);
	w = NEW_INT(n);
	weights = NEW_INT(n + 1);
	Gen = NEW_OBJECT(poset_classification);

	if (f_v) {
		cout << "subspace_orbits::init "
				"before Gen->read_arguments" << endl;
		}

	Gen->read_arguments(argc, argv, 0);

	if (f_v) {
		cout << "subspace_orbits::init "
				"after Gen->read_arguments" << endl;
		}

	if (f_v) {
		cout << "subspace_orbits::init "
				"LG->prefix=" << LG->prefix << endl;
		}

	sprintf(Gen->fname_base, "%s", LG->prefix);
	
	
	Gen->depth = depth;

	if (f_v) {
		cout << "subspace_orbits::init "
				"before init_group" << endl;
		}

	init_group(verbose_level);

	if (f_v) {
		cout << "subspace_orbits::init "
				"after init_group" << endl;
		}


	if (f_v) {
		cout << "subspace_orbits::init done" << endl;
		}
}


void subspace_orbits::init_group(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "subspace_orbits::init_group" << endl;
		}

	if (f_print_generators) {
		INT f_print_as_permutation = FALSE;
		INT f_offset = TRUE;
		INT offset = 1;
		INT f_do_it_anyway_even_for_big_degree = TRUE;
		INT f_print_cycles_of_length_one = TRUE;
		
		cout << "subspace_orbits->init_group printing generators "
				"for the group:" << endl;
		LG->Strong_gens->gens->print(cout, f_print_as_permutation, 
			f_offset, offset, 
			f_do_it_anyway_even_for_big_degree, 
			f_print_cycles_of_length_one);
		}

	if (f_v) {
		cout << "subspace_orbits->init_group before Gen->init" << endl;
		}

	
	Gen->init(LG->A_linear,
			LG->A2, LG->Strong_gens, Gen->depth, verbose_level);

#if 0
	Gen->init_check_func(
		subspace_orbits_test_func, 
		this /* candidate_check_data */);
#endif
	Gen->init_early_test_func(
		subspace_orbits_early_test_func, 
		this /*void *data */,  
		verbose_level);

	//Gen->init_incremental_check_func(
		//check_mindist_incremental, 
		//this /* candidate_check_data */);

	Gen->init_vector_space_action(n, 
		F, 
		subspace_orbits_rank_point_func, 
		subspace_orbits_unrank_point_func, 
		this, 
		verbose_level);
#if 0
	Gen->f_print_function = TRUE;
	Gen->print_function = print_set;
	Gen->print_function_data = this;
#endif	

	INT nb_poset_orbit_nodes = 1000;
	
	if (f_v) {
		cout << "subspace_orbits->init_group "
				"before Gen->init_poset_orbit_node" << endl;
		}
	Gen->init_poset_orbit_node(nb_poset_orbit_nodes, verbose_level - 1);
	if (f_v) {
		cout << "subspace_orbits->init_group "
				"calling Gen->init_root_node" << endl;
		}
	Gen->root[0].init_root_node(Gen, verbose_level - 1);
	
	schreier_depth = Gen->depth;
	f_use_invariant_subset_if_available = FALSE;
	f_implicit_fusion = FALSE;
	f_debug = FALSE;
	if (f_v) {
		cout << "subspace_orbits->init_group done" << endl;
		}
}



void subspace_orbits::compute_orbits(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT t0 = os_ticks();
	
	if (f_v) {
		cout << "subspace_orbits::compute_orbits "
				"calling generator_main" << endl;
		cout << "A=";
		Gen->A->print_info();
		cout << "A2=";
		Gen->A2->print_info();
		}
	Gen->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		verbose_level - 1);
	
	INT nb_orbits;
	
	if (f_v) {
		cout << "subspace_orbits::compute_orbits "
				"done with generator_main" << endl;
		}
	nb_orbits = Gen->nb_orbits_at_level(depth);
	if (f_v) {
		cout << "subspace_orbits::compute_orbits we found "
				<< nb_orbits << " orbits at depth " << depth << endl;
		}
}

void subspace_orbits::unrank_set_to_M(INT len, INT *S)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		PG_element_unrank_modified(*F, tmp_M + i * n, 1, n, S[i]);
		}
}

void subspace_orbits::unrank_set_to_matrix(INT len, INT *S, INT *M)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		PG_element_unrank_modified(*F, M + i * n, 1, n, S[i]);
		}
}

void subspace_orbits::rank_set_from_matrix(INT len, INT *S, INT *M)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		PG_element_rank_modified(*F, M + i * n, 1, n, S[i]);
		}
}

void subspace_orbits::Kramer_Mesner_matrix(INT t, INT k,
	INT f_print_matrix,
	INT f_read_solutions, const char *solution_fname,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "subspace_orbits::Kramer_Mesner_matrix "
				"t=" << t << " k=" << k << ":" << endl;
		}


	matrix Mtk;
	INT m, n;

	compute_Kramer_Mesner_matrix(Gen, 
		t, k, Mtk, TRUE /* f_subspaces */, q, verbose_level - 2);
		// in DISCRETA/discreta_global.C

	m = Mtk.s_m();
	n = Mtk.s_n();

	if (f_v) {
		cout << "The Kramer Mesner matrix has size "
				<< m << " x " << n << endl;
		//cout << Mtk << endl;
		}

	if (f_print_matrix) {
		cout << "The Kramer Mesner matrix has size "
				<< m << " x " << n << endl;
		cout << Mtk << endl;
		}

	if (f_v) {
		cout << "creating diophant:" << endl;
		}

	diophant *D;

	matrix_to_diophant(Mtk, D, verbose_level);

	INT l;
	INT i, row;
	INT *Len;
	
	l = Gen->nb_orbits_at_level(k);

	Len = NEW_INT(l);
	for (i = 0; i < l; i++) {
		Len[i] = Gen->orbit_length_as_INT(i, k);
		}

	cout << "Orbit lengths: ";
	INT_vec_print(cout, Len, l);
	cout << endl;

	{
	classify C;

	C.init(Len, l, FALSE, 0);
	cout << "classification of orbit lengths:" << endl;
	C.print_naked(TRUE /* f_backwards*/);
	cout << endl;
	}
	
	row = D->m;

	D->append_equation();

	for (i = 0; i < l; i++) {
		D->Aij(row, i) = Len[i];
		}
	D->type[row] = t_EQ;
	D->RHSi(row) = 0;

	D->sum = 0;
	D->f_x_max = TRUE;
	for (i = 0; i < n; i++) {
		D->x_max[i] = 1;
		}
	
	
	
	if (f_v) {
		cout << "diophant has been created" << endl;
		}


	if (f_read_solutions) {

		if (f_v) {
			cout << "reading solutions from file "
					<< solution_fname << endl;
			}
		D->read_solutions_from_file(solution_fname, verbose_level);
		INT *Sol;
		INT nb_sol;
		D->get_solutions(Sol, nb_sol, verbose_level);
		if (f_v) {
			cout << "we found " << nb_sol << " solutions in file "
					<< solution_fname << endl;
			}

		INT **Subspace_ranks;
		INT *Subspace_ranks1;
		INT nb_subspaces;
		INT j;

		Subspace_ranks = NEW_PINT(nb_sol);
		
		print_all_solutions(D, k, Sol, nb_sol,
				Subspace_ranks, nb_subspaces, verbose_level);

		cout << "Solutions by subspace ranks:" << endl;
		for (i = 0; i < nb_sol; i++) {
			cout << i << " / " << nb_sol << " : ";
			for (j = 0; j < nb_subspaces; j++) {
				cout << Subspace_ranks[i][j];
				if (j < nb_subspaces) {
					cout << ", ";
					}
				}
			cout << endl;
			}
		cout << "each solution consists of " << nb_subspaces
				<< " subspaces" << endl;
		Subspace_ranks1 = NEW_INT(nb_sol * nb_subspaces);
		for (i = 0; i < nb_sol; i++) {
			for (j = 0; j < nb_subspaces; j++) {
				Subspace_ranks1[i * nb_subspaces + j] =
						Subspace_ranks[i][j];
				}
			}
		char fname[1000];
		
		strcpy(fname, solution_fname);
		replace_extension_with(fname, "_designs.csv");
		INT_matrix_write_csv(fname, Subspace_ranks1,
				nb_sol, nb_subspaces);
		cout << "Written file " << fname << " of size "
				<< file_size(fname) << endl;
		
		FREE_INT(Subspace_ranks1);
		FREE_INT(Sol);
		}
	else {
	


		char fname[1000];

		sprintf(fname, "%s_KM_%ld_%ld.system", Gen->fname_base, t, k);
		cout << "saving diophant under the name " << fname << endl;
		D->save_in_general_format(fname, verbose_level);

#if 0

		cout << "eqn 6:" << endl;
			for (j = 0; j < nb_cols; j++) {
				if (Mtk.s_iji(6, j)) {
					cout << Mtk.s_iji(6, j) << " in col " << j << " : ";
					}
				}
		cout << endl;
#endif
		}


	FREE_INT(Len);
	cout << "closing diophant:" << endl;
	FREE_OBJECT(D);
	
	
#if 0
	Mtk_sup_to_inf(Gen, t, k, Mtk, Mtk_inf, verbose_level - 2);
	cout << "M_{" << t << "," << k << "} inf has been computed" << endl;
	//Mtk_inf.print(cout);
	
	//cout << endl;
	//cout << endl;
	
	INT nb_t_orbits;
	INT nb_k_orbits;
	INT first_t, first_k;
	INT len, rep, size;
	INT set1[1000];
	//INT set2[1000];
	
	first_t = Gen->first_oracle_node_at_level[t];
	first_k = Gen->first_oracle_node_at_level[k];
	nb_t_orbits = Mtk_inf.s_m();
	nb_k_orbits = Mtk_inf.s_n();
	for (j = 0; j < nb_k_orbits; j++) {
		cout << "   ";
		}
	cout << "| ";
	cout << " t-orbit orbit_lenth" << endl;
	for (i = 0; i < nb_t_orbits; i++) {
		len = Gen->orbit_length_as_INT(i, t);
		//cout << "i=" << i << " len=" << len << endl;
		Gen->get_set(first_t + i, set1, size);
		if (size != t) {
			cout << "size != t" << endl;
			exit(1);
			}
		for (j = 0; j < nb_k_orbits; j++) {
			a = Mtk_inf.s_iji(i, j);
			cout << setw(2) << a << " ";
			}
		cout << "| ";
		cout << setw(3) << i << " " << setw(3) << len << " ";
		if (t == 1) {
			rep = set1[0];
			schreier Schreier;

			Schreier.init(Gen->A2);
			Schreier.init_generators_by_hdl(
					Gen->root[0].nb_strong_generators,
					Gen->root[0].hdl_strong_generators,
					verbose_level - 1);
			Schreier.compute_point_orbit(rep, 0 /* verbose_level */);
			if (Schreier.orbit_len[0] != len) {
				cout << "Schreier.orbit_len[0] != len" << endl;
				exit(1);
				}
			INT *pts;
			INT len1;

			pts = NEW_INT(len);
			Schreier.get_orbit(0 /* orbit_idx */,
					pts, len1, 0 /* verbose_level */);
			
			//cout << "{";
			INT_vec_print(cout, pts, len);
			//cout << "}";
			FREE_INT(pts);
			}
		cout << endl;
		}
	cout << "t-orbits, t=" << t << " :" << endl;
	cout << "i : orbit_length of i-th orbit" << endl;
	for (i = 0; i < nb_t_orbits; i++) {
		len = Gen->orbit_length_as_INT(i, t);
		cout << i << " : " << len << endl;
		}
	cout << "k-orbits, k=" << k << " :" << endl;
	cout << "i : orbit_length of i-th orbit" << endl;
	for (i = 0; i < nb_k_orbits; i++) {
		len = Gen->orbit_length_as_INT(i, k);
		cout << i << " : " << len << endl;
		}
#endif
}

void subspace_orbits::print_all_solutions(
	diophant *D, INT k, INT *Sol, INT nb_sol,
	INT **Subspace_ranks, INT &nb_subspaces, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, nb1;

	if (f_v) {
		cout << "subspace_orbits::print_all_solutions" << endl;
		}
	for (i = 0; i < nb_sol; i++) {
		cout << "solution " << i << " / " << nb_sol << " : ";
		print_one_solution(D, k, Sol + i * D->sum,
				Subspace_ranks[i], nb1, verbose_level);
		if (i == 0) {
			nb_subspaces = nb1;
			}
		else {
			if (nb1 != nb_subspaces) {
				cout << "subspace_orbits::print_all_solutions "
						"nb1 != nb_subspaces" << endl;
				exit(1);
				}
			}
		}
}

void subspace_orbits::print_one_solution(
	diophant *D, INT k, INT *sol,
	INT *&subspace_ranks, INT &nb_subspaces,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, orbit_idx, len, rank, h, rk, cnt;
	INT *set;
	INT *M;
	grassmann *Gr;

	if (f_v) {
		cout << "subspace_orbits::print_one_solution" << endl;
		}

	set = NEW_INT(k);
	M = NEW_INT(k * Gen->vector_space_dimension);
	Gr = NEW_OBJECT(grassmann);
	Gr->init(Gen->vector_space_dimension, k, Gen->F,
			0/*verbose_level - 10*/);

	INT_vec_print(cout, sol, D->sum);
	cout << endl;

	nb_subspaces = 0;
	for (i = 0; i < D->sum; i++) {
		orbit_idx = sol[i];
		len = Gen->orbit_length_as_INT(orbit_idx, k);
		nb_subspaces += len;
		}
	cout << "nb_subspaces = " << nb_subspaces << endl;
	
	subspace_ranks = NEW_INT(nb_subspaces);

	cnt = 0;
	for (i = 0; i < D->sum; i++) {
		orbit_idx = sol[i];
		len = Gen->orbit_length_as_INT(orbit_idx, k);
		cout << "orbit " << orbit_idx << " of size " << len << endl;
		for (rank = 0; rank < len; rank++) {
			Gen->orbit_element_unrank(k, orbit_idx,
					rank, set, 0 /* verbose_level */);
			cout << rank << " / " << len << " : ";
			INT_vec_print(cout, set, k);
			cout << endl;
			for (h = 0; h < k; h++) {
				Gen->unrank_point(M + h * Gen->vector_space_dimension,
						set[h]);
				}
			cout << "generator matrix:" << endl;
			INT_matrix_print(M, k, Gen->vector_space_dimension);
			rk = Gr->rank_INT_here(M, 0);
			cout << "rank = " << rk << endl;
			
			subspace_ranks[cnt++] = rk;
			}
		}

	FREE_OBJECT(Gr);
	FREE_INT(M);
	FREE_INT(set);
}

INT subspace_orbits::test_dim_C_cap_Cperp_property(
		INT len, INT *S, INT d)
{
	//INT i;
	
#if 0
	cout << "subspace_orbits::test_dim_C_Cperp_property" << endl;
	cout << "Set ";
	INT_vec_print(cout, S, len);
	cout << endl;
#endif

	unrank_set_to_M(len, S);

#if 0
	cout << "coordinate matrix:" << endl;
	print_integer_matrix_width(cout, tmp_M, len, n, n, F->log10_of_q);
#endif

	INT k = len;
	INT k3;


	F->perp_standard_with_temporary_data(n, k, tmp_M, 
		tmp_M2, tmp_M3, base_cols, 
		0 /*verbose_level*/);

	//cout << "C perp:" << endl;
	//print_integer_matrix_width(cout, tmp_M + k * n,
	// n - k, n, n, P->F->log10_of_q);


	F->intersect_subspaces(n, k, tmp_M, n - k, tmp_M + k * n, 
		k3, tmp_M2, 0 /*verbose_level*/);

	//cout << "\\dim (C \\cap C^\\bot) = " << k3 << endl;

	//cout << "basis for C \\cap C^\\bot:" << endl;
	//print_integer_matrix_width(cout, tmp_M2,
	// k3, n, n, P->F->log10_of_q);

	if (k3 == d) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}

INT subspace_orbits::compute_minimum_distance(INT len, INT *S)
{
	INT i, d = 0;
	
#if 0
	cout << "subspace_orbits::compute_minimum_distance" << endl;
	cout << "Set ";
	INT_vec_print(cout, S, len);
	cout << endl;
#endif

	unrank_set_to_M(len, S);

#if 0
	cout << "coordinate matrix:" << endl;
	print_integer_matrix_width(cout, tmp_M, len, n, n, P->F->log10_of_q);
#endif


	F->code_projective_weight_enumerator(n, len, 
		tmp_M, // [k * n]
		weights, 
		0 /*verbose_level*/);


	//cout << "projective weights: " << endl;
	for (i = 1; i <= n; i++) {
		if (weights[i]) {
			d = i;
			break;
			}
		}
	return d;
}

void subspace_orbits::print_set(INT len, INT *S)
{
	INT i;
	
	cout << "subspace_orbits::print_set" << endl;
	cout << "Set ";
	INT_vec_print(cout, S, len);
	cout << endl;

	unrank_set_to_M(len, S);

	cout << "coordinate matrix after unrank:" << endl;
	print_integer_matrix_width(cout,
			tmp_M, len, n, n, F->log10_of_q);


	F->Gauss_easy(tmp_M, len, n);

	cout << "coordinate matrix in RREF:" << endl;
	print_integer_matrix_width(cout,
			tmp_M, len, n, n, F->log10_of_q);

	
	cout << "\\left[" << endl;
	INT_matrix_print_tex(cout, tmp_M, len, n);
	cout << "\\right]" << endl;


	if (len) {
		F->code_projective_weight_enumerator(n, len, 
			tmp_M, // [k * n]
			weights, // [n + 1]
			0 /*verbose_level*/);


		cout << "projective weights: " << endl;
		for (i = 0; i <= n; i++) {
			if (weights[i] == 0) {
				continue;
				}
			cout << i << " : " << weights[i] << endl;
			}
		}

	INT k = len;
	INT k3;


	F->perp_standard_with_temporary_data(n, k, tmp_M, 
		tmp_M2, tmp_M3, base_cols, 
		0 /*verbose_level*/);

	cout << "C perp:" << endl;
	print_integer_matrix_width(cout,
			tmp_M + k * n, n - k, n, n, F->log10_of_q);

	cout << "\\left[" << endl;
	INT_matrix_print_tex(cout, tmp_M + k * n, n - k, n);
	cout << "\\right]" << endl;

	INT *S1;
	INT *canonical_subset;
	INT *transporter;
	INT *M2;
	INT local_idx, global_idx;
	//INT f_implicit_fusion = FALSE;

	S1 = NEW_INT(n - k);
	canonical_subset = NEW_INT(n - k);
	transporter = NEW_INT(Gen->A->elt_size_in_INT);
	M2 = NEW_INT((n - k) * n);
	rank_set_from_matrix(n - k, S1, tmp_M + k * n);

	cout << "ranks of rows of the dual:" << endl;
	INT_vec_print(cout, S1, n - k);
	cout << endl;


	if (n - k > 0 && n - k <= depth && FALSE) {
		cout << "before Gen->trace_set" << endl;
		local_idx = Gen->trace_set(S1, n - k, n - k, 
			canonical_subset, transporter, 
			//f_implicit_fusion, 
			0 /*verbose_level - 3*/);

		global_idx = Gen->first_poset_orbit_node_at_level[n - k] + local_idx;
		cout << "after Gen->trace_set" << endl;
		cout << "local_idx = " << local_idx << endl;
		cout << "global_idx = " << global_idx << endl;
		cout << "canonical set:" << endl;
		INT_vec_print(cout, canonical_subset, n - k);
		cout << endl;

		unrank_set_to_matrix(n - k, canonical_subset, M2);
		cout << "C perp canonical:" << endl;
		print_integer_matrix_width(cout, M2, n - k, n, n, F->log10_of_q);

		cout << "\\left[" << endl;
		INT_matrix_print_tex(cout, M2, n - k, n);
		cout << "\\right]" << endl;

		cout << "transporter:" << endl;
		Gen->A->element_print_quick(transporter, cout);
		cout << "transporter in latex:" << endl;
		Gen->A->element_print_latex(transporter, cout);
		}

	FREE_INT(S1);
	FREE_INT(canonical_subset);
	FREE_INT(transporter);
	FREE_INT(M2);
	

	F->intersect_subspaces(n, k, tmp_M, n - k, tmp_M + k * n, 
		k3, tmp_M2, 0 /*verbose_level*/);

	cout << "\\dim (C \\cap C^\\bot) = " << k3 << endl;

	cout << "basis for C \\cap C^\\bot:" << endl;
	print_integer_matrix_width(cout, tmp_M2, k3, n, n, F->log10_of_q);
	
	
}

INT subspace_orbits::test_set(INT len, INT *S, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT ret = TRUE;
	INT rk;
	
	if (f_v) {
		cout << "subspace_orbits::test_set" << endl;
		cout << "Testing set ";
		INT_vec_print(cout, S, len);
		cout << endl;
		}
	unrank_set_to_M(len, S);
	if (f_vv) {
		cout << "coordinate matrix:" << endl;
		print_integer_matrix_width(cout,
				tmp_M, len, n, n, F->log10_of_q);
		}
	rk = F->Gauss_simple(tmp_M, len, n,
			base_cols, 0 /*verbose_level - 2*/);
	if (f_v) {
		cout << "the matrix has rank " << rk << endl;
		}
	if (rk < len) {
		ret = FALSE;
		}
	if (ret) {
		if (f_has_extra_test_func) {
			ret = (*extra_test_func)(this,
					len, S, extra_test_func_data, verbose_level);
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

INT subspace_orbits::test_minimum_distance(
		INT len, INT *S, INT mindist, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT ret = TRUE;
	INT i, h, wt, N, k;
	INT *msg;
	INT *word;
	INT *M;
	
	if (f_v) {
		cout << "subspace_orbits::test_minimum_distance" << endl;
		cout << "Testing set ";
		INT_vec_print(cout, S, len);
		cout << endl;
		}
	k = len;
	M = tmp_M;
	unrank_set_to_M(len, S);
	if (f_vv) {
		cout << "coordinate matrix:" << endl;
		print_integer_matrix_width(cout,
				M, len, n, n, F->log10_of_q);
		}
	N = nb_PG_elements(k - 1, q);
	msg = v;
	word = w;
	for (h = 0; h < N; h++) {
		PG_element_unrank_modified(*F, msg, 1, k, h);
		//AG_element_unrank(q, msg, 1, k, h);
		F->mult_vector_from_the_left(msg, M, word, k, n);
		wt = 0;
		for (i = 0; i < n; i++) {
			if (word[i]) {
				wt++;
				}
			}
		if (wt < mindist) {
			ret = FALSE;
			break;
			}
		}
	if (f_v) {
		if (ret) {
			cout << "is OK" << endl;
			}
		else {
			cout << "is not OK" << endl;
			}
		}
	return ret;
}

INT subspace_orbits::test_if_self_orthogonal(
		INT len, INT *S, INT f_doubly_even,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT ret;
	INT i, j, a, wt;
	INT *M;
	
	if (f_v) {
		cout << "subspace_orbits::test_if_self_orthogonal" << endl;
		cout << "f_doubly_even=" << f_doubly_even << endl;
		cout << "Testing set ";
		INT_vec_print(cout, S, len);
		cout << endl;
		}
	M = tmp_M;

	unrank_set_to_M(len, S);
	if (f_vv) {
		cout << "coordinate matrix:" << endl;
		print_integer_matrix_width(cout, M, len, n, n, F->log10_of_q);
		}

	ret = TRUE;
	if (f_doubly_even) {
		if (F->q != 2) {
			cout << "subspace_orbits::test_if_self_orthogonal "
					"doubly_even needs q = 2" << endl;
			exit(1);
			}
		/// check if each row of the generator matrix has weight
		/// divisible by 4:
		for (i = 0; i < len; i++) {
			wt = 0;
			for (j = 0; j < n; j++) {
				if (M[i * n + j]) {
					wt++;
					}
				}
			if (wt % 4) {
				ret = FALSE;
				break;
				}
			}
		}
	if (ret) {
		for (i = 0; i < len; i++) {
			for (j = i; j < len; j++) {
				a = F->dot_product(n, M + i * n, M + j * n);
				if (a) {
					ret = FALSE;
					break;
					}
				}
			if (j < len) {
				break;
				}
			}
		}
	if (f_v) {
		if (ret) {
			cout << "is OK" << endl;
			}
		else {
			cout << "is not OK" << endl;
			}
		}
	return ret;
}




// #############################################################################
// global functions:
// #############################################################################


INT subspace_orbits_rank_point_func(INT *v, void *data)
{
	subspace_orbits *G;
	poset_classification *gen;
	INT rk;
	
	G = (subspace_orbits *) data;
	gen = G->Gen;
	PG_element_rank_modified(*gen->F, v, 1,
			gen->vector_space_dimension, rk);
	return rk;
}

void subspace_orbits_unrank_point_func(INT *v, INT rk, void *data)
{
	subspace_orbits *G;
	poset_classification *gen;
	
	G = (subspace_orbits *) data;
	gen = G->Gen;
	PG_element_unrank_modified(*gen->F, v, 1,
			gen->vector_space_dimension, rk);
}

void subspace_orbits_early_test_func(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level)
{
	//verbose_level = 1;

	subspace_orbits *SubOrb;
	INT f_v = (verbose_level >= 1);
	INT i;

	SubOrb = (subspace_orbits *) data;

	if (f_v) {
		cout << "subspace_orbits_early_test_func" << endl;
		cout << "testing " << nb_candidates << " candidates" << endl;
		}
	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		S[len] = candidates[i];
		if (SubOrb->test_set(len + 1, S, verbose_level - 1)) {
			good_candidates[nb_good_candidates++] = candidates[i];
			}
		}
	if (f_v) {
		cout << "subspace_orbits_early_test_func" << endl;
		cout << "Out of " << nb_candidates << " candidates, "
				<< nb_good_candidates << " survive" << endl;
		}
}




