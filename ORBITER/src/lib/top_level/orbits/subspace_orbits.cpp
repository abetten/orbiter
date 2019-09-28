// subspace_orbits.cpp
// 
// Anton Betten
//
// started:    January 25, 2010
// moved here: March 29, 2012
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


subspace_orbits::subspace_orbits()
{
	LG = NULL;
	n = 0;
	F = NULL;
	q = 0;
	depth = 0;

	f_print_generators = FALSE;
	tmp_M = NULL;
	tmp_M2 = NULL;
	tmp_M3 = NULL;
	base_cols = NULL;
	v = NULL;
	w = NULL;
	weights = NULL;

	VS = NULL;
	Poset = NULL;
	Gen = NULL;
	
	schreier_depth = 0;
	f_use_invariant_subset_if_available = FALSE;
	f_implicit_fusion = FALSE;
	f_debug = FALSE;
	f_has_strong_generators = FALSE;
	Strong_gens = NULL;

	f_has_extra_test_func = FALSE;
	extra_test_func = NULL;
	extra_test_func_data = NULL;

	test_dim = 0;

}

subspace_orbits::~subspace_orbits()
{
	if (tmp_M) {
		FREE_int(tmp_M);
		}
	if (tmp_M2) {
		FREE_int(tmp_M2);
		}
	if (tmp_M3) {
		FREE_int(tmp_M3);
		}
	if (base_cols) {
		FREE_int(base_cols);
		}
	if (v) {
		FREE_int(v);
		}
	if (w) {
		FREE_int(w);
		}
	if (weights) {
		FREE_int(weights);
		}
	if (VS) {
		FREE_OBJECT(VS);
	}
	if (Poset) {
		FREE_OBJECT(Poset);
		}
	if (Gen) {
		FREE_OBJECT(Gen);
		}
}

void subspace_orbits::init(
	int argc, const char **argv,
	linear_group *LG, int depth, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

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


	
	tmp_M = NEW_int(n * n);
	tmp_M2 = NEW_int(n * n);
	tmp_M3 = NEW_int(n * n);
	base_cols = NEW_int(n);
	v = NEW_int(n);
	w = NEW_int(n);
	weights = NEW_int(n + 1);
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


void subspace_orbits::init_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "subspace_orbits::init_group" << endl;
		}

	if (f_print_generators) {
		int f_print_as_permutation = FALSE;
		int f_offset = TRUE;
		int offset = 1;
		int f_do_it_anyway_even_for_big_degree = TRUE;
		int f_print_cycles_of_length_one = TRUE;
		
		cout << "subspace_orbits->init_group "
				"printing generators "
				"for the group:" << endl;
		LG->Strong_gens->gens->print(cout,
			f_print_as_permutation,
			f_offset, offset, 
			f_do_it_anyway_even_for_big_degree, 
			f_print_cycles_of_length_one);
		}

	if (f_v) {
		cout << "subspace_orbits->init_group "
				"before Gen->init" << endl;
		}

	VS = NEW_OBJECT(vector_space);
	VS->init(F, n /* dimension */,
			verbose_level - 1);
	VS->init_rank_functions(
			subspace_orbits_rank_point_func,
			subspace_orbits_unrank_point_func,
			this,
			verbose_level - 1);

	Poset = NEW_OBJECT(poset);
	Poset->init_subspace_lattice(LG->A_linear,
			LG->A2, LG->Strong_gens,
			VS,
			verbose_level);
	Poset->add_testing_without_group(
			subspace_orbits_early_test_func,
				this /* void *data */,
				verbose_level);
	
	Gen->init(Poset, Gen->depth, verbose_level);


#if 0
	Gen->f_print_function = TRUE;
	Gen->print_function = print_set;
	Gen->print_function_data = this;
#endif	

	int nb_poset_orbit_nodes = 1000;
	
	if (f_v) {
		cout << "subspace_orbits->init_group "
				"before Gen->init_poset_orbit_node" << endl;
		}
	Gen->init_poset_orbit_node(
			nb_poset_orbit_nodes, verbose_level - 1);
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



void subspace_orbits::compute_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t0 = os_ticks();
	
	if (f_v) {
		cout << "subspace_orbits::compute_orbits "
				"calling generator_main" << endl;
		cout << "A=";
		Gen->Poset->A->print_info();
		cout << "A2=";
		Gen->Poset->A2->print_info();
		}
	Gen->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		verbose_level - 1);
	
	int nb_orbits;
	
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

void subspace_orbits::unrank_set_to_M(
		int len, int *S)
{
	int i;
	
	for (i = 0; i < len; i++) {
		F->PG_element_unrank_modified(tmp_M + i * n, 1, n, S[i]);
		}
}

void subspace_orbits::unrank_set_to_matrix(
		int len, int *S, int *M)
{
	int i;
	
	for (i = 0; i < len; i++) {
		F->PG_element_unrank_modified(M + i * n, 1, n, S[i]);
		}
}

void subspace_orbits::rank_set_from_matrix(
		int len, int *S, int *M)
{
	int i;
	
	for (i = 0; i < len; i++) {
		F->PG_element_rank_modified(M + i * n, 1, n, S[i]);
		}
}

void subspace_orbits::Kramer_Mesner_matrix(
	int t, int k,
	int f_print_matrix,
	int f_read_solutions, const char *solution_fname,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "subspace_orbits::Kramer_Mesner_matrix "
				"t=" << t << " k=" << k << ":" << endl;
		}


	matrix Mtk;
	int m, n;

	compute_Kramer_Mesner_matrix(Gen, 
		t, k, Mtk, TRUE /* f_subspaces */,
		q, verbose_level - 2);

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

	int l;
	int i, row;
	int *Len;
	
	l = Gen->nb_orbits_at_level(k);

	Len = NEW_int(l);
	for (i = 0; i < l; i++) {
		Len[i] = Gen->orbit_length_as_int(i, k);
		}

	cout << "Orbit lengths: ";
	int_vec_print(cout, Len, l);
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

	D->f_has_sum = TRUE;
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
		int *Sol;
		int nb_sol;
		D->get_solutions(Sol, nb_sol, verbose_level);
		if (f_v) {
			cout << "we found " << nb_sol << " solutions in file "
					<< solution_fname << endl;
			}

		int **Subspace_ranks;
		int *Subspace_ranks1;
		int nb_subspaces;
		int j;

		Subspace_ranks = NEW_pint(nb_sol);
		
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
		Subspace_ranks1 = NEW_int(nb_sol * nb_subspaces);
		for (i = 0; i < nb_sol; i++) {
			for (j = 0; j < nb_subspaces; j++) {
				Subspace_ranks1[i * nb_subspaces + j] =
						Subspace_ranks[i][j];
				}
			}
		char fname[1000];
		file_io Fio;
		
		strcpy(fname, solution_fname);
		replace_extension_with(fname, "_designs.csv");
		Fio.int_matrix_write_csv(fname, Subspace_ranks1,
				nb_sol, nb_subspaces);
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		
		FREE_int(Subspace_ranks1);
		FREE_int(Sol);
		}
	else {
	


		char fname[1000];

		sprintf(fname, "%s_KM_%d_%d.system", Gen->fname_base, t, k);
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


	FREE_int(Len);
	cout << "closing diophant:" << endl;
	FREE_OBJECT(D);
	
	
#if 0
	Mtk_sup_to_inf(Gen, t, k, Mtk, Mtk_inf, verbose_level - 2);
	cout << "M_{" << t << "," << k << "} inf has been computed" << endl;
	//Mtk_inf.print(cout);
	
	//cout << endl;
	//cout << endl;
	
	int nb_t_orbits;
	int nb_k_orbits;
	int first_t, first_k;
	int len, rep, size;
	int set1[1000];
	//int set2[1000];
	
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
		len = Gen->orbit_length_as_int(i, t);
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
			int *pts;
			int len1;

			pts = NEW_int(len);
			Schreier.get_orbit(0 /* orbit_idx */,
					pts, len1, 0 /* verbose_level */);
			
			//cout << "{";
			int_vec_print(cout, pts, len);
			//cout << "}";
			FREE_int(pts);
			}
		cout << endl;
		}
	cout << "t-orbits, t=" << t << " :" << endl;
	cout << "i : orbit_length of i-th orbit" << endl;
	for (i = 0; i < nb_t_orbits; i++) {
		len = Gen->orbit_length_as_int(i, t);
		cout << i << " : " << len << endl;
		}
	cout << "k-orbits, k=" << k << " :" << endl;
	cout << "i : orbit_length of i-th orbit" << endl;
	for (i = 0; i < nb_k_orbits; i++) {
		len = Gen->orbit_length_as_int(i, k);
		cout << i << " : " << len << endl;
		}
#endif
}

void subspace_orbits::print_all_solutions(
	diophant *D, int k, int *Sol, int nb_sol,
	int **Subspace_ranks, int &nb_subspaces, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, nb1;

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
	diophant *D, int k, int *sol,
	int *&subspace_ranks, int &nb_subspaces,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, orbit_idx, len, rank, rk, cnt;
	int *set;
	int *M;
	grassmann *Gr;

	if (f_v) {
		cout << "subspace_orbits::print_one_solution" << endl;
		}

	set = NEW_int(k);
	M = NEW_int(k * Gen->Poset->VS->dimension);
	Gr = NEW_OBJECT(grassmann);
	Gr->init(Gen->Poset->VS->dimension, k, Gen->Poset->VS->F,
			0/*verbose_level - 10*/);

	int_vec_print(cout, sol, D->sum);
	cout << endl;

	nb_subspaces = 0;
	for (i = 0; i < D->sum; i++) {
		orbit_idx = sol[i];
		len = Gen->orbit_length_as_int(orbit_idx, k);
		nb_subspaces += len;
		}
	cout << "nb_subspaces = " << nb_subspaces << endl;
	
	subspace_ranks = NEW_int(nb_subspaces);

	cnt = 0;
	for (i = 0; i < D->sum; i++) {
		orbit_idx = sol[i];
		len = Gen->orbit_length_as_int(orbit_idx, k);
		cout << "orbit " << orbit_idx << " of size " << len << endl;
		for (rank = 0; rank < len; rank++) {
			Gen->orbit_element_unrank(k, orbit_idx,
					rank, set, 0 /* verbose_level */);
			cout << rank << " / " << len << " : ";
			int_vec_print(cout, set, k);
			cout << endl;
			Gen->unrank_basis(M, set, k);
#if 0
			for (h = 0; h < k; h++) {
				Gen->unrank_point(M + h * Gen->vector_space_dimension,
						set[h]);
				}
#endif
			cout << "generator matrix:" << endl;
			int_matrix_print(M, k, Gen->Poset->VS->dimension);
			rk = Gr->rank_int_here(M, 0);
			cout << "rank = " << rk << endl;
			
			subspace_ranks[cnt++] = rk;
			}
		}

	FREE_OBJECT(Gr);
	FREE_int(M);
	FREE_int(set);
}

int subspace_orbits::test_dim_C_cap_Cperp_property(
		int len, int *S, int d)
{
	//int i;
	
#if 0
	cout << "subspace_orbits::test_dim_C_Cperp_property" << endl;
	cout << "Set ";
	int_vec_print(cout, S, len);
	cout << endl;
#endif

	unrank_set_to_M(len, S);

#if 0
	cout << "coordinate matrix:" << endl;
	print_integer_matrix_width(cout, tmp_M, len, n, n, F->log10_of_q);
#endif

	int k = len;
	int k3;


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

int subspace_orbits::compute_minimum_distance(int len, int *S)
{
	int i, d = 0;
	
#if 0
	cout << "subspace_orbits::compute_minimum_distance" << endl;
	cout << "Set ";
	int_vec_print(cout, S, len);
	cout << endl;
#endif

	unrank_set_to_M(len, S);

#if 0
	cout << "coordinate matrix:" << endl;
	print_integer_matrix_width(cout,
			tmp_M, len, n, n, P->F->log10_of_q);
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

void subspace_orbits::print_set(ostream &ost, int len, int *S)
{
	latex_interface L;
	int i;
	
	ost << "subspace_orbits::print_set" << endl;
	ost << "Set ";
	int_vec_print(ost, S, len);
	ost << endl;

	unrank_set_to_M(len, S);

	ost << "coordinate matrix after unrank:" << endl;
	print_integer_matrix_width(ost,
			tmp_M, len, n, n, F->log10_of_q);


	F->Gauss_easy(tmp_M, len, n);

	ost << "coordinate matrix in RREF:" << endl;
	print_integer_matrix_width(ost,
			tmp_M, len, n, n, F->log10_of_q);

	
	ost << "\\left[" << endl;
	L.int_matrix_print_tex(ost, tmp_M, len, n);
	ost << "\\right]" << endl;


	if (len) {
		F->code_projective_weight_enumerator(n, len, 
			tmp_M, // [k * n]
			weights, // [n + 1]
			0 /*verbose_level*/);


		ost << "projective weights: " << endl;
		for (i = 0; i <= n; i++) {
			if (weights[i] == 0) {
				continue;
				}
			ost << i << " : " << weights[i] << endl;
			}
		}

	int k = len;
	int k3;


	F->perp_standard_with_temporary_data(n, k, tmp_M, 
		tmp_M2, tmp_M3, base_cols, 
		0 /*verbose_level*/);

	ost << "C perp:" << endl;
	print_integer_matrix_width(ost,
			tmp_M + k * n, n - k, n, n, F->log10_of_q);

	ost << "\\left[" << endl;
	L.int_matrix_print_tex(ost, tmp_M + k * n, n - k, n);
	ost << "\\right]" << endl;

	int *S1;
	int *canonical_subset;
	int *transporter;
	int *M2;
	int local_idx, global_idx;
	//int f_implicit_fusion = FALSE;

	S1 = NEW_int(n - k);
	canonical_subset = NEW_int(n - k);
	transporter = NEW_int(Gen->Poset->A->elt_size_in_int);
	M2 = NEW_int((n - k) * n);
	rank_set_from_matrix(n - k, S1, tmp_M + k * n);

	ost << "ranks of rows of the dual:" << endl;
	int_vec_print(ost, S1, n - k);
	ost << endl;


	if (n - k > 0 && n - k <= depth && FALSE) {
		ost << "before Gen->trace_set" << endl;
		local_idx = Gen->trace_set(S1, n - k, n - k, 
			canonical_subset, transporter, 
			//f_implicit_fusion, 
			0 /*verbose_level - 3*/);

		global_idx =
				Gen->first_poset_orbit_node_at_level[n - k] + local_idx;
		ost << "after Gen->trace_set" << endl;
		ost << "local_idx = " << local_idx << endl;
		ost << "global_idx = " << global_idx << endl;
		ost << "canonical set:" << endl;
		int_vec_print(ost, canonical_subset, n - k);
		ost << endl;

		unrank_set_to_matrix(n - k, canonical_subset, M2);
		ost << "C perp canonical:" << endl;
		print_integer_matrix_width(ost,
				M2, n - k, n, n, F->log10_of_q);

		ost << "\\left[" << endl;
		L.int_matrix_print_tex(ost, M2, n - k, n);
		ost << "\\right]" << endl;

		ost << "transporter:" << endl;
		Gen->Poset->A->element_print_quick(transporter, ost);
		ost << "transporter in latex:" << endl;
		Gen->Poset->A->element_print_latex(transporter, ost);
		}

	FREE_int(S1);
	FREE_int(canonical_subset);
	FREE_int(transporter);
	FREE_int(M2);
	

	F->intersect_subspaces(n, k, tmp_M, n - k, tmp_M + k * n, 
		k3, tmp_M2, 0 /*verbose_level*/);

	ost << "\\dim (C \\cap C^\\bot) = " << k3 << endl;

	ost << "basis for C \\cap C^\\bot:" << endl;
	print_integer_matrix_width(ost,
			tmp_M2, k3, n, n, F->log10_of_q);
	
	
}

int subspace_orbits::test_set(int len, int *S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret = TRUE;
	int rk;
	
	if (f_v) {
		cout << "subspace_orbits::test_set" << endl;
		cout << "Testing set ";
		int_vec_print(cout, S, len);
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

int subspace_orbits::test_minimum_distance(
		int len, int *S, int mindist, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret = TRUE;
	int i, h, wt, N, k;
	int *msg;
	int *word;
	int *M;
	geometry_global Gg;
	
	if (f_v) {
		cout << "subspace_orbits::test_minimum_distance" << endl;
		cout << "Testing set ";
		int_vec_print(cout, S, len);
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
	N = Gg.nb_PG_elements(k - 1, q);
	msg = v;
	word = w;
	for (h = 0; h < N; h++) {
		F->PG_element_unrank_modified(msg, 1, k, h);
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

int subspace_orbits::test_if_self_orthogonal(
		int len, int *S, int f_doubly_even,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret;
	int i, j, a, wt;
	int *M;
	
	if (f_v) {
		cout << "subspace_orbits::test_if_self_orthogonal" << endl;
		cout << "f_doubly_even=" << f_doubly_even << endl;
		cout << "Testing set ";
		int_vec_print(cout, S, len);
		cout << endl;
		}
	M = tmp_M;

	unrank_set_to_M(len, S);
	if (f_vv) {
		cout << "coordinate matrix:" << endl;
		print_integer_matrix_width(cout,
				M, len, n, n, F->log10_of_q);
		}

	ret = TRUE;
	if (f_doubly_even) {
		if (F->q != 2) {
			cout << "subspace_orbits::test_if_self_orthogonal "
					"doubly_even needs q = 2" << endl;
			exit(1);
			}
		// check if each row of the generator matrix has weight
		// divisible by 4:
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


int subspace_orbits_rank_point_func(int *v, void *data)
{
	subspace_orbits *G;
	poset_classification *gen;
	int rk;
	
	G = (subspace_orbits *) data;
	gen = G->Gen;
	gen->Poset->VS->F->PG_element_rank_modified(v, 1,
			gen->Poset->VS->dimension, rk);
	return rk;
}

void subspace_orbits_unrank_point_func(int *v, int rk, void *data)
{
	subspace_orbits *G;
	poset_classification *gen;
	
	G = (subspace_orbits *) data;
	gen = G->Gen;
	gen->Poset->VS->F->PG_element_unrank_modified(v, 1,
			gen->Poset->VS->dimension, rk);
}

void subspace_orbits_early_test_func(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level)
{
	//verbose_level = 1;

	subspace_orbits *SubOrb;
	int f_v = (verbose_level >= 1);
	int i;

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

}}



