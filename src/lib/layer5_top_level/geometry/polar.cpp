// polar.cpp
// 
// Anton Betten
// started: Feb 8, 2010
// moved to DISCRETA: June 1, 2010
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_geometry {


static long int polar_callback_rank_point_func(int *v, void *data);
static void polar_callback_unrank_point_func(int *v, long int rk, void *data);
static void polar_callback_early_test_func(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);


polar::polar()
{
	epsilon = 0;
	n = 0; // vector space dimension
	k = 0;
	q = 0;
	depth = 0;

	f_print_generators = FALSE;

	A = NULL; // the orthogonal action



	Mtx = NULL; // only a copy of a pointer, not to be freed
	O = NULL; // only a copy of a pointer, not to be freed
	F = NULL; // only a copy of a pointer, not to be freed

	tmp_M = NULL; // [n * n]
	base_cols = NULL; // [n]

	VS = NULL;
	Control = NULL;
	Poset = NULL;
	Gen = NULL;

	schreier_depth = 0;
	f_use_invariant_subset_if_available = FALSE;
	f_debug = FALSE;

	f_has_strong_generators = FALSE;
	f_has_strong_generators_allocated = FALSE;
	Strong_gens = NULL;

	first_node = 0;
	nb_orbits = 0;
	nb_elements = 0;
}

polar::~polar()
{
	if (tmp_M) {
		FREE_int(tmp_M);
	}
	if (base_cols) {
		FREE_int(base_cols);
	}
	if (VS) {
		FREE_OBJECT(VS);
	}
	if (Control) {
		FREE_OBJECT(Control);
	}
	if (Poset) {
		FREE_OBJECT(Poset);
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
	int *group_generator_data, int group_generator_size, 
	int f_group_order_target, const char *group_order_target, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures_groups::vector_ge gens;

	if (f_v) {
		cout << "polar::init_group, calling "
				"A->init_group_from_generators_by_base_images" << endl;
		}
	A->init_group_from_generators_by_base_images(
		A->Sims,
		group_generator_data, group_generator_size, 
		f_group_order_target, group_order_target, 
		&gens, Strong_gens, 
		verbose_level);
	f_has_strong_generators = TRUE;
	f_has_strong_generators_allocated = TRUE;
}

void polar::init_group(
	int *group_generator_data, int group_generator_size, 
	int f_group_order_target, const char *group_order_target, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures_groups::vector_ge gens;

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

void polar::init(
		actions::action *A,
		orthogonal_geometry::orthogonal *O,
	int epsilon, int n, int k, field_theory::finite_field *F,
	int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);

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

	
	
	
	tmp_M = NEW_int(n * n);
	base_cols = NEW_int(n);

}

void polar::init2(int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//vector_ge *gens;
	//int *transversal_lengths;
	groups::strong_generators *gens;
	
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
		int f_print_as_permutation = TRUE;
		int f_offset = TRUE;
		int offset = 1;
		int f_do_it_anyway_even_for_big_degree = TRUE;
		int f_print_cycles_of_length_one = FALSE;
		
		cout << "printing generators for the group:" << endl;
		gens->gens->print(cout, f_print_as_permutation, 
			f_offset, offset, 
			f_do_it_anyway_even_for_big_degree, 
			f_print_cycles_of_length_one);
		}
	VS = NEW_OBJECT(algebra::vector_space);

	VS->init(F, n /* dimension */,
			verbose_level - 1);
	VS->init_rank_functions(
			polar_callback_rank_point_func,
			polar_callback_unrank_point_func,
			this,
			verbose_level - 1);

	Control = NEW_OBJECT(poset_classification::poset_classification_control);
	Control->f_depth = TRUE;
	Control->depth = depth;


	char label[1000];

	snprintf(label, sizeof(label), "polar_%d_%d_%d_%d", epsilon, n, k, q); // ToDo



	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
	Poset->init_subspace_lattice(A, A,
			gens,
			VS,
			verbose_level);

	Poset->add_testing_without_group(
			polar_callback_early_test_func,
				this /* void *data */,
				verbose_level);

	Gen = NEW_OBJECT(poset_classification::poset_classification);
	Gen->initialize_and_allocate_root_node(Control, Poset,
			depth /* sz */, verbose_level);


#if 0
	Gen->f_print_function = TRUE;
	Gen->print_function = print_set;
	Gen->print_function_data = this;
#endif	

	schreier_depth = depth;
	f_use_invariant_subset_if_available = FALSE;
	f_debug = FALSE;
}

void polar::compute_orbits(int t0, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "polar::compute_orbits calling generator_main" << endl;
		cout << "A=";
		Gen->get_A()->print_info();
		cout << "A2=";
		Gen->get_A2()->print_info();
		}
	Gen->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		verbose_level - 1);
		
	if (f_v) {
		cout << "done with generator_main" << endl;
		}
	first_node = Gen->first_node_at_level(depth);
	nb_orbits = Gen->nb_orbits_at_level(depth);

	int i;
	nb_elements = 0;
	for (i = 0; i < nb_orbits; i++) {
		nb_elements += Gen->orbit_length_as_int(i, depth);
		}
	if (f_v) {
		cout << "we found " << nb_orbits << " orbits containing "
				<< nb_elements << " elements at depth " << depth << endl;
		}
}

void polar::compute_cosets(int depth, int orbit_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, c, cc, node2, index_int;
	long int *the_set1;
	long int *the_set2;
	int *M1;
	int *M2;
	int *Elt1, *Elt2;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object go1, go2, index, rem, Rank;
	poset_classification::poset_orbit_node *O2;

	if (f_v) {
		cout << "polar::compute_cosets" << endl;
		}
	Elt1 = NEW_int(Gen->get_A()->elt_size_in_int);
	Elt2 = NEW_int(Gen->get_A()->elt_size_in_int);
	the_set1 = NEW_lint(depth);
	the_set2 = NEW_lint(depth);
	M1 = NEW_int(k * n);
	M2 = NEW_int(k * n);
	
	node2 = Gen->first_node_at_level(depth) + orbit_idx;
	O2 = Gen->get_node(node2);

	Gen->stabilizer_order(0, go1);
	Gen->stabilizer_order(node2, go2);
	D.integral_division(go1, go2, index, rem, 0);

	index_int = index.as_int();
	if (f_v) {
		cout << "polar::compute_cosets index=" << index_int << endl;
		}

	O2->store_set_to(Gen, depth - 1, the_set1);
	
	if (f_v) {
		cout << "the set representing orbit " << orbit_idx 
			<< " at level " << depth << " is ";
		Lint_vec_print(cout, the_set1, depth);
		cout << endl;
		}
	for (i = 0; i < k; i++) {
		unrank_point(M1 + i * n, the_set1[i]);
		//polar_callback_unrank_point_func(M1 + i * n, the_set1[i], this);
		}
	if (f_vv) {
		cout << "corresponding to the subspace with basis:" << endl;
		Int_vec_print_integer_matrix_width(cout, M1, k, n, n, F->log10_of_q);
		}

	geometry::grassmann Grass;
	
	Grass.init(n, k, F, 0 /*verbose_level*/);
	for (i = 0; i < k * n; i++) {
		Grass.M[i] = M1[i];
		}
	Grass.rank_longinteger(Rank, 0/*verbose_level - 3*/);
	cout << "Rank=" << Rank << endl;
	

	for (c = 0; c < index_int; c++) {

		//if (!(c == 2 || c == 4)) {continue;}
		if (f_v) {
			cout << "Coset " << c << endl;
			}
		Gen->coset_unrank(depth, orbit_idx, c, Elt1, 0/*verbose_level*/);

		if (f_vvv) {
			cout << "Left coset " << c << " is represented by" << endl;
			Gen->get_A()->element_print_quick(Elt1, cout);
			cout << endl;
			}

		Gen->get_A()->element_invert(Elt1, Elt2, 0);


		if (f_vvv) {
			cout << "Right coset " << c << " is represented by" << endl;
			Gen->get_A()->element_print_quick(Elt2, cout);
			cout << endl;
			}

		for (i = 0; i < k; i++) {
			A->element_image_of_low_level(M1 + i * n, M2 + i * n,
					Elt2, 0/* verbose_level*/);
			}
		if (f_vv) {
			cout << "basis of subspace that is the image "
					"under this element:" << endl;
			Int_vec_print_integer_matrix_width(cout, M2, k, n, n, F->log10_of_q);
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
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_lint(the_set1);
	FREE_lint(the_set2);
	FREE_int(M1);
	FREE_int(M2);
}

void polar::dual_polar_graph(int depth, int orbit_idx, 
		ring_theory::longinteger_object *&Rank_table, int &nb_maximals,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, c, node2, index_int;
	long int *the_set1;
	long int *the_set2;
	int *M1;
	int *M2;
	int *Elt1, *Elt2;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object go1, go2, index, rem, Rank;
	poset_classification::poset_orbit_node *O2;
	int *Adj;
	int **M;
	int witt;
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "polar::dual_polar_graph" << endl;
		}
	Elt1 = NEW_int(Gen->get_A()->elt_size_in_int);
	Elt2 = NEW_int(Gen->get_A()->elt_size_in_int);
	the_set1 = NEW_lint(depth);
	the_set2 = NEW_lint(depth);
	M1 = NEW_int(k * n);
	M2 = NEW_int(k * n);

	witt = Gg.Witt_index(epsilon, n - 1);
		
	node2 = Gen->first_node_at_level(depth) + orbit_idx;
	O2 = Gen->get_node(node2);

	Gen->stabilizer_order(0, go1);
	Gen->stabilizer_order(node2, go2);
	D.integral_division(go1, go2, index, rem, 0);

	index_int = index.as_int();
	if (f_v) {
		cout << "polar::dual_polar_graph index=" << index_int << endl;
		cout << "polar::dual_polar_graph witt=" << witt << endl;
		}

	nb_maximals = index_int;
	Rank_table = NEW_OBJECTS(ring_theory::longinteger_object, index_int);
	Adj = NEW_int(index_int * index_int);
	M = NEW_pint(index_int);

	for (i = 0; i < index_int; i++) {
		M[i] = NEW_int(k * n);
		}
	for (i = 0; i < index_int * index_int; i++) {
		Adj[i] = 0;
		}
	
	O2->store_set_to(Gen, depth - 1, the_set1);
	
	if (f_v) {
		cout << "the set representing orbit " << orbit_idx 
			<< " at level " << depth << " is ";
		Lint_vec_print(cout, the_set1, depth);
		cout << endl;
		}
	for (i = 0; i < k; i++) {
		unrank_point(M1 + i * n, the_set1[i]);
		//polar_callback_unrank_point_func(M1 + i * n, the_set1[i], this);
		}

	if (f_v) {
		cout << "corresponding to the subspace with basis:" << endl;
		Int_vec_print_integer_matrix_width(cout, M1, k, n, n, F->log10_of_q);
		}

	geometry::grassmann Grass;
	
	Grass.init(n, k, F, verbose_level - 2);
	for (i = 0; i < k * n; i++) {
		Grass.M[i] = M1[i];
		}
	Grass.rank_longinteger(Rank, 0/*verbose_level - 3*/);
	cout << "Rank=" << Rank << endl;
	

	for (c = 0; c < index_int; c++) {

		//if (!(c == 2 || c == 4)) {continue;}

		if (FALSE) {
			cout << "Coset " << c << endl;
			}
		Gen->coset_unrank(depth, orbit_idx, c, Elt1, 0/*verbose_level*/);

		if (FALSE) {
			cout << "Left coset " << c << " is represented by" << endl;
			Gen->get_A()->element_print_quick(Elt1, cout);
			cout << endl;
			}

		Gen->get_A()->element_invert(Elt1, Elt2, 0);


		if (FALSE) {
			cout << "Right coset " << c
					<< " is represented by" << endl;
			Gen->get_A()->element_print_quick(Elt2, cout);
			cout << endl;
			}

		for (i = 0; i < k; i++) {
			A->element_image_of_low_level(M1 + i * n,
					M2 + i * n, Elt2, 0/* verbose_level*/);
			}

		F->Linear_algebra->Gauss_easy(M2, k, n);
		
		if (f_vv) {
			cout << "subspace " << c << ":" << endl;
			Int_vec_print_integer_matrix_width(cout, M2, k, n, n, F->log10_of_q);
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
	
	int c1, c2, rk, nb_e, e;
	int *MM;
	int *Inc;
	
	MM = NEW_int(2 * k * n);
	
	nb_e = 0;
	for (c1 = 0; c1 < index_int; c1++) {
		for (c2 = 0; c2 < index_int; c2++) {
			for (i = 0; i < k * n; i++) {
				MM[i] = M[c1][i];
				}
			for (i = 0; i < k * n; i++) {
				MM[k * n + i] = M[c2][i];
				}
			rk = F->Linear_algebra->rank_of_rectangular_matrix(MM,
					2 * k, n, 0 /* verbose_level*/);
			//rk1 = rk - k;
			//Adj[c1 * index_int + c2] = rk1;
			if (rk == k + 1) {
				Adj[c1 * index_int + c2] = 1;
				}
			}
		}

	if (f_vv) {
		cout << "adjacency matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, Adj,
				index_int, index_int, index_int, 1);
		}

	if (f_vv) {
		cout << "neighborhood lists:" << endl;
		for (c1 = 0; c1 < index_int; c1++) {
			cout << "N(" << c1 << ")={";
			for (c2 = 0; c2 < index_int; c2++) {
				if (Adj[c1 * index_int + c2]) {
					cout << c2 << " ";
					}
				}
			cout << "}" << endl;
			}
		}

	nb_e = 0;
	for (c1 = 0; c1 < index_int; c1++) {
		for (c2 = c1 + 1; c2 < index_int; c2++) {
			if (Adj[c1 * index_int + c2]) {
				nb_e++;
				}
			}
		}
	if (f_vv) {
		cout << "with " << nb_e << " edges" << endl;
		}


	Inc = NEW_int(index_int * nb_e);
	for (i = 0; i < index_int * nb_e; i++) {
		Inc[i] = 0;
		}
	
	e = 0;
	for (c1 = 0; c1 < index_int; c1++) {
		for (c2 = c1 + 1; c2 < index_int; c2++) {
			if (Adj[c1 * index_int + c2]) {
				Inc[c1 * nb_e + e] = 1;
				Inc[c2 * nb_e + e] = 1;
				e++;
				}
			}
		}
	if (f_vv) {
		cout << "Incidence matrix:" << index_int
				<< " x " << nb_e << endl;
		Int_vec_print_integer_matrix_width(cout, Inc,
				index_int, nb_e, nb_e, 1);
		}

	{
	char fname[1000];

	snprintf(fname, sizeof(fname), "dual_polar_graph_O_%d_%d_%d.inc", epsilon, n, q);
	{
	ofstream f(fname);
	f << index_int << " " << nb_e << " " << 2 * nb_e << endl;
	for (i = 0; i < index_int * nb_e; i++) {
		if (Inc[i]) {
			f << i << " ";
			}
		}
	f << endl;
	f << -1 << endl;
	}

	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
	}

	FREE_int(Inc);
	FREE_int(MM);

	
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_lint(the_set1);
	FREE_lint(the_set2);
	FREE_int(M1);
	FREE_int(M2);
	FREE_int(Adj);
	for (i = 0; i < index_int; i++) {
		FREE_int(M[i]);
		}
	FREE_pint(M);
}

void polar::show_stabilizer(int depth, int orbit_idx, int verbose_level)
{
	int *Elt;
	long int goi, i, order;
	groups::strong_generators *Strong_gens;

	Elt = NEW_int(A->elt_size_in_int);	

	Gen->get_stabilizer_generators(Strong_gens,  
		depth, orbit_idx, 0 /* verbose_level*/);
	//Gen->get_stabilizer(gens, tl, depth, orbit_idx, verbose_level);

	groups::sims *S;
	S = A->create_sims_from_generators_with_target_group_order_factorized(
		Strong_gens->gens, Strong_gens->tl, A->base_len(),
		verbose_level);
	ring_theory::longinteger_object go;

	S->group_order(go);	
	cout << "polar::show_stabilizer created group of order " << go << endl;
	goi = go.as_int();
	for (i = 0; i < goi; i++) {
		S->element_unrank_lint(i, Elt);
		order = A->element_order(Elt);
		cout << "element " << i << " of order " << order << ":" << endl;
		A->element_print_quick(Elt, cout);
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
		}
	
	FREE_OBJECT(Strong_gens);
	FREE_OBJECT(S);
	FREE_int(Elt);
}

#if 0
void polar::get_maximals(int depth, int orbit_idx, int verbose_level)
{
	int node2;
	poset_orbit_node *O2;
	vector_ge *gens;
	int *tl;
	int *Elt;
	int goi, i, order;

	gens = NEW_OBJECT(vector_ge);
	tl = NEW_int(A->base_len);
	Elt = NEW_int(A->elt_size_in_int);	
	node2 = Gen->first_poset_orbit_node_at_level[depth] + orbit_idx;
	O2 = &Gen->root[node2];
	Gen->get_stabilizer(gens, tl, depth, orbit_idx, verbose_level);

	sims *S;
	S = create_sims_from_generators_with_target_group_order_factorized(
		A, gens, tl, A->base_len, verbose_level);
	longinteger_object go;

	S->group_order(go);	
	cout << "polar::show_stabilizer created group of order " << go << endl;
	goi = go.as_int();
	for (i = 0; i < goi; i++) {
		S->element_unrank_int(i, Elt);
		order = A->element_order(Elt);
		cout << "element " << i << " of order " << order << ":" << endl;
		A->element_print_quick(Elt, cout);
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
		}
	
	FREE_OBJECT(S);
	FREE_OBJECT(gens);
	FREE_int(tl);
	FREE_int(Elt);
}
#endif

#if 0
void polar::compute_Kramer_Mesner_matrix(int t, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "compute_Kramer_Mesner_matrix t=" << t
				<< " k=" << k << ":" << endl;
		}

	// compute Kramer Mesner matrices
	Vector V;
	int i;
	
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
	
	discreta_matrix Mtk, Mtk_inf;
	
	Mtk_from_MM(V, Mtk, t, k, TRUE, q, verbose_level - 2);
	cout << "M_{" << t << "," << k << "} sup:" << endl;
	Mtk.print(cout);
	
	
	Mtk_sup_to_inf(Gen, t, k, Mtk, Mtk_inf, verbose_level - 2);	
	cout << "M_{" << t << "," << k << "} inf:" << endl;
	Mtk_inf.print(cout);
	
}
#endif


#if 0
int polar::test(int *S, int len, int verbose_level)
// test if totally isotropic, i.e. contained in its own perp
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, rk;
	int f_OK = TRUE;

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
#endif

void polar::test_if_in_perp(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, f, c;
	

	if (f_v) {
		cout << "polar::test_if_in_perp done for ";
		orbiter_kernel_system::Orbiter->Lint_vec->set_print(cout, S, len);
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

	O->Hyperbolic_pair->unrank_point(tmp_M + 0 * n, 1, S[len - 1], 0);
	for (i = 0; i < nb_candidates; i++) {
		c = candidates[i];
		O->Hyperbolic_pair->unrank_point(tmp_M + 1 * n, 1, c, 0);
		if (f_vv) {
			cout << "candidate " << i << " = " << c << ":" << endl;
			Int_vec_print_integer_matrix_width(cout,
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
		orbiter_kernel_system::Orbiter->Lint_vec->set_print(cout, S, len);
		cout << "; # of candidates reduced from " << nb_candidates
				<< " to " << nb_good_candidates << endl;
		}
	if (f_vv) {
		cout << "good candidates: ";
		Lint_vec_print(cout, good_candidates, nb_good_candidates);
		cout << endl;
		}
}

void polar::test_if_closed_under_cosets(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, h, c, d, y, f_OK, idx, nb, nb0;
	int *M;
	int *N0;
	int *N;
	int *v;
	int *w;
	int *candidates_expanded;
	int nb_candidates_expanded;
	int *tmp_candidates;
	int nb_tmp_candidates;
	geometry::geometry_global Gg;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "polar::test_if_closed_under_cosets for ";
		Int_vec_set_print(cout, S, len);
		cout << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "candidates: ";
		Int_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		}
	if (len == 0) {
		for (i = 0; i < nb_candidates; i++) {
			good_candidates[i] = candidates[i];
			}
		nb_good_candidates = nb_candidates;
		return;
		}

	nb = Gg.nb_PG_elements(len - 1, F->q);
	if (len >= 2) {
		nb0 = Gg.nb_PG_elements(len - 2, F->q);
		}
	else {
		nb0 = 1;
		}
	M = NEW_int(len * n);
	N0 = NEW_int(nb0 * n);
	N = NEW_int(nb * n);
	v = NEW_int(n);
	w = NEW_int(n);
	candidates_expanded = NEW_int(2 * nb_candidates * (1 + nb0));
	tmp_candidates = NEW_int(2 * nb_candidates * (1 + nb0));
	for (i = 0; i < len; i++) {
		O->Hyperbolic_pair->unrank_point(M + i * n, 1, S[i], 0);
		}
	if (f_v) {
		cout << "the basis is ";
		Int_vec_print(cout, S, len);
		cout << endl;
		cout << "corresponding to the vectors:" << endl;
		Int_vec_print_integer_matrix_width(cout, M, len, n, n, F->log10_of_q);
		cout << "nb=" << nb << endl;
		}
	if (len >= 2) {
		for (i = 0; i < nb0; i++) {
			F->PG_element_unrank_modified(v, 1, len - 1, i);
			F->Linear_algebra->mult_vector_from_the_left(v, M, N0 + i * n, len - 1, n);
			}
		if (f_v) {
			cout << "the list of points N0:" << endl;
			Int_vec_print_integer_matrix_width(cout, N0, nb0, n, n, F->log10_of_q);
			}
		}
	for (i = 0; i < nb; i++) {
		F->PG_element_unrank_modified(v, 1, len, i);
		F->Linear_algebra->mult_vector_from_the_left(v, M, N + i * n, len, n);
		}
	if (f_v) {
		cout << "the list of points N:" << endl;
		Int_vec_print_integer_matrix_width(cout, N, nb, n, n, F->log10_of_q);
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
			if (Sorting.int_vec_search(S, len, c, idx)) {
				continue;
				}
			O->Hyperbolic_pair->unrank_point(v, 1, c, 0);
			if (f_v) {
				cout << "i=" << i;
				Int_vec_print(cout, v, n);
				cout << endl;
				}
			for (j = 0; j < nb0; j++) {
				for (y = 1; y < F->q; y++) {
					for (h = 0; h < n; h++) {
						w[h] = F->add(v[h], F->mult(y, N0[j * n + h]));
						}
					if (f_v) {
						cout << "j=" << j << " y=" << y << " : w=";
						Int_vec_print(cout, w, n);
						cout << endl;
						}
					d = O->Hyperbolic_pair->rank_point(w, 1, 0);
					if (f_v) {
						cout << "d=" << d << endl;
						}
					candidates_expanded[nb_candidates_expanded++] = d;
					} // next y
				} // next j
			} // next i
		if (f_v) {
			cout << "expanded candidate set:" << endl;
			Int_vec_print(cout,
					candidates_expanded, nb_candidates_expanded);
			cout << endl;
			}
		Sorting.int_vec_heapsort(candidates_expanded, nb_candidates_expanded);
		if (f_v) {
			cout << "expanded candidate set after sort:" << endl;
			Int_vec_print(cout, candidates_expanded, nb_candidates_expanded);
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
		if (Sorting.int_vec_search(S, len, c, idx)) {
			tmp_candidates[nb_tmp_candidates++] = c;
			continue;
			}
		O->Hyperbolic_pair->unrank_point(v, 1, c, 0);
		if (f_v) {
			cout << "i=" << i;
			Int_vec_print(cout, v, n);
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
					Int_vec_print(cout, w, n);
					cout << endl;
					}
				d = O->Hyperbolic_pair->rank_point(w, 1, 0);
				if (!Sorting.int_vec_search(candidates_expanded,
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
		Int_vec_print(cout, tmp_candidates, nb_tmp_candidates);
		cout << endl;
		}

	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		c = candidates[i];
		if (Sorting.int_vec_search(tmp_candidates, nb_tmp_candidates, c, idx)) {
			good_candidates[nb_good_candidates++] = c;
			continue;
			}
		}
	
	if (f_v) {
		cout << "polar::test_if_closed_under_cosets for ";
		Int_vec_set_print(cout, S, len);
		cout << "; # of candidates reduced from "
			<< nb_candidates << " to " << nb_good_candidates << endl;
		}
	if (f_vv) {
		cout << "good candidates: ";
		Int_vec_print(cout, good_candidates, nb_good_candidates);
		cout << endl;
		}
	FREE_int(M);
	FREE_int(N0);
	FREE_int(N);
	FREE_int(v);
	FREE_int(w);
	FREE_int(candidates_expanded);
	FREE_int(tmp_candidates);
}


void polar::get_stabilizer(int orbit_idx,
		data_structures_groups::group_container &G,
		ring_theory::longinteger_object &go_G)
{
	Gen->get_node(first_node + orbit_idx)->get_stabilizer(Gen,
			G, go_G, 0 /*verbose_level - 2*/);
}

void polar::get_orbit_length(int orbit_idx,
		ring_theory::longinteger_object &length)
{
	Gen->orbit_length(orbit_idx, depth, length);
}

int polar::get_orbit_length_as_int(int orbit_idx)
{
	return Gen->orbit_length_as_int(orbit_idx, depth);
}

void polar::orbit_element_unrank(int orbit_idx,
		long int rank, long int *set, int verbose_level)
{
	return Gen->orbit_element_unrank(depth,
			orbit_idx, rank, set, verbose_level);
}

void polar::orbit_element_rank(int &orbit_idx,
		long int &rank, long int *set, int verbose_level)
{
	return Gen->orbit_element_rank(depth,
			orbit_idx, rank, set, verbose_level);
}

void polar::unrank_point(int *v, int rk)
{
	O->Hyperbolic_pair->unrank_point(v, 1, rk, 0);
}

int polar::rank_point(int *v)
{
	return O->Hyperbolic_pair->rank_point(v, 1, 0);
}

void polar::list_whole_orbit(int depth,
		int orbit_idx, int f_limit, int limit)
{
	long int *set;
	int ii;
	long int len, j, h, jj;
	data_structures_groups::group_container G;
	ring_theory::longinteger_object go_G, Rank;
	int *M1;
	int *base_cols;

	set = NEW_lint(depth);
	M1 = NEW_int(depth * n);
	base_cols = NEW_int(n);
	get_stabilizer(orbit_idx, G, go_G);
	cout << "the stabilizer of orbit rep " << orbit_idx
			<< " has order " << go_G << endl;

	len = get_orbit_length_as_int(orbit_idx);
		
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
		Lint_vec_print(cout, set, depth);
		cout << endl;
			
		for (h = 0; h < depth; h++) {
			unrank_point(M1 + h * n, set[h]);
			}
		cout << "corresponding to the subspace with basis:" << endl;
		Int_vec_print_integer_matrix_width(cout, M1, k, n, n, F->log10_of_q);
			
		F->Linear_algebra->Gauss_simple(M1, depth, n, base_cols, 0/* verbose_level*/);

		cout << "basis in echelon form:" << endl;
		Int_vec_print_integer_matrix_width(cout, M1, k, n, n, F->log10_of_q);
		


		geometry::grassmann Grass;
	
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
	FREE_lint(set);
	FREE_int(M1);
	FREE_int(base_cols);
}


// #############################################################################
// global functions:
// #############################################################################

long int static polar_callback_rank_point_func(int *v, void *data)
{
	polar *P = (polar *) data;
	//generator *gen = P->Gen;
	long int rk;
	
	rk = P->O->Hyperbolic_pair->rank_point(v, 1, 0);
	return rk;
}

void static polar_callback_unrank_point_func(int *v, long int rk, void *data)
{
	polar *P = (polar *) data;
	//generator *gen = P->Gen;
	
	P->O->Hyperbolic_pair->unrank_point(v, 1, rk, 0);
	//PG_element_unrank_modified(*gen->F, v, 1,
	// gen->vector_space_dimension, rk);
}

#if 0
int polar_callback_test_func(int len, int *S,
		void *data, int verbose_level)
{
	polar *P = (polar *) data;
	int f_OK = TRUE;
	int f_v = (verbose_level >= 1);
	
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
#endif

void static polar_callback_early_test_func(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	polar *P = (polar *) data;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "polar_callback_early_test_func for set ";
		Lint_vec_print(cout, S, len);
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

}}}


