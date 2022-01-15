/*
 * hermitian_spreads_classify.cpp
 *
 *  Created on: Nov 6, 2019
 *      Author: anton
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


hermitian_spreads_classify::hermitian_spreads_classify()
{
	null();
}

hermitian_spreads_classify::~hermitian_spreads_classify()
{
	freeself();
}

void hermitian_spreads_classify::null()
{
	F = NULL;
	H = NULL;
	Pts = NULL;
	v = NULL;
	P = NULL;
	A = NULL;
	A2 = NULL;
	line_type = NULL;
	Intersection_sets = NULL;
	secants = NULL;
	Adj = NULL;
	//GU = NULL;
	sg = NULL;
	A2r = NULL;
	Poset = NULL;
	gen = NULL;
}

void hermitian_spreads_classify::freeself()
{
	int f_v = FALSE;
	int i;

	if (F) {
		FREE_OBJECT(F);
	}
	if (H) {
		FREE_OBJECT(H);
	}
	if (Pts) {
		FREE_lint(Pts);
	}
	if (v) {
		FREE_int(v);
	}
	if (P) {
		FREE_OBJECT(P);
	}
	if (A) {
		FREE_OBJECT(A);
	}
	if (A2) {
		FREE_OBJECT(A2);
	}
	if (line_type) {
		FREE_int(line_type);
	}
	if (Intersection_sets) {
		for (i = 0; i < nb_secants; i++) {
			FREE_lint(Intersection_sets[i]);
		}
	}
	if (f_v) {
		cout << "hermitian_spreads_classify::freeself deleting secants" << endl;
	}
	if (secants) {
		FREE_lint(secants);
	}
	if (f_v) {
		cout << "hermitian_spreads_classify::freeself deleting Adj" << endl;
	}
	if (Adj) {
		FREE_int(Adj);
	}
	if (f_v) {
		cout << "hermitian_spreads_classify::freeself deleting GU" << endl;
	}
#if 0
	if (GU) {
		delete GU;
		}
#endif
	if (f_v) {
		cout << "hermitian_spreads_classify::freeself deleting sg" << endl;
	}
	if (sg) {
		FREE_OBJECT(sg);
	}
	if (f_v) {
		cout << "hermitian_spreads_classify::freeself deleting A2r" << endl;
	}
	if (A2r) {
		FREE_OBJECT(A2r);
	}
	if (f_v) {
		cout << "hermitian_spreads_classify::freeself deleting gen" << endl;
	}
	if (Poset) {
		FREE_OBJECT(Poset);
	}
	if (gen) {
		FREE_OBJECT(gen);
	}
	null();
}

void hermitian_spreads_classify::init(int n, int Q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	sorting Sorting;

	if (f_v) {
		cout << "hermitian_spreads_classify::init" << endl;
		cout << "n=" << n << endl;
		cout << "Q=" << Q << endl;
	}
	hermitian_spreads_classify::n = n;
	hermitian_spreads_classify::Q = Q;
	F = NEW_OBJECT(finite_field);
	F->finite_field_init(Q, FALSE /* f_without_tables */, 0);

	len = n + 1;

	H = NEW_OBJECT(hermitian);
	H->init(F, len, verbose_level - 2);




	tally C;
	int f, j, a, b, idx;



	v = NEW_int(len);
	H->list_of_points_embedded_in_PG(Pts, nb_pts, verbose_level);
	cout << "We found " << nb_pts << " points, they are:" << endl;
	for (i = 0; i < nb_pts; i++) {
		cout << i << " : " << Pts[i] << " : ";
		F->PG_element_unrank_modified(v, 1, len, Pts[i]);
		Orbiter->Int_vec.print(cout, v, len);
		cout << endl;
	}


	P = NEW_OBJECT(projective_space);

	cout << "Creating projective_space" << endl;
	P->init(n, F,
		TRUE /* f_init_incidence_structure */,
		0 /* verbose_level */);
	cout << "Creating projective_space done" << endl;

	vector_ge *nice_gens;

	cout << "Creating linear group" << endl;
	A = NEW_OBJECT(action);
	A->init_general_linear_group(n + 1, F,
			TRUE /* f_semilinear */, TRUE /* f_basis */, TRUE /* f_init_sims */,
			nice_gens,
			verbose_level - 2);
	FREE_OBJECT(nice_gens);

	cout << "Creating action on lines" << endl;
	projective_space_init_line_action(P, A, A2, verbose_level);




	line_type = NEW_int(P->N_lines);

	P->line_intersection_type(Pts, nb_pts, line_type, verbose_level);


	C.init(line_type, P->N_lines, FALSE, 0);
	cout << "The line type is:" << endl;
	C.print(TRUE /* f_backwards*/);

	cout << "The secants are:" << endl;
	f = C.type_first[1];
	nb_secants = C.type_len[1];
	Intersection_sets = NEW_plint(nb_secants);
	sz = C.data_sorted[f];


	secants = NEW_lint(nb_secants);

	for (j = 0; j < nb_secants; j++) {
		a = C.sorting_perm_inv[f + j];
		secants[j] = a;
	}

	int intersection_set_size;

	for (j = 0; j < nb_secants; j++) {
		a = C.sorting_perm_inv[f + j];
		cout << j << " : " << a << " : ";

		P->intersection_of_subspace_with_point_set(
			P->Grass_lines, a, Pts, nb_pts,
			Intersection_sets[j], intersection_set_size,
			0 /* verbose_level */);
		if (intersection_set_size != sz) {
			cout << "intersection_set_size != sz" << endl;
			exit(1);
		}
		for (i = 0; i < sz; i++) {
			b = Intersection_sets[j][i];
			if (!Sorting.lint_vec_search_linear(Pts, nb_pts, b, idx)) {
				cout << "cannot find the point" << endl;
				exit(1);
			}
			Intersection_sets[j][i] = idx;
		}

		Orbiter->Lint_vec.print(cout, Intersection_sets[j], sz);
		cout << endl;
	}


	cout << "Computing Adjacency matrix:" << endl;
	Adj = NEW_int(nb_secants * nb_secants);
	for (i = 0; i < nb_secants * nb_secants; i++) {
		Adj[i] = 0;
	}
	for (i = 0; i < nb_secants; i++) {
		for (j = i + 1; j < nb_secants; j++) {
			if (Sorting.test_if_sets_are_disjoint_not_assuming_sorted(
					Intersection_sets[i], Intersection_sets[j], sz)) {
				Adj[i * nb_secants + j] = 1;
				Adj[j * nb_secants + i] = 1;
			}
		}
	}
	cout << "Adj" << endl;
	Orbiter->Int_vec.matrix_print(Adj, nb_secants, nb_secants);


	cout << "Computing the unitary group:" << endl;

	//int canonical_pt;
	sg = A->set_stabilizer_in_projective_space(P,
			Pts, nb_pts, /*canonical_pt,*/ NULL,
			verbose_level);
	//GU = P->set_stabilizer(Pts, nb_pts, verbose_level);
	longinteger_object go;

	sg->group_order(go);
	cout << "Group has been computed, group order = " << go << endl;



	cout << "strong generators are:" << endl;
	sg->print_generators(cout);



	//A2r = NEW_OBJECT(action);

	cout << "Creating restricted action on secants:" << endl;
	A2r = A2->create_induced_action_by_restriction(
		NULL,
		nb_secants, secants,
		FALSE /* f_induce_action */,
		0 /* verbose_level */);
	cout << "Creating restricted action on secants done." << endl;


	if (f_v) {
		cout << "hermitian_spread_classify::init done" << endl;
	}
}

void hermitian_spreads_classify::read_arguments(int argc, std::string *argv)
{
	int i;
	string_tools ST;

	Control = NEW_OBJECT(poset_classification_control);
	Poset = NEW_OBJECT(poset_with_group_action);
	gen = NEW_OBJECT(poset_classification);

#if 0
	for (i = 1; i < argc; i++) {
		cout << argv[i] << endl;
		}
#endif
	//gen->read_arguments(argc, argv, 1);


	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-poset_classification_control") == 0) {
			Control = NEW_OBJECT(poset_classification_control);
			i += Control->read_arguments(argc - (i + 1),
				argv + i + 1, 0 /*verbose_level*/);

			cout << "done with -poset_classification_control" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
}

void hermitian_spreads_classify::init2(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int depth;
	char prefix[1000];

	if (f_v) {
		cout << "hermitian_spreads_classify::init2" << endl;
		}
	//depth = order + 1;

	sprintf(prefix, "HS_%d_%d", n, Q);

	Poset->init_subset_lattice(A, A2r,
			sg,
			verbose_level);

	if (f_v) {
		cout << "hermitian_spreads_classify::init2 before "
				"Poset->add_testing_without_group" << endl;
	}
	Poset->add_testing_without_group(
			HS_early_test_func_callback,
				this /* void *data */,
				verbose_level);


	//gen->f_allowed_to_show_group_elements = TRUE;



#if 0
	gen->f_print_function = TRUE;
	gen->print_function = print_set;
	gen->print_function_data = this;
#endif


	if (f_v) {
		cout << "hermitian_spreads_classify::init2 done" << endl;
	}
}

void hermitian_spreads_classify::compute(int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int schreier_depth = depth;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	int t0;
	//int f_embedded = TRUE;
	os_interface Os;


	Control->f_depth = TRUE;
	Control->depth = depth;



	gen->initialize_and_allocate_root_node(Control, Poset,
		nb_pts / sz,
		//"", prefix,
		verbose_level - 2);



#if 0
	if (f_override_schreier_depth) {
		schreier_depth = override_schreier_depth;
	}
#endif
	if (f_v) {
		cout << "hermitian_spreads_classify::compute calling generator_main" << endl;
	}

	t0 = Os.os_ticks();
	gen->main(t0,
		schreier_depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level - 1);

	int length;

	if (f_v) {
		cout << "hermitian_spreads_classify::compute done with generator_main" << endl;
	}
	length = gen->nb_orbits_at_level(depth);

#if 0
	int f_sideways = FALSE;

	gen->draw_poset(gen->get_problem_label_with_path(), depth, 0 /* data1 */,
			f_embedded, f_sideways, 100, verbose_level);
	gen->print_data_structure_tex(depth, verbose_level);
#endif

	if (f_v) {
		cout << "hermitian_spreads_classify::compute "
				"We found " << length << " orbits" << endl;
	}
}


void hermitian_spreads_classify::early_test_func(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, i0;

	if (f_v) {
		cout << "hermitian_spreads_classify::early_test_func checking set ";
		Orbiter->Lint_vec.print(cout, S, len);
		cout << endl;
		cout << "candidate set of size " << nb_candidates << ":" << endl;
		Orbiter->Lint_vec.print(cout, candidates, nb_candidates);
		cout << endl;
	}

	nb_good_candidates = 0;
	for (j = 0; j < nb_candidates; j++) {

		a = candidates[j];

		if (len == 0) {
			i0 = 0;
		}
		else {
			i0 = len - 1;
		}
		for (i = i0; i < len; i++) {
			b = S[i];
			if (Adj[a * nb_secants + b] == 0) {
				break;
			}
			else {
			}
		} // next i



		if (i == len) {
			good_candidates[nb_good_candidates++] = candidates[j];
		}
	} // next j

}


void HS_early_test_func_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	hermitian_spreads_classify *HS = (hermitian_spreads_classify *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "HS_early_test_func for set ";
		Orbiter->Lint_vec.print(cout, S, len);
		cout << endl;
	}
	HS->early_test_func(S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
		verbose_level - 2);
	if (f_v) {
		cout << "HS_early_test_func done" << endl;
	}
}


void projective_space_init_line_action(projective_space *P,
		action *A_points, action *&A_on_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action_on_grassmannian *AoL;

	if (f_v) {
		cout << "projective_space_init_line_action" << endl;
	}
	A_on_lines = NEW_OBJECT(action);

	AoL = NEW_OBJECT(action_on_grassmannian);

	AoL->init(*A_points, P->Grass_lines, verbose_level - 5);


	if (f_v) {
		cout << "projective_space_init_line_action "
				"action on grassmannian established" << endl;
	}

	if (f_v) {
		cout << "projective_space_init_line_action "
				"initializing A_on_lines" << endl;
	}
	int f_induce_action = TRUE;
	sims S;
	longinteger_object go1;

	S.init(A_points, 0);
	S.init_generators(*A_points->Strong_gens->gens,
			0/*verbose_level*/);
	S.compute_base_orbits_known_length(
			A_points->get_transversal_length(),
			0/*verbose_level - 1*/);
	S.group_order(go1);
	if (f_v) {
		cout << "projective_space_init_line_action "
				"group order " << go1 << endl;
	}

	if (f_v) {
		cout << "projective_space_init_line_action "
				"initializing action on grassmannian" << endl;
	}
	A_on_lines->induced_action_on_grassmannian(A_points, AoL,
		f_induce_action, &S, verbose_level);
	if (f_v) {
		cout << "projective_space_init_line_action "
				"initializing A_on_lines done" << endl;
		A_on_lines->print_info();
	}

	if (f_v) {
		cout << "projective_space_init_line_action "
				"computing strong generators" << endl;
	}
	if (!A_on_lines->f_has_strong_generators) {
		cout << "projective_space_init_line_action "
				"induced action does not have strong generators" << endl;
	}
	if (f_v) {
		cout << "projective_space_init_line_action done" << endl;
	}
}



}}


