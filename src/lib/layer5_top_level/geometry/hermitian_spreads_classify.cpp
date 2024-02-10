/*
 * hermitian_spreads_classify.cpp
 *
 *  Created on: Nov 6, 2019
 *      Author: anton
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_geometry {

static void HS_early_test_func_callback(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
static void projective_space_init_line_action(
		geometry::projective_space *P,
		actions::action *A_points, actions::action *&A_on_lines, int verbose_level);

hermitian_spreads_classify::hermitian_spreads_classify()
{
	n = Q = len = 0;
	F = NULL;
	H = NULL;

	Pts = NULL;
	nb_pts = 0;
	v = NULL;
	line_type = NULL;
	P = NULL;
	sg = NULL;
	Intersection_sets = NULL;
	sz = 0;
	secants = NULL;
	nb_secants = 0;
	Adj = NULL;

	A = NULL;
	A2 = NULL;
	line_type = NULL;
	A2r = NULL;

	Control = NULL;
	Poset = NULL;
	gen = NULL;
}



hermitian_spreads_classify::~hermitian_spreads_classify()
{
	int f_v = false;
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
		cout << "hermitian_spreads_classify::~hermitian_spreads_classify deleting secants" << endl;
	}
	if (secants) {
		FREE_lint(secants);
	}
	if (f_v) {
		cout << "hermitian_spreads_classify::~hermitian_spreads_classify deleting Adj" << endl;
	}
	if (Adj) {
		FREE_int(Adj);
	}
	if (f_v) {
		cout << "hermitian_spreads_classify::~hermitian_spreads_classify deleting GU" << endl;
	}
	if (f_v) {
		cout << "hermitian_spreads_classify::~hermitian_spreads_classify deleting sg" << endl;
	}
	if (sg) {
		FREE_OBJECT(sg);
	}
	if (f_v) {
		cout << "hermitian_spreads_classify::~hermitian_spreads_classify deleting A2r" << endl;
	}
	if (A2r) {
		FREE_OBJECT(A2r);
	}
	if (f_v) {
		cout << "hermitian_spreads_classify::~hermitian_spreads_classify deleting gen" << endl;
	}
	if (Poset) {
		FREE_OBJECT(Poset);
	}
	if (gen) {
		FREE_OBJECT(gen);
	}
}

void hermitian_spreads_classify::init(
		int n, int Q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "hermitian_spreads_classify::init" << endl;
		cout << "n=" << n << endl;
		cout << "Q=" << Q << endl;
	}
	hermitian_spreads_classify::n = n;
	hermitian_spreads_classify::Q = Q;

	F = NEW_OBJECT(field_theory::finite_field);

	F->finite_field_init_small_order(
			Q,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			0);

	len = n + 1;

	H = NEW_OBJECT(geometry::hermitian);
	H->init(F, len, verbose_level - 2);




	data_structures::tally C;
	int f, j, a, b, idx;



	v = NEW_int(len);
	H->list_of_points_embedded_in_PG(Pts, nb_pts, verbose_level);
	cout << "We found " << nb_pts << " points, they are:" << endl;
	for (i = 0; i < nb_pts; i++) {
		cout << i << " : " << Pts[i] << " : ";
		F->Projective_space_basic->PG_element_unrank_modified(
				v, 1, len, Pts[i]);
		Int_vec_print(cout, v, len);
		cout << endl;
	}


	P = NEW_OBJECT(geometry::projective_space);

	cout << "Creating projective_space" << endl;
	P->projective_space_init(
			n, F,
		true /* f_init_incidence_structure */,
		0 /* verbose_level */);
	cout << "Creating projective_space done" << endl;

	data_structures_groups::vector_ge *nice_gens;

	cout << "Creating linear group" << endl;
	A = NEW_OBJECT(actions::action);
	A->Known_groups->init_general_linear_group(
			n + 1, F,
			true /* f_semilinear */, true /* f_basis */, true /* f_init_sims */,
			nice_gens,
			verbose_level - 2);
	FREE_OBJECT(nice_gens);

	cout << "Creating action on lines" << endl;
	projective_space_init_line_action(P, A, A2, verbose_level);




	line_type = NEW_int(P->Subspaces->N_lines);

	P->Subspaces->line_intersection_type(
			Pts, nb_pts, line_type, verbose_level);


	C.init(line_type, P->Subspaces->N_lines, false, 0);
	cout << "The line type is:" << endl;
	C.print(true /* f_backwards*/);

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
			P->Subspaces->Grass_lines, a, Pts, nb_pts,
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

		Lint_vec_print(cout, Intersection_sets[j], sz);
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
	Int_matrix_print(Adj, nb_secants, nb_secants);


	cout << "Computing the unitary group:" << endl;

	actions::action_global AcGl;

	//int canonical_pt;

	sg = AcGl.set_stabilizer_in_projective_space(A, P,
			Pts, nb_pts, /*canonical_pt,*/ //NULL,
			verbose_level);

	//GU = P->set_stabilizer(Pts, nb_pts, verbose_level);
	ring_theory::longinteger_object go;

	sg->group_order(go);
	cout << "Group has been computed, group order = " << go << endl;



	cout << "strong generators are:" << endl;
	sg->print_generators(cout);



	//A2r = NEW_OBJECT(action);

	std::string label_of_set;
	std::string label_of_set_tex;

	label_of_set.assign("_secants");
	label_of_set_tex.assign("\\_secants");


	cout << "Creating restricted action on secants:" << endl;
	A2r = A2->Induced_action->create_induced_action_by_restriction(
		NULL,
		nb_secants, secants, label_of_set, label_of_set_tex,
		false /* f_induce_action */,
		0 /* verbose_level */);
	cout << "Creating restricted action on secants done." << endl;


	if (f_v) {
		cout << "hermitian_spread_classify::init done" << endl;
	}
}

void hermitian_spreads_classify::read_arguments(
		int argc, std::string *argv)
{
	int i;
	data_structures::string_tools ST;

	Control = NEW_OBJECT(poset_classification::poset_classification_control);
	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
	gen = NEW_OBJECT(poset_classification::poset_classification);

#if 0
	for (i = 1; i < argc; i++) {
		cout << argv[i] << endl;
		}
#endif
	//gen->read_arguments(argc, argv, 1);


	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-poset_classification_control") == 0) {
			Control = NEW_OBJECT(poset_classification::poset_classification_control);
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

void hermitian_spreads_classify::init2(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int depth;
	string prefix;

	if (f_v) {
		cout << "hermitian_spreads_classify::init2" << endl;
		}
	//depth = order + 1;

	prefix = "HS_" + std::to_string(n) + "_" + std::to_string(Q);

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


	//gen->f_allowed_to_show_group_elements = true;



#if 0
	gen->f_print_function = true;
	gen->print_function = print_set;
	gen->print_function_data = this;
#endif


	if (f_v) {
		cout << "hermitian_spreads_classify::init2 done" << endl;
	}
}

void hermitian_spreads_classify::compute(
		int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int schreier_depth = depth;
	int f_use_invariant_subset_if_available = true;
	int f_debug = false;
	int t0;
	orbiter_kernel_system::os_interface Os;


	Control->f_depth = true;
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
	int f_sideways = false;

	gen->draw_poset(gen->get_problem_label_with_path(), depth, 0 /* data1 */,
			f_embedded, f_sideways, 100, verbose_level);
	gen->print_data_structure_tex(depth, verbose_level);
#endif

	if (f_v) {
		cout << "hermitian_spreads_classify::compute "
				"We found " << length << " orbits" << endl;
	}
}


void hermitian_spreads_classify::early_test_func(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, i0;

	if (f_v) {
		cout << "hermitian_spreads_classify::early_test_func checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size " << nb_candidates << ":" << endl;
		Lint_vec_print(cout, candidates, nb_candidates);
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


static void HS_early_test_func_callback(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	hermitian_spreads_classify *HS = (hermitian_spreads_classify *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "HS_early_test_func for set ";
		Lint_vec_print(cout, S, len);
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


static void projective_space_init_line_action(
		geometry::projective_space *P,
		actions::action *A_points, actions::action *&A_on_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_grassmannian *AoL;

	if (f_v) {
		cout << "projective_space_init_line_action" << endl;
	}
	//A_on_lines = NEW_OBJECT(actions::action);

	AoL = NEW_OBJECT(induced_actions::action_on_grassmannian);

	AoL->init(*A_points, P->Subspaces->Grass_lines, verbose_level - 5);


	if (f_v) {
		cout << "projective_space_init_line_action "
				"action on grassmannian established" << endl;
	}

	if (f_v) {
		cout << "projective_space_init_line_action "
				"initializing A_on_lines" << endl;
	}
	int f_induce_action = true;
	groups::sims S;
	ring_theory::longinteger_object go1;

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
	A_on_lines = A_points->Induced_action->induced_action_on_grassmannian_preloaded(AoL,
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



}}}


