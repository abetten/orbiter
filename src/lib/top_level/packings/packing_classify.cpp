// packing_classify.cpp
// 
// Anton Betten
// Feb 6, 2013
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace packings {


#if 0
static void callback_packing_compute_klein_invariants(
		isomorph *Iso, void *data, int verbose_level);
static void callback_packing_report(isomorph *Iso,
		void *data, int verbose_level);
static void packing_lifting_prepare_function_new(
	exact_cover *EC, int starter_case,
	long int *candidates, int nb_candidates,
	groups::strong_generators *Strong_gens,
	solvers::diophant *&Dio, long int *&col_labels,
	int &f_ruled_out,
	int verbose_level);
#endif
static void packing_early_test_function(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
#if 0
static int count(int *Inc, int n, int m, int *set, int t);
static int count_and_record(int *Inc, int n, int m,
		int *set, int t, int *occurances);
#endif

packing_classify::packing_classify()
{
	PA = NULL;
	T = NULL;
	F = NULL;
	spread_size = 0;
	nb_lines = 0;


	f_lexorder_test = TRUE;
	q = 0;
	size_of_packing = 0;
		// the number of spreads in a packing,
		// which is q^2 + q + 1

	Spread_table_with_selection = NULL;

	P3 = NULL;
	P5 = NULL;
	the_packing = NULL;
	spread_iso_type = NULL;
	dual_packing = NULL;
	list_of_lines = NULL;
	list_of_lines_klein_image = NULL;
	Gr = NULL;


	degree = NULL;

	Control = NULL;
	Poset = NULL;
	gen = NULL;

	nb_needed = 0;

	//null();
}

packing_classify::~packing_classify()
{
	freeself();
}

void packing_classify::null()
{
}

void packing_classify::freeself()
{
	if (Spread_table_with_selection) {
		FREE_OBJECT(Spread_table_with_selection);
	}
	if (P3) {
		FREE_OBJECT(P3);
	}
	if (P5) {
		FREE_OBJECT(P5);
	}
	if (the_packing) {
		FREE_lint(the_packing);
	}
	if (spread_iso_type) {
		FREE_lint(spread_iso_type);
	}
	if (dual_packing) {
		FREE_lint(dual_packing);
	}
	if (list_of_lines) {
		FREE_lint(list_of_lines);
	}
	if (list_of_lines_klein_image) {
		FREE_lint(list_of_lines_klein_image);
	}
	if (Gr) {
		FREE_OBJECT(Gr);
	}
	null();
}

void packing_classify::spread_table_init(
		projective_geometry::projective_space_with_action *PA,
		int dimension_of_spread_elements,
		int f_select_spread, std::string &select_spread_text,
		std::string &path_to_spread_tables,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_classify::spread_table_init "
				"dimension_of_spread_elements=" << dimension_of_spread_elements << endl;
	}
	int n, q;
	groups::matrix_group *Mtx;
	spreads::spread_classify *T;


	packing_classify::PA = PA;
	packing_classify::path_to_spread_tables.assign(path_to_spread_tables);
	n = PA->A->matrix_group_dimension();
	Mtx = PA->A->get_matrix_group();
	F = Mtx->GFq;
	q = F->q;
	if (f_v) {
		cout << "packing_classify::spread_table_init n=" << n
				<< " k=" << dimension_of_spread_elements
				<< " q=" << q << endl;
	}


	T = NEW_OBJECT(spreads::spread_classify);


	if (f_v) {
		cout << "packing_classify::spread_table_init before T->init" << endl;
	}


	T->init(PA,
			dimension_of_spread_elements,
			TRUE /* f_recoordinatize */,
			verbose_level - 1);

	if (f_v) {
		cout << "packing_classify::spread_table_init after T->init" << endl;
	}

#if 0
	if (f_v) {
		cout << "packing_classify::spread_table_init before T->init2" << endl;
	}
	T->init2(Control, verbose_level);
	if (f_v) {
		cout << "packing_classify::spread_table_init after T->init2" << endl;
	}
#endif



	spreads::spread_table_with_selection *Spread_table_with_selection;

	Spread_table_with_selection = NEW_OBJECT(spreads::spread_table_with_selection);

	if (f_v) {
		cout << "packing_classify::spread_table_init "
				"before Spread_table_with_selection->init" << endl;
	}
	Spread_table_with_selection->init(T,
		f_select_spread,
		select_spread_text,
		path_to_spread_tables,
		verbose_level);
	if (f_v) {
		cout << "packing_classify::spread_table_init "
				"after Spread_table_with_selection->init" << endl;
	}





	if (f_v) {
		cout << "packing_classify::spread_table_init before init" << endl;
	}
	init(
		PA,
		Spread_table_with_selection,
		TRUE,
		verbose_level);
	if (f_v) {
		cout << "packing_classify::spread_table_init after init" << endl;
	}

#if 0
	cout << "before IA->init" << endl;
	IA->init(T->A, P->A_on_spreads, P->gen,
		P->size_of_packing, P->prefix_with_directory, ECA,
		callback_packing_report,
		NULL /*callback_subset_orbits*/,
		P,
		verbose_level);
	cout << "after IA->init" << endl;
#endif

	if (f_v) {
		cout << "packing_classify::spread_table_init before P->compute_spread_table" << endl;
	}
	Spread_table_with_selection->compute_spread_table(verbose_level);
	if (f_v) {
		cout << "packing_classify::spread_table_init after P->compute_spread_table" << endl;
	}

	if (f_v) {
		cout << "packing_classify::spread_table_init done" << endl;
	}


}


void packing_classify::init(
		projective_geometry::projective_space_with_action *PA,
		spreads::spread_table_with_selection *Spread_table_with_selection,
		int f_lexorder_test,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_classify::init" << endl;
	}


	packing_classify::PA = PA;
	packing_classify::Spread_table_with_selection = Spread_table_with_selection;


	packing_classify::T = Spread_table_with_selection->T;
	F = Spread_table_with_selection->T->Mtx->GFq;
	q = Spread_table_with_selection->T->q;
	spread_size = Spread_table_with_selection->T->spread_size;
	size_of_packing = q * q + q + 1;
	nb_lines = Spread_table_with_selection->T->A2->degree;

	packing_classify::f_lexorder_test = f_lexorder_test;

	
	if (f_v) {
		cout << "packing_classify::init q=" << q << endl;
		cout << "packing_classify::init nb_lines=" << nb_lines << endl;
		cout << "packing_classify::init spread_size=" << spread_size << endl;
		cout << "packing_classify::init size_of_packing=" << size_of_packing << endl;
		//cout << "packing_classify::init input_prefix=" << input_prefix << endl;
		//cout << "packing_classify::init base_fname=" << base_fname << endl;
	}

	if (f_v) {
		cout << "packing_classify::init before init_P3_and_P5_and_Gr" << endl;
	}
	init_P3_and_P5_and_Gr(verbose_level - 1);
	if (f_v) {
		cout << "packing_classify::init after init_P3_and_P5_and_Gr" << endl;
	}


	if (f_v) {
		cout << "packing_classify::init done" << endl;
	}
}

void packing_classify::init2(
		poset_classification::poset_classification_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_classify::init2" << endl;
	}

	if (f_v) {
		cout << "packing_classify::init2 "
				"before create_action_on_spreads" << endl;
	}
	Spread_table_with_selection->create_action_on_spreads(verbose_level - 1);
	if (f_v) {
		cout << "packing_classify::init2 "
				"after create_action_on_spreads" << endl;
	}


	
	if (f_v) {
		cout << "packing_classify::init "
				"before prepare_generator" << endl;
	}
	prepare_generator(Control, verbose_level - 1);
	if (f_v) {
		cout << "packing_classify::init "
				"after prepare_generator" << endl;
	}

	if (f_v) {
		cout << "packing_classify::init done" << endl;
	}
}



void packing_classify::init_P3_and_P5_and_Gr(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_classify::init_P3_and_P5_and_Gr" << endl;
	}
	P3 = NEW_OBJECT(geometry::projective_space);
	
	P3->projective_space_init(3, F,
		TRUE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "packing_classify::init_P3_and_P5_and_Gr P3->N_points=" << P3->N_points << endl;
		cout << "packing_classify::init_P3_and_P5_and_Gr P3->N_lines=" << P3->N_lines << endl;
	}

	P5 = NEW_OBJECT(geometry::projective_space);

	P5->projective_space_init(5, F,
		TRUE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "packing_classify::init_P3_and_P5_and_Gr P5->N_points=" << P5->N_points << endl;
		cout << "packing_classify::init_P3_and_P5_and_Gr P5->N_lines=" << P5->N_lines << endl;
	}

	the_packing = NEW_lint(size_of_packing);
	spread_iso_type = NEW_lint(size_of_packing);
	dual_packing = NEW_lint(size_of_packing);
	list_of_lines = NEW_lint(size_of_packing * spread_size);
	list_of_lines_klein_image = NEW_lint(size_of_packing * spread_size);

	Gr = NEW_OBJECT(geometry::grassmann);

	Gr->init(6, 3, F, 0 /* verbose_level */);

	if (f_v) {
		cout << "packing_classify::init_P3_and_P5_and_Gr done" << endl;
	}
}









void packing_classify::prepare_generator(
		poset_classification::poset_classification_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "packing_classify::prepare_generator" << endl;
	}
	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
	Poset->init_subset_lattice(T->A, Spread_table_with_selection->A_on_spreads,
			T->A->Strong_gens,
			verbose_level);

	if (f_v) {
		cout << "packing_classify::prepare_generator before "
				"Poset->add_testing_without_group" << endl;
	}
	Poset->add_testing_without_group(
			packing_early_test_function,
				this /* void *data */,
				verbose_level);




#if 0
	Control->f_print_function = TRUE;
	Control->print_function = print_set;
	Control->print_function_data = this;
#endif


	if (f_v) {
		cout << "packing_classify::prepare_generator "
				"calling gen->initialize" << endl;
	}

	gen = NEW_OBJECT(poset_classification::poset_classification);

	gen->initialize_and_allocate_root_node(Control, Poset,
			size_of_packing,
			verbose_level - 1);

	if (f_v) {
		cout << "packing_classify::prepare_generator done" << endl;
	}
}


void packing_classify::compute(int search_depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int schreier_depth = search_depth;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	int t0;
	orbiter_kernel_system::os_interface Os;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "packing_classify::compute" << endl;
	}

	gen->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		verbose_level - 1);
	
	int length;
	
	if (f_v) {
		cout << "packing_classify::compute done with generator_main" << endl;
	}
	length = gen->nb_orbits_at_level(search_depth);
	if (f_v) {
		cout << "packing_classify::compute We found "
			<< length << " orbits on "
			<< search_depth << "-sets" << endl;
	}
	if (f_v) {
		cout << "packing_classify::compute done" << endl;
	}

}

void packing_classify::lifting_prepare_function_new(
	exact_cover *E, int starter_case,
	long int *candidates, int nb_candidates,
	groups::strong_generators *Strong_gens,
	solvers::diophant *&Dio, long int *&col_labels,
	int &f_ruled_out, 
	int verbose_level)
{
	verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_v3 = (verbose_level >= 3);
	long int *points_covered_by_starter;
	int nb_points_covered_by_starter;
	long int *free_points2;
	int nb_free_points2;
	long int *free_point_idx;
	long int *live_blocks2;
	int nb_live_blocks2;
	int nb_needed, /*nb_rows,*/ nb_cols;


	if (f_v) {
		cout << "packing_classify::lifting_prepare_function "
				"nb_candidates=" << nb_candidates << endl;
	}

	nb_needed = size_of_packing - E->starter_size;


	if (f_v) {
		cout << "packing_classify::lifting_prepare_function "
				"before compute_covered_points" << endl;
	}

	Spread_table_with_selection->compute_covered_points(points_covered_by_starter,
		nb_points_covered_by_starter, 
		E->starter, E->starter_size, 
		verbose_level - 1);


	if (f_v) {
		cout << "packing_classify::lifting_prepare_function "
				"before compute_free_points2" << endl;
	}

	Spread_table_with_selection->compute_free_points2(
		free_points2, nb_free_points2, free_point_idx,
		points_covered_by_starter, nb_points_covered_by_starter, 
		E->starter, E->starter_size, 
		verbose_level - 1);

	if (f_v) {
		cout << "packing_classify::lifting_prepare_function "
				"before compute_live_blocks2" << endl;
	}

	Spread_table_with_selection->compute_live_blocks2(
		E, starter_case, live_blocks2, nb_live_blocks2,
		points_covered_by_starter, nb_points_covered_by_starter, 
		E->starter, E->starter_size, 
		verbose_level - 1);


	if (f_v) {
		cout << "packing_classify::lifting_prepare_function "
				"after compute_live_blocks2" << endl;
	}

	//nb_rows = nb_free_points2;
	nb_cols = nb_live_blocks2;
	col_labels = NEW_lint(nb_cols);


	Lint_vec_copy(live_blocks2, col_labels, nb_cols);


	if (f_vv) {
		cout << "packing_classify::lifting_prepare_function_new candidates: ";
		Lint_vec_print(cout, col_labels, nb_cols);
		cout << " (nb_candidates=" << nb_cols << ")" << endl;
	}



	if (E->f_lex) {
		int nb_cols_before;

		nb_cols_before = nb_cols;
		E->lexorder_test(col_labels, nb_cols, Strong_gens->gens, 
			verbose_level - 2);
		if (f_v) {
			cout << "packing_classify::lifting_prepare_function_new after "
					"lexorder test nb_candidates before: " << nb_cols_before
					<< " reduced to  " << nb_cols << " (deleted "
					<< nb_cols_before - nb_cols << ")" << endl;
		}
	}

	if (f_vv) {
		cout << "packing_classify::lifting_prepare_function_new "
				"after lexorder test" << endl;
		cout << "packing::lifting_prepare_function_new "
				"nb_cols=" << nb_cols << endl;
	}

	Spread_table_with_selection->Spread_tables->make_exact_cover_problem(Dio,
			free_point_idx, nb_free_points2,
			live_blocks2, nb_live_blocks2,
			nb_needed,
			verbose_level);

	FREE_lint(points_covered_by_starter);
	FREE_lint(free_points2);
	FREE_lint(free_point_idx);
	FREE_lint(live_blocks2);
	if (f_v) {
		cout << "packing_classify::lifting_prepare_function done" << endl;
	}
}




void packing_classify::report_fixed_objects(int *Elt,
		char *fname_latex, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i, j, cnt;
	//int v[4];
	//file_io Fio;

	if (f_v) {
		cout << "packing_classify::report_fixed_objects" << endl;
	}


	{
		ofstream fp(fname_latex);
		char str[1000];
		string title, author, extra_praeamble;

		orbiter_kernel_system::latex_interface L;

		sprintf(str, "Fixed Objects");
		title.assign(str);
		author.assign("");

		L.head(fp,
			FALSE /* f_book */, TRUE /* f_title */,
			title, author /* const char *author */,
			FALSE /* f_toc */, FALSE /* f_landscape */, TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */, TRUE /* f_pagenumbers */,
			extra_praeamble /* extra_praeamble */);
		//latex_head_easy(fp);

	
		T->A->report_fixed_objects_in_P3(fp,
				P3,
				Elt,
				verbose_level);
	
#if 0
		fp << "\\section{Fixed Objects}" << endl;



		fp << "The element" << endl;
		fp << "$$" << endl;
		T->A->element_print_latex(Elt, fp);
		fp << "$$" << endl;
		fp << "has the following fixed objects:" << endl;


		fp << "\\subsection{Fixed Points}" << endl;

		cnt = 0;
		for (i = 0; i < P3->N_points; i++) {
			j = T->A->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				cnt++;
				}
			}

		fp << "There are " << cnt << " fixed points, they are: \\\\" << endl;
		for (i = 0; i < P3->N_points; i++) {
			j = T->A->element_image_of(i, Elt, 0 /* verbose_level */);
			F->PG_element_unrank_modified(v, 1, 4, i);
			if (j == i) {
				fp << i << " : ";
				int_vec_print(fp, v, 4);
				fp << "\\\\" << endl;
				cnt++;
				}
			}
	
		fp << "\\subsection{Fixed Lines}" << endl;

		{
		action *A2;

		A2 = T->A->induced_action_on_grassmannian(2, 0 /* verbose_level*/);

		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				cnt++;
				}
			}

		fp << "There are " << cnt << " fixed lines, they are: \\\\" << endl;
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				fp << i << " : $\\left[";
				A2->G.AG->G->print_single_generator_matrix_tex(fp, i);
				fp << "\\right]$\\\\" << endl;
				cnt++;
				}
			}

		FREE_OBJECT(A2);
		}
	
		fp << "\\subsection{Fixed Planes}" << endl;

		{
		action *A2;

		A2 = T->A->induced_action_on_grassmannian(3, 0 /* verbose_level*/);

		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				cnt++;
				}
			}

		fp << "There are " << cnt << " fixed planes, they are: \\\\" << endl;
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				fp << i << " : $\\left[";
				A2->G.AG->G->print_single_generator_matrix_tex(fp, i);
				fp << "\\right]$\\\\" << endl;
				cnt++;
				}
			}

		FREE_OBJECT(A2);
		}
#endif


		L.foot(fp);
	}
	orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname_latex << " of size "
			<< Fio.file_size(fname_latex) << endl;

	
	if (f_v) {
		cout << "packing::report_fixed_objects done" << endl;
	}
}


int packing_classify::test_if_orbit_is_partial_packing(
		groups::schreier *Orbits, int orbit_idx,
	long int *orbit1, int verbose_level)
{
	int f_v = FALSE; // (verbose_level >= 1);
	int len;

	if (f_v) {
		cout << "packing_classify::test_if_orbit_is_partial_packing "
				"orbit_idx = " << orbit_idx << endl;
	}
	Orbits->get_orbit(orbit_idx, orbit1, len, 0 /* verbose_level*/);
	return Spread_table_with_selection->Spread_tables->test_if_set_of_spreads_is_line_disjoint(orbit1, len);
}

int packing_classify::test_if_pair_of_orbits_are_adjacent(
		groups::schreier *Orbits, int a, int b,
	long int *orbit1, long int *orbit2,
	int verbose_level)
// tests if every spread from orbit a
// is line-disjoint from every spread from orbit b
{
	int f_v = FALSE; // (verbose_level >= 1);
	int len1, len2;

	if (f_v) {
		cout << "packing_classify::test_if_pair_of_orbits_are_adjacent "
				"a=" << a << " b=" << b << endl;
	}
	if (a == b) {
		return FALSE;
	}
	Orbits->get_orbit(a, orbit1, len1, 0 /* verbose_level*/);
	Orbits->get_orbit(b, orbit2, len2, 0 /* verbose_level*/);

	return Spread_table_with_selection->Spread_tables->test_if_pair_of_sets_are_adjacent(
			orbit1, len1,
			orbit2, len2,
			verbose_level);
}


int packing_classify::find_spread(long int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx;

	if (f_v) {
		cout << "packing_classify::find_spread" << endl;
	}
	idx = Spread_table_with_selection->find_spread(set, verbose_level);
	return idx;
}



// #############################################################################
// global functions:
// #############################################################################

#if 0
static void callback_packing_compute_klein_invariants(
		isomorph *Iso, void *data, int verbose_level)
{
	packing_classify *P = (packing_classify *) data;
	int f_split = FALSE;
	int split_r = 0;
	int split_m = 1;
	
	P->compute_klein_invariants(Iso, f_split, split_r, split_m,
			verbose_level);
}

static void callback_packing_report(isomorph *Iso,
		void *data, int verbose_level)
{
	packing_classify *P = (packing_classify *) data;
	
	P->report(Iso, verbose_level);
}


static void packing_lifting_prepare_function_new(
	exact_cover *EC, int starter_case,
	long int *candidates, int nb_candidates,
	groups::strong_generators *Strong_gens,
	solvers::diophant *&Dio, long int *&col_labels,
	int &f_ruled_out, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	packing_classify *P = (packing_classify *) EC->user_data;

	if (f_v) {
		cout << "packing_lifting_prepare_function_new "
				"nb_candidates=" << nb_candidates << endl;
	}

	P->lifting_prepare_function_new(
		EC, starter_case,
		candidates, nb_candidates, Strong_gens, 
		Dio, col_labels, f_ruled_out, 
		verbose_level);


	if (f_v) {
		cout << "packing_lifting_prepare_function_new "
				"after lifting_prepare_function_new" << endl;
	}

	if (f_v) {
		cout << "packing_lifting_prepare_function_new "
				"nb_rows=" << Dio->m
				<< " nb_cols=" << Dio->n << endl;
	}

	if (f_v) {
		cout << "packing_lifting_prepare_function_new done" << endl;
	}
}
#endif



static void packing_early_test_function(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	packing_classify *P = (packing_classify *) data;
	int f_v = (verbose_level >= 1);
	long int i, a, b;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "packing_early_test_function for set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
	}
	a = S[len - 1];
	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		b = candidates[i];

		if (b == a) {
			continue;
		}
#if 0
		if (P->bitvector_adjacency) {
			k = Combi.ij2k_lint(a, b, P->Spread_table_with_selection->Spread_tables->nb_spreads);
			if (bitvector_s_i(P->bitvector_adjacency, k)) {
				good_candidates[nb_good_candidates++] = b;
			}
		}
		else {
			if (P->Spread_table_with_selection->Spread_tables->test_if_spreads_are_disjoint(a, b)) {
				good_candidates[nb_good_candidates++] = b;
			}
		}
#else
		if (P->Spread_table_with_selection->is_adjacent(a, b)) {
			good_candidates[nb_good_candidates++] = b;
		}
#endif
	}
	if (f_v) {
		cout << "packing_early_test_function done" << endl;
	}
}




#if 0
static int count(int *Inc, int n, int m, int *set, int t)
{
	int i, j;
	int nb, h;
	
	nb = 0;
	for (j = 0; j < m; j++) {
		for (h = 0; h < t; h++) {
			i = set[h];
			if (Inc[i * m + j] == 0) {
				break;
			}
		}
		if (h == t) {
			nb++;
		}
	}
	return nb;
}

static int count_and_record(int *Inc,
		int n, int m, int *set, int t, int *occurances)
{
	int i, j;
	int nb, h;
	
	nb = 0;
	for (j = 0; j < m; j++) {
		for (h = 0; h < t; h++) {
			i = set[h];
			if (Inc[i * m + j] == 0) {
				break;
			}
		}
		if (h == t) {
			occurances[nb++] = j;
		}
	}
	return nb;
}
#endif


}}}

