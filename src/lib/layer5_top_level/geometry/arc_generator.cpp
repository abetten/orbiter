// arc_generator.cpp
// 
// Anton Betten
//
// previous version Dec 6, 2004
// revised June 19, 2006
// revised Aug 17, 2008
// moved here from hyperoval.cpp May 10, 2013
// moved to TOP_LEVEL from APPS/ARCS: February 23, 2017
//
// Searches for arcs and hyperovals in Desarguesian projective planes
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_geometry {


static void arc_generator_early_test_function(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
#if 0
static void arc_generator_lifting_prepare_function_new(
	exact_cover *EC, int starter_case,
	long int *candidates, int nb_candidates, groups::strong_generators *Strong_gens,
	solvers::diophant *&Dio, long int *&col_labels,
	int &f_ruled_out,
	int verbose_level);
#endif
static void arc_generator_print_arc(std::ostream &ost, int len, long int *S, void *data);
//static void arc_generator_print_point(long int pt, void *data);
//static void arc_generator_report(isomorph *Iso, void *data, int verbose_level);



arc_generator::arc_generator()
{
	Descr = NULL;
	PA = NULL;
	Control = NULL;

	nb_points_total = 0;
	nb_affine_lines = 0;


	forbidden_points = NULL;
	nb_forbidden_points = 0;
	f_is_forbidden = NULL;


	SG = NULL;

	Poset = NULL;

	line_type = NULL;

	gen = NULL;

}

arc_generator::~arc_generator()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_generator::~arc_generator" << endl;
	}
	if (gen) {
		if (f_v) {
			cout << "arc_generator::~arc_generator before FREE_OBJECT(gen)" << endl;
		}
		FREE_OBJECT(gen);
		if (f_v) {
			cout << "arc_generator::~arc_generator after FREE_OBJECT(gen)" << endl;
		}
	}
	
	if (Poset) {
		FREE_OBJECT(Poset);
	}
	if (line_type) {
		FREE_int(line_type);
	}
	if (f_v) {
		cout << "arc_generator::~arc_generator done" << endl;
	}
}

void arc_generator::main(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "arc_generator::main "
				"verbose_level=" << verbose_level << endl;
	}

	if (f_v) {
		cout << "arc_generator::main before compute_starter" << endl;
	}
	compute_starter(verbose_level);
	if (f_v) {
		cout << "arc_generator::main after compute_starter" << endl;
	}

	if (f_v) {
		cout << "arc_generator::main done" << endl;
	}
}

void arc_generator::init(
	arc_generator_description *Descr,
	projective_geometry::projective_space_with_action *PA,
	groups::strong_generators *SG,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "arc_generator::init" << endl;
	}
	arc_generator::Descr = Descr;
	arc_generator::PA = PA;


	arc_generator::SG = SG;


	
	if (f_v) {
		cout << "arc_generator::init" << endl;
	}

	nb_points_total = Gg.nb_PG_elements(PA->n, PA->q);
		// q * q + q + 1 for planes (n=3)

	if (f_v) {
		cout << "arc_generator::init nb_points_total = " << nb_points_total << endl;
	}

	if (Descr->f_affine) {
		nb_affine_lines = Gg.nb_affine_lines(PA->n, PA->q);
		if (f_v) {
			cout << "arc_generator::init nb_affine_lines = " << nb_affine_lines << endl;
		}
	}




	if (Descr->f_has_forbidden_point_set) {
		int i, a;

		Int_vec_scan(Descr->forbidden_point_set_string, forbidden_points, nb_forbidden_points);

		f_is_forbidden = NEW_int(PA->P->N_points);
		Int_vec_zero(f_is_forbidden, PA->P->N_points);
		for (i = 0; i < nb_forbidden_points; i++) {
			a = forbidden_points[i];
			f_is_forbidden[a] = TRUE;
			cout << "arc_generator::init point " << a << " is forbidden" << endl;
		}
	}
	if (PA->P->Implementation->Lines_on_point == NULL) {
		cout << "arc_generator::init "
				"P->Lines_on_point == NULL" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "arc_generator::init line_type" << endl;
	}

	line_type = NEW_int(PA->P->N_lines);


	if (!Descr->f_control) {
		cout << "arc_generator::init please use -control <label>" << endl;
		exit(1);
	}


	Control = Get_object_of_type_poset_classification_control(Descr->control_label);

	if (f_v) {
		cout << "arc_generator::init Control=" << endl;
		Control->print();
	}

	if (f_v) {
		cout << "arc_generator::init before prepare_generator" << endl;
	}

	prepare_generator(verbose_level - 2);

	if (f_v) {
		cout << "arc_generator::init after prepare_generator" << endl;
	}



	if (f_v) {
		cout << "arc_generator::init done" << endl;
	}
}


void arc_generator::prepare_generator(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_generator::prepare_generator" << endl;
		cout << "arc_generator::prepare_generator " << endl;
	}



	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
	Poset->init_subset_lattice(PA->A, PA->A, SG /* A->Strong_gens*/, verbose_level);

	Poset->f_print_function = FALSE;
	Poset->print_function = arc_generator_print_arc;
	Poset->print_function_data = this;


	if (!Descr->f_no_arc_testing) {
		if (f_v) {
			cout << "arc_generator::init before "
					"Poset->add_testing_without_group" << endl;
		}
		Poset->add_testing_without_group(
				arc_generator_early_test_function,
					this /* void *data */,
					verbose_level);
	}

	gen = NEW_OBJECT(poset_classification::poset_classification);

	
	gen->initialize_and_allocate_root_node(Control, Poset,
		Descr->target_size,
		verbose_level - 1);

	if (f_v) {
		cout << "arc_generator::init "
				"problem_label_with_path=" << gen->get_problem_label_with_path() << endl;
	}


#if 0
	if (f_read_data_file) {
		if (f_v) {
			cout << "arc_generator::init reading data file "
					<< fname_data_file << endl;
		}

		gen->read_data_file(depth_completed,
				fname_data_file, verbose_level - 1);
		if (f_v) {
			cout << "arc_generator::init after reading data file "
					<< fname_data_file << " depth_completed = "
					<< depth_completed << endl;
		}
		if (f_v) {
			cout << "arc_generator::init before "
					"gen->recreate_schreier_vectors_up_to_level" << endl;
		}
		gen->recreate_schreier_vectors_up_to_level(depth_completed - 1,
			verbose_level - 1);
		if (f_v) {
			cout << "arc_generator::init after "
					"gen->recreate_schreier_vectors_up_to_level" << endl;
		}

	}
#endif

	if (f_v) {
		cout << "arc_generator::prepare_generator done" << endl;
	}
}

void arc_generator::compute_starter(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::os_interface Os;
	int t0 = Os.os_ticks();
	

	if (f_v) {
		cout << "arc_generator::compute_starter" << endl;
	}

	int schreier_depth = 1000;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	int depth;
	//int f_embedded = TRUE;
	//int f_sideways = FALSE;


#if 0
	if (f_read_data_file) {
		int target_depth = 0; // ToDo

		if (f_v) {
			cout << "arc_generator::compute_starter "
					"before generator_main" << endl;
			cout << "arc_generator::compute_starter "
					"problem_label_with_path=" << gen->get_problem_label_with_path() << endl;
		}
		depth = gen->compute_orbits(
				depth_completed, target_depth,
				verbose_level - 2);
		if (f_v) {
			cout << "arc_generator::compute_starter "
					"after gen->compute_orbits" << endl;
		}
	}
#endif
	//else {
		if (f_v) {
			cout << "arc_generator::compute_starter "
					"before gen->main" << endl;
			cout << "arc_generator::compute_starter "
					"problem_label_with_path=" << gen->get_problem_label_with_path() << endl;
		}
		depth = gen->main(t0,
			schreier_depth,
			f_use_invariant_subset_if_available,
			f_debug,
			verbose_level);
		if (f_v) {
			cout << "arc_generator::compute_starter "
					"after gen->main" << endl;
		}
	//}

	if (f_v) {
		cout << "arc_generator::compute_starter "
				"finished, depth=" << depth << endl;
	}





	if (f_v) {
		cout << "arc_generator::compute_starter done" << endl;
	}

}

int arc_generator::test_nb_Eckardt_points(
		long int *S, int len, int pt, int nb_E, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = TRUE;
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "arc_generator::test_nb_Eckardt_points" << endl;
	}
	ret = Gg.test_nb_Eckardt_points(PA->PA2->P, S, len, pt, nb_E, verbose_level);
	if (f_v) {
		cout << "arc_generator::test_nb_Eckardt_points done" << endl;
	}
	return ret;
}

int arc_generator::conic_test(long int *S, int len, int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = TRUE;

	if (f_v) {
		cout << "arc_generator::conic_test" << endl;
	}
	ret = PA->P->conic_test(S, len, pt, verbose_level);
	if (f_v) {
		cout << "arc_generator::conic_test done" << endl;
	}
	return ret;
}

void arc_generator::early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b;
	int f_survive;
		
	if (f_v) {
		cout << "arc_generator::early_test_func checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size " << nb_candidates << ":" << endl;
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
	}


	compute_line_type(S, len, 0 /* verbose_level */);

	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		a = candidates[i];

		f_survive = TRUE;

		if (Descr->f_has_forbidden_point_set) {
			if (f_is_forbidden[a]) {
				f_survive = FALSE;
			}
		}


		if (f_survive && Descr->f_d) {
			// test that there are no more than d points per line:
			for (j = 0; j < PA->P->r; j++) {
				b = PA->P->Implementation->Lines_on_point[a * PA->P->r + j];
				if (line_type[b] == Descr->d) {
					if (Descr->f_affine && b < nb_affine_lines) {
						f_survive = FALSE;
					}
					else if (!Descr->f_affine) {
						f_survive = FALSE;
					}
					break;
				}
			}
		}

		if (f_survive && Descr->f_conic_test) {
			if (conic_test(S, len, a, verbose_level) == FALSE) {
				f_survive = FALSE;
			}
		}

		if (f_survive && Descr->f_test_nb_Eckardt_points) {
			if (test_nb_Eckardt_points(S, len, a,
					Descr->nb_E, verbose_level) == FALSE) {
				f_survive = FALSE;
			}
		}

		if (f_survive) {
			good_candidates[nb_good_candidates++] = candidates[i];
		}
	} // next i
	
}

void arc_generator::print(int len, long int *S)
{
	int i, a;
	
	if (len == 0) {
		return;
	}

	compute_line_type(S, len, 0 /* verbose_level */);

	cout << "set ";
	Lint_vec_print(cout, S, len);
	cout << " has line type ";

	data_structures::tally C;

	C.init(line_type, PA->P->N_lines, FALSE, 0);
	C.print_naked(TRUE);
	cout << endl;

	int *Coord;

	Coord = NEW_int(len * (PA->n + 1));
	cout << "the coordinates of the points are:" << endl;
	for (i = 0; i < len; i++) {
		a = S[i];
		point_unrank(Coord + i * (PA->n + 1), a);
	}
	for (i = 0; i < len; i++) {
		cout << S[i] << " : ";
		Int_vec_print(cout, Coord + i * (PA->n + 1), PA->n + 1);
		cout << endl;
	}



	if (Descr->f_d && Descr->d >= 3) {
	}
	else {
		long int **Pts_on_conic;
		int **Conic_eqn;
		int *nb_pts_on_conic;
		int len1;

	
		cout << "Conic intersections:" << endl;

		if (PA->P->n != 2) {
			cout << "conic intersections "
					"only defined in the plane" << endl;
			exit(1);
		}
		PA->P->conic_type(
			S, len, 
			6 /* threshold */,
			Pts_on_conic, Conic_eqn, nb_pts_on_conic, len1,
			0 /*verbose_level*/);
		cout << "The arc intersects " << len1
				<< " conics in 6 or more points. " << endl;

#if 0
		cout << "These intersections are" << endl;
		for (i = 0; i < len1; i++) {
			cout << "conic intersection of size "
					<< nb_pts_on_conic[i] << " : ";
			int_vec_print(cout, Pts_on_conic[i], nb_pts_on_conic[i]);
			cout << endl;
		}
#endif





		for (i = 0; i < len1; i++) {
			FREE_lint(Pts_on_conic[i]);
			FREE_int(Conic_eqn[i]);
		}
		FREE_int(nb_pts_on_conic);
		FREE_plint(Pts_on_conic);
		FREE_pint(Conic_eqn);
	}
	
#if 0
	if (f_simeon) {
		F->simeon(n, len, S, simeon_s, verbose_level);
	}
#endif

	FREE_int(Coord);
}

void arc_generator::print_set_in_affine_plane(int len, long int *S)
{
	PA->F->print_set_in_affine_plane(len, S);
}




void arc_generator::point_unrank(int *v, int rk)
{
	PA->F->PG_element_unrank_modified(v, 1 /* stride */, (PA->n + 1) /* len */, rk);
}

int arc_generator::point_rank(int *v)
{
	int rk;
	
	PA->F->PG_element_rank_modified(v, 1 /* stride */, (PA->n + 1), rk);
	return rk;
}

void arc_generator::compute_line_type(long int *set, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int a, b;

	if (f_v) {
		cout << "arc_generator::compute_line_type" << endl;
	}

	if (PA->P->Implementation->Lines_on_point == 0) {
		cout << "arc_generator::compute_line_type "
				"P->Lines_on_point == 0" << endl;
		exit(1);
	}
	Int_vec_zero(line_type, PA->P->N_lines);
	for (i = 0; i < len; i++) {
		a = set[i];
		for (j = 0; j < PA->P->r; j++) {
			b = PA->P->Implementation->Lines_on_point[a * PA->P->r + j];
			line_type[b]++;
		}
	}
	
}

void arc_generator::lifting_prepare_function_new(
		solvers_package::exact_cover *E, int starter_case,
	long int *candidates, int nb_candidates,
	groups::strong_generators *Strong_gens,
	solvers::diophant *&Dio, long int *&col_labels,
	int &f_ruled_out, 
	int verbose_level)
// compute the incidence matrix of tangent lines versus candidate points
// extended by external lines versus candidate points
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, a, b;
	int nb_needed;
	int starter_size;

	if (f_v) {
		cout << "arc_generator::lifting_prepare_function_new "
				"nb_candidates=" << nb_candidates << endl;
	}

	if (PA->n != 2) {
		cout << "arc_generator::lifting_prepare_function_new "
				"needs PA->n == 2" << endl;
		exit(1);
	}
	if (Descr->d != 2) {
		cout << "arc_generator::lifting_prepare_function_new "
				"needs d == 2" << endl;
		exit(1);
	}
	starter_size = Control->depth;
	if (f_v) {
		cout << "arc_generator::lifting_prepare_function_new "
				"starter_size=" << starter_size << endl;
	}

	nb_needed = Descr->target_size - starter_size;
	f_ruled_out = FALSE;



	compute_line_type(E->starter, starter_size, 0 /* verbose_level */);



	data_structures::tally C;

	C.init(line_type, PA->P->N_lines, FALSE, 0);
	if (f_v) {
		cout << "arc_generator::lifting_prepare_function_new "
				"line_type:" << endl;
		C.print_naked(TRUE);
		cout << endl;
	}


	// extract the tangent lines:

	int tangent_lines_fst, nb_tangent_lines;
	int *tangent_lines;
	int *tangent_line_idx;
	int external_lines_fst, nb_external_lines;
	int *external_lines;
	int *external_line_idx;
	int fst, len, idx;


	// find all tangent lines:

	fst = 0;
	len = 0;
	for (i = 0; i < C.nb_types; i++) {
		fst = C.type_first[i];
		len = C.type_len[i];
		idx = C.sorting_perm_inv[fst];
		if (line_type[idx] == 1) {
			break;
		}
	}
	if (i == C.nb_types) {
		cout << "arc_generator::lifting_prepare_function_new "
				"there are no tangent lines" << endl;
		exit(1);
	}
	tangent_lines_fst = fst;
	nb_tangent_lines = len;
	tangent_lines = NEW_int(nb_tangent_lines);
	tangent_line_idx = NEW_int(PA->P->N_lines);
	for (i = 0; i < PA->P->N_lines; i++) {
		tangent_line_idx[i] = -1;
	}
	for (i = 0; i < len; i++) {
		j = C.sorting_perm_inv[tangent_lines_fst + i];
		tangent_lines[i] = j;
		tangent_line_idx[j] = i;
	}


	// find all external lines:
	for (i = 0; i < C.nb_types; i++) {
		fst = C.type_first[i];
		len = C.type_len[i];
		idx = C.sorting_perm_inv[fst];
		if (line_type[idx] == 0) {
			break;
		}
	}
	if (i == C.nb_types) {
		cout << "arc_generator::lifting_prepare_function_new "
				"there are no external lines" << endl;
		exit(1);
	}
	external_lines_fst = fst;
	nb_external_lines = len;
	external_lines = NEW_int(nb_external_lines);
	external_line_idx = NEW_int(PA->P->N_lines);
	for (i = 0; i < PA->P->N_lines; i++) {
		external_line_idx[i] = -1;
	}
	for (i = 0; i < len; i++) {
		j = C.sorting_perm_inv[external_lines_fst + i];
		external_lines[i] = j;
		external_line_idx[j] = i;
	}


	
	col_labels = NEW_lint(nb_candidates);


	Lint_vec_copy(candidates, col_labels, nb_candidates);

	if (E->f_lex) {
		if (f_vv) {
			cout << "arc_generator::lifting_prepare_function_new "
					"before lexorder test" << endl;
		}
		E->lexorder_test(col_labels, nb_candidates, Strong_gens->gens, 
			verbose_level - 2);
		if (f_vv) {
			cout << "arc_generator::lifting_prepare_function_new "
					"after lexorder test" << endl;
		}
	}

	if (f_vv) {
		cout << "arc_generator::lifting_prepare_function_new "
				"after lexorder test" << endl;
		cout << "arc_generator::lifting_prepare_function_new "
				"nb_candidates=" << nb_candidates << endl;
	}

	// compute the incidence matrix between
	// tangent lines and candidate points as well as
	// external lines and candidate points:


	int nb_rows;
	int nb_cols;

	nb_rows = nb_tangent_lines + nb_external_lines;
	nb_cols = nb_candidates;

	Dio = NEW_OBJECT(solvers::diophant);
	Dio->open(nb_rows, nb_cols);
	Dio->sum = nb_needed;

	for (i = 0; i < nb_tangent_lines; i++) {
		Dio->type[i] = t_EQ;
		Dio->RHS[i] = 1;
	}

	for (i = 0; i < nb_external_lines; i++) {
		Dio->type[nb_tangent_lines + i] = t_ZOR;
		Dio->RHS[nb_tangent_lines + i] = 2;
	}

	Dio->fill_coefficient_matrix_with(0);


	for (i = 0; i < nb_candidates; i++) {
		a = col_labels[i];
		for (j = 0; j < PA->P->r; j++) {
			b = PA->P->Implementation->Lines_on_point[a * PA->P->r + j];
			if (line_type[b] == 2) {
				cout << "arc_generator::lifting_prepare_function "
						"candidate lies on a secant" << endl;
				exit(1);
			}
			idx = tangent_line_idx[b];
			if (idx >= 0) {
				Dio->Aij(idx, i) = 1;
			}
			idx = external_line_idx[b];
			if (idx >= 0) {
				Dio->Aij(nb_tangent_lines + idx, i) = 1;
			}
		}
	}


	FREE_int(tangent_lines);
	FREE_int(tangent_line_idx);
	FREE_int(external_lines);
	FREE_int(external_line_idx);
	
	if (f_v) {
		cout << "arc_generator::lifting_prepare_function_new "
				"done" << endl;
	}
}


void arc_generator::report(isomorph::isomorph &Iso, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname[1000];
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "arc_generator::report" << endl;
	}
	if (Descr->target_size == PA->q + 2) {
		snprintf(fname, sizeof(fname), "hyperovals_%d.tex", PA->q);
	}
	else {
		snprintf(fname, sizeof(fname), "arcs_%d_%d.tex", PA->q, Descr->target_size);
	}

	{
		ofstream f(fname);
		int f_book = TRUE;
		int f_title = TRUE;
		string title, author, extra_praeamble;
		char str[1000];


		author.assign("Orbiter");
		int f_toc = TRUE;
		int f_landscape = FALSE;
		int f_12pt = FALSE;
		int f_enlarged_page = TRUE;
		int f_pagenumbers = TRUE;
		orbiter_kernel_system::latex_interface L;

		if (Descr->target_size == PA->q + 2) {
			snprintf(str, sizeof(str), "Hyperovals over ${\\mathbb F}_{%d}$", PA->q);
			title.assign(str);
			}
		else {
			snprintf(str, sizeof(str), "Arcs over  ${\\mathbb F}_{%d}$ "
					"of size $%d$", PA->q, Descr->target_size);
			title.assign(str);
			}
		cout << "Writing file " << fname << " with "
				<< Iso.Folding->Reps->count << " arcs:" << endl;
		L.head(f, f_book, f_title,
			title, author,
			f_toc, f_landscape, f_12pt, f_enlarged_page, f_pagenumbers,
			extra_praeamble /* extra_praeamble */);


		report_do_the_work(f, Iso, verbose_level);




		L.foot(f);

	}

	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

}

void arc_generator::report_do_the_work(ostream &ost, isomorph::isomorph &Iso, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "arc_generator::report_do_the_work" << endl;
	}



	ost << "\\chapter{Summary}" << endl << endl;
	ost << "There are " << Iso.Folding->Reps->count
			<< " isomorphism types." << endl << endl;


	Iso.Lifting->setup_and_open_solution_database(verbose_level - 1);

	int i, first, /*c,*/ id;
	int u, v, h, rep, tt;
	ring_theory::longinteger_object go;
	long int data[1000];



	ring_theory::longinteger_object *Ago, *Ago_induced;
	int *Ago_int;

	Ago = NEW_OBJECTS(ring_theory::longinteger_object, Iso.Folding->Reps->count);
	Ago_induced = NEW_OBJECTS(ring_theory::longinteger_object, Iso.Folding->Reps->count);
	Ago_int = NEW_int(Iso.Folding->Reps->count);


	for (h = 0; h < Iso.Folding->Reps->count; h++) {
		rep = Iso.Folding->Reps->rep[h];
		first = Iso.Lifting->orbit_fst[rep];
		//c = Iso.starter_number[first];
		id = Iso.Lifting->orbit_perm[first];
		Iso.Lifting->load_solution(id, data, verbose_level - 1);

		groups::sims *Stab;
		
		Stab = Iso.Folding->Reps->stab[h];

		Iso.Folding->Reps->stab[h]->group_order(Ago[h]);
		Ago_int[h] = Ago[h].as_int();
		if (f_v) {
			cout << "arc_generator::report computing induced "
					"action on the set (in data)" << endl;
			}
		Iso.Folding->induced_action_on_set_basic(Stab, data, 0 /*verbose_level*/);
		
			
		Iso.Folding->AA->group_order(Ago_induced[h]);
	}


	data_structures::tally C_ago;

	C_ago.init(Ago_int, Iso.Folding->Reps->count, FALSE, 0);
	cout << "Classification by ago:" << endl;
	C_ago.print(FALSE /*f_backwards*/);



	ost << "\\chapter{Invariants}" << endl << endl;

	ost << "Classification by automorphism group order: ";
	C_ago.print_naked_tex(ost, FALSE /*f_backwards*/);
	ost << "\\\\" << endl;

	ost << "\\begin{center}" << endl;
	ost << "\\begin{tabular}{|c|l|}" << endl;
	ost << "\\hline" << endl;
	ost << "Ago & Isom. Types \\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;

	int cnt, length, t, vv, *set;

	cnt = 0;
	for (u = C_ago.nb_types - 1; u >= 0; u--) {
		first = C_ago.type_first[u];
		length = C_ago.type_len[u];
		t = C_ago.data_sorted[first];

		ost << t << " & ";

		set = NEW_int(length);
		for (v = 0; v < length; v++, cnt++) {
			vv = first + v;
			i = C_ago.sorting_perm_inv[vv];
			set[v] = i;
		}

		Sorting.int_vec_heapsort(set, length);

		for (v = 0; v < length; v++, cnt++) {

			ost << set[v];

			if (v < length - 1) {
				ost << ",";
				if ((v + 1) % 10 == 0) {
					ost << "\\\\" << endl;
					ost << " & " << endl;
				}
			}
		}
		ost << "\\\\" << endl;
		if (u > 0) {
			ost << "\\hline" << endl;
		}
		FREE_int(set);
	}
	ost << "\\hline" << endl;
	ost << "\\end{tabular}" << endl;
	ost << "\\end{center}" << endl << endl;


	ost << "\\clearpage" << endl << endl;

	ost << "\\begin{center}" << endl;
	ost << "\\begin{tabular}{|r|r|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "Isom. Type & $|\\mbox{Aut}|$ & $|\\mbox{Aut}|$ "
			"(induced)\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;

	cnt = 0;
	for (u = 0; u < C_ago.nb_types; u ++) {
		first = C_ago.type_first[u];
		length = C_ago.type_len[u];
		t = C_ago.data_sorted[first];

		set = NEW_int(length);
		for (v = 0; v < length; v++) {
			vv = first + v;
			i = C_ago.sorting_perm_inv[vv];
			set[v] = i;
		}

		Sorting.int_vec_heapsort(set, length);


		for (v = 0; v < length; v++) {
			vv = first + v;
			i = C_ago.sorting_perm_inv[vv];
			h = set[v];
			ost << setw(3) << h << " & ";
			Ago[h].print_not_scientific(ost);
			ost << " & ";
			Ago_induced[h].print_not_scientific(ost);
			ost << "\\\\" << endl;
			cnt++;
			if ((cnt % 30) == 0) {
				ost << "\\hline" << endl;
				ost << "\\end{tabular}" << endl;
				ost << "\\end{center}" << endl << endl;
				ost << "\\begin{center}" << endl;
				ost << "\\begin{tabular}{|r|r|r|}" << endl;
				ost << "\\hline" << endl;
				ost << "Isom. Type & $|\\mbox{Aut}|$ & $|\\mbox{Aut}|$ "
						"(induced)\\\\" << endl;
				ost << "\\hline" << endl;
				ost << "\\hline" << endl;
			}
		}
		FREE_int(set);
	}

	ost << "\\hline" << endl;
	ost << "\\end{tabular}" << endl;
	ost << "\\end{center}" << endl << endl;


	if (Descr->target_size == PA->q + 2) {
		ost << "\\chapter{The Hyperovals}" << endl << endl;
	}
	else {
		ost << "\\chapter{The Arcs}" << endl << endl;
	}

	ost << "\\clearpage" << endl << endl;


	for (h = 0; h < Iso.Folding->Reps->count; h++) {
		rep = Iso.Folding->Reps->rep[h];
		first = Iso.Lifting->orbit_fst[rep];
		//c = Iso.starter_number[first];
		id = Iso.Lifting->orbit_perm[first];
		Iso.Lifting->load_solution(id, data, verbose_level - 1);


		ost << "\\section{Isomorphism type " << h << "}" << endl;
		ost << "\\bigskip" << endl;


		if (Iso.Folding->Reps->stab[h]) {
			Iso.Folding->Reps->stab[h]->group_order(go);
			ost << "Stabilizer has order $";
			go.print_not_scientific(ost);
			ost << "$.\\\\" << endl;
		}
		else {
			//cout << endl;
		}

		groups::sims *Stab;
		
		Stab = Iso.Folding->Reps->stab[h];

		if (f_v) {
			cout << "arc_generator::report computing induced "
					"action on the set (in data)" << endl;
		}
		Iso.Folding->induced_action_on_set_basic(Stab, data, 0 /*verbose_level*/);
		
		ring_theory::longinteger_object go1;
			
		Iso.Folding->AA->group_order(go1);
		cout << "action " << Iso.Folding->AA->label
				<< " computed, group order is " << go1 << endl;

		ost << "Order of the group that is induced on the set is ";
		ost << "$";
		go1.print_not_scientific(ost);
		ost << "$.\\\\" << endl;
		

		groups::schreier Orb;
		//longinteger_object go2;
		
		Iso.Folding->AA->compute_all_point_orbits(Orb,
				Stab->gens, verbose_level - 2);
		ost << "With " << Orb.nb_orbits
				<< " orbits on the set.\\\\" << endl;

		data_structures::tally C_ol;

		C_ol.init(Orb.orbit_len, Orb.nb_orbits, FALSE, 0);

		ost << "Orbit lengths: ";
		//int_vec_print(f, Orb.orbit_len, Orb.nb_orbits);
		C_ol.print_naked_tex(ost, FALSE /*f_backwards*/);
		ost << " \\\\" << endl;
	
		tt = (Descr->target_size + 3) / 4;

		ost << "The points by ranks:\\\\" << endl;
		ost << "\\begin{center}" << endl;

		for (u = 0; u < 4; u++) {
			ost << "\\begin{tabular}[t]{|c|c|c|}" << endl;
			ost << "\\hline" << endl;
			ost << "$i$ & Rank & Unrank\\\\" << endl;
			ost << "\\hline" << endl;
			for (i = 0; i < tt; i++) {
				v = u * tt + i;
				if (v < Descr->target_size) {
					int vec[3];

					point_unrank(vec, data[v]);
					ost << "$" << v << "$ & $" << data[v] << "$ & $";
					Int_vec_print(ost, vec, 3);
					ost << "$\\\\" << endl;
				}
			}
			ost << "\\hline" << endl;
			ost << "\\end{tabular}" << endl;
		}
		ost << "\\end{center}" << endl;


		report_stabilizer(Iso, ost, h /* orbit */, 0 /* verbose_level */);


		report_decompositions(Iso, ost, h /* orbit */,
			data, verbose_level);

	}


	char prefix[1000];
	char label_of_structure_plural[1000];

	snprintf(prefix, sizeof(prefix), "arcs_%d_%d", PA->q, Descr->target_size);
	snprintf(label_of_structure_plural, sizeof(label_of_structure_plural), "Arcs");

	isomorph::isomorph_global IG;

	IG.init(Iso.A_base, Iso.A, Iso.Sub->gen, verbose_level);

	IG.report_data_in_source_code_inside_tex(Iso,
		prefix, label_of_structure_plural, ost,
		verbose_level);


	Iso.Lifting->close_solution_database(verbose_level - 1);

	FREE_int(Ago_int);
	FREE_OBJECTS(Ago);
	FREE_OBJECTS(Ago_induced);

	if (f_v) {
		cout << "arc_generator::report_do_the_work done" << endl;
	}
}

void arc_generator::report_decompositions(
		isomorph::isomorph &Iso, ostream &ost, int orbit,
	long int *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_generator::report_decompositions" << endl;
		}
	groups::sims *Stab;
	groups::strong_generators *gens;

	gens = NEW_OBJECT(groups::strong_generators);

	Stab = Iso.Folding->Reps->stab[orbit];
	gens->init_from_sims(Stab, 0 /* verbose_level */);

	apps_algebra::algebra_global_with_action Algebra;
	
	Algebra.report_tactical_decomposition_by_automorphism_group(
			ost, PA->P,
			PA->A /* A_on_points */, PA->A_on_lines,
			gens, 25 /* size_limit_for_printing */,
			verbose_level);


}

void arc_generator::report_stabilizer(isomorph::isomorph &Iso,
		ostream &ost, int orbit, int verbose_level)
{
	groups::sims *Stab;

	Stab = Iso.Folding->Reps->stab[orbit];
	groups::strong_generators *SG;

	SG = NEW_OBJECT(groups::strong_generators);
	SG->init_from_sims(Stab, verbose_level);

	SG->print_generators_in_latex_individually(ost);

	FREE_OBJECT(SG);
}



// #############################################################################
// global functions
// #############################################################################


static void arc_generator_early_test_function(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	arc_generator *Gen = (arc_generator *) data;
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "arc_generator_early_test_function for set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
	}
	Gen->early_test_func(S, len, 
		candidates, nb_candidates, 
		good_candidates, nb_good_candidates, 
		verbose_level - 2);
	if (f_v) {
		cout << "arc_generator_early_test_function nb_candidates=" << nb_candidates
				<< " nb_good_candidates=" << nb_good_candidates << endl;
	}
	if (f_v)  {
		cout << "arc_generator_early_test_function done" << endl;
	}
}

#if 0
static void arc_generator_lifting_prepare_function_new(
	exact_cover *EC, int starter_case,
	long int *candidates, int nb_candidates,
	groups::strong_generators *Strong_gens,
	solvers::diophant *&Dio, long int *&col_labels,
	int &f_ruled_out, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	arc_generator *Gen = (arc_generator *) EC->user_data;

	if (f_v) {
		cout << "arc_generator_lifting_prepare_function_new "
				"nb_candidates=" << nb_candidates << endl;
	}

	Gen->lifting_prepare_function_new(EC, starter_case, 
		candidates, nb_candidates, Strong_gens, 
		Dio, col_labels, f_ruled_out, 
		verbose_level - 1);


	if (f_v) {
		cout << "arc_generator_lifting_prepare_function_new "
				"nb_rows=" << Dio->m << " nb_cols=" << Dio->n << endl;
	}

	if (f_v) {
		cout << "arc_generator_lifting_prepare_function_new done" << endl;
	}
}
#endif


static void arc_generator_print_arc(ostream &ost, int len, long int *S, void *data)
{
	arc_generator *Gen = (arc_generator *) data;
	
	Gen->print(len, S);
	Gen->print_set_in_affine_plane(len, S);
}

#if 0
static void arc_generator_print_point(long int pt, void *data)
{
	arc_generator *Gen = (arc_generator *) data;
	int v[3];
	
	Gen->PA->F->PG_element_unrank_modified(
			v, 1 /* stride */, 3 /* len */, pt);
	cout << "(" << v[0] << "," << v[1] << "," << v[2] << ")" << endl;
}

static void arc_generator_report(
		isomorph *Iso, void *data, int verbose_level)
{
	arc_generator *Gen = (arc_generator *) data;
	
	Gen->report(*Iso, verbose_level);
}
#endif


}}}



