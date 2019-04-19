// arc_generator.C
// 
// Anton Betten
//
// previous version Dec 6, 2004
// revised June 19, 2006
// revised Aug 17, 2008
// moved here from hyperoval.C May 10, 2013
// moved to TOP_LEVEL from APPS/ARCS: February 23, 2017
//
// Searches for arcs and hyperovals in Desarguesian projective planes
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {

arc_generator::arc_generator()
{
	null();
}

arc_generator::~arc_generator()
{
	freeself();
}

void arc_generator::null()
{
	f_poly = FALSE;
	
	ECA = NULL;
	IA = NULL;
	gen = NULL;

	A = NULL;
	A_on_lines = NULL;
	Poset = NULL;
	P = NULL;
	line_type = NULL;
	f_d = FALSE;
	d = 0;
	f_n = FALSE;
	n = 2;
	verbose_level = 0;

	f_starter = FALSE;
	f_draw_poset = FALSE;
	f_list = FALSE;
	f_simeon = FALSE;

	f_target_size = FALSE;

	nb_recognize = 0;

	
	f_no_arc_testing = FALSE;

	f_read_data_file = FALSE;
	fname_data_file = NULL;
	depth_completed = 0;


}

void arc_generator::freeself()
{
	if (ECA) {
		FREE_OBJECT(ECA);
		}
	if (IA) {
		FREE_OBJECT(IA);
		}
	if (gen) {
		FREE_OBJECT(gen);
		}
	
	if (Grass) {
		FREE_OBJECT(Grass);
		}
	if (A) {
		FREE_OBJECT(A);
		}
	if (A_on_lines) {
		FREE_OBJECT(A_on_lines);
		}
	if (Poset) {
		FREE_OBJECT(Poset);
	}
	if (P) {
		FREE_OBJECT(P);
		}
	if (line_type) {
		FREE_int(line_type);
		}
	null();
}

void arc_generator::read_arguments(int argc, const char **argv)
{
	int i;
	int f_q = FALSE;

	ECA = NEW_OBJECT(exact_cover_arguments);
	IA = NEW_OBJECT(isomorph_arguments);

	for (i = 1; i < argc; i++) {
		
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
			}
		else if (strcmp(argv[i], "-starter") == 0) {
			f_starter = TRUE;
			cout << "-starter " << endl;
			}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset " << endl;
			}
		else if (strcmp(argv[i], "-no_arc_testing") == 0) {
			f_no_arc_testing = TRUE;
			cout << "-no_arc_testing " << endl;
			}
		else if (strcmp(argv[i], "-target_size") == 0) {
			f_target_size = TRUE;
			target_size = atoi(argv[++i]);
			cout << "-target_size " << target_size << endl;
			}
		else if (strcmp(argv[i], "-d") == 0) {
			f_d = TRUE;
			f_no_arc_testing = FALSE;
			d = atoi(argv[++i]);
			cout << "-d " << d << endl;
			}
		else if (strcmp(argv[i], "-list") == 0) {
			f_list = TRUE;
			list_depth = atoi(argv[++i]);
			cout << "-list " << list_depth << endl;
			}
		else if (strcmp(argv[i], "-simeon") == 0) {
			f_simeon = TRUE;
			simeon_s = atoi(argv[++i]);
			cout << "-simeon " << simeon_s << endl;
			}
		else if (strcmp(argv[i], "-recognize") == 0) {
			recognize[nb_recognize] = argv[++i];
			cout << "-recognize " << recognize[nb_recognize] << endl;
			nb_recognize++;
			}
		else if (strcmp(argv[i], "-read_data_file") == 0) {
			f_read_data_file = TRUE;
			fname_data_file = argv[++i];
			cout << "-read_data_file " << fname_data_file << endl;
			}

		}


	ECA->read_arguments(argc, argv, verbose_level);
	IA->read_arguments(argc, argv, verbose_level);


	if (!f_q) {
		cout << "Please specify the field size "
				"using the option -q <q>" << endl;
		exit(1);
		}
	if (!f_d) {
		cout << "Please specify the max intersection "
				"size using -d <d>" << endl;
		exit(1);
		}
	if (!f_n) {
		cout << "Please specify the dimension -n <n>" << endl;
		exit(1);
		}
	if (!ECA->f_starter_size) {
		cout << "please use option -starter_size <starter_size>" << endl;
		exit(1);
		}
	if (!ECA->f_has_input_prefix) {
		cout << "please use option -input_prefix <input_prefix>" << endl;
		exit(1);
		}


}


void arc_generator::main(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "arc_generator::main "
				"verbose_level=" << verbose_level << endl;
		}

	
	if (f_starter) {

		if (f_v) {
			cout << "arc_generator::main "
					"before compute_starter" << endl;
			}
		compute_starter(verbose_level);
		
		}
	else {
		cout << "not f_starter" << endl;
		}



	if (ECA->f_lift) {
	
		cout << "lift" << endl;
		
		ECA->target_size = target_size;
		ECA->user_data = (void *) this;
		ECA->A = A;
		ECA->A2 = A;
		ECA->prepare_function_new =
				arc_generator_lifting_prepare_function_new;
		ECA->early_test_function =
				arc_generator_early_test_function;
		ECA->early_test_function_data = (void *) this;
		
		ECA->compute_lifts(verbose_level);

		}


	IA->execute(verbose_level);




	if (f_v) {
		cout << "arc_generator::main done" << endl;
		}
}


void arc_generator::init(finite_field *F,
	const char *starter_directory_name,
	const char *base_fname,
	int starter_size,  
	int argc, const char **argv, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory_domain NT;
	geometry_global Gg;

	if (f_v) {
		cout << "arc_generator::init "
				"starter_size=" << starter_size << endl;
		cout << "arc_generator::init "
				"d=" << d << endl;
		}
	
	arc_generator::F = F;
	q = F->q;
	arc_generator::argc = argc;
	arc_generator::argv = argv;
	strcpy(arc_generator::starter_directory_name, starter_directory_name);
	strcpy(prefix, base_fname);
	sprintf(prefix_with_directory, "%s%s",
			starter_directory_name, base_fname);

	if (f_v) {
		cout << "arc_generator::init "
				"prefix_with_directory="
				<< prefix_with_directory << endl;
		}

	arc_generator::starter_size = starter_size;
	
	A = NEW_OBJECT(action);
	A_on_lines = NEW_OBJECT(action);
	AG = NEW_OBJECT(action_on_grassmannian);

	

	f_semilinear = TRUE;
	
	if (f_v) {
		cout << "arc_generator::init" << endl;
		}

	nb_points_total = Gg.nb_PG_elements(n, q); // q * q + q + 1;


	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
		}


	int f_basis = TRUE;
	if (f_v) {
		cout << "arc_generator::init "
				"calling init_projective_group" << endl;
		}
	vector_ge *nice_gens;

	A->init_projective_group(n + 1, F,
			f_semilinear, f_basis,
			nice_gens,
			0 /*verbose_level*/);
	FREE_OBJECT(nice_gens);
	if (f_v) {
		cout << "arc_generator::init "
				"after init_projective_group" << endl;
		}


	
	if (f_v) {
		cout << "arc_generator::init "
				"creating action on lines" << endl;
		}
	Grass = NEW_OBJECT(grassmann);

	Grass->init(n + 1 /*n*/, 2 /*k*/, F, verbose_level - 2);
	AG->init(*A, Grass, verbose_level - 2);
	
	A_on_lines->induced_action_on_grassmannian(A, AG, 
		FALSE /*f_induce_action*/, NULL /*sims *old_G */, 
		MINIMUM(verbose_level - 2, 2));
	
	if (f_v) {
		cout << "arc_generator::init "
				"action A_on_lines created: ";
		A_on_lines->print_info();
		}



	if (f_v) {
		cout << "arc_generator::init "
				"creating projective plane" << endl;
		}


	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "arc_generator::init "
				"before P->init" << endl;
		}
	P->init(n, F, 
		TRUE /* f_init_incidence_structure */, 
		0 /*verbose_level - 2*/);

	if (P->Lines_on_point == NULL) {
		cout << "arc_generator::init "
				"P->Lines_on_point == NULL" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "arc_generator::init "
				"after P->init" << endl;
		}

	line_type = NEW_int(P->N_lines);

	cout << "arc_generator::init "
			"before prepare_generator" << endl;
	prepare_generator(verbose_level);

	cout << "arc_generator::init "
			"before IA->init" << endl;

	IA->init(A, A, gen, 
		target_size, prefix_with_directory, ECA,
		arc_generator_report,
		NULL /* callback_subset_orbits */,
		this,
		verbose_level);

	if (f_v) {
		cout << "arc_generator::init done" << endl;
		}
}


void arc_generator::prepare_generator(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_generator::prepare_generator" << endl;
		cout << "arc_generator::prepare_generator "
				"starter_size = " << starter_size << endl;
		}


	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(A, A, A->Strong_gens, verbose_level);


	if (!f_no_arc_testing) {
		if (f_v) {
			cout << "arc_generator::init before "
					"Poset->add_testing_without_group" << endl;
			}
		Poset->add_testing_without_group(
				arc_generator_early_test_function,
					this /* void *data */,
					verbose_level);
	}


	gen = NEW_OBJECT(poset_classification);


	gen->read_arguments(argc, argv, 0);

	gen->f_print_function = FALSE;
	gen->print_function = arc_generator_print_arc;
	gen->print_function_data = this;

	
	gen->depth = starter_size;
	gen->initialize(Poset,
		starter_size, 
		starter_directory_name, prefix, verbose_level - 1);

#if 0
	if (f_no_arc_testing) {
		cout << "arc_generator::prepare_generator "
				"installing placebo_test_function" << endl;
		gen->init_check_func(placebo_test_function, 
			(void *)this /* candidate_check_data */);
		}
	else {
		cout << "arc_generator::prepare_generator "
				"installing ::check_arc" << endl;
		gen->init_check_func(::check_arc, 
			(void *)this /* candidate_check_data */);
		}
#endif

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

	if (f_v) {
		cout << "arc_generator::prepare_generator done" << endl;
		}
}

void arc_generator::compute_starter(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t0 = os_ticks();
	

	if (f_v) {
		cout << "arc_generator::compute_starter" << endl;
		}

#if 0
	gen->f_print_function = TRUE;
	gen->print_function = callback_arc_print;
	gen->print_function_data = this;
#endif

	int schreier_depth = 1000;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	int depth;
	int f_embedded = TRUE;
	int f_sideways = FALSE;


	if (f_read_data_file) {
		int target_depth;
		if (gen->f_max_depth) {
			target_depth = gen->max_depth;
			}
		else {
			target_depth = gen->depth;
			}
		if (f_v) {
			cout << "arc_generator::compute_starter "
					"before generator_main" << endl;
			cout << "arc_generator::compute_starter "
					"gen->compute_orbits=" << gen->fname_base << endl;
			}
		depth = gen->compute_orbits(depth_completed, target_depth,
				verbose_level);
		if (f_v) {
			cout << "arc_generator::compute_starter "
					"after gen->compute_orbits" << endl;
			}
	} else {
		if (f_v) {
			cout << "arc_generator::compute_starter "
					"before generator_main" << endl;
			cout << "arc_generator::compute_starter "
					"gen->fname_base=" << gen->fname_base << endl;
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
	}

	if (f_v) {
		cout << "arc_generator::compute_starter "
				"finished, depth=" << depth << endl;
		}



#if 0
	if (f_v) {
		cout << "arc_generator::compute_starter "
				"before gen->print_data_structure_tex" << endl;
		}

	//gen->print_data_structure_tex(depth, 0 /*gen->verbose_level */);
#endif

	if (f_draw_poset) {
		if (f_v) {
			cout << "arc_generator::compute_starter "
					"before gen->draw_poset" << endl;
			}

		gen->draw_poset(gen->fname_base, depth, 0 /* data1 */,
				f_embedded, f_sideways, 0 /* gen->verbose_level */);
		}


	if (nb_recognize) {
		int h;

		for (h = 0; h < nb_recognize; h++) {
			int *recognize_set;
			int recognize_set_sz;
			int orb;
			int *canonical_set;
			int *Elt_transporter;
			int *Elt_transporter_inv;

			cout << "recognize " << h << " / " << nb_recognize << endl;
			int_vec_scan(recognize[h], recognize_set, recognize_set_sz);
			cout << "input set = " << h << " / " << nb_recognize << " : ";
			int_vec_print(cout, recognize_set, recognize_set_sz);
			cout << endl;
		
			canonical_set = NEW_int(recognize_set_sz);
			Elt_transporter = NEW_int(gen->Poset->A->elt_size_in_int);
			Elt_transporter_inv = NEW_int(gen->Poset->A->elt_size_in_int);
			
			orb = gen->trace_set(recognize_set,
				recognize_set_sz, recognize_set_sz /* level */,
				canonical_set, Elt_transporter, 
				0 /*verbose_level */);

			cout << "canonical set = ";
			int_vec_print(cout, canonical_set, recognize_set_sz);
			cout << endl;
			cout << "is orbit " << orb << endl;
			cout << "transporter:" << endl;
			A->element_print_quick(Elt_transporter, cout);

			A->element_invert(Elt_transporter, Elt_transporter_inv, 0);
			cout << "transporter inverse:" << endl;
			A->element_print_quick(Elt_transporter_inv, cout);

		
			FREE_int(canonical_set);
			FREE_int(Elt_transporter);
			FREE_int(Elt_transporter_inv);
			FREE_int(recognize_set);
			}
		}
	if (f_list) {
		int f_show_orbit_decomposition = FALSE, f_show_stab = FALSE;
		int f_save_stab = FALSE, f_show_whole_orbit = FALSE;
		
		gen->generate_source_code(depth, verbose_level);
		
		gen->list_all_orbits_at_level(depth, 
			TRUE, 
			arc_generator_print_arc,
			this, 
			f_show_orbit_decomposition,
			f_show_stab,
			f_save_stab,
			f_show_whole_orbit);



		{
		spreadsheet *Sp;
		gen->make_spreadsheet_of_orbit_reps(Sp, depth);
		char fname_csv[1000];
		sprintf(fname_csv, "orbits_%d.csv", depth);
		Sp->save(fname_csv, verbose_level);
		FREE_OBJECT(Sp);
		}


#if 0
		int d;
		for (d = 0; d < 3; d++) {
			gen->print_schreier_vectors_at_depth(d, verbose_level);
			}
#endif
		}



	if (f_v) {
		cout << "arc_generator::compute_starter done" << endl;
		}

}

void arc_generator::early_test_func(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b;
		
	if (f_v) {
		cout << "arc_generator::early_test_func checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size " << nb_candidates << ":" << endl;
		int_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		}


	compute_line_type(S, len, 0 /* verbose_level */);

	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		a = candidates[i];

		// test that there are no more than d points per line:

		for (j = 0; j < P->r; j++) {
			b = P->Lines_on_point[a * P->r + j];
			if (line_type[b] == d) {
				break;
				}
			



			} // next j
		if (j == P->r) {
			good_candidates[nb_good_candidates++] = candidates[i];
			}
		} // next i
	
}

void arc_generator::print(int len, int *S)
{
	int i, a;
	
	if (len == 0) {
		return;
		}

	compute_line_type(S, len, 0 /* verbose_level */);

	cout << "set ";
	int_vec_print(cout, S, len);
	cout << " has line type ";

	classify C;

	C.init(line_type, P->N_lines, FALSE, 0);
	C.print_naked(TRUE);
	cout << endl;

	int *Coord;

	Coord = NEW_int(len * (n + 1));
	cout << "the coordinates of the points are:" << endl;
	for (i = 0; i < len; i++) {
		a = S[i];
		point_unrank(Coord + i * (n + 1), a);
		}
	for (i = 0; i < len; i++) {
		cout << S[i] << " : ";
		int_vec_print(cout, Coord + i * (n + 1), (n + 1));
		cout << endl;
		}



	if (f_d && d >= 3) {
		}
	else {
		int **Pts_on_conic;
		int *nb_pts_on_conic;
		int len1;

	
		cout << "Conic intersections:" << endl;

		if (P->n != 2) {
			cout << "conic intersections "
					"only defined in the plane" << endl;
			exit(1);
			}
		P->conic_type(
			S, len, 
			Pts_on_conic, nb_pts_on_conic, len1, 
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
			FREE_int(Pts_on_conic[i]);
			}
		FREE_int(nb_pts_on_conic);
		FREE_pint(Pts_on_conic);
		}
	
	if (f_simeon) {
		simeon(len, S, simeon_s, verbose_level);
		}

	FREE_int(Coord);
}

void arc_generator::print_set_in_affine_plane(int len, int *S)
{
	F->print_set_in_affine_plane(len, S);
}




void arc_generator::point_unrank(int *v, int rk)
{
	F->PG_element_unrank_modified(v,
			1 /* stride */, n + 1 /* len */, rk);
}

int arc_generator::point_rank(int *v)
{
	int rk;
	
	F->PG_element_rank_modified(v,
			1 /* stride */, n + 1, rk);
	return rk;
}

void arc_generator::compute_line_type(int *set, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b;

	if (f_v) {
		cout << "arc_generator::compute_line_type" << endl;
		}

	if (P->Lines_on_point == 0) {
		cout << "arc_generator::compute_line_type "
				"P->Lines_on_point == 0" << endl;
		exit(1);
		}
	int_vec_zero(line_type, P->N_lines);
	for (i = 0; i < len; i++) {
		a = set[i];
		for (j = 0; j < P->r; j++) {
			b = P->Lines_on_point[a * P->r + j];
			line_type[b]++;
			}
		}
	
}

void arc_generator::lifting_prepare_function_new(
	exact_cover *E, int starter_case,
	int *candidates, int nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, int *&col_labels, 
	int &f_ruled_out, 
	int verbose_level)
// compute the incidence matrix of tangent lines versus candidate points
// extended by external lines versus candidate points
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, a, b;
	int nb_needed;

	if (f_v) {
		cout << "arc_generator::lifting_prepare_function_new "
				"nb_candidates=" << nb_candidates << endl;
		}

	if (n != 2) {
		cout << "arc_generator::lifting_prepare_function_new "
				"needs n == 2" << endl;
		exit(1);
		}
	if (d != 2) {
		cout << "arc_generator::lifting_prepare_function_new "
				"needs d == 2" << endl;
		exit(1);
		}
	nb_needed = target_size - starter_size;
	f_ruled_out = FALSE;



	compute_line_type(E->starter, starter_size, 0 /* verbose_level */);



	classify C;

	C.init(line_type, P->N_lines, FALSE, 0);
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
	tangent_line_idx = NEW_int(P->N_lines);
	for (i = 0; i < P->N_lines; i++) {
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
	external_line_idx = NEW_int(P->N_lines);
	for (i = 0; i < P->N_lines; i++) {
		external_line_idx[i] = -1;
		}
	for (i = 0; i < len; i++) {
		j = C.sorting_perm_inv[external_lines_fst + i];
		external_lines[i] = j;
		external_line_idx[j] = i;
		}


	
	col_labels = NEW_int(nb_candidates);


	int_vec_copy(candidates, col_labels, nb_candidates);

	if (E->f_lex) {
		E->lexorder_test(col_labels, nb_candidates, Strong_gens->gens, 
			verbose_level - 2);
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

	Dio = NEW_OBJECT(diophant);
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
		for (j = 0; j < P->r; j++) {
			b = P->Lines_on_point[a * P->r + j];
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


void arc_generator::report(isomorph &Iso, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname[1000];

	if (f_v) {
		cout << "arc_generator::report" << endl;
		}
	if (target_size == q + 2) {
		sprintf(fname, "hyperovals_%d.tex", q);
		}
	else {
		sprintf(fname, "arcs_%d_%d.tex", q, target_size);
		}

	{
	ofstream f(fname);
	int f_book = TRUE;
	int f_title = TRUE;
	char title[1000];
	const char *author = "Orbiter";
	int f_toc = TRUE;
	int f_landscape = FALSE;
	int f_12pt = FALSE;
	int f_enlarged_page = TRUE;
	int f_pagenumbers = TRUE;
	latex_interface L;

	if (target_size == q + 2) {
		sprintf(title, "Hyperovals over ${\\mathbb F}_{%d}$", q);
		}
	else {
		sprintf(title, "Arcs over  ${\\mathbb F}_{%d}$ "
				"of size $%d$", q, target_size);
		}
	cout << "Writing file " << fname << " with "
			<< Iso.Reps->count << " arcs:" << endl;
	L.head(f, f_book, f_title,
		title, author, 
		f_toc, f_landscape, f_12pt, f_enlarged_page, f_pagenumbers,
		NULL /* extra_praeamble */);

	f << "\\chapter{Summary}" << endl << endl;
	f << "There are " << Iso.Reps->count
			<< " isomorphism types." << endl << endl;


	Iso.setup_and_open_solution_database(verbose_level - 1);

	int i, first, /*c,*/ id;
	int u, v, h, rep, tt;
	longinteger_object go;
	int data[1000];



	longinteger_object *Ago, *Ago_induced;
	int *Ago_int;

	Ago = NEW_OBJECTS(longinteger_object, Iso.Reps->count);
	Ago_induced = NEW_OBJECTS(longinteger_object, Iso.Reps->count);
	Ago_int = NEW_int(Iso.Reps->count);


	for (h = 0; h < Iso.Reps->count; h++) {
		rep = Iso.Reps->rep[h];
		first = Iso.orbit_fst[rep];
		//c = Iso.starter_number[first];
		id = Iso.orbit_perm[first];		
		Iso.load_solution(id, data);

		sims *Stab;
		
		Stab = Iso.Reps->stab[h];

		Iso.Reps->stab[h]->group_order(Ago[h]);
		Ago_int[h] = Ago[h].as_int();
		if (f_v) {
			cout << "arc_generator::report computing induced "
					"action on the set (in data)" << endl;
			}
		Iso.induced_action_on_set_basic(Stab, data, 0 /*verbose_level*/);
		
			
		Iso.AA->group_order(Ago_induced[h]);
		}


	classify C_ago;

	C_ago.init(Ago_int, Iso.Reps->count, FALSE, 0);
	cout << "Classification by ago:" << endl;
	C_ago.print(FALSE /*f_backwards*/);



	f << "\\chapter{Invariants}" << endl << endl;

	f << "Classification by automorphism group order: ";
	C_ago.print_naked_tex(f, FALSE /*f_backwards*/);
	f << "\\\\" << endl;

	f << "\\begin{center}" << endl;
	f << "\\begin{tabular}{|c|l|}" << endl;
	f << "\\hline" << endl;
	f << "Ago & Isom. Types \\\\" << endl;
	f << "\\hline" << endl;
	f << "\\hline" << endl;

	int cnt, length, t, vv, *set;

	cnt = 0;
	for (u = C_ago.nb_types - 1; u >= 0; u--) {
		first = C_ago.type_first[u];
		length = C_ago.type_len[u];
		t = C_ago.data_sorted[first];

		f << t << " & ";

		set = NEW_int(length);
		for (v = 0; v < length; v++, cnt++) {
			vv = first + v;
			i = C_ago.sorting_perm_inv[vv];
			set[v] = i;
			}

		int_vec_heapsort(set, length);

		for (v = 0; v < length; v++, cnt++) {

			f << set[v];

			if (v < length - 1) {
				f << ",";
				if ((v + 1) % 10 == 0) {
					f << "\\\\" << endl;
					f << " & " << endl;
					}
				}
			}
		f << "\\\\" << endl;
		if (u > 0) {
			f << "\\hline" << endl;
			}
		FREE_int(set);
		}
	f << "\\hline" << endl;
	f << "\\end{tabular}" << endl;
	f << "\\end{center}" << endl << endl;


	f << "\\clearpage" << endl << endl;

	f << "\\begin{center}" << endl;
	f << "\\begin{tabular}{|r|r|r|}" << endl;
	f << "\\hline" << endl;
	f << "Isom. Type & $|\\mbox{Aut}|$ & $|\\mbox{Aut}|$ "
			"(induced)\\\\" << endl;
	f << "\\hline" << endl;
	f << "\\hline" << endl;

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

		int_vec_heapsort(set, length);


		for (v = 0; v < length; v++) {
			vv = first + v;
			i = C_ago.sorting_perm_inv[vv];
			h = set[v];
			f << setw(3) << h << " & ";
			Ago[h].print_not_scientific(f);
			f << " & ";
			Ago_induced[h].print_not_scientific(f);
			f << "\\\\" << endl;
			cnt++;
			if ((cnt % 30) == 0) {
				f << "\\hline" << endl;
				f << "\\end{tabular}" << endl;
				f << "\\end{center}" << endl << endl;
				f << "\\begin{center}" << endl;
				f << "\\begin{tabular}{|r|r|r|}" << endl;
				f << "\\hline" << endl;
				f << "Isom. Type & $|\\mbox{Aut}|$ & $|\\mbox{Aut}|$ "
						"(induced)\\\\" << endl;
				f << "\\hline" << endl;
				f << "\\hline" << endl;
				}
			}
		FREE_int(set);
		}

	f << "\\hline" << endl;
	f << "\\end{tabular}" << endl;
	f << "\\end{center}" << endl << endl;


	if (target_size == q + 2) {
		f << "\\chapter{The Hyperovals}" << endl << endl;
		}
	else {
		f << "\\chapter{The Arcs}" << endl << endl;
		}

	f << "\\clearpage" << endl << endl;


	for (h = 0; h < Iso.Reps->count; h++) {
		rep = Iso.Reps->rep[h];
		first = Iso.orbit_fst[rep];
		//c = Iso.starter_number[first];
		id = Iso.orbit_perm[first];		
		Iso.load_solution(id, data);


		f << "\\section{Isomorphism type " << h << "}" << endl;
		f << "\\bigskip" << endl;


		if (Iso.Reps->stab[h]) {
			Iso.Reps->stab[h]->group_order(go);
			f << "Stabilizer has order $";
			go.print_not_scientific(f);
			f << "$.\\\\" << endl;
			}
		else {
			//cout << endl;
			}

		sims *Stab;
		
		Stab = Iso.Reps->stab[h];

		if (f_v) {
			cout << "arc_generator::report computing induced "
					"action on the set (in data)" << endl;
			}
		Iso.induced_action_on_set_basic(Stab, data,
				0 /*verbose_level*/);
		
		longinteger_object go1;
			
		Iso.AA->group_order(go1);
		cout << "action " << Iso.AA->label
				<< " computed, group order is " << go1 << endl;

		f << "Order of the group that is induced on the set is ";
		f << "$";
		go1.print_not_scientific(f);
		f << "$.\\\\" << endl;
		

		schreier Orb;
		//longinteger_object go2;
		
		Iso.AA->compute_all_point_orbits(Orb,
				Stab->gens, verbose_level - 2);
		f << "With " << Orb.nb_orbits
				<< " orbits on the set.\\\\" << endl;

		classify C_ol;

		C_ol.init(Orb.orbit_len, Orb.nb_orbits, FALSE, 0);

		f << "Orbit lengths: ";
		//int_vec_print(f, Orb.orbit_len, Orb.nb_orbits);
		C_ol.print_naked_tex(f, FALSE /*f_backwards*/);
		f << " \\\\" << endl;
	
		tt = (target_size + 3) / 4;

		f << "The points by ranks:\\\\" << endl;
		f << "\\begin{center}" << endl;

		for (u = 0; u < 4; u++) {
			f << "\\begin{tabular}[t]{|c|c|c|}" << endl;
			f << "\\hline" << endl;
			f << "$i$ & Rank & Unrank\\\\" << endl;
			f << "\\hline" << endl;
			for (i = 0; i < tt; i++) {
				v = u * tt + i;
				if (v < target_size) {
					int vec[3];

					point_unrank(vec, data[v]);
					f << "$" << v << "$ & $" << data[v] << "$ & $";
					int_vec_print(f, vec, 3);
					f << "$\\\\" << endl;
					}
				}
			f << "\\hline" << endl;
			f << "\\end{tabular}" << endl;
			}
		f << "\\end{center}" << endl; 


		report_stabilizer(Iso, f, h /* orbit */, 0 /* verbose_level */);


		report_decompositions(Iso, f, h /* orbit */, 
			data, verbose_level);

		}


	char prefix[1000];
	char label_of_structure_plural[1000];

	sprintf(prefix, "arcs_%d_%d", q, target_size);
	sprintf(label_of_structure_plural, "Arcs");
	isomorph_report_data_in_source_code_inside_tex(Iso, 
		prefix, label_of_structure_plural, f, 
		verbose_level);


	Iso.close_solution_database(verbose_level - 1);



	L.foot(f);
	
	FREE_int(Ago_int);
	FREE_OBJECTS(Ago);
	FREE_OBJECTS(Ago_induced);
	}

	cout << "Written file " << fname << " of size "
			<< file_size(fname) << endl;

}

void arc_generator::report_decompositions(
	isomorph &Iso, ofstream &f, int orbit,
	int *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_generator::report_decompositions" << endl;
		}
	incidence_structure *Inc;
	sims *Stab;
	strong_generators *gens;
	int *Mtx;
	int i, j, h;

	Inc = NEW_OBJECT(incidence_structure);
	gens = NEW_OBJECT(strong_generators);

	Stab = Iso.Reps->stab[orbit];
	gens->init_from_sims(Stab, 0 /* verbose_level */);

	Mtx = NEW_int(P->N_points * P->N_lines);
	int_vec_zero(Mtx, P->N_points * P->N_lines);

	for (j = 0; j < P->N_lines; j++) {
		for (h = 0; h < P->k; h++) {
			i = P->Lines[j * P->k + h];
			Mtx[i * P->N_lines + j] = 1;
			}
		}

	Inc->init_by_matrix(P->N_points, P->N_lines, Mtx, 0 /* verbose_level*/);
	

	partitionstack S;

	int N;

	if (f_v) {
		cout << "arc_generator::report_decompositions "
				"allocating partitionstack" << endl;
		}
	N = Inc->nb_points() + Inc->nb_lines();
	
	S.allocate(N, 0);
	// split off the column class:
	S.subset_continguous(Inc->nb_points(), Inc->nb_lines());
	S.split_cell(0);
	S.split_cell_front_or_back(data, target_size,
			TRUE /* f_front */, 0 /* verbose_level*/);
				
	int TDO_depth = N;
	//int TDO_ht;


	if (f_v) {
		cout << "arc_generator::report_decompositions "
				"before Inc->compute_TDO_safe" << endl;
		}
	Inc->compute_TDO_safe(S, TDO_depth, verbose_level - 3);
	//TDO_ht = S.ht;


	if (S.ht < 50) {
		f << "The TDO decomposition is" << endl;
		Inc->get_and_print_column_tactical_decomposition_scheme_tex(
				f, TRUE /* f_enter_math */,
				TRUE /* f_print_subscripts */, S);
		}
	else {
		f << "The TDO decomposition is very large (with "
				<< S.ht<< " classes).\\\\" << endl;
		}


	{
		schreier *Sch_points;
		schreier *Sch_lines;
		Sch_points = NEW_OBJECT(schreier);
		Sch_points->init(A /*A_on_points*/);
		Sch_points->initialize_tables();
		Sch_points->init_generators(*gens->gens /* *generators */);
		Sch_points->compute_all_point_orbits(0 /*verbose_level - 2*/);
		
		if (f_v) {
			cout << "found " << Sch_points->nb_orbits
					<< " orbits on points" << endl;
			}
		Sch_lines = NEW_OBJECT(schreier);
		Sch_lines->init(A_on_lines);
		Sch_lines->initialize_tables();
		Sch_lines->init_generators(*gens->gens /* *generators */);
		Sch_lines->compute_all_point_orbits(0 /*verbose_level - 2*/);
		
		if (f_v) {
			cout << "found " << Sch_lines->nb_orbits
					<< " orbits on lines" << endl;
			}
		S.split_by_orbit_partition(Sch_points->nb_orbits, 
			Sch_points->orbit_first, Sch_points->orbit_len, Sch_points->orbit,
			0 /* offset */, 
			verbose_level - 2);
		S.split_by_orbit_partition(Sch_lines->nb_orbits, 
			Sch_lines->orbit_first, Sch_lines->orbit_len, Sch_lines->orbit,
			Inc->nb_points() /* offset */, 
			verbose_level - 2);
		FREE_OBJECT(Sch_points);
		FREE_OBJECT(Sch_lines);
	}

	if (S.ht < 50) {
		f << "The TDA decomposition is" << endl;
		Inc->get_and_print_column_tactical_decomposition_scheme_tex(
				f, TRUE /* f_enter_math */,
				TRUE /* f_print_subscripts */, S);
		}
	else {
		f << "The TDA decomposition is very large (with "
				<< S.ht<< " classes).\\\\" << endl;
		}

	FREE_int(Mtx);
	FREE_OBJECT(gens);
	FREE_OBJECT(Inc);
}

void arc_generator::report_stabilizer(isomorph &Iso,
		ofstream &f, int orbit, int verbose_level)
{
	sims *Stab;
	longinteger_object go;
	int i;

	Stab = Iso.Reps->stab[orbit];
	Stab->group_order(go);

	f << "The stabilizer of order $" << go
			<< "$ is generated by:\\\\" << endl;
	for (i = 0; i < Stab->gens.len; i++) {
		
		int *fp, n, ord;
		
		fp = NEW_int(A->degree);
		n = A->find_fixed_points(Stab->gens.ith(i), fp, 0);
		//cout << "with " << n << " fixed points" << endl;
		FREE_int(fp);

		ord = A->element_order(Stab->gens.ith(i));

		f << "$$ g_{" << i + 1 << "}=" << endl;
		A->element_print_latex(Stab->gens.ith(i), f);
		f << "$$" << endl << "of order $" << ord << "$ and with "
				<< n << " fixed points." << endl;
		}
	f << endl << "\\bigskip" << endl;
}

void arc_generator::simeon(int len, int *S, int s, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 1);
	int k, nb_rows, nb_cols, nb_r1, nb_r2, row, col;
	int *Coord;
	int *M;
	int *A;
	int *C;
	int *T;
	int *Ac; // no not free
	int *U;
	int *U1;
	int nb_A, nb_U;
	int a, u, ac, i, d, idx, mtx_rank;

	if (f_v) {
		cout << "arc_generator::simeon s=" << s << endl;
		}
	k = n + 1;
	nb_cols = int_n_choose_k(len, k - 1);
	nb_r1 = int_n_choose_k(len, s);
	nb_r2 = int_n_choose_k(len - s, k - 2);
	nb_rows = nb_r1 * nb_r2;
	cout << "nb_r1=" << nb_r1 << endl;
	cout << "nb_r2=" << nb_r2 << endl;
	cout << "nb_rows=" << nb_rows << endl;
	cout << "nb_cols=" << nb_cols << endl;

	Coord = NEW_int(len * k);
	M = NEW_int(nb_rows * nb_cols);
	A = NEW_int(len);
	U = NEW_int(k - 2);
	U1 = NEW_int(k - 2);
	C = NEW_int(k - 1);
	T = NEW_int(k * k);

	int_vec_zero(M, nb_rows * nb_cols);


	// unrank all points of the arc:
	for (i = 0; i < len; i++) {
		point_unrank(Coord + i * k, S[i]);
		}

	
	nb_A = int_n_choose_k(len, k - 2);
	nb_U = int_n_choose_k(len - (k - 2), k - 1);
	if (nb_A * nb_U != nb_rows) {
		cout << "nb_A * nb_U != nb_rows" << endl;
		exit(1);
		}
	cout << "nb_A=" << nb_A << endl;
	cout << "nb_U=" << nb_U << endl;


	Ac = A + k - 2;

	row = 0;
	for (a = 0; a < nb_A; a++) {
		if (f_vv) {
			cout << "a=" << a << " / " << nb_A << ":" << endl;
			}
		unrank_k_subset(a, A, len, k - 2);
		set_complement(A, k - 2, Ac, ac, len);
		if (ac != len - (k - 2)) {
			cout << "arc_generator::simeon ac != len - (k - 2)" << endl;
			exit(1);
			}
		if (f_vv) {
			cout << "Ac=";
			int_vec_print(cout, Ac, ac);
			cout << endl;
			}


		for (u = 0; u < nb_U; u++, row++) {

			unrank_k_subset(u, U, len - (k - 2), k - 1);
			for (i = 0; i < k - 1; i++) {
				U1[i] = Ac[U[i]];
				}
			if (f_vv) {
				cout << "U1=";
				int_vec_print(cout, U1, k - 1);
				cout << endl;
				}

			for (col = 0; col < nb_cols; col++) {
				if (f_vv) {
					cout << "row=" << row << " / " << nb_rows
							<< " col=" << col << " / "
							<< nb_cols << ":" << endl;
					}
				unrank_k_subset(col, C, len, k - 1);
				if (f_vv) {
					cout << "C: ";
					int_vec_print(cout, C, k - 1);
					cout << endl;
					}

			
				// test if A is a subset of C:
				for (i = 0; i < k - 2; i++) {
					if (!int_vec_search_linear(C, k - 1, A[i], idx)) {
						//cout << "did not find A[" << i << "] in C" << endl;
						break;
						}
					}
				if (i == k - 2) {
					d = F->BallChowdhury_matrix_entry(
							Coord, C, U1, k, s /*sz_U */,
						T, 0 /*verbose_level*/);
					if (f_vv) {
						cout << "d=" << d << endl;
						}

					M[row * nb_cols + col] = d;
					} // next a
				} // next c
			} // next u
		} // next a

	cout << "simeon, the matrix M is:" << endl;
	//int_matrix_print(M, nb_rows, nb_cols);

	//print_integer_matrix_with_standard_labels(cout, M,
	//nb_rows, nb_cols, TRUE /* f_tex*/);
	//int_matrix_print_tex(cout, M, nb_rows, nb_cols);

	cout << "nb_rows=" << nb_rows << endl;
	cout << "nb_cols=" << nb_cols << endl;
	cout << "s=" << s << endl;
	
	mtx_rank = F->Gauss_easy(M, nb_rows, nb_cols);
	cout << "mtx_rank=" << mtx_rank << endl;
	//cout << "simeon, the reduced matrix M is:" << endl;
	//int_matrix_print(M, mtx_rank, nb_cols);


	FREE_int(Coord);
	FREE_int(M);
	FREE_int(A);
	FREE_int(C);
	//FREE_int(E);
#if 0
	FREE_int(A1);
	FREE_int(C1);
	FREE_int(E1);
	FREE_int(A2);
	FREE_int(C2);
#endif
	FREE_int(T);
}


// #############################################################################
// global functions
// #############################################################################


void arc_generator_early_test_function(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level)
{
	arc_generator *Gen = (arc_generator *) data;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "arc_generator_early_test_function for set ";
		print_set(cout, len, S);
		cout << endl;
		}
	Gen->early_test_func(S, len, 
		candidates, nb_candidates, 
		good_candidates, nb_good_candidates, 
		verbose_level - 2);
	if (f_v) {
		cout << "arc_generator_early_test_function done" << endl;
		}
}


void arc_generator_lifting_prepare_function_new(
	exact_cover *EC, int starter_case,
	int *candidates, int nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, int *&col_labels, 
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
		cout << "arc_generator_lifting_prepare_function_new "
				"done" << endl;
		}
}



void arc_generator_print_arc(int len, int *S, void *data)
{
	arc_generator *Gen = (arc_generator *) data;
	
	Gen->print_set_in_affine_plane(len, S);
}

void arc_generator_print_point(int pt, void *data)
{
	arc_generator *Gen = (arc_generator *) data;
	int v[3];
	
	Gen->F->PG_element_unrank_modified(
			v, 1 /* stride */, 3 /* len */, pt);
	cout << "(" << v[0] << "," << v[1] << "," << v[2] << ")" << endl;
}

void arc_generator_report(
		isomorph *Iso, void *data, int verbose_level)
{
	arc_generator *Gen = (arc_generator *) data;
	
	Gen->report(*Iso, verbose_level);
}

void arc_generator_print_arc(
		ostream &ost, int len, int *S, void *data)
{
	arc_generator *Gen = (arc_generator *) data;
	
	Gen->print(len, S);
}

}}



