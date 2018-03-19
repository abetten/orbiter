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

}

void arc_generator::freeself()
{
	if (ECA) {
		delete ECA;
		}
	if (IA) {
		delete IA;
		}
	if (gen) {
		delete gen;
		}
	
	if (Grass) {
		delete Grass;
		}
	if (A) {
		delete A;
		}
	if (A_on_lines) {
		delete A_on_lines;
		}
	if (P) {
		delete P;
		}
	if (line_type) {
		FREE_INT(line_type);
		}
	null();
}

void arc_generator::read_arguments(int argc, const char **argv)
{
	INT i;
	INT f_q = FALSE;

	ECA = new exact_cover_arguments;
	IA = new isomorph_arguments;

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

		}


	ECA->read_arguments(argc, argv, verbose_level);
	IA->read_arguments(argc, argv, verbose_level);


	if (!f_q) {
		cout << "Please specify the field size using the option -q <q>" << endl;
		exit(1);
		}
	if (!f_d) {
		cout << "Please specify the max intersection size using -d <d>" << endl;
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


void arc_generator::main(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "arc_generator::main verbose_level=" << verbose_level << endl;
		}

	
	if (f_starter) {

		if (f_v) {
			cout << "arc_generator::main before compute_starter" << endl;
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
		ECA->prepare_function_new = arc_generator_lifting_prepare_function_new;
		ECA->early_test_function = arc_generator_early_test_function;
		ECA->early_test_function_data = (void *) this;
		
		compute_lifts(ECA, verbose_level);
			// in TOP_LEVEL/extra.C

		}


	IA->execute(verbose_level);




	if (f_v) {
		cout << "arc_generator::main done" << endl;
		}
}


void arc_generator::init(finite_field *F,
	const BYTE *input_prefix, 
	const BYTE *base_fname,
	INT starter_size,  
	int argc, const char **argv, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_generator::init starter_size=" << starter_size << endl;
		cout << "arc_generator::init d=" << d << endl;
		}
	
	arc_generator::F = F;
	q = F->q;
	arc_generator::argc = argc;
	arc_generator::argv = argv;
	strcpy(starter_directory_name, input_prefix);
	strcpy(prefix, base_fname);
	sprintf(prefix_with_directory, "%s%s", starter_directory_name, base_fname);

	if (f_v) {
		cout << "arc_generator::init prefix_with_directory=" << prefix_with_directory << endl;
		}

	arc_generator::starter_size = starter_size;
	
	A = new action;
	A_on_lines = new action;
	AG = new action_on_grassmannian;

	

	f_semilinear = TRUE;
	
	if (f_v) {
		cout << "arc_generator::init" << endl;
		}

	nb_points_total = nb_PG_elements(n, q); // q * q + q + 1;


	if (is_prime(q)) {
		f_semilinear = FALSE;
		}


	INT f_basis = TRUE;
	if (f_v) {
		cout << "arc_generator::init calling init_projective_group" << endl;
		}
	A->init_projective_group(n + 1, F, f_semilinear, f_basis, 0 /*verbose_level*/);

	if (f_v) {
		cout << "arc_generator::init after init_projective_group" << endl;
		}


	
	if (f_v) {
		cout << "arc_generator::init creating action on lines" << endl;
		}
	Grass = new grassmann;

	Grass->init(n + 1 /*n*/, 2 /*k*/, F, verbose_level - 2);
	AG->init(*A, Grass, verbose_level - 2);
	
	A_on_lines->induced_action_on_grassmannian(A, AG, 
		FALSE /*f_induce_action*/, NULL /*sims *old_G */, 
		MINIMUM(verbose_level - 2, 2));
	
	if (f_v) {
		cout << "action A_on_lines created: ";
		A_on_lines->print_info();
		}

	

	if (f_v) {
		cout << "arc_generator::init creating projective plane" << endl;
		}


	P = new projective_space;

	if (f_v) {
		cout << "arc_generator::init before P->init" << endl;
		}
	P->init(n, F, 
		TRUE /* f_init_incidence_structure */, 
		0 /*verbose_level - 2*/);

	if (P->Lines_on_point == NULL) {
		cout << "arc_generator::init P->Lines_on_point == NULL" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "arc_generator::init after P->init" << endl;
		}

	line_type = NEW_INT(P->N_lines);

	cout << "arc_generator::init before prepare_generator" << endl;
	prepare_generator(verbose_level);

	cout << "arc_generator::init before IA->init" << endl;

	IA->init(A, A, gen, 
		target_size, prefix_with_directory, ECA,
		callback_arc_report,
		NULL /* callback_subset_orbits */,
		this,
		verbose_level);

	if (f_v) {
		cout << "arc_generator::init done" << endl;
		}
}


void arc_generator::prepare_generator(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_generator::prepare_generator" << endl;
		cout << "arc_generator::prepare_generator starter_size = " << starter_size << endl;
		}

	gen = new generator;


	gen->read_arguments(argc, argv, 0);

	gen->f_print_function = TRUE;
	gen->print_function = ::arc_print;
	gen->print_function_data = this;

	
	gen->depth = starter_size;
	gen->initialize(A, A,  
		A->Strong_gens, 
		starter_size, 
		starter_directory_name, prefix, verbose_level - 1);


	if (f_no_arc_testing) {
		cout << "arc_generator::prepare_generator installing placebo_test_function" << endl;
		gen->init_check_func(placebo_test_function, 
			(void *)this /* candidate_check_data */);
		}
	else {
		cout << "arc_generator::prepare_generator installing ::check_arc" << endl;
		gen->init_check_func(::check_arc, 
			(void *)this /* candidate_check_data */);
		}
	if (f_v) {
		cout << "arc_generator::prepare_generator done" << endl;
		}
}

void arc_generator::compute_starter(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT t0 = os_ticks();
	

	if (f_v) {
		cout << "arc_generator::compute_starter" << endl;
		}

	gen->f_print_function = TRUE;
	gen->print_function = print_arc;
	gen->print_function_data = this;
	

	INT schreier_depth = 1000;
	INT f_use_invariant_subset_if_available = TRUE;
	INT f_debug = FALSE;
	INT depth;
	INT f_embedded = TRUE;
	INT f_sideways = FALSE;


	if (f_v) {
		cout << "arc_generator::compute_starter before generator_main" << endl;
		cout << "arc_generator::compute_starter gen->fname_base=" << gen->fname_base << endl;
		}

	depth = gen->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		verbose_level);
	if (f_v) {
		cout << "arc_generator::compute_starter gen->main returns depth=" << depth << endl;
		}

	if (f_v) {
		cout << "arc_generator::compute_starter after gen->main" << endl;
		}


#if 0
	if (f_v) {
		cout << "arc_generator::compute_starter before gen->print_data_structure_tex" << endl;
		}

	//gen->print_data_structure_tex(depth, 0 /*gen->verbose_level */);
#endif

	if (f_draw_poset) {
		if (f_v) {
			cout << "arc_generator::compute_starter before gen->draw_poset" << endl;
			}

		gen->draw_poset(gen->fname_base, depth, 0 /* data1 */, f_embedded, f_sideways, 0 /* gen->verbose_level */);
		}


	if (nb_recognize) {
		INT h;

		for (h = 0; h < nb_recognize; h++) {
			INT *recognize_set;
			INT recognize_set_sz;
			INT orb;
			INT *canonical_set;
			INT *Elt_transporter;
			INT *Elt_transporter_inv;

			cout << "recognize " << h << " / " << nb_recognize << endl;
			INT_vec_scan(recognize[h], recognize_set, recognize_set_sz);
			cout << "input set = " << h << " / " << nb_recognize << " : ";
			INT_vec_print(cout, recognize_set, recognize_set_sz);
			cout << endl;
		
			canonical_set = NEW_INT(recognize_set_sz);
			Elt_transporter = NEW_INT(gen->A->elt_size_in_INT);
			Elt_transporter_inv = NEW_INT(gen->A->elt_size_in_INT);
			
			orb = gen->trace_set(recognize_set, recognize_set_sz, recognize_set_sz /* level */, 
				canonical_set, Elt_transporter, 
				0 /*verbose_level */);

			cout << "canonical set = ";
			INT_vec_print(cout, canonical_set, recognize_set_sz);
			cout << endl;
			cout << "is orbit " << orb << endl;
			cout << "transporter:" << endl;
			A->element_print_quick(Elt_transporter, cout);

			A->element_invert(Elt_transporter, Elt_transporter_inv, 0);
			cout << "transporter inverse:" << endl;
			A->element_print_quick(Elt_transporter_inv, cout);

		
			FREE_INT(canonical_set);
			FREE_INT(Elt_transporter);
			FREE_INT(Elt_transporter_inv);
			FREE_INT(recognize_set);
			}
		}
	if (f_list) {
		INT f_show_orbit_decomposition = FALSE, f_show_stab = FALSE, f_save_stab = FALSE, f_show_whole_orbit = FALSE;
		
		gen->generate_source_code(depth, verbose_level);
		
		gen->list_all_orbits_at_level(depth, 
			TRUE, 
			::arc_print, 
			this, 
			f_show_orbit_decomposition, f_show_stab, f_save_stab, f_show_whole_orbit);



		{
		spreadsheet *Sp;
		gen->make_spreadsheet_of_orbit_reps(Sp, depth);
		BYTE fname_csv[1000];
		sprintf(fname_csv, "orbits_%ld.csv", depth);
		Sp->save(fname_csv, verbose_level);
		delete Sp;
		}


#if 0
		INT d;
		for (d = 0; d < 3; d++) {
			gen->print_schreier_vectors_at_depth(d, verbose_level);
			}
#endif
		}



	if (f_v) {
		cout << "arc_generator::compute_starter done" << endl;
		}

}

void arc_generator::early_test_func(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a, b;
		
	if (f_v) {
		cout << "arc_generator::early_test_func checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size " << nb_candidates << ":" << endl;
		INT_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		}


	compute_line_type(S, len, 0 /* verbose_level */);

	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		a = candidates[i];

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

#if 0
INT arc_generator::check_arc(INT *S, INT len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT f_OK = TRUE;


	if (f_v) {
		cout << "checking set ";
		print_set(cout, len, S);
		}
	if (!rc.check_rank(len, S, verbose_level - 1)) {
		return FALSE;
		}
	
	if (f_v) {
		cout << "checking set ";
		print_set(cout, len, S);
		}
	if (f_v) {
		cout << endl;
		//print_integer_matrix(cout, S, 1, len);
		print_integer_matrix(cout, rc.M1, rc.m, len);
		if (len > 2) {
			print_set_in_affine_plane(len - 2, S + 2);
			}
		}



	if (f_OK) {
		if (f_v) {
			cout << "accepted" << endl;
			}
		return TRUE;
		}
	else {
		if (f_v) {
			cout << "rejected" << endl;
			}
		return FALSE;
		}
}
#endif

void arc_generator::print(INT len, INT *S)
{
	INT i, a;
	
	if (len == 0) {
		return;
		}

	compute_line_type(S, len, 0 /* verbose_level */);

	cout << "set ";
	INT_vec_print(cout, S, len);
	cout << " has line type ";

	classify C;

	C.init(line_type, P->N_lines, FALSE, 0);
	C.print_naked(TRUE);
	cout << endl;

	INT *Coord;

	Coord = NEW_INT(len * (n + 1));
	cout << "the coordinates of the points are:" << endl;
	for (i = 0; i < len; i++) {
		a = S[i];
		point_unrank(Coord + i * (n + 1), a);
		}
	for (i = 0; i < len; i++) {
		cout << S[i] << " : ";
		INT_vec_print(cout, Coord + i * (n + 1), (n + 1));
		cout << endl;
		}



	if (f_d && d >= 3) {
		}
	else {
		INT **Pts_on_conic;
		INT *nb_pts_on_conic;
		INT len1;

	
		cout << "Conic intersections:" << endl;

		if (P->n != 2) {
			cout << "conic intersections only defined in the plane" << endl;
			exit(1);
			}
		P->conic_type(
			S, len, 
			Pts_on_conic, nb_pts_on_conic, len1, 
			0 /*verbose_level*/);
		cout << "The arc intersects " << len1 << " conics in 6 or more points. " << endl;

#if 0
		cout << "These intersections are" << endl;
		for (i = 0; i < len1; i++) {
			cout << "conic intersection of size " << nb_pts_on_conic[i] << " : ";
			INT_vec_print(cout, Pts_on_conic[i], nb_pts_on_conic[i]);
			cout << endl;
			}
#endif



#if 0
		if (len1 == 0 && len == 6) {

			eckardt_point *E;
			INT nb_E;
			INT s, t, i, j;
	
			P->find_Eckardt_points_from_arc_not_on_conic(S, E, nb_E, verbose_level);
			cout << "We found " << nb_E << " Eckardt points" << endl;

			for (s = 0; s < nb_E; s++) {
				cout << s << " / " << nb_E << " : ";
				if (E[s].len == 3) {
					cout << "E_{";
					for (t = 0; t < 3; t++) {
						k2ij(E[s].index[t], i, j, 6);
						cout << i + 1 << j + 1;
						if (t < 2) {
							cout << ",";
							}
						}
					cout << "} B-pt=" << E[s].pt << endl;
					}
				else {
					cout << "E_{" << E[s].index[0] + 1 << E[s].index[1] + 1 << "}" << endl;
					}
				}

			cout << "We found " << nb_E << " Eckardt points" << endl;

			delete [] E;
			}
#endif


		for (i = 0; i < len1; i++) {
			FREE_INT(Pts_on_conic[i]);
			}
		FREE_INT(nb_pts_on_conic);
		FREE_PINT(Pts_on_conic);
		}
	
	if (f_simeon) {
		simeon(len, S, simeon_s, verbose_level);
		}

	FREE_INT(Coord);
}

void arc_generator::print_set_in_affine_plane(INT len, INT *S)
{
	::print_set_in_affine_plane(*F, len, S);
}




void arc_generator::point_unrank(INT *v, INT rk)
{
	PG_element_unrank_modified(*F, v, 1 /* stride */, n + 1 /* len */, rk);
}

INT arc_generator::point_rank(INT *v)
{
	INT rk;
	
	PG_element_rank_modified(*F, v, 1 /* stride */, n + 1, rk);
	return rk;
}

void arc_generator::compute_line_type(INT *set, INT len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a, b;

	if (f_v) {
		cout << "arc_generator::compute_line_type" << endl;
		}

	if (P->Lines_on_point == 0) {
		cout << "arc_generator::compute_line_type P->Lines_on_point == 0" << endl;
		exit(1);
		}
	INT_vec_zero(line_type, P->N_lines);
	for (i = 0; i < len; i++) {
		a = set[i];
		for (j = 0; j < P->r; j++) {
			b = P->Lines_on_point[a * P->r + j];
			line_type[b]++;
			}
		}
	
}

void arc_generator::lifting_prepare_function_new(exact_cover *E, INT starter_case, 
	INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, INT *&col_labels, 
	INT &f_ruled_out, 
	INT verbose_level)
// compute the incidence matrix of tangent lines versus candidate points
// extended by external lines versus candidate points
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, a, b;
	INT nb_needed;

	if (f_v) {
		cout << "arc_generator::lifting_prepare_function_new nb_candidates=" << nb_candidates << endl;
		}

	if (n != 2) {
		cout << "arc_generator::lifting_prepare_function_new needs n == 2" << endl;
		exit(1);
		}
	if (d != 2) {
		cout << "arc_generator::lifting_prepare_function_new needs d == 2" << endl;
		exit(1);
		}
	nb_needed = target_size - starter_size;
	f_ruled_out = FALSE;



	compute_line_type(E->starter, starter_size, 0 /* verbose_level */);



	classify C;

	C.init(line_type, P->N_lines, FALSE, 0);
	if (f_v) {
		cout << "arc_generator::lifting_prepare_function_new line_type:" << endl;
		C.print_naked(TRUE);
		cout << endl;
		}


	// extract the tangent lines:

	INT tangent_lines_fst, nb_tangent_lines;
	INT *tangent_lines;
	INT *tangent_line_idx;
	INT external_lines_fst, nb_external_lines;
	INT *external_lines;
	INT *external_line_idx;
	INT fst, len, idx;


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
		cout << "arc_generator::lifting_prepare_function_new there are no tangent lines" << endl;
		exit(1);
		}
	tangent_lines_fst = fst;
	nb_tangent_lines = len;
	tangent_lines = NEW_INT(nb_tangent_lines);
	tangent_line_idx = NEW_INT(P->N_lines);
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
		cout << "arc_generator::lifting_prepare_function_new there are no external lines" << endl;
		exit(1);
		}
	external_lines_fst = fst;
	nb_external_lines = len;
	external_lines = NEW_INT(nb_external_lines);
	external_line_idx = NEW_INT(P->N_lines);
	for (i = 0; i < P->N_lines; i++) {
		external_line_idx[i] = -1;
		}
	for (i = 0; i < len; i++) {
		j = C.sorting_perm_inv[external_lines_fst + i];
		external_lines[i] = j;
		external_line_idx[j] = i;
		}


	
	col_labels = NEW_INT(nb_candidates);


	INT_vec_copy(candidates, col_labels, nb_candidates);

	if (E->f_lex) {
		E->lexorder_test(col_labels, nb_candidates, Strong_gens->gens, 
			verbose_level - 2);
		}

	if (f_vv) {
		cout << "arc_generator::lifting_prepare_function_new after lexorder test" << endl;
		cout << "arc_generator::lifting_prepare_function_new nb_candidates=" << nb_candidates << endl;
		}

	// compute the incidence matrix between
	// tangent lines and candidate points as well as external lines and candidate points:


	INT nb_rows;
	INT nb_cols;

	nb_rows = nb_tangent_lines + nb_external_lines;
	nb_cols = nb_candidates;

	Dio = new diophant;
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
				cout << "arc_generator::lifting_prepare_function candidate lies on a secant" << endl;
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


	FREE_INT(tangent_lines);
	FREE_INT(tangent_line_idx);
	FREE_INT(external_lines);
	FREE_INT(external_line_idx);
	
	if (f_v) {
		cout << "arc_generator::lifting_prepare_function_new done" << endl;
		}
}

INT arc_generator::arc_test(INT *S, INT len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT ret = TRUE;
	INT i;

	if (f_v) {
		cout << "arc_generator::arc_test for set ";
		INT_vec_print(cout, S, len);
		cout << endl;
		}

	if (f_v) {
		cout << "before compute_line_type" << endl;
		}
	compute_line_type(S, len, 0 /* verbose_level */);

	for (i = 0; i < P->N_lines; i++) {
		if (line_type[i] > d) {
			ret = FALSE;
			break;
			}
		}
	return ret;
}

void arc_generator::report(isomorph &Iso, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE fname[1000];

	if (f_v) {
		cout << "arc_generator::report" << endl;
		}
	if (target_size == q + 2) {
		sprintf(fname, "hyperovals_%ld.tex", q);
		}
	else {
		sprintf(fname, "arcs_%ld_%ld.tex", q, target_size);
		}

	{
	ofstream f(fname);
	INT f_book = TRUE;
	INT f_title = TRUE;
	BYTE title[1000];
	const BYTE *author = "Orbiter";
	INT f_toc = TRUE;
	INT f_landscape = FALSE;
	INT f_12pt = FALSE;
	INT f_enlarged_page = TRUE;
	INT f_pagenumbers = TRUE;

	if (target_size == q + 2) {
		sprintf(title, "Hyperovals over ${\\mathbb F}_{%ld}$", q);
		}
	else {
		sprintf(title, "Arcs over  ${\\mathbb F}_{%ld}$ of size $%ld$", q, target_size);
		}
	cout << "Writing file " << fname << " with " << Iso.Reps->count << " arcs:" << endl;
	latex_head(f, f_book, f_title, 
		title, author, 
		f_toc, f_landscape, f_12pt, f_enlarged_page, f_pagenumbers);

	f << "\\chapter{Summary}" << endl << endl;
	f << "There are " << Iso.Reps->count << " isomorphism types." << endl << endl;


	Iso.setup_and_open_solution_database(verbose_level - 1);

	INT i, first, /*c,*/ id;
	INT u, v, h, rep, tt;
	longinteger_object go;
	INT data[1000];



	longinteger_object *Ago, *Ago_induced;
	INT *Ago_INT;

	Ago = new longinteger_object[Iso.Reps->count];
	Ago_induced = new longinteger_object[Iso.Reps->count];
	Ago_INT = NEW_INT(Iso.Reps->count);


	for (h = 0; h < Iso.Reps->count; h++) {
		rep = Iso.Reps->rep[h];
		first = Iso.orbit_fst[rep];
		//c = Iso.starter_number[first];
		id = Iso.orbit_perm[first];		
		Iso.load_solution(id, data);

		sims *Stab;
		
		Stab = Iso.Reps->stab[h];

		Iso.Reps->stab[h]->group_order(Ago[h]);
		Ago_INT[h] = Ago[h].as_INT();
		if (f_v) {
			cout << "arc_generator::report computing induced action on the set (in data)" << endl;
			}
		Iso.induced_action_on_set_basic(Stab, data, 0 /*verbose_level*/);
		
			
		Iso.AA->group_order(Ago_induced[h]);
		}


	classify C_ago;

	C_ago.init(Ago_INT, Iso.Reps->count, FALSE, 0);
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

	INT cnt, length, t, vv, *set;

	cnt = 0;
	for (u = C_ago.nb_types - 1; u >= 0; u--) {
		first = C_ago.type_first[u];
		length = C_ago.type_len[u];
		t = C_ago.data_sorted[first];

		f << t << " & ";

		set = NEW_INT(length);
		for (v = 0; v < length; v++, cnt++) {
			vv = first + v;
			i = C_ago.sorting_perm_inv[vv];
			set[v] = i;
			}

		INT_vec_heapsort(set, length);

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
		FREE_INT(set);
		}
	f << "\\hline" << endl;
	f << "\\end{tabular}" << endl;
	f << "\\end{center}" << endl << endl;


	f << "\\clearpage" << endl << endl;

	f << "\\begin{center}" << endl;
	f << "\\begin{tabular}{|r|r|r|}" << endl;
	f << "\\hline" << endl;
	f << "Isom. Type & $|\\mbox{Aut}|$ & $|\\mbox{Aut}|$ (induced)\\\\" << endl;
	f << "\\hline" << endl;
	f << "\\hline" << endl;

	cnt = 0;
	for (u = 0; u < C_ago.nb_types; u ++) {
		first = C_ago.type_first[u];
		length = C_ago.type_len[u];
		t = C_ago.data_sorted[first];

		set = NEW_INT(length);
		for (v = 0; v < length; v++) {
			vv = first + v;
			i = C_ago.sorting_perm_inv[vv];
			set[v] = i;
			}

		INT_vec_heapsort(set, length);


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
				f << "Isom. Type & $|\\mbox{Aut}|$ & $|\\mbox{Aut}|$ (induced)\\\\" << endl;
				f << "\\hline" << endl;
				f << "\\hline" << endl;
				}
			}
		FREE_INT(set);
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
			cout << "arc_generator::report computing induced action on the set (in data)" << endl;
			}
		Iso.induced_action_on_set_basic(Stab, data, 0 /*verbose_level*/);
		
		longinteger_object go1;
			
		Iso.AA->group_order(go1);
		cout << "action " << Iso.AA->label << " computed, group order is " << go1 << endl;

		f << "Order of the group that is induced on the set is ";
		f << "$";
		go1.print_not_scientific(f);
		f << "$.\\\\" << endl;
		

		schreier Orb;
		//longinteger_object go2;
		
		Iso.AA->compute_all_point_orbits(Orb, Stab->gens, verbose_level - 2);
		f << "With " << Orb.nb_orbits << " orbits on the set.\\\\" << endl;

		classify C_ol;

		C_ol.init(Orb.orbit_len, Orb.nb_orbits, FALSE, 0);

		f << "Orbit lengths: ";
		//INT_vec_print(f, Orb.orbit_len, Orb.nb_orbits);
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
					INT vec[3];

					point_unrank(vec, data[v]);
					f << "$" << v << "$ & $" << data[v] << "$ & $";
					INT_vec_print(f, vec, 3);
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


	BYTE prefix[1000];
	BYTE label_of_structure_plural[1000];

	sprintf(prefix, "arcs_%ld_%ld", q, target_size);
	sprintf(label_of_structure_plural, "Arcs");
	isomorph_report_data_in_source_code_inside_tex(Iso, 
		prefix, label_of_structure_plural, f, 
		verbose_level);


	Iso.close_solution_database(verbose_level - 1);



	latex_foot(f);
	
	FREE_INT(Ago_INT);
	delete [] Ago;
	delete [] Ago_induced;
	}

	cout << "Written file " << fname << " of size " << file_size(fname) << endl;

}

void arc_generator::report_decompositions(isomorph &Iso, ofstream &f, INT orbit, 
	INT *data, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_generator::report_decompositions" << endl;
		}
	incidence_structure *Inc;
	sims *Stab;
	strong_generators *gens;
	INT *Mtx;
	INT i, j, h;

	Inc = new incidence_structure;
	gens = new strong_generators;

	Stab = Iso.Reps->stab[orbit];
	gens->init_from_sims(Stab, 0 /* verbose_level */);

	Mtx = NEW_INT(P->N_points * P->N_lines);
	INT_vec_zero(Mtx, P->N_points * P->N_lines);

	for (j = 0; j < P->N_lines; j++) {
		for (h = 0; h < P->k; h++) {
			i = P->Lines[j * P->k + h];
			Mtx[i * P->N_lines + j] = 1;
			}
		}

	Inc->init_by_matrix(P->N_points, P->N_lines, Mtx, 0 /* verbose_level*/);
	

	partitionstack S;

	INT N;

	if (f_v) {
		cout << "arc_generator::report_decompositions allocating partitionstack" << endl;
		}
	N = Inc->nb_points() + Inc->nb_lines();
	
	S.allocate(N, 0);
	// split off the column class:
	S.subset_continguous(Inc->nb_points(), Inc->nb_lines());
	S.split_cell(0);
	S.split_cell_front_or_back(data, target_size, TRUE /* f_front */, 0 /* verbose_level*/);
				
	INT TDO_depth = N;
	//INT TDO_ht;


	if (f_v) {
		cout << "arc_generator::report_decompositions before Inc->compute_TDO_safe" << endl;
		}
	Inc->compute_TDO_safe(S, TDO_depth, verbose_level - 3);
	//TDO_ht = S.ht;


	if (S.ht < 50) {
		f << "The TDO decomposition is" << endl;
		Inc->get_and_print_column_tactical_decomposition_scheme_tex(f, TRUE /* f_enter_math */, TRUE /* f_print_subscripts */, S);
		}
	else {
		f << "The TDO decomposition is very large (with " << S.ht<< " classes).\\\\" << endl;
		}


	{
		schreier *Sch_points;
		schreier *Sch_lines;
		Sch_points = new schreier;
		Sch_points->init(A /*A_on_points*/);
		Sch_points->initialize_tables();
		Sch_points->init_generators(*gens->gens /* *generators */);
		Sch_points->compute_all_point_orbits(0 /*verbose_level - 2*/);
		
		if (f_v) {
			cout << "found " << Sch_points->nb_orbits << " orbits on points" << endl;
			}
		Sch_lines = new schreier;
		Sch_lines->init(A_on_lines);
		Sch_lines->initialize_tables();
		Sch_lines->init_generators(*gens->gens /* *generators */);
		Sch_lines->compute_all_point_orbits(0 /*verbose_level - 2*/);
		
		if (f_v) {
			cout << "found " << Sch_lines->nb_orbits << " orbits on lines" << endl;
			}
		S.split_by_orbit_partition(Sch_points->nb_orbits, 
			Sch_points->orbit_first, Sch_points->orbit_len, Sch_points->orbit,
			0 /* offset */, 
			verbose_level - 2);
		S.split_by_orbit_partition(Sch_lines->nb_orbits, 
			Sch_lines->orbit_first, Sch_lines->orbit_len, Sch_lines->orbit,
			Inc->nb_points() /* offset */, 
			verbose_level - 2);
		delete Sch_points;
		delete Sch_lines;
	}

	if (S.ht < 50) {
		f << "The TDA decomposition is" << endl;
		Inc->get_and_print_column_tactical_decomposition_scheme_tex(f, TRUE /* f_enter_math */, TRUE /* f_print_subscripts */, S);
		}
	else {
		f << "The TDA decomposition is very large (with " << S.ht<< " classes).\\\\" << endl;
		}

	FREE_INT(Mtx);
	delete gens;
	delete Inc;
}

void arc_generator::report_stabilizer(isomorph &Iso, ofstream &f, INT orbit, INT verbose_level)
{
	sims *Stab;
	longinteger_object go;
	INT i;

	Stab = Iso.Reps->stab[orbit];
	Stab->group_order(go);

	f << "The stabilizer of order $" << go << "$ is generated by:\\\\" << endl;
	for (i = 0; i < Stab->gens.len; i++) {
		
		INT *fp, n, ord;
		
		fp = NEW_INT(A->degree);
		n = A->find_fixed_points(Stab->gens.ith(i), fp, 0);
		//cout << "with " << n << " fixed points" << endl;
		FREE_INT(fp);

		ord = A->element_order(Stab->gens.ith(i));

		f << "$$ g_{" << i + 1 << "}=" << endl;
		A->element_print_latex(Stab->gens.ith(i), f);
		f << "$$" << endl << "of order $" << ord << "$ and with " << n << " fixed points." << endl;
		}
	f << endl << "\\bigskip" << endl;
}

void arc_generator::simeon(INT len, INT *S, INT s, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = FALSE; //(verbose_level >= 1);
	INT k, nb_rows, nb_cols, nb_r1, nb_r2, row, col;
	INT *Coord;
	INT *M;
	INT *A;
	INT *C;
	INT *T;
	INT *Ac; // no not free
	INT *U;
	INT *U1;
	INT nb_A, nb_U;
	INT a, u, ac, i, d, idx, mtx_rank;

	if (f_v) {
		cout << "arc_generator::simeon s=" << s << endl;
		}
	k = n + 1;
	nb_cols = INT_n_choose_k(len, k - 1);
	nb_r1 = INT_n_choose_k(len, s);
	nb_r2 = INT_n_choose_k(len - s, k - 2);
	nb_rows = nb_r1 * nb_r2;
	cout << "nb_r1=" << nb_r1 << endl;
	cout << "nb_r2=" << nb_r2 << endl;
	cout << "nb_rows=" << nb_rows << endl;
	cout << "nb_cols=" << nb_cols << endl;

	Coord = NEW_INT(len * k);
	M = NEW_INT(nb_rows * nb_cols);
	A = NEW_INT(len);
	U = NEW_INT(k - 2);
	U1 = NEW_INT(k - 2);
	C = NEW_INT(k - 1);
	T = NEW_INT(k * k);

	INT_vec_zero(M, nb_rows * nb_cols);


	// unrank all points of the arc:
	for (i = 0; i < len; i++) {
		point_unrank(Coord + i * k, S[i]);
		}

	
	nb_A = INT_n_choose_k(len, k - 2);
	nb_U = INT_n_choose_k(len - (k - 2), k - 1);
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
			INT_vec_print(cout, Ac, ac);
			cout << endl;
			}


		for (u = 0; u < nb_U; u++, row++) {

			unrank_k_subset(u, U, len - (k - 2), k - 1);
			for (i = 0; i < k - 1; i++) {
				U1[i] = Ac[U[i]];
				}
			if (f_vv) {
				cout << "U1=";
				INT_vec_print(cout, U1, k - 1);
				cout << endl;
				}

			for (col = 0; col < nb_cols; col++) {
				if (f_vv) {
					cout << "row=" << row << " / " << nb_rows << " col=" << col << " / " << nb_cols << ":" << endl;
					}
				unrank_k_subset(col, C, len, k - 1);
				if (f_vv) {
					cout << "C: ";
					INT_vec_print(cout, C, k - 1);
					cout << endl;
					}

			
				// test if A is a subset of C:
				for (i = 0; i < k - 2; i++) {
					if (!INT_vec_search_linear(C, k - 1, A[i], idx)) {
						//cout << "did not find A[" << i << "] in C" << endl;
						break;
						}
					}
				if (i == k - 2) {
					d = F->BallChowdhury_matrix_entry(Coord, C, U1, k, s /*sz_U */, 
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
	//INT_matrix_print(M, nb_rows, nb_cols);

	//print_integer_matrix_with_standard_labels(cout, M, nb_rows, nb_cols, TRUE /* f_tex*/);
	//INT_matrix_print_tex(cout, M, nb_rows, nb_cols);

	cout << "nb_rows=" << nb_rows << endl;
	cout << "nb_cols=" << nb_cols << endl;
	cout << "s=" << s << endl;
	
	mtx_rank = F->Gauss_easy(M, nb_rows, nb_cols);
	cout << "mtx_rank=" << mtx_rank << endl;
	//cout << "simeon, the reduced matrix M is:" << endl;
	//INT_matrix_print(M, mtx_rank, nb_cols);


	FREE_INT(Coord);
	FREE_INT(M);
	FREE_INT(A);
	FREE_INT(C);
	//FREE_INT(E);
#if 0
	FREE_INT(A1);
	FREE_INT(C1);
	FREE_INT(E1);
	FREE_INT(A2);
	FREE_INT(C2);
#endif
	FREE_INT(T);
}

#if 0
INT arc_generator::simeon_matrix_entry(INT *Coord, INT *C, INT *E, INT *S, INT len, INT s, 
	INT *T, INT verbose_level)
{
	INT k, u, d, d1, a, i;
	
	k = n + 1;
	d = 1;
	for (u = 0; u < s; u++) {
		a = E[len - s + u];
		INT_vec_copy(Coord + a * k, T, k);
		for (i = 0; i < k - 1; i++) {
			a = C[i];
			INT_vec_copy(Coord + a * k, T + (i + 1) * k, k);
			}
		if (TRUE) {
			cout << "u=" << u << " / " << s << " the matrix is:" << endl;
			INT_matrix_print(T, k, k);
			}
		d1 = F->matrix_determinant(T, k, 0 /* verbose_level */);
		if (TRUE) {
			cout << "determinant = " << d1 << endl;
			}
		d = F->mult(d, d1);
		}
	if (TRUE) {
		cout << "d=" << d << endl;
		}
	return d;
}
#endif


// ##################################################################################################
// global functions
// ##################################################################################################

INT callback_arc_test(exact_cover *EC, INT *S, INT len, void *data, INT verbose_level)
{
	arc_generator *Gen = (arc_generator *) data;
	INT f_OK;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "checking set ";
		print_set(cout, len, S);
		cout << endl;
		}
	f_OK = Gen->arc_test(S, len, verbose_level - 1);
	if (f_OK) {
		if (f_v) {
			cout << "accepted" << endl;
			}
		return TRUE;
		}
	else {
		if (f_v) {
			cout << "rejected" << endl;
			}
		return FALSE;
		}
}


INT check_arc(INT len, INT *S, void *data, INT verbose_level)
{
	arc_generator *Gen = (arc_generator *) data;
	INT f_OK;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "check_arc checking set ";
		print_set(cout, len, S);
		cout << endl;
		}
	f_OK = Gen->arc_test(S, len, verbose_level - 1);
	if (f_OK) {
		if (f_v) {
			cout << "check_arc accepted" << endl;
			}
		return TRUE;
		}
	else {
		if (f_v) {
			cout << "check_arc rejected" << endl;
			}
		return FALSE;
		}
}

INT placebo_test_function(INT len, INT *S, void *data, INT verbose_level)
{
	//arc_generator *Gen = (arc_generator *) data;
	INT f_OK;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "checking set ";
		print_set(cout, len, S);
		cout << endl;
		}
	f_OK = TRUE;
	if (f_OK) {
		if (f_v) {
			cout << "accepted" << endl;
			}
		return TRUE;
		}
	else {
		if (f_v) {
			cout << "rejected" << endl;
			}
		return FALSE;
		}
}

void arc_generator_early_test_function(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level)
{
	arc_generator *Gen = (arc_generator *) data;
	INT f_v = (verbose_level >= 1);
	
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

void placebo_early_test_function(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level)
{
	//arc_generator *Gen = (arc_generator *) data;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "placebo_early_test_function for set ";
		print_set(cout, len, S);
		cout << endl;
		}

	INT_vec_copy(candidates, good_candidates, nb_candidates);
	nb_good_candidates = nb_candidates;

	if (f_v) {
		cout << "placebo_early_test_function done" << endl;
		}
}

void arc_generator_lifting_prepare_function_new(exact_cover *EC, INT starter_case, 
	INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, INT *&col_labels, 
	INT &f_ruled_out, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	arc_generator *Gen = (arc_generator *) EC->user_data;

	if (f_v) {
		cout << "arc_generator_lifting_prepare_function_new nb_candidates=" << nb_candidates << endl;
		}

	Gen->lifting_prepare_function_new(EC, starter_case, 
		candidates, nb_candidates, Strong_gens, 
		Dio, col_labels, f_ruled_out, 
		verbose_level - 1);


	if (f_v) {
		cout << "arc_generator_lifting_prepare_function_new nb_rows=" << Dio->m << " nb_cols=" << Dio->n << endl;
		}

	if (f_v) {
		cout << "arc_generator_lifting_prepare_function_new done" << endl;
		}
}



void print_arc(INT len, INT *S, void *data)
{
	arc_generator *Gen = (arc_generator *) data;
	
	Gen->print_set_in_affine_plane(len, S);
}

void print_point(INT pt, void *data)
{
	arc_generator *Gen = (arc_generator *) data;
	INT v[3];
	
	PG_element_unrank_modified(*Gen->F, v, 1 /* stride */, 3 /* len */, pt);
	cout << "(" << v[0] << "," << v[1] << "," << v[2] << ")" << endl;
}

void callback_arc_report(isomorph *Iso, void *data, INT verbose_level)
{
	arc_generator *Gen = (arc_generator *) data;
	
	Gen->report(*Iso, verbose_level);
}

void arc_print(INT len, INT *S, void *data)
{
	arc_generator *Gen = (arc_generator *) data;
	
	Gen->print(len, S);
}




