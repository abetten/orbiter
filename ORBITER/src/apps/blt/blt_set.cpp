// blt_set.C
// 
// Anton Betten
//
// started 8/13/2006
//
// moved here from blt.C 5/24/09
//
//
//
//

#include "orbiter.h"
#include "blt.h"

void blt_set::read_arguments(int argc, const char **argv)
{
	INT i;
		
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-schreier") == 0) {
			f_override_schreier_depth = TRUE;
			override_schreier_depth = atoi(argv[++i]);
			cout << "-schreier " << override_schreier_depth << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_override_n = TRUE;
			override_n = atoi(argv[++i]);
			cout << "-override_n " << override_n << endl;
			}
		else if (strcmp(argv[i], "-epsilon") == 0) {
			f_override_epsilon = TRUE;
			override_epsilon = atoi(argv[++i]);
			cout << "-override_epsilon " << override_epsilon << endl;
			}
		else if (strcmp(argv[i], "-BLT") == 0) {
			f_BLT = TRUE;
			cout << "-BLT " << endl;
			}
		else if (strcmp(argv[i], "-ovoid") == 0) {
			f_ovoid = TRUE;
			cout << "-ovoid " << endl;
			}
		else if (strcmp(argv[i], "-semilinear") == 0) {
			f_semilinear = TRUE;
			cout << "-semilinear" << endl;
			}
		}
	if (!f_BLT && !f_ovoid) {
		cout << "please use either -BLT or -ovoid" << endl;
		exit(1);
		}
}

blt_set::blt_set()
{
	null();
}

blt_set::~blt_set()
{
	freeself();
}

void blt_set::null()
{
	//override_poly = NULL;
	f_semilinear = FALSE;
	gen = NULL;
	F = NULL;
	A = NULL;
	O = NULL;
	f_BLT = FALSE;
	f_ovoid = FALSE;
	f_semilinear = FALSE;
	f_orthogonal_allocated = FALSE;
	nb_sol = 0;
	f_override_schreier_depth = FALSE;
	f_override_n = FALSE;
	override_n = 0;
	f_override_epsilon = FALSE;
	override_epsilon = 0;
	Pts = NULL;
	Candidates = NULL;
}

void blt_set::freeself()
{
	INT f_v = FALSE;

	if (f_v) {
		cout << "blt_set::freeself before A" << endl;
		}
	if (A) {
		delete A;
		A = NULL;
		}
	if (f_v) {
		cout << "blt_set::freeself before gen" << endl;
		}
	if (gen) {
		delete gen;
		gen = NULL;
		}
	if (f_orthogonal_allocated) {
		if (f_v) {
			cout << "blt_set::freeself before O" << endl;
			}
		if (O) {
			delete O;
			}
		f_orthogonal_allocated = FALSE;
		O = NULL;
		}
	if (Pts) {
		FREE_INT(Pts);
		}
	if (Candidates) {
		FREE_INT(Candidates);
		}
	null();
	if (f_v) {
		cout << "blt_set::freeself done" << endl;
		}
	
}



void blt_set::init_basic(finite_field *F, 
	const BYTE *input_prefix, 
	const BYTE *base_fname,
	INT starter_size,  
	int argc, const char **argv, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "blt_set::init_basic" << endl;
		cout << "blt_set::init_basic "
				"verbose_level = " << verbose_level << endl;
		}

	if (f_vv) {
		cout << "blt_set::init_basic "
				"before read_arguments" << endl;
		}

	read_arguments(argc, argv);


	gen = new generator;
	gen->read_arguments(argc, argv, 0);
	


	blt_set::F = F;
	blt_set::q = F->q;

	strcpy(starter_directory_name, input_prefix);
	strcpy(prefix, base_fname);
	sprintf(prefix_with_directory, "%s%s",
			starter_directory_name, base_fname);
	blt_set::starter_size = starter_size;

	target_size = q + 1;
	strcpy(gen->fname_base, prefix_with_directory);
		

	if (f_vv) {
		cout << "blt_set::init_basic q=" << q
				<< " target_size = " << target_size << endl;
		}
	
	n = 0;
	epsilon = 0;
	
	
	if (f_BLT) {
		epsilon = 0;
		n = 5;
		}
	else if (f_ovoid) {
		if (f_override_n) {
			n = override_n;
			if (f_vv) {
				cout << "blt_set::init_basic "
						"override value of n=" << n << endl;
				}
			}
		if (f_override_epsilon) {
			epsilon = override_epsilon;
			if (f_vv) {
				cout << "blt_set::init_basic "
						"override value of epsilon=" << epsilon << endl;
				}
			}
		}
	else {
		cout << "neither f_BLT nor f_ovoid is TRUE" << endl;
		exit(1);
		}
	
	f_semilinear = TRUE;
	if (is_prime(q)) {
		f_semilinear = FALSE;
		}
	if (f_vv) {
		cout << "blt_set::init_basic "
				"f_semilinear=" << f_semilinear << endl;
		}
	if (f_v) {
		cout << "blt_set::init_basic finished" << endl;
		}
}

void blt_set::init_group(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_basis = TRUE;

	if (f_v) {
		cout << "blt_set::init_group" << endl;
		}
	if (f_vv) {
		cout << "blt_set::init_group "
				"epsilon=" << epsilon << endl;
		cout << "blt_set::init_group "
				"n=" << n << endl;
		cout << "blt_set::init_group "
				"q=" << q << endl;
		cout << "blt_set::init_group "
				"f_semilinear=" << f_semilinear << endl;
		}
	if (f_vv) {
		cout << "blt_set::init_group "
				"before A->init_orthogonal_group" << endl;
		}
	A = new action;

	A->init_orthogonal_group(epsilon, n, F, 
		TRUE /* f_on_points */, 
		FALSE /* f_on_lines */, 
		FALSE /* f_on_points_and_lines */, 
		f_semilinear, f_basis, verbose_level - 1);
	degree = A->degree;
	if (f_vv) {
		cout << "blt_set::init_group "
				"after A->init_orthogonal_group" << endl;
		cout << "blt_set::init_group "
				"degree = " << degree << endl;
		}
	
	if (f_vv) {
		cout << "blt_set::init_group "
				"computing lex least base" << endl;
		}
	A->lex_least_base_in_place(0 /*verbose_level - 2*/);
	if (f_vv) {
		cout << "blt_set::init_group "
				"computing lex least base done" << endl;
		cout << "blt_set::init_group base: ";
		INT_vec_print(cout, A->base, A->base_len);
		cout << endl;
		}
	
	action_on_orthogonal *AO;

	AO = A->G.AO;
	O = AO->O;

	if (f_v) {
		cout << "blt_set::init_group "
				"degree = " << A->degree << endl;
		}
		
	//init_orthogonal_hash(verbose_level);

	if (A->degree < 200) {
		if (f_v) {
			cout << "blt_set::init_group "
					"before test_Orthogonal" << endl;
			}
		test_Orthogonal(epsilon, n - 1, q);
		}
	//A->Sims->print_all_group_elements();

	if (FALSE) {
		cout << "blt_set::init_group before "
				"A->Sims->print_all_transversal_elements" << endl;
		A->Sims->print_all_transversal_elements();
		cout << "blt_set::init_group after "
				"A->Sims->print_all_transversal_elements" << endl;
		}


	if (FALSE /*f_vv*/) {
		O->F->print(FALSE);
		}


	
	if (f_v) {
		cout << "blt_set::init_group "
				"allocating Pts and Candidates" << endl;
		}
	//int Pts_size = target_size * n;
	Pts = NEW_INT(target_size * n);

#if 0
	for (int i=0; i<Pts_size; i++) Pts[i] = 0;
			// set all the points to an
			// initial value of zero
			// in order to prevent 
			// finite_field::mult(INT i, INT j)
			// from throwing an error.
#endif

	Candidates = NEW_INT(degree * n);
	
	if (f_v) {
		cout << "blt_set::init_group finished" << endl;
		}
}

#if 0
void blt_set::init_orthogonal(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "blt_set::init_orthogonal" << endl;
		}
	if (f_vv) {
		cout << "epsilon=" << epsilon << endl;
		cout << "n=" << n << endl;
		cout << "q=" << q << endl;
		cout << "f_semilinear=" << f_semilinear << endl;
		}

	f_orthogonal_allocated = TRUE;
	O = new orthogonal;
	O->init(epsilon, n, F, verbose_level - 3);
	if (f_vv) {
		cout << "created O^" << plus_minus_string(epsilon)
				<< "(" << n << "," << q << ") with "
			<< O->nb_points << " points and " << O->nb_lines
			<< " lines" << endl << endl;
		}

	init_orthogonal_hash(verbose_level);

	if (f_vv) {
		O->F->print(FALSE);
		}

	if (f_v) {
		cout << "blt_set::init_orthogonal finished" << endl;
		}
}
#endif

void blt_set::init_orthogonal_hash(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set::init_orthogonal_hash" << endl;
		}

	init_hash_table_parabolic(*O->F, 4, 0/*verbose_level*/);

	if (f_v) {
		cout << "blt_set::init_orthogonal finished" << endl;
		}
}

void blt_set::init2(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "blt_set::init2" << endl;
		}


	if (gen->f_max_depth) {
		gen->depth = gen->max_depth;
		}
	else {
		gen->depth = starter_size;
		}
	
	if (f_v) {
		cout << "blt_set::init2 depth = " << gen->depth << endl;
		}

	
	gen->init(A, A, A->Strong_gens, 
		gen->depth /* sz */, verbose_level);
	
#if 0
	// not needed since we have an early_test_func:
	gen->init_check_func(::check_conditions, 
		(void *)this /* candidate_check_data */);
#endif

	// we have an early test function:

	gen->init_early_test_func(
		early_test_func_callback, 
		this,  
		verbose_level);

	// We also have an incremental check function. 
	// This is only used by the clique finder:
	gen->init_incremental_check_func(
		check_function_incremental_callback, 
		this /* candidate_check_data */);


	gen->f_print_function = TRUE;
	gen->print_function = print_set;
	gen->print_function_data = (void *) this;
	
	
	INT nb_nodes = ONE_MILLION;
	
	if (f_vv) {
		cout << "blt_set::init2 calling init_poset_orbit_node with "
				<< nb_nodes << " nodes" << endl;
		}
	
	gen->init_poset_orbit_node(nb_nodes, verbose_level - 1);

	if (f_vv) {
		cout << "blt_set::init2 after init_root_node" << endl;
		}
	
	//cout << "verbose_level = " << verbose_level << endl;
	//cout << "verbose_level_group_theory = "
	//<< verbose_level_group_theory << endl;
	
	gen->root[0].init_root_node(gen, 0/*verbose_level - 2*/);
	if (f_v) {
		cout << "blt_set::init2 done" << endl;
		}
}





void blt_set::create_graphs(
	INT orbit_at_level_r, INT orbit_at_level_m, 
	INT level_of_candidates_file, 
	const BYTE *output_prefix, 
	INT f_lexorder_test, INT f_eliminate_graphs_if_possible, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_v3 = (verbose_level >= 3);


	if (f_v) {
		cout << "blt_set::create_graphs" << endl;
		cout << "blt_set::create_graphs "
				"starter_size = " << starter_size << endl;
		cout << "blt_set::create_graphs "
				"f_lexorder_test=" << f_lexorder_test << endl;
		}


	//f_memory_debug = TRUE;


	BYTE fname[1000];
	BYTE fname_list_of_cases[1000];
	BYTE fname_time[1000];
	BYTE graph_fname_base[1000];
	INT orbit;
	INT nb_orbits;
	INT *list_of_cases;
	INT nb_of_cases;

	INT *Time;
	INT time_idx;




	sprintf(fname, "%s_lvl_%ld", prefix_with_directory, starter_size);
	sprintf(fname_list_of_cases, "%slist_of_cases_%s_%ld_%ld_%ld.txt",
			output_prefix, prefix, starter_size,
			orbit_at_level_r, orbit_at_level_m);
	sprintf(fname_time, "%stime_%s_%ld_%ld_%ld.csv",
			output_prefix, prefix, starter_size,
			orbit_at_level_r, orbit_at_level_m);

	nb_orbits = count_number_of_orbits_in_file(fname, 0);
	if (f_v) {
		cout << "blt_set::create_graphs There are "
				<< nb_orbits << " starters" << endl;
		}
	if (nb_orbits < 0) {
		cout << "Something is wrong, nb_orbits is negative" << endl;
		exit(1);
		}


	Time = NEW_INT(nb_orbits * 2);
	INT_vec_zero(Time, nb_orbits * 2);
	time_idx = 0;

	nb_of_cases = 0;
	list_of_cases = NEW_INT(nb_orbits);
	for (orbit = 0; orbit < nb_orbits; orbit++) {
		if ((orbit % orbit_at_level_m) != orbit_at_level_r) {
			continue;
			}
		if (f_v3) {
			cout << "blt_set::create_graphs creating graph associated "
					"with orbit " << orbit << " / " << nb_orbits
					<< ":" << endl;
			}

		
		colored_graph *CG = NULL;
		INT nb_vertices = -1;

		INT t0 = os_ticks();
		
		if (create_graph(orbit, level_of_candidates_file, 
			output_prefix, 
			f_lexorder_test, f_eliminate_graphs_if_possible, 
			nb_vertices, graph_fname_base,
			CG,  
			verbose_level - 2)) {
			list_of_cases[nb_of_cases++] = orbit;

			BYTE fname[1000];

			sprintf(fname, "%s%s.bin", output_prefix, CG->fname_base);
			CG->save(fname, verbose_level - 2);
			
			nb_vertices = CG->nb_points;
			}

		if (CG) {
			delete CG;
			}

		INT t1 = os_ticks();

		Time[time_idx * 2 + 0] = orbit;
		Time[time_idx * 2 + 1] = t1 - t0;
		time_idx++;
		
		if (f_vv) {
			if (nb_vertices >= 0) {
				cout << "blt_set::create_graphs creating graph "
						"associated with orbit " << orbit << " / "
						<< nb_orbits << " with " << nb_vertices
						<< " vertices created" << endl;
				}
			else {
				cout << "blt_set::create_graphs creating graph "
						"associated with orbit " << orbit << " / "
						<< nb_orbits << " is ruled out" << endl;
				}
			}
		}

	if (f_v) {
		cout << "blt_set::create_graphs writing file "
				<< fname_time << endl;
		}
	INT_matrix_write_csv(fname_time, Time, time_idx, 2);
	if (f_v) {
		cout << "blt_set::create_graphs Written file "
				<< fname_time << " of size "
				<< file_size(fname_time) << endl;
		}

	write_set_to_file(fname_list_of_cases,
			list_of_cases, nb_of_cases,
			0 /*verbose_level */);
	if (f_v) {
		cout << "blt_set::create_graphs Written file "
				<< fname_list_of_cases << " of size "
				<< file_size(fname_list_of_cases) << endl;
		}

	FREE_INT(Time);
	FREE_INT(list_of_cases);

	//registry_dump_sorted();
}

void blt_set::create_graphs_list_of_cases(
	const BYTE *case_label, 
	const BYTE *list_of_cases_text, 
	INT level_of_candidates_file, 
	const BYTE *output_prefix, 
	INT f_lexorder_test, INT f_eliminate_graphs_if_possible, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_v3 = (verbose_level >= 3);


	if (f_v) {
		cout << "blt_set::create_graphs_list_of_cases" << endl;
		cout << "blt_set::create_graphs_list_of_cases "
				"case_label = " << case_label << endl;
		}

	
	//f_memory_debug = TRUE;

	INT *list_of_cases = NULL;
	INT nb_of_cases;


	INT_vec_scan(list_of_cases_text, list_of_cases, nb_of_cases);
	if (f_v) {
		cout << "blt_set::create_graphs_list_of_cases "
				"nb_of_cases = " << nb_of_cases << endl;
		cout << "blt_set::create_graphs_list_of_cases "
				"starter_size = " << starter_size << endl;
		cout << "blt_set::create_graphs_list_of_cases "
				"f_lexorder_test=" << f_lexorder_test << endl;
		}

	BYTE fname[1000];
	BYTE fname_list_of_cases[1000];
	BYTE graph_fname_base[1000];
	INT orbit;
	INT nb_orbits;
	INT *list_of_cases_created;
	INT nb_of_cases_created;
	INT c;




	sprintf(fname, "%s_lvl_%ld", prefix_with_directory, starter_size);
	sprintf(fname_list_of_cases, "%s%s_list_of_cases.txt",
			output_prefix, case_label);

	nb_orbits = count_number_of_orbits_in_file(fname, 0);
	if (f_v) {
		cout << "blt_set::create_graphs_list_of_cases "
				"There are " << nb_orbits << " starters" << endl;
		}
	if (nb_orbits < 0) {
		cout << "Something is wrong, nb_orbits is negative" << endl;
		cout << "fname = " << fname << endl;
		exit(1);
		}


	nb_of_cases_created = 0;
	list_of_cases_created = NEW_INT(nb_orbits);
	for (c = 0; c < nb_of_cases; c++) {
		orbit = list_of_cases[c];
		if (f_v3) {
			cout << "blt_set::create_graphs_list_of_cases case "
					<< c << " / " << nb_of_cases << " creating graph "
							"associated with orbit " << orbit << " / "
							<< nb_orbits << ":" << endl;
			}

		
		colored_graph *CG = NULL;
		INT nb_vertices = -1;


		if (create_graph(orbit, level_of_candidates_file, 
			output_prefix, 
			f_lexorder_test, f_eliminate_graphs_if_possible, 
			nb_vertices, graph_fname_base,
			CG,  
			verbose_level - 2)) {
			list_of_cases_created[nb_of_cases_created++] = orbit;

			BYTE fname[1000];

			sprintf(fname, "%s%s.bin", output_prefix, CG->fname_base);
			CG->save(fname, verbose_level - 2);
			
			nb_vertices = CG->nb_points;
			//delete CG;
			}

		if (CG) {
			delete CG;
			}
		if (f_vv) {
			if (nb_vertices >= 0) {
				cout << "blt_set::create_graphs_list_of_cases "
						"case " << c << " / " << nb_of_cases
						<< " creating graph associated with orbit "
						<< orbit << " / " << nb_orbits << " with "
						<< nb_vertices << " vertices created" << endl;
				}
			else {
				cout << "blt_set::create_graphs_list_of_cases "
						"case " << c << " / " << nb_of_cases
						<< " creating graph associated with orbit "
						<< orbit << " / " << nb_orbits
						<< " is ruled out" << endl;
				}
			}
		}

	write_set_to_file(fname_list_of_cases,
			list_of_cases_created, nb_of_cases_created,
			0 /*verbose_level */);
	if (f_v) {
		cout << "blt_set::create_graphs_list_of_cases "
				"Written file " << fname_list_of_cases
				<< " of size " << file_size(fname_list_of_cases) << endl;
		}
	if (f_v) {
		cout << "blt_set::create_graphs_list_of_cases "
				"we created " << nb_of_cases_created
				<< " / " << nb_of_cases << " cases" << endl;
		}

	FREE_INT(list_of_cases_created);

	//registry_dump_sorted();
}


INT blt_set::create_graph(
	INT orbit_at_level, INT level_of_candidates_file, 
	const BYTE *output_prefix, 
	INT f_lexorder_test, INT f_eliminate_graphs_if_possible, 
	INT &nb_vertices, BYTE *graph_fname_base,
	colored_graph *&CG,  
	INT verbose_level)
// returns TRUE if a graph was written, FALSE otherwise
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_v3 = (verbose_level >= 3);


	if (f_v) {
		cout << "blt_set::create_graph" << endl;
		cout << "blt_set::create_graph "
				"f_lexorder_test=" << f_lexorder_test << endl;
		cout << "blt_set::create_graph "
				"orbit_at_level=" << orbit_at_level << endl;
		cout << "blt_set::create_graph "
				"level_of_candidates_file="
				<< level_of_candidates_file << endl;
		}

	CG = NULL;
	
	INT ret;

	orbit_rep *R;



	INT max_starter;
	INT nb;

	nb_vertices = 0;


	R = new orbit_rep;
	if (f_v) {
		cout << "blt_set::create_graph before "
				"R->init_from_file" << endl;
		}
	R->init_from_file(A, prefix_with_directory, 
		starter_size, orbit_at_level, level_of_candidates_file, 
		early_test_func_callback, 
		this /* early_test_func_callback_data */, 
		verbose_level - 1
		);
	if (f_v) {
		cout << "blt_set::create_graph "
				"after R->init_from_file" << endl;
		}
	nb = q + 1 - starter_size;


	if (f_vv) {
		cout << "blt_set::create_graph Case "
				<< orbit_at_level << " / " << R->nb_cases
				<< " Read starter : ";
		INT_vec_print(cout, R->rep, starter_size);
		cout << endl;
		}

	max_starter = R->rep[starter_size - 1];

	if (f_vv) {
		cout << "blt_set::create_graph Case " << orbit_at_level
				<< " / " << R->nb_cases << " max_starter="
				<< max_starter << endl;
		cout << "blt_set::create_graph Case " << orbit_at_level
				<< " / " << R->nb_cases << " Group order="
				<< R->stab_go << endl;
		cout << "blt_set::create_graph Case " << orbit_at_level
				<< " / " << R->nb_cases << " nb_candidates="
				<< R->nb_candidates << " at level "
				<< starter_size << endl;
		}



	if (f_lexorder_test) {
		INT nb_candidates2;
	
		if (f_v3) {
			cout << "blt_set::create_graph Case " << orbit_at_level
					<< " / " << R->nb_cases << " Before lexorder_test" << endl;
			}
		A->lexorder_test(R->candidates,
			R->nb_candidates, nb_candidates2,
			R->Strong_gens->gens, max_starter, verbose_level - 3);
		if (f_vv) {
			cout << "blt_set::create_graph "
					"After lexorder_test nb_candidates="
					<< nb_candidates2 << " eliminated "
					<< R->nb_candidates - nb_candidates2
					<< " candidates" << endl;
			}
		R->nb_candidates = nb_candidates2;
		}


	// we must do this. 
	// For instance, what of we have no points left, 
	// then the minimal color stuff break down.
	//if (f_eliminate_graphs_if_possible) {
		if (R->nb_candidates < nb) {
			if (f_v) {
				cout << "blt_set::create_graph "
						"Case " << orbit_at_level << " / "
						<< R->nb_cases << " nb_candidates < nb, "
								"the case is eliminated" << endl;
				}
			delete R;
			return FALSE;
			}
		//}


	nb_vertices = R->nb_candidates;


	INT *point_color;
	INT nb_colors;

	INT *lines_on_pt;
	
	lines_on_pt = NEW_INT(1 /*starter_size*/ * (q + 1));
	O->lines_on_point_by_line_rank(
			R->rep[0],
			lines_on_pt, 0 /* verbose_level */);

	if (f_v3) {
		cout << "Case " << orbit_at_level
				<< " Lines on partial BLT set:" << endl;
		INT_matrix_print(lines_on_pt, 1 /*starter_size*/, q + 1);
		}

	INT special_line;

	special_line = lines_on_pt[0];

	compute_colors(orbit_at_level, 
		R->rep, starter_size, 
		special_line, 
		R->candidates, R->nb_candidates, 
		point_color, nb_colors, 
		verbose_level);


	classify C;

	C.init(point_color, R->nb_candidates, FALSE, 0);
	if (f_v3) {
		cout << "blt_set::create_graph Case " << orbit_at_level
				<< " / " << R->nb_cases
				<< " point colors (1st classification): ";
		C.print(FALSE /* f_reverse */);
		cout << endl;
		}


	classify C2;

	C2.init(point_color, R->nb_candidates, TRUE, 0);
	if (f_vv) {
		cout << "blt_set::create_graph Case " << orbit_at_level
				<< " / " << R->nb_cases
				<< " point colors (2nd classification): ";
		C2.print(FALSE /* f_reverse */);
		cout << endl;
		}



	INT f, /*l,*/ idx;

	f = C2.second_type_first[0];
	//l = C2.second_type_len[0];
	idx = C2.second_sorting_perm_inv[f + 0];
#if 0
	if (C.type_len[idx] != minimal_type_multiplicity) {
		cout << "idx != minimal_type" << endl;
		cout << "idx=" << idx << endl;
		cout << "minimal_type=" << minimal_type << endl;
		cout << "C.type_len[idx]=" << C.type_len[idx] << endl;
		cout << "minimal_type_multiplicity="
				<< minimal_type_multiplicity << endl;
		exit(1);
		}
#endif
	INT minimal_type, minimal_type_multiplicity;
	
	minimal_type = idx;
	minimal_type_multiplicity = C2.type_len[idx];

	if (f_vv) {
		cout << "blt_set::create_graph Case " << orbit_at_level
				<< " / " << R->nb_cases << " minimal type is "
				<< minimal_type << endl;
		cout << "blt_set::create_graph Case " << orbit_at_level
				<< " / " << R->nb_cases << " minimal_type_multiplicity "
				<< minimal_type_multiplicity << endl;
		}

	if (f_eliminate_graphs_if_possible) {
		if (minimal_type_multiplicity == 0) {
			cout << "blt_set::create_graph Case " << orbit_at_level
					<< " / " << R->nb_cases << " Color class "
					<< minimal_type << " is empty, the case is "
							"eliminated" << endl;
			ret = FALSE;
			goto finish;
			}
		}



	if (f_vv) {
		cout << "blt_set::create_graph Case " << orbit_at_level
				<< " / " << R->nb_cases << " Computing adjacency list, "
						"nb_points=" << R->nb_candidates << endl;
		}

	UBYTE *bitvector_adjacency;
	INT bitvector_length_in_bits;
	INT bitvector_length;

	compute_adjacency_list_fast(R->rep[0], 
		R->candidates, R->nb_candidates, point_color, 
		bitvector_adjacency, bitvector_length_in_bits, bitvector_length, 
		verbose_level - 2);

	if (f_vv) {
		cout << "blt_set::create_graph Case " << orbit_at_level
				<< " / " << R->nb_cases << " Computing adjacency "
						"list done" << endl;
		cout << "blt_set::create_graph Case " << orbit_at_level
				<< " / " << R->nb_cases << " bitvector_length="
				<< bitvector_length << endl;
		}


	if (f_v) {
		cout << "blt_set::create_graph creating colored_graph" << endl;
		}

	CG = new colored_graph;

	CG->init(R->nb_candidates /* nb_points */, nb_colors, 
		point_color, bitvector_adjacency, TRUE, verbose_level - 2);
		// the adjacency becomes part of the colored_graph object
	
	INT i;
	for (i = 0; i < R->nb_candidates; i++) {
		CG->points[i] = R->candidates[i];
		}
	CG->init_user_data(R->rep, starter_size, verbose_level - 2);
	sprintf(CG->fname_base, "graph_BLT_%ld_%ld_%ld",
			q, starter_size, orbit_at_level);
		

	if (f_v) {
		cout << "blt_set::create_graph colored_graph created" << endl;
		}

	FREE_INT(lines_on_pt);
	FREE_INT(point_color);


	ret = TRUE;

finish:
	delete R;
	return ret;
}



void blt_set::compute_adjacency_list_fast(
	INT first_point_of_starter,
	INT *points, INT nb_points, INT *point_color, 
	UBYTE *&bitvector_adjacency,
	INT &bitvector_length_in_bits,
	INT &bitvector_length,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT L;
	INT i, j, k, c1, c2;
	INT *Pts;
	INT *form_value;
	INT v1[5];
	INT m[5];
	INT f12, f13, f23, d;
	UINT cnt;
	INT two;
	INT *Pi, *Pj;

	if (f_v) {
		cout << "blt_set::compute_adjacency_list_fast" << endl;
		}
	L = (nb_points * (nb_points - 1)) >> 1;

	bitvector_length_in_bits = L;
	bitvector_length = (L + 7) >> 3;
	bitvector_adjacency = NEW_UBYTE(bitvector_length);
	for (i = 0; i < bitvector_length; i++) {
		bitvector_adjacency[i] = 0;
		}
	
	Pts = NEW_INT(nb_points * 5);
	form_value = NEW_INT(nb_points);
	O->unrank_point(v1, 1, first_point_of_starter, 0);
	if (f_v) {
		cout << "blt_set::compute_adjacency_list_fast "
				"unranking points" << endl;
		}
	for (i = 0; i < nb_points; i++) {
		O->unrank_point(Pts + i * 5, 1, points[i], 0);
		form_value[i] = O->evaluate_bilinear_form(
				v1, Pts + i * 5, 1);
		}

	if (f_v) {
		cout << "blt_set::compute_adjacency_list_fast "
				"computing adjacencies" << endl;
		}

	cnt = 0;
	two = F->add(1, 1);
	
	for (i = 0; i < nb_points; i++) {
		f12 = form_value[i];
		c1 = point_color[i];
		Pi = Pts + i * 5;
		m[0] = F->mult(Pi[0], two);
		m[1] = Pi[2];
		m[2] = Pi[1];
		m[3] = Pi[4];
		m[4] = Pi[3];
		
		for (j = i + 1; j < nb_points; j++, cnt++) {
			k = ij2k(i, j, nb_points);
		
			if ((cnt & ((1 << 25) - 1)) == 0 && cnt) {
				cout << "blt_set::compute_adjacency_list_fast "
						"nb_points=" << nb_points << " adjacency "
						<< cnt << " / " << L << " i=" << i
						<< " j=" << j << endl;
				}
			c2 = point_color[j];
			if (c1 == c2) {
				bitvector_m_ii(bitvector_adjacency, k, 0);
				continue;
				}
			f13 = form_value[j];
			Pj = Pts + j * 5;
			f23 = F->add5(
				F->mult(m[0], Pj[0]), 
				F->mult(m[1], Pj[1]), 
				F->mult(m[2], Pj[2]), 
				F->mult(m[3], Pj[3]), 
				F->mult(m[4], Pj[4])
				);
			d = F->product3(f12, f13, f23);
			if (d == 0) {
				bitvector_m_ii(bitvector_adjacency, k, 0);
				}
			else {
				if (O->f_is_minus_square[d]) {
					bitvector_m_ii(bitvector_adjacency, k, 0);
					}
				else {
					bitvector_m_ii(bitvector_adjacency, k, 1);
					}
				}
			
			} // next j
		} // next i



	FREE_INT(Pts);
	FREE_INT(form_value);
	if (f_v) {
		cout << "blt_set::compute_adjacency_list_fast done" << endl;
		}
}



void blt_set::compute_colors(INT orbit_at_level, 
	INT *starter, INT starter_sz, 
	INT special_line, 
	INT *candidates, INT nb_candidates, 
	INT *&point_color, INT &nb_colors, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT p1, p2;
	INT v1[5];
	INT v2[5];
	INT v3[5];
	INT *pts_on_special_line;
	INT idx, i;


	if (f_v) {
		cout << "blt_set::compute_colors" << endl;
		}
	O->unrank_line(p1, p2, special_line, 0/*verbose_level*/);
	if (f_vv) {
		cout << "after unrank_line " << special_line << ":" << endl;
		cout << "p1=" << p1 << " p2=" << p2 << endl;
		}
	O->unrank_point(v1, 1, p1, 0);
	O->unrank_point(v2, 1, p2, 0);
	if (f_vv) {
		cout << "p1=" << p1 << " ";
		INT_vec_print(cout, v1, 5);
		cout << endl;
		cout << "p2=" << p2 << " ";
		INT_vec_print(cout, v2, 5);
		cout << endl;
		}
	if (p1 != starter[0]) {
		cout << "p1 != starter[0]" << endl;
		exit(1);
		}
	
	pts_on_special_line = NEW_INT(q + 1);
	O->points_on_line(p1, p2, pts_on_special_line,
			0/*verbose_level*/);
	
	if (f_vv) {
		cout << "pts_on_special_line:" << endl;
		INT_vec_print(cout, pts_on_special_line, q + 1);
		cout << endl;
		}

	if (!INT_vec_search(pts_on_special_line, q + 1, starter[0], idx)) {
		cout << "cannot find the first point on the line" << endl;
		exit(1);
		}
	for (i = idx; i < q + 1; i++) {
		pts_on_special_line[i] = pts_on_special_line[i + 1];
		}
	if (f_vv) {
		cout << "pts_on_special_line without the first "
				"starter point:" << endl;
		INT_vec_print(cout, pts_on_special_line, q);
		cout << endl;
		}
	
	INT a, b, t, c, j, h;
	INT *starter_t;
	
	starter_t = NEW_INT(starter_sz);
	starter_t[0] = -1;
	for (i = 1; i < starter_sz; i++) {
		O->unrank_point(v3, 1, starter[i], 0);
		a = O->evaluate_bilinear_form(v1, v3, 1);
		b = O->evaluate_bilinear_form(v2, v3, 1);
		if (a == 0) {
			cout << "a == 0, this should not be" << endl;
			exit(1);
			}
		// <v3,t*v1+v2> = t*<v3,v1>+<v3,v2> = t*a+b = 0
		// Thus, t = -b/a
		t = O->F->mult(O->F->negate(b), O->F->inverse(a));
		starter_t[i] = t;
		}

	if (f_vv) {
		cout << "starter_t:" << endl;
		INT_vec_print(cout, starter_t, starter_sz);
		cout << endl;
		}

	INT *free_pts;
	INT *open_colors;
	INT *open_colors_inv;

	free_pts = NEW_INT(q);
	open_colors = NEW_INT(q);
	open_colors_inv = NEW_INT(q);

	point_color = NEW_INT(nb_candidates);

	nb_colors = q - starter_sz + 1;
	j = 0;
	for (i = 0; i < q; i++) {
		for (h = 1; h < starter_sz; h++) {
			if (starter_t[h] == i)
				break;
			}
		if (h == starter_sz) {
			free_pts[j] = pts_on_special_line[i];
			open_colors[j] = i;
			j++;
			}
		}
	if (j != nb_colors) {
		cout << "extension_data::setup error: j != nb_colors" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "The " << nb_colors << " free points are :" << endl;
		INT_vec_print(cout, free_pts, nb_colors);
		cout << endl;
		cout << "The " << nb_colors << " open colors are :" << endl;
		INT_vec_print(cout, open_colors, nb_colors);
		cout << endl;
		}
	for ( ; j < q; j++) {
		open_colors[j] = starter_t[j - nb_colors + 1];
		}
	if (f_vv) {
		cout << "open_colors :" << endl;
		INT_vec_print(cout, open_colors, q);
		cout << endl;
		}
	for (i = 0; i < q; i++) {
		j = open_colors[i];
		open_colors_inv[j] = i;
		}
	if (f_vv) {
		cout << "open_colors_inv :" << endl;
		INT_vec_print(cout, open_colors_inv, q);
		cout << endl;
		}


	for (i = 0; i < nb_candidates; i++) {
		O->unrank_point(v3, 1, candidates[i], 0);
		if (f_vv) {
			cout << "candidate " << i << " / " << nb_candidates
					<< " is " << candidates[i] << " = ";
			INT_vec_print(cout, v3, 5);
			cout << endl;
			}
		a = O->evaluate_bilinear_form(v1, v3, 1);
		b = O->evaluate_bilinear_form(v2, v3, 1);
		if (a == 0) {
			cout << "a == 0, this should not be" << endl;
			exit(1);
			}
		// <v3,t*v1+v2> = t*<v3,v1>+<v3,v2> = t*a+b = 0
		// Thus, t = -b/a
		t = O->F->mult(O->F->negate(b), O->F->inverse(a));
		c = open_colors_inv[t];
		if (c >= nb_colors) {
			cout << "c >= nb_colors" << endl;
			cout << "i=" << i << endl;
			cout << "candidates[i]=" << candidates[i] << endl;
			cout << "as vector: ";
			INT_vec_print(cout, v3, 5);
			cout << endl;
			cout << "a=" << a << endl;
			cout << "b=" << b << endl;
			cout << "t=" << t << endl;
			cout << "c=" << c << endl;
			cout << "nb_colors=" << nb_colors << endl;
			
			exit(1);
			}
		point_color[i] = c;
		}

	if (f_vv) {
		cout << "point colors:" << endl;
		INT_vec_print(cout, point_color, nb_candidates);
		cout << endl;
		}

	FREE_INT(pts_on_special_line);
	FREE_INT(starter_t);
	FREE_INT(free_pts);
	FREE_INT(open_colors);
	FREE_INT(open_colors_inv);
	if (f_v) {
		cout << "blt_set::compute_colors done" << endl;
		}
}



void blt_set::early_test_func(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, a;
	INT f_OK;
	INT v[5];
	INT *v1, *v2, *v3;
	INT m1[5];
	INT m3[5];
	INT two;
	INT fxy, fxz, fyz;
		
	if (f_v) {
		cout << "blt_set::early_test_func checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		INT_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		if (f_vv) {
			for (i = 0; i < nb_candidates; i++) {
				O->unrank_point(v, 1, candidates[i],
						0/*verbose_level - 4*/);
				cout << "candidate " << i << "="
						<< candidates[i] << ": ";
				INT_vec_print(cout, v, 5);
				cout << endl;
				}
			}
		}
	for (i = 0; i < len; i++) {
		O->unrank_point(Pts + i * 5, 1,
				S[i], 0/*verbose_level - 4*/);
		}
	for (i = 0; i < nb_candidates; i++) {
		O->unrank_point(Candidates + i * 5, 1, candidates[i],
				0/*verbose_level - 4*/);
		}
	
	two = O->F->add(1, 1);


	if (len == 0) {
		INT_vec_copy(candidates, good_candidates, nb_candidates);
		nb_good_candidates = nb_candidates;
		}
	else {
		nb_good_candidates = 0;
	
		if (f_vv) {
			cout << "blt_set::early_test_func before testing" << endl;
			}
		for (j = 0; j < nb_candidates; j++) {


			if (f_vv) {
				cout << "blt_set::early_test_func "
						"testing " << j << " / "
						<< nb_candidates << endl;
				}

			v1 = Pts;
			v3 = Candidates + j * 5;

			m1[0] = O->F->mult(two, v1[0]);
			m1[1] = v1[2];
			m1[2] = v1[1];
			m1[3] = v1[4];
			m1[4] = v1[3];

			//fxz = evaluate_bilinear_form(v1, v3, 1);
			// too slow !!!
			fxz = O->F->add5(
					O->F->mult(m1[0], v3[0]), 
					O->F->mult(m1[1], v3[1]), 
					O->F->mult(m1[2], v3[2]), 
					O->F->mult(m1[3], v3[3]), 
					O->F->mult(m1[4], v3[4]) 
				);


			if (fxz == 0) {
				f_OK = FALSE;
				}
			else {
				m3[0] = O->F->mult(two, v3[0]);
				m3[1] = v3[2];
				m3[2] = v3[1];
				m3[3] = v3[4];
				m3[4] = v3[3];

				f_OK = TRUE;
				for (i = 1; i < len; i++) {
					//fxy = evaluate_bilinear_form(v1, v2, 1);

					v2 = Pts + i * 5;
		
					fxy = O->F->add5(
						O->F->mult(m1[0], v2[0]), 
						O->F->mult(m1[1], v2[1]), 
						O->F->mult(m1[2], v2[2]), 
						O->F->mult(m1[3], v2[3]), 
						O->F->mult(m1[4], v2[4]) 
						);
		
					//fyz = evaluate_bilinear_form(v2, v3, 1);
					fyz = O->F->add5(
							O->F->mult(m3[0], v2[0]), 
							O->F->mult(m3[1], v2[1]), 
							O->F->mult(m3[2], v2[2]), 
							O->F->mult(m3[3], v2[3]), 
							O->F->mult(m3[4], v2[4]) 
						);

					a = O->F->product3(fxy, fxz, fyz);

					if (a == 0) {
						f_OK = FALSE;
						break;
						}
					if (O->f_is_minus_square[a]) {
						f_OK = FALSE;
						break;
						}

					}
				}
			if (f_OK) {
				good_candidates[nb_good_candidates++] =
						candidates[j];
				}
			} // next j
		} // else
}

INT blt_set::check_function_incremental(
		INT len, INT *S, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, a;
	INT f_OK;
	INT *v1, *v2, *v3;
	INT m1[5];
	INT m3[5];
	INT two;
	INT fxy, fxz, fyz;
		
	if (f_v) {
		cout << "blt_set::check_function_incremental "
				"checking set ";
		print_set(cout, len, S);
		cout << endl;
		}

	for (i = 0; i < len; i++) {
		O->unrank_point(Pts + i * 5, 1, S[i], 0/*verbose_level - 4*/);
		}

	two = O->F->add(1, 1);

	v1 = Pts;
	v3 = Pts + (len - 1) * 5;

	m1[0] = O->F->mult(two, v1[0]);
	m1[1] = v1[2];
	m1[2] = v1[1];
	m1[3] = v1[4];
	m1[4] = v1[3];

	//fxz = evaluate_bilinear_form(v1, v3, 1);
	// too slow !!!
	fxz = O->F->add5(
			O->F->mult(m1[0], v3[0]), 
			O->F->mult(m1[1], v3[1]), 
			O->F->mult(m1[2], v3[2]), 
			O->F->mult(m1[3], v3[3]), 
			O->F->mult(m1[4], v3[4]) 
		);

	m3[0] = O->F->mult(two, v3[0]);
	m3[1] = v3[2];
	m3[2] = v3[1];
	m3[3] = v3[4];
	m3[4] = v3[3];

	f_OK = TRUE;
	for (i = 1; i < len - 1; i++) {
		//fxy = evaluate_bilinear_form(v1, v2, 1);

		v2 = Pts + i * 5;
		
		fxy = O->F->add5(
			O->F->mult(m1[0], v2[0]), 
			O->F->mult(m1[1], v2[1]), 
			O->F->mult(m1[2], v2[2]), 
			O->F->mult(m1[3], v2[3]), 
			O->F->mult(m1[4], v2[4]) 
			);
		
		//fyz = evaluate_bilinear_form(v2, v3, 1);
		fyz = O->F->add5(
				O->F->mult(m3[0], v2[0]), 
				O->F->mult(m3[1], v2[1]), 
				O->F->mult(m3[2], v2[2]), 
				O->F->mult(m3[3], v2[3]), 
				O->F->mult(m3[4], v2[4]) 
			);

		a = O->F->product3(fxy, fxz, fyz);

		if (a == 0) {
			f_OK = FALSE;
			break;
			}
		
		if (O->f_is_minus_square[a]) {
			f_OK = FALSE;
			break;
			}

		}
	return f_OK;
}

INT blt_set::pair_test(INT a, INT x, INT y, INT verbose_level)
// We assume that a is an element
// of a set S of size at least two such that
// S \cup \{ x \} is BLT and 
// S \cup \{ y \} is BLT.
// In order to test if S \cup \{ x, y \}
// is BLT, we only need to test
// the triple \{ x,y,a\}
{
	INT v1[5], v2[5], v3[5];
	INT f12, f13, f23;
	INT d;

	O->unrank_point(v1, 1, a, 0);
	O->unrank_point(v2, 1, x, 0);
	O->unrank_point(v3, 1, y, 0);
	f12 = O->evaluate_bilinear_form(v1, v2, 1);
	f13 = O->evaluate_bilinear_form(v1, v3, 1);
	f23 = O->evaluate_bilinear_form(v2, v3, 1);
	d = O->F->product3(f12, f13, f23);
	if (d == 0) {
		return FALSE;
		}
	if (O->f_is_minus_square[d]) {
		return FALSE;
		}
	else {
		return TRUE;
		}
	
}

INT blt_set::check_conditions(INT len, INT *S, INT verbose_level)
{
	INT f_OK = TRUE;
	INT f_BLT_test = FALSE;
	INT f_collinearity_test = FALSE;
	//INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	//f_v = TRUE;
	//f_vv = TRUE;
	
	if (f_vv) {
		cout << "checking set ";
		print_set(cout, len, S);
		}
	if (!collinearity_test(S, len, verbose_level)) {
		f_OK = FALSE;
		f_collinearity_test = TRUE;
		}
	if (f_BLT) {
		if (!O->BLT_test(len, S, verbose_level)) {
			f_OK = FALSE;
			f_BLT_test = TRUE;
			}
		}


	if (f_OK) {
		if (f_vv) {
			cout << "OK" << endl;
			}
		return TRUE;
		}
	else {
		if (f_vv) {
			cout << "not OK because of ";
			if (f_BLT_test) {
				cout << "BLT test";
				}
			if (f_collinearity_test) {
				cout << "collinearity test";
				}
			cout << endl;
			}
		return FALSE;
		}
}

INT blt_set::collinearity_test(INT *S, INT len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, x, y;
	INT f_OK = TRUE;
	INT fxy;
	
	if (f_v) {
		cout << "collinearity test for" << endl;
		for (i = 0; i < len; i++) {
			O->unrank_point(O->v1, 1, S[i], 0);
			INT_vec_print(cout, O->v1, n);
			cout << endl;
			}
		}
	y = S[len - 1];
	O->unrank_point(O->v1, 1, y, 0);
	
	for (i = 0; i < len - 1; i++) {
		x = S[i];
		O->unrank_point(O->v2, 1, x, 0);
		fxy = O->evaluate_bilinear_form(O->v1, O->v2, 1);
		
		if (fxy == 0) {
			f_OK = FALSE;
			if (f_v) {
				cout << "not OK; ";
				cout << "{x,y}={" << x << ","
						<< y << "} are collinear" << endl;
				INT_vec_print(cout, O->v1, n);
				cout << endl;
				INT_vec_print(cout, O->v2, n);
				cout << endl;
				cout << "fxy=" << fxy << endl;
				}
			break;
			}
		}
	
	if (f_v) {
		if (!f_OK) {
			cout << "collinearity test fails" << endl;
			}
		}
	return f_OK;
}

void blt_set::print(INT *S, INT len)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		O->unrank_point(O->v1, 1, S[i], 0);
		INT_vec_print(cout, O->v1, n);
		cout << endl;
		}
}



