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
	int i;
		
	
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
	Poset = NULL;
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
	int f_v = FALSE;

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
	if (Poset) {
		FREE_OBJECT(Poset);
		Poset = NULL;
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
		FREE_int(Pts);
		}
	if (Candidates) {
		FREE_int(Candidates);
		}
	null();
	if (f_v) {
		cout << "blt_set::freeself done" << endl;
		}
	
}



void blt_set::init_basic(finite_field *F, 
	const char *input_prefix, 
	const char *base_fname,
	int starter_size,  
	int argc, const char **argv, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

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


	gen = NEW_OBJECT(poset_classification);
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

void blt_set::init_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_basis = TRUE;

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
	A = NEW_OBJECT(action);

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
	A->lex_least_base_in_place(verbose_level - 2);
	if (f_vv) {
		cout << "blt_set::init_group "
				"computing lex least base done" << endl;
		cout << "blt_set::init_group base: ";
		int_vec_print(cout, A->base, A->base_len);
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
		O->F->print();
		}


	
	if (f_v) {
		cout << "blt_set::init_group "
				"allocating Pts and Candidates" << endl;
		}
	//int Pts_size = target_size * n;
	Pts = NEW_int(target_size * n);

#if 0
	for (int i=0; i<Pts_size; i++) Pts[i] = 0;
			// set all the points to an
			// initial value of zero
			// in order to prevent 
			// finite_field::mult(int i, int j)
			// from throwing an error.
#endif

	Candidates = NEW_int(degree * n);
	
	if (f_v) {
		cout << "blt_set::init_group finished" << endl;
		}
}


void blt_set::init_orthogonal_hash(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set::init_orthogonal_hash" << endl;
		}

	init_hash_table_parabolic(*O->F, 4, 0/*verbose_level*/);

	if (f_v) {
		cout << "blt_set::init_orthogonal finished" << endl;
		}
}

void blt_set::init2(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

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

	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(A, A,
			A->Strong_gens,
			verbose_level);
	
	gen->init(Poset,
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
	
	
	int nb_nodes = ONE_MILLION;
	
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
	int orbit_at_level_r, int orbit_at_level_m, 
	int level_of_candidates_file, 
	const char *output_prefix, 
	int f_lexorder_test, int f_eliminate_graphs_if_possible, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);


	if (f_v) {
		cout << "blt_set::create_graphs" << endl;
		cout << "blt_set::create_graphs "
				"starter_size = " << starter_size << endl;
		cout << "blt_set::create_graphs "
				"f_lexorder_test=" << f_lexorder_test << endl;
		}


	//f_memory_debug = TRUE;


	char fname[1000];
	char fname_list_of_cases[1000];
	char fname_time[1000];
	char graph_fname_base[1000];
	int orbit;
	int nb_orbits;
	int *list_of_cases;
	int nb_of_cases;

	int *Time;
	int time_idx;




	sprintf(fname, "%s_lvl_%d", prefix_with_directory, starter_size);
	sprintf(fname_list_of_cases, "%slist_of_cases_%s_%d_%d_%d.txt",
			output_prefix, prefix, starter_size,
			orbit_at_level_r, orbit_at_level_m);
	sprintf(fname_time, "%stime_%s_%d_%d_%d.csv",
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


	Time = NEW_int(nb_orbits * 2);
	int_vec_zero(Time, nb_orbits * 2);
	time_idx = 0;

	nb_of_cases = 0;
	list_of_cases = NEW_int(nb_orbits);
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
		int nb_vertices = -1;

		int t0 = os_ticks();
		
		if (create_graph(orbit, level_of_candidates_file, 
			output_prefix, 
			f_lexorder_test, f_eliminate_graphs_if_possible, 
			nb_vertices, graph_fname_base,
			CG,  
			verbose_level - 2)) {
			list_of_cases[nb_of_cases++] = orbit;

			char fname[1000];

			sprintf(fname, "%s%s.bin", output_prefix, CG->fname_base);
			CG->save(fname, verbose_level - 2);
			
			nb_vertices = CG->nb_points;
			}

		if (CG) {
			delete CG;
			}

		int t1 = os_ticks();

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
	int_matrix_write_csv(fname_time, Time, time_idx, 2);
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

	FREE_int(Time);
	FREE_int(list_of_cases);

	//registry_dump_sorted();
}

void blt_set::create_graphs_list_of_cases(
	const char *case_label, 
	const char *list_of_cases_text, 
	int level_of_candidates_file, 
	const char *output_prefix, 
	int f_lexorder_test, int f_eliminate_graphs_if_possible, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);


	if (f_v) {
		cout << "blt_set::create_graphs_list_of_cases" << endl;
		cout << "blt_set::create_graphs_list_of_cases "
				"case_label = " << case_label << endl;
		}

	
	//f_memory_debug = TRUE;

	int *list_of_cases = NULL;
	int nb_of_cases;


	int_vec_scan(list_of_cases_text, list_of_cases, nb_of_cases);
	if (f_v) {
		cout << "blt_set::create_graphs_list_of_cases "
				"nb_of_cases = " << nb_of_cases << endl;
		cout << "blt_set::create_graphs_list_of_cases "
				"starter_size = " << starter_size << endl;
		cout << "blt_set::create_graphs_list_of_cases "
				"f_lexorder_test=" << f_lexorder_test << endl;
		}

	char fname[1000];
	char fname_list_of_cases[1000];
	char graph_fname_base[1000];
	int orbit;
	int nb_orbits;
	int *list_of_cases_created;
	int nb_of_cases_created;
	int c;




	sprintf(fname, "%s_lvl_%d", prefix_with_directory, starter_size);
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
	list_of_cases_created = NEW_int(nb_orbits);
	for (c = 0; c < nb_of_cases; c++) {
		orbit = list_of_cases[c];
		if (f_v3) {
			cout << "blt_set::create_graphs_list_of_cases case "
					<< c << " / " << nb_of_cases << " creating graph "
							"associated with orbit " << orbit << " / "
							<< nb_orbits << ":" << endl;
			}

		
		colored_graph *CG = NULL;
		int nb_vertices = -1;


		if (create_graph(orbit, level_of_candidates_file, 
			output_prefix, 
			f_lexorder_test, f_eliminate_graphs_if_possible, 
			nb_vertices, graph_fname_base,
			CG,  
			verbose_level - 2)) {
			list_of_cases_created[nb_of_cases_created++] = orbit;

			char fname[1000];

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

	FREE_int(list_of_cases_created);

	//registry_dump_sorted();
}


int blt_set::create_graph(
	int orbit_at_level, int level_of_candidates_file, 
	const char *output_prefix, 
	int f_lexorder_test, int f_eliminate_graphs_if_possible, 
	int &nb_vertices, char *graph_fname_base,
	colored_graph *&CG,  
	int verbose_level)
// returns TRUE if a graph was written, FALSE otherwise
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);


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
	
	int ret;

	orbit_rep *R;



	int max_starter;
	int nb;

	nb_vertices = 0;


	R = NEW_OBJECT(orbit_rep);
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
		int_vec_print(cout, R->rep, starter_size);
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
		int nb_candidates2;
	
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
			FREE_OBJECT(R);
			return FALSE;
			}
		//}


	nb_vertices = R->nb_candidates;


	int *point_color;
	int nb_colors;

	int *lines_on_pt;
	
	lines_on_pt = NEW_int(1 /*starter_size*/ * (q + 1));
	O->lines_on_point_by_line_rank(
			R->rep[0],
			lines_on_pt, 0 /* verbose_level */);

	if (f_v3) {
		cout << "Case " << orbit_at_level
				<< " Lines on partial BLT set:" << endl;
		int_matrix_print(lines_on_pt, 1 /*starter_size*/, q + 1);
		}

	int special_line;

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



	int f, /*l,*/ idx;

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
	int minimal_type, minimal_type_multiplicity;
	
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

	uchar *bitvector_adjacency;
	int bitvector_length_in_bits;
	int bitvector_length;

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

	CG = NEW_OBJECT(colored_graph);

	CG->init(R->nb_candidates /* nb_points */, nb_colors, 
		point_color, bitvector_adjacency, TRUE, verbose_level - 2);
		// the adjacency becomes part of the colored_graph object
	
	int i;
	for (i = 0; i < R->nb_candidates; i++) {
		CG->points[i] = R->candidates[i];
		}
	CG->init_user_data(R->rep, starter_size, verbose_level - 2);
	sprintf(CG->fname_base, "graph_BLT_%d_%d_%d",
			q, starter_size, orbit_at_level);
		

	if (f_v) {
		cout << "blt_set::create_graph colored_graph created" << endl;
		}

	FREE_int(lines_on_pt);
	FREE_int(point_color);


	ret = TRUE;

finish:
	FREE_OBJECT(R);
	return ret;
}



void blt_set::compute_adjacency_list_fast(
	int first_point_of_starter,
	int *points, int nb_points, int *point_color, 
	uchar *&bitvector_adjacency,
	int &bitvector_length_in_bits,
	int &bitvector_length,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int L;
	int i, j, k, c1, c2;
	int *Pts;
	int *form_value;
	int v1[5];
	int m[5];
	int f12, f13, f23, d;
	uint cnt;
	int two;
	int *Pi, *Pj;

	if (f_v) {
		cout << "blt_set::compute_adjacency_list_fast" << endl;
		}
	L = (nb_points * (nb_points - 1)) >> 1;

	bitvector_length_in_bits = L;
	bitvector_length = (L + 7) >> 3;
	bitvector_adjacency = NEW_uchar(bitvector_length);
	for (i = 0; i < bitvector_length; i++) {
		bitvector_adjacency[i] = 0;
		}
	
	Pts = NEW_int(nb_points * 5);
	form_value = NEW_int(nb_points);
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



	FREE_int(Pts);
	FREE_int(form_value);
	if (f_v) {
		cout << "blt_set::compute_adjacency_list_fast done" << endl;
		}
}



void blt_set::compute_colors(int orbit_at_level, 
	int *starter, int starter_sz, 
	int special_line, 
	int *candidates, int nb_candidates, 
	int *&point_color, int &nb_colors, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int p1, p2;
	int v1[5];
	int v2[5];
	int v3[5];
	int *pts_on_special_line;
	int idx, i;


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
		int_vec_print(cout, v1, 5);
		cout << endl;
		cout << "p2=" << p2 << " ";
		int_vec_print(cout, v2, 5);
		cout << endl;
		}
	if (p1 != starter[0]) {
		cout << "p1 != starter[0]" << endl;
		exit(1);
		}
	
	pts_on_special_line = NEW_int(q + 1);
	O->points_on_line(p1, p2, pts_on_special_line,
			0/*verbose_level*/);
	
	if (f_vv) {
		cout << "pts_on_special_line:" << endl;
		int_vec_print(cout, pts_on_special_line, q + 1);
		cout << endl;
		}

	if (!int_vec_search(pts_on_special_line, q + 1, starter[0], idx)) {
		cout << "cannot find the first point on the line" << endl;
		exit(1);
		}
	for (i = idx; i < q + 1; i++) {
		pts_on_special_line[i] = pts_on_special_line[i + 1];
		}
	if (f_vv) {
		cout << "pts_on_special_line without the first "
				"starter point:" << endl;
		int_vec_print(cout, pts_on_special_line, q);
		cout << endl;
		}
	
	int a, b, t, c, j, h;
	int *starter_t;
	
	starter_t = NEW_int(starter_sz);
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
		int_vec_print(cout, starter_t, starter_sz);
		cout << endl;
		}

	int *free_pts;
	int *open_colors;
	int *open_colors_inv;

	free_pts = NEW_int(q);
	open_colors = NEW_int(q);
	open_colors_inv = NEW_int(q);

	point_color = NEW_int(nb_candidates);

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
		int_vec_print(cout, free_pts, nb_colors);
		cout << endl;
		cout << "The " << nb_colors << " open colors are :" << endl;
		int_vec_print(cout, open_colors, nb_colors);
		cout << endl;
		}
	for ( ; j < q; j++) {
		open_colors[j] = starter_t[j - nb_colors + 1];
		}
	if (f_vv) {
		cout << "open_colors :" << endl;
		int_vec_print(cout, open_colors, q);
		cout << endl;
		}
	for (i = 0; i < q; i++) {
		j = open_colors[i];
		open_colors_inv[j] = i;
		}
	if (f_vv) {
		cout << "open_colors_inv :" << endl;
		int_vec_print(cout, open_colors_inv, q);
		cout << endl;
		}


	for (i = 0; i < nb_candidates; i++) {
		O->unrank_point(v3, 1, candidates[i], 0);
		if (f_vv) {
			cout << "candidate " << i << " / " << nb_candidates
					<< " is " << candidates[i] << " = ";
			int_vec_print(cout, v3, 5);
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
			int_vec_print(cout, v3, 5);
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
		int_vec_print(cout, point_color, nb_candidates);
		cout << endl;
		}

	FREE_int(pts_on_special_line);
	FREE_int(starter_t);
	FREE_int(free_pts);
	FREE_int(open_colors);
	FREE_int(open_colors_inv);
	if (f_v) {
		cout << "blt_set::compute_colors done" << endl;
		}
}



void blt_set::early_test_func(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, a;
	int f_OK;
	int v[5];
	int *v1, *v2, *v3;
	int m1[5];
	int m3[5];
	int two;
	int fxy, fxz, fyz;
		
	if (f_v) {
		cout << "blt_set::early_test_func checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		int_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		if (f_vv) {
			for (i = 0; i < nb_candidates; i++) {
				O->unrank_point(v, 1, candidates[i],
						0/*verbose_level - 4*/);
				cout << "candidate " << i << "="
						<< candidates[i] << ": ";
				int_vec_print(cout, v, 5);
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
		int_vec_copy(candidates, good_candidates, nb_candidates);
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

int blt_set::check_function_incremental(
		int len, int *S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a;
	int f_OK;
	int *v1, *v2, *v3;
	int m1[5];
	int m3[5];
	int two;
	int fxy, fxz, fyz;
		
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

int blt_set::pair_test(int a, int x, int y, int verbose_level)
// We assume that a is an element
// of a set S of size at least two such that
// S \cup \{ x \} is BLT and 
// S \cup \{ y \} is BLT.
// In order to test if S \cup \{ x, y \}
// is BLT, we only need to test
// the triple \{ x,y,a\}
{
	int v1[5], v2[5], v3[5];
	int f12, f13, f23;
	int d;

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

int blt_set::check_conditions(int len, int *S, int verbose_level)
{
	int f_OK = TRUE;
	int f_BLT_test = FALSE;
	int f_collinearity_test = FALSE;
	//int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
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

int blt_set::collinearity_test(int *S, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, x, y;
	int f_OK = TRUE;
	int fxy;
	
	if (f_v) {
		cout << "collinearity test for" << endl;
		for (i = 0; i < len; i++) {
			O->unrank_point(O->v1, 1, S[i], 0);
			int_vec_print(cout, O->v1, n);
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
				int_vec_print(cout, O->v1, n);
				cout << endl;
				int_vec_print(cout, O->v2, n);
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

void blt_set::print(int *S, int len)
{
	int i;
	
	for (i = 0; i < len; i++) {
		O->unrank_point(O->v1, 1, S[i], 0);
		int_vec_print(cout, O->v1, n);
		cout << endl;
		}
}



