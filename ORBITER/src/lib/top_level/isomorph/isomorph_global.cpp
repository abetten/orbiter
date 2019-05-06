// isomorph_global.C
// 
// Anton Betten
// started Aug 1, 2012
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {

void isomorph_read_statistic_files(
	action *A_base, action *A,
	poset_classification *gen,
	int size, const char *prefix_classify,
	const char *prefix, int level,
	const char **fname, int nb_files,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_implicit_fusion = FALSE;
	

	if (f_v) {
		cout << "isomorph_read_statistic_files" << endl;
		cout << "nb_files = " << nb_files << endl;
		cout << "prefix_classify = " << prefix_classify << endl;
		cout << "prefix = " << prefix << endl;
		}

	
	discreta_init();

	{
	isomorph Iso;
	int f_use_database_for_starter = TRUE;
	file_io Fio;


	if (f_v) {
		cout << "size = " << size << endl;
		}
	
	if (f_v) {
		cout << "isomorph_read_statistic_files "
				"before Iso.init" << endl;
		}
	Iso.init(prefix, A_base, A, gen, size, level,
		f_use_database_for_starter,
		f_implicit_fusion, verbose_level);
		// sets level and initializes file names

	

	if (f_v) {
		cout << "isomorph_read_statistic_files "
				"before Iso.read_data_files_for_starter" << endl;
		}
	Iso.read_data_files_for_starter(level,
		prefix_classify, verbose_level);
		// in isomorph.C
	



	// Row,Case_nb,Nb_sol,Nb_backtrack,Nb_col,Dt,Dt_in_sec

	spreadsheet *S;
	int i, h, Case_nb, Nb_sol, Nb_backtrack, Nb_col, Dt, Dt_in_sec;
	int case_nb, nb_sol, nb_backtrack, nb_col, dt, dt_in_sec;
	int *Stats;

	S = NEW_OBJECTS(spreadsheet, nb_files);
	for (i = 0; i < nb_files; i++) {
		cout << "reading file " << fname[i] << ":" << endl;
		S[i].read_spreadsheet(fname[i], 0 /* verbose_level */);
		}

	cout << "Allocating array Stats for " << Iso.nb_starter
			<< " starter cases" << endl;
	 
	Stats = NEW_int(6 * Iso.nb_starter);
	int_vec_zero(Stats, 6 * Iso.nb_starter);
	for (i = 0; i < Iso.nb_starter; i++) {
		Stats[i * 6 + 0] = -1;
		}
	
	cout << "Reading all the statistic files" << endl;

	for (h = 0; h < nb_files; h++) {
		Case_nb = S[h].find_by_column("Case_nb");
		Nb_sol = S[h].find_by_column("Nb_sol");
		Nb_backtrack = S[h].find_by_column("Nb_backtrack");
		Nb_col = S[h].find_by_column("Nb_col");
		Dt = S[h].find_by_column("Dt");
		Dt_in_sec = S[h].find_by_column("Dt_in_sec");
		for (i = 1; i < S[h].nb_rows; i++) {
			case_nb = S[h].get_int(i, Case_nb);
			nb_sol = S[h].get_int(i, Nb_sol);
			nb_backtrack = S[h].get_int(i, Nb_backtrack);
			nb_col = S[h].get_int(i, Nb_col);
			dt = S[h].get_int(i, Dt);
			dt_in_sec = S[h].get_int(i, Dt_in_sec);
			Stats[case_nb * 6 + 0] = 1;
			Stats[case_nb * 6 + 1] = nb_sol;
			Stats[case_nb * 6 + 2] = nb_backtrack;
			Stats[case_nb * 6 + 3] = nb_col;
			Stats[case_nb * 6 + 4] = dt;
			Stats[case_nb * 6 + 5] = dt_in_sec;
			}
		}

	cout << "Read all the statistic files" << endl;
	for (i = 0; i < Iso.nb_starter; i++) {
		if (Stats[i * 6 + 0] == -1) {
			cout << "The run is incomplete, I don't have data for case "
					<< i << " for instance" << endl;
			exit(1);
			}
		}

	cout << "The run is complete" << endl;



	cout << "The cases where solutions exist are:" << endl;
	for (i = 0; i < Iso.nb_starter; i++) {
		if (Stats[i * 6 + 1]) {
			cout << setw(5) << i << " : " << setw(5)
					<< Stats[i * 6 + 1] << " : ";

			}
		}

	int Nb_cases = 0;

	Nb_cases = 0;
	for (i = 0; i < Iso.nb_starter; i++) {
		if (Stats[i * 6 + 1]) {
			Nb_cases++;
			}
		}

	int *Stats_short;

	
	Stats_short = NEW_int(6 * Nb_cases);
	h = 0;
	for (i = 0; i < Iso.nb_starter; i++) {
		if (Stats[i * 6 + 1]) {
			int_vec_copy(Stats + 6 * i, Stats_short + 6 * h, 6);
			Stats_short[h * 6 + 0] = i;
			h++;
			}
		}

	// Row,Case_nb,Nb_sol,Nb_backtrack,Nb_col,Dt,Dt_in_sec

	const char *Column_label[] = {
		"Case_nb", 
		"Nb_sol", 
		"Nb_backtrack", 
		"Nb_col", 
		"Dt", 
		"Dt_in_sec" 
		};
	const char *fname_collected = "stats_collected.csv";
	Fio.int_matrix_write_csv_with_labels(fname_collected,
			Stats_short, Nb_cases, 6, Column_label);

	cout << "Written file " << fname_collected << " of size "
			<< Fio.file_size(fname_collected) << endl;
	
	
	Nb_sol = 0;
	Nb_col = 0;
	Dt_in_sec = 0;
	for (i = 0; i < Iso.nb_starter; i++) {
		Nb_sol += Stats[i * 6 + 1];
		Nb_col += Stats[i * 6 + 3];
		Dt_in_sec += Stats[i * 6 + 5];
		}

	cout << "In total we have:" << endl;
	cout << "Nb_sol = " << Nb_sol << endl;
	cout << "Nb_col = " << Nb_col << endl;
	cout << "Nb_col (average) = "
			<< (double) Nb_col / Iso.nb_starter << endl;
	cout << "Dt_in_sec = " << Dt_in_sec << endl;

	

	FREE_OBJECTS(S);
#if 0
	if (f_v) {
		cout << "isomorph_read_statistic_files "
				"before Iso.count_solutions" << endl;
		}
	int f_get_statistics = FALSE;


	Iso.count_solutions(nb_files, fname,
			f_get_statistics, verbose_level);
			// in isomorph_files.C
			//
			// now we know Iso.N, the number of solutions
			// from the clique finder
		
	registry_dump_sorted_by_size();
		
	Iso.build_up_database(nb_files, fname, verbose_level);
			// in isomorph_files.C
#endif

	}
	cout << "isomorph_read_statistic_files done" << endl;

	//discreta_exit();
}

void isomorph_build_db(
	action *A_base, action *A, poset_classification *gen,
	int size, const char *prefix_classify,
	const char *prefix_iso, int level,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_implicit_fusion = FALSE;
	int i;
	
	if (f_v) {
		cout << "isomorph_build_db" << endl;
		cout << "size = " << size << endl;
		cout << "level = " << level << endl;
		cout << "prefix_classify = " << prefix_classify << endl;
		cout << "prefix_iso = " << prefix_iso << endl;
		}

	
	discreta_init();

	{
	isomorph Iso;
	int f_use_database_for_starter = TRUE;
	
	if (f_v) {
		cout << "isomorph_build_db before Iso.init" << endl;
		}
	Iso.init(prefix_iso, A_base, A, gen, size, level,
		f_use_database_for_starter,
		f_implicit_fusion, verbose_level);
		// sets size, level and initializes file names

	
	Iso.read_data_files_for_starter(level, prefix_classify,
			verbose_level);
		// in isomorph.C, used gen->read_level_file_binary
	
	for (i = 0; i <= level; i++) {
		if (f_v) {
			cout << "isomorph_build_db creating level database for "
					"level " << i << " / " << level << endl;
			}
		Iso.create_level_database(i, verbose_level);
		}
	

	}
	if (f_v) {
		cout << "isomorph_build_db done" << endl;
		}

	//discreta_exit();
}

void isomorph_read_solution_files(
	action *A_base, action *A, poset_classification *gen,
	int size, const char *prefix_classify,
	const char *prefix_iso, int level,
	const char **fname, int nb_files, 
	int f_has_final_test_function, 
	int (*final_test_function)(int *data, int sz,
			void *final_test_data, int verbose_level),
	void *final_test_data, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_implicit_fusion = FALSE;
	

	if (f_v) {
		cout << "isomorph_read_solution_files" << endl;
		cout << "nb_files = " << nb_files << endl;
		cout << "prefix_classify = " << prefix_classify << endl;
		cout << "prefix_iso = " << prefix_iso << endl;
		}

	
	discreta_init();

	{
	isomorph Iso;
	int f_use_database_for_starter = TRUE;


	if (f_v) {
		cout << "size = " << size << endl;
		}
	
	if (f_v) {
		cout << "isomorph_read_solution_files "
				"before Iso.init" << endl;
		}
	Iso.init(prefix_iso, A_base, A, gen, size, level,
			f_use_database_for_starter,
		f_implicit_fusion, verbose_level);

	

	if (f_v) {
		cout << "isomorph_read_solution_files "
				"before Iso.read_data_files_for_starter" << endl;
		}
	Iso.read_data_files_for_starter(level,
			prefix_classify, verbose_level);
		// in isomorph.C
	


	if (f_v) {
		cout << "isomorph_read_solution_files "
				"before Iso.count_solutions" << endl;
		}
	int f_get_statistics = FALSE;
	Iso.count_solutions(nb_files, fname, f_get_statistics, 
			f_has_final_test_function,
			final_test_function, final_test_data,
			verbose_level);
			// in isomorph_files.C
			//
			// now we know Iso.N, the number of solutions
			// from the clique finder
		
	//registry_dump_sorted_by_size();
		
	Iso.build_up_database(nb_files, fname, 
		f_has_final_test_function,
		final_test_function, final_test_data,
		verbose_level);
			// in isomorph_files.C
	

	}
	cout << "isomorph_read_solution_files done" << endl;

	//discreta_exit();
}

void isomorph_init_solutions_from_memory(
	action *A_base, action *A, poset_classification *gen,
	int size, const char *prefix_classify,
	const char *prefix_iso, int level,
	int **Solutions, int *Nb_sol, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_implicit_fusion = FALSE;

	if (f_v) {
		cout << "isomorph_init_solutions_from_memory" << endl;
		cout << "prefix_classify = " << prefix_classify << endl;
		cout << "prefix_iso = " << prefix_iso << endl;
		}

	
	discreta_init();

	{
	isomorph Iso;
	int f_use_database_for_starter = TRUE;


	if (f_v) {
		cout << "size = " << size << endl;
		}
	
	if (f_v) {
		cout << "isomorph_init_solutions_from_memory "
				"before Iso.init" << endl;
		}
	Iso.init(prefix_iso, A_base, A, gen, size,
		level, f_use_database_for_starter,
		f_implicit_fusion, 0/*verbose_level - 2*/);
	if (f_v) {
		cout << "isomorph_init_solutions_from_memory "
				"after Iso.init" << endl;
		}

	

	if (f_v) {
		cout << "isomorph_init_solutions_from_memory "
				"before Iso.read_data_files_for_starter" << endl;
		}
	Iso.read_data_files_for_starter(level,
			prefix_classify, 0/*verbose_level - 4*/);
		// in isomorph.C
	if (f_v) {
		cout << "isomorph_init_solutions_from_memory "
				"after Iso.read_data_files_for_starter" << endl;
		}
	


	if (f_v) {
		cout << "isomorph_init_solutions_from_memory "
				"before Iso.init_solutions" << endl;
		}
	//int f_get_statistics = FALSE;
	Iso.init_solutions(Solutions, Nb_sol, verbose_level - 1);
			// in isomorph_files.C
			//
			// now we know Iso.N, the number of solutions
			// from the clique finder
		
	if (f_v) {
		cout << "isomorph_init_solutions_from_memory "
				"after Iso.init_solutions" << endl;
		}
	
	}
	if (f_v) {
		cout << "isomorph_init_solutions_from_memory done" << endl;
		}

	//discreta_exit();
}

void isomorph_read_solution_files_from_clique_finder_case_by_case(
		action *A_base, action *A, poset_classification *gen,
	int size, const char *prefix_classify, const char *prefix_iso, int level, 
	const char **fname, int *list_of_cases, int nb_files, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_implicit_fusion = FALSE;

	if (f_v) {
		cout << "isomorph_read_solution_files_from_clique_"
				"finder_case_by_case" << endl;
		cout << "nb_files = " << nb_files << endl;
		cout << "prefix_iso = " << prefix_iso << endl;
		}

	
	discreta_init();

	{
	isomorph Iso;
	int f_use_database_for_starter = TRUE;


	if (f_v) {
		cout << "size = " << size << endl;
		}
	
	if (f_v) {
		cout << "isomorph_read_solution_files_from_clique_"
				"finder_case_by_case before Iso.init" << endl;
		}
	Iso.init(prefix_iso, A_base, A, gen, size, level,
			f_use_database_for_starter,
		f_implicit_fusion, 0/*verbose_level - 2*/);
	if (f_v) {
		cout << "isomorph_read_solution_files_from_clique_"
				"finder_case_by_case after Iso.init" << endl;
		}
	

	if (f_v) {
		cout << "isomorph_read_solution_files_from_clique_"
				"finder_case_by_case before Iso.read_data_files_"
				"for_starter" << endl;
		}
	Iso.read_data_files_for_starter(level, prefix_classify,
			0/*verbose_level - 4*/);
		// in isomorph.C
	if (f_v) {
		cout << "isomorph_read_solution_files_from_clique_"
				"finder_case_by_case after Iso.read_data_files_"
				"for_starter" << endl;
		}
	


	if (f_v) {
		cout << "isomorph_read_solution_files_from_clique_finder_"
				"case_by_case before Iso.count_solutions_"
				"from_clique_finder" << endl;
		}
	//int f_get_statistics = FALSE;
	Iso.count_solutions_from_clique_finder_case_by_case(nb_files,
			list_of_cases, fname, /*f_get_statistics,*/
			verbose_level - 1);
			// in isomorph_files.C
			//
			// now we know Iso.N, the number of solutions
			// from the clique finder
		
	//registry_dump_sorted_by_size();
		
	if (f_v) {
		cout << "isomorph_read_solution_files_from_clique_finder_"
				"case_by_case before Iso.read_solutions_from_clique_"
				"finder_case_by_case" << endl;
		}
	Iso.read_solutions_from_clique_finder_case_by_case(nb_files,
			list_of_cases, fname, verbose_level - 1);
			// in isomorph_files.C
	if (f_v) {
		cout << "isomorph_read_solution_files_from_clique_finder_"
				"case_by_case after Iso.read_solutions_from_clique_"
				"finder_case_by_case" << endl;
		}
	

	}
	if (f_v) {
		cout << "isomorph_read_solution_files_from_clique_finder_"
				"case_by_case done" << endl;
		}

	//discreta_exit();
}

void isomorph_read_solution_files_from_clique_finder(
	action *A_base, action *A, poset_classification *gen,
	int size, const char *prefix_classify, const char *prefix_iso, int level, 
	const char **fname, int nb_files, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_implicit_fusion = FALSE;

	if (f_v) {
		cout << "isomorph_read_solution_files_from_"
				"clique_finder" << endl;
		cout << "nb_files = " << nb_files << endl;
		cout << "prefix_iso = " << prefix_iso << endl;
		}

	
	discreta_init();

	{
	isomorph Iso;
	int f_use_database_for_starter = TRUE;


	if (f_v) {
		cout << "size = " << size << endl;
		}
	
	if (f_v) {
		cout << "isomorph_read_solution_files_from_clique_finder "
				"before Iso.init" << endl;
		}
	Iso.init(prefix_iso, A_base, A, gen,
		size, level, f_use_database_for_starter,
		f_implicit_fusion, 0/*verbose_level - 2*/);
	if (f_v) {
		cout << "isomorph_read_solution_files_from_clique_finder "
				"after Iso.init" << endl;
		}
	

	if (f_v) {
		cout << "isomorph_read_solution_files_from_clique_finder "
				"before Iso.read_data_files_for_starter" << endl;
		}
	Iso.read_data_files_for_starter(level, prefix_classify,
			0/*verbose_level - 4*/);
		// in isomorph.C
	if (f_v) {
		cout << "isomorph_read_solution_files_from_clique_finder "
				"after Iso.read_data_files_for_starter" << endl;
		}
	


	if (f_v) {
		cout << "isomorph_read_solution_files_from_clique_finder "
				"before Iso.count_solutions_from_clique_finder" << endl;
		}
	//int f_get_statistics = FALSE;
	Iso.count_solutions_from_clique_finder(nb_files, fname,
			verbose_level - 1);
			// in isomorph_files.C
			//
			// now we know Iso.N, the number of solutions
			// from the clique finder
		
	//registry_dump_sorted_by_size();
		
	if (f_v) {
		cout << "isomorph_read_solution_files_from_clique_finder "
				"before Iso.read_solutions_from_clique_finder" << endl;
		}
	Iso.read_solutions_from_clique_finder(nb_files,
			fname, verbose_level - 1);
			// in isomorph_files.C
	if (f_v) {
		cout << "isomorph_read_solution_files_from_clique_finder "
				"after Iso.read_solutions_from_clique_finder" << endl;
		}
	

	}
	if (f_v) {
		cout << "isomorph_read_solution_files_from_clique_finder "
				"done" << endl;
		}

	//discreta_exit();
}

void isomorph_compute_orbits(
	action *A_base, action *A, poset_classification *gen,
	int size, const char *prefix_classify,
	const char *prefix_iso, int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_implicit_fusion = FALSE;

	if (f_v) {
		cout << "isomorph_compute_orbits" << endl;
		}

	
	discreta_init();

	{
	isomorph Iso;
	int f_use_database_for_starter = TRUE;

	
	if (f_v) {
		cout << "isomorph_compute_orbits before Iso.init" << endl;
		}
	Iso.init(prefix_iso, A_base, A, gen, size, level,
		f_use_database_for_starter,
		f_implicit_fusion, verbose_level);
		// sets q, level and initializes file names

	

	Iso.read_data_files_for_starter(level,
			prefix_classify, verbose_level);
	
	Iso.init_solution(verbose_level);
	
	Iso.orbits_of_stabilizer(verbose_level);

	Iso.write_orbit_data(verbose_level);


	

	}
	cout << "isomorph_compute_orbits done" << endl;

	//discreta_exit();
}


void isomorph_testing(
	action *A_base, action *A, poset_classification *gen,
	int size, const char *prefix_classify,
	const char *prefix_iso, int level,
	int f_play_back, const char *old_event_file,
	int print_mod, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_implicit_fusion = FALSE;
	int t0;


	t0 = os_ticks();


	if (f_v) {
		cout << "isomorph_testing" << endl;
		}

	
	discreta_init();

	{
	isomorph Iso;
	int f_use_database_for_starter = FALSE;
	file_io Fio;

	
	if (f_v) {
		cout << "isomorph_testing before Iso.init" << endl;
		}
	Iso.init(prefix_iso, 
		A_base, A, gen,
		size, level, 
		f_use_database_for_starter, 
		f_implicit_fusion, 
		verbose_level - 1);

	

	Iso.read_data_files_for_starter(level,
			prefix_classify, verbose_level - 1);
	//Iso.compute_nb_starter(search_depth, verbose_level);

	Iso.init_solution(verbose_level - 1);

	Iso.load_table_of_solutions(verbose_level - 1);
	
	Iso.read_orbit_data(verbose_level - 1);

	Iso.depth_completed = level /*- 2*/;

	if (f_v) {
		cout << "isomorph_testing before Iso.gen->recreate_schreier_"
				"vectors_up_to_level" << endl;
		}
	Iso.gen->recreate_schreier_vectors_up_to_level(level - 1,
			verbose_level - 1);

	int i;
	
	if (f_v) {
		for (i = 0; i <= level + 1; i++) {
			cout << "gen->first_oracle_node_at_level[" << i << "]="
					<< gen->first_poset_orbit_node_at_level[i] << endl;
			}
		cout << "Iso.depth_completed=" << Iso.depth_completed << endl;
		}
	
#if 0
	cout << "Node 28:" << endl;
	Iso.gen->root[28].print_node(Iso.gen);
#endif

	Iso.iso_test_init(verbose_level - 1);

	int f_implicit_fusion = FALSE;
	
	Iso.gen->f_allowed_to_show_group_elements = FALSE;
	
	Iso.read_starter_nb_orbits(verbose_level); // added Oct 30, 2014 


	if (f_v) {
		cout << "isomorph_testing before Iso.isomorph_testing" << endl;
		}
	Iso.isomorph_testing(t0, f_play_back, old_event_file,
			f_implicit_fusion, print_mod, verbose_level);
	
	Iso.Reps->save(verbose_level - 1);


	int data1[1000];
	int id, orbit;

	Iso.setup_and_open_solution_database(verbose_level - 1);
	
	char fname[1000];
	sprintf(fname, "%sorbits.txt", prefix_iso);
	{
	ofstream fp(fname);
	fp << "# " << Iso.size << endl;
	for (orbit = 0; orbit < Iso.Reps->count; orbit++) {
	
		
		id = Iso.orbit_perm[Iso.orbit_fst[Iso.Reps->rep[orbit]]];
	
		Iso.load_solution(id, data1);
		if (FALSE) {
			cout << "read representative of orbit " << orbit
					<< " (id=" << id << ")" << endl;
			int_vec_print(cout, data1, Iso.size);
			cout << endl;
			}

#if 0
		for (i = 0; i < Iso.size; i++) {
			cout << setw(8) << data1[i] << ", ";
			}
		cout << endl;
#endif
		fp << Iso.size;
		for (i = 0; i < Iso.size; i++) {
			fp << " " << data1[i];
			}
		longinteger_object go;

		Iso.Reps->stab[orbit]->group_order(go);
		fp << " ";
		go.print_not_scientific(fp);
		fp << endl;

		//write_set_to_file(fname, data1, Iso.size, verbose_level - 1);
		}
	fp << "-1 " << Iso.Reps->count << endl;
	
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	
	Iso.close_solution_database(verbose_level - 1);

#if 0
	Iso.print_set_function = callback_print_isomorphism_type_extend_regulus;
	Iso.print_set_data = this;
	Iso.print_isomorphism_types(verbose_level);
#endif


	}
	cout << "isomorph_testing done" << endl;

	//discreta_exit();
}

void isomorph_classification_graph(
	action *A_base, action *A, poset_classification *gen,
	int size, const char *prefix_classify,
	const char *prefix_iso, int level,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_implicit_fusion = FALSE;
	//int t0;


	//t0 = os_ticks();


	if (f_v) {
		cout << "isomorph_classification_graph" << endl;
		}

	
	discreta_init();

	{
	isomorph Iso;
	int f_use_database_for_starter = FALSE;

	
	if (f_v) {
		cout << "isomorph_classification_graph "
				"before Iso.init" << endl;
		}
	Iso.init(prefix_iso, 
		A_base, A, gen,
		size, level, 
		f_use_database_for_starter, 
		f_implicit_fusion, 
		verbose_level - 1);

	

	Iso.read_everything_including_classification(
			prefix_classify, verbose_level);

	



	Iso.write_classification_matrix(verbose_level);
	Iso.write_classification_graph(verbose_level);
	Iso.decomposition_matrix(verbose_level);
	


	}
	cout << "isomorph_classification_graph done" << endl;

	//discreta_exit();
}


void isomorph_identify(
	action *A_base, action *A, poset_classification *gen,
	int size, const char *prefix_classify,
	const char *prefix_iso, int level,
	int identify_nb_files, const char **fname, int *Iso_type, 
	int f_save, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_implicit_fusion = FALSE;
	int *the_set;
	int set_size;
	char fname_transporter[1000];




	if (f_v) {
		cout << "isomorph_identify" << endl;
		}
	
	
	discreta_init();

	{
	isomorph Iso;
	int f_use_database_for_starter = FALSE;
	file_io Fio;

	
	if (f_v) {
		cout << "isomorph_identify before Iso.init" << endl;
		}
	Iso.init(prefix_iso, A_base, A, gen, 
		size, level, 
		f_use_database_for_starter, 
		f_implicit_fusion, 
		verbose_level);



	Iso.read_everything_including_classification(
			prefix_classify, verbose_level);


	int i;
	
	for (i = 0; i < identify_nb_files; i++) {
	
		Fio.read_set_from_file(fname[i], the_set, set_size, verbose_level);
		if (f_v) {
			cout << "isomorph_identify read file " << fname[i] << endl;
			cout << "the_set = ";
			int_vec_print(cout, the_set, set_size);
			cout << endl;
			}


		sprintf(fname_transporter, "transporter_%s", fname[i]);


		if (f_v) {
			cout << "isomorph_identify before Iso.identify" << endl;
			}
		Iso_type[i] = Iso.identify(the_set,
				f_implicit_fusion, verbose_level - 2);
		if (f_v) {
			cout << "isomorph_identify after Iso.identify" << endl;
			}
	
		if (f_save) {

#if 0
			FILE *f2;
			f2 = fopen(fname_transporter, "wb");
			Iso.A_base->element_write_file_fp(Iso.transporter,
					f2, 0/* verbose_level*/);
	
			fclose(f2);
#else
			Iso.A_base->element_write_file(Iso.transporter,
					fname_transporter, 0 /* verbose_level*/);
#endif
			cout << "isomorph_identify written file " << fname_transporter
					<< " of size " << Fio.file_size(fname_transporter) << endl;
			}


		if (f_v) {
			cout << "isomorph_identify The set in " << fname[i]
				<< " belongs to isomorphism type " << Iso_type[i] << endl;
			}

		FREE_int(the_set);
		}


	if (f_v) {
		cout << "isomorph_identify Summary:" << endl;
		for (i = 0; i < identify_nb_files; i++) {
			cout << i << " : " << fname[i] << " : " << Iso_type[i] << endl;
			}
		}


	}
	cout << "isomorph_identify done" << endl;
	//discreta_exit();
}

void isomorph_identify_table(
	action *A_base, action *A, poset_classification *gen,
	int size, const char *prefix_classify,
	const char *prefix_iso, int level,
	int nb_rows, int *Table, int *Iso_type, 
	int verbose_level)
// Table[nb_rows * size]
{
	int f_v = (verbose_level >= 1);
	int f_implicit_fusion = FALSE;
	int *the_set;
	int set_size;




	if (f_v) {
		cout << "isomorph_identify_table" << endl;
		}
	
	
	discreta_init();

	{
	isomorph Iso;
	int f_use_database_for_starter = FALSE;

	
	if (f_v) {
		cout << "isomorph_identify_table before Iso.init" << endl;
		}
	Iso.init(prefix_iso, A_base, A, gen, 
		size, level, 
		f_use_database_for_starter, 
		f_implicit_fusion, 
		verbose_level);
		// sets level and initializes file names


	

	Iso.read_everything_including_classification(
			prefix_classify, verbose_level);

	int i;
	


	set_size = size;
	the_set = NEW_int(set_size);

	for (i = 0; i < nb_rows; i++) {
	
		int_vec_copy(Table + i * set_size, the_set, set_size);
		
		if (f_v) {
			cout << "isomorph_identify_table "
					"Identifying set no " << i << endl;
			cout << "the_set = ";
			int_vec_print(cout, the_set, set_size);
			cout << endl;
			}



		if (f_v) {
			cout << "isomorph_identify_table "
					"before Iso.identify" << endl;
			}
		Iso_type[i] = Iso.identify(the_set,
				f_implicit_fusion, verbose_level - 2);
		if (f_v) {
			cout << "isomorph_identify_table "
					"after Iso.identify" << endl;
			}
	


		if (f_v) {
			cout << "isomorph_identify_table The set no " << i
					<< " belongs to isomorphism type "
					<< Iso_type[i] << endl;
			}

		}
	FREE_int(the_set);


	if (f_v) {
		cout << "isomorph_identify_table Summary:" << endl;
		for (i = 0; i < nb_rows; i++) {
			cout << i << " : " << Iso_type[i] << endl;
			}
		}


	}
	cout << "isomorph_identify_table done" << endl;
	//discreta_exit();
}

void isomorph_worker(
	action *A_base, action *A, poset_classification *gen,
	int size, const char *prefix_classify, const char *prefix_iso, 
	void (*work_callback)(isomorph *Iso, void *data, int verbose_level), 
	void *work_data, 
	int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph_worker" << endl;
		cout << "isomorph_worker size=" << size << endl;
		cout << "isomorph_worker level=" << level << endl;
		}

	
	discreta_init();

	{
	isomorph Iso;
	int f_use_database_for_starter = FALSE;
	int f_implicit_fusion = FALSE;
	
	if (f_v) {
		cout << "isomorph_worker before Iso.init" << endl;
		}
	Iso.init(prefix_iso, 
		A_base, A, gen,
		size, level, 
		f_use_database_for_starter, 
		f_implicit_fusion, 
		verbose_level);
		// sets level and initializes file names

	Iso.read_everything_including_classification(
			prefix_classify, verbose_level);


	

	Iso.setup_and_open_solution_database(verbose_level - 1);
	Iso.setup_and_open_level_database(verbose_level - 1);

#if 0
	Iso.print_set_function = callback_print_isomorphism_type_extend_regulus;
	Iso.print_set_data = this;
	Iso.print_isomorphism_types(f_select, select_first,
			select_len, verbose_level);
#endif


	(*work_callback)(&Iso, work_data, verbose_level);


	Iso.close_solution_database(verbose_level - 1);
	Iso.close_level_database(verbose_level - 1);
	


	

	}
	cout << "isomorph_worker done" << endl;

}

void isomorph_compute_down_orbits(
	action *A_base, action *A, poset_classification *gen,
	int size, const char *prefix_classify, const char *prefix, 
	void *data, 
	int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph_compute_down_orbits "
				"level = " << level << endl;
		cout << "isomorph_compute_down_orbits "
				"verbose_level = " << verbose_level << endl;
		}
	isomorph_worker(A_base, A, gen, 
		size, prefix_classify, prefix, 
		isomorph_compute_down_orbits_worker, 
		data, 
		level, verbose_level);
	if (f_v) {
		cout << "isomorph_compute_down_orbits done" << endl;
		}
}

void isomorph_compute_down_orbits_worker(
		isomorph *Iso, void *data, int verbose_level)
// data is not needed
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int orbit;
	int *Nb_orbits;
	int nb_orbits = 0;
	int nb_special_orbits = 0;
	int **Down_orbit_identify;
	int *Down_identify;
	int h, i, idx;
	file_io Fio;

	if (f_v) {
		cout << "isomorph_compute_down_orbits_worker" << endl;
		}

	//f_memory_debug = TRUE;
	Nb_orbits = NEW_int(Iso->Reps->count * 2);
	Down_orbit_identify = NEW_pint(Iso->Reps->count);
	for (orbit = 0; orbit < Iso->Reps->count; orbit++) {

		int cnt_orbits, cnt_special_orbits;
		int *special_orbit_identify;

		isomorph_compute_down_orbits_for_isomorphism_type(
				Iso, orbit, cnt_orbits, cnt_special_orbits,
				special_orbit_identify, verbose_level - 1);

		if (f_vv) {
			cout << "isomorph_compute_down_orbits_worker orbit "
					<< orbit << " / " << Iso->Reps->count
					<< " cnt_orbits=" << cnt_orbits
					<< " cnt_special_orbits=" << cnt_special_orbits << endl;
			}
		Nb_orbits[orbit * 2 + 0] = cnt_orbits;
		Nb_orbits[orbit * 2 + 1] = cnt_special_orbits;
		Down_orbit_identify[orbit] = special_orbit_identify;
		
		nb_orbits += cnt_orbits;
		nb_special_orbits += cnt_special_orbits;

#if 0
		if (orbit && ((orbit % 100) == 0)) {
			registry_dump_sorted();
			}
#endif

		}

	Fio.int_matrix_write_csv("Nb_down_orbits.csv",
			Nb_orbits, Iso->Reps->count, 2);
	
	if (f_v) {
		cout << "isomorph_compute_down_orbits_worker" << endl;
		cout << "nb_orbits=" << nb_orbits << endl;
		cout << "nb_special_orbits=" << nb_special_orbits << endl;
		}

	Down_identify = NEW_int(nb_special_orbits * 3);
	h = 0;
	for (orbit = 0; orbit < Iso->Reps->count; orbit++) {
		for (i = 0; i < Nb_orbits[orbit * 2 + 1]; i++) {
			idx = Down_orbit_identify[orbit][i];
			Down_identify[h * 3 + 0] = orbit;
			Down_identify[h * 3 + 1] = i;
			Down_identify[h * 3 + 2] = idx;
			h++;
			}
		}
	
	Fio.int_matrix_write_csv("Down_identify.csv",
			Down_identify, nb_special_orbits, 3);

	for (orbit = 0; orbit < Iso->Reps->count; orbit++) {
		FREE_int(Down_orbit_identify[orbit]);
		}
	FREE_pint(Down_orbit_identify);
	FREE_int(Down_identify);
	FREE_int(Nb_orbits);
	if (f_v) {
		cout << "isomorph_compute_down_orbits_worker done" << endl;
		}
}

void isomorph_compute_down_orbits_for_isomorphism_type(
	isomorph *Iso, int orbit,
	int &cnt_orbits, int &cnt_special_orbits,
	int *&special_orbit_identify, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int id, rep, first; //, c;
	int data[1000];
	combinatorics_domain Combi;
	sorting Sorting;

	if (f_v) {
		cout << "isomorph_compute_down_orbits_for_isomorphism_type "
				"orbit=" << orbit << endl;
		}

	cnt_orbits = 0;
	cnt_special_orbits = 0;

	rep = Iso->Reps->rep[orbit];
	first = Iso->orbit_fst[rep];
	//c = Iso->starter_number[first];
	id = Iso->orbit_perm[first];		
	Iso->load_solution(id, data);

	

	sims *Stab;
	strong_generators *Strong_gens;

	Stab = Iso->Reps->stab[orbit];

	if (f_vv) {
		cout << "isomorph_compute_down_orbits_for_isomorphism_type "
				"computing induced action on the set (in data)" << endl;
		}


	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->init_from_sims(Stab, verbose_level - 2);
	

	Iso->induced_action_on_set(Stab, data, 0 /*verbose_level*/);
		
	if (f_vv) {
		cout << "data after induced_action_on_set:" << endl;
		int_vec_print(cout, data, Iso->size);
		cout << endl;
		}
		
	longinteger_object go1;
			
	Iso->AA->group_order(go1);

	if (f_vv) {
		cout << "action " << Iso->AA->label << " computed, "
				"group order is " << go1 << endl;

		cout << "Order of the group that is induced on the object is ";
		cout << "$";
		go1.print_not_scientific(cout);
		cout << "$\\\\" << endl;
		}

	if (FALSE /*go1.is_one()*/) {
		cnt_orbits = Combi.int_n_choose_k(Iso->size, Iso->level);
		cnt_special_orbits = 1;
		}
	else {
		int *orbit_reps;
		int nb_orbits;

		if (f_vv) {
			cout << "isomorph_compute_down_orbits_for_isomorphism_type "
					"orbit=" << orbit << " / " << Iso->Reps->count
					<< " computing orbits on subsets" << endl;
			}
		poset *Poset;
		Poset = NEW_OBJECT(poset);
		Poset->init_subset_lattice(
				Iso->A_base, Iso->AA, Strong_gens,
				verbose_level);
		Poset->orbits_on_k_sets(
			Iso->level, orbit_reps, nb_orbits, verbose_level - 5);
		FREE_OBJECT(Poset);
		if (f_vv) {
			cout << "isomorph_compute_down_orbits_for_isomorphism_type "
					"orbit=" << orbit << " / " << Iso->Reps->count
					<< " computing orbits on subsets done" << endl;
			}

		if (f_vvv) {
			cout << "Orbit reps: nb_orbits=" << nb_orbits << endl;
			int_matrix_print(orbit_reps, nb_orbits, Iso->level);
			}

		if (f_vv) {
			cout << "Number of orbits on $" << Iso->level << "$-sets is "
					<< nb_orbits << ".\\\\" << endl;
			}

		int *rearranged_set;
		int *transporter;
		int u;
		int case_nb;
		int f_implicit_fusion = FALSE;
		int idx;
		
		rearranged_set = NEW_int(Iso->size);
		transporter = NEW_int(Iso->A_base->elt_size_in_int);

		cnt_orbits = nb_orbits;
		cnt_special_orbits = 0;
		special_orbit_identify = NEW_int(nb_orbits);
		for (u = 0; u < nb_orbits; u++) {

			if (f_vv) {
				cout << "iso type " << orbit << " / " << Iso->Reps->count
						<< " down_orbit " << u << " / "
						<< nb_orbits << ":" << endl;
				int_vec_print(cout, orbit_reps + u * Iso->level, Iso->level);
				cout << endl;
				}



			Sorting.rearrange_subset(Iso->size, Iso->level, data,
				orbit_reps + u * Iso->level, rearranged_set,
				0/*verbose_level - 3*/);
				// in GALOIS/sorting.C


			//int_vec_print(cout, rearranged_set, Iso.size);
			//cout << endl;
			int f_failure_to_find_point, f_found;

			Iso->A_base->element_one(transporter, 0);
			case_nb = Iso->trace_set(rearranged_set, transporter, 
				f_implicit_fusion, f_failure_to_find_point,
				0 /*verbose_level - 2*/);

			//cout << "f_failure_to_find_point="
			// << f_failure_to_find_point << endl;
			//cout << "case_nb=" << case_nb << endl;
			if (f_failure_to_find_point) {
				cout << "isomorph_compute_down_orbits_for_isomorphism_type "
						"f_failure_to_find_point" << endl;
				exit(1);
				}	

			f_found = Iso->find_extension_easy_new(rearranged_set,
					case_nb, idx, 0 /* verbose_level */);
#if 0
			f_found = Iso.identify_solution_relaxed(prefix, transporter, 
				f_implicit_fusion, orbit_no0,
				f_failure_to_find_point, 3 /*verbose_level*/);
#endif

			//cout << "f_found=" << f_found << endl;
			if (!f_found) {
				if (f_vv) {
					cout << "isomorph_compute_down_orbits_for_"
							"isomorphism_type not found" << endl;
					}
				continue;
				}
			else {
				if (f_vv) {
					cout << "iso type " << orbit << " / " << Iso->Reps->count
							<< " down orbit " << u << " / " << nb_orbits
							<< " leads to orbit " << idx << endl;
					}
				}
			special_orbit_identify[cnt_special_orbits] = idx;
			cnt_special_orbits++;
			} // next u
		if (f_v) {
			cout << "Number of special orbits on $" << Iso->level
					<< "$-sets is " << cnt_special_orbits
					<< ".\\\\" << endl;
			}

		int *soi;
		int i;

		soi = NEW_int(cnt_special_orbits);
		for (i = 0; i < cnt_special_orbits; i++) {
			soi[i] = special_orbit_identify[i];
			}
		FREE_int(special_orbit_identify);
		special_orbit_identify = soi;


		FREE_int(rearranged_set);
		FREE_int(transporter);
		FREE_int(orbit_reps);
	}
	FREE_OBJECT(Strong_gens);

	if (f_v) {
		cout << "isomorph_compute_down_orbits_for_isomorphism_type "
				"done" << endl;
		}
}

void isomorph_report_data_in_source_code_inside_tex(
		isomorph &Iso, const char *prefix,
		char *label_of_structure_plural, ostream &f,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int *selection;
	int selection_size;
	int i;

	if (f_v) {
		cout << "isomorph_report_data_in_source_code" << endl;
		}
	selection_size = Iso.Reps->count;
	selection = NEW_int(selection_size);
	for (i = 0; i < selection_size; i++) {
		selection[i] = i;
		}
	isomorph_report_data_in_source_code_inside_tex_with_selection(
		Iso, prefix,
		label_of_structure_plural, f, 
		selection_size, selection, verbose_level);
	FREE_int(selection);
}


void isomorph_report_data_in_source_code_inside_tex_with_selection(
		isomorph &Iso, const char *prefix,
		char *label_of_structure_plural, ostream &fp,
		int selection_size, int *selection,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int h, rep, first, /*c,*/ id, i, s;
	int data[1000];

	if (f_v) {
		cout << "isomorph_report_data_in_source_code" << endl;
		}

	fp << "\\section{The " << label_of_structure_plural
			<< " in Numeric Form}" << endl << endl;

	//fp << "\\clearpage" << endl << endl;
	for (s = 0; s < selection_size; s++) {
		h = selection[s];
		rep = Iso.Reps->rep[h];
		first = Iso.orbit_fst[rep];
		//c = Iso.starter_number[first];
		id = Iso.orbit_perm[first];		
		Iso.load_solution(id, data);
		for (i = 0; i < Iso.size; i++) {
			fp << data[i];
			if (i < Iso.size - 1) {
				fp << ", ";
				}
			}
		fp << "\\\\" << endl;
		}
	fp << "\\begin{verbatim}" << endl << endl;
	fp << "int " << prefix << "_size = " << Iso.size << ";" << endl;
	fp << "int " << prefix << "_nb_reps = " << selection_size << ";" << endl;
	fp << "int " << prefix << "_reps[] = {" << endl;
	for (s = 0; s < selection_size; s++) {
		h = selection[s];
		rep = Iso.Reps->rep[h];
		first = Iso.orbit_fst[rep];
		//c = Iso.starter_number[first];
		id = Iso.orbit_perm[first];		
		Iso.load_solution(id, data);
		fp << "\t";
		for (i = 0; i < Iso.size; i++) {
			fp << data[i];
			fp << ", ";
			}
		fp << endl;
		}
	fp << "};" << endl;
	fp << "const char *" << prefix << "_stab_order[] = {" << endl;
	for (s = 0; s < selection_size; s++) {
		h = selection[s];

		longinteger_object go;
		
		rep = Iso.Reps->rep[h];
		first = Iso.orbit_fst[rep];
		//c = Iso.starter_number[first];
		id = Iso.orbit_perm[first];		
		Iso.load_solution(id, data);
		if (Iso.Reps->stab[h]) {
			Iso.Reps->stab[h]->group_order(go);
			fp << "\"";
			go.print_not_scientific(fp);
			fp << "\"," << endl;
			}
		else {
			fp << "\"";
			fp << "1";
			fp << "\"," << endl;
			}
		}
	fp << "};" << endl;
	
	{
	int *stab_gens_first;
	int *stab_gens_len;
	int fst;

	stab_gens_first = NEW_int(selection_size);
	stab_gens_len = NEW_int(selection_size);
	fst = 0;
	fp << "int " << prefix << "_stab_gens[] = {" << endl;
	for (s = 0; s < selection_size; s++) {
		h = selection[s];
		vector_ge *gens;
		int *tl;
		int j;

		gens = NEW_OBJECT(vector_ge);
		tl = NEW_int(Iso.A_base->base_len);
		
		if (f_vv) {
			cout << "isomorph_report_data_in_source_code_inside_"
					"tex_with_selection before extract_strong_"
					"generators_in_order" << endl;
			}
		Iso.Reps->stab[h]->extract_strong_generators_in_order(
				*gens, tl, 0);

		stab_gens_first[s] = fst;
		stab_gens_len[s] = gens->len;
		fst += gens->len;
		
		for (j = 0; j < gens->len; j++) {
			if (f_vv) {
				cout << "isomorph_report_data_in_source_code_inside_"
						"tex_with_selection before extract_strong_"
						"generators_in_order generator " << j
						<< " / " << gens->len << endl;
				}
			fp << "";
			Iso.A_base->element_print_for_make_element(gens->ith(j), fp);
			fp << endl;
			}

		FREE_int(tl);
		FREE_OBJECT(gens);
		}
	fp << "};" << endl;
	fp << "int " << prefix << "_stab_gens_fst[] = { ";
	for (s = 0; s < selection_size; s++) {
		fp << stab_gens_first[s];
		if (s < selection_size - 1) {
			fp << ", ";
			}
		}
	fp << "};" << endl;
	fp << "int " << prefix << "_stab_gens_len[] = { ";
	for (s = 0; s < selection_size; s++) {
		fp << stab_gens_len[s];
		if (s < selection_size - 1) {
			fp << ", ";
			}
		}
	fp << "};" << endl;
	fp << "int " << prefix << "_make_element_size = "
			<< Iso.A_base->make_element_size << ";" << endl;
	}
	fp << "\\end{verbatim}" << endl << endl;
}



}}



