// isomorph_arguments.C
//
// Anton Betten
// January 27, 2016

#include "orbiter.h"



isomorph_arguments::isomorph_arguments()
{
	null();
}

isomorph_arguments::~isomorph_arguments()
{
	freeself();
}

void isomorph_arguments::null()
{
	f_init_has_been_called = FALSE;

	ECA = NULL;
	f_build_db = FALSE;

	f_read_solutions = FALSE;
	f_read_solutions_from_clique_finder = FALSE;
	f_read_solutions_from_clique_finder_list_of_cases = FALSE;
	fname_list_of_cases = NULL;
	f_read_solutions_after_split = FALSE;
	read_solutions_split_m = 0;
	
	f_read_statistics_after_split = FALSE;
	read_statistics_split_m = 0;

	f_compute_orbits = FALSE;
	f_isomorph_testing = FALSE;
	f_classification_graph = FALSE;
	f_event_file = FALSE; // -e <event file> option
	event_file_name = NULL;
	print_mod = 500;
	f_report = FALSE;
	f_subset_orbits = FALSE;
	f_subset_orbits_file = FALSE;
	f_down_orbits = FALSE;

	prefix_iso = "./ISO/";

	f_has_final_test_function = FALSE;
	final_test_function = NULL;
	final_test_data = NULL;
}

void isomorph_arguments::freeself()
{
	null();
}

void isomorph_arguments::read_arguments(int argc, const char **argv, 
	INT verbose_level)
{
	INT i;

	for (i = 1; i < argc; i++) {
		if (argv[i][0] != '-') {
			continue;
			}
		else if (strcmp(argv[i], "-build_db") == 0) {
			f_build_db = TRUE;
			cout << "-build_db " << endl;
			}
		else if (strcmp(argv[i], "-read_solutions") == 0) {
			f_read_solutions = TRUE;
			cout << "-read_solutions " << endl;
			}
		else if (strcmp(argv[i], "-read_solutions_from_clique_finder") == 0) {
			f_read_solutions_from_clique_finder = TRUE;
			cout << "-read_solutions_from_clique_finder " << endl;
			}
		else if (strcmp(argv[i], "-read_solutions_from_clique_finder_list_of_cases") == 0) {
			f_read_solutions_from_clique_finder_list_of_cases = TRUE;
			fname_list_of_cases = argv[++i];
			cout << "-read_solutions_from_clique_finder_list_of_cases " << fname_list_of_cases << endl;
			}
		else if (strcmp(argv[i], "-read_solutions_after_split") == 0) {
			f_read_solutions_after_split = TRUE;
			read_solutions_split_m = atoi(argv[++i]);
			cout << "-read_solutions_after_split " << read_solutions_split_m << endl;
			}
		else if (strcmp(argv[i], "-read_statistics_after_split") == 0) {
			f_read_statistics_after_split = TRUE;
			read_statistics_split_m = atoi(argv[++i]);
			cout << "-read_statistics_after_split " << read_statistics_split_m << endl;
			}

		else if (strcmp(argv[i], "-compute_orbits") == 0) {
			f_compute_orbits = TRUE;
			cout << "-compute_orbits " << endl;
			}
		else if (strcmp(argv[i], "-isomorph_testing") == 0) {
			f_isomorph_testing = TRUE;
			cout << "-isomorph_testing " << endl;
			}
		else if (strcmp(argv[i], "-classification_graph") == 0) {
			f_classification_graph = TRUE;
			cout << "-make_classification_graph " << endl;
			}
		else if (strcmp(argv[i], "-e") == 0) {
			i++;
			f_event_file = TRUE;
			event_file_name = argv[i];
			cout << "-e " << event_file_name << endl;
			}
		else if (strcmp(argv[i], "-print_interval") == 0) {
			print_mod = atoi(argv[++i]);
			cout << "-print_interval " << print_mod << endl;
			}
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report " << endl;
			}
		else if (strcmp(argv[i], "-subset_orbits") == 0) {
			f_subset_orbits = TRUE;
			cout << "-subset_orbits " << endl;
			}
		else if (strcmp(argv[i], "-subset_orbits_file") == 0) {
			f_subset_orbits_file = TRUE;
			subset_orbits_fname = argv[++i];
			cout << "-subset_orbits_fname " << endl;
			}
		else if (strcmp(argv[i], "-down_orbits") == 0) {
			f_down_orbits = TRUE;
			cout << "-down_orbits " << endl;
			}
		else if (strcmp(argv[i], "-prefix_iso") == 0) {
			prefix_iso = argv[++i];
			cout << "-prefix_iso " << prefix_iso << endl;
			}
		}
}


void isomorph_arguments::init(action *A, action *A2,
	poset_classification *gen,
	INT target_size, const BYTE *prefix_with_directory, exact_cover_arguments *ECA, 
	void (*callback_report)(isomorph *Iso, void *data, INT verbose_level), 
	void (*callback_subset_orbits)(isomorph *Iso, void *data, INT verbose_level), 
	void *callback_data, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph_arguments::init" << endl;
		}
	isomorph_arguments::A = A;
	isomorph_arguments::A2 = A2;
	isomorph_arguments::gen = gen;
	isomorph_arguments::target_size = target_size;
	isomorph_arguments::prefix_with_directory = prefix_with_directory;
	isomorph_arguments::ECA = ECA;
	isomorph_arguments::callback_report = callback_report;
	isomorph_arguments::callback_subset_orbits = callback_subset_orbits;
	isomorph_arguments::callback_data = callback_data;

	if (!ECA->f_has_solution_prefix) {
		cout << "isomorph_arguments::init please use -solution_prefix <solution_prefix>" << endl;
		exit(1);
		}
	if (!ECA->f_has_base_fname) {
		cout << "isomorph_arguments::init please use -base_fname <base_fname>" << endl;
		exit(1);
		}

	f_init_has_been_called = TRUE;

	if (f_v) {
		cout << "isomorph_arguments::init done" << endl;
		}
}
	
void isomorph_arguments::execute(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph_arguments::execute" << endl;
		}
	
	if (!f_init_has_been_called) {
		cout << "isomorph_arguments::execute please call init before execute" << endl;
		exit(1);
		}
	
	if (f_build_db) {

		cout << "isomorph_arguments::execute build_db" << endl;
		
		isomorph_build_db(A, A, gen, 
			target_size, prefix_with_directory, prefix_iso, 
			ECA->starter_size, verbose_level);
		}
	else if (f_read_solutions) {

		const BYTE *fname[1];
		BYTE fname1[1000];
		INT nb_files = 1;
		
		//sprintf(fname1, "%s%s_solutions_%ld_0_1.txt", ECA->solution_prefix, ECA->base_fname, ECA->starter_size);
		sprintf(fname1, "%s%s_depth_%ld_solutions.txt", ECA->solution_prefix, ECA->base_fname, ECA->starter_size);
		fname[0] = fname1;


		isomorph_read_solution_files(A, A2, gen, 
			target_size, prefix_with_directory, prefix_iso, ECA->starter_size, 
			fname, nb_files, 
			f_has_final_test_function, final_test_function, final_test_data, 
			verbose_level);

		}
	else if (f_read_solutions_from_clique_finder) {

		const BYTE *fname[1];
		BYTE fname1[1000];
		INT nb_files = 1;
		
		sprintf(fname1, "%s%s_solutions_%ld_0_1.txt", ECA->solution_prefix, ECA->base_fname, ECA->starter_size);
		//sprintf(fname1, "%s%s_depth_%ld_solutions.txt", ECA->solution_prefix, ECA->base_fname, ECA->starter_size);
		fname[0] = fname1;


		isomorph_read_solution_files_from_clique_finder(A, A2, gen, 
			target_size, prefix_with_directory, prefix_iso, ECA->starter_size, 
			fname, nb_files, verbose_level);
		}
	else if (f_read_solutions_from_clique_finder_list_of_cases) {


		cout << "f_read_solutions_from_clique_finder_list_of_cases" << endl;
		INT *list_of_cases;
		INT nb_cases;

		read_set_from_file(fname_list_of_cases, list_of_cases, nb_cases, verbose_level);
		cout << "nb_cases=" << nb_cases << endl;

		BYTE **fname;
		BYTE fname1[1000];
		INT i, c;
		
		fname = NEW_PBYTE(nb_cases);
		for (i = 0; i < nb_cases; i++) {
			c = list_of_cases[i];
			//cout << "read_solutions_from_clique_finder_list_of_cases case " << i << " / " << nb_cases << ":" << endl;
			sprintf(fname1, "%s%s_solutions_%ld.txt", ECA->solution_prefix, ECA->base_fname, c);

			fname[i] = NEW_BYTE(strlen(fname1) + 1);
			strcpy(fname[i], fname1);
			}

		isomorph_read_solution_files_from_clique_finder_case_by_case(A, A2, gen, 
			target_size, prefix_with_directory, prefix_iso, ECA->starter_size, 
			(const BYTE **) fname, list_of_cases, nb_cases, verbose_level);

		for (i = 0; i < nb_cases; i++) {
			FREE_BYTE(fname[i]);
			}
		FREE_PBYTE(fname);
		FREE_INT(list_of_cases);
		}


	else if (f_read_solutions_after_split) {


		BYTE **fname;
		BYTE fname1[1000];
		INT nb_files = 0;
		INT i;

		nb_files = read_solutions_split_m;
		fname = NEW_PBYTE(nb_files);
		for (i = 0; i < read_solutions_split_m; i++) {
			sprintf(fname1, "%s%s_solutions_%ld_%ld_%ld.txt", ECA->solution_prefix, ECA->base_fname, ECA->starter_size, i, read_solutions_split_m);
			fname[i] = NEW_BYTE(strlen(fname1) + 1);
			strcpy(fname[i], fname1);
			}
		cout << "Reading the following " << nb_files << " files:" << endl;
		for (i = 0; i < nb_files; i++) {
			cout << i << " : " << fname[i] << endl;
			}


		
		isomorph_read_solution_files_from_clique_finder(A, A2, gen, 
			target_size, prefix_with_directory, prefix_iso, ECA->starter_size, 
			(const BYTE **) fname, nb_files, verbose_level);
		}


	else if (f_read_statistics_after_split) {
		BYTE **fname;
		BYTE fname1[1000];
		INT nb_files = 0;
		INT i;


		cout << "f_read_statistics_after_split" << endl;
		
		nb_files = read_statistics_split_m;
		fname = NEW_PBYTE(nb_files);
		for (i = 0; i < read_statistics_split_m; i++) {
			sprintf(fname1, "%s%s_solutions_%ld_%ld_%ld_stats.txt", ECA->solution_prefix, ECA->base_fname, ECA->starter_size, i, read_solutions_split_m);
			fname[i] = NEW_BYTE(strlen(fname1) + 1);
			strcpy(fname[i], fname1);
			}
		cout << "Reading the following " << nb_files << " files:" << endl;
		for (i = 0; i < nb_files; i++) {
			cout << i << " : " << fname[i] << endl;
			}

		
		isomorph_read_statistic_files(A, A2, gen, 
			target_size, prefix_with_directory, prefix_iso, ECA->starter_size, 
			(const BYTE **) fname, nb_files, verbose_level);

		for (i = 0; i < nb_files; i++) {
			FREE_BYTE(fname[i]);
			}
		FREE_PBYTE(fname);
		}

	else if (f_compute_orbits) {


		isomorph_compute_orbits(A, A2, gen, 
			target_size, prefix_with_directory, prefix_iso, 
			ECA->starter_size, verbose_level);
		}
	else if (f_isomorph_testing) {


		isomorph_testing(A, A2, gen, 
			target_size, prefix_with_directory, prefix_iso, 
			ECA->starter_size, 
			f_event_file, event_file_name, print_mod, 
			verbose_level);
		}
	else if (f_classification_graph) {

		isomorph_classification_graph(A, A2, gen, 
			target_size, 
			prefix_with_directory, prefix_iso, 
			ECA->starter_size, 
			verbose_level);
		}
	else if (f_report) {

		if (callback_report == NULL) {
			cout << "isomorph_arguments::execute callback_report == NULL" << endl;
			exit(1);
			}
		isomorph_worker(A, A2, gen, 
			target_size, prefix_with_directory, prefix_iso, 
			callback_report, callback_data, 
			ECA->starter_size, verbose_level);
		}
	else if (f_subset_orbits) {

		isomorph_worker_data WD;

		WD.the_set = NULL;
		WD.set_size = 0;
		WD.callback_data = callback_data;
		
		if (f_subset_orbits_file) {
			read_set_from_file(subset_orbits_fname, WD.the_set, WD.set_size, verbose_level);
			}
		isomorph_worker(A, A2, gen, 
			target_size, prefix_with_directory, prefix_iso, 
			callback_subset_orbits, &WD, 
			ECA->starter_size, verbose_level);

		if (WD.the_set) {
			FREE_INT(WD.the_set);
			}
		}
	else if (f_down_orbits) {

		isomorph_compute_down_orbits(A, A2, gen, 
			target_size, 
			prefix_with_directory, prefix_iso, 
			callback_data, 
			ECA->starter_size, verbose_level);
		}


	if (f_v) {
		cout << "isomorph_arguments::execute done" << endl;
		}
}


