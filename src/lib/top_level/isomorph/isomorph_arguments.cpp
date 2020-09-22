// isomorph_arguments.cpp
//
// Anton Betten
// January 27, 2016

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


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
	//fname_list_of_cases = NULL;
	f_read_solutions_after_split = FALSE;
	read_solutions_split_m = 0;
	
	f_read_statistics_after_split = FALSE;
	//read_statistics_split_m = 0;

	f_compute_orbits = FALSE;
	f_isomorph_testing = FALSE;
	f_classification_graph = FALSE;
	f_event_file = FALSE; // -e <event file> option
	event_file_name = NULL;
	print_mod = 500;
	f_isomorph_report = FALSE;
	f_subset_orbits = FALSE;
	f_subset_orbits_file = FALSE;
	f_down_orbits = FALSE;

	f_prefix_iso = FALSE;
	//prefix_iso = "./ISO/";

	f_has_final_test_function = FALSE;
	final_test_function = NULL;
	final_test_data = NULL;

	f_prefix_with_directory = FALSE;

}

void isomorph_arguments::freeself()
{
	null();
}

int isomorph_arguments::read_arguments(int argc, const char **argv,
	int verbose_level)
{
	int i;

	for (i = 0; i < argc; i++) {
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
			fname_list_of_cases.assign(argv[++i]);
			cout << "-read_solutions_from_clique_finder_list_of_cases " << fname_list_of_cases << endl;
		}
		else if (strcmp(argv[i], "-read_solutions_after_split") == 0) {
			f_read_solutions_after_split = TRUE;
			read_solutions_split_m = atoi(argv[++i]);
			cout << "-read_solutions_after_split " << read_solutions_split_m << endl;
		}
		else if (strcmp(argv[i], "-read_statistics_after_split") == 0) {
			f_read_statistics_after_split = TRUE;
			read_solutions_split_m = atoi(argv[++i]);
			cout << "-read_statistics_after_split " << read_solutions_split_m << endl;
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
		else if (strcmp(argv[i], "-isomorph_report") == 0) {
			f_isomorph_report = TRUE;
			cout << "-report " << endl;
		}
		else if (strcmp(argv[i], "-subset_orbits") == 0) {
			f_subset_orbits = TRUE;
			cout << "-subset_orbits " << endl;
		}
		else if (strcmp(argv[i], "-subset_orbits_file") == 0) {
			f_subset_orbits_file = TRUE;
			subset_orbits_fname.assign(argv[++i]);
			cout << "-subset_orbits_fname " << endl;
		}
		else if (strcmp(argv[i], "-down_orbits") == 0) {
			f_down_orbits = TRUE;
			cout << "-down_orbits " << endl;
		}
		else if (strcmp(argv[i], "-prefix_iso") == 0) {
			f_prefix_iso = TRUE;
			prefix_iso.assign(argv[++i]);
			cout << "-prefix_iso " << prefix_iso << endl;
		}
		else if (strcmp(argv[i], "-prefix_with_directory") == 0) {
			f_prefix_with_directory = TRUE;
			prefix_with_directory.assign(argv[++i]);
			cout << "-prefix_with_directory " << prefix_with_directory << endl;
		}
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "isomorph_arguments::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "isomorph_arguments::read_arguments done" << endl;
	return i + 1;
}


void isomorph_arguments::init(action *A, action *A2,
	poset_classification *gen,
	int target_size,
	poset_classification_control *Control,
	exact_cover_arguments *ECA,
	void (*callback_report)(isomorph *Iso, void *data, int verbose_level), 
	void (*callback_subset_orbits)(isomorph *Iso, void *data, int verbose_level), 
	void *callback_data, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph_arguments::init" << endl;
		}
	isomorph_arguments::A = A;
	isomorph_arguments::A2 = A2;
	isomorph_arguments::gen = gen;
	isomorph_arguments::target_size = target_size;
	isomorph_arguments::Control = Control;
	//sprintf(prefix_with_directory, "%s%s", Control->path, Control->problem_label);
	//isomorph_arguments::prefix_with_directory = prefix_with_directory;
	isomorph_arguments::ECA = ECA;
	isomorph_arguments::callback_report = callback_report;
	isomorph_arguments::callback_subset_orbits = callback_subset_orbits;
	isomorph_arguments::callback_data = callback_data;

	if (!ECA->f_has_solution_prefix) {
		cout << "isomorph_arguments::init please "
				"use -solution_prefix <solution_prefix>" << endl;
		exit(1);
		}
	if (!ECA->f_has_base_fname) {
		cout << "isomorph_arguments::init please "
				"use -base_fname <base_fname>" << endl;
		exit(1);
		}

	f_init_has_been_called = TRUE;

	if (f_v) {
		cout << "isomorph_arguments::init done" << endl;
		}
}
	
void isomorph_arguments::execute(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "isomorph_arguments::execute" << endl;
	}
	
	if (!f_init_has_been_called) {
		cout << "isomorph_arguments::execute please "
				"call init before execute" << endl;
		exit(1);
		}
	
	if (f_build_db) {

		if (f_v) {
			cout << "isomorph_arguments::execute build_db" << endl;
			cout << "isomorph_arguments::execute before isomorph_build_db" << endl;
		}
		isomorph_build_db(A, A, gen, 
			target_size, prefix_with_directory, prefix_iso, 
			ECA->starter_size, verbose_level);
		if (f_v) {
			cout << "isomorph_arguments::execute after isomorph_build_db" << endl;
		}
	}
	else if (f_read_solutions) {

		string fname;
		int nb_files = 1;
		char str[1000];
		
		fname.assign(ECA->solution_prefix);
		fname.append(ECA->base_fname);
		sprintf(str, "_depth_%d_solutions.txt", ECA->starter_size);


		if (f_v) {
			cout << "isomorph_arguments::execute before isomorph_read_solution_files" << endl;
		}
		isomorph_read_solution_files(A, A2, gen, 
			target_size, prefix_with_directory,
			prefix_iso, ECA->starter_size,
			&fname, nb_files,
			f_has_final_test_function,
			final_test_function, final_test_data,
			verbose_level);
		if (f_v) {
			cout << "isomorph_arguments::execute after isomorph_read_solution_files" << endl;
		}

	}
	else if (f_read_solutions_from_clique_finder) {

		string fname1;
		char str[1000];
		
		fname1.assign(ECA->solution_prefix);
		fname1.append(ECA->base_fname);
		sprintf(str, "_solutions_%d_0_1.txt", ECA->starter_size);
		fname1.append(str);


		if (f_v) {
			cout << "isomorph_arguments::execute before isomorph_read_solution_files_from_clique_finder" << endl;
		}
		isomorph_read_solution_files_from_clique_finder(
			A, A2, gen,
			target_size, prefix_with_directory,
			prefix_iso, ECA->starter_size,
			&fname1, 1 /*nb_files*/, verbose_level);
		if (f_v) {
			cout << "isomorph_arguments::execute after isomorph_read_solution_files_from_clique_finder" << endl;
		}
	}
	else if (f_read_solutions_from_clique_finder_list_of_cases) {


		if (f_v) {
			cout << "f_read_solutions_from_clique_finder_list_of_cases" << endl;
		}
		long int *list_of_cases;
		int nb_cases;

		if (f_v) {
			cout << "isomorph_arguments::execute before Fio.read_set_from_file" << endl;
		}
		Fio.read_set_from_file(fname_list_of_cases,
				list_of_cases, nb_cases, verbose_level);
		if (f_v) {
			cout << "nb_cases=" << nb_cases << endl;
		}

		string *fname;
		int i, c;
		
		fname = new string[nb_cases];
		for (i = 0; i < nb_cases; i++) {
			c = list_of_cases[i];

			char str[1000];
			sprintf(str, "_solutions_%d.txt", c);

			fname[i].assign(ECA->solution_prefix);
			fname[i].append(ECA->base_fname);
			fname[i].append(str);
			}

		if (f_v) {
			cout << "isomorph_arguments::execute before "
					"isomorph_read_solution_files_from_clique_finder_case_by_case" << endl;
		}
		isomorph_read_solution_files_from_clique_finder_case_by_case(A, A2, gen, 
			target_size, prefix_with_directory, prefix_iso, ECA->starter_size, 
			fname, list_of_cases, nb_cases, verbose_level);

		delete [] fname;
		FREE_lint(list_of_cases);
	}


	else if (f_read_solutions_after_split) {


		string *fname;
		int nb_files = 0;
		int i;

		nb_files = read_solutions_split_m;

		fname = new string[read_solutions_split_m];

		for (i = 0; i < read_solutions_split_m; i++) {


			char str[1000];
			sprintf(str, "_solutions_%d_%d_%d.txt", ECA->starter_size, i, read_solutions_split_m);

			fname[i].assign(ECA->solution_prefix);
			fname[i].append(ECA->base_fname);
			fname[i].append(str);

			}
		if (f_v) {
			cout << "Reading the following " << nb_files << " files:" << endl;
			for (i = 0; i < nb_files; i++) {
				cout << i << " : " << fname[i] << endl;
			}
		}


		
		if (f_v) {
			cout << "isomorph_arguments::execute before "
					"isomorph_read_solution_files_from_clique_finder" << endl;
		}


		isomorph_read_solution_files_from_clique_finder(A, A2, gen, 
			target_size, prefix_with_directory, prefix_iso, ECA->starter_size, 
			fname, nb_files, verbose_level);

		delete [] fname;
		}


	else if (f_read_statistics_after_split) {


		if (f_v) {
			cout << "f_read_statistics_after_split" << endl;
		}


		string *fname;
		int nb_files = 0;
		int i;

		nb_files = read_solutions_split_m;

		fname = new string[read_solutions_split_m];


		for (i = 0; i < read_solutions_split_m; i++) {


			char str[1000];
			sprintf(str, "_solutions_%d_%d_%d_stats.txt", ECA->starter_size, i, read_solutions_split_m);

			fname[i].assign(ECA->solution_prefix);
			fname[i].append(ECA->base_fname);
			fname[i].append(str);

			}
		if (f_v) {
			cout << "Reading the following " << nb_files << " files:" << endl;
			for (i = 0; i < nb_files; i++) {
				cout << i << " : " << fname[i] << endl;
			}
		}

		
		if (f_v) {
			cout << "isomorph_arguments::execute before "
					"isomorph_read_statistic_files" << endl;
		}
		isomorph_read_statistic_files(A, A2, gen, 
			target_size, prefix_with_directory,
			prefix_iso, ECA->starter_size,
			fname, nb_files, verbose_level);

		delete [] fname;

	}

	else if (f_compute_orbits) {


		if (f_v) {
			cout << "isomorph_arguments::execute before "
					"isomorph_compute_orbits" << endl;
		}
		isomorph_compute_orbits(A, A2, gen, 
			target_size, prefix_with_directory, prefix_iso, 
			ECA->starter_size, verbose_level);
		if (f_v) {
			cout << "isomorph_arguments::execute after "
					"isomorph_compute_orbits" << endl;
		}
	}
	else if (f_isomorph_testing) {


		if (f_v) {
			cout << "isomorph_arguments::execute before isomorph_testing" << endl;
		}
		isomorph_testing(A, A2, gen, 
			target_size, prefix_with_directory, prefix_iso, 
			ECA->starter_size, 
			f_event_file, event_file_name, print_mod, 
			verbose_level);
		if (f_v) {
			cout << "isomorph_arguments::execute after isomorph_testing" << endl;
		}
	}
	else if (f_classification_graph) {

		if (f_v) {
			cout << "isomorph_arguments::execute before "
					"isomorph_classification_graph" << endl;
		}
		isomorph_classification_graph(A, A2, gen, 
			target_size, 
			prefix_with_directory, prefix_iso, 
			ECA->starter_size, 
			verbose_level);
		if (f_v) {
			cout << "isomorph_arguments::execute after "
					"isomorph_classification_graph" << endl;
		}
	}
	else if (f_isomorph_report) {

		if (callback_report == NULL) {
			cout << "isomorph_arguments::execute "
					"callback_report == NULL" << endl;
			exit(1);
			}
		if (f_v) {
			cout << "isomorph_arguments::execute before isomorph_worker" << endl;
		}
		isomorph_worker(A, A2, gen, 
			target_size, prefix_with_directory, prefix_iso, 
			callback_report, callback_data, 
			ECA->starter_size, verbose_level);
		if (f_v) {
			cout << "isomorph_arguments::execute after isomorph_worker" << endl;
		}
	}
	else if (f_subset_orbits) {

		isomorph_worker_data WD;

		WD.the_set = NULL;
		WD.set_size = 0;
		WD.callback_data = callback_data;
		
		if (f_subset_orbits_file) {
			Fio.read_set_from_file(subset_orbits_fname,
					WD.the_set, WD.set_size, verbose_level);
			}
		if (f_v) {
			cout << "isomorph_arguments::execute before isomorph_worker" << endl;
		}
		isomorph_worker(A, A2, gen, 
			target_size, prefix_with_directory, prefix_iso, 
			callback_subset_orbits, &WD, 
			ECA->starter_size, verbose_level);
		if (f_v) {
			cout << "isomorph_arguments::execute after isomorph_worker" << endl;
		}

		if (WD.the_set) {
			FREE_lint(WD.the_set);
		}
	}
	else if (f_down_orbits) {

		if (f_v) {
			cout << "isomorph_arguments::execute before isomorph_compute_down_orbits" << endl;
		}
		isomorph_compute_down_orbits(A, A2, gen, 
			target_size, 
			prefix_with_directory, prefix_iso, 
			callback_data, 
			ECA->starter_size, verbose_level);
		if (f_v) {
			cout << "isomorph_arguments::execute after isomorph_compute_down_orbits" << endl;
		}
	}


	if (f_v) {
		cout << "isomorph_arguments::execute done" << endl;
	}
}

}}

