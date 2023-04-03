// isomorph_arguments.cpp
//
// Anton Betten
// January 27, 2016

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace isomorph {


isomorph_arguments::isomorph_arguments()
{
	f_init_has_been_called = false;

	f_use_database_for_starter = false;
	f_implicit_fusion = false;

	f_build_db = false;

	f_read_solutions = false;

	f_list_of_cases = false;
	//std::string list_of_cases_fname;

	//f_read_solutions_from_clique_finder = false;
	//f_read_solutions_from_clique_finder_list_of_cases = false;
	//fname_list_of_cases = NULL;
	f_read_solutions_after_split = false;
	read_solutions_split_m = 0;
	
	f_read_statistics_after_split = false;

	f_recognize = false;
	//std::string recognize_label;

	f_compute_orbits = false;
	f_isomorph_testing = false;
	f_classification_graph = false;
	f_event_file = false; // -e <event file> option
	//event_file_name;
	print_mod = 500;

	f_isomorph_report = false;

	f_export_source_code = false;

	f_subset_orbits = false;
	f_subset_orbits_file = false;
	//std::string subset_orbits_fname;
	f_eliminate_graphs_if_possible = false;
	f_down_orbits = false;

	f_prefix_iso = false;
	//std::string prefix_iso;
	//prefix_iso = "./ISO/";


	A = NULL;
	A2 = NULL;
	gen = NULL;
	target_size = 0;
	Control = NULL;

	f_prefix_with_directory = false;
	//std::string prefix_with_directory;

	f_prefix_classify = false;
	//std::string prefix_classify;

	f_solution_prefix = false;
	//std::string solution_prefix;

	f_base_fname = false;
	//std::string base_fname;

	ECA = NULL;

	callback_report = NULL;
	callback_subset_orbits = NULL;
	callback_data = NULL;

	f_has_final_test_function = false;
	final_test_function = NULL;
	final_test_data = NULL;

}

isomorph_arguments::~isomorph_arguments()
{
}

int isomorph_arguments::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int i;
	data_structures::string_tools ST;

	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-use_database_for_starter") == 0) {
			f_use_database_for_starter = true;
			cout << "-use_database_for_starter " << endl;
		}
		else if (ST.stringcmp(argv[i], "-implicit_fusion") == 0) {
			f_implicit_fusion = true;
			cout << "-implicit_fusion " << endl;
		}
		else if (ST.stringcmp(argv[i], "-build_db") == 0) {
			f_build_db = true;
			cout << "-build_db " << endl;
		}
		else if (ST.stringcmp(argv[i], "-read_solutions") == 0) {
			f_read_solutions = true;
			cout << "-read_solutions " << endl;
		}
		else if (ST.stringcmp(argv[i], "-list_of_cases") == 0) {
			f_list_of_cases = true;
			list_of_cases_fname.assign(argv[++i]);
			cout << "-list_of_cases " << list_of_cases_fname << endl;
		}


		else if (ST.stringcmp(argv[i], "-read_solutions_after_split") == 0) {
			f_read_solutions_after_split = true;
			read_solutions_split_m = ST.strtoi(argv[++i]);
			cout << "-read_solutions_after_split " << read_solutions_split_m << endl;
		}
		else if (ST.stringcmp(argv[i], "-read_statistics_after_split") == 0) {
			f_read_statistics_after_split = true;
			read_solutions_split_m = ST.strtoi(argv[++i]);
			cout << "-read_statistics_after_split " << read_solutions_split_m << endl;
		}
		else if (ST.stringcmp(argv[i], "-recognize") == 0) {
			f_recognize = true;
			recognize_label.assign(argv[++i]);
			cout << "-recognize " << recognize_label << endl;
		}

		else if (ST.stringcmp(argv[i], "-compute_orbits") == 0) {
			f_compute_orbits = true;
			cout << "-compute_orbits " << endl;
		}
		else if (ST.stringcmp(argv[i], "-isomorph_testing") == 0) {
			f_isomorph_testing = true;
			cout << "-isomorph_testing " << endl;
		}
		else if (ST.stringcmp(argv[i], "-classification_graph") == 0) {
			f_classification_graph = true;
			cout << "-classification_graph " << endl;
		}
		else if (ST.stringcmp(argv[i], "-e") == 0) {
			i++;
			f_event_file = true;
			event_file_name.assign(argv[i]);
			cout << "-e " << event_file_name << endl;
		}
		else if (ST.stringcmp(argv[i], "-print_interval") == 0) {
			print_mod = ST.strtoi(argv[++i]);
			cout << "-print_interval " << print_mod << endl;
		}
		else if (ST.stringcmp(argv[i], "-isomorph_report") == 0) {
			f_isomorph_report = true;
			cout << "-isomorph_report " << endl;
		}
		else if (ST.stringcmp(argv[i], "-export_source_code") == 0) {
			f_export_source_code = true;
			cout << "-export_source_code " << endl;
		}
		else if (ST.stringcmp(argv[i], "-subset_orbits") == 0) {
			f_subset_orbits = true;
			cout << "-subset_orbits " << endl;
		}
		else if (ST.stringcmp(argv[i], "-subset_orbits_file") == 0) {
			f_subset_orbits_file = true;
			subset_orbits_fname.assign(argv[++i]);
			cout << "-subset_orbits_fname " << endl;
		}
		else if (ST.stringcmp(argv[i], "-down_orbits") == 0) {
			f_down_orbits = true;
			cout << "-down_orbits " << endl;
		}
		else if (ST.stringcmp(argv[i], "-prefix_iso") == 0) {
			f_prefix_iso = true;
			prefix_iso.assign(argv[++i]);
			cout << "-prefix_iso " << prefix_iso << endl;
		}
		else if (ST.stringcmp(argv[i], "-prefix_with_directory") == 0) {
			f_prefix_with_directory = true;
			prefix_with_directory.assign(argv[++i]);
			cout << "-prefix_with_directory " << prefix_with_directory << endl;
		}
		else if (ST.stringcmp(argv[i], "-prefix_classify") == 0) {
			f_prefix_classify = true;
			prefix_classify.assign(argv[++i]);
			cout << "-prefix_classify " << prefix_classify << endl;
		}
		else if (ST.stringcmp(argv[i], "-solution_prefix") == 0) {
			f_solution_prefix = true;
			solution_prefix.assign(argv[++i]);
			cout << "-solution_prefix " << solution_prefix << endl;
		}
		else if (ST.stringcmp(argv[i], "-base_fname") == 0) {
			f_base_fname = true;
			base_fname.assign(argv[++i]);
			cout << "-base_fname " << base_fname << endl;
		}


		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "isomorph_arguments::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "isomorph_arguments::read_arguments done" << endl;
	return i + 1;
}

void isomorph_arguments::print()
{
	if (f_use_database_for_starter) {
		cout << "-use_database_for_starter " << endl;
	}
	if (f_implicit_fusion) {
		cout << "-implicit_fusion " << endl;
	}
	if (f_build_db) {
		cout << "-build_db " << endl;
	}
	if (f_read_solutions) {
		cout << "-read_solutions " << endl;
	}
	if (f_list_of_cases) {
		cout << "-list_of_cases " << list_of_cases_fname << endl;
	}
	if (f_read_solutions_after_split) {
		cout << "-read_solutions_after_split " << read_solutions_split_m << endl;
	}
	if (f_read_statistics_after_split) {
		cout << "-read_statistics_after_split " << read_solutions_split_m << endl;
	}
	if (f_recognize) {
		cout << "-recognize " << recognize_label << endl;
	}

	if (f_compute_orbits) {
		cout << "-compute_orbits " << endl;
	}
	if (f_isomorph_testing) {
		cout << "-isomorph_testing " << endl;
	}
	if (f_classification_graph) {
		cout << "-classification_graph " << endl;
	}
	if (f_event_file) {
		cout << "-e " << event_file_name << endl;
	}
	if (print_mod) {
		cout << "-print_interval " << print_mod << endl;
	}
	if (f_isomorph_report) {
		cout << "-isomorph_report " << endl;
	}
	if (f_export_source_code) {
		cout << "-export_source_code " << endl;
	}
	if (f_subset_orbits) {
		cout << "-subset_orbits " << endl;
	}
	if (f_subset_orbits_file) {
		cout << "-subset_orbits_fname " << endl;
	}
	if (f_down_orbits) {
		cout << "-down_orbits " << endl;
	}
	if (f_prefix_iso) {
		cout << "-prefix_iso " << prefix_iso << endl;
	}
	if (f_prefix_with_directory) {
		cout << "-prefix_with_directory " << prefix_with_directory << endl;
	}
	if (f_prefix_classify) {
		cout << "-prefix_classify " << prefix_classify << endl;
	}
	if (f_solution_prefix) {
		cout << "-solution_prefix " << solution_prefix << endl;
	}
	if (f_base_fname) {
		cout << "-base_fname " << base_fname << endl;
	}

}

void isomorph_arguments::init(
		actions::action *A,
		actions::action *A2,
		poset_classification::poset_classification *gen,
	int target_size,
	poset_classification::poset_classification_control *Control,
	solvers_package::exact_cover_arguments *ECA,
	void (*callback_report)(
			isomorph *Iso, void *data, int verbose_level),
	void (*callback_subset_orbits)(
			isomorph *Iso, void *data, int verbose_level),
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
	//isomorph_arguments::prefix_with_directory = prefix_with_directory;
	isomorph_arguments::ECA = ECA;
	isomorph_arguments::callback_report = callback_report;
	isomorph_arguments::callback_subset_orbits = callback_subset_orbits;
	isomorph_arguments::callback_data = callback_data;

	if (!f_solution_prefix) {
		cout << "isomorph_arguments::init please "
				"use -solution_prefix <solution_prefix>" << endl;
		exit(1);
	}
	if (!f_base_fname) {
		cout << "isomorph_arguments::init please "
				"use -base_fname <base_fname>" << endl;
		exit(1);
	}

	f_init_has_been_called = true;

	if (f_v) {
		cout << "isomorph_arguments::init done" << endl;
	}
}

#if 0
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
		snprintf(str, sizeof(str), "_depth_%d_solutions.txt", ECA->starter_size);


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
		snprintf(str, sizeof(str), "_solutions_%d_0_1.txt", ECA->starter_size);
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
			snprintf(str, sizeof(str), "_solutions_%d.txt", c);

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
			snprintf(str, sizeof(str), "_solutions_%d_%d_%d.txt", ECA->starter_size, i, read_solutions_split_m);

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
			snprintf(str, sizeof(str), "_solutions_%d_%d_%d_stats.txt", ECA->starter_size, i, read_solutions_split_m);

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
#endif

}}}


