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
	f_prefix_iso = false;
	//std::string prefix_iso;
	//prefix_iso = "./ISO/";


	f_prefix_with_directory = false;
	//std::string prefix_with_directory;

	f_prefix_classify = false;
	//std::string prefix_classify;

	f_solution_prefix = false;
	//std::string solution_prefix;

	f_base_fname = false;
	//std::string base_fname;

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



	//

	f_init_has_been_called = false;

	A = NULL;
	A2 = NULL;
	gen = NULL;
	target_size = 0;
	Control = NULL;

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

int isomorph_arguments::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int i;
	data_structures::string_tools ST;

	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-prefix_iso") == 0) {
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
		else if (ST.stringcmp(argv[i], "-use_database_for_starter") == 0) {
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


}}}


