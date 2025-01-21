/*
 * spread_classify_activity_description.cpp
 *
 *  Created on: Aug 31, 2022
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace spreads {

spread_classify_activity_description::spread_classify_activity_description()
{
	Record_birth();
	f_compute_starter = false;

	f_prepare_lifting_single_case = false;
	prepare_lifting_single_case_case_number = 0;

	f_prepare_lifting_all_cases = false;

	f_split = false;
	split_r = 0;
	split_m = 1;

	f_isomorph = false;
	//std::string isomorph_label;

	f_build_db = false;
	f_read_solutions = false;
	f_compute_orbits = false;
	f_isomorph_testing = false;
	f_isomorph_report = false;

}

spread_classify_activity_description::~spread_classify_activity_description()
{
	Record_death();
}

int spread_classify_activity_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "spread_classify_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-compute_starter") == 0) {
			f_compute_starter = true;
			if (f_v) {
				cout << "-compute_starter " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-prepare_lifting_single_case") == 0) {
			f_prepare_lifting_single_case = true;
			prepare_lifting_single_case_case_number = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-prepare_lifting_single_case " << prepare_lifting_single_case_case_number << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-prepare_lifting_all_cases") == 0) {
			f_prepare_lifting_all_cases = true;
			if (f_v) {
				cout << "-prepare_lifting_all_cases " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-split") == 0) {
			f_split = true;
			split_r = ST.strtoi(argv[++i]);
			split_m = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-split " << split_r << " " << split_m << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-isomorph") == 0) {
			f_isomorph = true;
			isomorph_label.assign(argv[++i]);
			if (f_v) {
				cout << "-isomorph " << isomorph_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-build_db") == 0) {
			f_build_db = true;
			if (f_v) {
				cout << "-build_db " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-read_solutions") == 0) {
			f_read_solutions = true;
			if (f_v) {
				cout << "-read_solutions " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-compute_orbits") == 0) {
			f_compute_orbits = true;
			if (f_v) {
				cout << "-compute_orbits " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-isomorph_testing") == 0) {
			f_isomorph_testing = true;
			if (f_v) {
				cout << "-isomorph_testing " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-isomorph_report") == 0) {
			f_isomorph_report = true;
			if (f_v) {
				cout << "-isomorph_report " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "spread_classify_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "spread_classify_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void spread_classify_activity_description::print()
{
	if (f_compute_starter) {
		cout << "-compute_starter " << endl;
	}
	if (f_prepare_lifting_single_case) {
		cout << "-prepare_lifting_single_case " << prepare_lifting_single_case_case_number << endl;
	}
	if (f_prepare_lifting_all_cases) {
		cout << "-prepare_lifting_all_cases " << endl;
	}
	if (f_split) {
		cout << "-split " << split_r << " " << split_m << endl;
	}
	if (f_isomorph) {
		cout << "-isomorph " << isomorph_label << endl;
	}
	if (f_build_db) {
		cout << "-build_db " << endl;
	}
	if (f_read_solutions) {
		cout << "-read_solutions " << endl;
	}
	if (f_compute_orbits) {
		cout << "-compute_orbits " << endl;
	}
	if (f_isomorph_testing) {
		cout << "-isomorph_testing " << endl;
	}
	if (f_isomorph_report) {
		cout << "-isomorph_report " << endl;
	}
}

}}}

