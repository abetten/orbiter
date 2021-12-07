/*
 * combinatorial_object_activity_description.cpp
 *
 *  Created on: Mar 20, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



combinatorial_object_activity_description::combinatorial_object_activity_description()
{
	f_save = FALSE;

	f_save_as = FALSE;
	//std::string save_as_fname;

	f_extract_subset = FALSE;
	//std::string extract_subset_set;
	//std::string extract_subset_fname;

	f_line_type = FALSE;

	f_conic_type = FALSE;
	conic_type_threshold = 0;

	f_non_conical_type = FALSE;

	f_ideal = FALSE;
	ideal_degree = 0;


	f_canonical_form_PG = FALSE;
	//std::string &canonical_form_PG_PG_label;
	Canonical_form_PG_Descr = NULL;

	f_canonical_form = FALSE;
	Canonical_form_Descr = NULL;

}

combinatorial_object_activity_description::~combinatorial_object_activity_description()
{
}


int combinatorial_object_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "combinatorial_object_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		// activities for COC:


		if (stringcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			if (f_v) {
				cout << "-save " << endl;
			}
		}
		else if (stringcmp(argv[i], "-save_as") == 0) {
			f_save_as = TRUE;
			save_as_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-save_as " << save_as_fname << endl;
			}
		}
		else if (stringcmp(argv[i], "-extract_subset") == 0) {
			f_extract_subset = TRUE;
			extract_subset_set.assign(argv[++i]);
			extract_subset_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-extract_subset " << extract_subset_set << " " << extract_subset_fname << endl;
			}
		}

		else if (stringcmp(argv[i], "-line_type") == 0) {
			f_line_type = TRUE;
			if (f_v) {
				cout << "-line_type " << endl;
			}
		}
		else if (stringcmp(argv[i], "-conic_type") == 0) {
			f_conic_type = TRUE;
			conic_type_threshold = strtoi(argv[++i]);
			if (f_v) {
				cout << "-conic_type " << conic_type_threshold << endl;
			}
		}
		else if (stringcmp(argv[i], "-non_conical_type") == 0) {
			f_non_conical_type = TRUE;
			if (f_v) {
				cout << "-non_conical_type " << endl;
			}
		}
		else if (stringcmp(argv[i], "-ideal") == 0) {
			f_ideal = TRUE;
			ideal_degree = strtoi(argv[++i]);
			if (f_v) {
				cout << "-ideal " << ideal_degree << endl;
			}
		}

		// activities for IS:

		else if (stringcmp(argv[i], "-canonical_form_PG") == 0) {
			f_canonical_form_PG = TRUE;
			if (f_v) {
				cout << "-canonical_form_PG, reading extra arguments" << endl;
			}

			canonical_form_PG_PG_label.assign(argv[++i]);

			Canonical_form_PG_Descr = NEW_OBJECT(projective_space_object_classifier_description);

			i += Canonical_form_PG_Descr->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
			if (f_v) {
				cout << "done reading -canonical_form_PG " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}

		else if (stringcmp(argv[i], "-canonical_form") == 0) {
			f_canonical_form = TRUE;
			if (f_v) {
				cout << "-canonical_form, reading extra arguments" << endl;
			}

			Canonical_form_Descr = NEW_OBJECT(projective_space_object_classifier_description);

			i += Canonical_form_Descr->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
			if (f_v) {
				cout << "done reading -canonical_form " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}



		else if (stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "combinatorial_object_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	if (f_v) {
		cout << "combinatorial_object_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void combinatorial_object_activity_description::print()
{
	if (f_save) {
		cout << "-save " << endl;
	}
	if (f_save_as) {
		cout << "-save_as " << save_as_fname << endl;
	}
	if (f_extract_subset) {
		cout << "-extract_subset " << extract_subset_set << " " << extract_subset_fname << endl;
	}
	if (f_line_type) {
		cout << "-line_type " << endl;
	}
	if (f_conic_type) {
		cout << "-conic_type " << conic_type_threshold << endl;
	}
	if (f_non_conical_type) {
		cout << "-f_non_conical_type" << endl;
	}
	if (f_ideal) {
		cout << "-ideal " << ideal_degree << endl;
	}
	if (f_canonical_form_PG) {
		cout << "-canonical_form_PG " << canonical_form_PG_PG_label << endl;
		Canonical_form_PG_Descr->print();
	}
	if (f_canonical_form) {
		cout << "-canonical_form " << endl;
		Canonical_form_Descr->print();
	}
}



}}
