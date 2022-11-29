/*
 * classification_of_cubic_surfaces_with_double_sixes_activity_description.cpp
 *
 *  Created on: Apr 1, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_double_sixes {



classification_of_cubic_surfaces_with_double_sixes_activity_description::classification_of_cubic_surfaces_with_double_sixes_activity_description()
{
	f_report = FALSE;
	report_options = NULL;

	f_identify_Eckardt = FALSE;

	f_identify_F13 = FALSE;

	f_identify_Bes = FALSE;

	f_identify_general_abcd = FALSE;

	f_isomorphism_testing = FALSE;
	isomorphism_testing_surface1 = NULL;
	isomorphism_testing_surface2 = NULL;

	f_recognize = FALSE;
	recognize_surface = NULL;

	f_create_source_code = FALSE;

	f_sweep_Cayley = FALSE;

}

classification_of_cubic_surfaces_with_double_sixes_activity_description::~classification_of_cubic_surfaces_with_double_sixes_activity_description()
{
}

int classification_of_cubic_surfaces_with_double_sixes_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		cout << "classification_of_cubic_surfaces_with_double_sixes_activity_description::read_arguments, next argument is " << argv[i] << endl;

		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			report_options = NEW_OBJECT(poset_classification::poset_classification_report_options);
			if (f_v) {
				cout << "-report " << endl;
			}
			i += report_options->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -report " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}

			if (f_v) {
				cout << "-report" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-identify_Eckardt") == 0) {
			f_identify_Eckardt = TRUE;
			if (f_v) {
				cout << "-identify_Eckardt " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-identify_F13") == 0) {
			f_identify_F13 = TRUE;
			if (f_v) {
				cout << "-identify_F13 " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-identify_Bes") == 0) {
			f_identify_Bes = TRUE;
			if (f_v) {
				cout << "-identify_Bes " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-identify_general_abcd") == 0) {
			f_identify_general_abcd = TRUE;
			if (f_v) {
				cout << "-identify_general_abcd " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-isomorphism_testing") == 0) {
			f_isomorphism_testing = TRUE;
			if (f_v) {
				cout << "-isomorphism_testing" << endl;
				cout << "-isomorphism_testing reading description of first surface" << endl;
			}
			isomorphism_testing_surface1 = NEW_OBJECT(cubic_surfaces_in_general::surface_create_description);
			i += isomorphism_testing_surface1->
					read_arguments(argc - (i + 1), argv + i + 1,
					verbose_level);
			if (f_v) {
				cout << "-isomorphism_testing after reading description of first surface" << endl;
				cout << "the current argument is " << argv[i] << endl;
				cout << "-isomorphism_testing reading description of second surface" << endl;
			}
			isomorphism_testing_surface2 = NEW_OBJECT(cubic_surfaces_in_general::surface_create_description);
			i += isomorphism_testing_surface2->
					read_arguments(argc - (i + 1), argv + i + 1,
					verbose_level);
			if (f_v) {
				cout << "done with -isomorphism_testing" << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-isomorphism_testing " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-recognize") == 0) {
			f_recognize = TRUE;
			if (f_v) {
				cout << "-recognize reading description of surface" << endl;
			}
			recognize_surface = NEW_OBJECT(cubic_surfaces_in_general::surface_create_description);
			i += recognize_surface->
					read_arguments(argc - (i + 1), argv + i + 1,
					verbose_level);
			if (f_v) {
				cout << "done with -surface_recognize" << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-recognize " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-create_source_code") == 0) {
			f_create_source_code = TRUE;
		}
		else if (ST.stringcmp(argv[i], "-sweep_Cayley") == 0) {
			f_sweep_Cayley = TRUE;
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "classification_of_cubic_surfaces_with_double_sixes_activity_description::read_arguments -end" << endl;
			}
			break;
		}

		else {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity_description::read_arguments "
					"looping, i=" << i << endl;
		}
	} // next i
	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void classification_of_cubic_surfaces_with_double_sixes_activity_description::print()
{
	if (f_report) {
		cout << "-report ";
		report_options->print();
	}
	if (f_identify_Eckardt) {
		cout << "-identify_Eckardt " << endl;
	}
	if (f_identify_F13) {
		cout << "-identify_F13 " << endl;
	}
	if (f_identify_Bes) {
		cout << "-identify_Bes " << endl;
	}
	if (f_identify_general_abcd) {
		cout << "-identify_general_abcd " << endl;
	}
	if (f_isomorphism_testing) {
		cout << "-isomorphism_testing " << endl;
	}
	if (f_recognize) {
		cout << "-recognize " << endl;
	}
	if (f_create_source_code) {
		cout << "-create_source_code " << endl;
	}
	if (f_sweep_Cayley) {
		cout << "-sweep_Cayley " << endl;
	}
}




}}}}


