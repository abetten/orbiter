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
	f_report = false;
	report_options = NULL;

	f_identify_Eckardt = false;

	f_identify_F13 = false;

	f_identify_Bes = false;

	f_identify_general_abcd = false;

	f_isomorphism_testing = false;
	//std::string isomorphism_testing_surface1_label;
	//std::string isomorphism_testing_surface2_label;

	f_recognize = false;
	//std::string recognize_surface_label;

	f_create_source_code = false;

	f_sweep_Cayley = false;

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

		cout << "classification_of_cubic_surfaces_with_double_sixes_activity_description::read_arguments, "
				"next argument is " << argv[i] << endl;

		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;
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
			f_identify_Eckardt = true;
			if (f_v) {
				cout << "-identify_Eckardt " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-identify_F13") == 0) {
			f_identify_F13 = true;
			if (f_v) {
				cout << "-identify_F13 " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-identify_Bes") == 0) {
			f_identify_Bes = true;
			if (f_v) {
				cout << "-identify_Bes " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-identify_general_abcd") == 0) {
			f_identify_general_abcd = true;
			if (f_v) {
				cout << "-identify_general_abcd " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-isomorphism_testing") == 0) {
			f_isomorphism_testing = true;
			isomorphism_testing_surface1_label.assign(argv[++i]);
			isomorphism_testing_surface2_label.assign(argv[++i]);
			if (f_v) {
				cout << "-isomorphism_testing "
						<< isomorphism_testing_surface1_label << " "
						<< isomorphism_testing_surface2_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-recognize") == 0) {
			f_recognize = true;
			recognize_surface_label.assign(argv[++i]);
			if (f_v) {
				cout << "-recognize "
						<< recognize_surface_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-create_source_code") == 0) {
			f_create_source_code = true;
			if (f_v) {
				cout << "-create_source_code " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-sweep_Cayley") == 0) {
			f_sweep_Cayley = true;
			if (f_v) {
				cout << "-sweep_Cayley " << endl;
			}
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
		cout << endl;
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
		cout << "-isomorphism_testing "
				<< isomorphism_testing_surface1_label << " "
				<< isomorphism_testing_surface2_label << endl;
	}
	if (f_recognize) {
		cout << "-recognize "
				<< recognize_surface_label << endl;
	}
	if (f_create_source_code) {
		cout << "-create_source_code " << endl;
	}
	if (f_sweep_Cayley) {
		cout << "-sweep_Cayley " << endl;
	}
}




}}}}


