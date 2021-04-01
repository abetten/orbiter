/*
 * classification_of_cubic_surfaces_with_double_sixes_activity_description.cpp
 *
 *  Created on: Apr 1, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


classification_of_cubic_surfaces_with_double_sixes_activity_description::classification_of_cubic_surfaces_with_double_sixes_activity_description()
{
	f_report = FALSE;

	f_identify_HCV = FALSE;

	f_identify_F13 = FALSE;

	f_identify_Bes = FALSE;

	f_identify_general_abcd = FALSE;

	f_isomorphism_testing = FALSE;
	isomorphism_testing_surface1 = NULL;
	isomorphism_testing_surface2 = NULL;

	f_recognize = FALSE;
	recognize_surface = NULL;

}

classification_of_cubic_surfaces_with_double_sixes_activity_description::~classification_of_cubic_surfaces_with_double_sixes_activity_description()
{
}

int classification_of_cubic_surfaces_with_double_sixes_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "classification_of_cubic_surfaces_with_double_sixes_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		cout << "classification_of_cubic_surfaces_with_double_sixes_activity_description::read_arguments, next argument is " << argv[i] << endl;

		if (stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report " << endl;
		}
		else if (stringcmp(argv[i], "-identify_HCV") == 0) {
			f_identify_HCV = TRUE;
			cout << "-identify_HCV " << endl;
		}
		else if (stringcmp(argv[i], "-identify_F13") == 0) {
			f_identify_F13 = TRUE;
			cout << "-identify_F13 " << endl;
		}
		else if (stringcmp(argv[i], "-identify_Bes") == 0) {
			f_identify_Bes = TRUE;
			cout << "-identify_Bes " << endl;
		}
		else if (stringcmp(argv[i], "-identify_general_abcd") == 0) {
			f_identify_general_abcd = TRUE;
			cout << "-identify_general_abcd " << endl;
		}
		else if (stringcmp(argv[i], "-isomorphism_testing") == 0) {
			f_isomorphism_testing = TRUE;
			cout << "-isomorphism_testing" << endl;
			cout << "-isomorphism_testing reading description of first surface" << endl;
			isomorphism_testing_surface1 = NEW_OBJECT(surface_create_description);
			i += isomorphism_testing_surface1->
					read_arguments(argc - (i - 1), argv + i,
					verbose_level);
			cout << "-isomorphism_testing after reading description of first surface" << endl;
			cout << "the current argument is " << argv[i] << endl;
			cout << "-isomorphism_testing reading description of second surface" << endl;
			isomorphism_testing_surface2 = NEW_OBJECT(surface_create_description);
			i += isomorphism_testing_surface2->
					read_arguments(argc - (i - 1), argv + i,
					verbose_level);
			cout << "done with -isomorphism_testing" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-isomorphism_testing " << endl;
		}
		else if (stringcmp(argv[i], "-recognize") == 0) {
			f_recognize = TRUE;
			cout << "-recognize reading description of surface" << endl;
			recognize_surface = NEW_OBJECT(surface_create_description);
			i += recognize_surface->
					read_arguments(argc - (i - 1), argv + i,
					verbose_level);
			cout << "done with -surface_recognize" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-recognize " << endl;
		}
		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity_description::read_arguments -end" << endl;
			break;
		}

		else {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity_description::read_arguments looping, i=" << i << endl;
	} // next i
	cout << "classification_of_cubic_surfaces_with_double_sixes_activity_description::read_arguments done" << endl;
	return i + 1;
}



}}

