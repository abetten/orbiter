/*
 * cubic_surface_activity_description.cpp
 *
 *  Created on: Mar 18, 2021
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {

cubic_surface_activity_description::cubic_surface_activity_description()
{
	f_report = FALSE;

	f_export_points = FALSE;

	f_clebsch = FALSE;

	f_codes = FALSE;

	f_all_quartic_curves = FALSE;

}

cubic_surface_activity_description::~cubic_surface_activity_description()
{

}


int cubic_surface_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "cubic_surface_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report " << endl;
		}
		else if (stringcmp(argv[i], "-export_points") == 0) {
			f_export_points = TRUE;
			cout << "-export_points " << endl;
		}
		else if (stringcmp(argv[i], "-clebsch") == 0) {
			f_clebsch = TRUE;
			cout << "-clebsch " << endl;
		}
		else if (stringcmp(argv[i], "-codes") == 0) {
			f_codes = TRUE;
			cout << "-codes " << endl;
		}
		else if (stringcmp(argv[i], "-all_quartic_curves") == 0) {
			f_all_quartic_curves = TRUE;
			cout << "-all_quartic_curves " << endl;
		}

		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "cubic_surface_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		cout << "cubic_surface_activity_description::read_arguments looping, i=" << i << endl;
	} // next i

	cout << "cubic_surface_activity_description::read_arguments done" << endl;
	return i + 1;
}



}}

