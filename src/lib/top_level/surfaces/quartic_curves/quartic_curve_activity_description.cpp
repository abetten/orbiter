/*
 * quartic_curve_activity_description.cpp
 *
 *  Created on: May 21, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {
namespace applications_in_algebraic_geometry {



quartic_curve_activity_description::quartic_curve_activity_description()
{

	f_report = FALSE;

	f_report_with_group = FALSE;

	f_export_points = FALSE;

	f_create_surface = FALSE;


}

quartic_curve_activity_description::~quartic_curve_activity_description()
{
}

int quartic_curve_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;
	data_structures::string_tools ST;

	cout << "quartic_curve_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report " << endl;
		}
		else if (ST.stringcmp(argv[i], "-report_with_group") == 0) {
			f_report_with_group = TRUE;
			cout << "-report_with_group " << endl;
		}
		else if (ST.stringcmp(argv[i], "-export_points") == 0) {
			f_export_points = TRUE;
			cout << "-export_points " << endl;
		}
		else if (ST.stringcmp(argv[i], "-create_surface") == 0) {
			f_create_surface = TRUE;
			cout << "-create_surface " << endl;
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "quartic_curve_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		cout << "quartic_curve_activity_description::read_arguments looping, i=" << i << endl;
	} // next i

	cout << "quartic_curve_activity_description::read_arguments done" << endl;
	return i + 1;
}

void quartic_curve_activity_description::print()
{
	if (f_report) {
		cout << "-report " << endl;
	}
	if (f_report_with_group) {
		cout << "-report_with_group " << endl;
	}
	if (f_export_points) {
		cout << "-export_points " << endl;
	}
	if (f_create_surface) {
		cout << "-create_surface " << endl;
	}
}





}}}

