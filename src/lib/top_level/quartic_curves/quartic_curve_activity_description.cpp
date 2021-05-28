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



quartic_curve_activity_description::quartic_curve_activity_description()
{

	f_report = FALSE;

	f_report_with_group = FALSE;

	f_export_points = FALSE;


}

quartic_curve_activity_description::~quartic_curve_activity_description()
{
}

int quartic_curve_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "quartic_curve_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report " << endl;
		}
		else if (stringcmp(argv[i], "-report_with_group") == 0) {
			f_report_with_group = TRUE;
			cout << "-report_with_group " << endl;
		}
		else if (stringcmp(argv[i], "-export_points") == 0) {
			f_export_points = TRUE;
			cout << "-export_points " << endl;
		}
		else if (stringcmp(argv[i], "-end") == 0) {
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




}}
