/*
 * cubic_surface_activity_description.cpp
 *
 *  Created on: Mar 18, 2021
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_in_general {


cubic_surface_activity_description::cubic_surface_activity_description()
{
	f_report = FALSE;

	f_export_something = FALSE;
	//std::string export_something_what;

	f_export_gap = FALSE;

	f_all_quartic_curves = FALSE;

	f_export_all_quartic_curves = FALSE;

}

cubic_surface_activity_description::~cubic_surface_activity_description()
{

}


int cubic_surface_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "cubic_surface_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			if (f_v) {
				cout << "-report " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_something") == 0) {
			f_export_something = TRUE;
			export_something_what.assign(argv[++i]);
			if (f_v) {
				cout << "-export_something " << export_something_what << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_gap") == 0) {
			f_export_gap = TRUE;
			if (f_v) {
				cout << "-export_gap " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-all_quartic_curves") == 0) {
			f_all_quartic_curves = TRUE;
			if (f_v) {
				cout << "-all_quartic_curves " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_all_quartic_curves") == 0) {
			f_export_all_quartic_curves = TRUE;
			if (f_v) {
				cout << "-export_all_quartic_curves " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "cubic_surface_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		if (f_v) {
			cout << "cubic_surface_activity_description::read_arguments looping, i=" << i << endl;
		}
	} // next i

	if (f_v) {
		cout << "cubic_surface_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void cubic_surface_activity_description::print()
{
	if (f_report) {
		cout << "-report " << endl;
	}
	if (f_export_something) {
		cout << "-export_something " << export_something_what << endl;
	}
	if (f_export_gap) {
		cout << "-export_gap " << endl;
	}
	if (f_all_quartic_curves) {
		cout << "-all_quartic_curves " << endl;
	}
	if (f_export_all_quartic_curves) {
		cout << "-export_all_quartic_curves " << endl;
	}
}



}}}}


