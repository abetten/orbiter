/*
 * orbits_activity_description.cpp
 *
 *  Created on: Nov 8, 2022
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;
using namespace orbiter::layer1_foundations;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


orbits_activity_description::orbits_activity_description()
{

	f_report = FALSE;

	f_export_something = FALSE;
	//std::string export_something_what;
	export_something_data1 = 0;

	f_export_trees = FALSE;

	f_draw_tree = FALSE;
	draw_tree_idx = 0;

	f_stabilizer = FALSE;
	stabilizer_point = 0;

	f_stabilizer_of_orbit_rep = FALSE;
	stabilizer_of_orbit_rep_orbit_idx = 0;

}


orbits_activity_description::~orbits_activity_description()
{
}


int orbits_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level > 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "orbits_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			if (f_v) {
				cout << "-report" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_trees") == 0) {
			f_export_trees = TRUE;
			if (f_v) {
				cout << "-export_trees" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_something") == 0) {
			f_export_something = TRUE;
			export_something_what.assign(argv[++i]);
			export_something_data1 = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-export_something " << export_something_what << " " << export_something_data1 << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-draw_tree") == 0) {
			f_draw_tree = TRUE;
			draw_tree_idx = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-export_trees" << draw_tree_idx << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-stabilizer") == 0) {
			f_stabilizer = TRUE;
			stabilizer_point = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-stabilizer " << stabilizer_point << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-stabilizer_of_orbit_rep") == 0) {
			f_stabilizer_of_orbit_rep = TRUE;
			stabilizer_of_orbit_rep_orbit_idx = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-stabilizer_of_orbit_rep " << stabilizer_of_orbit_rep_orbit_idx << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "orbits_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}


	} // next i

	if (f_v) {
		cout << "action_on_forms_description::read_arguments done" << endl;
	}
	return i + 1;
}


void orbits_activity_description::print()
{
	if (f_report) {
		cout << "-report" << endl;
	}
	if (f_export_trees) {
		cout << "-export_trees" << endl;
	}
	if (f_export_something) {
		cout << "-export_something " << export_something_what << " " << export_something_data1 << endl;
	}
	if (f_draw_tree) {
		cout << "-draw_tree " << draw_tree_idx << endl;
	}
	if (f_stabilizer) {
		cout << "-stabilizer " << stabilizer_point << endl;
	}
	if (f_stabilizer_of_orbit_rep) {
		cout << "-stabilizer_of_orbit_rep " << stabilizer_of_orbit_rep_orbit_idx << endl;
	}
}



}}}



