/*
 * blt_set_activity_description.cpp
 *
 *  Created on: Jan 13, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {

blt_set_activity_description::blt_set_activity_description()
{
	f_report = false;

	f_export_gap = false;

	f_create_flock = false;
	create_flock_point_idx = 0;

	f_BLT_test = false;

	f_export_set_in_PG = false;

	f_plane_invariant = false;

}

blt_set_activity_description::~blt_set_activity_description()
{
}

int blt_set_activity_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "blt_set_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;
			if (f_v) {
				cout << "-report " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_gap") == 0) {
			f_export_gap = true;
			if (f_v) {
				cout << "-export_gap " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-create_flock") == 0) {
			f_create_flock = true;
			create_flock_point_idx = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-create_flock " << create_flock_point_idx << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-BLT_test") == 0) {
			f_BLT_test = true;
			if (f_v) {
				cout << "-BLT_test " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_set_in_PG") == 0) {
			f_export_set_in_PG = true;
			if (f_v) {
				cout << "-export_set_in_PG " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-plane_invariant") == 0) {
			f_plane_invariant = true;
			if (f_v) {
				cout << "-plane_invariant " << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "blt_set_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "blt_set_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void blt_set_activity_description::print()
{
	if (f_report) {
		cout << "-report " << endl;
	}
	if (f_export_gap) {
		cout << "-export_gap " << endl;
	}
	if (f_create_flock) {
		cout << "-create_flock " << create_flock_point_idx << endl;
	}
	if (f_BLT_test) {
		cout << "-BLT_test " << endl;
	}
	if (f_export_set_in_PG) {
		cout << "-export_set_in_PG " << endl;
	}
	if (f_plane_invariant) {
		cout << "-plane_invariant " << endl;
	}
}


}}}


