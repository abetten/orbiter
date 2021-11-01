/*
 * combinatorial_object_activity_description.cpp
 *
 *  Created on: Mar 20, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


combinatorial_object_activity_description::combinatorial_object_activity_description()
{
	f_save = FALSE;

	f_line_type = FALSE;

	f_conic_type = FALSE;
	conic_type_threshold = 0;

	f_non_conical_type = FALSE;

	f_ideal = FALSE;
	ideal_degree = 0;
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
		if (stringcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			if (f_v) {
				cout << "-save " << endl;
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
}



}}
