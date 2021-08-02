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

	f_conic_type = FALSE;
	conic_type_threshold = 0;

	f_non_conical_type = FALSE;
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
			cout << "-save " << endl;
		}
		else if (stringcmp(argv[i], "-conic_type") == 0) {
			f_conic_type = TRUE;
			conic_type_threshold = strtoi(argv[++i]);
			cout << "-conic_type " << conic_type_threshold << endl;
		}
		else if (stringcmp(argv[i], "-non_conical_type") == 0) {
			f_non_conical_type = TRUE;
			cout << "-non_conical_type " << endl;
		}
		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
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
	if (f_conic_type) {
		cout << "-conic_type " << conic_type_threshold << endl;
	}
	if (f_non_conical_type) {
		cout << "-f_non_conical_type" << endl;
	}
}



}}
