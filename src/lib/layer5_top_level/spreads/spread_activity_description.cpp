/*
 * spread_activity_description.cpp
 *
 *  Created on: Sep 19, 2022
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace spreads {

spread_activity_description::spread_activity_description()
{
	f_report = FALSE;
}

spread_activity_description::~spread_activity_description()
{
}

int spread_activity_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "spread_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			if (f_v) {
				cout << "-report " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "spread_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "spread_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void spread_activity_description::print()
{
	if (f_report) {
		cout << "-report " << endl;
	}
}


}}}


