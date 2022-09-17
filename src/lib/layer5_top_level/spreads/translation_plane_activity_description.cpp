/*
 * translation_plane_activity_description.cpp
 *
 *  Created on: Sep 16, 2022
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace spreads {

translation_plane_activity_description::translation_plane_activity_description()
{
	f_export_incma = FALSE;
	f_report = FALSE;
}

translation_plane_activity_description::~translation_plane_activity_description()
{
}

int translation_plane_activity_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "translation_plane_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-export_incma") == 0) {
			f_export_incma = TRUE;
			if (f_v) {
				cout << "-export_incma " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			if (f_v) {
				cout << "-report " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
	} // next i
	if (f_v) {
		cout << "translation_plane_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void translation_plane_activity_description::print()
{
	if (f_export_incma) {
		cout << "-export_incma " << endl;
	}
	if (f_report) {
		cout << "-report " << endl;
	}
}


}}}


