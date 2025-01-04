/*
 * vector_ge_activity_description.cpp
 *
 *  Created on: Dec 24, 2024
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {





vector_ge_activity_description::vector_ge_activity_description()
{
	Record_birth();

	f_report = false;

	f_export_GAP = false;

	f_transform_variety = false;
	//std::string transform_variety_label;

	f_multiply = false;

	f_conjugate = false;

	f_conjugate_inverse = false;


}

vector_ge_activity_description::~vector_ge_activity_description()
{
	Record_death();
}


int vector_ge_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "vector_ge_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;
			if (f_v) {
				cout << "-report " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_GAP") == 0) {
			f_export_GAP = true;
			if (f_v) {
				cout << "-export_GAP " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-transform_variety") == 0) {
			f_transform_variety = true;
			transform_variety_label.assign(argv[++i]);
			if (f_v) {
				cout << "-transform_variety " << transform_variety_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-multiply") == 0) {
			f_multiply = true;
			if (f_v) {
				cout << "-multiply " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-conjugate") == 0) {
			f_conjugate = true;
			if (f_v) {
				cout << "-conjugate " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-conjugate_inverse") == 0) {
			f_conjugate_inverse = true;
			if (f_v) {
				cout << "-conjugate_inverse " << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "vector_ge_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "vector_ge_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void vector_ge_activity_description::print()
{

	if (f_report) {
		cout << "-report " << endl;
	}
	if (f_export_GAP) {
		cout << "-export_GAP " << endl;
	}
	if (f_transform_variety) {
		cout << "-transform_variety " << transform_variety_label << endl;
	}
	if (f_multiply) {
		cout << "-multiply " << endl;
	}
	if (f_conjugate) {
		cout << "-conjugate " << endl;
	}
	if (f_conjugate_inverse) {
		cout << "-conjugate_inverse " << endl;
	}

}





}}}

