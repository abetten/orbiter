/*
 * ovoid_classify_description.cpp
 *
 *  Created on: Jul 28, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_geometry {


ovoid_classify_description::ovoid_classify_description()
{
	Record_birth();
	f_control = false;
	//std::string control_label;
	f_epsilon = false;
	epsilon = 0;
	f_d = false;
	d = 0;
}



ovoid_classify_description::~ovoid_classify_description()
{
	Record_death();
}

int ovoid_classify_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;


	if (f_v) {
		cout << "ovoid_classify_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-control") == 0) {
			f_control = true;
			control_label.assign(argv[++i]);
			if (f_v) {
				cout << "-control " << control_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-epsilon") == 0) {
			f_epsilon = true;
			epsilon = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-epsilon " << epsilon << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-d") == 0) {
			f_d = true;
			d = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-d " << d << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "ovoid_classify_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i

	if (f_v) {
		cout << "ovoid_classify_description::read_arguments done" << endl;
	}
	return i;
}

void ovoid_classify_description::print()
{
	if (f_control) {
		cout << "-control " << control_label << endl;
	}
	else if (f_epsilon) {
		cout << "-epsilon " << epsilon << endl;
	}
	else if (f_d) {
		cout << "-d " << d << endl;
	}
}




}}}



