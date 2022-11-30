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
	f_control = FALSE;
	//std::string control_label;
	f_epsilon = FALSE;
	epsilon = 0;
	f_d = FALSE;
	d = 0;
}



ovoid_classify_description::~ovoid_classify_description()
{
}

int ovoid_classify_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;


	if (f_v) {
		cout << "ovoid_classify_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-control") == 0) {
			f_control = TRUE;
			control_label.assign(argv[++i]);
			if (f_v) {
				cout << "-control " << control_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-epsilon") == 0) {
			f_epsilon = TRUE;
			epsilon = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-epsilon " << epsilon << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-d") == 0) {
			f_d = TRUE;
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



