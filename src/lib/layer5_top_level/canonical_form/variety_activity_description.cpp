/*
 * variety_activity_description.cpp
 *
 *  Created on: Jul 15, 2024
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;
using namespace orbiter::layer1_foundations;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {


variety_activity_description::variety_activity_description()
{

	f_compute_group = false;

	f_report = false;

	f_classify = false;

	f_apply_transformation = false;
	//std::string apply_transformation_group_element;

	f_singular_points = false;

}


variety_activity_description::~variety_activity_description()
{
}


int variety_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level > 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "variety_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-compute_group") == 0) {
			f_compute_group = true;
			if (f_v) {
				cout << "-compute_group " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;
			if (f_v) {
				cout << "-report " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-classify") == 0) {
			f_classify = true;
			if (f_v) {
				cout << "-classify " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-apply_transformation") == 0) {
			f_apply_transformation = true;
			apply_transformation_group_element.assign(argv[++i]);
			if (f_v) {
				cout << "-apply_transformation " << apply_transformation_group_element << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-singular_points") == 0) {
			f_singular_points = true;
			if (f_v) {
				cout << "-singular_points " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "variety_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i

	if (f_v) {
		cout << "variety_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}


void variety_activity_description::print()
{
	if (f_compute_group) {
		cout << "-compute_group " << endl;
	}
	if (f_report) {
		cout << "-report " << endl;
	}
	if (f_classify) {
		cout << "-classify " << endl;
	}
	if (f_apply_transformation) {
		cout << "-apply_transformation " << apply_transformation_group_element << endl;
	}
	if (f_singular_points) {
		cout << "-singular_points " << endl;
	}
}



}}}




