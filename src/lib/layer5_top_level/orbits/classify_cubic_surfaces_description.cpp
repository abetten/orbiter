/*
 * classify_cubic_surfaces_description.cpp
 *
 *  Created on: May 2, 2025
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orbits {


classify_cubic_surfaces_description::classify_cubic_surfaces_description()
{
	Record_birth();

	f_use_double_sixes = false;

	f_projective_space = false;
	//std::string projective_space_label;

	f_poset_classification_control = false;
	//std::string poset_classification_control_object;

}

classify_cubic_surfaces_description::~classify_cubic_surfaces_description()
{
	Record_death();
}

int classify_cubic_surfaces_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "classify_cubic_surfaces_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-use_double_sixes") == 0) {
			f_use_double_sixes = true;
			if (f_v) {
				cout << "-use_double_sixes " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-projective_space") == 0) {
			f_projective_space = true;
			projective_space_label.assign(argv[++i]);
			if (f_v) {
				cout << "-projective_space " << projective_space_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-poset_classification_control") == 0) {
			f_poset_classification_control = true;
			poset_classification_control_object.assign(argv[++i]);
			if (f_v) {
				cout << "-poset_classification_control " << poset_classification_control_object << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "classify_cubic_surfaces_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "classify_cubic_surfaces_description::read_arguments done" << endl;
	}
	return i + 1;
}


void classify_cubic_surfaces_description::print()
{
	if (f_use_double_sixes) {
		cout << "-use_double_sixes " << endl;
	}
	if (f_projective_space) {
		cout << "-projective_space " << projective_space_label << endl;
	}
	if (f_poset_classification_control) {
		cout << "-poset_classification_control " << poset_classification_control_object << endl;
	}
}


}}}

