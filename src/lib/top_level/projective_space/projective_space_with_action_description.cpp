/*
 * projective_space_with_action_description.cpp
 *
 *  Created on: Jan 5, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {
namespace projective_geometry {




projective_space_with_action_description::projective_space_with_action_description()
{
	n = 0;
	//input_q;
	F = NULL;
	f_use_projectivity_subgroup = FALSE;

}

projective_space_with_action_description::~projective_space_with_action_description()
{
}


int projective_space_with_action_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i = 0;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "projective_space_with_action_description::read_arguments" << endl;
	}
	n = ST.strtoi(argv[i++]);
	if (f_v) {
		cout << "n = " << n << endl;
	}
	input_q.assign(argv[i++]);
	if (f_v) {
		cout << "q = " << input_q << endl;
		cout << "projective_space_with_action_description::read_arguments done" << endl;
	}
	for (; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-use_projectivity_subgroup") == 0) {
			f_use_projectivity_subgroup = TRUE;
			if (f_v) {
				cout << "-f_use_projectivity_subgroup" << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}

		else {
			cout << "projective_space_with_action_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}

		if (f_v) {
			cout << "projective_space_with_action_description::read_arguments looping, i=" << i << endl;
		}
	} // next i

	if (f_v) {
		cout << "projective_space_with_action_description::read_arguments done" << endl;
	}
	return i + 1;
}



void projective_space_with_action_description::print()
{
	cout << n << " " << input_q;
	if (f_use_projectivity_subgroup) {
		cout << " -f_use_projectivity_subgroup";
	}
	cout << endl;
}

}}}

