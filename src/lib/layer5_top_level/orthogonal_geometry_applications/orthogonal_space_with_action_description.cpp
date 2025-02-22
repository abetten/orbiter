/*
 * orthogonal_space_with_action_description.cpp
 *
 *  Created on: Jan 12, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {


orthogonal_space_with_action_description::orthogonal_space_with_action_description()
{
	Record_birth();
	epsilon = 0;
	n = 0;

	//input_q;
	F = NULL;

	f_label_txt = false;
	//std::string label_txt;
	f_label_tex = false;
	//std::string label_tex;


	f_without_group = false;

	f_create_extension_fields = false;

}




orthogonal_space_with_action_description::~orthogonal_space_with_action_description()
{
	Record_death();
}


int orthogonal_space_with_action_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "projective_space_object_classifier_description::read_arguments" << endl;
	}
	//cout << "next argument is " << argv[0] << endl;
	epsilon = ST.strtoi(argv[0]);
	//cout << "epsilon = " << epsilon << endl;
	n = ST.strtoi(argv[1]);
	//cout << "n = " << n << endl;
	input_q.assign(argv[2]);
	//cout << "q = " << input_q << endl;
	//cout << "orthogonal_space_with_action_description::read_arguments done" << endl;

	for (i = 3; i < argc; i++) {

		//cout << "projective_space_object_classifier_description::read_arguments, next argument is " << argv[i] << endl;

		if (ST.stringcmp(argv[i], "-label_txt") == 0) {
			f_label_txt = true;
			label_txt.assign(argv[++i]);
			if (f_v) {
				cout << "-label_txt " << label_txt << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-label_tex") == 0) {
			f_label_tex = true;
			label_tex.assign(argv[++i]);
			if (f_v) {
				cout << "-label_tet " << label_tex << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-without_group") == 0) {
			f_without_group = true;
			if (f_v) {
				cout << "-without_group "<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-create_extension_fields") == 0) {
			f_create_extension_fields = true;
			if (f_v) {
				cout << "-create_extension_fields "<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}

		else {
			cout << "projective_space_object_classifier_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		//cout << "projective_space_object_classifier_description::read_arguments looping, i=" << i << endl;
	} // next i
	if (f_v) {
		cout << "projective_space_object_classifier_description::read_arguments done" << endl;
	}
	return i + 1;
}

void orthogonal_space_with_action_description::print()
{
	//cout << "orthogonal_space_with_action_description::print:" << endl;

	cout << "epsilon = " << epsilon << endl;
	cout << "n = " << n << endl;
	cout << "q = " << input_q << endl;
	if (f_label_txt) {
		cout << "-label_txt " << label_txt << endl;
	}
	if (f_label_tex) {
		cout << "-label_tex " << label_tex << endl;
	}
	if (f_without_group) {
		cout << "without group" << endl;
	}
	if (f_create_extension_fields) {
		cout << "-create_extension_fields "<< endl;
	}
}


}}}

