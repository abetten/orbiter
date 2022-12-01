/*
 * action_on_forms_activity_description.cpp
 *
 *  Created on: Oct 24, 2022
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;
using namespace orbiter::layer1_foundations;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


action_on_forms_activity_description::action_on_forms_activity_description()
{

	f_algebraic_normal_form = FALSE;
	//std::string algebraic_normal_form_input;

	f_orbits_on_functions = FALSE;
	//std::string orbits_on_functions_input;

	f_associated_set_in_plane = FALSE;
	//std::string associated_set_in_plane_input;

	f_differential_uniformity = FALSE;
	//std::string differential_uniformity_input;


}


action_on_forms_activity_description::~action_on_forms_activity_description()
{
}


int action_on_forms_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level > 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "action_on_forms_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-algebraic_normal_form") == 0) {
			f_algebraic_normal_form = TRUE;
			algebraic_normal_form_input.assign(argv[++i]);
			cout << "-algebraic_normal_form " << algebraic_normal_form_input << endl;
		}
		else if (ST.stringcmp(argv[i], "-orbits_on_functions") == 0) {
			f_orbits_on_functions = TRUE;
			orbits_on_functions_input.assign(argv[++i]);
			cout << "-orbits_on_functions " << orbits_on_functions_input << endl;
		}
		else if (ST.stringcmp(argv[i], "-associated_set_in_plane") == 0) {
			f_associated_set_in_plane = TRUE;
			associated_set_in_plane_input.assign(argv[++i]);
			cout << "-associated_set_in_plane " << associated_set_in_plane_input << endl;
		}
		else if (ST.stringcmp(argv[i], "-differential_uniformity") == 0) {
			f_differential_uniformity = TRUE;
			differential_uniformity_input.assign(argv[++i]);
			cout << "-differential_uniformity " << differential_uniformity_input << endl;
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "action_on_forms_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i

	if (f_v) {
		cout << "action_on_forms_description::read_arguments done" << endl;
	}
	return i + 1;
}


void action_on_forms_activity_description::print()
{
	if (f_algebraic_normal_form) {
		cout << "-algebraic_normal_form " << algebraic_normal_form_input << endl;
	}
	if (f_orbits_on_functions) {
		cout << "-orbits_on_functions " << orbits_on_functions_input << endl;
	}
	if (f_associated_set_in_plane) {
		cout << "-associated_set_in_plane " << associated_set_in_plane_input << endl;
	}
	if (f_differential_uniformity) {
		cout << "-differential_uniformity " << differential_uniformity_input << endl;
	}
}



}}}




