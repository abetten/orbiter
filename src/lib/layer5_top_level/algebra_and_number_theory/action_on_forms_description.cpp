/*
 * action_on_forms_description.cpp
 *
 *  Created on: Oct 23, 2022
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;
using namespace orbiter::layer1_foundations;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


action_on_forms_description::action_on_forms_description()
{

	f_space = FALSE;
	//std::string space_label;


	f_degree = FALSE;
	degree = 0;


}


action_on_forms_description::~action_on_forms_description()
{
}


int action_on_forms_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level > 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "action_on_forms_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-space") == 0) {
			f_space = TRUE;
			space_label.assign(argv[++i]);
			cout << "-space " << space_label << endl;
		}
		else if (ST.stringcmp(argv[i], "-degree") == 0) {
			f_degree = TRUE;
			degree = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-degree " << degree << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "action_on_forms_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i

	if (f_v) {
		cout << "action_on_forms_description::read_arguments done" << endl;
	}
	return i + 1;
}


void action_on_forms_description::print()
{
	if (f_space) {
		cout << "-space " << space_label << endl;
	}
	if (f_degree) {
		cout << "-degree " << degree << endl;
	}
}



}}}


