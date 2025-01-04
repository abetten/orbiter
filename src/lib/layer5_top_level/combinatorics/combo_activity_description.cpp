/*
 * combo_activity_description.cpp
 *
 *  Created on: Jan 3, 2025
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {



combo_activity_description::combo_activity_description()
{
	Record_birth();

	f_report = false;
	Objects_report_options = NULL;

}

combo_activity_description::~combo_activity_description()
{
	Record_death();
}


int combo_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "combo_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {



		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;

			Objects_report_options = NEW_OBJECT(combinatorics::canonical_form_classification::objects_report_options);
			i += Objects_report_options->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
		}


		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "combo_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "combo_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void combo_activity_description::print()
{
	if (f_report) {
		cout << "-report " << endl;
		Objects_report_options->print();
	}
}



}}}


