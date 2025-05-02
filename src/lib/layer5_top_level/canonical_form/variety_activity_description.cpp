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
	Record_birth();

	f_compute_group = false;

	f_compute_set_stabilizer = false;

	f_nauty_control = false;
	Nauty_interface_control = NULL;

	f_report = false;

	f_export = false;

	f_classify = false;

	f_apply_transformation = false;
	//std::string apply_transformation_group_element;

	f_singular_points = false;

	f_output_fname_base = false;
	//std::string output_fname_base;

}


variety_activity_description::~variety_activity_description()
{
	Record_death();
}


int variety_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level > 1);
	int i;
	other::data_structures::string_tools ST;

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
		else if (ST.stringcmp(argv[i], "-compute_set_stabilizer") == 0) {
			f_compute_set_stabilizer = true;
			if (f_v) {
				cout << "-compute_set_stabilizer " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-nauty_control") == 0) {
			if (f_v) {
				cout << "-nauty_control " << endl;
			}
			f_nauty_control = true;
			Nauty_interface_control = NEW_OBJECT(other::l1_interfaces::nauty_interface_control);

			i += Nauty_interface_control->parse_arguments(
					argc - (i + 1), argv + i + 1,
					verbose_level);

			if (f_v) {
				cout << "done reading -nauty_control " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}
		else if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;
			if (f_v) {
				cout << "-report " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export") == 0) {
			f_export = true;
			if (f_v) {
				cout << "-export " << endl;
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
		else if (ST.stringcmp(argv[i], "-output_fname_base") == 0) {
			f_output_fname_base = true;
			output_fname_base.assign(argv[++i]);
			if (f_v) {
				cout << "-output_fname_base " << output_fname_base << endl;
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
	if (f_compute_set_stabilizer) {
		cout << "-compute_set_stabilizer " << endl;
	}
	if (f_nauty_control) {
		cout << "-nauty_control " << endl;
		Nauty_interface_control->print();
	}
	if (f_report) {
		cout << "-report " << endl;
	}
	if (f_export) {
		cout << "-export " << endl;
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
	if (f_output_fname_base) {
		cout << "-output_fname_base " << output_fname_base << endl;
	}
}



}}}




