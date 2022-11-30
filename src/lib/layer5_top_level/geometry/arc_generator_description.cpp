/*
 * arc_generator_description.cpp
 *
 *  Created on: Jun 26, 2020
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_geometry {


arc_generator_description::arc_generator_description()
{

	f_control = FALSE;
	//std::string control_label;

	//f_poset_classification_control = FALSE;
	//Control = NULL;

	f_d = FALSE;
	d = 0;

	f_target_size = FALSE;
	target_size = 0;

	f_conic_test = FALSE;

	f_test_nb_Eckardt_points = FALSE;
	nb_E = 0;
	//Surf = NULL;
	f_affine = FALSE;
	f_no_arc_testing = FALSE;

	f_has_forbidden_point_set = FALSE;
	//forbidden_point_set_string = NULL;

	f_override_group = FALSE;
	//std::string override_group_label;

}

arc_generator_description::~arc_generator_description()
{
}

int arc_generator_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;


	if (f_v) {
		cout << "arc_generator_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

#if 0
		if (ST.stringcmp(argv[i], "-poset_classification_control") == 0) {
			f_poset_classification_control = TRUE;
			Control = NEW_OBJECT(poset_classification::poset_classification_control);
			if (f_v) {
				cout << "-poset_classification_control " << endl;
			}
			i += Control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -poset_classification_control " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}
#endif
		if (ST.stringcmp(argv[i], "-control") == 0) {
			f_control = TRUE;
			control_label.assign(argv[++i]);
			if (f_v) {
				cout << "-control " << control_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-d") == 0) {
			f_d = TRUE;
			d = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-d " << d << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-target_size") == 0) {
			f_target_size = TRUE;
			target_size = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-target_size " << target_size << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-conic_test") == 0) {
			f_conic_test = TRUE;
			if (f_v) {
				cout << "-conic_test " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-test_nb_Eckardt_points") == 0) {
			f_test_nb_Eckardt_points = TRUE;
			nb_E = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-test_nb_Eckardt_points " << nb_E << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-affine") == 0) {
			f_affine = TRUE;
			if (f_v) {
				cout << "-affine " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-no_arc_testing") == 0) {
			f_no_arc_testing = TRUE;
			if (f_v) {
				cout << "-no_arc_testing " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-forbidden_point_set") == 0) {
			f_has_forbidden_point_set = TRUE;
			orbiter_kernel_system::os_interface Os;

			i++;

			Os.get_string_from_command_line(forbidden_point_set_string, argc, argv, i, verbose_level);
			i--;
			if (f_v) {
				cout << "-f_has_forbidden_point_set " << forbidden_point_set_string << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-override_group") == 0) {
			f_override_group = TRUE;
			override_group_label.assign(argv[++i]);
			if (f_v) {
				cout << "-override_group " << override_group_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "arc_generator_description::read_arguments unknown argument " << argv[i] << endl;
			exit(1);
		}
	} // next i

	if (f_v) {
		cout << "arc_generator_description::read_arguments done" << endl;
	}
	return i + 1;
}

void arc_generator_description::print()
{
#if 0
	if (f_poset_classification_control) {
		Control->print();
	}
#endif

	if (f_control) {
		cout << "-control " << control_label << endl;
	}

	if (f_d) {
		cout << "-d " << d << endl;
	}
	if (f_target_size) {
		cout << "-target_size " << target_size << endl;
	}
	if (f_conic_test) {
		cout << "-conic_test " << endl;
	}
	if (f_test_nb_Eckardt_points) {
		cout << "-test_nb_Eckardt_points " << nb_E << endl;
	}
	if (f_affine) {
		cout << "-affine " << endl;
	}
	if (f_no_arc_testing) {
		cout << "-no_arc_testing " << endl;
	}
	if (f_has_forbidden_point_set) {
		cout << "-has_forbidden_point_set " << forbidden_point_set_string << endl;
	}
	if (f_override_group) {
		cout << "-override_group " << override_group_label << endl;
	}
}


}}}



