/*
 * arc_generator_description.cpp
 *
 *  Created on: Jun 26, 2020
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


arc_generator_description::arc_generator_description()
{
	f_q = FALSE;
	LG = NULL;
	F = NULL;
	q = 0;

	f_poset_classification_control = FALSE;
	Control = NULL;
	f_d = FALSE;
	d = 0;
	f_n = FALSE;
	n = 0;
	f_target_size = FALSE;
	target_size = 0;
	f_conic_test = FALSE;
	f_test_nb_Eckardt_points = FALSE;
	nb_E = 0;
	Surf = NULL;
	f_affine = FALSE;
	f_no_arc_testing = FALSE;
	f_has_forbidden_point_set = FALSE;
	//forbidden_point_set_string = NULL;

}

arc_generator_description::~arc_generator_description()
{
}

int arc_generator_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int i;



	cout << "arc_generator_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = strtoi(argv[++i]);
			cout << "-q " << q << endl;
		}
		else if (stringcmp(argv[i], "-poset_classification_control") == 0) {
			f_poset_classification_control = TRUE;
			Control = NEW_OBJECT(poset_classification_control);
			cout << "-poset_classification_control " << endl;
			i += Control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -poset_classification_control " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-q " << q << endl;
		}
		else if (stringcmp(argv[i], "-d") == 0) {
			f_d = TRUE;
			d = strtoi(argv[++i]);
			cout << "-d " << d << endl;
		}
		else if (stringcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = strtoi(argv[++i]);
			cout << "-n " << n << endl;
		}
		else if (stringcmp(argv[i], "-target_size") == 0) {
			f_target_size = TRUE;
			target_size = strtoi(argv[++i]);
			cout << "-target_size " << target_size << endl;
		}
		else if (stringcmp(argv[i], "-conic_test") == 0) {
			f_conic_test = TRUE;
			cout << "-conic_test " << endl;
		}
		else if (stringcmp(argv[i], "-test_nb_Eckardt_points") == 0) {
			f_test_nb_Eckardt_points = TRUE;
			nb_E = strtoi(argv[++i]);
			cout << "-test_nb_Eckardt_points " << nb_E << endl;
		}
		else if (stringcmp(argv[i], "-affine") == 0) {
			f_affine = TRUE;
			cout << "-affine " << endl;
		}
		else if (stringcmp(argv[i], "-no_arc_testing") == 0) {
			f_no_arc_testing = TRUE;
			cout << "-no_arc_testing " << endl;
		}
		else if (stringcmp(argv[i], "-forbidden_point_set") == 0) {
			f_has_forbidden_point_set = TRUE;
			os_interface Os;

			i++;

			Os.get_string_from_command_line(forbidden_point_set_string, argc, argv, i, verbose_level);
			i--;
			cout << "-f_has_forbidden_point_set " << forbidden_point_set_string << endl;
		}
		else if (stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "ignoring argument " << argv[i] << endl;
		}
	} // next i

	cout << "arc_generator_description::read_arguments done" << endl;
	return i + 1;
}



}}



