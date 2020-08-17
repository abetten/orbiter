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
	forbidden_point_set_string = NULL;

}

arc_generator_description::~arc_generator_description()
{
}

int arc_generator_description::read_arguments(int argc, const char **argv,
	int verbose_level)
{
	int i;



	cout << "arc_generator_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
		}
		else if (strcmp(argv[i], "-d") == 0) {
			f_d = TRUE;
			d = atoi(argv[++i]);
			cout << "-d " << d << endl;
		}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
		}
		else if (strcmp(argv[i], "-target_size") == 0) {
			f_target_size = TRUE;
			target_size = atoi(argv[++i]);
			cout << "-target_size " << target_size << endl;
		}
		else if (strcmp(argv[i], "-conic_test") == 0) {
			f_conic_test = TRUE;
			cout << "-conic_test " << endl;
		}
		else if (strcmp(argv[i], "-test_nb_Eckardt_points") == 0) {
			f_test_nb_Eckardt_points = TRUE;
			nb_E = atoi(argv[++i]);
			cout << "-test_nb_Eckardt_points " << nb_E << endl;
		}
		else if (strcmp(argv[i], "-affine") == 0) {
			f_affine = TRUE;
			cout << "-affine " << endl;
		}
		else if (strcmp(argv[i], "-no_arc_testing") == 0) {
			f_no_arc_testing = TRUE;
			cout << "-no_arc_testing " << endl;
		}
		else if (strcmp(argv[i], "-forbidden_point_set") == 0) {
			f_has_forbidden_point_set = TRUE;
			forbidden_point_set_string = argv[++i];
			cout << "-f_has_forbidden_point_set " << forbidden_point_set_string << endl;
		}

		else if (strcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "ignoring argument " << argv[i] << endl;
		}
	} // next i

	cout << "arc_generator_description::read_arguments done" << endl;
	return i;
}



}}



