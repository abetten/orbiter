/*
 * diophant_activity_description.cpp
 *
 *  Created on: May 29, 2020
 *      Author: betten
 */


#include "foundations.h"


using namespace std;

namespace orbiter {
namespace foundations {



diophant_activity_description::diophant_activity_description()
{
	f_input_file = FALSE;
	input_file = NULL;
	f_print = FALSE;
	f_solve_mckay = FALSE;
	f_solve_standard = FALSE;
	f_draw = FALSE;
	f_perform_column_reductions = FALSE;
}

diophant_activity_description::~diophant_activity_description()
{
}

int diophant_activity_description::read_arguments(
	int argc, const char **argv,
	int verbose_level)
{
	int i;

	cout << "diophant_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (strcmp(argv[i], "-input_file") == 0) {
			f_input_file = TRUE;
			input_file = argv[++i];
			cout << "-input_file " << input_file << endl;
		}
		else if (strcmp(argv[i], "-print") == 0) {
			f_print = TRUE;
			cout << "-print " << endl;
		}
		else if (strcmp(argv[i], "-solve_mckay") == 0) {
			f_solve_mckay = TRUE;
			cout << "-solve_mckay " << endl;
		}
		else if (strcmp(argv[i], "-solve_standard") == 0) {
			f_solve_standard = TRUE;
			cout << "-solve_standard " << endl;
		}
		else if (strcmp(argv[i], "-draw") == 0) {
			f_draw = TRUE;
			cout << "-draw " << endl;
		}
		else if (strcmp(argv[i], "-perform_column_reductions") == 0) {
			f_perform_column_reductions = TRUE;
			cout << "-perform_column_reductions " << endl;
		}
		else {
			cout << "diophant_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "diophant_activity_description::read_arguments done" << endl;
	return i;
}


}}


