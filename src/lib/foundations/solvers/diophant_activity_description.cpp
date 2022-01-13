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
	//input_file = NULL;
	f_print = FALSE;
	f_solve_mckay = FALSE;
	f_solve_standard = FALSE;
	f_solve_DLX = FALSE;

	f_draw_as_bitmap = FALSE;
	box_width = 1;
	bit_depth = 8;
	f_draw = FALSE;
	f_perform_column_reductions = FALSE;

	f_project_to_single_equation_and_solve = FALSE;
	eqn_idx = 0;
	solve_case_idx = 0;

	f_project_to_two_equations_and_solve = FALSE;
	eqn1_idx = 0;
	eqn2_idx = 0;
	solve_case_idx_r = 0;
	solve_case_idx_m = 0;

	f_test_single_equation = FALSE;
	max_number_of_coefficients = 0;

}

diophant_activity_description::~diophant_activity_description()
{
}

int diophant_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "diophant_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-input_file") == 0) {
			f_input_file = TRUE;
			input_file.assign(argv[++i]);
			if (f_v) {
				cout << "-input_file " << input_file << endl;
			}
		}
		else if (stringcmp(argv[i], "-print") == 0) {
			f_print = TRUE;
			if (f_v) {
				cout << "-print " << endl;
			}
		}
		else if (stringcmp(argv[i], "-solve_mckay") == 0) {
			f_solve_mckay = TRUE;
			if (f_v) {
				cout << "-solve_mckay " << endl;
			}
		}
		else if (stringcmp(argv[i], "-solve_standard") == 0) {
			f_solve_standard = TRUE;
			if (f_v) {
				cout << "-solve_standard " << endl;
			}
		}
		else if (stringcmp(argv[i], "-solve_DLX") == 0) {
			f_solve_DLX = TRUE;
			if (f_v) {
				cout << "-solve_DLX " << endl;
			}
		}
		else if (stringcmp(argv[i], "-draw_as_bitmap") == 0) {
			f_draw_as_bitmap = TRUE;
			box_width = strtoi(argv[++i]);
			bit_depth = strtoi(argv[++i]);
			if (f_v) {
				cout << "-draw_as_bitmap " << box_width << " " << bit_depth << endl;
			}
		}
		else if (stringcmp(argv[i], "-draw") == 0) {
			f_draw = TRUE;
			if (f_v) {
				cout << "-draw " << endl;
			}
		}
		else if (stringcmp(argv[i], "-perform_column_reductions") == 0) {
			f_perform_column_reductions = TRUE;
			if (f_v) {
				cout << "-perform_column_reductions " << endl;
			}
		}
		else if (stringcmp(argv[i], "-test_single_equation") == 0) {
			f_test_single_equation = TRUE;
			max_number_of_coefficients = strtoi(argv[++i]);
			if (f_v) {
				cout << "-test_single_equation " << max_number_of_coefficients << endl;
			}
		}
		else if (stringcmp(argv[i], "-project_to_single_equation_and_solve") == 0) {
			f_project_to_single_equation_and_solve = TRUE;
			eqn_idx = strtoi(argv[++i]);
			solve_case_idx = strtoi(argv[++i]);
			if (f_v) {
				cout << "-project_to_single_equation_and_solve " << eqn_idx << " " << solve_case_idx << endl;
			}
		}
		else if (stringcmp(argv[i], "-project_to_two_equations_and_solve") == 0) {
			f_project_to_two_equations_and_solve = TRUE;
			eqn1_idx = strtoi(argv[++i]);
			eqn2_idx = strtoi(argv[++i]);
			solve_case_idx_r = strtoi(argv[++i]);
			solve_case_idx_m = strtoi(argv[++i]);
			if (f_v) {
				cout << "-project_to_single_equation_and_solve " << eqn1_idx << " " << eqn2_idx
					<< " " << solve_case_idx_r
					<< " " << solve_case_idx_m
					<< endl;
			}
		}
		else if (stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "diophant_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "diophant_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void diophant_activity_description::print()
{
	if (f_input_file) {
		cout << "-input_file " << input_file << endl;
	}
	if (f_print) {
		cout << "-print " << endl;
	}
	if (f_solve_mckay) {
		cout << "-solve_mckay " << endl;
	}
	if (f_solve_standard) {
		cout << "-solve_standard " << endl;
	}
	if (f_solve_DLX) {
		cout << "-solve_DLX " << endl;
	}
	if (f_draw_as_bitmap) {
		cout << "-draw_as_bitmap " << box_width << " " << bit_depth << endl;
	}
	if (f_draw) {
		cout << "-draw " << endl;
	}
	if (f_perform_column_reductions) {
		cout << "-perform_column_reductions " << endl;
	}
	if (f_test_single_equation) {
		cout << "-test_single_equation " << max_number_of_coefficients << endl;
	}
	if (f_project_to_single_equation_and_solve) {
		cout << "-project_to_single_equation_and_solve " << eqn_idx << " " << solve_case_idx << endl;
	}
	if (f_project_to_two_equations_and_solve) {
		cout << "-project_to_single_equation_and_solve " << eqn1_idx << " " << eqn2_idx
				<< " " << solve_case_idx_r
				<< " " << solve_case_idx_m
				<< endl;
	}
}


}}


