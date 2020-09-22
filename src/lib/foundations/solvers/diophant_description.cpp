/*
 * diophant_description.cpp
 *
 *  Created on: May 28, 2020
 *      Author: betten
 */



#include "foundations.h"


using namespace std;

namespace orbiter {
namespace foundations {



diophant_description::diophant_description()
{
	f_q = FALSE;
	input_q = 0;
	f_override_polynomial = FALSE;
	//override_polynomial = NULL;
	//F = NULL;

	f_maximal_arc = FALSE;
	maximal_arc_sz = 0;
	maximal_arc_d = 0;
	maximal_arc_secants_text = NULL;
	external_lines_as_subset_of_secants_text = NULL;

	f_label = FALSE;
	//label = NULL;
	//from_scratch_m = 0;
	//from_scratch_n = 0;

	f_coefficient_matrix = FALSE;
	coefficient_matrix_m = 0;
	coefficient_matrix_n = 0;
	coefficient_matrix_text = NULL;

	f_coefficient_matrix_csv = FALSE;
	//coefficient_matrix_csv = NULL;

	f_RHS_constant = FALSE;
	//RHS_constant_text = NULL;


	f_RHS = FALSE;
	//RHS_text = NULL;

	f_RHS_csv = FALSE;
	//RHS_csv_text = NULL;

	f_x_max_global = FALSE;
	x_max_global = 0;

	f_x_min_global = FALSE;
	x_min_global = 0;

	f_x_bounds = FALSE;
	x_bounds_text = NULL;

	f_x_bounds_csv = FALSE;
	//x_bounds_csv = NULL;

	f_has_sum = FALSE;
	has_sum = 0;
}


diophant_description::~diophant_description()
{
}

int diophant_description::read_arguments(
	int argc, const char **argv,
	int verbose_level)
{
	int i;

	cout << "diophant_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {


		if (strcmp(argv[i], "-maximal_arc") == 0) {
			f_maximal_arc = TRUE;
			maximal_arc_sz = atoi(argv[++i]);
			maximal_arc_d = atoi(argv[++i]);
			maximal_arc_secants_text = argv[++i];
			external_lines_as_subset_of_secants_text = argv[++i];
			cout << "-maximal_arc " << maximal_arc_sz << " " << maximal_arc_d
					<< " " << maximal_arc_secants_text
					<< " " << external_lines_as_subset_of_secants_text << endl;
		}
		else if (strcmp(argv[i], "-label") == 0) {
			f_label = TRUE;
			label.assign(argv[++i]);
			cout << "-label " << label << endl;
		}
		else if (strcmp(argv[i], "-coefficient_matrix") == 0) {
			f_coefficient_matrix = TRUE;
			coefficient_matrix_m = atoi(argv[++i]);
			coefficient_matrix_n = atoi(argv[++i]);
			coefficient_matrix_text = argv[++i];
			cout << "-coefficient_matrix " << coefficient_matrix_m << " "
					<< coefficient_matrix_n << " " << coefficient_matrix_text << endl;
		}
		else if (strcmp(argv[i], "-coefficient_matrix_csv") == 0) {
			f_coefficient_matrix_csv = TRUE;
			coefficient_matrix_csv.assign(argv[++i]);
			cout << "-coefficient_matrix_csv " << coefficient_matrix_csv << endl;
		}
		else if (strcmp(argv[i], "-RHS") == 0) {
			f_RHS = TRUE;
			RHS_text.assign(argv[++i]);
			cout << "-RHS " << RHS_text << endl;
		}
		else if (strcmp(argv[i], "-RHS_csv") == 0) {
			f_RHS_csv = TRUE;
			RHS_csv_text.assign(argv[++i]);
			cout << "-RHS_csv " << RHS_csv_text << endl;
		}
		else if (strcmp(argv[i], "-RHS_constant") == 0) {
			f_RHS_constant = TRUE;
			RHS_constant_text.assign(argv[++i]);
			cout << "-RHS_constant " << RHS_constant_text << endl;
		}
		else if (strcmp(argv[i], "-x_max_global") == 0) {
			f_x_max_global = TRUE;
			x_max_global = atoi(argv[++i]);
			cout << "-x_max_global " << x_max_global << endl;
		}
		else if (strcmp(argv[i], "-x_min_global") == 0) {
			f_x_min_global = TRUE;
			x_min_global = atoi(argv[++i]);
			cout << "-x_min_global " << x_min_global << endl;
		}
		else if (strcmp(argv[i], "-x_bounds") == 0) {
			f_x_bounds = TRUE;
			x_bounds_text = argv[++i];
			cout << "-x_bounds " << x_bounds_text << endl;
		}
		else if (strcmp(argv[i], "-x_bounds_csv") == 0) {
			f_x_bounds_csv = TRUE;
			x_bounds_csv.assign(argv[++i]);
			cout << "-x_bounds_csv " << x_bounds_csv << endl;
		}
		else if (strcmp(argv[i], "-has_sum") == 0) {
			f_has_sum = TRUE;
			has_sum = atoi(argv[++i]);
			cout << "-has_sum " << has_sum << endl;
		}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			input_q = atoi(argv[++i]);
			cout << "-q" << input_q << endl;
		}
		else if (strcmp(argv[i], "-override_polynomial") == 0) {
			f_override_polynomial = TRUE;
			override_polynomial.assign(argv[++i]);
			cout << "-override_polynomial" << override_polynomial << endl;
		}
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "diophant_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "diophant_description::read_arguments done" << endl;
	return i;
}

}}

