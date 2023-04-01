/*
 * diophant_description.cpp
 *
 *  Created on: May 28, 2020
 *      Author: betten
 */



#include "foundations.h"


using namespace std;

namespace orbiter {
namespace layer1_foundations {
namespace solvers {



diophant_description::diophant_description()
{
	f_field = FALSE;
	//std::string field_label;

	f_maximal_arc = FALSE;
	maximal_arc_sz = 0;
	maximal_arc_d = 0;
	//maximal_arc_secants_text;
	//external_lines_as_subset_of_secants_text;

	f_label = FALSE;
	//label = NULL;
	//from_scratch_m = 0;
	//from_scratch_n = 0;

	f_coefficient_matrix = FALSE;
	//coefficient_matrix_label;

	f_problem_of_Steiner_type = FALSE;
	problem_of_Steiner_type_nb_t_orbits = 0;
	//std::string problem_of_Steiner_type_covering_matrix_fname;

	f_coefficient_matrix_csv = FALSE;
	//coefficient_matrix_csv;

	f_RHS_constant = FALSE;
	//RHS_constant_text;


	f_RHS = FALSE;
	//RHS_text;

	f_RHS_csv = FALSE;
	//RHS_csv_text;

	f_x_max_global = FALSE;
	x_max_global = 0;

	f_x_min_global = FALSE;
	x_min_global = 0;

	f_x_bounds = FALSE;
	//x_bounds_text;

	f_x_bounds_csv = FALSE;
	//x_bounds_csv;

	f_has_sum = FALSE;
	has_sum = 0;
}


diophant_description::~diophant_description()
{
}

int diophant_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "diophant_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {


		if (ST.stringcmp(argv[i], "-field") == 0) {
			f_field = TRUE;
			field_label = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-field" << field_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-maximal_arc") == 0) {
			f_maximal_arc = TRUE;
			maximal_arc_sz = ST.strtoi(argv[++i]);
			maximal_arc_d = ST.strtoi(argv[++i]);
			maximal_arc_secants_text.assign(argv[++i]);
			external_lines_as_subset_of_secants_text.assign(argv[++i]);
			if (f_v) {
				cout << "-maximal_arc " << maximal_arc_sz << " " << maximal_arc_d
					<< " " << maximal_arc_secants_text
					<< " " << external_lines_as_subset_of_secants_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-label") == 0) {
			f_label = TRUE;
			label.assign(argv[++i]);
			if (f_v) {
				cout << "-label " << label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-coefficient_matrix") == 0) {
			f_coefficient_matrix = TRUE;
			coefficient_matrix_label.assign(argv[++i]);
			if (f_v) {
				cout << "-coefficient_matrix " << coefficient_matrix_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-problem_of_Steiner_type") == 0) {
			f_problem_of_Steiner_type = TRUE;
			problem_of_Steiner_type_nb_t_orbits = ST.strtoi(argv[++i]);
			problem_of_Steiner_type_covering_matrix_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-problem_of_Steiner_type " << problem_of_Steiner_type_nb_t_orbits
					<< " " << problem_of_Steiner_type_covering_matrix_fname << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-coefficient_matrix_csv") == 0) {
			f_coefficient_matrix_csv = TRUE;
			coefficient_matrix_csv.assign(argv[++i]);
			if (f_v) {
				cout << "-coefficient_matrix_csv " << coefficient_matrix_csv << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-RHS") == 0) {
			f_RHS = TRUE;
			RHS_text.assign(argv[++i]);
			if (f_v) {
				cout << "-RHS " << RHS_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-RHS_csv") == 0) {
			f_RHS_csv = TRUE;
			RHS_csv_text.assign(argv[++i]);
			if (f_v) {
				cout << "-RHS_csv " << RHS_csv_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-RHS_constant") == 0) {
			f_RHS_constant = TRUE;
			RHS_constant_text.assign(argv[++i]);
			if (f_v) {
				cout << "-RHS_constant " << RHS_constant_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-x_max_global") == 0) {
			f_x_max_global = TRUE;
			x_max_global = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-x_max_global " << x_max_global << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-x_min_global") == 0) {
			f_x_min_global = TRUE;
			x_min_global = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-x_min_global " << x_min_global << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-x_bounds") == 0) {
			f_x_bounds = TRUE;
			x_bounds_text.assign(argv[++i]);
			if (f_v) {
				cout << "-x_bounds " << x_bounds_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-x_bounds_csv") == 0) {
			f_x_bounds_csv = TRUE;
			x_bounds_csv.assign(argv[++i]);
			if (f_v) {
				cout << "-x_bounds_csv " << x_bounds_csv << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-has_sum") == 0) {
			f_has_sum = TRUE;
			has_sum = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-has_sum " << has_sum << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "diophant_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "diophant_description::read_arguments done" << endl;
	}
	return i + 1;
}


void diophant_description::print()
{
	if (f_field) {
		cout << "-field" << field_label << endl;
	}
	if (f_maximal_arc) {
		cout << "-maximal_arc " << maximal_arc_sz << " " << maximal_arc_d
				<< " " << maximal_arc_secants_text
				<< " " << external_lines_as_subset_of_secants_text << endl;
	}
	if (f_label) {
		cout << "-label " << label << endl;
	}
	if (f_coefficient_matrix) {
		cout << "-coefficient_matrix " << coefficient_matrix_label << endl;
	}
	if (f_problem_of_Steiner_type) {
		cout << "-problem_of_Steiner_type " << problem_of_Steiner_type_nb_t_orbits
				<< " " << problem_of_Steiner_type_covering_matrix_fname << endl;
	}

	if (f_coefficient_matrix_csv) {
		cout << "-coefficient_matrix_csv " << coefficient_matrix_csv << endl;
	}
	if (f_RHS) {
		cout << "-RHS " << RHS_text << endl;
	}
	if (f_RHS_csv) {
		cout << "-RHS_csv " << RHS_csv_text << endl;
	}
	if (f_RHS_constant) {
		cout << "-RHS_constant " << RHS_constant_text << endl;
	}
	if (f_x_max_global) {
		cout << "-x_max_global " << x_max_global << endl;
	}
	if (f_x_min_global) {
		cout << "-x_min_global " << x_min_global << endl;
	}
	if (f_x_bounds) {
		cout << "-x_bounds " << x_bounds_text << endl;
	}
	if (f_x_bounds_csv) {
		cout << "-x_bounds_csv " << x_bounds_csv << endl;
	}
	if (f_has_sum) {
		cout << "-has_sum " << has_sum << endl;
	}
}


}}}

