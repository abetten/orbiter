/*
 * tdo_refinement_description.cpp
 *
 *  Created on: May 23, 2020
 *      Author: betten
 */





#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {

tdo_refinement_description::tdo_refinement_description()
{
	fname_in = NULL;
	f_lambda3 = FALSE;
	lambda3 = 0;
	block_size = 0;
	f_scale = FALSE;
	scaling = 0;
	f_range = FALSE;
	range_first = 0;
	range_len = 1;
	f_select = FALSE;
	select_label = NULL;
	f_omit1 = FALSE;
	omit1 = 0;
	f_omit2 = FALSE;
	omit2 = 0;
	f_D1_upper_bound_x0 = FALSE;
	D1_upper_bound_x0 = 0;
	f_reverse = FALSE;
	f_reverse_inverse = FALSE;
	f_use_packing_numbers = FALSE;
	f_dual_is_linear_space = FALSE;
	f_do_the_geometric_test = FALSE;
	f_once = FALSE;
	f_use_mckay_solver = FALSE;
	f_input_file = FALSE;
	fname_in = NULL;

	Sol = NULL;
}

tdo_refinement_description::~tdo_refinement_description()
{

	if (Sol) {
		FREE_OBJECT(Sol);
	}
}

int tdo_refinement_description::read_arguments(int argc, const char **argv, int verbose_level)
{
	int i;

#if 0
	if (argc <= 1) {
		print_usage();
		exit(1);
		}
#endif

	Sol = NEW_OBJECT(solution_file_data);
	Sol->nb_solution_files = 0;


	for (i = 0; i < argc - 1; i++) {
		if (strcmp(argv[i], "-lambda3") == 0) {
			f_lambda3 = TRUE;
			lambda3 = atoi(argv[++i]);
			block_size = atoi(argv[++i]);
			cout << "-lambda3 " << lambda3 << " " << block_size << endl;
		}
		if (strcmp(argv[i], "-scale") == 0) {
			f_scale = TRUE;
			scaling = atoi(argv[++i]);
			cout << "-scale " << scaling << endl;
		}
		if (strcmp(argv[i], "-solution") == 0) {
			//f_solution = TRUE;
			Sol->system_no[Sol->nb_solution_files] = atoi(argv[++i]);
			Sol->solution_file[Sol->nb_solution_files] = argv[++i];
			cout << "-solution " << Sol->system_no[Sol->nb_solution_files]
				<< " " << Sol->solution_file[Sol->nb_solution_files] << endl;
			Sol->nb_solution_files++;
		}
		else if (strcmp(argv[i], "-range") == 0) {
			f_range = TRUE;
			range_first = atoi(argv[++i]);
			range_len = atoi(argv[++i]);
			cout << "-range " << range_first << " " << range_len << endl;
		}
		else if (strcmp(argv[i], "-select") == 0) {
			f_select = TRUE;
			select_label = argv[++i];
			cout << "-select " << select_label << endl;
		}
		else if (strcmp(argv[i], "-o1") == 0) {
			f_omit1 = TRUE;
			omit1 = atoi(argv[++i]);
			cout << "-o1 " << omit1 << endl;
		}
		else if (strcmp(argv[i], "-o2") == 0) {
			f_omit2 = TRUE;
			omit2 = atoi(argv[++i]);
			cout << "-o2 " << omit2 << endl;
		}
		if (strcmp(argv[i], "-D1_upper_bound_x0") == 0) {
			f_D1_upper_bound_x0 = TRUE;
			D1_upper_bound_x0 = atoi(argv[++i]);
			cout << "-D1_upper_bound_x0 " << D1_upper_bound_x0 << endl;
		}
		else if (strcmp(argv[i], "-reverse") == 0) {
			f_reverse = TRUE;
			cout << "-reverse" << endl;
		}
		else if (strcmp(argv[i], "-reverse_inverse") == 0) {
			f_reverse_inverse = TRUE;
			cout << "-reverse_inverse" << endl;
		}
		else if (strcmp(argv[i], "-nopacking") == 0) {
			f_use_packing_numbers = FALSE;
			cout << "-nopacking" << endl;
		}
		else if (strcmp(argv[i], "-dual_is_linear_space") == 0) {
			f_dual_is_linear_space = TRUE;
			cout << "-dual_is_linear_space" << endl;
		}
		else if (strcmp(argv[i], "-geometric_test") == 0) {
			f_do_the_geometric_test = TRUE;
			cout << "-geometric_test" << endl;
		}
		else if (strcmp(argv[i], "-once") == 0) {
			f_once = TRUE;
			cout << "-once" << endl;
		}
		else if (strcmp(argv[i], "-mckay") == 0) {
			f_use_mckay_solver = TRUE;
			cout << "-mckay" << endl;
		}
		else if (strcmp(argv[i], "-input_file") == 0) {
			f_input_file = TRUE;
			fname_in = argv[++i];
			cout << "-input_file" << fname_in << endl;
		}
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "tdo_refinement_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "tdo_refinement_description::read_arguments done" << endl;
	return i;
}



}}

