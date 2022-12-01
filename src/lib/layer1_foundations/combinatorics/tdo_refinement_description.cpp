/*
 * tdo_refinement_description.cpp
 *
 *  Created on: May 23, 2020
 *      Author: betten
 */





#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {


tdo_refinement_description::tdo_refinement_description()
{
	//fname_in;
	f_lambda3 = FALSE;
	lambda3 = 0;
	block_size = 0;
	f_scale = FALSE;
	scaling = 0;
	f_range = FALSE;
	range_first = 0;
	range_len = 1;
	f_select = FALSE;
	//select_label;
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
	//fname_in;

	Sol = NULL;
}

tdo_refinement_description::~tdo_refinement_description()
{

	if (Sol) {
		FREE_OBJECT(Sol);
	}
}

int tdo_refinement_description::read_arguments(int argc, std::string *argv, int verbose_level)
{
	int i;
	data_structures::string_tools ST;

	Sol = NEW_OBJECT(solution_file_data);
	Sol->nb_solution_files = 0;


	for (i = 0; i < argc - 1; i++) {
		if (ST.stringcmp(argv[i], "-lambda3") == 0) {
			f_lambda3 = TRUE;
			lambda3 = ST.strtoi(argv[++i]);
			block_size = ST.strtoi(argv[++i]);
			cout << "-lambda3 " << lambda3 << " " << block_size << endl;
		}
		if (ST.stringcmp(argv[i], "-scale") == 0) {
			f_scale = TRUE;
			scaling = ST.strtoi(argv[++i]);
			cout << "-scale " << scaling << endl;
		}
		if (ST.stringcmp(argv[i], "-solution") == 0) {
			//f_solution = TRUE;
			Sol->system_no.push_back(ST.strtoi(argv[++i]));
			string s;

			s.assign(argv[++i]);
			Sol->solution_file.push_back(s);

			cout << "-solution " << Sol->system_no[Sol->system_no.size() - 1]
				<< " " << Sol->solution_file[Sol->solution_file.size() - 1] << endl;
			Sol->nb_solution_files++;
		}
		else if (ST.stringcmp(argv[i], "-range") == 0) {
			f_range = TRUE;
			range_first = ST.strtoi(argv[++i]);
			range_len = ST.strtoi(argv[++i]);
			cout << "-range " << range_first << " " << range_len << endl;
		}
		else if (ST.stringcmp(argv[i], "-select") == 0) {
			f_select = TRUE;
			select_label.assign(argv[++i]);
			cout << "-select " << select_label << endl;
		}
		else if (ST.stringcmp(argv[i], "-o1") == 0) {
			f_omit1 = TRUE;
			omit1 = ST.strtoi(argv[++i]);
			cout << "-o1 " << omit1 << endl;
		}
		else if (ST.stringcmp(argv[i], "-o2") == 0) {
			f_omit2 = TRUE;
			omit2 = ST.strtoi(argv[++i]);
			cout << "-o2 " << omit2 << endl;
		}
		if (ST.stringcmp(argv[i], "-D1_upper_bound_x0") == 0) {
			f_D1_upper_bound_x0 = TRUE;
			D1_upper_bound_x0 = ST.strtoi(argv[++i]);
			cout << "-D1_upper_bound_x0 " << D1_upper_bound_x0 << endl;
		}
		else if (ST.stringcmp(argv[i], "-reverse") == 0) {
			f_reverse = TRUE;
			cout << "-reverse" << endl;
		}
		else if (ST.stringcmp(argv[i], "-reverse_inverse") == 0) {
			f_reverse_inverse = TRUE;
			cout << "-reverse_inverse" << endl;
		}
		else if (ST.stringcmp(argv[i], "-nopacking") == 0) {
			f_use_packing_numbers = FALSE;
			cout << "-nopacking" << endl;
		}
		else if (ST.stringcmp(argv[i], "-dual_is_linear_space") == 0) {
			f_dual_is_linear_space = TRUE;
			cout << "-dual_is_linear_space" << endl;
		}
		else if (ST.stringcmp(argv[i], "-geometric_test") == 0) {
			f_do_the_geometric_test = TRUE;
			cout << "-geometric_test" << endl;
		}
		else if (ST.stringcmp(argv[i], "-once") == 0) {
			f_once = TRUE;
			cout << "-once" << endl;
		}
		else if (ST.stringcmp(argv[i], "-mckay") == 0) {
			f_use_mckay_solver = TRUE;
			cout << "-mckay" << endl;
		}
		else if (ST.stringcmp(argv[i], "-input_file") == 0) {
			f_input_file = TRUE;
			fname_in.assign(argv[++i]);
			cout << "-input_file" << fname_in << endl;
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "tdo_refinement_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "tdo_refinement_description::read_arguments done" << endl;
	return i + 1;
}

void tdo_refinement_description::print()
{
	if (f_lambda3) {
		cout << "-lambda3 " << lambda3 << " " << block_size << endl;
	}
	if (f_scale) {
		cout << "-scale " << scaling << endl;
	}
#if 0
	if (stringcmp(argv[i], "-solution") == 0) {
		//f_solution = TRUE;
		Sol->system_no.push_back(strtoi(argv[++i]));
		string s;

		s.assign(argv[++i]);
		Sol->solution_file.push_back(s);

		cout << "-solution " << Sol->system_no[Sol->system_no.size() - 1]
			<< " " << Sol->solution_file[Sol->solution_file.size() - 1] << endl;
		Sol->nb_solution_files++;
	}
#endif
	if (f_range) {
		cout << "-range " << range_first << " " << range_len << endl;
	}
	if (f_select) {
		cout << "-select " << select_label << endl;
	}
	if (f_omit1) {
		cout << "-o1 " << omit1 << endl;
	}
	if (f_omit2) {
		cout << "-o2 " << omit2 << endl;
	}
	if (f_D1_upper_bound_x0) {
		cout << "-D1_upper_bound_x0 " << D1_upper_bound_x0 << endl;
	}
	if (f_reverse) {
		cout << "-reverse" << endl;
	}
	if (f_reverse_inverse) {
		cout << "-reverse_inverse" << endl;
	}
	if (!f_use_packing_numbers) {
		cout << "-nopacking" << endl;
	}
	if (f_dual_is_linear_space) {
		cout << "-dual_is_linear_space" << endl;
	}
	if (f_do_the_geometric_test) {
		cout << "-geometric_test" << endl;
	}
	if (f_once) {
		cout << "-once" << endl;
	}
	if (f_use_mckay_solver) {
		cout << "-mckay" << endl;
	}
	if (f_input_file) {
		cout << "-input_file" << fname_in << endl;
	}

}



}}}


