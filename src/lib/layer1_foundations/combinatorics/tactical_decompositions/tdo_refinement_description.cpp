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
namespace tactical_decompositions {


tdo_refinement_description::tdo_refinement_description()
{
	Record_birth();
	//fname_in;
	f_lambda3 = false;
	lambda3 = 0;
	block_size = 0;
	f_scale = false;
	scaling = 0;
	f_range = false;
	range_first = 0;
	range_len = 1;
	f_select = false;
	//select_label;
	f_omit1 = false;
	omit1 = 0;
	f_omit2 = false;
	omit2 = 0;
	f_D1_upper_bound_x0 = false;
	D1_upper_bound_x0 = 0;
	f_reverse = false;
	f_reverse_inverse = false;
	f_use_packing_numbers = false;
	f_dual_is_linear_space = false;
	f_do_the_geometric_test = false;
	f_once = false;
	f_use_mckay_solver = false;
	f_input_file = false;
	//fname_in;

	Sol = NULL;
}

tdo_refinement_description::~tdo_refinement_description()
{
	Record_death();

	if (Sol) {
		FREE_OBJECT(Sol);
	}
}

int tdo_refinement_description::read_arguments(
		int argc, std::string *argv, int verbose_level)
{
	int i;
	other::data_structures::string_tools ST;
	int f_v = (verbose_level >= 1);

	Sol = NEW_OBJECT(solution_file_data);
	Sol->nb_solution_files = 0;


	for (i = 0; i < argc - 1; i++) {
		if (ST.stringcmp(argv[i], "-lambda3") == 0) {
			f_lambda3 = true;
			lambda3 = ST.strtoi(argv[++i]);
			block_size = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-lambda3 " << lambda3 << " " << block_size << endl;
			}
		}
		if (ST.stringcmp(argv[i], "-scale") == 0) {
			f_scale = true;
			scaling = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-scale " << scaling << endl;
			}
		}
		if (ST.stringcmp(argv[i], "-solution") == 0) {
			//f_solution = true;
			Sol->system_no.push_back(ST.strtoi(argv[++i]));
			string s;

			s.assign(argv[++i]);
			Sol->solution_file.push_back(s);

			if (f_v) {
				cout << "-solution " << Sol->system_no[Sol->system_no.size() - 1]
					<< " " << Sol->solution_file[Sol->solution_file.size() - 1] << endl;
				Sol->nb_solution_files++;
			}
		}
		else if (ST.stringcmp(argv[i], "-range") == 0) {
			f_range = true;
			range_first = ST.strtoi(argv[++i]);
			range_len = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-range " << range_first << " " << range_len << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-select") == 0) {
			f_select = true;
			select_label.assign(argv[++i]);
			if (f_v) {
				cout << "-select " << select_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-o1") == 0) {
			f_omit1 = true;
			omit1 = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-o1 " << omit1 << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-o2") == 0) {
			f_omit2 = true;
			omit2 = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-o2 " << omit2 << endl;
			}
		}
		if (ST.stringcmp(argv[i], "-D1_upper_bound_x0") == 0) {
			f_D1_upper_bound_x0 = true;
			D1_upper_bound_x0 = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-D1_upper_bound_x0 " << D1_upper_bound_x0 << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-reverse") == 0) {
			f_reverse = true;
			if (f_v) {
				cout << "-reverse" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-reverse_inverse") == 0) {
			f_reverse_inverse = true;
			if (f_v) {
				cout << "-reverse_inverse" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-nopacking") == 0) {
			f_use_packing_numbers = false;
			if (f_v) {
				cout << "-nopacking" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-dual_is_linear_space") == 0) {
			f_dual_is_linear_space = true;
			if (f_v) {
				cout << "-dual_is_linear_space" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-geometric_test") == 0) {
			f_do_the_geometric_test = true;
			if (f_v) {
				cout << "-geometric_test" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-once") == 0) {
			f_once = true;
			if (f_v) {
				cout << "-once" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-mckay") == 0) {
			f_use_mckay_solver = true;
			if (f_v) {
				cout << "-mckay" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-input_file") == 0) {
			f_input_file = true;
			fname_in.assign(argv[++i]);
			if (f_v) {
				cout << "-input_file" << fname_in << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "tdo_refinement_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "tdo_refinement_description::read_arguments done" << endl;
	}
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
		//f_solution = true;
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
		cout << "-input_file " << fname_in << endl;
	}

}



}}}}



