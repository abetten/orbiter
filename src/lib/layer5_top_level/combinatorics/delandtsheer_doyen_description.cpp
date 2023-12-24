/*
 * delandtsheer_doyen_description.cpp
 *
 *  Created on: May 24, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {


delandtsheer_doyen_description::delandtsheer_doyen_description()
{
#if 0
	f_depth = false;
	depth = 0;
#endif

	f_d1 = false;
	d1 = 0;
	f_d2 = false;
	d2 = 0;
	f_q1 = false;
	q1 = 0;
	f_q2 = false;
	q2 = 0;

	f_group_label = false;
	//group_label = NULL;

	f_mask_label = false;
	//mask_label = NULL;

	f_problem_label = false;
	//std::string problem_label;

	DELANDTSHEER_DOYEN_X = -1;
	DELANDTSHEER_DOYEN_Y = -1;
	f_K = false;
	K = 0;


	f_pair_search_control = false;
	std::string pair_search_control_label;
	//Pair_search_control = NULL;

	f_search_control = false;
	std::string search_control_label;
	//Search_control = NULL;

	f_R = false;
	nb_row_types = 0;
	row_type = NULL;     		// [nb_row_types + 1]

	f_C = false;
	nb_col_types = 0;
	col_type = NULL;     		// [nb_col_types + 1]

	f_nb_orbits_on_blocks = false;
	nb_orbits_on_blocks = 1;

	// mask related test:
	nb_mask_tests = 0;

	f_search_partial_base_lines = false;

	f_singletons = false;
	singletons_starter_size = 0;

	f_subgroup = false;
	//subgroup_gens = NULL;
	//subgroup_order = NULL;

	f_search_wrt_subgroup = false;

}


delandtsheer_doyen_description::~delandtsheer_doyen_description()
{
	if (row_type) {
		FREE_int(row_type);
	}
	if (col_type) {
		FREE_int(col_type);
	}
}


int delandtsheer_doyen_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);


	int i, j;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "delandtsheer_doyen_description::read_arguments" << endl;
	}


	nb_mask_tests = 0;



	for (i = 0; i < argc; i++) {


		if (ST.stringcmp(argv[i], "-group_label") == 0) {
			f_group_label = true;
			group_label.assign(argv[++i]);
			if (f_v) {
				cout << "-group_label " << group_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-mask_label") == 0) {
			f_mask_label = true;
			mask_label.assign(argv[++i]);
			if (f_v) {
				cout << "-mask_label " << mask_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-problem_label") == 0) {
			f_problem_label = true;
			problem_label.assign(argv[++i]);
			if (f_v) {
				cout << "-problem_label " << problem_label << endl;
			}
		}
#if 0
		else if (ST.stringcmp(argv[i], "-depth") == 0) {
			f_depth = true;
			depth = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-depth " << depth << endl;
			}
		}
#endif
		else if (ST.stringcmp(argv[i], "-d1") == 0) {
			f_d1 = true;
			d1 = ST.strtoi(argv[++i]);
			cout << "-d1 " << d1 << endl;
		}
		else if (ST.stringcmp(argv[i], "-d2") == 0) {
			f_d2 = true;
			d2 = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-d2 " << d2 << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-q1") == 0) {
			f_q1 = true;
			q1 = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-q1 " << q1 << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-q2") == 0) {
			f_q2 = true;
			q2 = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-q2 " << q2 << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-DDx") == 0) {
			DELANDTSHEER_DOYEN_X = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-DDx " << DELANDTSHEER_DOYEN_X << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-DDy") == 0) {
			DELANDTSHEER_DOYEN_Y = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-DDy " << DELANDTSHEER_DOYEN_Y << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-K") == 0) {
			f_K = true;
			K = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-K " << K << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-R") == 0) {
			f_R = true;
			nb_row_types = ST.strtoi(argv[++i]);
			row_type = NEW_int(nb_row_types + 1);
			row_type[0] = 0;
			for (j = 1; j <= nb_row_types; j++) {
				row_type[j] = ST.strtoi(argv[++i]);
				//row_type_cur[j] = 0;
			}
			if (f_v) {
				cout << "-R ";
				Int_vec_print(cout, row_type + 1, nb_row_types);
				cout << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-C") == 0) {
			f_C = true;
			nb_col_types = ST.strtoi(argv[++i]);
			col_type = NEW_int(nb_col_types + 1);
			col_type[0] = 0;
			for (j = 1; j <= nb_col_types; j++) {
				col_type[j] = ST.strtoi(argv[++i]);
				//col_type_cur[j] = 0;
			}
			if (f_v) {
				cout << "-C ";
				Int_vec_print(cout, col_type + 1, nb_col_types);
				cout << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-nb_orbits_on_blocks") == 0) {
			f_nb_orbits_on_blocks = true;
			nb_orbits_on_blocks = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-nb_orbits_on_blocks " << nb_orbits_on_blocks << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-masktest") == 0) {
			string who;
			string what;

			mask_test_level[nb_mask_tests] = ST.strtoi(argv[++i]);
			who.assign(argv[++i]);
			what.assign(argv[++i]);
			mask_test_value[nb_mask_tests] = ST.strtoi(argv[++i]);

			if (ST.stringcmp(who, "x") == 0)
				mask_test_who[nb_mask_tests] = 1;
			else if (ST.stringcmp(who, "y") == 0)
				mask_test_who[nb_mask_tests] = 2;
			else if (ST.stringcmp(who, "x+y") == 0)
				mask_test_who[nb_mask_tests] = 3;
			else if (ST.stringcmp(who, "s") == 0)
				mask_test_who[nb_mask_tests] = 4;
			else {
				cout << "masktest: unknown 'who' option: " << who << endl;
				cout << "must be one of 'x', 'y', 'x+y' or 's'" << endl;
				exit(1);
				}
			if (ST.stringcmp(what, "eq") == 0)
				mask_test_what[nb_mask_tests] = 1;
			else if (ST.stringcmp(what, "ge") == 0)
				mask_test_what[nb_mask_tests] = 2;
			else if (ST.stringcmp(what, "le") == 0)
				mask_test_what[nb_mask_tests] = 3;
			else {
				cout << "masktest: unknown 'what' option: " << who << endl;
				cout << "must be one of 'eq', 'ge' or 'le'" << endl;
				exit(1);
				}
			if (f_v) {
				cout << "-masktest "
					<< mask_test_level[nb_mask_tests] << " "
					<< mask_test_who[nb_mask_tests] << " "
					<< mask_test_what[nb_mask_tests] << " "
					<< mask_test_value[nb_mask_tests] << endl;
			}
			nb_mask_tests++;
			if (f_v) {
				cout << "nb_mask_tests=" << nb_mask_tests << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-search_partial_base_lines") == 0) {
			f_search_partial_base_lines = true;
			if (f_v) {
				cout << "-search_partial_base_lines " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-singletons") == 0) {
			f_singletons = true;
			singletons_starter_size = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-singletons " << singletons_starter_size << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subgroup") == 0) {
			f_subgroup = true;
			subgroup_gens.assign(argv[++i]);
			subgroup_order.assign(argv[++i]);
			if (f_v) {
				cout << "-subgroup " << subgroup_gens << " " << subgroup_order << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-pair_search_control") == 0) {
			f_pair_search_control = true;
			pair_search_control_label.assign(argv[++i]);
			if (f_v) {
				cout << "-pair_search_control " << pair_search_control_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-search_control") == 0) {
			f_search_control = true;
			search_control_label.assign(argv[++i]);
			if (f_v) {
				cout << "-search_control " << search_control_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-search_wrt_subgroup") == 0) {
			f_search_wrt_subgroup = true;
			if (f_v) {
				cout << "-search_wrt_subgroup " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "delandtsheer_doyen_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i


	if (!f_group_label) {
		cout << "please use -group_label <label> to specify a label for the group used";
		exit(1);
	}
	if (!f_mask_label) {
		cout << "please use -mask_label <label> to specify a label for the mask used";
		exit(1);
	}

	cout << "delandtsheer_doyen_description::read_arguments done" << endl;
	return i + 1;
}

void delandtsheer_doyen_description::print()
{

	if (f_group_label) {
		cout << "-group_label " << group_label << endl;
	}
	if (f_mask_label) {
		cout << "-mask_label " << mask_label << endl;
	}
	if (f_problem_label) {
		cout << "-problem_label " << problem_label << endl;
	}
#if 0
	if (f_depth) {
		cout << "-depth " << depth << endl;
	}
#endif
	if (f_d1) {
		cout << "-d1 " << d1 << endl;
	}
	if (f_d2) {
		cout << "-d2 " << d2 << endl;
	}
	if (f_q1) {
		cout << "-q1 " << q1 << endl;
	}
	if (f_q2) {
		cout << "-q2 " << q2 << endl;
	}
	cout << "-DDx " << DELANDTSHEER_DOYEN_X << endl;
	cout << "-DDy " << DELANDTSHEER_DOYEN_Y << endl;
	if (f_K) {
		cout << "-K " << K << endl;
	}
	if (f_R) {
		cout << "-R ";
		Int_vec_print(cout, row_type + 1, nb_row_types);
		cout << endl;
	}
	if (f_C) {
		cout << "-C ";
		Int_vec_print(cout, col_type + 1, nb_col_types);
		cout << endl;
	}
	if (f_nb_orbits_on_blocks) {
		cout << "-nb_orbits_on_blocks " << nb_orbits_on_blocks << endl;
	}
	for (int i = 0; i < nb_mask_tests; i++) {
		cout << "-masktest "
			<< mask_test_level[i] << " "
			<< mask_test_who[i] << " "
			<< mask_test_what[i] << " "
			<< mask_test_value[i] << endl;

	}
	if (f_search_partial_base_lines) {
		cout << "-search_partial_base_lines " << endl;
	}
	if (f_singletons) {
		cout << "-singletons " << singletons_starter_size << endl;
	}
	if (f_subgroup) {
		cout << "-subgroup " << subgroup_gens << " " << subgroup_order << endl;
	}
	if (f_pair_search_control) {
		cout << "-pair_search_control " << pair_search_control_label << endl;
	}
	if (f_search_control) {
		cout << "-search_control " << search_control_label << endl;
	}
	if (f_search_wrt_subgroup) {
		cout << "-search_wrt_subgroup " << endl;
	}

}




}}}


