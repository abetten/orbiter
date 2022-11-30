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

	f_depth = FALSE;
	depth = 0;

	f_d1 = FALSE;
	d1 = 0;
	f_d2 = FALSE;
	d2 = 0;
	f_q1 = FALSE;
	q1 = 0;
	f_q2 = FALSE;
	q2 = 0;

	f_group_label = FALSE;
	//group_label = NULL;

	f_mask_label = FALSE;
	//mask_label = NULL;

	f_problem_label = FALSE;
	//std::string problem_label;

	DELANDTSHEER_DOYEN_X = -1;
	DELANDTSHEER_DOYEN_Y = -1;
	f_K = FALSE;
	K = 0;


	f_pair_search_control = FALSE;
	Pair_search_control = NULL;

	f_search_control = FALSE;
	Search_control = NULL;

	f_R = FALSE;
	nb_row_types = 0;
	row_type = NULL;     		// [nb_row_types + 1]

	f_C = FALSE;
	nb_col_types = 0;
	col_type = NULL;     		// [nb_col_types + 1]

	f_nb_orbits_on_blocks = FALSE;
	nb_orbits_on_blocks = 1;

	// mask related test:
	nb_mask_tests = 0;

	f_singletons = FALSE;
	f_subgroup = FALSE;
	//subgroup_gens = NULL;
	//subgroup_order = NULL;

	f_search_wrt_subgroup = FALSE;

}


delandtsheer_doyen_description::~delandtsheer_doyen_description()
{
	if (row_type) {
		FREE_int(row_type);
	}
	if (col_type) {
		FREE_int(col_type);
	}
	if (Pair_search_control) {
		FREE_OBJECT(Pair_search_control);
	}
	if (Search_control) {
		FREE_OBJECT(Search_control);
	}
}


int delandtsheer_doyen_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i, j;
	data_structures::string_tools ST;

	cout << "delandtsheer_doyen_description::read_arguments" << endl;


	nb_mask_tests = 0;



	for (i = 0; i < argc; i++) {


		if (ST.stringcmp(argv[i], "-group_label") == 0) {
			f_group_label = TRUE;
			group_label.assign(argv[++i]);
			cout << "-group_label " << group_label << endl;
		}
		else if (ST.stringcmp(argv[i], "-mask_label") == 0) {
			f_mask_label = TRUE;
			mask_label.assign(argv[++i]);
			cout << "-mask_label " << mask_label << endl;
		}
		else if (ST.stringcmp(argv[i], "-problem_label") == 0) {
			f_problem_label = TRUE;
			problem_label.assign(argv[++i]);
			cout << "-problem_label " << problem_label << endl;
		}
		else if (ST.stringcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			depth = ST.strtoi(argv[++i]);
			cout << "-depth " << depth << endl;
		}
		else if (ST.stringcmp(argv[i], "-d1") == 0) {
			f_d1 = TRUE;
			d1 = ST.strtoi(argv[++i]);
			cout << "-d1 " << d1 << endl;
		}
		else if (ST.stringcmp(argv[i], "-d2") == 0) {
			f_d2 = TRUE;
			d2 = ST.strtoi(argv[++i]);
			cout << "-d2 " << d2 << endl;
		}
		else if (ST.stringcmp(argv[i], "-q1") == 0) {
			f_q1 = TRUE;
			q1 = ST.strtoi(argv[++i]);
			cout << "-q1 " << q1 << endl;
		}
		else if (ST.stringcmp(argv[i], "-q2") == 0) {
			f_q2 = TRUE;
			q2 = ST.strtoi(argv[++i]);
			cout << "-q2 " << q2 << endl;
		}
		else if (ST.stringcmp(argv[i], "-DDx") == 0) {
			DELANDTSHEER_DOYEN_X = ST.strtoi(argv[++i]);
			cout << "-DDx " << DELANDTSHEER_DOYEN_X << endl;
		}
		else if (ST.stringcmp(argv[i], "-DDy") == 0) {
			DELANDTSHEER_DOYEN_Y = ST.strtoi(argv[++i]);
			cout << "-DDy " << DELANDTSHEER_DOYEN_Y << endl;
		}
		else if (ST.stringcmp(argv[i], "-K") == 0) {
			f_K = TRUE;
			K = ST.strtoi(argv[++i]);
			cout << "-K " << K << endl;
		}
		else if (ST.stringcmp(argv[i], "-R") == 0) {
			f_R = TRUE;
			nb_row_types = ST.strtoi(argv[++i]);
			row_type = NEW_int(nb_row_types + 1);
			row_type[0] = 0;
			for (j = 1; j <= nb_row_types; j++) {
				row_type[j] = ST.strtoi(argv[++i]);
				//row_type_cur[j] = 0;
				}
			cout << "-R ";
			Int_vec_print(cout, row_type + 1, nb_row_types);
			cout << endl;
		}
		else if (ST.stringcmp(argv[i], "-C") == 0) {
			f_C = TRUE;
			nb_col_types = ST.strtoi(argv[++i]);
			col_type = NEW_int(nb_col_types + 1);
			col_type[0] = 0;
			for (j = 1; j <= nb_col_types; j++) {
				col_type[j] = ST.strtoi(argv[++i]);
				//col_type_cur[j] = 0;
				}
			cout << "-C ";
			Int_vec_print(cout, col_type + 1, nb_col_types);
			cout << endl;
		}
		else if (ST.stringcmp(argv[i], "-nb_orbits_on_blocks") == 0) {
			f_nb_orbits_on_blocks = TRUE;
			nb_orbits_on_blocks = ST.strtoi(argv[++i]);
			cout << "-nb_orbits_on_blocks " << nb_orbits_on_blocks << endl;
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
			cout << "-masktest "
				<< mask_test_level[nb_mask_tests] << " "
				<< mask_test_who[nb_mask_tests] << " "
				<< mask_test_what[nb_mask_tests] << " "
				<< mask_test_value[nb_mask_tests] << endl;
			nb_mask_tests++;
			cout << "nb_mask_tests=" << nb_mask_tests << endl;
		}
		else if (ST.stringcmp(argv[i], "-singletons") == 0) {
			f_singletons = TRUE;
			cout << "-singletons" << endl;
		}
		else if (ST.stringcmp(argv[i], "-subgroup") == 0) {
			f_subgroup = TRUE;
			subgroup_gens.assign(argv[++i]);
			subgroup_order.assign(argv[++i]);
			cout << "-subgroup " << subgroup_gens << " " << subgroup_order << endl;
		}
		else if (ST.stringcmp(argv[i], "-pair_search_control") == 0) {
			f_pair_search_control = TRUE;
			Pair_search_control = NEW_OBJECT(poset_classification::poset_classification_control);
			i += Pair_search_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-pair_search_control" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-search_control") == 0) {
			f_search_control = TRUE;
			Search_control = NEW_OBJECT(poset_classification::poset_classification_control);
			i += Search_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-search_control" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-search_wrt_subgroup") == 0) {
			f_search_wrt_subgroup = TRUE;
			cout << "-search_wrt_subgroup " << endl;
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "delandtsheer_doyen_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i

	if (!f_pair_search_control) {
		Pair_search_control = NEW_OBJECT(poset_classification::poset_classification_control);
	}

	if (!f_search_control) {
		Search_control = NEW_OBJECT(poset_classification::poset_classification_control);
	}
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
	if (f_depth) {
		cout << "-depth " << depth << endl;
	}
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
	if (f_singletons) {
		cout << "-singletons" << endl;
	}
	if (f_subgroup) {
		cout << "-subgroup " << subgroup_gens << " " << subgroup_order << endl;
	}
	if (f_pair_search_control) {
		cout << "-pair_search_control" << endl;
		Pair_search_control->print();
	}
	if (f_search_control) {
		cout << "-search_control" << endl;
		Search_control->print();
	}
	if (f_search_wrt_subgroup) {
		cout << "-search_wrt_subgroup " << endl;
	}

}




}}}


