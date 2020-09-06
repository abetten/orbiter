/*
 * delandtsheer_doyen_description.cpp
 *
 *  Created on: May 24, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {

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

	// mask related test:
	nb_mask_tests = 0;

	f_singletons = FALSE;
	f_subgroup = FALSE;
	//subgroup_gens = NULL;
	//subgroup_order = NULL;

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
	int argc, const char **argv,
	int verbose_level)
{
	int i, j;

	cout << "delandtsheer_doyen_description::read_arguments" << endl;


	nb_mask_tests = 0;



	for (i = 0; i < argc; i++) {

#if 0
		if (argv[i][0] != '-') {
			continue;
			}
#endif

		if (strcmp(argv[i], "-group_label") == 0) {
			f_group_label = TRUE;
			group_label.assign(argv[++i]);
			cout << "-group_label " << group_label << endl;
		}
		else if (strcmp(argv[i], "-mask_label") == 0) {
			f_mask_label = TRUE;
			mask_label.assign(argv[++i]);
			cout << "-mask_label " << mask_label << endl;
		}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			depth = atoi(argv[++i]);
			cout << "-depth " << depth << endl;
		}
		else if (strcmp(argv[i], "-d1") == 0) {
			f_d1 = TRUE;
			d1 = atoi(argv[++i]);
			cout << "-d1 " << d1 << endl;
		}
		else if (strcmp(argv[i], "-d2") == 0) {
			f_d2 = TRUE;
			d2 = atoi(argv[++i]);
			cout << "-d2 " << d2 << endl;
		}
		else if (strcmp(argv[i], "-q1") == 0) {
			f_q1 = TRUE;
			q1 = atoi(argv[++i]);
			cout << "-q1 " << q1 << endl;
		}
		else if (strcmp(argv[i], "-q2") == 0) {
			f_q2 = TRUE;
			q2 = atoi(argv[++i]);
			cout << "-q2 " << q2 << endl;
		}
		else if (strcmp(argv[i], "-DDx") == 0) {
			DELANDTSHEER_DOYEN_X = atoi(argv[++i]);
			cout << "-DDx " << DELANDTSHEER_DOYEN_X << endl;
		}
		else if (strcmp(argv[i], "-DDy") == 0) {
			DELANDTSHEER_DOYEN_Y = atoi(argv[++i]);
			cout << "-DDy " << DELANDTSHEER_DOYEN_Y << endl;
		}
		else if (strcmp(argv[i], "-K") == 0) {
			f_K = TRUE;
			K = atoi(argv[++i]);
			cout << "-K " << K << endl;
		}
		else if (strcmp(argv[i], "-R") == 0) {
			f_R = TRUE;
			nb_row_types = atoi(argv[++i]);
			row_type = NEW_int(nb_row_types + 1);
			row_type[0] = 0;
			for (j = 1; j <= nb_row_types; j++) {
				row_type[j] = atoi(argv[++i]);
				//row_type_cur[j] = 0;
				}
			cout << "-R ";
			int_vec_print(cout, row_type + 1, nb_row_types);
			cout << endl;
		}
		else if (strcmp(argv[i], "-C") == 0) {
			f_C = TRUE;
			nb_col_types = atoi(argv[++i]);
			col_type = NEW_int(nb_col_types + 1);
			col_type[0] = 0;
			for (j = 1; j <= nb_col_types; j++) {
				col_type[j] = atoi(argv[++i]);
				//col_type_cur[j] = 0;
				}
			cout << "-C ";
			int_vec_print(cout, col_type + 1, nb_col_types);
			cout << endl;
		}
		else if (strcmp(argv[i], "-masktest") == 0) {
			const char *who;
			const char *what;

			mask_test_level[nb_mask_tests] = atoi(argv[++i]);
			who = argv[++i];
			what = argv[++i];
			mask_test_value[nb_mask_tests] = atoi(argv[++i]);

			if (strcmp(who, "x") == 0)
				mask_test_who[nb_mask_tests] = 1;
			else if (strcmp(who, "y") == 0)
				mask_test_who[nb_mask_tests] = 2;
			else if (strcmp(who, "x+y") == 0)
				mask_test_who[nb_mask_tests] = 3;
			else if (strcmp(who, "s") == 0)
				mask_test_who[nb_mask_tests] = 4;
			else {
				cout << "masktest: unknown 'who' option: " << who << endl;
				cout << "must be one of 'x', 'y', 'x+y' or 's'" << endl;
				exit(1);
				}
			if (strcmp(what, "eq") == 0)
				mask_test_what[nb_mask_tests] = 1;
			else if (strcmp(what, "ge") == 0)
				mask_test_what[nb_mask_tests] = 2;
			else if (strcmp(what, "le") == 0)
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
		else if (strcmp(argv[i], "-singletons") == 0) {
			f_singletons = TRUE;
			cout << "-singletons" << endl;
		}
		else if (strcmp(argv[i], "-subgroup") == 0) {
			f_subgroup = TRUE;
			subgroup_gens.assign(argv[++i]);
			subgroup_order.assign(argv[++i]);
			cout << "-subgroup " << subgroup_gens << " " << subgroup_order << endl;
		}
		else if (strcmp(argv[i], "-pair_search_control") == 0) {
			f_pair_search_control = TRUE;
			Pair_search_control = NEW_OBJECT(poset_classification_control);
			i += Pair_search_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-pair_search_control" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-search_control") == 0) {
			f_search_control = TRUE;
			Search_control = NEW_OBJECT(poset_classification_control);
			i += Search_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-search_control" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "delandtsheer_doyen_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i

	if (!f_pair_search_control) {
		Pair_search_control = NEW_OBJECT(poset_classification_control);
	}

	if (!f_search_control) {
		Search_control = NEW_OBJECT(poset_classification_control);
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




}}

