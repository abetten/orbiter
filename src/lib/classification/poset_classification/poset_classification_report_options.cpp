/*
 * poset_classification_report_options.cpp
 *
 *  Created on: Jul 31, 2021
 *      Author: betten
 */





#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

using namespace std;

namespace orbiter {
namespace classification {


poset_classification_report_options::poset_classification_report_options()
{
	f_select_orbits_by_level = FALSE;
	select_orbits_by_level_level = 0;
	f_select_orbits_by_stabilizer_order = FALSE;
	select_orbits_by_stabilizer_order_so = 0;
	f_select_orbits_by_stabilizer_order_multiple_of = FALSE;
	select_orbits_by_stabilizer_order_so_multiple_of = 0;
	f_include_projective_stabilizer = FALSE;
}

poset_classification_report_options::~poset_classification_report_options()
{
}


int poset_classification_report_options::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "poset_classification_report_options::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-select_orbits_by_level") == 0) {
			f_select_orbits_by_level = TRUE;
			select_orbits_by_level_level = ST.strtoi(argv[++i]);
		}
		else if (ST.stringcmp(argv[i], "-select_orbits_by_stabilizer_order") == 0) {
			f_select_orbits_by_stabilizer_order = TRUE;
			select_orbits_by_stabilizer_order_so = ST.strtoi(argv[++i]);
		}
		else if (ST.stringcmp(argv[i], "-select_orbits_by_stabilizer_order_multiple_of") == 0) {
			f_select_orbits_by_stabilizer_order_multiple_of = TRUE;
			select_orbits_by_stabilizer_order_so_multiple_of = ST.strtoi(argv[++i]);
		}
		else if (ST.stringcmp(argv[i], "-include_projective_stabilizer") == 0) {
			f_include_projective_stabilizer = TRUE;
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "poset_classification_report_options::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}

	} // next i
	if (f_v) {
		cout << "poset_classification_report_options::read_arguments done" << endl;
	}
	return i + 1;
}

void poset_classification_report_options::print()
{
	//cout << "poset_classification_report_options::print:" << endl;



	if (f_select_orbits_by_level) {
		cout << "-select_orbits_by_level " << select_orbits_by_level_level << endl;
	}
	if (f_select_orbits_by_stabilizer_order) {
		cout << "-select_orbits_by_stabilizer_order " << select_orbits_by_stabilizer_order_so << endl;
	}
	if (f_select_orbits_by_stabilizer_order_multiple_of) {
		cout << "-select_orbits_by_stabilizer_order_multiple_of " << select_orbits_by_stabilizer_order_so_multiple_of << endl;
	}
	if (f_include_projective_stabilizer) {
		cout << "-include_projective_stabilizer" << endl;
	}

}

int poset_classification_report_options::is_selected_by_group_order(long int so)
{
	if (f_select_orbits_by_stabilizer_order) {
		if (select_orbits_by_stabilizer_order_so == so) {
			return TRUE;
		}
		else {
			return FALSE;
		}
	}
	else if (f_select_orbits_by_stabilizer_order_multiple_of) {
		if ((so % select_orbits_by_stabilizer_order_so_multiple_of) == 0) {
			return TRUE;
		}
		else {
			return FALSE;
		}
	}
	else {
		return TRUE;
	}
}



}}

