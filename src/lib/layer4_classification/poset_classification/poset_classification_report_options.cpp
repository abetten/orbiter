/*
 * poset_classification_report_options.cpp
 *
 *  Created on: Jul 31, 2021
 *      Author: betten
 */





#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


poset_classification_report_options::poset_classification_report_options()
{
	f_select_orbits_by_level = false;
	select_orbits_by_level_level = 0;

	f_select_orbits_by_stabilizer_order = false;
	select_orbits_by_stabilizer_order_so = 0;

	f_select_orbits_by_stabilizer_order_multiple_of = false;
	select_orbits_by_stabilizer_order_so_multiple_of = 0;

	f_include_projective_stabilizer = false;

	f_draw_poset = false;
	f_type_aux = false;
	f_type_ordinary = false;
	f_type_tree = false;
	f_type_detailed = false;

	f_fname = false;
	//std::string fname;
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
			f_select_orbits_by_level = true;
			select_orbits_by_level_level = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-select_orbits_by_level "
						<< select_orbits_by_level_level << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-select_orbits_by_stabilizer_order") == 0) {
			f_select_orbits_by_stabilizer_order = true;
			select_orbits_by_stabilizer_order_so = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-select_orbits_by_stabilizer_order "
						<< select_orbits_by_stabilizer_order_so << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-select_orbits_by_stabilizer_order_multiple_of") == 0) {
			f_select_orbits_by_stabilizer_order_multiple_of = true;
			select_orbits_by_stabilizer_order_so_multiple_of = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-select_orbits_by_stabilizer_order_multiple_of "
						<< select_orbits_by_stabilizer_order_so_multiple_of << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-include_projective_stabilizer") == 0) {
			f_include_projective_stabilizer = true;
			if (f_v) {
				cout << "-include_projective_stabilizer" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = true;
			if (f_v) {
				cout << "-draw_poset" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-type_aux") == 0) {
			f_type_aux = true;
			if (f_v) {
				cout << "-type_aux" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-type_ordinary") == 0) {
			f_type_ordinary = true;
			if (f_v) {
				cout << "-type_ordinary" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-type_tree") == 0) {
			f_type_tree = true;
			if (f_v) {
				cout << "-type_tree" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-type_detailed") == 0) {
			f_type_detailed = true;
			if (f_v) {
				cout << "-type_detailed" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-fname") == 0) {
			f_fname = true;
			fname.assign(argv[++i]);
			if (f_v) {
				cout << "-fname" << fname << endl;
			}
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
	if (f_draw_poset) {
		cout << "-draw_poset" << endl;
	}
	if (f_type_aux) {
		cout << "-type_aux" << endl;
	}
	if (f_type_ordinary) {
		cout << "-type_ordinary" << endl;
	}
	if (f_type_tree) {
		cout << "-type_tree" << endl;
	}
	if (f_type_detailed) {
		cout << "-type_detailed" << endl;
	}
	if (f_fname) {
		cout << "-fname" << fname << endl;
	}
}

int poset_classification_report_options::is_selected_by_group_order(
		long int so)
{
	if (f_select_orbits_by_stabilizer_order) {
		if (select_orbits_by_stabilizer_order_so == so) {
			return true;
		}
		else {
			return false;
		}
	}
	else if (f_select_orbits_by_stabilizer_order_multiple_of) {
		if ((so % select_orbits_by_stabilizer_order_so_multiple_of) == 0) {
			return true;
		}
		else {
			return false;
		}
	}
	else {
		return true;
	}
}



}}}

