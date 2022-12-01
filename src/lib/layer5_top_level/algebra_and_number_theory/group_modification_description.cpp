/*
 * group_modification_description.cpp
 *
 *  Created on: Dec 1, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;
using namespace orbiter::layer1_foundations;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


group_modification_description::group_modification_description()
{
	f_restricted_action = FALSE;
	//std::string restricted_action_set_text;

	f_on_k_subspaces = FALSE;
	on_k_subspaces_k = 0;

	f_on_k_subsets = FALSE;
	on_k_subsets_k = 0;

	f_on_wedge_product = FALSE;

	f_create_special_subgroup = FALSE;

	f_point_stabilizer = FALSE;
	point_stabilizer_index = 0;

	//std::vector<std::string> from;

}


group_modification_description::~group_modification_description()
{
}


int group_modification_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level > 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "group_modification_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-restricted_action") == 0) {
			f_restricted_action = TRUE;
			restricted_action_set_text.assign(argv[++i]);
			if (f_v) {
				cout << "-restricted_action " << restricted_action_set_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_k_subspaces") == 0) {
			f_on_k_subspaces = TRUE;
			on_k_subspaces_k = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-on_k_subspaces " << on_k_subspaces_k << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_k_subsets") == 0) {
			f_on_k_subsets = TRUE;
			on_k_subsets_k = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-on_k_subsets " << on_k_subsets_k << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_wedge_product") == 0) {
			f_on_wedge_product = TRUE;
			if (f_v) {
				cout << "-on_wedge_product " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-create_special_subgroup") == 0) {
			f_create_special_subgroup = TRUE;
			if (f_v) {
				cout << "-create_special_subgroup " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-point_stabilizer") == 0) {
			f_point_stabilizer = TRUE;
			point_stabilizer_index = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-point_stabilizer " << point_stabilizer_index << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-from") == 0) {
			std::string from_text;
			from_text.assign(argv[++i]);
			from.push_back(from_text);
			if (f_v) {
				cout << "-from " << from_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "group_modification_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i

	if (f_v) {
		cout << "group_modification_description::read_arguments done" << endl;
	}
	return i + 1;
}


void group_modification_description::print()
{
	if (f_restricted_action) {
		cout << "-restricted_action " << restricted_action_set_text << endl;
	}
	if (f_on_k_subspaces) {
		cout << "-on_k_subspaces " << on_k_subspaces_k << endl;
	}
	if (f_on_k_subsets) {
		cout << "-on_k_subsets " << on_k_subsets_k << endl;
	}
	if (f_on_wedge_product) {
		cout << "-on_wedge_product " << endl;
	}
	if (f_create_special_subgroup) {
		cout << "-create_special_subgroup " << endl;
	}
	if (f_point_stabilizer) {
		cout << "-point_stabilizer " << point_stabilizer_index << endl;
	}

	if (from.size()) {
		int i;
		for (i = 0; i < from.size(); i++) {
			cout << "-from " << from[i] << endl;
		}
	}
}



}}}

