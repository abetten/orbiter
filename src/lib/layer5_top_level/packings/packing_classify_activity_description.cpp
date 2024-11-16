/*
 * packing_classify_activity_description.cpp
 *
 *  Created on: Nov 14, 2024
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace packings {

packing_classify_activity_description::packing_classify_activity_description()
{

	f_report = false;

	f_classify = false;
	//std::string classify_control_label;

	f_make_graph_of_disjoint_spreads = false;

	f_export_group_on_spreads = false;

}

packing_classify_activity_description::~packing_classify_activity_description()
{
}


int packing_classify_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "packing_classify_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		cout << "packing_classify_activity_description::read_arguments, "
				"next argument is " << argv[i] << endl;


		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;
			if (f_v) {
				cout << "-report" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-classify") == 0) {
			f_classify = true;
			classify_control_label.assign(argv[++i]);
			if (f_v) {
				cout << "-classify "
					<< classify_control_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-make_graph_of_disjoint_spreads") == 0) {
			f_make_graph_of_disjoint_spreads = true;
			if (f_v) {
				cout << "-make_graph_of_disjoint_spreads" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_group_on_spreads") == 0) {
			f_export_group_on_spreads = true;
			if (f_v) {
				cout << "-export_group_on_spreads" << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "packing_classify_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		if (f_v) {
			cout << "packing_classify_activity_description::read_arguments looping, i=" << i << endl;
		}
	} // next i
	if (f_v) {
		cout << "packing_classify_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void packing_classify_activity_description::print()
{
	if (f_report) {
		cout << "-report" << endl;
	}
	if (f_classify) {
		cout << "-classify "
				<< classify_control_label << endl;
	}
	if (f_make_graph_of_disjoint_spreads) {
		cout << "-make_graph_of_disjoint_spreads" << endl;
	}
	if (f_export_group_on_spreads) {
		cout << "-export_group_on_spreads" << endl;
	}
}



}}}


