/*
 * orbits_activity_description.cpp
 *
 *  Created on: Nov 8, 2022
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;
using namespace orbiter::layer1_foundations;

namespace orbiter {
namespace layer5_applications {
namespace orbits {


orbits_activity_description::orbits_activity_description()
{

	f_report = false;

	f_export_something = false;
	//std::string export_something_what;
	export_something_data1 = 0;

	f_export_trees = false;

	f_export_source_code = false;

	f_export_levels = false;
	export_levels_orbit_idx = 0;

	f_draw_tree = false;
	draw_tree_idx = 0;

	f_stabilizer = false;
	stabilizer_point = 0;

	f_stabilizer_of_orbit_rep = false;
	stabilizer_of_orbit_rep_orbit_idx = 0;

	f_Kramer_Mesner_matrix = false;
	Kramer_Mesner_t = 0;
	Kramer_Mesner_k = 0;

	f_recognize = false;
	//std::vector<std::string> recognize;

	f_transporter = false;
	//std::string transporter_label_of_set;

	f_report_options = false;
	report_options = NULL;

}


orbits_activity_description::~orbits_activity_description()
{
}


int orbits_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level > 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "orbits_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;
			if (f_v) {
				cout << "-report" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_trees") == 0) {
			f_export_trees = true;
			if (f_v) {
				cout << "-export_trees" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_source_code") == 0) {
			f_export_source_code = true;
			if (f_v) {
				cout << "-export_source_code" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_levels") == 0) {
			f_export_levels = true;
			export_levels_orbit_idx = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-export_levels " << export_levels_orbit_idx << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_something") == 0) {
			f_export_something = true;
			export_something_what.assign(argv[++i]);
			export_something_data1 = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-export_something " << export_something_what << " " << export_something_data1 << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-draw_tree") == 0) {
			f_draw_tree = true;
			draw_tree_idx = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-export_trees" << draw_tree_idx << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-stabilizer") == 0) {
			f_stabilizer = true;
			stabilizer_point = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-stabilizer " << stabilizer_point << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-stabilizer_of_orbit_rep") == 0) {
			f_stabilizer_of_orbit_rep = true;
			stabilizer_of_orbit_rep_orbit_idx = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-stabilizer_of_orbit_rep " << stabilizer_of_orbit_rep_orbit_idx << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Kramer_Mesner_matrix") == 0) {
			f_Kramer_Mesner_matrix = true;
			Kramer_Mesner_t = ST.strtoi(argv[++i]);
			Kramer_Mesner_k = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Kramer_Mesner_matrix " << Kramer_Mesner_t << " " << Kramer_Mesner_k << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-recognize") == 0) {

			f_recognize = true;
			string s;

			s.assign(argv[++i]);
			recognize.push_back(s);
			if (f_v) {
				cout << "-recognize " << recognize[recognize.size() - 1] << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-transporter") == 0) {

			f_transporter = true;
			transporter_label_of_set.assign(argv[++i]);
			if (f_v) {
				cout << "-transporter " << transporter_label_of_set << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-report_options") == 0) {
			f_report_options = true;

			report_options = NEW_OBJECT(poset_classification::poset_classification_report_options);
			if (f_v) {
				cout << "-report_options " << endl;
			}
			i += report_options->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -report_options " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}

			if (f_v) {
				cout << "-report_options" << endl;
				report_options->print();
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "orbits_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}


	} // next i

	if (f_v) {
		cout << "action_on_forms_description::read_arguments done" << endl;
	}
	return i + 1;
}


void orbits_activity_description::print()
{
	if (f_report) {
		cout << "-report" << endl;
	}
	if (f_export_trees) {
		cout << "-export_trees" << endl;
	}
	if (f_export_source_code) {
		cout << "-export_source_code" << endl;
	}
	if (f_export_levels) {
		cout << "-export_levels " << export_levels_orbit_idx << endl;
	}
	if (f_export_something) {
		cout << "-export_something " << export_something_what << " " << export_something_data1 << endl;
	}
	if (f_draw_tree) {
		cout << "-draw_tree " << draw_tree_idx << endl;
	}
	if (f_stabilizer) {
		cout << "-stabilizer " << stabilizer_point << endl;
	}
	if (f_stabilizer_of_orbit_rep) {
		cout << "-stabilizer_of_orbit_rep " << stabilizer_of_orbit_rep_orbit_idx << endl;
	}
	if (f_Kramer_Mesner_matrix) {
		cout << "-Kramer_Mesner_matrix t=" << Kramer_Mesner_t << " k=" << Kramer_Mesner_k << endl;
	}
	if (f_recognize) {
		int i;

		cout << "-recognize number of sets = " << recognize.size() << endl;
		for (i = 0; i < recognize.size(); i++) {
			cout << i << " : " << recognize[i] << endl;
		}
	}
	if (f_transporter) {
		cout << "-transporter " << transporter_label_of_set << endl;
	}
	if (f_report_options) {
		cout << "-report_options" << endl;
		report_options->print();
	}


}



}}}



