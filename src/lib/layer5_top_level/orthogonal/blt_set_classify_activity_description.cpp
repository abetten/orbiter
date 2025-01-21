/*
 * blt_set_classify_activity_description.cpp
 *
 *  Created on: Aug 2, 2022
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {

blt_set_classify_activity_description::blt_set_classify_activity_description()
{
	Record_birth();
	f_compute_starter = false;
	starter_control = NULL;

	f_poset_classification_activity = false;
	//std::string poset_classification_activity_label;

	f_create_graphs = false;

	f_split = false;
	split_r = 0;
	split_m = 1;

	f_isomorph = false;
	//std::string isomorph_label;

	f_build_db = false;
	f_read_solutions = false;
	f_compute_orbits = false;
	f_isomorph_testing = false;
	f_isomorph_report = false;

}

blt_set_classify_activity_description::~blt_set_classify_activity_description()
{
	Record_death();
}

int blt_set_classify_activity_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "blt_set_classify_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-compute_starter") == 0) {
			f_compute_starter = true;
			if (f_v) {
				cout << "-compute_starter " << endl;
			}
			starter_control = NEW_OBJECT(poset_classification::poset_classification_control);

			i += starter_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

		}
		else if (ST.stringcmp(argv[i], "-poset_classification_activity") == 0) {
			f_poset_classification_activity = true;
			poset_classification_activity_label.assign(argv[++i]);
			if (f_v) {
				cout << "-poset_classification_activity "
						<< poset_classification_activity_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-create_graphs") == 0) {
			f_create_graphs = true;
			if (f_v) {
				cout << "-create_graphs " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-split") == 0) {
			f_split = true;
			split_r = ST.strtoi(argv[++i]);
			split_m = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-split " << split_r << " " << split_m << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-isomorph") == 0) {
			f_isomorph = true;
			isomorph_label.assign(argv[++i]);
			if (f_v) {
				cout << "-isomorph " << isomorph_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-build_db") == 0) {
			f_build_db = true;
			if (f_v) {
				cout << "-build_db " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-read_solutions") == 0) {
			f_read_solutions = true;
			if (f_v) {
				cout << "-read_solutions " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-compute_orbits") == 0) {
			f_compute_orbits = true;
			if (f_v) {
				cout << "-compute_orbits " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-isomorph_testing") == 0) {
			f_isomorph_testing = true;
			if (f_v) {
				cout << "-isomorph_testing " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-isomorph_report") == 0) {
			f_isomorph_report = true;
			if (f_v) {
				cout << "-isomorph_report " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "blt_set_classify_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "blt_set_classify_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void blt_set_classify_activity_description::print()
{
	if (f_compute_starter) {
		cout << "-compute_starter " << endl;
	}
	if (f_poset_classification_activity) {
		cout << "-poset_classification_activity "
				<< poset_classification_activity_label << endl;
	}
	if (f_create_graphs) {
		cout << "-create_graphs " << endl;
	}
	if (f_split) {
		cout << "-split " << split_r << " " << split_m << endl;
	}
	if (f_isomorph) {
		cout << "-isomorph " << isomorph_label << endl;
	}
	if (f_build_db) {
		cout << "-build_db " << endl;
	}
	if (f_read_solutions) {
		cout << "-read_solutions " << endl;
	}
	if (f_compute_orbits) {
		cout << "-compute_orbits " << endl;
	}
	if (f_isomorph_testing) {
		cout << "-isomorph_testing " << endl;
	}
	if (f_isomorph_report) {
		cout << "-isomorph_report " << endl;
	}
}


}}}




