/*
 * spread_classify_activity_description.cpp
 *
 *  Created on: Aug 31, 2022
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace spreads {

spread_classify_activity_description::spread_classify_activity_description()
{
	f_compute_starter = FALSE;
	starter_control = NULL;

	f_prepare_lifting_single_case = FALSE;
	prepare_lifting_single_case_case_number = 0;

	f_prepare_lifting_all_cases = FALSE;

	f_split = FALSE;
	split_r = 0;
	split_m = 1;

	f_isomorph = FALSE;
	//std::string prefix_classify;
	//std::string prefix_iso;
	Isomorph_arguments = NULL;

}

spread_classify_activity_description::~spread_classify_activity_description()
{
}

int spread_classify_activity_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "spread_classify_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-compute_starter") == 0) {
			f_compute_starter = TRUE;
			if (f_v) {
				cout << "-compute_starter " << endl;
			}
			starter_control = NEW_OBJECT(poset_classification::poset_classification_control);

			i += starter_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

		}
		else if (ST.stringcmp(argv[i], "-prepare_lifting_single_case") == 0) {
			f_prepare_lifting_single_case = TRUE;
			prepare_lifting_single_case_case_number = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-prepare_lifting_single_case " << prepare_lifting_single_case_case_number << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-prepare_lifting_all_cases") == 0) {
			f_prepare_lifting_all_cases = TRUE;
			if (f_v) {
				cout << "-prepare_lifting_all_cases " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-split") == 0) {
			f_split = TRUE;
			split_r = ST.strtoi(argv[++i]);
			split_m = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-split " << split_r << " " << split_m << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-isomorph") == 0) {
			f_isomorph = TRUE;
			//prefix_classify.assign(argv[++i]);
			//prefix_iso.assign(argv[++i]);
			Isomorph_arguments = NEW_OBJECT(layer4_classification::isomorph::isomorph_arguments);

			i += Isomorph_arguments->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);
			if (f_v) {
				cout << "-isomorph " << endl; //prefix_classify << " " << prefix_iso << endl;
				Isomorph_arguments->print();
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "spread_classify_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "spread_classify_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void spread_classify_activity_description::print()
{
	if (f_compute_starter) {
		cout << "-compute_starter " << endl;
	}
	if (f_prepare_lifting_single_case) {
		cout << "-prepare_lifting_single_case " << prepare_lifting_single_case_case_number << endl;
	}
	if (f_prepare_lifting_all_cases) {
		cout << "-prepare_lifting_all_cases " << endl;
	}
	if (f_split) {
		cout << "-split " << split_r << " " << split_m << endl;
	}
	if (f_isomorph) {
		cout << "-isomorph " << endl; // prefix_classify << " " << prefix_iso << endl;
		Isomorph_arguments->print();
	}
}


}}}

