/*
 * spread_classify_description.cpp
 *
 *  Created on: Aug 31, 2022
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace spreads {

spread_classify_description::spread_classify_description()
{
	f_projective_space = false;
	//std::string projective_space_label;

	f_starter_size = false;
	starter_size = 0;

	f_k = false;
	k = 0;

	f_poset_classification_control = false;
	//std::string poset_classification_control_label;
	//Control = NULL;

	f_output_prefix = false;
	//std::string output_prefix;

	f_recoordinatize = false;
}

spread_classify_description::~spread_classify_description()
{
}

int spread_classify_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "spread_classify_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-projective_space") == 0) {
			f_projective_space = true;
			projective_space_label.assign(argv[++i]);
			if (f_v) {
				cout << "-projective_space " << projective_space_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-starter_size") == 0) {
			f_starter_size = true;
			starter_size = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-starter_size " << starter_size << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-k") == 0) {
			f_k = true;
			k = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-k " << k << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-poset_classification_control") == 0) {
			f_poset_classification_control = true;
			poset_classification_control_label.assign(argv[++i]);
#if 0
			Control = NEW_OBJECT(poset_classification::poset_classification_control);
			if (f_v) {
				cout << "-poset_classification_control " << endl;
			}
			i += Control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);
#endif

			if (f_v) {
				cout << "-poset_classification_control " << poset_classification_control_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-output_prefix") == 0) {
			f_output_prefix = true;
			output_prefix.assign(argv[++i]);
			if (f_v) {
				cout << "-output_prefix " << output_prefix << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-recoordinatize") == 0) {
			f_recoordinatize = true;
			if (f_v) {
				cout << "-recoordinatize " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "spread_classify_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "spread_classify_description::read_arguments done" << endl;
	}
	return i + 1;
}

void spread_classify_description::print()
{
	if (f_projective_space) {
		cout << "-projective_space " << projective_space_label << endl;
	}
	if (f_starter_size) {
		cout << "-starter_size " << starter_size << endl;
	}
	if (f_k) {
		cout << "-k " << k << endl;
	}
	if (f_poset_classification_control) {
		cout << "-poset_classification_control " << poset_classification_control_label << endl;
	}
	if (f_output_prefix) {
		cout << "-output_prefix " << output_prefix << endl;
	}
	if (f_recoordinatize) {
		cout << "-recoordinatize " << endl;
	}
}


}}}




