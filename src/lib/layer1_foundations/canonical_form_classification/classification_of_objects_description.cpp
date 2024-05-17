/*
 * classification_of_objects_description.cpp
 *
 *  Created on: Sep 13, 2020
 *      Author: betten
 */


#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace canonical_form_classification {



classification_of_objects_description::classification_of_objects_description()
{

	f_save_classification = false;
	//std::string save_prefix;

	f_max_TDO_depth = false;
	max_TDO_depth = 0;

	f_save_canonical_labeling = false;

	f_save_ago = false;

	f_save_transversal = false;

}

classification_of_objects_description::~classification_of_objects_description()
{

}

int classification_of_objects_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_objects_description::read_arguments" << endl;
	}
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "classification_of_objects_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (f_v) {
			cout << "classification_of_objects_description::read_arguments, next argument is " << argv[i] << endl;
		}

		if (ST.stringcmp(argv[i], "-save_classification") == 0) {
			f_save_classification = true;
			save_prefix.assign(argv[++i]);
			if (f_v) {
				cout << "-save_classification" << save_prefix << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-max_TDO_depth") == 0) {
			f_max_TDO_depth = true;
			max_TDO_depth = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-max_TDO_depth " << max_TDO_depth << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-save_canonical_labeling") == 0) {
			f_save_canonical_labeling = true;
			if (f_v) {
				cout << "-save_canonical_labeling " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-save_ago") == 0) {
			f_save_ago = true;
			if (f_v) {
				cout << "-save_ago " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-save_transversal") == 0) {
			f_save_transversal = true;
			if (f_v) {
				cout << "-save_transversal " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "projective_space_object_classifier_description::read_arguments -end" << endl;
			}
			break;
		}

		else {
			if (f_v) {
				cout << "classification_of_objects_description::read_arguments "
						"unrecognized option " << argv[i] << endl;
			}
			exit(1);
		}
		if (f_v) {
			cout << "classification_of_objects_description::read_arguments "
					"looping, i=" << i << endl;
		}
	} // next i
	if (f_v) {
		cout << "classification_of_objects_description::read_arguments done" << endl;
	}
	return i + 1;
}

void classification_of_objects_description::print()
{

	if (f_save_classification) {
		cout << "-save_classification " << save_prefix << endl;
	}

	if (f_max_TDO_depth) {
		cout << "-max_TDO_depth " << max_TDO_depth << endl;
	}

	if (f_save_canonical_labeling) {
		cout << "-save_canonical_labeling " << endl;
	}

	if (f_save_ago) {
		cout << "-save_ago " << endl;
	}

	if (f_save_transversal) {
		cout << "-save_transversal " << endl;
	}

}


}}}



