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

#if 0
	f_label = false;
	//std::string label;
#endif

	f_save_classification = false;
	//std::string save_prefix;

	f_max_TDO_depth = false;
	max_TDO_depth = 0;

#if 0
	f_classification_prefix = false;
	//std::string classification_prefix;
#endif

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
	int i;
	data_structures::string_tools ST;

	cout << "classification_of_objects_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		cout << "classification_of_objects_description::read_arguments, next argument is " << argv[i] << endl;

#if 0
		if (ST.stringcmp(argv[i], "-label") == 0) {
			f_label = true;
			label.assign(argv[++i]);
			cout << "-label" << label << endl;
		}
#endif

		if (ST.stringcmp(argv[i], "-save_classification") == 0) {
			f_save_classification = true;
			save_prefix.assign(argv[++i]);
			cout << "-save_classification" << save_prefix << endl;
		}
#if 0
		else if (stringcmp(argv[i], "-fixed_structure_of_element_of_order") == 0) {
			fixed_structure_order_list[fixed_structure_order_list_sz] = strtoi(argv[++i]);
			cout << "-fixed_structure_of_element_of_order "
					<< fixed_structure_order_list[fixed_structure_order_list_sz] << endl;
			fixed_structure_order_list_sz++;
		}
#endif

		else if (ST.stringcmp(argv[i], "-max_TDO_depth") == 0) {
			f_max_TDO_depth = true;
			max_TDO_depth = ST.strtoi(argv[++i]);
			cout << "-max_TDO_depth " << max_TDO_depth << endl;
		}

#if 0
		else if (ST.stringcmp(argv[i], "-classification_prefix") == 0) {
			f_classification_prefix = true;
			classification_prefix.assign(argv[++i]);
			cout << "-classification_prefix " << classification_prefix << endl;
		}
#endif

		else if (ST.stringcmp(argv[i], "-save_canonical_labeling") == 0) {
			f_save_canonical_labeling = true;
			cout << "-save_canonical_labeling " << endl;
		}

		else if (ST.stringcmp(argv[i], "-save_ago") == 0) {
			f_save_ago = true;
			cout << "-save_ago " << endl;
		}

		else if (ST.stringcmp(argv[i], "-save_transversal") == 0) {
			f_save_transversal = true;
			cout << "-save_transversal " << endl;
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "projective_space_object_classifier_description::read_arguments -end" << endl;
			break;
		}

		else {
			cout << "classification_of_objects_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		cout << "classification_of_objects_description::read_arguments looping, i=" << i << endl;
	} // next i
	cout << "classification_of_objects_description::read_arguments done" << endl;
	return i + 1;
}

void classification_of_objects_description::print()
{

#if 0
	if (f_label) {
		cout << "-label " << label << endl;
	}
#endif
	if (f_save_classification) {
		cout << "-save_classification " << save_prefix << endl;
	}

	if (f_max_TDO_depth) {
		cout << "-max_TDO_depth " << max_TDO_depth << endl;
	}

#if 0
	if (f_classification_prefix) {
		cout << "-classification_prefix " << classification_prefix << endl;
	}
#endif

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



