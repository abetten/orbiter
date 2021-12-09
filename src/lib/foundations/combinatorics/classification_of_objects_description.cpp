/*
 * classification_of_objects_description.cpp
 *
 *  Created on: Sep 13, 2020
 *      Author: betten
 */


#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {



classification_of_objects_description::classification_of_objects_description()
{

	f_label = FALSE;
	//std::string label;

	f_save_classification = FALSE;
	//std::string save_prefix;

	//f_report = FALSE;
	////std::string report_prefix;

	fixed_structure_order_list_sz = 0;
	//fixed_structure_order_list;

	f_max_TDO_depth = FALSE;
	max_TDO_depth = 0;

	f_classification_prefix = FALSE;
	//std::string classification_prefix;

	//f_save_incma_in_and_out = FALSE;
	//std::string save_incma_in_and_out_prefix;

	f_save_canonical_labeling = FALSE;

	f_save_ago = FALSE;

	f_save_transversal = FALSE;

	f_load_canonical_labeling = FALSE;

	f_load_ago = FALSE;

	f_save_cumulative_canonical_labeling = FALSE;

	f_save_cumulative_ago = FALSE;

	f_save_cumulative_data = FALSE;

	f_save_fibration = FALSE;
}

classification_of_objects_description::~classification_of_objects_description()
{

}

int classification_of_objects_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "classification_of_objects_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		cout << "classification_of_objects_description::read_arguments, next argument is " << argv[i] << endl;

		if (stringcmp(argv[i], "-label") == 0) {
			f_label = TRUE;
			label.assign(argv[++i]);
			cout << "-label" << label << endl;
		}

		else if (stringcmp(argv[i], "-save_classification") == 0) {
			f_save_classification = TRUE;
			save_prefix.assign(argv[++i]);
			cout << "-save_classification" << save_prefix << endl;
		}



#if 0
		else if (stringcmp(argv[i], "-all_k_subsets") == 0) {
			f_all_k_subsets = TRUE;
			k = strtoi(argv[++i]);
			cout << "-all_k_subsets " << k << endl;
		}
#endif

		else if (stringcmp(argv[i], "-fixed_structure_of_element_of_order") == 0) {
			fixed_structure_order_list[fixed_structure_order_list_sz] = strtoi(argv[++i]);
			cout << "-fixed_structure_of_element_of_order "
					<< fixed_structure_order_list[fixed_structure_order_list_sz] << endl;
			fixed_structure_order_list_sz++;
		}

		else if (stringcmp(argv[i], "-max_TDO_depth") == 0) {
			f_max_TDO_depth = TRUE;
			max_TDO_depth = strtoi(argv[++i]);
			cout << "-max_TDO_depth " << max_TDO_depth << endl;
		}

		else if (stringcmp(argv[i], "-classification_prefix") == 0) {
			f_classification_prefix = TRUE;
			classification_prefix.assign(argv[++i]);
			cout << "-classification_prefix " << classification_prefix << endl;
		}

#if 0
		else if (stringcmp(argv[i], "-save_incma_in_and_out") == 0) {
			f_save_incma_in_and_out = TRUE;
			save_incma_in_and_out_prefix.assign(argv[++i]);
			cout << "-save_incma_in_and_out" << save_incma_in_and_out_prefix << endl;
		}
#endif

		else if (stringcmp(argv[i], "-save_canonical_labeling") == 0) {
			f_save_canonical_labeling = TRUE;
			cout << "-save_canonical_labeling " << endl;
		}

		else if (stringcmp(argv[i], "-save_ago") == 0) {
			f_save_ago = TRUE;
			cout << "-save_ago " << endl;
		}

		else if (stringcmp(argv[i], "-save_transversal") == 0) {
			f_save_transversal = TRUE;
			cout << "-save_transversal " << endl;
		}

		else if (stringcmp(argv[i], "-load_canonical_labeling") == 0) {
			f_load_canonical_labeling = TRUE;
			//load_canonical_labeling_fname.assign(argv[++i]);
			cout << "-load_canonical_labeling " << endl;
		}

		else if (stringcmp(argv[i], "-load_ago") == 0) {
			f_load_ago = TRUE;
			//load_ago_fname.assign(argv[++i]);
			cout << "-load_ago " << endl;
		}

		else if (stringcmp(argv[i], "-save_cumulative_canonical_labeling") == 0) {
			f_save_cumulative_canonical_labeling = TRUE;
			cumulative_canonical_labeling_fname.assign(argv[++i]);
			cout << "-save_cumulative_canonical_labeling " << cumulative_canonical_labeling_fname << endl;
		}

		else if (stringcmp(argv[i], "-save_cumulative_ago") == 0) {
			f_save_cumulative_ago = TRUE;
			cumulative_ago_fname.assign(argv[++i]);
			cout << "-save_cumulative_ago " << cumulative_ago_fname << endl;
		}

		else if (stringcmp(argv[i], "-save_cumulative_data") == 0) {
			f_save_cumulative_data = TRUE;
			cumulative_data_fname.assign(argv[++i]);
			cout << "-save_cumulative_data " << cumulative_data_fname << endl;
		}

		else if (stringcmp(argv[i], "-save_fibration") == 0) {
			f_save_fibration = TRUE;
			fibration_fname.assign(argv[++i]);
			cout << "-save_fibration " << fibration_fname << endl;
		}

		else if (stringcmp(argv[i], "-end") == 0) {
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

	if (f_label) {
		cout << "-label" << label << endl;
	}
	if (f_save_classification) {
		cout << "-save_classification" << save_prefix << endl;
	}

#if 0
	if (stringcmp(argv[i], "-fixed_structure_of_element_of_order") == 0) {
		fixed_structure_order_list[fixed_structure_order_list_sz] = strtoi(argv[++i]);
		cout << "-fixed_structure_of_element_of_order "
				<< fixed_structure_order_list[fixed_structure_order_list_sz] << endl;
		fixed_structure_order_list_sz++;
	}
	if (f_report) {
		cout << "-report " << report_prefix << endl;
	}
#endif

	if (f_max_TDO_depth) {
		cout << "-max_TDO_depth " << max_TDO_depth << endl;
	}

	if (f_classification_prefix) {
		cout << "-classification_prefix " << classification_prefix << endl;
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

	if (f_load_canonical_labeling) {
		cout << "-load_canonical_labeling " << endl;
	}

	if (f_load_ago) {
		cout << "-load_ago " << endl;
	}

	if (f_save_cumulative_canonical_labeling) {
		cout << "-save_cumulative_canonical_labeling "
				<< cumulative_canonical_labeling_fname << endl;
	}

	if (f_save_cumulative_ago) {
		cout << "-save_cumulative_ago " << cumulative_ago_fname << endl;
	}

	if (f_save_cumulative_data) {
		cout << "-save_cumulative_data " << cumulative_data_fname << endl;
	}

	if (f_save_fibration) {
		cout << "-save_fibration " << fibration_fname << endl;
	}
}


}}



