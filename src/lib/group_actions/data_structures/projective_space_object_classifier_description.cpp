/*
 * projective_space_object_classifier_description.cpp
 *
 *  Created on: Sep 13, 2020
 *      Author: betten
 */


#include "foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace group_actions {


projective_space_object_classifier_description::projective_space_object_classifier_description()
{
	f_input = FALSE;
	Data = NULL;


	f_save = FALSE;
	//std::string save_prefix;

	f_report = FALSE;
	//std::string report_prefix;

	fixed_structure_order_list_sz = 0;
	//fixed_structure_order_list;

	f_max_TDO_depth = FALSE;
	max_TDO_depth = 0;

	f_classification_prefix = FALSE;
	//std::string classification_prefix;

	f_save_incma_in_and_out = FALSE;
	//std::string save_incma_in_and_out_prefix;

	f_save_canonical_labeling = FALSE;

	f_save_ago = FALSE;

	f_load_canonical_labeling = FALSE;
	//std::string load_canonical_labeling_fname
}

projective_space_object_classifier_description::~projective_space_object_classifier_description()
{

}

int projective_space_object_classifier_description::read_arguments(
	int argc, const char **argv,
	int verbose_level)
{
	int i;

	cout << "projective_space_object_classifier_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		cout << "projective_space_object_classifier_description::read_arguments, next argument is " << argv[i] << endl;

		if (strcmp(argv[i], "-input") == 0) {
			f_input = TRUE;
			cout << "-input" << endl;
			Data = NEW_OBJECT(data_input_stream);
			i += Data->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);
			cout << "-input" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}


		else if (strcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			save_prefix.assign(argv[++i]);
			cout << "-save " << save_prefix << endl;
		}


#if 0
		else if (strcmp(argv[i], "-all_k_subsets") == 0) {
			f_all_k_subsets = TRUE;
			k = atoi(argv[++i]);
			cout << "-all_k_subsets " << k << endl;
		}
#endif

		else if (strcmp(argv[i], "-fixed_structure_of_element_of_order") == 0) {
			fixed_structure_order_list[fixed_structure_order_list_sz] = atoi(argv[++i]);
			cout << "-fixed_structure_of_element_of_order "
					<< fixed_structure_order_list[fixed_structure_order_list_sz] << endl;
			fixed_structure_order_list_sz++;
		}
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			report_prefix.assign(argv[++i]);
			cout << "-report " << report_prefix << endl;
		}

		else if (strcmp(argv[i], "-max_TDO_depth") == 0) {
			f_max_TDO_depth = TRUE;
			max_TDO_depth = atoi(argv[++i]);
			cout << "-max_TDO_depth " << max_TDO_depth << endl;
		}

		else if (strcmp(argv[i], "-classification_prefix") == 0) {
			f_classification_prefix = TRUE;
			classification_prefix.assign(argv[++i]);
			cout << "-classification_prefix " << classification_prefix << endl;
		}

		else if (strcmp(argv[i], "-save_incma_in_and_out") == 0) {
			f_save_incma_in_and_out = TRUE;
			save_incma_in_and_out_prefix.assign(argv[++i]);
			cout << "-save_incma_in_and_out" << save_incma_in_and_out_prefix << endl;
		}

		else if (strcmp(argv[i], "-save_canonical_labeling") == 0) {
			f_save_canonical_labeling = TRUE;
			cout << "-save_canonical_labeling " << endl;
		}

		else if (strcmp(argv[i], "-save_ago") == 0) {
			f_save_ago = TRUE;
			cout << "-save_ago " << endl;
		}

		else if (strcmp(argv[i], "-load_canonical_labeling") == 0) {
			f_load_canonical_labeling = TRUE;
			load_canonical_labeling_fname.assign(argv[++i]);
			cout << "-load_canonical_labeling " << load_canonical_labeling_fname << endl;
		}

		else if (strcmp(argv[i], "-end") == 0) {
			cout << "projective_space_object_classifier_description::read_arguments -end" << endl;
			break;
		}

		else {
			cout << "projective_space_object_classifier_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		cout << "projective_space_object_classifier_description::read_arguments looping, i=" << i << endl;
	} // next i
	cout << "projective_space_object_classifier_description::read_arguments done" << endl;
	return i + 1;
}


}}



