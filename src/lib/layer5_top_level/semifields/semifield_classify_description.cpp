/*
 * semifield_classify_description.cpp
 *
 *  Created on: Nov 22, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace semifields {



semifield_classify_description::semifield_classify_description()
{
	f_order = FALSE;
	order = 0;
	f_dim_over_kernel = FALSE;
	dim_over_kernel = 0;
	f_prefix = FALSE;
	//prefix = "";
	f_orbits_light = FALSE;
	f_test_semifield = FALSE;
	//test_semifield_data = NULL;
	f_identify_semifield = FALSE;
	//identify_semifield_data = NULL;
	f_identify_semifields_from_file = FALSE;
	//identify_semifields_from_file_fname = NULL;
	f_load_classification = FALSE;
	f_report = FALSE;
	f_decomposition_matrix_level_3 = FALSE;
	f_level_two_prefix = FALSE;
	//level_two_prefix;
	f_level_three_prefix = FALSE;
	//level_three_prefix;
}

semifield_classify_description::~semifield_classify_description()
{

}


int semifield_classify_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;
	data_structures::string_tools ST;

	cout << "semifield_classify_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-order") == 0) {
			f_order = TRUE;
			order = ST.strtoi(argv[++i]);
			cout << "-order " << order << endl;
		}
		else if (ST.stringcmp(argv[i], "-dim_over_kernel") == 0) {
			f_dim_over_kernel = TRUE;
			dim_over_kernel = ST.strtoi(argv[++i]);
			cout << "-dim_over_kernel " << dim_over_kernel << endl;
		}
		else if (ST.stringcmp(argv[i], "-prefix") == 0) {
			f_prefix = TRUE;
			prefix.assign(argv[++i]);
			cout << "-prefix " << prefix << endl;
		}
		else if (ST.stringcmp(argv[i], "-orbits_light") == 0) {
			f_orbits_light = TRUE;
			cout << "-orbits_light " << endl;
		}
		else if (ST.stringcmp(argv[i], "-test_semifield") == 0) {
			f_test_semifield = TRUE;
			test_semifield_data.assign(argv[++i]);
			cout << "-test_semifield " << test_semifield_data << endl;
		}
		else if (ST.stringcmp(argv[i], "-identify_semifield") == 0) {
			f_identify_semifield = TRUE;
			identify_semifield_data.assign(argv[++i]);
			cout << "-identify_semifield " << identify_semifield_data << endl;
		}
		else if (ST.stringcmp(argv[i], "-identify_semifields_from_file") == 0) {
			f_identify_semifields_from_file = TRUE;
			identify_semifields_from_file_fname.assign(argv[++i]);
			cout << "-identify_semifields_from_file "
					<< identify_semifields_from_file_fname << endl;
		}
#if 0
		else if (stringcmp(argv[i], "-trace_record_prefix") == 0) {
			f_trace_record_prefix = TRUE;
			trace_record_prefix = argv[++i];
			cout << "-trace_record_prefix " << trace_record_prefix << endl;
		}
		else if (stringcmp(argv[i], "-FstLen") == 0) {
			f_FstLen = TRUE;
			fname_FstLen = argv[++i];
			cout << "-FstLen " << fname_FstLen << endl;
		}
		else if (stringcmp(argv[i], "-Data") == 0) {
			f_Data = TRUE;
			fname_Data = argv[++i];
			cout << "-Data " << fname_Data << endl;
		}
#endif
		else if (ST.stringcmp(argv[i], "-load_classification") == 0) {
			f_load_classification = TRUE;
			cout << "-load_classification " << endl;
		}
		else if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report " << endl;
		}
		else if (ST.stringcmp(argv[i], "-decomposition_matrix_level_3") == 0) {
			f_decomposition_matrix_level_3 = TRUE;
			cout << "-decomposition_matrix_level_3 " << endl;
		}
		else if (ST.stringcmp(argv[i], "-level_two_prefix") == 0) {
			f_level_two_prefix = TRUE;
			level_two_prefix.assign(argv[++i]);
			cout << "-level_two_prefix " << level_two_prefix << endl;
		}
		else if (ST.stringcmp(argv[i], "-level_three_prefix") == 0) {
			f_level_three_prefix = TRUE;
			level_three_prefix.assign(argv[++i]);
			cout << "-level_three_prefix " << level_three_prefix << endl;
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "semifield_classify_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "semifield_classify_description::read_arguments done" << endl;
	return i + 1;
}


}}}
