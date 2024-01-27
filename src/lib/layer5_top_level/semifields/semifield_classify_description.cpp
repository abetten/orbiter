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
	f_order = false;
	order = 0;
	f_dim_over_kernel = false;
	dim_over_kernel = 0;
	f_prefix = false;
	//prefix = "";
	f_orbits_light = false;
	f_test_semifield = false;
	//test_semifield_data = NULL;
	f_identify_semifield = false;
	//identify_semifield_data = NULL;
	f_identify_semifields_from_file = false;
	//identify_semifields_from_file_fname = NULL;
	f_load_classification = false;
	f_report = false;
	f_decomposition_matrix_level_3 = false;
	f_level_two_prefix = false;
	//level_two_prefix;
	f_level_three_prefix = false;
	//level_three_prefix;
}

semifield_classify_description::~semifield_classify_description()
{

}


int semifield_classify_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "semifield_classify_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-order") == 0) {
			f_order = true;
			order = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-order " << order << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-dim_over_kernel") == 0) {
			f_dim_over_kernel = true;
			dim_over_kernel = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-dim_over_kernel " << dim_over_kernel << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-prefix") == 0) {
			f_prefix = true;
			prefix.assign(argv[++i]);
			if (f_v) {
				cout << "-prefix " << prefix << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-orbits_light") == 0) {
			f_orbits_light = true;
			if (f_v) {
				cout << "-orbits_light " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-test_semifield") == 0) {
			f_test_semifield = true;
			test_semifield_data.assign(argv[++i]);
			if (f_v) {
				cout << "-test_semifield " << test_semifield_data << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-identify_semifield") == 0) {
			f_identify_semifield = true;
			identify_semifield_data.assign(argv[++i]);
			if (f_v) {
				cout << "-identify_semifield " << identify_semifield_data << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-identify_semifields_from_file") == 0) {
			f_identify_semifields_from_file = true;
			identify_semifields_from_file_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-identify_semifields_from_file "
						<< identify_semifields_from_file_fname << endl;
			}
		}
#if 0
		else if (stringcmp(argv[i], "-trace_record_prefix") == 0) {
			f_trace_record_prefix = true;
			trace_record_prefix = argv[++i];
			cout << "-trace_record_prefix " << trace_record_prefix << endl;
		}
		else if (stringcmp(argv[i], "-FstLen") == 0) {
			f_FstLen = true;
			fname_FstLen = argv[++i];
			cout << "-FstLen " << fname_FstLen << endl;
		}
		else if (stringcmp(argv[i], "-Data") == 0) {
			f_Data = true;
			fname_Data = argv[++i];
			cout << "-Data " << fname_Data << endl;
		}
#endif
		else if (ST.stringcmp(argv[i], "-load_classification") == 0) {
			f_load_classification = true;
			if (f_v) {
				cout << "-load_classification " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;
			if (f_v) {
				cout << "-report " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-decomposition_matrix_level_3") == 0) {
			f_decomposition_matrix_level_3 = true;
			if (f_v) {
				cout << "-decomposition_matrix_level_3 " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-level_two_prefix") == 0) {
			f_level_two_prefix = true;
			level_two_prefix.assign(argv[++i]);
			if (f_v) {
				cout << "-level_two_prefix " << level_two_prefix << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-level_three_prefix") == 0) {
			f_level_three_prefix = true;
			level_three_prefix.assign(argv[++i]);
			if (f_v) {
				cout << "-level_three_prefix " << level_three_prefix << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "semifield_classify_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "semifield_classify_description::read_arguments done" << endl;
	}
	return i + 1;
}

void semifield_classify_description::print()
{
	if (f_order) {
		cout << "-order " << order << endl;
	}
	if (f_dim_over_kernel) {
		cout << "-dim_over_kernel " << dim_over_kernel << endl;
	}
	if (f_prefix) {
		cout << "-prefix " << prefix << endl;
	}
	if (f_orbits_light) {
		cout << "-orbits_light " << endl;
	}
	if (f_test_semifield) {
		cout << "-test_semifield " << test_semifield_data << endl;
	}
	if (f_identify_semifield) {
		cout << "-identify_semifield " << identify_semifield_data << endl;
	}
	if (f_identify_semifields_from_file) {
		cout << "-identify_semifields_from_file "
				<< identify_semifields_from_file_fname << endl;
	}
	if (f_load_classification) {
		cout << "-load_classification " << endl;
	}
	if (f_report) {
		cout << "-report " << endl;
	}
	if (f_decomposition_matrix_level_3) {
		cout << "-decomposition_matrix_level_3 " << endl;
	}
	if (f_level_two_prefix) {
		cout << "-level_two_prefix " << level_two_prefix << endl;
	}
	if (f_level_three_prefix) {
		cout << "-level_three_prefix " << level_three_prefix << endl;
	}
}

}}}
