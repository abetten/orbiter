/*
 * crc_options_description.cpp
 *
 *  Created on: Aug 12, 2022
 *      Author: betten
 */





#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace coding_theory {




crc_options_description::crc_options_description()
{
	f_input = false;
	//std::string input_fname;

	f_output = false;
	//std::string output_fname;

	f_crc_type = false;
	//std::string crc_type;

	f_block_length = false;
	block_length = 0;

	f_block_based_error_generator = false;

	f_file_based_error_generator = false;
	file_based_error_generator_threshold = 0;

	f_nb_repeats = false;
	nb_repeats = 0;

	f_threshold = false;
	threshold = 0;

	f_error_log = false;
	//std::string error_log_fname;

	f_selected_block = false;
	selected_block = 0;
}

crc_options_description::~crc_options_description()
{
}


int crc_options_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "crc_options_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-input") == 0) {
			f_input = true;
			input_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-input " << input_fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-output") == 0) {
			f_output = true;
			output_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-output " << output_fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-crc_type") == 0) {
			f_crc_type = true;
			crc_type.assign(argv[++i]);
			if (f_v) {
				cout << "-crc_type " << crc_type << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-block_length") == 0) {
			f_block_length = true;
			block_length = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-block_length " << block_length << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-block_based_error_generator") == 0) {
			f_block_based_error_generator = true;
			if (f_v) {
				cout << "-block_based_error_generator " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-file_based_error_generator") == 0) {
			f_file_based_error_generator = true;
			file_based_error_generator_threshold = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-file_based_error_generator " << file_based_error_generator_threshold << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-nb_repeats") == 0) {
			f_nb_repeats = true;
			nb_repeats = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-nb_repeats " << nb_repeats << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-threshold") == 0) {
			f_threshold = true;
			threshold = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-threshold " << threshold << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-error_log") == 0) {
			f_error_log = true;
			error_log_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-error_log " << error_log_fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-selected_block") == 0) {
			f_selected_block = true;
			selected_block = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-selected_block " << selected_block << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
			}
		else {
			cout << "crc_options_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}

	} // next i
	if (f_v) {
		cout << "coding_theoretic_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void crc_options_description::print()
{
	if (f_input) {
		cout << "-input " << input_fname << endl;
	}
	if (f_output) {
		cout << "-output " << output_fname << endl;
	}
	if (f_crc_type) {
		cout << "-crc_type " << crc_type << endl;
	}
	if (f_block_length) {
		cout << "-block_length " << block_length << endl;
	}
	if (f_block_based_error_generator) {
		cout << "-block_based_error_generator " << endl;
	}
	if (f_file_based_error_generator) {
		cout << "-file_based_error_generator " << file_based_error_generator_threshold << endl;
	}
	if (f_nb_repeats) {
		cout << "-nb_repeats " << nb_repeats << endl;
	}
	if (f_threshold) {
		cout << "-threshold " << threshold << endl;
	}
	if (f_error_log) {
		cout << "-error_log " << error_log_fname << endl;
	}
	if (f_selected_block) {
		cout << "-selected_block " << selected_block << endl;
	}


}



}}}

