/*
 * vector_builder_description.cpp
 *
 *  Created on: Nov 1, 2021
 *      Author: betten
 */



#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {


vector_builder_description::vector_builder_description()
{
	f_field = FALSE;
	//std::string field_label;

	f_dense = FALSE;
	//std::string dense_text;

	f_compact = FALSE;
	//std::string compact_text;

	f_repeat = FALSE;
	//std::string repeat_text;
	repeat_length = 0;

	f_format = FALSE;
	format_k = 0;

	f_file = FALSE;
	//std::string file_name;

	f_sparse = FALSE;
	sparse_len = 0;
	//std::string sparse_pairs;

}

vector_builder_description::~vector_builder_description()
{
}


int vector_builder_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i = 0;

	if (f_v) {
		cout << "vector_builder_description::read_arguments" << endl;
		cout << "vector_builder_description::read_arguments i = " << i << endl;
		cout << "vector_builder_description::read_arguments argc = " << argc << endl;
	}
	for (i = 0; i < argc; i++) {
		if (stringcmp(argv[i], "-field") == 0) {
			f_field = TRUE;
			field_label.assign(argv[++i]);
			if (f_v) {
				cout << "-field " << field_label << endl;
			}
		}
		else if (stringcmp(argv[i], "-dense") == 0) {
			f_dense = TRUE;
			dense_text.assign(argv[++i]);
			if (f_v) {
				cout << "-dense " << dense_text << endl;
			}
		}
		else if (stringcmp(argv[i], "-compact") == 0) {
			f_compact = TRUE;
			compact_text.assign(argv[++i]);
			if (f_v) {
				cout << "-compact " << compact_text << endl;
			}
		}
		else if (stringcmp(argv[i], "-repeat") == 0) {
			f_repeat = TRUE;
			repeat_text.assign(argv[++i]);
			repeat_length = strtoi(argv[++i]);
			if (f_v) {
				cout << "-repeat " << repeat_text << " " << repeat_length << endl;
			}
		}
		else if (stringcmp(argv[i], "-format") == 0) {
			f_format = TRUE;
			format_k = strtoi(argv[++i]);
			if (f_v) {
				cout << "-format " << format_k << endl;
			}
		}
		else if (stringcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			file_name.assign(argv[++i]);
			if (f_v) {
				cout << "-file " << file_name << endl;
			}
		}
		else if (stringcmp(argv[i], "-sparse") == 0) {
			f_sparse = TRUE;
			sparse_len = strtoi(argv[++i]);
			sparse_pairs.assign(argv[++i]);
			if (f_v) {
				cout << "-sparse " << sparse_len << " " << sparse_pairs << endl;
			}
		}

		else if (stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "vector_builder_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "vector_builder_description::read_arguments done" << endl;
	}
	return i + 1;
}

void vector_builder_description::print()
{
	cout << "vector_builder_description:" << endl;
	if (f_field) {
		cout << "-field " << field_label << endl;
	}
	if (f_dense) {
		cout << "-dense " << dense_text << endl;
	}
	if (f_compact) {
		cout << "-compact " << compact_text << endl;
	}
	if (f_repeat) {
		cout << "-repeat " << repeat_text << " " << repeat_length << endl;
	}
	if (f_format) {
		cout << "-format " << format_k << endl;
	}
	if (f_file) {
		cout << "-file " << file_name << endl;
	}
	if (f_sparse) {
		cout << "-file " << sparse_len << " " << sparse_pairs << endl;
	}
}


}}


