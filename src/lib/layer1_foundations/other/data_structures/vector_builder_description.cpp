/*
 * vector_builder_description.cpp
 *
 *  Created on: Nov 1, 2021
 *      Author: betten
 */



#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace data_structures {


vector_builder_description::vector_builder_description()
{
	Record_birth();

	f_field = false;
	//std::string field_label;

	f_allow_negatives = false;

	f_dense = false;
	//std::string dense_text;

	f_compact = false;
	//std::string compact_text;

	f_repeat = false;
	//std::string repeat_text;
	repeat_length = 0;

	f_format = false;
	format_k = 0;

	f_file = false;
	//std::string file_name;

	f_file_column = false;
	//std::string file_column_name;
	//std::string file_column_label;

	f_load_csv_no_border = false;
	//std::string load_csv_no_border_fname;

	f_load_csv_data_column = false;
	//std::string load_csv_data_column_fname;
	load_csv_data_column_idx = 0;

	f_sparse = false;
	sparse_len = 0;
	//std::string sparse_pairs;

	f_concatenate = false;
	//std::string concatenate_list;

	f_loop = false;
	loop_start = 0;
	loop_upper_bound = 0;
	loop_increment = 0;

	f_index_of_support = false;
	//std::string index_of_support_input;

	f_permutation_matrix = false;
	//std::string permutation_matrix_data;

	f_permutation_matrix_inverse = false;
	//std::string permutation_matrix_inverse_data;


}

vector_builder_description::~vector_builder_description()
{
	Record_death();
}


int vector_builder_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i = 0;
	string_tools ST;

	if (f_v) {
		cout << "vector_builder_description::read_arguments" << endl;
		cout << "vector_builder_description::read_arguments i = " << i << endl;
		cout << "vector_builder_description::read_arguments argc = " << argc << endl;
	}
	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-field") == 0) {
			f_field = true;
			field_label.assign(argv[++i]);
			if (f_v) {
				cout << "-field " << field_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-allow_negatives") == 0) {
			f_allow_negatives = true;
			if (f_v) {
				cout << "-allow_negatives " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-dense") == 0) {
			f_dense = true;
			dense_text.assign(argv[++i]);
			if (f_v) {
				cout << "-dense " << dense_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-compact") == 0) {
			f_compact = true;
			compact_text.assign(argv[++i]);
			if (f_v) {
				cout << "-compact " << compact_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-repeat") == 0) {
			f_repeat = true;
			repeat_text.assign(argv[++i]);
			repeat_length = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-repeat " << repeat_text << " " << repeat_length << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-format") == 0) {
			f_format = true;
			format_k = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-format " << format_k << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-file") == 0) {
			f_file = true;
			file_name.assign(argv[++i]);
			if (f_v) {
				cout << "-file " << file_name << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-file_column") == 0) {
			f_file_column = true;
			file_column_name.assign(argv[++i]);
			file_column_label.assign(argv[++i]);
			if (f_v) {
				cout << "-file_column " << file_column_name << " " << file_column_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-load_csv_no_border") == 0) {
			f_load_csv_no_border = true;
			load_csv_no_border_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-load_csv_no_border " << load_csv_no_border_fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-load_csv_data_column") == 0) {
			f_load_csv_data_column = true;
			load_csv_data_column_fname.assign(argv[++i]);
			load_csv_data_column_idx = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-load_csv_data_column "
						<< load_csv_data_column_fname
						<< " " << load_csv_data_column_idx << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-sparse") == 0) {
			f_sparse = true;
			sparse_len = ST.strtoi(argv[++i]);
			sparse_pairs.assign(argv[++i]);
			if (f_v) {
				cout << "-sparse " << sparse_len << " " << sparse_pairs << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-concatenate") == 0) {
			f_concatenate = true;
			concatenate_list.assign(argv[++i]);
			if (f_v) {
				cout << "-concatenate " << concatenate_list << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-loop") == 0) {
			f_loop = true;
			loop_start = ST.strtoi(argv[++i]);
			loop_upper_bound = ST.strtoi(argv[++i]);
			loop_increment = ST.strtoi(argv[++i]);
			cout << "-loop "
					<< loop_start << " "
					<< loop_upper_bound << " "
					<< loop_increment << endl;
		}
		else if (ST.stringcmp(argv[i], "-index_of_support") == 0) {
			f_index_of_support = true;
			index_of_support_input.assign(argv[++i]);
			if (f_v) {
				cout << "-index_of_support " << index_of_support_input << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-permutation_matrix") == 0) {
			f_permutation_matrix = true;
			permutation_matrix_data.assign(argv[++i]);
			if (f_v) {
				cout << "-permutation_matrix " << permutation_matrix_data << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-permutation_matrix_inverse") == 0) {
			f_permutation_matrix_inverse = true;
			permutation_matrix_inverse_data.assign(argv[++i]);
			if (f_v) {
				cout << "-permutation_matrix_inverse " << permutation_matrix_inverse_data << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
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
	if (f_allow_negatives) {
		cout << "-allow_negatives " << endl;
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
	if (f_file_column) {
		cout << "-file_column " << file_column_name << " " << file_column_label << endl;
	}
	if (f_load_csv_no_border) {
		cout << "-load_csv_no_border " << load_csv_no_border_fname << endl;
	}
	if (f_load_csv_data_column) {
		cout << "-load_csv_data_column "
				<< load_csv_data_column_fname << " " << load_csv_data_column_idx << endl;
	}
	if (f_sparse) {
		cout << "-sparse " << sparse_len << " " << sparse_pairs << endl;
	}
	if (f_concatenate) {
		cout << "-concatenate " << concatenate_list << endl;
	}
	if (f_loop) {
		cout << "-loop "
				<< loop_start << " "
				<< loop_upper_bound << " "
				<< loop_increment << endl;
	}
	if (f_index_of_support) {
		cout << "-index_of_support " << index_of_support_input << endl;
	}
	if (f_permutation_matrix) {
		cout << "-permutation_matrix " << permutation_matrix_data << endl;
	}
	if (f_permutation_matrix_inverse) {
		cout << "-permutation_matrix_inverse " << permutation_matrix_inverse_data << endl;
	}

}


}}}}



