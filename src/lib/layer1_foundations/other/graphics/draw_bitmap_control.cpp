/*
 * draw_bitmap_control.cpp
 *
 *  Created on: Mar 18, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace graphics {

draw_bitmap_control::draw_bitmap_control()
{
	Record_birth();
	f_input_csv_file = false;
	//std::string input_csv_file_name;

	f_secondary_input_csv_file = false;
	//std::string secondary_input_csv_file_name;


	f_input_object = false;
	//std::string input_object_label;

	f_partition = false;
	part_width = 4;
	//std::string part_row;
	//std::string part_col;

	f_box_width = false;
	box_width = 10;

	f_invert_colors = false;
	bit_depth = 8;

	f_grayscale = false;

	// not a command line argument
	f_input_matrix = false;
	M = NULL;
	M2 = NULL;
	m = 0;
	n = 0;

}

draw_bitmap_control::~draw_bitmap_control()
{
	Record_death();

}


int draw_bitmap_control::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "draw_bitmap_control::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-input_csv_file") == 0) {
			f_input_csv_file = true;
			input_csv_file_name.assign(argv[++i]);
			if (f_v) {
				cout << "-input_csv_file " << input_csv_file_name << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-secondary_input_csv_file") == 0) {
			f_secondary_input_csv_file = true;
			secondary_input_csv_file_name.assign(argv[++i]);
			if (f_v) {
				cout << "-secondary_input_csv_file " << secondary_input_csv_file_name << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-input_object") == 0) {
			f_input_object = true;
			input_object_label.assign(argv[++i]);
			if (f_v) {
				cout << "-input_object " << input_object_label << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-partition") == 0) {
			f_partition = true;
			part_width = ST.strtoi(argv[++i]);
			part_row.assign(argv[++i]);
			part_col.assign(argv[++i]);
			if (f_v) {
				cout << "-partition " << part_width << " " << part_row << " " << part_col << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-box_width") == 0) {
			f_box_width = true;
			box_width = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-box_width " << box_width << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-bit_depth") == 0) {
			bit_depth = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-bit_depth " << bit_depth << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-invert_colors") == 0) {
			f_invert_colors = true;
			if (f_v) {
				cout << "-invert_colors " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-grayscale") == 0) {
			f_grayscale = true;
			if (f_v) {
				cout << "-grayscale " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "draw_bitmap_control::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}

	}
	if (f_v) {
		cout << "draw_bitmap_control::read_arguments done" << endl;
	}
	return i + 1;
}


void draw_bitmap_control::print()
{
	if (f_input_csv_file) {
		cout << "-input_csv_file " << input_csv_file_name << endl;
	}
	if (f_secondary_input_csv_file) {
		cout << "-secondary_input_csv_file " << secondary_input_csv_file_name << endl;
	}
	if (f_input_object) {
		cout << "-input_object " << input_object_label << endl;
	}
	if (f_partition) {
		cout << "-partition " << part_width << " " << part_row << " " << part_col << endl;
	}
	if (f_box_width) {
		cout << "-box_width " << box_width << endl;
	}
	if (bit_depth) {
		cout << "-bit_depth " << bit_depth << endl;
	}
	if (f_invert_colors) {
		cout << "-invert_colors " << endl;
	}
	if (f_grayscale) {
		cout << "-grayscale " << endl;
	}
}



}}}}



