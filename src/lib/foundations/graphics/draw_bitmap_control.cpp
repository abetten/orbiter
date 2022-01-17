/*
 * draw_bitmap_control.cpp
 *
 *  Created on: Mar 18, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {

draw_bitmap_control::draw_bitmap_control()
{
	f_input_csv_file = FALSE;
	//std::string input_csv_file_name;

	f_secondary_input_csv_file = FALSE;
	//std::string secondary_input_csv_file_name;


	f_input_matrix = FALSE;
	M = NULL;
	M2 = NULL;
	m = 0;
	n = 0;

	f_partition = FALSE;
	part_width = 4;
	//std::string part_row;
	//std::string part_col;

	f_box_width = FALSE;
	box_width = 10;
	f_invert_colors = FALSE;
	bit_depth = 8;

}

draw_bitmap_control::~draw_bitmap_control()
{

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
			f_input_csv_file = TRUE;
			input_csv_file_name.assign(argv[++i]);
			if (f_v) {
				cout << "-input_csv_file " << input_csv_file_name << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-secondary_input_csv_file") == 0) {
			f_secondary_input_csv_file = TRUE;
			secondary_input_csv_file_name.assign(argv[++i]);
			if (f_v) {
				cout << "-secondary_input_csv_file " << secondary_input_csv_file_name << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-partition") == 0) {
			f_partition = TRUE;
			part_width = ST.strtoi(argv[++i]);
			part_row.assign(argv[++i]);
			part_col.assign(argv[++i]);
			//Orbiter->Int_vec.scan(part_row, Row_parts, nb_row_parts);
			//Orbiter->Int_vec.scan(part_col, Col_parts, nb_col_parts);
			if (f_v) {
				cout << "-partition " << part_width << " " << part_row << " " << part_col << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-box_width") == 0) {
			f_box_width = TRUE;
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
			f_invert_colors = TRUE;
			if (f_v) {
				cout << "-invert_colors " << endl;
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
}



}}

