/*
 * draw_incidence_structure_description.cpp
 *
 *  Created on: Dec 5, 2021
 *      Author: betten
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace graphics {


draw_incidence_structure_description::draw_incidence_structure_description()
{
	f_width = FALSE;
	width = 40;

	f_width_10 = FALSE;
	width_10 = 10;

	f_outline_thin = FALSE;

	f_unit_length = FALSE;;
	unit_length.assign("0.065mm");

	f_thick_lines = FALSE;
	thick_lines.assign("0.5mm");

	f_thin_lines = FALSE;
	thin_lines.assign("0.15mm");

	f_geo_line_width = FALSE;
	geo_line_width.assign("0.25mm");


	v = 0;
	b = 0;
	V = 0;
	B = 0;
	Vi = NULL;
	Bj = NULL;

#if 0
	int *R;
	int *X;
	int dim_X;
#endif

	f_labelling_points = FALSE;
	point_labels = NULL;

	f_labelling_blocks = FALSE;
	block_labels = NULL;
}

draw_incidence_structure_description::~draw_incidence_structure_description()
{

}



int draw_incidence_structure_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "draw_incidence_structure_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-width") == 0) {
			f_width = TRUE;
			width = ST.strtoi(argv[++i]);
			cout << "-width " << width << endl;
		}
		else if (ST.stringcmp(argv[i], "-width_10") == 0) {
			f_width_10 = TRUE;
			width_10 = ST.strtoi(argv[++i]);
			cout << "-width_10 " << width_10 << endl;
		}
		else if (ST.stringcmp(argv[i], "-outline_thin") == 0) {
			f_outline_thin = TRUE;
			cout << "-outline_thin " << endl;
		}
		else if (ST.stringcmp(argv[i], "-unit_length") == 0) {
			f_unit_length = TRUE;
			unit_length.assign(argv[++i]);
			cout << "-unit_length " << unit_length << endl;
		}
		else if (ST.stringcmp(argv[i], "-thick_lines") == 0) {
			f_thick_lines = TRUE;
			thick_lines.assign(argv[++i]);
			cout << "-thick_lines " << thick_lines << endl;
		}
		else if (ST.stringcmp(argv[i], "-thin_lines") == 0) {
			f_thin_lines = TRUE;
			thin_lines.assign(argv[++i]);
			cout << "-thin_lines " << thin_lines << endl;
		}
		else if (ST.stringcmp(argv[i], "-geo_line_width") == 0) {
			f_geo_line_width = TRUE;
			geo_line_width.assign(argv[++i]);
			cout << "-geo_line_width " << geo_line_width << endl;
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "draw_incidence_structure_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "draw_incidence_structure_description::read_arguments done" << endl;
	}
	return i + 1;
}

#if 0
int width;

int width_10;

int f_outline_thin;

std::string unit_length;
std::string thick_lines;
std::string thin_lines;
std::string geo_line_width;

int v;
int b;
int V;
int B;
int *Vi;
int *Bj;

#if 0
int *R;
int *X;
int dim_X;
#endif

int f_labelling_points;
std::string *point_labels;

int f_labelling_blocks;
std::string *block_labels;
#endif

void draw_incidence_structure_description::print()
{
	if (f_width) {
		cout << "-width " << width << endl;
	}
	if (f_width_10) {
		cout << "-width_10 " << width_10 << endl;
	}
	if (f_outline_thin) {
		cout << "-outline_thin " << endl;
	}
	if (f_unit_length) {
		cout << "-unit_length " << unit_length << endl;
	}
	if (f_thick_lines) {
		cout << "-thick_lines " << thick_lines << endl;
	}
	if (f_thin_lines) {
		cout << "-thin_lines " << thin_lines << endl;
	}
	if (f_geo_line_width) {
		cout << "-geo_line_width " << geo_line_width << endl;
	}
}





}}}


