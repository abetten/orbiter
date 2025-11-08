/*
 * tree_draw_options.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: betten
 */






#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace graphics {


tree_draw_options::tree_draw_options()
{
	Record_birth();
	f_file = false;
	//file_name;

	f_restrict = false;
	restrict_excluded_color = 0;

	f_select_path = false;
	//select_path_text;

	f_has_draw_vertex_callback = false;
	draw_vertex_callback = NULL;

	f_draw_options = false;
	//std::string draw_options_label;

	f_line_width = false;
	line_width = 100;

}


tree_draw_options::~tree_draw_options()
{
	Record_death();
}


int tree_draw_options::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "tree_draw_options::read_arguments" << endl;
	}

	for (i = 0; i < argc; i++) {


		if (ST.stringcmp(argv[i], "-file") == 0) {
			f_file = true;
			file_name.assign(argv[++i]);
			if (f_v) {
				cout << "-file " << file_name << " " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-restrict") == 0) {
			f_restrict = true;
			restrict_excluded_color = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-restrict " << restrict_excluded_color << " " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-select_path") == 0) {
			f_select_path = true;
			select_path_text.assign(argv[++i]);
			if (f_v) {
				cout << "-select_path " << select_path_text << " " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-draw_options") == 0) {
			f_draw_options = true;
			draw_options_label.assign(argv[++i]);
			if (f_v) {
				cout << "-draw_options " << draw_options_label << " " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-line_width") == 0) {
			f_line_width = true;
			line_width = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-line_width " << line_width << " " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "tree_draw_options::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	if (f_v) {
		cout << "tree_draw_options::read_arguments done" << endl;
	}
	return i + 1;
}

void tree_draw_options::print()
{
	if (f_file) {
		cout << "-file " << file_name << " " << endl;
	}
	if (f_restrict) {
		cout << "-restrict " << restrict_excluded_color << " " << endl;
	}
	if (f_select_path) {
		cout << "-select_path " << select_path_text << " " << endl;
	}
	if (f_has_draw_vertex_callback) {
		cout << "has draw_vertex_callback function" << endl;
	}
	if (f_draw_options) {
		cout << "-draw_options " << draw_options_label << " " << endl;
	}
	if (f_line_width) {
		cout << "-line_width " << line_width << " " << endl;
	}
}





}}}}


