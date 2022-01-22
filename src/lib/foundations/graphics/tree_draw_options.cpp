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

tree_draw_options::tree_draw_options()
{
	f_file = FALSE;
	//file_name;

	f_restrict = FALSE;
	restrict_excluded_color = 0;

	f_select_path = FALSE;
	//select_path_text;

	f_has_draw_vertex_callback = FALSE;
	draw_vertex_callback = NULL;

}


tree_draw_options::~tree_draw_options()
{
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
			f_file = TRUE;
			file_name.assign(argv[++i]);
			if (f_v) {
				cout << "-file " << file_name << " " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-restrict") == 0) {
			f_restrict = TRUE;
			restrict_excluded_color = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-restrict " << restrict_excluded_color << " " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-select_path") == 0) {
			f_select_path = TRUE;
			select_path_text.assign(argv[++i]);
			if (f_v) {
				cout << "-select_path " << select_path_text << " " << endl;
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
}





}}
