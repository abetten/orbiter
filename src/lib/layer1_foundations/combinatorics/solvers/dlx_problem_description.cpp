/*
 * dlx_problem_description.cpp
 *
 *  Created on: Jan 12, 2022
 *      Author: betten
 */




#include "foundations.h"


using namespace std;

namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace solvers {


dlx_problem_description::dlx_problem_description()
{
	Record_birth();
	f_label_txt = false;
	//std::string label_txt;
	f_label_tex = false;
	//std::string label_tex;

	f_data_label = false;
	//std::string data_label;

	f_data_matrix = false;
	data_matrix = NULL;
	data_matrix_m = 0;
	data_matrix_n = 0;


	f_write_solutions = false;
	f_write_tree = false;

	f_tracking_depth = false;
	tracking_depth = 0;

}

dlx_problem_description::~dlx_problem_description()
{
	Record_death();
}

int dlx_problem_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "dlx_problem_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-label_txt") == 0) {
			f_label_txt = true;
			label_txt.assign(argv[++i]);
			if (f_v) {
				cout << "-label_txt " << label_txt << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-label_tex") == 0) {
			f_label_tex = true;
			label_tex.assign(argv[++i]);
			if (f_v) {
				cout << "-label_tex " << label_tex << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-data_label") == 0) {
			f_data_label = true;
			data_label.assign(argv[++i]);
			if (f_v) {
				cout << "-data_label " << data_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-write_solutions") == 0) {
			f_write_solutions = true;
			if (f_v) {
				cout << "-write_solutions " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-write_tree") == 0) {
			f_write_tree = true;
			if (f_v) {
				cout << "-write_tree " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-tracking_depth") == 0) {
			f_tracking_depth = true;
			tracking_depth = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-tracking_depth " << tracking_depth << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "dlx_problem_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "dlx_problem_description::read_arguments done" << endl;
	}
	return i + 1;
}

void dlx_problem_description::print()
{
	if (f_label_txt) {
		cout << "-label_txt " << label_txt << endl;
	}
	if (f_label_tex) {
		cout << "-label_tex " << label_tex << endl;
	}
	if (f_data_label) {
		cout << "-data_label " << data_label << endl;
	}
	if (f_data_matrix) {
		cout << "-data_matrix of size " << data_matrix_m << " x " << data_matrix_n << endl;
	}
	if (f_write_solutions) {
		cout << "-write_solutions " << endl;
	}
	if (f_write_tree) {
		cout << "-write_tree " << endl;
	}
	if (f_tracking_depth) {
		cout << "-tracking_depth " << tracking_depth << endl;
	}
}



}}}}



