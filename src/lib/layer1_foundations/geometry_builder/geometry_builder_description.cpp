/*
 * geometry_builder_description.cpp
 *
 *  Created on: Aug 24, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace geometry_builder {



geometry_builder_description::geometry_builder_description()
{
	f_V = FALSE;
	//std::string V_text;
	f_B = FALSE;
	//std::string B_text;
	f_TDO = FALSE;
	//std::string TDO_text;
	f_fuse = FALSE;
	//std::string fuse_text;

	f_girth_test = FALSE;
	girth = 0;

	f_lambda = FALSE;
	lambda = 0;

	f_find_square = TRUE; /* JS 120100 */

	f_simple = FALSE; /* JS 180100 */

	f_search_tree = FALSE;
	f_search_tree_flags = FALSE;

	f_orderly = FALSE;
	f_special_test_not_orderly = FALSE;

	//std::vector<std::string> test_lines;
	//std::vector<std::string> test2_lines;


	f_split = FALSE;
	split_line = 0;
	split_remainder = 0;
	split_modulo = 1;

	//std::vector<int> print_at_line;

	f_fname_GEO = FALSE;
	//std::string fname_GEO;

	f_output_to_inc_file = FALSE;
	f_output_to_sage_file = FALSE;
	f_output_to_blocks_file = FALSE;
	f_output_to_blocks_latex_file = FALSE;


}

geometry_builder_description::~geometry_builder_description()
{
}

int geometry_builder_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "geometry_builder_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-V") == 0) {
			f_V = TRUE;
			V_text.assign(argv[++i]);
			if (f_v) {
				cout << "-V " << V_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-B") == 0) {
			f_B = TRUE;
			B_text.assign(argv[++i]);
			if (f_v) {
				cout << "-B " << B_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-TDO") == 0) {
			f_TDO = TRUE;
			TDO_text.assign(argv[++i]);
			if (f_v) {
				cout << "-TDO " << TDO_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-fuse") == 0) {
			f_fuse = TRUE;
			fuse_text.assign(argv[++i]);
			if (f_v) {
				cout << "-fuse " << fuse_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-girth") == 0) {
			f_girth_test = TRUE;
			girth = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-girth_test " << girth << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-lambda") == 0) {
			f_lambda = TRUE;
			lambda = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-lambda " << lambda << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-no_square_test") == 0) {
			f_find_square = FALSE;
			if (f_v) {
				cout << "-no_square_test " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-simple") == 0) {
			f_simple = TRUE;
			if (f_v) {
				cout << "-simple " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-search_tree") == 0) {
			f_search_tree = TRUE;
			if (f_v) {
				cout << "-search_tree " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-search_tree_flags") == 0) {
			f_search_tree_flags = TRUE;
			if (f_v) {
				cout << "-search_tree_flags " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-orderly") == 0) {
			f_orderly = TRUE;
			if (f_v) {
				cout << "-orderly " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-special_test_not_orderly") == 0) {
			f_special_test_not_orderly = TRUE;
			if (f_v) {
				cout << "-special_test_not_orderly " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-test") == 0) {
			string lines;
			lines.assign(argv[++i]);
			test_lines.push_back(lines);
			if (f_v) {
				cout << "-test " << lines << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-test2") == 0) {
			string lines;
			lines.assign(argv[++i]);
			test2_lines.push_back(lines);
			if (f_v) {
				cout << "-test2 " << lines << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-split") == 0) {
			f_split = TRUE;
			split_line = ST.strtoi(argv[++i]);
			split_remainder = ST.strtoi(argv[++i]);
			split_modulo = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-split " << split_line << " "
						<< split_remainder << " " << split_modulo << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-print_at_line") == 0) {
			int a;
			a = ST.strtoi(argv[++i]);
			print_at_line.push_back(a);
			if (f_v) {
				cout << "-print_at_line " << a << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-fname_GEO") == 0) {
			f_fname_GEO = TRUE;
			fname_GEO.assign(argv[++i]);
			if (f_v) {
				cout << "-fname_GEO " << fname_GEO << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-output_to_inc_file") == 0) {
			f_output_to_inc_file = TRUE;
			if (f_v) {
				cout << "-output_to_inc_file " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-output_to_sage_file") == 0) {
			f_output_to_sage_file = TRUE;
			if (f_v) {
				cout << "-output_to_sage_file " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-output_to_blocks_file") == 0) {
			f_output_to_blocks_file = TRUE;
			if (f_v) {
				cout << "-output_to_blocks_file " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-output_to_blocks_latex_file") == 0) {
			f_output_to_blocks_latex_file = TRUE;
			if (f_v) {
				cout << "-output_to_blocks_latex_file " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "geometry_builder_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "geometry_builder_description::read_arguments done" << endl;
	}
	return i + 1;
}

void geometry_builder_description::print()
{
	if (f_V) {
		cout << "-V " << V_text << endl;
	}
	if (f_B) {
		cout << "-B " << B_text << endl;
	}
	if (f_TDO) {
		cout << "-TDO " << TDO_text << endl;
	}
	if (f_fuse) {
		cout << "-fuse " << fuse_text << endl;
	}
	if (f_girth_test) {
		cout << "-girth " << girth << endl;
	}
	if (f_lambda) {
		cout << "-lambda " << lambda << endl;
	}
	if (f_find_square == FALSE) {
		cout << "-no_square_test " << endl;
	}
	if (f_simple) {
		cout << "-simple " << endl;
	}
	if (f_search_tree) {
		cout << "-search_tree " << endl;
	}
	if (f_search_tree_flags) {
		cout << "-search_tree_flags " << endl;
	}
	if (f_orderly) {
		cout << "-orderly " << endl;
	}
	if (f_special_test_not_orderly) {
		cout << "-special_test_not_orderly " << endl;
	}
	if (test_lines.size()) {
		int i;

		for (i = 0; i < test_lines.size(); i++) {
			cout << "-test " << test_lines[i] << " " << endl;
		}
	}
	if (test2_lines.size()) {
		int i;

		for (i = 0; i < test2_lines.size(); i++) {
			cout << "-test2 " << test2_lines[i] << " " << endl;
		}
	}
	if (f_split) {
		cout << "-split " << split_line << " " << split_remainder << " " << split_modulo << endl;
	}
	if (print_at_line.size()) {
		int i;

		for (i = 0; i < print_at_line.size(); i++) {
			cout << "-print_at_line " << print_at_line[i] << endl;
		}
	}
	if (f_fname_GEO) {
		cout << "-fname_GEO " << fname_GEO << endl;
	}
	if (f_output_to_inc_file) {
		cout << "-output_to_inc_file " << endl;
	}
	if (f_output_to_sage_file) {
		cout << "-output_to_sage_file " << endl;
	}
	if (f_output_to_blocks_file) {
		cout << "-output_to_blocks_file " << endl;
	}
	if (f_output_to_blocks_latex_file) {
		cout << "-output_to_blocks_latex_file " << endl;
	}
}




}}}

