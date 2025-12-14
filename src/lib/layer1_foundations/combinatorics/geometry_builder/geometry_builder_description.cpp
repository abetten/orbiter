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
namespace combinatorics {
namespace geometry_builder {



geometry_builder_description::geometry_builder_description()
{
	Record_birth();
	f_V = false;
	//std::string V_text;
	f_B = false;
	//std::string B_text;
	f_TDO = false;
	//std::string TDO_text;
	f_fuse = false;
	//std::string fuse_text;

	f_girth_test = false;
	girth = 0;

	f_lambda = false;
	lambda = 0;

	f_find_square = true; /* JS 120100 */

	f_simple = false; /* JS 180100 */

	f_search_tree = false;
	f_search_tree_flags = false;

	f_orderly = false;
	f_special_test_not_orderly = false;

	f_has_test_lines = false;
	//std::vector<std::string> test_lines;

	f_has_test2_lines = false;
	//std::vector<std::string> test2_lines;


	f_split = false;
	split_line = 0;
	split_remainder = 0;
	split_modulo = 1;

	//std::vector<int> print_at_line;

	f_fname_GEO = false;
	//std::string fname_GEO;

	f_output_to_inc_file = false;
	f_output_to_sage_file = false;
	f_output_to_blocks_file = false;
	f_output_to_blocks_latex_file = false;

	f_save_canonical_forms = false;


}

geometry_builder_description::~geometry_builder_description()
{
	Record_death();
}

int geometry_builder_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "geometry_builder_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-V") == 0) {
			f_V = true;
			V_text.assign(argv[++i]);
			if (f_v) {
				cout << "-V " << V_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-B") == 0) {
			f_B = true;
			B_text.assign(argv[++i]);
			if (f_v) {
				cout << "-B " << B_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-TDO") == 0) {
			f_TDO = true;
			TDO_text.assign(argv[++i]);
			if (f_v) {
				cout << "-TDO " << TDO_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-fuse") == 0) {
			f_fuse = true;
			fuse_text.assign(argv[++i]);
			if (f_v) {
				cout << "-fuse " << fuse_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-girth") == 0) {
			f_girth_test = true;
			girth = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-girth_test " << girth << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-lambda") == 0) {
			f_lambda = true;
			lambda = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-lambda " << lambda << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-no_square_test") == 0) {
			f_find_square = false;
			if (f_v) {
				cout << "-no_square_test " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-simple") == 0) {
			f_simple = true;
			if (f_v) {
				cout << "-simple " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-search_tree") == 0) {
			f_search_tree = true;
			if (f_v) {
				cout << "-search_tree " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-search_tree_flags") == 0) {
			f_search_tree_flags = true;
			if (f_v) {
				cout << "-search_tree_flags " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-orderly") == 0) {
			f_orderly = true;
			if (f_v) {
				cout << "-orderly " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-special_test_not_orderly") == 0) {
			f_special_test_not_orderly = true;
			if (f_v) {
				cout << "-special_test_not_orderly " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-test") == 0) {
			f_has_test_lines = true;
			string lines;
			lines.assign(argv[++i]);
			test_lines.push_back(lines);
			if (f_v) {
				cout << "-test " << lines << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-test2") == 0) {
			f_has_test2_lines = true;
			string lines;
			lines.assign(argv[++i]);
			test2_lines.push_back(lines);
			if (f_v) {
				cout << "-test2 " << lines << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-split") == 0) {
			f_split = true;
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
			f_fname_GEO = true;
			fname_GEO.assign(argv[++i]);
			if (f_v) {
				cout << "-fname_GEO " << fname_GEO << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-output_to_inc_file") == 0) {
			f_output_to_inc_file = true;
			if (f_v) {
				cout << "-output_to_inc_file " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-output_to_sage_file") == 0) {
			f_output_to_sage_file = true;
			if (f_v) {
				cout << "-output_to_sage_file " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-output_to_blocks_file") == 0) {
			f_output_to_blocks_file = true;
			if (f_v) {
				cout << "-output_to_blocks_file " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-output_to_blocks_latex_file") == 0) {
			f_output_to_blocks_latex_file = true;
			if (f_v) {
				cout << "-output_to_blocks_latex_file " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-save_canonical_forms") == 0) {
			f_save_canonical_forms = true;
			if (f_v) {
				cout << "-save_canonical_forms " << endl;
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
	if (f_find_square == false) {
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
	if (f_has_test_lines) {
		int i;

		for (i = 0; i < test_lines.size(); i++) {
			cout << "-test " << test_lines[i] << " " << endl;
		}
	}
	if (f_has_test2_lines) {
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
	if (f_save_canonical_forms) {
		cout << "-save_canonical_forms " << endl;
	}
}




}}}}


