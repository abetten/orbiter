/*
 * geometry_builder_description.cpp
 *
 *  Created on: Aug 24, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {



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

	//std::vector<std::string> test_lines;
	//std::vector<std::string> test_flags;
	//std::vector<std::string> test2_lines;
	//std::vector<std::string> test2_flags;

	f_fname_GEO = FALSE;
	//std::string fname_GEO;
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

	if (f_v) {
		cout << "geometry_builder_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {
		if (stringcmp(argv[i], "-V") == 0) {
			f_V = TRUE;
			V_text.assign(argv[++i]);
			if (f_v) {
				cout << "-V " << V_text << endl;
			}
		}
		else if (stringcmp(argv[i], "-B") == 0) {
			f_B = TRUE;
			B_text.assign(argv[++i]);
			if (f_v) {
				cout << "-B " << B_text << endl;
			}
		}
		else if (stringcmp(argv[i], "-TDO") == 0) {
			f_TDO = TRUE;
			TDO_text.assign(argv[++i]);
			if (f_v) {
				cout << "-TDO " << TDO_text << endl;
			}
		}
		else if (stringcmp(argv[i], "-fuse") == 0) {
			f_fuse = TRUE;
			fuse_text.assign(argv[++i]);
			if (f_v) {
				cout << "-fuse " << fuse_text << endl;
			}
		}
		else if (stringcmp(argv[i], "-test") == 0) {
			string lines, flags;
			lines.assign(argv[++i]);
			flags.assign(argv[++i]);
			test_lines.push_back(lines);
			test_flags.push_back(flags);
			if (f_v) {
				cout << "-test " << lines << " " << flags << endl;
			}
		}
		else if (stringcmp(argv[i], "-test2") == 0) {
			string lines, flags;
			lines.assign(argv[++i]);
			flags.assign(argv[++i]);
			test2_lines.push_back(lines);
			test2_flags.push_back(flags);
			if (f_v) {
				cout << "-test2 " << lines << " " << flags << endl;
			}
		}
		else if (stringcmp(argv[i], "-fname_GEO") == 0) {
			f_fname_GEO = TRUE;
			fname_GEO.assign(argv[++i]);
			if (f_v) {
				cout << "-fname_GEO " << fname_GEO << endl;
			}
		}
		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "geometry_builder_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
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
	if (test_lines.size()) {
		int i;

		for (i = 0; i < test_lines.size(); i++) {
			cout << "-test " << test_lines[i] << " " << test_flags[i] << endl;
		}
	}
	if (test2_lines.size()) {
		int i;

		for (i = 0; i < test2_lines.size(); i++) {
			cout << "-test2 " << test2_lines[i] << " " << test2_flags[i] << endl;
		}
	}
	if (f_fname_GEO) {
		cout << "-fname_GEO " << fname_GEO << endl;
	}
}




}}

