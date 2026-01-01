/*
 * spread_table_description.cpp
 *
 *  Created on: Dec 21, 2025
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace finite_geometries {




spread_table_description::spread_table_description()
{
	Record_birth();

	f_space = false;
	//std::string space_label;

	f_rk = false;
	rk = 0;

	f_iso_types = false;
	//std::string iso_types_string;


	f_path = false;
	//std::string path;

	f_control = false;
	//std::string control_label;


	f_load = false;

	f_restricted_table = false;
	//std::string restricted_table_label;
	//std::string restricted_table_subset;

}

spread_table_description::~spread_table_description()
{
	Record_death();

}


int spread_table_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int i;
	other::data_structures::string_tools ST;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_table_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-space") == 0) {
			f_space = true;
			space_label.assign(argv[++i]);
			if (f_v) {
				cout << "-space " << space_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-rk") == 0) {
			f_rk = true;
			rk = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-rk " << rk << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-iso_types") == 0) {
			f_iso_types = true;
			iso_types_string.assign(argv[++i]);
			if (f_v) {
				cout << "-iso_types " << iso_types_string << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-path") == 0) {
			f_path = true;
			path.assign(argv[++i]);
			if (f_v) {
				cout << "-path " << path << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-control") == 0) {
			f_control = true;
			control_label.assign(argv[++i]);
			if (f_v) {
				cout << "-control " << control_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-load") == 0) {
			f_load = true;
			if (f_v) {
				cout << "-load " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-restricted_table") == 0) {
			f_restricted_table = true;
			restricted_table_label.assign(argv[++i]);
			restricted_table_subset.assign(argv[++i]);
			if (f_v) {
				cout << "-restricted_table " << restricted_table_label
						<< " " << restricted_table_subset << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "spread_table_description::read_arguments unknown command " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "spread_table_description::read_arguments done" << endl;
	return i + 1;
}

void spread_table_description::print()
{

	if (f_space) {
		cout << "-space " << space_label << endl;
	}
	if (f_rk) {
		cout << "-rk " << rk << endl;
	}
	if (f_iso_types) {
		cout << "-iso_types " << iso_types_string << endl;
	}
	if (f_path) {
		cout << "-path " << path << endl;
	}
	if (f_control) {
		cout << "-control " << control_label << endl;
	}
	if (f_load) {
		cout << "-load " << endl;
	}
	if (f_restricted_table) {
		cout << "-restricted_table " << restricted_table_label
				<< " " << restricted_table_subset << endl;
	}
}




}}}}




