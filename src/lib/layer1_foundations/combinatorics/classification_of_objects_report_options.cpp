/*
 * classification_of_objects_report_options.cpp
 *
 *  Created on: Dec 16, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {



classification_of_objects_report_options::classification_of_objects_report_options()
{
	f_prefix = false;
	//std::string prefix;

	f_export_flag_orbits = false;

	f_show_incidence_matrices = false;

	f_show_TDO = false;

	f_show_TDA = false;

	f_export_group_orbiter = false;
	f_export_group_GAP = false;

	f_lex_least = false;
	//std::string lex_least_geometry_builder;

}

classification_of_objects_report_options::~classification_of_objects_report_options()
{

}

int classification_of_objects_report_options::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;
	data_structures::string_tools ST;

	cout << "classification_of_objects_report_options::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		cout << "classification_of_objects_report_options::read_arguments, next argument is " << argv[i] << endl;

		if (ST.stringcmp(argv[i], "-prefix") == 0) {
			f_prefix = true;
			prefix.assign(argv[++i]);
			cout << "-prefix " << prefix << endl;
		}
		else if (ST.stringcmp(argv[i], "-export_flag_orbits") == 0) {
			f_export_flag_orbits = true;
			cout << "-export_flag_orbits" << endl;
		}
		else if (ST.stringcmp(argv[i], "-dont_export_flag_orbits") == 0) {
			f_export_flag_orbits = false;
			cout << "-export_flag_orbits" << endl;
		}
		else if (ST.stringcmp(argv[i], "-show_incidence_matrices") == 0) {
			f_show_incidence_matrices = true;
			cout << "-show_incidence_matrices" << endl;
		}
		else if (ST.stringcmp(argv[i], "-dont_show_incidence_matrices") == 0) {
			f_show_incidence_matrices = false;
			cout << "-dont_show_incidence_matrices" << endl;
		}
		else if (ST.stringcmp(argv[i], "-show_TDA") == 0) {
			f_show_TDA = true;
			cout << "-show_TDA" << endl;
		}
		else if (ST.stringcmp(argv[i], "-dont_show_TDA") == 0) {
			f_show_TDA = false;
			cout << "-dont_show_TDA" << endl;
		}
		else if (ST.stringcmp(argv[i], "-show_TDO") == 0) {
			f_show_TDO = true;
			cout << "-show_TDO" << endl;
		}
		else if (ST.stringcmp(argv[i], "-dont_show_TDO") == 0) {
			f_show_TDO = false;
			cout << "-dont_show_TDO" << endl;
		}
		else if (ST.stringcmp(argv[i], "-export_group_orbiter") == 0) {
			f_export_group_orbiter = true;
			cout << "-export_group_orbiter" << endl;
		}
		else if (ST.stringcmp(argv[i], "-export_group_GAP") == 0) {
			f_export_group_GAP = true;
			cout << "-export_group_GAP" << endl;
		}
		else if (ST.stringcmp(argv[i], "-dont_export_group") == 0) {
			f_export_group_orbiter = false;
			f_export_group_GAP = false;
			cout << "-dont_export_group" << endl;
		}
		else if (ST.stringcmp(argv[i], "-lex_least") == 0) {
			f_lex_least = true;
			lex_least_geometry_builder.assign(argv[++i]);
			cout << "-lex_least" << lex_least_geometry_builder << endl;
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "classification_of_objects_report_options::read_arguments -end" << endl;
			break;
		}

		else {
			cout << "classification_of_objects_report_options::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		cout << "classification_of_objects_report_options::read_arguments looping, i=" << i << endl;
	} // next i
	cout << "classification_of_objects_report_options::read_arguments done" << endl;
	return i + 1;
}

void classification_of_objects_report_options::print()
{
	if (f_prefix) {
		cout << "-prefix " << prefix << endl;
	}
	if (f_export_flag_orbits) {
		cout << "-export_flag_orbits " << endl;
	}
	if (f_show_incidence_matrices) {
		cout << "-show_incidence_matrices " << endl;
	}
	if (f_show_TDO) {
		cout << "-show_TDO " << endl;
	}
	if (f_show_TDA) {
		cout << "-show_TDA " << endl;
	}
	if (f_export_group_orbiter) {
		cout << "-export_group_orbiter " << endl;
	}
	if (f_export_group_GAP) {
		cout << "-export_group_GAP " << endl;
	}
	if (f_lex_least) {
		cout << "-lex_least " << lex_least_geometry_builder << endl;
	}

}


}}}



