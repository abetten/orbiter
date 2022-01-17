/*
 * classification_of_objects_report_options.cpp
 *
 *  Created on: Dec 16, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {
namespace combinatorics {



classification_of_objects_report_options::classification_of_objects_report_options()
{
	f_prefix = FALSE;
	//std::string prefix;

	f_export_flag_orbits = FALSE;

	f_show_incidence_matrices = FALSE;

	f_show_TDO = FALSE;

	f_show_TDA = FALSE;

	f_export_group = FALSE;

	f_lex_least = FALSE;
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
			f_prefix = TRUE;
			prefix.assign(argv[++i]);
			cout << "-prefix" << prefix << endl;
		}
		else if (ST.stringcmp(argv[i], "-export_flag_orbits") == 0) {
			f_export_flag_orbits = TRUE;
			cout << "-export_flag_orbits" << endl;
		}
		else if (ST.stringcmp(argv[i], "-dont_export_flag_orbits") == 0) {
			f_export_flag_orbits = FALSE;
			cout << "-export_flag_orbits" << endl;
		}
		else if (ST.stringcmp(argv[i], "-show_incidence_matrices") == 0) {
			f_show_incidence_matrices = TRUE;
			cout << "-show_incidence_matrices" << endl;
		}
		else if (ST.stringcmp(argv[i], "-dont_show_incidence_matrices") == 0) {
			f_show_incidence_matrices = FALSE;
			cout << "-dont_show_incidence_matrices" << endl;
		}
		else if (ST.stringcmp(argv[i], "-show_TDA") == 0) {
			f_show_TDA = TRUE;
			cout << "-show_TDA" << endl;
		}
		else if (ST.stringcmp(argv[i], "-dont_show_TDA") == 0) {
			f_show_TDA = FALSE;
			cout << "-dont_show_TDA" << endl;
		}
		else if (ST.stringcmp(argv[i], "-show_TDO") == 0) {
			f_show_TDO = TRUE;
			cout << "-show_TDO" << endl;
		}
		else if (ST.stringcmp(argv[i], "-dont_show_TDO") == 0) {
			f_show_TDO = FALSE;
			cout << "-dont_show_TDO" << endl;
		}
		else if (ST.stringcmp(argv[i], "-export_group") == 0) {
			f_export_group = TRUE;
			cout << "-export_group" << endl;
		}
		else if (ST.stringcmp(argv[i], "-dont_export_group") == 0) {
			f_export_group = FALSE;
			cout << "-dont_export_group" << endl;
		}
		else if (ST.stringcmp(argv[i], "-lex_least") == 0) {
			f_lex_least = TRUE;
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
		cout << "-prefix" << prefix << endl;
	}
	if (f_export_flag_orbits) {
		cout << "-export_flag_orbits" << endl;
	}
	if (f_show_incidence_matrices) {
		cout << "-show_incidence_matrices" << endl;
	}
	if (f_show_TDO) {
		cout << "-show_TDO" << endl;
	}
	if (f_show_TDA) {
		cout << "-show_TDA" << endl;
	}
	if (f_export_group) {
		cout << "-export_group" << endl;
	}
	if (f_lex_least) {
		cout << "-lex_least" << lex_least_geometry_builder << endl;
	}

}


}}}



