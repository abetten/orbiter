/*
 * objects_report_options.cpp
 *
 *  Created on: Dec 16, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace canonical_form_classification {



objects_report_options::objects_report_options()
{
	Record_birth();

	f_export_flag_orbits = false;

	f_show_incidence_matrices = false;

	f_show_TDO = false;

	f_show_TDA = false;

	f_export_labels = false;

	f_export_group_orbiter = false;
	f_export_group_GAP = false;

	f_lex_least = false;
	//std::string lex_least_geometry_builder;

	f_incidence_draw_options = false;
	//std::string incidence_draw_options_label;

	f_canonical_forms = false;

}

objects_report_options::~objects_report_options()
{
	Record_death();

}

int objects_report_options::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "objects_report_options::read_arguments" << endl;
	}
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "objects_report_options::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (f_v) {
			cout << "objects_report_options::read_arguments, "
					"next argument is " << argv[i] << endl;
		}

		if (ST.stringcmp(argv[i], "-export_flag_orbits") == 0) {
			f_export_flag_orbits = true;
			if (f_v) {
				cout << "-export_flag_orbits" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-dont_export_flag_orbits") == 0) {
			f_export_flag_orbits = false;
			if (f_v) {
				cout << "-export_flag_orbits" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-show_incidence_matrices") == 0) {
			f_show_incidence_matrices = true;
			if (f_v) {
				cout << "-show_incidence_matrices" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-dont_show_incidence_matrices") == 0) {
			f_show_incidence_matrices = false;
			if (f_v) {
				cout << "-dont_show_incidence_matrices" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-show_TDA") == 0) {
			f_show_TDA = true;
			if (f_v) {
				cout << "-show_TDA" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_labels") == 0) {
			f_export_labels = true;
			if (f_v) {
				cout << "-export_labels" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-dont_show_TDA") == 0) {
			f_show_TDA = false;
			if (f_v) {
				cout << "-dont_show_TDA" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-show_TDO") == 0) {
			f_show_TDO = true;
			if (f_v) {
				cout << "-show_TDO" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-dont_show_TDO") == 0) {
			f_show_TDO = false;
			if (f_v) {
				cout << "-dont_show_TDO" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_group_orbiter") == 0) {
			f_export_group_orbiter = true;
			if (f_v) {
				cout << "-export_group_orbiter" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_group_GAP") == 0) {
			f_export_group_GAP = true;
			if (f_v) {
				cout << "-export_group_GAP" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-dont_export_group") == 0) {
			f_export_group_orbiter = false;
			f_export_group_GAP = false;
			if (f_v) {
				cout << "-dont_export_group" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-lex_least") == 0) {
			f_lex_least = true;
			lex_least_geometry_builder.assign(argv[++i]);
			if (f_v) {
				cout << "-lex_least" << lex_least_geometry_builder << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-incidence_draw_options") == 0) {
			f_incidence_draw_options = true;
			incidence_draw_options_label.assign(argv[++i]);
			if (f_v) {
				cout << "-incidence_draw_options " << incidence_draw_options_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-canonical_forms") == 0) {
			f_canonical_forms = true;
			if (f_v) {
				cout << "-canonical_forms " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "objects_report_options::read_arguments -end" << endl;
			}
			break;
		}

		else {
			cout << "objects_report_options::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		if (f_v) {
			cout << "objects_report_options::read_arguments looping, i=" << i << endl;
		}
	} // next i
	if (f_v) {
		cout << "objects_report_options::read_arguments done" << endl;
	}
	return i + 1;
}

void objects_report_options::print()
{
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
	if (f_export_labels) {
		cout << "-export_labels" << endl;
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
	if (f_incidence_draw_options) {
		cout << "-incidence_draw_options " << incidence_draw_options_label << endl;
	}
	if (f_canonical_forms) {
		cout << "-canonical_forms " << endl;
	}

}


}}}}




