/*
 * finite_field_description.cpp
 *
 *  Created on: Dec 2, 2020
 *      Author: betten
 */






#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace field_theory {


finite_field_description::finite_field_description()
{
	f_q = FALSE;
	//std::string q_text;
	//q = 0;

	f_override_polynomial = FALSE;
	//std::string override_polynomial;

	f_without_tables = FALSE;

	f_compute_related_fields = TRUE;

	f_symbol = FALSE;
	//std::string symbol_label;

}

finite_field_description::~finite_field_description()
{
}

int finite_field_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "finite_field_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q_text.assign(argv[++i]);
			if (f_v) {
				cout << "-q " << q_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-override_polynomial") == 0) {
			f_override_polynomial = TRUE;
			override_polynomial.assign(argv[++i]);
			if (f_v) {
				cout << "-override_polynomial " << override_polynomial << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-without_tables") == 0) {
			f_without_tables = TRUE;
			if (f_v) {
				cout << "-without_tables " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-compute_related_fields") == 0) {
			f_compute_related_fields = TRUE;
			if (f_v) {
				cout << "-compute_related_fields " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-symbol") == 0) {
			f_symbol = TRUE;
			symbol_label.assign(argv[++i]);
			if (f_v) {
				cout << "-symbol " << symbol_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "finite_field_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "finite_field_description::read_arguments done" << endl;
	}
	return i + 1;
}

void finite_field_description::print()
{
	//cout << "finite_field_description::print:" << endl;

	if (f_q) {
		cout << "-q " << q_text << endl;
	}
	if (f_override_polynomial) {
		cout << "-override_polynomial " << override_polynomial << endl;
	}
	if (f_without_tables) {
		cout << "-without_tables" << endl;
	}
	if (f_compute_related_fields) {
		cout << "-compute_related_fields " << endl;
	}
	if (f_symbol) {
		cout << "-symbol " << symbol_label << endl;
	}
}

}}}

