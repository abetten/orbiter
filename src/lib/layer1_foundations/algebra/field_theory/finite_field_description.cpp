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
namespace algebra {
namespace field_theory {


finite_field_description::finite_field_description()
{
	Record_birth();
	f_q = false;
	//std::string q_text;
	//q = 0;

	f_override_polynomial = false;
	//std::string override_polynomial;

	f_without_tables = false;

	f_compute_related_fields = true;

	f_symbol = false;
	//std::string symbol_label;

	f_print_as_exponentials = false;
	f_print_numerically = false;

}

finite_field_description::~finite_field_description()
{
	Record_death();
}

int finite_field_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "finite_field_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-q") == 0) {
			f_q = true;
			q_text.assign(argv[++i]);
			if (f_v) {
				cout << "-q " << q_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-override_polynomial") == 0) {
			f_override_polynomial = true;
			override_polynomial.assign(argv[++i]);
			if (f_v) {
				cout << "-override_polynomial " << override_polynomial << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-without_tables") == 0) {
			f_without_tables = true;
			if (f_v) {
				cout << "-without_tables " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-compute_related_fields") == 0) {
			f_compute_related_fields = true;
			if (f_v) {
				cout << "-compute_related_fields " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-symbol") == 0) {
			f_symbol = true;
			symbol_label.assign(argv[++i]);
			if (f_v) {
				cout << "-symbol " << symbol_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-print_as_exponentials") == 0) {
			f_print_as_exponentials = true;
			f_print_numerically = false;
			if (f_v) {
				cout << "-print_as_exponentials " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-print_numerically") == 0) {
			f_print_numerically = true;
			f_print_as_exponentials = false;
			if (f_v) {
				cout << "-print_numerically " << endl;
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
	if (f_print_as_exponentials) {
		cout << "-print_as_exponentials " << endl;
	}
	if (f_print_numerically) {
		cout << "-print_numerically " << endl;
	}
}

}}}}


