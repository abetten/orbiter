/*
 * finite_field_description.cpp
 *
 *  Created on: Dec 2, 2020
 *      Author: betten
 */






#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {


finite_field_description::finite_field_description()
{
	f_q = FALSE;
	q = 0;

	f_override_polynomial = FALSE;
	//std::string override_polynomial;

}

finite_field_description::~finite_field_description()
{
}


int finite_field_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "finite_field_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {
		if (stringcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = strtoi(argv[++i]);
			cout << "-q " << q << endl;
		}
		else if (stringcmp(argv[i], "-override_polynomial") == 0) {
			f_override_polynomial = TRUE;
			override_polynomial.assign(argv[++i]);
			cout << "-override_polynomial" << override_polynomial << endl;
		}
		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "finite_field_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "finite_field_description::read_arguments done" << endl;
	return i + 1;
}

void finite_field_description::print()
{
	cout << "finite_field_description::print:" << endl;

	if (f_q) {
		cout << "q: " << q << endl;
	}
	if (f_override_polynomial) {
		cout << "override_polynomial: " << override_polynomial << endl;
	}
}

}}
