/*
 * packing_was_fixpoints_activity_description.cpp
 *
 *  Created on: Apr 3, 2021
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



packing_was_fixpoints_activity_description::packing_was_fixpoints_activity_description()
{
	f_report = FALSE;

	f_print_packing = FALSE;
	//std::string print_packing_text;
}

packing_was_fixpoints_activity_description::~packing_was_fixpoints_activity_description()
{

}

int packing_was_fixpoints_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "packing_was_fixpoints_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		cout << "packing_was_fixpoints_activity_description::read_arguments, next argument is " << argv[i] << endl;


		if (stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report" << endl;
		}
		else if (stringcmp(argv[i], "-print_packing") == 0) {
			f_print_packing = TRUE;
			print_packing_text.assign(argv[++i]);
			cout << "-print_packing" << print_packing_text << endl;
		}
		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "packing_was_fixpoints_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		cout << "packing_was_fixpoints_activity_description::read_arguments looping, i=" << i << endl;
	} // next i
	cout << "packing_was_fixpoints_activity_description::read_arguments done" << endl;
	return i + 1;
}

void packing_was_fixpoints_activity_description::print()
{
	if (f_report) {
		cout << "-report" << endl;
	}
	if (f_print_packing) {
		cout << "-print_packing " << print_packing_text << endl;
	}
}




}}





