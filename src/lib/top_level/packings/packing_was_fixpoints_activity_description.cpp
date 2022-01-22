/*
 * packing_was_fixpoints_activity_description.cpp
 *
 *  Created on: Apr 3, 2021
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace packings {



packing_was_fixpoints_activity_description::packing_was_fixpoints_activity_description()
{
	f_report = FALSE;

	f_print_packing = FALSE;
	//std::string print_packing_text;

	f_compare_files_of_packings = FALSE;
	//std::string compare_files_of_packings_fname1;
	//std::string compare_files_of_packings_fname2;

}

packing_was_fixpoints_activity_description::~packing_was_fixpoints_activity_description()
{

}

int packing_was_fixpoints_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;
	data_structures::string_tools ST;

	cout << "packing_was_fixpoints_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		cout << "packing_was_fixpoints_activity_description::read_arguments, next argument is " << argv[i] << endl;


		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report" << endl;
		}
		else if (ST.stringcmp(argv[i], "-print_packing") == 0) {
			f_print_packing = TRUE;
			print_packing_text.assign(argv[++i]);
			cout << "-print_packing" << print_packing_text << endl;
		}
		else if (ST.stringcmp(argv[i], "-compare_files_of_packings") == 0) {
			f_compare_files_of_packings = TRUE;
			compare_files_of_packings_fname1.assign(argv[++i]);
			compare_files_of_packings_fname2.assign(argv[++i]);
			cout << "-compare_files_of_packings" << compare_files_of_packings_fname1 << " " << compare_files_of_packings_fname2 << endl;
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
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
	if (f_compare_files_of_packings) {
		cout << "-compare_files_of_packings" << compare_files_of_packings_fname1 << " " << compare_files_of_packings_fname2 << endl;
	}
}




}}}





