/*
 * spread_table_activity_description.cpp
 *
 *  Created on: Apr 3, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



spread_table_activity_description::spread_table_activity_description()
{
	f_find_spread = FALSE;
	//std::string find_spread_text

	f_find_spread_and_dualize = FALSE;
	//std::string find_spread_and_dualize_text

	f_dualize_packing = FALSE;
	//std::string dualize_packing_text;

	f_print_spreads = FALSE;
	//std::string print_spreads_idx_text;

	f_export_spreads_to_csv = FALSE;
	//std::string export_spreads_to_csv_fname;
	//std::string export_spreads_to_csv_idx_text;


	f_find_spreads_containing_two_lines = FALSE;
	find_spreads_containing_two_lines_line1 = 0;
	find_spreads_containing_two_lines_line2 = 0;

	f_find_spreads_containing_one_line = FALSE;
	find_spreads_containing_one_line_line_idx = 0;

}

spread_table_activity_description::~spread_table_activity_description()
{

}

int spread_table_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "spread_table_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		cout << "spread_table_activity_description::read_arguments, next argument is " << argv[i] << endl;

		if (stringcmp(argv[i], "-find_spread") == 0) {
			f_find_spread = TRUE;
			find_spread_text.assign(argv[++i]);
			cout << "-find_spread " << find_spread_text << endl;
		}
		else if (stringcmp(argv[i], "-find_spread_and_dualize") == 0) {
			f_find_spread_and_dualize = TRUE;
			find_spread_and_dualize_text.assign(argv[++i]);
			cout << "-find_spread_and_dualize " << find_spread_and_dualize_text << endl;
		}
		else if (stringcmp(argv[i], "-dualize_packing") == 0) {
			f_dualize_packing = TRUE;
			dualize_packing_text.assign(argv[++i]);
			cout << "-dualize_packing " << dualize_packing_text << endl;
		}
		else if (stringcmp(argv[i], "-print_spreads") == 0) {
			f_print_spreads = TRUE;
			print_spreads_idx_text.assign(argv[++i]);
			cout << "-print_spreads " << print_spreads_idx_text << endl;
		}
		else if (stringcmp(argv[i], "-export_spreads_to_csv") == 0) {
			f_export_spreads_to_csv = TRUE;
			export_spreads_to_csv_fname.assign(argv[++i]);
			export_spreads_to_csv_idx_text.assign(argv[++i]);
			cout << "-export_spreads_to_csv " << export_spreads_to_csv_fname
					<< " " << export_spreads_to_csv_idx_text << endl;
		}
		else if (stringcmp(argv[i], "-find_spreads_containing_two_lines") == 0) {
			f_find_spreads_containing_two_lines = TRUE;
			find_spreads_containing_two_lines_line1 = strtoi(argv[++i]);
			find_spreads_containing_two_lines_line2 = strtoi(argv[++i]);
			cout << "-find_spreads_containing_two_lines "
					<< " " << find_spreads_containing_two_lines_line1
					<< " " << find_spreads_containing_two_lines_line2
					<< endl;
		}

		else if (stringcmp(argv[i], "-find_spreads_containing_one_line") == 0) {
			f_find_spreads_containing_one_line = TRUE;
			find_spreads_containing_one_line_line_idx = strtoi(argv[++i]);
			cout << "-find_spreads_containing_one_line "
					<< " " << find_spreads_containing_one_line_line_idx
					<< endl;
		}

		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "spread_table_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		cout << "spread_table_activity_description::read_arguments looping, i=" << i << endl;
	} // next i
	cout << "spread_table_activity_description::read_arguments done" << endl;
	return i + 1;
}

void spread_table_activity_description::print()
{
	if (f_find_spread) {
		cout << "-find_spread " << find_spread_text << endl;
	}
	if (f_find_spread_and_dualize) {
		cout << "-find_spread_and_dualize " << find_spread_and_dualize_text << endl;
	}
	if (f_dualize_packing) {
		cout << "-dualize_packing " << dualize_packing_text << endl;
	}
	if (f_print_spreads) {
		cout << "-print_spreads " << print_spreads_idx_text << endl;
	}
	if (f_export_spreads_to_csv) {
		cout << "-export_spreads_to_csv " << export_spreads_to_csv_fname
				<< " " << export_spreads_to_csv_idx_text << endl;
	}
	if (f_find_spreads_containing_two_lines) {
		cout << "-find_spreads_containing_two_lines "
				<< " " << find_spreads_containing_two_lines_line1
				<< " " << find_spreads_containing_two_lines_line2
				<< endl;
	}

	if (f_find_spreads_containing_one_line) {
		cout << "-find_spreads_containing_one_line "
				<< " " << find_spreads_containing_one_line_line_idx
				<< endl;
	}
}




}}



