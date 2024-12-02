/*
 * spread_table_activity_description.cpp
 *
 *  Created on: Apr 3, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace spreads {



spread_table_activity_description::spread_table_activity_description()
{
	Record_birth();
	f_find_spread = false;
	//std::string find_spread_text

	f_find_spread_and_dualize = false;
	//std::string find_spread_and_dualize_text

	f_dualize_packing = false;
	//std::string dualize_packing_text;

	f_print_spreads = false;
	//std::string print_spreads_idx_text;

	f_export_spreads_to_csv = false;
	//std::string export_spreads_to_csv_fname;
	//std::string export_spreads_to_csv_idx_text;


	f_find_spreads_containing_two_lines = false;
	find_spreads_containing_two_lines_line1 = 0;
	find_spreads_containing_two_lines_line2 = 0;

	f_find_spreads_containing_one_line = false;
	find_spreads_containing_one_line_line_idx = 0;

}

spread_table_activity_description::~spread_table_activity_description()
{
	Record_death();

}

int spread_table_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;
	other::data_structures::string_tools ST;

	cout << "spread_table_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		cout << "spread_table_activity_description::read_arguments, next argument is " << argv[i] << endl;

		if (ST.stringcmp(argv[i], "-find_spread") == 0) {
			f_find_spread = true;
			find_spread_text.assign(argv[++i]);
			cout << "-find_spread " << find_spread_text << endl;
		}
		else if (ST.stringcmp(argv[i], "-find_spread_and_dualize") == 0) {
			f_find_spread_and_dualize = true;
			find_spread_and_dualize_text.assign(argv[++i]);
			cout << "-find_spread_and_dualize " << find_spread_and_dualize_text << endl;
		}
		else if (ST.stringcmp(argv[i], "-dualize_packing") == 0) {
			f_dualize_packing = true;
			dualize_packing_text.assign(argv[++i]);
			cout << "-dualize_packing " << dualize_packing_text << endl;
		}
		else if (ST.stringcmp(argv[i], "-print_spreads") == 0) {
			f_print_spreads = true;
			print_spreads_idx_text.assign(argv[++i]);
			cout << "-print_spreads " << print_spreads_idx_text << endl;
		}
		else if (ST.stringcmp(argv[i], "-export_spreads_to_csv") == 0) {
			f_export_spreads_to_csv = true;
			export_spreads_to_csv_fname.assign(argv[++i]);
			export_spreads_to_csv_idx_text.assign(argv[++i]);
			cout << "-export_spreads_to_csv " << export_spreads_to_csv_fname
					<< " " << export_spreads_to_csv_idx_text << endl;
		}
		else if (ST.stringcmp(argv[i], "-find_spreads_containing_two_lines") == 0) {
			f_find_spreads_containing_two_lines = true;
			find_spreads_containing_two_lines_line1 = ST.strtoi(argv[++i]);
			find_spreads_containing_two_lines_line2 = ST.strtoi(argv[++i]);
			cout << "-find_spreads_containing_two_lines "
					<< " " << find_spreads_containing_two_lines_line1
					<< " " << find_spreads_containing_two_lines_line2
					<< endl;
		}

		else if (ST.stringcmp(argv[i], "-find_spreads_containing_one_line") == 0) {
			f_find_spreads_containing_one_line = true;
			find_spreads_containing_one_line_line_idx = ST.strtoi(argv[++i]);
			cout << "-find_spreads_containing_one_line "
					<< " " << find_spreads_containing_one_line_line_idx
					<< endl;
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
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




}}}



