/*
 * vector_ge_description.cpp
 *
 *  Created on: Jul 2, 2022
 *      Author: betten
 */



#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace data_structures_groups {


vector_ge_description::vector_ge_description()
{

	f_action = false;
	//std::string action_label;

	f_read_csv = false;
	//std::string read_csv_fname;
	//std::string read_csv_column_label;

	f_vector_data = false;
	//std::string vector_data_label;

}

vector_ge_description::~vector_ge_description()
{
}

int vector_ge_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "vector_ge_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-action") == 0) {
			f_action = true;
			action_label.assign(argv[++i]);
			if (f_v) {
				cout << "-action " << action_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-read_csv") == 0) {
			f_read_csv = true;
			read_csv_fname.assign(argv[++i]);
			read_csv_column_label.assign(argv[++i]);
			if (f_v) {
				cout << "-read_csv " << read_csv_fname << " " << read_csv_column_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-vector_data") == 0) {
			f_vector_data = true;
			vector_data_label.assign(argv[++i]);
			if (f_v) {
				cout << "-vector_data " << vector_data_label << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "group_theoretic_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "group_theoretic_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void vector_ge_description::print()
{
	if (f_action) {
		cout << "-action " << action_label << endl;
	}
	if (f_read_csv) {
		cout << "-read_csv " << read_csv_fname << " " << read_csv_column_label << endl;
	}
	if (f_vector_data) {
		cout << "-vector_data " << vector_data_label << endl;
	}

}





}}}


