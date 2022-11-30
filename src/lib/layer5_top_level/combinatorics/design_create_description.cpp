/*
 * design_create_description.cpp
 *
 *  Created on: Sep 19, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {


design_create_description::design_create_description()
{
	f_field = FALSE;
	//std::string field_label;

	f_catalogue = FALSE;
	iso = 0;
	f_family = FALSE;
	//family_name;

	f_list_of_blocks_coded = FALSE;
	list_of_blocks_coded_v = 0;
	list_of_blocks_coded_k = 0;
	//std::string list_of_blocks_coded_label;

	f_list_of_sets_coded = FALSE;
	list_of_sets_coded_v = 0;
	//std::string list_of_sets_coded_label;


	f_list_of_blocks_coded_from_file = FALSE;
	//std::string list_of_blocks_coded_from_file_fname;

	f_list_of_blocks_from_file = FALSE;
	list_of_blocks_from_file_v = 0;
	//std::string list_of_blocks_from_file_fname;

	f_wreath_product_designs = FALSE;
	wreath_product_designs_n = 0;
	wreath_product_designs_k = 0;

	f_no_group = FALSE;
}

design_create_description::~design_create_description()
{
}

int design_create_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	cout << "design_create_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-field") == 0) {
			f_field = TRUE;
			field_label.assign(argv[++i]);
			if (f_v) {
				cout << "-field " << field_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-catalogue") == 0) {
			f_catalogue = TRUE;
			iso = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-catalogue " << iso << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-family") == 0) {
			f_family = TRUE;
			family_name.assign(argv[++i]);
			if (f_v) {
				cout << "-family " << family_name << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-list_of_blocks_coded") == 0) {
			f_list_of_blocks_coded = TRUE;
			list_of_blocks_coded_v = ST.strtoi(argv[++i]);
			list_of_blocks_coded_k = ST.strtoi(argv[++i]);
			list_of_blocks_coded_label.assign(argv[++i]);
			if (f_v) {
				cout << "-list_of_blocks " << list_of_blocks_coded_v
						<< " " << list_of_blocks_coded_k
						<< " " << list_of_blocks_coded_label
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-list_of_sets_coded") == 0) {
			f_list_of_sets_coded = TRUE;
			list_of_sets_coded_v = ST.strtoi(argv[++i]);
			list_of_sets_coded_label.assign(argv[++i]);
			if (f_v) {
				cout << "-list_of_sets_coded " << list_of_sets_coded_v
						<< " " << list_of_sets_coded_label
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-list_of_blocks_coded_from_file") == 0) {
			f_list_of_blocks_coded_from_file = TRUE;
			list_of_blocks_coded_v = ST.strtoi(argv[++i]);
			list_of_blocks_coded_k = ST.strtoi(argv[++i]);
			list_of_blocks_coded_from_file_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-list_of_blocks_coded_from_file " << list_of_blocks_coded_v
						<< " " << list_of_blocks_coded_k
						<< " " << list_of_blocks_coded_from_file_fname
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-list_of_blocks_from_file") == 0) {
			f_list_of_blocks_from_file = TRUE;
			list_of_blocks_from_file_v = ST.strtoi(argv[++i]);
			list_of_blocks_from_file_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-list_of_blocks_from_file " << list_of_blocks_from_file_v
						<< " " << list_of_blocks_from_file_fname
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-wreath_product_designs") == 0) {
			f_wreath_product_designs = TRUE;
			wreath_product_designs_n = ST.strtoi(argv[++i]);
			wreath_product_designs_k = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-wreath_product_designs "
						<< " " << wreath_product_designs_n
						<< " " << wreath_product_designs_k
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-no_group") == 0) {
			f_no_group = TRUE;
			if (f_v) {
				cout << "-no_group " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "design_create_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "design_create_description::read_arguments done" << endl;
	return i + 1;
}


void design_create_description::print()
{
	if (f_field) {
		cout << "-field " << field_label << endl;
	}
	if (f_catalogue) {
		cout << "-catalogue " << iso << endl;
	}
	if (f_family) {
		cout << "-family " << family_name << endl;
	}
	if (f_list_of_blocks_coded) {
		cout << "-list_of_blocks " << list_of_blocks_coded_v
				<< " " << list_of_blocks_coded_k
				<< " " << list_of_blocks_coded_label
				<< endl;
	}
	if (f_list_of_sets_coded) {
		cout << "-list_of_sets " << list_of_sets_coded_v
				<< " " << list_of_sets_coded_label
				<< endl;
	}
	if (f_list_of_blocks_coded_from_file) {
		cout << "-list_of_blocks_coded_from_file " << list_of_blocks_coded_v
				<< " " << list_of_blocks_coded_k
				<< " " << list_of_blocks_coded_from_file_fname
				<< endl;
	}
	if (f_list_of_blocks_from_file) {
		cout << "-list_of_blocks_from_file " << list_of_blocks_from_file_v
				<< " " << list_of_blocks_from_file_fname
				<< endl;
	}
	if (f_wreath_product_designs) {
		cout << "-wreath_product_designs "
				<< " " << wreath_product_designs_n
				<< " " << wreath_product_designs_k
				<< endl;
	}
	if (f_no_group) {
		cout << "-no_group " << endl;
	}
}


}}}




