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
	Record_birth();
	f_label = false;
	//std::string label_txt;
	//std::string label_tex;

	f_field = false;
	//std::string field_label;

	f_catalogue = false;
	iso = 0;

	f_family = false;
	//family_name;

	f_list_of_base_blocks = false;
	//std::string list_of_base_blocks_group_label;
	//std::string list_of_base_blocks_fname;
	//std::string list_of_base_blocks_col;
	//std::string list_of_base_blocks_selection_fname;
	//std::string list_of_base_blocks_selection_col;
	list_of_base_blocks_selection_idx = 0;

	f_list_of_blocks_coded = false;
	list_of_blocks_coded_v = 0;
	list_of_blocks_coded_k = 0;
	//std::string list_of_blocks_coded_label;

	f_list_of_sets_coded = false;
	list_of_sets_coded_v = 0;
	//std::string list_of_sets_coded_label;


	f_list_of_blocks_coded_from_file = false;
	//std::string list_of_blocks_coded_from_file_fname;

	f_list_of_blocks_from_file = false;
	list_of_blocks_from_file_v = 0;
	//std::string list_of_blocks_from_file_fname;
	//std::string list_of_blocks_from_file_column;

	f_wreath_product_designs = false;
	wreath_product_designs_n = 0;
	wreath_product_designs_k = 0;

	f_linear_space_from_latin_square = false;
	//std::string linear_space_from_latin_square_name;

	f_orbit_on_sets = false;
	//std::string orbit_on_sets_orbits;
	orbit_on_sets_size = -1;
	orbit_on_sets_idx = -1;

	f_no_group = false;

	f_block_partition = false;
	block_partition_sz = 0;

}

design_create_description::~design_create_description()
{
	Record_death();
}

int design_create_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	cout << "design_create_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-label") == 0) {
			f_label = true;
			label_txt.assign(argv[++i]);
			label_tex.assign(argv[++i]);
			if (f_v) {
				cout << "-label "
						<< " " << label_txt
						<< " " << label_tex
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-field") == 0) {
			f_field = true;
			field_label.assign(argv[++i]);
			if (f_v) {
				cout << "-field " << field_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-catalogue") == 0) {
			f_catalogue = true;
			iso = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-catalogue " << iso << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-family") == 0) {
			f_family = true;
			family_name.assign(argv[++i]);
			if (f_v) {
				cout << "-family " << family_name << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-list_of_base_blocks") == 0) {
			f_list_of_base_blocks = true;
			list_of_base_blocks_group_label.assign(argv[++i]);
			list_of_base_blocks_fname.assign(argv[++i]);
			list_of_base_blocks_col.assign(argv[++i]);
			list_of_base_blocks_selection_fname.assign(argv[++i]);
			list_of_base_blocks_selection_col.assign(argv[++i]);
			list_of_base_blocks_selection_idx = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-list_of_base_blocks"
						<< " " << list_of_base_blocks_group_label
						<< " " << list_of_base_blocks_fname
						<< " " << list_of_base_blocks_col
						<< " " << list_of_base_blocks_selection_fname
						<< " " << list_of_base_blocks_selection_col
						<< " " << list_of_base_blocks_selection_idx
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-list_of_blocks_coded") == 0) {
			f_list_of_blocks_coded = true;
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
			f_list_of_sets_coded = true;
			list_of_sets_coded_v = ST.strtoi(argv[++i]);
			list_of_sets_coded_label.assign(argv[++i]);
			if (f_v) {
				cout << "-list_of_sets_coded " << list_of_sets_coded_v
						<< " " << list_of_sets_coded_label
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-list_of_blocks_coded_from_file") == 0) {
			f_list_of_blocks_coded_from_file = true;
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
			f_list_of_blocks_from_file = true;
			list_of_blocks_from_file_v = ST.strtoi(argv[++i]);
			list_of_blocks_from_file_fname.assign(argv[++i]);
			list_of_blocks_from_file_column.assign(argv[++i]);
			if (f_v) {
				cout << "-list_of_blocks_from_file " << list_of_blocks_from_file_v
						<< " " << list_of_blocks_from_file_fname
						<< " " << list_of_blocks_from_file_column
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-wreath_product_designs") == 0) {
			f_wreath_product_designs = true;
			wreath_product_designs_n = ST.strtoi(argv[++i]);
			wreath_product_designs_k = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-wreath_product_designs "
						<< " " << wreath_product_designs_n
						<< " " << wreath_product_designs_k
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-linear_space_from_latin_square") == 0) {
			f_linear_space_from_latin_square = true;
			linear_space_from_latin_square_name.assign(argv[++i]);
			if (f_v) {
				cout << "-linear_space_from_latin_square "
						<< " " << linear_space_from_latin_square_name
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-orbit_on_sets") == 0) {
			f_orbit_on_sets = true;
			orbit_on_sets_orbits.assign(argv[++i]);
			orbit_on_sets_size = ST.strtoi(argv[++i]);
			orbit_on_sets_idx = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-orbit_on_sets "
						<< " " << orbit_on_sets_orbits
						<< " " << orbit_on_sets_size
						<< " " << orbit_on_sets_idx
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-no_group") == 0) {
			f_no_group = true;
			if (f_v) {
				cout << "-no_group " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-block_partition") == 0) {
			f_block_partition = true;
			block_partition_sz = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-block_partition " << block_partition_sz << endl;
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
	if (f_label) {
		cout << "-label "
				<< " " << label_txt
				<< " " << label_tex
				<< endl;
	}
	if (f_field) {
		cout << "-field " << field_label << endl;
	}
	if (f_catalogue) {
		cout << "-catalogue " << iso << endl;
	}
	if (f_family) {
		cout << "-family " << family_name << endl;
	}
	if (f_list_of_base_blocks) {
		cout << "-list_of_base_blocks"
				<< " " << list_of_base_blocks_group_label
				<< " " << list_of_base_blocks_fname
				<< " " << list_of_base_blocks_col
				<< " " << list_of_base_blocks_selection_fname
				<< " " << list_of_base_blocks_selection_col
				<< " " << list_of_base_blocks_selection_idx
				<< endl;
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
				<< " " << list_of_blocks_from_file_column
				<< endl;
	}
	if (f_wreath_product_designs) {
		cout << "-wreath_product_designs "
				<< " " << wreath_product_designs_n
				<< " " << wreath_product_designs_k
				<< endl;
	}
	if (f_linear_space_from_latin_square) {
		cout << "-linear_space_from_latin_square "
				<< " " << linear_space_from_latin_square_name
				<< endl;
	}
	if (f_orbit_on_sets) {
		cout << "-orbit_on_sets "
				<< " " << orbit_on_sets_orbits
				<< " " << orbit_on_sets_size
				<< " " << orbit_on_sets_idx
				<< endl;
	}
	if (f_no_group) {
		cout << "-no_group " << endl;
	}
	if (f_block_partition) {
		cout << "-block_partition " << block_partition_sz << endl;
	}
}


}}}




