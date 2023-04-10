/*
 * symbolic_object_builder_description.cpp
 *
 *  Created on: Apr 7, 2023
 *      Author: betten
 */






#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace data_structures {


symbolic_object_builder_description::symbolic_object_builder_description()
{
	f_text = false;
	//std::string text_txt;

	f_ring = false;
	//std::string ring_label;

	f_file = false;
	//std::string file_name;

	f_matrix = false;
	nb_rows = 0;

	f_test = false;
	//std::string test_object1;

}

symbolic_object_builder_description::~symbolic_object_builder_description()
{
}


int symbolic_object_builder_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i = 0;
	string_tools ST;

	if (f_v) {
		cout << "symbolic_object_builder_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-text") == 0) {
			f_text = true;
			text_txt.assign(argv[++i]);
			if (f_v) {
				cout << "-text " << text_txt << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-ring") == 0) {
			f_ring = true;
			ring_label.assign(argv[++i]);
			if (f_v) {
				cout << "-ring " << ring_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-file") == 0) {
			f_file = true;
			file_name.assign(argv[++i]);
			if (f_v) {
				cout << "-file " << file_name << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-matrix") == 0) {
			f_matrix = true;
			nb_rows = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-matrix " << nb_rows << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-test") == 0) {
			f_test = true;
			test_object1 = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-test " << test_object1 << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "symbolic_object_builder_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "symbolic_object_builder_description::read_arguments done" << endl;
	}
	return i + 1;
}

void symbolic_object_builder_description::print()
{
	if (f_text) {
		cout << "-text " << text_txt << endl;
	}
	if (f_ring) {
		cout << "-ring " << ring_label << endl;
	}
	if (f_file) {
		cout << "-file " << file_name << endl;
	}
	if (f_matrix) {
		cout << "-matrix " << nb_rows << endl;
	}
	if (f_test) {
		cout << "-test " << test_object1 << endl;
	}

}


}}}

