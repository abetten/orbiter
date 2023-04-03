/*
 * crc_process_description.cpp
 *
 *  Created on: Dec 9, 2022
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_coding_theory {


crc_process_description::crc_process_description()
{
	f_code = false;
	//std::string code_label;

	f_crc_options = false;
	Crc_options = NULL;

}


int crc_process_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "crc_process_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		code_modification_description M;

		if (ST.stringcmp(argv[i], "-code") == 0) {
			f_code = true;
			code_label.assign(argv[++i]);
			if (f_v) {
				cout << "-code " << code_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-crc_options") == 0) {
			f_crc_options = true;
			Crc_options = NEW_OBJECT(coding_theory::crc_options_description);
			if (f_v) {
				cout << "-crc_options" << endl;
			}
			i += Crc_options->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -crc_options " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
			if (f_v) {
				cout << "-crc_options " << endl;
				Crc_options->print();
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "crc_process_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "crc_process_description::read_arguments done" << endl;
	}
	return i + 1;
}

void crc_process_description::print()
{
	if (f_code) {
		cout << "-code " << code_label << endl;
	}
	if (f_crc_options) {
		cout << "-crc_options " << endl;
		Crc_options->print();
	}
}




}}}



