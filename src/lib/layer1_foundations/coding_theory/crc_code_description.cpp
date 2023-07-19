/*
 * crc_code_description.cpp
 *
 *  Created on: Jul 19, 2023
 *      Author: betten
 */






#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace coding_theory {




crc_code_description::crc_code_description()
{

	f_type = false;
	//std::string type;

	f_block_length = false;
	block_length = 0;

}

crc_code_description::~crc_code_description()
{
}


int crc_code_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "crc_code_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-type") == 0) {
			f_type = true;
			type.assign(argv[++i]);
			if (f_v) {
				cout << "-type " << type << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-block_length") == 0) {
			f_block_length = true;
			block_length = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-block_length " << block_length << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
			}
		else {
			cout << "crc_code_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}

	} // next i
	if (f_v) {
		cout << "crc_code_description::read_arguments done" << endl;
	}
	return i + 1;
}

void crc_code_description::print()
{
	if (f_type) {
		cout << "-type " << type << endl;
	}
	if (f_block_length) {
		cout << "-block_length " << block_length << endl;
	}

}



}}}



