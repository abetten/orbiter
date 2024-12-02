/*
 * blt_set_classify_description.cpp
 *
 *  Created on: Aug 2, 2022
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {

blt_set_classify_description::blt_set_classify_description()
{
	Record_birth();

	f_orthogonal_space = false;
	//std::string orthogonal_space_label;

	f_starter_size = false;
	starter_size = 0;
}

blt_set_classify_description::~blt_set_classify_description()
{
	Record_death();
}

int blt_set_classify_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "blt_set_classify_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-starter_size") == 0) {
			f_starter_size = true;
			starter_size = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-starter_size " << starter_size << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-orthogonal_space") == 0) {
			f_orthogonal_space = true;
			orthogonal_space_label.assign(argv[++i]);
			if (f_v) {
				cout << "-orthogonal_space " << orthogonal_space_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "blt_set_classify_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "blt_set_classify_description::read_arguments done" << endl;
	}
	return i + 1;
}

void blt_set_classify_description::print()
{
	if (f_starter_size) {
		cout << "-starter_size " << starter_size << endl;
	}
	if (f_orthogonal_space) {
		cout << "-orthogonal_space " << orthogonal_space_label << endl;
	}
}


}}}


