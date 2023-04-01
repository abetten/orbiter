// BLT_set_create_description.cpp
// 
// Anton Betten
//
// March 17, 2018
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {


BLT_set_create_description::BLT_set_create_description()
{
	//f_q = FALSE;
	//q = 0;
	f_catalogue = FALSE;
	iso = 0;
	f_family = FALSE;
	//family_name;

	f_flock = FALSE;
	//std::string flock_label;

	f_space = FALSE;
	//std::string space_label;

	//f_space_pointer = FALSE;
	//space_pointer = NULL;

	f_invariants = FALSE;
}

BLT_set_create_description::~BLT_set_create_description()
{
}

int BLT_set_create_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "BLT_set_create_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-catalogue") == 0) {
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
		else if (ST.stringcmp(argv[i], "-flock") == 0) {
			f_flock = TRUE;
			flock_label.assign(argv[++i]);
			if (f_v) {
				cout << "-flock " << flock_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-space") == 0) {
			f_space = TRUE;
			space_label.assign(argv[++i]);
			if (f_v) {
				cout << "-space " << space_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-invariants") == 0) {
			f_invariants = TRUE;
			if (f_v) {
				cout << "-invariants " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "BLT_set_create_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "BLT_set_create_description::read_arguments done" << endl;
	}
	return i + 1;
}

void BLT_set_create_description::print()
{
	if (f_catalogue) {
		cout << "-catalogue " << iso << endl;
	}
	if (f_family) {
		cout << "-family " << family_name << endl;
	}
	if (f_flock) {
		cout << "-flock " << flock_label << endl;
	}
	if (f_space) {
		cout << "-space " << space_label << endl;
	}
	if (f_invariants) {
		cout << "-invariants " << endl;
	}
}

}}}

