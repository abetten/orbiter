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
namespace top_level {


BLT_set_create_description::BLT_set_create_description()
{
	//f_q = FALSE;
	//q = 0;
	f_catalogue = FALSE;
	iso = 0;
	f_family = FALSE;
	//family_name;
	null();
}

BLT_set_create_description::~BLT_set_create_description()
{
	freeself();
}

void BLT_set_create_description::null()
{
}

void BLT_set_create_description::freeself()
{
	null();
}

int BLT_set_create_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "BLT_set_create_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-catalogue") == 0) {
			f_catalogue = TRUE;
			iso = strtoi(argv[++i]);
			if (f_v) {
				cout << "-catalogue " << iso << endl;
			}
		}
		else if (stringcmp(argv[i], "-family") == 0) {
			f_family = TRUE;
			family_name.assign(argv[++i]);
			if (f_v) {
				cout << "-family " << family_name << endl;
			}
		}
		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
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
	else if (f_family) {
		cout << "-family " << family_name << endl;
	}
}

}}

