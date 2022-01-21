// spread_create_description.cpp
// 
// Anton Betten
//
// March 22, 2018
//
//
// 
//
//

#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace top_level {
namespace spreads {


spread_create_description::spread_create_description()
{
	null();
}

spread_create_description::~spread_create_description()
{
	freeself();
}

void spread_create_description::null()
{
	f_q = FALSE;
	q = 0;
	f_k = FALSE;
	k = 0;
	f_catalogue = FALSE;
	iso = 0;
	f_family = FALSE;
	//family_name;
}

void spread_create_description::freeself()
{
	null();
}

int spread_create_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int i;
	data_structures::string_tools ST;

	cout << "spread_create_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = ST.strtoi(argv[++i]);
			cout << "-q " << q << endl;
		}
		else if (ST.stringcmp(argv[i], "-k") == 0) {
			f_k = TRUE;
			k = ST.strtoi(argv[++i]);
			cout << "-k " << k << endl;
		}
		else if (ST.stringcmp(argv[i], "-catalogue") == 0) {
			f_catalogue = TRUE;
			iso = ST.strtoi(argv[++i]);
			cout << "-catalogue " << iso << endl;
		}
		else if (ST.stringcmp(argv[i], "-family") == 0) {
			f_family = TRUE;
			family_name.assign(argv[++i]);
			cout << "-family " << family_name << endl;
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
	} // next i
	cout << "spread_create_description::read_arguments done" << endl;
	return i + 1;
}

void spread_create_description::print()
{
	if (f_q) {
		cout << "-q " << q << endl;
	}
	if (f_k) {
		cout << "-k " << k << endl;
	}
	if (f_catalogue) {
		cout << "-catalogue " << iso << endl;
	}
	if (f_family) {
		cout << "-family " << family_name << endl;
	}
}


}}}
