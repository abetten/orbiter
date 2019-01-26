// spread_create_description.C
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


namespace orbiter {

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
	family_name = NULL;
}

void spread_create_description::freeself()
{
	null();
}

int spread_create_description::read_arguments(int argc, const char **argv, 
	int verbose_level)
{
	int i;

	cout << "spread_create_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (argv[i][0] != '-') {
			continue;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-k") == 0) {
			f_k = TRUE;
			k = atoi(argv[++i]);
			cout << "-k " << k << endl;
			}
		else if (strcmp(argv[i], "-catalogue") == 0) {
			f_catalogue = TRUE;
			iso = atoi(argv[++i]);
			cout << "-catalogue " << iso << endl;
			}
		else if (strcmp(argv[i], "-family") == 0) {
			f_family = TRUE;
			family_name = argv[++i];
			cout << "-family " << family_name << endl;
			}
		else if (strcmp(argv[i], "-end") == 0) {
			return i;
			}
		} // next i
	cout << "spread_create_description::read_arguments done" << endl;
	return i;
}


}
