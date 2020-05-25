/*
 * design_create_description.cpp
 *
 *  Created on: Sep 19, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


design_create_description::design_create_description()
{
	f_q = FALSE;
	q = 0;
	f_catalogue = FALSE;
	iso = 0;
	f_family = FALSE;
	family_name = NULL;
	//null();
}

design_create_description::~design_create_description()
{
	freeself();
}

void design_create_description::null()
{
}

void design_create_description::freeself()
{
	null();
}

int design_create_description::read_arguments(int argc, const char **argv,
	int verbose_level)
{
	int i;

	cout << "design_create_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (argv[i][0] != '-') {
			continue;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
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
			break;
			}
		} // next i
	cout << "design_create_description::read_arguments done" << endl;
	return i + 1;
}


int design_create_description::get_q()
{
	if (!f_q) {
		cout << "design_create_description::get_q "
				"q has not been set yet" << endl;
		exit(1);
	}
	return q;
}

}}




