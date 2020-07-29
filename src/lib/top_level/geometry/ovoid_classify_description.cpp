/*
 * ovoid_classify_description.cpp
 *
 *  Created on: Jul 28, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


ovoid_classify_description::ovoid_classify_description()
{
	Control = NULL;
	f_epsilon = FALSE;
	epsilon = 0;
	f_d = FALSE;
	d = 0;
}



ovoid_classify_description::~ovoid_classify_description()
{
}

int ovoid_classify_description::read_arguments(int argc, const char **argv,
	int verbose_level)
{
	int i;



	cout << "ovoid_classify_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (strcmp(argv[i], "-epsilon") == 0) {
			f_epsilon = TRUE;
			epsilon = atoi(argv[++i]);
			cout << "-epsilon " << epsilon << endl;
		}
		else if (strcmp(argv[i], "-d") == 0) {
			f_d = TRUE;
			d = atoi(argv[++i]);
			cout << "-d " << d << endl;
		}

		else if (strcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "ignoring argument " << argv[i] << endl;
		}
	} // next i

	cout << "ovoid_classify_description::read_arguments done" << endl;
	return i;
}



}}



