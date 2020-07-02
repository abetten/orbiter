/*
 * regular_linear_space_description.cpp
 *
 *  Created on: Jun 17, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


regular_linear_space_description::regular_linear_space_description()
{
	f_m = FALSE;
	m = 0;
	f_n = FALSE;
	n = 0;
	f_k = FALSE;
	k = 0;
	f_r = FALSE;
	r = 0;
	f_target_size = FALSE;
	target_size = 0;

	starter_size = 0;
	initial_pair_covering = NULL;

	f_has_control = FALSE;
	Control = NULL;

};

regular_linear_space_description::~regular_linear_space_description()
{
	if (initial_pair_covering) {
		FREE_int(initial_pair_covering);
	}
	if (Control) {
		FREE_OBJECT(Control);
	}

};


int regular_linear_space_description::read_arguments(
	int argc, const char **argv,
	int verbose_level)
{
	int i;

	cout << "regular_linear_space_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (strcmp(argv[i], "-m") == 0) {
			f_m = TRUE;
			m = atoi(argv[++i]);
			cout << "-m " << m << endl;
		}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
		}
		else if (strcmp(argv[i], "-k") == 0) {
			f_k = TRUE;
			k = atoi(argv[++i]);
			cout << "-k " << k << endl;
		}
		else if (strcmp(argv[i], "-r") == 0) {
			f_r = TRUE;
			r = atoi(argv[++i]);
			cout << "-r " << r << endl;
		}
		else if (strcmp(argv[i], "-target_size") == 0) {
			f_target_size = TRUE;
			target_size = atoi(argv[++i]);
			cout << "-target_size " << target_size << endl;
		}
		else if (strcmp(argv[i], "-control") == 0) {
			f_has_control = TRUE;
			Control = NEW_OBJECT(poset_classification_control);
			i += Control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -poset_classification_control " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			return i;
		}
		else {
			cout << "regular_linear_space_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "regular_linear_space_description::read_arguments done" << endl;
	return i;
}

}}
