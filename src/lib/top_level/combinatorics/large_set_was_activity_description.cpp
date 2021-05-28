/*
 * large_set_was_activity_description.cpp
 *
 *  Created on: May 27, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


large_set_was_activity_description::large_set_was_activity_description()
{
	f_normalizer_on_orbits_of_a_given_length = FALSE;
	normalizer_on_orbits_of_a_given_length_length = 0;
}

large_set_was_activity_description::~large_set_was_activity_description()
{

}


int large_set_was_activity_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "large_set_was_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {


		if (stringcmp(argv[i], "-normalizer_on_orbits_of_a_given_length") == 0) {
			f_normalizer_on_orbits_of_a_given_length = TRUE;
			normalizer_on_orbits_of_a_given_length_length = strtoi(argv[++i]);
			cout << "-normalizer_on_orbits_of_a_given_length " << normalizer_on_orbits_of_a_given_length_length
					<< endl;
		}
		if (stringcmp(argv[i], "-end") == 0) {
			break;
		}
	} // next i
	cout << "large_set_was_activity_description::read_arguments done" << endl;
	return i + 1;
}




}}




