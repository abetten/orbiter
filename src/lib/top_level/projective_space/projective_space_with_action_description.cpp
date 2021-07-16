/*
 * projective_space_with_action_description.cpp
 *
 *  Created on: Jan 5, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {




projective_space_with_action_description::projective_space_with_action_description()
{
	n = 0;
	//input_q;
	F = NULL;

}

projective_space_with_action_description::~projective_space_with_action_description()
{
}


int projective_space_with_action_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i = 0;

	if (f_v) {
		cout << "projective_space_with_action_description::read_arguments" << endl;
		cout << "next argument is " << argv[i] << endl;
	}
	n = strtoi(argv[i++]);
	if (f_v) {
		cout << "n = " << n << endl;
	}
	input_q.assign(argv[i++]);
	if (f_v) {
		cout << "q = " << input_q << endl;
		cout << "projective_space_with_action_description::read_arguments done" << endl;
	}
	return i;
}

void projective_space_with_action_description::print()
{
	cout << n << " " << input_q << endl;
}

}}
