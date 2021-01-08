/*
 * projective_space_with_action_description.cpp
 *
 *  Created on: Jan 5, 2021
 *      Author: betten
 */




#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;



namespace orbiter {
namespace group_actions {


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
	int i = 0;

	cout << "projective_space_with_action_description::read_arguments" << endl;
	cout << "next argument is " << argv[i] << endl;
	n = strtoi(argv[i++]);
	cout << "n = " << n << endl;
	input_q.assign(argv[i++]);
	cout << "q = " << input_q << endl;
	cout << "projective_space_with_action_description::read_arguments done" << endl;
	return i;
}

void projective_space_with_action_description::print()
{
	cout << "projective_space_with_action_description::print:" << endl;

	cout << "n = " << n << endl;
	cout << "q = " << input_q << endl;
}

}}
