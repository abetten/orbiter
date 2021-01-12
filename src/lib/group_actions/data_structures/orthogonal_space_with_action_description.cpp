/*
 * orthogonal_space_with_action_description.cpp
 *
 *  Created on: Jan 12, 2021
 *      Author: betten
 */




#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;



namespace orbiter {
namespace group_actions {


orthogonal_space_with_action_description::orthogonal_space_with_action_description()
{
	epsilon = 0;
	n = 0;
	//input_q;
	F = NULL;

}

orthogonal_space_with_action_description::~orthogonal_space_with_action_description()
{
}


int orthogonal_space_with_action_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i = 0;

	cout << "orthogonal_space_with_action_description::read_arguments" << endl;
	cout << "next argument is " << argv[i] << endl;
	epsilon = strtoi(argv[i++]);
	cout << "epsilon = " << epsilon << endl;
	n = strtoi(argv[i++]);
	cout << "n = " << n << endl;
	input_q.assign(argv[i++]);
	cout << "q = " << input_q << endl;
	cout << "orthogonal_space_with_action_description::read_arguments done" << endl;
	return i;
}

void orthogonal_space_with_action_description::print()
{
	cout << "orthogonal_space_with_action_description::print:" << endl;

	cout << "epsilon = " << epsilon << endl;
	cout << "n = " << n << endl;
	cout << "q = " << input_q << endl;
}

}}
