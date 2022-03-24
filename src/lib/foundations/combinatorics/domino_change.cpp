/*
 * domino_change.cpp
 *
 *  Created on: Mar 2, 2020
 *      Author: betten
 */





#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {


domino_change::domino_change()
{
	type_of_change = 0;
	cost_after_change = 0;
}

domino_change::~domino_change()
{

}

void domino_change::init(domino_assignment *DA,
		int type_of_change, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "domino_change::init" << endl;
	}

	domino_change::type_of_change = type_of_change;
	cost_after_change = DA->cost_function();

	cout << "domino_change::init cost = " << cost_after_change << endl;


	if (f_v) {
		cout << "domino_change::init done" << endl;
	}
}

}}}




