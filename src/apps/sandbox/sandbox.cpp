/*
 * sandbox.cpp
 *
 *  Created on: Apr 2, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;
using namespace orbiter;


int main()
{
	finite_field F;
	F.finite_field_init(16, 0);

	cout << "8 x 15 = " << F.mult(8, 15) << endl;

}

