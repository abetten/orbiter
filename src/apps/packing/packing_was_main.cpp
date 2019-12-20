/*
 * packing_was_main.cpp
 *
 * was = with assumed symmetry
 *
 *  Created on: Aug 7, 2019
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;


// global data:

int t0; // the system time when the program started


int main(int argc, const char **argv)
{
	packing_was P;
	os_interface Os;

	t0 = Os.os_ticks();

	P.init(argc, argv);

	the_end(t0);
	//the_end_quietly(t0);

}


