// kramer_mesner.cpp
// 
// Anton Betten
// April 20, 2009
//
//
// 
//
//


#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;


// global data:

int t0; // the system time when the program started

int main(int argc, const char **argv);

int main(int argc, const char **argv)
{
	t0 = os_ticks();
	
	int verbose_level;
	
	{
	kramer_mesner KM;
	
	cout << "km.C: before read_arguments" << endl;
	KM.read_arguments(argc, argv, verbose_level);

	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	sims *S;

	cout << "km.C: before init_group" << endl;
	KM.init_group(S, verbose_level);


	cout << "km.C: before orbits" << endl;
	KM.orbits(argc, argv, S, verbose_level);

	delete S;
	}

	the_end(t0);
	//the_end_quietly(t0);
}


