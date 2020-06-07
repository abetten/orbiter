// orbiter.cpp
//
// by Anton Betten
//
// started: 4/3/2020
//

#include "orbiter.h"

using namespace std;
using namespace orbiter;
using namespace orbiter::interfaces;


int main(int argc, const char **argv)
{

	//cout << "orbiter.out main" << endl;

	orbiter_session Session;
	int i;


	// setup:


	i = Session.read_arguments(argc, argv, 1);

	int verbose_level;

	verbose_level = Session.verbose_level;

	int f_v = (Session.verbose_level > 1);

	if (f_v) {
		cout << "Welcome to Orbiter!" << endl;
	}

	if (Session.f_seed) {
		os_interface Os;

		if (f_v) {
			cout << "seeding random number generator with " << Session.the_seed << endl;
		}
		srand(Session.the_seed);
		Os.random_integer(1000);
	}

	// main dispatch:

	Session.work(argc, argv, i, verbose_level);


	// finish:

	if (f_memory_debug) {
		global_mem_object_registry.dump();
	}

	the_end(Session.t0);

}


