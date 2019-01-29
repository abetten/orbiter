// simeon.cpp
//
// classifies arcs and uses a method of Simeon Ball and Raymond Hill
// to create larger arcs
//
// Anton Betten
// September 25, 2017
//
// the class arc_lifting_simeon can be found in
// src/top_level/geometry/arc_lifting_simeon.cpp

#include "orbiter.h"

using namespace orbiter;
using namespace orbiter::top_level;


int main()
{
	int i;
#if 0
	int q = 11;
	int d = 2; // largest number of points per line
	int n = 2; // projective dimension
	int k = 9; // size of the arc
#else
	int q = 11;
	int d = 2; // largest number of points per line
	int n = 2; // projective dimension
	int k = 5; // size of the arc
#endif
	int verbose_level = 5;
	
	arc_lifting_simeon *Simeon;

	Simeon = NEW_OBJECT(arc_lifting_simeon);

	cout << "before Simeon->init" << endl;
	Simeon->init(q, d, n, k,
			verbose_level);
	cout << "after Simeon->init" << endl;


	int nb_orbits;

	nb_orbits = Simeon->Gen->nb_orbits_at_level(k);
	cout << "We found " << nb_orbits
			<< " orbits of subsets of size " << k << endl;

	for (i = 0; i < nb_orbits; i++) {

		set_and_stabilizer *SaS;

		SaS = Simeon->Gen->get_set_and_stabilizer(
				k /* level */,
				i /* orbit_at_level */,
				0 /* verbose_level */);
		cout << "orbit " << i << " / " << nb_orbits << " : ";
		SaS->print_set_tex(cout);
		cout << endl;

		SaS->print_generators_tex(cout);

		if (i == 0) {
			Simeon->do_covering_problem(SaS);
		}

	}


}

