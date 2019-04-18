/*
 * semifield_lifting.cpp
 *
 *  Created on: Apr 17, 2019
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



semifield_lifting::semifield_lifting()
{
	SC = NULL;
	L2 = NULL;
	Prev = NULL;

	level = 0;
	prev_level_nb_orbits = 0;

	f_prefix = FALSE;
	prefix = NULL;

	Candidates = NULL;
	Nb_candidates = NULL;

	Downstep_nodes = NULL;
	nb_middle_layer_nodes = 0;
	Middle_layer_nodes = NULL;

	nb_orbits = 0;
	Po = NULL;
	So = NULL;
	Mo = NULL;
	Pt = NULL;
	Stabilizer_gens = NULL;
}

semifield_lifting::~semifield_lifting()
{

}


}}
