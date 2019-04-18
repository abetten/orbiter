/*
 * semifield_downstep_node.cpp
 *
 *  Created on: Apr 17, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



semifield_downstep_node::semifield_downstep_node()
{
	SC = NULL;
	SL = NULL;
	F = NULL;
	k = 0;
	k2 = 0;
	level = 0;
	orbit_number = 0;
	Candidates = NULL;
	nb_candidates = 0;
	subspace_basis = NULL;
	subspace_base_cols = NULL;
	on_cosets = NULL;
	A_on_cosets = NULL;
	Sch = NULL;
	first_middle_orbit = 0;
	//null();
}

semifield_downstep_node::~semifield_downstep_node()
{
	//freeself();
}



}}

