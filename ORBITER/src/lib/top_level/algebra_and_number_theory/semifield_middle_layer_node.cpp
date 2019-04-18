/*
 * semifield_middle_layer_node.cpp
 *
 *  Created on: Apr 17, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



semifield_middle_layer_node::semifield_middle_layer_node()
{
	downstep_primary_orbit = 0;
	downstep_secondary_orbit = 0;
	pt_local = 0;
	pt = 0;
	downstep_orbit_len = 0;
	f_long_orbit = FALSE;
	upstep_orbit = 0;
	f_fusion_node = FALSE;
	fusion_with = 0;
	fusion_elt = NULL;

	//longinteger_object go;
	gens = NULL;

	//null();
}

semifield_middle_layer_node::~semifield_middle_layer_node()
{
	//freeself();
}




}}

