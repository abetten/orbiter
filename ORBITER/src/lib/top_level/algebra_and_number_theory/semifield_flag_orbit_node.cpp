/*
 * semifield_flag_orbit_node.cpp
 *
 *  Created on: Apr 17, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



semifield_flag_orbit_node::semifield_flag_orbit_node()
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

semifield_flag_orbit_node::~semifield_flag_orbit_node()
{
	//freeself();
}


void semifield_flag_orbit_node::init(
	int downstep_primary_orbit, int downstep_secondary_orbit,
	int pt_local, long int pt, int downstep_orbit_len, int f_long_orbit,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_flag_orbit_node:init" << endl;
	}
	semifield_flag_orbit_node::downstep_primary_orbit = downstep_primary_orbit;
	semifield_flag_orbit_node::downstep_secondary_orbit = downstep_secondary_orbit;
	semifield_flag_orbit_node::pt_local = pt_local;
	semifield_flag_orbit_node::pt = pt;
	semifield_flag_orbit_node::downstep_orbit_len = downstep_orbit_len;
	semifield_flag_orbit_node::f_long_orbit = f_long_orbit;
	if (f_v) {
		cout << "semifield_flag_orbit_node:init done" << endl;
	}

}

void semifield_flag_orbit_node::group_order(longinteger_object &go)
{
	if (f_long_orbit) {
		go.create(1);
		}
	else {
		semifield_flag_orbit_node::go.assign_to(go);
		}
}

int semifield_flag_orbit_node::group_order_as_int()
{
	if (f_long_orbit) {
		return 1;
		}
	else {
		return go.as_int();;
		}
}



}}

