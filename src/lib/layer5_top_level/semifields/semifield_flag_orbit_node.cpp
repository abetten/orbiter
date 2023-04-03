/*
 * semifield_flag_orbit_node.cpp
 *
 *  Created on: Apr 17, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace semifields {



semifield_flag_orbit_node::semifield_flag_orbit_node()
{
	downstep_primary_orbit = 0;
	downstep_secondary_orbit = 0;
	pt_local = 0;
	pt = 0;
	downstep_orbit_len = 0;
	f_long_orbit = false;
	upstep_orbit = 0;
	f_fusion_node = false;
	fusion_with = 0;
	fusion_elt = NULL;

	//longinteger_object go;
	gens = NULL;

}

semifield_flag_orbit_node::~semifield_flag_orbit_node()
{
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

void semifield_flag_orbit_node::group_order(ring_theory::longinteger_object &go)
{
	if (f_long_orbit) {
		go.create(1, __FILE__, __LINE__);
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

void semifield_flag_orbit_node::write_to_file_binary(
		semifield_lifting *SL, ofstream &fp,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_flag_orbit_node::write_to_file_binary" << endl;
		}
	fp.write((char *) &downstep_primary_orbit, sizeof(int));
	fp.write((char *) &downstep_secondary_orbit, sizeof(int));
	fp.write((char *) &pt_local, sizeof(int));
	fp.write((char *) &pt, sizeof(long int));
	fp.write((char *) &downstep_orbit_len, sizeof(int));
	fp.write((char *) &f_long_orbit, sizeof(int));
	fp.write((char *) &f_fusion_node, sizeof(int));
	if (f_fusion_node) {
		fp.write((char *) &fusion_with, sizeof(int));
		SL->SC->A->Group_element->element_write_to_file_binary(fusion_elt, fp, 0);
		}
	else {
		fp.write((char *) &upstep_orbit, sizeof(int));
		}
	if (!f_long_orbit) {
		gens->write_to_file_binary(fp, verbose_level - 1);
		}
	if (f_v) {
		cout << "semifield_flag_orbit_node::write_to_file_binary done" << endl;
		}
}

void semifield_flag_orbit_node::read_from_file_binary(
		semifield_lifting *SL, ifstream &fp,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_flag_orbit_node::read_from_file_binary" << endl;
		}
	fp.read((char *) &downstep_primary_orbit, sizeof(int));
	fp.read((char *) &downstep_secondary_orbit, sizeof(int));
	fp.read((char *) &pt_local, sizeof(int));
	fp.read((char *) &pt, sizeof(long int));
	fp.read((char *) &downstep_orbit_len, sizeof(int));
	fp.read((char *) &f_long_orbit, sizeof(int));
	fp.read((char *) &f_fusion_node, sizeof(int));
	if (f_fusion_node) {
		fp.read((char *) &fusion_with, sizeof(int));
		fusion_elt = NEW_int(SL->SC->A->elt_size_in_int);
		SL->SC->A->Group_element->element_read_from_file_binary(fusion_elt, fp, 0);
		}
	else {
		fp.read((char *) &upstep_orbit, sizeof(int));
		}
	if (!f_long_orbit) {
		gens = NEW_OBJECT(groups::strong_generators);
		gens->read_from_file_binary(SL->SC->A, fp, verbose_level - 1);
		}
	if (f_v) {
		cout << "semifield_flag_orbit_node::read_from_file_binary done" << endl;
		}
}





}}}

