// flag_orbit_node.C
// 
// Anton Betten
// September 23, 2017
//
//
// 
//
//

#include "orbiter.h"

flag_orbit_node::flag_orbit_node()
{
	null();
}

flag_orbit_node::~flag_orbit_node()
{
	freeself();
}

void flag_orbit_node::null()
{
	Flag_orbits = NULL;
	flag_orbit_index = -1;
	downstep_primary_orbit = -1;
	downstep_secondary_orbit = -1;
	upstep_primary_orbit = -1;
	upstep_secondary_orbit = -1;
	downstep_orbit_len = 0;
	f_fusion_node = FALSE;
	fusion_with = -1;
	fusion_elt = NULL;
	gens = NULL;
}

void flag_orbit_node::freeself()
{
	if (fusion_elt) {
		FREE_INT(fusion_elt);
		}
	if (gens) {
		delete gens;
		}
	null();
}

void flag_orbit_node::init(flag_orbits *Flag_orbits, INT flag_orbit_index, 
	INT downstep_primary_orbit, INT downstep_secondary_orbit, 
	INT downstep_orbit_len, INT f_long_orbit, 
	INT *pt_representation, strong_generators *Strong_gens, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag_orbit_node::init" << endl;
		}
	flag_orbit_node::Flag_orbits = Flag_orbits;
	flag_orbit_node::flag_orbit_index = flag_orbit_index;
	flag_orbit_node::downstep_primary_orbit = downstep_primary_orbit;
	flag_orbit_node::downstep_secondary_orbit = downstep_secondary_orbit;
	flag_orbit_node::downstep_orbit_len = downstep_orbit_len;
	flag_orbit_node::f_long_orbit = FALSE;
	INT_vec_copy(pt_representation, Flag_orbits->Pt + flag_orbit_index * Flag_orbits->pt_representation_sz, Flag_orbits->pt_representation_sz);
	gens = Strong_gens;
	if (f_v) {
		cout << "flag_orbit_node::init done" << endl;
		}
}

void flag_orbit_node::write_file(ofstream &fp, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "flag_orbit_node::write_file" << endl;
		}
	fp.write((char *) &downstep_primary_orbit, sizeof(INT));
	fp.write((char *) &downstep_secondary_orbit, sizeof(INT));
	fp.write((char *) &downstep_orbit_len, sizeof(INT));
	fp.write((char *) &upstep_primary_orbit, sizeof(INT));
	fp.write((char *) &upstep_secondary_orbit, sizeof(INT));
	fp.write((char *) &f_fusion_node, sizeof(INT));
	if (f_fusion_node) {
		fp.write((char *) &fusion_with, sizeof(INT));
		Flag_orbits->A->element_write_to_file_binary(fusion_elt, fp, 0);
		}
	gens->write_to_file_binary(fp, 0 /* verbose_level */);

	if (f_v) {
		cout << "flag_orbit_node::write_file finished" << endl;
		}
}

void flag_orbit_node::read_file(ifstream &fp, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "flag_orbit_node::read_file" << endl;
		}
	
	fp.read((char *) &downstep_primary_orbit, sizeof(INT));
	fp.read((char *) &downstep_secondary_orbit, sizeof(INT));
	fp.read((char *) &downstep_orbit_len, sizeof(INT));
	fp.read((char *) &upstep_primary_orbit, sizeof(INT));
	fp.read((char *) &upstep_secondary_orbit, sizeof(INT));
	fp.read((char *) &f_fusion_node, sizeof(INT));
	if (f_fusion_node) {
		fp.read((char *) &fusion_with, sizeof(INT));
		fusion_elt = NEW_INT(Flag_orbits->A->elt_size_in_INT);
		Flag_orbits->A->element_read_from_file_binary(fusion_elt, fp, 0);
		}

	if (FALSE) {
		cout << "flag_orbit_node::read_file before gens->read_from_file_binary" << endl;
		}
	gens = new strong_generators;
	gens->read_from_file_binary(Flag_orbits->A, fp, verbose_level);

	if (f_v) {
		cout << "flag_orbit_node::read_file finished" << endl;
		}
}


