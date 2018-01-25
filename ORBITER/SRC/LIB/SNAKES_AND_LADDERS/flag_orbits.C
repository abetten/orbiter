// flag_orbits.C
// 
// Anton Betten
// September 23, 2017
//
//
// 
//
//

#include "orbiter.h"

flag_orbits::flag_orbits()
{
	null();
}

flag_orbits::~flag_orbits()
{
	freeself();
}

void flag_orbits::null()
{
	A = NULL;
	A2 = NULL;
	nb_primary_orbits_lower = 0;
	nb_primary_orbits_upper = 0;
	nb_flag_orbits = 0;
	Flag_orbit_node = NULL;
	pt_representation_sz = 0;
	Pt = NULL;
}

void flag_orbits::freeself()
{
	if (Flag_orbit_node) {
		delete [] Flag_orbit_node;
		}
	if (Pt) {
		FREE_INT(Pt);
		}
	null();
}

void flag_orbits::init(action *A, action *A2, 
	INT nb_primary_orbits_lower, 
	INT pt_representation_sz, INT nb_flag_orbits, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag_orbits::init" << endl;
		}
	flag_orbits::A = A;
	flag_orbits::A2 = A2;
	flag_orbits::nb_primary_orbits_lower = nb_primary_orbits_lower;
	flag_orbits::pt_representation_sz = pt_representation_sz;
	flag_orbits::nb_flag_orbits = nb_flag_orbits;
	Pt = NEW_INT(nb_flag_orbits * pt_representation_sz);
	Flag_orbit_node = new flag_orbit_node[nb_flag_orbits];
	if (f_v) {
		cout << "flag_orbits::init done" << endl;
		}
}

void flag_orbits::write_file(ofstream &fp, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "flag_orbits::write_file" << endl;
		}
	fp.write((char *) &nb_primary_orbits_lower, sizeof(INT));
	fp.write((char *) &nb_primary_orbits_upper, sizeof(INT));
	fp.write((char *) &nb_flag_orbits, sizeof(INT));
	fp.write((char *) &pt_representation_sz, sizeof(INT));

	for (i = 0; i < nb_flag_orbits * pt_representation_sz; i++) {
		fp.write((char *) &Pt[i], sizeof(INT));
		}
	for (i = 0; i < nb_flag_orbits; i++) {
		Flag_orbit_node[i].write_file(fp, 0 /*verbose_level*/);
		}

	if (f_v) {
		cout << "flag_orbits::write_file finished" << endl;
		}
}

void flag_orbits::read_file(ifstream &fp, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "flag_orbits::read_file" << endl;
		}
	fp.read((char *) &nb_primary_orbits_lower, sizeof(INT));
	fp.read((char *) &nb_primary_orbits_upper, sizeof(INT));
	fp.read((char *) &nb_flag_orbits, sizeof(INT));
	fp.read((char *) &pt_representation_sz, sizeof(INT));

	Pt = NEW_INT(nb_flag_orbits * pt_representation_sz);
	for (i = 0; i < nb_flag_orbits * pt_representation_sz; i++) {
		fp.read((char *) &Pt[i], sizeof(INT));
		}
	Flag_orbit_node = new flag_orbit_node[nb_flag_orbits];
	for (i = 0; i < nb_flag_orbits; i++) {
		if (FALSE) {
			cout << "flag_orbits::read_file node " << i << " / " << nb_flag_orbits << endl;
			}
		Flag_orbit_node[i].Flag_orbits = this;
		Flag_orbit_node[i].flag_orbit_index = i;
		Flag_orbit_node[i].read_file(fp, 0 /*verbose_level */);
		}

	if (f_v) {
		cout << "flag_orbits::read_file finished" << endl;
		}
}


