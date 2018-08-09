/*
 * classify.h
 *
 *  Created on: Aug 9, 2018
 *      Author: Anton Betten
 *
 * started:  September 20, 2007
 * pulled out of snakesandladders.h: Aug 9, 2018
 */

#ifndef ORBITER_SRC_LIB_SNAKES_AND_LADDERS_CLASSIFY_CLASSIFY_H_
#define ORBITER_SRC_LIB_SNAKES_AND_LADDERS_CLASSIFY_CLASSIFY_H_



// #############################################################################
// classification.C:
// #############################################################################

class classification {
public:
	action *A; // do not free
	action *A2; // do not free

	longinteger_object go;
	INT max_orbits;
	INT nb_orbits;
	orbit_node *Orbit; // [max_orbits]
	INT representation_sz;
	INT *Rep; // [nb_orbits * representation_sz]

	classification();
	~classification();
	void null();
	void freeself();
	void init(action *A, action *A2, INT max_orbits, INT representation_sz,
			longinteger_object &go, INT verbose_level);
	set_and_stabilizer *get_set_and_stabilizer(INT orbit_index,
			INT verbose_level);
	void print_latex(ostream &ost, const BYTE *title, INT f_with_stabilizers);
	void write_file(ofstream &fp, INT verbose_level);
	void read_file(ifstream &fp, INT verbose_level);

};

// #############################################################################
// flag_orbits.C:
// #############################################################################

class flag_orbits {
public:
	action *A; // do not free
	action *A2; // do not free

	INT nb_primary_orbits_lower;
	INT nb_primary_orbits_upper;

	INT nb_flag_orbits;
	flag_orbit_node *Flag_orbit_node;
	INT pt_representation_sz;
	INT *Pt; // [nb_flag_orbits * pt_representation_sz]

	flag_orbits();
	~flag_orbits();
	void null();
	void freeself();
	void init(action *A, action *A2, INT nb_primary_orbits_lower,
			INT pt_representation_sz, INT nb_flag_orbits, INT verbose_level);
	void write_file(ofstream &fp, INT verbose_level);
	void read_file(ifstream &fp, INT verbose_level);

};

// #############################################################################
// flag_orbit_node.C:
// #############################################################################

class flag_orbit_node {
public:
	flag_orbits *Flag_orbits;

	INT flag_orbit_index;

	INT downstep_primary_orbit;
	INT downstep_secondary_orbit;
	INT downstep_orbit_len;
	INT f_long_orbit;
	INT upstep_primary_orbit;
	INT upstep_secondary_orbit;
	INT f_fusion_node;
	INT fusion_with;
	INT *fusion_elt;

	longinteger_object go;
	strong_generators *gens;

	flag_orbit_node();
	~flag_orbit_node();
	void null();
	void freeself();
	void init(flag_orbits *Flag_orbits, INT flag_orbit_index,
			INT downstep_primary_orbit, INT downstep_secondary_orbit,
			INT downstep_orbit_len, INT f_long_orbit, INT *pt_representation,
			strong_generators *Strong_gens, INT verbose_level);
	void write_file(ofstream &fp, INT verbose_level);
	void read_file(ifstream &fp, INT verbose_level);

};

// #############################################################################
// orbit_node.C:
// #############################################################################

class orbit_node {
public:
	classification *C;
	INT orbit_index;
	strong_generators *gens;

	orbit_node();
	~orbit_node();
	void null();
	void freeself();
	void init(classification *C, INT orbit_index, strong_generators *gens,
			INT *Rep, INT verbose_level);
	void write_file(ofstream &fp, INT verbose_level);
	void read_file(ifstream &fp, INT verbose_level);
};



#endif /* ORBITER_SRC_LIB_SNAKES_AND_LADDERS_CLASSIFY_CLASSIFY_H_ */
