/*
 * classify.h
 *
 *  Created on: Aug 9, 2018
 *      Author: Anton Betten
 *
 * started:  September 20, 2007
 * pulled out of snakesandladders.h: Aug 9, 2018
 */



// #############################################################################
// classification.C:
// #############################################################################

//! a poset classification data structure


class classification {
public:
	action *A; // do not free
	action *A2; // do not free

	longinteger_object go;
	int max_orbits;
	int nb_orbits;
	orbit_node *Orbit; // [max_orbits]
	int representation_sz;
	int *Rep; // [nb_orbits * representation_sz]

	classification();
	~classification();
	void null();
	void freeself();
	void init(action *A, action *A2, int max_orbits, int representation_sz,
			longinteger_object &go, int verbose_level);
	set_and_stabilizer *get_set_and_stabilizer(int orbit_index,
			int verbose_level);
	void print_latex(ostream &ost, const char *title, int f_with_stabilizers);
	void write_file(ofstream &fp, int verbose_level);
	void read_file(ifstream &fp, int verbose_level);

};

// #############################################################################
// flag_orbits.C:
// #############################################################################

//! related to the class classification


class flag_orbits {
public:
	action *A; // do not free
	action *A2; // do not free

	int nb_primary_orbits_lower;
	int nb_primary_orbits_upper;

	int nb_flag_orbits;
	flag_orbit_node *Flag_orbit_node;
	int pt_representation_sz;
	int *Pt; // [nb_flag_orbits * pt_representation_sz]

	flag_orbits();
	~flag_orbits();
	void null();
	void freeself();
	void init(action *A, action *A2, int nb_primary_orbits_lower,
			int pt_representation_sz, int nb_flag_orbits, int verbose_level);
	void write_file(ofstream &fp, int verbose_level);
	void read_file(ifstream &fp, int verbose_level);

};

// #############################################################################
// flag_orbit_node.C:
// #############################################################################

//! related to the class flag_orbits


class flag_orbit_node {
public:
	flag_orbits *Flag_orbits;

	int flag_orbit_index;

	int downstep_primary_orbit;
	int downstep_secondary_orbit;
	int downstep_orbit_len;
	int f_long_orbit;
	int upstep_primary_orbit;
	int upstep_secondary_orbit;
	int f_fusion_node;
	int fusion_with;
	int *fusion_elt;

	longinteger_object go;
	strong_generators *gens;

	flag_orbit_node();
	~flag_orbit_node();
	void null();
	void freeself();
	void init(flag_orbits *Flag_orbits, int flag_orbit_index,
			int downstep_primary_orbit, int downstep_secondary_orbit,
			int downstep_orbit_len, int f_long_orbit, int *pt_representation,
			strong_generators *Strong_gens, int verbose_level);
	void write_file(ofstream &fp, int verbose_level);
	void read_file(ifstream &fp, int verbose_level);

};

// #############################################################################
// orbit_node.C:
// #############################################################################

//! related to the class classification

class orbit_node {
public:
	classification *C;
	int orbit_index;
	strong_generators *gens;

	orbit_node();
	~orbit_node();
	void null();
	void freeself();
	void init(classification *C, int orbit_index, strong_generators *gens,
			int *Rep, int verbose_level);
	void write_file(ofstream &fp, int verbose_level);
	void read_file(ifstream &fp, int verbose_level);
};



