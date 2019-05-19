/*
 * classify.h
 *
 *  Created on: Aug 9, 2018
 *      Author: Anton Betten
 *
 * started:  September 20, 2007
 * pulled out of snakesandladders.h: Aug 9, 2018
 */

namespace orbiter {
namespace classification {



// #############################################################################
// classification_step.C:
// #############################################################################

//! a single step classification of combinatorial objects


class classification_step {
public:
	action *A; // do not free
	action *A2; // do not free

	int f_lint;
	longinteger_object go;
	int max_orbits;
	int nb_orbits;
	orbit_node *Orbit; // [max_orbits]
	int representation_sz;
	int *Rep; // [nb_orbits * representation_sz]
	long int *Rep_lint; // [nb_orbits * representation_sz]

	classification_step();
	~classification_step();
	void null();
	void freeself();
	void init(action *A, action *A2, int max_orbits, int representation_sz,
			longinteger_object &go, int verbose_level);
	void init_lint(action *A, action *A2, int max_orbits, int representation_sz,
			longinteger_object &go, int verbose_level);
	set_and_stabilizer *get_set_and_stabilizer(int orbit_index,
			int verbose_level);
	void print_latex(std::ostream &ost,
		const char *title, int f_print_stabilizer_gens,
		int f_has_print_function,
		void (*print_function)(std::ostream &ost, int i,
				classification_step *Step, void *print_function_data),
		void *print_function_data);
	void write_file(std::ofstream &fp, int verbose_level);
	void read_file(std::ifstream &fp,
			action *A, action *A2, longinteger_object &go,
			int verbose_level);
	void generate_source_code(const char *fname_base, int verbose_level);
	int *Rep_ith(int i);
	long int *Rep_lint_ith(int i);

};

// #############################################################################
// flag_orbits.C:
// #############################################################################

//! stores the set of flag orbits; related to the class classification_step


class flag_orbits {
public:
	action *A; // do not free
	action *A2; // do not free

	int nb_primary_orbits_lower;
	int nb_primary_orbits_upper;

	int f_lint;
	int nb_flag_orbits;
	flag_orbit_node *Flag_orbit_node;
	int pt_representation_sz;
	int *Pt; // [nb_flag_orbits * pt_representation_sz]
	long int *Pt_lint; // [nb_flag_orbits * pt_representation_sz]

	flag_orbits();
	~flag_orbits();
	void null();
	void freeself();
	void init(action *A, action *A2, int nb_primary_orbits_lower,
			int pt_representation_sz, int nb_flag_orbits, int verbose_level);
	void init_lint(action *A, action *A2,
		int nb_primary_orbits_lower,
		int pt_representation_sz, int nb_flag_orbits,
		int verbose_level);
	int find_node_by_po_so(int po, int so, int &idx,
		int verbose_level);
	void write_file(std::ofstream &fp, int verbose_level);
	void read_file(std::ifstream &fp,
			action *A, action *A2,
			int verbose_level);
	void print_latex(std::ostream &ost,
		const char *title, int f_print_stabilizer_gens);

};

// #############################################################################
// flag_orbit_node.C:
// #############################################################################

//! to represent a flag orbit; related to the class flag_orbits


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
	void init_lint(
		flag_orbits *Flag_orbits, int flag_orbit_index,
		int downstep_primary_orbit, int downstep_secondary_orbit,
		int downstep_orbit_len, int f_long_orbit,
		long int *pt_representation, strong_generators *Strong_gens,
		int verbose_level);
	void write_file(std::ofstream &fp, int verbose_level);
	void read_file(std::ifstream &fp, int verbose_level);
	void print_latex(flag_orbits *Flag_orbits,
			std::ostream &ost,
			int f_print_stabilizer_gens);

};

// #############################################################################
// orbit_node.C:
// #############################################################################

//! to encode one group orbit, associated to the class classification_step

class orbit_node {
public:
	classification_step *C;
	int orbit_index;
	strong_generators *gens;

	orbit_node();
	~orbit_node();
	void null();
	void freeself();
	void init(classification_step *C,
			int orbit_index, strong_generators *gens,
			int *Rep, int verbose_level);
	void init_lint(classification_step *C, int orbit_index,
		strong_generators *gens, long int *Rep, int verbose_level);
	void write_file(std::ofstream &fp, int verbose_level);
	void read_file(std::ifstream &fp, int verbose_level);
};

}}


