/*
 * classify.h
 *
 *  Created on: Aug 9, 2018
 *      Author: Anton Betten
 *
 * started:  September 20, 2007
 * pulled out of snakesandladders.h: Aug 9, 2018
 */


#ifndef ORBITER_SRC_LIB_CLASSIFICATION_CLASSIFY_CLASSIFY_H_
#define ORBITER_SRC_LIB_CLASSIFICATION_CLASSIFY_CLASSIFY_H_


namespace orbiter {
namespace layer4_classification {
namespace invariant_relations {



// #############################################################################
// classification_step.cpp
// #############################################################################

//! a single step classification of combinatorial objects


class classification_step {
public:
	actions::action *A; // do not free
	actions::action *A2; // do not free

	ring_theory::longinteger_object go;
	int max_orbits;
	int nb_orbits;
	orbit_node *Orbit; // [max_orbits]
	int representation_sz;
	long int *Rep; // [nb_orbits * representation_sz]

	classification_step();
	~classification_step();
	void init(
			actions::action *A,
			actions::action *A2,
			int max_orbits, int representation_sz,
			ring_theory::longinteger_object &go,
			int verbose_level);
	data_structures_groups::set_and_stabilizer
		*get_set_and_stabilizer(
			int orbit_index,
			int verbose_level);
	void write_file(
			std::ofstream &fp, int verbose_level);
	void read_file(
			std::ifstream &fp,
			actions::action *A,
			actions::action *A2,
			ring_theory::longinteger_object &go,
			int verbose_level);
	void generate_source_code(
			std::string &fname_base, int verbose_level);
	void generate_source_code(
			std::ostream &ost, std::string &prefix, int verbose_level);
	long int *Rep_ith(int i);
	void print_group_orders();
	void print_summary(std::ostream &ost);
	void print_latex(std::ostream &ost,
			std::string &title,
			int f_print_stabilizer_gens,
		int f_has_print_function,
		void (*print_function)(std::ostream &ost, int i,
				classification_step *Step, void *print_function_data),
		void *print_function_data);

};

// #############################################################################
// flag_orbits.cpp
// #############################################################################

//! stores the set of flag orbits; related to the class classification_step


class flag_orbits {
public:
	actions::action *A; // do not free
	actions::action *A2; // do not free

	int nb_primary_orbits_lower;
	int nb_primary_orbits_upper;

	int upper_bound_for_number_of_traces;
	void (*func_to_free_received_trace)(
			void *trace_result, void *data, int verbose_level);
	void (*func_latex_report_trace)(
			std::ostream &ost, void *trace_result,
			void *data, int verbose_level);
	void *free_received_trace_data;

	int nb_flag_orbits;
	flag_orbit_node *Flag_orbit_node;
	int pt_representation_sz;
	long int *Pt; // [nb_flag_orbits * pt_representation_sz]


	flag_orbits();
	~flag_orbits();
	void init(actions::action *A,
			actions::action *A2,
			int nb_primary_orbits_lower,
			int pt_representation_sz, int nb_flag_orbits,
			int upper_bound_for_number_of_traces,
			void (*func_to_free_received_trace)(
					void *trace_result, void *data, int verbose_level),
			void (*func_latex_report_trace)(
					std::ostream &ost, void *trace_result,
					void *data, int verbose_level),
			void *free_received_trace_data,
			int verbose_level);
	int find_node_by_po_so(int po, int so, int &idx,
		int verbose_level);
	void write_file(std::ofstream &fp, int verbose_level);
	void read_file(std::ifstream &fp,
			actions::action *A, actions::action *A2,
			int verbose_level);
	void print_latex(std::ostream &ost,
			std::string &title, int f_print_stabilizer_gens);

};

// #############################################################################
// flag_orbit_node.cpp
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

	ring_theory::longinteger_object go;
	groups::strong_generators *gens;

	int nb_received;
	void **Receptacle; // [upper_bound_for_number_of_traces]

	flag_orbit_node();
	~flag_orbit_node();
	void init(flag_orbits *Flag_orbits,
			int flag_orbit_index,
			int downstep_primary_orbit, int downstep_secondary_orbit,
			int downstep_orbit_len,
			int f_long_orbit,
			long int *pt_representation,
			groups::strong_generators *Strong_gens,
			int verbose_level);
	void receive_trace_result(
			void *trace_result, int verbose_level);
	void write_file(std::ofstream &fp, int verbose_level);
	void read_file(std::ifstream &fp, int verbose_level);
	void print_latex(flag_orbits *Flag_orbits,
			std::ostream &ost,
			int f_print_stabilizer_gens);

};

// #############################################################################
// orbit_node.cpp
// #############################################################################

//! to encode one group orbit, associated to the class classification_step

class orbit_node {
public:
	classification_step *C;
	int orbit_index;
	groups::strong_generators *gens;
	void *extra_data;

	orbit_node();
	~orbit_node();
	void init(classification_step *C,
			int orbit_index,
			groups::strong_generators *gens,
			long int *Rep, void *extra_data,
			int verbose_level);
	void write_file(std::ofstream &fp, int verbose_level);
	void read_file(std::ifstream &fp, int verbose_level);
};

}}}


#endif /* ORBITER_SRC_LIB_CLASSIFICATION_CLASSIFY_CLASSIFY_H_ */




