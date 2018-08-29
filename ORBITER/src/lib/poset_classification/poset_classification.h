// poset_classification.h
//
// Anton Betten
//
// started:  September 20, 2007


typedef class poset_classification poset_classification;
typedef class poset_orbit_node poset_orbit_node;
typedef class extension extension;
typedef class set_stabilizer_compute set_stabilizer_compute;
typedef class upstep_work upstep_work;
typedef class compute_stabilizer compute_stabilizer;
typedef class flag_orbits flag_orbits;
typedef class flag_orbit_node flag_orbit_node;
typedef class classification classification;
typedef class orbit_node orbit_node;

enum trace_result { 
	found_automorphism, 
	not_canonical, 
	no_result_extension_not_found, 
	no_result_fusion_node_installed, 
	no_result_fusion_node_already_installed
};

enum find_isomorphism_result { 
	fi_found_isomorphism, 
	fi_not_isomorphic, 
	fi_no_result 
};

#include "./classify/classify.h"
#include "./other/other.h"
#include "./snakes_and_ladders/snakes_and_ladders.h"
#include "./set_stabilizer/set_stabilizer.h"

