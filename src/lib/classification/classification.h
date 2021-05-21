// classification.h
//
// Anton Betten
//
// started:  September 20, 2007


#ifndef ORBITER_SRC_LIB_CLASSIFICATION_CLASSIFICATION_H_
#define ORBITER_SRC_LIB_CLASSIFICATION_CLASSIFICATION_H_



using namespace orbiter::foundations;
using namespace orbiter::group_actions;


namespace orbiter {

//! classification of combinatorial objects

namespace classification {

// classify
class classification_step;
class flag_orbit_node;
class flag_orbits;
class orbit_node;

// poset_classification
class classification_base_case;
class extension;
class orbit_based_testing;
class poset_classification;
class poset_classification_control;
class poset_description;
class poset_orbit_node;
class poset_with_group_action;
class upstep_work;


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

}}

#include "./classify/classify.h"
#include "./poset_classification/poset_classification.h"



#endif /* ORBITER_SRC_LIB_CLASSIFICATION_CLASSIFICATION_H_ */



