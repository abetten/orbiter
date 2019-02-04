// classification.h
//
// Anton Betten
//
// started:  September 20, 2007

using namespace orbiter::foundations;
using namespace orbiter::group_actions;


namespace orbiter {

//! poset classification for combinatorial objects

namespace classification {


class extension;
class orbit_based_testing;
class poset_classification;
class poset_orbit_node;
class set_stabilizer_compute;
class upstep_work;
class compute_stabilizer;
class flag_orbits;
class flag_orbit_node;
class classification_step;
class orbit_node;
class poset;
class poset_description;


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
#include "./other/other.h"
#include "./poset_classification/poset_classification.h"
#include "./set_stabilizer/set_stabilizer.h"

