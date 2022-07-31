// classification.h
//
// Anton Betten
//
// started:  September 20, 2007


#ifndef ORBITER_SRC_LIB_CLASSIFICATION_CLASSIFICATION_H_
#define ORBITER_SRC_LIB_CLASSIFICATION_CLASSIFICATION_H_



using namespace orbiter::layer1_foundations;
using namespace orbiter::layer2_discreta;
using namespace orbiter::layer3_group_actions;


namespace orbiter {

//! classification of combinatorial objects

namespace layer4_classification {


//! classification by using an invariant relation

namespace invariant_relations {

	// classify
	class classification_step;
	class flag_orbit_node;
	class flag_orbits;
	class orbit_node;

}

// isomorph
class isomorph_arguments;
class isomorph;
struct isomorph_worker_data;
class representatives;


// orbits
class orbit_of_equations;
class orbit_of_sets;
class orbit_of_subspaces;





//! To compute orbits on sets and subspaces using poset classification

namespace poset_classification {

	// poset_classification
	class classification_base_case;
	class extension;
	class orbit_based_testing;
	class orbit_tracer;
	class poset_classification;
	class poset_classification_control;
	class poset_classification_report_options;
	class poset_of_orbits;
	class poset_description;
	class poset_orbit_node;
	class poset_with_group_action;
	class upstep_work;

}

// set_stabilizer
class compute_stabilizer;
class stabilizer_orbits_and_types;
class substructure_classifier;
class substructure_stats_and_selection;

// solver:
class exact_cover_arguments;
class exact_cover;



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

#include "../layer4_classification/classify/classify.h"
#include "../layer4_classification/isomorph/isomorph.h"
#include "../layer4_classification/orbits/orbits.h"
#include "../layer4_classification/poset_classification/poset_classification.h"
#include "../layer4_classification/set_stabilizer/set_stabilizer.h"
#include "../layer4_classification/solver/solver.h"



#endif /* ORBITER_SRC_LIB_CLASSIFICATION_CLASSIFICATION_H_ */



