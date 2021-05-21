// top_level.h
//
// Anton Betten
//
// started:  September 23 2010
//
// based on global.h, which was taken from reader.h: 3/22/09



#ifndef ORBITER_SRC_LIB_TOP_LEVEL_TOP_LEVEL_H_
#define ORBITER_SRC_LIB_TOP_LEVEL_TOP_LEVEL_H_



using namespace orbiter::foundations;
using namespace orbiter::group_actions;
using namespace orbiter::classification;
using namespace orbiter::discreta;


namespace orbiter {

//! classes for combinatorial objects and their classification

namespace top_level {






// algbra_and_number_theory
class algebra_global_with_action;
class character_table_burnside;
class group_theoretic_activity_description;
class group_theoretic_activity;
class orbits_on_polynomials;
class orthogonal_space_activity_description;
class orthogonal_space_activity;
class young;


// combinatorics
class boolean_function_classify;
class delandtsheer_doyen_description;
class delandtsheer_doyen;
class design_create_description;
class design_create;
class difference_set_in_heisenberg_group;
class hadamard_classify;
class hall_system_classify;
class large_set_classify;
class regular_linear_space_description;
class regular_ls_classify;
class tactical_decomposition;


// geometry
class arc_generator_description;
class arc_generator;
class arc_lifting_simeon;
class blt_set_classify;
class BLT_set_create_description;
class BLT_set_create;
class blt_set_with_action;
class choose_points_or_lines;
class classify_cubic_curves;
class cubic_curve_with_action;
class hermitian_spread_classify;
class linear_set_classify;
class ovoid_classify_description;
class ovoid_classify;
class polar;
class search_blocking_set;
class singer_cycle;
class tensor_classify;



// graph_theory.h:
class cayley_graph_search;
class create_graph_description;
class create_graph;
class graph_classification_activity_description;
class graph_classification_activity;
class graph_classify_description;
class graph_classify;
class graph_theoretic_activity_description;
class graph_theoretic_activity;


// interfaces
class interface_algebra;
class interface_coding_theory;
class interface_combinatorics;
class interface_cryptography;
class interface_povray;
class interface_projective;
class interface_symbol_table;
class interface_toolkit;
class orbiter_top_level_session;


// isomorph
class isomorph_arguments;
class isomorph;
struct isomorph_worker_data;
class representatives;


// orbits
class orbit_of_equations;
class orbit_of_sets;
class orbit_of_subspaces;



// projective_space.h:
class canonical_form_classifier_description;
class canonical_form_classifier;
class canonical_form_nauty;
class canonical_form_substructure;
class compute_stabilizer;
class object_in_projective_space_with_action;
class projective_space_activity_description;
class projective_space_activity;
class projective_space_job_description;
class projective_space_job;
class projective_space_object_classifier_description;
class projective_space_object_classifier;
class projective_space_with_action_description;
class projective_space_with_action;
class quartic_curve_activity_description;
class quartic_curve_activity;
class quartic_curve_create_description;
class quartic_curve_create;
class quartic_curve_domain_with_action;


// semifield
class semifield_classify_description;
class semifield_classify_with_substructure;
class semifield_classify;
class semifield_downstep_node;
class semifield_flag_orbit_node;
class semifield_level_two;
class semifield_lifting;
class semifield_substructure;
class semifield_trace;
class trace_record;


// solver:
class exact_cover_arguments;
class exact_cover;


// spreads and packings:
class spread_table_activity_description;
class spread_table_activity;
class spread_table_with_selection;
class packing_was_fixpoints;
class packing_long_orbits_description;
class regular_packing;
class packing_was_description;
class packing_classify;
class packing_invariants;
class invariants_packing;
class spread_create_description;
class spread_create;
class spread_lifting;
class packing_was;
class packing_long_orbits;
class recoordinatize;
class spread_classify;
class packing_was_activity_description;
class packing_was_activity;
class packing_was_fixpoints_activity_description;
class packing_was_fixpoints_activity;

// surfaces:
class arc_orbits_on_pairs;
class arc_partition;
class classification_of_cubic_surfaces_with_double_sixes_activity_description;
class classification_of_cubic_surfaces_with_double_sixes_activity;
class surface_with_action;
class surface_object_with_action;
class classify_trihedral_pairs;
class classify_double_sixes;
class surface_create_description;
class surface_create;
class arc_lifting;
class six_arcs_not_on_a_conic;
class surface_study;
class quartic_curve;
class surface_classify_using_arc;
class surface_create_by_arc_lifting;
class surface_clebsch_map;
class trihedral_pair_with_action;
class surfaces_arc_lifting_upstep;
class surfaces_arc_lifting;
class surfaces_arc_lifting_trace;
class surfaces_arc_lifting_definition_node;
class cubic_surface_activity_description;
class cubic_surface_activity;
class surface_domain_high_level;
class surface_classify_wedge;



// pointer types:
typedef class surfaces_arc_lifting_trace psurfaces_arc_lifting_trace;


// #############################################################################
// global variable:
// #############################################################################



extern orbiter_top_level_session *The_Orbiter_top_level_session; // global top level Orbiter session



// #############################################################################
// representatives.cpp
// #############################################################################

//! auxiliary class for class isomorph



class representatives {
public:
	action *A;

	std::string prefix;
	std::string fname_rep;
	std::string fname_stabgens;
	std::string fname_fusion;
	std::string fname_fusion_ge;



	// flag orbits:
	int nb_objects;
	int *fusion; // [nb_objects]
		// fusion[i] == -2 means that the flag orbit i
		// has not yet been processed by the
		// isomorphism testing procedure.
		// fusion[i] = i means that flag orbit [i]
		// in an orbit representative
		// Otherwise, fusion[i] is an earlier flag_orbit,
		// and handle[i] is a group element that maps
		// to it
	int *handle; // [nb_objects]
		// handle[i] is only relevant if fusion[i] != i,
		// i.e., if flag orbit i is not a representative
		// of an isomorphism type.
		// In this case, handle[i] is the (handle of a)
		// group element moving flag orbit i to flag orbit fusion[i].


	// classified objects:
	int count;
	int *rep; // [count]
	sims **stab; // [count]



	//char *elt;
	int *Elt1;
	int *tl; // [A->base_len]

	int nb_open;
	int nb_reps;
	int nb_fused;


	representatives();
	void null();
	~representatives();
	void free();
	void init(action *A, int nb_objects, std::string &prefix, int verbose_level);
	void write_fusion(int verbose_level);
	void read_fusion(int verbose_level);
	void write_representatives_and_stabilizers(int verbose_level);
	void read_representatives_and_stabilizers(int verbose_level);
	void save(int verbose_level);
	void load(int verbose_level);
	void calc_fusion_statistics();
	void print_fusion_statistics();
};

}}


#include "./algebra_and_number_theory/tl_algebra_and_number_theory.h"
#include "./combinatorics/tl_combinatorics.h"
#include "./geometry/tl_geometry.h"
#include "./graph_theory/graph_theory.h"
#include "./interfaces/interfaces.h"
#include "./isomorph/isomorph.h"
#include "./orbits/orbits.h"
#include "./projective_space/projective_space.h"
#include "./semifields/semifields.h"
#include "./solver/solver.h"
#include "./spreads_and_packings/spreads_and_packings.h"
#include "./surfaces/surfaces.h"



#endif /* ORBITER_SRC_LIB_TOP_LEVEL_TOP_LEVEL_H_ */




