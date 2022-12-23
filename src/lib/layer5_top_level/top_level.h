// top_level.h
//
// Anton Betten
//
// started:  September 23 2010
//
// based on global.h, which was taken from reader.h: 3/22/09



#ifndef ORBITER_SRC_LIB_TOP_LEVEL_TOP_LEVEL_H_
#define ORBITER_SRC_LIB_TOP_LEVEL_TOP_LEVEL_H_



using namespace orbiter::layer1_foundations;
using namespace orbiter::layer2_discreta;
using namespace orbiter::layer3_group_actions;
using namespace orbiter::layer4_classification;


namespace orbiter {

//! classes for combinatorial objects and their classification

namespace layer5_applications {



//! Applications in algebra and number theory


namespace apps_algebra {

	// algebra_and_number_theory
	class action_on_forms_activity_description;
	class action_on_forms_activity;
	class action_on_forms_description;
	class action_on_forms;
	class algebra_global_with_action;
	class any_group;
	class character_table_burnside;
	class element_processing_description;
	class group_modification_description;
	class group_theoretic_activity_description;
	class group_theoretic_activity;
	class modified_group_create;
	class orbit_cascade;
	class orbits_activity_description;
	class orbits_activity;
	class orbits_create_description;
	class orbits_create;
	class orbits_on_polynomials;
	class orbits_on_subspaces;
	class polynomial_ring_activity;
	class vector_ge_builder;
	class young;

}

//! Applications in coding theory



namespace apps_coding_theory {

	class code_modification_description;
	class coding_theoretic_activity_description;
	class coding_theoretic_activity;
	class crc_process_description;
	class crc_process;
	class create_code_description;
	class create_code;
}


//! Classification problems in combinatorics, including design theory

namespace apps_combinatorics {

	// combinatorics
	class boolean_function_classify;
	class combinatorial_object_activity_description;
	class combinatorial_object_activity;
	class combinatorics_global;
	class delandtsheer_doyen_description;
	class delandtsheer_doyen;
	class design_activity_description;
	class design_activity;
	class design_create_description;
	class design_create;
	class design_tables;
	class difference_set_in_heisenberg_group;
	class flag_orbits_incidence_structure;
	class hadamard_classify;
	class hall_system_classify;
	class large_set_activity_description;
	class large_set_activity;
	class large_set_classify;
	class large_set_was_activity_description;
	class large_set_was_activity;
	class large_set_was_description;
	class large_set_was;
	class object_with_properties;
	class regular_linear_space_description;
	class regular_ls_classify;
	class tactical_decomposition;

}

//! Geometry

namespace apps_geometry {

	// geometry
	class arc_generator_description;
	class arc_generator;
	class arc_lifting_simeon;
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
	class top_level_geometry_global;

}

//! Graph theory

namespace apps_graph_theory {

	// graph_theory.h:
	class cayley_graph_search;
	class create_graph_description;
	class create_graph;
	class graph_classification_activity_description;
	class graph_classification_activity;
	class graph_classify_description;
	class graph_classify;
	class graph_modification_description;
	class graph_theoretic_activity_description;
	class graph_theoretic_activity;

}

//! Orbiter command line, orbiter dash code, symbol definitions and activities

namespace user_interface {

	// interfaces
	class activity_description;
	class interface_algebra;
	class interface_coding_theory;
	class interface_combinatorics;
	class interface_cryptography;
	class interface_povray;
	class interface_projective;
	class interface_symbol_table;
	class interface_toolkit;
	class orbiter_command;
	class orbiter_top_level_session;
	class symbol_definition;

}


//! Applications in orthogonal geometry

namespace orthogonal_geometry_applications {


	// orthogonal
	class blt_set_classify_activity_description;
	class blt_set_classify_description;
	class blt_set_classify;
	class BLT_set_create_description;
	class BLT_set_create;
	class blt_set_with_action;
	class orthogonal_space_activity_description;
	class orthogonal_space_activity;
	class orthogonal_space_with_action_description;
	class orthogonal_space_with_action;

}

//! packings in projective space

namespace packings {

	// packings:
	class invariants_packing;
	class packing_classify;
	class packing_invariants;
	class packing_long_orbits_description;
	class packing_long_orbits;
	class packing_was_activity_description;
	class packing_was_activity;
	class packing_was_description;
	class packing_was_fixpoints_activity_description;
	class packing_was_fixpoints_activity;
	class packing_was_fixpoints;
	class packing_was;
	class packings_global;
	class regular_packing;

}


//! Applications in projective space

namespace projective_geometry {

	// projective_space.h:
	class canonical_form_classifier_description;
	class canonical_form_classifier;
	class canonical_form_nauty;
	class canonical_form_substructure;
	class object_in_projective_space_with_action;
	class projective_space_activity_description;
	class projective_space_activity;
	class projective_space_globals;
	class projective_space_with_action_description;
	class projective_space_with_action;
	class quartic_curve_object;

}

//! Classification of finite semifields


namespace semifields {


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

}


//! Spreads in projective space

namespace spreads {

	// spreads:
	class recoordinatize;
	class spread_activity_description;
	class spread_activity;
	class spread_classify_activity_description;
	class spread_classify_activity;
	class spread_classify_description;
	class spread_classify;
	class spread_create_description;
	class spread_create;
	class spread_lifting;
	class spread_table_activity_description;
	class spread_table_activity;
	class spread_table_with_selection;
	class translation_plane_activity_description;
	class translation_plane_activity;

}


//! cubic surfaces, quartic curves, etc.

namespace applications_in_algebraic_geometry {


	//! classes related to plane quartic curves with 28 bitangents

	namespace quartic_curves {

		// surfaces/quartic curves
		class quartic_curve_activity_description;
		class quartic_curve_activity;
		class quartic_curve_create_description;
		class quartic_curve_create;
		class quartic_curve_domain_with_action;
		class quartic_curve_from_surface;
		class quartic_curve_object_with_action;

	}

	//! cubic surfaces and related six-arcs and trihedral pairs

	namespace cubic_surfaces_and_arcs {

		// surfaces/surfaces_and_arcs:
		class arc_lifting;
		class arc_orbits_on_pairs;
		class arc_partition;
		class classify_trihedral_pairs;
		class six_arcs_not_on_a_conic;
		class surface_classify_using_arc;
		class surface_create_by_arc_lifting;
		class surfaces_arc_lifting_definition_node;
		class surfaces_arc_lifting_trace;
		class surfaces_arc_lifting_upstep;
		class surfaces_arc_lifting;
		class trihedral_pair_with_action;
		// pointer types:
		typedef class surfaces_arc_lifting_trace psurfaces_arc_lifting_trace;

	}

	//! cubic surfaces and related Schlaefli double-sixes

	namespace cubic_surfaces_and_double_sixes {

		// surfaces/surfaces_and_double_sixes:
		class classification_of_cubic_surfaces_with_double_sixes_activity_description;
		class classification_of_cubic_surfaces_with_double_sixes_activity;
		class classify_double_sixes;
		class surface_classify_wedge;

	}

	//! cubic surfaces in general

	namespace cubic_surfaces_in_general {

		// surfaces/surfaces_general:
		class cubic_surface_activity_description;
		class cubic_surface_activity;
		class surface_clebsch_map;
		class surface_create_description;
		class surface_create;
		class surface_domain_high_level;
		class surface_object_with_action;
		class surface_study;
		class surface_with_action;

	}

}

#define Get_object_of_type_any_group(label) user_interface::The_Orbiter_top_level_session->get_object_of_type_any_group(label)
#define Get_vector_or_set(label, set, sz) user_interface::The_Orbiter_top_level_session->get_vector_or_set(label, set, sz, 0)
#define Get_object_of_type_finite_field(label) user_interface::The_Orbiter_top_level_session->get_object_of_type_finite_field(label)
#define Get_object_of_type_any_group(label) user_interface::The_Orbiter_top_level_session->get_object_of_type_any_group(label)
#define Get_object_of_type_spread(label) user_interface::The_Orbiter_top_level_session->get_object_of_type_spread(label)
//#define Get_object_of_type_ring(label) user_interface::The_Orbiter_top_level_session->get_object_of_type_ring(label)
#define Get_object_of_type_poset_classification_control(label) user_interface::The_Orbiter_top_level_session->get_object_of_type_poset_classification_control(label)
#define Get_object_of_type_vector_ge(label) user_interface::The_Orbiter_top_level_session->get_object_of_type_vector_ge(label)
#define Get_object_of_projective_space(label) user_interface::The_Orbiter_top_level_session->get_object_of_type_projective_space(label)
#define Get_object_of_cubic_surface(label) user_interface::The_Orbiter_top_level_session->get_object_of_type_cubic_surface(label)
#define Get_object_of_type_code(label) user_interface::The_Orbiter_top_level_session->get_object_of_type_code(label)



}}


#include "./algebra_and_number_theory/tl_algebra_and_number_theory.h"
#include "./coding_theory_apps/coding_theory_apps.h"
#include "./combinatorics/tl_combinatorics.h"
#include "./geometry/tl_geometry.h"
#include "./graph_theory/graph_theory.h"
#include "./interfaces/interfaces.h"
#include "./orthogonal/tl_orthogonal.h"
#include "./projective_space/projective_space.h"
#include "./semifields/semifields.h"
#include "./spreads/spreads.h"
#include "./packings/packings.h"
#include "./surfaces/quartic_curves/quartic_curves.h"
#include "./surfaces/surfaces_and_arcs/surfaces_and_arcs.h"
#include "./surfaces/surfaces_and_double_sixes/surfaces_and_double_sixes.h"
#include "./surfaces/surfaces_general/surfaces_general.h"



#endif /* ORBITER_SRC_LIB_TOP_LEVEL_TOP_LEVEL_H_ */




