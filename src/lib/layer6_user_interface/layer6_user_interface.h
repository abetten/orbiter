/*
 * layer6_user_interface.h
 *
 *  Created on: Mar 27, 2026
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER6_USER_INTERFACE_LAYER6_USER_INTERFACE_H_
#define SRC_LIB_LAYER6_USER_INTERFACE_LAYER6_USER_INTERFACE_H_



using namespace orbiter::layer1_foundations;
using namespace orbiter::layer2_discreta;
using namespace orbiter::layer3_group_actions;
using namespace orbiter::layer4_classification;
//using namespace orbiter::layer5_applications;


namespace orbiter {

	//! user interface on top of the orbiter library

	namespace layer6_user_interface {



		//! starting point for global commands

		namespace activities {

			class interface_algebra;
			class interface_coding_theory;
			class interface_combinatorics;
			class interface_cryptography;
			class interface_povray;
			class interface_projective;
			class interface_symbol_table;
			class interface_toolkit;

		}


		//! starting point for commands related to objects form layer 1

		namespace activities_layer1 {

			class diophant_activity_description;
			class diophant_activity;
			class finite_field_activity_description;
			class finite_field_activity;
			class polynomial_ring_activity_description;
			class symbolic_object_activity_description;
			class symbolic_object_activity;

		}

		//! starting point for commands related to objects form layer 5

		namespace activities_layer5 {

			class action_on_forms_activity_description;
			class action_on_forms_activity;
			class blt_set_activity_description;
			class blt_set_activity;
			class blt_set_classify_activity_description;
			class blt_set_classify_description;
			class classification_of_cubic_surfaces_with_double_sixes_activity_description;
			class classification_of_cubic_surfaces_with_double_sixes_activity;
			class coding_theoretic_activity_description;
			class coding_theoretic_activity;
			class combinatorial_object_activity_description;
			class combinatorial_object_activity;
			class combo_activity_description;
			class combo_activity;
			class cubic_surface_activity_description;
			class cubic_surface_activity;
			class design_activity_description;
			class design_activity;
			class graph_classification_activity_description;
			class graph_classification_activity;
			class graph_theoretic_activity_description;
			class graph_theoretic_activity;
			class group_theoretic_activity_description;
			class group_theoretic_activity;
			class large_set_activity_description;
			class large_set_activity;
			class large_set_was_activity_description;
			class large_set_was_activity;
			class orbits_activity_description;
			class orbits_activity;
			class orthogonal_space_activity_description;
			class orthogonal_space_activity;
			class packing_classify_activity_description;
			class packing_classify_activity;
			class packing_was_activity_description;
			class packing_was_activity;
			class packing_was_fixpoints_activity_description;
			class packing_was_fixpoints_activity;
			class plesken_ring_activity_description;
			class plesken_ring_activity;
			class polynomial_ring_activity;
			class projective_space_activity_description;
			class projective_space_activity;
			class quartic_curve_activity_description;
			class quartic_curve_activity;
			class spread_activity_description;
			class spread_activity;
			class spread_classify_activity_description;
			class spread_classify_activity;
			class spread_table_activity_description;
			class spread_table_activity;
			class translation_plane_activity_description;
			class translation_plane_activity;
			class variety_activity_description;
			class variety_activity;
			class vector_ge_activity_description;
			class vector_ge_activity;

		}

		//! the command line parser, symbol table and user interface

		namespace control_everything {

			class activity_description;
			class orbiter_command;

		}

	}



}


#include "layer6_user_interface/activities/activities.h"
#include "layer6_user_interface/activities_layer1/activities_layer1.h"
#include "layer6_user_interface/activities_layer5/activities_layer5.h"
#include "layer6_user_interface/control_everything/control_everything.h"





#endif /* SRC_LIB_LAYER6_USER_INTERFACE_LAYER6_USER_INTERFACE_H_ */
