/*
 * symbol_definition.cpp
 *
 *  Created on: Jun 20, 2021
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace user_interface {


symbol_definition::symbol_definition()
{
	Record_birth();
	Sym = NULL;

	//std::string define_label;

	f_finite_field = false;
	Finite_field_description = NULL;

	f_polynomial_ring = false;
	Polynomial_ring_description = NULL;

	f_projective_space = false;
	Projective_space_with_action_description = NULL;

	f_orthogonal_space = false;
	Orthogonal_space_with_action_description = NULL;

	f_BLT_set_classifier = false;
	//std::string BLT_set_classifier_label_orthogonal_geometry;
	Blt_set_classify_description = NULL;

	f_spread_classifier = false;
	Spread_classify_description = NULL;


	f_linear_group = false;
	Linear_group_description = NULL;

	f_permutation_group = false;
	Permutation_group_description = NULL;

	f_group_modification = false;
	Group_modification_description = NULL;

	f_collection = false;
	//std::string list_of_objects;

	f_geometric_object = false;
	//std::string geometric_object_projective_space_label;
	Geometric_object_description = NULL;

	f_graph = false;
	Create_graph_description = NULL;

	f_code = false;
	Create_code_description = NULL;

	f_spread = false;
	Spread_create_description = NULL;

	f_cubic_surface = false;
	Surface_Descr = NULL;

	f_quartic_curve = false;
	Quartic_curve_descr = NULL;

	f_BLT_set = false;
	BLT_Set_create_description = NULL;


	f_translation_plane = false;
	//std::string translation_plane_spread_label;
	//std::string translation_plane_group_n_label;
	//std::string translation_plane_group_np1_label;


	f_spread_table = false;
	//std::string spread_table_label_PA;
	dimension_of_spread_elements = 0;
	//std::string spread_selection_text;
	//std::string spread_table_prefix;
	//std::string spread_table_control;




	f_packing_classify = false;
	//std::string packing_classify_label_PA3;
	//std::string packing_classify_label_PA5;
	//std::string packing_classify_label_spread_table;


	f_packing_was = false;
	//std::string packing_was_label_packing_classify;
	packing_was_descr = NULL;

	f_packing_was_choose_fixed_points = false;
	//std::string packing_with_assumed_symmetry_label;
	packing_with_assumed_symmetry_choose_fixed_points_clique_size = 0;
	packing_with_assumed_symmetry_choose_fixed_points_control = NULL;


	f_packing_long_orbits = false;
	//std::string packing_long_orbits_choose_fixed_points_label
	Packing_long_orbits_description = NULL;

	f_graph_classification = false;
	Graph_classify_description = NULL;

	f_diophant = false;
	Diophant_description = NULL;

	f_design = false;
	Design_create_description = NULL;


	f_design_table = false;
	//std::string design_table_label_design;
	//std::string design_table_label;
	//std::string design_table_group;


	f_large_set_was = false;
	//std::string  large_set_was_label_design_table;
	large_set_was_descr = NULL;


	f_set = false;
	Set_builder_description = false;

	f_vector = false;
	Vector_builder_description = false;

	f_symbolic_object = false;
	Symbolic_object_builder_description = NULL;


	f_combinatorial_object = false;
	Data_input_stream_description = false;

	f_geometry_builder = false;
	Geometry_builder_description = NULL;

	f_vector_ge = false;
	Vector_ge_description = NULL;

	f_action_on_forms = false;
	Action_on_forms_descr = NULL;

	f_orbits = false;
	Orbits_create_description = NULL;

	f_poset_classification_control = false;
	Poset_classification_control = NULL;

	f_poset_classification_report_options = false;
	Poset_classification_report_options = NULL;

	f_draw_options = false;
	Draw_options = NULL;

	f_draw_incidence_structure_options = false;
	Draw_incidence_structure_description = NULL;


	f_arc_generator_control = false;
	Arc_generator_control = NULL;

	f_poset_classification_activity = false;
	Poset_classification_activity = NULL;


	f_crc_code = false;
	Crc_code_description = NULL;

	f_mapping = false;
	Mapping_description = NULL;

	f_variety = false;
	Variety_description = NULL;

}


symbol_definition::~symbol_definition()
{
	Record_death();

}

void symbol_definition::read_definition(
		interface_symbol_table *Sym,
		int argc, std::string *argv, int &i,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "symbol_definition::read_definition "
				"i=" << i << " argc=" << argc << endl;
	}

	symbol_definition::Sym = Sym;

	//f_define = true;
	define_label.assign(argv[++i]);
	if (f_v) {
		cout << "symbol_definition::read_definition "
				"define_label=" << define_label << endl;
	}
	i++;
	if (f_v) {
		cout << "-define " << define_label << endl;
	}
	if (ST.stringcmp(argv[i], "-finite_field") == 0) {
		f_finite_field = true;
		Finite_field_description =
				NEW_OBJECT(algebra::field_theory::finite_field_description);
		if (f_v) {
			cout << "reading -finite_field" << endl;
		}
		i += Finite_field_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-finite_field" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-polynomial_ring") == 0) {
		f_polynomial_ring = true;
		Polynomial_ring_description =
				NEW_OBJECT(algebra::ring_theory::polynomial_ring_description);
		if (f_v) {
			cout << "reading -polynomial_ring" << endl;
		}
		i += Polynomial_ring_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-polynomial_ring" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-projective_space") == 0) {
		f_projective_space = true;
		Projective_space_with_action_description =
				NEW_OBJECT(projective_geometry::projective_space_with_action_description);
		if (f_v) {
			cout << "reading -projective_space" << endl;
		}
		i += Projective_space_with_action_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-projective_space" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-orthogonal_space") == 0) {
		f_orthogonal_space = true;
		Orthogonal_space_with_action_description =
				NEW_OBJECT(orthogonal_geometry_applications::orthogonal_space_with_action_description);
		if (f_v) {
			cout << "reading -orthogonal_space" << endl;
		}
		i += Orthogonal_space_with_action_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-orthogonal_space" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-BLT_set_classifier") == 0) {
		f_BLT_set_classifier = true;
		Blt_set_classify_description =
				NEW_OBJECT(orthogonal_geometry_applications::blt_set_classify_description);
		if (f_v) {
			cout << "reading -BLT_set_classifier" << endl;
		}
		i += Blt_set_classify_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-BLT_set_classifier" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-spread_classifier") == 0) {
		f_spread_classifier = true;
		Spread_classify_description =
				NEW_OBJECT(spreads::spread_classify_description);
		if (f_v) {
			cout << "reading -spread_classifier" << endl;
		}
		i += Spread_classify_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-spread_classifier" << endl;
			Spread_classify_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}

	else if (ST.stringcmp(argv[i], "-linear_group") == 0) {
		f_linear_group = true;
		Linear_group_description =
				NEW_OBJECT(group_constructions::linear_group_description);
		if (f_v) {
			cout << "reading -linear_group" << endl;
		}
		i += Linear_group_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-linear_group" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}

	else if (ST.stringcmp(argv[i], "-permutation_group") == 0) {
		f_permutation_group = true;
		Permutation_group_description =
				NEW_OBJECT(group_constructions::permutation_group_description);
		if (f_v) {
			cout << "reading -permutation_group" << endl;
		}
		i += Permutation_group_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-permutation_group" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}

	else if (ST.stringcmp(argv[i], "-modified_group") == 0) {
		f_group_modification = true;
		Group_modification_description =
				NEW_OBJECT(group_constructions::group_modification_description);
		if (f_v) {
			cout << "reading -modified_group" << endl;
		}
		i += Group_modification_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-modified_group" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}

	else if (ST.stringcmp(argv[i], "-geometric_object") == 0) {
		f_geometric_object = true;

		geometric_object_projective_space_label.assign(argv[++i]);
		Geometric_object_description =
				NEW_OBJECT(geometry::other_geometry::geometric_object_description);
		if (f_v) {
			cout << "symbol_definition::read_definition reading -geometric_object" << endl;
		}
		i += Geometric_object_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-geometric_object" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}


	else if (ST.stringcmp(argv[i], "-collection") == 0) {
		if (f_v) {
			cout << "symbol_definition::read_definition -collection" << endl;
		}

		f_collection = true;

		list_of_objects.assign(argv[++i]);
		i++;

	}
	else if (ST.stringcmp(argv[i], "-graph") == 0) {

		f_graph = true;
		Create_graph_description =
				NEW_OBJECT(apps_graph_theory::create_graph_description);
		if (f_v) {
			cout << "symbol_definition::read_definition reading -graph" << endl;
		}

		i += Create_graph_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-graph" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-graph " << endl;
			Create_graph_description->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-code") == 0) {

		f_code = true;
		Create_code_description =
				NEW_OBJECT(apps_coding_theory::create_code_description);
		if (f_v) {
			cout << "symbol_definition::read_definition reading -code" << endl;
		}

		i += Create_code_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-code" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-code " << endl;
			Create_code_description->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-spread") == 0) {

		f_spread = true;
		Spread_create_description =
				NEW_OBJECT(spreads::spread_create_description);
		if (f_v) {
			cout << "reading -spread" << endl;
		}

		i += Spread_create_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-spread" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-spread " << endl;
			Spread_create_description->print();
		}
	}

	else if (ST.stringcmp(argv[i], "-cubic_surface") == 0) {

		f_cubic_surface = true;
		Surface_Descr =
				NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description);
		if (f_v) {
			cout << "reading -cubic_surface" << endl;
		}

		i += Surface_Descr->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-cubic_surface" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-cubic_surface " << endl;
			Surface_Descr->print();
		}
	}

	else if (ST.stringcmp(argv[i], "-quartic_curve") == 0) {

		f_quartic_curve = true;
		Quartic_curve_descr =
				NEW_OBJECT(applications_in_algebraic_geometry::quartic_curves::quartic_curve_create_description);
		if (f_v) {
			cout << "reading -quartic_curve" << endl;
		}

		i += Quartic_curve_descr->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-quartic_curve" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-quartic_curve " << endl;
			Quartic_curve_descr->print();
		}
	}

	else if (ST.stringcmp(argv[i], "-BLT_set") == 0) {

		f_BLT_set = true;
		BLT_Set_create_description =
				NEW_OBJECT(orthogonal_geometry_applications::BLT_set_create_description);
		if (f_v) {
			cout << "reading -BLT_set" << endl;
		}

		i += BLT_Set_create_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-BLT_set" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-BLT_set " << endl;
			BLT_Set_create_description->print();
		}
	}

	else if (ST.stringcmp(argv[i], "-translation_plane") == 0) {
		f_translation_plane = true;
		translation_plane_spread_label.assign(argv[++i]);
		translation_plane_group_n_label.assign(argv[++i]);
		translation_plane_group_np1_label.assign(argv[++i]);
		i++; // eat -end
		i++;
		if (f_v) {
			cout << "-translation_plane "
					<< " " << translation_plane_spread_label
					<< " " << translation_plane_group_n_label
					<< " " << translation_plane_group_np1_label
					<< endl;
		}
	}


	else if (ST.stringcmp(argv[i], "-spread_table") == 0) {
		f_spread_table = true;

		spread_table_label_PA.assign(argv[++i]);
		dimension_of_spread_elements = ST.strtoi(argv[++i]);
		spread_selection_text.assign(argv[++i]);
		spread_table_prefix.assign(argv[++i]);
		spread_table_control.assign(argv[++i]);


		i++;

		if (f_v) {
			cout << "dimension_of_spread_elements = " << dimension_of_spread_elements
					<< " " << spread_selection_text
					<< " " << spread_table_prefix
					<< " " << spread_table_control
					<< endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}


		if (f_v) {
			cout << "-spread_table " << spread_table_label_PA
					<< " " << dimension_of_spread_elements
					<< " " << spread_selection_text
					<< " " << spread_table_prefix
					<< " " << spread_table_control
					<< endl;
		}
	}

	else if (ST.stringcmp(argv[i], "-packing_classify") == 0) {
		f_packing_classify = true;

		packing_classify_label_PA3.assign(argv[++i]);
		packing_classify_label_PA5.assign(argv[++i]);
		packing_classify_label_spread_table.assign(argv[++i]);

		i++;

		if (f_v) {
			cout << "-packing_classify "
					<< " " << packing_classify_label_PA3
					<< " " << packing_classify_label_PA5
					<< " " << packing_classify_label_spread_table
					<< endl;
		}
	}


	else if (ST.stringcmp(argv[i], "-packing_with_symmetry_assumption") == 0) {
		f_packing_was = true;

		packing_was_label_packing_classify.assign(argv[++i]);

		packing_was_descr =
				NEW_OBJECT(packings::packing_was_description);
		if (f_v) {
			cout << "reading -packing_with_symmetry_assumption" << endl;
		}
		i += packing_was_descr->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-packing_with_symmetry_assumption" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-packing_with_symmetry_assumption " << packing_was_label_packing_classify
					<< endl;
			packing_was_descr->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-packing_choose_fixed_points") == 0) {
		f_packing_was_choose_fixed_points = true;

		packing_with_assumed_symmetry_label.assign(argv[++i]);
		packing_with_assumed_symmetry_choose_fixed_points_clique_size = ST.strtoi(argv[++i]);

		packing_with_assumed_symmetry_choose_fixed_points_control =
				NEW_OBJECT(poset_classification::poset_classification_control);
		if (f_v) {
			cout << "reading -packing_with_symmetry_assumption_choose_fixed_points" << endl;
		}
		i += packing_with_assumed_symmetry_choose_fixed_points_control->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-packing_with_symmetry_assumption_choose_fixed_points" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-packing_with_symmetry_assumption_choose_fixed_points "
					<< packing_with_assumed_symmetry_label
					<< " " << packing_with_assumed_symmetry_choose_fixed_points_clique_size
					<< endl;
			packing_with_assumed_symmetry_choose_fixed_points_control->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-packing_long_orbits") == 0) {
		f_packing_long_orbits = true;

		packing_long_orbits_choose_fixed_points_label.assign(argv[++i]);

		Packing_long_orbits_description =
				NEW_OBJECT(packings::packing_long_orbits_description);
		if (f_v) {
			cout << "reading -packing_long_orbits" << endl;
		}
		i += Packing_long_orbits_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-packing_long_orbits" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-packing_long_orbits "
					<< packing_long_orbits_choose_fixed_points_label
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-graph_classification") == 0) {
		f_graph_classification = true;

		Graph_classify_description =
				NEW_OBJECT(apps_graph_theory::graph_classify_description);
		if (f_v) {
			cout << "reading -graph_classification" << endl;
		}
		i += Graph_classify_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-graph_classification" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-graph_classification "
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-diophant") == 0) {
		f_diophant = true;

		Diophant_description =
				NEW_OBJECT(combinatorics::solvers::diophant_description);
		if (f_v) {
			cout << "reading -diophant_description" << endl;
		}
		i += Diophant_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-diophant_description" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-diophant_description "
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-design") == 0) {

		f_design = true;
		Design_create_description =
				NEW_OBJECT(apps_combinatorics::design_create_description);
		if (f_v) {
			cout << "reading -design" << endl;
		}

		i += Design_create_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-design" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-design_table") == 0) {
		f_design_table = true;

		design_table_label_design.assign(argv[++i]);
		design_table_label.assign(argv[++i]);
		design_table_group.assign(argv[++i]);


		i++;
		i++; //-end

		if (f_v) {
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-design_table " << design_table_label_design
					<< " " << design_table_label
					<< " " << design_table_group
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-large_set_with_symmetry_assumption") == 0) {
		f_large_set_was = true;

		large_set_was_label_design_table.assign(argv[++i]);

		large_set_was_descr =
				NEW_OBJECT(apps_combinatorics::large_set_was_description);
		if (f_v) {
			cout << "reading -large_set_with_symmetry_assumption" << endl;
		}
		i += large_set_was_descr->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-large_set_with_symmetry_assumption" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-large_set_with_symmetry_assumption " << large_set_was_label_design_table
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-set") == 0) {
		f_set = true;


		Set_builder_description =
				NEW_OBJECT(other::data_structures::set_builder_description);
		if (f_v) {
			cout << "reading -set" << endl;
		}
		i += Set_builder_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-set" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-set " << endl;
			Set_builder_description->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-vector") == 0) {
		f_vector = true;


		Vector_builder_description =
				NEW_OBJECT(other::data_structures::vector_builder_description);
		if (f_v) {
			cout << "reading -vector" << endl;
		}
		i += Vector_builder_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-vector" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-vector " << endl;
			Vector_builder_description->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-symbolic_object") == 0) {
		f_symbolic_object = true;


		Symbolic_object_builder_description =
				NEW_OBJECT(algebra::expression_parser::symbolic_object_builder_description);
		if (f_v) {
			cout << "reading -symbolic_object" << endl;
		}
		i += Symbolic_object_builder_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-symbolic_object" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-symbolic_object " << endl;
			Symbolic_object_builder_description->print();
		}
	}


	else if (ST.stringcmp(argv[i], "-combinatorial_object") == 0) {
		f_combinatorial_object = true;


		Data_input_stream_description =
				NEW_OBJECT(combinatorics::canonical_form_classification::data_input_stream_description);
		if (f_v) {
			cout << "reading -combinatorial_objects" << endl;
		}
		i += Data_input_stream_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-vector" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-combinatorial_objects " << endl;
			Data_input_stream_description->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-geometry_builder") == 0) {
		f_geometry_builder = true;


		Geometry_builder_description =
				NEW_OBJECT(combinatorics::geometry_builder::geometry_builder_description);
		if (f_v) {
			cout << "reading -geometry_builder" << endl;
		}
		i += Geometry_builder_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-vector" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-geometry_builder " << endl;
			Geometry_builder_description->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-vector_ge") == 0) {
		f_vector_ge = true;


		Vector_ge_description =
				NEW_OBJECT(data_structures_groups::vector_ge_description);
		if (f_v) {
			cout << "reading -vector_ge" << endl;
		}
		i += Vector_ge_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-vector_ge" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-vector_ge " << endl;
			Vector_ge_description->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-action_on_forms") == 0) {
		f_action_on_forms = true;

		Action_on_forms_descr =
				NEW_OBJECT(apps_algebra::action_on_forms_description);
		if (f_v) {
			cout << "reading -action_on_forms" << endl;
		}
		i += Action_on_forms_descr->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-action_on_forms" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-action_on_forms " << endl;
			Action_on_forms_descr->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-orbits") == 0) {
		f_orbits = true;

		Orbits_create_description =
				NEW_OBJECT(orbits::orbits_create_description);
		if (f_v) {
			cout << "reading -orbits" << endl;
		}
		i += Orbits_create_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-orbits" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-orbits " << endl;
			Orbits_create_description->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-poset_classification_control") == 0) {
		f_poset_classification_control = true;

		Poset_classification_control =
				NEW_OBJECT(poset_classification::poset_classification_control);
		if (f_v) {
			cout << "reading -poset_classification_control" << endl;
		}
		i += Poset_classification_control->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-poset_classification_control" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-poset_classification_control " << endl;
			Poset_classification_control->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-poset_classification_report_options") == 0) {
		f_poset_classification_report_options = true;

		Poset_classification_report_options =
				NEW_OBJECT(poset_classification::poset_classification_report_options);
		if (f_v) {
			cout << "reading -poset_classification_report_options" << endl;
		}
		i += Poset_classification_report_options->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-poset_classification_report_options" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-poset_classification_report_options " << endl;
			Poset_classification_report_options->print();
		}
	}


	else if (ST.stringcmp(argv[i], "-draw_options") == 0) {
		f_draw_options = true;

		Draw_options =
				NEW_OBJECT(other::graphics::layered_graph_draw_options);
		if (f_v) {
			cout << "reading -draw_options" << endl;
		}
		i += Draw_options->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-draw_options" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-draw_options " << endl;
			Draw_options->print();
		}
	}

	else if (ST.stringcmp(argv[i], "-draw_incidence_structure_options") == 0) {
		f_draw_incidence_structure_options = true;

		Draw_incidence_structure_description =
				NEW_OBJECT(other::graphics::draw_incidence_structure_description);
		if (f_v) {
			cout << "reading -draw_incidence_structure_options" << endl;
		}
		i += Draw_incidence_structure_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-draw_incidence_structure_options" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-draw_incidence_structure_options " << endl;
			Draw_incidence_structure_description->print();
		}
	}



	else if (ST.stringcmp(argv[i], "-arc_generator_control") == 0) {
		f_arc_generator_control = true;

		Arc_generator_control =
				NEW_OBJECT(apps_geometry::arc_generator_description);
		if (f_v) {
			cout << "reading -arc_generator_control" << endl;
		}
		i += Arc_generator_control->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-arc_generator_control" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-arc_generator_control " << endl;
			Arc_generator_control->print();
		}
	}



	else if (ST.stringcmp(argv[i], "-poset_classification_activity") == 0) {
		f_poset_classification_activity = true;

		Poset_classification_activity =
				NEW_OBJECT(poset_classification::poset_classification_activity_description);
		if (f_v) {
			cout << "reading -poset_classification_activity" << endl;
		}
		i += Poset_classification_activity->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-poset_classification_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-poset_classification_activity " << endl;
			Poset_classification_activity->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-crc_code") == 0) {

		f_crc_code = true;
		Crc_code_description =
				NEW_OBJECT(combinatorics::coding_theory::crc_code_description);
		if (f_v) {
			cout << "symbol_definition::read_definition reading -crc_code" << endl;
		}

		i += Crc_code_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-crc_code" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-crc_code " << endl;
			Crc_code_description->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-mapping") == 0) {

		f_mapping = true;
		Mapping_description =
				NEW_OBJECT(apps_geometry::mapping_description);
		if (f_v) {
			cout << "symbol_definition::read_definition reading -mapping" << endl;
		}

		i += Mapping_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-mapping" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-mapping " << endl;
			Mapping_description->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-variety") == 0) {

		f_variety = true;
		Variety_description =
				NEW_OBJECT(geometry::algebraic_geometry::variety_description);
		if (f_v) {
			cout << "reading -variety" << endl;
		}

		i += Variety_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-variety" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-variety " << endl;
			Variety_description->print();
		}
	}


	else {
		cout << "unrecognized command after -define" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "symbol_definition::read_definition done" << endl;
	}
}


void symbol_definition::perform_definition(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::perform_definition" << endl;
	}

	if (f_finite_field) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_finite_field" << endl;
		}
		definition_of_finite_field(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_finite_field" << endl;
		}
	}
	else if (f_polynomial_ring) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_polynomial_ring" << endl;
		}
		definition_of_polynomial_ring(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_polynomial_ring" << endl;
		}
	}
	else if (f_projective_space) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_projective_space" << endl;
		}
		definition_of_projective_space(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_projective_space" << endl;
		}
	}
	else if (f_orthogonal_space) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_orthogonal_space" << endl;
		}
		definition_of_orthogonal_space(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_orthogonal_space" << endl;
		}
	}
	else if (f_BLT_set_classifier) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_BLT_set_classifier" << endl;
		}
		definition_of_BLT_set_classifier(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_BLT_set_classifier" << endl;
		}
	}
	else if (f_spread_classifier) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_spread_classifier" << endl;
		}
		definition_of_spread_classifier(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_spread_classifier" << endl;
		}
	}

	else if (f_linear_group) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_linear_group" << endl;
		}
		definition_of_linear_group(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_linear_group" << endl;
		}
	}
	else if (f_permutation_group) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_permutation_group" << endl;
		}
		definition_of_permutation_group(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_permutation_group" << endl;
		}
	}
	else if (f_group_modification) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_modified_group" << endl;
		}
		definition_of_modified_group(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_modified_group" << endl;
		}
	}

	else if (f_geometric_object) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_geometric_object" << endl;
		}
		definition_of_geometric_object(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_geometric_object" << endl;
		}
	}
	else if (f_collection) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_collection" << endl;
		}
		definition_of_collection(list_of_objects, verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_collection" << endl;
		}
	}
	else if (f_graph) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_graph" << endl;
		}
		definition_of_graph(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_graph" << endl;
		}
	}
	else if (f_code) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_code" << endl;
		}
		definition_of_code(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_code" << endl;
		}
	}
	else if (f_spread) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_spread" << endl;
		}
		definition_of_spread(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_spread" << endl;
		}
	}
	else if (f_cubic_surface) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_cubic_surface" << endl;
		}
		definition_of_cubic_surface(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_cubic_surface" << endl;
		}
	}
	else if (f_quartic_curve) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_quartic_curve" << endl;
		}
		definition_of_quartic_curve(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_quartic_curve" << endl;
		}
	}
	else if (f_BLT_set) {

		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_BLT_set" << endl;
		}
		definition_of_BLT_set(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_BLT_set" << endl;
		}
	}
	else if (f_translation_plane) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_translation_plane" << endl;
		}
		definition_of_translation_plane(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_translation_plane" << endl;
		}
	}
	else if (f_spread_table) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_projective_space" << endl;
		}
		definition_of_spread_table(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_projective_space" << endl;
		}
	}
	else if (f_packing_classify) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_packing_classify" << endl;
		}
		definition_of_packing_classify(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_packing_classify" << endl;
		}
	}
	else if (f_packing_was) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_packing_was" << endl;
		}
		definition_of_packing_was(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_packing_was" << endl;
		}
	}
	else if (f_packing_was_choose_fixed_points) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_packing_was_choose_fixed_points" << endl;
		}
		definition_of_packing_was_choose_fixed_points(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_packing_was_choose_fixed_points" << endl;
		}
	}
	else if (f_packing_long_orbits) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_packing_long_orbits" << endl;
		}
		definition_of_packing_long_orbits(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_packing_long_orbits" << endl;
		}
	}
	else if (f_graph_classification) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_graph_classification" << endl;
		}
		definition_of_graph_classification(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_graph_classification" << endl;
		}
	}
	else if (f_diophant) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_graph_classification" << endl;
		}
		definition_of_diophant(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_graph_classification" << endl;
		}
	}
	else if (f_design) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_design" << endl;
		}
		definition_of_design(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_design" << endl;
		}
	}
	else if (f_design_table) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_design_table" << endl;
		}
		definition_of_design_table(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_design_table" << endl;
		}
	}
	else if (f_large_set_was) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_packing_was" << endl;
		}
		definition_of_large_set_was(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_packing_was" << endl;
		}
	}
	else if (f_set) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_set" << endl;
		}
		definition_of_set(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_set" << endl;
		}
	}
	else if (f_vector) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_vector" << endl;
		}
		definition_of_vector(define_label, Vector_builder_description, verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_vector" << endl;
		}
	}
	else if (f_symbolic_object) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_symbolic_object" << endl;
		}
		definition_of_symbolic_object(define_label, Symbolic_object_builder_description, verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_symbolic_object" << endl;
		}
	}
	else if (f_combinatorial_object) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_combinatorial_object" << endl;
		}
		definition_of_combinatorial_object(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_combinatorial_object" << endl;
		}
	}
	else if (f_geometry_builder) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before do_geometry_builder" << endl;
		}
		do_geometry_builder(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after do_geometry_builder" << endl;
		}
	}
	else if (f_vector_ge) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_vector_ge" << endl;
		}
		definition_of_vector_ge(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_vector_ge" << endl;
		}
	}
	else if (f_action_on_forms) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"f_action_on_forms" << endl;
		}
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_action_on_forms" << endl;
		}
		definition_of_action_on_forms(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_action_on_forms" << endl;
		}
	}
	else if (f_orbits) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"f_orbits" << endl;
		}
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_orbits" << endl;
		}
		definition_of_orbits(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_orbits" << endl;
		}
	}
	else if (f_poset_classification_control) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"f_poset_classification_control" << endl;
		}
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_poset_classification_control" << endl;
		}
		definition_of_poset_classification_control(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_poset_classification_control" << endl;
		}
	}
	else if (f_poset_classification_report_options) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"f_poset_classification_report_options" << endl;
		}
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_poset_classification_report_options" << endl;
		}
		definition_of_poset_classification_report_options(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_poset_classification_report_options" << endl;
		}
	}

	else if (f_draw_options) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"f_draw_options" << endl;
		}
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_draw_options" << endl;
		}
		definition_of_draw_options(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_draw_options" << endl;
		}
	}
	else if (f_draw_incidence_structure_options) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"f_draw_incidence_structure_options" << endl;
		}
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_draw_incidence_structure_options" << endl;
		}
		definition_of_draw_incidence_structure_options(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_draw_incidence_structure_options" << endl;
		}
	}
	else if (f_arc_generator_control) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"f_arc_generator_control" << endl;
		}
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_arc_generator_control" << endl;
		}
		definition_of_arc_generator_control(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_arc_generator_control" << endl;
		}
	}




	else if (f_poset_classification_activity) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"f_poset_classification_activity" << endl;
		}
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_poset_classification_activity" << endl;
		}
		definition_of_poset_classification_activity(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_poset_classification_activityt" << endl;
		}
	}
	else if (f_crc_code) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_crc_code" << endl;
		}
		definition_of_crc_code(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_crc_code" << endl;
		}
	}
	else if (f_mapping) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_mapping" << endl;
		}
		definition_of_mapping(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_mapping" << endl;
		}
	}
	else if (f_variety) {
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"before definition_of_variety" << endl;
		}
		definition_of_variety(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition "
					"after definition_of_variety" << endl;
		}
	}

	else {
		if (f_v) {
			cout << "symbol_definition::perform_definition no definition" << endl;
			exit(1);
		}
	}
}


void symbol_definition::print()
{
	cout << "-define " << define_label << " " << endl;
	if (f_finite_field) {
		cout << "-finite_field ";
		Finite_field_description->print();
	}
	else if (f_polynomial_ring) {
		cout << "-polynomial_ring ";
		Polynomial_ring_description->print();
	}
	else if (f_projective_space) {
		cout << "-projective_space ";
		Projective_space_with_action_description->print();
	}
	else if (f_orthogonal_space) {
		cout << "-orthogonal_space ";
		Orthogonal_space_with_action_description->print();
	}
	else if (f_BLT_set_classifier) {
		cout << "-BLT_set_classifier " << " ";
		Blt_set_classify_description->print();
	}
	else if (f_spread_classifier) {
		cout << "-spread_classifier" << endl;
		Spread_classify_description->print();
	}
	else if (f_linear_group) {
		cout << "-linear_group ";
		Linear_group_description->print();
	}
	else if (f_permutation_group) {
		cout << "-permutation_group ";
		Permutation_group_description->print();
	}
	else if (f_group_modification) {
		cout << "-modified_group ";
		Group_modification_description->print();
	}
	else if (f_geometric_object) {
		cout << "-geometric_object ";
		Geometric_object_description->print();
	}
	else if (f_collection) {
		cout << "-collection ";
		//cout << list_of_objects << endl;
	}
	else if (f_graph) {
		cout << "-graph ";
		Create_graph_description->print();
	}
	else if (f_code) {
		cout << "-code ";
		Create_code_description->print();
	}
	else if (f_spread) {
		cout << "-spread ";
		Spread_create_description->print();
	}
	else if (f_cubic_surface) {
		cout << "-cubic_surface " << endl;
		Surface_Descr->print();
	}
	else if (f_quartic_curve) {
		cout << "-quartic_curve " << endl;
		Quartic_curve_descr->print();
	}
	else if (f_BLT_set) {
		cout << "-BLT_set " << endl;
		BLT_Set_create_description->print();
	}
	else if (f_translation_plane) {
		cout << "-translation_plane "
				<< " " << translation_plane_spread_label
				<< " " << translation_plane_group_n_label
				<< " " << translation_plane_group_np1_label
				<< endl;
	}
	else if (f_spread_table) {
		cout << "-spread_table " << spread_table_label_PA
				<< " " << dimension_of_spread_elements
				<< " " << spread_selection_text
				<< " " << spread_table_prefix
				<< " " << spread_table_control
				<< endl;
	}
	else if (f_packing_classify) {
		cout << "-packing_classify "
				<< " " << packing_classify_label_PA3
				<< " " << packing_classify_label_PA5
				<< " " << packing_classify_label_spread_table
				<< endl;
	}
	else if (f_packing_was) {
		cout << "-packing_with_symmetry_assumption " << packing_was_label_packing_classify
				<< endl;
		packing_was_descr->print();
	}
	else if (f_packing_was_choose_fixed_points) {
		cout << "-packing_was_choose_fixed_points ";
		cout << packing_with_assumed_symmetry_label;
		cout << " " << packing_with_assumed_symmetry_choose_fixed_points_clique_size << " " << endl;
		packing_with_assumed_symmetry_choose_fixed_points_control->print();
		//std::string packing_with_assumed_symmetry_label;
		//int packing_with_assumed_symmetry_choose_fixed_points_clique_size;
		//poset_classification_control *packing_with_assumed_symmetry_choose_fixed_points_control;
	}
	else if (f_packing_long_orbits) {
		cout << "-packing_long_orbits " << packing_long_orbits_choose_fixed_points_label << endl;
		Packing_long_orbits_description->print();
	}
	else if (f_graph_classification) {
		cout << "-graph_classification ";
		Graph_classify_description->print();
	}
	else if (f_diophant) {
		cout << "-diophant ";
		Diophant_description->print();
	}
	else if (f_design) {
		cout << "-design ";
		Design_create_description->print();
	}
	else if (f_design_table) {
		cout << "-design_table "
				<< design_table_label_design
				<< " " << design_table_label
				<< " " << design_table_group << endl;
	}
	else if (f_large_set_was) {
		cout << "-large_set_was " << large_set_was_label_design_table << endl;
		large_set_was_descr->print();
	}
	else if (f_set) {
		cout << "-set ";
		Set_builder_description->print();
	}
	else if (f_vector) {
		cout << "-vector ";
		Vector_builder_description->print();
	}
	else if (f_symbolic_object) {
		cout << "-symbolic_object ";
		Symbolic_object_builder_description->print();
	}
	else if (f_combinatorial_object) {
		cout << "-combinatorial_object ";
		Data_input_stream_description->print();
	}
	else if (f_geometry_builder) {
		cout << "-geometry_builder ";
		Geometry_builder_description->print();
	}
	else if (f_vector_ge) {
		cout << "-vector_g ";
		Vector_ge_description->print();
	}
	else if (f_action_on_forms) {
		cout << "-action_on_forms ";
		Action_on_forms_descr->print();
	}
	else if (f_orbits) {
		cout << "-orbits ";
		Orbits_create_description->print();
	}
	else if (f_poset_classification_control) {
		cout << "-poset_classification_control ";
		Poset_classification_control->print();
	}
	else if (f_poset_classification_report_options) {
		cout << "-poset_classification_report_options ";
		Poset_classification_report_options->print();
	}
	else if (f_draw_options) {
		cout << "-draw_options ";
		Draw_options->print();
	}
	else if (f_draw_incidence_structure_options) {
		cout << "-draw_incidence_structure_options " << endl;
		Draw_incidence_structure_description->print();
	}

	else if (f_arc_generator_control) {
		cout << "-arc_generator_control" << endl;
		Arc_generator_control->print();
	}

	else if (f_poset_classification_activity) {
		cout << "-poset_classification_activity ";
		Poset_classification_activity->print();
	}
	else if (f_crc_code) {
		cout << "-crc_code ";
		Crc_code_description->print();
	}
	else if (f_mapping) {
		cout << "-mapping ";
		Mapping_description->print();
	}
	else if (f_variety) {
		cout << "-variety ";
		Variety_description->print();
	}
	else {
		cout << "symbol_definition::print unknown type" << endl;
		exit(1);
	}
}





void symbol_definition::definition_of_finite_field(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_finite_field" << endl;
	}
	Finite_field_description->print();
	algebra::field_theory::finite_field *F;

	F = NEW_OBJECT(algebra::field_theory::finite_field);
	if (f_v) {
		cout << "symbol_definition::definition_of_finite_field "
				"before F->init" << endl;
	}
	F->init(Finite_field_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_finite_field "
				"after F->init" << endl;
	}

	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_finite_field(
			define_label, F, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_finite_field "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_finite_field done" << endl;
	}
}

void symbol_definition::definition_of_polynomial_ring(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_polynomial_ring" << endl;
	}
	Polynomial_ring_description->print();
	algebra::ring_theory::homogeneous_polynomial_domain *HPD;

	HPD = NEW_OBJECT(algebra::ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "symbol_definition::definition_of_polynomial_ring "
				"before HPD->init" << endl;
	}
	HPD->init(Polynomial_ring_description, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_polynomial_ring "
				"after F->init" << endl;
	}

	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_polynomial_ring(
			define_label, HPD, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_polynomial_ring "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_polynomial_ring done" << endl;
	}
}



void symbol_definition::definition_of_projective_space(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space "
				"verbose_level=" << verbose_level << endl;
	}


	if (Projective_space_with_action_description->f_override_verbose_level) {
		verbose_level = Projective_space_with_action_description->override_verbose_level;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space "
				"before load_finite_field_PG" << endl;
	}
	load_finite_field_PG(verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space "
				"after load_finite_field_PG" << endl;
	}


#if 0
	int f_semilinear;
	number_theory::number_theory_domain NT;


	if (NT.is_prime(Projective_space_with_action_description->F->q)) {
		f_semilinear = false;
	}
	else {
		f_semilinear = true;
	}

	if (Projective_space_with_action_description->f_use_projectivity_subgroup) {
		f_semilinear = false;
	}
#endif

	projective_geometry::projective_space_with_action *PA;

	PA = NEW_OBJECT(projective_geometry::projective_space_with_action);

	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space "
				"before PA->init_from_description" << endl;
	}
	PA->init_from_description(Projective_space_with_action_description, verbose_level);
#if 0
	PA->init(Projective_space_with_action_description->F,
			Projective_space_with_action_description->n,
		f_semilinear,
		true /*f_init_incidence_structure*/,
		verbose_level - 2);
#endif
	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space "
				"after PA->init_from_description" << endl;
	}

	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_projective_space(define_label, PA, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space done" << endl;
	}
}

void symbol_definition::print_definition_of_projective_space(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::print_definition_of_projective_space" << endl;
	}
	Projective_space_with_action_description->print();
}

void symbol_definition::definition_of_orthogonal_space(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_orthogonal_space" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_orthogonal_space "
				"before get_or_create_finite_field" << endl;
	}
	Orthogonal_space_with_action_description->F = get_or_create_finite_field(
			Orthogonal_space_with_action_description->input_q,
			verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_orthogonal_space "
				"after get_or_create_finite_field" << endl;
	}

	//int f_semilinear;
	algebra::number_theory::number_theory_domain NT;

#if 0
	if (NT.is_prime(Orthogonal_space_with_action_description->F->q)) {
		f_semilinear = false;
	}
	else {
		f_semilinear = true;
	}
#endif

	orthogonal_geometry_applications::orthogonal_space_with_action *OA;

	OA = NEW_OBJECT(orthogonal_geometry_applications::orthogonal_space_with_action);

	if (f_v) {
		cout << "symbol_definition::definition_of_orthogonal_space "
				"before OA->init" << endl;
	}
	OA->init(Orthogonal_space_with_action_description,
		verbose_level - 1);
	if (f_v) {
		cout << "symbol_definition::definition_of_orthogonal_space "
				"after OA->init" << endl;
	}

	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_orthogonal_space(
			define_label, OA, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_orthogonal_space "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_orthogonal_space done" << endl;
	}
}


void symbol_definition::definition_of_BLT_set_classifier(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set_classifier" << endl;
	}




	orthogonal_geometry_applications::blt_set_classify *BLT_classify;

	BLT_classify = NEW_OBJECT(orthogonal_geometry_applications::blt_set_classify);

	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set_classifier "
				"before BLT_classify->init" << endl;
	}
	BLT_classify->init(
			Blt_set_classify_description,
			verbose_level - 1);
	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set_classifier "
				"after BLT_classify->init" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set_classifier "
				"before BLT_classify->init_basic" << endl;
	}


	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_BLT_set_classify(
			define_label, BLT_classify, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set_classifier "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set_classifier done" << endl;
	}
}

void symbol_definition::definition_of_spread_classifier(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_spread_classifier" << endl;
	}


	spreads::spread_classify *Spread_classify;

	Spread_classify = NEW_OBJECT(spreads::spread_classify);

	if (f_v) {
		cout << "symbol_definition::definition_of_spread_classifier "
				"before Spread_classify->init_basic" << endl;
	}
	Spread_classify->init_basic(
			Spread_classify_description,
			verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_spread_classifier "
				"after Spread_classify->init_basic" << endl;
	}

	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_spread_classify(
			define_label, Spread_classify, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_spread_classifier "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_spread_classifier done" << endl;
	}
}


void symbol_definition::definition_of_linear_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_linear_group" << endl;
	}

	if (Linear_group_description->input_q.length()) {


		if (f_v) {
			cout << "symbol_definition::definition_of_linear_group "
					"before get_or_create_finite_field" << endl;
		}
		Linear_group_description->F = get_or_create_finite_field(
				Linear_group_description->input_q,
				verbose_level);
		if (f_v) {
			cout << "symbol_definition::definition_of_linear_group "
					"after get_or_create_finite_field" << endl;
		}
	}


	group_constructions::linear_group *LG;

	LG = NEW_OBJECT(group_constructions::linear_group);
	if (f_v) {
		cout << "symbol_definition::definition "
				"before LG->linear_group_init, "
				"creating the group" << endl;
	}

	LG->linear_group_init(
			Linear_group_description, verbose_level - 2);

	if (f_v) {
		cout << "symbol_definition::definition "
				"after LG->linear_group_init" << endl;
	}


	// create any_group object from linear_group:


	groups::any_group *AG;

	AG = NEW_OBJECT(groups::any_group);
	if (f_v) {
		cout << "symbol_definition::definition "
				"before AG->init_linear_group" << endl;
	}
	AG->init_linear_group(LG, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition "
				"after AG->init_linear_group" << endl;
	}




	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_any_group(
			define_label, AG, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_linear_group done" << endl;
	}
}

void symbol_definition::definition_of_permutation_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_permutation_group" << endl;
	}


	group_constructions::permutation_group_create *PGC;

	PGC = NEW_OBJECT(group_constructions::permutation_group_create);
	if (f_v) {
		cout << "symbol_definition::definition_of_permutation_group "
				"before PGC->permutation_group_init" << endl;
	}

	PGC->permutation_group_init(Permutation_group_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_permutation_group "
				"after PGC->permutation_group_init" << endl;
	}


	// create any_group object from permutation_group_create:


	groups::any_group *AG;

	AG = NEW_OBJECT(groups::any_group);
	AG->init_permutation_group(PGC, verbose_level);



	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_any_group(
			define_label, AG, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_permutation_group "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_permutation_group done" << endl;
	}
}


void symbol_definition::definition_of_modified_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_modified_group" << endl;
	}


	group_constructions::modified_group_create *MGC;

	MGC = NEW_OBJECT(group_constructions::modified_group_create);
	if (f_v) {
		cout << "symbol_definition::definition_of_modified_group "
				"before PGC->permutation_group_init, "
				"before PGC->permutation_group_init" << endl;
	}

	apps_algebra::algebra_global_with_action Algebra_global_with_action;

	Algebra_global_with_action.modified_group_init(
			MGC, Group_modification_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_modified_group "
				"after PGC->permutation_group_init, "
				"after PGC->permutation_group_init" << endl;
	}

	groups::any_group *AG;

	AG = NEW_OBJECT(groups::any_group);
	AG->init_modified_group(MGC, verbose_level);

	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);

	Symb->init_any_group(
			define_label, AG, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_modified_group "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_modified_group done" << endl;
	}
}

void symbol_definition::definition_of_geometric_object(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_geometric_object" << endl;
	}


	//geometry::geometric_object_create *GOC;

	//GOC = NEW_OBJECT(geometry::geometric_object_create);


	projective_geometry::projective_space_with_action *PA;

	PA = Get_projective_space(geometric_object_projective_space_label);

	geometry::other_geometry::geometric_object_create *GeoObj;

	GeoObj = NEW_OBJECT(geometry::other_geometry::geometric_object_create);

	if (f_v) {
		cout << "symbol_definition::definition_of_geometric_object "
				"before GeoObj->init" << endl;
	}

	GeoObj->init(
			Geometric_object_description, PA->P, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_geometric_object "
				"after GeoObj->init" << endl;
	}

	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);



	Symb->init_geometric_object(
			define_label, GeoObj, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_geometric_object "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_geometric_object done" << endl;
	}
}





void symbol_definition::definition_of_collection(
		std::string &list_of_objects,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_collection" << endl;
	}

	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_collection(
			define_label, list_of_objects, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_formula "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_collection done" << endl;
	}
}

void symbol_definition::definition_of_graph(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_graph" << endl;
	}

	apps_graph_theory::create_graph *Gr;

	Gr = NEW_OBJECT(apps_graph_theory::create_graph);

	if (f_v) {
		cout << "symbol_definition::definition_of_graph "
				"before Gr->init" << endl;
	}
	Gr->init(Create_graph_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_graph "
				"after Gr->init" << endl;
	}
	if (f_v) {
		cout << "Gr->N=" << Gr->N << endl;
		cout << "Gr->label=" << Gr->label << endl;
		cout << "Gr->f_has_CG=" << Gr->f_has_CG << endl;
		//cout << "Adj:" << endl;
		//int_matrix_print(Gr->Adj, Gr->N, Gr->N);
	}


	if (f_v) {
		cout << "Gr->CG->nb_points = " << Gr->CG->nb_points << endl;
	}


	if (f_v) {
		cout << "symbol_definition::definition_of_graph "
				"we created a graph on " << Gr->N
				<< " points, called " << Gr->label << endl;

#if 0
		cout << "list of points:" << endl;

		cout << COC->nb_pts << endl;
		for (i = 0; i < COC->nb_pts; i++) {
			cout << COC->Pts[i] << " ";
			}
		cout << endl;
#endif
	}


	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_graph(
			define_label, Gr->CG, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_graph "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_graph done" << endl;
	}
}


void symbol_definition::definition_of_code(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_code" << endl;
	}

	apps_coding_theory::create_code *Code;

	Code = NEW_OBJECT(apps_coding_theory::create_code);

	if (f_v) {
		cout << "symbol_definition::definition_of_code "
				"before Code->init" << endl;
	}
	Code->init(Create_code_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_code "
				"after Code->init" << endl;
	}
	if (f_v) {
		cout << "Code->label_txt" << Code->label_txt << endl;
	}



	if (f_v) {
		cout << "symbol_definition::definition_of_code "
				"we have created a code called " << Code->label_txt << endl;

	}


	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_code(
			define_label, Code, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_code "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_code done" << endl;
	}
}



void symbol_definition::definition_of_spread(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_spread" << endl;
	}


	spreads::spread_create *Spread;


	Spread = NEW_OBJECT(spreads::spread_create);

	if (f_v) {
		cout << "symbol_definition::definition_of_spread "
				"before Spread->init" << endl;
	}
	Spread->init(Spread_create_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_spread "
				"after Spread->init" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_spread "
				"we have created a spread called " << Spread->label_txt << endl;

	}


	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_spread(
			define_label, Spread, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_spread before "
				"add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_spread done" << endl;
	}
}


void symbol_definition::definition_of_cubic_surface(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_cubic_surface" << endl;
	}


	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create *SC;

	SC = NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create);

	SC->create_cubic_surface(
			Surface_Descr,
			verbose_level);


	if (f_v) {
		cout << "symbol_definition::definition_of_cubic_surface "
				"we have created a cubic surface called " << SC->SO->label_txt << endl;

	}


	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_cubic_surface(
			define_label, SC, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_cubic_surface "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_cubic_surface done" << endl;
	}
}



void symbol_definition::definition_of_quartic_curve(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_quartic_curve" << endl;
	}


	applications_in_algebraic_geometry::quartic_curves::quartic_curve_create *QC;


	QC = NEW_OBJECT(applications_in_algebraic_geometry::quartic_curves::quartic_curve_create);

	if (f_v) {
		cout << "symbol_definition::definition_of_quartic_curve "
				"before QC->create_quartic_curve" << endl;
	}
	QC->create_quartic_curve(
			Quartic_curve_descr,
			verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_quartic_curve "
				"after QC->create_quartic_curve" << endl;
	}



	if (f_v) {
		cout << "symbol_definition::definition_of_quartic_curve "
				"we have created a quartic curve called " << QC->label_txt << endl;

	}


	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_quartic_curve(
			define_label, QC, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_quartic_curve "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_quartic_curve done" << endl;
	}
}



void symbol_definition::definition_of_BLT_set(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set" << endl;
	}



	if (!BLT_Set_create_description->f_space) {
		cout << "please specify the orthogonal space using -space <label>" << endl;
		exit(1);
	}

	orthogonal_geometry_applications::orthogonal_space_with_action *OA;

	OA = Get_orthogonal_space(BLT_Set_create_description->space_label);

	orthogonal_geometry_applications::BLT_set_create *BC;

	BC = NEW_OBJECT(orthogonal_geometry_applications::BLT_set_create);


	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set "
				"before BC->init" << endl;
	}
	BC->init(
			BLT_Set_create_description,
			OA,
			verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set "
				"after BC->init" << endl;
	}



	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set "
				"we created a BLT-set called " << BC->label_txt << endl;

	}


	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_BLT_set(define_label, BC, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set done" << endl;
	}
}




void symbol_definition::definition_of_translation_plane(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_translation_plane" << endl;
	}

	//translation_plane_spread_label
	//translation_plane_group_n_label
	//translation_plane_group_np1_label


	spreads::spread_create *Spread;
	groups::any_group *Gn;
	groups::any_group *Gnp1;
	combinatorics_with_groups::translation_plane_via_andre_model *TP;


	Spread = Get_object_of_type_spread(translation_plane_spread_label);

	if (f_v) {
		cout << "symbol_definition::definition_of_translation_plane "
				"found spread " << Spread->label_txt << endl;
	}

	Gn = Get_any_group(translation_plane_group_n_label);

	if (f_v) {
		cout << "symbol_definition::definition_of_translation_plane "
				"found group Gn " << Gn->label << endl;
	}

	Gnp1 = Get_any_group(translation_plane_group_np1_label);

	if (f_v) {
		cout << "symbol_definition::definition_of_translation_plane "
				"found group Gnp1 " << Gnp1->label << endl;
	}

	TP = NEW_OBJECT(combinatorics_with_groups::translation_plane_via_andre_model);

	actions::action *An1;

	An1 = Gnp1->A_base;

	if (f_v) {
		cout << "symbol_definition::definition_of_translation_plane "
				"before TP->init" << endl;
	}
	TP->init(
			Spread->k,
			Spread->label_txt,
			Spread->label_tex,
			Spread->Sg,
			Spread->Andre,
			Spread->A,
			An1,
			verbose_level);

	// TP is in layer 4 and hence does not know about the spread class,
	// so we need to pass the arguments individually.

	if (f_v) {
		cout << "symbol_definition::definition_of_translation_plane "
				"after TP->init" << endl;
		cout << "symbol_definition::definition_of_translation_plane "
				"TP->label_txt=" << TP->label_txt << endl;
		cout << "symbol_definition::definition_of_translation_plane "
				"TP->label_tex=" << TP->label_tex << endl;
		//Spread->Andre->report(cout, verbose_level);
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_translation_plane "
				"we created a translation plane called " << TP->label_txt << endl;

	}

	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_translation_plane(
			define_label, TP, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_translation_plane "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_translation_plane done" << endl;
	}
}




void symbol_definition::definition_of_spread_table(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_spread_table" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_spread_table "
				"using existing PA "
				<< spread_table_label_PA << endl;
	}
	projective_geometry::projective_space_with_action *PA;

	PA = Get_projective_space(spread_table_label_PA);



#if 0
	packings::packing_classify *P;

	if (f_v) {
		cout << "symbol_definition::definition_of_spread_table "
				"before P->spread_table_init" << endl;
	}

	P = NEW_OBJECT(packings::packing_classify);

	P->spread_table_init(
			PA,
			dimension_of_spread_elements,
			true /* f_select_spread */, spread_selection_text,
			spread_tables_prefix,
			spread_table_control,
			verbose_level);


	if (f_v) {
		cout << "symbol_definition::definition_of_spread_table "
				"after P->spread_table_init" << endl;
	}
#endif

	spreads::spread_table_with_selection *Spread_table_with_selection;

	Spread_table_with_selection = NEW_OBJECT(spreads::spread_table_with_selection);

	Spread_table_with_selection->do_spread_table_init(
			PA,
			dimension_of_spread_elements,
			true /* f_select_spread */, spread_selection_text,
			spread_table_prefix,
			spread_table_control,
			verbose_level);



	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_spread_table(define_label, Spread_table_with_selection, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_spread_table "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_spread_table done" << endl;
	}
}



void symbol_definition::definition_of_packing_classify(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_classify" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_classify "
				"using existing PA "
				<< spread_table_label_PA << endl;
	}
	projective_geometry::projective_space_with_action *PA3;
	projective_geometry::projective_space_with_action *PA5;

	PA3 = Get_projective_space(packing_classify_label_PA3);
	PA5 = Get_projective_space(packing_classify_label_PA5);

	spreads::spread_table_with_selection *Spread_table;
	Spread_table = Get_spread_table(packing_classify_label_spread_table);


	packings::packing_classify *PC;


	PC = NEW_OBJECT(packings::packing_classify);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_classify "
				"before PC->init" << endl;
	}
	PC->init(
			PA3,
			PA5,
		Spread_table,
		true,
		verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_packing_classify "
				"after PC->init" << endl;
	}





	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_packing_classify(define_label, PC, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_packing_classify "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_packing_classify done" << endl;
	}
}



void symbol_definition::definition_of_packing_was(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was "
				"using existing spread table "
				<< packing_was_label_packing_classify << endl;
	}

	packings::packing_classify *P;

	P = Get_packing_classify(packing_was_label_packing_classify);




	packings::packing_was *PW;

	PW = NEW_OBJECT(packings::packing_was);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was "
				"before PW->init" << endl;
	}

	PW->init(packing_was_descr, P, verbose_level);

	if (f_v) {
		cout << "symbol_definition::perform_activity "
				"after PW->init" << endl;
	}




	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_packing_was(
			define_label, PW, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was done" << endl;
	}
}



void symbol_definition::definition_of_packing_was_choose_fixed_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was_choose_fixed_points" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was_choose_fixed_points "
				"using existing object "
				<< packing_with_assumed_symmetry_label << endl;
	}
	int idx;
	packings::packing_was *PW;

	idx = Sym->Orbiter_top_level_session->find_symbol(
			packing_with_assumed_symmetry_label);
	PW = (packings::packing_was *)
			Sym->Orbiter_top_level_session->get_object(idx);


	packings::packing_was_fixpoints *PWF;

	PWF = NEW_OBJECT(packings::packing_was_fixpoints);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was_choose_fixed_points "
				"before PWF->init" << endl;
	}

	PWF->init(PW,
			packing_with_assumed_symmetry_choose_fixed_points_clique_size,
			packing_with_assumed_symmetry_choose_fixed_points_control,
			verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was_choose_fixed_points "
				"after PWF->init" << endl;
	}

	if (packing_with_assumed_symmetry_choose_fixed_points_clique_size > 0) {
		PWF->compute_cliques_on_fixpoint_graph(
				packing_with_assumed_symmetry_choose_fixed_points_clique_size,
				packing_with_assumed_symmetry_choose_fixed_points_control,
				verbose_level);
	}
	else {
		cout << "symbol_definition::definition_of_packing_was_choose_fixed_points "
				"clique size on fixed spreads is zero, so nothing to do" << endl;
	}




	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_packing_was_choose_fixed_points(
			define_label, PWF, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was_choose_fixed_points "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was_choose_fixed_points done" << endl;
	}
}





void symbol_definition::definition_of_packing_long_orbits(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_long_orbits" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_long_orbits "
				"using existing object "
				<< packing_long_orbits_choose_fixed_points_label << endl;
	}
	int idx;

	packings::packing_was_fixpoints *PWF;

	idx = Sym->Orbiter_top_level_session->find_symbol(
			packing_long_orbits_choose_fixed_points_label);
	PWF = (packings::packing_was_fixpoints *)
			Sym->Orbiter_top_level_session->get_object(idx);


	packings::packing_long_orbits *PL;

	PL = NEW_OBJECT(packings::packing_long_orbits);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_long_orbits "
				"before PL->init" << endl;
	}

	PL->init(PWF, Packing_long_orbits_description, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_long_orbits "
				"after PL->init" << endl;
	}




	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_packing_long_orbits(
			define_label, PL, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_packing_long_orbits "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_packing_long_orbits done" << endl;
	}
}


void symbol_definition::definition_of_graph_classification(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_graph_classification" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_graph_classification "
				"using existing object "
				<< packing_long_orbits_choose_fixed_points_label << endl;
	}


	apps_graph_theory::graph_classify *GC;


	GC = NEW_OBJECT(apps_graph_theory::graph_classify);

	if (f_v) {
		cout << "symbol_definition::definition_of_graph_classification "
				"before GC->init" << endl;
	}

	GC->init(
			Graph_classify_description, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_graph_classification "
				"after GC->init" << endl;
	}




	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_graph_classify(
			define_label, GC, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_graph_classification "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_graph_classification done" << endl;
	}
}

void symbol_definition::definition_of_diophant(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_diophant" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_diophant "
				"using existing object "
				<< packing_long_orbits_choose_fixed_points_label << endl;
	}


	combinatorics::solvers::diophant_create *Dio;


	Dio = NEW_OBJECT(combinatorics::solvers::diophant_create);

	if (f_v) {
		cout << "symbol_definition::definition_of_diophant "
				"before Dio->init" << endl;
	}

	Dio->init(Diophant_description, verbose_level);


	if (f_v) {
		cout << "symbol_definition::definition_of_diophant "
				"after Dio->init" << endl;
	}




	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_diophant(
			define_label, Dio, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_diophant "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_diophant done" << endl;
	}
}



void symbol_definition::definition_of_design(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_design" << endl;
	}


	apps_combinatorics::design_create *DC;


	DC = NEW_OBJECT(apps_combinatorics::design_create);

	if (f_v) {
		cout << "symbol_definition::definition_of_design "
				"before DC->init" << endl;
	}

	DC->init(
			Design_create_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_design "
				"after DC->init" << endl;
	}




	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_design(
			define_label, DC, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_design "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_design done" << endl;
	}
}



void symbol_definition::definition_of_design_table(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_design_table" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_design_table "
				"using existing design " << design_table_label_design << endl;
	}
	int idx;
	apps_combinatorics::design_create *DC;

	idx = Sym->Orbiter_top_level_session->find_symbol(design_table_label_design);
	DC = (apps_combinatorics::design_create *)
			Sym->Orbiter_top_level_session->get_object(idx);




	groups::any_group *AG;

	idx = other::orbiter_kernel_system::Orbiter->find_symbol(design_table_group);

	layer1_foundations::other::orbiter_kernel_system::symbol_table_object_type t;

	t = other::orbiter_kernel_system::Orbiter->get_object_type(idx);

	if (t != layer1_foundations::other::orbiter_kernel_system::symbol_table_object_type::t_any_group) {
		cout << "object must be of type group, but is ";
		other::orbiter_kernel_system::Orbiter->print_type(t);
		cout << endl;
		exit(1);
	}
	AG = (groups::any_group *)
		other::orbiter_kernel_system::Orbiter->get_object(idx);



	apps_combinatorics::combinatorics_global Combi;
	apps_combinatorics::design_tables *T;


	if (f_v) {
		cout << "symbol_definition::definition_of_design_table "
				"before Combi.create_design_table" << endl;
	}

	Combi.create_design_table(DC,
			design_table_label,
			T,
			AG->Subgroup_gens,
			verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_design_table "
				"after Combi.create_design_table" << endl;
	}



	apps_combinatorics::large_set_classify *LS;

	LS = NEW_OBJECT(apps_combinatorics::large_set_classify);

	if (f_v) {
		cout << "symbol_definition::definition_of_design_table "
				"before LS->init" << endl;
	}

	LS->init(DC,
			T,
			verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_design_table "
				"after LS->init" << endl;
	}



	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_design_table(
			define_label, LS, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_design_table "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_design_table done" << endl;
	}
}


void symbol_definition::definition_of_large_set_was(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_large_set_was" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_large_set_was "
				"using existing spread table "
				<< large_set_was_label_design_table << endl;
	}
	int idx;
	apps_combinatorics::large_set_classify *LS;

	idx = Sym->Orbiter_top_level_session->find_symbol(large_set_was_label_design_table);
	LS = (apps_combinatorics::large_set_classify *)
			Sym->Orbiter_top_level_session->get_object(idx);






	apps_combinatorics::large_set_was *LSW;

	LSW = NEW_OBJECT(apps_combinatorics::large_set_was);

	if (f_v) {
		cout << "symbol_definition::definition_of_large_set_was "
				"before LSW->init" << endl;
	}

	LSW->init(large_set_was_descr, LS, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_large_set_was "
				"after LSW->init" << endl;
	}




	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_large_set_was(
			define_label, LSW, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_large_set_was "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_large_set_was done" << endl;
	}
}

void symbol_definition::definition_of_set(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_set" << endl;
	}


	other::data_structures::set_builder *SB;

	SB = NEW_OBJECT(other::data_structures::set_builder);

	if (f_v) {
		cout << "symbol_definition::definition_of_set "
				"before SB->init" << endl;
	}

	SB->init(Set_builder_description, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_set "
				"after SB->init" << endl;
	}


	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_set(
			define_label, SB, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_set "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_set done" << endl;
	}
}

void symbol_definition::definition_of_vector(
		std::string &label,
		other::data_structures::vector_builder_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_vector" << endl;
	}


	algebra::field_theory::finite_field *F = NULL;

	if (Descr->f_field) {


		F = get_or_create_finite_field(
				Descr->field_label,
				verbose_level);

	}
	else {
		if (f_v) {
			cout << "symbol_definition::definition_of_vector "
					"not over a field" << endl;
		}

	}


	other::data_structures::vector_builder *VB;

	VB = NEW_OBJECT(other::data_structures::vector_builder);

	if (f_v) {
		cout << "symbol_definition::definition_of_vector "
				"before VB->init" << endl;
	}

	VB->init(Descr, F, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_vector "
				"after VB->init" << endl;
	}


	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_vector(
			label, VB, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_vector "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_vector done" << endl;
	}
}


void symbol_definition::definition_of_symbolic_object(
		std::string &label,
		algebra::expression_parser::symbolic_object_builder_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_symbolic_object" << endl;
	}



	algebra::expression_parser::symbolic_object_builder *SB;

	SB = NEW_OBJECT(algebra::expression_parser::symbolic_object_builder);

	if (f_v) {
		cout << "symbol_definition::definition_of_symbolic_object "
				"before SB->init" << endl;
	}

	SB->init(Descr, define_label, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_symbolic_object "
				"after SB->init" << endl;
	}


	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_symbolic_object(
			label, SB, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_symbolic_object "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_symbolic_object done" << endl;
	}
}


void symbol_definition::definition_of_combinatorial_object(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_combinatorial_object" << endl;
	}

	apps_combinatorics::combinatorial_object_stream *Combo;

	Combo = NEW_OBJECT(apps_combinatorics::combinatorial_object_stream);


	Combo->init(
			Data_input_stream_description,
			verbose_level);

	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_combinatorial_object(
			define_label, Combo, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_combinatorial_object "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_combinatorial_object done" << endl;
	}
}

void symbol_definition::do_geometry_builder(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::do_geometry_builder" << endl;
	}

	combinatorics::geometry_builder::geometry_builder *GB;

	GB = NEW_OBJECT(combinatorics::geometry_builder::geometry_builder);

	GB->init_description(Geometry_builder_description, verbose_level);

	GB->gg->main2(verbose_level);


	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_geometry_builder_object(
			define_label, GB, verbose_level);
	if (f_v) {
		cout << "symbol_definition::do_geometry_builder "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::do_geometry_builder done" << endl;
	}
}

void symbol_definition::load_finite_field_PG(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::load_finite_field_PG" << endl;
	}
	other::data_structures::string_tools ST;


	if (Projective_space_with_action_description->f_field_label) {
		if (f_v) {
			cout << "symbol_definition::load_finite_field_PG "
					"using existing finite field "
					<< Projective_space_with_action_description->field_label << endl;
		}
		Projective_space_with_action_description->F =
				Get_finite_field(
						Projective_space_with_action_description->field_label);
	}
	else if (Projective_space_with_action_description->f_q) {

		int q = Projective_space_with_action_description->q;

		if (f_v) {
			cout << "symbol_definition::load_finite_field_PG "
					"creating the finite field of order " << q << endl;
		}
		Projective_space_with_action_description->F = NEW_OBJECT(algebra::field_theory::finite_field);
		Projective_space_with_action_description->F->finite_field_init_small_order(q,
				false /* f_without_tables */,
				true /* f_compute_related_fields */,
				verbose_level - 1);
		if (f_v) {
			cout << "symbol_definition::load_finite_field_PG "
					"the finite field of order " << q
					<< " has been created" << endl;
		}

	}
	else {
		if (f_v) {
			cout << "symbol_definition::load_finite_field_PG "
					"please use one of -q <int> or -field <string>" << endl;
		}
		exit(1);
	}

	if (f_v) {
		cout << "symbol_definition::load_finite_field_PG done" << endl;
	}
}



algebra::field_theory::finite_field *symbol_definition::get_or_create_finite_field(
		std::string &input_q,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::get_or_create_finite_field" << endl;
	}
	algebra::field_theory::finite_field *F;
	other::data_structures::string_tools ST;

	if (ST.starts_with_a_number(input_q)) {
		int q;

		q = ST.strtoi(input_q);
		if (f_v) {
			cout << "symbol_definition::get_or_create_finite_field "
					"creating the finite field of order " << q << endl;
		}
		F = NEW_OBJECT(algebra::field_theory::finite_field);

		F->finite_field_init_small_order(q,
				false /* f_without_tables */,
				true /* f_compute_related_fields */,
				verbose_level - 1);

		if (f_v) {
			cout << "symbol_definition::get_or_create_finite_field "
					"the finite field of order " << q
					<< " has been created" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "symbol_definition::get_or_create_finite_field "
					"using existing finite field " << input_q << endl;
		}
		int idx;
		idx = Sym->Orbiter_top_level_session->find_symbol(input_q);
		if (idx < 0) {
			cout << "symbol_definition::get_or_create_finite_field "
					"done cannot find finite field object" << endl;
			exit(1);
		}
		F = (algebra::field_theory::finite_field *)
				Sym->Orbiter_top_level_session->get_object(idx);
	}

	if (f_v) {
		cout << "symbol_definition::get_or_create_finite_field done" << endl;
	}
	return F;
}


void symbol_definition::definition_of_vector_ge(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_vector_ge" << endl;
	}


	apps_algebra::vector_ge_builder *VB;

	VB = NEW_OBJECT(apps_algebra::vector_ge_builder);

	if (f_v) {
		cout << "symbol_definition::definition_of_vector_ge "
				"before VB->init" << endl;
	}

	VB->init(Vector_ge_description, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_vector_ge "
				"after VB->init" << endl;
	}


	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_vector_ge(
			define_label, VB, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_vector_ge "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_vector_ge done" << endl;
	}
}

void symbol_definition::definition_of_action_on_forms(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_action_on_forms" << endl;
	}


	apps_algebra::action_on_forms *AF;

	AF = NEW_OBJECT(apps_algebra::action_on_forms);

	if (f_v) {
		cout << "symbol_definition::definition_of_action_on_forms "
				"before AF->create_action_on_forms" << endl;
	}

	AF->create_action_on_forms(
			Action_on_forms_descr,
			verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_action_on_forms "
				"after AF->create_action_on_forms" << endl;
	}


	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_action_on_forms(
			define_label, AF, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_action_on_forms "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_action_on_forms done" << endl;
	}
}

void symbol_definition::definition_of_orbits(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_orbits" << endl;
	}


	orbits::orbits_create *OC;

	OC = NEW_OBJECT(orbits::orbits_create);

	if (f_v) {
		cout << "symbol_definition::definition_of_orbits "
				"before OC->init" << endl;
	}

	OC->init(
			Orbits_create_description,
			verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_orbits "
				"after OC->init" << endl;
	}


	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_orbits(
			define_label, OC, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_orbits "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_orbits done" << endl;
	}
}

void symbol_definition::definition_of_poset_classification_control(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_poset_classification_control" << endl;
	}




	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_poset_classification_control(
			define_label, Poset_classification_control, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_poset_classification_control "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_poset_classification_control done" << endl;
	}
}

void symbol_definition::definition_of_poset_classification_report_options(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_poset_classification_report_options" << endl;
	}




	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_poset_classification_report_options(
			define_label, Poset_classification_report_options, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_poset_classification_report_options "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_poset_classification_report_options done" << endl;
	}
}


void symbol_definition::definition_of_draw_options(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_draw_options" << endl;
	}




	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_draw_options(
			define_label, Draw_options, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_draw_options "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_draw_options done" << endl;
	}
}



void symbol_definition::definition_of_draw_incidence_structure_options(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_draw_incidence_structure_options" << endl;
	}




	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_draw_incidence_structure_options(
			define_label, Draw_incidence_structure_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_draw_incidence_structure_options "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_draw_incidence_structure_options done" << endl;
	}
}





void symbol_definition::definition_of_arc_generator_control(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_arc_generator_control" << endl;
	}




	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_arc_generator_control(
			define_label, Arc_generator_control, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_arc_generator_control "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_arc_generator_control done" << endl;
	}
}


void symbol_definition::definition_of_poset_classification_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_poset_classification_activity" << endl;
	}




	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_poset_classification_activity(
			define_label, Poset_classification_activity, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_poset_classification_activity "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_poset_classification_activity done" << endl;
	}
}

void symbol_definition::definition_of_crc_code(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_crc_code" << endl;
	}

	combinatorics::coding_theory::crc_object *Code;

	Code = NEW_OBJECT(combinatorics::coding_theory::crc_object);

	if (f_v) {
		cout << "symbol_definition::definition_of_crc_code "
				"before Code->init" << endl;
	}
	Code->init(Crc_code_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_crc_code "
				"after Code->init" << endl;
	}
	if (f_v) {
		cout << "Code->label_txt" << Code->label_txt << endl;
	}



	if (f_v) {
		cout << "symbol_definition::definition_of_crc_code "
				"we have created a code called " << Code->label_txt << endl;

	}


	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_crc_code(
			define_label, Code, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_crc_code "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_crc_code done" << endl;
	}
}


void symbol_definition::definition_of_mapping(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_mapping" << endl;
	}

	apps_geometry::mapping *Mapping;

	Mapping = NEW_OBJECT(apps_geometry::mapping);

	if (f_v) {
		cout << "symbol_definition::definition_of_mapping "
				"before Mapping->init" << endl;
	}
	Mapping->init(Mapping_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_mapping "
				"after Mapping->init" << endl;
	}
	if (f_v) {
		cout << "Mapping->label_txt" << Mapping->label_txt << endl;
	}



	if (f_v) {
		cout << "symbol_definition::definition_of_mapping "
				"we have created a mapping called " << Mapping->label_txt << endl;

	}


	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_mapping(
			define_label, Mapping, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_mapping "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_mapping done" << endl;
	}
}


void symbol_definition::definition_of_variety(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_variety" << endl;
	}

	int cnt = 0;
	int po_go = 0;
	int po_index = 0;
	int po = 0;
	int so = 0;


	projective_geometry::projective_space_with_action *PA;

	if (f_v) {
		cout << "symbol_definition::definition_of_variety before getting PA" << endl;
	}
	if (Variety_description->f_projective_space) {
		PA = Get_projective_space(Variety_description->projective_space_label);
	}
	else {
		cout << "symbol_definition::definition_of_variety we don't have a projective space" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "symbol_definition::definition_of_variety after getting PA" << endl;
	}


	canonical_form::variety_object_with_action *Variety;

	Variety = NEW_OBJECT(canonical_form::variety_object_with_action);

	if (f_v) {
		cout << "symbol_definition::definition_of_variety "
				"before Variety->create_variety" << endl;
	}
	Variety->create_variety(
			PA, cnt, po_go, po_index, po, so, Variety_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_variety "
				"after Variety->create_variety" << endl;
	}
	if (f_v) {
		cout << "Variety->Variety_object->label_txt"
				<< Variety->Variety_object->label_txt << endl;
	}



	if (f_v) {
		cout << "symbol_definition::definition_of_variety "
				"we have created a variety called "
				<< Variety->Variety_object->label_txt << endl;

	}


	other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_variety(
			define_label, Variety, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_variety "
				"before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_variety done" << endl;
	}
}





}}}




