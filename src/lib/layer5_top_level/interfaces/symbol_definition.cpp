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
	Sym = NULL;

	//std::string define_label;

	f_finite_field = FALSE;
	Finite_field_description = NULL;

	f_polynomial_ring = FALSE;
	Polynomial_ring_description = NULL;

	f_projective_space = FALSE;
	Projective_space_with_action_description = NULL;

	f_orthogonal_space = FALSE;
	Orthogonal_space_with_action_description = NULL;

	f_BLT_set_classifier = FALSE;
	//std::string BLT_set_classifier_label_orthogonal_geometry;
	Blt_set_classify_description = NULL;

	f_spread_classifier = FALSE;
	Spread_classify_description = NULL;


	f_linear_group = FALSE;
	Linear_group_description = NULL;

	f_permutation_group = FALSE;
	Permutation_group_description = NULL;

	f_group_modification = FALSE;
	Group_modification_description = NULL;

	f_formula = FALSE;
	Formula = NULL;
	//std::string label;
	//std::string label_tex;
	//std::string managed_variables;
	//std::string formula_text;

	f_collection = FALSE;
	//std::string list_of_objects;

	f_geometric_object = FALSE;
	//std::string geometric_object_projective_space_label;
	Geometric_object_description = NULL;

	f_graph = FALSE;
	Create_graph_description = NULL;

	f_code = FALSE;
	Create_code_description = NULL;

	f_spread = FALSE;
	Spread_create_description = NULL;

	f_cubic_surface = FALSE;
	Surface_Descr = NULL;

	f_quartic_curve = FALSE;
	Quartic_curve_descr = NULL;


	f_translation_plane = FALSE;
	//std::string translation_plane_spread_label;
	//std::string translation_plane_group_n_label;
	//std::string translation_plane_group_np1_label;


	f_spread_table = FALSE;
	//std::string spread_table_label_PA;
	dimension_of_spread_elements = 0;
	//std::string spread_selection_text;
	//std::string spread_tables_prefix;


	f_packing_was = FALSE;
	//std::string packing_was_label_spread_table;
	packing_was_descr = NULL;

	f_packing_was_choose_fixed_points = FALSE;
	//std::string packing_with_assumed_symmetry_label;
	packing_with_assumed_symmetry_choose_fixed_points_clique_size = 0;
	packing_with_assumed_symmetry_choose_fixed_points_control = NULL;


	f_packing_long_orbits = FALSE;
	//std::string packing_long_orbits_choose_fixed_points_label
	Packing_long_orbits_description = NULL;

	f_graph_classification = FALSE;
	Graph_classify_description = NULL;

	f_diophant = FALSE;
	Diophant_description = NULL;

	f_design = FALSE;
	Design_create_description = NULL;


	f_design_table = FALSE;
	//std::string design_table_label_design;
	//std::string design_table_label;
	//std::string design_table_group;


	f_large_set_was = FALSE;
	//std::string  large_set_was_label_design_table;
	large_set_was_descr = NULL;


	f_set = FALSE;
	Set_builder_description = FALSE;

	f_vector = FALSE;
	Vector_builder_description = FALSE;

	f_combinatorial_objects = FALSE;
	Data_input_stream_description = FALSE;

	f_geometry_builder = FALSE;
	Geometry_builder_description = NULL;

	f_vector_ge = FALSE;
	Vector_ge_description = NULL;

	f_action_on_forms = FALSE;
	Action_on_forms_descr = NULL;

	f_orbits = FALSE;
	Orbits_create_description = NULL;

	f_poset_classification_control = FALSE;
	Poset_classification_control = NULL;

}


symbol_definition::~symbol_definition()
{

}

void symbol_definition::read_definition(
		interface_symbol_table *Sym,
		int argc, std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "symbol_definition::read_definition i=" << i << " argc=" << argc << endl;
	}

	symbol_definition::Sym = Sym;

	//f_define = TRUE;
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
		f_finite_field = TRUE;
		Finite_field_description = NEW_OBJECT(field_theory::finite_field_description);
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
		f_polynomial_ring = TRUE;
		Polynomial_ring_description = NEW_OBJECT(ring_theory::polynomial_ring_description);
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
		f_projective_space = TRUE;
		Projective_space_with_action_description = NEW_OBJECT(projective_geometry::projective_space_with_action_description);
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
		f_orthogonal_space = TRUE;
		Orthogonal_space_with_action_description = NEW_OBJECT(orthogonal_geometry_applications::orthogonal_space_with_action_description);
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
		f_BLT_set_classifier = TRUE;
		BLT_set_classifier_label_orthogonal_geometry.assign(argv[++i]);
		Blt_set_classify_description = NEW_OBJECT(orthogonal_geometry_applications::blt_set_classify_description);
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
		f_spread_classifier = TRUE;
		Spread_classify_description = NEW_OBJECT(spreads::spread_classify_description);
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
		f_linear_group = TRUE;
		Linear_group_description = NEW_OBJECT(groups::linear_group_description);
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
		f_permutation_group = TRUE;
		Permutation_group_description = NEW_OBJECT(groups::permutation_group_description);
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
		f_group_modification = TRUE;
		Group_modification_description = NEW_OBJECT(apps_algebra::group_modification_description);
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

	else if (ST.stringcmp(argv[i], "-formula") == 0) {
		if (f_v) {
			cout << "-formula" << endl;
		}
		f_formula = TRUE;

		label.assign(argv[++i]);
		label_tex.assign(argv[++i]);
		managed_variables.assign(argv[++i]);
		formula_text.assign(argv[++i]);

		i++;



		Formula = NEW_OBJECT(expression_parser::formula);
		Formula->init(label, label_tex, managed_variables, formula_text, verbose_level);

	}

	else if (ST.stringcmp(argv[i], "-geometric_object") == 0) {
		f_geometric_object = TRUE;

		geometric_object_projective_space_label.assign(argv[++i]);
		Geometric_object_description = NEW_OBJECT(geometry::geometric_object_description);
		if (f_v) {
			cout << "reading -geometric_object" << endl;
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
			cout << "-collection" << endl;
		}

		f_collection = TRUE;

		list_of_objects.assign(argv[++i]);
		i++;

	}
	else if (ST.stringcmp(argv[i], "-graph") == 0) {

		f_graph = TRUE;
		Create_graph_description = NEW_OBJECT(apps_graph_theory::create_graph_description);
		if (f_v) {
			cout << "reading -graph" << endl;
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

		f_code = TRUE;
		Create_code_description = NEW_OBJECT(apps_coding_theory::create_code_description);
		if (f_v) {
			cout << "reading -code" << endl;
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

		f_spread = TRUE;
		Spread_create_description = NEW_OBJECT(spreads::spread_create_description);
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

		f_cubic_surface = TRUE;
		Surface_Descr = NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description);
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

		f_quartic_curve = TRUE;
		Quartic_curve_descr = NEW_OBJECT(applications_in_algebraic_geometry::quartic_curves::quartic_curve_create_description);
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


	else if (ST.stringcmp(argv[i], "-translation_plane") == 0) {
		f_translation_plane = TRUE;
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
		f_spread_table = TRUE;

		spread_table_label_PA.assign(argv[++i]);
		dimension_of_spread_elements = ST.strtoi(argv[++i]);
		spread_selection_text.assign(argv[++i]);
		spread_tables_prefix.assign(argv[++i]);

		i++;

		if (f_v) {
			cout << "dimension_of_spread_elements = " << dimension_of_spread_elements
					<< " " << spread_selection_text
					<< " " << spread_tables_prefix << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}


		if (f_v) {
			cout << "-spread_table " << spread_table_label_PA
					<< " " << dimension_of_spread_elements
					<< " " << spread_selection_text
					<< " " << spread_tables_prefix
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-packing_with_symmetry_assumption") == 0) {
		f_packing_was = TRUE;

		packing_was_label_spread_table.assign(argv[++i]);

		packing_was_descr = NEW_OBJECT(packings::packing_was_description);
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
			cout << "-packing_with_symmetry_assumption " << packing_was_label_spread_table
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-packing_choose_fixed_points") == 0) {
		f_packing_was_choose_fixed_points = TRUE;

		packing_with_assumed_symmetry_label.assign(argv[++i]);
		packing_with_assumed_symmetry_choose_fixed_points_clique_size = ST.strtoi(argv[++i]);

		packing_with_assumed_symmetry_choose_fixed_points_control = NEW_OBJECT(poset_classification::poset_classification_control);
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
		f_packing_long_orbits = TRUE;

		packing_long_orbits_choose_fixed_points_label.assign(argv[++i]);

		Packing_long_orbits_description = NEW_OBJECT(packings::packing_long_orbits_description);
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
		f_graph_classification = TRUE;

		Graph_classify_description = NEW_OBJECT(apps_graph_theory::graph_classify_description);
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
		f_diophant = TRUE;

		Diophant_description = NEW_OBJECT(solvers::diophant_description);
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

		f_design = TRUE;
		Design_create_description = NEW_OBJECT(apps_combinatorics::design_create_description);
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
		f_design_table = TRUE;

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
		f_large_set_was = TRUE;

		large_set_was_label_design_table.assign(argv[++i]);

		large_set_was_descr = NEW_OBJECT(apps_combinatorics::large_set_was_description);
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
		f_set = TRUE;


		Set_builder_description = NEW_OBJECT(data_structures::set_builder_description);
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
			cout << "-set ";
			Set_builder_description->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-vector") == 0) {
		f_vector = TRUE;


		Vector_builder_description = NEW_OBJECT(data_structures::vector_builder_description);
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
			cout << "-vector ";
			Vector_builder_description->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-combinatorial_objects") == 0) {
		f_combinatorial_objects = TRUE;


		Data_input_stream_description = NEW_OBJECT(data_structures::data_input_stream_description);
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
			cout << "-combinatorial_objects ";
			Data_input_stream_description->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-geometry_builder") == 0) {
		f_geometry_builder = TRUE;


		Geometry_builder_description = NEW_OBJECT(geometry_builder::geometry_builder_description);
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
			cout << "-geometry_builder ";
			Geometry_builder_description->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-vector_ge") == 0) {
		f_vector_ge = TRUE;


		Vector_ge_description = NEW_OBJECT(data_structures_groups::vector_ge_description);
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
			cout << "-vector_ge ";
			Vector_ge_description->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-action_on_forms") == 0) {
		f_action_on_forms = TRUE;

		Action_on_forms_descr = NEW_OBJECT(apps_algebra::action_on_forms_description);
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
			cout << "-action_on_forms ";
			Action_on_forms_descr->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-orbits") == 0) {
		f_orbits = TRUE;

		Orbits_create_description = NEW_OBJECT(apps_algebra::orbits_create_description);
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
			cout << "-orbits ";
			Orbits_create_description->print();
		}
	}
	else if (ST.stringcmp(argv[i], "-poset_classification_control") == 0) {
		f_poset_classification_control = TRUE;

		Poset_classification_control = NEW_OBJECT(poset_classification::poset_classification_control);
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
			cout << "-poset_classification_control ";
			Poset_classification_control->print();
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


void symbol_definition::perform_definition(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::perform_definition" << endl;
	}

	if (f_finite_field) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_finite_field" << endl;
		}
		definition_of_finite_field(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_finite_field" << endl;
		}
	}
	else if (f_polynomial_ring) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_polynomial_ring" << endl;
		}
		definition_of_polynomial_ring(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_polynomial_ring" << endl;
		}
	}
	else if (f_projective_space) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_projective_space" << endl;
		}
		definition_of_projective_space(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_projective_space" << endl;
		}
	}
	else if (f_orthogonal_space) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_orthogonal_space" << endl;
		}
		definition_of_orthogonal_space(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_orthogonal_space" << endl;
		}
	}
	else if (f_BLT_set_classifier) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_BLT_set_classifier" << endl;
		}
		definition_of_BLT_set_classifier(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_BLT_set_classifier" << endl;
		}
	}
	else if (f_spread_classifier) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_spread_classifier" << endl;
		}
		definition_of_spread_classifier(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_spread_classifier" << endl;
		}
	}

	else if (f_linear_group) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_linear_group" << endl;
		}
		definition_of_linear_group(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_linear_group" << endl;
		}
	}
	else if (f_permutation_group) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_permutation_group" << endl;
		}
		definition_of_permutation_group(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_permutation_group" << endl;
		}
	}
	else if (f_group_modification) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_modified_group" << endl;
		}
		definition_of_modified_group(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_modified_group" << endl;
		}
	}


	else if (f_formula) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_formula" << endl;
		}
		definition_of_formula(Formula, verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_formula" << endl;
		}
	}

	else if (f_geometric_object) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_geometric_object" << endl;
		}
		definition_of_geometric_object(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_geometric_object" << endl;
		}
	}
	else if (f_collection) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_collection" << endl;
		}
		definition_of_collection(list_of_objects, verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_collection" << endl;
		}
	}
	else if (f_graph) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_graph" << endl;
		}
		definition_of_graph(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_graph" << endl;
		}
	}
	else if (f_code) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_code" << endl;
		}
		definition_of_code(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_code" << endl;
		}
	}
	else if (f_spread) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_spread" << endl;
		}
		definition_of_spread(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_spread" << endl;
		}
	}
	else if (f_cubic_surface) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_cubic_surface" << endl;
		}
		definition_of_cubic_surface(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_cubic_surface" << endl;
		}
	}
	else if (f_quartic_curve) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_quartic_curve" << endl;
		}
		definition_of_quartic_curve(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_quartic_curve" << endl;
		}
	}
	else if (f_translation_plane) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_translation_plane" << endl;
		}
		definition_of_translation_plane(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_translation_plane" << endl;
		}
	}
	else if (f_spread_table) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_projective_space" << endl;
		}
		definition_of_spread_table(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_projective_space" << endl;
		}
	}
	else if (f_packing_was) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_packing_was" << endl;
		}
		definition_of_packing_was(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_packing_was" << endl;
		}
	}
	else if (f_packing_was_choose_fixed_points) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_packing_was_choose_fixed_points" << endl;
		}
		definition_of_packing_was_choose_fixed_points(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_packing_was_choose_fixed_points" << endl;
		}
	}
	else if (f_packing_long_orbits) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_packing_long_orbits" << endl;
		}
		definition_of_packing_long_orbits(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_packing_long_orbits" << endl;
		}
	}
	else if (f_graph_classification) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_graph_classification" << endl;
		}
		definition_of_graph_classification(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_graph_classification" << endl;
		}
	}
	else if (f_diophant) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_graph_classification" << endl;
		}
		definition_of_diophant(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_graph_classification" << endl;
		}
	}
	else if (f_design) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_design" << endl;
		}
		definition_of_design(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_design" << endl;
		}
	}
	else if (f_design_table) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_design_table" << endl;
		}
		definition_of_design_table(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_design_table" << endl;
		}
	}
	else if (f_large_set_was) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_packing_was" << endl;
		}
		definition_of_large_set_was(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_packing_was" << endl;
		}
	}
	else if (f_set) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_set" << endl;
		}
		definition_of_set(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_set" << endl;
		}
	}
	else if (f_vector) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_vector" << endl;
		}
		definition_of_vector(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_vector" << endl;
		}
	}
	else if (f_combinatorial_objects) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_combinatorial_object" << endl;
		}
		definition_of_combinatorial_object(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_combinatorial_object" << endl;
		}
	}
	else if (f_geometry_builder) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before do_geometry_builder" << endl;
		}
		do_geometry_builder(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after do_geometry_builder" << endl;
		}
	}
	else if (f_vector_ge) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_vector_ge" << endl;
		}
		definition_of_vector_ge(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_vector_ge" << endl;
		}
	}
	else if (f_action_on_forms) {
		if (f_v) {
			cout << "symbol_definition::perform_definition f_action_on_forms" << endl;
		}
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_action_on_forms" << endl;
		}
		definition_of_action_on_forms(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_action_on_forms" << endl;
		}
	}
	else if (f_orbits) {
		if (f_v) {
			cout << "symbol_definition::perform_definition f_orbits" << endl;
		}
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_orbits" << endl;
		}
		definition_of_orbits(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_orbits" << endl;
		}
	}
	else if (f_poset_classification_control) {
		if (f_v) {
			cout << "symbol_definition::perform_definition f_poset_classification_control" << endl;
		}
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_poset_classification_control" << endl;
		}
		definition_of_poset_classification_control(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_poset_classification_control" << endl;
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
	cout << "-define " << define_label << " ";
	if (f_finite_field) {
		cout << "-finite_field ";
		Finite_field_description->print();
	}
	if (f_polynomial_ring) {
		cout << "-polynomial_ring ";
		Polynomial_ring_description->print();
	}
	if (f_projective_space) {
		cout << "-projective_space ";
		Projective_space_with_action_description->print();
	}
	if (f_orthogonal_space) {
		cout << "-orthogonal_space ";
		Orthogonal_space_with_action_description->print();
	}
	if (f_BLT_set_classifier) {
		cout << "-BLT_set_classifier " << BLT_set_classifier_label_orthogonal_geometry << " ";
		Blt_set_classify_description->print();
	}
	if (f_spread_classifier) {
		cout << "-spread_classifier" << endl;
		Spread_classify_description->print();
	}
	if (f_linear_group) {
		cout << "-linear_group ";
		Linear_group_description->print();
	}
	if (f_permutation_group) {
		cout << "-permutation_group ";
		Permutation_group_description->print();
	}
	if (f_group_modification) {
		cout << "-modified_group ";
		Group_modification_description->print();
	}
	if (f_formula) {
		cout << "-formula " << label << " " << label_tex << " " << managed_variables << " " << formula_text;
		//formula *F;
		//std::string label;
		//std::string label_tex;
		//std::string managed_variables;
		//std::string formula_text;
	}
	if (f_geometric_object) {
		cout << "-geometric_object ";
		Geometric_object_description->print();
	}
	if (f_collection) {
		cout << "-collection ";
		//cout << list_of_objects << endl;
	}
	if (f_graph) {
		cout << "-graph ";
		Create_graph_description->print();
	}
	if (f_code) {
		cout << "-code ";
		Create_code_description->print();
	}
	if (f_spread) {
		cout << "-spread ";
		Spread_create_description->print();
	}
	if (f_cubic_surface) {
		cout << "-cubic_surface " << endl;
		Surface_Descr->print();
	}
	if (f_quartic_curve) {
		cout << "-quartic_curve " << endl;
		Quartic_curve_descr->print();
	}
	if (f_translation_plane) {
		cout << "-translation_plane "
				<< " " << translation_plane_spread_label
				<< " " << translation_plane_group_n_label
				<< " " << translation_plane_group_np1_label
				<< endl;
	}
	if (f_spread_table) {
		cout << "-spread_table ";
		cout << spread_table_label_PA << " " << dimension_of_spread_elements << " " << spread_selection_text << " " << spread_tables_prefix << endl;
	}
	if (f_packing_was) {
		cout << "-packing_was " << packing_was_label_spread_table << endl;
		packing_was_descr->print();
	}
	if (f_packing_was_choose_fixed_points) {
		cout << "-packing_was_choose_fixed_points ";
		cout << packing_with_assumed_symmetry_label;
		cout << " " << packing_with_assumed_symmetry_choose_fixed_points_clique_size << " " << endl;
		packing_with_assumed_symmetry_choose_fixed_points_control->print();
		//std::string packing_with_assumed_symmetry_label;
		//int packing_with_assumed_symmetry_choose_fixed_points_clique_size;
		//poset_classification_control *packing_with_assumed_symmetry_choose_fixed_points_control;
	}
	if (f_packing_long_orbits) {
		cout << "-packing_long_orbits " << packing_long_orbits_choose_fixed_points_label << endl;
		Packing_long_orbits_description->print();
	}
	if (f_graph_classification) {
		cout << "-graph_classification ";
		Graph_classify_description->print();
	}
	if (f_diophant) {
		cout << "-diophant ";
		Diophant_description->print();
	}
	if (f_design) {
		cout << "-design ";
		Design_create_description->print();
	}
	if (f_design_table) {
		cout << "-design_table "
				<< design_table_label_design
				<< " " << design_table_label
				<< " " << design_table_group << endl;
	}
	if (f_large_set_was) {
		cout << "-large_set_was " << large_set_was_label_design_table << endl;
		large_set_was_descr->print();
	}
	if (f_set) {
		cout << "-set ";
		Set_builder_description->print();
	}
	if (f_vector) {
		cout << "-vector ";
		Vector_builder_description->print();
	}
	if (f_combinatorial_objects) {
		cout << "-combinatorial_objects ";
		Data_input_stream_description->print();
	}
	if (f_geometry_builder) {
		cout << "-geometry_builder ";
		Geometry_builder_description->print();
	}
	if (f_vector_ge) {
		cout << "-vector_g ";
		Vector_ge_description->print();
	}
	if (f_action_on_forms) {
		cout << "-action_on_forms ";
		Action_on_forms_descr->print();
	}
	if (f_orbits) {
		cout << "-orbits ";
		Orbits_create_description->print();
	}
	if (f_poset_classification_control) {
		cout << "-poset_classification_control ";
		Poset_classification_control->print();
	}
}





void symbol_definition::definition_of_finite_field(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_finite_field" << endl;
	}
	Finite_field_description->print();
	field_theory::finite_field *F;

	F = NEW_OBJECT(field_theory::finite_field);
	if (f_v) {
		cout << "symbol_definition::definition_of_finite_field before F->init" << endl;
	}
	F->init(Finite_field_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_finite_field after F->init" << endl;
	}

	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_finite_field(define_label, F, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_finite_field before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_finite_field done" << endl;
	}
}

void symbol_definition::definition_of_polynomial_ring(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_polynomial_ring" << endl;
	}
	Polynomial_ring_description->print();
	ring_theory::homogeneous_polynomial_domain *HPD;

	HPD = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "symbol_definition::definition_of_polynomial_ring before HPD->init" << endl;
	}
	HPD->init(Polynomial_ring_description, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_polynomial_ring after F->init" << endl;
	}

	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_polynomial_ring(define_label, HPD, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_polynomial_ring before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_polynomial_ring done" << endl;
	}
}



void symbol_definition::definition_of_projective_space(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space, verbose_level=" << verbose_level << endl;
	}


	if (Projective_space_with_action_description->f_override_verbose_level) {
		verbose_level = Projective_space_with_action_description->override_verbose_level;
	}
	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space before load_finite_field_PG" << endl;
	}
	load_finite_field_PG(verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space after load_finite_field_PG" << endl;
	}


	int f_semilinear;
	number_theory::number_theory_domain NT;


	if (NT.is_prime(Projective_space_with_action_description->F->q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}

	if (Projective_space_with_action_description->f_use_projectivity_subgroup) {
		f_semilinear = FALSE;
	}

	projective_geometry::projective_space_with_action *PA;

	PA = NEW_OBJECT(projective_geometry::projective_space_with_action);

	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space before PA->init" << endl;
	}
	PA->init(Projective_space_with_action_description->F,
			Projective_space_with_action_description->n,
		f_semilinear,
		TRUE /*f_init_incidence_structure*/,
		verbose_level - 2);
	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space after PA->init" << endl;
	}

	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_projective_space(define_label, PA, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space done" << endl;
	}
}

void symbol_definition::print_definition_of_projective_space(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::print_definition_of_projective_space" << endl;
	}
	Projective_space_with_action_description->print();
}

void symbol_definition::definition_of_orthogonal_space(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_orthogonal_space" << endl;
	}


	load_finite_field(
			Orthogonal_space_with_action_description->input_q,
			Orthogonal_space_with_action_description->F,
			verbose_level);


	//int f_semilinear;
	number_theory::number_theory_domain NT;

#if 0
	if (NT.is_prime(Orthogonal_space_with_action_description->F->q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}
#endif

	orthogonal_geometry_applications::orthogonal_space_with_action *OA;

	OA = NEW_OBJECT(orthogonal_geometry_applications::orthogonal_space_with_action);

	if (f_v) {
		cout << "symbol_definition::definition_of_orthogonal_space before OA->init" << endl;
	}
	OA->init(Orthogonal_space_with_action_description,
		verbose_level - 1);
	if (f_v) {
		cout << "symbol_definition::definition_of_orthogonal_space after OA->init" << endl;
	}

	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_orthogonal_space(define_label, OA, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_orthogonal_space before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_orthogonal_space done" << endl;
	}
}


void symbol_definition::definition_of_BLT_set_classifier(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set_classifier" << endl;
	}

	//std::string BLT_set_classifier_label_orthogonal_geometry;
	//orthogonal_geometry_applications::blt_set_classify_description *Blt_set_classify_description;


	orthogonal_geometry_applications::orthogonal_space_with_action *OA;

	OA = The_Orbiter_top_level_session->get_object_of_type_orthogonal_space_with_action(BLT_set_classifier_label_orthogonal_geometry);

	orthogonal_geometry_applications::blt_set_classify *BLT_classify;

	BLT_classify = NEW_OBJECT(orthogonal_geometry_applications::blt_set_classify);

#if 0
	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set_classifier before BLT_classify->init" << endl;
	}
	BLT_classify->init(Orthogonal_space_with_action_description,
		verbose_level - 1);
	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set_classifier after BLT_classify->init" << endl;
	}
#endif

	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set_classifier before BLT_classify->init_basic" << endl;
	}

	if (!Blt_set_classify_description->f_starter_size) {
		cout << "please use option -starter_size <s>" << endl;
		exit(1);
	}
	BLT_classify->init_basic(
			OA,
			OA->A,
			OA->A->Strong_gens,
			Blt_set_classify_description->starter_size,
			verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set_classifier after BLT_classify->init_basic" << endl;
	}



	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_BLT_set_classify(define_label, BLT_classify, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set_classifier before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_BLT_set_classifier done" << endl;
	}
}

void symbol_definition::definition_of_spread_classifier(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_spread_classifier" << endl;
	}


	spreads::spread_classify *Spread_classify;

	Spread_classify = NEW_OBJECT(spreads::spread_classify);

	if (f_v) {
		cout << "symbol_definition::definition_of_spread_classifier before Spread_classify->init_basic" << endl;
	}
	Spread_classify->init_basic(
			Spread_classify_description,
			verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_spread_classifier after Spread_classify->init_basic" << endl;
	}

	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_spread_classify(define_label, Spread_classify, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_spread_classifier before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_spread_classifier done" << endl;
	}
}


void symbol_definition::definition_of_linear_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_linear_group" << endl;
	}

#if 0
	if (f_v) {
		cout << "symbol_definition::definition_of_linear_group before load_finite_field" << endl;
	}
	load_finite_field(Linear_group_description->input_q,
			Linear_group_description->F, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_linear_group after load_finite_field" << endl;
	}
#endif


	groups::linear_group *LG;

	LG = NEW_OBJECT(groups::linear_group);
	if (f_v) {
		cout << "symbol_definition::definition before LG->linear_group_init, "
				"creating the group" << endl;
	}

	LG->linear_group_init(Linear_group_description, verbose_level - 2);

	if (f_v) {
		cout << "symbol_definition::definition after LG->linear_group_init" << endl;
	}


	// create any_group object from linear_group:


	apps_algebra::any_group *AG;

	AG = NEW_OBJECT(apps_algebra::any_group);
	if (f_v) {
		cout << "symbol_definition::definition before AG->init_linear_group" << endl;
	}
	AG->init_linear_group(LG, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition after AG->init_linear_group" << endl;
	}




	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_any_group(define_label, AG, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_linear_group done" << endl;
	}
}

void symbol_definition::definition_of_permutation_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_permutation_group" << endl;
	}


	groups::permutation_group_create *PGC;

	PGC = NEW_OBJECT(groups::permutation_group_create);
	if (f_v) {
		cout << "symbol_definition::definition_of_permutation_group before PGC->permutation_group_init, "
				"before PGC->permutation_group_init" << endl;
	}

	PGC->permutation_group_init(Permutation_group_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_permutation_group after PGC->permutation_group_init, "
				"after PGC->permutation_group_init" << endl;
	}


	// create any_group object from permutation_group_create:


	apps_algebra::any_group *AG;

	AG = NEW_OBJECT(apps_algebra::any_group);
	AG->init_permutation_group(PGC, verbose_level);



	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_any_group(define_label, AG, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_permutation_group before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_permutation_group done" << endl;
	}
}


void symbol_definition::definition_of_modified_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_modified_group" << endl;
	}


	apps_algebra::modified_group_create *MGC;

	MGC = NEW_OBJECT(apps_algebra::modified_group_create);
	if (f_v) {
		cout << "symbol_definition::definition_of_modified_group before PGC->permutation_group_init, "
				"before PGC->permutation_group_init" << endl;
	}

	MGC->modified_group_init(Group_modification_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_modified_group after PGC->permutation_group_init, "
				"after PGC->permutation_group_init" << endl;
	}

	apps_algebra::any_group *AG;

	AG = NEW_OBJECT(apps_algebra::any_group);
	AG->init_modified_group(MGC, verbose_level);

	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);

	Symb->init_any_group(define_label, AG, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_modified_group before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_modified_group done" << endl;
	}
}

void symbol_definition::definition_of_geometric_object(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_geometric_object" << endl;
	}


	//geometry::geometric_object_create *GOC;

	//GOC = NEW_OBJECT(geometry::geometric_object_create);


	projective_geometry::projective_space_with_action *PA;

	PA = The_Orbiter_top_level_session->get_object_of_type_projective_space(geometric_object_projective_space_label);

	geometry::geometric_object_create *GeoObj;

	GeoObj = NEW_OBJECT(geometry::geometric_object_create);

	if (f_v) {
		cout << "symbol_definition::definition_of_geometric_object before GeoObj->init" << endl;
	}

	GeoObj->init(Geometric_object_description, PA->P, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_geometric_object after GeoObj->init" << endl;
	}

	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);



	Symb->init_geometric_object(define_label, GeoObj, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_geometric_object before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_geometric_object done" << endl;
	}
}






void symbol_definition::definition_of_formula(
		expression_parser::formula *Formula,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_formula" << endl;
	}

	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);

	Symb->init_formula(define_label, Formula, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_formula before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_formula done" << endl;
	}
}

void symbol_definition::definition_of_collection(std::string &list_of_objects,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_collection" << endl;
	}

	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_collection(define_label, list_of_objects, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_formula before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_collection done" << endl;
	}
}

void symbol_definition::definition_of_graph(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_graph" << endl;
	}

	apps_graph_theory::create_graph *Gr;

	Gr = NEW_OBJECT(apps_graph_theory::create_graph);

	if (f_v) {
		cout << "symbol_definition::definition_of_graph before Gr->init" << endl;
	}
	Gr->init(Create_graph_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_graph after Gr->init" << endl;
	}
	if (f_v) {
		cout << "Gr->N=" << Gr->N << endl;
		cout << "Gr->label=" << Gr->label << endl;
		cout << "Gr->f_has_CG=" << Gr->f_has_CG << endl;
		//cout << "Adj:" << endl;
		//int_matrix_print(Gr->Adj, Gr->N, Gr->N);
	}



	if (f_v) {
		cout << "symbol_definition::definition_of_graph we created a graph on " << Gr->N
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


	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_graph(define_label, Gr->CG, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_graph before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_graph done" << endl;
	}
}


void symbol_definition::definition_of_code(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_code" << endl;
	}

	apps_coding_theory::create_code *Code;

	Code = NEW_OBJECT(apps_coding_theory::create_code);

	if (f_v) {
		cout << "symbol_definition::definition_of_code before Code->init" << endl;
	}
	Code->init(Create_code_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_code after Code->init" << endl;
	}
	if (f_v) {
		cout << "Code->label_txt" << Code->label_txt << endl;
	}



	if (f_v) {
		cout << "symbol_definition::definition_of_code we created a code called " << Code->label_txt << endl;

	}


	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_code(define_label, Code, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_code before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_code done" << endl;
	}
}


void symbol_definition::definition_of_spread(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_spread" << endl;
	}


	spreads::spread_create *Spread;


	Spread = NEW_OBJECT(spreads::spread_create);

	if (f_v) {
		cout << "symbol_definition::definition_of_spread before Spread->init" << endl;
	}
	Spread->init(Spread_create_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_spread after Spread->init" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_spread we created a spread called " << Spread->label_txt << endl;

	}


	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_spread(define_label, Spread, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_spread before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_spread done" << endl;
	}
}


void symbol_definition::definition_of_cubic_surface(int verbose_level)
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
		cout << "symbol_definition::definition_of_cubic_surface we created a cubic surface called " << SC->label_txt << endl;

	}


	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_cubic_surface(define_label, SC, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_cubic_surface before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_cubic_surface done" << endl;
	}
}



void symbol_definition::definition_of_quartic_curve(int verbose_level)
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
		cout << "symbol_definition::definition_of_quartic_curve we created a quartic curve called " << QC->label_txt << endl;

	}


	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_quartic_curve(define_label, QC, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_quartic_curve before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_quartic_curve done" << endl;
	}
}




void symbol_definition::definition_of_translation_plane(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_translation_plane" << endl;
	}

	//translation_plane_spread_label
	//translation_plane_group_n_label
	//translation_plane_group_np1_label


	spreads::spread_create *Spread;
	apps_algebra::any_group *Gn;
	apps_algebra::any_group *Gnp1;
	data_structures_groups::translation_plane_via_andre_model *TP;


	Spread = Get_object_of_type_spread(translation_plane_spread_label);

	if (f_v) {
		cout << "symbol_definition::definition_of_translation_plane found spread " << Spread->label_txt << endl;
	}

	Gn = Get_object_of_type_any_group(translation_plane_group_n_label);

	if (f_v) {
		cout << "symbol_definition::definition_of_translation_plane found group Gn " << Gn->label << endl;
	}

	Gnp1 = Get_object_of_type_any_group(translation_plane_group_np1_label);

	if (f_v) {
		cout << "symbol_definition::definition_of_translation_plane found group Gnp1 " << Gnp1->label << endl;
	}

	TP = NEW_OBJECT(data_structures_groups::translation_plane_via_andre_model);

	actions::action *An1;

	An1 = Gnp1->A_base;

	if (f_v) {
		cout << "symbol_definition::definition_of_translation_plane before TP->init" << endl;
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
		cout << "symbol_definition::definition_of_translation_plane after TP->init" << endl;
		cout << "symbol_definition::definition_of_translation_plane TP->label_txt=" << TP->label_txt << endl;
		cout << "symbol_definition::definition_of_translation_plane TP->label_tex=" << TP->label_tex << endl;
		//Spread->Andre->report(cout, verbose_level);
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_translation_plane we created a translation plane called " << TP->label_txt << endl;

	}

	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_translation_plane(define_label, TP, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_translation_plane before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_translation_plane done" << endl;
	}
}




void symbol_definition::definition_of_spread_table(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_spread_table" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_spread_table "
				"using existing PA " << spread_table_label_PA << endl;
	}
	int idx;
	projective_geometry::projective_space_with_action *PA;

	idx = Sym->Orbiter_top_level_session->find_symbol(spread_table_label_PA);
	PA = (projective_geometry::projective_space_with_action *) Sym->Orbiter_top_level_session->get_object(idx);




	packings::packing_classify *P;

	if (f_v) {
		cout << "symbol_definition::definition_of_spread_table before P->spread_table_init" << endl;
	}

	P = NEW_OBJECT(packings::packing_classify);

	P->spread_table_init(
			PA,
			dimension_of_spread_elements,
			TRUE /* f_select_spread */, spread_selection_text,
			spread_tables_prefix,
			verbose_level);


	if (f_v) {
		cout << "symbol_definition::definition_of_spread_table after do_spread_table_init" << endl;
	}




	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_spread_table(define_label, P, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_spread_table before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_spread_table done" << endl;
	}
}


void symbol_definition::definition_of_packing_was(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was "
				"using existing spread table " << packing_was_label_spread_table << endl;
	}
	int idx;
	packings::packing_classify *P;

	idx = Sym->Orbiter_top_level_session->find_symbol(packing_was_label_spread_table);
	P = (packings::packing_classify *) Sym->Orbiter_top_level_session->get_object(idx);






	packings::packing_was *PW;

	PW = NEW_OBJECT(packings::packing_was);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was before PW->init" << endl;
	}

	PW->init(packing_was_descr, P, verbose_level);

	if (f_v) {
		cout << "symbol_definition::perform_activity after PW->init" << endl;
	}




	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_packing_was(define_label, PW, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was done" << endl;
	}
}



void symbol_definition::definition_of_packing_was_choose_fixed_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was_choose_fixed_points" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was_choose_fixed_points "
				"using existing object " << packing_with_assumed_symmetry_label << endl;
	}
	int idx;
	packings::packing_was *PW;

	idx = Sym->Orbiter_top_level_session->find_symbol(packing_with_assumed_symmetry_label);
	PW = (packings::packing_was *) Sym->Orbiter_top_level_session->get_object(idx);


	packings::packing_was_fixpoints *PWF;

	PWF = NEW_OBJECT(packings::packing_was_fixpoints);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was_choose_fixed_points before PWF->init" << endl;
	}

	PWF->init(PW,
			packing_with_assumed_symmetry_choose_fixed_points_clique_size,
			packing_with_assumed_symmetry_choose_fixed_points_control,
			verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was_choose_fixed_points after PWF->init" << endl;
	}

	if (packing_with_assumed_symmetry_choose_fixed_points_clique_size > 0) {
		PWF->compute_cliques_on_fixpoint_graph(
				packing_with_assumed_symmetry_choose_fixed_points_clique_size,
				packing_with_assumed_symmetry_choose_fixed_points_control,
				verbose_level);
	}
	else {
		cout << "symbol_definition::definition_of_packing_was_choose_fixed_points clique size on fixed spreads is zero, so nothing to do" << endl;
	}




	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_packing_was_choose_fixed_points(define_label, PWF, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was_choose_fixed_points before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was_choose_fixed_points done" << endl;
	}
}





void symbol_definition::definition_of_packing_long_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_long_orbits" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_long_orbits "
				"using existing object " << packing_long_orbits_choose_fixed_points_label << endl;
	}
	int idx;

	packings::packing_was_fixpoints *PWF;

	idx = Sym->Orbiter_top_level_session->find_symbol(packing_long_orbits_choose_fixed_points_label);
	PWF = (packings::packing_was_fixpoints *) Sym->Orbiter_top_level_session->get_object(idx);


	packings::packing_long_orbits *PL;

	PL = NEW_OBJECT(packings::packing_long_orbits);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_long_orbits before PL->init" << endl;
	}

	PL->init(PWF, Packing_long_orbits_description, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_long_orbits after PL->init" << endl;
	}




	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_packing_long_orbits(define_label, PL, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_packing_long_orbits before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_packing_long_orbits done" << endl;
	}
}


void symbol_definition::definition_of_graph_classification(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_graph_classification" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_graph_classification "
				"using existing object " << packing_long_orbits_choose_fixed_points_label << endl;
	}


	apps_graph_theory::graph_classify *GC;


	GC = NEW_OBJECT(apps_graph_theory::graph_classify);

	if (f_v) {
		cout << "symbol_definition::definition_of_graph_classification before GC->init" << endl;
	}

	GC->init(Graph_classify_description, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_graph_classification after GC->init" << endl;
	}




	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_graph_classify(define_label, GC, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_graph_classification before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_graph_classification done" << endl;
	}
}

void symbol_definition::definition_of_diophant(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_diophant" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_diophant "
				"using existing object " << packing_long_orbits_choose_fixed_points_label << endl;
	}


	solvers::diophant_create *Dio;


	Dio = NEW_OBJECT(solvers::diophant_create);

	if (f_v) {
		cout << "symbol_definition::definition_of_diophant before Dio->init" << endl;
	}

	Dio->init(Diophant_description, verbose_level);


	if (f_v) {
		cout << "symbol_definition::definition_of_diophant after Dio->init" << endl;
	}




	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_diophant(define_label, Dio, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_diophant before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_diophant done" << endl;
	}
}



void symbol_definition::definition_of_design(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_design" << endl;
	}


	apps_combinatorics::design_create *DC;


	DC = NEW_OBJECT(apps_combinatorics::design_create);

	if (f_v) {
		cout << "symbol_definition::definition_of_design before DC->init" << endl;
	}

	DC->init(Design_create_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_design after DC->init" << endl;
	}




	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_design(define_label, DC, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_design before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_design done" << endl;
	}
}



void symbol_definition::definition_of_design_table(int verbose_level)
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
	DC = (apps_combinatorics::design_create *) Sym->Orbiter_top_level_session->get_object(idx);




	apps_algebra::any_group *AG;

	idx = orbiter_kernel_system::Orbiter->find_symbol(design_table_group);

	symbol_table_object_type t;

	t = orbiter_kernel_system::Orbiter->get_object_type(idx);

	if (t != t_any_group) {
		cout << "object must be of type group, but is ";
		orbiter_kernel_system::Orbiter->print_type(t);
		cout << endl;
		exit(1);
	}
	AG = (apps_algebra::any_group *) orbiter_kernel_system::Orbiter->get_object(idx);



	apps_combinatorics::combinatorics_global Combi;
	apps_combinatorics::design_tables *T;


	if (f_v) {
		cout << "symbol_definition::definition_of_design_table before Combi.create_design_table" << endl;
	}

	Combi.create_design_table(DC,
			design_table_label,
			T,
			AG->Subgroup_gens,
			verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_design_table after Combi.create_design_table" << endl;
	}



	apps_combinatorics::large_set_classify *LS;

	LS = NEW_OBJECT(apps_combinatorics::large_set_classify);

	if (f_v) {
		cout << "symbol_definition::definition_of_design_table before LS->init" << endl;
	}

	LS->init(DC,
			T,
			verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_design_table after LS->init" << endl;
	}



	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_design_table(define_label, LS, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_design_table before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_design_table done" << endl;
	}
}


void symbol_definition::definition_of_large_set_was(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_large_set_was" << endl;
	}

	if (f_v) {
		cout << "symbol_definition::definition_of_large_set_was "
				"using existing spread table " << packing_was_label_spread_table << endl;
	}
	int idx;
	apps_combinatorics::large_set_classify *LS;

	idx = Sym->Orbiter_top_level_session->find_symbol(large_set_was_label_design_table);
	LS = (apps_combinatorics::large_set_classify *) Sym->Orbiter_top_level_session->get_object(idx);






	apps_combinatorics::large_set_was *LSW;

	LSW = NEW_OBJECT(apps_combinatorics::large_set_was);

	if (f_v) {
		cout << "symbol_definition::definition_of_large_set_was before LSW->init" << endl;
	}

	LSW->init(large_set_was_descr, LS, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_large_set_was after LSW->init" << endl;
	}




	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_large_set_was(define_label, LSW, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_large_set_was before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_large_set_was done" << endl;
	}
}

void symbol_definition::definition_of_set(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_set" << endl;
	}


	data_structures::set_builder *SB;

	SB = NEW_OBJECT(data_structures::set_builder);

	if (f_v) {
		cout << "symbol_definition::definition_of_set before SB->init" << endl;
	}

	SB->init(Set_builder_description, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_set after SB->init" << endl;
	}


	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_set(define_label, SB, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_set before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_set done" << endl;
	}
}

void symbol_definition::definition_of_vector(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_vector" << endl;
	}


	field_theory::finite_field *F = NULL;

	if (Vector_builder_description->f_field) {

		int idx;

		idx = Sym->Orbiter_top_level_session->find_symbol(Vector_builder_description->field_label);
		F = (field_theory::finite_field *) Sym->Orbiter_top_level_session->get_object(idx);
		if (f_v) {
			cout << "symbol_definition::definition_of_vector over a field" << endl;
		}


	}
	else {
		if (f_v) {
			cout << "symbol_definition::definition_of_vector not over a field" << endl;
		}

	}


	data_structures::vector_builder *VB;

	VB = NEW_OBJECT(data_structures::vector_builder);

	if (f_v) {
		cout << "symbol_definition::definition_of_vector before VB->init" << endl;
	}

	VB->init(Vector_builder_description, F, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_vector after VB->init" << endl;
	}


	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_vector(define_label, VB, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_vector before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_vector done" << endl;
	}
}

void symbol_definition::definition_of_combinatorial_object(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_combinatorial_object" << endl;
	}

	data_structures::data_input_stream *IS;

	IS = NEW_OBJECT(data_structures::data_input_stream);

	if (f_v) {
		cout << "symbol_definition::definition_of_combinatorial_object before IS->init" << endl;
	}

	IS->init(Data_input_stream_description, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_combinatorial_object after IS->init" << endl;
	}


	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_combinatorial_objects(define_label, IS, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_combinatorial_object before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_combinatorial_object done" << endl;
	}
}

void symbol_definition::do_geometry_builder(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::do_geometry_builder" << endl;
	}

	geometry_builder::geometry_builder *GB;

	GB = NEW_OBJECT(geometry_builder::geometry_builder);

	GB->init_description(Geometry_builder_description, verbose_level);

	GB->gg->main2(verbose_level);


	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_geometry_builder_object(define_label, GB, verbose_level);
	if (f_v) {
		cout << "symbol_definition::do_geometry_builder before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::do_geometry_builder done" << endl;
	}
}

void symbol_definition::load_finite_field_PG(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::load_finite_field_PG" << endl;
	}
	data_structures::string_tools ST;


	if (Projective_space_with_action_description->f_field) {
		if (f_v) {
			cout << "symbol_definition::load_finite_field_PG "
					"using existing finite field "
					<< Projective_space_with_action_description->field_label << endl;
		}
		Projective_space_with_action_description->F =
				Get_object_of_type_finite_field(
						Projective_space_with_action_description->field_label);
	}
	else if (Projective_space_with_action_description->f_q) {

		int q = Projective_space_with_action_description->q;

		if (f_v) {
			cout << "symbol_definition::load_finite_field_PG "
					"creating the finite field of order " << q << endl;
		}
		Projective_space_with_action_description->F = NEW_OBJECT(field_theory::finite_field);
		Projective_space_with_action_description->F->finite_field_init(q, FALSE /* f_without_tables */, verbose_level - 1);
		if (f_v) {
			cout << "symbol_definition::load_finite_field_PG "
					"the finite field of order " << q << " has been created" << endl;
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



void symbol_definition::load_finite_field(std::string &input_q,
		field_theory::finite_field *&F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::load_finite_field" << endl;
	}
	data_structures::string_tools ST;

	if (ST.starts_with_a_number(input_q)) {
		int q;

		q = ST.strtoi(input_q);
		if (f_v) {
			cout << "symbol_definition::load_finite_field "
					"creating the finite field of order " << q << endl;
		}
		F = NEW_OBJECT(field_theory::finite_field);
		F->finite_field_init(q, FALSE /* f_without_tables */, verbose_level - 1);
		if (f_v) {
			cout << "symbol_definition::load_finite_field "
					"the finite field of order " << q << " has been created" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "symbol_definition::load_finite_field "
					"using existing finite field " << input_q << endl;
		}
		int idx;
		idx = Sym->Orbiter_top_level_session->find_symbol(input_q);
		if (idx < 0) {
			cout << "symbol_definition::load_finite_field done cannot find finite field object" << endl;
			exit(1);
		}
		F = (field_theory::finite_field *) Sym->Orbiter_top_level_session->get_object(idx);
	}

	if (f_v) {
		cout << "symbol_definition::load_finite_field done" << endl;
	}
}


void symbol_definition::definition_of_vector_ge(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_vector_ge" << endl;
	}


	apps_algebra::vector_ge_builder *VB;

	VB = NEW_OBJECT(apps_algebra::vector_ge_builder);

	if (f_v) {
		cout << "symbol_definition::definition_of_vector_ge before VB->init" << endl;
	}

	VB->init(Vector_ge_description, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_vector_ge after VB->init" << endl;
	}


	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_vector_ge(define_label, VB, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_vector_ge before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_vector_ge done" << endl;
	}
}

void symbol_definition::definition_of_action_on_forms(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_action_on_forms" << endl;
	}


	apps_algebra::action_on_forms *AF;

	AF = NEW_OBJECT(apps_algebra::action_on_forms);

	if (f_v) {
		cout << "symbol_definition::definition_of_action_on_forms before AF->create_action_on_forms" << endl;
	}

	AF->create_action_on_forms(
			Action_on_forms_descr,
			verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_action_on_forms after AF->create_action_on_forms" << endl;
	}


	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_action_on_forms(define_label, AF, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_action_on_forms before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_action_on_forms done" << endl;
	}
}

void symbol_definition::definition_of_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_orbits" << endl;
	}


	apps_algebra::orbits_create *OC;

	OC = NEW_OBJECT(apps_algebra::orbits_create);

	if (f_v) {
		cout << "symbol_definition::definition_of_orbits before OC->init" << endl;
	}

	OC->init(
			Orbits_create_description,
			verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_orbits after OC->init" << endl;
	}


	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_orbits(define_label, OC, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_orbits before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_orbits done" << endl;
	}
}

void symbol_definition::definition_of_poset_classification_control(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_poset_classification_control" << endl;
	}




	orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

	Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);
	Symb->init_poset_classification_control(define_label, Poset_classification_control, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_poset_classification_control before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_poset_classification_control done" << endl;
	}
}






}}}




