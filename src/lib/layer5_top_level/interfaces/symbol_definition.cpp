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


	load_finite_field(Projective_space_with_action_description->input_q,
			Projective_space_with_action_description->F,
			verbose_level);


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


	load_finite_field(Orthogonal_space_with_action_description->input_q,
			Orthogonal_space_with_action_description->F,
			verbose_level);


	int f_semilinear;
	number_theory::number_theory_domain NT;


	if (NT.is_prime(Orthogonal_space_with_action_description->F->q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}

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

void symbol_definition::definition_of_linear_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_linear_group" << endl;
	}

	load_finite_field(Linear_group_description->input_q,
			Linear_group_description->F,
			verbose_level);



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
		cout << "symbol_definition::definition_of_permutation_group before PGC->permutation_group_init, "
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
		cout << "symbol_definition::definition_of_modified_group before PGC->permutation_group_init, "
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


	geometry::geometric_object_create *GOC;

	GOC = NEW_OBJECT(geometry::geometric_object_create);


	projective_geometry::projective_space_with_action *PA;

	PA = The_Orbiter_top_level_session->get_object_of_type_projective_space(geometric_object_projective_space_label);

	geometry::geometric_object_create *GeoObj;

	GeoObj = NEW_OBJECT(geometry::geometric_object_create);

	if (f_v) {
		cout << "symbol_definition::definition_of_geometric_object before GeoObj->init" << endl;
	}

	GeoObj->init(Geometric_object_description, PA->P, verbose_level);

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





}}}




