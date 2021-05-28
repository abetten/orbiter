/*
 * interface_symbol_table.cpp
 *
 *  Created on: Dec 2, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {




interface_symbol_table::interface_symbol_table()
{
	f_define = FALSE;
	//define_label

	f_finite_field = FALSE;
	Finite_field_description = NULL;

	f_projective_space = FALSE;
	Projective_space_with_action_description = NULL;

	f_orthogonal_space = FALSE;
	Orthogonal_space_with_action_description = NULL;

	f_linear_group = FALSE;
	Linear_group_description = NULL;

	f_combinatorial_object = FALSE;
	Combinatorial_object_description = NULL;

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
	//std::string design_table_go_text;
	//std::string design_table_generators_data;


	f_large_set_was = FALSE;
	//std::string  large_set_was_label_design_table;
	large_set_was_descr = NULL;



	f_print_symbols = FALSE;
	f_with = FALSE;
	//std::vector<std::string> with_labels;





	f_finite_field_activity = FALSE;
	Finite_field_activity_description = NULL;

	f_projective_space_activity = FALSE;
	Projective_space_activity_description = NULL;

	f_orthogonal_space_activity = FALSE;
	Orthogonal_space_activity_description = NULL;

	f_group_theoretic_activity = FALSE;
	Group_theoretic_activity_description = NULL;

	f_cubic_surface_activity = FALSE;
	Cubic_surface_activity_description = NULL;

	f_quartic_curve_activity = FALSE;
	Quartic_curve_activity_description = NULL;

	f_combinatorial_object_activity = FALSE;
	Combinatorial_object_activity_description = NULL;

	f_graph_theoretic_activity = FALSE;
	Graph_theoretic_activity_description = NULL;

	f_classification_of_cubic_surfaces_with_double_sixes_activity = FALSE;
	Classification_of_cubic_surfaces_with_double_sixes_activity_description = NULL;

	f_spread_table_activity = FALSE;
	Spread_table_activity_description = NULL;

	f_packing_with_symmetry_assumption_activity = FALSE;
	Packing_was_activity_description = NULL;

	f_packing_fixed_points_activity = FALSE;
	Packing_was_fixpoints_activity_description = NULL;

	f_graph_classification_activity = FALSE;
	Graph_classification_activity_description = NULL;

	f_diophant_activity = FALSE;
	Diophant_activity_description = NULL;

	f_design_activity = FALSE;
	Design_activity_description = NULL;


	f_large_set_was_activity = FALSE;
	Large_set_was_activity_description = NULL;

}


void interface_symbol_table::print_help(int argc,
		std::string *argv, int i, int verbose_level)
{
	if (stringcmp(argv[i], "-define") == 0) {
		cout << "-define <string : label> description -end" << endl;
	}
	else if (stringcmp(argv[i], "-print_symbols") == 0) {
		cout << "-print_symbols" << endl;
	}
	else if (stringcmp(argv[i], "-with") == 0) {
		cout << "-with <string : label> *[ -and <string : label> ] -do ... -end" << endl;
	}
}

int interface_symbol_table::recognize_keyword(int argc,
		std::string *argv, int i, int verbose_level)
{
	if (i >= argc) {
		return false;
	}
	if (stringcmp(argv[i], "-define") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-print_symbols") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-with") == 0) {
		return true;
	}
	return false;
}

void interface_symbol_table::read_arguments(
		orbiter_top_level_session *Orbiter_top_level_session,
		int argc, std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::read_arguments" << endl;
	}


	if (f_v) {
		cout << "interface_symbol_table::read_arguments the next argument is " << argv[i] << endl;
	}

	if (stringcmp(argv[i], "-define") == 0) {
		read_definition(Orbiter_top_level_session, argc, argv, i, verbose_level);
	}
	else if (stringcmp(argv[i], "-print_symbols") == 0) {
		f_print_symbols = TRUE;
		cout << "-print_symbols" << endl;
		i++;
	}
	else if (stringcmp(argv[i], "-with") == 0) {
		read_with(Orbiter_top_level_session, argc, argv, i, verbose_level);
	}

	if (f_v) {
		cout << "interface_symbol_table::read_arguments done" << endl;
	}
	//return i;
}

void interface_symbol_table::read_definition(
		orbiter_top_level_session *Orbiter_top_level_session,
		int argc, std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::read_definition" << endl;
	}

	//f_define = TRUE;
	define_label.assign(argv[++i]);
	if (f_v) {
		cout << "interface_symbol_table::read_definition define_label=" << define_label << endl;
	}
	i++;
	cout << "-define " << define_label << endl;
	if (stringcmp(argv[i], "-finite_field") == 0) {
		f_finite_field = TRUE;
		Finite_field_description = NEW_OBJECT(finite_field_description);
		cout << "reading -finite_field" << endl;
		i += Finite_field_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-finite_field" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		if (f_v) {
			cout << "interface_symbol_table::read_definition before definition_of_finite_field" << endl;
		}
		definition_of_finite_field(Orbiter_top_level_session, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_definition after definition_of_finite_field" << endl;
		}
	}
	else if (stringcmp(argv[i], "-projective_space") == 0) {
		f_projective_space = TRUE;
		Projective_space_with_action_description = NEW_OBJECT(projective_space_with_action_description);
		cout << "reading -projective_space" << endl;
		i += Projective_space_with_action_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-projective_space" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		if (f_v) {
			cout << "interface_symbol_table::read_definition before definition_of_projective_space" << endl;
		}
		definition_of_projective_space(Orbiter_top_level_session, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_definition after definition_of_projective_space" << endl;
		}
	}
	else if (stringcmp(argv[i], "-orthogonal_space") == 0) {
		f_orthogonal_space = TRUE;
		Orthogonal_space_with_action_description = NEW_OBJECT(orthogonal_space_with_action_description);
		cout << "reading -orthogonal_space" << endl;
		i += Orthogonal_space_with_action_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-orthogonal_space" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		if (f_v) {
			cout << "interface_symbol_table::read_definition before definition_of_orthogonal_space" << endl;
		}
		definition_of_orthogonal_space(Orbiter_top_level_session, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_definition after definition_of_orthogonal_space" << endl;
		}
	}
	else if (stringcmp(argv[i], "-linear_group") == 0) {
		f_linear_group = TRUE;
		Linear_group_description = NEW_OBJECT(linear_group_description);
		cout << "reading -linear_group" << endl;
		i += Linear_group_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-linear_group" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		if (f_v) {
			cout << "interface_symbol_table::read_definition before definition_of_linear_group" << endl;
		}
		definition_of_linear_group(Orbiter_top_level_session, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_definition after definition_of_linear_group" << endl;
		}
	}
	else if (stringcmp(argv[i], "-formula") == 0) {
		cout << "-formula" << endl;
		string label;
		string label_tex;
		string managed_variables;
		string formula_text;

		label.assign(argv[++i]);
		label_tex.assign(argv[++i]);
		managed_variables.assign(argv[++i]);
		formula_text.assign(argv[++i]);

		i++;


		formula *F;

		F = NEW_OBJECT(formula);
		F->init(label, label_tex, managed_variables, formula_text, verbose_level);

		if (f_v) {
			cout << "interface_symbol_table::read_definition before definition_of_formula" << endl;
		}
		definition_of_formula(Orbiter_top_level_session, F, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_definition after definition_of_formula" << endl;
		}
	}
	else if (stringcmp(argv[i], "-collection") == 0) {
		cout << "-collection" << endl;
		string list_of_objects;

		list_of_objects.assign(argv[++i]);
		i++;

		if (f_v) {
			cout << "interface_symbol_table::read_definition before definition_of_collection" << endl;
		}
		definition_of_collection(Orbiter_top_level_session, list_of_objects, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_definition after definition_of_collection" << endl;
		}
	}
	else if (stringcmp(argv[i], "-combinatorial_object") == 0) {

		f_combinatorial_object = TRUE;
		Combinatorial_object_description = NEW_OBJECT(combinatorial_object_description);
		cout << "reading -combinatorial_object" << endl;
		i += Combinatorial_object_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-combinatorial_object" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		if (f_v) {
			cout << "interface_symbol_table::read_definition before definition_of_combinatorial_object" << endl;
		}
		definition_of_combinatorial_object(Orbiter_top_level_session, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_definition after definition_of_combinatorial_object" << endl;
		}
	}

	else if (stringcmp(argv[i], "-graph") == 0) {

		f_graph = TRUE;
		Create_graph_description = NEW_OBJECT(create_graph_description);
		cout << "reading -graph" << endl;

		i += Create_graph_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-graph" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		if (f_v) {
			cout << "interface_symbol_table::read_definition before definition_of_graph" << endl;
		}
		definition_of_graph(Orbiter_top_level_session, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_definition after definition_of_graph" << endl;
		}
	}
	else if (stringcmp(argv[i], "-spread_table") == 0) {
		f_spread_table = TRUE;

		spread_table_label_PA.assign(argv[++i]);
		dimension_of_spread_elements = strtoi(argv[++i]);
		spread_selection_text.assign(argv[++i]);
		spread_tables_prefix.assign(argv[++i]);
		cout << "dimension_of_spread_elements = " << dimension_of_spread_elements
				<< " " << spread_selection_text
				<< " " << spread_tables_prefix << endl;

		i++;

		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		cout << "-spread_table " << spread_table_label_PA
				<< " " << dimension_of_spread_elements
				<< " " << spread_selection_text
				<< " " << spread_tables_prefix
				<< endl;
		if (f_v) {
			cout << "interface_symbol_table::read_definition before definition_of_projective_space" << endl;
		}
		definition_of_spread_table(Orbiter_top_level_session, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_definition after definition_of_projective_space" << endl;
		}
	}
	else if (stringcmp(argv[i], "-packing_with_symmetry_assumption") == 0) {
		f_packing_was = TRUE;

		packing_was_label_spread_table.assign(argv[++i]);

		packing_was_descr = NEW_OBJECT(packing_was_description);
		cout << "reading -packing_with_symmetry_assumption" << endl;
		i += packing_was_descr->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-packing_with_symmetry_assumption" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		cout << "-packing_with_symmetry_assumption " << packing_was_label_spread_table
				<< endl;
		if (f_v) {
			cout << "interface_symbol_table::read_definition before definition_of_packing_was" << endl;
		}
		definition_of_packing_was(Orbiter_top_level_session, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_definition after definition_of_packing_was" << endl;
		}
	}
	else if (stringcmp(argv[i], "-packing_choose_fixed_points") == 0) {
		f_packing_was_choose_fixed_points = TRUE;

		packing_with_assumed_symmetry_label.assign(argv[++i]);
		packing_with_assumed_symmetry_choose_fixed_points_clique_size = strtoi(argv[++i]);

		packing_with_assumed_symmetry_choose_fixed_points_control = NEW_OBJECT(poset_classification_control);
		cout << "reading -packing_with_symmetry_assumption_choose_fixed_points" << endl;
		i += packing_with_assumed_symmetry_choose_fixed_points_control->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

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
		if (f_v) {
			cout << "interface_symbol_table::read_definition before definition_of_packing_was_choose_fixed_points" << endl;
		}
		definition_of_packing_was_choose_fixed_points(Orbiter_top_level_session, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_definition after definition_of_packing_was_choose_fixed_points" << endl;
		}
	}
	else if (stringcmp(argv[i], "-packing_long_orbits") == 0) {
		f_packing_long_orbits = TRUE;

		packing_long_orbits_choose_fixed_points_label.assign(argv[++i]);

		Packing_long_orbits_description = NEW_OBJECT(packing_long_orbits_description);
		cout << "reading -packing_long_orbits" << endl;
		i += Packing_long_orbits_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-packing_long_orbits" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		cout << "-packing_long_orbits "
				<< packing_long_orbits_choose_fixed_points_label
				<< endl;
		if (f_v) {
			cout << "interface_symbol_table::read_definition before definition_of_packing_long_orbits" << endl;
		}
		definition_of_packing_long_orbits(Orbiter_top_level_session, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_definition after definition_of_packing_long_orbits" << endl;
		}
	}
	else if (stringcmp(argv[i], "-graph_classification") == 0) {
		f_graph_classification = TRUE;

		Graph_classify_description = NEW_OBJECT(graph_classify_description);
		cout << "reading -graph_classification" << endl;
		i += Graph_classify_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-graph_classification" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		cout << "-graph_classification "
				<< endl;
		if (f_v) {
			cout << "interface_symbol_table::read_definition before definition_of_graph_classification" << endl;
		}
		definition_of_graph_classification(Orbiter_top_level_session, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_definition after definition_of_graph_classification" << endl;
		}
	}
	else if (stringcmp(argv[i], "-diophant") == 0) {
		f_diophant = TRUE;

		Diophant_description = NEW_OBJECT(diophant_description);
		cout << "reading -diophant_description" << endl;
		i += Diophant_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-diophant_description" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		cout << "-diophant_description "
				<< endl;
		if (f_v) {
			cout << "interface_symbol_table::read_definition before definition_of_graph_classification" << endl;
		}
		definition_of_diophant(Orbiter_top_level_session, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_definition after definition_of_graph_classification" << endl;
		}
	}
	else if (stringcmp(argv[i], "-design") == 0) {

		f_design = TRUE;
		Design_create_description = NEW_OBJECT(design_create_description);
		cout << "reading -design" << endl;

		i += Design_create_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-design" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		if (f_v) {
			cout << "interface_symbol_table::read_definition before definition_of_design" << endl;
		}
		definition_of_design(Orbiter_top_level_session, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_definition after definition_of_design" << endl;
		}
	}
	else if (stringcmp(argv[i], "-design_table") == 0) {
		f_design_table = TRUE;

		design_table_label_design.assign(argv[++i]);
		design_table_label.assign(argv[++i]);
		design_table_go_text.assign(argv[++i]);
		design_table_generators_data.assign(argv[++i]);


		i++;

		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		cout << "-design_table " << design_table_label_design
				<< " " << design_table_label
				<< " " << design_table_go_text
				<< " " << design_table_generators_data
				<< endl;
		if (f_v) {
			cout << "interface_symbol_table::read_definition before definition_of_design_table" << endl;
		}
		definition_of_design_table(Orbiter_top_level_session, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_definition after definition_of_design_table" << endl;
		}
	}
	else if (stringcmp(argv[i], "-large_set_with_symmetry_assumption") == 0) {
		f_large_set_was = TRUE;

		large_set_was_label_design_table.assign(argv[++i]);

		large_set_was_descr = NEW_OBJECT(large_set_was_description);
		cout << "reading -large_set_with_symmetry_assumption" << endl;
		i += large_set_was_descr->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-large_set_with_symmetry_assumption" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		cout << "-large_set_with_symmetry_assumption " << large_set_was_label_design_table
				<< endl;
		if (f_v) {
			cout << "interface_symbol_table::read_definition before definition_of_packing_was" << endl;
		}
		definition_of_large_set_was(Orbiter_top_level_session, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_definition after definition_of_packing_was" << endl;
		}
	}

	else {
		cout << "unrecognized command after -define" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "interface_symbol_table::read_definition done" << endl;
	}
}





void interface_symbol_table::read_with(
		orbiter_top_level_session *Orbiter_top_level_session,
		int argc, std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::read_with" << endl;
	}

	f_with = TRUE;
	string s;

	s.assign(argv[++i]);
	with_labels.push_back(s);

	while (TRUE) {
		i++;
		if (stringcmp(argv[i], "-and") == 0) {
			string s;

			s.assign(argv[++i]);
			with_labels.push_back(s);
		}
		else if (stringcmp(argv[i], "-do") == 0) {
			i++;
			read_activity_arguments(argc, argv, i, verbose_level);
			break;
		}
		else {
			cout << "syntax error after -with, seeing " << argv[i] << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "interface_symbol_table::read_with done" << endl;
	}

}
void interface_symbol_table::read_activity_arguments(int argc,
		std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::read_activity_arguments" << endl;
	}
	if (stringcmp(argv[i], "-finite_field_activity") == 0) {
		f_finite_field_activity = TRUE;
		Finite_field_activity_description =
				NEW_OBJECT(finite_field_activity_description);
		cout << "reading -finite_field_activity" << endl;
		i += Finite_field_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-finite_field_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		//i++;
	}
	else if (stringcmp(argv[i], "-projective_space_activity") == 0) {
		f_projective_space_activity = TRUE;
		Projective_space_activity_description =
				NEW_OBJECT(projective_space_activity_description);
		cout << "reading -projective_space_activity" << endl;
		i += Projective_space_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-projective_space_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		//i++;
	}
	else if (stringcmp(argv[i], "-orthogonal_space_activity") == 0) {
		f_orthogonal_space_activity = TRUE;
		Orthogonal_space_activity_description =
				NEW_OBJECT(orthogonal_space_activity_description);
		cout << "reading -orthogonal_space_activity" << endl;
		i += Orthogonal_space_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-orthogonal_space_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		//i++;
	}
	else if (stringcmp(argv[i], "-group_theoretic_activities") == 0) {
		f_group_theoretic_activity = TRUE;
		Group_theoretic_activity_description =
				NEW_OBJECT(group_theoretic_activity_description);
		cout << "reading -group_theoretic_activities" << endl;
		i += Group_theoretic_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-group_theoretic_activities" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		//i++;
	}
	else if (stringcmp(argv[i], "-cubic_surface_activity") == 0) {
		f_cubic_surface_activity = TRUE;
		Cubic_surface_activity_description =
				NEW_OBJECT(cubic_surface_activity_description);
		cout << "reading -cubic_surface_activity" << endl;
		i += Cubic_surface_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-cubic_surface_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
	}
	else if (stringcmp(argv[i], "-quartic_curve_activity") == 0) {
		f_quartic_curve_activity = TRUE;
		Quartic_curve_activity_description =
				NEW_OBJECT(quartic_curve_activity_description);
		cout << "reading -quartic_curve_activity" << endl;
		i += Quartic_curve_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-quartic_curve_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
	}
	else if (stringcmp(argv[i], "-combinatorial_object_activity") == 0) {
		f_combinatorial_object_activity = TRUE;
		Combinatorial_object_activity_description =
				NEW_OBJECT(combinatorial_object_activity_description);
		cout << "reading -combinatorial_object_activity" << endl;
		i += Combinatorial_object_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-combinatorial_object_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
	}
	else if (stringcmp(argv[i], "-graph_theoretic_activity") == 0) {
		f_graph_theoretic_activity = TRUE;
		Graph_theoretic_activity_description =
				NEW_OBJECT(graph_theoretic_activity_description);
		cout << "reading -graph_theoretic_activity" << endl;
		i += Graph_theoretic_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-graph_theoretic_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
	}
	else if (stringcmp(argv[i], "-classification_of_cubic_surfaces_with_double_sixes_activity") == 0) {
		f_classification_of_cubic_surfaces_with_double_sixes_activity = TRUE;
		Classification_of_cubic_surfaces_with_double_sixes_activity_description =
				NEW_OBJECT(classification_of_cubic_surfaces_with_double_sixes_activity_description);
		cout << "reading -classification_of_cubic_surfaces_with_double_sixes_activity" << endl;
		i += Classification_of_cubic_surfaces_with_double_sixes_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-classification_of_cubic_surfaces_with_double_sixes_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
	}
	else if (stringcmp(argv[i], "-spread_table_activity") == 0) {
		f_spread_table_activity = TRUE;
		Spread_table_activity_description =
				NEW_OBJECT(spread_table_activity_description);
		cout << "reading -classification_of_cubic_surfaces_with_double_sixes_activity" << endl;
		i += Spread_table_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-spread_table_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
	}
	else if (stringcmp(argv[i], "-packing_with_symmetry_assumption_activity") == 0) {
		f_packing_with_symmetry_assumption_activity = TRUE;
		Packing_was_activity_description =
				NEW_OBJECT(packing_was_activity_description);
		cout << "reading -packing_with_symmetry_assumption_activity" << endl;
		i += Packing_was_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-packing_with_symmetry_assumption_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
	}
	else if (stringcmp(argv[i], "-packing_fixed_points_activity") == 0) {
		f_packing_fixed_points_activity = TRUE;
		Packing_was_fixpoints_activity_description =
				NEW_OBJECT(packing_was_fixpoints_activity_description);
		cout << "reading -packing_fixed_points_activity" << endl;
		i += Packing_was_fixpoints_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-packing_fixed_points_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
	}
	else if (stringcmp(argv[i], "-graph_classification_activity") == 0) {
		f_graph_classification_activity = TRUE;
		Graph_classification_activity_description =
				NEW_OBJECT(graph_classification_activity_description);
		cout << "reading -graph_classification_activity" << endl;
		i += Graph_classification_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-graph_classification_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
	}
	else if (stringcmp(argv[i], "-diophant_activity") == 0) {
		f_diophant_activity = TRUE;
		Diophant_activity_description =
				NEW_OBJECT(diophant_activity_description);
		cout << "reading -diophant_activity" << endl;
		i += Diophant_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-diophant_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
	}
	else if (stringcmp(argv[i], "-design_activity") == 0) {
		f_design_activity = TRUE;
		Design_activity_description =
				NEW_OBJECT(design_activity_description);
		cout << "reading -design_activity" << endl;
		i += Design_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-design_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
	}
	else if (stringcmp(argv[i], "-large_set_with_symmetry_assumption_activity") == 0) {
		f_large_set_was_activity = TRUE;
		Large_set_was_activity_description =
				NEW_OBJECT(large_set_was_activity_description);
		cout << "reading -large_set_with_symmetry_assumption_activity" << endl;
		i += Large_set_was_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		cout << "-large_set_with_symmetry_assumption_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
	}

	else {
		cout << "expecting activity after -do but seeing " << argv[i] << endl;
		exit(1);
	}

}

void interface_symbol_table::worker(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::worker" << endl;
	}

	if (f_define) {

#if 0
		if (f_v) {
			cout << "interface_symbol_table::worker f_define define_label=" << define_label << endl;
		}
		definition(Orbiter_top_level_session, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::worker f_define define_label=" << define_label << " done" << endl;
		}
#endif
	}
	else if (f_print_symbols) {

		Orbiter_top_level_session->print_symbol_table();
	}
	else if (f_finite_field_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_finite_field_activity" << endl;
		}
		do_finite_field_activity(Orbiter_top_level_session, verbose_level);

	}
	else if (f_projective_space_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_projective_space_activity" << endl;
		}
		do_projective_space_activity(Orbiter_top_level_session, verbose_level);

	}
	else if (f_orthogonal_space_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_orthogonal_space_activity" << endl;
		}
		do_orthogonal_space_activity(Orbiter_top_level_session, verbose_level);

	}
	else if (f_group_theoretic_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_group_theoretic_activity" << endl;
		}
		do_group_theoretic_activity(Orbiter_top_level_session, verbose_level);

	}
	else if (f_cubic_surface_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_cubic_surface_activity" << endl;
		}
		do_cubic_surface_activity(Orbiter_top_level_session, verbose_level);

	}
	else if (f_quartic_curve_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_quartic_curve_activity" << endl;
		}
		do_quartic_curve_activity(Orbiter_top_level_session, verbose_level);

	}
	else if (f_combinatorial_object_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_combinatorial_object_activity" << endl;
		}
		do_combinatorial_object_activity(Orbiter_top_level_session, verbose_level);

	}
	else if (f_graph_theoretic_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_graph_theoretic_activity" << endl;
		}
		do_graph_theoretic_activity(Orbiter_top_level_session, verbose_level);

	}
	else if (f_classification_of_cubic_surfaces_with_double_sixes_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_classification_of_cubic_surfaces_with_double_sixes_activity" << endl;
		}

		do_classification_of_cubic_surfaces_with_double_sixes_activity(Orbiter_top_level_session, verbose_level);
	}
	else if (f_spread_table_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_spread_table_activity" << endl;
		}

		do_spread_table_activity(Orbiter_top_level_session, verbose_level);
	}
	else if (f_packing_with_symmetry_assumption_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_packing_with_symmetry_activity" << endl;
		}

		do_packing_was_activity(Orbiter_top_level_session, verbose_level);
	}
	else if (f_packing_fixed_points_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_packing_with_symmetry_activity" << endl;
		}

		do_packing_fixed_points_activity(Orbiter_top_level_session, verbose_level);
	}
	else if (f_graph_classification_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_graph_classification_activity" << endl;
		}

		do_graph_classification_activity(Orbiter_top_level_session, verbose_level);
	}
	else if (f_diophant_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_diophant_activity" << endl;
		}

		do_diophant_activity(Orbiter_top_level_session, verbose_level);
	}
	else if (f_design_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_design_activity" << endl;
		}

		do_design_activity(Orbiter_top_level_session, verbose_level);

	}
	else if (f_large_set_was_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_large_set_was_activity" << endl;
		}

		do_large_set_was_activity(Orbiter_top_level_session, verbose_level);
	}


	if (f_v) {
		cout << "interface_symbol_table::worker done" << endl;
	}
}



}}
