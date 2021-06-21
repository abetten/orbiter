/*
 * symbol_definition.cpp
 *
 *  Created on: Jun 20, 2021
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


symbol_definition::symbol_definition()
{
	Sym = NULL;

	//f_define = FALSE;
	//define_label

	f_finite_field = FALSE;
	Finite_field_description = NULL;

	f_projective_space = FALSE;
	Projective_space_with_action_description = NULL;

	f_orthogonal_space = FALSE;
	Orthogonal_space_with_action_description = NULL;

	f_linear_group = FALSE;
	Linear_group_description = NULL;

	f_formula = FALSE;
	F = NULL;
	//std::string label;
	//std::string label_tex;
	//std::string managed_variables;
	//std::string formula_text;

	f_collection = FALSE;
	//std::string list_of_objects;

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



}


symbol_definition::~symbol_definition()
{

}

void symbol_definition::read_definition(
		interface_symbol_table *Sym,
		int argc, std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::read_definition" << endl;
	}

	symbol_definition::Sym = Sym;

	//f_define = TRUE;
	define_label.assign(argv[++i]);
	if (f_v) {
		cout << "symbol_definition::read_definition define_label=" << define_label << endl;
	}
	i++;
	if (f_v) {
		cout << "-define " << define_label << endl;
	}
	if (stringcmp(argv[i], "-finite_field") == 0) {
		f_finite_field = TRUE;
		Finite_field_description = NEW_OBJECT(finite_field_description);
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
	else if (stringcmp(argv[i], "-projective_space") == 0) {
		f_projective_space = TRUE;
		Projective_space_with_action_description = NEW_OBJECT(projective_space_with_action_description);
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
	else if (stringcmp(argv[i], "-orthogonal_space") == 0) {
		f_orthogonal_space = TRUE;
		Orthogonal_space_with_action_description = NEW_OBJECT(orthogonal_space_with_action_description);
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
	else if (stringcmp(argv[i], "-linear_group") == 0) {
		f_linear_group = TRUE;
		Linear_group_description = NEW_OBJECT(linear_group_description);
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
	else if (stringcmp(argv[i], "-formula") == 0) {
		if (f_v) {
			cout << "-formula" << endl;
		}
		f_formula = TRUE;

		label.assign(argv[++i]);
		label_tex.assign(argv[++i]);
		managed_variables.assign(argv[++i]);
		formula_text.assign(argv[++i]);

		i++;



		F = NEW_OBJECT(formula);
		F->init(label, label_tex, managed_variables, formula_text, verbose_level);

	}
	else if (stringcmp(argv[i], "-collection") == 0) {
		if (f_v) {
			cout << "-collection" << endl;
		}

		f_collection = TRUE;

		list_of_objects.assign(argv[++i]);
		i++;

	}
	else if (stringcmp(argv[i], "-combinatorial_object") == 0) {

		f_combinatorial_object = TRUE;
		Combinatorial_object_description = NEW_OBJECT(combinatorial_object_description);
		if (f_v) {
			cout << "reading -combinatorial_object" << endl;
		}
		i += Combinatorial_object_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-combinatorial_object" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}

	else if (stringcmp(argv[i], "-graph") == 0) {

		f_graph = TRUE;
		Create_graph_description = NEW_OBJECT(create_graph_description);
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
	else if (stringcmp(argv[i], "-spread_table") == 0) {
		f_spread_table = TRUE;

		spread_table_label_PA.assign(argv[++i]);
		dimension_of_spread_elements = strtoi(argv[++i]);
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
	else if (stringcmp(argv[i], "-packing_with_symmetry_assumption") == 0) {
		f_packing_was = TRUE;

		packing_was_label_spread_table.assign(argv[++i]);

		packing_was_descr = NEW_OBJECT(packing_was_description);
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
	else if (stringcmp(argv[i], "-packing_choose_fixed_points") == 0) {
		f_packing_was_choose_fixed_points = TRUE;

		packing_with_assumed_symmetry_label.assign(argv[++i]);
		packing_with_assumed_symmetry_choose_fixed_points_clique_size = strtoi(argv[++i]);

		packing_with_assumed_symmetry_choose_fixed_points_control = NEW_OBJECT(poset_classification_control);
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
	else if (stringcmp(argv[i], "-packing_long_orbits") == 0) {
		f_packing_long_orbits = TRUE;

		packing_long_orbits_choose_fixed_points_label.assign(argv[++i]);

		Packing_long_orbits_description = NEW_OBJECT(packing_long_orbits_description);
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
	else if (stringcmp(argv[i], "-graph_classification") == 0) {
		f_graph_classification = TRUE;

		Graph_classify_description = NEW_OBJECT(graph_classify_description);
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
	else if (stringcmp(argv[i], "-diophant") == 0) {
		f_diophant = TRUE;

		Diophant_description = NEW_OBJECT(diophant_description);
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
	else if (stringcmp(argv[i], "-design") == 0) {

		f_design = TRUE;
		Design_create_description = NEW_OBJECT(design_create_description);
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
	else if (stringcmp(argv[i], "-design_table") == 0) {
		f_design_table = TRUE;

		design_table_label_design.assign(argv[++i]);
		design_table_label.assign(argv[++i]);
		design_table_go_text.assign(argv[++i]);
		design_table_generators_data.assign(argv[++i]);


		i++;

		if (f_v) {
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-design_table " << design_table_label_design
					<< " " << design_table_label
					<< " " << design_table_go_text
					<< " " << design_table_generators_data
					<< endl;
		}
	}
	else if (stringcmp(argv[i], "-large_set_with_symmetry_assumption") == 0) {
		f_large_set_was = TRUE;

		large_set_was_label_design_table.assign(argv[++i]);

		large_set_was_descr = NEW_OBJECT(large_set_was_description);
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
	else if (f_formula) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_formula" << endl;
		}
		definition_of_formula(F, verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_formula" << endl;
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
	else if (f_combinatorial_object) {
		if (f_v) {
			cout << "symbol_definition::perform_definition before definition_of_combinatorial_object" << endl;
		}
		definition_of_combinatorial_object(verbose_level);
		if (f_v) {
			cout << "symbol_definition::perform_definition after definition_of_combinatorial_object" << endl;
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
	else if (f_projective_space) {
		cout << "-projective_space ";
		Projective_space_with_action_description->print();
	}
	else if (f_orthogonal_space) {
		cout << "-orthogonal_space ";
		Orthogonal_space_with_action_description->print();
	}
	else if (f_linear_group) {
		cout << "-linear_group ";
		Linear_group_description->print();
	}
	else if (f_formula) {
		cout << "-formula " << label << " " << label_tex << " " << managed_variables << " " << formula_text;
		//formula *F;
		//std::string label;
		//std::string label_tex;
		//std::string managed_variables;
		//std::string formula_text;
	}
	else if (f_collection) {
		cout << "-collection ";
		//cout << list_of_objects << endl;
	}
	else if (f_combinatorial_object) {
		cout << "-combinatorial_object ";
		Combinatorial_object_description->print();
	}
	else if (f_graph) {
		cout << "-graph ";
		Create_graph_description->print();
	}
	else if (f_spread_table) {
		cout << "-spread_table ";
		//std::string spread_table_label_PA;
		//int dimension_of_spread_elements;
		//std::string spread_selection_text;
		//std::string spread_tables_prefix;
	}
	else if (f_packing_was) {
		cout << "-packing_was ";
		//std::string packing_was_label_spread_table;
		//packing_was_description * packing_was_descr;
	}
	else if (f_packing_was_choose_fixed_points) {
		cout << "-packing_was_choose_fixed_points ";
		//std::string packing_with_assumed_symmetry_label;
		//int packing_with_assumed_symmetry_choose_fixed_points_clique_size;
		//poset_classification_control *packing_with_assumed_symmetry_choose_fixed_points_control;
	}
	else if (f_packing_long_orbits) {
		cout << "-packing_long_orbits ";
		//std::string packing_long_orbits_choose_fixed_points_label;
		//packing_long_orbits_description * Packing_long_orbits_description;
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
		cout << "-design_table ";
		//std::string design_table_label_design;
		//std::string design_table_label;
		//std::string design_table_go_text;
		//std::string design_table_generators_data;
	}
	else if (f_large_set_was) {
		cout << "-large_set_was ";
		//std::string  large_set_was_label_design_table;
		large_set_was_descr->print();
	}
	else {
		cout << "symbol_definition::print unknown type" << endl;
		exit(1);
	}
}





void symbol_definition::definition_of_finite_field(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_finite_field" << endl;
	}
	Finite_field_description->print();
	finite_field *F;

	F = NEW_OBJECT(finite_field);
	F->init(Finite_field_description, verbose_level);

	orbiter_symbol_table_entry Symb;
	Symb.init_finite_field(define_label, F, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_finite_field before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_finite_field done" << endl;
	}
}

void symbol_definition::definition_of_projective_space(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space" << endl;
	}
	finite_field *F;

	if (string_starts_with_a_number(Projective_space_with_action_description->input_q)) {
		int q;

		q = strtoi(Projective_space_with_action_description->input_q);
		if (f_v) {
			cout << "symbol_definition::definition_of_projective_space "
					"creating finite field of order " << q << endl;
		}
		F = NEW_OBJECT(finite_field);
		F->finite_field_init(q, 0);
	}
	else {
		if (f_v) {
			cout << "symbol_definition::definition_of_projective_space "
					"using existing finite field " << Projective_space_with_action_description->input_q << endl;
		}
		int idx;
		idx = Sym->Orbiter_top_level_session->find_symbol(Projective_space_with_action_description->input_q);
		F = (finite_field *) Sym->Orbiter_top_level_session->get_object(idx);
	}

	Projective_space_with_action_description->F = F;

	int f_semilinear;
	number_theory_domain NT;


	if (NT.is_prime(F->q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}

	projective_space_with_action *PA;

	PA = NEW_OBJECT(projective_space_with_action);

	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space before PA->init" << endl;
	}
	PA->init(Projective_space_with_action_description->F, Projective_space_with_action_description->n,
		f_semilinear,
		TRUE /*f_init_incidence_structure*/,
		0 /* verbose_level */);
	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space after PA->init" << endl;
	}

	orbiter_symbol_table_entry Symb;
	Symb.init_projective_space(define_label, PA, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);

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
	finite_field *F;

	if (string_starts_with_a_number(Orthogonal_space_with_action_description->input_q)) {
		int q;

		q = strtoi(Orthogonal_space_with_action_description->input_q);
		if (f_v) {
			cout << "symbol_definition::definition_of_orthogonal_space "
					"creating finite field of order " << q << endl;
		}
		F = NEW_OBJECT(finite_field);
		F->finite_field_init(q, 0);
	}
	else {
		if (f_v) {
			cout << "symbol_definition::definition_of_orthogonal_space "
					"using existing finite field " << Orthogonal_space_with_action_description->input_q << endl;
		}
		int idx;
		idx = Sym->Orbiter_top_level_session->find_symbol(Orthogonal_space_with_action_description->input_q);
		F = (finite_field *) Sym->Orbiter_top_level_session->get_object(idx);
	}

	Orthogonal_space_with_action_description->F = F;

	int f_semilinear;
	number_theory_domain NT;


	if (NT.is_prime(F->q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}

	orthogonal_space_with_action *OA;

	OA = NEW_OBJECT(orthogonal_space_with_action);

	if (f_v) {
		cout << "symbol_definition::definition_of_orthogonal_space before OA->init" << endl;
	}
	OA->init(Orthogonal_space_with_action_description,
		verbose_level - 2);
	if (f_v) {
		cout << "symbol_definition::definition_of_orthogonal_space after OA->init" << endl;
	}

	orbiter_symbol_table_entry Symb;
	Symb.init_orthogonal_space(define_label, OA, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_orthogonal_space before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);

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

	finite_field *F;

	if (string_starts_with_a_number(Linear_group_description->input_q)) {
		int q;

		q = strtoi(Linear_group_description->input_q);
		if (f_v) {
			cout << "symbol_definition::definition "
					"creating finite field of order " << q << endl;
		}
		F = NEW_OBJECT(finite_field);
		F->finite_field_init(q, 0);
	}
	else {
		if (f_v) {
			cout << "symbol_definition::definition "
					"using existing finite field " << Linear_group_description->input_q << endl;
		}
		int idx;
		idx = Sym->Orbiter_top_level_session->find_symbol(Linear_group_description->input_q);
		F = (finite_field *) Sym->Orbiter_top_level_session->get_object(idx);
	}



	Linear_group_description->F = F;
	//q = Descr->input_q;

	linear_group *LG;

	LG = NEW_OBJECT(linear_group);
	if (f_v) {
		cout << "symbol_definition::definition before LG->init, "
				"creating the group" << endl;
	}

	LG->linear_group_init(Linear_group_description, verbose_level - 5);

	orbiter_symbol_table_entry Symb;
	Symb.init_linear_group(define_label, LG, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_linear_group done" << endl;
	}
}

void symbol_definition::definition_of_formula(formula *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_formula" << endl;
	}

	orbiter_symbol_table_entry Symb;
	Symb.init_formula(define_label, F, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_formula before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);

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

	orbiter_symbol_table_entry Symb;
	Symb.init_collection(define_label, list_of_objects, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_formula before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_collection done" << endl;
	}
}

void symbol_definition::definition_of_combinatorial_object(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_combinatorial_object" << endl;
	}

	combinatorial_object_create *COC;

	COC = NEW_OBJECT(combinatorial_object_create);

	if (f_v) {
		cout << "symbol_definition::definition_of_combinatorial_object before COC->init" << endl;
	}
	COC->init(Combinatorial_object_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_combinatorial_object after COC->init" << endl;
	}



	if (f_v) {
		cout << "symbol_definition::definition_of_combinatorial_object we created a set of " << COC->nb_pts
				<< " points, called " << COC->fname << endl;

#if 0
		cout << "list of points:" << endl;

		cout << COC->nb_pts << endl;
		for (i = 0; i < COC->nb_pts; i++) {
			cout << COC->Pts[i] << " ";
			}
		cout << endl;
#endif
	}


	orbiter_symbol_table_entry Symb;
	Symb.init_combinatorial_object(define_label, COC, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_combinatorial_object before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_combinatorial_object done" << endl;
	}
}


void symbol_definition::definition_of_graph(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbol_definition::definition_of_graph" << endl;
	}

	create_graph *Gr;

	Gr = NEW_OBJECT(create_graph);

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


	orbiter_symbol_table_entry Symb;
	Symb.init_graph(define_label, Gr, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_graph before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



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
	projective_space_with_action *PA;

	idx = Sym->Orbiter_top_level_session->find_symbol(spread_table_label_PA);
	PA = (projective_space_with_action *) Sym->Orbiter_top_level_session->get_object(idx);




	packing_classify *P;

	if (f_v) {
		cout << "symbol_definition::definition_of_spread_table before P->spread_table_init" << endl;
	}

	P = NEW_OBJECT(packing_classify);

	P->spread_table_init(
			PA,
			dimension_of_spread_elements,
			TRUE /* f_select_spread */, spread_selection_text,
			spread_tables_prefix,
			verbose_level);


	if (f_v) {
		cout << "symbol_definition::definition_of_spread_table after do_spread_table_init" << endl;
	}




	orbiter_symbol_table_entry Symb;
	Symb.init_spread_table(define_label, P, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_spread_table before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



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
	packing_classify *P;

	idx = Sym->Orbiter_top_level_session->find_symbol(packing_was_label_spread_table);
	P = (packing_classify *) Sym->Orbiter_top_level_session->get_object(idx);






	packing_was *PW;

	PW = NEW_OBJECT(packing_was);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was before PW->init" << endl;
	}

	PW->init(packing_was_descr, P, verbose_level);

	if (f_v) {
		cout << "symbol_definition::perform_activity after PW->init" << endl;
	}




	orbiter_symbol_table_entry Symb;
	Symb.init_packing_was(define_label, PW, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



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
	packing_was *PW;

	idx = Sym->Orbiter_top_level_session->find_symbol(packing_with_assumed_symmetry_label);
	PW = (packing_was *) Sym->Orbiter_top_level_session->get_object(idx);


	packing_was_fixpoints *PWF;

	PWF = NEW_OBJECT(packing_was_fixpoints);

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




	orbiter_symbol_table_entry Symb;
	Symb.init_packing_was_choose_fixed_points(define_label, PWF, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_packing_was_choose_fixed_points before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



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

	packing_was_fixpoints *PWF;

	idx = Sym->Orbiter_top_level_session->find_symbol(packing_long_orbits_choose_fixed_points_label);
	PWF = (packing_was_fixpoints *) Sym->Orbiter_top_level_session->get_object(idx);


	packing_long_orbits *PL;

	PL = NEW_OBJECT(packing_long_orbits);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_long_orbits before PL->init" << endl;
	}

	PL->init(PWF, Packing_long_orbits_description, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_packing_long_orbits after PL->init" << endl;
	}




	orbiter_symbol_table_entry Symb;

	Symb.init_packing_long_orbits(define_label, PL, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_packing_long_orbits before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



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


	graph_classify *GC;


	GC = NEW_OBJECT(graph_classify);

	if (f_v) {
		cout << "symbol_definition::definition_of_graph_classification before GC->init" << endl;
	}

	GC->init(Graph_classify_description, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_graph_classification after GC->init" << endl;
	}




	orbiter_symbol_table_entry Symb;

	Symb.init_graph_classify(define_label, GC, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_graph_classification before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



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


	diophant_create *Dio;


	Dio = NEW_OBJECT(diophant_create);

	if (f_v) {
		cout << "symbol_definition::definition_of_diophant before Dio->init" << endl;
	}

	Dio->init(Diophant_description, verbose_level);


	if (f_v) {
		cout << "symbol_definition::definition_of_diophant after Dio->init" << endl;
	}




	orbiter_symbol_table_entry Symb;

	Symb.init_diophant(define_label, Dio, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_diophant before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



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


	design_create *DC;


	DC = NEW_OBJECT(design_create);

	if (f_v) {
		cout << "symbol_definition::definition_of_design before DC->init" << endl;
	}

	DC->init(Design_create_description, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_design after DC->init" << endl;
	}




	orbiter_symbol_table_entry Symb;

	Symb.init_design(define_label, DC, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_design before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



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
	design_create *DC;

	idx = Sym->Orbiter_top_level_session->find_symbol(design_table_label_design);
	DC = (design_create *) Sym->Orbiter_top_level_session->get_object(idx);






	strong_generators *Gens;
	Gens = NEW_OBJECT(strong_generators);

	if (f_v) {
		cout << "symbol_definition::definition_of_design_table before Gens->init_from_data_with_go" << endl;
	}
	Gens->init_from_data_with_go(
			DC->A, design_table_generators_data,
			design_table_go_text,
			verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_design_table after Gens->init_from_data_with_go" << endl;
	}


	combinatorics_global Combi;
	design_tables *T;


	if (f_v) {
		cout << "symbol_definition::definition_of_design_table before Combi.create_design_table" << endl;
	}

	Combi.create_design_table(DC,
			design_table_label,
			T,
			Gens,
			verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_design_table after Combi.create_design_table" << endl;
	}



	large_set_classify *LS;

	LS = NEW_OBJECT(large_set_classify);

	LS->init(DC,
			T,
			verbose_level);



	orbiter_symbol_table_entry Symb;
	Symb.init_design_table(define_label, LS, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_design_table before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



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
	large_set_classify *LS;

	idx = Sym->Orbiter_top_level_session->find_symbol(large_set_was_label_design_table);
	LS = (large_set_classify *) Sym->Orbiter_top_level_session->get_object(idx);






	large_set_was *LSW;

	LSW = NEW_OBJECT(large_set_was);

	if (f_v) {
		cout << "symbol_definition::definition_of_large_set_was before LSW->init" << endl;
	}

	LSW->init(large_set_was_descr, LS, verbose_level);

	if (f_v) {
		cout << "symbol_definition::definition_of_large_set_was after LSW->init" << endl;
	}




	orbiter_symbol_table_entry Symb;
	Symb.init_large_set_was(define_label, LSW, verbose_level);
	if (f_v) {
		cout << "symbol_definition::definition_of_large_set_was before add_symbol_table_entry" << endl;
	}
	Sym->Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



	if (f_v) {
		cout << "symbol_definition::definition_of_large_set_was done" << endl;
	}
}


}}


