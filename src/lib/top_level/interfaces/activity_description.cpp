/*
 * activity_description.cpp
 *
 *  Created on: Jun 20, 2021
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


activity_description::activity_description()
{
	Sym = NULL;

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

activity_description::~activity_description()
{

}


void activity_description::read_arguments(
		interface_symbol_table *Sym,
		int argc, std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::read_arguments" << endl;
	}

	activity_description::Sym = Sym;

	if (stringcmp(argv[i], "-finite_field_activity") == 0) {
		f_finite_field_activity = TRUE;
		Finite_field_activity_description =
				NEW_OBJECT(finite_field_activity_description);
		if (f_v) {
			cout << "reading -finite_field_activity" << endl;
		}
		i += Finite_field_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-finite_field_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (stringcmp(argv[i], "-projective_space_activity") == 0) {
		f_projective_space_activity = TRUE;
		Projective_space_activity_description =
				NEW_OBJECT(projective_space_activity_description);
		if (f_v) {
			cout << "reading -projective_space_activity" << endl;
		}
		i += Projective_space_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-projective_space_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (stringcmp(argv[i], "-orthogonal_space_activity") == 0) {
		f_orthogonal_space_activity = TRUE;
		Orthogonal_space_activity_description =
				NEW_OBJECT(orthogonal_space_activity_description);
		if (f_v) {
			cout << "reading -orthogonal_space_activity" << endl;
		}
		i += Orthogonal_space_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-orthogonal_space_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (stringcmp(argv[i], "-group_theoretic_activity") == 0) {
		f_group_theoretic_activity = TRUE;
		Group_theoretic_activity_description =
				NEW_OBJECT(group_theoretic_activity_description);
		if (f_v) {
			cout << "reading -group_theoretic_activities" << endl;
		}
		i += Group_theoretic_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-group_theoretic_activities" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (stringcmp(argv[i], "-cubic_surface_activity") == 0) {
		f_cubic_surface_activity = TRUE;
		Cubic_surface_activity_description =
				NEW_OBJECT(cubic_surface_activity_description);
		if (f_v) {
			cout << "reading -cubic_surface_activity" << endl;
		}
		i += Cubic_surface_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-cubic_surface_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (stringcmp(argv[i], "-quartic_curve_activity") == 0) {
		f_quartic_curve_activity = TRUE;
		Quartic_curve_activity_description =
				NEW_OBJECT(quartic_curve_activity_description);
		if (f_v) {
			cout << "reading -quartic_curve_activity" << endl;
		}
		i += Quartic_curve_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-quartic_curve_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (stringcmp(argv[i], "-combinatorial_object_activity") == 0) {
		f_combinatorial_object_activity = TRUE;
		Combinatorial_object_activity_description =
				NEW_OBJECT(combinatorial_object_activity_description);
		if (f_v) {
			cout << "reading -combinatorial_object_activity" << endl;
		}
		i += Combinatorial_object_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-combinatorial_object_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (stringcmp(argv[i], "-graph_theoretic_activity") == 0) {
		f_graph_theoretic_activity = TRUE;
		Graph_theoretic_activity_description =
				NEW_OBJECT(graph_theoretic_activity_description);
		if (f_v) {
			cout << "reading -graph_theoretic_activity" << endl;
		}
		i += Graph_theoretic_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-graph_theoretic_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (stringcmp(argv[i], "-classification_of_cubic_surfaces_with_double_sixes_activity") == 0) {
		f_classification_of_cubic_surfaces_with_double_sixes_activity = TRUE;
		Classification_of_cubic_surfaces_with_double_sixes_activity_description =
				NEW_OBJECT(classification_of_cubic_surfaces_with_double_sixes_activity_description);
		if (f_v) {
			cout << "reading -classification_of_cubic_surfaces_with_double_sixes_activity" << endl;
		}
		i += Classification_of_cubic_surfaces_with_double_sixes_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-classification_of_cubic_surfaces_with_double_sixes_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (stringcmp(argv[i], "-spread_table_activity") == 0) {
		f_spread_table_activity = TRUE;
		Spread_table_activity_description =
				NEW_OBJECT(spread_table_activity_description);
		if (f_v) {
			cout << "reading -classification_of_cubic_surfaces_with_double_sixes_activity" << endl;
		}
		i += Spread_table_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-spread_table_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (stringcmp(argv[i], "-packing_with_symmetry_assumption_activity") == 0) {
		f_packing_with_symmetry_assumption_activity = TRUE;
		Packing_was_activity_description =
				NEW_OBJECT(packing_was_activity_description);
		if (f_v) {
			cout << "reading -packing_with_symmetry_assumption_activity" << endl;
		}
		i += Packing_was_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-packing_with_symmetry_assumption_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (stringcmp(argv[i], "-packing_fixed_points_activity") == 0) {
		f_packing_fixed_points_activity = TRUE;
		Packing_was_fixpoints_activity_description =
				NEW_OBJECT(packing_was_fixpoints_activity_description);
		if (f_v) {
			cout << "reading -packing_fixed_points_activity" << endl;
		}
		i += Packing_was_fixpoints_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-packing_fixed_points_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (stringcmp(argv[i], "-graph_classification_activity") == 0) {
		f_graph_classification_activity = TRUE;
		Graph_classification_activity_description =
				NEW_OBJECT(graph_classification_activity_description);
		if (f_v) {
			cout << "reading -graph_classification_activity" << endl;
		}
		i += Graph_classification_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-graph_classification_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (stringcmp(argv[i], "-diophant_activity") == 0) {
		f_diophant_activity = TRUE;
		Diophant_activity_description =
				NEW_OBJECT(diophant_activity_description);
		if (f_v) {
			cout << "reading -diophant_activity" << endl;
		}
		i += Diophant_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-diophant_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (stringcmp(argv[i], "-design_activity") == 0) {
		f_design_activity = TRUE;
		Design_activity_description =
				NEW_OBJECT(design_activity_description);
		if (f_v) {
			cout << "reading -design_activity" << endl;
		}
		i += Design_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-design_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (stringcmp(argv[i], "-large_set_with_symmetry_assumption_activity") == 0) {
		f_large_set_was_activity = TRUE;
		Large_set_was_activity_description =
				NEW_OBJECT(large_set_was_activity_description);
		if (f_v) {
			cout << "reading -large_set_with_symmetry_assumption_activity" << endl;
		}
		i += Large_set_was_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-large_set_with_symmetry_assumption_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}

	else {
		cout << "unrecognized activity after -do : " << argv[i] << endl;
		exit(1);
	}

	if (f_v) {
		cout << "activity_description::read_arguments done" << endl;
	}
}


void activity_description::worker(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::worker" << endl;
	}

	if (f_finite_field_activity) {

		if (f_v) {
			cout << "activity_description::worker f_finite_field_activity" << endl;
		}
		do_finite_field_activity(verbose_level);

	}
	else if (f_projective_space_activity) {

		if (f_v) {
			cout << "activity_description::worker f_projective_space_activity" << endl;
		}
		do_projective_space_activity(verbose_level);

	}
	else if (f_orthogonal_space_activity) {

		if (f_v) {
			cout << "activity_description::worker f_orthogonal_space_activity" << endl;
		}
		do_orthogonal_space_activity(verbose_level);

	}
	else if (f_group_theoretic_activity) {

		if (f_v) {
			cout << "activity_description::worker f_group_theoretic_activity" << endl;
		}
		do_group_theoretic_activity(verbose_level);

	}
	else if (f_cubic_surface_activity) {

		if (f_v) {
			cout << "activity_description::worker f_cubic_surface_activity" << endl;
		}
		do_cubic_surface_activity(verbose_level);

	}
	else if (f_quartic_curve_activity) {

		if (f_v) {
			cout << "activity_description::worker f_quartic_curve_activity" << endl;
		}
		do_quartic_curve_activity(verbose_level);

	}
	else if (f_combinatorial_object_activity) {

		if (f_v) {
			cout << "activity_description::worker f_combinatorial_object_activity" << endl;
		}
		do_combinatorial_object_activity(verbose_level);

	}
	else if (f_graph_theoretic_activity) {

		if (f_v) {
			cout << "activity_description::worker f_graph_theoretic_activity" << endl;
		}
		do_graph_theoretic_activity(verbose_level);

	}
	else if (f_classification_of_cubic_surfaces_with_double_sixes_activity) {

		if (f_v) {
			cout << "activity_description::worker f_classification_of_cubic_surfaces_with_double_sixes_activity" << endl;
		}

		do_classification_of_cubic_surfaces_with_double_sixes_activity(verbose_level);
	}
	else if (f_spread_table_activity) {

		if (f_v) {
			cout << "activity_description::worker f_spread_table_activity" << endl;
		}

		do_spread_table_activity(verbose_level);
	}
	else if (f_packing_with_symmetry_assumption_activity) {

		if (f_v) {
			cout << "activity_description::worker f_packing_with_symmetry_activity" << endl;
		}

		do_packing_was_activity(verbose_level);
	}
	else if (f_packing_fixed_points_activity) {

		if (f_v) {
			cout << "activity_description::worker f_packing_with_symmetry_activity" << endl;
		}

		do_packing_fixed_points_activity(verbose_level);
	}
	else if (f_graph_classification_activity) {

		if (f_v) {
			cout << "activity_description::worker f_graph_classification_activity" << endl;
		}

		do_graph_classification_activity(verbose_level);
	}
	else if (f_diophant_activity) {

		if (f_v) {
			cout << "activity_description::worker f_diophant_activity" << endl;
		}

		do_diophant_activity(verbose_level);
	}
	else if (f_design_activity) {

		if (f_v) {
			cout << "activity_description::worker f_design_activity" << endl;
		}

		do_design_activity(verbose_level);

	}
	else if (f_large_set_was_activity) {

		if (f_v) {
			cout << "activity_description::worker f_large_set_was_activity" << endl;
		}

		do_large_set_was_activity(verbose_level);
	}


	if (f_v) {
		cout << "activity_description::worker done" << endl;
	}
}

void activity_description::print()
{

	cout << "-with ";
	Sym->print_with();
	cout << "-do " << endl;

	if (f_finite_field_activity) {
		cout << "-finite_field_activity ";
		Finite_field_activity_description->print();
	}
	else if (f_projective_space_activity) {
		cout << "-projective_space_activity ";
		Projective_space_activity_description->print();
	}
	else if (f_orthogonal_space_activity) {
		cout << "-orthogonal_space_activity ";
		Orthogonal_space_activity_description->print();
	}
	else if (f_group_theoretic_activity) {
		cout << "-group_theoretic_activities ";
		Group_theoretic_activity_description->print();
	}
	else if (f_cubic_surface_activity) {
		cout << "-cubic_surface_activity ";
		Cubic_surface_activity_description->print();
	}
	else if (f_quartic_curve_activity) {
		cout << "-quartic_curve_activity ";
		Quartic_curve_activity_description->print();
	}
	else if (f_combinatorial_object_activity) {
		cout << "-combinatorial_object_activity ";
		Combinatorial_object_activity_description->print();
	}
	else if (f_graph_theoretic_activity) {
		cout << "-graph_theoretic_activity ";
		Graph_theoretic_activity_description->print();
	}
	else if (f_classification_of_cubic_surfaces_with_double_sixes_activity) {
		cout << "-classification_of_cubic_surfaces_with_double_sixes_activity ";
		Classification_of_cubic_surfaces_with_double_sixes_activity_description->print();
	}
	else if (f_spread_table_activity) {
		cout << "-spread_table_activity ";
		Spread_table_activity_description->print();
	}
	else if (f_packing_with_symmetry_assumption_activity) {
		cout << "-packing_with_symmetry_assumption_activity ";
		Packing_was_activity_description->print();
	}
	else if (f_packing_fixed_points_activity) {
		cout << "-packing_fixed_points_activity ";
		Packing_was_fixpoints_activity_description->print();
	}
	else if (f_graph_classification_activity) {
		cout << "-graph_classification_activity ";
		Graph_classification_activity_description->print();
	}
	else if (f_diophant_activity) {
		cout << "-diophant_activity ";
		Diophant_activity_description->print();
	}
	else if (f_design_activity) {
		cout << "-design_activity ";
		Design_activity_description->print();
	}
	else if (f_large_set_was_activity) {
		cout << "-large_set_with_symmetry_assumption_activity ";
		Large_set_was_activity_description->print();
	}
}



void activity_description::do_finite_field_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_finite_field_activity "
				"finite field activity for the following objects: ";
		Sym->print_with();
	}

	int *Idx;


	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "-finite_field_activity requires at least one input" << endl;
		exit(1);
	}
	finite_field *F;

	F = (finite_field *) Sym->Orbiter_top_level_session->get_object(Idx[0]);

	finite_field_activity FA;
	FA.init(Finite_field_activity_description, F, verbose_level);
#if 0
	Finite_field_activity_description->f_q = TRUE;
	Finite_field_activity_description->q = F->q;
	FA.Descr = Finite_field_activity_description;
	FA.F = F;
#endif

	if (Sym->with_labels.size() == 2) {
		cout << "-finite_field_activity has two inputs" << endl;
		FA.F_secondary = (finite_field *) Sym->Orbiter_top_level_session->get_object(Idx[1]);
	}



	if (f_v) {
		cout << "activity_description::do_finite_field_activity "
				"before FA.perform_activity" << endl;
	}
	FA.perform_activity(verbose_level);
	if (f_v) {
		cout << "activity_description::do_finite_field_activity "
				"after FA.perform_activity" << endl;
	}

	FREE_int(Idx);

}


void activity_description::do_projective_space_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_projective_space_activity "
				"projective space activity for the following objects: ";
		Sym->print_with();
	}

	int *Idx;


	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "-finite_field_activity requires at least one input" << endl;
		exit(1);
	}
	projective_space_with_action *PA;

	PA = (projective_space_with_action *) Sym->Orbiter_top_level_session->get_object(Idx[0]);

	projective_space_activity Activity;
	Activity.Descr = Projective_space_activity_description;
	Activity.PA = PA;

#if 0
	if (with_labels.size() == 2) {
		cout << "-finite_field_activity has two inputs" << endl;
		FA.F_secondary = (finite_field *) Orbiter_top_level_session->get_object(Idx[1]);
	}
#endif


	if (f_v) {
		cout << "activity_description::do_projective_space_activity "
				"before Activity.perform_activity" << endl;
	}
	Activity.perform_activity(verbose_level);
	if (f_v) {
		cout << "activity_description::do_projective_space_activity "
				"after Activity.perform_activity" << endl;
	}

	FREE_int(Idx);

}

void activity_description::do_orthogonal_space_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_orthogonal_space_activity "
				"orthogonal space activity for the following objects: ";
		Sym->print_with();
	}

	int *Idx;


	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "-finite_field_activity requires at least one input" << endl;
		exit(1);
	}
	orthogonal_space_with_action *OA;

	OA = (orthogonal_space_with_action *) Sym->Orbiter_top_level_session->get_object(Idx[0]);

	orthogonal_space_activity Activity;
	Activity.Descr = Orthogonal_space_activity_description;
	Activity.OA = OA;

#if 0
	if (with_labels.size() == 2) {
		cout << "-finite_field_activity has two inputs" << endl;
		FA.F_secondary = (finite_field *) Orbiter_top_level_session->get_object(Idx[1]);
	}
#endif


	if (f_v) {
		cout << "activity_description::do_orthogonal_space_activity "
				"before Activity.perform_activity" << endl;
	}
	Activity.perform_activity(verbose_level);
	if (f_v) {
		cout << "activity_description::do_orthogonal_space_activity "
				"after Activity.perform_activity" << endl;
	}

	FREE_int(Idx);

}

void activity_description::do_group_theoretic_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_group_theoretic_activity "
				"finite field activity for the following objects:";
		Sym->print_with();
	}

	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "-group_theoretic_activity requires at least one input" << endl;
		exit(1);
	}


	symbol_table_object_type type;

	type = Sym->Orbiter_top_level_session->get_object_type(Idx[0]);

	if (type != t_any_group) {
		cout << "activity_description::do_group_theoretic_activity type is not t_any_group" << endl;
		exit(1);
	}

	any_group *AG;

	AG = (any_group *) Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		group_theoretic_activity Activity;

		Activity.init_group(Group_theoretic_activity_description, AG, verbose_level);

		if (f_v) {
			cout << "activity_description::do_group_theoretic_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_group_theoretic_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_group_theoretic_activity done" << endl;
	}

}

void activity_description::do_cubic_surface_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_cubic_surface_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "-group_theoretic_activity requires at least one input" << endl;
		exit(1);
	}

	surface_create *SC;

	SC = (surface_create *) Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		cubic_surface_activity Activity;

		Activity.init(Cubic_surface_activity_description, SC, verbose_level);

		if (f_v) {
			cout << "activity_description::do_cubic_surface_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_cubic_surface_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_cubic_surface_activity done" << endl;
	}

}


void activity_description::do_quartic_curve_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_quartic_curve_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "-group_theoretic_activity requires at least one input" << endl;
		exit(1);
	}

	quartic_curve_create *QC;

	QC = (quartic_curve_create *) Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		quartic_curve_activity Activity;

		Activity.init(Quartic_curve_activity_description, QC, verbose_level);

		if (f_v) {
			cout << "activity_description::do_quartic_curve_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_quartic_curve_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_quartic_curve_activity done" << endl;
	}

}


void activity_description::do_combinatorial_object_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_combinatorial_object_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "-combinatorial_object_activity requires at least one input" << endl;
		exit(1);
	}

	symbol_table_object_type t;

	t = Sym->Orbiter_top_level_session->get_object_type(Idx[0]);
	if (t == t_geometric_object) {
		geometric_object_create *GOC;

		GOC = (geometric_object_create *) Sym->Orbiter_top_level_session->get_object(Idx[0]);
		{
			combinatorial_object_activity Activity;

			Activity.init(Combinatorial_object_activity_description, GOC, verbose_level);

			if (f_v) {
				cout << "activity_description::do_combinatorial_object_activity "
						"before Activity.perform_activity" << endl;
			}
			Activity.perform_activity(verbose_level);
			if (f_v) {
				cout << "activity_description::do_combinatorial_object_activity "
						"after Activity.perform_activity" << endl;
			}

		}
	}
	else if (t == t_combinatorial_objects) {
		data_input_stream *IS;

		IS = (data_input_stream *) Sym->Orbiter_top_level_session->get_object(Idx[0]);
		{
			combinatorial_object_activity Activity;

			Activity.init_input_stream(Combinatorial_object_activity_description, IS, verbose_level);

			if (f_v) {
				cout << "activity_description::do_combinatorial_object_activity "
						"before Activity.perform_activity" << endl;
			}
			Activity.perform_activity(verbose_level);
			if (f_v) {
				cout << "activity_description::do_combinatorial_object_activity "
						"after Activity.perform_activity" << endl;
			}

		}
	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_combinatorial_object_activity done" << endl;
	}

}

void activity_description::do_graph_theoretic_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_graph_theoretic_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	//create_graph *Gr;
	colored_graph *CG;

	CG = (colored_graph *) Sym->Orbiter_top_level_session->get_object(Idx[0]);
	if (f_v) {
		cout << "activity_description::do_graph_theoretic_activity CG->label = " << CG->label << endl;
	}

	{
		graph_theoretic_activity Activity;

		Activity.init(Graph_theoretic_activity_description, CG, verbose_level);

		if (f_v) {
			cout << "activity_description::do_graph_theoretic_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_graph_theoretic_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_graph_theoretic_activity done" << endl;
	}

}

void activity_description::do_classification_of_cubic_surfaces_with_double_sixes_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_classification_of_cubic_surfaces_with_double_sixes_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	surface_classify_wedge *SCW;

	SCW = (surface_classify_wedge *) Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		classification_of_cubic_surfaces_with_double_sixes_activity Activity;

		Activity.init(Classification_of_cubic_surfaces_with_double_sixes_activity_description, SCW, verbose_level);

		if (f_v) {
			cout << "activity_description::do_classification_of_cubic_surfaces_with_double_sixes_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_classification_of_cubic_surfaces_with_double_sixes_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_classification_of_cubic_surfaces_with_double_sixes_activity done" << endl;
	}

}

void activity_description::do_spread_table_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_spread_table_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	packing_classify *P;

	P = (packing_classify *) Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		spread_table_activity Activity;

		Activity.init(Spread_table_activity_description, P, verbose_level);

		if (f_v) {
			cout << "activity_description::do_spread_table_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_spread_table_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_spread_table_activity done" << endl;
	}

}

void activity_description::do_packing_was_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_packing_was_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	packing_was *PW;

	PW = (packing_was *) Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		packing_was_activity Activity;

		Activity.init(Packing_was_activity_description, PW, verbose_level);

		if (f_v) {
			cout << "activity_description::do_packing_was_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_packing_was_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_packing_was_activity done" << endl;
	}

}



void activity_description::do_packing_fixed_points_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_packing_fixed_points_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	packing_was_fixpoints *PWF;

	PWF = (packing_was_fixpoints *) Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		packing_was_fixpoints_activity Activity;

		Activity.init(Packing_was_fixpoints_activity_description, PWF, verbose_level);

		if (f_v) {
			cout << "activity_description::do_packing_fixed_points_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_packing_fixed_points_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_packing_fixed_points_activity done" << endl;
	}

}


void activity_description::do_graph_classification_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_graph_classification_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	graph_classify *GC;

	GC = (graph_classify *) Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		graph_classification_activity Activity;

		Activity.init(Graph_classification_activity_description, GC, verbose_level);

		if (f_v) {
			cout << "activity_description::do_graph_classification_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_graph_classification_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_graph_classification_activity done" << endl;
	}

}

void activity_description::do_diophant_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_diophant_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	diophant_create *Dio;

	Dio = (diophant_create *) Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		diophant_activity Activity;


		if (f_v) {
			cout << "activity_description::do_diophant_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(Diophant_activity_description, Dio->D, verbose_level);
		if (f_v) {
			cout << "activity_description::do_diophant_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_diophant_activity done" << endl;
	}

}

void activity_description::do_design_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_design_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	design_create *DC;

	DC = (design_create *) Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		design_activity Activity;


		if (f_v) {
			cout << "activity_description::do_design_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(Design_activity_description, DC, verbose_level);
		if (f_v) {
			cout << "activity_description::do_design_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_design_activity done" << endl;
	}

}

void activity_description::do_large_set_was_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_large_set_was_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	large_set_was *LSW;

	LSW = (large_set_was *) Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		large_set_was_activity Activity;

		if (f_v) {
			cout << "activity_description::do_large_set_was_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(Large_set_was_activity_description, LSW, verbose_level);
		if (f_v) {
			cout << "activity_description::do_large_set_was_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_large_set_was_activity done" << endl;
	}

}




}}


