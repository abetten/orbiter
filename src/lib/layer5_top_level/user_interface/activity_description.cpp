/*
 * activity_description.cpp
 *
 *  Created on: Jun 20, 2021
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace user_interface {


activity_description::activity_description()
{
	Record_birth();
	Sym = NULL;

	f_finite_field_activity = false;
	Finite_field_activity_description = NULL;

	f_polynomial_ring_activity = false;
	Polynomial_ring_activity_description = NULL;

	f_projective_space_activity = false;
	Projective_space_activity_description = NULL;

	f_orthogonal_space_activity = false;
	Orthogonal_space_activity_description = NULL;

	f_group_theoretic_activity = false;
	Group_theoretic_activity_description = NULL;

	f_coding_theoretic_activity = false;
	Coding_theoretic_activity_description = NULL;

	f_cubic_surface_activity = false;
	Cubic_surface_activity_description = NULL;

	f_quartic_curve_activity = false;
	Quartic_curve_activity_description = NULL;

	f_blt_set_activity = false;
	Blt_set_activity_description = NULL;

	f_combinatorial_object_activity = false;
	Combinatorial_object_activity_description = NULL;

	f_graph_theoretic_activity = false;
	Graph_theoretic_activity_description = NULL;

	f_classification_of_cubic_surfaces_with_double_sixes_activity = false;
	Classification_of_cubic_surfaces_with_double_sixes_activity_description = NULL;

	f_spread_table_activity = false;
	Spread_table_activity_description = NULL;



	// orbiter_activities2.csv


	f_packing_classify_activity_description = false;
	Packing_classify_activity_description = NULL;

	f_packing_with_symmetry_assumption_activity = false;
	Packing_was_activity_description = NULL;

	f_packing_fixed_points_activity = false;
	Packing_was_fixpoints_activity_description = NULL;

	f_graph_classification_activity = false;
	Graph_classification_activity_description = NULL;

	f_diophant_activity = false;
	Diophant_activity_description = NULL;

	f_design_activity = false;
	Design_activity_description = NULL;


	f_large_set_was_activity = false;
	Large_set_was_activity_description = NULL;

	f_symbolic_object_activity = false;
	Symbolic_object_activity_description = NULL;

	f_BLT_set_classify_activity = false;
	Blt_set_classify_activity_description = NULL;

	f_spread_classify_activity = false;
	Spread_classify_activity_description = NULL;

	f_spread_activity = false;
	Spread_activity_description = NULL;

	f_translation_plane_activity = false;
	Translation_plane_activity_description = NULL;

	f_action_on_forms_activity = false;
	Action_on_forms_activity_description = NULL;

	f_orbits_activity = false;
	Orbits_activity_description = NULL;

	f_variety_activity = false;
	Variety_activity_description = NULL;

	f_vector_ge_activity = false;
	Vector_ge_activity_description = NULL;

	f_combo_activity = false;
	Combo_activity_description = NULL;


}

activity_description::~activity_description()
{
	Record_death();

}


void activity_description::read_arguments(
		//interface_symbol_table *Sym,
		int argc, std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::read_arguments" << endl;
	}
	other::data_structures::string_tools ST;

	//activity_description::Sym = Sym;

	if (ST.stringcmp(argv[i], "-finite_field_activity") == 0) {
		f_finite_field_activity = true;
		Finite_field_activity_description =
				NEW_OBJECT(algebra::field_theory::finite_field_activity_description);
		if (f_v) {
			cout << "reading -finite_field_activity" << endl;
		}
		i += Finite_field_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-finite_field_activity" << endl;
			Finite_field_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-ring_theoretic_activity") == 0) {
		f_polynomial_ring_activity = true;
		Polynomial_ring_activity_description =
				NEW_OBJECT(algebra::ring_theory::polynomial_ring_activity_description);
		if (f_v) {
			cout << "reading -ring_theoretic_activity" << endl;
		}
		i += Polynomial_ring_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-ring_theoretic_activity" << endl;
			Polynomial_ring_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}

	else if (ST.stringcmp(argv[i], "-projective_space_activity") == 0) {
		f_projective_space_activity = true;
		Projective_space_activity_description =
				NEW_OBJECT(projective_geometry::projective_space_activity_description);
		if (f_v) {
			cout << "reading -projective_space_activity" << endl;
		}
		i += Projective_space_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-projective_space_activity" << endl;
			Projective_space_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-orthogonal_space_activity") == 0) {
		f_orthogonal_space_activity = true;
		Orthogonal_space_activity_description =
				NEW_OBJECT(orthogonal_geometry_applications::orthogonal_space_activity_description);
		if (f_v) {
			cout << "reading -orthogonal_space_activity" << endl;
		}
		i += Orthogonal_space_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-orthogonal_space_activity" << endl;
			Orthogonal_space_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-group_theoretic_activity") == 0) {
		f_group_theoretic_activity = true;
		Group_theoretic_activity_description =
				NEW_OBJECT(apps_algebra::group_theoretic_activity_description);
		if (f_v) {
			cout << "reading -group_theoretic_activities" << endl;
		}
		i += Group_theoretic_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-group_theoretic_activities" << endl;
			Group_theoretic_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-coding_theoretic_activity") == 0) {
		f_coding_theoretic_activity = true;
		Coding_theoretic_activity_description =
				NEW_OBJECT(apps_coding_theory::coding_theoretic_activity_description);
		if (f_v) {
			cout << "reading -coding_theoretic_activities" << endl;
		}
		i += Coding_theoretic_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-coding_theoretic_activities" << endl;
			Coding_theoretic_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-cubic_surface_activity") == 0) {
		f_cubic_surface_activity = true;
		Cubic_surface_activity_description =
				NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_in_general::cubic_surface_activity_description);
		if (f_v) {
			cout << "reading -cubic_surface_activity" << endl;
		}
		i += Cubic_surface_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-cubic_surface_activity" << endl;
			Cubic_surface_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-quartic_curve_activity") == 0) {
		f_quartic_curve_activity = true;
		Quartic_curve_activity_description =
				NEW_OBJECT(applications_in_algebraic_geometry::quartic_curves::quartic_curve_activity_description);
		if (f_v) {
			cout << "reading -quartic_curve_activity" << endl;
		}
		i += Quartic_curve_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-quartic_curve_activity" << endl;
			Quartic_curve_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-blt_set_activity") == 0) {
		f_blt_set_activity = true;
		Blt_set_activity_description =
				NEW_OBJECT(orthogonal_geometry_applications::blt_set_activity_description);
		if (f_v) {
			cout << "reading -blt_set_activity" << endl;
		}
		i += Blt_set_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-blt_set_activity" << endl;
			Blt_set_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}

	else if (ST.stringcmp(argv[i], "-combinatorial_object_activity") == 0) {
		f_combinatorial_object_activity = true;
		Combinatorial_object_activity_description =
				NEW_OBJECT(apps_combinatorics::combinatorial_object_activity_description);
		if (f_v) {
			cout << "reading -combinatorial_object_activity" << endl;
		}
		i += Combinatorial_object_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-combinatorial_object_activity" << endl;
			Combinatorial_object_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-graph_theoretic_activity") == 0) {
		f_graph_theoretic_activity = true;
		Graph_theoretic_activity_description =
				NEW_OBJECT(apps_graph_theory::graph_theoretic_activity_description);
		if (f_v) {
			cout << "reading -graph_theoretic_activity" << endl;
		}
		i += Graph_theoretic_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-graph_theoretic_activity" << endl;
			Graph_theoretic_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-classification_of_cubic_surfaces_with_double_sixes_activity") == 0) {
		f_classification_of_cubic_surfaces_with_double_sixes_activity = true;
		Classification_of_cubic_surfaces_with_double_sixes_activity_description =
				NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_and_double_sixes::classification_of_cubic_surfaces_with_double_sixes_activity_description);
		if (f_v) {
			cout << "reading -classification_of_cubic_surfaces_with_double_sixes_activity" << endl;
		}
		i += Classification_of_cubic_surfaces_with_double_sixes_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-classification_of_cubic_surfaces_with_double_sixes_activity" << endl;
			Classification_of_cubic_surfaces_with_double_sixes_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-spread_table_activity") == 0) {
		f_spread_table_activity = true;
		Spread_table_activity_description =
				NEW_OBJECT(spreads::spread_table_activity_description);
		if (f_v) {
			cout << "reading -classification_of_cubic_surfaces_with_double_sixes_activity" << endl;
		}
		i += Spread_table_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-spread_table_activity" << endl;
			Spread_table_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-packing_classify_activity") == 0) {
		f_packing_classify_activity_description = true;
		Packing_classify_activity_description =
				NEW_OBJECT(packings::packing_classify_activity_description);
		if (f_v) {
			cout << "reading -packing_classify_activity_description" << endl;
		}
		i += Packing_classify_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-packing_classify_activity_description" << endl;
			Packing_classify_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}

	else if (ST.stringcmp(argv[i], "-packing_with_symmetry_assumption_activity") == 0) {
		f_packing_with_symmetry_assumption_activity = true;
		Packing_was_activity_description =
				NEW_OBJECT(packings::packing_was_activity_description);
		if (f_v) {
			cout << "reading -packing_with_symmetry_assumption_activity" << endl;
		}
		i += Packing_was_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-packing_with_symmetry_assumption_activity" << endl;
			Packing_was_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-packing_fixed_points_activity") == 0) {
		f_packing_fixed_points_activity = true;
		Packing_was_fixpoints_activity_description =
				NEW_OBJECT(packings::packing_was_fixpoints_activity_description);
		if (f_v) {
			cout << "reading -packing_fixed_points_activity" << endl;
		}
		i += Packing_was_fixpoints_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-packing_fixed_points_activity" << endl;
			Packing_was_fixpoints_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-graph_classification_activity") == 0) {
		f_graph_classification_activity = true;
		Graph_classification_activity_description =
				NEW_OBJECT(apps_graph_theory::graph_classification_activity_description);
		if (f_v) {
			cout << "reading -graph_classification_activity" << endl;
		}
		i += Graph_classification_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-graph_classification_activity" << endl;
			Graph_classification_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-diophant_activity") == 0) {
		f_diophant_activity = true;
		Diophant_activity_description =
				NEW_OBJECT(combinatorics::solvers::diophant_activity_description);
		if (f_v) {
			cout << "reading -diophant_activity" << endl;
		}
		i += Diophant_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-diophant_activity" << endl;
			Diophant_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-design_activity") == 0) {
		f_design_activity = true;
		Design_activity_description =
				NEW_OBJECT(apps_combinatorics::design_activity_description);
		if (f_v) {
			cout << "reading -design_activity" << endl;
		}
		i += Design_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-design_activity" << endl;
			Design_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-large_set_with_symmetry_assumption_activity") == 0) {
		f_large_set_was_activity = true;
		Large_set_was_activity_description =
				NEW_OBJECT(apps_combinatorics::large_set_was_activity_description);
		if (f_v) {
			cout << "reading -large_set_with_symmetry_assumption_activity" << endl;
		}
		i += Large_set_was_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-large_set_with_symmetry_assumption_activity" << endl;
			Large_set_was_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-symbolic_object_activity") == 0) {
		f_symbolic_object_activity = true;
		Symbolic_object_activity_description =
				NEW_OBJECT(algebra::expression_parser::symbolic_object_activity_description);
		if (f_v) {
			cout << "reading -symbolic_object_activity" << endl;
		}
		i += Symbolic_object_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-symbolic_object_activity" << endl;
			Symbolic_object_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-BLT_set_classify_activity") == 0) {
		f_BLT_set_classify_activity = true;
		Blt_set_classify_activity_description =
				NEW_OBJECT(orthogonal_geometry_applications::blt_set_classify_activity_description);
		if (f_v) {
			cout << "reading -BLT_set_classify_activity" << endl;
		}
		i += Blt_set_classify_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-BLT_set_classify_activity" << endl;
			Blt_set_classify_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-spread_classify_activity") == 0) {
		f_spread_classify_activity = true;
		Spread_classify_activity_description =
				NEW_OBJECT(spreads::spread_classify_activity_description);
		if (f_v) {
			cout << "reading -spread_classify_activity" << endl;
		}
		i += Spread_classify_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-spread_classify_activity" << endl;
			Spread_classify_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-spread_activity") == 0) {
		f_spread_activity = true;
		Spread_activity_description =
				NEW_OBJECT(spreads::spread_activity_description);
		if (f_v) {
			cout << "reading -spread_activity" << endl;
		}
		i += Spread_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-spread_activity" << endl;
			Spread_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-translation_plane_activity") == 0) {
		f_translation_plane_activity = true;
		Translation_plane_activity_description =
				NEW_OBJECT(spreads::translation_plane_activity_description);
		if (f_v) {
			cout << "reading -translation_plane_activity" << endl;
		}
		i += Translation_plane_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-translation_plane_activity" << endl;
			Translation_plane_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-action_on_forms_activity") == 0) {
		f_action_on_forms_activity = true;
		Action_on_forms_activity_description =
				NEW_OBJECT(apps_algebra::action_on_forms_activity_description);
		if (f_v) {
			cout << "reading -action_on_forms_activity" << endl;
		}
		i += Action_on_forms_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-action_on_forms_activity" << endl;
			Action_on_forms_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-orbits_activity") == 0) {
		f_orbits_activity = true;
		Orbits_activity_description =
				NEW_OBJECT(orbits::orbits_activity_description);
		if (f_v) {
			cout << "reading -orbits_activity" << endl;
		}
		i += Orbits_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-orbits_activity" << endl;
			Orbits_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-variety_activity") == 0) {
		f_variety_activity = true;
		Variety_activity_description =
				NEW_OBJECT(canonical_form::variety_activity_description);
		if (f_v) {
			cout << "reading -variety_activity" << endl;
		}
		i += Variety_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-variety_activity" << endl;
			Variety_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-vector_ge_activity") == 0) {
		f_vector_ge_activity = true;
		Vector_ge_activity_description =
				NEW_OBJECT(apps_algebra::vector_ge_activity_description);
		if (f_v) {
			cout << "reading -vector_ge_activity" << endl;
		}
		i += Vector_ge_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-vector_ge_activity" << endl;
			Vector_ge_activity_description->print();
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-combo_activity") == 0) {
		f_combo_activity = true;
		Combo_activity_description =
				NEW_OBJECT(apps_combinatorics::combo_activity_description);
		if (f_v) {
			cout << "reading -combo_activity" << endl;
		}
		i += Combo_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		i++;

		if (f_v) {
			cout << "-combo_activity" << endl;
			Combo_activity_description->print();
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


void activity_description::worker(
		std::vector<std::string> &with_labels,
		int &nb_output,
		other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::worker" << endl;
		cout << "activity_description::worker verbose_level = " << verbose_level << endl;
		cout << "activity_description::worker number of inputs = " << with_labels.size() << endl;
		int i;
		for (i = 0; i < with_labels.size(); i++) {
			cout << with_labels[i];
			if (i < with_labels.size() - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}


	other::orbiter_kernel_system::activity_output *AO = NULL;

	if (f_finite_field_activity) {

		if (f_v) {
			cout << "activity_description::worker f_finite_field_activity" << endl;
		}
		do_finite_field_activity(AO, verbose_level);

	}

	else if (f_polynomial_ring_activity) {

		if (f_v) {
			cout << "activity_description::worker ring_theoretic_activity" << endl;
		}
		do_ring_theoretic_activity(nb_output, Output, verbose_level);

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
		do_group_theoretic_activity(nb_output, Output, verbose_level);

	}
	else if (f_coding_theoretic_activity) {

		if (f_v) {
			cout << "activity_description::worker f_coding_theoretic_activity" << endl;
		}
		do_coding_theoretic_activity(verbose_level);

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
	else if (f_blt_set_activity) {

		if (f_v) {
			cout << "activity_description::worker f_blt_set_activity" << endl;
		}
		do_blt_set_activity(verbose_level);

	}
	else if (f_combinatorial_object_activity) {

		if (f_v) {
			cout << "activity_description::worker f_combinatorial_object_activity" << endl;
		}
		do_combinatorial_object_activity(nb_output, Output, AO, verbose_level);

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
	else if (f_packing_classify_activity_description) {

		if (f_v) {
			cout << "activity_description::worker f_packing_classify_activity_description" << endl;
		}

		do_packing_classify_activity(verbose_level);
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
	else if (f_symbolic_object_activity) {

		if (f_v) {
			cout << "activity_description::worker f_symbolic_object_activity" << endl;
		}

		do_symbolic_object_activity(verbose_level);
	}
	else if (f_BLT_set_classify_activity) {

		if (f_v) {
			cout << "activity_description::worker f_BLT_set_classify_activity" << endl;
		}

		do_BLT_set_classify_activity(verbose_level);
	}
	else if (f_spread_classify_activity) {

		if (f_v) {
			cout << "activity_description::worker f_spread_classify_activity" << endl;
		}

		do_spread_classify_activity(verbose_level);
	}
	else if (f_spread_activity) {

		if (f_v) {
			cout << "activity_description::worker f_spread_activity" << endl;
		}

		do_spread_activity(verbose_level);
	}
	else if (f_translation_plane_activity) {

		if (f_v) {
			cout << "activity_description::worker f_translation_plane_activity" << endl;
		}

		do_translation_plane_activity(verbose_level);
	}
	else if (f_action_on_forms_activity) {

		if (f_v) {
			cout << "activity_description::worker f_action_on_forms_activity" << endl;
		}

		do_action_on_forms_activity(verbose_level);
	}
	else if (f_orbits_activity) {

		if (f_v) {
			cout << "activity_description::worker f_orbits_activity" << endl;
		}

		do_orbits_activity(verbose_level);
	}
	else if (f_variety_activity) {

		if (f_v) {
			cout << "activity_description::worker f_variety_activity" << endl;
		}

		do_variety_activity(verbose_level);
	}
	else if (f_vector_ge_activity) {

		if (f_v) {
			cout << "activity_description::worker f_vector_ge_activity" << endl;
		}

		do_vector_ge_activity(with_labels, nb_output, Output, verbose_level);
	}
	else if (f_combo_activity) {

		if (f_v) {
			cout << "activity_description::worker f_combo_activity" << endl;
		}

		do_combo_activity(nb_output, Output, verbose_level);
	}



	if (AO) {

		if (f_v) {
			cout << "activity_description::worker before AO->save" << endl;
		}
		AO->save(verbose_level);
		if (f_v) {
			cout << "activity_description::worker after AO->save" << endl;
		}

		FREE_OBJECT(AO);
	}

	if (f_v) {
		cout << "activity_description::worker done" << endl;
	}
}

void activity_description::print()
{

	if (Sym) {
		cout << "-with ";
		Sym->print_with();
		cout << "-do " << endl;
	}

	if (f_finite_field_activity) {
		cout << "-finite_field_activity ";
		Finite_field_activity_description->print();
	}

	else if (f_polynomial_ring_activity) {
		cout << "-ring_theoretic_activity ";
		Polynomial_ring_activity_description->print();
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
		cout << "-group_theoretic_activity ";
		Group_theoretic_activity_description->print();
	}
	else if (f_coding_theoretic_activity) {
		cout << "-coding_theoretic_activity ";
		Coding_theoretic_activity_description->print();
	}
	else if (f_cubic_surface_activity) {
		cout << "-cubic_surface_activity ";
		Cubic_surface_activity_description->print();
	}
	else if (f_quartic_curve_activity) {
		cout << "-quartic_curve_activity ";
		Quartic_curve_activity_description->print();
	}
	else if (f_blt_set_activity) {
		cout << "-blt_set_activity" << endl;
		Blt_set_activity_description->print();
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
	else if (f_packing_classify_activity_description) {
		cout << "-packing_classify_activity_description ";
		Packing_classify_activity_description->print();
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
	else if (f_symbolic_object_activity) {
		cout << "-symbolic_object_activity ";
		Symbolic_object_activity_description->print();
	}
	else if (f_BLT_set_classify_activity) {
		cout << "-BLT_set_classify_activity ";
		Blt_set_classify_activity_description->print();
	}
	else if (f_spread_classify_activity) {
		cout << "-spread_classify_activity ";
		Spread_classify_activity_description->print();
	}
	else if (f_spread_activity) {
		cout << "-spread_activity ";
		Spread_activity_description->print();
	}
	else if (f_translation_plane_activity) {
		cout << "-translation_plane_activity ";
		Translation_plane_activity_description->print();
	}
	else if (f_action_on_forms_activity) {
		cout << "-action_on_forms_activity" << endl;
		Action_on_forms_activity_description->print();
	}
	else if (f_orbits_activity) {
		cout << "-orbits_activity" << endl;
		Orbits_activity_description->print();
	}
	else if (f_variety_activity) {
		cout << "-variety_activity" << endl;
		Variety_activity_description->print();
	}
	else if (f_vector_ge_activity) {
		cout << "-vector_ge_activity" << endl;
		Vector_ge_activity_description->print();
	}
	else if (f_combo_activity) {
		cout << "-combo_activity" << endl;
		Combo_activity_description->print();
	}
	else {
		cout << "activity_description::print unknown type of activity" << endl;
		exit(1);
	}

}



void activity_description::do_finite_field_activity(
		other::orbiter_kernel_system::activity_output *&AO,
		int verbose_level)
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
	algebra::field_theory::finite_field *F;

	F = (algebra::field_theory::finite_field *) Sym->Orbiter_top_level_session->get_object(Idx[0]);

	algebra::field_theory::finite_field_activity FA;
	FA.init(Finite_field_activity_description, F, verbose_level);

	if (Sym->with_labels.size() == 2) {
		cout << "-finite_field_activity has two inputs" << endl;
		FA.F_secondary = (algebra::field_theory::finite_field *) Sym->Orbiter_top_level_session->get_object(Idx[1]);
	}



	if (f_v) {
		cout << "activity_description::do_finite_field_activity "
				"before FA.perform_activity" << endl;
	}
	FA.perform_activity(AO, verbose_level);
	if (f_v) {
		cout << "activity_description::do_finite_field_activity "
				"after FA.perform_activity" << endl;
	}

	FREE_int(Idx);

}


void activity_description::do_ring_theoretic_activity(
		int &nb_output,
		other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_ring_theoretic_activity "
				"ring theoretic activity for the following objects: ";
		Sym->print_with();
	}

	int *Idx;


	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "-finite_field_activity requires at least one input" << endl;
		exit(1);
	}
	algebra::ring_theory::homogeneous_polynomial_domain *HPD;

	HPD = (algebra::ring_theory::homogeneous_polynomial_domain *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);

	apps_algebra::polynomial_ring_activity Activity;
	Activity.init(Polynomial_ring_activity_description, HPD, verbose_level);

#if 0
	if (Sym->with_labels.size() == 2) {
		cout << "-finite_field_activity has two inputs" << endl;
		FA.F_secondary = (field_theory::finite_field *) Sym->Orbiter_top_level_session->get_object(Idx[1]);
	}
#endif


	if (f_v) {
		cout << "activity_description::do_ring_theoretic_activity "
				"before Activity.perform_activity" << endl;
	}
	Activity.perform_activity(verbose_level);
	if (f_v) {
		cout << "activity_description::do_ring_theoretic_activity "
				"after Activity.perform_activity" << endl;
	}

	nb_output = Activity.nb_output;
	Output = Activity.Output;

	FREE_int(Idx);

}


void activity_description::do_projective_space_activity(
		int verbose_level)
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
	projective_geometry::projective_space_with_action *PA;

	PA = (projective_geometry::projective_space_with_action *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);

	projective_geometry::projective_space_activity Activity;
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

void activity_description::do_orthogonal_space_activity(
		int verbose_level)
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
	orthogonal_geometry_applications::orthogonal_space_with_action *OA;

	OA = (orthogonal_geometry_applications::orthogonal_space_with_action *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);

	orthogonal_geometry_applications::orthogonal_space_activity Activity;
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
	if (f_v) {
		cout << "activity_description::do_orthogonal_space_activity done" << endl;
	}

}

void activity_description::do_group_theoretic_activity(
		int &nb_output,
		other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_group_theoretic_activity "
				"group theoretic activity for the following objects:";
		Sym->print_with();
	}

	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "-group_theoretic_activity requires at least one input" << endl;
		exit(1);
	}


	layer1_foundations::other::orbiter_kernel_system::symbol_table_object_type type;

	type = Sym->Orbiter_top_level_session->get_object_type(Idx[0]);

	if (type != layer1_foundations::other::orbiter_kernel_system::symbol_table_object_type::t_any_group) {
		cout << "activity_description::do_group_theoretic_activity type is not t_any_group" << endl;
		exit(1);
	}

	groups::any_group *AG;

	AG = (groups::any_group *) Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		apps_algebra::group_theoretic_activity Activity;

		Activity.init_group(Group_theoretic_activity_description, AG, verbose_level);




		if (Sym->with_labels.size() >= 2) {

			layer1_foundations::other::orbiter_kernel_system::symbol_table_object_type type;

			type = Sym->Orbiter_top_level_session->get_object_type(Idx[1]);

			if (type != layer1_foundations::other::orbiter_kernel_system::symbol_table_object_type::t_any_group) {
				cout << "activity_description::do_group_theoretic_activity secondary type is not t_any_group" << endl;
				exit(1);
			}

			groups::any_group *AG_secondary;

			AG_secondary = (groups::any_group *)
					Sym->Orbiter_top_level_session->get_object(Idx[1]);

			if (f_v) {
				cout << "activity_description::do_group_theoretic_activity "
						"before Activity.init_secondary_group" << endl;
			}
			Activity.init_secondary_group(Group_theoretic_activity_description, AG_secondary, verbose_level);

		}


		if (f_v) {
			cout << "activity_description::do_group_theoretic_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_group_theoretic_activity "
					"after Activity.perform_activity" << endl;
		}

		nb_output = Activity.nb_output;
		Output = Activity.Output;

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_group_theoretic_activity done" << endl;
	}

}

void activity_description::do_coding_theoretic_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_coding_theoretic_activity "
				"coding theoretic activity for the following objects:";
		Sym->print_with();
	}

	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "-coding_theoretic_activity requires at least one input" << endl;
		exit(1);
	}

	{
		int i;
		apps_coding_theory::coding_theoretic_activity Activity;


		for (i = 0; i < Sym->with_labels.size(); i++) {
			layer1_foundations::other::orbiter_kernel_system::symbol_table_object_type type;

			type = Sym->Orbiter_top_level_session->get_object_type(Idx[i]);

			if (type == layer1_foundations::other::orbiter_kernel_system::symbol_table_object_type::t_finite_field) {

				if (f_v) {
					cout << "activity_description::do_coding_theoretic_activity type is t_finite_field" << endl;
				}

				algebra::field_theory::finite_field *F;

				F = (algebra::field_theory::finite_field *)
						Sym->Orbiter_top_level_session->get_object(Idx[i]);

				Activity.init_field(Coding_theoretic_activity_description, F, verbose_level);
			}

			else if (type == layer1_foundations::other::orbiter_kernel_system::symbol_table_object_type::t_code) {

				if (f_v) {
					cout << "activity_description::do_coding_theoretic_activity type is t_code" << endl;
				}

				apps_coding_theory::create_code *Code;

				Code = (apps_coding_theory::create_code *)
						Sym->Orbiter_top_level_session->get_object(Idx[i]);

				Activity.init_code(Coding_theoretic_activity_description, Code, verbose_level);

			}

		}


		if (f_v) {
			cout << "activity_description::do_coding_theoretic_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_coding_theoretic_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_coding_theoretic_activity done" << endl;
	}

}

void activity_description::do_cubic_surface_activity(
		int verbose_level)
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

	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create *SC;

	SC = (applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		applications_in_algebraic_geometry::cubic_surfaces_in_general::cubic_surface_activity Activity;

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


void activity_description::do_quartic_curve_activity(
		int verbose_level)
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
		cout << "-quartic_curve_activity requires at least one input" << endl;
		exit(1);
	}

	applications_in_algebraic_geometry::quartic_curves::quartic_curve_create *QC;

	QC = (applications_in_algebraic_geometry::quartic_curves::quartic_curve_create *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		applications_in_algebraic_geometry::quartic_curves::quartic_curve_activity Activity;

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


void activity_description::do_blt_set_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_blt_set_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "-blt_set_activity requires at least one input" << endl;
		exit(1);
	}

	orthogonal_geometry_applications::BLT_set_create *BC;

	BC = (orthogonal_geometry_applications::BLT_set_create *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		orthogonal_geometry_applications::blt_set_activity Activity;

		Activity.init(Blt_set_activity_description, BC, verbose_level);

		if (f_v) {
			cout << "activity_description::do_blt_set_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_blt_set_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_blt_set_activity done" << endl;
	}

}


void activity_description::do_combinatorial_object_activity(
		int &nb_output,
		other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
		other::orbiter_kernel_system::activity_output *&AO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_combinatorial_object_activity "
				"activity for the following objects:";
		Sym->print_with();
		cout << "activity_description::do_combinatorial_object_activity "
				"verbose_level = " << verbose_level << endl;
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "-combinatorial_object_activity requires at least one input" << endl;
		exit(1);
	}

	layer1_foundations::other::orbiter_kernel_system::symbol_table_object_type t;

	t = Sym->Orbiter_top_level_session->get_object_type(Idx[0]);
	if (t == layer1_foundations::other::orbiter_kernel_system::symbol_table_object_type::t_geometric_object) {
		geometry::other_geometry::geometric_object_create *GOC;

		GOC = (geometry::other_geometry::geometric_object_create *)
				Sym->Orbiter_top_level_session->get_object(Idx[0]);
		{
			apps_combinatorics::combinatorial_object_activity Activity;

			Activity.init(Combinatorial_object_activity_description, GOC, verbose_level);


			if (f_v) {
				cout << "activity_description::do_combinatorial_object_activity "
						"before Activity.perform_activity" << endl;
			}
			Activity.perform_activity(AO, verbose_level);
			if (f_v) {
				cout << "activity_description::do_combinatorial_object_activity "
						"after Activity.perform_activity" << endl;
			}

			if (AO) {
				AO->fname_base = GOC->label_txt;
			}


		}
	}
	else if (t == layer1_foundations::other::orbiter_kernel_system::symbol_table_object_type::t_combinatorial_object) {
		apps_combinatorics::combinatorial_object_stream *Combo;

		Combo = (apps_combinatorics::combinatorial_object_stream *)
				Sym->Orbiter_top_level_session->get_object(Idx[0]);
		{
			apps_combinatorics::combinatorial_object_activity Activity;

			Activity.init_combo(Combinatorial_object_activity_description, Combo, verbose_level);


			if (f_v) {
				cout << "activity_description::do_combinatorial_object_activity "
						"before Activity.perform_activity" << endl;
			}
			Activity.perform_activity(AO, verbose_level);
			if (f_v) {
				cout << "activity_description::do_combinatorial_object_activity "
						"after Activity.perform_activity" << endl;
			}

			if (AO) {
				AO->fname_base = Combo->IS->Descr->label_txt;
			}

			// pass back the output (if there is one):
			nb_output = Activity.nb_output;
			Output = Activity.Output;

		}
	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_combinatorial_object_activity done" << endl;
	}

}

void activity_description::do_graph_theoretic_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_graph_theoretic_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	int nb = Sym->with_labels.size();


	if (nb < 1) {
		cout << "activity_description::do_graph_theoretic_activity "
				"this activity requires at least one input" << endl;
		exit(1);
	}


	//create_graph *Gr;
	combinatorics::graph_theory::colored_graph **CG;
	int i;

	CG = (combinatorics::graph_theory::colored_graph **) NEW_pvoid(nb);

	for (i = 0; i < nb; i++) {
		CG[i] = (combinatorics::graph_theory::colored_graph *)
			Sym->Orbiter_top_level_session->get_object(Idx[i]);
	}
	if (f_v) {
		cout << "activity_description::do_graph_theoretic_activity graphs = ";
		for (i = 0; i < nb; i++) {
			cout << CG[i]->label;
			if (i < nb - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}

	{
		apps_graph_theory::graph_theoretic_activity Activity;
		std::vector<std::string> feedback;

		Activity.init(Graph_theoretic_activity_description, nb, CG, verbose_level);

		if (f_v) {
			cout << "activity_description::do_graph_theoretic_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(feedback, verbose_level);
		if (f_v) {
			cout << "activity_description::do_graph_theoretic_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_pvoid((void **) CG);
	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_graph_theoretic_activity done" << endl;
	}

}

void activity_description::do_classification_of_cubic_surfaces_with_double_sixes_activity(
		int verbose_level)
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

	applications_in_algebraic_geometry::cubic_surfaces_and_double_sixes::surface_classify_wedge *SCW;

	SCW = (applications_in_algebraic_geometry::cubic_surfaces_and_double_sixes::surface_classify_wedge *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		applications_in_algebraic_geometry::cubic_surfaces_and_double_sixes::classification_of_cubic_surfaces_with_double_sixes_activity
			Activity;

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

void activity_description::do_spread_table_activity(
		int verbose_level)
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

	packings::packing_classify *P;

	P = (packings::packing_classify *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		spreads::spread_table_activity Activity;

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



void activity_description::do_packing_classify_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_packing_classify_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	packings::packing_classify *P;

	P = (packings::packing_classify *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		packings::packing_classify_activity Activity;

		Activity.init(Packing_classify_activity_description, P, verbose_level);

		if (f_v) {
			cout << "activity_description::do_packing_classify_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_packing_classify_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_packing_classify_activity done" << endl;
	}

}

void activity_description::do_packing_was_activity(
		int verbose_level)
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

	packings::packing_was *PW;

	PW = (packings::packing_was *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		packings::packing_was_activity Activity;

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



void activity_description::do_packing_fixed_points_activity(
		int verbose_level)
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

	packings::packing_was_fixpoints *PWF;

	PWF = (packings::packing_was_fixpoints *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		packings::packing_was_fixpoints_activity Activity;

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


void activity_description::do_graph_classification_activity(
		int verbose_level)
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

	apps_graph_theory::graph_classify *GC;

	GC = (apps_graph_theory::graph_classify *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		apps_graph_theory::graph_classification_activity Activity;

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

void activity_description::do_diophant_activity(
		int verbose_level)
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

	combinatorics::solvers::diophant_create *Dio;

	Dio = (combinatorics::solvers::diophant_create *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		combinatorics::solvers::diophant_activity Activity;


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

void activity_description::do_design_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_design_activity "
				"activity for the following objects:";
		Sym->print_with();
	}


	combinatorics::design_theory::design_object *Design_object;

	Design_object = Get_design(Sym->with_labels[0]);

#if 0
	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	apps_combinatorics::design_create *DC;

	DC = (apps_combinatorics::design_create *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		apps_combinatorics::design_activity Activity;


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
#endif

	{
		apps_combinatorics::design_activity Activity;


		if (f_v) {
			cout << "activity_description::do_design_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(Design_activity_description, Design_object, verbose_level);
		if (f_v) {
			cout << "activity_description::do_design_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	if (f_v) {
		cout << "activity_description::do_design_activity done" << endl;
	}

}

void activity_description::do_large_set_was_activity(
		int verbose_level)
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

	apps_combinatorics::large_set_was *LSW;

	LSW = (apps_combinatorics::large_set_was *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		apps_combinatorics::large_set_was_activity Activity;

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

void activity_description::do_symbolic_object_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_symbolic_object_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	algebra::expression_parser::symbolic_object_builder *f;

	f = (algebra::expression_parser::symbolic_object_builder *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{
		algebra::expression_parser::symbolic_object_activity Activity;


		Activity.init(
				Symbolic_object_activity_description,
				f,
				verbose_level);

		if (f_v) {
			cout << "activity_description::do_symbolic_object_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_symbolic_object_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_symbolic_object_activity done" << endl;
	}

}

void activity_description::do_BLT_set_classify_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_BLT_set_classify_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	orthogonal_geometry_applications::blt_set_classify *B;

	B = (orthogonal_geometry_applications::blt_set_classify *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{

		orthogonal_geometry_applications::blt_set_classify_activity Activity;

		Activity.init(Blt_set_classify_activity_description,
				B /* blt_set_classify *BLT_classify*/,
				B->OA,
				verbose_level);


		if (f_v) {
			cout << "activity_description::do_BLT_set_classify_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_BLT_set_classify_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_BLT_set_classify_activity done" << endl;
	}

}

void activity_description::do_spread_classify_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_spread_classify_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	spreads::spread_classify *SC;

	SC = (spreads::spread_classify *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{

		spreads::spread_classify_activity Activity;

		Activity.init(
				Spread_classify_activity_description,
				SC,
				verbose_level);


		if (f_v) {
			cout << "activity_description::do_spread_classify_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_spread_classify_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_spread_classify_activity done" << endl;
	}

}

void activity_description::do_spread_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_spread_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	spreads::spread_create *SC;

	SC = (spreads::spread_create *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{

		spreads::spread_activity Activity;

		Activity.init(
				Spread_activity_description,
				SC,
				verbose_level);


		if (f_v) {
			cout << "activity_description::do_spread_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_spread_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_spread_activity done" << endl;
	}

}


void activity_description::do_translation_plane_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_translation_plane_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	combinatorics_with_groups::translation_plane_via_andre_model *TP;

	TP = (combinatorics_with_groups::translation_plane_via_andre_model *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{

		spreads::translation_plane_activity Activity;

		Activity.init(
				Translation_plane_activity_description,
				TP,
				verbose_level);


		if (f_v) {
			cout << "activity_description::do_translation_plane_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_translation_plane_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_translation_plane_activity done" << endl;
	}

}


void activity_description::do_action_on_forms_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_action_on_forms_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	apps_algebra::action_on_forms *AF;

	AF = (apps_algebra::action_on_forms *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{

		apps_algebra::action_on_forms_activity Activity;

		Activity.init(
				Action_on_forms_activity_description,
				AF,
				verbose_level);


		if (f_v) {
			cout << "activity_description::do_action_on_forms_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_action_on_forms_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_action_on_forms_activity done" << endl;
	}

}


void activity_description::do_orbits_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_orbits_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	orbits::orbits_create *OC;

	OC = (orbits::orbits_create *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{

		orbits::orbits_activity Activity;

		Activity.init(
				Orbits_activity_description,
				OC,
				verbose_level);


		if (f_v) {
			cout << "activity_description::do_orbits_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_orbits_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_orbits_activity done" << endl;
	}

}


void activity_description::do_variety_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_variety_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	canonical_form::variety_object_with_action *V;

	V = (canonical_form::variety_object_with_action *)
			Sym->Orbiter_top_level_session->get_object(Idx[0]);
	{

		canonical_form::variety_activity Activity;

		Activity.init(
				Variety_activity_description,
				1,
				V,
				verbose_level);



		if (f_v) {
			cout << "activity_description::do_variety_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_variety_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_variety_activity done" << endl;
	}

}

void activity_description::do_vector_ge_activity(
		std::vector<std::string> &with_labels,
		int &nb_output,
		other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_vector_ge_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}

	int nb_objects;
	apps_algebra::vector_ge_builder **pVB;
	//data_structures_groups::vector_ge *vec;

	nb_objects = Sym->with_labels.size();
	pVB = (apps_algebra::vector_ge_builder **) NEW_pvoid(nb_objects);
	int i;

	for (i = 0; i < nb_objects; i++) {
		apps_algebra::vector_ge_builder *VB;

		VB = (apps_algebra::vector_ge_builder *)
				Sym->Orbiter_top_level_session->get_object(Idx[i]);
		pVB[i] = VB;

	}

	//vec = VB->V;
	{

		apps_algebra::vector_ge_activity Activity;

		Activity.init(
				Vector_ge_activity_description,
				pVB,
				nb_objects,
				with_labels,
				verbose_level);



		if (f_v) {
			cout << "activity_description::do_vector_ge_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "activity_description::do_vector_ge_activity "
					"after Activity.perform_activity" << endl;
		}

		// pass back the output (if there is one):
		nb_output = Activity.nb_output;
		Output = Activity.Output;

	}

	FREE_pvoid((void **) pVB);

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_vector_ge_activity done" << endl;
	}

}



void activity_description::do_combo_activity(
		int &nb_output,
		other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_description::do_combo_activity "
				"activity for the following objects:";
		Sym->print_with();
	}



	int *Idx;

	Sym->Orbiter_top_level_session->find_symbols(Sym->with_labels, Idx);

	if (Sym->with_labels.size() < 1) {
		cout << "activity requires at least one input" << endl;
		exit(1);
	}


	int nb_objects;
	canonical_form::combinatorial_object_with_properties **pOwP;

	nb_objects = Sym->with_labels.size();
	pOwP = (canonical_form::combinatorial_object_with_properties **) NEW_pvoid(nb_objects);
	int i;

	for (i = 0; i < nb_objects; i++) {

		pOwP[i] = (canonical_form::combinatorial_object_with_properties *)
				Sym->Orbiter_top_level_session->get_object(Idx[i]);

	}

	other::orbiter_kernel_system::activity_output *AO;
	{

		apps_combinatorics::combo_activity Activity;

		Activity.init(
				Combo_activity_description,
				pOwP,
				nb_objects,
				verbose_level);


		if (f_v) {
			cout << "activity_description::do_combo_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(AO, verbose_level);
		if (f_v) {
			cout << "activity_description::do_combo_activity "
					"after Activity.perform_activity" << endl;
		}

		// pass back the output (if there is one):
		nb_output = Activity.nb_output;
		Output = Activity.Output;

	}

	FREE_pvoid((void **) pOwP);

	FREE_int(Idx);

	if (f_v) {
		cout << "activity_description::do_combo_activity done" << endl;
	}

}






}}}


