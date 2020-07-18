/*
 * group_theoretic_activity_description.cpp
 *
 *  Created on: Apr 26, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


group_theoretic_activity_description::group_theoretic_activity_description()
{
	f_poset_classification_control = FALSE;
	Control = NULL;
	f_orbits_on_points = FALSE;
	f_export_trees = FALSE;
	f_shallow_tree = FALSE;
	f_stabilizer = FALSE;
	f_orbits_on_subsets = FALSE;
	orbits_on_subsets_size = 0;
	f_draw_poset = FALSE;
	f_draw_full_poset = FALSE;
	f_classes = FALSE;
	f_group_table = FALSE;
	f_normalizer = FALSE;
	f_centralizer_of_element = FALSE;
	element_description_text = NULL;
	element_label = NULL;
	f_normalizer_of_cyclic_subgroup = FALSE;
	f_report = FALSE;
	f_sylow = FALSE;
	f_test_if_geometric = FALSE;
	test_if_geometric_depth = 0;
	f_draw_tree = FALSE;
	f_orbit_of = FALSE;
	orbit_of_idx = 0;
	f_orbits_on_set_system_from_file = FALSE;
	orbits_on_set_system_from_file_fname = NULL;
	orbits_on_set_system_first_column = 0;
	orbits_on_set_system_number_of_columns = 0;
	f_orbit_of_set_from_file = FALSE;
	orbit_of_set_from_file_fname = NULL;
	f_search_subgroup = FALSE;
	f_find_singer_cycle = FALSE;
	f_search_element_of_order = FALSE;
	search_element_order = 0;
	f_linear_codes = FALSE;
	linear_codes_minimum_distance = 0;
	linear_codes_target_size = 0;
	f_print_elements = FALSE;
	f_print_elements_tex = FALSE;
	f_multiply = FALSE;
	multiply_a = NULL;
	multiply_b = NULL;
	f_inverse = FALSE;
	inverse_a = NULL;
	f_export_gap = FALSE;
	f_export_magma = FALSE;
	f_order_of_products = FALSE;
	order_of_products_elements = NULL;
	//f_embedded = FALSE;
	//f_sideways = FALSE;
	//x_stretch = 1.;
	//f_print_generators = FALSE;
	f_classify_arcs = FALSE;
	Arc_generator_description = NULL;
	//f_classify_nonconical_arcs = FALSE;
	//classify_arcs_target_size = 0;
	//classify_arcs_d = 0;
	f_exact_cover = FALSE;
	ECA = NULL;
	f_isomorph_arguments = FALSE;
	IA = NULL;
	f_surface_classify = FALSE;
	f_surface_report = FALSE;
	f_surface_identify_Sa = FALSE;
	f_surface_isomorphism_testing = FALSE;
		surface_descr_isomorph1 = NULL;
		surface_descr_isomorph2 = NULL;
	f_surface_recognize = FALSE;
		surface_descr = NULL;
	f_classify_surfaces_through_arcs_and_trihedral_pairs = FALSE;
	f_trihedra1_control = FALSE;
	Trihedra1_control = NULL;
	f_trihedra2_control = FALSE;
	Trihedra2_control = NULL;
	f_control_six_arcs = FALSE;
			Control_six_arcs = NULL;
	f_create_surface = FALSE;
	surface_description = NULL;
	f_surface_quartic = FALSE;
	f_surface_clebsch = FALSE;
	f_surface_codes = FALSE;
	nb_transform = 0;
	//const char *transform_coeffs[1000];
	//int f_inverse_transform[1000];


	f_orbits_on_subspaces = FALSE;
	orbits_on_subspaces_depth = 0;
	f_mindist = FALSE;
	mindist = 0;
	f_self_orthogonal = FALSE;
	f_doubly_even = FALSE;

	f_spread_classify = FALSE;
	spread_classify_k = 0;

	f_packing_classify = FALSE;
	dimension_of_spread_elements = 0;
	spread_selection_text = NULL;
	spread_tables_prefix = NULL;
	f_packing_with_assumed_symmetry = FALSE;
	packing_was_descr = NULL;


	f_tensor_classify = FALSE;
	tensor_classify_depth = 0;
	f_tensor_permutations = FALSE;
}

group_theoretic_activity_description::~group_theoretic_activity_description()
{
	freeself();
}

void group_theoretic_activity_description::null()
{
}

void group_theoretic_activity_description::freeself()
{
	null();
}

void group_theoretic_activity_description::read_arguments_from_string(
		const char *str, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	int argc;
	char **argv;
	int i;

	if (f_v) {
		cout << "group_theoretic_activity_description::read_arguments_from_string" << endl;
	}
	chop_string(str, argc, argv);

	if (f_vv) {
		cout << "argv:" << endl;
		for (i = 0; i < argc; i++) {
			cout << i << " : " << argv[i] << endl;
		}
	}


	read_arguments(
		argc, (const char **) argv,
		verbose_level);

	for (i = 0; i < argc; i++) {
		FREE_char(argv[i]);
	}
	FREE_pchar(argv);
	if (f_v) {
		cout << "group_theoretic_activity_description::read_arguments_from_string "
				"done" << endl;
	}
}

int group_theoretic_activity_description::read_arguments(
	int argc, const char **argv,
	int verbose_level)
{
	int i;

	cout << "group_theoretic_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {
		if (strcmp(argv[i], "-orbits_on_subsets") == 0) {
			f_orbits_on_subsets = TRUE;
			orbits_on_subsets_size = atoi(argv[++i]);
			cout << "-orbits_on_subsets " << orbits_on_subsets_size << endl;
		}
		else if (strcmp(argv[i], "-poset_classification_control") == 0) {
			f_poset_classification_control = TRUE;
			Control = NEW_OBJECT(poset_classification_control);
			i += Control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -poset_classification_control " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-orbits_on_points") == 0) {
			f_orbits_on_points = TRUE;
			cout << "-orbits_on_points" << endl;
		}
		else if (strcmp(argv[i], "-export_trees") == 0) {
			f_export_trees = TRUE;
			cout << "-export_trees" << endl;
		}
		else if (strcmp(argv[i], "-shallow_tree") == 0) {
			f_shallow_tree = TRUE;
			cout << "-shallow_tree" << endl;
		}
		else if (strcmp(argv[i], "-stabilizer") == 0) {
			f_stabilizer = TRUE;
			cout << "-stabilizer" << endl;
		}
		else if (strcmp(argv[i], "-test_if_geometric") == 0) {
			f_test_if_geometric = TRUE;
			test_if_geometric_depth = atoi(argv[++i]);
			cout << "-test_if_geometric" << endl;
		}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset" << endl;
		}
		else if (strcmp(argv[i], "-draw_full_poset") == 0) {
			f_draw_full_poset = TRUE;
			cout << "-draw_full_poset" << endl;
		}
		else if (strcmp(argv[i], "-classes") == 0) {
			f_classes = TRUE;
			cout << "-classes" << endl;
		}
		else if (strcmp(argv[i], "-group_table") == 0) {
			f_group_table = TRUE;
			cout << "-group_table" << endl;
		}
		else if (strcmp(argv[i], "-normalizer") == 0) {
			f_normalizer = TRUE;
			cout << "-normalizer" << endl;
		}
		else if (strcmp(argv[i], "-centralizer_of_element") == 0) {
			f_centralizer_of_element = TRUE;
			element_label = argv[++i];
			element_description_text = argv[++i];
			cout << "-centralizer_of_element " << element_label << " " << element_description_text << endl;
		}
		else if (strcmp(argv[i], "-normalizer_of_cyclic_subgroup") == 0) {
			f_normalizer_of_cyclic_subgroup = TRUE;
			element_label = argv[++i];
			element_description_text = argv[++i];
			cout << "-normalizer_of_cyclic_subgroup " << element_label << " " << element_description_text << endl;
		}
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report" << endl;
		}
		else if (strcmp(argv[i], "-sylow") == 0) {
			f_sylow = TRUE;
			cout << "-sylow" << endl;
		}
		else if (strcmp(argv[i], "-f_draw_tree") == 0) {
			f_draw_tree = TRUE;
			cout << "-f_draw_tree " << endl;
		}
		else if (strcmp(argv[i], "-orbit_of") == 0) {
			f_orbit_of = TRUE;
			orbit_of_idx = atoi(argv[++i]);
			cout << "-orbit_of " << orbit_of_idx << endl;
		}
		else if (strcmp(argv[i], "-orbit_of_set_from_file") == 0) {
			f_orbit_of_set_from_file = TRUE;
			orbit_of_set_from_file_fname = argv[++i];
			cout << "-orbit_of_set_from_file"
					<< orbit_of_set_from_file_fname << endl;
		}
		else if (strcmp(argv[i], "-orbits_on_set_system_from_file") == 0) {
			f_orbits_on_set_system_from_file = TRUE;
			orbits_on_set_system_from_file_fname = argv[++i];
			orbits_on_set_system_first_column = atoi(argv[++i]);
			orbits_on_set_system_number_of_columns = atoi(argv[++i]);
			cout << "-orbits_on_set_system_from_file"
					<< orbits_on_set_system_from_file_fname
					<< " " << orbits_on_set_system_first_column << " "
					<< orbits_on_set_system_number_of_columns << endl;
		}
		else if (strcmp(argv[i], "-search_subgroup") == 0) {
			f_search_subgroup = TRUE;
			cout << "-search_subgroup " << endl;
		}
		else if (strcmp(argv[i], "-find_singer_cycle") == 0) {
			f_find_singer_cycle = TRUE;
			cout << "-find_singer_cycle " << endl;
		}
		else if (strcmp(argv[i], "-search_element_of_order") == 0) {
			f_search_element_of_order = TRUE;
			search_element_order = atoi(argv[++i]);
			cout << "-search_element_of_order " << search_element_order << endl;
		}
		else if (strcmp(argv[i], "-print_elements") == 0) {
			f_print_elements = TRUE;
			cout << "-print_elements " << endl;
		}
		else if (strcmp(argv[i], "-print_elements_tex") == 0) {
			f_print_elements_tex = TRUE;
			cout << "-print_elements_tex " << endl;
		}
		else if (strcmp(argv[i], "-order_of_products") == 0) {
			f_order_of_products = TRUE;
			order_of_products_elements = argv[++i];
			cout << "-order_of_products " << order_of_products_elements << endl;
		}
		else if (strcmp(argv[i], "-multiply") == 0) {
			f_multiply = TRUE;
			multiply_a = argv[++i];
			multiply_b = argv[++i];
			cout << "-multiply " << multiply_a << " " << multiply_b << endl;
		}
		else if (strcmp(argv[i], "-inverse") == 0) {
			f_inverse = TRUE;
			inverse_a = argv[++i];
			cout << "-inverse " << inverse_a << endl;
		}
		else if (strcmp(argv[i], "-export_gap") == 0) {
			f_export_gap = TRUE;
			cout << "-export_gap " << endl;
		}
		else if (strcmp(argv[i], "-export_magma") == 0) {
			f_export_magma = TRUE;
			cout << "-export_magma " << endl;
		}
#if 0
		else if (strcmp(argv[i], "-group_table") == 0) {
			f_group_table = TRUE;
			cout << "-group_table" << endl;
		}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "-embedded" << endl;
		}
		else if (strcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			cout << "-sideways" << endl;
		}
		else if (strcmp(argv[i], "-x_stretch") == 0) {
			sscanf(argv[++i], "%lf", &x_stretch);
			cout << "-x_stretch" << x_stretch << endl;
		}
		else if (strcmp(argv[i], "-print_generators") == 0) {
			f_print_generators = TRUE;
			cout << "-print_generators" << endl;
		}
#endif

		// classification tasks:

		// linear codes:

		else if (strcmp(argv[i], "-linear_codes") == 0) {
			f_linear_codes = TRUE;
			linear_codes_minimum_distance = atoi(argv[++i]);
			linear_codes_target_size = atoi(argv[++i]);
			cout << "-linear_codes " << linear_codes_minimum_distance << " " << linear_codes_target_size << endl;
		}


		// arcs:


		else if (strcmp(argv[i], "-classify_arcs") == 0) {
			f_classify_arcs = TRUE;
			Arc_generator_description = NEW_OBJECT(arc_generator_description);
			cout << "-classify_arcs" << endl;
			i += Arc_generator_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -classify_arcs " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}

		else if (strcmp(argv[i], "-exact_cover") == 0) {
			f_exact_cover = TRUE;
			ECA = NEW_OBJECT(exact_cover_arguments);
			i += ECA->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done with -exact_cover" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-isomorph_arguments") == 0) {
			f_isomorph_arguments = TRUE;
			IA = NEW_OBJECT(isomorph_arguments);
			i += IA->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done with -isomorph_arguments" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}


		// cubic surfaces:
		else if (strcmp(argv[i], "-surface_classify") == 0) {
			f_surface_classify = TRUE;
			cout << "-surface_classify " << endl;
		}
		else if (strcmp(argv[i], "-surface_report") == 0) {
			f_surface_report = TRUE;
			cout << "-surface_report " << endl;
		}
		else if (strcmp(argv[i], "-surface_identify_Sa") == 0) {
			f_surface_identify_Sa = TRUE;
			cout << "-surface_identify_Sa " << endl;
		}
		else if (strcmp(argv[i], "-surface_isomorphism_testing") == 0) {
			f_surface_isomorphism_testing = TRUE;
			cout << "-surface_isomorphism_testing reading description of first surface" << endl;
			surface_descr_isomorph1 = NEW_OBJECT(surface_create_description);
			i += surface_descr_isomorph1->
					read_arguments(argc - (i - 1), argv + i,
					verbose_level) - 1;
			cout << "-isomorph after reading description of first surface" << endl;
			i += 2;
			cout << "the current argument is " << argv[i] << endl;
			cout << "-isomorph reading description of second surface" << endl;
			surface_descr_isomorph2 = NEW_OBJECT(surface_create_description);
			i += surface_descr_isomorph2->
					read_arguments(argc - (i - 1), argv + i,
					verbose_level);
			cout << "-surface_isomorphism_testing " << endl;
		}
		else if (strcmp(argv[i], "-surface_recognize") == 0) {
			f_surface_recognize = TRUE;
			cout << "-surface_recognize reading description of surface" << endl;
			surface_descr = NEW_OBJECT(surface_create_description);
			i += surface_descr->
					read_arguments(argc - (i - 1), argv + i,
					verbose_level);
			//i += 2;
			cout << "-surface_recognize " << endl;
		}
		else if (strcmp(argv[i], "-classify_surfaces_through_arcs_and_trihedral_pairs") == 0) {
			f_classify_surfaces_through_arcs_and_trihedral_pairs = TRUE;
			//q = atoi(argv[++i]);
			cout << "-classify_surfaces_through_arcs_and_trihedral_pairs " << endl;
		}
		else if (strcmp(argv[i], "-create_surface") == 0) {
			f_create_surface = TRUE;
			surface_description = NEW_OBJECT(surface_create_description);
			cout << "-create_surface" << endl;
			i += surface_description->read_arguments(
					argc - (i - 1), argv + i,
					verbose_level);
			cout << "done with -create_surface" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-surface_quartic") == 0) {
			f_surface_quartic = TRUE;
			cout << "-surface_quartic" << endl;
		}
		else if (strcmp(argv[i], "-surface_clebsch") == 0) {
			f_surface_clebsch = TRUE;
			cout << "=surface_clebsch" << endl;
		}
		else if (strcmp(argv[i], "-surface_codes") == 0) {
			f_surface_codes = TRUE;
			cout << "-surface_codes" << endl;
		}
		else if (strcmp(argv[i], "-transform") == 0) {
			transform_coeffs[nb_transform] = argv[++i];
			f_inverse_transform[nb_transform] = FALSE;
			cout << "-transform " << transform_coeffs[nb_transform] << endl;
			nb_transform++;
		}
		else if (strcmp(argv[i], "-transform_inverse") == 0) {
			transform_coeffs[nb_transform] = argv[++i];
			f_inverse_transform[nb_transform] = TRUE;
			cout << "-transform_inverse "
					<< transform_coeffs[nb_transform] << endl;
			nb_transform++;
		}
		else if (strcmp(argv[i], "-trihedra1_control") == 0) {
			f_trihedra1_control = TRUE;
			Trihedra1_control = NEW_OBJECT(poset_classification_control);
			i += Trihedra1_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -trihedra1_control " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-trihedra2_control") == 0) {
			f_trihedra2_control = TRUE;
			Trihedra2_control = NEW_OBJECT(poset_classification_control);
			i += Trihedra2_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -trihedra2_control " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-control_six_arcs") == 0) {
			f_control_six_arcs = TRUE;
			Control_six_arcs = NEW_OBJECT(poset_classification_control);
			i += Control_six_arcs->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -control_six_arcs " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}



		else if (strcmp(argv[i], "-orbits_on_subspaces") == 0) {
			f_orbits_on_subspaces = TRUE;
			orbits_on_subspaces_depth = atoi(argv[++i]);
			cout << "-orbits_on_subspaces " << orbits_on_subspaces_depth << endl;
		}
		else if (strcmp(argv[i], "-mindist") == 0) {
			f_mindist = TRUE;
			mindist = atoi(argv[++i]);
			cout << "-mindist" << mindist << endl;
		}
		else if (strcmp(argv[i], "-self_orthogonal") == 0) {
			f_self_orthogonal = TRUE;
			cout << "-self_orthogonal" << endl;
		}
		else if (strcmp(argv[i], "-doubly_even") == 0) {
			f_doubly_even = TRUE;
			cout << "-doubly_even" << endl;
		}


		// spreads:

		else if (strcmp(argv[i], "-spread_classify") == 0) {
			f_spread_classify = TRUE;
			spread_classify_k = atoi(argv[++i]);
			cout << "-spread_classify " << spread_classify_k << endl;
		}

		// packings:
		else if (strcmp(argv[i], "-packing_classify") == 0) {
			f_packing_classify = TRUE;
			dimension_of_spread_elements = atoi(argv[++i]);
			spread_selection_text = argv[++i];
			spread_tables_prefix = argv[++i];
			cout << "-packing_classify "
					<< dimension_of_spread_elements
					<< " " << spread_selection_text
					<< " " << spread_tables_prefix
					<< endl;
		}
		else if (strcmp(argv[i], "-packing_with_assumed_symmetry") == 0) {
			f_packing_with_assumed_symmetry = TRUE;
			packing_was_descr = NEW_OBJECT(packing_was_description);
			cout << "-packing_with_assumed_symmetry " << endl;
			i += packing_was_descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -packing_with_assumed_symmetry " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}


		// tensors:

		else if (strcmp(argv[i], "-tensor_classify") == 0) {
			f_tensor_classify = TRUE;
			tensor_classify_depth = atoi(argv[++i]);
			cout << "-tensor_classify " << tensor_classify_depth << endl;
		}
		else if (strcmp(argv[i], "-tensor_permutations") == 0) {
			f_tensor_permutations = TRUE;
			cout << "-tensor_permutations " << endl;
		}
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "group_theoretic_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "group_theoretic_activity_description::read_arguments done" << endl;
	return i + 1;
}



}}
