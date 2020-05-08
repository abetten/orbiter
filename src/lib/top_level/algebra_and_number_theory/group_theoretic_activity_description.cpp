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
	f_orbits_on_points = FALSE;
	f_export_trees = FALSE;
	f_shallow_tree = FALSE;
	f_stabilizer = FALSE;
	f_orbits_on_subsets = FALSE;
	orbits_on_subsets_size = 0;
	f_draw_poset = FALSE;
	f_draw_full_poset = FALSE;
	f_classes = FALSE;
	f_normalizer = FALSE;
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
	f_print_elements = FALSE;
	f_print_elements_tex = FALSE;
	f_multiply = FALSE;
	multiply_a = NULL;
	multiply_b = NULL;
	f_inverse = FALSE;
	inverse_a = NULL;
	f_order_of_products = FALSE;
	order_of_products_elements = NULL;
	f_group_table = FALSE;
	f_embedded = FALSE;
	f_sideways = FALSE;
	x_stretch = 1.;
	f_print_generators = FALSE;
	f_classify_arcs = FALSE;
	classify_arcs_target_size = 0;
	f_classify_arcs_d = FALSE;
	classify_arcs_d = 0;
	f_not_on_conic = FALSE;
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
	f_orbits_on_subspaces = FALSE;
	orbits_on_subspaces_depth = 0;
	f_mindist = FALSE;
	mindist = 0;
	f_self_orthogonal = FALSE;
	f_doubly_even = FALSE;
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
		else if (strcmp(argv[i], "-normalizer") == 0) {
			f_normalizer = TRUE;
			cout << "-normalizer" << endl;
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
		else if (strcmp(argv[i], "-classify_arcs") == 0) {
			f_classify_arcs = TRUE;
			classify_arcs_target_size = atoi(argv[++i]);
			cout << "-classify_arcs" << endl;
		}
		else if (strcmp(argv[i], "-classify_arcs_d") == 0) {
			f_classify_arcs_d = TRUE;
			classify_arcs_d = atoi(argv[++i]);
			cout << "-classify_arcs_d " << classify_arcs_d << endl;
		}
		else if (strcmp(argv[i], "-not_on_conic") == 0) {
			f_not_on_conic = TRUE;
			cout << "-not_on_conic" << endl;
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
					verbose_level) - 1;
			cout << "-surface_isomorphism_testing " << endl;
		}
		else if (strcmp(argv[i], "-surface_recognize") == 0) {
			f_surface_recognize = TRUE;
			cout << "-surface_recognize reading description of surface" << endl;
			surface_descr = NEW_OBJECT(surface_create_description);
			i += surface_descr->
					read_arguments(argc - (i - 1), argv + i,
					verbose_level) - 1;
			i += 2;
			cout << "-surface_recognize " << endl;
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
