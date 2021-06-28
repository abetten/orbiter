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
	f_multiply = FALSE;
	//multiply_a = NULL;
	//multiply_b = NULL;
	f_inverse = FALSE;
	//inverse_a = NULL;

	f_raise_to_the_power = FALSE;
	//std::string raise_to_the_power_a_text;
	//std::string raise_to_the_power_exponent_text;

	f_export_gap = FALSE;

	f_export_magma = FALSE;


	f_search_element_of_order = FALSE;
	search_element_order = 0;

	f_find_standard_generators = FALSE;
	find_standard_generators_order_a = 0;
	find_standard_generators_order_b = 0;
	find_standard_generators_order_ab = 0;

	f_element_rank = FALSE;
	// std::string element_rank_data;

	f_element_unrank = FALSE;
	//std::string element_unrank_data;

	f_find_singer_cycle = FALSE;



	//

	f_poset_classification_control = FALSE;
	Control = NULL;

	f_orbits_on_points = FALSE;

	f_export_trees = FALSE;
	//f_shallow_tree = FALSE;

	f_stabilizer = FALSE;

	f_orbits_on_subsets = FALSE;
	orbits_on_subsets_size = 0;

	f_orbits_on_subspaces = FALSE;
	orbits_on_subspaces_depth = 0;

	f_classes_based_on_normal_form = FALSE;
	f_classes = FALSE;
	f_group_table = FALSE;
	f_normalizer = FALSE;
	f_centralizer_of_element = FALSE;
	//element_description_text = NULL;
	//element_label = NULL;
	f_conjugacy_class_of_element = FALSE;
	f_orbits_on_group_elements_under_conjugation = FALSE;
	//std::string orbits_on_group_elements_under_conjugation_fname;
	//orbits_on_group_elements_under_conjugation_transporter_fname

	f_normalizer_of_cyclic_subgroup = FALSE;
	f_find_subgroup = FALSE;
	find_subgroup_order = 0;
	f_report = FALSE;
	f_sylow = FALSE;
	f_test_if_geometric = FALSE;
	test_if_geometric_depth = 0;
	f_draw_tree = FALSE;
	f_orbit_of = FALSE;
	orbit_of_idx = 0;
	f_orbits_on_set_system_from_file = FALSE;
	//orbits_on_set_system_from_file_fname = NULL;
	orbits_on_set_system_first_column = 0;
	orbits_on_set_system_number_of_columns = 0;
	f_orbit_of_set_from_file = FALSE;
	//orbit_of_set_from_file_fname = NULL;
	//f_search_subgroup = FALSE;


	f_conjugacy_class_of = FALSE;
	//std::string conjugacy_class_of_data;

	f_isomorphism_Klein_quadric = FALSE;
	//std::string isomorphism_Klein_quadric_fname;

	f_linear_codes = FALSE;
	linear_codes_minimum_distance = 0;
	linear_codes_target_size = 0;
	f_print_elements = FALSE;
	f_print_elements_tex = FALSE;

	f_order_of_products = FALSE;
	//order_of_products_elements = NULL;
	f_reverse_isomorphism_exterior_square = FALSE;


	// classification:

	f_classify_arcs = FALSE;
	Arc_generator_description = NULL;
	f_exact_cover = FALSE;
	ECA = NULL;
	f_isomorph_arguments = FALSE;
	IA = NULL;





	f_mindist = FALSE;
	mindist = 0;
	f_self_orthogonal = FALSE;
	f_doubly_even = FALSE;


	f_tensor_classify = FALSE;
	tensor_classify_depth = 0;
	f_tensor_permutations = FALSE;

	f_classify_ovoids = FALSE;
	Ovoid_classify_description = NULL;

	f_classify_cubic_curves = FALSE;


	f_orbits_on_polynomials = FALSE;
	orbits_on_polynomials_degree = 0;
	f_recognize_orbits_on_polynomials = FALSE;
	//std::string recognize_orbits_on_polynomials_text;
	f_orbits_on_polynomials_draw_tree = FALSE;
	orbits_on_polynomials_draw_tree_idx = 0;

	f_representation_on_polynomials = FALSE;
	representation_on_polynomials_degree = 0;


	f_Andre_Bruck_Bose_construction = FALSE;
	Andre_Bruck_Bose_construction_spread_no = 0;
	// Andre_Bruck_Bose_construction_label

	f_BLT_starter = FALSE;
	BLT_starter_size = 0;

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


int group_theoretic_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "group_theoretic_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {
		if (stringcmp(argv[i], "-multiply") == 0) {
			f_multiply = TRUE;
			multiply_a.assign(argv[++i]);
			multiply_b.assign(argv[++i]);
			if (f_v) {
				cout << "-multiply " << multiply_a << " " << multiply_b << endl;
			}
		}
		else if (stringcmp(argv[i], "-inverse") == 0) {
			f_inverse = TRUE;
			inverse_a.assign(argv[++i]);
			if (f_v) {
				cout << "-inverse " << inverse_a << endl;
			}
		}
		else if (stringcmp(argv[i], "-raise_to_the_power") == 0) {
			f_raise_to_the_power = TRUE;
			raise_to_the_power_a_text.assign(argv[++i]);
			raise_to_the_power_exponent_text.assign(argv[++i]);
			if (f_v) {
				cout << "-raise_to_the_power " << raise_to_the_power_a_text
						<< " " << raise_to_the_power_exponent_text << endl;
			}
		}

		else if (stringcmp(argv[i], "-export_gap") == 0) {
			f_export_gap = TRUE;
			if (f_v) {
				cout << "-export_gap " << endl;
			}
		}
		else if (stringcmp(argv[i], "-export_magma") == 0) {
			f_export_magma = TRUE;
			if (f_v) {
				cout << "-export_magma " << endl;
			}
		}

		else if (stringcmp(argv[i], "-search_element_of_order") == 0) {
			f_search_element_of_order = TRUE;
			search_element_order = strtoi(argv[++i]);
			if (f_v) {
				cout << "-search_element_of_order " << search_element_order << endl;
			}
		}
		else if (stringcmp(argv[i], "-find_standard_generators") == 0) {
			f_find_standard_generators = TRUE;
			find_standard_generators_order_a = strtoi(argv[++i]);
			find_standard_generators_order_b = strtoi(argv[++i]);
			find_standard_generators_order_ab = strtoi(argv[++i]);
			if (f_v) {
				cout << "-find_standard_generators " << find_standard_generators_order_a
						<< " " << find_standard_generators_order_b
						<< " " << find_standard_generators_order_ab
						<< endl;
			}
		}

		else if (stringcmp(argv[i], "-element_rank") == 0) {
			f_element_rank = TRUE;
			element_rank_data.assign(argv[++i]);
			if (f_v) {
				cout << "-element_rank " << element_rank_data << endl;
			}
		}
		else if (stringcmp(argv[i], "-element_unrank") == 0) {
			f_element_unrank = TRUE;
			element_unrank_data.assign(argv[++i]);
			if (f_v) {
				cout << "-element_unrank " << element_unrank_data << endl;
			}
		}
		else if (stringcmp(argv[i], "-find_singer_cycle") == 0) {
			f_find_singer_cycle = TRUE;
			if (f_v) {
				cout << "-find_singer_cycle " << endl;
			}
		}




		else if (stringcmp(argv[i], "-poset_classification_control") == 0) {
			f_poset_classification_control = TRUE;
			Control = NEW_OBJECT(poset_classification_control);
			if (f_v) {
				cout << "-poset_classification_control " << endl;
			}
			i += Control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -poset_classification_control " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}
		else if (stringcmp(argv[i], "-orbits_on_points") == 0) {
			f_orbits_on_points = TRUE;
			if (f_v) {
				cout << "-orbits_on_points" << endl;
			}
		}

		else if (stringcmp(argv[i], "-orbits_on_subsets") == 0) {
			f_orbits_on_subsets = TRUE;
			orbits_on_subsets_size = strtoi(argv[++i]);
			if (f_v) {
				cout << "-orbits_on_subsets " << orbits_on_subsets_size << endl;
			}
		}

		else if (stringcmp(argv[i], "-orbits_on_subspaces") == 0) {
			f_orbits_on_subspaces = TRUE;
			orbits_on_subspaces_depth = strtoi(argv[++i]);
			if (f_v) {
				cout << "-orbits_on_subspaces " << orbits_on_subspaces_depth << endl;
			}
		}

		else if (stringcmp(argv[i], "-export_trees") == 0) {
			f_export_trees = TRUE;
			if (f_v) {
				cout << "-export_trees" << endl;
			}
		}

		else if (stringcmp(argv[i], "-stabilizer") == 0) {
			f_stabilizer = TRUE;
			if (f_v) {
				cout << "-stabilizer" << endl;
			}
		}
		else if (stringcmp(argv[i], "-test_if_geometric") == 0) {
			f_test_if_geometric = TRUE;
			test_if_geometric_depth = strtoi(argv[++i]);
			if (f_v) {
				cout << "-test_if_geometric" << endl;
			}
		}
		else if (stringcmp(argv[i], "-classes_based_on_normal_form") == 0) {
			f_classes_based_on_normal_form = TRUE;
			if (f_v) {
				cout << "-classes_based_on_normal_form" << endl;
			}
		}
		else if (stringcmp(argv[i], "-classes") == 0) {
			f_classes = TRUE;
			if (f_v) {
				cout << "-classes" << endl;
			}
		}
		else if (stringcmp(argv[i], "-group_table") == 0) {
			f_group_table = TRUE;
			if (f_v) {
				cout << "-group_table" << endl;
			}
		}
		else if (stringcmp(argv[i], "-normalizer") == 0) {
			f_normalizer = TRUE;
			if (f_v) {
				cout << "-normalizer" << endl;
			}
		}
		else if (stringcmp(argv[i], "-centralizer_of_element") == 0) {
			f_centralizer_of_element = TRUE;
			element_label.assign(argv[++i]);
			element_description_text.assign(argv[++i]);
			if (f_v) {
				cout << "-centralizer_of_element " << element_label
						<< " " << element_description_text << endl;
			}
		}
		else if (stringcmp(argv[i], "-conjugacy_class_of_element") == 0) {
			f_conjugacy_class_of_element = TRUE;
			element_label.assign(argv[++i]);
			element_description_text.assign(argv[++i]);
			if (f_v) {
				cout << "-conjugacy_class_of_element " << element_label
						<< " " << element_description_text << endl;
			}
		}
		else if (stringcmp(argv[i], "-orbits_on_group_elements_under_conjugation") == 0) {
			f_orbits_on_group_elements_under_conjugation = TRUE;
			orbits_on_group_elements_under_conjugation_fname.assign(argv[++i]);
			orbits_on_group_elements_under_conjugation_transporter_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-orbits_on_group_elements_under_conjugation "
						<< " " << orbits_on_group_elements_under_conjugation_fname
						<< " " << orbits_on_group_elements_under_conjugation_transporter_fname
						<< endl;
			}
		}


		else if (stringcmp(argv[i], "-normalizer_of_cyclic_subgroup") == 0) {
			f_normalizer_of_cyclic_subgroup = TRUE;
			element_label.assign(argv[++i]);
			element_description_text.assign(argv[++i]);
			if (f_v) {
				cout << "-normalizer_of_cyclic_subgroup " << element_label
						<< " " << element_description_text << endl;
			}
		}
		else if (stringcmp(argv[i], "-find_subgroup") == 0) {
			f_find_subgroup = TRUE;
			find_subgroup_order = strtoi(argv[++i]);
			if (f_v) {
				cout << "-find_subgroup " << find_subgroup_order << endl;
			}
		}
		else if (stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			if (f_v) {
				cout << "-report" << endl;
			}
		}
		else if (stringcmp(argv[i], "-sylow") == 0) {
			f_sylow = TRUE;
			if (f_v) {
				cout << "-sylow" << endl;
			}
		}
		else if (stringcmp(argv[i], "-f_draw_tree") == 0) {
			f_draw_tree = TRUE;
			if (f_v) {
				cout << "-f_draw_tree " << endl;
			}
		}
		else if (stringcmp(argv[i], "-orbit_of") == 0) {
			f_orbit_of = TRUE;
			orbit_of_idx = strtoi(argv[++i]);
			if (f_v) {
				cout << "-orbit_of " << orbit_of_idx << endl;
			}
		}
		else if (stringcmp(argv[i], "-orbit_of_set_from_file") == 0) {
			f_orbit_of_set_from_file = TRUE;
			orbit_of_set_from_file_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-orbit_of_set_from_file"
						<< orbit_of_set_from_file_fname << endl;
			}
		}
		else if (stringcmp(argv[i], "-orbits_on_set_system_from_file") == 0) {
			f_orbits_on_set_system_from_file = TRUE;
			orbits_on_set_system_from_file_fname.assign(argv[++i]);
			orbits_on_set_system_first_column = strtoi(argv[++i]);
			orbits_on_set_system_number_of_columns = strtoi(argv[++i]);
			if (f_v) {
				cout << "-orbits_on_set_system_from_file"
						<< orbits_on_set_system_from_file_fname
						<< " " << orbits_on_set_system_first_column << " "
						<< orbits_on_set_system_number_of_columns << endl;
			}
		}

		else if (stringcmp(argv[i], "-conjugacy_class_of") == 0) {
			f_conjugacy_class_of = TRUE;
			conjugacy_class_of_data.assign(argv[++i]);
			if (f_v) {
				cout << "-conjugacy_class_of " << conjugacy_class_of_data << endl;
			}
		}

		else if (stringcmp(argv[i], "-isomorphism_Klein_quadric") == 0) {
			f_isomorphism_Klein_quadric = TRUE;
			isomorphism_Klein_quadric_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-isomorphism_Klein_quadric " << isomorphism_Klein_quadric_fname << endl;
			}
		}

		else if (stringcmp(argv[i], "-print_elements") == 0) {
			f_print_elements = TRUE;
			if (f_v) {
				cout << "-print_elements " << endl;
			}
		}
		else if (stringcmp(argv[i], "-print_elements_tex") == 0) {
			f_print_elements_tex = TRUE;
			if (f_v) {
				cout << "-print_elements_tex " << endl;
			}
		}
		else if (stringcmp(argv[i], "-order_of_products") == 0) {
			f_order_of_products = TRUE;
			order_of_products_elements.assign(argv[++i]);
			if (f_v) {
				cout << "-order_of_products " << order_of_products_elements << endl;
			}
		}
		else if (stringcmp(argv[i], "-reverse_isomorphism_exterior_square") == 0) {
			f_reverse_isomorphism_exterior_square = TRUE;
			if (f_v) {
				cout << "-reverse_isomorphism_exterior_square " << endl;
			}
		}



		// classification tasks:

		// linear codes:

		else if (stringcmp(argv[i], "-linear_codes") == 0) {
			f_linear_codes = TRUE;
			linear_codes_minimum_distance = strtoi(argv[++i]);
			linear_codes_target_size = strtoi(argv[++i]);
			if (f_v) {
				cout << "-linear_codes " << linear_codes_minimum_distance
					<< " " << linear_codes_target_size << endl;
			}
		}


		// arcs:


		else if (stringcmp(argv[i], "-classify_arcs") == 0) {
			f_classify_arcs = TRUE;
			Arc_generator_description = NEW_OBJECT(arc_generator_description);
			if (f_v) {
				cout << "-classify_arcs" << endl;
			}
			i += Arc_generator_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -classify_arcs " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}

		else if (stringcmp(argv[i], "-exact_cover") == 0) {
			f_exact_cover = TRUE;
			ECA = NEW_OBJECT(exact_cover_arguments);
			i += ECA->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done with -exact_cover" << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}
		else if (stringcmp(argv[i], "-isomorph_arguments") == 0) {
			f_isomorph_arguments = TRUE;
			IA = NEW_OBJECT(isomorph_arguments);
			i += IA->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done with -isomorph_arguments" << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}







		else if (stringcmp(argv[i], "-mindist") == 0) {
			f_mindist = TRUE;
			mindist = strtoi(argv[++i]);
			if (f_v) {
				cout << "-mindist" << mindist << endl;
			}
		}
		else if (stringcmp(argv[i], "-self_orthogonal") == 0) {
			f_self_orthogonal = TRUE;
			if (f_v) {
				cout << "-self_orthogonal" << endl;
			}
		}
		else if (stringcmp(argv[i], "-doubly_even") == 0) {
			f_doubly_even = TRUE;
			if (f_v) {
				cout << "-doubly_even" << endl;
			}
		}



		// tensors:

		else if (stringcmp(argv[i], "-tensor_classify") == 0) {
			f_tensor_classify = TRUE;
			tensor_classify_depth = strtoi(argv[++i]);
			if (f_v) {
				cout << "-tensor_classify " << tensor_classify_depth << endl;
			}
		}
		else if (stringcmp(argv[i], "-tensor_permutations") == 0) {
			f_tensor_permutations = TRUE;
			if (f_v) {
				cout << "-tensor_permutations " << endl;
			}
		}


		// ovoids:

		else if (stringcmp(argv[i], "-classify_ovoids") == 0) {
			f_classify_ovoids = TRUE;
			Ovoid_classify_description = NEW_OBJECT(ovoid_classify_description);
			if (f_v) {
				cout << "-classify_ovoids" << endl;
			}
			i += Ovoid_classify_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -classify_ovoids " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}

		// cubic curves
		else if (stringcmp(argv[i], "-classify_cubic_curves") == 0) {
			f_classify_cubic_curves = TRUE;
			Arc_generator_description = NEW_OBJECT(arc_generator_description);
			if (f_v) {
				cout << "-classify_cubic_curves" << endl;
			}
			i += Arc_generator_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -classify_cubic_curves " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-classify_cubic_curves " << endl;
			}
		}



		// other:
		else if (stringcmp(argv[i], "-orbits_on_polynomials") == 0) {
			f_orbits_on_polynomials = TRUE;
			orbits_on_polynomials_degree = strtoi(argv[++i]);
			if (f_v) {
				cout << "-orbits_on_polynomials " << endl;
			}
		}
		else if (stringcmp(argv[i], "-orbits_on_polynomials_draw_tree") == 0) {
			f_orbits_on_polynomials_draw_tree = TRUE;
			orbits_on_polynomials_draw_tree_idx = strtoi(argv[++i]);
			if (f_v) {
				cout << "-orbits_on_polynomials_draw_tree " << orbits_on_polynomials_draw_tree_idx << endl;
			}
		}


		else if (stringcmp(argv[i], "-recognize_orbits_on_polynomials") == 0) {
			f_recognize_orbits_on_polynomials = TRUE;
			recognize_orbits_on_polynomials_text.assign(argv[++i]);
			if (f_v) {
				cout << "-recognize_orbits_on_polynomials " << endl;
			}
		}


		else if (stringcmp(argv[i], "-representation_on_polynomials") == 0) {
			f_representation_on_polynomials = TRUE;
			representation_on_polynomials_degree = strtoi(argv[++i]);
			if (f_v) {
				cout << "-representation_on_polynomials " << representation_on_polynomials_degree << endl;
			}
		}

		else if (stringcmp(argv[i], "-Andre_Bruck_Bose_construction") == 0) {
			f_Andre_Bruck_Bose_construction = TRUE;
			Andre_Bruck_Bose_construction_spread_no = strtoi(argv[++i]);
			Andre_Bruck_Bose_construction_label.assign(argv[++i]);
			if (f_v) {
				cout << "-Andre_Bruck_Bose_construction " << Andre_Bruck_Bose_construction_spread_no
					<< " " << Andre_Bruck_Bose_construction_label << endl;
			}
		}

		else if (stringcmp(argv[i], "-BLT_starter") == 0) {
			f_BLT_starter = TRUE;
			BLT_starter_size = strtoi(argv[++i]);
			if (f_v) {
				cout << "-BLT_starter " << BLT_starter_size << endl;
			}
		}


		else if (stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "group_theoretic_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "group_theoretic_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void group_theoretic_activity_description::print()
{
	if (f_multiply) {
		cout << "-multiply " << multiply_a << " " << multiply_b << endl;
	}
	if (f_inverse) {
		cout << "-inverse " << inverse_a << endl;
	}
	if (f_raise_to_the_power) {
		cout << "-raise_to_the_power " << raise_to_the_power_a_text
					<< " " << raise_to_the_power_exponent_text << endl;
	}

	if (f_export_gap) {
		cout << "-export_gap " << endl;
	}
	if (f_export_magma) {
		cout << "-export_magma " << endl;
	}

	if (f_search_element_of_order) {
		cout << "-search_element_of_order " << search_element_order << endl;
	}
	if (f_find_standard_generators) {
		cout << "-find_standard_generators " << find_standard_generators_order_a
					<< " " << find_standard_generators_order_b
					<< " " << find_standard_generators_order_ab
					<< endl;
	}

	if (f_element_rank) {
		cout << "-element_rank " << element_rank_data << endl;
	}
	if (f_element_unrank) {
		cout << "-element_unrank " << element_unrank_data << endl;
	}
	if (f_find_singer_cycle) {
		cout << "-find_singer_cycle " << endl;
	}




	if (f_poset_classification_control) {
		Control->print();
	}
	if (f_orbits_on_points) {
		cout << "-orbits_on_points" << endl;
	}

	if (f_orbits_on_subsets) {
		cout << "-orbits_on_subsets " << orbits_on_subsets_size << endl;
	}

	if (f_orbits_on_subspaces) {
		cout << "-orbits_on_subspaces " << orbits_on_subspaces_depth << endl;
	}

	if (f_export_trees) {
		cout << "-export_trees" << endl;
	}

	if (f_stabilizer) {
		cout << "-stabilizer" << endl;
	}
	if (f_test_if_geometric) {
		cout << "-test_if_geometric" << endl;
	}
	if (f_classes_based_on_normal_form) {
		cout << "-classes_based_on_normal_form" << endl;
	}
	if (f_classes) {
		cout << "-classes" << endl;
	}
	if (f_group_table) {
		cout << "-group_table" << endl;
	}
	if (f_normalizer) {
		cout << "-normalizer" << endl;
	}
	if (f_centralizer_of_element) {
		cout << "-centralizer_of_element " << element_label
				<< " " << element_description_text << endl;
	}
	if (f_conjugacy_class_of_element) {
		cout << "-conjugacy_class_of_element " << element_label
					<< " " << element_description_text << endl;
	}
	if (f_orbits_on_group_elements_under_conjugation) {
		cout << "-orbits_on_group_elements_under_conjugation "
					<< " " << orbits_on_group_elements_under_conjugation_fname
					<< " " << orbits_on_group_elements_under_conjugation_transporter_fname
					<< endl;
	}


	if (f_normalizer_of_cyclic_subgroup) {
		cout << "-normalizer_of_cyclic_subgroup " << element_label
					<< " " << element_description_text << endl;
	}
	if (f_find_subgroup) {
		cout << "-find_subgroup " << find_subgroup_order << endl;
	}
	if (f_report) {
		cout << "-report" << endl;
	}
	if (f_sylow) {
		cout << "-sylow" << endl;
	}
	if (f_draw_tree) {
		cout << "-f_draw_tree " << endl;
	}
	if (f_orbit_of) {
		cout << "-orbit_of " << orbit_of_idx << endl;
	}
	if (f_orbit_of_set_from_file) {
		cout << "-orbit_of_set_from_file"
					<< orbit_of_set_from_file_fname << endl;
	}
	if (f_orbits_on_set_system_from_file) {
		cout << "-orbits_on_set_system_from_file"
				<< orbits_on_set_system_from_file_fname
				<< " " << orbits_on_set_system_first_column << " "
				<< orbits_on_set_system_number_of_columns << endl;
	}

	if (f_conjugacy_class_of) {
		cout << "-conjugacy_class_of " << conjugacy_class_of_data << endl;
	}

	if (f_isomorphism_Klein_quadric) {
		cout << "-isomorphism_Klein_quadric " << isomorphism_Klein_quadric_fname << endl;
	}

	if (f_print_elements) {
		cout << "-print_elements " << endl;
	}
	if (f_print_elements_tex) {
		cout << "-print_elements_tex " << endl;
	}
	if (f_order_of_products) {
		cout << "-order_of_products " << order_of_products_elements << endl;
	}
	if (f_reverse_isomorphism_exterior_square) {
		cout << "-reverse_isomorphism_exterior_square " << endl;
	}



	// classification tasks:

	// linear codes:

	if (f_linear_codes) {
			cout << "-linear_codes " << linear_codes_minimum_distance
				<< " " << linear_codes_target_size << endl;
	}


	// arcs:


	if (f_classify_arcs) {
		cout << "-classify_arcs " << endl;
		Arc_generator_description->print();
	}

	if (f_exact_cover) {
		cout << "-exact_cover" << endl;
	}
	if (f_isomorph_arguments) {
		cout << "-isomorph_arguments" << endl;
	}







	if (f_mindist) {
		cout << "-mindist" << mindist << endl;
	}
	if (f_self_orthogonal) {
		cout << "-self_orthogonal" << endl;
	}
	if (f_doubly_even) {
		cout << "-doubly_even" << endl;
	}



	// tensors:

	if (f_tensor_classify) {
		cout << "-tensor_classify " << tensor_classify_depth << endl;
	}
	if (f_tensor_permutations) {
		cout << "-tensor_permutations " << endl;
	}


	// ovoids:

	if (f_classify_ovoids) {
		cout << "-classify_ovoids" << endl;
		Ovoid_classify_description->print();
	}

	// cubic curves
	if (f_classify_cubic_curves) {
		cout << "-classify_cubic_curves" << endl;
		Arc_generator_description->print();
	}



	// other:
	if (f_orbits_on_polynomials) {
		cout << "-orbits_on_polynomials " << endl;
	}
	if (f_orbits_on_polynomials_draw_tree) {
			cout << "-orbits_on_polynomials_draw_tree " << orbits_on_polynomials_draw_tree_idx << endl;
	}


	if (f_recognize_orbits_on_polynomials) {
		cout << "-recognize_orbits_on_polynomials " << endl;
	}


	if (f_representation_on_polynomials) {
		cout << "-representation_on_polynomials " << representation_on_polynomials_degree << endl;
	}

	if (f_Andre_Bruck_Bose_construction) {
		cout << "-Andre_Bruck_Bose_construction " << Andre_Bruck_Bose_construction_spread_no
			<< " " << Andre_Bruck_Bose_construction_label << endl;
	}

	if (f_BLT_starter) {
		cout << "-BLT_starter " << BLT_starter_size << endl;
	}
}





}}
