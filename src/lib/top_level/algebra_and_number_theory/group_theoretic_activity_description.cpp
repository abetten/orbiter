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
	int i;

	cout << "group_theoretic_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {
		if (stringcmp(argv[i], "-multiply") == 0) {
			f_multiply = TRUE;
			multiply_a.assign(argv[++i]);
			multiply_b.assign(argv[++i]);
			cout << "-multiply " << multiply_a << " " << multiply_b << endl;
		}
		else if (stringcmp(argv[i], "-inverse") == 0) {
			f_inverse = TRUE;
			inverse_a.assign(argv[++i]);
			cout << "-inverse " << inverse_a << endl;
		}
		else if (stringcmp(argv[i], "-raise_to_the_power") == 0) {
			f_raise_to_the_power = TRUE;
			raise_to_the_power_a_text.assign(argv[++i]);
			raise_to_the_power_exponent_text.assign(argv[++i]);
			cout << "-raise_to_the_power " << raise_to_the_power_a_text
					<< " " << raise_to_the_power_exponent_text << endl;
		}

		else if (stringcmp(argv[i], "-export_gap") == 0) {
			f_export_gap = TRUE;
			cout << "-export_gap " << endl;
		}
		else if (stringcmp(argv[i], "-export_magma") == 0) {
			f_export_magma = TRUE;
			cout << "-export_magma " << endl;
		}

		else if (stringcmp(argv[i], "-search_element_of_order") == 0) {
			f_search_element_of_order = TRUE;
			search_element_order = strtoi(argv[++i]);
			cout << "-search_element_of_order " << search_element_order << endl;
		}
		else if (stringcmp(argv[i], "-element_rank") == 0) {
			f_element_rank = TRUE;
			element_rank_data.assign(argv[++i]);
			cout << "-element_rank " << element_rank_data << endl;
		}
		else if (stringcmp(argv[i], "-element_unrank") == 0) {
			f_element_unrank = TRUE;
			element_unrank_data.assign(argv[++i]);
			cout << "-element_unrank " << element_unrank_data << endl;
		}
		else if (stringcmp(argv[i], "-find_singer_cycle") == 0) {
			f_find_singer_cycle = TRUE;
			cout << "-find_singer_cycle " << endl;
		}




		else if (stringcmp(argv[i], "-poset_classification_control") == 0) {
			f_poset_classification_control = TRUE;
			Control = NEW_OBJECT(poset_classification_control);
			cout << "-poset_classification_control " << endl;
			i += Control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -poset_classification_control " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (stringcmp(argv[i], "-orbits_on_points") == 0) {
			f_orbits_on_points = TRUE;
			cout << "-orbits_on_points" << endl;
		}

		else if (stringcmp(argv[i], "-orbits_on_subsets") == 0) {
			f_orbits_on_subsets = TRUE;
			orbits_on_subsets_size = strtoi(argv[++i]);
			cout << "-orbits_on_subsets " << orbits_on_subsets_size << endl;
		}

		else if (stringcmp(argv[i], "-orbits_on_subspaces") == 0) {
			f_orbits_on_subspaces = TRUE;
			orbits_on_subspaces_depth = strtoi(argv[++i]);
			cout << "-orbits_on_subspaces " << orbits_on_subspaces_depth << endl;
		}

		else if (stringcmp(argv[i], "-export_trees") == 0) {
			f_export_trees = TRUE;
			cout << "-export_trees" << endl;
		}

		else if (stringcmp(argv[i], "-stabilizer") == 0) {
			f_stabilizer = TRUE;
			cout << "-stabilizer" << endl;
		}
		else if (stringcmp(argv[i], "-test_if_geometric") == 0) {
			f_test_if_geometric = TRUE;
			test_if_geometric_depth = strtoi(argv[++i]);
			cout << "-test_if_geometric" << endl;
		}
		else if (stringcmp(argv[i], "-classes_based_on_normal_form") == 0) {
			f_classes_based_on_normal_form = TRUE;
			cout << "-classes_based_on_normal_form" << endl;
		}
		else if (stringcmp(argv[i], "-classes") == 0) {
			f_classes = TRUE;
			cout << "-classes" << endl;
		}
		else if (stringcmp(argv[i], "-group_table") == 0) {
			f_group_table = TRUE;
			cout << "-group_table" << endl;
		}
		else if (stringcmp(argv[i], "-normalizer") == 0) {
			f_normalizer = TRUE;
			cout << "-normalizer" << endl;
		}
		else if (stringcmp(argv[i], "-centralizer_of_element") == 0) {
			f_centralizer_of_element = TRUE;
			element_label.assign(argv[++i]);
			element_description_text.assign(argv[++i]);
			cout << "-centralizer_of_element " << element_label
					<< " " << element_description_text << endl;
		}
		else if (stringcmp(argv[i], "-conjugacy_class_of_element") == 0) {
			f_conjugacy_class_of_element = TRUE;
			element_label.assign(argv[++i]);
			element_description_text.assign(argv[++i]);
			cout << "-conjugacy_class_of_element " << element_label
					<< " " << element_description_text << endl;
		}
		else if (stringcmp(argv[i], "-orbits_on_group_elements_under_conjugation") == 0) {
			f_orbits_on_group_elements_under_conjugation = TRUE;
			orbits_on_group_elements_under_conjugation_fname.assign(argv[++i]);
			orbits_on_group_elements_under_conjugation_transporter_fname.assign(argv[++i]);
			cout << "-orbits_on_group_elements_under_conjugation "
					<< " " << orbits_on_group_elements_under_conjugation_fname
					<< " " << orbits_on_group_elements_under_conjugation_transporter_fname
					<< endl;
		}


		else if (stringcmp(argv[i], "-normalizer_of_cyclic_subgroup") == 0) {
			f_normalizer_of_cyclic_subgroup = TRUE;
			element_label.assign(argv[++i]);
			element_description_text.assign(argv[++i]);
			cout << "-normalizer_of_cyclic_subgroup " << element_label
					<< " " << element_description_text << endl;
		}
		else if (stringcmp(argv[i], "-find_subgroup") == 0) {
			f_find_subgroup = TRUE;
			find_subgroup_order = strtoi(argv[++i]);
			cout << "-find_subgroup " << find_subgroup_order << endl;
		}
		else if (stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report" << endl;
		}
		else if (stringcmp(argv[i], "-sylow") == 0) {
			f_sylow = TRUE;
			cout << "-sylow" << endl;
		}
		else if (stringcmp(argv[i], "-f_draw_tree") == 0) {
			f_draw_tree = TRUE;
			cout << "-f_draw_tree " << endl;
		}
		else if (stringcmp(argv[i], "-orbit_of") == 0) {
			f_orbit_of = TRUE;
			orbit_of_idx = strtoi(argv[++i]);
			cout << "-orbit_of " << orbit_of_idx << endl;
		}
		else if (stringcmp(argv[i], "-orbit_of_set_from_file") == 0) {
			f_orbit_of_set_from_file = TRUE;
			orbit_of_set_from_file_fname.assign(argv[++i]);
			cout << "-orbit_of_set_from_file"
					<< orbit_of_set_from_file_fname << endl;
		}
		else if (stringcmp(argv[i], "-orbits_on_set_system_from_file") == 0) {
			f_orbits_on_set_system_from_file = TRUE;
			orbits_on_set_system_from_file_fname.assign(argv[++i]);
			orbits_on_set_system_first_column = strtoi(argv[++i]);
			orbits_on_set_system_number_of_columns = strtoi(argv[++i]);
			cout << "-orbits_on_set_system_from_file"
					<< orbits_on_set_system_from_file_fname
					<< " " << orbits_on_set_system_first_column << " "
					<< orbits_on_set_system_number_of_columns << endl;
		}
#if 0
		else if (stringcmp(argv[i], "-search_subgroup") == 0) {
			f_search_subgroup = TRUE;
			cout << "-search_subgroup " << endl;
		}
#endif

		else if (stringcmp(argv[i], "-conjugacy_class_of") == 0) {
			f_conjugacy_class_of = TRUE;
			conjugacy_class_of_data.assign(argv[++i]);
			cout << "-conjugacy_class_of " << conjugacy_class_of_data << endl;
		}

		else if (stringcmp(argv[i], "-isomorphism_Klein_quadric") == 0) {
			f_isomorphism_Klein_quadric = TRUE;
			isomorphism_Klein_quadric_fname.assign(argv[++i]);
			cout << "-isomorphism_Klein_quadric " << isomorphism_Klein_quadric_fname << endl;
		}

		else if (stringcmp(argv[i], "-print_elements") == 0) {
			f_print_elements = TRUE;
			cout << "-print_elements " << endl;
		}
		else if (stringcmp(argv[i], "-print_elements_tex") == 0) {
			f_print_elements_tex = TRUE;
			cout << "-print_elements_tex " << endl;
		}
		else if (stringcmp(argv[i], "-order_of_products") == 0) {
			f_order_of_products = TRUE;
			order_of_products_elements.assign(argv[++i]);
			cout << "-order_of_products " << order_of_products_elements << endl;
		}
		else if (stringcmp(argv[i], "-reverse_isomorphism_exterior_square") == 0) {
			f_reverse_isomorphism_exterior_square = TRUE;
			cout << "-reverse_isomorphism_exterior_square " << endl;
		}



		// classification tasks:

		// linear codes:

		else if (stringcmp(argv[i], "-linear_codes") == 0) {
			f_linear_codes = TRUE;
			linear_codes_minimum_distance = strtoi(argv[++i]);
			linear_codes_target_size = strtoi(argv[++i]);
			cout << "-linear_codes " << linear_codes_minimum_distance
					<< " " << linear_codes_target_size << endl;
		}


		// arcs:


		else if (stringcmp(argv[i], "-classify_arcs") == 0) {
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

		else if (stringcmp(argv[i], "-exact_cover") == 0) {
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
		else if (stringcmp(argv[i], "-isomorph_arguments") == 0) {
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







		else if (stringcmp(argv[i], "-mindist") == 0) {
			f_mindist = TRUE;
			mindist = strtoi(argv[++i]);
			cout << "-mindist" << mindist << endl;
		}
		else if (stringcmp(argv[i], "-self_orthogonal") == 0) {
			f_self_orthogonal = TRUE;
			cout << "-self_orthogonal" << endl;
		}
		else if (stringcmp(argv[i], "-doubly_even") == 0) {
			f_doubly_even = TRUE;
			cout << "-doubly_even" << endl;
		}



		// tensors:

		else if (stringcmp(argv[i], "-tensor_classify") == 0) {
			f_tensor_classify = TRUE;
			tensor_classify_depth = strtoi(argv[++i]);
			cout << "-tensor_classify " << tensor_classify_depth << endl;
		}
		else if (stringcmp(argv[i], "-tensor_permutations") == 0) {
			f_tensor_permutations = TRUE;
			cout << "-tensor_permutations " << endl;
		}


		// ovoids:

		else if (stringcmp(argv[i], "-classify_ovoids") == 0) {
			f_classify_ovoids = TRUE;
			Ovoid_classify_description = NEW_OBJECT(ovoid_classify_description);
			cout << "-classify_ovoids" << endl;
			i += Ovoid_classify_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -classify_ovoids " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}

		// cubic curves
		else if (stringcmp(argv[i], "-classify_cubic_curves") == 0) {
			f_classify_cubic_curves = TRUE;
			Arc_generator_description = NEW_OBJECT(arc_generator_description);
			cout << "-classify_cubic_curves" << endl;
			i += Arc_generator_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -classify_cubic_curves " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-classify_cubic_curves " << endl;
		}



		// other:
		else if (stringcmp(argv[i], "-orbits_on_polynomials") == 0) {
			f_orbits_on_polynomials = TRUE;
			orbits_on_polynomials_degree = strtoi(argv[++i]);
			cout << "-orbits_on_polynomials " << endl;
		}
		else if (stringcmp(argv[i], "-orbits_on_polynomials_draw_tree") == 0) {
			f_orbits_on_polynomials_draw_tree = TRUE;
			orbits_on_polynomials_draw_tree_idx = strtoi(argv[++i]);
			cout << "-orbits_on_polynomials_draw_tree " << orbits_on_polynomials_draw_tree_idx << endl;
		}


		else if (stringcmp(argv[i], "-recognize_orbits_on_polynomials") == 0) {
			f_recognize_orbits_on_polynomials = TRUE;
			recognize_orbits_on_polynomials_text.assign(argv[++i]);
			cout << "-recognize_orbits_on_polynomials " << endl;
		}


		else if (stringcmp(argv[i], "-representation_on_polynomials") == 0) {
			f_representation_on_polynomials = TRUE;
			representation_on_polynomials_degree = strtoi(argv[++i]);
			cout << "-representation_on_polynomials " << representation_on_polynomials_degree << endl;
		}

		else if (stringcmp(argv[i], "-Andre_Bruck_Bose_construction") == 0) {
			f_Andre_Bruck_Bose_construction = TRUE;
			Andre_Bruck_Bose_construction_spread_no = strtoi(argv[++i]);
			Andre_Bruck_Bose_construction_label.assign(argv[++i]);
			cout << "-Andre_Bruck_Bose_construction " << Andre_Bruck_Bose_construction_spread_no
					<< " " << Andre_Bruck_Bose_construction_label << endl;
		}

		else if (stringcmp(argv[i], "-BLT_starter") == 0) {
			f_BLT_starter = TRUE;
			BLT_starter_size = strtoi(argv[++i]);
			cout << "-BLT_starter " << BLT_starter_size << endl;
		}


		else if (stringcmp(argv[i], "-end") == 0) {
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
