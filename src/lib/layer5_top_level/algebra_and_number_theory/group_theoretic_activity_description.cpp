/*
 * group_theoretic_activity_description.cpp
 *
 *  Created on: Apr 26, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {





group_theoretic_activity_description::group_theoretic_activity_description()
{

	f_report = false;
	f_report_sylow = false;
	f_report_group_table = false;
	//f_report_classes = false;

	f_export_group_table = false;


	f_random_element = false;
	//std::string random_element_label;


	f_permutation_representation_of_element = false;
	//std::string permutation_representation_element_text;


	f_apply = false;
	//std::string apply_input;
	//std::string apply_element;

	f_element_processing = false;
	element_processing_descr = NULL;

	f_multiply = false;
	//multiply_a = NULL;
	//multiply_b = NULL;

	f_inverse = false;
	//inverse_a = NULL;

	f_consecutive_powers = false;
	//std::string consecutive_powers_a_text;
	//std::string consecutive_powers_exponent_text;

	f_raise_to_the_power = false;
	//std::string raise_to_the_power_a_text;
	//std::string raise_to_the_power_exponent_text;

	f_export_orbiter = false;

	f_export_gap = false;

	f_export_magma = false;

	// GAP
	f_canonical_image_GAP = false;
	//std::string canonical_image_GAP_input_set;

	f_canonical_image = false;
	//std::string canonical_image_input_set;

	f_search_element_of_order = false;
	search_element_order = 0;

	f_find_standard_generators = false;
	find_standard_generators_order_a = 0;
	find_standard_generators_order_b = 0;
	find_standard_generators_order_ab = 0;


	f_element_rank = false;
	// std::string element_rank_data;

	f_element_unrank = false;
	//std::string element_unrank_data;

	f_find_singer_cycle = false;

	f_classes_based_on_normal_form = false;


	// Magma:
	f_normalizer = false;

	// Magma:
	f_centralizer_of_element = false;
	//std::string centralizer_of_element_label;
	//std::string centralizer_of_element_data;

#if 0
	f_orbits_on_group_elements_under_conjugation = false;
	//std::string orbits_on_group_elements_under_conjugation_fname;
	//orbits_on_group_elements_under_conjugation_transporter_fname
#endif

	// Magma:
	f_normalizer_of_cyclic_subgroup = false;
	//std::string normalizer_of_cyclic_subgroup_label;
	//std::string normalizer_of_cyclic_subgroup_data;

	// Magma:
	f_classes = false;

	f_subgroup_lattice_magma = false;


	// undocumented:
	f_find_subgroup = false;
	find_subgroup_order = 0;


	//f_test_if_geometric = false;
	//test_if_geometric_depth = 0;



	f_conjugacy_class_of = false;
	//std::string conjugacy_class_of_label;
	//std::string conjugacy_class_of_data;

	f_isomorphism_Klein_quadric = false;
	//std::string isomorphism_Klein_quadric_fname;

	f_print_elements = false;
	f_print_elements_tex = false;

	f_save_elements_csv = false;
	//std::string save_elements_csv_fname;

	f_export_inversion_graphs = false;
	//std::string export_inversion_graphs_fname;


	f_evaluate_word = false;
	//std::string evaluate_word_word;
	//std::string evaluate_word_gens;

	f_multiply_all_elements_in_lex_order = false;


	f_stats = false;
	//std::string stats_fname_base;


	f_move_a_to_b = false;
	move_a_to_b_a = -1;
	move_a_to_b_b = -1;

	f_rational_normal_form = false;
	//std::string rational_normal_form_input;


	f_find_conjugating_element = false;
	//std::string find_conjugating_element_element_from;
	//std::string find_conjugating_element_element_to;


	f_group_of_automorphisms_by_images_of_generators = false;
	//std::string group_of_automorphisms_by_images_of_generators_label;
	//std::string group_of_automorphisms_by_images_of_generators_elements;
	//std::string group_of_automorphisms_by_images_of_generators_images;




	//  3:

#if 0
	f_multiply_elements_csv_column_major_ordering = false;
	//std::string multiply_elements_csv_column_major_ordering_fname1;
	//std::string multiply_elements_csv_column_major_ordering_fname2;
	//std::string multiply_elements_csv_column_major_ordering_fname3;

	f_multiply_elements_csv_row_major_ordering = false;
	//std::string multiply_elements_csv_row_major_ordering_fname1;
	//std::string multiply_elements_csv_row_major_ordering_fname2;
	//std::string multiply_elements_csv_row_major_ordering_fname3;

	f_apply_elements_csv_to_set = false;
	//std::string apply_elements_csv_to_set_fname1;
	//std::string apply_elements_csv_to_set_fname2;
	//std::string apply_elements_csv_to_set_set;
#endif

	f_order_of_products = false;
	//order_of_products_elements = NULL;

	f_reverse_isomorphism_exterior_square = false;

	f_is_subgroup_of = false;

	f_coset_reps = false;


	// orbit stuff:


	f_subgroup_lattice = false;

	f_subgroup_lattice_load = false;
	//std::string subgroup_lattice_load_fname;


	f_subgroup_lattice_draw_by_orbits = false;

	f_subgroup_lattice_draw_by_groups = false;

	f_subgroup_lattice_intersection_orbit_orbit = false;
	subgroup_lattice_intersection_orbit_orbit_orbit1 = -1;
	subgroup_lattice_intersection_orbit_orbit_orbit2 = -1;

	f_subgroup_lattice_find_overgroup_in_orbit = false;
	subgroup_lattice_find_overgroup_in_orbit_orbit_global1 = -1;
	subgroup_lattice_find_overgroup_in_orbit_group1 = -1;
	subgroup_lattice_find_overgroup_in_orbit_orbit_global2 = -1;



	f_subgroup_lattice_create_flag_transitive_geometry_with_partition = false;
	subgroup_lattice_create_flag_transitive_geometry_with_partition_P_orbit = -1;
	subgroup_lattice_create_flag_transitive_geometry_with_partition_Q_orbit = -1;
	subgroup_lattice_create_flag_transitive_geometry_with_partition_R_orbit = -1;
	subgroup_lattice_create_flag_transitive_geometry_with_partition_R_group = -1;
	subgroup_lattice_create_flag_transitive_geometry_with_partition_intersection_size = -1;


	f_subgroup_lattice_create_coset_geometry = false;
	subgroup_lattice_create_coset_geometry_P_orb_global = -1;
	subgroup_lattice_create_coset_geometry_P_group = -1;
	subgroup_lattice_create_coset_geometry_Q_orb_global = -1;
	subgroup_lattice_create_coset_geometry_Q_group = -1;
	subgroup_lattice_create_coset_geometry_intersection_size = -1;



	f_subgroup_lattice_identify_subgroup = false;
	//std::string subgroup_lattice_identify_subgroup_subgroup_label;


#if 0
	// old style orbit function, usage is discouraged.
	// Better to use the -orbits command.

	f_orbit_of = false;
	orbit_of_point_idx = 0;


	f_orbits_on_set_system_from_file = false;
	//orbits_on_set_system_from_file_fname = NULL;
	orbits_on_set_system_first_column = 0;
	orbits_on_set_system_number_of_columns = 0;

	f_orbit_of_set_from_file = false;
	//orbit_of_set_from_file_fname = NULL;
	//f_search_subgroup = false;

#endif


	// classification:

	f_linear_codes = false;
	//std::string linear_codes_control;
	linear_codes_minimum_distance = 0;
	linear_codes_target_size = 0;

	f_tensor_permutations = false;

	f_classify_ovoids = false;
	Ovoid_classify_description = NULL;

	//f_classify_cubic_curves = false;

	f_representation_on_polynomials = false;
	//std::string representation_on_polynomials_ring;


}

group_theoretic_activity_description::~group_theoretic_activity_description()
{
}


int group_theoretic_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "group_theoretic_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;
			if (f_v) {
				cout << "-report" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-report_sylow") == 0) {
			f_report_sylow = true;
			if (f_v) {
				cout << "-report_sylow" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-report_group_table") == 0) {
			f_report_group_table = true;
			if (f_v) {
				cout << "-report_group_table" << endl;
			}
		}
#if 0
		else if (ST.stringcmp(argv[i], "-report_classes") == 0) {
			f_report_classes = true;
			if (f_v) {
				cout << "-report_classes" << endl;
			}
		}
#endif
		else if (ST.stringcmp(argv[i], "-export_group_table") == 0) {
			f_export_group_table = true;
			if (f_v) {
				cout << "-export_group_table" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-random_element") == 0) {
			f_random_element = true;
			random_element_label.assign(argv[++i]);
			if (f_v) {
				cout << "-random_element " << random_element_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-permutation_representation_of_element") == 0) {
			f_permutation_representation_of_element = true;
			permutation_representation_element_text.assign(argv[++i]);
			if (f_v) {
				cout << "-permutation_representation_of_element "
						<< permutation_representation_element_text
						<< " " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-apply") == 0) {
			f_apply = true;
			apply_input.assign(argv[++i]);
			apply_element.assign(argv[++i]);
			if (f_v) {
				cout << "-apply " << apply_input << " " << apply_element << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-element_processing") == 0) {
			f_element_processing = true;
			element_processing_descr = NEW_OBJECT(element_processing_description);
			if (f_v) {
				cout << "-element_processing" << endl;
			}
			i += element_processing_descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -element_processing " << endl;
				element_processing_descr->print();
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}


		else if (ST.stringcmp(argv[i], "-multiply") == 0) {
			f_multiply = true;
			multiply_a.assign(argv[++i]);
			multiply_b.assign(argv[++i]);
			if (f_v) {
				cout << "-multiply " << multiply_a << " " << multiply_b << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-inverse") == 0) {
			f_inverse = true;
			inverse_a.assign(argv[++i]);
			if (f_v) {
				cout << "-inverse " << inverse_a << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-consecutive_powers") == 0) {
			f_consecutive_powers = true;
			consecutive_powers_a_text.assign(argv[++i]);
			consecutive_powers_exponent_text.assign(argv[++i]);
			if (f_v) {
				cout << "-consecutive_powers " << consecutive_powers_a_text
						<< " " << consecutive_powers_exponent_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-raise_to_the_power") == 0) {
			f_raise_to_the_power = true;
			raise_to_the_power_a_text.assign(argv[++i]);
			raise_to_the_power_exponent_text.assign(argv[++i]);
			if (f_v) {
				cout << "-raise_to_the_power " << raise_to_the_power_a_text
						<< " " << raise_to_the_power_exponent_text << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-export_orbiter") == 0) {
			f_export_orbiter = true;
			if (f_v) {
				cout << "-export_orbiter " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_gap") == 0) {
			f_export_gap = true;
			if (f_v) {
				cout << "-export_gap " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_magma") == 0) {
			f_export_magma = true;
			if (f_v) {
				cout << "-export_magma " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-canonical_image_GAP") == 0) {
			f_canonical_image_GAP = true;
			canonical_image_GAP_input_set.assign(argv[++i]);
			if (f_v) {
				cout << "-canonical_image_GAP "
						<< canonical_image_GAP_input_set << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-canonical_image") == 0) {
			f_canonical_image = true;
			canonical_image_input_set.assign(argv[++i]);
			if (f_v) {
				cout << "-canonical_image " << canonical_image_input_set << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-search_element_of_order") == 0) {
			f_search_element_of_order = true;
			search_element_order = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-search_element_of_order " << search_element_order << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-find_standard_generators") == 0) {
			f_find_standard_generators = true;
			find_standard_generators_order_a = ST.strtoi(argv[++i]);
			find_standard_generators_order_b = ST.strtoi(argv[++i]);
			find_standard_generators_order_ab = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-find_standard_generators "
						<< find_standard_generators_order_a
						<< " " << find_standard_generators_order_b
						<< " " << find_standard_generators_order_ab
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-element_rank") == 0) {
			f_element_rank = true;
			element_rank_data.assign(argv[++i]);
			if (f_v) {
				cout << "-element_rank " << element_rank_data << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-element_unrank") == 0) {
			f_element_unrank = true;
			element_unrank_data.assign(argv[++i]);
			if (f_v) {
				cout << "-element_unrank " << element_unrank_data << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-find_singer_cycle") == 0) {
			f_find_singer_cycle = true;
			if (f_v) {
				cout << "-find_singer_cycle " << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-classes_based_on_normal_form") == 0) {
			f_classes_based_on_normal_form = true;
			if (f_v) {
				cout << "-classes_based_on_normal_form" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-normalizer") == 0) {
			f_normalizer = true;
			if (f_v) {
				cout << "-normalizer" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-centralizer_of_element") == 0) {
			f_centralizer_of_element = true;
			centralizer_of_element_label.assign(argv[++i]);
			centralizer_of_element_data.assign(argv[++i]);
			if (f_v) {
				cout << "-centralizer_of_element " << centralizer_of_element_label
						<< " " << centralizer_of_element_data << endl;
			}
		}
#if 0
		else if (ST.stringcmp(argv[i], "-orbits_on_group_elements_under_conjugation") == 0) {
			f_orbits_on_group_elements_under_conjugation = true;
			orbits_on_group_elements_under_conjugation_fname.assign(argv[++i]);
			orbits_on_group_elements_under_conjugation_transporter_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-orbits_on_group_elements_under_conjugation "
						<< " " << orbits_on_group_elements_under_conjugation_fname
						<< " " << orbits_on_group_elements_under_conjugation_transporter_fname
						<< endl;
			}
		}
#endif

		else if (ST.stringcmp(argv[i], "-normalizer_of_cyclic_subgroup") == 0) {
			f_normalizer_of_cyclic_subgroup = true;
			normalizer_of_cyclic_subgroup_label.assign(argv[++i]);
			normalizer_of_cyclic_subgroup_data.assign(argv[++i]);
			if (f_v) {
				cout << "-normalizer_of_cyclic_subgroup "
						<< normalizer_of_cyclic_subgroup_label
						<< " " << normalizer_of_cyclic_subgroup_data << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-classes") == 0) {
			f_classes = true;
			if (f_v) {
				cout << "-classes " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subgroup_lattice_magma") == 0) {
			f_subgroup_lattice_magma = true;
			if (f_v) {
				cout << "-subgroup_lattice_magma " << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-find_subgroup") == 0) {
			f_find_subgroup = true;
			find_subgroup_order = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-find_subgroup " << find_subgroup_order << endl;
			}
		}
#if 0
		else if (ST.stringcmp(argv[i], "-test_if_geometric") == 0) {
			f_test_if_geometric = true;
			test_if_geometric_depth = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-test_if_geometric" << endl;
			}
		}
#endif
		else if (ST.stringcmp(argv[i], "-conjugacy_class_of") == 0) {
			f_conjugacy_class_of = true;
			conjugacy_class_of_label.assign(argv[++i]);
			conjugacy_class_of_data.assign(argv[++i]);
			if (f_v) {
				cout << "-conjugacy_class_of " << conjugacy_class_of_label
						<< " " << conjugacy_class_of_data << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-isomorphism_Klein_quadric") == 0) {
			f_isomorphism_Klein_quadric = true;
			isomorphism_Klein_quadric_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-isomorphism_Klein_quadric " << isomorphism_Klein_quadric_fname << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-print_elements") == 0) {
			f_print_elements = true;
			if (f_v) {
				cout << "-print_elements " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-print_elements_tex") == 0) {
			f_print_elements_tex = true;
			if (f_v) {
				cout << "-print_elements_tex " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-save_elements_csv") == 0) {
			f_save_elements_csv = true;
			save_elements_csv_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-save_elements_csv " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_inversion_graphs") == 0) {
			f_export_inversion_graphs = true;
			export_inversion_graphs_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-export_inversion_graphs " << export_inversion_graphs_fname << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-evaluate_word") == 0) {
			f_evaluate_word = true;
			evaluate_word_word.assign(argv[++i]);
			evaluate_word_gens.assign(argv[++i]);
			if (f_v) {
				cout << "-evaluate_word "
						<< " " << evaluate_word_word
						<< " " << evaluate_word_gens
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-multiply_all_elements_in_lex_order") == 0) {
			f_multiply_all_elements_in_lex_order = true;
			if (f_v) {
				cout << "-multiply_all_elements_in_lex_order " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-stats") == 0) {
			f_stats = true;
			stats_fname_base.assign(argv[++i]);
			if (f_v) {
				cout << "-stats " << stats_fname_base << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-move_a_to_b") == 0) {
			f_move_a_to_b = true;
			move_a_to_b_a = ST.strtoi(argv[++i]);
			move_a_to_b_b = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-move_a_to_b " << move_a_to_b_a << " " << move_a_to_b_b << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-rational_normal_form") == 0) {
			f_rational_normal_form = true;
			rational_normal_form_input.assign(argv[++i]);
			if (f_v) {
				cout << "-rational_normal_form "
						<< rational_normal_form_input
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-find_conjugating_element") == 0) {
			f_find_conjugating_element = true;
			find_conjugating_element_element_from.assign(argv[++i]);
			find_conjugating_element_element_to.assign(argv[++i]);
			if (f_v) {
				cout << "-find_conjugating_element "
						<< find_conjugating_element_element_from
						<< " " << find_conjugating_element_element_to
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-group_of_automorphisms_by_images_of_generators") == 0) {
			f_group_of_automorphisms_by_images_of_generators = true;
			group_of_automorphisms_by_images_of_generators_label.assign(argv[++i]);
			group_of_automorphisms_by_images_of_generators_elements.assign(argv[++i]);
			group_of_automorphisms_by_images_of_generators_images.assign(argv[++i]);
			if (f_v) {
				cout << "-group_of_automorphisms_by_images_of_generators "
						<< " " << group_of_automorphisms_by_images_of_generators_label
						<< " " << group_of_automorphisms_by_images_of_generators_elements
						<< " " << group_of_automorphisms_by_images_of_generators_images
						<< endl;
			}
		}

		// 3:

#if 0
		else if (ST.stringcmp(argv[i], "-multiply_elements_csv_column_major_ordering") == 0) {
			f_multiply_elements_csv_column_major_ordering = true;
			multiply_elements_csv_column_major_ordering_fname1.assign(argv[++i]);
			multiply_elements_csv_column_major_ordering_fname2.assign(argv[++i]);
			multiply_elements_csv_column_major_ordering_fname3.assign(argv[++i]);
			if (f_v) {
				cout << "-multiply_elements_csv_column_major_ordering "
						<< multiply_elements_csv_column_major_ordering_fname1 << " "
						<< multiply_elements_csv_column_major_ordering_fname2 << " "
						<< multiply_elements_csv_column_major_ordering_fname3 << " "
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-multiply_elements_csv_row_major_ordering") == 0) {
			f_multiply_elements_csv_row_major_ordering = true;
			multiply_elements_csv_row_major_ordering_fname1.assign(argv[++i]);
			multiply_elements_csv_row_major_ordering_fname2.assign(argv[++i]);
			multiply_elements_csv_row_major_ordering_fname3.assign(argv[++i]);
			if (f_v) {
				cout << "-multiply_elements_csv_row_major_ordering "
						<< multiply_elements_csv_row_major_ordering_fname1 << " "
						<< multiply_elements_csv_row_major_ordering_fname2 << " "
						<< multiply_elements_csv_row_major_ordering_fname3 << " "
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-apply_elements_csv_to_set") == 0) {
			f_apply_elements_csv_to_set = true;
			apply_elements_csv_to_set_fname1.assign(argv[++i]);
			apply_elements_csv_to_set_fname2.assign(argv[++i]);
			apply_elements_csv_to_set_set.assign(argv[++i]);
			if (f_v) {
				cout << "-apply_elements_csv_to_set "
						<< apply_elements_csv_to_set_fname1 << " "
						<< apply_elements_csv_to_set_fname2 << " "
						<< apply_elements_csv_to_set_set << " "
						<< endl;
			}
		}
#endif



		else if (ST.stringcmp(argv[i], "-order_of_products") == 0) {
			f_order_of_products = true;
			order_of_products_elements.assign(argv[++i]);
			if (f_v) {
				cout << "-order_of_products " << order_of_products_elements << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-reverse_isomorphism_exterior_square") == 0) {
			f_reverse_isomorphism_exterior_square = true;
			if (f_v) {
				cout << "-reverse_isomorphism_exterior_square " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-is_subgroup_of") == 0) {
			f_is_subgroup_of = true;
			if (f_v) {
				cout << "-is_subgroup_of " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-coset_reps") == 0) {
			f_coset_reps = true;
			if (f_v) {
				cout << "-coset_reps " << endl;
			}
		}


		// orbit stuff:



		else if (ST.stringcmp(argv[i], "-subgroup_lattice") == 0) {
			f_subgroup_lattice = true;
			if (f_v) {
				cout << "-subgroup_lattice " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subgroup_lattice_load") == 0) {
			f_subgroup_lattice_load = true;
			subgroup_lattice_load_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-subgroup_lattice_load " << subgroup_lattice_load_fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subgroup_lattice_draw_by_orbits") == 0) {
			f_subgroup_lattice_draw_by_orbits = true;
			if (f_v) {
				cout << "-subgroup_lattice_draw_by_orbits " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subgroup_lattice_draw_by_groups") == 0) {
			f_subgroup_lattice_draw_by_groups = true;
			if (f_v) {
				cout << "-subgroup_lattice_draw_by_groups " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subgroup_lattice_intersection_orbit_orbit") == 0) {
			f_subgroup_lattice_intersection_orbit_orbit = true;
			subgroup_lattice_intersection_orbit_orbit_orbit1 = ST.strtoi(argv[++i]);
			subgroup_lattice_intersection_orbit_orbit_orbit2 = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-subgroup_lattice_intersection_orbit_orbit "
						<< subgroup_lattice_intersection_orbit_orbit_orbit1
						<< " " << subgroup_lattice_intersection_orbit_orbit_orbit2 << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subgroup_lattice_find_overgroup_in_orbit") == 0) {
			f_subgroup_lattice_find_overgroup_in_orbit = true;
			subgroup_lattice_find_overgroup_in_orbit_orbit_global1 = ST.strtoi(argv[++i]);
			subgroup_lattice_find_overgroup_in_orbit_group1 = ST.strtoi(argv[++i]);
			subgroup_lattice_find_overgroup_in_orbit_orbit_global2 = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-subgroup_lattice_find_overgroup_in_orbit "
						<< subgroup_lattice_find_overgroup_in_orbit_orbit_global1
						<< " " << subgroup_lattice_find_overgroup_in_orbit_group1
						<< " " << subgroup_lattice_find_overgroup_in_orbit_orbit_global2 << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subgroup_lattice_create_flag_transitive_geometry_with_partition") == 0) {
			f_subgroup_lattice_create_flag_transitive_geometry_with_partition = true;
			subgroup_lattice_create_flag_transitive_geometry_with_partition_P_orbit = ST.strtoi(argv[++i]);
			subgroup_lattice_create_flag_transitive_geometry_with_partition_Q_orbit = ST.strtoi(argv[++i]);
			subgroup_lattice_create_flag_transitive_geometry_with_partition_R_orbit = ST.strtoi(argv[++i]);
			subgroup_lattice_create_flag_transitive_geometry_with_partition_R_group = ST.strtoi(argv[++i]);
			subgroup_lattice_create_flag_transitive_geometry_with_partition_intersection_size = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-subgroup_lattice_create_flag_transitive_geometry_with_partition "
						<< subgroup_lattice_create_flag_transitive_geometry_with_partition_P_orbit
						<< " " << subgroup_lattice_create_flag_transitive_geometry_with_partition_Q_orbit
						<< " " << subgroup_lattice_create_flag_transitive_geometry_with_partition_R_orbit
						<< " " << subgroup_lattice_create_flag_transitive_geometry_with_partition_R_group
						<< " " << subgroup_lattice_create_flag_transitive_geometry_with_partition_intersection_size
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subgroup_lattice_create_coset_geometry") == 0) {
			f_subgroup_lattice_create_coset_geometry = true;
			subgroup_lattice_create_coset_geometry_P_orb_global = ST.strtoi(argv[++i]);
			subgroup_lattice_create_coset_geometry_P_group = ST.strtoi(argv[++i]);
			subgroup_lattice_create_coset_geometry_Q_orb_global = ST.strtoi(argv[++i]);
			subgroup_lattice_create_coset_geometry_Q_group = ST.strtoi(argv[++i]);
			subgroup_lattice_create_coset_geometry_intersection_size = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-subgroup_lattice_create_coset_geometry "
						<< subgroup_lattice_create_coset_geometry_P_orb_global
						<< " " << subgroup_lattice_create_coset_geometry_P_group
						<< " " << subgroup_lattice_create_coset_geometry_Q_orb_global
						<< " " << subgroup_lattice_create_coset_geometry_Q_group
						<< " " << subgroup_lattice_create_coset_geometry_intersection_size
						<< endl;
			}
		}



		else if (ST.stringcmp(argv[i], "-subgroup_lattice_identify_subgroup") == 0) {
			f_subgroup_lattice_identify_subgroup = true;
			subgroup_lattice_identify_subgroup_subgroup_label.assign(argv[++i]);
			if (f_v) {
				cout << "-subgroup_lattice_identify_subgroup "
						<< subgroup_lattice_identify_subgroup_subgroup_label
						<< endl;
			}
		}




#if 0
		else if (ST.stringcmp(argv[i], "-orbit_of") == 0) {
			f_orbit_of = true;
			orbit_of_point_idx = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-orbit_of " << orbit_of_point_idx << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-orbit_of_set_from_file") == 0) {
			f_orbit_of_set_from_file = true;
			orbit_of_set_from_file_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-orbit_of_set_from_file"
						<< orbit_of_set_from_file_fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-orbits_on_set_system_from_file") == 0) {
			f_orbits_on_set_system_from_file = true;
			orbits_on_set_system_from_file_fname.assign(argv[++i]);
			orbits_on_set_system_first_column = ST.strtoi(argv[++i]);
			orbits_on_set_system_number_of_columns = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-orbits_on_set_system_from_file"
						<< orbits_on_set_system_from_file_fname
						<< " " << orbits_on_set_system_first_column << " "
						<< orbits_on_set_system_number_of_columns << endl;
			}
		}
#endif


		// classification tasks:

		// linear codes:

		else if (ST.stringcmp(argv[i], "-linear_codes") == 0) {
			f_linear_codes = true;
			linear_codes_control.assign(argv[++i]);
			linear_codes_minimum_distance = ST.strtoi(argv[++i]);
			linear_codes_target_size = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-linear_codes " << linear_codes_control
						<< " " << linear_codes_minimum_distance
						<< " " << linear_codes_target_size << endl;
			}
		}





		// tensors:

		else if (ST.stringcmp(argv[i], "-tensor_permutations") == 0) {
			f_tensor_permutations = true;
			if (f_v) {
				cout << "-tensor_permutations " << endl;
			}
		}


		// ovoids:

		else if (ST.stringcmp(argv[i], "-classify_ovoids") == 0) {
			f_classify_ovoids = true;
			Ovoid_classify_description = NEW_OBJECT(apps_geometry::ovoid_classify_description);
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





		else if (ST.stringcmp(argv[i], "-representation_on_polynomials") == 0) {
			f_representation_on_polynomials = true;
			representation_on_polynomials_ring.assign(argv[++i]);
			if (f_v) {
				cout << "-representation_on_polynomials "
						<< representation_on_polynomials_ring << endl;
			}
		}




		else if (ST.stringcmp(argv[i], "-end") == 0) {
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
	if (f_report) {
		cout << "-report" << endl;
	}
	if (f_report_sylow) {
		cout << "-report_sylow" << endl;
	}
	if (f_report_group_table) {
		cout << "-report_group_table" << endl;
	}
#if 0
	if (f_report_classes) {
		cout << "-report_classes" << endl;
	}
#endif
	if (f_export_group_table) {
		cout << "-export_group_table" << endl;
	}
	if (f_random_element) {
		cout << "-random_element " << random_element_label << endl;
	}
	if (f_permutation_representation_of_element) {
		cout << "-permutation_representation_of_element "
				<< permutation_representation_element_text
				<< " " << endl;
	}
	if (f_apply) {
		cout << "-apply " << apply_input << " " << apply_element << endl;
	}
	if (f_element_processing) {
		cout << "-element_processing " << endl;
		element_processing_descr->print();
	}
	if (f_multiply) {
		cout << "-multiply " << multiply_a << " " << multiply_b << endl;
	}
	if (f_inverse) {
		cout << "-inverse " << inverse_a << endl;
	}

	if (f_consecutive_powers) {
		cout << "-consecutive_powers " << consecutive_powers_a_text
				<< " " << consecutive_powers_exponent_text << endl;
	}

	if (f_raise_to_the_power) {
		cout << "-raise_to_the_power " << raise_to_the_power_a_text
					<< " " << raise_to_the_power_exponent_text << endl;
	}

	if (f_export_orbiter) {
		cout << "-export_orbiter " << endl;
	}
	if (f_export_gap) {
		cout << "-export_gap " << endl;
	}
	if (f_export_magma) {
		cout << "-export_magma " << endl;
	}
	if (f_canonical_image_GAP) {
		cout << "-canonical_image_GAP " << canonical_image_GAP_input_set << endl;
	}
	if (f_canonical_image) {
		cout << "-canonical_image " << canonical_image_input_set << endl;
	}

	if (f_search_element_of_order) {
		cout << "-search_element_of_order " << search_element_order << endl;
	}
	if (f_find_standard_generators) {
		cout << "-find_standard_generators "
				<< find_standard_generators_order_a
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
	if (f_classes_based_on_normal_form) {
		cout << "-classes_based_on_normal_form" << endl;
	}
	if (f_normalizer) {
		cout << "-normalizer" << endl;
	}
	if (f_centralizer_of_element) {
		cout << "-centralizer_of_element " << centralizer_of_element_label
				<< " " << centralizer_of_element_data << endl;
	}
#if 0
	if (f_orbits_on_group_elements_under_conjugation) {
		cout << "-orbits_on_group_elements_under_conjugation "
			<< " " << orbits_on_group_elements_under_conjugation_fname
			<< " " << orbits_on_group_elements_under_conjugation_transporter_fname
			<< endl;
	}
#endif

	if (f_normalizer_of_cyclic_subgroup) {
		cout << "-normalizer_of_cyclic_subgroup " << normalizer_of_cyclic_subgroup_label
				<< " " << normalizer_of_cyclic_subgroup_data << endl;
	}
	if (f_classes) {
		cout << "-classes " << endl;
	}
	if (f_subgroup_lattice_magma) {
		cout << "-subgroup_lattice_magma " << endl;
	}

	if (f_find_subgroup) {
		cout << "-find_subgroup " << find_subgroup_order << endl;
	}
#if 0
	if (f_test_if_geometric) {
		cout << "-test_if_geometric " << test_if_geometric_depth << endl;
	}
#endif
	if (f_conjugacy_class_of) {
		cout << "-conjugacy_class_of " << conjugacy_class_of_label
				<< " " << conjugacy_class_of_data << endl;
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
	if (f_save_elements_csv) {
		cout << "-save_elements_csv " << save_elements_csv_fname << endl;
	}
	if (f_export_inversion_graphs) {
		cout << "-export_inversion_graphs " << export_inversion_graphs_fname << endl;
	}
	if (f_evaluate_word) {
		cout << "-evaluate_word "
				<< " " << evaluate_word_word
				<< " " << evaluate_word_gens
				<< endl;
	}
	if (f_multiply_all_elements_in_lex_order) {
		cout << "-multiply_all_elements_in_lex_order " << endl;
	}
	if (f_stats) {
		cout << "-stats " << stats_fname_base << endl;
	}
	if (f_move_a_to_b) {
		cout << "-move_a_to_b " << move_a_to_b_a << " " << move_a_to_b_b << endl;
	}
	if (f_rational_normal_form) {
		cout << "-rational_normal_form "
				<< rational_normal_form_input
				<< endl;
	}

	if (f_find_conjugating_element) {
		cout << "-find_conjugating_element "
				<< find_conjugating_element_element_from
				<< " " << find_conjugating_element_element_to
				<< endl;
	}

	if (f_group_of_automorphisms_by_images_of_generators) {
		cout << "-group_of_automorphisms_by_images_of_generators "
				<< " " << group_of_automorphisms_by_images_of_generators_label
				<< " " << group_of_automorphisms_by_images_of_generators_elements
				<< " " << group_of_automorphisms_by_images_of_generators_images
				<< endl;
	}


	// 3:

#if 0
	if (f_multiply_elements_csv_column_major_ordering) {
		cout << "-multiply_elements_csv_column_major_ordering "
				<< multiply_elements_csv_column_major_ordering_fname1 << " "
				<< multiply_elements_csv_column_major_ordering_fname2 << " "
				<< multiply_elements_csv_column_major_ordering_fname3 << " "
				<< endl;
	}
	if (f_multiply_elements_csv_row_major_ordering) {
		cout << "-multiply_elements_csv_row_major_ordering "
				<< multiply_elements_csv_row_major_ordering_fname1 << " "
				<< multiply_elements_csv_row_major_ordering_fname2 << " "
				<< multiply_elements_csv_row_major_ordering_fname3 << " "
				<< endl;
	}


	if (f_apply_elements_csv_to_set) {
		cout << "-apply_elements_csv_to_set "
				<< apply_elements_csv_to_set_fname1 << " "
				<< apply_elements_csv_to_set_fname2 << " "
				<< apply_elements_csv_to_set_set << " "
				<< endl;
	}
#endif

	if (f_order_of_products) {
		cout << "-order_of_products " << order_of_products_elements << endl;
	}
	if (f_reverse_isomorphism_exterior_square) {
		cout << "-reverse_isomorphism_exterior_square " << endl;
	}
	if (f_is_subgroup_of) {
		cout << "-is_subgroup_of " << endl;
	}
	if (f_coset_reps) {
		cout << "-coset_reps " << endl;
	}


	// orbit stuff:


	if (f_subgroup_lattice) {
		cout << "-subgroup_lattice " << endl;
	}

	if (f_subgroup_lattice_load) {
		cout << "-subgroup_lattice_load " << subgroup_lattice_load_fname << endl;
	}
	if (f_subgroup_lattice_draw_by_orbits) {
		cout << "-subgroup_lattice_draw_by_orbits " << endl;
	}
	if (f_subgroup_lattice_draw_by_groups) {
		cout << "-subgroup_lattice_draw_by_groups " << endl;
	}
	if (f_subgroup_lattice_intersection_orbit_orbit) {
		cout << "-subgroup_lattice_intersection_orbit_orbit "
				<< subgroup_lattice_intersection_orbit_orbit_orbit1
				<< " " << subgroup_lattice_intersection_orbit_orbit_orbit2 << endl;
	}
	if (f_subgroup_lattice_find_overgroup_in_orbit) {
		cout << "-subgroup_lattice_find_overgroup_in_orbit "
				<< subgroup_lattice_find_overgroup_in_orbit_orbit_global1
				<< " " << subgroup_lattice_find_overgroup_in_orbit_group1
				<< " " << subgroup_lattice_find_overgroup_in_orbit_orbit_global2 << endl;
	}
	if (f_subgroup_lattice_create_flag_transitive_geometry_with_partition) {
		cout << "-subgroup_lattice_create_flag_transitive_geometry_with_partition "
				<< subgroup_lattice_create_flag_transitive_geometry_with_partition_P_orbit
				<< " " << subgroup_lattice_create_flag_transitive_geometry_with_partition_Q_orbit
				<< " " << subgroup_lattice_create_flag_transitive_geometry_with_partition_R_orbit
				<< " " << subgroup_lattice_create_flag_transitive_geometry_with_partition_R_group
				<< " " << subgroup_lattice_create_flag_transitive_geometry_with_partition_intersection_size
				<< endl;
	}
	if (f_subgroup_lattice_create_coset_geometry) {
		cout << "-subgroup_lattice_create_coset_geometry "
				<< subgroup_lattice_create_coset_geometry_P_orb_global
				<< " " << subgroup_lattice_create_coset_geometry_P_group
				<< " " << subgroup_lattice_create_coset_geometry_Q_orb_global
				<< " " << subgroup_lattice_create_coset_geometry_Q_group
				<< " " << subgroup_lattice_create_coset_geometry_intersection_size
				<< endl;
	}
	if (f_subgroup_lattice_identify_subgroup) {
		cout << "-subgroup_lattice_identify_subgroup "
				<< subgroup_lattice_identify_subgroup_subgroup_label
				<< endl;
	}

#if 0
	if (f_orbit_of) {
		cout << "-orbit_of " << orbit_of_point_idx << endl;
	}
	if (f_orbits_on_set_system_from_file) {
		cout << "-orbits_on_set_system_from_file"
				<< orbits_on_set_system_from_file_fname
				<< " " << orbits_on_set_system_first_column << " "
				<< orbits_on_set_system_number_of_columns << endl;
	}
	if (f_orbit_of_set_from_file) {
		cout << "-orbit_of_set_from_file"
					<< orbit_of_set_from_file_fname << endl;
	}
#endif




	// linear codes:

	if (f_linear_codes) {
			cout << "-linear_codes " << linear_codes_control
					<< " " << linear_codes_minimum_distance
					<< " " << linear_codes_target_size << endl;
	}

	if (f_tensor_permutations) {
		cout << "-tensor_permutations " << endl;
	}



	// ovoids:

	if (f_classify_ovoids) {
		cout << "-classify_ovoids" << endl;
		Ovoid_classify_description->print();
	}


	if (f_representation_on_polynomials) {
		cout << "-representation_on_polynomials "
				<< representation_on_polynomials_ring << endl;
	}


}





}}}

