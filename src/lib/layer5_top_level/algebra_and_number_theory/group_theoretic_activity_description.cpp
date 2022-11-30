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

	f_apply = FALSE;
	//std::string apply_input;
	//std::string apply_element;

	f_multiply = FALSE;
	//multiply_a = NULL;
	//multiply_b = NULL;

	f_inverse = FALSE;
	//inverse_a = NULL;

	f_consecutive_powers = FALSE;
	//std::string consecutive_powers_a_text;
	//std::string consecutive_powers_exponent_text;

	f_raise_to_the_power = FALSE;
	//std::string raise_to_the_power_a_text;
	//std::string raise_to_the_power_exponent_text;

	f_export_orbiter = FALSE;

	f_export_gap = FALSE;

	f_export_magma = FALSE;

	f_canonical_image = FALSE;
	//std::string canonical_image_input_set;

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

	f_classes_based_on_normal_form = FALSE;


	// Magma:
	f_normalizer = FALSE;

	// Magma:
	f_centralizer_of_element = FALSE;
	//element_description_text = NULL;
	//element_label = NULL;

	f_permutation_representation_of_element = FALSE;
	//std::string permutation_representation_element_text;

	f_conjugacy_class_of_element = FALSE;

	f_orbits_on_group_elements_under_conjugation = FALSE;
	//std::string orbits_on_group_elements_under_conjugation_fname;
	//orbits_on_group_elements_under_conjugation_transporter_fname

	// Magma:
	f_normalizer_of_cyclic_subgroup = FALSE;

	// Magma:
	f_classes = FALSE;

	f_find_subgroup = FALSE;
	find_subgroup_order = 0;

	f_report = FALSE;
	f_report_sylow = FALSE;
	f_report_group_table = FALSE;
	f_report_classes = FALSE;

	f_export_group_table = FALSE;


	f_test_if_geometric = FALSE;
	test_if_geometric_depth = 0;

	f_conjugacy_class_of = FALSE;
	//std::string conjugacy_class_of_data;

	f_isomorphism_Klein_quadric = FALSE;
	//std::string isomorphism_Klein_quadric_fname;

	f_print_elements = FALSE;
	f_print_elements_tex = FALSE;

	f_save_elements_csv = FALSE;
	//std::string save_elements_csv_fname;

	f_export_inversion_graphs = FALSE;
	//std::string export_inversion_graphs_fname;


	f_multiply_elements_csv_column_major_ordering = FALSE;
	//std::string multiply_elements_csv_column_major_ordering_fname1;
	//std::string multiply_elements_csv_column_major_ordering_fname2;
	//std::string multiply_elements_csv_column_major_ordering_fname3;

	f_multiply_elements_csv_row_major_ordering = FALSE;
	//std::string multiply_elements_csv_row_major_ordering_fname1;
	//std::string multiply_elements_csv_row_major_ordering_fname2;
	//std::string multiply_elements_csv_row_major_ordering_fname3;

	f_apply_elements_csv_to_set = FALSE;
	//std::string apply_elements_csv_to_set_fname1;
	//std::string apply_elements_csv_to_set_fname2;
	//std::string apply_elements_csv_to_set_set;

	f_order_of_products = FALSE;
	//order_of_products_elements = NULL;

	f_reverse_isomorphism_exterior_square = FALSE;

	f_is_subgroup_of = FALSE;
	f_coset_reps = FALSE;



	f_orbit_of = FALSE;
	orbit_of_point_idx = 0;

	f_orbits_on_set_system_from_file = FALSE;
	//orbits_on_set_system_from_file_fname = NULL;
	orbits_on_set_system_first_column = 0;
	orbits_on_set_system_number_of_columns = 0;

	f_orbit_of_set_from_file = FALSE;
	//orbit_of_set_from_file_fname = NULL;
	//f_search_subgroup = FALSE;




	// classification:

	f_linear_codes = FALSE;
	//std::string linear_codes_control;
	linear_codes_minimum_distance = 0;
	linear_codes_target_size = 0;

	f_tensor_permutations = FALSE;

	f_classify_ovoids = FALSE;
	Ovoid_classify_description = NULL;

	f_classify_cubic_curves = FALSE;

	f_representation_on_polynomials = FALSE;
	representation_on_polynomials_degree = 0;


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

		if (ST.stringcmp(argv[i], "-apply") == 0) {
			f_apply = TRUE;
			apply_input.assign(argv[++i]);
			apply_element.assign(argv[++i]);
			if (f_v) {
				cout << "-apply " << apply_input << " " << apply_element << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-multiply") == 0) {
			f_multiply = TRUE;
			multiply_a.assign(argv[++i]);
			multiply_b.assign(argv[++i]);
			if (f_v) {
				cout << "-multiply " << multiply_a << " " << multiply_b << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-inverse") == 0) {
			f_inverse = TRUE;
			inverse_a.assign(argv[++i]);
			if (f_v) {
				cout << "-inverse " << inverse_a << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-consecutive_powers") == 0) {
			f_consecutive_powers = TRUE;
			consecutive_powers_a_text.assign(argv[++i]);
			consecutive_powers_exponent_text.assign(argv[++i]);
			if (f_v) {
				cout << "-consecutive_powers " << consecutive_powers_a_text
						<< " " << consecutive_powers_exponent_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-raise_to_the_power") == 0) {
			f_raise_to_the_power = TRUE;
			raise_to_the_power_a_text.assign(argv[++i]);
			raise_to_the_power_exponent_text.assign(argv[++i]);
			if (f_v) {
				cout << "-raise_to_the_power " << raise_to_the_power_a_text
						<< " " << raise_to_the_power_exponent_text << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-export_orbiter") == 0) {
			f_export_orbiter = TRUE;
			if (f_v) {
				cout << "-export_orbiter " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_gap") == 0) {
			f_export_gap = TRUE;
			if (f_v) {
				cout << "-export_gap " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_magma") == 0) {
			f_export_magma = TRUE;
			if (f_v) {
				cout << "-export_magma " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-canonical_image") == 0) {
			f_canonical_image = TRUE;
			canonical_image_input_set.assign(argv[++i]);
			if (f_v) {
				cout << "-canonical_image " << canonical_image_input_set << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-search_element_of_order") == 0) {
			f_search_element_of_order = TRUE;
			search_element_order = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-search_element_of_order " << search_element_order << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-find_standard_generators") == 0) {
			f_find_standard_generators = TRUE;
			find_standard_generators_order_a = ST.strtoi(argv[++i]);
			find_standard_generators_order_b = ST.strtoi(argv[++i]);
			find_standard_generators_order_ab = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-find_standard_generators " << find_standard_generators_order_a
						<< " " << find_standard_generators_order_b
						<< " " << find_standard_generators_order_ab
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-element_rank") == 0) {
			f_element_rank = TRUE;
			element_rank_data.assign(argv[++i]);
			if (f_v) {
				cout << "-element_rank " << element_rank_data << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-element_unrank") == 0) {
			f_element_unrank = TRUE;
			element_unrank_data.assign(argv[++i]);
			if (f_v) {
				cout << "-element_unrank " << element_unrank_data << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-find_singer_cycle") == 0) {
			f_find_singer_cycle = TRUE;
			if (f_v) {
				cout << "-find_singer_cycle " << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-classes_based_on_normal_form") == 0) {
			f_classes_based_on_normal_form = TRUE;
			if (f_v) {
				cout << "-classes_based_on_normal_form" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-normalizer") == 0) {
			f_normalizer = TRUE;
			if (f_v) {
				cout << "-normalizer" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-centralizer_of_element") == 0) {
			f_centralizer_of_element = TRUE;
			element_label.assign(argv[++i]);
			element_description_text.assign(argv[++i]);
			if (f_v) {
				cout << "-centralizer_of_element " << element_label
						<< " " << element_description_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-permutation_representation_of_element") == 0) {
			f_permutation_representation_of_element = TRUE;
			permutation_representation_element_text.assign(argv[++i]);
			if (f_v) {
				cout << "-permutation_representation_of_element " << permutation_representation_element_text
						<< " " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-conjugacy_class_of_element") == 0) {
			f_conjugacy_class_of_element = TRUE;
			element_label.assign(argv[++i]);
			element_description_text.assign(argv[++i]);
			if (f_v) {
				cout << "-conjugacy_class_of_element " << element_label
						<< " " << element_description_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-orbits_on_group_elements_under_conjugation") == 0) {
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


		else if (ST.stringcmp(argv[i], "-normalizer_of_cyclic_subgroup") == 0) {
			f_normalizer_of_cyclic_subgroup = TRUE;
			element_label.assign(argv[++i]);
			element_description_text.assign(argv[++i]);
			if (f_v) {
				cout << "-normalizer_of_cyclic_subgroup " << element_label
						<< " " << element_description_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-classes") == 0) {
			f_classes = TRUE;
			if (f_v) {
				cout << "-classes " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-find_subgroup") == 0) {
			f_find_subgroup = TRUE;
			find_subgroup_order = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-find_subgroup " << find_subgroup_order << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			if (f_v) {
				cout << "-report" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-report_sylow") == 0) {
			f_report_sylow = TRUE;
			if (f_v) {
				cout << "-report_sylow" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-report_group_table") == 0) {
			f_report_group_table = TRUE;
			if (f_v) {
				cout << "-report_group_table" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-report_classes") == 0) {
			f_report_classes = TRUE;
			if (f_v) {
				cout << "-report_classes" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_group_table") == 0) {
			f_export_group_table = TRUE;
			if (f_v) {
				cout << "-export_group_table" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-test_if_geometric") == 0) {
			f_test_if_geometric = TRUE;
			test_if_geometric_depth = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-test_if_geometric" << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-conjugacy_class_of") == 0) {
			f_conjugacy_class_of = TRUE;
			conjugacy_class_of_data.assign(argv[++i]);
			if (f_v) {
				cout << "-conjugacy_class_of " << conjugacy_class_of_data << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-isomorphism_Klein_quadric") == 0) {
			f_isomorphism_Klein_quadric = TRUE;
			isomorphism_Klein_quadric_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-isomorphism_Klein_quadric " << isomorphism_Klein_quadric_fname << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-print_elements") == 0) {
			f_print_elements = TRUE;
			if (f_v) {
				cout << "-print_elements " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-print_elements_tex") == 0) {
			f_print_elements_tex = TRUE;
			if (f_v) {
				cout << "-print_elements_tex " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-save_elements_csv") == 0) {
			f_save_elements_csv = TRUE;
			save_elements_csv_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-save_elements_csv " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_inversion_graphs") == 0) {
			f_export_inversion_graphs = TRUE;
			export_inversion_graphs_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-export_inversion_graphs " << export_inversion_graphs_fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-multiply_elements_csv_column_major_ordering") == 0) {
			f_multiply_elements_csv_column_major_ordering = TRUE;
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
			f_multiply_elements_csv_row_major_ordering = TRUE;
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
			f_apply_elements_csv_to_set = TRUE;
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



		else if (ST.stringcmp(argv[i], "-order_of_products") == 0) {
			f_order_of_products = TRUE;
			order_of_products_elements.assign(argv[++i]);
			if (f_v) {
				cout << "-order_of_products " << order_of_products_elements << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-reverse_isomorphism_exterior_square") == 0) {
			f_reverse_isomorphism_exterior_square = TRUE;
			if (f_v) {
				cout << "-reverse_isomorphism_exterior_square " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-is_subgroup_of") == 0) {
			f_is_subgroup_of = TRUE;
			if (f_v) {
				cout << "-is_subgroup_of " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-coset_reps") == 0) {
			f_coset_reps = TRUE;
			if (f_v) {
				cout << "-coset_reps " << endl;
			}
		}



		else if (ST.stringcmp(argv[i], "-orbit_of") == 0) {
			f_orbit_of = TRUE;
			orbit_of_point_idx = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-orbit_of " << orbit_of_point_idx << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-orbit_of_set_from_file") == 0) {
			f_orbit_of_set_from_file = TRUE;
			orbit_of_set_from_file_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-orbit_of_set_from_file"
						<< orbit_of_set_from_file_fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-orbits_on_set_system_from_file") == 0) {
			f_orbits_on_set_system_from_file = TRUE;
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



		// classification tasks:

		// linear codes:

		else if (ST.stringcmp(argv[i], "-linear_codes") == 0) {
			f_linear_codes = TRUE;
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
			f_tensor_permutations = TRUE;
			if (f_v) {
				cout << "-tensor_permutations " << endl;
			}
		}


		// ovoids:

		else if (ST.stringcmp(argv[i], "-classify_ovoids") == 0) {
			f_classify_ovoids = TRUE;
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
			f_representation_on_polynomials = TRUE;
			representation_on_polynomials_degree = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-representation_on_polynomials " << representation_on_polynomials_degree << endl;
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
	if (f_apply) {
		cout << "-apply " << apply_input << " " << apply_element << endl;
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
	if (f_canonical_image) {
		cout << "-canonical_image " << canonical_image_input_set << endl;
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
	if (f_classes_based_on_normal_form) {
		cout << "-classes_based_on_normal_form" << endl;
	}
	if (f_normalizer) {
		cout << "-normalizer" << endl;
	}
	if (f_centralizer_of_element) {
		cout << "-centralizer_of_element " << element_label
				<< " " << element_description_text << endl;
	}
	if (f_permutation_representation_of_element) {
		cout << "-permutation_representation_of_element " << permutation_representation_element_text
				<< " " << endl;
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
	if (f_classes) {
		cout << "-classes " << endl;
	}
	if (f_find_subgroup) {
		cout << "-find_subgroup " << find_subgroup_order << endl;
	}
	if (f_report) {
		cout << "-report" << endl;
	}
	if (f_report_sylow) {
		cout << "-report_sylow" << endl;
	}
	if (f_report_group_table) {
		cout << "-report_group_table" << endl;
	}
	if (f_report_classes) {
		cout << "-report_classes" << endl;
	}
	if (f_export_group_table) {
		cout << "-export_group_table" << endl;
	}
	if (f_test_if_geometric) {
		cout << "-test_if_geometric " << test_if_geometric_depth << endl;
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
	if (f_save_elements_csv) {
		cout << "-save_elements_csv " << save_elements_csv_fname << endl;
	}
	if (f_export_inversion_graphs) {
		cout << "-export_inversion_graphs " << export_inversion_graphs_fname << endl;
	}

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


	if (f_orbit_of) {
		cout << "-orbit_of " << orbit_of_point_idx << endl;
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
		cout << "-representation_on_polynomials " << representation_on_polynomials_degree << endl;
	}


}





}}}

